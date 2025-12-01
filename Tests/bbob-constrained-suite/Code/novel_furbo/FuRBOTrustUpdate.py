from botorch.generation.sampling import ConstrainedMaxPosteriorSampling
from botorch.generation.sampling import MaxPosteriorSampling

import math
import torch

###
from utilities import multivariate_ellipsoid
from utilities import get_fitted_model
#####
# Helper: PCA on samples

def _compute_pca_from_samples(samples: torch.Tensor, eps: float = 1e-8):
    device = samples.device
    dtype = samples.dtype
    mu = samples.mean(dim=0)
    Xc = samples - mu
    n = samples.shape[0]
    d = samples.shape[1]
    # covariance computation
    C = (Xc.t() @ Xc) / max(n - 1, 1)
    C = C + eps * torch.eye(d, device=device, dtype=dtype)
    eigvals, eigvecs = torch.linalg.eigh(C)  # eigenvals in ascending order
    eigvals = eigvals.flip(0)
    eigvecs = eigvecs.flip(1)
    return mu, eigvecs, eigvals

#  Convert eigenvalues into axis lengths
def _make_radii_from_eigvals(eigvals: torch.Tensor, base_scale: float, min_length: float, max_length: float):

    lengths = base_scale * torch.sqrt(torch.clamp(eigvals, min=1e-12))
    lengths = lengths.clamp(min=min_length, max=max_length)
    return lengths

# Extract GP lengthscales
def _get_gp_lengthscales(model):

    try:
        # single GP with SE/ RBF base kernel
        ls = model.covar_module.base_kernel.lengthscale
    except Exception:
        try:
            ls = model.models[0].covar_module.base_kernel.lengthscale
        except Exception:
            return None
    ls = ls.detach().squeeze()
    return ls



#######
# PCA-based multi-trust-region adaptation function

def multinormal_radius(state,              # FuRBO state
                       n_samples_factor: int = 1000,
                       percentage: float = 0.1,
                       base_scale: float = 2.0,
                       min_axis_fraction: float = 1e-3,
                       max_axis_fraction: float = 0.5,
                       eps: float = 1e-8,
                       **tkwargs
                       ):
    d = state.dim
    n_samples = max(4, n_samples_factor * d)
    lb = torch.zeros(d, **tkwargs)
    ub = torch.ones(d, **tkwargs)
 
    # Clamping values (normalized domain)
    min_axis_length = min_axis_fraction
    max_axis_length = max_axis_fraction

    # Ensure tr_radii is in the correct shape (tr_number, d)
    if state.tr_radii.ndim == 1:
        scalar_radii = state.tr_radii.clone()

        new_radii = []
        for i in range(state.tr_number):
            repeated = scalar_radii[i].repeat(d)
            new_radii.append(repeated)

        state.tr_radii = torch.stack(new_radii, dim=0).to(**tkwargs)


    # For each trust region
    for ind in range(state.tr_number):
        # Sample points inside the current trust region
        samples = multivariate_ellipsoid( center=state.tr_center[ind], radii=state.tr_radii[ind],R=state.tr_R[ind],n_samples=n_samples,
                                         lb=lb, ub=ub,**tkwargs)

        # Evaluate the samples using surrogate models of objective and constraints
        state.Y_model.eval()
        with torch.no_grad():
            posterior = state.Y_model.posterior(samples)
            samples_yy = posterior.mean.squeeze()

        state.C_model.eval()
        with torch.no_grad():
            posterior = state.C_model.posterior(samples)
            samples_cc = posterior.mean

        # Normalize constraint outputs and reduce to scalar
        samples_cc = samples_cc / (torch.abs(samples_cc).max(dim=0).values + eps)
        samples_cc = torch.max(samples_cc, dim=1).values

        # Determine number of points to select for TR adaptation
        n_samples_tr = max(int(n_samples * percentage), 4)
        # Select top samples based on feasibility and objective
        if torch.any(samples_cc < 0):
            # Some points are feasible
            feasible_idx = torch.where(samples_cc <= 0)[0]
            infeasible_idx = torch.where(samples_cc > 0)[0]

            feasible_vals = samples_yy[feasible_idx]
            infeasible_vals = samples_cc[infeasible_idx]

            feasible_sorted_ids = torch.argsort(feasible_vals)  # ascending (smaller is better)
            infeasible_sorted_ids = torch.argsort(infeasible_vals)

            chosen_idx = torch.cat([
                feasible_idx[feasible_sorted_ids],
                infeasible_idx[infeasible_sorted_ids]
            ])[:n_samples_tr]
        else:
            # No feasible points; select points with smallest constraint violation
            if n_samples_tr > len(samples_cc):
                n_samples_tr = len(samples_cc)

            top_vals, top_idx = torch.topk(samples_cc, n_samples_tr, largest=False)
            chosen_idx = top_idx

        chosen_samples = samples[chosen_idx]

        # Compute PCA for chosen samples
        mu, R, eigvals = _compute_pca_from_samples(chosen_samples, eps=eps)

        # Fit local GP per trust region to adapt its radius
        local_gp = get_fitted_model(chosen_samples, samples_yy[chosen_idx].unsqueeze(-1), state.dim)
        state.local_Y_gps[ind] = local_gp

        # Extract lengthscales
        gp_ls = _get_gp_lengthscales(local_gp)
        if gp_ls is None:
            print("gp_ls is NONE")
            gp_ls = torch.ones(d, **tkwargs)

        # Convert PCA eigenvalues into axis lengths and scale by GP lengthscales
        axis_lengths = _make_radii_from_eigvals(eigvals, base_scale=base_scale,
                                                min_length=min_axis_length,
                                                max_length=max_axis_length)
        axis_lengths = axis_lengths.to(dtype=mu.dtype, device=mu.device)
        adaptive_radii = axis_lengths * gp_ls  

        # Update trust region state
        state.tr_R[ind] = R
        state.tr_center[ind] = mu
        state.tr_radii[ind] = adaptive_radii     

    return state