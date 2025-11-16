# FuRBO sampling strategies
# 
##########
# Imports
import torch

from botorch.generation.sampling import ConstrainedMaxPosteriorSampling

# function to sample in ball
def sample_in_ball(center, r, n_points, lb, ub, **tkwargs):
    """Sample uniformly in a d-dimensional ball"""
    dim = center.shape[-1]
    # draw n_points, d dimensional vectors 
    g = torch.randn(n_points, dim, **tkwargs)
    g_norm = g.norm(dim=1, keepdim=True).clamp_min(1e-12) # length of each vector 
    #every row lies on unit sphere
    directions = g / g_norm
    # sample unfirom random samples again for radius
    u = torch.rand(n_points, 1, **tkwargs)
    #ensure uniform sampling
    rho = u.pow(1.0 / dim)
    X = center + r * rho * directions
    # clip to global domain
    X = torch.max(torch.min(X, ub), lb)
    return X

# Utility functions
def get_initial_points_sobol(state,
                             **tkwargs):
    '''Function to generate the initial experimental design'''
    X_init = state.sobol.draw(n=state.n_init).to(**tkwargs)
    return X_init

def generate_batch_thompson_sampling(state,
                                     n_candidates,
                                     tr_shape = "hypershpere",
                                     **tkwargs):
    '''Function to find net candidate optimum'''
    assert state.X.min() >= 0.0 and state.X.max() <= 1.0 and torch.all(torch.isfinite(state.Y))

    # Initialize tensor with samples to evaluate
    X_next = torch.ones((state.batch_size*state.tr_number, state.dim), **tkwargs)
    
    # Iterate over the several trust regions
    for i in range(state.tr_number):


        if tr_shape == "hypershpere":
            lb = torch.zeros(state.dim, **tkwargs)
            ub = torch.ones(state.dim, **tkwargs)
            # Center and radius of spherical trust region
            center = state.best_X[i]
            r = state.radius  # radius

            # Generate candidate points uniformly in the ball 
            X_cand = sample_in_ball(center=center, r=r, n_points=n_candidates, lb=lb, ub=ub, **tkwargs)
        else:

            tr_lb = state.tr_lb[i]
            tr_ub = state.tr_ub[i]

            # Thompson Sampling w/ Constraints (like SCBO)
            pert = state.sobol.draw(n_candidates).to(**tkwargs)
            pert = tr_lb + (tr_ub - tr_lb) * pert

            # Create a perturbation mask
            prob_perturb = min(20.0 / state.dim, 1.0)
            mask = torch.rand(n_candidates, state.dim, **tkwargs) <= prob_perturb
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            mask[ind, torch.randint(0, state.dim - 1, size=(len(ind),), device=tkwargs['device'])] = 1

            # Create candidate points from the perturbations and the mask
            X_cand = state.best_X[i].expand(n_candidates, state.dim).clone()
            X_cand[mask] = pert[mask]
        
        # Sample on the candidate points using Constrained Max Posterior Sampling
        constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(
            model=state.Y_model, constraint_model=state.C_model, replacement=False
            )
        with torch.no_grad():
            X_next[i*state.batch_size:i*state.batch_size+state.batch_size, :] = constrained_thompson_sampling(X_cand, num_samples=state.batch_size)
        
    return X_next

