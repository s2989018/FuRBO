import torch
import numpy as np
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

# ===============================================================
# Helper functions for MC objective
# ===============================================================
def obj_from_samples(samples, X=None, **kwargs):
    """Return objective values from MC samples (first column)"""
    return samples[..., 0]

def make_con(j):
    """Return constraint j from MC samples (starting at 1)"""
    return lambda samples, j=j, X=None, **kwargs: samples[..., j + 1]

# ===============================================================
# Custom qLogExpectedImprovement (stable Log-EI)
# ===============================================================
class qLogExpectedImprovement(MCAcquisitionFunction):
    def __init__(self, model, best_f, objective=None, sampler=None):
        def _make_sampler(n=256):
            try:
                return SobolQMCNormalSampler(num_samples=n)
            except TypeError:
                pass
            try:
                return SobolQMCNormalSampler(sample_shape=torch.Size([n]))
            except TypeError:
                pass
            try:
                return SobolQMCNormalSampler(sample_shape=(n,))
            except Exception:
                pass
            # Fallback
            try:
                from botorch.sampling import IIDNormalSampler
                return IIDNormalSampler(num_samples=n)
            except Exception:
                raise RuntimeError("Cannot construct MC sampler; check BoTorch version")

        if sampler is None:
            sampler = _make_sampler(256)
        super().__init__(model=model, sampler=sampler, objective=objective)
        self.best_f = best_f

    def forward(self, X):
        samples = self.get_samples(X)
        obj = self.objective(samples=samples, X=X)

        improvement = (self.best_f - obj).clamp_min(0)
        if improvement.dim() >= 2:
            per_sample = improvement.mean(dim=-1)
            ei = per_sample.mean(dim=0)
        else:
            ei = improvement
        return torch.log(ei + 1e-8)

    def get_samples(self, X):
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=next(self.model.parameters()).dtype)
        posterior = self.model.posterior(X)
        try:
            return self.sampler(posterior)
        except Exception:
            n = getattr(self.sampler, 'num_samples', 256)
            try:
                return posterior.rsample(torch.Size([n]))
            except Exception:
                raise RuntimeError('Unable to draw MC samples from posterior')

# ===============================================================
# Safe GP fit
# ===============================================================
def safe_fit_gp(gp, name="gp"):
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    try:
        fit_gpytorch_mll(mll)
    except Exception as e:
        print(f"[WARN] GP fit failed for {name}, continuing. Error: {e}")

# ===============================================================
# Main function
# ===============================================================
def run_botorch_qlogei(p, seed=0, budget=None):
    """
    Run qLogEI optimization on a COCO problem object `p`.

    p: COCO problem object from a suite
    seed: random seed
    budget: number of evaluations (default: 10 * p.dimension)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    dim = p.dimension
    if budget is None:
        budget = 10 * dim

    # -------------------------------
    # Initial random design
    # -------------------------------
    n_init = min(10, 5 * dim)
    X = np.random.uniform(p.lower_bounds, p.upper_bounds, size=(n_init, dim))

    y_obj = []
    y_con = []

    for x in X:
        y_obj.append(p(x))
        y_con.append(p.constraint(x))

    X_t = torch.tensor(X, dtype=torch.double)
    y_obj = torch.tensor(y_obj, dtype=torch.double).unsqueeze(-1)
    y_con = torch.tensor(y_con, dtype=torch.double)
    if y_con.dim() == 1:
        y_con = y_con.unsqueeze(1)

    # -------------------------------
    # Fit initial GP models
    # -------------------------------
    gp_obj = SingleTaskGP(X_t, y_obj, outcome_transform=Standardize(m=1))
    safe_fit_gp(gp_obj, "objective")

    gp_cons = []
    for j in range(y_con.shape[1]):
        gpj = SingleTaskGP(X_t, y_con[:, j:j+1], outcome_transform=Standardize(m=1))
        safe_fit_gp(gpj, f"constraint_{j}")
        gp_cons.append(gpj)

    model = ModelListGP(gp_obj, *gp_cons)

    # -------------------------------
    # Constrained MC Objective
    # -------------------------------
    constraints = [make_con(j) for j in range(y_con.shape[1])]
    objective = ConstrainedMCObjective(objective=obj_from_samples, constraints=constraints)

    # -------------------------------
    # BO Loop
    # -------------------------------
    eval_count = n_init
    bounds = torch.tensor([p.lower_bounds, p.upper_bounds], dtype=torch.double)

    while eval_count < budget:
        feas = torch.all(y_con <= 0, dim=1)
        best_f = y_obj[feas].min() if torch.any(feas) else y_obj.min()

        acq = qLogExpectedImprovement(model=model, best_f=best_f, objective=objective)

        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=64
        )

        candidate = candidate.reshape(1, -1)
        x_new = candidate.detach().cpu().numpy()[0]

        f_new = p(x_new)
        g_new = p.constraint(x_new)

        # Add new point
        X_t = torch.cat([X_t, candidate], dim=0)
        y_obj = torch.cat([y_obj, torch.tensor([[f_new]], dtype=torch.double)], dim=0)

        g_new_t = torch.tensor([g_new], dtype=torch.double)
        if g_new_t.dim() == 1:
            g_new_t = g_new_t.unsqueeze(0)
        y_con = torch.cat([y_con, g_new_t], dim=0)

        # Refit GPs
        gp_obj = SingleTaskGP(X_t, y_obj, outcome_transform=Standardize(m=1))
        safe_fit_gp(gp_obj, "objective")

        gp_cons = []
        for j in range(y_con.shape[1]):
            gpj = SingleTaskGP(X_t, y_con[:, j:j+1], outcome_transform=Standardize(m=1))
            safe_fit_gp(gpj, f"constraint_{j}")
            gp_cons.append(gpj)

        model = ModelListGP(gp_obj, *gp_cons)
        eval_count += 1

    return y_obj, y_con, X_t
