import torch
import numpy as np
import cocoex
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

def evaluate_objective(x, coco_fun, coco_instance, dim=None):

    inst_id = f"i{coco_instance+1:02d}"
    fun_id = f"f{coco_fun:03d}"

    suite = cocoex.Suite("bbob-constrained", "", "")
    for p in suite:
        parts = p.id.split('_')
        if len(parts) >= 4 and parts[1] == fun_id and parts[2] == inst_id:
            if dim is None or parts[3] == f"d{int(dim):02d}":
                arr = np.asarray(x, dtype=np.float64)
                return float(p(arr))

    raise ValueError(f"COCO problem f{coco_fun} i{coco_instance+1} not found in suite")


def evaluate_constraints(x, coco_fun, coco_instance, dim=None):

    inst_id = f"i{coco_instance+1:02d}"
    fun_id = f"f{coco_fun:03d}"

    suite = cocoex.Suite("bbob-constrained", "", "")
    for p in suite:
        parts = p.id.split('_')
        if len(parts) >= 4 and parts[1] == fun_id and parts[2] == inst_id:
            if dim is None or parts[3] == f"d{int(dim):02d}":
                arr = np.asarray(x, dtype=np.float64)
                c = p.constraint(arr)
                return np.asarray(c, dtype=np.float64)

    raise ValueError(f"COCO problem f{coco_fun} i{coco_instance+1} not found in suite")


class qLogExpectedImprovement(MCAcquisitionFunction):
    """Monte-Carlo Log Expected Improvement acquisition with robust sampler handling."""

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
            from botorch.sampling import IIDNormalSampler

            return IIDNormalSampler(num_samples=n)

        sampler = sampler or _make_sampler(256)
        super().__init__(model=model, sampler=sampler, objective=objective)
        self.best_f = best_f

    def forward(self, X):
        samples = self.get_samples(X)
        obj = self.objective(samples=samples, X=X)

        if obj.dim() >= 2:
            improvement = (self.best_f - obj).clamp_min(0)
            if improvement.dim() >= 2:
                per_sample = improvement.mean(dim=-1)
            else:
                per_sample = improvement
            ei = per_sample.mean(dim=0)
        else:
            ei = (self.best_f - obj).clamp_min(0)

        return torch.log(ei + 1e-8)

    def get_samples(self, X):
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=next(self.model.parameters()).dtype)

        try:
            posterior = self.model.posterior(X)
        except Exception:
            posterior = self.model.posterior(X.unsqueeze(0))

        # Try sampler call patterns, fallback to posterior.rsample
        if hasattr(self, 'sampler') and self.sampler is not None:
            try:
                return self.sampler(posterior)
            except Exception:
                pass

            n = getattr(self.sampler, 'num_samples', None) or getattr(self.sampler, '_num_samples', None)
            if n is None:
                sh = getattr(self.sampler, 'sample_shape', None)
                if isinstance(sh, (tuple, list, torch.Size)) and len(sh) > 0:
                    try:
                        n = int(sh[0])
                    except Exception:
                        n = None
            n = n or 256

            for call in (lambda p: self.sampler(p, num_samples=n), lambda p: self.sampler(p, sample_shape=torch.Size([n]))):
                try:
                    return call(posterior)
                except Exception:
                    pass

        try:
            return posterior.rsample(torch.Size([256]))
        except Exception:
            raise RuntimeError('Unable to draw MC samples from the model posterior')


# ===============================================================
#  Safe GP fit (avoid crashes)
# ===============================================================
def safe_fit_gp(gp, name="gp"):
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    try:
        fit_gpytorch_mll(mll)
    except Exception as e:
        print(f"[WARN] GP fit failed for {name}, continuing. Error: {e}")


# ===============================================================
#               Baseline qLogEI for COCO constrained problem
# ===============================================================
def run_botorch_qlogei(dim, budget, coco_fun, coco_instance, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_init = min(10, 5 * dim)
    X = np.random.uniform(-5, 5, size=(n_init, dim))

    obj_vals = []
    con_vals = []
    for x in X:
        obj_vals.append(evaluate_objective(x, coco_fun, coco_instance, dim))
        con_vals.append(evaluate_constraints(x, coco_fun, coco_instance, dim))

    X_t = torch.tensor(X, dtype=torch.double)
    y_obj = torch.tensor(obj_vals, dtype=torch.double).unsqueeze(-1)
    y_con = torch.tensor(con_vals, dtype=torch.double)
    if y_con.dim() == 1:
        y_con = y_con.unsqueeze(1)

    gp_obj = SingleTaskGP(X_t, y_obj, outcome_transform=Standardize(m=1))
    safe_fit_gp(gp_obj, "objective")

    gp_cons = []
    for j in range(y_con.shape[1]):
        gpj = SingleTaskGP(X_t, y_con[:, j:j+1], outcome_transform=Standardize(m=1))
        safe_fit_gp(gpj, f"constraint_{j}")
        gp_cons.append(gpj)

    model = ModelListGP(gp_obj, *gp_cons)

    def obj_from_samples(samples, X=None, **kwargs):
        return samples[..., 0]

    def make_con(j):
        return lambda samples, j=j, X=None, **kwargs: samples[..., j + 1]

    constraints = [make_con(j) for j in range(y_con.shape[1])]
    objective = ConstrainedMCObjective(objective=obj_from_samples, constraints=constraints)

    eval_count = n_init
    bounds = torch.tensor([[-5.0] * dim, [5.0] * dim], dtype=torch.double)

    while eval_count < budget:
        feas = torch.all(y_con <= 0, dim=1)
        best_f = y_obj[feas].min() if torch.any(feas) else y_obj.min()

        acq = qLogExpectedImprovement(model=model, best_f=best_f, objective=objective)
        candidate, _ = optimize_acqf(acq_function=acq, bounds=bounds, q=1, num_restarts=5, raw_samples=64)

        candidate = candidate.reshape(1, -1)
        x_new = candidate.detach().cpu().numpy()[0]

        f_new = evaluate_objective(x_new, coco_fun, coco_instance, dim)
        g_new = evaluate_constraints(x_new, coco_fun, coco_instance, dim)

        X_t = torch.cat([X_t, candidate], dim=0)
        y_obj = torch.cat([y_obj, torch.tensor([[f_new]], dtype=torch.double)], dim=0)

        g_new_t = torch.tensor([g_new], dtype=torch.double)
        if g_new_t.dim() == 1:
            g_new_t = g_new_t.unsqueeze(0)
        y_con = torch.cat([y_con, g_new_t], dim=0)

        gp_obj = SingleTaskGP(X_t, y_obj, outcome_transform=Standardize(m=1))
        safe_fit_gp(gp_obj, "objective")

        gp_cons = []
        for j in range(y_con.shape[1]):
            gpj = SingleTaskGP(X_t, y_con[:, j:j+1], outcome_transform=Standardize(m=1))
            safe_fit_gp(gpj, f"constraint_{j}")
            gp_cons.append(gpj)

        model = ModelListGP(gp_obj, *gp_cons)
        eval_count += 1

    return y_obj, y_con, X_t.numpy()
