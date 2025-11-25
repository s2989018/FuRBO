##########
# FuRBO sampling strategies (rotated PCA-based)

##########

from abc import ABC, abstractmethod
from botorch.acquisition.objective import IdentityMCObjective, MCAcquisitionObjective, PosteriorTransform
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import MultiTaskGP
from botorch.generation.utils import _flip_sub_unique
from torch.nn import Module
from torch import Tensor
from typing import Optional, Union
import torch


def get_initial_points_sobol(FuRBO, **tkwargs):
    X_init = FuRBO.sobol.draw(n=FuRBO.n_init).to(**tkwargs)
    return X_init

# Generate initial points inside rotated PCA-based trust regions.
def get_initial_points_rotated_TR(state, n_init=None, **tkwargs):

    if n_init is None:
        n_init = state.n_init

    X_init = torch.empty((n_init, state.dim), **tkwargs)
    
    for i in range(state.tr_number):
        # Determine number of points per TR
        points_per_TR = n_init // state.tr_number
        if i == state.tr_number - 1:
            points_per_TR = n_init - (points_per_TR * i)

        # Sample in unit cube and scale by TR radius
        u = torch.rand(points_per_TR, state.dim, **tkwargs) * 2 - 1
        u = u / u.norm(dim=1, keepdim=True)  
        u = u * state.tr_radii[i]          

        # Rotate via PCA and translate to TR center
        X_cand = u @ state.tr_R[i].T + state.tr_center[i]

        X_cand = X_cand.clamp(0.0, 1.0)

        start = i * (n_init // state.tr_number)
        end = start + points_per_TR
        X_init[start:end, :] = X_cand

    return X_init

##########
# Candidate generation (Thompson sampling) in rotated PCA-based TRs

def generate_batch_thompson_sampling_rotated_TR(state, n_candidates, **tkwargs):

    assert state.X.min() >= 0.0 and state.X.max() <= 1.0 and torch.all(torch.isfinite(state.Y))
    tr_number = state.tr_number
    batch_size = state.batch_size
    batch_per_TR = batch_size // tr_number

    X_next = torch.empty((batch_size, state.dim), **tkwargs)

  
    for i in range(state.tr_number):
        if i == tr_number - 1:
            points_this_TR = batch_size - (batch_per_TR * i)
        else:
            points_this_TR = batch_per_TR

        pert = torch.rand(n_candidates, state.dim, **tkwargs) * 2 - 1
        pert = pert / pert.norm(dim=1, keepdim=True)
        pert = pert * state.tr_radii[i]  

        # Rotate and translate to TR center
        X_cand = pert @ state.tr_R[i].T + state.tr_center[i]

        # Create a perturbation mask
        prob_perturb = min(20.0 / state.dim, 0.5)
        print(prob_perturb)
        mask = torch.rand(n_candidates, state.dim, **tkwargs) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]

        # Create candidate points from the perturbations and the mask
        if state.tr_best_X[i] is not None:
            mask[ind, torch.randint(0, state.dim, size=(len(ind),), device=tkwargs['device'])] = 1
            X_cand[mask] = state.tr_best_X[i].expand(n_candidates, state.dim)[mask]

        # Clip to [0,1]
        X_cand = X_cand.clamp(0.0, 1.0)

  
        sampler = ConstrainedMaxPosteriorSampling(
            model=state.Y_model, constraint_model=state.C_model, replacement=False
        )
        with torch.no_grad():
            start = i * batch_per_TR
            end = start + points_this_TR
            X_next[start:end, :] = sampler(X_cand, num_samples=points_this_TR)

        with torch.no_grad():
            posterior = state.Y_model.posterior(X_next[start:end, :])
            var = posterior.variance

    return X_next

##########
# Sampling strategy classes

class SamplingStrategy(Module, ABC):
    """Abstract base class for sampling-based generation strategies."""

    @abstractmethod
    def forward(self, X: Tensor, num_samples: int = 1) -> Tensor:
        pass

class MaxPosteriorSampling(SamplingStrategy):
    """Sample points according to their max posterior values."""

    def __init__(self, model: Model, objective: Optional[MCAcquisitionObjective]=None,
                 posterior_transform: Optional[PosteriorTransform]=None, replacement: bool=True):
        super().__init__()
        self.model = model
        self.objective = IdentityMCObjective() if objective is None else objective
        self.posterior_transform = posterior_transform
        self.replacement = replacement

    def forward(self, X: Tensor, num_samples: int = 1, observation_noise: bool = False) -> Tensor:
        posterior = self.model.posterior(
            X, observation_noise=observation_noise, posterior_transform=self.posterior_transform
        )
        samples = posterior.rsample(sample_shape=torch.Size([num_samples]))
        return self.maximize_samples(X, samples, num_samples)

    def maximize_samples(self, X: Tensor, samples: Tensor, num_samples: int = 1):
        obj = self.objective(samples, X=X)
        if self.replacement:
            idcs = torch.argmax(obj, dim=-1)
        else:
            _, idcs_full = torch.topk(obj, num_samples, dim=-1)
            ridx, cindx = torch.tril_indices(num_samples, num_samples)
            sub_idcs = idcs_full[ridx, ..., cindx]
            if sub_idcs.ndim == 1:
                idcs = _flip_sub_unique(sub_idcs, num_samples)
            elif sub_idcs.ndim == 2:
                n_b = sub_idcs.size(-1)
                idcs = torch.stack([_flip_sub_unique(sub_idcs[:, i], num_samples) for i in range(n_b)], dim=-1)
            else:
                raise NotImplementedError("MaxPosteriorSampling without replacement for >1 batch dim.")
        if idcs.ndim > 1:
            idcs = idcs.permute(*range(1, idcs.ndim), 0)
        idcs = idcs.unsqueeze(-1).expand(*idcs.shape, X.size(-1))
        Xe = X.expand(*obj.shape[1:], X.size(-1))
        return torch.gather(Xe, -2, idcs)

class ConstrainedMaxPosteriorSampling(MaxPosteriorSampling):
    """Constrained max posterior sampling."""

    def __init__(self, model: Model, constraint_model: Union[ModelListGP, MultiTaskGP],
                 objective: Optional[MCAcquisitionObjective]=None,
                 posterior_transform: Optional[PosteriorTransform]=None,
                 replacement: bool=True):
        if objective is not None:
            raise NotImplementedError("`objective` not supported for ConstrainedMaxPosteriorSampling.")
        super().__init__(model=model, objective=objective, posterior_transform=posterior_transform, replacement=replacement)
        self.constraint_model = constraint_model

    def _convert_samples_to_scores(self, Y_samples, C_samples) -> Tensor:
        is_feasible = (C_samples <= 0).all(dim=-1)
        has_feasible_candidate = is_feasible.any(dim=-1)
        scores = Y_samples.clone()
        scores[~is_feasible] = -float("inf")
        if not has_feasible_candidate.all():
            total_violation = C_samples[~has_feasible_candidate].clamp(min=0).sum(dim=-1, keepdim=True)
            scores[~has_feasible_candidate] = -total_violation
        return scores

    def forward(self, X: Tensor, num_samples: int = 1, observation_noise: bool = False) -> Tensor:
        posterior = self.model.posterior(X=X, observation_noise=observation_noise,
                                         posterior_transform=self.posterior_transform)
        Y_samples = posterior.rsample(sample_shape=torch.Size([num_samples]))
        C_tmp = []
        for c in self.constraint_model.models:
            c_posterior = c.posterior(
                X=X, observation_noise=observation_noise
                )
            C_tmp.append(c_posterior.rsample(sample_shape=torch.Size([num_samples])))

        C_samples = torch.cat(C_tmp, dim=2)
        scores = self._convert_samples_to_scores(Y_samples=Y_samples, C_samples=C_samples)
        return self.maximize_samples(X=X, samples=scores, num_samples=num_samples)
