from typing import Optional

import torch

from src.gaussian_noise import GaussianDiffusion, BetaSchedule, ModelMeanType, ModelVarType


class Sampler:

    def __init__(self, model, sample_shape: tuple[int, ...], diffusion: Optional[GaussianDiffusion] = None,
                 ddim: bool = False,
                 device: Optional[torch.device] = None):
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.device = device
        self.sample_shape = sample_shape
        self.ddim = ddim
        self.model = model.to(self.device)
        if diffusion is None:
            diffusion = GaussianDiffusion(1000, BetaSchedule.COSINE, ModelMeanType.EPSILON,
                                          ModelVarType.FIXED_SMALL,
                                          device=self.device)
        self.diffusion = diffusion

    @torch.no_grad()
    def sample(self, num_samples: int, model_kwargs: Optional[dict] = None) -> torch.Tensor:
        if model_kwargs is None:
            model_kwargs = {}

        sample_fn = self.diffusion.ddim_sample_loop if self.ddim else self.diffusion.p_sample_loop
        shape = (num_samples, *self.sample_shape)
        samples, _ = sample_fn(self.model, shape, model_kwargs=model_kwargs)
        return samples