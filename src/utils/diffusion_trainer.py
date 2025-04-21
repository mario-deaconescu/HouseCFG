from abc import ABC, abstractmethod
from typing import Optional, Callable, TypeVar, TypedDict, Generic, final

import torch
from torch._inductor.scheduler import Scheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset

from src.gaussian_noise import GaussianDiffusion
from src.schedule_sampler import ScheduleSampler, LossSampler
from src.utils.trainer import Trainer

T = TypeVar('T')
T_BATCH = TypeVar('T_BATCH')

class DiffusionStepOutput(TypedDict):
    x: torch.Tensor
    model_kwargs: dict

class DiffusionTrainer(Generic[T, T_BATCH], Trainer[T, T_BATCH]):

    def __init__(self, model: torch.nn.Module, dataset: Dataset, diffusion: GaussianDiffusion,
                 timestep_sampler: ScheduleSampler,
                 epochs: int, batch_size: int, lr: float,
                 device: Optional[torch.device] = None,
                 collate_fn: Optional[Callable[[list[T]], T_BATCH]] = None,
                 get_batch_size: Optional[Callable[[T_BATCH], int]] = None,
                 checkpoint_path: Optional[str] = None, model_dict: Optional[dict] = None,
                 log_interval: Optional[int] = None,
                 num_workers: int = 8, scheduler: Optional[LRScheduler] = None, optimizer: Optional[Optimizer] = None):
        super().__init__(model, dataset, epochs, batch_size, lr, device, collate_fn, checkpoint_path, model_dict,
                         log_interval, num_workers, scheduler, optimizer)
        self.get_batch_size = get_batch_size
        self.diffusion = diffusion
        self.timestep_sampler = timestep_sampler

    @abstractmethod
    def diffusion_step(self, batch: T_BATCH) -> DiffusionStepOutput:
        pass

    @final
    def step(self, batch: T_BATCH) -> torch.Tensor:
        if self.get_batch_size is not None:
            batch_size = self.get_batch_size(batch)
        else:
            batch_size = batch.shape[0].shape[0]
        t, sampler_weights = self.timestep_sampler.sample(batch_size, device=self.device)
        diffusion_step_output = self.diffusion_step(batch)
        x_noisy = diffusion_step_output['x']
        model_kwargs = diffusion_step_output['model_kwargs']
        losses = self.diffusion.training_losses(self.model, x_noisy, t, model_kwargs)['loss']

        if isinstance(self.timestep_sampler, LossSampler):
            self.timestep_sampler.update_with_all_losses(t, losses.detach())
        loss = (losses * sampler_weights).mean()

        return loss

