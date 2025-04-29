from typing import Optional, Callable

import torch
from torch._inductor.scheduler import Scheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset

from src.gaussian_noise import GaussianDiffusion, BetaSchedule, ModelMeanType, ModelVarType
from src.rplan.dataset import RPlanImageDataset
from src.rplan.types import MaskPlan, ImagePlan, ImagePlanCollated, RoomType
from src.rplan_masks.karras.cfg import CFGUnet
from src.rplan_masks.karras.denoise import GithubUnet
from src.schedule_sampler import ScheduleSampler, LossSampler
from src.utils.diffusion_trainer import DiffusionTrainer, DiffusionStepOutput

def custom_eps_loss(output: torch.Tensor, eps: torch.Tensor, x_0: torch.Tensor, x_t: torch.Tensor, **model_kwargs):
    room_types = x_0[:, 0, :, :]
    walls = x_0[:, 1, :, :]
    doors = x_0[:, 2, :, :]
    wall_weights = torch.where(walls > 0, 10, 1)
    door_weights = torch.where(doors > -1, 10, 1)
    # room_type_frequencies = RoomType.frequencies()
    # max_count = max(room_type_frequencies.values())
    # normalized_frequencies = {room_type: max_count / count for room_type, count in room_type_frequencies.items()}
    room_type_loss = torch.nn.functional.mse_loss(output[:, 0, :, :], eps[:, 0, :, :], reduction='none').mean(dim=(1, 2))
    wall_loss = torch.nn.functional.mse_loss(output[:, 1, :, :], eps[:, 1, :, :], reduction='none') * wall_weights
    wall_loss = wall_loss.mean(dim=(1, 2))
    door_loss = torch.nn.functional.mse_loss(output[:, 2, :, :], eps[:, 2, :, :], reduction='none') * door_weights
    door_loss = door_loss.mean(dim=(1, 2))
    loss = wall_loss + door_loss + room_type_loss
    return loss


class CfgBubbleTrainer(DiffusionTrainer[ImagePlan, ImagePlanCollated]):

    def __init__(self, batch_size: int, lr: float, mask_size: int = 64, epochs: int = 30,
                 model: Optional[torch.nn.Module] = None, dataset: Optional[Dataset] = None,
                 diffusion: Optional[GaussianDiffusion] = None,
                 timestep_sampler: Optional[ScheduleSampler] = None,
                 device: Optional[torch.device] = None,
                 collate_fn: Optional[Callable[[list[ImagePlan]], ImagePlanCollated]] = None,
                 checkpoint_path: Optional[str] = None, model_dict: Optional[dict] = None,
                 save_interval: Optional[int] = 500,
                 num_workers: int = 8, scheduler: Optional[LRScheduler] = None, optimizer: Optional[Optimizer] = None):
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
        if model is None:
            model = CFGUnet(dim=mask_size, init_dim=32, channels=5, out_dim=3, cond_drop_prob=0.15, use_bubbles=True).to(device)
        if dataset is None:
            dataset = RPlanImageDataset('data/rplan', load_base_rplan=True, random_flip=True, random_scale=0.6,
                                        no_doors=False,
                                        no_front_door=False,
                                        random_translate=True,
                                        random_rotate=True, shuffle_rooms=True, max_workers=num_workers,
                                        mask_size=(mask_size, mask_size))
        if diffusion is None:
            diffusion = GaussianDiffusion(1000, BetaSchedule.COSINE, ModelMeanType.EPSILON,
                                          ModelVarType.FIXED_SMALL,
                                          device=device)
        if timestep_sampler is None:
            timestep_sampler = LossSampler(diffusion)
        if collate_fn is None:
            collate_fn = ImagePlan.collate
        if checkpoint_path is None:
            checkpoint_path = 'checkpoints_unet_bubbles'
        super().__init__(model, dataset, diffusion, timestep_sampler, epochs, batch_size, lr, device, collate_fn,
                         lambda batch: batch[0].shape[0],
                         checkpoint_path, model_dict,
                         save_interval, num_workers, scheduler, optimizer)
        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 10, T_mult=2)

    def diffusion_step(self, batch: ImagePlanCollated) -> DiffusionStepOutput:
        images, walls, doors, room_types, bubbles, masks = batch
        images, walls, doors, masks = images.to(self.device), walls.to(self.device), doors.to(
            self.device), masks.to(self.device)
        if bubbles is not None:
            bubbles = bubbles.to(self.device)
            # bubbles = bubbles[:, 0, :, :].unsqueeze(1)
        if room_types is not None:
            room_types = room_types.to(self.device)
        masks = masks.float()
        x = torch.cat([images, walls, doors], dim=1)
        model_kwargs = {
            'masks': masks,
            'room_types': room_types,
            'bubbles': bubbles,
            'use_bubbles': True,
            'custom_eps_loss': custom_eps_loss
        }

        return {
            'x': x,
            'model_kwargs': model_kwargs
        }
