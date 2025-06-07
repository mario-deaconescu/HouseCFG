from typing import Optional, Callable

import numpy as np
import torch
from torch._inductor.scheduler import Scheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset

from src.rplan.area_constants import MEAN_AREA_PER_ROOM_TYPE, TOTAL_MEAN_AREA
from src.gaussian_noise import GaussianDiffusion, BetaSchedule, ModelMeanType, ModelVarType
from src.rplan.dataset import RPlanImageDataset
from src.rplan.types import MaskPlan, ImagePlan, ImagePlanCollated, RoomType
from src.rplan_masks.karras.cfg import CFGUnet
from src.schedule_sampler import ScheduleSampler, LossSampler
from src.utils.diffusion_trainer import DiffusionTrainer, DiffusionStepOutput

def custom_eps_loss(output: torch.Tensor, eps: torch.Tensor, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, **model_kwargs):
    mse = torch.nn.functional.mse_loss(output, eps, reduction='none').mean(dim=(2, 3))
    mse = mse.mean(dim=1)
    return mse





class CfgTrainer(DiffusionTrainer[ImagePlan, ImagePlanCollated]):

    def __init__(self, epochs: int, batch_size: int, lr: float, mask_size: int = 64,
                 model: Optional[torch.nn.Module] = None, dataset: Optional[Dataset] = None,
                 diffusion: Optional[GaussianDiffusion] = None,
                 timestep_sampler: Optional[ScheduleSampler] = None,
                 device: Optional[torch.device] = None,
                 collate_fn: Optional[Callable[[list[ImagePlan]], ImagePlanCollated]] = None,
                 checkpoint_path: Optional[str] = None, model_dict: Optional[dict] = None,
                 save_interval: Optional[int] = 500, constraint_loss: Optional[float] = None,
                 num_workers: int = 8, scheduler: Optional[LRScheduler] = None, optimizer: Optional[Optimizer] = None):
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
        if model is None:
            model = CFGUnet(dim=mask_size, channels=4, out_dim=3, cond_drop_prob=0.2).to(device)
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
            checkpoint_path = 'checkpoints_unet'

        self.constraint_loss = constraint_loss
        if constraint_loss is not None:
            mean_areas: list[Optional[float]] = [None for _ in range(RoomType.restricted_length())]
            for i in range(RoomType.restricted_length()):
                room_type = RoomType.from_index_restricted(i)
                mean_areas[i] = MEAN_AREA_PER_ROOM_TYPE[room_type]

            for area in mean_areas:
                assert area is not None, 'None area found for room type in area data'

            self.mean_areas = torch.tensor(mean_areas, device=device, dtype=torch.float32)


        super().__init__(model, dataset, diffusion, timestep_sampler, epochs, batch_size, lr, device, collate_fn,
                         lambda batch: batch[0].shape[0],
                         checkpoint_path, model_dict,
                         save_interval, num_workers, scheduler, optimizer)

    def custom_eps_loss_with_room_types(self, output: torch.Tensor, eps: torch.Tensor, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor,
                 **model_kwargs):
        mse = torch.nn.functional.mse_loss(output, eps, reduction='none').mean(dim=(2, 3))
        mse = mse.mean(dim=1)

        constraints = model_kwargs.get('room_types', None)  # [B, num_room_types]
        if constraints is None or self.constraint_loss is None:
            return mse

        pred_x_0 = self.diffusion.predict_x_0_from_eps(x_t, t, eps)
        plan_areas = (pred_x_0[:, 0, :, :] > -0.9).float().sum(dim=(1, 2))
        pred_x_0_expanded = pred_x_0[:, 0, :, :].unsqueeze(1).expand(1, RoomType.restricted_length() + 1, -1, -1)
        print('here')

        room_type_centers = np.linspace(-1, 1, RoomType.restricted_length() + 1)
        sigma = 0.2 / 15
        room_type_centers = torch.tensor(room_type_centers, device=pred_x_0.device, dtype=pred_x_0.dtype)
        room_type_centers = room_type_centers.view(1, -1, 1, 1)
        room_type_centers = room_type_centers.expand(pred_x_0.shape[0], -1, pred_x_0.shape[2], pred_x_0.shape[3])
        print(room_type_centers.shape)
        room_type_probs = torch.exp(
            -((pred_x_0_expanded - room_type_centers) ** 2) / (2 * sigma ** 2))
        print(room_type_probs.shape)
        room_type_probs = room_type_probs / (room_type_probs.sum(dim=1,
                                                                 keepdim=True) + 1e-8)  # Shape: (batch_size, num_room_types, height, width)

        room_type_areas = room_type_probs.sum(dim=(2, 3))[:, 1:] / plan_areas.view(-1, 1)  # Shape: (batch_size, num_room_types)

        target_room_type_areas = constraints * self.mean_areas.view(1, -1) / TOTAL_MEAN_AREA

        area_loss = torch.nn.functional.mse_loss(room_type_areas, target_room_type_areas, reduction='none').mean(
            dim=1)  # Shape: (batch_size,)

        print(area_loss, room_type_areas, target_room_type_areas)

        return mse + area_loss * self.constraint_loss


    def diffusion_step(self, batch: ImagePlanCollated) -> DiffusionStepOutput:
        images, walls, doors, room_types, bubbles, masks = batch
        images, walls, doors, masks = images.to(self.device), walls.to(self.device), doors.to(
            self.device), masks.to(self.device)
        if bubbles is not None:
            bubbles = bubbles.to(self.device)
        if room_types is not None:
            room_types = room_types.to(self.device)
        masks = masks.float()
        x = torch.cat([images, walls, doors], dim=1)
        model_kwargs = {
            'masks': masks,
            'room_types': room_types,
            'bubbles': bubbles,
            'custom_eps_loss': self.custom_eps_loss_with_room_types
        }

        return {
            'x': x,
            'model_kwargs': model_kwargs
        }
