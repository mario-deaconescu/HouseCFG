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
from src.rplan_masks.openai.unet import UNetModel
from src.schedule_sampler import ScheduleSampler, LossSampler
from src.utils.diffusion_trainer import DiffusionTrainer, DiffusionStepOutput


def custom_eps_loss(output: torch.Tensor, eps: torch.Tensor, x_0: torch.Tensor, x_t: torch.Tensor, **model_kwargs):
    room_types = x_0[:, 0, :, :]
    walls = x_0[:, 1, :, :]
    doors = x_0[:, 2, :, :]
    # room_type_frequencies = RoomType.frequencies()
    # max_count = max(room_type_frequencies.values())
    # normalized_frequencies = {room_type: max_count / count for room_type, count in room_type_frequencies.items()}
    room_type_loss = torch.nn.functional.mse_loss(output[:, 0, :, :], eps[:, 0, :, :], reduction='none').mean(
        dim=(1, 2))

    wall_loss = torch.nn.functional.mse_loss(output[:, 1, :, :], eps[:, 1, :, :], reduction='none')
    wall_positive_loss = wall_loss * (walls > 0)
    wall_negative_loss = wall_loss * (walls < 0)
    wall_loss = wall_positive_loss.mean(dim=(1, 2)) + wall_negative_loss.mean(dim=(1, 2))

    door_loss = torch.nn.functional.mse_loss(output[:, 2, :, :], eps[:, 2, :, :], reduction='none')
    door_positive_loss = door_loss * (doors > 0)
    front_door_loss = door_loss * (doors == 0)
    door_negative_loss = door_loss * (doors < 0)
    door_loss = door_positive_loss.mean(dim=(1, 2)) + door_negative_loss.mean(dim=(1, 2)) + front_door_loss.mean(
        dim=(1, 2))

    loss = wall_loss + door_loss * .5 + room_type_loss
    return loss


class OpenaiTrainer(DiffusionTrainer[ImagePlan, ImagePlanCollated]):

    def __init__(self, batch_size: int, lr: float, mask_size: int = 64, epochs: int = 30,
                 model: Optional[torch.nn.Module] = None, dataset: Optional[Dataset] = None,
                 diffusion: Optional[GaussianDiffusion] = None,
                 timestep_sampler: Optional[ScheduleSampler] = None,
                 fp_16: bool = False,
                 custom_loss: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 device: Optional[torch.device] = None,
                 collate_fn: Optional[Callable[[list[ImagePlan]], ImagePlanCollated]] = None,
                 checkpoint_path: Optional[str] = None, model_dict: Optional[dict] = None,
                 save_interval: Optional[int] = 500,
                 num_workers: int = 8, scheduler: Optional[LRScheduler] = None, optimizer: Optional[Optimizer] = None):
        self.custom_loss = custom_loss
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
        if model is None:
            model = UNetModel(image_size=mask_size, in_channels=6, model_channels=192, out_channels=3, num_res_blocks=3,
                              attention_resolutions=[32, 16, 8], num_head_channels=64, resblock_updown=True, use_scale_shift_norm=True,
                              use_new_attention_order=True, use_fp16=fp_16, dropout=0.1, cond_drop_prob=0.1).to(device)
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
        assert bubbles is not None, "Bubbles must be provided for OpenAI trainer"
        bubbles = bubbles.to(self.device)
        bubbles = bubbles[:, 0, :, :].unsqueeze(1)
        masks = masks.float()
        x = torch.cat([images, walls, doors], dim=1)
        model_kwargs = {
            'bubbles': bubbles,
            'masks': masks,
        }
        if self.custom_loss is not None:
            model_kwargs['custom_eps_loss'] = self.custom_loss

        return {
            'x': x,
            'model_kwargs': model_kwargs
        }
