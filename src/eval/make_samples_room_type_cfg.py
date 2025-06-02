import os
from typing import Optional

import numpy as np
import torch
from tqdm.contrib.concurrent import process_map

from src.eval.make_eval_gt import save_img_plan
from src.gaussian_noise import BetaSchedule, ModelMeanType, ModelVarType
from src.respace import SpacedDiffusion, space_timesteps
from src.rplan.dataset import RPlanImageDataset
from src.rplan.types import ImagePlan
from src.rplan_masks.sample_unet_room_types import sample_plans_room_types


def make_samples_room_type_cfg(model, output_path: str, num_samples: int = 1000, batch_size: int = 1,
                               num_timesteps: int = 1000,
                               data_path='../data/rplan', mask_size=64, condition_scale=1, rescaled_phi=0, ddim=True,
                               thin_walls=True, simplify: Optional[float] = None, target_size: Optional[tuple[int, int]] = None,
                               num_workers: int = 8):
    T = 1000
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    diffusion = SpacedDiffusion(space_timesteps(T, f'ddim{num_timesteps}'), T, BetaSchedule.COSINE,
                                ModelMeanType.EPSILON,
                                ModelVarType.FIXED_SMALL,
                                device=device)

    generated = 0
    epoch = 0

    os.makedirs(output_path, exist_ok=True)
    dataset = RPlanImageDataset(data_path=data_path, mask_size=(mask_size, mask_size), shuffle_rooms=True,
                                random_scale=0.6)
    while generated < num_samples:
        if condition_scale > 0:
            random_samples_types = [dataset[np.random.randint(0, len(dataset))] for _ in range(batch_size)]
            room_types = [sample.room_types for sample in random_samples_types]
        else:
            room_types = None
        samples = sample_plans_room_types(diffusion, model, num_samples=batch_size, device=device, data_path=data_path,
                                          mask_size=mask_size, room_types=room_types,
                                          condition_scale=condition_scale, rescaled_phi=rescaled_phi, ddim=ddim)

        plans = [ImagePlan(walls=walls, image=rooms, door_image=doors) for rooms, walls, doors in samples]

        plans = [plan.to_plan(thin_walls=thin_walls, simplify=simplify, target_size=target_size)[1] for plan in plans]

        process_map(save_img_plan, [output_path] * len(plans), plans, range(len(plans)), [epoch] * len(plans),
                    max_workers=num_workers, chunksize=100, desc=f"Generating Step {epoch}: {generated}/{num_samples}",
                    total=len(plans))
        generated += len(plans)
        epoch += 1


def eval_samples_room_type_cfg(model, num_samples: int = 1000, batch_size: int = 1,
                               num_timesteps: int = 1000,
                               data_path='../data/rplan', mask_size=64, condition_scale=1, rescaled_phi=0, ddim=True,
                               thin_walls=True, simplify: Optional[float] = None, target_size: Optional[tuple[int, int]] = None,
                               num_workers: int = 8, output_path: Optional[str] = None):
    T = 1000
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    diffusion = SpacedDiffusion(space_timesteps(T, f'ddim{num_timesteps}'), T, BetaSchedule.COSINE,
                                ModelMeanType.EPSILON,
                                ModelVarType.FIXED_SMALL,
                                device=device)

    generated = 0
    epoch = 0
    dataset = RPlanImageDataset(data_path=data_path, mask_size=(mask_size, mask_size), shuffle_rooms=True,
                                random_scale=0.6)

    input_room_types = []
    output_room_types = []
    while generated < num_samples:
        random_samples_types = [dataset[np.random.randint(0, len(dataset))] for _ in range(batch_size)]
        room_types = [sample.room_types for sample in random_samples_types]
        input_room_types.extend(room_types)
        samples = sample_plans_room_types(diffusion, model, num_samples=batch_size, device=device, data_path=data_path,
                                          mask_size=mask_size, room_types=room_types,
                                          condition_scale=condition_scale, rescaled_phi=rescaled_phi, ddim=ddim)

        plans = [ImagePlan(walls=walls, image=rooms, door_image=doors) for rooms, walls, doors in samples]

        plans = [plan.to_plan(thin_walls=thin_walls, simplify=simplify, target_size=target_size)[0] for plan in plans]

        plans = [ImagePlan.from_plan(plan, mask_size=mask_size, with_bubbles=False) for plan in plans]

        output_room_types.extend([plan.room_types for plan in plans])

        generated += len(plans)
        epoch += 1

    input_room_types = np.array(input_room_types)
    output_room_types = np.array(output_room_types)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(output_path, input_room_types=input_room_types, output_room_types=output_room_types)

    return input_room_types, output_room_types