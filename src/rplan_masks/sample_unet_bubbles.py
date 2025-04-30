from functools import partial
from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.gaussian_noise import GaussianDiffusion, BetaSchedule, ModelMeanType, ModelVarType
from src.rplan.dataset import RPlanMasksDataset, RPlanImageDataset
from src.rplan.types import MaskPlan, RoomType, ImagePlan
from src.rplan_masks.karras.cfg import CFGUnet
from src.rplan_masks.karras.denoise import GithubUnet


@torch.no_grad()
def sample_plans_bubbles(diffusion: GaussianDiffusion, model, num_samples: int = 1, data_path='data/rplan', mask_size: int = 64,
                bubbles: Optional[np.ndarray] = None, condition_scale: float = 1.0, force_bubbles: bool = False,
                rescaled_phi: float = 0.0, ddim: bool = False,
                device: torch.device = torch.device('cpu')):
    dataset = RPlanImageDataset(data_path=data_path, mask_size=(mask_size, mask_size), shuffle_rooms=True,
                                random_scale=0.6)
    random_samples = [dataset[np.random.randint(0, len(dataset))] for _ in range(num_samples)]
    input_bubbles = bubbles
    if bubbles is not None:
        for i in range(len(random_samples)):
            random_samples[i].bubbles = bubbles[i]
    # random_sample = dataset[0]
    # random_sample = from_export('notebooks/data/export')
    # with open('data/rplan/1.json') as f:
    #     raw_plan = RawPlan(json.load(f))
    #
    # plan = Plan.from_raw(raw_plan)
    # random_sample = TorchTransformerPlan.from_plan(plan, 32, 100, front_door_at_end=True)
    random_sample_batched = ImagePlan.collate(random_samples)

    images, walls, doors, _, bubbles, masks = random_sample_batched
    images, walls, doors, bubbles, masks = images.to(device), walls.to(device), doors.to(device), bubbles.to(
        device), masks.to(device)
    # room_types = room_types.float()
    # real_room_types = [RoomType.from_one_hot(room_type) for room_type in room_types[0]]
    masks = masks.float()
    # masks = torch.ones_like(masks, device=device)
    x = torch.cat([images, walls, doors], dim=1)

    model_kwargs = {
        'masks': masks,
        # 'bubbles': bubbles if input_bubbles is not None else None,
        'bubbles': bubbles if force_bubbles or (input_bubbles is not None) else None, # TODO
        'cond_scale': condition_scale,
        'rescaled_phi': rescaled_phi,
        # 'src_key_padding_mask': src_key_padding_mask,
        # 'nodes_to_graph': nodes_to_graph,
    }

    if ddim:
        samples, _ = diffusion.ddim_sample_loop(model, x.shape, model_kwargs=model_kwargs)
    else:
        samples, _ = diffusion.p_sample_loop(model, x.shape, model_kwargs=model_kwargs)
    samples = samples.cpu().numpy()
    return samples
