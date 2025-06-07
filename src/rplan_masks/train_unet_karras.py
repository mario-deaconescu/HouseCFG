import os
import time
from datetime import datetime

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch_geometric.nn.lr_scheduler import CosineWithWarmupRestartsLR

from src.gaussian_noise import GaussianDiffusion, BetaSchedule, ModelMeanType, ModelVarType
from src.rplan.dataset import RPlanMasksDataset, RPlanImageDataset
from src.rplan.types import MaskPlan, ImagePlan
from src.rplan_masks.karras.cfg import CFGUnet
from src.rplan_masks.karras.denoise import GithubUnet
from src.rplan_masks.karras.karras_unet import KarrasUnet
from src.rplan_masks.unet import RPlanMaskUnet
from src.rplan_masks.unet_github import GitHubUNet

from src.rplan_masks.unet_trainer import UnetTrainer
from src.schedule_sampler import LossSampler, UniformSampler

def custom_eps_loss(output: torch.Tensor, eps: torch.Tensor, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, **model_kwargs):
    # valid_indices = ~model_kwargs['src_key_padding_mask']
    # eps[~valid_indices] = 0
    mse = torch.nn.functional.mse_loss(output, eps, reduction='none').mean(dim=(2, 3))
    # mse = mse * valid_indices
    mse = mse.mean(dim=1)
    return mse

def train_unet_karras(epochs: int = 50, num_timesteps: int = 1000, batch_size: int = 32, lr: float = 1e-4, num_workers: int = 8,
          model_dict: dict = None, mask_size: int = 64, checkpoint_path: str = 'checkpoints_unet'):
    device = torch.device('cuda' if torch.cuda.is_available() else torch.device(
        'mps') if torch.backends.mps.is_available() else torch.device('cpu'))

    dataset = RPlanImageDataset('data/rplan', load_base_rplan=True, random_flip=True, random_scale=0.6,
                                no_doors=False,
                                no_front_door=False,
                                random_translate=True,
                                random_rotate=True, shuffle_rooms=True, max_workers=num_workers,
                                mask_size=(mask_size, mask_size))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            persistent_workers=True, collate_fn=ImagePlan.collate)

    diffusion = GaussianDiffusion(num_timesteps, BetaSchedule.COSINE, ModelMeanType.EPSILON, ModelVarType.FIXED_SMALL,
                                  device=device)

    # model = RPlanMaskUnet(mask_size, 32, 5, room_type_channels=8, time_emb_dim=256, norm='instance').to(device)
    model = CFGUnet(dim=64, channels=4, out_dim=3, cond_drop_prob=0.2).to(device)

    if model_dict is not None:
        model.load_state_dict(model_dict)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, 10, T_mult=2)

    loss_sampler = LossSampler(diffusion)

    checkpoints_path = os.path.join(checkpoint_path, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

    for epoch in range(epochs):
        start_time = time.time()
        for i, batch in enumerate(dataloader):
            images, walls, doors, room_types, masks = batch
            images, walls, doors, room_types, masks = images.to(device), walls.to(device), doors.to(device), room_types.to(device), masks.to(device)
            load_time = time.time() - start_time
            start_time = time.time()
            masks = masks.float()
            # masks[masks == 0] = -1
            # edges = edges.transpose(0, 1).float()
            # edges = torch.stack([edges[:, 0], torch.ones(edges.size(0), device=device), edges[:, 1]], dim=1)
            x = torch.cat([images, walls, doors], dim=1)

            optimizer.zero_grad()

            model_kwargs = {
                'masks': masks,
                'room_types': room_types,
                # 'edges': edges,
                # 'nodes_to_graph': nodes_to_graph,
                # 'src_key_padding_mask': src_key_padding_mask,
                'custom_eps_loss': custom_eps_loss
            }

            # num_batches = (nodes_to_graph.max() + 1).item()
            t, sampler_weights = loss_sampler.sample(masks.shape[0], device=device)
            # t = t_batch[nodes_to_graph.detach().cpu().numpy()]
            # sampler_weights = sampler_weights[nodes_to_graph.detach().cpu().numpy()]

            losses = diffusion.training_losses(model, x, t, model_kwargs=model_kwargs)['loss']

            loss_sampler.update_with_all_losses(t, losses.detach())
            loss = (losses * sampler_weights).mean()
            loss.backward()

            optimizer.step()

            train_time = time.time() - start_time

            print(f'Epoch {epoch}, Batch {i}, Loss {loss.item()}, Load time {load_time}, Train time {train_time}')

            start_time = time.time()

            if i % 500 == 0 and (i > 0 or epoch > 0):
                os.makedirs(checkpoints_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(checkpoints_path, f'model_{epoch}_{i}.pt'))

        scheduler.step()


if __name__ == '__main__':
    # model_path = 'checkpoints_unet/2025-04-13_17-43-10/model_2_2500.pt'
    # state_dict = torch.load(model_path)
    state_dict = None
    # train_unet_karras(batch_size=16, model_dict=state_dict, lr=1e-4, mask_size=64)
    UnetTrainer(batch_size=16, lr=1e-4, mask_size=64, epochs=100).train()
