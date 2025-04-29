from typing import Optional

import torch

from src.rplan_masks.cfg_bubble_trainer import CfgBubbleTrainer
from src.rplan_masks.karras.cfg import CFGUnet
from src.rplan_masks.unet_bubble_trainer import UnetBubbleTrainer


def train_unet_bubbles(model: Optional[torch.nn.Module] = None):
    trainer = CfgBubbleTrainer(model=model, lr=1e-4, mask_size=64, epochs=30, batch_size=16)
    trainer.train()

if __name__ == '__main__':
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = CFGUnet(dim=64, channels=6, out_dim=3, cond_drop_prob=0, bubble_dim=2).to(device)

    # model.load_state_dict(torch.load(
    #     '/Users/mariodeaconescu/Library/CloudStorage/GoogleDrive-mariodeaconescu2003@gmail.com/My Drive/ColabFiles/diffusion_generation/export/checkpoints/2025-03-07_16-42-58/model_49.pt', map_location=device))
    model_path = '/Users/mariodeaconescu/Library/CloudStorage/GoogleDrive-mariodeaconescu2003@gmail.com/My Drive/ColabFiles/diffusion_generation/export/checkpoints_unet_bubbles/2025-04-21_00-54-32/model_29_500.pt'
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    train_unet_bubbles(model)
