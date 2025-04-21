import torch

from src.gaussian_noise import BetaSchedule, ModelMeanType, ModelVarType
from src.respace import SpacedDiffusion, space_timesteps
from src.rplan_masks.karras.cfg import CFGUnet, CFGUnetWithScale
from src.utils.sampler import Sampler


@torch.no_grad()
def sample_unet_bubbles(num_samples: int = 1, timesteps: int = 100, mask_size: int = 64, ddim: bool = True):
    T = 1000

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    diffusion = SpacedDiffusion(space_timesteps(T, f'ddim{timesteps}'), T, BetaSchedule.COSINE, ModelMeanType.EPSILON,
                                ModelVarType.FIXED_SMALL,
                                device=device)
    model = CFGUnet(dim=mask_size, channels=6, out_dim=3, cond_drop_prob=0.15).to(device)

    # model.load_state_dict(torch.load(
    #     '/Users/mariodeaconescu/Library/CloudStorage/GoogleDrive-mariodeaconescu2003@gmail.com/My Drive/ColabFiles/diffusion_generation/export/checkpoints/2025-03-07_16-42-58/model_49.pt', map_location=device))
    model_path = '/Users/mariodeaconescu/Library/CloudStorage/GoogleDrive-mariodeaconescu2003@gmail.com/My Drive/ColabFiles/diffusion_generation/export/checkpoints_unet_bubbles/2025-04-21_00-54-32/model_6_0.pt'
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    full_model = CFGUnetWithScale(model).to(device)

    sampler = Sampler(full_model, (3, mask_size, mask_size), diffusion, ddim=ddim, device=device)
    return sampler.sample(
        num_samples,
        model_kwargs={
            'cond_scale': 0.5,
        }
    )


if __name__ == '__main__':
    sample_unet_bubbles(1)
