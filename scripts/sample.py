import argparse
from functools import partial

import torch

from src.eval.make_samples_combined import make_samples_combined
from src.eval.make_samples_openai import make_samples_bubbles
from src.eval.make_samples_room_type_cfg import make_samples_room_type_cfg
from src.rplan_masks.karras.cfg import CFGUnetWithScale
from src.rplan_masks.openai.unet import UNetModel
from src.rplan_masks.karras.cfg import CFGUnet
from src.utils.scripts import make_parser


def main(args: argparse.Namespace):
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    if args.type == 'room_types':
        model = CFGUnet(dim=64, channels=4, out_dim=3, cond_drop_prob=0)
        sampler = partial(make_samples_room_type_cfg, eval_constraints=args.eval_constraints)
    elif args.type == 'bubbles_old':
        UNetModel(image_size=64, in_channels=6, model_channels=192, out_channels=3, num_res_blocks=3,
                  attention_resolutions=[32, 16, 8], num_head_channels=64, resblock_updown=True,
                  use_scale_shift_norm=True,
                  use_new_attention_order=True, use_fp16=False, dropout=0.1, cond_drop_prob=0)
        sampler = make_samples_bubbles
    elif args.type == 'bubbles' or args.model_choice == 'bubbles_cfg':
        model = CFGUnet(dim=64, channels=6, out_dim=3, cond_drop_prob=0, bubble_dim=1)
        sampler = partial(make_samples_combined,
                          eval_constraints=args.eval_constraints)
    else:
        raise ValueError(f"Unknown model choice: {args.model_choice}")

    model = model.to(device)
    state_dict = torch.load(f"./models/{args.model}", map_location=device)

    if args.type == 'bubbles':
        state_dict['null_bubble_diagram'] = torch.zeros_like(model.null_bubble_diagram)
    model.load_state_dict(state_dict)
    full_model = CFGUnetWithScale(model).to(device)

    sampler(full_model, args.output, data_path=args.input, mask_size=64, num_samples=args.num_samples,
            ddim=not args.ddpm, num_timesteps=args.steps, condition_scale=args.condition_scale,
            target_size=(256, 256), batch_size=args.batch_size,
            eval_constraints=args.eval_constraints)


if __name__ == '__main__':
    parser = make_parser()
    parser.add_argument('--model', '-m', type=str, required=True, help='Path to the model checkpoint.')
    parser.add_argument('--output', '-o', type=str, required=False, help='Path to save the generated samples.',
                        default='output')
    parser.add_argument('--num_samples', '-n', type=int, default=1, help='Number of samples to generate.')
    parser.add_argument('--ddpm', action='store_true', help='Use DDPM sampling method.', default=False)
    parser.add_argument('--steps', '-s', type=int, default=100, help='Number of sampling steps.')
    parser.add_argument('--condition_scale', '-c', type=float, default=1.0, )
    parser.add_argument('--eval_constraints', action='store_true', help='Evaluate constraints during sampling.',
                        default=False)
    args = parser.parse_args()
    main(args)
