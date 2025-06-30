import argparse

import torch

from src.rplan_masks.combined_trainer import CombinedTrainer
from src.rplan_masks.openai_trainer import OpenaiTrainer
from src.rplan_masks.cfg_trainer import CfgTrainer
from src.utils.scripts import make_parser


def main(args: argparse.Namespace):
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    if args.type == 'room_types':
        trainer_type = CfgTrainer
    elif args.type == 'bubbles_old':
        trainer_type = OpenaiTrainer
    elif args.type == 'bubbles' or args.model_choice == 'bubbles_cfg':
        trainer_type = CombinedTrainer
    else:
        raise ValueError(f"Unknown model choice: {args.model_choice}")

    trainer = trainer_type(epochs=args.epoch, batch_size=args.batch_size, lr=args.learning_rate, mask_size=64,
                           checkpoint_path=args.output, save_interval=args.save_interval)

    if not args.no_scheduler:
        trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(trainer.optimizer, 20, T_mult=2)

    trainer.train()


if __name__ == '__main__':
    parser = make_parser()
    parser.add_argument('--output', '-o', type=str, required=False, help='Path to save the model checkpoints.',
                        default='checkpoints')
    parser.add_argument('--epoch', '-e', type=int, default=20, help='Total number of epochs to train.')
    parser.add_argument('--save_interval', '-s', type=int, default=1000, help='Number of batches between saves.')
    parser.add_argument('--learning_rate', '-l', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--no_scheduler', action='store_true', default=False,)
    args = parser.parse_args()
    main(args)
