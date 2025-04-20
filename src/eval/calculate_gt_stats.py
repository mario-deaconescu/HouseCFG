import torch
from cleanfid import fid

def calculate_gt_stats(input_path: str = 'eval/gt', output_path: str = 'eval_gt_stats', batch_size: int = 64, num_workers:int=8):
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    fid.make_custom_stats(output_path, input_path, device=device, batch_size=batch_size)


if __name__ == '__main__':
    calculate_gt_stats()
