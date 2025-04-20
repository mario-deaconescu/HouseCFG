import numpy as np
import torch


def dec2bin(xinp, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1, device=xinp.device, dtype=xinp.dtype)
    return xinp.unsqueeze(-1).bitwise_and(mask).ne(0).float()


def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)


def np_dec2bin(xinp, bits):
    # Use bitwise operations
    mask = 2 ** np.arange(bits - 1, -1, -1)
    return (xinp[..., None] & mask).astype(bool)

def np_bin2dec(b, bits):
    mask = 2 ** np.arange(bits - 1, -1, -1)
    return np.sum(mask * b, -1)