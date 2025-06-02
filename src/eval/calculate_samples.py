import os
import pickle
import shutil
from enum import Enum
from typing import Optional

import numpy as np
import torch
from cleanfid import fid
from tqdm import tqdm


class Stats(Enum):
    FID = "fid"
    KID = "kid"

class StatsScore:
    def __init__(self, scores: np.ndarray):
        self._scores = scores
        self.mean = np.mean(scores)
        self.std = np.std(scores, ddof=1)  # Sample standard deviation
        self.sem = self.std / np.sqrt(len(scores))  # Standard error of the mean

    def format(self, precision: int = 2) -> str:
        return f"{self.mean:.{precision}f} ± {self.sem:.{precision}f}"



def fid_for_run(path: str, device: torch.device = torch.device("cpu"), num_workers: int = 0) -> float:
    return fid.compute_fid(path, dataset_name="eval_gt_stats", verbose=False, batch_size=32, device=device,
                           num_workers=num_workers,
                           mode="clean", dataset_split="custom")

def kid_for_run(path: str, device: torch.device = torch.device("cpu"), num_workers: int = 0) -> float:
    return fid.compute_kid(path, dataset_name="eval_gt_stats", verbose=False, batch_size=32, device=device,
                           num_workers=num_workers,
                           mode="clean", dataset_split="custom")


def calculate_stats(path: str, stats: Stats, runs: int = 5, percentage: float = 0.8,
                  device: torch.device = torch.device("cpu"), save_path: Optional[str] = None) -> StatsScore:
    if save_path is not None:
        full_path = os.path.join(save_path, os.path.basename(path), f"{stats.value}.pkl")
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        if os.path.exists(full_path):
            with open(full_path, "rb") as f:
                return pickle.load(f)
    temp_dirs = [f"./tmp_{i}" for i in range(runs)]
    all_images = os.listdir(path)
    num_to_sample = int(len(all_images) * percentage)
    for temp_dir in temp_dirs:
        os.makedirs(temp_dir, exist_ok=True)
        for fname in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, fname))
        random_images = np.random.choice(all_images, num_to_sample, replace=False)
        for fname in random_images:
            src = os.path.join(path, fname)
            dst = os.path.join(temp_dir, fname)
            os.link(src, dst)

    func = fid_for_run if stats == Stats.FID else kid_for_run
    results = list(
        tqdm((func(temp_dir, device) for temp_dir in temp_dirs), desc=os.path.basename(path), total=runs,
             unit="run"))
    results = np.array(results)

    for temp_dir in temp_dirs:
        shutil.rmtree(temp_dir, ignore_errors=True)

    score = StatsScore(results)

    if save_path is not None:
        full_path = os.path.join(save_path, os.path.basename(path), f"{stats.value}.pkl")
        with open(full_path, "wb") as f:
            pickle.dump(score, f)

    return score
