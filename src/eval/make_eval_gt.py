import os

import cv2

from src.rplan.dataset import RPlanImageDataset
from src.rplan.types import ImagePlan
from tqdm.contrib.concurrent import process_map

def save_img_plan(output_path:str, plan: ImagePlan, plan_idx: int, epoch: int):
    image = plan.to_image()
    cv2.imwrite(os.path.join(output_path, f"{epoch}_{plan_idx}.png"), image)

def make_eval_gt(output_path: str, input_path: str = 'data/rplan', epochs: int = 1, num_workers: int = 8, mask_size: int = 64):
    os.makedirs(output_path, exist_ok=True)
    dataset = RPlanImageDataset(input_path, load_base_rplan=True, random_flip=True, random_scale=0.6,
                                no_doors=False,
                                no_front_door=False,
                                random_translate=True,
                                random_rotate=True, shuffle_rooms=True, max_workers=num_workers,
                                mask_size=(mask_size, mask_size))

    for epoch in range(epochs):
        process_map(save_img_plan(), [output_path] * len(dataset), dataset, range(len(dataset)), [epoch] * len(dataset), max_workers=num_workers, chunksize=100, desc=f"Epoch {epoch + 1}/{epochs}", total=len(dataset))

if __name__ == '__main__':
    make_eval_gt(epochs=10)
