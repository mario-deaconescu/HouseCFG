import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Generic, TypeVar, Optional, Callable

import torch
from torch._inductor.scheduler import Scheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset

T = TypeVar('T')
T_BATCH = TypeVar('T_BATCH')


class Trainer(ABC, Generic[T, T_BATCH]):
    dataset: Dataset
    epochs: int
    batch_size: int
    lr: float
    num_workers: int = 8
    save_interval: Optional[int] = None
    model_dict: Optional[dict] = None
    checkpoint_path: Optional[str] = None
    device: torch.device
    optimizer: Optimizer
    scheduler: Optional[LRScheduler]

    def __init__(self, model: torch.nn.Module, dataset: Dataset, epochs: int, batch_size: int, lr: float,
                 device: Optional[torch.device] = None,
                 collate_fn: Optional[Callable[[list[T]], T_BATCH]] = None,
                 checkpoint_path: Optional[str] = None, model_dict: Optional[dict] = None,
                 log_interval: Optional[int] = None,
                 num_workers: int = 8, scheduler: Optional[LRScheduler] = None, optimizer: Optional[Optimizer] = None):
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        else:
            self.device = device
        self.model = model.to(self.device)
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.collate_fn = collate_fn
        self.checkpoints_path = os.path.join(checkpoint_path, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        self.model_dict = model_dict
        self.num_workers = num_workers
        self.scheduler = scheduler
        self.log_interval = log_interval
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = optimizer
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                                      num_workers=self.num_workers, collate_fn=self.collate_fn,
                                                      persistent_workers=True)
        if self.model_dict is not None:
            self.model.load_state_dict(self.model_dict)

    @abstractmethod
    def step(self, batch: T_BATCH) -> torch.Tensor:
        pass

    def step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def train(self):
        loss_history = []
        for epoch in range(self.epochs):
            start_time = time.time()
            for batch_idx, batch in enumerate(self.dataloader):
                end_time = time.time()
                load_time = end_time - start_time
                start_time = time.time()
                self.optimizer.zero_grad()
                loss = self.step(batch)
                loss_history.append(loss.item())
                loss.backward()
                self.optimizer.step()
                train_time = time.time() - start_time
                print(
                    f'Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}, Load time {load_time}, Train time {train_time}')
                start_time = time.time()
                if self.save_interval is not None and batch_idx % self.log_interval == 0 and (
                        batch_idx > 0 or epoch > 0):
                    os.makedirs(self.checkpoints_path, exist_ok=True)
                    torch.save(self.model.state_dict(),
                               os.path.join(self.checkpoints_path, f'model_{epoch}_{batch_idx}.pt'))

            self.step_scheduler()

        return loss_history
