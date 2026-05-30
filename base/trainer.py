"""Abstract training loop shared by KG trainers."""

import time
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from data.dataset import Dataset
from data.dataloader import collate
from utils.checkpoint import save_checkpoint, best_model_path, last_model_path
from utils.device import report_num_trainable_parameters
from utils.logger import logger


class Trainer(ABC):
    """Abstract base class for KG trainers. Defines the training loop and evaluation logic, while leaving model-specific details to subclasses."""

    def __init__(self, args, ngpus_per_node, model, criterion):
        self.args = args
        self.ngpus_per_node = ngpus_per_node
        self.model = model
        self.criterion = criterion
        self.best_metric = None
        self.best_checkpoint_path = None
        self.train_time = 0.0
        self.valid_time = 0.0
        self.total_time = 0.0

        self._setup_training()

        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        report_num_trainable_parameters(self.model)

        train_dataset = Dataset(path=args.train_path, task=args.dataset)
        valid_dataset = Dataset(path=args.valid_path, task=args.dataset) if args.valid_path else None
        num_training_steps = args.epochs * len(train_dataset) // max(args.batch_size, 1)
        args.warmup = min(args.warmup, num_training_steps // 10)
        logger.info('Total training steps: {}, warmup steps: {}'.format(num_training_steps, args.warmup))
        self.scheduler = self._create_lr_scheduler(num_training_steps)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )

        self.valid_loader = None
        if valid_dataset:
            self.valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=args.batch_size * 2,
                shuffle=True,
                collate_fn=collate,
                num_workers=args.workers,
                pin_memory=True,
            )

    def train_loop(self) -> dict:
        """Main training loop that iterates over epochs, runs training and evaluation, and handles checkpointing."""

        if self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        total_start_time = time.time()

        for epoch in range(self.args.epochs):
            epoch_train_start = time.time()
            self.train_epoch(epoch)
            self.train_time += time.time() - epoch_train_start
            self._run_eval(epoch=epoch)

        self.total_time = time.time() - total_start_time
        logger.info(f"[Timing] Training time (s): {round(self.train_time, 2)}")
        logger.info(f"[Timing] Valid time (s): {round(self.valid_time, 2)}")
        logger.info(f"[Timing] Total run time (s): {round(self.total_time, 2)}")

        return {
            'best_epoch': None if self.best_metric is None else self.best_metric.get('epoch'),
            'best_mrr': None if self.best_metric is None else self.best_metric.get('score'),
            'train_time': self.train_time,
            'valid_time': self.valid_time,
            'total_time': self.total_time,
            'best_checkpoint_path': self.best_checkpoint_path,
        }

    @abstractmethod
    def train_epoch(self, epoch) -> float:
        """Train for one epoch. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def eval_epoch(self, epoch) -> dict:
        """Evaluate for one epoch. Must be implemented by subclasses."""
        raise NotImplementedError

    @torch.no_grad()
    def _run_eval(self, epoch, step=0) -> dict:
        """Run evaluation and handle checkpointing based on the results."""

        eval_start = time.time()
        metric_dict = self.eval_epoch(epoch)
        self.valid_time += time.time() - eval_start
        monitor_value = self._extract_monitor_value(metric_dict)
        is_best = monitor_value is not None and (self.best_metric is None or monitor_value > self.best_metric.get('score', float('-inf')))
        if is_best:
            self.best_metric = {'score': monitor_value, 'metrics': metric_dict, 'epoch': epoch}

        saved_checkpoint_path = save_checkpoint({
            'epoch': epoch,
            'best_epoch': epoch if is_best else None,
            'best_metric': self.best_metric,
            'args': self.args.__dict__,
            'state_dict': self.model.state_dict(),
        }, is_best=is_best, filename=last_model_path(self.args.output_dir))
        if is_best:
            self.best_checkpoint_path = best_model_path(self.args.output_dir)
        elif self.best_checkpoint_path is None:
            self.best_checkpoint_path = saved_checkpoint_path
        return metric_dict

    def _extract_monitor_value(self, metric_dict, valid_metric='mrr') -> float | None:
        """Extract the value to monitor for checkpointing from the metric dictionary."""

        if not metric_dict:
            return None
        if valid_metric in metric_dict:
            return metric_dict[valid_metric]
        if 'loss' in metric_dict:
            return -metric_dict['loss']
        for value in metric_dict.values():
            if isinstance(value, (int, float)):
                return value
        return None

    def _setup_training(self) -> None:
        """Set up the model for training, including moving to GPU(s) if available."""

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model).cuda()
        elif torch.cuda.is_available():
            self.model.cuda()
        else:
            logger.info('No gpu will be used')

    def _create_lr_scheduler(self, num_training_steps) -> Any:
        """Create a learning rate scheduler based on the specified type."""

        if self.args.lr_scheduler == 'linear':
            return get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.args.warmup,
                num_training_steps=num_training_steps,
            )
        elif self.args.lr_scheduler == 'cosine':
            return get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.args.warmup,
                num_training_steps=num_training_steps,
            )
        else:
            assert False, 'Unknown lr scheduler: {}'.format(self.args.lr_scheduler)