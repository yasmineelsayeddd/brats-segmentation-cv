"""Training loop for segmentation models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from src.utils.artifacts import experiment_dir, save_json, save_rows_csv, save_yaml
from src.utils.logging import setup_logger

logger = setup_logger("trainer")


class Trainer:
    """Generic segmentation trainer with AMP and GPU-side metrics."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None = None,
        device: str = "cuda",
        num_classes: int = 4,
        checkpoint_dir: str | Path = "checkpoints",
        output_dir: str | Path = "outputs",
        experiment_name: str = "model",
        config: dict[str, Any] | None = None,
        early_stopping_patience: int | None = None,
        tensorboard: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.output_dir = experiment_dir(output_dir, experiment_name)
        self.early_stopping_patience = early_stopping_patience
        self.best_val_dice = -1.0
        self.epochs_without_improvement = 0
        self.history: list[dict[str, Any]] = []
        self.writer = None
        # AMP scaler — no-op on CPU
        self.scaler = GradScaler("cuda") if device == "cuda" else None

        if config is not None:
            save_yaml(config, self.output_dir / "config.yaml")
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
            except ImportError:
                logger.warning("TensorBoard requested but unavailable; continuing without it.")

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict[str, float]:
        self.model.train(train)
        total_loss = 0.0
        n_batches = 0

        # Accumulate dice + accuracy on GPU — avoids 1026 CPU transfers per epoch
        dice_num = torch.zeros(self.num_classes - 1, device=self.device)
        dice_den = torch.zeros(self.num_classes - 1, device=self.device)
        correct = torch.tensor(0, device=self.device)
        total_px = torch.tensor(0, device=self.device)

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for images, masks in loader:
                images = images.to(self.device, non_blocking=True)
                masks  = masks.to(self.device, non_blocking=True)

                with autocast("cuda", enabled=self.scaler is not None):
                    logits = self.model(images)
                    loss   = self.criterion(logits, masks)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()

                preds = logits.argmax(dim=1)
                total_loss += loss.item()
                n_batches  += 1

                # GPU-side metrics (no .cpu() call per batch)
                for i, c in enumerate(range(1, self.num_classes)):
                    pred_c = preds == c
                    mask_c = masks == c
                    dice_num[i] += 2.0 * (pred_c & mask_c).sum()
                    dice_den[i] += pred_c.sum() + mask_c.sum()
                correct  += (preds == masks).sum()
                total_px += masks.numel()

        if n_batches == 0:
            raise ValueError("DataLoader produced zero batches")

        dice = ((dice_num + 1e-6) / (dice_den + 1e-6)).mean().item()
        acc  = (correct / total_px).item()
        return {"loss": total_loss / n_batches, "dice": dice, "acc": acc}

    def _save_checkpoint(self, epoch: int) -> Path:
        ckpt_path = self.checkpoint_dir / f"{self.experiment_name}_best.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_dice": self.best_val_dice,
                "experiment_name": self.experiment_name,
            },
            ckpt_path,
        )
        save_json({"best_checkpoint": str(ckpt_path), "best_val_dice": self.best_val_dice}, self.output_dir / "best.json")
        return ckpt_path

    def _save_last(self, epoch: int) -> None:
        """Full training state, written every epoch so an interrupted run can resume cleanly."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_dice": self.best_val_dice,
            "epochs_without_improvement": self.epochs_without_improvement,
            "history": self.history,
            "experiment_name": self.experiment_name,
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(state, self.checkpoint_dir / f"{self.experiment_name}_last.pth")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        experiment_name: str | None = None,
        start_epoch: int = 1,
    ) -> list[dict[str, Any]]:
        if experiment_name is not None and experiment_name != self.experiment_name:
            self.experiment_name = experiment_name

        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics   = self._run_epoch(val_loader,   train=False)
            if self.scheduler is not None:
                self.scheduler.step()

            record = {
                "epoch":      epoch,
                "train_loss": train_metrics["loss"],
                "train_dice": train_metrics["dice"],
                "train_acc":  train_metrics["acc"],
                "val_loss":   val_metrics["loss"],
                "val_dice":   val_metrics["dice"],
                "val_acc":    val_metrics["acc"],
                "lr":         self.optimizer.param_groups[0]["lr"],
                "elapsed_s":  time.time() - t0,
            }
            self.history.append(record)

            if self.writer is not None:
                for key, value in record.items():
                    if key != "epoch":
                        self.writer.add_scalar(key, value, epoch)

            logger.info(
                f"Epoch {epoch:03d}/{epochs} train_loss={record['train_loss']:.4f} "
                f"val_dice={record['val_dice']:.4f} lr={record['lr']:.2e} ({record['elapsed_s']:.0f}s)"
            )

            if val_metrics["dice"] > self.best_val_dice:
                self.best_val_dice = val_metrics["dice"]
                self.epochs_without_improvement = 0
                ckpt_path = self._save_checkpoint(epoch)
                logger.info(f"New best val_dice={self.best_val_dice:.4f}; saved to {ckpt_path}")
            else:
                self.epochs_without_improvement += 1

            save_rows_csv(self.history, self.output_dir / "history.csv")
            save_json(self.history, self.output_dir / "history.json")
            self._save_last(epoch)

            if self.early_stopping_patience is not None and self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping after {self.epochs_without_improvement} epochs without improvement.")
                break

        if self.writer is not None:
            self.writer.close()
        return self.history
