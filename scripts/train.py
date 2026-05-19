"""Train a segmentation model.

Usage:
    python -m scripts.train --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.data.brats import BraTSDataset
from src.data.splits import load_split
from src.models import build_model
from src.training.augmentation import AlbumentationsWrapper, get_train_transforms
from src.training.losses import DiceCELoss
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.seed import set_seed


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    split = load_split(cfg["data"]["split_file"])
    train_transform = None
    if cfg["training"].get("augment", True):
        train_transform = AlbumentationsWrapper(get_train_transforms(cfg["data"]["image_size"]))
    train_ds = BraTSDataset(cfg["data"]["data_root"], patient_ids=split["train"], transform=train_transform)
    val_ds = BraTSDataset(cfg["data"]["data_root"], patient_ids=split["val"], transform=None)
    import os as _os
    pin = torch.cuda.is_available()
    nw  = min(cfg["data"].get("num_workers", 4), _os.cpu_count() or 2)
    return (
        DataLoader(
            train_ds,
            batch_size=cfg["data"]["batch_size"],
            shuffle=True,
            num_workers=nw,
            pin_memory=pin,
            persistent_workers=nw > 0,
            prefetch_factor=4 if nw > 0 else None,
        ),
        DataLoader(
            val_ds,
            batch_size=cfg["data"]["batch_size"],
            shuffle=False,
            num_workers=nw,
            pin_memory=pin,
            persistent_workers=nw > 0,
            prefetch_factor=4 if nw > 0 else None,
        ),
    )


def main(config_path: str = "configs/default.yaml", resume: str | None = None) -> list[dict]:
    cfg = load_config(config_path)
    set_seed(cfg["experiment"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = build_model(
        cfg["model"]["arch"],
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_channels=cfg["model"].get("base_channels", 64),
        dropout=cfg["model"].get("dropout", 0.2),
    )
    print(f"Model parameters: {model.parameter_count():,}")

    criterion = DiceCELoss(
        dice_weight=cfg["training"].get("dice_weight", 0.5),
        ce_weight=cfg["training"].get("ce_weight", 0.5),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 1e-5),
    )
    scheduler = None
    if cfg["training"].get("scheduler", "cosine") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["training"]["epochs"],
            eta_min=cfg["training"].get("min_lr", 1e-6),
        )

    start_epoch = 1
    resume_state = None
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        if scheduler is not None:
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            else:  # older _best.pth without scheduler state — fast-forward instead
                for _ in range(ckpt["epoch"]):
                    scheduler.step()
        resume_state = {
            "best_val_dice": ckpt.get("best_val_dice", ckpt.get("val_dice", -1.0)),
            "epochs_without_improvement": ckpt.get("epochs_without_improvement", 0),
            "history": ckpt.get("history", []),
        }
        print(
            f"Resumed from {resume} (epoch {ckpt['epoch']}, "
            f"val_dice={resume_state['best_val_dice']:.4f}) — continuing from epoch {start_epoch}"
        )

    train_loader, val_loader = build_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_classes=cfg["model"]["out_channels"],
        checkpoint_dir=cfg.get("checkpoint_dir", "checkpoints"),
        output_dir=cfg["experiment"].get("output_dir", "outputs"),
        experiment_name=cfg["experiment"]["name"],
        config=cfg,
        early_stopping_patience=cfg["training"].get("early_stopping_patience"),
        tensorboard=cfg.get("logging", {}).get("tensorboard", False),
    )
    if resume_state is not None:
        trainer.best_val_dice = resume_state["best_val_dice"]
        trainer.epochs_without_improvement = resume_state["epochs_without_improvement"]
        trainer.history = resume_state["history"]
    history = trainer.fit(train_loader, val_loader, epochs=cfg["training"]["epochs"], start_epoch=start_epoch)
    print(f"\nBest val Dice: {trainer.best_val_dice:.4f}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint .pth to resume from")
    args = parser.parse_args()
    main(args.config, resume=args.resume)
