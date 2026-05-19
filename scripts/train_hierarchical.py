"""Train a binary U-Net for one BraTS hierarchical region (WT, TC, or ET).

Reuses the existing training pipeline exactly. The only difference: the
multi-class BraTS mask is collapsed to a binary mask for the target
region before being fed to the loss. Three of these — one per region —
form the deep hierarchical cascade.

Regions (BraTS standard, nested):
  WT (Whole Tumor)    = any non-background class (labels 1, 2, 3)
  TC (Tumor Core)     = NCR/NET + enhancing  (labels 1, 3)  — excludes edema
  ET (Enhancing)      = enhancing only        (label 3)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.brats import BraTSDataset
from src.data.splits import load_split
from src.models import build_model
from src.training.augmentation import AlbumentationsWrapper, get_train_transforms
from src.training.losses import DiceCELoss
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.seed import set_seed


REGION_TO_CLASSES: dict[str, list[int]] = {
    "WT": [1, 2, 3],
    "TC": [1, 3],
    "ET": [3],
}


class BinaryRegionWrapper(Dataset):
    """Wraps a BraTSDataset; returns (image, binary_mask) for the target region."""

    def __init__(self, base: Dataset, classes_in_region: list[int]):
        self.base = base
        self.classes = classes_in_region

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, mask = self.base[idx]
        binary_mask = torch.zeros_like(mask, dtype=mask.dtype)
        for c in self.classes:
            binary_mask[mask == c] = 1
        return image, binary_mask


def main(config_path: str, region: str, resume: str | None = None) -> list[dict]:
    if region not in REGION_TO_CLASSES:
        raise ValueError(f"region must be one of {list(REGION_TO_CLASSES)}; got {region}")

    cfg = load_config(config_path)
    set_seed(cfg["experiment"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Force binary classification + region-specific experiment name
    cfg["model"]["out_channels"] = 2
    arch = cfg["model"]["arch"]
    # Vanilla "unet" keeps short name for backward compat with existing checkpoints;
    # other architectures get the arch baked in so multiple cascades coexist on the volume.
    if arch == "unet":
        cfg["experiment"]["name"] = f"hierarchical_{region.lower()}"
    else:
        cfg["experiment"]["name"] = f"hierarchical_{arch}_{region.lower()}"

    classes = REGION_TO_CLASSES[region]
    print(f"Training binary segmentation for region {region} (mask classes: {classes})")

    model = build_model(
        cfg["model"]["arch"],
        in_channels=cfg["model"]["in_channels"],
        out_channels=2,
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
            else:
                for _ in range(ckpt["epoch"]):
                    scheduler.step()
        resume_state = {
            "best_val_dice": ckpt.get("best_val_dice", ckpt.get("val_dice", -1.0)),
            "epochs_without_improvement": ckpt.get("epochs_without_improvement", 0),
            "history": ckpt.get("history", []),
        }
        print(f"Resumed from {resume} (epoch {ckpt['epoch']}, val_dice={resume_state['best_val_dice']:.4f})")

    # Build dataloaders with binary mask wrapper
    split = load_split(cfg["data"]["split_file"])
    train_transform = None
    if cfg["training"].get("augment", True):
        train_transform = AlbumentationsWrapper(get_train_transforms(cfg["data"]["image_size"]))
    train_base = BraTSDataset(cfg["data"]["data_root"], patient_ids=split["train"], transform=train_transform)
    val_base = BraTSDataset(cfg["data"]["data_root"], patient_ids=split["val"], transform=None)
    train_ds = BinaryRegionWrapper(train_base, classes)
    val_ds = BinaryRegionWrapper(val_base, classes)

    pin = torch.cuda.is_available()
    nw = min(cfg["data"].get("num_workers", 4), os.cpu_count() or 2)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True,
        num_workers=nw, pin_memory=pin,
        persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
        num_workers=nw, pin_memory=pin,
        persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_classes=2,
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

    history = trainer.fit(
        train_loader, val_loader,
        epochs=cfg["training"]["epochs"], start_epoch=start_epoch,
    )
    print(f"\n[{region}] Best val Dice: {trainer.best_val_dice:.4f}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--region", required=True, choices=["WT", "TC", "ET"])
    parser.add_argument("--resume", default=None, help="Path to checkpoint .pth to resume from")
    args = parser.parse_args()
    main(args.config, args.region, resume=args.resume)
