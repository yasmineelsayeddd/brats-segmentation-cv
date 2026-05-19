"""L11 — MC-dropout uncertainty evaluation.

Keeps dropout layers active at inference, runs N stochastic forward passes,
reports segmentation metrics on the mean prediction plus the
confidence-error correlation between predictive entropy and the per-pixel
error map.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.brats import BraTSDataset
from src.data.splits import load_split
from src.evaluation.metrics import (
    aggregate_summaries,
    confidence_error_correlation,
    summarize_segmentation,
)
from src.models import build_model
from src.utils.artifacts import experiment_dir, save_json
from src.utils.config import load_config


def enable_mc_dropout(model: nn.Module) -> None:
    """eval() for everything, then re-enable Dropout layers only (keeps BN frozen)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


def main(config_path: str, checkpoint: str, split_name: str = "test", n_samples: int = 20) -> dict:
    cfg = load_config(config_path)
    split = load_split(cfg["data"]["split_file"])
    ds = BraTSDataset(cfg["data"]["data_root"], patient_ids=split[split_name])
    loader = DataLoader(
        ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 2),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(
        cfg["model"]["arch"],
        cfg["model"]["in_channels"],
        cfg["model"]["out_channels"],
        cfg["model"].get("base_channels", 64),
        cfg["model"].get("dropout", 0.2),
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    enable_mc_dropout(model)

    summaries = []
    correlations: list[float] = []
    start = time.time()
    eps = 1e-12
    num_classes = cfg["model"]["out_channels"]
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            probs_sum = None
            for _ in range(n_samples):
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                probs_sum = probs if probs_sum is None else probs_sum + probs
            mean_probs = probs_sum / n_samples
            # Predictive entropy as the per-pixel uncertainty map
            entropy = -(mean_probs * torch.log(mean_probs.clamp(min=eps))).sum(dim=1).cpu()
            preds = mean_probs.argmax(dim=1).cpu()
            for pred, mask, unc in zip(preds, masks, entropy, strict=False):
                summaries.append(summarize_segmentation(pred, mask, num_classes))
                correlations.append(confidence_error_correlation(pred, mask, unc))

    metrics = aggregate_summaries(summaries)
    metrics["confidence_error_corr"] = float(np.nanmean(correlations))
    metrics["n_mc_samples"] = n_samples
    metrics["experiment"] = cfg["experiment"]["name"]
    metrics["split"] = split_name
    metrics["inference_time_s"] = time.time() - start
    metrics["checkpoint_path"] = str(checkpoint)

    out_dir = experiment_dir(cfg["experiment"].get("output_dir", "outputs"), cfg["experiment"]["name"])
    save_json(metrics, out_dir / f"{split_name}_uncertainty_metrics.json")
    print(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/uncertainty_unet.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n-samples", type=int, default=20)
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.split, args.n_samples)
