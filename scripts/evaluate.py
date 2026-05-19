"""Evaluate a trained segmentation checkpoint."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.data.brats import BraTSDataset
from src.data.splits import load_split
from src.evaluation.metrics import aggregate_summaries, summarize_segmentation
from src.models import build_model
from src.utils.artifacts import experiment_dir, save_json, save_rows_csv
from src.utils.config import load_config


def main(config_path: str, checkpoint: str, split_name: str = "test") -> dict[str, float]:
    cfg = load_config(config_path)
    split = load_split(cfg["data"]["split_file"])
    ds = BraTSDataset(cfg["data"]["data_root"], patient_ids=split[split_name])
    loader = DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"].get("num_workers", 2))
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
    model.eval()

    summaries = []
    start = time.time()
    with torch.no_grad():
        for images, masks in loader:
            logits = model(images.to(device))
            preds = logits.argmax(dim=1).cpu()
            for pred, mask in zip(preds, masks, strict=False):
                summaries.append(summarize_segmentation(pred, mask, cfg["model"]["out_channels"]))
    metrics = aggregate_summaries(summaries)
    metrics.update(
        {
            "experiment": cfg["experiment"]["name"],
            "split": split_name,
            "inference_time_s": time.time() - start,
            "checkpoint_path": str(checkpoint),
        }
    )
    out_dir = experiment_dir(cfg["experiment"].get("output_dir", "outputs"), cfg["experiment"]["name"])
    save_json(metrics, out_dir / f"{split_name}_metrics.json")
    save_rows_csv([metrics], out_dir / f"{split_name}_metrics.csv")
    print(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.split)
