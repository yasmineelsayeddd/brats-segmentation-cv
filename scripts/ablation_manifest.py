"""Write the required ablation manifest for report tracking."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.artifacts import save_rows_csv

REQUIRED_EXPERIMENTS = [
    "classical_baselines",
    "unet_baseline",
    "unet_best_fusion_loss_aug",
    "unetpp",
    "attention_unet",
    "uncertainty_unet",
    "yolo_detector",
    "yolo_unet_cascade",
    "tta_morph_postprocess",
    "gradcam_visuals",
]


def main(output: str = "outputs/ablation_manifest.csv") -> None:
    rows = [{"experiment": name, "required": True, "status": "pending"} for name in REQUIRED_EXPERIMENTS]
    save_rows_csv(rows, Path(output))
    print(f"Saved ablation manifest to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/ablation_manifest.csv")
    args = parser.parse_args()
    main(args.output)
