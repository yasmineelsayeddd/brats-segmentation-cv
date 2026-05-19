"""Run reproducible classical baselines on prepared BraTS slices."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classical import run_classical_methods
from src.data.brats import BraTSDataset
from src.data.splits import load_split
from src.utils.artifacts import experiment_dir, save_rows_csv
from src.utils.config import load_config


def main(config_path: str, split_name: str = "test", limit: int | None = None) -> list[dict]:
    cfg = load_config(config_path)
    split = load_split(cfg["data"]["split_file"])
    ds = BraTSDataset(cfg["data"]["data_root"], patient_ids=split[split_name])
    rows = []
    n = len(ds) if limit is None else min(limit, len(ds))
    for idx in range(n):
        image, mask = ds[idx]
        for row in run_classical_methods(image.numpy(), mask.numpy()):
            row = dict(row)
            row["slice_index"] = idx
            row["split"] = split_name
            rows.append(row)
    out = experiment_dir(cfg["experiment"].get("output_dir", "outputs"), "classical_baselines")
    save_rows_csv(rows, out / f"{split_name}_classical_metrics.csv")
    print(f"Saved {len(rows)} rows to {out}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(args.config, args.split, args.limit)
