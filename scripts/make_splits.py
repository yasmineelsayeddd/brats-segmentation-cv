"""Generate a patient-level train/val/test split JSON from prepared BraTS data.

Usage:
    python scripts/make_splits.py \
        --data-root data/brats2020_2d \
        --output    configs/splits/default.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.splits import (
    get_patient_ids_from_metadata,
    patient_level_split,
    save_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val/test split file.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    patient_ids = get_patient_ids_from_metadata(args.data_root)
    split = patient_level_split(
        patient_ids,
        ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        seed=args.seed,
    )
    save_split(split, args.output)

    print(f"Total patients: {len(patient_ids)}")
    for name, ids in split.items():
        print(f"  {name}: {len(ids)} patients")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
