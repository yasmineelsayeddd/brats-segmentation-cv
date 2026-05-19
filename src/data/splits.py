"""Patient-level train/val/test splits for BraTS.

Splitting must happen at the patient level (not slice level). If the same patient
appears in both train and validation, the model can memorise patient-specific
anatomy and validation metrics become a lie.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def patient_level_split(
    patient_ids: list[str],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split patients into train/val/test groups.

    Args:
        patient_ids: All patient IDs to split.
        ratios: (train, val, test) fractions, must sum to 1.0.
        seed: Reproducibility seed.

    Returns:
        Dict with keys "train", "val", "test" mapping to lists of patient IDs.
    """
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")

    rng = np.random.default_rng(seed)
    shuffled = list(patient_ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    return {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train : n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val :]),
    }


def save_split(split: dict[str, list[str]], path: str | Path) -> None:
    """Persist a split to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        json.dump(split, f, indent=2)


def load_split(path: str | Path) -> dict[str, list[str]]:
    """Load a previously saved split."""
    with Path(path).open() as f:
        return json.load(f)


def get_patient_ids_from_metadata(data_root: str | Path) -> list[str]:
    """Read the unique patient IDs from a prepared data folder's metadata.json."""
    metadata_path = Path(data_root) / "metadata.json"
    with metadata_path.open() as f:
        metadata = json.load(f)
    return sorted({s["patient"] for s in metadata["slices"]})
