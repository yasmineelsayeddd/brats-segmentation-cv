"""PyTorch Dataset for prepared BraTS 2020 2D slices."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    """Loads 2D BraTS slices produced by scripts/prepare_data.py.

    Each sample:
        image: torch.FloatTensor of shape (4, H, W) — modalities [flair, t1, t1ce, t2]
        mask:  torch.LongTensor of shape (H, W) — labels in {0, 1, 2, 3}
    """

    NUM_CLASSES = 4
    CLASS_NAMES = ["background", "NCR/NET", "edema", "enhancing_tumor"]

    def __init__(
        self,
        data_root: str | Path,
        patient_ids: list[str] | None = None,
        transform: Callable | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.transform = transform

        metadata_path = self.data_root / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found in {self.data_root}. "
                "Run scripts/prepare_data.py first."
            )
        with metadata_path.open() as f:
            metadata = json.load(f)

        all_slices = metadata["slices"]
        if patient_ids is not None:
            allowed = set(patient_ids)
            self.samples = [s for s in all_slices if s["patient"] in allowed]
        else:
            self.samples = all_slices

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        patient = sample["patient"]
        slice_idx = sample["slice"]

        image_path = self.data_root / patient / f"slice_{slice_idx:03d}_image.npy"
        mask_path = self.data_root / patient / f"slice_{slice_idx:03d}_mask.npy"

        image = np.load(image_path)  # (4, H, W)
        mask = np.load(mask_path)    # (H, W)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).long()
        return image_tensor, mask_tensor

    @property
    def patient_ids(self) -> list[str]:
        """Unique patient IDs present in this dataset."""
        return sorted({s["patient"] for s in self.samples})
