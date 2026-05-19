"""Augmentation pipeline for BraTS 2D slices."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import albumentations as A
except ImportError:
    A = None


def get_train_transforms(image_size: int = 240) -> Any:
    """Intensity-only augmentation via albumentations.

    Spatial transforms (flip, rotate) are applied directly in AlbumentationsWrapper
    using numpy to avoid cv2.warpAffine failures with 4-channel float32 images.
    """
    if A is None:
        raise ImportError("albumentations is required for training augmentations.")
    return A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.GaussNoise(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ],
    )


def get_val_transforms(image_size: int = 240) -> None:
    return None


class AlbumentationsWrapper:
    """Wrap transforms for (C,H,W) image and (H,W) mask arrays.

    Spatial transforms run in numpy (works with any channel count).
    Intensity transforms run through albumentations.
    """

    def __init__(self, transforms: Any | None) -> None:
        self.transforms = transforms

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.transforms is None:
            return image, mask

        # ── Spatial transforms (numpy — safe for 4-channel images) ──────────
        if np.random.random() < 0.5:                          # horizontal flip
            image = image[:, :, ::-1].copy()
            mask  = mask[:, ::-1].copy()
        if np.random.random() < 0.2:                          # vertical flip
            image = image[:, ::-1, :].copy()
            mask  = mask[::-1, :].copy()
        if np.random.random() < 0.3:                          # 90° rotation
            k = np.random.randint(1, 4)
            image = np.rot90(image, k, axes=(1, 2)).copy()
            mask  = np.rot90(mask,  k, axes=(0, 1)).copy()

        # ── Intensity transforms (albumentations, no cv2 geometric ops) ─────
        img_hwc = image.transpose(1, 2, 0).astype(np.float32)
        result  = self.transforms(image=img_hwc, mask=mask)
        return result["image"].transpose(2, 0, 1), result["mask"]
