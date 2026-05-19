"""Plotting helpers for BraTS slices."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from src.data.brats import BraTSDataset

MASK_CMAP = ListedColormap(
    [
        (0, 0, 0, 0),        # background — transparent
        (0.9, 0.2, 0.2, 0.6),  # NCR/NET — red
        (0.2, 0.8, 0.2, 0.6),  # edema — green
        (0.9, 0.9, 0.2, 0.6),  # enhancing tumor — yellow
    ]
)


def plot_modalities(image: np.ndarray, mask: np.ndarray | None = None, title: str = ""):
    """Plot the 4 modalities of a single slice side-by-side, optionally with mask overlay.

    Args:
        image: (4, H, W) array — modalities [flair, t1, t1ce, t2].
        mask: Optional (H, W) array with class indices.
        title: Figure title.
    """
    modality_names = ["FLAIR", "T1", "T1ce", "T2"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, ax in enumerate(axes):
        ax.imshow(image[i], cmap="gray")
        if mask is not None:
            ax.imshow(mask, cmap=MASK_CMAP, vmin=0, vmax=3, interpolation="none")
        ax.set_title(modality_names[i])
        ax.axis("off")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_sample_grid(dataset: BraTSDataset, n: int = 8, seed: int = 0):
    """Plot a grid of n random samples with mask overlays."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        image, mask = dataset[idx]
        image_np = image.numpy()
        mask_np = mask.numpy()
        for col, modality_name in enumerate(["FLAIR", "T1", "T1ce", "T2"]):
            ax = axes[row, col]
            ax.imshow(image_np[col], cmap="gray")
            ax.imshow(mask_np, cmap=MASK_CMAP, vmin=0, vmax=3, interpolation="none")
            if row == 0:
                ax.set_title(modality_name)
            ax.axis("off")

    plt.tight_layout()
    return fig


def class_distribution(dataset: BraTSDataset) -> dict[int, int]:
    """Count pixels per class across the entire dataset (slow — use a subset for EDA)."""
    counts = {i: 0 for i in range(dataset.NUM_CLASSES)}
    for i in range(len(dataset)):
        _, mask = dataset[i]
        for cls, cnt in zip(*np.unique(mask.numpy(), return_counts=True), strict=False):
            counts[int(cls)] += int(cnt)
    return counts
