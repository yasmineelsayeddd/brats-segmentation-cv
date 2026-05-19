"""Morphological post-processing for segmentation masks."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def clean_segmentation(mask: np.ndarray, min_size: int = 32, closing_iters: int = 1) -> np.ndarray:
    """Remove tiny islands and close small holes class-by-class."""
    mask = np.asarray(mask)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    structure = np.ones((3, 3), dtype=bool)
    for cls in sorted(int(c) for c in np.unique(mask) if c != 0):
        binary = mask == cls
        if closing_iters > 0:
            binary = ndimage.binary_closing(binary, structure=structure, iterations=closing_iters)
        labels, n = ndimage.label(binary)
        for label in range(1, n + 1):
            component = labels == label
            if component.sum() >= min_size:
                cleaned[component] = cls
    return cleaned
