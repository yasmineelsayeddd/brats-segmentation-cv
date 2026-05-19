"""Reproducible classical segmentation baselines."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from src.evaluation.metrics import summarize_segmentation


def _threshold_otsu(image: np.ndarray) -> float:
    try:
        from skimage.filters import threshold_otsu

        return float(threshold_otsu(image))
    except ImportError:
        hist, edges = np.histogram(image.ravel(), bins=128)
        total = image.size
        sum_total = (hist * edges[:-1]).sum()
        weight_bg = 0.0
        sum_bg = 0.0
        max_var = -1.0
        threshold = float(image.mean())
        for count, edge in zip(hist, edges[:-1], strict=False):
            weight_bg += count
            if weight_bg == 0:
                continue
            weight_fg = total - weight_bg
            if weight_fg == 0:
                break
            sum_bg += count * edge
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_total - sum_bg) / weight_fg
            between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
            if between > max_var:
                max_var = between
                threshold = float(edge)
        return threshold


def _remove_small(binary: np.ndarray, min_size: int = 32) -> np.ndarray:
    labels, n = ndimage.label(binary)
    out = np.zeros_like(binary, dtype=bool)
    for label in range(1, n + 1):
        component = labels == label
        if component.sum() >= min_size:
            out |= component
    return out


def otsu_baseline(image: np.ndarray) -> np.ndarray:
    t1ce = image[2]
    binary = _remove_small(t1ce > _threshold_otsu(t1ce), min_size=64)
    return binary.astype(np.uint8)


def multi_otsu_baseline(image: np.ndarray) -> np.ndarray:
    t1ce = image[2]
    try:
        from skimage.filters import threshold_multiotsu

        thresholds = threshold_multiotsu(t1ce, classes=3)
        binary = t1ce > thresholds[-1]
    except ImportError:
        binary = t1ce > np.percentile(t1ce, 85)
    return _remove_small(binary, min_size=32).astype(np.uint8)


def region_growing_baseline(image: np.ndarray) -> np.ndarray:
    t1ce = image[2]
    seed = np.unravel_index(np.argmax(t1ce), t1ce.shape)
    seed_value = t1ce[seed]
    tolerance = max(float(t1ce.std()) * 0.35, 1e-6)
    binary = np.abs(t1ce - seed_value) <= tolerance
    labels, _ = ndimage.label(binary)
    return (labels == labels[seed]).astype(np.uint8)


def watershed_baseline(image: np.ndarray) -> np.ndarray:
    t1ce = ndimage.gaussian_filter(image[2], sigma=1.5)
    foreground = t1ce > _threshold_otsu(t1ce)
    distance = ndimage.distance_transform_edt(foreground)
    markers, _ = ndimage.label(distance > np.percentile(distance[distance > 0], 90) if distance.any() else foreground)
    try:
        from skimage.segmentation import watershed

        labels = watershed(-distance, markers, mask=foreground)
    except ImportError:
        labels = markers
    if labels.max() == 0:
        return foreground.astype(np.uint8)
    sizes = [(labels == label).sum() for label in range(1, labels.max() + 1)]
    best = int(np.argmax(sizes)) + 1
    return (labels == best).astype(np.uint8)


def canny_morph_baseline(image: np.ndarray) -> np.ndarray:
    t1ce = image[2]
    try:
        from skimage.feature import canny

        edges = canny(t1ce, sigma=1.0)
    except ImportError:
        gx = ndimage.sobel(t1ce, axis=1)
        gy = ndimage.sobel(t1ce, axis=0)
        edges = np.hypot(gx, gy) > np.percentile(np.hypot(gx, gy), 90)
    closed = ndimage.binary_fill_holes(ndimage.binary_dilation(edges, iterations=2))
    bright = t1ce > np.percentile(t1ce, 75)
    return _remove_small(closed & bright, min_size=32).astype(np.uint8)


METHODS = {
    "otsu": otsu_baseline,
    "multi_otsu": multi_otsu_baseline,
    "region_growing": region_growing_baseline,
    "watershed": watershed_baseline,
    "canny_morph": canny_morph_baseline,
}


def run_classical_methods(image: np.ndarray, mask: np.ndarray) -> list[dict[str, float | str]]:
    """Run all classical methods against a binary tumor target."""
    gt = (mask > 0).astype(np.uint8)
    rows: list[dict[str, float | str]] = []
    for name, method in METHODS.items():
        pred = method(image)
        summary = summarize_segmentation(pred, gt, num_classes=2)
        row: dict[str, float | str] = {"experiment": name}
        row.update(summary.to_flat_dict())
        rows.append(row)
    return rows
