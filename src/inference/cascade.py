"""YOLO detection followed by U-Net segmentation in the detected ROI."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from src.data.yolo import mask_to_bbox
from src.inference.segmentation import SegmentationResult, segment_slice

BBox = tuple[int, int, int, int]
Detector = Callable[[np.ndarray], BBox | None]


def expand_bbox(bbox: BBox, shape: tuple[int, int], margin: int = 16) -> BBox:
    x1, y1, x2, y2 = bbox
    h, w = shape
    return max(x1 - margin, 0), max(y1 - margin, 0), min(x2 + margin, w), min(y2 + margin, h)


def crop_to_bbox(image: np.ndarray, bbox: BBox) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return image[:, y1:y2, x1:x2]


def paste_roi(roi_mask: np.ndarray, full_shape: tuple[int, int], bbox: BBox) -> np.ndarray:
    out = np.zeros(full_shape, dtype=roi_mask.dtype)
    x1, y1, x2, y2 = bbox
    out[y1:y2, x1:x2] = roi_mask[: y2 - y1, : x2 - x1]
    return out


def bbox_from_mask_or_none(mask: np.ndarray, margin: int = 16) -> BBox | None:
    return mask_to_bbox(mask, margin=margin)


def cascade_segment(
    segmenter: torch.nn.Module,
    image: np.ndarray,
    detector: Detector | None = None,
    device: str = "cpu",
    margin: int = 16,
) -> SegmentationResult:
    """Run detector-crop-segment-reconstruct.

    If detection fails, the function falls back to full-image segmentation.
    """
    if image.ndim != 3:
        raise ValueError("image must have shape (C,H,W)")
    bbox = detector(image) if detector is not None else None
    if bbox is None:
        return segment_slice(segmenter, image, device=device)
    bbox = expand_bbox(bbox, image.shape[-2:], margin=margin)
    roi = crop_to_bbox(image, bbox)
    roi_result = segment_slice(segmenter, roi, device=device)
    full_mask = paste_roi(roi_result.mask, image.shape[-2:], bbox)
    full_probs = np.zeros((roi_result.probabilities.shape[0], *image.shape[-2:]), dtype=np.float32)
    x1, y1, x2, y2 = bbox
    full_probs[:, y1:y2, x1:x2] = roi_result.probabilities[:, : y2 - y1, : x2 - x1]
    full_probs[0, full_probs.sum(axis=0) == 0] = 1.0
    confidence = full_probs.max(axis=0).astype(np.float32)
    return SegmentationResult(mask=full_mask, probabilities=full_probs, confidence=confidence)
