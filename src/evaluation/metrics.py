"""Segmentation evaluation metrics for BraTS-style multi-class masks."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage

CLASS_NAMES = ["background", "NCR/NET", "edema", "enhancing_tumor"]


def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def dice_per_class(
    pred: np.ndarray | torch.Tensor,
    gt: np.ndarray | torch.Tensor,
    num_classes: int = 4,
    smooth: float = 1e-6,
    ignore_background: bool = True,
) -> np.ndarray:
    """Compute Dice coefficient per class."""
    pred = _to_numpy(pred)
    gt = _to_numpy(gt)
    scores = []
    for c in range(1 if ignore_background else 0, num_classes):
        p = pred == c
        g = gt == c
        intersection = np.logical_and(p, g).sum()
        scores.append((2 * intersection + smooth) / (p.sum() + g.sum() + smooth))
    return np.array(scores, dtype=np.float32)


def iou_per_class(
    pred: np.ndarray | torch.Tensor,
    gt: np.ndarray | torch.Tensor,
    num_classes: int = 4,
    smooth: float = 1e-6,
    ignore_background: bool = True,
) -> np.ndarray:
    """Intersection-over-Union per class."""
    pred = _to_numpy(pred)
    gt = _to_numpy(gt)
    scores = []
    for c in range(1 if ignore_background else 0, num_classes):
        p = pred == c
        g = gt == c
        intersection = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        scores.append((intersection + smooth) / (union + smooth))
    return np.array(scores, dtype=np.float32)


def _surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask.astype(bool)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    return np.logical_xor(mask, eroded)


def hd95_binary(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float] = (1.0, 1.0)) -> float:
    """Symmetric 95th percentile Hausdorff distance for binary masks.

    Returns NaN when either mask is empty, matching common medical segmentation
    reporting practice for undefined class distances.
    """
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    if not pred.any() or not gt.any():
        return float("nan")
    pred_surface = _surface(pred)
    gt_surface = _surface(gt)
    dt_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    dt_gt = ndimage.distance_transform_edt(~gt_surface, sampling=spacing)
    distances = np.concatenate([dt_gt[pred_surface], dt_pred[gt_surface]])
    return float(np.percentile(distances, 95))


def hd95_per_class(
    pred: np.ndarray | torch.Tensor,
    gt: np.ndarray | torch.Tensor,
    num_classes: int = 4,
    ignore_background: bool = True,
    spacing: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    pred = _to_numpy(pred)
    gt = _to_numpy(gt)
    values = [
        hd95_binary(pred == c, gt == c, spacing=spacing)
        for c in range(1 if ignore_background else 0, num_classes)
    ]
    return np.array(values, dtype=np.float32)


def mean_dice(
    pred: np.ndarray | torch.Tensor,
    gt: np.ndarray | torch.Tensor,
    num_classes: int = 4,
    ignore_background: bool = True,
) -> float:
    return float(np.nanmean(dice_per_class(pred, gt, num_classes, ignore_background=ignore_background)))


def mean_iou(
    pred: np.ndarray | torch.Tensor,
    gt: np.ndarray | torch.Tensor,
    num_classes: int = 4,
    ignore_background: bool = True,
) -> float:
    return float(np.nanmean(iou_per_class(pred, gt, num_classes, ignore_background=ignore_background)))


def pixel_accuracy(pred: np.ndarray | torch.Tensor, gt: np.ndarray | torch.Tensor) -> float:
    pred = _to_numpy(pred)
    gt = _to_numpy(gt)
    return float((pred == gt).mean())


@dataclass(frozen=True)
class MetricSummary:
    """Flat metric bundle ready for CSV/JSON output."""

    mean_dice: float
    mean_iou: float
    pixel_accuracy: float
    dice_per_class: list[float]
    iou_per_class: list[float]
    hd95_per_class: list[float]

    def to_flat_dict(self, prefix: str = "") -> dict[str, float]:
        names = CLASS_NAMES[1:]
        out: dict[str, float] = {
            f"{prefix}mean_dice": self.mean_dice,
            f"{prefix}mean_iou": self.mean_iou,
            f"{prefix}pixel_accuracy": self.pixel_accuracy,
        }
        for name, dice, iou, hd in zip(names, self.dice_per_class, self.iou_per_class, self.hd95_per_class, strict=True):
            out[f"{prefix}dice_{name}"] = float(dice)
            out[f"{prefix}iou_{name}"] = float(iou)
            out[f"{prefix}hd95_{name}"] = float(hd)
        return out


def summarize_segmentation(
    pred: np.ndarray | torch.Tensor,
    gt: np.ndarray | torch.Tensor,
    num_classes: int = 4,
) -> MetricSummary:
    dice = dice_per_class(pred, gt, num_classes=num_classes)
    iou = iou_per_class(pred, gt, num_classes=num_classes)
    hd95 = hd95_per_class(pred, gt, num_classes=num_classes)
    return MetricSummary(
        mean_dice=float(np.nanmean(dice)),
        mean_iou=float(np.nanmean(iou)),
        pixel_accuracy=pixel_accuracy(pred, gt),
        dice_per_class=[float(x) for x in dice],
        iou_per_class=[float(x) for x in iou],
        hd95_per_class=[float(x) for x in hd95],
    )


def aggregate_summaries(summaries: Iterable[MetricSummary]) -> dict[str, float]:
    rows = [s.to_flat_dict() for s in summaries]
    if not rows:
        raise ValueError("Cannot aggregate an empty metric list")
    keys = rows[0].keys()
    return {key: float(np.nanmean([row[key] for row in rows])) for key in keys}


def confidence_error_correlation(
    pred: np.ndarray | torch.Tensor,
    gt: np.ndarray | torch.Tensor,
    uncertainty: np.ndarray | torch.Tensor,
) -> float:
    """Correlation between uncertainty and binary error map."""
    pred_np = _to_numpy(pred).ravel()
    gt_np = _to_numpy(gt).ravel()
    unc = _to_numpy(uncertainty).ravel().astype(np.float64)
    err = (pred_np != gt_np).astype(np.float64)
    if np.std(unc) < 1e-12 or np.std(err) < 1e-12:
        return 0.0
    return float(np.corrcoef(unc, err)[0, 1])
