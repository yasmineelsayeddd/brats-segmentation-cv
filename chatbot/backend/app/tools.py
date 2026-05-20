from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import ml_config


def _get_device() -> str:
    import torch
    cfg = ml_config().get("device", "auto")
    if cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def _load_model():
    import torch
    from src.models.unet import build_model

    cfg = ml_config()
    checkpoint_path = cfg.get("checkpoint_path")
    num_classes = cfg.get("num_classes", 4)

    model = build_model(arch="unet", in_channels=4, out_channels=num_classes)
    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)
        print(f"[tools] Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"[tools] No checkpoint at {checkpoint_path}, using untrained weights")

    model.eval()
    return model


_model_cache = None


def _model():
    global _model_cache
    if _model_cache is None:
        _model_cache = _load_model()
    return _model_cache


def segment_scan(image_path: str) -> str:
    """Segment a brain MRI slice and return tumor region statistics.

    Args:
        image_path: Path to a .npy file with shape (4, H, W) containing
                    [flair, t1, t1ce, t2] modalities.
    """
    import torch
    from src.inference.segmentation import segment_slice

    image = np.load(image_path)
    if image.shape[0] != 4:
        return "Error: Expected 4-channel image (flair, t1, t1ce, t2)"

    result = segment_slice(_model(), image, device=_get_device())

    class_names = ["background", "NCR/NET", "edema", "enhancing_tumor"]
    total_pixels = result.mask.size
    stats = {}
    for i, name in enumerate(class_names):
        count = int((result.mask == i).sum())
        stats[name] = {
            "pixel_count": count,
            "percentage": round(count / total_pixels * 100, 2),
        }

    stats["mean_confidence"] = round(float(result.confidence.mean()), 4)
    stats["output_shape"] = list(result.mask.shape)

    return json.dumps(stats, indent=2)


def analyze_uncertainty(image_path: str, passes: int = 30) -> str:
    """Run MC-dropout uncertainty analysis on a brain MRI slice.

    Returns predictive entropy map statistics showing where the model is uncertain.

    Args:
        image_path: Path to a .npy file with shape (4, H, W).
        passes: Number of MC dropout passes (default 30).
    """
    from src.inference.uncertainty import mc_dropout_predict

    image = np.load(image_path)
    result = mc_dropout_predict(_model(), image, device=_get_device(), passes=passes)

    if result.uncertainty is None:
        return "Error: No uncertainty map generated"

    unc = result.uncertainty
    stats = {
        "mean_entropy": round(float(unc.mean()), 4),
        "max_entropy": round(float(unc.max()), 4),
        "std_entropy": round(float(unc.std()), 4),
        "high_uncertainty_pct": round(float((unc > np.percentile(unc, 90)).sum() / unc.size * 100), 2),
        "output_shape": list(unc.shape),
    }

    return json.dumps(stats, indent=2)


def compute_metrics(pred_path: str, gt_path: str) -> str:
    """Compute segmentation metrics (Dice, IoU, HD95) against ground truth.

    Args:
        pred_path: Path to predicted mask .npy file (H, W).
        gt_path: Path to ground truth mask .npy file (H, W).
    """
    from src.evaluation.metrics import summarize_segmentation

    pred = np.load(pred_path)
    gt = np.load(gt_path)

    summary = summarize_segmentation(pred, gt, num_classes=ml_config().get("num_classes", 4))
    result = {
        "mean_dice": round(summary.mean_dice, 4),
        "mean_iou": round(summary.mean_iou, 4),
        "pixel_accuracy": round(summary.pixel_accuracy, 4),
        "per_class": {},
    }

    class_names = ["NCR/NET", "edema", "enhancing_tumor"]
    for name, dice, iou, hd in zip(class_names, summary.dice_per_class, summary.iou_per_class, summary.hd95_per_class):
        result["per_class"][name] = {
            "dice": round(dice, 4),
            "iou": round(iou, 4),
            "hd95": round(hd, 4) if not np.isnan(hd) else "N/A",
        }

    return json.dumps(result, indent=2)


def cascade_detect(image_path: str) -> str:
    """Run cascade detection (YOLO crop + U-Net segmentation) on a brain MRI slice.

    Uses a mask-based detector as fallback if YOLO is not available.

    Args:
        image_path: Path to a .npy file with shape (4, H, W).
    """
    from src.inference.cascade import cascade_segment

    image = np.load(image_path)
    result = cascade_segment(_model(), image, detector=None, device=_get_device())

    class_names = ["background", "NCR/NET", "edema", "enhancing_tumor"]
    total_pixels = result.mask.size
    stats = {
        "method": "cascade (full-image fallback)",
        "mean_confidence": round(float(result.confidence.mean()), 4),
        "regions": {},
    }

    for i, name in enumerate(class_names):
        count = int((result.mask == i).sum())
        stats["regions"][name] = {
            "pixel_count": count,
            "percentage": round(count / total_pixels * 100, 2),
        }

    return json.dumps(stats, indent=2)


def explain_findings(metrics_json: str) -> str:
    """Provide clinical interpretation of segmentation metrics.

    Args:
        metrics_json: JSON string of metrics from compute_metrics or segment_scan.
    """
    try:
        data = json.loads(metrics_json)
    except json.JSONDecodeError:
        return "Error: Invalid JSON input"

    findings = []

    if "mean_dice" in data:
        dice = data["mean_dice"]
        if dice >= 0.85:
            findings.append(f"Overall Dice score of {dice:.2f} indicates excellent segmentation quality.")
        elif dice >= 0.70:
            findings.append(f"Dice score of {dice:.2f} shows good segmentation with minor boundary discrepancies.")
        else:
            findings.append(f"Dice score of {dice:.2f} suggests the model may need refinement on this case.")

    if "per_class" in data:
        for cls, metrics in data["per_class"].items():
            d = metrics.get("dice", 0)
            if isinstance(d, (int, float)):
                if d >= 0.80:
                    findings.append(f"{cls} region well-delineated (Dice: {d:.2f}).")
                elif d >= 0.60:
                    findings.append(f"{cls} region moderately segmented (Dice: {d:.2f}). Consider manual review.")
                else:
                    findings.append(f"{cls} region poorly segmented (Dice: {d:.2f}). High uncertainty expected.")

    if "mean_entropy" in data:
        ent = data["mean_entropy"]
        if ent > 0.5:
            findings.append("High predictive entropy detected. Model shows significant uncertainty — manual verification recommended.")
        elif ent > 0.2:
            findings.append("Moderate uncertainty present. Review high-entropy regions carefully.")
        else:
            findings.append("Low predictive entropy. Model predictions are confident.")

    if "enhancing_tumor" in data.get("regions", {}):
        tumor_pct = data["regions"]["enhancing_tumor"].get("percentage", 0)
        if tumor_pct > 5:
            findings.append(f"Enhancing tumor occupies {tumor_pct:.1f}% of slice — substantial tumor burden.")
        elif tumor_pct > 1:
            findings.append(f"Enhancing tumor occupies {tumor_pct:.1f}% of slice — moderate tumor burden.")
        elif tumor_pct > 0:
            findings.append(f"Small enhancing tumor region detected ({tumor_pct:.1f}% of slice).")
        else:
            findings.append("No enhancing tumor detected in this slice.")

    if not findings:
        findings.append("No interpretable metrics found in input.")

    return "\n".join(findings)
