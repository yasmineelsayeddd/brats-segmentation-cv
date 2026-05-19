"""Classical hierarchical segmentation pipeline (pure course content).

Mirrors the structure of the deep hierarchical cascade but replaces each
binary U-Net with a classical pipeline using techniques from Lectures 2
(Otsu thresholding, region selection) and 3 (mathematical morphology).

  Stage 1 (WT): Otsu threshold on FLAIR + morphological opening/closing,
                keep the largest connected component.
  Stage 2 (TC): Otsu threshold on T1ce WITHIN the WT mask + morphology.
  Stage 3 (ET): Higher-percentile threshold on T1ce WITHIN the TC mask +
                morphology.

Reports binary Dice/IoU/HD95 per region against the ground truth
hierarchical mask. Zero deep learning, zero training. The output table
gives a direct apples-to-apples comparison with the deep hierarchical
cascade — same hierarchical structure, different operators.

Course references:
  L1 — intensity normalization (per-modality 1-99% rescale)
  L2 — Otsu thresholding (skimage.filters.threshold_otsu)
  L3 — morphological opening, closing, connected components
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import (
    binary_closing,
    binary_opening,
    disk,
    remove_small_objects,
)
from torch.utils.data import DataLoader

from src.data.brats import BraTSDataset
from src.data.splits import load_split
from src.evaluation.metrics import dice_per_class, hd95_binary, iou_per_class
from src.utils.artifacts import save_json
from src.utils.config import load_config

# Channel order in BraTSDataset 4-channel input (verify against data prep)
FLAIR_CHANNEL = 0
T1CE_CHANNEL = 2


def _normalize_01(arr: np.ndarray) -> np.ndarray:
    """Per-image 1-99% rescaling to [0, 1] (L1 intensity normalization)."""
    lo, hi = np.percentile(arr, [1, 99])
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component (L3 connected components)."""
    if not mask.any():
        return mask
    labeled, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    return labeled == largest


def _morph_cleanup(mask: np.ndarray, open_r: int = 1, close_r: int = 2, min_size: int = 16) -> np.ndarray:
    """Opening (remove specks) + closing (fill small holes) + small-object removal."""
    if not mask.any():
        return mask
    cleaned = binary_opening(mask, disk(open_r))
    cleaned = binary_closing(cleaned, disk(close_r))
    cleaned = remove_small_objects(cleaned, min_size=min_size)
    return cleaned


def _safe_otsu(arr: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Otsu threshold (L2). Restricts to `mask` if given; falls back to image median."""
    if mask is not None:
        values = arr[mask]
    else:
        values = arr.ravel()
    if values.size == 0 or values.max() - values.min() < 1e-6:
        return float(np.median(arr))
    try:
        return float(threshold_otsu(values))
    except Exception:
        return float(np.median(values))


def classical_hierarchical_segment(
    image_4c: np.ndarray,
    wt_percentile: float = 90.0,
    tc_percentile: float = 55.0,
    et_percentile: float = 60.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the 3-stage classical pipeline on one 4-channel slice.

    Returns (WT_mask, TC_mask, ET_mask) as boolean arrays of shape (H, W),
    each nested inside the previous (ET ⊂ TC ⊂ WT).

    Note: naive Otsu on a full BraTS slice finds the brain-vs-background
    split, not tumor-vs-brain (brain itself is the dominant bright region
    on FLAIR). We therefore (a) build a brain mask from non-zero pixels
    (BraTS is already skull-stripped, background is ~0), then (b) take
    the top X% of intensities within the relevant mask at each stage.
    """
    flair = _normalize_01(image_4c[FLAIR_CHANNEL])
    t1ce = _normalize_01(image_4c[T1CE_CHANNEL])

    # Brain mask — BraTS data is skull-stripped, background pixels are ~0
    brain_mask = flair > 0.05
    if not brain_mask.any():
        empty = np.zeros_like(flair, dtype=bool)
        return empty, empty, empty

    # --- Stage 1 (WT) — top wt_percentile% of FLAIR intensities within brain ---
    # Tumor + edema are the brightest tier on FLAIR
    t_wt = float(np.percentile(flair[brain_mask], wt_percentile))
    wt = (flair > t_wt) & brain_mask
    wt = _morph_cleanup(wt, open_r=1, close_r=2, min_size=32)
    wt = _largest_component(wt)

    # --- Stage 2 (TC) — top tc_percentile% of T1ce intensities within WT ---
    # Tumor core (NCR + enhancing) is brighter on T1ce than edema
    if not wt.any():
        empty = np.zeros_like(wt)
        return wt, empty, empty
    t_tc = float(np.percentile(t1ce[wt], tc_percentile))
    tc = (t1ce > t_tc) & wt
    tc = _morph_cleanup(tc, open_r=1, close_r=1, min_size=8)

    # --- Stage 3 (ET) — top et_percentile% of T1ce within TC ---
    # Enhancing tumor is the brightest on T1ce (gadolinium uptake)
    if not tc.any():
        empty = np.zeros_like(wt)
        return wt, tc, empty
    t_et = float(np.percentile(t1ce[tc], et_percentile))
    et = (t1ce > t_et) & tc
    et = _morph_cleanup(et, open_r=0, close_r=1, min_size=4)

    return wt.astype(bool), tc.astype(bool), et.astype(bool)


def gt_hierarchical(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert BraTS multi-class mask (0/1/2/3) into nested binary regions."""
    wt = mask > 0
    tc = (mask == 1) | (mask == 3)
    et = mask == 3
    return wt.astype(bool), tc.astype(bool), et.astype(bool)


def main(
    config_path: str,
    split_name: str = "test",
    wt_percentile: float = 90.0,
    tc_percentile: float = 55.0,
    et_percentile: float = 60.0,
) -> dict:
    cfg = load_config(config_path)
    split = load_split(cfg["data"]["split_file"])
    ds = BraTSDataset(cfg["data"]["data_root"], patient_ids=split[split_name])
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg["data"].get("num_workers", 2))

    region_dice: dict[str, list[float]] = {"WT": [], "TC": [], "ET": []}
    region_iou: dict[str, list[float]] = {"WT": [], "TC": [], "ET": []}
    region_hd95: dict[str, list[float]] = {"WT": [], "TC": [], "ET": []}

    start = time.time()
    for images, masks in loader:
        # batch_size=1
        img = images[0].numpy()
        gt = masks[0].numpy()
        wt_p, tc_p, et_p = classical_hierarchical_segment(
            img,
            wt_percentile=wt_percentile,
            tc_percentile=tc_percentile,
            et_percentile=et_percentile,
        )
        wt_g, tc_g, et_g = gt_hierarchical(gt)

        for name, pred, gtm in [("WT", wt_p, wt_g), ("TC", tc_p, tc_g), ("ET", et_p, et_g)]:
            # binary scoring: predicted vs ground truth as {0,1} arrays, 2 classes
            d = float(dice_per_class(pred.astype(np.uint8), gtm.astype(np.uint8),
                                     num_classes=2, ignore_background=True)[0])
            i = float(iou_per_class(pred.astype(np.uint8), gtm.astype(np.uint8),
                                    num_classes=2, ignore_background=True)[0])
            h = float(hd95_binary(pred, gtm))
            region_dice[name].append(d)
            region_iou[name].append(i)
            region_hd95[name].append(h)

    metrics: dict = {}
    for name in ("WT", "TC", "ET"):
        metrics[f"classical_{name}_dice_mean"] = float(np.nanmean(region_dice[name]))
        metrics[f"classical_{name}_iou_mean"] = float(np.nanmean(region_iou[name]))
        metrics[f"classical_{name}_hd95_mean"] = float(np.nanmean(region_hd95[name]))
    metrics["pipeline"] = (
        f"P{wt_percentile:.0f}(FLAIR|brain)->WT, P{tc_percentile:.0f}(T1ce|WT)->TC, "
        f"P{et_percentile:.0f}(T1ce|TC)->ET, with opening+closing+component-filter"
    )
    metrics["split"] = split_name
    metrics["wt_percentile"] = wt_percentile
    metrics["tc_percentile"] = tc_percentile
    metrics["et_percentile"] = et_percentile
    metrics["num_slices"] = len(ds)
    metrics["inference_time_s"] = time.time() - start

    out_path = Path(cfg["experiment"].get("output_dir", "outputs")) / "classical_hierarchical" / f"{split_name}_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(metrics, out_path)
    print(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--wt-percentile", type=float, default=90.0,
                        help="Percentile threshold on FLAIR within brain for the whole-tumor stage")
    parser.add_argument("--tc-percentile", type=float, default=55.0,
                        help="Percentile threshold on T1ce within WT for the tumor-core stage")
    parser.add_argument("--et-percentile", type=float, default=60.0,
                        help="Percentile threshold on T1ce within TC for the enhancing-tumor stage")
    args = parser.parse_args()
    main(args.config, args.split, args.wt_percentile, args.tc_percentile, args.et_percentile)
