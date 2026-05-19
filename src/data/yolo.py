"""YOLO dataset conversion helpers for tumor detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def mask_to_bbox(mask: np.ndarray, margin: int = 8) -> tuple[int, int, int, int] | None:
    """Return x1, y1, x2, y2 for all non-background mask pixels."""
    mask = np.asarray(mask)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    x1 = max(int(xs.min()) - margin, 0)
    y1 = max(int(ys.min()) - margin, 0)
    x2 = min(int(xs.max()) + margin + 1, w)
    y2 = min(int(ys.max()) + margin + 1, h)
    return x1, y1, x2, y2


def bbox_to_yolo(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[float, float, float, float]:
    """Convert x1,y1,x2,y2 to normalized YOLO cx,cy,w,h."""
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2
    cy = y1 + bh / 2
    return cx / width, cy / height, bw / width, bh / height


def modality_rgb(image: np.ndarray) -> np.ndarray:
    """Build pseudo-RGB slice from [FLAIR, T1ce, T2]."""
    image = np.asarray(image)
    channels = [image[0], image[2], image[3]]
    out = []
    for ch in channels:
        ch = ch.astype(np.float32)
        lo, hi = np.percentile(ch, [1, 99])
        ch = np.clip((ch - lo) / (hi - lo + 1e-8), 0, 1)
        out.append((ch * 255).astype(np.uint8))
    return np.stack(out, axis=-1)


def write_yolo_sample(
    image: np.ndarray,
    mask: np.ndarray,
    image_path: str | Path,
    label_path: str | Path,
    margin: int = 8,
) -> bool:
    """Write one image/label pair. Returns False when mask has no tumor."""
    bbox = mask_to_bbox(mask, margin=margin)
    if bbox is None:
        return False
    image_path = Path(image_path)
    label_path = Path(label_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = modality_rgb(image)
    Image.fromarray(rgb).save(image_path)
    h, w = mask.shape
    cx, cy, bw, bh = bbox_to_yolo(bbox, width=w, height=h)
    label_path.write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")
    return True
