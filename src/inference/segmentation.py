"""Stable segmentation inference API for later demo integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SegmentationResult:
    mask: np.ndarray
    probabilities: np.ndarray
    confidence: np.ndarray
    uncertainty: np.ndarray | None = None


def segment_slice(
    model: torch.nn.Module,
    image: np.ndarray | torch.Tensor,
    device: str = "cpu",
) -> SegmentationResult:
    """Segment one prepared 4-channel 2D slice."""
    model = model.to(device)
    model.eval()
    tensor = torch.as_tensor(image, dtype=torch.float32)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor.to(device))
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    mask = probs.argmax(axis=0).astype(np.uint8)
    confidence = probs.max(axis=0).astype(np.float32)
    return SegmentationResult(mask=mask, probabilities=probs.astype(np.float32), confidence=confidence)
