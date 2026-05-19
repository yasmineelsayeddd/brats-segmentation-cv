"""MC-dropout uncertainty inference."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.inference.segmentation import SegmentationResult


def predictive_entropy(probabilities: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Pixelwise entropy over class probabilities."""
    return -(probabilities * np.log(probabilities + eps)).sum(axis=0).astype(np.float32)


def _enable_dropout(module: torch.nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, torch.nn.Dropout | torch.nn.Dropout2d | torch.nn.Dropout3d):
            child.train()


def mc_dropout_predict(
    model: torch.nn.Module,
    image: np.ndarray | torch.Tensor,
    device: str = "cpu",
    passes: int = 30,
) -> SegmentationResult:
    """Run repeated stochastic passes and return mean probabilities + uncertainty."""
    if passes <= 0:
        raise ValueError("passes must be positive")
    model = model.to(device)
    model.eval()
    _enable_dropout(model)
    tensor = torch.as_tensor(image, dtype=torch.float32)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    probs = []
    with torch.no_grad():
        for _ in range(passes):
            logits = model(tensor.to(device))
            probs.append(F.softmax(logits, dim=1)[0].cpu().numpy())
    mean_probs = np.mean(np.stack(probs, axis=0), axis=0).astype(np.float32)
    mask = mean_probs.argmax(axis=0).astype(np.uint8)
    confidence = mean_probs.max(axis=0).astype(np.float32)
    entropy = predictive_entropy(mean_probs)
    return SegmentationResult(mask=mask, probabilities=mean_probs, confidence=confidence, uncertainty=entropy)
