"""Loss functions for multi-class segmentation.

DiceLoss + CrossEntropyLoss combined (the standard for medical seg).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft multi-class Dice loss.

    Args:
        smooth: Laplace smoothing to avoid 0/0.
        ignore_background: If True, class 0 is excluded from the mean.
    """

    def __init__(self, smooth: float = 1.0, ignore_background: bool = True) -> None:
        super().__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C, H, W) raw model output
            targets: (B, H, W) long tensor with class indices
        """
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets → (B, C, H, W)
        targets_oh = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        start_cls = 1 if self.ignore_background else 0
        dice_per_class = []
        for c in range(start_cls, num_classes):
            p = probs[:, c]
            g = targets_oh[:, c]
            intersection = (p * g).sum()
            dice_per_class.append(
                1.0 - (2.0 * intersection + self.smooth) / (p.sum() + g.sum() + self.smooth)
            )

        return torch.stack(dice_per_class).mean()


class DiceCELoss(nn.Module):
    """Weighted combination of Dice and Cross-Entropy losses.

    Args:
        dice_weight: Weight for Dice loss.
        ce_weight:   Weight for Cross-Entropy loss.
        class_weights: Optional per-class CE weights (1-D tensor of length C).
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        class_weights: torch.Tensor | None = None,
        ignore_background: bool = True,
    ) -> None:
        super().__init__()
        self.dice = DiceLoss(ignore_background=ignore_background)
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dw = dice_weight
        self.cw = ce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dw * self.dice(logits, targets) + self.cw * self.ce(logits, targets)
