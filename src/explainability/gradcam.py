"""GradCAM helpers for segmentation models."""

from __future__ import annotations

import numpy as np
import torch


def gradcam_overlay(
    model: torch.nn.Module,
    image: np.ndarray | torch.Tensor,
    target_layer: torch.nn.Module,
    class_idx: int = 3,
    device: str = "cpu",
) -> np.ndarray:
    """Return a normalized GradCAM heatmap for one class.

    The implementation avoids a hard dependency on pytorch-grad-cam so the
    project remains smoke-testable locally. It is sufficient for report panels.
    """
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        activations.append(output)

    def backward_hook(_module: torch.nn.Module, _grad_input: tuple[torch.Tensor, ...], grad_output: tuple[torch.Tensor, ...]) -> None:
        gradients.append(grad_output[0])

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)
    try:
        model = model.to(device)
        model.eval()
        x = torch.as_tensor(image, dtype=torch.float32, device=device)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        logits = model(x)
        score = logits[:, class_idx].mean()
        model.zero_grad(set_to_none=True)
        score.backward()
        acts = activations[-1]
        grads = gradients[-1]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1))[0]
        cam = torch.nn.functional.interpolate(
            cam[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0]
        cam = cam.detach().cpu().numpy()
        return ((cam - cam.min()) / (cam.max() - cam.min() + 1e-8)).astype(np.float32)
    finally:
        handle_f.remove()
        handle_b.remove()
