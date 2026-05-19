"""U-Net family models for multi-class brain tumor segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Conv -> BN -> ReLU -> Conv -> BN -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None) -> None:
        super().__init__()
        mid = mid_channels or out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool2d -> DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """Bilinear upsample -> concatenate skip -> DoubleConv."""

    def __init__(self, x_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(x_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """Standard 2D U-Net with configurable width."""

    def __init__(self, in_channels: int = 4, out_channels: int = 4, base_channels: int = 64) -> None:
        super().__init__()
        b = base_channels
        self.enc1 = DoubleConv(in_channels, b)
        self.enc2 = Down(b, b * 2)
        self.enc3 = Down(b * 2, b * 4)
        self.enc4 = Down(b * 4, b * 8)
        self.bottleneck = Down(b * 8, b * 16)
        self.dec4 = Up(b * 16, b * 8, b * 8)
        self.dec3 = Up(b * 8, b * 4, b * 4)
        self.dec2 = Up(b * 4, b * 2, b * 2)
        self.dec1 = Up(b * 2, b, b)
        self.head = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b = self.bottleneck(s4)
        x = self.dec4(b, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.head(x)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AttentionGate(nn.Module):
    """Attention gate for filtering encoder skip features."""

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(nn.Conv2d(inter_channels, 1, kernel_size=1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        alpha = self.psi(self.relu(self.gate_proj(gate) + self.skip_proj(skip)))
        return skip * alpha


class AttentionUp(nn.Module):
    """Upsampling block with an attention-filtered skip connection."""

    def __init__(self, x_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.attn = AttentionGate(x_channels, skip_channels, max(out_channels // 2, 1))
        self.up = Up(x_channels, skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.up(x, self.attn(x, skip))


class AttentionUNet(UNet):
    """U-Net with attention gates on decoder skip connections."""

    def __init__(self, in_channels: int = 4, out_channels: int = 4, base_channels: int = 64) -> None:
        super().__init__(in_channels, out_channels, base_channels)
        b = base_channels
        self.dec4 = AttentionUp(b * 16, b * 8, b * 8)
        self.dec3 = AttentionUp(b * 8, b * 4, b * 4)
        self.dec2 = AttentionUp(b * 4, b * 2, b * 2)
        self.dec1 = AttentionUp(b * 2, b, b)


class DropoutUNet(UNet):
    """U-Net variant with dropout for MC-dropout uncertainty inference."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(in_channels, out_channels, base_channels)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b = self.dropout(self.bottleneck(s4))
        x = self.dropout(self.dec4(b, s4))
        x = self.dropout(self.dec3(x, s3))
        x = self.dropout(self.dec2(x, s2))
        x = self.dec1(x, s1)
        return self.head(x)


class UNetPlusPlus(nn.Module):
    """Compact U-Net++ style nested skip model."""

    def __init__(self, in_channels: int = 4, out_channels: int = 4, base_channels: int = 64) -> None:
        super().__init__()
        b = base_channels
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.x00 = DoubleConv(in_channels, b)
        self.x10 = DoubleConv(b, b * 2)
        self.x20 = DoubleConv(b * 2, b * 4)
        self.x30 = DoubleConv(b * 4, b * 8)
        self.x40 = DoubleConv(b * 8, b * 16)
        self.x01 = DoubleConv(b + b * 2, b)
        self.x11 = DoubleConv(b * 2 + b * 4, b * 2)
        self.x21 = DoubleConv(b * 4 + b * 8, b * 4)
        self.x31 = DoubleConv(b * 8 + b * 16, b * 8)
        self.x02 = DoubleConv(b * 2 + b * 2, b)
        self.x12 = DoubleConv(b * 4 + b * 4, b * 2)
        self.x22 = DoubleConv(b * 8 + b * 8, b * 4)
        self.x03 = DoubleConv(b * 3 + b * 2, b)
        self.x13 = DoubleConv(b * 6 + b * 4, b * 2)
        self.x04 = DoubleConv(b * 4 + b * 2, b)
        self.head = nn.Conv2d(b, out_channels, kernel_size=1)

    @staticmethod
    def _cat(*tensors: torch.Tensor) -> torch.Tensor:
        target = tensors[0].shape[-2:]
        aligned = [
            t if t.shape[-2:] == target else F.interpolate(t, size=target, mode="bilinear", align_corners=True)
            for t in tensors
        ]
        return torch.cat(aligned, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x20 = self.x20(self.pool(x10))
        x30 = self.x30(self.pool(x20))
        x40 = self.x40(self.pool(x30))
        x01 = self.x01(self._cat(x00, self.up(x10)))
        x11 = self.x11(self._cat(x10, self.up(x20)))
        x21 = self.x21(self._cat(x20, self.up(x30)))
        x31 = self.x31(self._cat(x30, self.up(x40)))
        x02 = self.x02(self._cat(x00, x01, self.up(x11)))
        x12 = self.x12(self._cat(x10, x11, self.up(x21)))
        x22 = self.x22(self._cat(x20, x21, self.up(x31)))
        x03 = self.x03(self._cat(x00, x01, x02, self.up(x12)))
        x13 = self.x13(self._cat(x10, x11, x12, self.up(x22)))
        x04 = self.x04(self._cat(x00, x01, x02, x03, self.up(x13)))
        return self.head(x04)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(
    arch: str,
    in_channels: int = 4,
    out_channels: int = 4,
    base_channels: int = 64,
    dropout: float = 0.2,
) -> nn.Module:
    """Factory for all report-required segmentation architectures."""
    name = arch.lower().replace("-", "_")
    if name in {"unet", "unet_baseline"}:
        return UNet(in_channels, out_channels, base_channels)
    if name in {"unetpp", "unet_plus_plus", "unet++"}:
        return UNetPlusPlus(in_channels, out_channels, base_channels)
    if name in {"attention_unet", "attention"}:
        return AttentionUNet(in_channels, out_channels, base_channels)
    if name in {"uncertainty_unet", "dropout_unet", "mc_dropout_unet"}:
        return DropoutUNet(in_channels, out_channels, base_channels, dropout=dropout)
    raise ValueError(f"Unknown model architecture: {arch}")
