import torch

from src.models import AttentionUNet, DropoutUNet, UNet, UNetPlusPlus, build_model
from src.training.losses import DiceCELoss


def test_unet_forward_240_and_odd_size():
    model = UNet(in_channels=4, out_channels=4, base_channels=8)
    for shape in [(2, 4, 240, 240), (1, 4, 65, 67)]:
        y = model(torch.randn(*shape))
        assert y.shape == (shape[0], 4, shape[2], shape[3])


def test_required_model_factories_forward():
    x = torch.randn(1, 4, 64, 64)
    for arch in ["unet", "unetpp", "attention_unet", "uncertainty_unet"]:
        model = build_model(arch, base_channels=4)
        y = model(x)
        assert y.shape == (1, 4, 64, 64)


def test_loss_is_finite():
    logits = torch.randn(2, 4, 32, 32)
    targets = torch.randint(0, 4, (2, 32, 32))
    loss = DiceCELoss()(logits, targets)
    assert torch.isfinite(loss)
