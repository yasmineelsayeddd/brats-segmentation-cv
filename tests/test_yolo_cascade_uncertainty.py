import numpy as np
import torch

from src.data.yolo import bbox_to_yolo, mask_to_bbox
from src.inference.cascade import cascade_segment, paste_roi
from src.inference.uncertainty import mc_dropout_predict
from src.models import DropoutUNet, UNet


def test_bbox_generation_empty_and_multicomponent():
    empty = np.zeros((16, 16), dtype=np.uint8)
    assert mask_to_bbox(empty) is None
    mask = empty.copy()
    mask[2:4, 3:5] = 1
    mask[10:12, 12:14] = 2
    assert mask_to_bbox(mask, margin=0) == (3, 2, 14, 12)
    yolo = bbox_to_yolo((3, 2, 14, 12), width=16, height=16)
    assert all(0 <= x <= 1 for x in yolo)


def test_paste_roi_returns_full_size():
    roi = np.ones((4, 5), dtype=np.uint8)
    out = paste_roi(roi, (12, 12), (2, 3, 7, 7))
    assert out.shape == (12, 12)
    assert out.sum() == roi.sum()


def test_cascade_falls_back_to_full_image():
    model = UNet(base_channels=4)
    image = np.random.randn(4, 32, 32).astype(np.float32)
    result = cascade_segment(model, image, detector=lambda _image: None)
    assert result.mask.shape == (32, 32)
    assert result.probabilities.shape == (4, 32, 32)


def test_uncertainty_shapes_and_ranges():
    model = DropoutUNet(base_channels=4, dropout=0.5)
    image = np.random.randn(4, 32, 32).astype(np.float32)
    result = mc_dropout_predict(model, image, passes=2)
    assert result.mask.shape == (32, 32)
    assert result.confidence.min() >= 0
    assert result.confidence.max() <= 1
    assert result.uncertainty is not None
    assert np.isfinite(result.uncertainty).all()
