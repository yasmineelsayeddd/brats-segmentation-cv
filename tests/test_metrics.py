import numpy as np

from src.evaluation.metrics import dice_per_class, hd95_per_class, iou_per_class, pixel_accuracy


def test_metrics_on_toy_masks():
    gt = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    pred = gt.copy()
    assert np.allclose(dice_per_class(pred, gt), [1, 1, 1])
    assert np.allclose(iou_per_class(pred, gt), [1, 1, 1])
    assert pixel_accuracy(pred, gt) == 1.0


def test_hd95_returns_nan_for_missing_class():
    gt = np.zeros((8, 8), dtype=np.uint8)
    pred = np.zeros((8, 8), dtype=np.uint8)
    values = hd95_per_class(pred, gt, num_classes=2)
    assert np.isnan(values[0])
