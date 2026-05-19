import json

import numpy as np

from src.data.brats import BraTSDataset
from src.data.splits import patient_level_split


def test_patient_split_has_no_leakage():
    split = patient_level_split([f"p{i}" for i in range(20)], seed=7)
    train = set(split["train"])
    val = set(split["val"])
    test = set(split["test"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)


def test_synthetic_dataset_loads(tmp_path):
    patient = tmp_path / "p001"
    patient.mkdir()
    np.save(patient / "slice_000_image.npy", np.zeros((4, 16, 16), dtype=np.float32))
    np.save(patient / "slice_000_mask.npy", np.zeros((16, 16), dtype=np.uint8))
    (tmp_path / "metadata.json").write_text(
        json.dumps({"slices": [{"patient": "p001", "slice": 0}]}),
        encoding="utf-8",
    )
    ds = BraTSDataset(tmp_path)
    image, mask = ds[0]
    assert image.shape == (4, 16, 16)
    assert mask.shape == (16, 16)
