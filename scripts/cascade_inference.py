"""Cascade inference: run 3 binary U-Nets sequentially, evaluate per-region.

For each input slice:
  Stage 1: WT model predicts a whole-tumor binary mask.
  Stage 2: TC model predicts; result is intersected with the WT mask
           (cascade constraint: TC ⊂ WT).
  Stage 3: ET model predicts; result is intersected with the TC mask
           (cascade constraint: ET ⊂ TC).

Reports Dice/IoU/HD95 per region against the ground-truth hierarchical
regions. Works for both vanilla and uncertainty cascade variants — just
point it at the three checkpoint paths.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.brats import BraTSDataset
from src.data.splits import load_split
from src.evaluation.metrics import dice_per_class, hd95_binary, iou_per_class
from src.models import build_model
from src.utils.artifacts import save_json
from src.utils.config import load_config


def load_binary_unet(arch: str, base_channels: int, dropout: float, in_channels: int,
                     ckpt_path: str, device: str) -> torch.nn.Module:
    model = build_model(
        arch,
        in_channels=in_channels,
        out_channels=2,
        base_channels=base_channels,
        dropout=dropout,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def gt_hierarchical(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BraTS multi-class -> nested binary regions."""
    wt = mask > 0
    tc = (mask == 1) | (mask == 3)
    et = mask == 3
    return wt.astype(bool), tc.astype(bool), et.astype(bool)


def main(
    config_path: str,
    wt_ckpt: str,
    tc_ckpt: str,
    et_ckpt: str,
    out_path: str,
    split_name: str = "test",
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(config_path)
    arch = cfg["model"]["arch"]
    base_channels = cfg["model"].get("base_channels", 64)
    dropout = cfg["model"].get("dropout", 0.2)
    in_channels = cfg["model"]["in_channels"]

    wt_model = load_binary_unet(arch, base_channels, dropout, in_channels, wt_ckpt, device)
    tc_model = load_binary_unet(arch, base_channels, dropout, in_channels, tc_ckpt, device)
    et_model = load_binary_unet(arch, base_channels, dropout, in_channels, et_ckpt, device)

    split = load_split(cfg["data"]["split_file"])
    ds = BraTSDataset(cfg["data"]["data_root"], patient_ids=split[split_name])
    loader = DataLoader(
        ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
        num_workers=cfg["data"].get("num_workers", 2),
    )

    region_dice: dict[str, list[float]] = {"WT": [], "TC": [], "ET": []}
    region_iou: dict[str, list[float]] = {"WT": [], "TC": [], "ET": []}
    region_hd95: dict[str, list[float]] = {"WT": [], "TC": [], "ET": []}

    start = time.time()
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)

            wt_pred = wt_model(images).argmax(dim=1).cpu().numpy().astype(bool)
            tc_raw = tc_model(images).argmax(dim=1).cpu().numpy().astype(bool)
            et_raw = et_model(images).argmax(dim=1).cpu().numpy().astype(bool)

            # Cascade constraints (nested)
            tc_pred = tc_raw & wt_pred
            et_pred = et_raw & tc_pred

            masks_np = masks.numpy()
            for b in range(wt_pred.shape[0]):
                wt_g, tc_g, et_g = gt_hierarchical(masks_np[b])
                for name, pred, gt in [
                    ("WT", wt_pred[b], wt_g),
                    ("TC", tc_pred[b], tc_g),
                    ("ET", et_pred[b], et_g),
                ]:
                    d = float(dice_per_class(pred.astype(np.uint8), gt.astype(np.uint8),
                                             num_classes=2, ignore_background=True)[0])
                    i = float(iou_per_class(pred.astype(np.uint8), gt.astype(np.uint8),
                                            num_classes=2, ignore_background=True)[0])
                    h = float(hd95_binary(pred, gt))
                    region_dice[name].append(d)
                    region_iou[name].append(i)
                    region_hd95[name].append(h)

    metrics: dict = {}
    for name in ("WT", "TC", "ET"):
        metrics[f"cascade_{name}_dice_mean"] = float(np.nanmean(region_dice[name]))
        metrics[f"cascade_{name}_iou_mean"] = float(np.nanmean(region_iou[name]))
        metrics[f"cascade_{name}_hd95_mean"] = float(np.nanmean(region_hd95[name]))
    metrics["split"] = split_name
    metrics["arch"] = arch
    metrics["wt_checkpoint"] = str(wt_ckpt)
    metrics["tc_checkpoint"] = str(tc_ckpt)
    metrics["et_checkpoint"] = str(et_ckpt)
    metrics["num_slices"] = len(ds)
    metrics["inference_time_s"] = time.time() - start

    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    save_json(metrics, out_path_p)
    print(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--wt-ckpt", required=True)
    parser.add_argument("--tc-ckpt", required=True)
    parser.add_argument("--et-ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()
    main(args.config, args.wt_ckpt, args.tc_ckpt, args.et_ckpt, args.out, args.split)
