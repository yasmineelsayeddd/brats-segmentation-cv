"""Extract 2D axial slices from BraTS 2020 3D NIfTI volumes.

Expected input layout (BraTS 2020 Training Data from Kaggle):

    <input_dir>/
        BraTS20_Training_001/
            BraTS20_Training_001_flair.nii
            BraTS20_Training_001_t1.nii
            BraTS20_Training_001_t1ce.nii
            BraTS20_Training_001_t2.nii
            BraTS20_Training_001_seg.nii
        BraTS20_Training_002/
            ...

Output layout:

    <output_dir>/
        BraTS20_Training_001/
            slice_060_image.npy  # shape (4, H, W), modalities stacked: [flair, t1, t1ce, t2]
            slice_060_mask.npy   # shape (H, W), labels in {0, 1, 2, 3}
            ...
        metadata.json            # list of (patient_id, slice_index) tuples + label remapping info

Label remapping:
    BraTS native labels are {0, 1, 2, 4}. We remap to {0, 1, 2, 3} for clean indexing:
        0 -> background
        1 -> NCR/NET (non-enhancing tumor core)
        2 -> ED (peritumoral edema)
        3 -> ET (GD-enhancing tumor), originally 4 in BraTS

Normalization:
    Per-modality z-score using only brain voxels (non-zero values).

Usage:
    python scripts/prepare_data.py \
        --input  data/raw/MICCAI_BraTS2020_TrainingData \
        --output data/brats2020_2d
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

MODALITIES = ["flair", "t1", "t1ce", "t2"]


def load_modality(patient_dir: Path, patient_id: str, modality: str) -> np.ndarray:
    """Load a single 3D modality volume."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("nibabel is required for BraTS NIfTI loading. Install requirements.txt first.") from exc

    path = patient_dir / f"{patient_id}_{modality}.nii"
    if not path.exists():
        path = patient_dir / f"{patient_id}_{modality}.nii.gz"
    return nib.load(str(path)).get_fdata().astype(np.float32)


def normalize_modality(volume: np.ndarray) -> np.ndarray:
    """Z-score normalize using only brain voxels (non-zero)."""
    mask = volume > 0
    if not mask.any():
        return volume
    mean = volume[mask].mean()
    std = volume[mask].std()
    if std < 1e-8:
        return volume - mean
    return (volume - mean) / std


def remap_labels(mask: np.ndarray) -> np.ndarray:
    """Remap BraTS labels {0,1,2,4} -> {0,1,2,3}."""
    remapped = mask.copy().astype(np.uint8)
    remapped[mask == 4] = 3
    return remapped


def process_patient(
    patient_dir: Path,
    output_dir: Path,
    keep_empty_ratio: float = 0.0,
    image_dtype: str = "float32",
    min_tumor_pixels: int = 1,
) -> list[tuple[str, int]]:
    """Extract 2D slices from one patient.

    Args:
        patient_dir: Directory containing one patient's .nii files.
        output_dir: Where to write the per-slice .npy files.
        keep_empty_ratio: Fraction of tumor-free slices to keep (0 = drop all).
        image_dtype: dtype used for saved image slices. Use float16 on Kaggle to reduce disk.
        min_tumor_pixels: Minimum tumor pixels needed to keep a tumor slice.

    Returns:
        List of (patient_id, slice_index) pairs that were saved.
    """
    patient_id = patient_dir.name
    out_subdir = output_dir / patient_id
    out_subdir.mkdir(parents=True, exist_ok=True)

    volumes = [load_modality(patient_dir, patient_id, m) for m in MODALITIES]
    seg = load_modality(patient_dir, patient_id, "seg")

    volumes = [normalize_modality(v) for v in volumes]
    seg = remap_labels(seg)

    num_slices = seg.shape[2]
    saved: list[tuple[str, int]] = []

    for idx in range(num_slices):
        image_slice = np.stack([v[..., idx] for v in volumes], axis=0)  # (4, H, W)
        mask_slice = seg[..., idx]

        tumor_pixels = int((mask_slice > 0).sum())
        has_tumor = tumor_pixels >= min_tumor_pixels
        has_brain = (image_slice != 0).any(axis=0).any()

        if not has_brain:
            continue
        if not has_tumor and np.random.random() > keep_empty_ratio:
            continue

        np.save(out_subdir / f"slice_{idx:03d}_image.npy", image_slice.astype(image_dtype))
        np.save(out_subdir / f"slice_{idx:03d}_mask.npy", mask_slice.astype(np.uint8))
        saved.append((patient_id, idx))

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 2D slices from BraTS 2020.")
    parser.add_argument("--input", type=Path, required=True, help="Raw BraTS 2020 training directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for 2D slices.")
    parser.add_argument(
        "--keep-empty-ratio",
        type=float,
        default=0.0,
        help="Fraction of tumor-free brain slices to keep (default 0).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-dtype",
        choices=["float16", "float32"],
        default="float32",
        help="Saved image dtype. float16 roughly halves prepared data size.",
    )
    parser.add_argument(
        "--min-tumor-pixels",
        type=int,
        default=1,
        help="Minimum non-background tumor pixels required to keep a tumor slice.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted(p for p in args.input.iterdir() if p.is_dir())
    if not patient_dirs:
        raise SystemExit(f"No patient subdirectories found in {args.input}")

    all_slices: list[tuple[str, int]] = []
    for patient_dir in tqdm(patient_dirs, desc="Patients"):
        try:
            saved = process_patient(
                patient_dir,
                args.output,
                args.keep_empty_ratio,
                image_dtype=args.image_dtype,
                min_tumor_pixels=args.min_tumor_pixels,
            )
            all_slices.extend(saved)
        except FileNotFoundError as e:
            print(f"Skipping {patient_dir.name}: {e}")

    metadata = {
        "modalities": MODALITIES,
        "label_remap": {"0": "background", "1": "NCR/NET", "2": "edema", "3": "enhancing_tumor"},
        "num_patients": len(patient_dirs),
        "num_slices": len(all_slices),
        "keep_empty_ratio": args.keep_empty_ratio,
        "image_dtype": args.image_dtype,
        "min_tumor_pixels": args.min_tumor_pixels,
        "slices": [{"patient": p, "slice": s} for p, s in all_slices],
    }
    with (args.output / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {len(all_slices)} slices from {len(patient_dirs)} patients to {args.output}")


if __name__ == "__main__":
    main()
