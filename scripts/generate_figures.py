"""Generate report-ready figures from the local outputs JSON files.

Reads the per-experiment history.json + best.json + test_metrics.json
files already downloaded into ``modal_checkpoints_local`` and writes
publication-quality matplotlib figures (PNG) into ``report_figures/``.

Figures produced:
  fig_training_curves.png    — val Dice over epochs, 5 multi-class models overlaid
  fig_val_test_gap.png       — generalization gap bar chart
  fig_per_class_dice.png     — per-class test Dice grouped bar chart
  fig_hierarchical.png       — WT/TC/ET Dice across 3 pipelines (classical, vanilla cascade, uncertainty cascade)
  fig_analysis_deltas.png    — TTA / cascade / SAM / active contour vs baseline (signed deltas)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOCAL_ROOT = Path("C:/Users/yasmi/Downloads/brats_modal/outputs")
OUT_DIR = Path("C:/Users/yasmi/Desktop/uni/senior year/SPRING/Computer Vision/Project/report_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MULTI_CLASS = [
    ("unet_baseline", "U-Net (baseline)"),
    ("unet_best_fusion_loss_aug", "U-Net + best loss/aug"),
    ("attention_unet", "Attention U-Net"),
    ("uncertainty_unet", "MC-dropout U-Net"),
    ("unetpp", "U-Net++"),
]

PLOT_STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}
plt.rcParams.update(PLOT_STYLE)


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fig_training_curves() -> None:
    """Val Dice vs epoch for all 5 multi-class models, overlaid."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(MULTI_CLASS)))
    for (exp, label), color in zip(MULTI_CLASS, colors, strict=False):
        hist = _load_json(LOCAL_ROOT / exp / "history.json")
        if hist is None:
            continue
        epochs = [h["epoch"] for h in hist]
        val = [h["val_dice"] for h in hist]
        ax.plot(epochs, val, label=label, color=color, linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Dice")
    ax.set_title("Training Trajectories — 5 U-Net Variants")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")
    fig.savefig(OUT_DIR / "fig_training_curves.png")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'fig_training_curves.png'}")


def fig_val_test_gap() -> None:
    """Generalization gap (val_dice − mean test Dice) for each model."""
    rows: list[tuple[str, float, float]] = []  # (label, val, test)
    for exp, label in MULTI_CLASS:
        best = _load_json(LOCAL_ROOT / exp / "best.json")
        test = _load_json(LOCAL_ROOT / exp / "test_metrics.json")
        if best is None or test is None:
            continue
        rows.append((label, best["best_val_dice"], test["mean_dice"]))

    rows.sort(key=lambda r: r[1] - r[2])  # smallest gap first
    labels = [r[0] for r in rows]
    val_scores = [r[1] for r in rows]
    test_scores = [r[2] for r in rows]
    gaps = [v - t for v, t in zip(val_scores, test_scores, strict=False)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, val_scores, w, label="Validation", color="#1f77b4")
    ax.bar(x + w / 2, test_scores, w, label="Test", color="#ff7f0e")
    for i, gap in enumerate(gaps):
        ax.text(i, max(val_scores[i], test_scores[i]) + 0.005, f"Δ={gap:.3f}",
                ha="center", fontsize=9, color="#666")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Mean Dice")
    ax.set_title("Validation vs Test Dice — Generalization Gap")
    ax.set_ylim(0.55, 0.90)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    fig.savefig(OUT_DIR / "fig_val_test_gap.png")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'fig_val_test_gap.png'}")


def fig_per_class_dice() -> None:
    """Per-class test Dice grouped bar chart, 5 models × 3 classes."""
    classes = ["NCR/NET", "edema", "enhancing_tumor"]
    class_labels = ["NCR/NET", "Edema", "Enhancing"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    n_models = len(MULTI_CLASS)
    n_classes = len(classes)
    x = np.arange(n_classes)
    w = 0.8 / n_models
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    for i, ((exp, label), color) in enumerate(zip(MULTI_CLASS, colors, strict=False)):
        test = _load_json(LOCAL_ROOT / exp / "test_metrics.json")
        if test is None:
            continue
        vals = [test[f"dice_{c}"] for c in classes]
        ax.bar(x + (i - n_models / 2) * w + w / 2, vals, w, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(class_labels)
    ax.set_ylabel("Test Dice")
    ax.set_title("Per-Class Test Dice — 5 U-Net Variants")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", ncol=2)
    ax.set_ylim(0.55, 0.85)
    fig.savefig(OUT_DIR / "fig_per_class_dice.png")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'fig_per_class_dice.png'}")


def fig_hierarchical() -> None:
    """WT/TC/ET Dice across the three hierarchical pipelines.

    Reads val metrics for the cascades and test metrics for classical. If
    cascade test metrics exist (after cascade_test_eval), uses those
    instead for apples-to-apples comparison.
    """
    # Classical (test)
    classical = _load_json(LOCAL_ROOT / "classical_hierarchical" / "test_metrics.json") or {}

    # Vanilla cascade — try test first, fall back to val (best.json)
    vanilla_test = _load_json(LOCAL_ROOT / "cascade_vanilla" / "test_metrics.json")
    if vanilla_test is None:
        vanilla = {
            "WT": _load_json(LOCAL_ROOT / "hierarchical_wt" / "best.json"),
            "TC": _load_json(LOCAL_ROOT / "hierarchical_tc" / "best.json"),
            "ET": _load_json(LOCAL_ROOT / "hierarchical_et" / "best.json"),
        }
        vanilla_dice = {r: (vanilla[r]["best_val_dice"] if vanilla[r] else 0.0) for r in ("WT", "TC", "ET")}
        vanilla_label = "Vanilla U-Net Cascade (val)"
    else:
        vanilla_dice = {r: vanilla_test[f"cascade_{r}_dice_mean"] for r in ("WT", "TC", "ET")}
        vanilla_label = "Vanilla U-Net Cascade"

    # Uncertainty cascade
    uncertainty_test = _load_json(LOCAL_ROOT / "cascade_uncertainty" / "test_metrics.json")
    if uncertainty_test is None:
        unc = {
            "WT": _load_json(LOCAL_ROOT / "hierarchical_uncertainty_unet_wt" / "best.json"),
            "TC": _load_json(LOCAL_ROOT / "hierarchical_uncertainty_unet_tc" / "best.json"),
            "ET": _load_json(LOCAL_ROOT / "hierarchical_uncertainty_unet_et" / "best.json"),
        }
        unc_dice = {r: (unc[r]["best_val_dice"] if unc[r] else 0.0) for r in ("WT", "TC", "ET")}
        unc_label = "Uncertainty U-Net Cascade (val)"
    else:
        unc_dice = {r: uncertainty_test[f"cascade_{r}_dice_mean"] for r in ("WT", "TC", "ET")}
        unc_label = "Uncertainty U-Net Cascade"

    classical_dice = {r: classical.get(f"classical_{r}_dice_mean", 0.0) for r in ("WT", "TC", "ET")}

    regions = ["WT", "TC", "ET"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(regions))
    w = 0.27

    ax.bar(x - w, [classical_dice[r] for r in regions], w, label="Classical (Otsu + Morphology)", color="#888888")
    ax.bar(x, [vanilla_dice[r] for r in regions], w, label=vanilla_label, color="#1f77b4")
    ax.bar(x + w, [unc_dice[r] for r in regions], w, label=unc_label, color="#2ca02c")

    ax.set_xticks(x)
    ax.set_xticklabels(["Whole Tumor", "Tumor Core", "Enhancing Tumor"])
    ax.set_ylabel("Dice")
    ax.set_title("Hierarchical Region Segmentation — Classical vs Deep Cascades")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower left")
    ax.set_ylim(0, 1.0)

    for i, r in enumerate(regions):
        for offset, val in zip([-w, 0, w], [classical_dice[r], vanilla_dice[r], unc_dice[r]], strict=False):
            ax.text(i + offset, val + 0.015, f"{val:.2f}", ha="center", fontsize=8.5)

    fig.savefig(OUT_DIR / "fig_hierarchical.png")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'fig_hierarchical.png'}")


def fig_analysis_deltas() -> None:
    """Δ Dice over uncertainty_unet baseline for each post-hoc analysis."""
    base = _load_json(LOCAL_ROOT / "uncertainty_unet" / "test_metrics.json")
    if base is None:
        return
    base_dice = base["mean_dice"]

    rows: list[tuple[str, float]] = []  # (label, dice)
    analyses = [
        ("test_tta_metrics.json", "tta_mean_dice", "TTA (4 flips)"),
        ("test_tta_metrics.json", "tta_morph_mean_dice", "TTA + Morph cleanup"),
        ("test_uncertainty_metrics.json", "mean_dice", "MC dropout (20 samples)"),
        ("test_cascade_metrics.json", "mean_dice", "YOLO→U-Net cascade"),
    ]
    for filename, key, label in analyses:
        d = _load_json(LOCAL_ROOT / "uncertainty_unet" / filename)
        if d and key in d:
            rows.append((label, d[key] - base_dice))

    if not rows:
        return
    labels, deltas = zip(*rows)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
    ax.barh(range(len(labels)), deltas, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"Δ mean Dice vs MC-dropout U-Net baseline ({base_dice:.4f})")
    ax.set_title("Post-Hoc Analyses — Signed Improvement vs Baseline (test set)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    for i, d in enumerate(deltas):
        ax.text(d + (0.005 if d >= 0 else -0.005), i, f"{d:+.3f}",
                va="center", ha="left" if d >= 0 else "right", fontsize=9)
    fig.savefig(OUT_DIR / "fig_analysis_deltas.png")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'fig_analysis_deltas.png'}")


if __name__ == "__main__":
    fig_training_curves()
    fig_val_test_gap()
    fig_per_class_dice()
    fig_hierarchical()
    fig_analysis_deltas()
    print(f"\nAll figures written to {OUT_DIR}")
