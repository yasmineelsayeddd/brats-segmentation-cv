"""Train and evaluate BraTS segmentation models on Modal — persistent volumes, no session limits.

One-time local setup:
    pip install modal
    modal token new          # opens a browser to authenticate

Step 1 — download the prepared BraTS dataset into a persistent volume (run once, ~3 min):
    modal run --detach modal_train.py::download_data

Step 2 — train multi-class U-Net variants:
    modal run --detach modal_train.py --all                 # all 5 variants in parallel
    modal run --detach modal_train.py --config unetpp.yaml  # or just one

Step 3 — train hierarchical cascade (3 binary U-Nets, runs in parallel):
    modal run --detach modal_train.py --hierarchical-all                                     # vanilla U-Net cascade
    modal run --detach modal_train.py --hierarchical-all --hierarchical-config uncertainty_unet.yaml  # MC-dropout variant

Step 4 — evaluate on the held-out test split:
    modal run --detach modal_train.py --eval-all                                # all 5 multi-class
    modal run --detach modal_train.py --cascade-test-eval --variant vanilla     # cascade test eval
    modal run --detach modal_train.py --uncertainty                             # L11 MC-dropout uncertainty
    modal run --detach modal_train.py --classical                               # L3 classical hierarchical

Step 5 — pull results back to your machine:
    modal volume get brats-checkpoints / ./modal_checkpoints

Track running jobs at https://modal.com/apps
"""

from __future__ import annotations

from pathlib import Path

import modal

# Kaggle API token for the public prepared dataset
KAGGLE_TOKEN = "KGAT_bb60250db72735c2a11893aa4a1e0db7"
DATASET_SLUG = "yasmineelqorashy/brats2020-2d-prepared"

# Multi-class U-Net variants — each tests a different architectural hypothesis
CONFIGS = [
    "default.yaml",                     # vanilla baseline
    "unetpp.yaml",                      # UNet++
    "unet_best_fusion_loss_aug.yaml",   # tuned loss/aug
    "attention_unet.yaml",              # Attention U-Net
    "uncertainty_unet.yaml",            # MC-dropout U-Net
]

LOCAL_REPO = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "scikit-image",
        "opencv-python-headless",
        "albumentations",
        "Pillow",
        "pyyaml",
        "tqdm",
        "tensorboard",
        "kaggle",
    )
    .add_local_dir(LOCAL_REPO / "src", "/root/project/src")
    .add_local_dir(LOCAL_REPO / "scripts", "/root/project/scripts")
    .add_local_dir(LOCAL_REPO / "configs", "/root/project/configs")
)

app = modal.App("brats-segmentation")

data_vol = modal.Volume.from_name("brats-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("brats-checkpoints", create_if_missing=True)


def _resolve_data_paths() -> tuple[Path, Path]:
    """Find metadata.json + default.json on the data volume."""
    data_vol.reload()
    try:
        meta = next(Path("/data").rglob("metadata.json"))
        split = next(Path("/data").rglob("default.json"))
    except StopIteration as exc:
        raise RuntimeError("BraTS dataset not on volume — run download_data first.") from exc
    return meta.parent, split


def _patched_config(config_name: str) -> tuple[Path, str]:
    """Load a repo config, patch volume paths, return (runtime_cfg_path, exp_name)."""
    import yaml
    project = Path("/root/project")
    data_root, split = _resolve_data_paths()
    with open(project / "configs" / config_name) as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["data_root"] = str(data_root)
    cfg["data"]["split_file"] = str(split)
    cfg["checkpoint_dir"] = "/ckpt"
    cfg["experiment"]["output_dir"] = "/ckpt/outputs"
    exp = cfg["experiment"]["name"]
    runtime_cfg = Path(f"/tmp/runtime_{exp}.yaml")
    with open(runtime_cfg, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return runtime_cfg, exp


# =============================================================================
# Data — one-time download from Kaggle into the persistent volume
# =============================================================================
@app.function(image=image, volumes={"/data": data_vol}, timeout=3600)
def download_data() -> None:
    """One-time: download the prepared BraTS dataset from Kaggle into the data volume."""
    import os
    import subprocess

    existing = list(Path("/data").rglob("metadata.json"))
    if existing:
        print(f"Dataset already on volume: {existing[0].parent}")
        return

    os.environ["KAGGLE_API_TOKEN"] = KAGGLE_TOKEN
    print(f"Downloading {DATASET_SLUG} ...")
    subprocess.check_call(
        ["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", "/data", "--unzip"]
    )
    data_vol.commit()
    meta = [str(m) for m in Path("/data").rglob("metadata.json")]
    print(f"Done. metadata.json found at: {meta}")


# =============================================================================
# Multi-class U-Net training (5 variants — L4, L6, L7, L10)
# =============================================================================
@app.function(
    image=image,
    gpu="A10",
    volumes={"/data": data_vol, "/ckpt": ckpt_vol},
    cpu=8.0,
    memory=16384,
    timeout=86400,  # 24 h ceiling
)
def train_model(config_name: str) -> None:
    """Train one multi-class U-Net variant; auto-resumes from a checkpoint if present."""
    import os
    import subprocess
    import sys

    project = Path("/root/project")
    os.chdir(project)
    runtime_cfg, exp_name = _patched_config(config_name)

    ckpt_vol.reload()
    last = Path("/ckpt") / f"{exp_name}_last.pth"
    best = Path("/ckpt") / f"{exp_name}_best.pth"
    resume_arg: list[str] = []
    if last.exists():
        resume_arg = ["--resume", str(last)]
        print(f"[{exp_name}] resuming from {last.name}")
    elif best.exists():
        resume_arg = ["--resume", str(best)]
        print(f"[{exp_name}] resuming from {best.name} (first resume)")
    else:
        print(f"[{exp_name}] training from scratch")

    subprocess.check_call(
        [sys.executable, "-m", "scripts.train", "--config", str(runtime_cfg), *resume_arg],
        cwd=str(project),
    )
    ckpt_vol.commit()
    print(f"[{exp_name}] finished — checkpoint saved to brats-checkpoints volume.")


# =============================================================================
# Multi-class evaluation
# =============================================================================
@app.function(
    image=image,
    gpu="A10",
    volumes={"/data": data_vol, "/ckpt": ckpt_vol},
    cpu=4.0,
    timeout=3600,
)
def evaluate_model(config_name: str, split: str = "test") -> None:
    """Evaluate a trained multi-class U-Net on the given split (default: held-out test)."""
    import os
    import subprocess
    import sys

    project = Path("/root/project")
    os.chdir(project)
    runtime_cfg, exp = _patched_config(config_name)
    ckpt = Path("/ckpt") / f"{exp}_best.pth"
    if not ckpt.exists():
        raise RuntimeError(f"Checkpoint missing: {ckpt}")
    print(f"[{exp}] evaluating on {split} split ...")
    subprocess.check_call(
        [sys.executable, "-m", "scripts.evaluate",
         "--config", str(runtime_cfg),
         "--checkpoint", str(ckpt),
         "--split", split],
        cwd=str(project),
    )
    ckpt_vol.commit()
    print(f"[{exp}] metrics saved to /ckpt/outputs/{exp}/{split}_metrics.json")


# =============================================================================
# Hierarchical cascade training (3 binary U-Nets per variant)
# =============================================================================
@app.function(
    image=image,
    gpu="A10",
    volumes={"/data": data_vol, "/ckpt": ckpt_vol},
    cpu=4.0,
    timeout=86400,
)
def train_hierarchical(region: str, config_name: str = "default.yaml") -> None:
    """Train one binary U-Net for a BraTS hierarchical region (WT, TC, or ET)."""
    import os
    import subprocess
    import sys

    import yaml

    project = Path("/root/project")
    os.chdir(project)

    data_root, split_file = _resolve_data_paths()
    with open(project / "configs" / config_name) as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["data_root"] = str(data_root)
    cfg["data"]["split_file"] = str(split_file)
    cfg["checkpoint_dir"] = "/ckpt"
    cfg["experiment"]["output_dir"] = "/ckpt/outputs"

    runtime_cfg = Path(f"/tmp/hierarchical_{region.lower()}.yaml")
    with open(runtime_cfg, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    arch = cfg["model"]["arch"]
    exp = f"hierarchical_{region.lower()}" if arch == "unet" else f"hierarchical_{arch}_{region.lower()}"

    last = Path("/ckpt") / f"{exp}_last.pth"
    best = Path("/ckpt") / f"{exp}_best.pth"
    resume_arg: list[str] = []
    if last.exists():
        resume_arg = ["--resume", str(last)]
        print(f"[{exp}] resuming from {last.name}")
    elif best.exists():
        resume_arg = ["--resume", str(best)]
        print(f"[{exp}] resuming from {best.name} (first resume)")
    else:
        print(f"[{exp}] training from scratch")

    subprocess.check_call(
        [sys.executable, "-m", "scripts.train_hierarchical",
         "--config", str(runtime_cfg),
         "--region", region, *resume_arg],
        cwd=str(project),
    )
    ckpt_vol.commit()
    print(f"[{exp}] finished — checkpoint saved to brats-checkpoints volume.")


# =============================================================================
# Cascade test-set inference (both vanilla and uncertainty variants)
# =============================================================================
@app.function(
    image=image,
    gpu="A10",
    volumes={"/data": data_vol, "/ckpt": ckpt_vol},
    cpu=4.0,
    timeout=3600,
)
def cascade_test_eval(variant: str = "vanilla", split: str = "test") -> None:
    """Run cascade test-set evaluation. variant ∈ {vanilla, uncertainty}."""
    import os
    import subprocess
    import sys

    import yaml

    project = Path("/root/project")
    os.chdir(project)

    if variant == "vanilla":
        config_name = "default.yaml"
        ckpt_prefix = "hierarchical"
        out_subdir = "cascade_vanilla"
    elif variant == "uncertainty":
        config_name = "uncertainty_unet.yaml"
        ckpt_prefix = "hierarchical_uncertainty_unet"
        out_subdir = "cascade_uncertainty"
    else:
        raise ValueError(f"variant must be 'vanilla' or 'uncertainty', got {variant}")

    data_root, split_file = _resolve_data_paths()
    ckpt_vol.reload()
    with open(project / "configs" / config_name) as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["data_root"] = str(data_root)
    cfg["data"]["split_file"] = str(split_file)
    cfg["experiment"]["output_dir"] = "/ckpt/outputs"

    runtime_cfg = Path(f"/tmp/cascade_eval_{variant}.yaml")
    with open(runtime_cfg, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    wt_ckpt = Path("/ckpt") / f"{ckpt_prefix}_wt_best.pth"
    tc_ckpt = Path("/ckpt") / f"{ckpt_prefix}_tc_best.pth"
    et_ckpt = Path("/ckpt") / f"{ckpt_prefix}_et_best.pth"
    for p in (wt_ckpt, tc_ckpt, et_ckpt):
        if not p.exists():
            raise RuntimeError(f"Missing checkpoint: {p}")

    out_path = Path("/ckpt/outputs") / out_subdir / f"{split}_metrics.json"
    print(f"[cascade-{variant}] running test-set eval ({split}) ...")
    subprocess.check_call(
        [sys.executable, "-m", "scripts.cascade_inference",
         "--config", str(runtime_cfg),
         "--wt-ckpt", str(wt_ckpt),
         "--tc-ckpt", str(tc_ckpt),
         "--et-ckpt", str(et_ckpt),
         "--out", str(out_path),
         "--split", split],
        cwd=str(project),
    )
    ckpt_vol.commit()
    print(f"[cascade-{variant}] done — metrics at {out_path}")


# =============================================================================
# L11 — MC-dropout uncertainty evaluation
# =============================================================================
@app.function(
    image=image,
    gpu="A10",
    volumes={"/data": data_vol, "/ckpt": ckpt_vol},
    cpu=4.0,
    timeout=3600,
)
def uncertainty_eval(config_name: str = "uncertainty_unet.yaml", n_samples: int = 20, split: str = "test") -> None:
    """L11 — Monte Carlo dropout uncertainty evaluation."""
    import os
    import subprocess
    import sys

    project = Path("/root/project")
    os.chdir(project)
    runtime_cfg, exp = _patched_config(config_name)
    ckpt = Path("/ckpt") / f"{exp}_best.pth"
    if not ckpt.exists():
        raise RuntimeError(f"Checkpoint missing: {ckpt}")
    print(f"[{exp}] MC-dropout eval, {n_samples} samples on {split} split ...")
    subprocess.check_call(
        [sys.executable, "-m", "scripts.uncertainty_eval",
         "--config", str(runtime_cfg),
         "--checkpoint", str(ckpt),
         "--split", split,
         "--n-samples", str(n_samples)],
        cwd=str(project),
    )
    ckpt_vol.commit()


# =============================================================================
# L3 — Classical hierarchical thresholding pipeline (pure CV, CPU only)
# =============================================================================
@app.function(
    image=image,
    volumes={"/data": data_vol, "/ckpt": ckpt_vol},
    cpu=8.0,
    timeout=3600,
)
def classical_hierarchical_eval(split: str = "test", et_percentile: float = 75.0) -> None:
    """L3 — classical hierarchical thresholding (Otsu + morphology, no learning)."""
    import os
    import subprocess
    import sys

    import yaml

    project = Path("/root/project")
    os.chdir(project)

    data_root, split_file = _resolve_data_paths()
    with open(project / "configs" / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["data_root"] = str(data_root)
    cfg["data"]["split_file"] = str(split_file)
    cfg["experiment"]["output_dir"] = "/ckpt/outputs"

    runtime_cfg = Path("/tmp/classical_runtime.yaml")
    with open(runtime_cfg, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"Running classical hierarchical pipeline on {split} split ...")
    subprocess.check_call(
        [sys.executable, "-m", "scripts.classical_hierarchical",
         "--config", str(runtime_cfg),
         "--split", split,
         "--et-percentile", str(et_percentile)],
        cwd=str(project),
    )
    ckpt_vol.commit()
    print("Classical hierarchical eval done.")


# =============================================================================
# CLI entrypoint
# =============================================================================
@app.local_entrypoint()
def main(
    config: str = "",
    all: bool = False,
    eval_all: bool = False,
    eval: str = "",
    split: str = "test",
    hierarchical_all: bool = False,
    hierarchical: str = "",
    hierarchical_config: str = "default.yaml",
    cascade_test_eval_flag: bool = False,
    variant: str = "vanilla",
    uncertainty: bool = False,
    mc_samples: int = 20,
    classical: bool = False,
) -> None:
    if hierarchical_all:
        for region in ("WT", "TC", "ET"):
            train_hierarchical.spawn(region, hierarchical_config)
        print(f"Spawned 3 binary U-Net trainings ({hierarchical_config}) for WT, TC, ET.")
    elif hierarchical:
        train_hierarchical.spawn(hierarchical, hierarchical_config)
        print(f"Spawned hierarchical training for region {hierarchical} ({hierarchical_config}).")
    elif cascade_test_eval_flag:
        cascade_test_eval.spawn(variant, split)
        print(f"Spawned cascade test eval ({variant}, {split}).")
    elif uncertainty:
        uncertainty_eval.spawn("uncertainty_unet.yaml", mc_samples, split)
        print(f"Spawned MC-dropout uncertainty eval ({mc_samples} samples, {split}).")
    elif classical:
        classical_hierarchical_eval.spawn(split)
        print(f"Spawned classical hierarchical pipeline on {split}.")
    elif eval_all:
        handles = [evaluate_model.spawn(c, split) for c in CONFIGS]
        print(f"Spawned {len(handles)} evaluations on the {split} split.")
    elif eval:
        evaluate_model.spawn(eval, split)
        print(f"Spawned evaluation for {eval} on {split}.")
    elif all:
        handles = [train_model.spawn(c) for c in CONFIGS]
        print(f"Spawned {len(handles)} training jobs — running in parallel on Modal.")
    elif config:
        train_model.spawn(config)
        print(f"Spawned training for {config}.")
    else:
        print(
            "Usage:\n"
            "  --config <name>.yaml        train one multi-class model\n"
            "  --all                       train all 5 multi-class models\n"
            "  --eval <name>.yaml          evaluate one model\n"
            "  --eval-all                  evaluate all 5 models\n"
            "  --hierarchical-all          train all 3 cascade stages (WT, TC, ET)\n"
            "  --hierarchical <REGION>     train one cascade stage\n"
            "  --hierarchical-config <c>   config for hierarchical (default: default.yaml)\n"
            "  --cascade-test-eval-flag --variant {vanilla,uncertainty}   cascade test eval\n"
            "  --uncertainty               MC-dropout uncertainty eval\n"
            "  --classical                 classical hierarchical pipeline\n"
        )
        return
    print("Jobs run server-side. Track them at https://modal.com/apps")
