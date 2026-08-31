---
sidebar_position: 1
---

# Assets

Teleopit's Git repository contains code, not large robot meshes, policies or
motion data. [Installation](../../getting-started/installation) shows the shortest
download command for each user workflow; this page is the complete inventory
and maintainer reference.

## What's Not in Git

- `assets/robots/` - Canonical robot XML/meshes
- `teleopit/retargeting/gmr/assets/` - GMR retargeting assets, IK configs, and non-canonical robot descriptions
- `data/`, `ckpt/`, checkpoints, caches
- Demo media (`assets/demo.gif`, `assets/demo.mp4`)

## Asset Inventory

| Group | Local result | Used for |
|-------|--------------|----------|
| `ckpt` | `ckpt/track_g1.{onnx,pt}`, `ckpt/track_g1_neck_o6.{onnx,pt}` | Ready-to-run inference models and matching PyTorch checkpoints |
| `robots` | Robot XML variants and meshes under `assets/robots/` | Training, MuJoCo inference, GMR and dataset FK |
| `gmr` | `teleopit/retargeting/gmr/assets/` | Retargeting models and IK configuration |
| `bvh` | `data/sample_bvh/*.bvh` | Sample motions used by the installation check and simulation tutorial |
| `data` | `data/datasets/<dataset>/shard_*.h5` | Minimal distributed motion datasets; precompute before training |

The current G1 robot bundle includes:

| Model XML | Setup |
|-----------|-------|
| `assets/robots/unitree_g1/g1_29dof.xml` | Base G1 model and the default |
| `assets/robots/unitree_g1/g1_29dof_dex3.xml` | G1 with Dex3 hand geometry and inertial properties |
| `assets/robots/unitree_g1/g1_29dof_neck_o6.xml` | G1 with neck active vision and O6 hand models |

The default is not a model allowlist. Training can select another
task-compatible XML with `--robot_xml`. XML files in the GMR asset directory
belong to their retargeting configurations and are separate from the runtime
robot bundle. Use the `track_g1` policy pair with the base model and the
`track_g1_neck_o6` pair with the neck-and-O6 model.

## Repositories

### ModelScope (default download source)

| Repository | Type | Contents |
|-----------|------|----------|
| `BingqianWu/Teleopit-models` | model | Checkpoints, GMR retargeting assets, sample BVH |
| `BingqianWu/Teleopit-datasets` | dataset | Training/validation datasets |

### HuggingFace (alternative)

| Repository | Type | Contents |
|-----------|------|----------|
| `12e21/Teleopit-models` | model | Checkpoints, GMR retargeting assets, sample BVH |
| `12e21/Teleopit-datasets` | dataset | Training/validation datasets |

### Asset Group and Repository Mapping

| Group | Repository | Remote Path |
|-------|-----------|-------------|
| `ckpt` | Teleopit-models | `checkpoints/track_g1.{onnx,pt}`, `checkpoints/track_g1_neck_o6.{onnx,pt}` |
| `robots` | Teleopit-models | `archives/robot_assets.tar.gz` |
| `gmr` | Teleopit-models | `archives/gmr_assets.tar.gz` |
| `bvh` | Teleopit-models | `archives/sample_bvh.tar.gz` |
| `data` | Teleopit-datasets | `data/datasets/*/*.h5` (`lafan1`, `pico_record`, `seed`, `twist2`) |

## Download Behavior

Use the project download script (defaults to ModelScope):

```bash
# Download everything
python scripts/setup/download_assets.py

# Only inference essentials
python scripts/setup/download_assets.py --only robots gmr ckpt bvh

# Only training data
python scripts/setup/download_assets.py --only data

# Download from HuggingFace instead
python scripts/setup/download_assets.py --source huggingface
```

Local paths after download:

| Remote | Local |
|--------|-------|
| `checkpoints/track_g1.onnx` | `ckpt/track_g1.onnx` |
| `checkpoints/track_g1.pt` | `ckpt/track_g1.pt` |
| `checkpoints/track_g1_neck_o6.onnx` | `ckpt/track_g1_neck_o6.onnx` |
| `checkpoints/track_g1_neck_o6.pt` | `ckpt/track_g1_neck_o6.pt` |
| `archives/robot_assets.tar.gz` | `assets/robots/` (extracted) |
| `archives/gmr_assets.tar.gz` | `teleopit/retargeting/gmr/assets/` (extracted) |
| `archives/sample_bvh.tar.gz` | `data/sample_bvh/` (extracted) |
| `data/datasets/*/*.h5` | `data/datasets/` |

## Upload to ModelScope

### Step 1: Prepare Upload Directory

```bash
python scripts/setup/prepare_modelscope_assets.py --only ckpt robots gmr bvh --clean
python scripts/setup/prepare_modelscope_assets.py --only data
```

Output goes to `data/modelscope_upload/`.

### Step 2: Upload

```bash
# Model repo
modelscope upload --repo-type model BingqianWu/Teleopit-models \
    data/modelscope_upload/checkpoints checkpoints --sync
modelscope upload --repo-type model BingqianWu/Teleopit-models \
    data/modelscope_upload/archives archives

# Dataset repo
modelscope upload --repo-type dataset BingqianWu/Teleopit-datasets \
    data/modelscope_upload/data data
```

The checkpoint upload intentionally uses `--sync`. Its deletion scope is the
remote `checkpoints/` directory, so obsolete policy names are removed without
touching `archives/`. Do not add `--sync` to the archive upload unless the local
staging directory contains every remote archive that must be retained.

### Step 3: Tag Version

Only the model repo supports tags (dataset repo does not).

```bash
python - <<'EOF'
from modelscope.hub.api import HubApi
api = HubApi()
url = api.create_model_tag("BingqianWu/Teleopit-models", "vX.Y.Z")
print(url)
EOF
```

Tags should match Git tags for traceability.

## Upload to HuggingFace

### Step 1: Prepare and Upload

```bash
# Prepare and upload model assets (--clean ensures no leftover files)
python scripts/setup/upload_hf_assets.py --only ckpt robots gmr bvh --clean

# Prepare and upload dataset
python scripts/setup/upload_hf_assets.py --only data --clean
```

Use `--dry-run` to stage files locally without uploading.

:::warning
Always use `--clean` when running `--only`, otherwise the staging directory may carry leftover files from a previous run, causing unintended uploads.
:::

### Step 2: Tag Version

```bash
python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
api.create_tag("12e21/Teleopit-models", tag="vX.Y.Z", repo_type="model")
EOF
```

## Pre-Push Check

```bash
python scripts/dev/check_large_tracked_files.py
```

This blocks large binary files and checks tracked file size limits.
