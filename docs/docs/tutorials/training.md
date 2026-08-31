---
sidebar_position: 4
---

# Train a Motion Controller

This guide starts with downloaded motion data and ends with an ONNX controller
that Teleopit can run in simulation or on a G1.

The normal training path assumes an NVIDIA GPU. Motion data is loaded into
memory at startup, so larger combined datasets also need enough system and GPU
memory.

## Before You Start

Follow [Installation](../getting-started/installation) with:

- the `train` profile, and
- the `robots data` asset bundle.

Verify the training package:

```bash
python -c "import train_mimic.tasks; print('training OK')"
```

## 1. Prepare the Downloaded Dataset

Downloaded datasets are compact distribution files. Training uses a second
directory with joint velocities and body kinematics precomputed:

```bash
python train_mimic/scripts/data/precompute_dataset.py \
    data/datasets \
    --outdir data/datasets_precomputed \
    --jobs 8
```

Use `data/datasets_precomputed` for every training, playback and benchmark
command below. Pointing training at the original `data/datasets` directory is
an error, not a supported shortcut.

For custom BVH, PKL, NPZ or Pico-recorded data, see
[Motion Datasets](../reference/resources/motion-datasets).

## 2. Choose the Robot Model

Use `--robot_xml` to select the MuJoCo model used by training. If the argument
is omitted, it defaults to:

```text
assets/robots/unitree_g1/g1_29dof.xml
```

The current `robots` asset bundle includes these ready-to-use examples:

| Model XML | Setup |
|-----------|-------|
| `assets/robots/unitree_g1/g1_29dof.xml` | Base G1 model and the default |
| `assets/robots/unitree_g1/g1_29dof_dex3.xml` | G1 with Dex3 hand geometry and inertial properties |
| `assets/robots/unitree_g1/g1_29dof_neck_o6.xml` | G1 with neck active vision and O6 hand models |

This table describes the models shipped in the current asset bundle; it is not
a hard-coded model allowlist. Another XML can be passed when its joint and body
definitions are compatible with the selected task configuration and dataset.

The full-training command below explicitly selects the base model as a copyable
example. Replace that path with the model you want to train. The other commands
do not repeat this option; playback and benchmark load the robot from the
selected task configuration.

## 3. Run a Short Smoke Test

Before starting a long job, verify that the dataset, simulator and logger work
together:

```bash
python train_mimic/scripts/train.py \
    --num_envs 64 \
    --max_iterations 100 \
    --motion_file data/datasets_precomputed
```

The test is successful when environments step, losses are reported and a run
directory appears under `logs/rsl_rl/g1_general_tracking/`.

## 4. Start a Full Run

```bash
python train_mimic/scripts/train.py \
    --robot_xml assets/robots/unitree_g1/g1_29dof.xml \
    --num_envs 4096 \
    --max_iterations 30000 \
    --motion_file data/datasets_precomputed
```

Reduce `--num_envs` if GPU memory is insufficient. The default logger is
TensorBoard; choose `--logger wandb` or `--logger swanlab` when required.

`--max_iterations` means additional iterations. For example, resuming
`model_12000.pt` with `--max_iterations 18000` continues to iteration 30000.

## 5. Watch the Checkpoint in Simulation

```bash
python train_mimic/scripts/play.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --motion_file data/datasets_precomputed
```

Playback starts clips from their beginning and removes training noise. Use it
to catch an obviously unstable policy before exporting.

## 6. Run the Benchmark

```bash
python train_mimic/scripts/benchmark.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --motion_file data/datasets_precomputed \
    --num_envs 32
```

The benchmark evaluates one deterministic 10-second rollout for every eligible
clip. It reports:

- mean per-joint position error (`MPJPE`),
- root position, rotation and velocity error, and
- rollout success rate.

Results are written as a text summary, JSON, per-clip CSV and per-rollout CSV.

## 7. Export ONNX

```bash
python train_mimic/scripts/save_onnx.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --output ckpt/track_g1.onnx \
    --history_length 10
```

The result must be a dual-input TemporalCNN with `obs` and `obs_history`.
Teleopit validates the 167D observation signature at startup and rejects an
incompatible export.

Test the export in the normal runtime:

```bash
python scripts/run/run_sim.py \
    controller.policy_path=ckpt/track_g1.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh
```

## Scale to Multiple GPUs

For one machine:

```bash
python train_mimic/scripts/train.py \
    --gpu_ids 0 1 2 3 \
    --num_envs 1024 \
    --max_iterations 30000 \
    --motion_file data/datasets_precomputed
```

`--num_envs` is per GPU.

For multiple machines, launch the same script with `torchrun`:

```bash
torchrun \
    --nnodes=$PET_NNODES \
    --nproc_per_node=$PET_NPROC_PER_NODE \
    --node_rank=$PET_NODE_RANK \
    --master_addr=$PET_MASTER_ADDR \
    --master_port=$PET_MASTER_PORT \
    train_mimic/scripts/train.py \
    --num_envs 1024 \
    --max_iterations 1000 \
    --motion_file data/datasets_precomputed
```

Here `--num_envs` is per process, so the total scales with the world size.

## Common Problems

| Symptom | What to check |
|---------|---------------|
| Loader says the dataset is minimal | Run `precompute_dataset.py` and use its output directory |
| Out of GPU memory | Lower `--num_envs` or train with fewer precomputed data shards |
| Out of system memory during startup | Train on fewer precomputed shards or add RAM |
| Training is unexpectedly slow | Check that PyTorch detects CUDA and that the training device is a CUDA GPU |

For task internals and model dimensions, see
[Architecture](../reference/architecture).
