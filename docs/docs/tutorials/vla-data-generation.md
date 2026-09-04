---
sidebar_position: 6
---

# Automatic VLA data generation

Teleopit includes an automatic data-generation path for the 43-DOF tabletop
scenes. It is simulation-only: no PICO headset, XR bridge or real robot is
needed. The default planner is CuRobo `MotionGen`: it builds a collision world
from the MuJoCo table/floor, solves approach/grasp/lift/place segments in joint
space, and executes the result through the 43-DOF PD loop. It records MuJoCo
camera frames, state, target action, object pose and a language instruction.

CuRobo requires a CUDA-enabled PyTorch environment. If it is unavailable, the
command fails explicitly; use `--planner scripted` only for a data-pipeline
smoke test, never as collision-planned training data.

On a CUDA workstation, install the pinned CuRobo checkout with
`bash scripts/setup/setup_curobo.sh`. The default `.venv_scene` environment is
otherwise intentionally usable without CuRobo.

First install/build the selected object scene, then generate episodes:

```bash
.venv/bin/python scripts/setup/download_scene_object.py can
.venv_scene/bin/python scripts/setup/build_scene_with_object.py can
.venv_scene/bin/python scripts/run/generate_vla_scene_data.py \
  --scene can --planner curobo --episodes 10 --output-dir outputs/vla_scene_data
```

The ignored output contains `schema.json`, `episodes.jsonl`, one compressed
NPZ per attempt, an image directory and an MP4 diagnostic per attempt.
`success` requires real finger contact, at least 3 cm of object lift, at least
10 cm horizontal transfer, and settling back near the tabletop. Failed
attempts remain in the output for debugging and must be excluded from training.

To collect a fixed number of valid demonstrations while retaining failed
attempts for inspection, set a larger attempt budget:

```bash
.venv_scene/bin/python scripts/run/generate_vla_scene_data.py \
  --scene cube --planner curobo \
  --successful-episodes 100 --episodes 140 \
  --output-dir outputs/vla_scene_data_curobo
```

The command exits with status 2 if the success target is not reached within
the attempt budget.

## Replay and validation

采集后建议先离线重放，确认动作确实能在同一场景中复现，并检查状态是否
与记录一致：

```bash
.venv_scene/bin/python scripts/run/replay_vla_scene_data.py \
  --scene can \
  --episode outputs/vla_scene_data_curobo/episode_000000.npz \
  --render-dir outputs/vla_scene_replays/episode_000000
```

工具从场景初始状态重新执行 43-D action，输出 JPEG 帧、`replay.mp4` 和
`replay_report.json`。报告包含状态最大绝对误差、RMSE、物体位姿误差和
物体位移；命令仅在 `success` 且状态误差不超过 `--state-tolerance` 时返回
零退出码，可用于批量过滤无效 episode。

The planner is a replaceable interface. `CuroboSceneTrajectoryPlanner` is the
production backend; `ScriptedPickPlacePlanner` exists only for plumbing tests.
