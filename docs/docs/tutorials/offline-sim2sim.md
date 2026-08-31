---
sidebar_position: 1
---

# Run a Motion Controller in Simulation

Use this guide to watch a trained controller reproduce a motion in MuJoCo. This
is the quickest way to answer two basic questions before adding VR or a real
robot:

- Does the policy load and keep the G1 stable?
- Does the retargeted motion look like the source motion?

## Before You Start

Complete [Installation](../getting-started/installation) with the base profile
and the `robots gmr ckpt bvh` asset bundle.

## 1. Run the Sample Motion

```bash
python scripts/run/run_sim.py \
    controller.policy_path=ckpt/track_g1.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh \
    playback.keyboard.enabled=true
```

The `sim2sim` window is the result that matters: it shows the G1 produced by
physics and the policy, not just a kinematic target.

| Key | Action |
|-----|--------|
| `Space` or `P` | Pause or resume |
| `R` | Replay from the first frame |
| `Q` | Stop |

The run is healthy when the robot remains stable and follows the overall timing
and pose of the clip. Small tracking error is normal; falling, frozen joints or
a clearly wrong facing direction is not.

## 2. Compare the Three Views

Open all views when you need to find where a bad result starts:

```bash
python scripts/run/run_sim.py \
    controller.policy_path=ckpt/track_g1.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh \
    viewers=all
```

| View | What you are looking at |
|------|-------------------------|
| `mocap` | The human skeleton read from the BVH file |
| `retarget` | The kinematic G1 pose produced by GMR |
| `sim2sim` | The G1 after policy inference and MuJoCo physics |

If `mocap` is wrong, check the BVH format. If `mocap` looks right but
`retarget` does not, inspect the retargeting setup. If only `sim2sim` is wrong,
check the policy and observation configuration.

You can also select views explicitly:

```bash
# Only the physics result
python scripts/run/run_sim.py \
    controller.policy_path=ckpt/track_g1.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh \
    viewers=sim2sim

# No windows; useful for a server or timing test
python scripts/run/run_sim.py \
    controller.policy_path=ckpt/track_g1.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh \
    viewers=none
```

Closing every active viewer ends the simulation.

## 3. Try Your Own BVH

For a LAFAN1-style file:

```bash
python scripts/run/run_sim.py \
    controller.policy_path=ckpt/track_g1.onnx \
    input.bvh_file=/path/to/motion.bvh \
    input.bvh_format=lafan1
```

For an `hc_mocap` file:

```bash
python scripts/run/run_sim.py \
    controller.policy_path=ckpt/track_g1.onnx \
    input.bvh_file=/path/to/motion.bvh \
    input.bvh_format=hc_mocap
```

Teleopit does not guess an unknown skeleton layout. A file can be valid BVH and
still need an adapter before it matches a supported format.

## 4. Save a Video

Use the renderer when you want repeatable output instead of interactive
windows:

```bash
MUJOCO_GL=egl python scripts/render/render_sim.py \
    --bvh data/sample_bvh/aiming1_subject1.bvh \
    --policy ckpt/track_g1.onnx
```

Add `--format hc_mocap` for that input format. The renderer writes synchronized
`mocap`, `retarget` and `sim2sim` videos.

## Useful Playback Options

```bash
# Hold the final pose instead of exiting
playback.pause_on_end=true

# Stop after 300 simulation steps; 0 means no step limit
num_steps=300

# Keep wall-clock timing even with no viewer
realtime=true
```

For every available field, see
[Configuration](../reference/configuration/overview).
