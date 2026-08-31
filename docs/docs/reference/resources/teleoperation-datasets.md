---
sidebar_position: 3
---

# Teleoperation Datasets

Teleoperation datasets are manually recorded sim2real episodes. They synchronize
the G1 state, the reference consumed by the motion tracker, optional hand and
neck commands, and RealSense RGB video. This format is intended for review and
external policy work; it is not a motion dataset for controller training.

## Record Episodes

Recording is available only for onboard Pico sim2real deployment with an
interactive terminal and a fresh RealSense stream:

```bash
pip install -e '.[recording]'
python scripts/run/run_sim2real.py --config-name sim2real_record \
    controller.policy_path=policy.onnx
```

The equivalent manual configuration requires `recording.enabled=true`,
`input.provider=pico4`, `input.video.enabled=true`, and
`input.video.source=realsense`.

Use `R` to start an episode, `S` to save it, `D` to discard it, and `Q` to shut
down. `STANDING`, `MOCAP`, `ARMS`, and paused mocap are recordable. Recording
does not start without a fresh camera frame. If the camera stays stale for one
second during an episode, that episode is discarded; Pico input and G1 control
continue, and recording does not restart automatically when video recovers.

## Dataset Layout

The recording runtime writes an editable dataset rather than one self-contained
HDF5 file:

```text
data/recordings/sim2real_hdf5/
├── schema.json
├── episodes.jsonl
├── data/
│   └── episode_000000.h5
└── videos/
    └── d435i_rgb/
        └── episode_000000.mp4
```

`schema.json` defines the dataset FPS, `robot_type`, `hand_type`, `neck_type`,
and every feature's shape, dtype, names, and groups. Hardware types must match
the active runtime configuration.

`episodes.jsonl` is the editable episode manifest. Each line maps one episode
to its HDF5 and MP4 files and stores the task prompt. Task text is not copied
into HDF5 attributes, so it can be edited without rewriting frame data.

## Frame Fields

Each HDF5 file contains only frame-aligned arrays:

| Field | Shape | Meaning |
|-------|-------|---------|
| `frame_index` | scalar | Camera/action frame index |
| `timestamp` | scalar | Monotonic timestamp in seconds |
| `observation.state` | `(68,)` | G1 joint state, base orientation/angular velocity, and projected gravity |
| `observation.state.hand` | `(12,)`, optional | Left/right LinkerHand hardware joint readback |
| `observation.state.neck` | `(2,)`, optional | OpenNeck servo yaw/pitch readback in degrees |
| `observation.mode` | scalar | `STANDING`, `MOCAP`, `ARMS`, or paused mocap code |
| `action` | `(36,)` | Root pose plus 29-joint reference consumed by the motion tracker |
| `action.hand` | `(12,)`, optional | Left/right LinkerHand target when hand control is enabled |
| `action.neck` | `(2,)`, optional | Mechanically clamped OpenNeck yaw/pitch target in degrees |

`observation.state` is ordered as `joint_pos(29)`, `joint_vel(29)`,
`base_quat_wxyz(4)`, `base_ang_vel(3)`, and `projected_gravity(3)`.
`observation.state.hand` uses the LinkerHand SDK's 0-255 joint values, ordered
as six left-hand channels followed by six right-hand channels.
`observation.state.neck` is `[yaw_deg, pitch_deg]` returned by OpenNeck
`read_deg()`.
`observation.mode` uses `standing=0`, `mocap=1`, `arms=2`, and `pause=3`.
`action` is `root_pos(3) + root_quat_wxyz(4) + reference_joint_pos(29)`.

Camera RGB is stored only in the MP4 sidecar; HDF5 does not duplicate raw image
frames. Optional state and action fields appear exactly when the corresponding
hardware is enabled.

## Commit and Recovery Rules

The recorder commits HDF5 and video files before appending the manifest entry.
An interrupted, uncommitted episode is removed on the next recording-worker
startup and does not consume an episode index. An incompatible existing
`schema.json` stops only the non-critical recording worker, while G1 control
continues.

## Review Recordings

```bash
python scripts/view/view_recording.py \
    --recording data/recordings/sim2real_hdf5
```

The reviewer validates manifest paths, HDF5 shapes, dtypes, finite values, and
MP4 alignment before playback. Measured root XYZ is not recorded, so the
observed robot is anchored to the reference root position; global root
translation cannot be evaluated from this format.
