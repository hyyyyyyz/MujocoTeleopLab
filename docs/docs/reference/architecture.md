---
sidebar_position: 2
---

# Architecture

This page defines Teleopit's runtime pipelines, repository layout, supported
technical surface, and public entry points.

## Pipeline

![Teleopit runtime pipelines](/img/diagrams/architecture-pipeline.svg)

The main tracking path converts BVH or live PICO body motion into a time-aligned
G1 reference. `VelCmdObservationBuilder` combines that reference with robot
state, and the dual-input TemporalCNN ONNX controller produces 29 joint offsets.
The same observation and controller path drives MuJoCo and the real G1.

Pico hand and active-vision paths are optional process-isolated workers. They
reuse the same in-process `PicoBridge` receiver and never add fields to the 167D
tracking-policy observation. A hand or neck failure must not stop G1 body
control. These optional hardware paths are supported by onboard deployment;
external-host Pico deployment supports whole-body control only.

Host-policy deployment is independent from the Pico runtime. A separate host
environment receives JPEG RGB, measured G1 joint positions, raw measured O6
readback, measured OpenNeck angles, and an observation-time source reference
root pose. The body/hand/neck arrays form the 43D model observation; the
session-local source pose only anchors reconstruction of source-relative root
output. The host returns canonical `float32[T,50]` action chunks over strict
ZeroMQ/msgpack messages. The onboard validator and scheduler convert the body
portion into a 36D reference for the existing motion tracker; host output never
bypasses that tracker or becomes a direct motor command.

The Teleopit and host environments share semantic data and one identical
`hand_calibration.json`, but do not import each other's Python packages. The
current client/server code and protocol tests define the network structure, so
both repositories must change together when that protocol changes.

## Runtime Boundaries

- Offline core components communicate through `InProcessBus` without copying
  array payloads.
- Sim2real robot control, reference generation, camera, recording, hand, neck,
  and host-policy client work are process-isolated where blocking or hardware
  failure could disturb the 50 Hz control loop.
- Local sim2real workers use localhost ZeroMQ and shared-memory video rings.
- The external host-policy boundary uses msgpack and non-pickle float32 arrays.
- Shared component contracts are `typing.Protocol` definitions in
  `teleopit/interfaces.py`.

## Repository Layout

```text
teleopit/                              — Core inference and deployment package
├── interfaces.py                     — Robot, controller, input and retargeting protocols
├── pipeline.py                       — Thin offline simulation facade
├── runtime/                          — Config/path resolution, factories and CLI validation
├── configs/                          — Hydra runtime configuration
├── bus/                              — In-process zero-copy publish/subscribe
├── inputs/                           — BVH, PICO and realtime input adapters
├── retargeting/gmr/                  — Self-contained whole-body GMR implementation
├── controllers/                      — Observation builder and ONNX policy controller
├── robots/                           — MuJoCo robot adapter
├── sim/                              — 200 Hz PD / 50 Hz policy simulation loop
├── sim2real/
│   ├── mp/                           — Process supervisor, IPC and robot-control state machine
│   ├── hands/                        — Optional LinkerHand drivers and input mapping
│   └── neck/                         — Optional OpenNeck mapping and worker
├── high_level_policy/                — Host protocol, frame transforms and action scheduler
└── recording/                        — Sim2real dataset schema and recording workers

train_mimic/                          — Training package
├── app.py                            — Shared train/play/benchmark assembly
├── tasks/tracking/                   — General-Tracking-G1 task and TemporalCNN model
├── data/                             — Dataset construction and motion loading
└── scripts/                          — Training, playback, benchmark and ONNX export

scripts/                              — User-facing runtime and maintenance entry points
├── run/                              — Simulation, sim2real and recording commands
├── setup/                            — Asset download and hardware setup
├── render/                           — Offline video rendering
├── view/                             — Recording review
└── dev/                              — Validation and calibration utilities

third_party/                          — Optional hardware SDKs and somehand
tests/                                — Unit, protocol and integration tests
```

## Technical Specifications

| Specification | Supported value |
|---------------|-----------------|
| Robot | Unitree G1 with 29 actuated joints |
| Simulator | MuJoCo |
| Whole-body retargeting | GMR (General Motion Retargeting) |
| Policy / PD rates | 50 Hz / 200 Hz |
| Training task | `General-Tracking-G1` |
| Inference observation | `velcmd_history` (167D) |
| ONNX signature | Dual input: `obs` (167D) + `obs_history` |
| Policy action | 29D joint offsets from `default_dof_pos` |
| Actor / critic | TemporalCNN (2048, 1024, 512, 256, 128) |
| Training sampling | `rewind` by default; `uniform` supported; playback uses `start`; benchmark pins exact clips and disables clip-end resampling |
| Training window | `window_steps=[0]` |
| Distributed motion data | Minimal recursive HDF5 `shard_*.h5` files |
| Optional hands | LinkerHand L6/O6 with gripper or PICO hand-pose input |
| Optional active vision | OpenNeck yaw/pitch in physical degrees |
| Host-policy observation | JPEG RGB + G1 joint position (29D) + raw O6 readback (12D) + OpenNeck degrees (2D); request also carries the camera-time active reference root pose (7D) |
| Host-policy action | `float32[T,50]`, 30 Hz source horizon, `T` in `[1,50]` |
| Host-policy body control | 36D root/joint reference through the existing 50 Hz motion tracker |

## Constraints

- `controller.policy_path` must be explicit and point to an existing file.
- Offline BVH runs require an explicit, existing `input.bvh_file`.
- `viewers` is the only viewer configuration key.
- Observation definitions and ONNX signatures must match exactly; startup fails
  instead of padding or trimming data.
- `default_dof_pos` must come from the selected robot's default standing angles.
- Sim2real requires the same dual-input observation contract used in simulation.
- Host message-envelope or schema mismatches are rejected while the robot
  remains in `STANDING`. Shape, finiteness, session, sequence, quaternion,
  staleness, and safety violations reject the whole action chunk.
- Host actions are validated, scheduled, and rate-limited onboard. The host
  cannot bypass the motion tracker or send G1 motor commands.
- Policy entry remains an internal `STANDING` flow while one host session waits
  for its first valid chunk. That chunk enters `POLICY` directly, with no
  candidate alignment, entry Kp ramp, or second session/reset. The 50 Hz limiter
  starts from the measured robot reference captured at session start.
- Temporal root, yaw, and joint-reference discontinuities are accepted at chunk
  boundaries and inside chunks, then rate-limited at the 50 Hz scheduler output
  so recorded pause/resume transitions remain usable.
- PICO input, RealSense preview, recording, hand, and neck failures are
  non-critical; the Unitree remote and robot-control loop remain available.

## Public Entry Points

Supported run modes are offline sim2sim, offline sim2real playback, PICO
sim2sim, PICO G1 sim2real, and independent host-policy G1 sim2real.

Runtime commands:

- `scripts/run/run_sim.py` — offline BVH and live PICO sim2sim
- `scripts/run/run_sim2real.py` — BVH or PICO G1 sim2real
- `scripts/run/run_high_level_policy_sim2real.py` — independent host-policy G1 deployment
- `scripts/run/record_pico_motion.py` — record retargeted motion clips from PICO
- `scripts/render/render_sim.py` — render mocap, retargeting, and sim2sim videos
- `scripts/view/view_recording.py` — review synchronized sim2real recordings

Training and data commands:

- `train_mimic/scripts/train.py`, `play.py`, `benchmark.py`, `save_onnx.py`
- `train_mimic/scripts/data/build_dataset.py`
- `train_mimic/scripts/data/precompute_dataset.py`

Public Python surfaces:

- Protocols in `teleopit/interfaces.py`
- `TeleopPipeline`
- `VelCmdObservationBuilder`
- `RLPolicyController`
