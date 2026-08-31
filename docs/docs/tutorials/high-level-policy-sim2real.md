---
sidebar_position: 4
---

# From Teleoperation Data to Imitation Learning / VLA Deployment

This guide connects the complete workflow: record Pico demonstrations with
Teleopit, train an ACT or GR00T N1.7 policy in `lerobot-teleopit`, and run the
result on a physical Unitree G1.

```text
Pico demonstration
  -> Teleopit v4 recording
  -> LeRobot Dataset
  -> ACT or GR00T checkpoint
  -> host policy server
  -> Teleopit onboard motion tracker
  -> G1 + LinkerHand O6 + OpenNeck
```

Teleopit owns recording and real-time robot control.
[`lerobot-teleopit`](https://github.com/BotRunner64/lerobot-teleopit) owns
dataset conversion, model training and the host policy server. Keep their
Python environments separate; the host sends reference motion, not G1 motor
commands.

## Before You Start

- [VR Teleoperation on Unitree G1](pico-sim2real) works reliably.
- The onboard setup has two LinkerHand O6 hands, OpenNeck and a RealSense RGB
  camera. The current training and deployment path requires all of them.
- The onboard computer is prepared through
  [Installation](../getting-started/installation) with recording, OpenNeck,
  LinkerHand and somehand support, plus `ckpt/track_g1_neck_o6.onnx`.
- The [Standalone Standing Test](standalone-standing) is stable with the same
  G1 network interface and low-level tracking policy.
- The host workstation is prepared separately through the
  [`lerobot-teleopit` installation guide](https://github.com/BotRunner64/lerobot-teleopit#installation).

:::danger Keep the Unitree remote in your hand
Use `L1+R1` to enter `DAMPING` whenever motion is unexpected. Keep clear space
around the robot, have another person ready to support or stop it, and never
run two programs that can command the G1 at the same time.
:::

## 1. Record and Review Demonstrations

Run the recording configuration on the G1 onboard computer. This example uses
Pico hand-pose retargeting; use `hands.mode=gripper` when demonstrations should
use the controller triggers instead.

```bash
python scripts/run/run_sim2real.py \
  --config-name sim2real_record \
  controller.policy_path=ckpt/track_g1_neck_o6.onnx \
  real_robot.network_interface=eth0 \
  hands.enabled=true \
  hands.driver=linkerhand_o6 \
  hands.mode=vr_hand_pose \
  neck.enabled=true \
  recording.output_dir=data/recordings/my_task \
  recording.task="pick up the object"
```

Use the G1 remote to enter `MOCAP` or `ARMS`, then use the recording terminal:

| Key | Action |
|-----|--------|
| `R` | Start an episode after a fresh RealSense frame is available |
| `S` | Save the active episode |
| `D` | Discard the active episode |
| `Q` | Shut down the runtime |

Record one task per dataset and keep `recording.task` consistent. Save only
successful demonstrations, while varying useful factors such as starting pose,
object position and execution speed.

Review the synchronized video, measured state and reference before training:

```bash
python scripts/view/view_recording.py \
  --recording data/recordings/my_task
```

Discard episodes with tracking loss, camera interruption or unsafe references.
For the recording schema and recovery rules, see
[Teleoperation Datasets](../reference/resources/teleoperation-datasets).

## 2. Hand the Dataset to `lerobot-teleopit`

Copy the complete recording directory to the host without flattening or
renaming its contents. A typical source directory is:

```text
lerobot-teleopit/data/raw/my_task/
├── schema.json
├── episodes.jsonl
├── data/
└── videos/d435i_rgb/
```

The current converter requires a Teleopit v4 dataset with LinkerHand O6 and
OpenNeck state/action fields. Missing fields are rejected rather than padded.
If the same task was recorded in several directories, use the host repository's
`merge_raw_datasets.py` tool before conversion.

Run all remaining host commands inside the independent `lerobot-teleopit`
environment. Its
[Dataset Conversion and Training guide](https://github.com/BotRunner64/lerobot-teleopit/blob/main/docs/training-entrypoint.md)
covers dependencies, merge options, training scales, multi-GPU settings and
logging.

## 3. Convert and Train on the Host

The shortest conversion command is:

```bash
python scripts/convert_dataset.py \
  --source data/raw/my_task \
  --output data/lerobot/my_task \
  --repo-id local/my_task \
  --workers 4
```

Choose one training command. For ACT:

```bash
python scripts/train_policy.py \
  --policy act \
  --dataset-root data/lerobot/my_task \
  --devices 0
```

For GR00T N1.7:

```bash
python scripts/train_policy.py \
  --policy groot \
  --dataset-root data/lerobot/my_task \
  --devices 0,1,2,3
```

Append `--dry-run` to verify the resolved launch without starting training.
Unless `--output-dir` is set, runs are created under `outputs/train/`. The
deployable artifact is the run's `checkpoints/last/pretrained_model/`
directory.

## 4. Validate the Robot Path with ReplayPolicy

Before loading a learned checkpoint, replay a recorded episode from the host:

```bash
python scripts/run_policy_server.py \
  --backend replay \
  --dataset-root data/lerobot/my_task \
  --repo-id local/my_task \
  --episode 0 \
  --start-frame 0 \
  --chunk-size 15 \
  --bind tcp://0.0.0.0:5555
```

Bind to `0.0.0.0` only on the trusted robot network. The service has no
authentication and must not be exposed to the public internet.

On the G1 onboard computer, start Teleopit's dedicated runtime. Replace
`HOST_IP` with the workstation address and use the same task wording as the
dataset:

```bash
python scripts/run/run_high_level_policy_sim2real.py \
  controller.policy_path=ckpt/track_g1_neck_o6.onnx \
  high_level_policy.endpoint=tcp://HOST_IP:5555 \
  high_level_policy.task="pick up the object" \
  real_robot.network_interface=eth0
```

Starting the process leaves the robot in `IDLE`. Use the Unitree remote:

| Control | Action |
|---------|--------|
| `Start` | Enter `STANDING` |
| `Y` | Start a policy session; the first valid chunk enters `POLICY` |
| `B` | Pause or resume after a fresh chunk is available |
| `X` | End the session and return to `STANDING` |
| `L1+R1` | Immediately enter `DAMPING` |

ReplayPolicy should reproduce the recorded reference closely enough to verify
the network, action convention and onboard execution path. Stop here if it
does not. A learned policy cannot fix a recording, conversion, coordinate or
low-level tracking problem.

## 5. Deploy the Trained Policy

Press `X` to return the G1 to `STANDING`, then stop ReplayPolicy. On the host,
start the learned-policy server with the `pretrained_model` directory itself:

```bash
python scripts/run_policy_server.py \
  --backend lerobot \
  --checkpoint outputs/train/<run>/checkpoints/last/pretrained_model \
  --device cuda \
  --bind tcp://0.0.0.0:5555
```

ACT and GR00T use the same server command. Press `Y` on the Unitree remote to
create a new policy session. Begin with a familiar scene from the training
distribution and small, recoverable motions.

To record the observations and actions exchanged during a run, add this host
option:

```bash
--record-dir outputs/policy-recordings
```

Teleopit validates and rate-limits each returned plan before the 50 Hz motion
tracker consumes it. Malformed output is rejected rather than padded or
trimmed. A host, network, camera or action-watchdog fault pauses the session
and holds the latest commands; it does not automatically enter `STANDING`.
Restore the failed path and press `B` to resume, or use `X` or `L1+R1` as
appropriate.

## Common Problems

| Symptom | What to check |
|---------|---------------|
| Pressing `Y` never enters `POLICY` | Host IP and firewall, server logs, a fresh RealSense frame, matching code versions and identical `hand_calibration.json` files |
| `POLICY` becomes paused | Host inference latency, request timeout, stale camera/result, action watchdog or a required worker exit |

For model action coordinates and host-side behavior, see the
[`lerobot-teleopit` Action Space guide](https://github.com/BotRunner64/lerobot-teleopit/blob/main/docs/planar-relative-root-actions.md).
For onboard timing and safety settings, see
[Configuration Fields](../reference/configuration/fields#host-high-level-policy-independent-sim2real).
