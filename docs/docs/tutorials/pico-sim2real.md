---
sidebar_position: 3
---

# VR Teleoperation on Unitree G1

This guide moves the Pico workflow from MuJoCo to a physical Unitree G1. First
choose where Teleopit will run, then verify standing control before handing the
robot over to live body tracking.

:::danger Keep the Unitree remote in your hand
Use `L1+R1` to enter `DAMPING` whenever motion is unexpected. Keep clear space
around the robot and have another person ready to support or stop it.
:::

## Choose a Deployment

### External host: whole-body tracking only

Run Teleopit on a workstation or laptop connected to G1 by Ethernet. The Pico
headset must be able to reach this computer over the network.

This deployment is for G1 whole-body control only. Keep LinkerHand, OpenNeck,
RealSense preview and recording disabled. Find the wired interface connected
to G1:

```bash
ifconfig
```

Use that interface name in the commands below. The examples use `enp130s0`.

### Onboard computer: full embodiment

Run Teleopit directly on the G1 onboard computer when you also need LinkerHand,
OpenNeck, RealSense preview or data collection. The Pico headset must be able to
reach the onboard computer.

When the onboard setup includes both O6 hands and OpenNeck, set the low-level
tracking policy to
`controller.policy_path=ckpt/track_g1_neck_o6.onnx`.

The G1 DDS interface is `eth0` by default. Apart from the network interface and
the optional onboard hardware settings, the body-control configuration and
launch command are the same as for an external host.

## Before You Start

Do not continue until all of these are true:

- [VR Teleoperation in Simulation](pico-sim2sim) works reliably.
- You installed the `pico4` profile and built `g1_bridge_sdk` as described in
  [Installation](../getting-started/installation).
- `ckpt/track_g1.onnx`, the robot files and GMR assets are present.
- The machine running Teleopit has a wired DDS connection to G1.
- No other program is commanding the robot.

## 1. Check Standing Control

Check state reception and policy timing without sending motor commands. On an
external host, replace `enp130s0` with the interface reported by `ifconfig`:

```bash
python scripts/run/standalone_standing.py \
    --policy ckpt/track_g1.onnx \
    --network-interface enp130s0 \
    --dry-run
```

On the onboard computer, use `--network-interface eth0`.

If the dry run succeeds, repeat the command without `--dry-run` in a safe
hardware setup:

```bash
python scripts/run/standalone_standing.py \
    --policy ckpt/track_g1.onnx \
    --network-interface enp130s0
```

Stop here if standing control is not stable. Follow the
[Standalone Standing Test](standalone-standing) before adding Pico input.

## 2. Start Pico Sim2Real

External-host example:

```bash
python scripts/run/run_sim2real.py \
    --config-name pico4_sim2real \
    controller.policy_path=ckpt/track_g1.onnx \
    real_robot.network_interface=enp130s0
```

Onboard-computer example:

```bash
python scripts/run/run_sim2real.py \
    --config-name pico4_sim2real \
    controller.policy_path=ckpt/track_g1.onnx \
    real_robot.network_interface=eth0
```

Starting the program does not immediately give Pico control of the robot.

## 3. Use the G1 State Machine

![Pico G1 control state machine](/img/diagrams/pico-g1-state-machine.svg)

Labels beginning with **G1 remote** refer to the Unitree remote. Labels
beginning with **Pico controller** refer to the VR controllers. The computer
keyboard does not switch robot modes.

Press **G1 remote** `Start` to enter `STANDING`. Wait until the robot is stable,
stand in a neutral pose and make sure Pico tracking is valid. Then press
**G1 remote** `Y` to enter `MOCAP`, and begin with small, slow movements. Press
**G1 remote** `X` when you want to end the VR session and return to `STANDING`.

`MOCAP` follows the whole body. `ARMS` keeps the body, waist and legs in the
standing pose while both arms continue to follow. `PAUSED` holds the current
reference; resuming returns to the previous `MOCAP` or `ARMS` state.

Teleopit checks several consecutive Pico frames before entering `MOCAP`. If
that check fails, the robot stays in `STANDING`.

:::tip Pause and resume
G1 remote `B` or Pico controller `A` pauses and resumes the current session.
Resume while standing still and close to the held pose. Use G1 remote `X`
instead when you want to end the session.
:::

If Pico input stops, body control holds the last reference and the G1 remote
remains available. Use `X` to return to `STANDING`, or `L1+R1` to enter
`DAMPING`; do not wait for an automatic mode change.

## Onboard Only: LinkerHand

Skip this section unless LinkerHand hardware is connected to the onboard
computer. Install the hand packages from
[Installation](../getting-started/installation), then bring up both CAN
interfaces:

```bash
sudo /usr/sbin/ip link set can0 up type can bitrate 1000000
sudo /usr/sbin/ip link set can1 up type can bitrate 1000000
```

Test both hands before starting G1 control:

```bash
python scripts/dev/test_linkerhand.py \
    --driver linkerhand_o6 \
    --hand-type both \
    --left-can can0 \
    --right-can can1
```

Enable O6 hand-pose control by adding these overrides to the sim2real command:

```text
hands.enabled=true
hands.driver=linkerhand_o6
hands.mode=vr_hand_pose
hands.linkerhand_o6.left_can=can0
hands.linkerhand_o6.right_can=can1
```

With `hands.mode=gripper`, hold the controller's side grip trigger to enable
that hand, then use the index trigger to control how far it closes. Releasing
the side grip trigger commands that hand to open. LinkerHand L6 is also
supported through the matching `hands.linkerhand_l6.*` settings.

## Onboard Only: OpenNeck

Install and calibrate OpenNeck:

```bash
pip install -e '.[openneck]'
openneck calibrate
```

Then add these overrides to the sim2real command:

```text
neck.enabled=true
neck.port=/dev/ttyACM0
```

OpenNeck follows the Pico HMD relative to the operator's upper body. It reuses
the existing Pico receiver.

## Onboard Only: RealSense Preview

Install `pyrealsense2`, then add:

```text
input.video.enabled=true
input.video.device=<optional-realsense-serial>
```

The camera view is sent to the headset. A timeout restarts the camera in the
background without stopping Pico tracking or G1 control.

## Onboard Only: Record and Review Data

Recording requires a fresh RealSense RGB frame:

```bash
python scripts/run/run_sim2real.py \
    --config-name sim2real_record \
    controller.policy_path=ckpt/track_g1.onnx \
    real_robot.network_interface=eth0 \
    recording.task="walk forward"
```

Use terminal `R` to start an episode, `S` to save it, `D` to discard it and
`Q` to shut down. If no fresh camera frame arrives for one second, the active
episode is discarded while robot control continues. Start a new episode
manually after video recovers.

Review the saved recording with:

```bash
pip install -e '.[review]'
python scripts/view/view_recording.py \
    --recording data/recordings/sim2real_hdf5
```

The viewer synchronizes camera video, measured and reference G1 poses, and
optional hand and neck signals. See
[Teleoperation Datasets](../reference/resources/teleoperation-datasets)
for the stored fields.

## Common Problems

| Problem | Solution |
|---------|----------|
| RealSense does not work on Arm | Remove the PyPI wheel with `pip uninstall pyrealsense2`, then install the conda-forge Arm build with `conda install -c conda-forge pyrealsense2` |

## Other G1 Workflows

- [Standalone Standing Test](standalone-standing)
- [BVH Playback on Unitree G1](bvh-sim2real)
- [From Teleoperation Data to Imitation Learning / VLA Deployment](high-level-policy-sim2real)
