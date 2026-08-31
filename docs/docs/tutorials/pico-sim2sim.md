---
sidebar_position: 2
---

# VR Teleoperation in Simulation

Use Pico tracking to control a simulated G1 before connecting a physical robot.
Do not skip this step: it lets you fix headset, network and body-tracking
problems without putting hardware at risk.

## Supported Headsets

- Pico 4
- Pico 4 Ultra
- Pico 4 Ultra Enterprise
- Pico 4 Pro

All headsets must have full-body tracking enabled and run a Pico system version
that supports the current body-tracking interface.

## Before You Start

You need:

- the headset and the computer running Teleopit on the same network,
- the `pico4` install profile and `robots gmr ckpt bvh` assets, and
- a working result from
  [Run a Motion Controller in Simulation](offline-sim2sim).

## 1. Prepare the Headset

1. Download the headset APK from
   [pico-bridge Releases](https://github.com/BotRunner64/pico-bridge/releases).
2. Install it:

   ```bash
   adb install pico-bridge.apk
   ```

3. Open the pico-bridge app in the headset.
4. Turn on full-body tracking.

Teleopit uses pico-bridge 0.2.1. The receiver runs inside Teleopit on your
computer; there is no second relay program to start.

## 2. Check That the Computer Receives Pico Data

This diagnostic prints body-frame and connection information without starting
the robot controller:

```bash
python scripts/dev/test_pico_bridge.py --no-video
```

Move slightly and confirm that new valid frames continue to arrive. Press
`Ctrl+C` to stop the diagnostic.

If discovery chooses the wrong network address, pass the address that the
headset can reach:

```bash
python scripts/dev/test_pico_bridge.py \
    --no-video \
    --bridge-advertise-ip=192.168.1.20
```

## 3. Start the Simulation

```bash
python scripts/run/run_sim.py \
    --config-name pico4_sim \
    controller.policy_path=ckpt/track_g1.onnx
```

The robot intentionally starts in `STANDING`; live body tracking does not take
control until you ask for it.

## 4. Use the Simulation State Machine

![Pico simulation control state machine](/img/diagrams/pico-sim-state-machine.svg)

Labels beginning with **Keyboard** refer to the computer keyboard. Labels
beginning with **Pico controller** refer to the VR controllers. The Unitree G1
remote is not used in simulation.

Stand in a comfortable neutral pose and wait for stable tracking before using
**Keyboard** `Y` to enter `MOCAP`. Move slowly at first. Use **Keyboard** `X` to
end the VR session and return to `STANDING`; **Keyboard** `Q` quits the
simulation from any state.

`MOCAP` follows the whole body. `ARMS` keeps the body, waist and legs in the
standing pose while both arms continue to follow. `PAUSED` holds the current
reference and returns to the previous `MOCAP` or `ARMS` state when resumed.

Each new `STANDING -> MOCAP` session recalibrates the live root pose. You may
turn to a new heading while standing, then enter `MOCAP` again.

:::tip Pausing is not the same as stopping VR control
Keyboard or Pico controller `A` freezes and resumes the current mocap pose.
Use Keyboard `X` when you want to end the session and return to `STANDING`.
:::

## Choose the Viewer Layout

Pico simulation opens the mocap, retarget and physics views by default. Use a
smaller layout when you no longer need all three:

```bash
# Physics result only
python scripts/run/run_sim.py \
    --config-name pico4_sim \
    controller.policy_path=ckpt/track_g1.onnx \
    viewers=sim2sim

# Headless
python scripts/run/run_sim.py \
    --config-name pico4_sim \
    controller.policy_path=ckpt/track_g1.onnx \
    viewers=none
```

## Optional Headset Video

To send the simulated `d435i_rgb` camera view back to the headset:

```bash
python scripts/run/run_sim.py \
    --config-name pico4_sim \
    controller.policy_path=ckpt/track_g1.onnx \
    input.video.enabled=true
```

Use `input.video.source=test-pattern` to check only the video connection.
Video failure disables the preview but does not stop tracking or control.

## Network Overrides

Most setups only need automatic discovery. Use these overrides when the
diagnostic shows a network problem:

```bash
# Advertise a specific host address to the headset
input.bridge_advertise_ip=192.168.1.20

# Disable discovery and bind explicitly
input.bridge_discovery=false
input.bridge_host=0.0.0.0
input.bridge_port=63901

# Wait longer for the first body frame
input.pico4_timeout=30
```

## Common Problems

| Problem | Solution |
|---------|----------|
| No body frames arrive | Upgrade the Pico headset to the latest available system version, restart it, enable full-body tracking again, and rerun `scripts/dev/test_pico_bridge.py --no-video` |

Once this workflow is reliable, continue with
[VR Teleoperation on Unitree G1](pico-sim2real).
