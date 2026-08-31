---
sidebar_position: 3
---

# XRoboToolkit Pico Teleoperation in MuJoCo

This is the Pico-to-MuJoCo G1 workflow. It uses the XRoboToolkit PC Service
and headset app for tracking, then uses Teleopit's existing G1 retargeting and
MuJoCo controller. It does **not** connect to a physical robot.

## One-time Local Setup

The configured workspace already contains the downloaded PC Service and Pico
APK under the ignored `.tools/xrobotoolkit/` directory. To recreate that setup
on this computer, run:

```bash
bash scripts/setup/setup_xrobotoolkit.sh
```

The script downloads the official Ubuntu 24.04 x86_64 PC Service, unpacks it
locally (no system installation), compiles the official `xrobotoolkit_sdk`
Python binding, and installs it into `.venv`. It requires `c++`, `cmake`, and
`dpkg-deb`.

## Start the PC Service

In terminal 1, from the project root:

```bash
.tools/xrobotoolkit/service/opt/apps/roboticsservice/runService.sh
```

Leave this terminal running. A successful service listens locally on TCP 60061.

## Configure the Pico Headset

Install `.tools/xrobotoolkit/XRoboToolkit-PICO-1.1.1.apk` using ADB, or download
the same release from [XRoboToolkit Unity Client](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/tag/v1.1.1).

```bash
.tools/platform-tools/adb install -r .tools/xrobotoolkit/XRoboToolkit-PICO-1.1.1.apk
```

The headset and PC must be on the same LAN. In the headset:

1. Open **XRoboToolkit** from Unknown Sources.
2. Select the discovered PC address, or enter the PC's LAN IPv4 address, then select **Connect**. Its status must become **WORKING**.
3. Under **Tracking**, enable **Head** and **Controller**.
4. Under **PICO Motion Tracker**, select **Full-body**. Pair and calibrate the two ankle trackers in the Pico system Motion Tracker app first.
5. Enable **Send**. Do not enable Remote Vision for this first tracking test.

## Verify Tracking Before Simulation

In terminal 2:

```bash
.venv/bin/python scripts/dev/test_xrobotoolkit.py --seconds 20
```

Move slightly. The command must report increasing sequence numbers and a stable
frame rate before proceeding.

## Run MuJoCo G1 Teleoperation

In the same terminal:

```bash
.venv/bin/python scripts/run/run_sim.py \
  --config-dir "$PWD/local" \
  --config-name pico4_xrobotoolkit_mujoco \
  controller.policy_path=ckpt/track_g1.onnx
```

The simulator starts in `STANDING`. Use the computer keyboard: `Y` begins
whole-body mocap, `A` pauses/resumes, `B` toggles arms-only mode, `X` returns
to standing, and `Q` quits. The Pico controller `A` and `B` provide the same
pause and arms toggles when the headset app is sending controller data.

For all three debug windows use `--config-name xrobotoolkit_sim`; for a headless
smoke test add `viewers=none +num_steps=8`.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| PC Service starts but no frames | Headset UI says `WORKING`; Full-body and Send are enabled; both ankle trackers are paired and calibrated. |
| SDK import fails | Rerun `bash scripts/setup/setup_xrobotoolkit.sh` from the repository root. |
| Service cannot bind port 60061 | Stop the existing `RoboticsServiceProcess`; only one PC Service instance may run. |
| Simulation waits for data | Run `scripts/dev/test_xrobotoolkit.py` first; do not diagnose from MuJoCo. |
