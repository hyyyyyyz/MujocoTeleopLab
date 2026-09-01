---
sidebar_position: 3
---

# XRoboToolkit Pico Teleoperation in MuJoCo

This is the Pico-to-MuJoCo G1 workflow. It uses the XRoboToolkit PC Service
and headset app for tracking, then uses Teleopit's existing G1 retargeting and
MuJoCo controller. It does **not** connect to a physical robot.

For table-top object interaction with Dex3 hands, joystick locomotion, and the
released decoupled whole-body controller, use the separate
[scene teleoperation workflow](scene-teleop.md). The two runtimes must not run
at the same time.

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

Leave this terminal running. The SDK uses local TCP 60061; the headset service
also accepts its LAN connection on TCP 63901.

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
5. Enable **Send**.

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
PICO_VIDEO_HOST=<Pico IPv4> \
.venv/bin/python scripts/run/run_sim.py \
  --config-dir "$PWD/local" \
  --config-name pico4_xrobotoolkit_mujoco \
  controller.policy_path=ckpt/track_g1.onnx
```

The simulator starts in `STANDING`. Its controller interaction follows the
SIMPLE workflow. After the robot has stood stably for 4--6 seconds and
**Controller** is enabled in the XRoboToolkit headset app, use the Pico
controller directly:

| SIMPLE controller input | Teleopit action |
| --- | --- |
| Hold left **Menu** (three-line button) + squeeze the right index trigger | Start whole-body `MOCAP`. This is an edge-triggered activation: release both after the short press. |
| Squeeze both side grips (left + right middle fingers) | Safely end the current mocap session and return to `STANDING`, with a fresh reference alignment. |
| `B` | Reserved for XRoboToolkit Remote Vision stereo/single-view switching; Teleopit does not bind it. |

The activation chord is ignored until the PC Service has received at least one
Full-body frame. Teleopit tracks body motion directly: it does not reproduce
SIMPLE's joystick locomotion/turn commands, `X` squat, or trigger-driven hand
grippers because this G1 sim2sim workflow has no decoupled-WBC navigation or
hand actuator stack. The computer keyboard remains an optional fallback:
`Y` starts mocap, `A` pauses/resumes, `B` toggles arms-only mode, `X` returns
to standing, and `Q` quits.

## MuJoCo First-person Video

The `pico4_xrobotoolkit_mujoco` local preset streams the simulated G1
`d435i_rgb` camera to the headset. Current XRoboToolkit headset builds expose
the media listener directly on TCP 12345. Set `PICO_VIDEO_HOST` to the
headset's current LAN IPv4 address when launching; the preset reads that
variable and deliberately has no hard-coded address because Pico DHCP leases
can change. The PC continuously reconnects to that listener and sends a
2560x720 H.264 stereo stream made from the simulated mono view.

If UFW is enabled, allow only the local LAN to reach the XRoboToolkit service
ports (replace the subnet with your own LAN). Direct video is an outbound
connection from the PC to the headset; no inbound TCP 13579 rule is required.

```bash
sudo ufw allow from 10.0.90.0/23 to any port 63901 proto tcp
```

On the headset, open **Remote Vision**, select **ZEDMINI** (the source label
selects its H.264 receiver), and press **Listen**. If the Pico obtains a new
DHCP address, change `PICO_VIDEO_HOST` in the launch command and restart the
simulator; do not reuse the previous session's address. The older headset flow
that asks for a PC address is also supported through TCP 13579, but is not the
default for this local preset.

For the three standard debug windows (mocap, retarget, and sim2sim), use the
base `--config-name xrobotoolkit_sim`. Video is disabled in that base profile.
If you explicitly enable `input.video.enabled=true` without setting
`input.video.direct_host`, Teleopit falls back to the legacy `OPEN_CAMERA`
negotiation on TCP 13579. Use the local `pico4_xrobotoolkit_mujoco` preset above
when you need the current direct TCP 12345 Remote Vision listener. For a
headless smoke test add `viewers=none +num_steps=8`.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| PC Service starts but no frames | Headset UI says `WORKING`; Full-body and Send are enabled; both ankle trackers are paired and calibrated. |
| SDK import fails | Rerun `bash scripts/setup/setup_xrobotoolkit.sh` from the repository root. |
| Service cannot bind port 60061 | Stop the existing `RoboticsServiceProcess`; only one PC Service instance may run. |
| Simulation waits for data | Run `scripts/dev/test_xrobotoolkit.py` first; do not diagnose from MuJoCo. |
| Remote Vision shows an empty white panel | Verify that the simulation was launched with `pico4_xrobotoolkit_mujoco`, select `ZEDMINI`, press **Listen**, and set `PICO_VIDEO_HOST` to the headset's current LAN IPv4 address. |
