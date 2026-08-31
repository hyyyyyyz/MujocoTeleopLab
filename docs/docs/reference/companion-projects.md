---
sidebar_position: 4
---

# Companion Projects

Teleopit integrates four focused components for robot communication, hand
retargeting, active vision, and PICO transport. They are kept outside the
`teleopit` Python package so each project can own its hardware protocol and
public API.

| Component | Source | Function | Use in Teleopit |
|-----------|--------|----------|-----------------|
| G1 Bridge SDK | [Teleopit source tree](https://github.com/BotRunner64/Teleopit/tree/master/third_party/g1_bridge_sdk) | Native C++/pybind11 bridge over Unitree SDK2 and Cyclone DDS | Real-time G1 state, remote input, mode selection, and 200 Hz low-level commands |
| somehand | [GitHub](https://github.com/BotRunner64/somehand) | Dexterous-hand retargeting library | Maps live Pico hand landmarks to LinkerHand L6/O6 targets |
| OpenNeck | [GitHub](https://github.com/BotRunner64/OpenNeck) | Calibrated two-axis neck driver | Converts physical yaw/pitch degrees to safe servo commands |
| PICO Bridge | [GitHub](https://github.com/BotRunner64/pico-bridge) | Headset app and Python receiver for PICO tracking and video | Supplies body, controller, hand, and HMD frames and optionally returns RGB video |

## G1 Bridge SDK

G1 Bridge SDK is maintained directly in Teleopit under
`third_party/g1_bridge_sdk`; it is not a separate repository. Its setup script
downloads [Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2), then
builds and installs the local pybind11 extension:

```bash
bash scripts/setup/setup_g1_bridge.sh
```

All DDS publish/subscribe work runs on native C++ threads. Teleopit's
`UnitreeG1` adapter reads joint state, base orientation, angular velocity, and
wireless-remote input through the bridge, and sends 29-joint position targets
with per-joint PD gains. This is the hardware boundary used by sim2real
teleoperation, the standalone standing check, and host-policy deployment.

## somehand

somehand provides configurable human-to-robot hand retargeting. Teleopit pins
the compatible source as the `third_party/somehand` Git submodule and uses its
0.3.0 public `somehand.api` surface.

In `hands.mode=vr_hand_pose`, Teleopit converts PICO's 26-joint hand state to
21 landmarks, calls somehand for continuous retargeting, and sends the result
to LinkerHand L6 or O6. Teleopit owns the live Pico receiver and the landmark
conversion; it does not start somehand's standalone Pico input path.

Install the dexterous-hand dependencies with:

```bash
git submodule update --init --recursive
pip install -e third_party/linkerhand-python-sdk
pip install -e third_party/somehand
```

## OpenNeck

OpenNeck owns serial communication, degree-to-servo-step conversion, and
calibrated mechanical limits for the two-axis active-vision gimbal. Teleopit
supports the OpenNeck 0.2.0 physical-angle API and calls `move_deg()`; removed
normalized control fields are not compatible.

For Pico teleoperation, Teleopit computes HMD rotation relative to the
same-frame `Body.Spine3` orientation, applies the configured dead zone and
pitch gain, and sends yaw/pitch degrees from a non-critical neck worker. Host
policy deployment sends the validated neck fields from its canonical action.

```bash
pip install -e '.[openneck]'
openneck calibrate
```

## PICO Bridge

PICO Bridge contains both the headset application and the importable Python PC
receiver. Teleopit supports release 0.2.1, installed by the `pico4` extra:

```bash
pip install -e '.[pico4]'
```

One in-process `PicoBridge` instance supplies full-body, controller, hand, and
independent HMD data to Teleopit. Whole-body retargeting, hand control, and
OpenNeck all reuse that receiver. When video is enabled, Teleopit can also push
MuJoCo or RealSense RGB frames back to the headset through
`push_video_frame()`.

Download the headset APK from the
[PICO Bridge releases](https://github.com/BotRunner64/pico-bridge/releases).
