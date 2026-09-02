---
sidebar_position: 4
---

# XRoboToolkit Table-top Scene Teleoperation

Use this workflow for physics-based MuJoCo table-top manipulation scenes: a
43-DOF Unitree G1 with two Dex3 hands, a balance/walk policy, arm IK, and
collidable objects.
It is separate from the 29-DOF whole-body tracking workflow described in the
[XRoboToolkit MuJoCo tutorial](xrobotoolkit-sim2sim.md). Do not launch both
workflows at the same time because they would compete for XRoboToolkit.
This workflow is simulation-only; it sends no commands to a physical G1.

## What It Includes

The initial scenes are released `decoupled_wbc` G1 assets:

| Scene | Contents |
| --- | --- |
| `cube` | Table and a small free cube for the first pick-and-place test. |
| `bottle` | Table and a free bottle. |
| `box` | Table and a larger free box. |

The scene runtime runs MuJoCo physics at 200 Hz and the released Balance/Walk
whole-body policy at 50 Hz. The 43 commanded joints are 29 body joints plus
seven Dex3 joints per hand. The robot and objects reset together, so a failed
grasp never requires restarting the application.

## One-time Setup

First set up the usual XRoboToolkit PC Service and its Python 3.12 binding:

```bash
bash scripts/setup/setup_xrobotoolkit.sh
```

Then create the isolated Python 3.10 environment used by the upstream WBC,
Pinocchio IK, and the 43-DOF MuJoCo assets:

```bash
bash scripts/setup/setup_scene_teleop.sh
```

The setup script clones `third_party/decoupled_wbc` if it is absent. That
directory and `.venv_scene` are local dependencies and intentionally ignored
by Git. `.venv_scene` is reserved for the Python 3.10 WBC/IK/MuJoCo stack; it
does not install the main `teleopit` package. The scene launchers import the
repository checkout directly, while the regular `.venv` remains the home for
the Python 3.12 XRoboToolkit bridge and Teleopit dependencies.

## Pico Setup

1. Start the local PC Service:

   ```bash
   .tools/xrobotoolkit/service/opt/apps/roboticsservice/runService.sh
   ```

2. In **XRoboToolkit** on Pico, connect to the PC until it reports
   **WORKING**.
3. Enable **Head**, **Controller**, and **Send**.
4. In **Remote Vision**, select **ZEDMINI** and press **Listen**. Note the
   Pico's current LAN IPv4 address.

Full-body tracking and ankle trackers are not required for this scene
workflow. It uses the HMD and both controllers only.

## Start the Cube Scene

From the repository root, substitute the Pico address if it changes:

```bash
PICO_VIDEO_HOST=<Pico IPv4> bash scripts/run/start_scene_teleop.sh --scene cube
```

`PICO_VIDEO_HOST` is required for this launcher; it deliberately has no stale
address fallback.  For a local MuJoCo viewer or headless check without Pico
Remote Vision, opt out explicitly:

```bash
SCENE_NO_VIDEO=1 bash scripts/run/start_scene_teleop.sh --scene cube
```

The launcher also refuses a second scene/bridge instance because both would
compete for the same localhost XR packet port. Stop the existing launcher
before starting a new one.

This single command starts two local processes:

1. `.venv` reads XRoboToolkit controller/HMD data and sends it only to a
   localhost UDP bridge.
2. `.venv_scene` runs the 43-DOF WBC, MuJoCo scene, local viewer, and Remote
   Vision sender.

The direct Remote Vision connection is outbound from the PC to Pico TCP 12345.
The scene uses SIMPLE's ZEDMINI media profile: the renderer produces one
1280x720 eye, duplicates it side-by-side, and sends a 2560x720 H.264 stream at
60 FPS.  If you need to reduce load for a diagnostic run, pass
`--video-fps 30`, but keep the default 60 FPS for normal headset use.
In parallel, the scene sender also runs a best-effort legacy operator-control
listener on TCP 13579 so older headset builds that negotiate via `OPEN_CAMERA`
keep working; a legacy request replaces the direct target for the rest of that
Listen session. Binding TCP 13579 may fail (for example when another
XRoboToolkit service already owns it) without affecting the direct path.
If a firewall is enabled, the PC Service still needs the same LAN rules as the
standard XRoboToolkit tutorial.

Once the stream is live, turn your head to look around the simulated scene.
The scene runtime applies the HMD orientation from **Head** to its
`scene_head_camera`; the first valid HMD orientation in a process is the
neutral reference, and only the subsequent yaw/pitch/roll change is applied.
HMD translation is intentionally ignored, so the camera stays mounted to the
MuJoCo torso. Restart the scene process to establish a new neutral orientation.

For a renderer-free smoke test, run:

```bash
.venv_scene/bin/python scripts/run/run_scene_teleop.py \
  --scene cube --headless --seconds 10 --no-video
```

For a stronger, deterministic acceptance check, run the controller-pose,
grasp, contact, and object-motion smoke test:

```bash
.venv_scene/bin/python scripts/dev/smoke_scene_teleop.py
```

It must report a standing root height, at least one cube contact body, and a
cube displacement of at least 1 cm. It is intended for validating this first
cube scene before you use Pico, or after updating the scene-control stack.

### Run while Pico is charging

Pico is not required for the local physics checks. Skip the Python 3.12
XRoboToolkit bridge and the outbound video worker explicitly, then run the
isolated MuJoCo process headlessly:

```bash
SCENE_NO_BRIDGE=1 SCENE_NO_VIDEO=1 \
bash scripts/run/start_scene_teleop.sh \
  --scene cube --headless --seconds 10 --no-realtime
```

This exercises scene loading, the 43-DOF actuator map, WBC initialization,
PD stepping, and clean shutdown. It waits safely for no XR packets, so it does
not require a powered headset or a running PC Service. The deterministic
`smoke_scene_teleop.py` command above additionally drives a synthetic
controller sequence through arm IK and cube contact without opening any
network socket.

## Pico Controller Map

The mapping intentionally follows SIMPLE's Pico decoupled-WBC interaction.
All controls are edge-triggered where noted; release a chord after pressing it.

| Pico input | Scene action |
| --- | --- |
| Left **Menu** + left index trigger | Toggle walking-input lock. Balance remains active while locked, so the robot remains standing. |
| Left **Menu** + right index trigger | Toggle arm/hand teleoperation and calibrate the arm reference on entry. |
| Left stick | Walk forward/backward and strafe. |
| Right stick left/right | Turn the walking reference. |
| Left/right index trigger | Index-pinch gesture for that hand. |
| Index trigger + side grip | Power-grip gesture for that hand. |
| Hold **X** | Lower the commanded base height. |
| Hold **Y** | Raise the commanded base height to standing height. |
| Both side grips | Reset robot, objects, WBC state, and arm calibration. |
| **B** | Reserved for XRoboToolkit Remote Vision single/stereo switching. |

After the robot stands steadily, use Menu + left index once if you want to
walk, then use Menu + right index once. Keep the two controllers in a
comfortable neutral pose when starting: it becomes the arm reference. Move
them gradually toward the object, then use the hand gesture to close around
it. The local MuJoCo viewer remains open for debugging, while Remote Vision
provides the headset view.

## Select or Add Scenes

Use one of the bundled scene names:

```bash
bash scripts/run/start_scene_teleop.sh --scene bottle
bash scripts/run/start_scene_teleop.sh --scene box
```

For a new task, pass a custom XML that exposes exactly the released 43 joint
actuator names (29 G1 body + 14 Dex3 hand joints):

```bash
bash scripts/run/start_scene_teleop.sh --scene-xml /absolute/path/my_scene.xml
```

The runtime inserts its `scene_head_camera` at launch, so the source XML stays
read-only. Keep free objects dynamic and collidable; both grips provide the
normal episode reset action.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `No XR packet` or no controller response | PC Service and Pico must be `WORKING`; enable Head, Controller, and Send. Full-body is not needed. |
| Terminal says `waiting for bridge packets` | The scene process cannot see the local Python 3.12 bridge. Restart `start_scene_teleop.sh`; when working, it reports `Scene XR input: ... Hz`. |
| WBC import or asset error | Re-run `bash scripts/setup/setup_scene_teleop.sh`; do not install the WBC stack into the normal `.venv`. |
| Remote Vision is white/blank | The terminal reports `Scene Remote Vision waiting for Pico Listen` until the listener opens. In Pico choose ZEDMINI and Listen, then verify it changes to `connected`/`live`; set `PICO_VIDEO_HOST` to the headset's current IP. |
| PC Service connection conflict | Close the old 29-DOF `run_sim.py` workflow before starting the scene workflow. |
| Robot/object state is unusable | Hold both Pico side grips to reset the whole scene. |
