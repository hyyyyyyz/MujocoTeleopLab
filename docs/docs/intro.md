---
sidebar_position: 1
slug: /
---

# MujocoTeleopLab

> **Looking for Chinese docs?** Use the Chinese-language documentation build
> for this project.

MujocoTeleopLab is a simulation-first research workspace for **PICO/
XRoboToolkit teleoperation, tabletop manipulation, and VLA data generation**.
Its main target is a 43-DOF Unitree G1/Dex3 MuJoCo scene: an operator can
control the robot and interact with objects, while automatic CuRobo planning
can generate collision-aware trajectories for dataset collection.

The repository also includes the compatible `teleopit` motion-retargeting
runtime for standard G1 whole-body experiments. That package name is retained
for API/configuration compatibility; the project itself is maintained as
MujocoTeleopLab and is not an official Teleopit release.

The same motion controller runs in MuJoCo first, so you can check tracking and
controls before connecting a physical robot.

## Start Here

If this is your first time using MujocoTeleopLab:

1. [Install MujocoTeleopLab](getting-started/installation) for the job you want to do
   and complete the check at the end of that page.
2. Continue with one of the seven guides below.

| I want to... | Follow this guide |
|--------------|-------------------|
| Check a motion controller in MuJoCo | [Run a Motion Controller in Simulation](tutorials/offline-sim2sim) |
| Try Pico VR control without a real robot | [VR Teleoperation in Simulation](tutorials/pico-sim2sim) |
| Use XRoboToolkit Pico tracking with the standard G1 sim2sim controller | [XRoboToolkit Pico Teleoperation in MuJoCo](tutorials/xrobotoolkit-sim2sim) |
| Manipulate table-top objects with Pico, Dex3, and the scene WBC in MuJoCo | [XRoboToolkit Table-top Scene Teleoperation](tutorials/scene-teleop) |
| Control a physical G1 with Pico VR | [VR Teleoperation on Unitree G1](tutorials/pico-sim2real) |
| Train and export my own controller | [Train a Motion Controller](tutorials/training) |

:::warning Before using a real robot
Make the Pico workflow work in simulation first. Keep the Unitree remote in
hand during hardware operation; `L1+R1` is the emergency path to `DAMPING`.
:::

## Looking for Implementation Details?

The user guides intentionally keep internals out of the main flow. See
[Architecture](reference/architecture) for the runtime pipeline and technical
specifications, [Assets](reference/resources/assets) for every
downloaded file, or
[Configuration](reference/configuration/overview) for Hydra options.
