---
sidebar_position: 1
slug: /
---

# Teleopit

> **Looking for Chinese docs?** [中文文档点此进入](https://BotRunner64.github.io/Teleopit/zh-Hans/)

Teleopit is a **full-embodiment humanoid teleoperation system for the Unitree
G1**. With a supported Pico headset, an operator can drive the robot's
whole-body motion in real time. In onboard deployments, optional LinkerHand
hands reproduce hand gestures, and an optional OpenNeck gimbal turns head
motion into active camera control.

The same motion controller runs in MuJoCo first, so you can check tracking and
controls before connecting a physical robot.

## Start Here

If this is your first time using Teleopit:

1. [Install Teleopit](getting-started/installation) for the job you want to do
   and complete the check at the end of that page.
2. Continue with one of the four guides below.

| I want to... | Follow this guide |
|--------------|-------------------|
| Check a motion controller in MuJoCo | [Run a Motion Controller in Simulation](tutorials/offline-sim2sim) |
| Try Pico VR control without a real robot | [VR Teleoperation in Simulation](tutorials/pico-sim2sim) |
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
