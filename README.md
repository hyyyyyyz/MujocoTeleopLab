<h1 align="center">MujocoTeleopLab</h1>

<p align="center">
  Simulation-first research workspace for PICO/XRoboToolkit teleoperation and
  automatic VLA data generation.
  <br/>
  MuJoCo tabletop scenes, Unitree G1/Dex3 manipulation, reusable open-source
  assets, and collision-aware CuRobo planning.
</p>

<p align="center">
  <a href="https://github.com/hyyyyyyz/MujocoTeleopLab">Source code</a> &bull;
  <a href="docs/docs/intro.md">Documentation</a> &bull;
  <a href="docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/intro.md">中文文档</a>
</p>

---

## About MujocoTeleopLab

MujocoTeleopLab is an independent project for building and testing robot
manipulation tasks in simulation. Its main workflow is:

```text
PICO + XRoboToolkit → 43-DOF G1/Dex3 MuJoCo scene → task trajectories → VLA dataset
```

The repository keeps the `teleopit` Python package name for compatibility with
the existing runtime and configuration APIs, but this project is maintained
under the MujocoTeleopLab name. The scene teleoperation runtime, asset catalog,
object import tools, and VLA/CuRobo data pipeline are project-specific work.

Some low-level retargeting and robot-control components originate from the
open-source Teleopit ecosystem. They are retained under their applicable
licenses and are listed as dependencies or upstream sources; this repository is
not an official Teleopit release.

Third-party scene and object files are downloaded, not silently copied into
Git. Their source revision, license status, checksum, and task semantics are
recorded in [`assets/manifests/`](assets/manifests/README.md).

## Quick Start — Minimal Sim2Sim

**1. Install**

```bash
pip install -e .
```

**2. Download assets**

```bash
pip install modelscope
python scripts/setup/download_assets.py --only robots gmr ckpt bvh
```

The default Unitree G1 robot model is downloaded to
`assets/robots/unitree_g1/g1_29dof.xml`, with additional model variants in the
same directory. Training can select a task-compatible XML with `--robot_xml`;
the quick-start command below uses the default model and its matching
`ckpt/track_g1.onnx` policy. The neck-and-O6 variant uses
`g1_29dof_neck_o6.xml` with `ckpt/track_g1_neck_o6.onnx`.

**3. Run**

```bash
python scripts/run/run_sim.py \
    controller.policy_path=ckpt/track_g1.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh
```

You should see a MuJoCo viewer with the robot tracking the BVH motion.

To show the simulated D435i RGB camera view, add the explicit `camera` viewer:

```bash
python scripts/run/run_sim.py \
    controller.policy_path=ckpt/track_g1.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh \
    'viewers=[sim2sim,camera]'
```

For sim2real, viewers are disabled by default. Add `viewers=retarget` to show
the retargeted reference in an optional MuJoCo window.

For Pico-to-MuJoCo teleoperation through XRoboToolkit, including simulated
first-person Remote Vision and SIMPLE-style controller activation, see the
[XRoboToolkit MuJoCo tutorial](docs/docs/tutorials/xrobotoolkit-sim2sim.md).

For MuJoCo table-top manipulation with a 43-DOF G1/Dex3 model, collidable
objects, SIMPLE-style joystick walking, trigger hand gestures, and Pico Remote
Vision with HMD camera control, use the separate scene runtime:

```bash
bash scripts/setup/setup_scene_teleop.sh
PICO_VIDEO_HOST=<Pico IPv4> bash scripts/run/start_scene_teleop.sh --scene cube
```

This path is intentionally separate from the 29-DOF whole-body tracker. See
the [XRoboToolkit scene teleoperation tutorial](docs/docs/tutorials/scene-teleop.md)
for setup, controller mapping, and custom-scene requirements.

Simulation assets are cataloged separately from code. See
[`assets/manifests/`](assets/manifests/README.md) for the provenance,
license status, task semantics, and directory rules for scenes and objects.
The catalog does not redistribute third-party meshes or XML; validate it with
`./.venv/bin/python scripts/dev/validate_asset_manifests.py`.

## Documentation

The local documentation covers installation profiles, PICO/XRoboToolkit scene
teleoperation, asset governance, VLA data generation, configuration, and
architecture. Start with [`docs/docs/intro.md`](docs/docs/intro.md).

## Current capabilities

- PICO/XRoboToolkit scene teleoperation with SIMPLE-style activation, walking,
  camera control, hand gestures, and MuJoCo Remote Vision.
- Reusable `cube`, `bottle`, and `box` tasks plus governed robosuite object
  imports for `bottle`, `can`, and `lemon`.
- Automatic VLA scene-data generation with language/task metadata, images,
  state/action arrays, and a CuRobo collision-planning backend when CUDA is
  available.
- Compatible 29-DOF G1 motion-retargeting and ONNX sim2sim runtime for
  regression tests and baseline comparisons.

The project is under active development. Generated assets and datasets are
downloaded locally and are intentionally excluded from the source repository;
use the setup scripts and manifests to reproduce them on another machine.

## Development history

### 2026-09 — MujocoTeleopLab foundation

- Established the independent MujocoTeleopLab project and simulation-first
  workflow for PICO/XRoboToolkit tabletop manipulation.
- Added governed scene/object manifests, reproducible asset download/build
  scripts, and robosuite object integration.
- Added the VLA scene-data recorder and CuRobo planner interface, with clear
  CUDA requirements and a dependency-light smoke-test planner.
- Kept the compatible `teleopit` package and 29-DOF runtime as a baseline;
  future scene, task, and dataset work is developed under this repository's
  own roadmap.

## License

[Apache 2.0](LICENSE)
