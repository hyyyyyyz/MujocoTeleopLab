---
sidebar_position: 2
---

# Sourcing additional simulation assets

When adding tabletop objects or task scenes, prefer a source repository with
an explicit asset license and a stable revision. Record the source in
`assets/manifests/` before adapting anything. The repository should contain
only the manifest and adapter code; downloaded meshes, textures and large
archives stay in ignored paths.

## Candidate sources

| Source | What it provides | License evidence | Fit for this project |
| --- | --- | --- | --- |
| [robosuite](https://github.com/ARISE-Initiative/robosuite) | MuJoCo arenas, primitive generators, bottles, cans, bread, cereal, milk, lemons, plates, doors | Repository `LICENSE` is MIT; it separately notes included MuJoCo code is Apache-2.0. Verify each asset directory before redistribution. | Best first source for small, self-contained object adapters. Its XML is designed as composable bodies, but it is not a drop-in 43-DOF G1 scene. |
| [RoboCasa](https://github.com/robocasa/robocasa) | Kitchen fixtures, 3,200+ objects, 2,500+ scenes and everyday-task definitions | Upstream README states code is MIT and assets/datasets are CC BY 4.0. | Good for later kitchen tasks; download is about 10 GB and the assets need a conversion layer and attribution manifest. |
| [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | High-quality robot models and example scenes | Each model directory has its own `LICENSE`; the README shows different models under BSD-3-Clause, Apache-2.0, MIT, etc. | Useful for robot/model references, not the first choice for tabletop object packs. Never assume the top-level license covers a model directory. |
| [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) | MuJoCo locomotion/manipulation environment code and integrations | Upstream README states Apache-2.0 for repository content and CC0 for the Polyhaven rough-terrain texture. | Useful for environment patterns and randomization; not a ready-made G1/Dex3 object library. |
| [ManiSkill](https://github.com/haosulab/ManiSkill) | Large manipulation benchmark and rigid-body environments | Upstream README states environments use permissive licenses, while assets are CC BY-NC 4.0. | Good research reference, but NC terms are unsuitable for a generally redistributable asset bundle. Keep as reference unless a specific asset's terms are cleared. |

This table records upstream statements and integration scope; it is not a
legal determination for every nested file. A release still needs a per-file
license/notice review.

## Proposed import path

The first concrete imports are robosuite v1.5.2 bottle, can and lemon. Run
`.venv/bin/python scripts/setup/download_scene_object.py`; the script pins the
upstream commit, verifies SHA256 for every file, downloads the upstream MIT
notice, and emits a small body fragment under ignored
`assets/objects/robosuite/<object>/v1/`. It does not install robosuite itself.
The generated fragment can be compiled independently with
`.venv_scene/bin/python scripts/dev/validate_scene_object.py`.

To compose an object into the existing G1/Dex3 tabletop scene, run
`.venv_scene/bin/python scripts/setup/build_scene_with_object.py <object>`
(`bottle`, `can` or `lemon`). It writes the ignored
`outputs/scenes/robosuite_<object>_43dof.xml`, which can be passed to the
existing launcher with `--scene-xml`.

For the first import, use only a small primitive or one clearly licensed
robosuite object. Convert it into a Teleopit object fragment with:

1. a visual mesh (optional) and a simplified primitive collision geom;
2. explicit mass, friction, solver settings and scale in metres;
3. stable body, geom and free-joint names;
4. a manifest entry with upstream URL, commit/tag, SPDX status and SHA256;
5. a MuJoCo smoke test that checks XML loading, contact and reset behavior.

The existing scene runtime can then compose that fragment into a scene XML
while preserving its required 43 actuator names. This keeps the asset import
independent of PICO/XRoboToolkit input and avoids copying an entire simulator
framework into Teleopit.

## Recommended order of work

1. Import one robosuite primitive/object and validate a cube-like pick task.
2. Add a source-neutral object adapter and object-specific manifest entries.
3. Add task metadata and success predicates (`push`, `lift`, `place`).
4. Evaluate RoboCasa kitchen assets only in a separate optional download group.
