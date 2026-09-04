---
sidebar_position: 4
---

# Dexterous grasp sources

The optional grasp adapter can consume SIMPLE/Bodex phase caches without
copying third-party data into this repository:

```python
from teleopit.scenes.grasp_assets import load_simple_bodex

record = load_simple_bodex(
    "${SIMPLE_ROOT}/data/assets/graspnet/dex_grasp/dex3/<object>/<object>__0.npy",
    source_joint_names=SOURCE_JOINT_NAMES,
    target_joint_names=TARGET_JOINT_NAMES,
)
```

The loader validates `pregrasp`, `grasp`, `squeeze` and `lift` states, finite
values, dimensions, duplicate names and optional joint permutations. It only
reads a local file supplied by the user; it does not download or redistribute
SIMPLE, Bodex, GraspNet or DexGraspNet data.

## What can be used

- CuRobo is Apache-2.0 and is already the motion-planning backend.
- robosuite is MIT and is suitable for MuJoCo object/scene assets.
- SIMPLE's planner structure can be reimplemented: candidate selection,
  pregrasp, grasp, squeeze, lift, lower, release and retreat.
- SIMPLE/Bodex caches can be consumed locally after the user verifies the
  upstream data terms.

## What must not be copied blindly

- GraspNet baseline uses an academic/non-commercial research-only license.
- DexGraspNet's repository does not expose a clear top-level redistributable
  data license; treat its data as external until the authors confirm terms.
- ShadowHand grasps from DexGraspNet do not directly fit G1/Dex3. They require
  a hand-model retargeting and contact/physics validation step.

The manifest at `assets/manifests/grasp_sources.yaml` records these decisions.
Only metadata and adapters belong in Git; large grasp files remain local or in
an approved artifact store.

When using the CuRobo scene generator, pass a local cache explicitly:

```bash
.venv_scene/bin/python scripts/run/generate_vla_scene_data.py \
  --scene cube --planner curobo \
  --grasp-asset /path/to/SIMPLE/data/assets/graspnet/dex_grasp/dex3_right/<uid>/<uid>__0.npy
```

The imported phase hand postures replace the generic close/squeeze posture;
the object is still simulated as a free MuJoCo body and contact validation is
unchanged. The asset must correspond to the current object geometry and hand
model; a grasp generated for another shape is not expected to transfer safely.
