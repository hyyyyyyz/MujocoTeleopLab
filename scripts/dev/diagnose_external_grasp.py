#!/usr/bin/env python3
"""Inspect static Bodex hand/object geometry for one generated scene.

This diagnostic never changes object state beyond the requested MuJoCo reset
and never runs a controller.  It applies each named grasp phase to the robot
actuator qpos, forwards MuJoCo, and prints all finger/object contacts and the
closest finger geom distances.  It is intended to distinguish a bad grasp
frame from a collision-monitoring bug before running the expensive CuRobo
trajectory generator.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teleopit.scenes.grasp_assets import load_simple_bodex
from teleopit.scenes.runtime import SceneTeleopRuntime
from teleopit.scenes.vla_datagen import KinematicObjectAttachment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-xml", type=Path, required=True)
    parser.add_argument("--grasp-asset", type=Path, required=True)
    parser.add_argument("--object-name", default="toy_rhinocero")
    args = parser.parse_args()

    runtime = SceneTeleopRuntime(scene_xml=args.scene_xml)
    record = load_simple_bodex(args.grasp_asset)
    attachment = KinematicObjectAttachment(runtime, args.object_name)
    model, data, mujoco = runtime.model, runtime.data, runtime._mujoco
    object_joint = f"robosuite_{args.object_name}_free"
    object_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, object_joint)
    object_body = int(model.jnt_bodyid[object_jid])
    object_geoms = {
        int(g): mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or str(g)
        for g in range(model.ngeom)
        if int(model.geom_bodyid[g]) == object_body
    }
    finger_geoms = set(attachment._finger_geom_ids)
    finger_bodies = set(attachment._finger_body_ids)
    print(f"object body={object_body} geoms={object_geoms}")
    print("finger bodies:", [(b, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)) for b in sorted(finger_bodies)])
    print("finger geoms:", [(g, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)) for g in sorted(finger_geoms)])

    for phase in ("pregrasp", "grasp", "squeeze", "lift"):
        runtime.reset()
        target = getattr(record, phase)
        for name, value in target.items():
            if name in runtime._qpos_adr:
                data.qpos[runtime._qpos_adr[name]] = float(value)
        mujoco.mj_forward(model, data)
        print(f"\n[{phase}]")
        print("object pose", np.asarray(data.qpos[int(model.jnt_qposadr[object_jid]):int(model.jnt_qposadr[object_jid])+7]))
        for body_name in ("right_wrist_yaw_link", "right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link", "right_hand_ring_1_link", "right_hand_pinky_1_link"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if bid >= 0:
                print("body", body_name, np.asarray(data.xpos[bid]))
        pairs = []
        for i in range(int(data.ncon)):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in object_geoms and g2 in finger_geoms) or (g2 in object_geoms and g1 in finger_geoms):
                pairs.append((object_geoms.get(g1, g1), mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2 if g1 in object_geoms else g1), float(c.dist), np.asarray(c.pos).copy()))
        print("contacts", pairs)
        closest = []
        for fg in sorted(finger_geoms):
            for og in sorted(object_geoms):
                dist = float(mujoco.mj_geomDistance(model, data, fg, og, 1.0, None))
                closest.append((dist, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, fg), object_geoms[og]))
        print("closest", sorted(closest)[:12])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
