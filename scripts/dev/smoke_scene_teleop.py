#!/usr/bin/env python3
"""Verify the G1/Dex3 cube scene can contact and move its table-top object.

Run this with ``.venv_scene/bin/python``.  It deliberately sends the same
controller-pose and trigger/grip sequence that the Pico bridge produces, but
does not need XRoboToolkit or a viewer.  It is a regression check for the full
scene control path: wrist pre-processing, arm/hand IK, WBC, MuJoCo PD,
collision, and object dynamics.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teleopit.scenes.controller import SimpleSceneController
from teleopit.scenes.runtime import SceneTeleopRuntime, scene_xml_path
from teleopit.scenes.xr_packet import SceneXRPacket


_BASE_PACKET: dict[str, object] = {
    "left_pose": [-0.2, 0.0, -0.3, 0.0, 0.0, 0.0, 1.0],
    "right_pose": [0.2, 0.0, -0.3, 0.0, 0.0, 0.0, 1.0],
    "head_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    "left_axis": [0.0, 0.0],
    "right_axis": [0.0, 0.0],
    "left_trigger": 0.0,
    "right_trigger": 0.0,
    "left_grip": 0.0,
    "right_grip": 0.0,
    "a": False,
    "b": False,
    "x": False,
    "y": False,
    "left_menu": False,
}


def _packet(sequence: int, timestamp_s: float, **overrides: object) -> SceneXRPacket:
    return SceneXRPacket.from_mapping(
        _BASE_PACKET | {"sequence": sequence, "timestamp_s": timestamp_s} | overrides
    )


def _object_contact_bodies(runtime: SceneTeleopRuntime) -> set[str]:
    """Return bodies touching the cube, excluding the cube itself and table."""
    mujoco = runtime._mujoco
    cube_body_id = mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_BODY, "cube_body")
    if cube_body_id < 0:
        raise ValueError("The cube scene is missing cube_body")
    contacts: set[str] = set()
    for contact_index in range(runtime.data.ncon):
        contact = runtime.data.contact[contact_index]
        first_body = int(runtime.model.geom_bodyid[contact.geom1])
        second_body = int(runtime.model.geom_bodyid[contact.geom2])
        if first_body == cube_body_id:
            other_body = second_body
        elif second_body == cube_body_id:
            other_body = first_body
        else:
            continue
        name = mujoco.mj_id2name(runtime.model, mujoco.mjtObj.mjOBJ_BODY, other_body) or "<unnamed>"
        if name != "table_body":
            contacts.add(name)
    return contacts


def _free_joint_position(runtime: SceneTeleopRuntime, joint_name: str) -> np.ndarray:
    """Return one named free joint's world position without assuming qpos order.

    The released cube currently follows the 43 robot coordinates in ``qpos``,
    but MuJoCo permits a scene to declare free objects before the robot (or to
    add another free object in between).  The smoke check is meant to validate
    object motion, not one particular XML declaration order, so resolve the
    joint address through the compiled model just as the runtime resolves the
    G1 floating root.
    """

    mujoco = runtime._mujoco
    joint_id = mujoco.mj_name2id(
        runtime.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
    )
    if joint_id < 0:
        raise ValueError(f"The cube scene is missing {joint_name}")
    if runtime.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"{joint_name} must be a MuJoCo free joint")
    qpos_address = int(runtime.model.jnt_qposadr[joint_id])
    position = np.asarray(
        runtime.data.qpos[qpos_address : qpos_address + 3], dtype=np.float64
    ).copy()
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError(
            f"{joint_name} produced an invalid world position {position!r}"
        )
    return position


def _advance(
    runtime: SceneTeleopRuntime,
    controller: SimpleSceneController,
    *,
    sequence: int,
    timestamp_s: float,
    steps: int,
    **overrides: object,
) -> tuple[int, float, set[str]]:
    contacts: set[str] = set()
    for _ in range(steps):
        command = controller.update(_packet(sequence, timestamp_s, **overrides))
        sequence += 1
        timestamp_s += 1.0 / 50.0
        runtime.update_policy(command, active=True)
        for _ in range(4):
            runtime._apply_pd()
            runtime._mujoco.mj_step(runtime.model, runtime.data)
            contacts.update(_object_contact_bodies(runtime))
    return sequence, timestamp_s, contacts


def main() -> int:
    runtime = SceneTeleopRuntime(scene_xml=scene_xml_path("cube"))
    controller = SimpleSceneController()

    start = controller.update(_packet(1, 1.0, left_menu=True, right_trigger=1.0))
    if not start.activation_toggled or not controller.active:
        raise AssertionError("The SIMPLE arm-activation chord was not recognized")
    runtime._start_teleoperation(start)

    sequence, timestamp_s, _ = _advance(
        runtime, controller, sequence=2, timestamp_s=1.02, steps=100
    )
    cube_before = _free_joint_position(runtime, "cube_joint")
    approach = {"right_pose": [0.05, -0.10, -0.38, 0.0, 0.0, 0.0, 1.0]}
    sequence, timestamp_s, contacts = _advance(
        runtime,
        controller,
        sequence=sequence,
        timestamp_s=timestamp_s,
        steps=200,
        **approach,
    )
    _, _, grip_contacts = _advance(
        runtime,
        controller,
        sequence=sequence,
        timestamp_s=timestamp_s,
        steps=200,
        right_trigger=1.0,
        right_grip=1.0,
        **approach,
    )
    contacts.update(grip_contacts)

    cube_after = _free_joint_position(runtime, "cube_joint")
    cube_displacement_m = float(np.linalg.norm(cube_after - cube_before))
    root_height_m = float(runtime.data.qpos[runtime._root_qpos_adr + 2])
    print(
        "Scene smoke passed: "
        f"root_z={root_height_m:.3f} m, cube_displacement={cube_displacement_m:.3f} m, "
        f"cube_contact_bodies={sorted(contacts)}"
    )
    if not 0.65 <= root_height_m <= 0.85:
        raise AssertionError(f"G1 did not remain upright (root_z={root_height_m:.3f} m)")
    if not contacts:
        raise AssertionError("The Dex3 hand never contacted the cube")
    if cube_displacement_m < 0.01:
        raise AssertionError(
            f"The grip sequence did not move the cube enough ({cube_displacement_m:.4f} m)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
