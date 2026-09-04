#!/usr/bin/env python3
"""Generate image-language-action demonstrations for a 43-DOF scene.

The default CuRobo planner runs entirely in MuJoCo and requires a CUDA CuRobo
installation, but does not require PICO/XRoboToolkit. Heavy scene assets stay outside Git; the output directory
is also ignored.  The resulting layout is intentionally simple and editable:

    episode_000000.npz  (state, action, object_pose, timestamps)
    episode_000000/000000.jpg ...
    episodes.jsonl       (language/task metadata and success flag)
    schema.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teleopit.scenes.controller import SimpleSceneController
from teleopit.scenes.runtime import SceneTeleopRuntime, scene_xml_path
from teleopit.scenes.vla_datagen import (
    CuroboSceneTrajectoryPlanner,
    JointWaypoint,
    ScriptedPickPlacePlanner,
    WristWaypoint,
    interpolate_waypoints,
)
from teleopit.scenes.xr_packet import SceneXRPacket


def _packet(sequence: int, timestamp_s: float, pose: np.ndarray, trigger: float, grip: float, *, left_menu: bool = False) -> SceneXRPacket:
    values = {
        "sequence": sequence,
        "timestamp_s": timestamp_s,
        "left_pose": [-0.2, 0.0, -0.3, 0.0, 0.0, 0.0, 1.0],
        "right_pose": pose.tolist(),
        "head_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "left_axis": [0.0, 0.0],
        "right_axis": [0.0, 0.0],
        "left_trigger": 0.0,
        "right_trigger": trigger,
        "left_grip": 0.0,
        "right_grip": grip,
        "a": False,
        "b": False,
        "x": False,
        "y": False,
        "left_menu": left_menu,
    }
    return SceneXRPacket.from_mapping(values)


def _object_pose(runtime: SceneTeleopRuntime, object_name: str) -> np.ndarray:
    mujoco = runtime._mujoco
    joint_name = "cube_joint" if object_name == "cube" else f"robosuite_{object_name}_free"
    joint_id = mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"scene is missing object free joint {joint_name!r}")
    address = int(runtime.model.jnt_qposadr[joint_id])
    return np.asarray(runtime.data.qpos[address : address + 7], dtype=np.float32).copy()


def _capture(renderer: object, runtime: SceneTeleopRuntime) -> np.ndarray:
    renderer.update_scene(runtime.data, camera="scene_head_camera")
    return np.asarray(renderer.render(), dtype=np.uint8).copy()


def _place_object_on_table(runtime: SceneTeleopRuntime, object_name: str) -> None:
    """Deterministically reset the manipulated object onto the tabletop.

    Older generated scene XMLs placed robosuite objects at z=0.9 regardless
    of their size, leaving them suspended above the table.  Set only the free
    joint's initial z and velocity here so old assets remain usable while the
    asset builder is corrected for future scenes.
    """
    mujoco = runtime._mujoco
    model, data = runtime.model, runtime.data
    joint_name = "cube_joint" if object_name == "cube" else f"robosuite_{object_name}_free"
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"scene is missing object free joint {joint_name!r}")
    object_body = int(model.jnt_bodyid[joint_id])
    object_geom_ids = [
        geom_id for geom_id in range(model.ngeom) if int(model.geom_bodyid[geom_id]) == object_body
    ]
    collision_geom = next(
        (geom_id for geom_id in object_geom_ids if str(model.geom(geom_id).name).endswith("_collision")),
        object_geom_ids[0] if object_geom_ids else None,
    )
    if collision_geom is None:
        raise ValueError(f"scene object body {object_body} has no geometry")
    half_height = float(model.geom_size[collision_geom][2])
    table_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table_body")
    if table_body < 0:
        raise ValueError("scene is missing table_body")
    table_top = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == table_body and str(model.geom(geom_id).name) == "table_top"
    ]
    if not table_top:
        raise ValueError("scene is missing table_top")
    table_geom = table_top[0]
    table_height = float(data.geom_xpos[table_geom][2] + model.geom_size[table_geom][2])
    object_qpos = int(model.jnt_qposadr[joint_id])
    object_qvel = int(model.jnt_dofadr[joint_id])
    data.qpos[object_qpos + 2] = table_height + half_height + 0.002
    data.qvel[object_qvel : object_qvel + 6] = 0.0
    mujoco.mj_forward(model, data)


def generate_planned_episode(runtime: SceneTeleopRuntime, *, planner: object, object_name: str, episode_index: int, output_dir: Path, image_stride: int, hz: float) -> dict[str, object]:
    runtime.reset()
    _place_object_on_table(runtime, object_name)
    controller = SimpleSceneController()
    waypoints = planner.plan(object_name=object_name, episode_index=episode_index, runtime=runtime) if isinstance(planner, CuroboSceneTrajectoryPlanner) else planner.plan(object_name=object_name, episode_index=episode_index)
    if not waypoints:
        raise RuntimeError("planner returned an empty trajectory")
    # Enter arm mode using the same SIMPLE chord as the PICO bridge.
    first_pose = waypoints[0].pose if isinstance(waypoints[0], WristWaypoint) else waypoints[0].wrist_pose
    activation = controller.update(_packet(1, 0.0, np.asarray(first_pose), 1.0, 0.0, left_menu=True))
    if not activation.activation_toggled:
        raise RuntimeError("failed to activate scripted scene teleoperation")
    if isinstance(waypoints[0], WristWaypoint):
        runtime._start_teleoperation(activation)

    renderer = None
    try:
        import mujoco

        renderer = mujoco.Renderer(runtime.model, height=224, width=224)
    except Exception as exc:
        raise RuntimeError("MuJoCo offscreen renderer is required for VLA image capture") from exc

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    objects: list[np.ndarray] = []
    timestamps: list[float] = []
    image_dir = output_dir / f"episode_{episode_index:06d}"
    image_dir.mkdir(parents=True, exist_ok=True)
    sequence = 2
    timestamp = 1.0 / hz
    if isinstance(waypoints[0], WristWaypoint):
        samples = ((pose, trigger, grip, None) for pose, trigger, grip in interpolate_waypoints(waypoints, hz=hz))
    else:
        samples = ((np.asarray(item.wrist_pose, dtype=np.float64), item.trigger, item.grip, item) for item in waypoints)
    for frame, (pose, trigger, grip, joint_waypoint) in enumerate(samples):
        packet = _packet(sequence, timestamp, pose, trigger, grip)
        command = controller.update(packet)
        if isinstance(joint_waypoint, JointWaypoint):
            # Keep the balance/WBC target alive while CuRobo overrides only
            # the planned arm joints.  Bypassing update_policy here leaves
            # legs at the raw XML reset pose and makes the robot push/fall
            # instead of executing a stable manipulation.
            runtime.update_policy(command, active=True)
            runtime._target_by_joint.update(joint_waypoint.positions)
            # CuRobo's URDF intentionally plans the 10-DOF waist/right-arm
            # chain.  Apply the controller-compatible Dex3 finger IK for the
            # trigger/grip state separately so the object is actually held.
            right_hand_q = runtime._retargeting_ik.right_hand_ik_solver(command.right_fingers)
            left_hand_q = runtime._retargeting_ik.left_hand_ik_solver(command.left_fingers)
            runtime._target_by_joint.update(dict(zip(runtime._right_hand_joint_names, right_hand_q, strict=True)))
            runtime._target_by_joint.update(dict(zip(runtime._left_hand_joint_names, left_hand_q, strict=True)))
        else:
            runtime.update_policy(command, active=True)
        for _ in range(runtime._control_decimation):
            runtime._apply_pd()
            runtime._mujoco.mj_step(runtime.model, runtime.data)
        state = np.array([runtime.data.qpos[runtime._qpos_adr[name]] for name in runtime._actuator_names], dtype=np.float32)
        action = np.array([runtime._target_by_joint[name] for name in runtime._actuator_names], dtype=np.float32)
        states.append(state)
        actions.append(action)
        objects.append(_object_pose(runtime, object_name))
        timestamps.append(timestamp)
        if frame % max(1, image_stride) == 0:
            frame_rgb = _capture(renderer, runtime)
            Image.fromarray(frame_rgb).save(image_dir / f"{frame:06d}.jpg", quality=85)
        sequence += 1
        timestamp += 1.0 / hz
    renderer.close()

    object_positions = np.asarray(objects, dtype=np.float64)[:, :3]
    object_delta_xy = float(np.linalg.norm(object_positions[-1, :2] - object_positions[0, :2]))
    initial_z = float(object_positions[0, 2])
    max_lift = float(np.max(object_positions[:, 2]) - initial_z)
    final_height_error = float(abs(object_positions[-1, 2] - initial_z))
    # A valid pick/place must show a lift and a rightward horizontal transfer,
    # then settle back near the original tabletop height.  Pure pushes and
    # drops therefore no longer pass the collection success flag.
    lifted = bool(max_lift >= 0.03)
    placed = bool(
        object_delta_xy >= 0.10
        and final_height_error <= 0.06
    )
    success = bool(lifted and placed)
    np.savez_compressed(
        output_dir / f"episode_{episode_index:06d}.npz",
        observation_state=np.asarray(states, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32),
        object_pose=np.asarray(objects, dtype=np.float32),
        timestamp_s=np.asarray(timestamps, dtype=np.float64),
    )
    return {
        "success": success,
        "object_horizontal_displacement_m": object_delta_xy,
        "object_max_lift_m": max_lift,
        "object_final_height_error_m": final_height_error,
        "lifted": lifted,
        "placed": placed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("cube", "bottle", "can", "lemon"), default="can")
    parser.add_argument("--scene-xml", type=Path)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vla_scene_data"))
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--image-stride", type=int, default=5)
    parser.add_argument(
        "--planner",
        choices=("curobo", "scripted"),
        default="curobo",
        help="Planning backend. 'curobo' is the collision-aware production path; 'scripted' is only a plumbing smoke test.",
    )
    args = parser.parse_args(argv)
    if args.episodes <= 0 or args.hz <= 0 or args.image_stride <= 0:
        parser.error("episodes, hz and image-stride must be positive")
    if args.scene == "cube":
        object_name = "cube"
        xml = args.scene_xml.resolve() if args.scene_xml else scene_xml_path("cube")
    else:
        object_name = args.scene
        xml = args.scene_xml.resolve() if args.scene_xml else scene_xml_path(f"robosuite-{args.scene}")
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    schema = {
        "format": "teleopit_scene_vla_v1",
        "scene": args.scene,
        "fps": args.hz,
        "state_shape": [43],
        "action_shape": [43],
        "image_size": [224, 224],
        "language_field": "task",
    }
    (output_dir / "schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    episodes_path = output_dir / "episodes.jsonl"
    runtime = SceneTeleopRuntime(scene_xml=xml, input_timeout_s=1.0)
    if args.planner == "curobo":
        from teleopit.scenes.vla_datagen import CuroboSceneTrajectoryPlanner

        urdf = PROJECT_ROOT / "third_party/decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand.urdf"
        planner = CuroboSceneTrajectoryPlanner(runtime, urdf_path=str(urdf))
    else:
        planner = ScriptedPickPlacePlanner()
    with episodes_path.open("w", encoding="utf-8") as stream:
        for index in range(args.episodes):
            started = time.time()
            metrics = generate_planned_episode(runtime, planner=planner, object_name=object_name, episode_index=index, output_dir=output_dir, image_stride=args.image_stride, hz=args.hz)
            stream.write(json.dumps({
                "episode_index": index,
                "scene": args.scene,
                "task": f"pick up the {args.scene} and place it to the right",
                **metrics,
                "duration_s": round(time.time() - started, 3),
                "data": f"episode_{index:06d}.npz",
                "images": f"episode_{index:06d}",
            }) + "\n")
    print(f"Generated {args.episodes} VLA episode(s) under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
