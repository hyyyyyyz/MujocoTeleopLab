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
import shutil
import subprocess
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
    KinematicObjectAttachment,
    place_object_on_table,
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


def _make_video(frame_dir: Path, video_path: Path, fps: float) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    try:
        subprocess.run(
            [
                ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
                "-i", str(frame_dir / "%06d.jpg"), "-c:v", "libx264",
                "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                str(video_path),
            ],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return video_path.is_file()


def generate_planned_episode(runtime: SceneTeleopRuntime, *, planner: object, object_name: str, episode_index: int, output_dir: Path, image_stride: int, hz: float, video_width: int = 640, video_height: int = 360) -> dict[str, object]:
    runtime.reset()
    variation = (
        planner.episode_variation(episode_index)
        if isinstance(planner, CuroboSceneTrajectoryPlanner)
        else {"object_dx": 0.0, "object_dy": 0.0}
    )
    place_object_on_table(
        runtime,
        object_name,
        offset_xy=(variation["object_dx"], variation["object_dy"]),
    )
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

        renderer = mujoco.Renderer(runtime.model, height=video_height, width=video_width)
    except Exception as exc:
        raise RuntimeError("MuJoCo offscreen renderer is required for VLA image capture") from exc

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    objects: list[np.ndarray] = []
    timestamps: list[float] = []
    grasp_states: list[bool] = []
    diagnostic_printed = False
    post_step_diagnostic_printed = False
    attachment = KinematicObjectAttachment(runtime, object_name)
    image_dir = output_dir / f"episode_{episode_index:06d}"
    image_dir.mkdir(parents=True, exist_ok=True)
    sequence = 2
    timestamp = 1.0 / hz
    video_frame_index = 0
    if isinstance(waypoints[0], WristWaypoint):
        samples = ((pose, trigger, grip, None) for pose, trigger, grip in interpolate_waypoints(waypoints, hz=hz))
    else:
        samples = ((np.asarray(item.wrist_pose, dtype=np.float64), item.trigger, item.grip, item) for item in waypoints)
    for frame, (pose, trigger, grip, joint_waypoint) in enumerate(samples):
        packet = _packet(sequence, timestamp, pose, trigger, grip)
        command = controller.update(packet)
        released_this_frame = False
        grasp_requested = bool(
            joint_waypoint.grasp if isinstance(joint_waypoint, JointWaypoint) else trigger > 0.5 and grip > 0.5
        )
        if grasp_requested and not attachment.attached:
            # The grasp assist is explicit and deterministic: the planner
            # owns the grasp phase, while the object follows the wrist once
            # that phase begins instead of being numerically ejected by a
            # contact-only finger closure.
            attachment.try_attach(max_distance_m=0.18)
            if not diagnostic_printed:
                hand_id = __import__('mujoco').mj_name2id(runtime.model, __import__('mujoco').mjtObj.mjOBJ_BODY, 'right_wrist_yaw_link')
                print('GRASP_DIAG pre_step hand=', runtime.data.xpos[hand_id].tolist(), 'object=', _object_pose(runtime, object_name)[:3].tolist(), 'hand_xmat=', runtime.data.xmat[hand_id].reshape(3,3).tolist(), 'contact=', attachment.attached, flush=True)
                diagnostic_printed = True
        elif not grasp_requested and attachment.attached:
            attachment.release()
            released_this_frame = True
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
            # SIMPLE's MP agent uses the robot EEF controller's calibrated
            # close pose during the grasp/lift/place phases.  Sparse Pico
            # fingertip gestures are appropriate for teleop, but are not a
            # deterministic pinch for scripted data generation; they let the
            # cube slip or tip.  Use the released Dex3 close pose only while
            # a JointWaypoint explicitly requests a grasp, and retain IK for
            # the open/approach frames.
            if joint_waypoint.grasp:
                right_hand_q = np.asarray(
                    [0.02331954, -0.02398408, -0.22170663, 0.25662386, 1.3371105, 0.3085137, 0.9805285],
                    dtype=np.float64,
                )
                left_hand_q = np.zeros(len(runtime._left_hand_joint_names), dtype=np.float64)
            else:
                right_hand_q = runtime._retargeting_ik.right_hand_ik_solver(command.right_fingers)
                left_hand_q = runtime._retargeting_ik.left_hand_ik_solver(command.left_fingers)
            runtime._target_by_joint.update(dict(zip(runtime._right_hand_joint_names, right_hand_q, strict=True)))
            runtime._target_by_joint.update(dict(zip(runtime._left_hand_joint_names, left_hand_q, strict=True)))
        else:
            runtime.update_policy(command, active=True)
        for _ in range(runtime._control_decimation):
            runtime._apply_pd()
            runtime._mujoco.mj_step(runtime.model, runtime.data)
        if grasp_requested and not post_step_diagnostic_printed:
            mujoco_mod = __import__('mujoco')
            hand_id = mujoco_mod.mj_name2id(runtime.model, mujoco_mod.mjtObj.mjOBJ_BODY, 'right_wrist_yaw_link')
            finger_ids = [mujoco_mod.mj_name2id(runtime.model, mujoco_mod.mjtObj.mjOBJ_BODY, n) for n in ('right_hand_index_1_link','right_hand_middle_1_link','right_hand_thumb_2_link')]
            print('GRASP_DIAG post_step hand=', runtime.data.xpos[hand_id].tolist(), 'fingers=', [runtime.data.xpos[i].tolist() for i in finger_ids], 'object=', _object_pose(runtime, object_name)[:3].tolist(), 'contact=', attachment.attached, 'ncon=', int(runtime.data.ncon), flush=True)
            post_step_diagnostic_printed = True
        if released_this_frame:
            # Let MuJoCo integrate the released free body.  Do not snap it to
            # the tabletop: SIMPLE records the real drop/settling dynamics.
            pass
        else:
            attachment.update()
        state = np.array([runtime.data.qpos[runtime._qpos_adr[name]] for name in runtime._actuator_names], dtype=np.float32)
        action = np.array([runtime._target_by_joint[name] for name in runtime._actuator_names], dtype=np.float32)
        states.append(state)
        actions.append(action)
        objects.append(_object_pose(runtime, object_name))
        timestamps.append(timestamp)
        grasp_states.append(attachment.attached)
        # The MP4 is a smooth 50 Hz diagnostic recording.  Training images
        # remain strided to keep the dataset compact, but video capture is
        # deliberately independent of ``image_stride``.
        frame_rgb = _capture(renderer, runtime)
        if frame % max(1, image_stride) == 0:
            Image.fromarray(frame_rgb).resize((224, 224), Image.Resampling.BILINEAR).save(
                image_dir / f"{frame:06d}.jpg", quality=85
            )
        video_frame_dir = output_dir / ".video_frames" / f"episode_{episode_index:06d}"
        video_frame_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame_rgb).save(video_frame_dir / f"{video_frame_index:06d}.jpg", quality=92)
        video_frame_index += 1
        sequence += 1
        timestamp += 1.0 / hz
    renderer.close()
    video_frame_dir = output_dir / ".video_frames" / f"episode_{episode_index:06d}"
    video_path = output_dir / f"episode_{episode_index:06d}.mp4"
    video_ok = _make_video(video_frame_dir, video_path, fps=hz)
    shutil.rmtree(video_frame_dir, ignore_errors=True)

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
    grasped = bool(any(grasp_states))
    success = bool(grasped and lifted and placed)
    np.savez_compressed(
        output_dir / f"episode_{episode_index:06d}.npz",
        observation_state=np.asarray(states, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32),
        object_pose=np.asarray(objects, dtype=np.float32),
        timestamp_s=np.asarray(timestamps, dtype=np.float64),
        success=np.asarray(success, dtype=np.bool_),
        object_horizontal_displacement_m=np.asarray(object_delta_xy, dtype=np.float32),
        object_max_lift_m=np.asarray(max_lift, dtype=np.float32),
        object_final_height_error_m=np.asarray(final_height_error, dtype=np.float32),
        grasp_state=np.asarray(grasp_states, dtype=np.bool_),
    )
    return {
        "success": success,
        "object_horizontal_displacement_m": object_delta_xy,
        "object_max_lift_m": max_lift,
        "object_final_height_error_m": final_height_error,
        "lifted": lifted,
        "placed": placed,
        "grasped": grasped,
        "video": str(video_path.name) if video_ok else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("cube", "bottle", "can", "lemon"), default="can")
    parser.add_argument("--scene-xml", type=Path)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vla_scene_data"))
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--image-stride", type=int, default=5)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=360)
    parser.add_argument(
        "--planner",
        choices=("curobo", "scripted"),
        default="curobo",
        help="Planning backend. 'curobo' is the collision-aware production path; 'scripted' is only a plumbing smoke test.",
    )
    args = parser.parse_args(argv)
    if args.episodes <= 0 or args.hz <= 0 or args.image_stride <= 0:
        parser.error("episodes, hz and image-stride must be positive")
    if args.video_width <= 0 or args.video_height <= 0 or args.video_width % 2 or args.video_height % 2:
        parser.error("video dimensions must be positive even integers")
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
            metrics = generate_planned_episode(runtime, planner=planner, object_name=object_name, episode_index=index, output_dir=output_dir, image_stride=args.image_stride, hz=args.hz, video_width=args.video_width, video_height=args.video_height)
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
