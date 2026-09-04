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
    if ffmpeg is not None:
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
            return video_path.is_file()
        except (OSError, subprocess.CalledProcessError):
            pass

    # The scene venv intentionally carries PyAV, while some minimal servers do
    # not install the ffmpeg command-line binary.  Keep video recording
    # self-contained so every accepted trajectory has a smooth MP4 diagnostic.
    try:
        import av

        frames = sorted(frame_dir.glob("*.jpg"))
        if not frames:
            return False
        container = av.open(str(video_path), mode="w")
        # PyAV expects an integer or Fraction for ``rate``; passing a Python
        # float raises ``AttributeError: numerator`` on current releases.
        stream = container.add_stream("libx264", rate=max(1, int(round(fps))))
        stream.width, stream.height = Image.open(frames[0]).size
        stream.pix_fmt = "yuv420p"
        for frame_path in frames:
            with Image.open(frame_path) as image:
                rgb = image.convert("RGB")
                video_frame = av.VideoFrame.from_ndarray(np.asarray(rgb), format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return video_path.is_file()
    except Exception:
        try:
            video_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False


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
    # Calibrate against SIMPLE's neutral controller pose.  Calibrating on the
    # first planned waypoint would make the approach a zero-motion command.
    neutral_pose = np.asarray((0.2, 0.0, -0.3, 0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    activation = controller.update(_packet(1, 0.0, neutral_pose, 1.0, 0.0, left_menu=True))
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
    failure_reason: str | None = None
    contact_loss_frames = 0
    previous_phase: str | None = None
    attachment = KinematicObjectAttachment(runtime, object_name)
    image_dir = output_dir / f"episode_{episode_index:06d}"
    image_dir.mkdir(parents=True, exist_ok=True)
    sequence = 2
    timestamp = 1.0 / hz
    # Match SIMPLE's neutral warm-up before sending the approach trajectory.
    for _ in range(int(round(2.0 * hz))):
        neutral_packet = _packet(sequence, timestamp, neutral_pose, 0.0, 0.0)
        neutral_command = controller.update(neutral_packet)
        runtime.update_policy(neutral_command, active=True)
        for _ in range(runtime._control_decimation):
            runtime._apply_pd()
            runtime._mujoco.mj_step(runtime.model, runtime.data)
        sequence += 1
        timestamp += 1.0 / hz
    # Replay must start exactly where the first recorded action starts. The
    # neutral warm-up moves the robot away from the XML reset pose, and CuRobo
    # episodes may also use a varied object placement. Persist the complete
    # MuJoCo state at this boundary rather than asking replay to guess it.
    initial_qpos = np.asarray(runtime.data.qpos, dtype=np.float64).copy()
    initial_qvel = np.asarray(runtime.data.qvel, dtype=np.float64).copy()
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
        phase = joint_waypoint.phase if isinstance(joint_waypoint, JointWaypoint) else "scripted"
        if phase in ("lift", "transport") and previous_phase not in ("lift", "transport"):
            if not attachment.ever_contacted:
                failure_reason = "no_finger_object_contact_before_lift"
                break
        previous_phase = phase
        if grasp_requested and not attachment.attached:
            # This is detection only: the object remains a free MuJoCo body.
            # The planner owns the close/squeeze phase, while attachment state
            # records whether real contact has actually occurred.
            attachment.try_attach(max_distance_m=0.18)
        elif not grasp_requested and attachment.attached:
            attachment.release()
            released_this_frame = True
            contact_loss_frames = 0
        elif not grasp_requested:
            contact_loss_frames = 0
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
            # SIMPLE's MP agent closes and then adds a directional Dex3
            # squeeze stroke before lift. Sparse Pico fingertip gestures are
            # appropriate for teleop, but are not a deterministic pinch for
            # scripted data generation. Joint waypoints therefore carry the
            # planned hand posture; open/approach frames retain regular IK.
            if joint_waypoint.right_hand_positions is not None:
                right_hand_q = np.asarray(joint_waypoint.right_hand_positions, dtype=np.float64)
            elif joint_waypoint.grasp:
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
        if released_this_frame:
            # Let MuJoCo integrate the released free body.  Do not snap it to
            # the tabletop: SIMPLE records the real drop/settling dynamics.
            pass
        else:
            attachment.update()
            if grasp_requested and phase in ("lift", "transport") and not attachment.attached:
                contact_loss_frames += 1
            else:
                contact_loss_frames = 0
            # MuJoCo contact manifolds can disappear for one or two frames
            # while the object rolls between fingertips.  Treat only a
            # sustained 0.2 s loss during lift/transport as a dropped object.
            if contact_loss_frames >= max(1, int(round(hz * 0.2))):
                failure_reason = "finger_object_contact_lost"
                break
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

    if not objects:
        raise RuntimeError("planner produced no executable frames")
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
    if not success and failure_reason is None:
        failure_reason = "object_motion_predicate_failed"
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
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
    )
    return {
        "success": success,
        "object_horizontal_displacement_m": object_delta_xy,
        "object_max_lift_m": max_lift,
        "object_final_height_error_m": final_height_error,
        "lifted": lifted,
        "placed": placed,
        "grasped": grasped,
        "failure_reason": failure_reason,
        "video": str(video_path.name) if video_ok else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("cube", "bottle", "can", "lemon"), default="can")
    parser.add_argument("--scene-xml", type=Path)
    parser.add_argument(
        "--grasp-asset",
        type=Path,
        help="Optional local SIMPLE/Bodex .npy/.npz phase cache. The file is "
        "read locally and is never copied into the output or Git.",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--successful-episodes",
        type=int,
        help="Collect this many successful episodes, retrying failed attempts. "
        "When set, --episodes is used as the maximum number of attempts.",
    )
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
    if args.successful_episodes is not None and args.successful_episodes <= 0:
        parser.error("successful-episodes must be positive")
    if args.successful_episodes is not None and args.episodes < args.successful_episodes:
        parser.error("episodes must be at least successful-episodes when retrying")
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
        grasp_asset = str(args.grasp_asset.resolve()) if args.grasp_asset is not None else None
        planner = CuroboSceneTrajectoryPlanner(runtime, urdf_path=str(urdf), grasp_asset=grasp_asset)
    else:
        planner = ScriptedPickPlacePlanner()
    target_successes = args.successful_episodes
    max_attempts = args.episodes
    successes = 0
    attempts = 0
    with episodes_path.open("w", encoding="utf-8") as stream:
        while attempts < max_attempts and (target_successes is None or successes < target_successes):
            index = attempts
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
            stream.flush()
            attempts += 1
            successes += int(bool(metrics["success"]))
            if target_successes is not None and not metrics["success"]:
                print(f"Attempt {attempts}/{max_attempts} failed; retrying (successful={successes}/{target_successes})")
    if target_successes is not None and successes < target_successes:
        print(
            f"Collected {successes}/{target_successes} successful episode(s) "
            f"after {attempts} attempt(s); failed attempts are retained for review."
        )
        return 2
    print(
        f"Generated {attempts} VLA episode(s) under {output_dir}"
        + (f" ({successes} successful)" if target_successes is not None else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
