#!/usr/bin/env python3
"""Replay and validate one automatically generated tabletop VLA episode.

The replay starts from the scene XML reset pose, applies the recorded 43-D
joint targets through the same 200 Hz MuJoCo PD loop, and optionally renders a
camera movie.  It never modifies the source NPZ; the JSON report contains
state/object errors and an independent success decision so bad episodes can be
filtered before training.

Example::

    .venv_scene/bin/python scripts/run/replay_vla_scene_data.py \
        --scene can --episode outputs/vla_scene_data_curobo/episode_000000.npz \
        --render-dir outputs/vla_scene_replays/episode_000000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teleopit.scenes.runtime import SceneTeleopRuntime, scene_xml_path


def _object_pose(runtime: SceneTeleopRuntime, object_name: str) -> np.ndarray:
    mujoco = runtime._mujoco
    joint_name = "cube_joint" if object_name == "cube" else f"robosuite_{object_name}_free"
    joint_id = mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"scene is missing object free joint {joint_name!r}")
    address = int(runtime.model.jnt_qposadr[joint_id])
    return np.asarray(runtime.data.qpos[address : address + 7], dtype=np.float32).copy()


def _render_frame(renderer: object, runtime: SceneTeleopRuntime, path: Path) -> None:
    renderer.update_scene(runtime.data, camera="scene_head_camera")
    frame = np.asarray(renderer.render(), dtype=np.uint8)
    Image.fromarray(frame).save(path, quality=85)


def _make_video(frame_dir: Path, video_path: Path, fps: float) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    video_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "%06d.jpg"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return video_path.is_file()


def replay_episode(
    *,
    scene: str,
    episode_path: Path,
    render_dir: Path,
    image_stride: int,
    hz: float,
    state_tolerance: float,
    object_success_threshold: float,
    make_video: bool,
) -> dict[str, object]:
    with np.load(episode_path, allow_pickle=False) as archive:
        required = {"observation_state", "action", "object_pose", "timestamp_s"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"episode is missing required arrays: {missing}")
        recorded_state = np.asarray(archive["observation_state"], dtype=np.float64)
        actions = np.asarray(archive["action"], dtype=np.float64)
        recorded_object = np.asarray(archive["object_pose"], dtype=np.float64)
        timestamps = np.asarray(archive["timestamp_s"], dtype=np.float64)
    if recorded_state.ndim != 2 or recorded_state.shape[1] != 43:
        raise ValueError(f"observation_state must have shape [T, 43], got {recorded_state.shape}")
    if actions.shape != recorded_state.shape:
        raise ValueError(f"action shape must match observation_state, got {actions.shape}")
    if recorded_object.shape != (len(actions), 7):
        raise ValueError(f"object_pose must have shape [T, 7], got {recorded_object.shape}")
    if timestamps.shape != (len(actions),) or not np.all(np.isfinite(timestamps)):
        raise ValueError("timestamp_s must be a finite vector matching episode length")
    if not np.all(np.isfinite(actions)) or not np.all(np.isfinite(recorded_state)):
        raise ValueError("episode contains non-finite state/action values")

    if scene == "cube":
        object_name = "cube"
    else:
        object_name = scene
    runtime = SceneTeleopRuntime(scene_xml=scene_xml_path(scene if scene == "cube" else f"robosuite-{scene}"), input_timeout_s=1.0)
    runtime.reset()
    try:
        import mujoco

        renderer = mujoco.Renderer(runtime.model, height=224, width=224)
    except Exception as exc:
        raise RuntimeError("MuJoCo offscreen renderer is required for replay rendering") from exc

    render_dir.mkdir(parents=True, exist_ok=True)
    replay_state: list[np.ndarray] = []
    replay_object: list[np.ndarray] = []
    rendered_frame = 0
    for frame, target in enumerate(actions):
        runtime._target_by_joint = dict(zip(runtime._actuator_names, target, strict=True))
        for _ in range(runtime._control_decimation):
            runtime._apply_pd()
            runtime._mujoco.mj_step(runtime.model, runtime.data)
        replay_state.append(
            np.asarray(
                [runtime.data.qpos[runtime._qpos_adr[name]] for name in runtime._actuator_names],
                dtype=np.float64,
            )
        )
        replay_object.append(_object_pose(runtime, object_name).astype(np.float64))
        if frame % image_stride == 0:
            # Keep rendered files contiguous so ffmpeg's image-sequence
            # demuxer does not stop at the first stride gap (000000, 000005,
            # ... would otherwise produce a one-frame video).
            _render_frame(renderer, runtime, render_dir / f"{rendered_frame:06d}.jpg")
            rendered_frame += 1
    renderer.close()

    replay_state_array = np.asarray(replay_state)
    replay_object_array = np.asarray(replay_object)
    state_error = np.abs(replay_state_array - recorded_state)
    object_error = np.linalg.norm(replay_object_array[:, :3] - recorded_object[:, :3], axis=1)
    displacement = float(np.linalg.norm(replay_object_array[-1, :3] - replay_object_array[0, :3]))
    report: dict[str, object] = {
        "format": "teleopit_scene_vla_replay_v1",
        "scene": scene,
        "episode": str(episode_path),
        "frames": int(len(actions)),
        "control_hz": hz,
        "image_stride": image_stride,
        "state_max_abs_error": float(np.max(state_error, initial=0.0)),
        "state_rmse": float(np.sqrt(np.mean(np.square(state_error)))),
        "object_pose_max_position_error_m": float(np.max(object_error, initial=0.0)),
        "object_displacement_m": displacement,
        "success": bool(displacement >= object_success_threshold),
        "state_match": bool(np.max(state_error, initial=0.0) <= state_tolerance),
        "video": None,
    }
    if make_video:
        video_path = render_dir / "replay.mp4"
        if _make_video(render_dir, video_path, fps=hz / image_stride):
            report["video"] = str(video_path)
    (render_dir / "replay_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("cube", "bottle", "can", "lemon"), default="can")
    parser.add_argument("--episode", type=Path, required=True, help="Recorded episode_*.npz")
    parser.add_argument("--render-dir", type=Path, default=None)
    parser.add_argument("--image-stride", type=int, default=5)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--state-tolerance", type=float, default=0.05)
    parser.add_argument("--object-success-threshold", type=float, default=0.01)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args(argv)
    if args.image_stride <= 0 or args.hz <= 0 or args.state_tolerance < 0 or args.object_success_threshold < 0:
        parser.error("image-stride and hz must be positive; tolerances must be non-negative")
    episode = args.episode if args.episode.is_absolute() else PROJECT_ROOT / args.episode
    if not episode.is_file():
        parser.error(f"episode does not exist: {episode}")
    render_dir = args.render_dir or episode.parent / f"{episode.stem}_replay"
    if not render_dir.is_absolute():
        render_dir = PROJECT_ROOT / render_dir
    report = replay_episode(
        scene=args.scene,
        episode_path=episode.resolve(),
        render_dir=render_dir.resolve(),
        image_stride=args.image_stride,
        hz=args.hz,
        state_tolerance=args.state_tolerance,
        object_success_threshold=args.object_success_threshold,
        make_video=not args.no_video,
    )
    print(json.dumps(report, indent=2))
    return 0 if bool(report["success"]) and bool(report["state_match"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
