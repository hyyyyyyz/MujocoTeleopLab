#!/usr/bin/env python3
"""Read-only synchronized reviewer for Teleopit sim2real recordings."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import html
import json
from pathlib import Path
import sys
import time
from typing import Any

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teleopit.constants import FULL_QPOS_DIM, NUM_JOINTS
from teleopit.recording.hdf5 import (
    ACTION_KEY,
    FRAME_INDEX_KEY,
    HAND_ACTION_KEY,
    HAND_STATE_KEY,
    HDF5_RECORDING_FORMAT,
    HDF5_RECORDING_VERSION,
    MODE_KEY,
    NECK_ACTION_KEY,
    NECK_STATE_KEY,
    STATE_KEY,
    TIMESTAMP_KEY,
)
from teleopit.runtime.assets import UNITREE_G1_XML, missing_gmr_assets_message


DEFAULT_RECORDING_ROOT = PROJECT_ROOT / "data" / "recordings" / "sim2real_hdf5"
DEFAULT_XML = UNITREE_G1_XML

JOINT_GROUPS: tuple[tuple[str, slice], ...] = (
    ("left leg", slice(0, 6)),
    ("right leg", slice(6, 12)),
    ("waist", slice(12, 15)),
    ("left arm", slice(15, 22)),
    ("right arm", slice(22, 29)),
)
GROUP_COLORS = ("#3b82f6", "#06b6d4", "#f59e0b", "#ec4899", "#8b5cf6")


@dataclass(frozen=True)
class RecordingEpisode:
    episode_index: int
    frames: int
    task: str
    data_path: Path
    video_path: Path

    def label(self, fps: int) -> str:
        duration_s = self.frames / fps
        return f"#{self.episode_index:06d} · {self.task} · {duration_s:.1f}s"


@dataclass(frozen=True)
class RecordingDataset:
    root: Path
    schema: dict[str, Any]
    features: dict[str, Any]
    fps: int
    image_key: str
    image_shape: tuple[int, int, int]
    mode_names: dict[int, str]
    joint_names: tuple[str, ...]
    hand_names: tuple[str, ...]
    has_hand_action: bool
    has_neck_action: bool
    episodes: tuple[RecordingEpisode, ...]


@dataclass(frozen=True)
class EpisodeReviewData:
    episode: RecordingEpisode
    frame_index: np.ndarray
    timestamps: np.ndarray
    state: np.ndarray
    mode: np.ndarray
    action: np.ndarray
    hand_state: np.ndarray | None
    hand_action: np.ndarray | None
    neck_state: np.ndarray | None
    neck_action: np.ndarray | None
    joint_error: np.ndarray
    group_error: dict[str, np.ndarray]
    root_orientation_error_rad: np.ndarray
    joint_rmse_rad: float
    root_orientation_rmse_rad: float
    max_joint_error_rad: float
    max_joint_error_frame: int
    max_joint_error_name: str


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Recording {label} not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid recording {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Recording {label} must contain a JSON object: {path}")
    return payload


def _feature_shape(features: dict[str, Any], key: str) -> tuple[int, ...]:
    feature = features.get(key)
    if not isinstance(feature, dict):
        raise ValueError(f"Recording schema is missing feature {key!r}")
    shape = feature.get("shape")
    if not isinstance(shape, list) or not all(isinstance(value, int) for value in shape):
        raise ValueError(f"Recording schema feature {key!r} has invalid shape {shape!r}")
    return tuple(shape)


def _feature_dtype(features: dict[str, Any], key: str) -> np.dtype:
    feature = features.get(key)
    raw_dtype = feature.get("dtype") if isinstance(feature, dict) else None
    if not isinstance(raw_dtype, str):
        raise ValueError(
            f"Recording schema feature {key!r} has invalid dtype {raw_dtype!r}"
        )
    try:
        return np.dtype(raw_dtype)
    except TypeError as exc:
        raise ValueError(
            f"Recording schema feature {key!r} has invalid dtype {raw_dtype!r}"
        ) from exc


def _feature_names(features: dict[str, Any], key: str, expected: int) -> tuple[str, ...]:
    feature = features.get(key)
    names = feature.get("names") if isinstance(feature, dict) else None
    if not isinstance(names, list) or len(names) != expected:
        raise ValueError(
            f"Recording schema feature {key!r} must define {expected} names, got {names!r}"
        )
    return tuple(str(name) for name in names)


def _resolve_recording_path(root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Recording manifest {label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"Recording manifest {label} must be relative to {root}: {value!r}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Recording manifest {label} escapes dataset root: {value!r}") from exc
    if not resolved.is_file():
        raise ValueError(f"Recording manifest {label} not found: {resolved}")
    return resolved


def _validate_hdf5_episode(
    data_path: Path,
    *,
    frames: int,
    features: dict[str, Any],
    keys: tuple[str, ...],
) -> None:
    try:
        with h5py.File(data_path, "r") as h5:
            for key in keys:
                if key not in h5:
                    raise ValueError(f"Recording episode {data_path} is missing HDF5 dataset {key!r}")
                expected_shape = (frames, *_feature_shape(features, key))
                actual_shape = tuple(h5[key].shape)
                if actual_shape != expected_shape:
                    raise ValueError(
                        f"Recording episode {data_path} dataset {key!r} shape {actual_shape} "
                        f"!= manifest/schema shape {expected_shape}"
                    )
                expected_dtype = _feature_dtype(features, key)
                actual_dtype = h5[key].dtype
                if actual_dtype != expected_dtype:
                    raise ValueError(
                        f"Recording episode {data_path} dataset {key!r} dtype {actual_dtype} "
                        f"!= schema dtype {expected_dtype}"
                    )
    except OSError as exc:
        raise ValueError(f"Cannot open recording episode HDF5: {data_path}") from exc


def load_recording_dataset(recording_root: str | Path) -> RecordingDataset:
    """Load and validate the dataset-level schema and episode manifest."""

    root = Path(recording_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Recording root not found: {root}")

    schema = _read_json_object(root / "schema.json", label="schema.json")
    if schema.get("format") != HDF5_RECORDING_FORMAT:
        raise ValueError(
            f"Unsupported recording format {schema.get('format')!r}; expected {HDF5_RECORDING_FORMAT!r}"
        )
    if schema.get("version") != HDF5_RECORDING_VERSION:
        raise ValueError(
            f"Unsupported recording version {schema.get('version')!r}; expected {HDF5_RECORDING_VERSION}"
        )

    fps = schema.get("fps")
    if not isinstance(fps, int) or fps <= 0:
        raise ValueError(f"Recording schema fps must be a positive integer, got {fps!r}")
    features = schema.get("features")
    if not isinstance(features, dict):
        raise ValueError("Recording schema features must be an object")

    expected_shapes = {
        FRAME_INDEX_KEY: (),
        TIMESTAMP_KEY: (),
        STATE_KEY: (68,),
        MODE_KEY: (),
        ACTION_KEY: (FULL_QPOS_DIM,),
    }
    for key, expected_shape in expected_shapes.items():
        actual_shape = _feature_shape(features, key)
        if actual_shape != expected_shape:
            raise ValueError(
                f"Recording schema feature {key!r} shape {actual_shape} != {expected_shape}"
            )

    video_keys = [
        str(key)
        for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]
    if len(video_keys) != 1:
        raise ValueError(
            f"Recording reviewer requires exactly one video feature, found {video_keys}"
        )
    image_key = video_keys[0]
    image_shape = _feature_shape(features, image_key)
    if len(image_shape) != 3 or image_shape[2] != 3 or min(image_shape) <= 0:
        raise ValueError(
            f"Recording video feature {image_key!r} must have [height, width, 3] shape, got {image_shape}"
        )

    mode_feature = features[MODE_KEY]
    raw_mode_values = mode_feature.get("values") if isinstance(mode_feature, dict) else None
    if not isinstance(raw_mode_values, dict) or not raw_mode_values:
        raise ValueError(f"Recording schema feature {MODE_KEY!r} must define mode values")
    mode_names: dict[int, str] = {}
    for name, value in raw_mode_values.items():
        if not isinstance(value, int) or value in mode_names:
            raise ValueError(f"Recording schema has invalid or duplicate mode code {value!r}")
        mode_names[value] = str(name)

    action_names = _feature_names(features, ACTION_KEY, FULL_QPOS_DIM)
    joint_names = action_names[7:]
    if len(joint_names) != NUM_JOINTS:
        raise ValueError(
            f"Recording action names must contain {NUM_JOINTS} reference joints, got {len(joint_names)}"
        )

    hand_type = str(schema.get("hand_type", "none")).strip().lower()
    neck_type = str(schema.get("neck_type", "none")).strip().lower()
    has_hand_action = hand_type != "none"
    has_neck_action = neck_type != "none"
    hand_names: tuple[str, ...] = ()
    if has_hand_action:
        if _feature_shape(features, HAND_STATE_KEY) != (12,):
            raise ValueError(f"Recording schema feature {HAND_STATE_KEY!r} must be 12D")
        _feature_names(features, HAND_STATE_KEY, 12)
        if _feature_shape(features, HAND_ACTION_KEY) != (12,):
            raise ValueError(f"Recording schema feature {HAND_ACTION_KEY!r} must be 12D")
        hand_names = _feature_names(features, HAND_ACTION_KEY, 12)
    elif HAND_STATE_KEY in features or HAND_ACTION_KEY in features:
        raise ValueError(f"Recording schema hand_type={hand_type!r} must not define hand features")
    if has_neck_action:
        if _feature_shape(features, NECK_STATE_KEY) != (2,):
            raise ValueError(f"Recording schema feature {NECK_STATE_KEY!r} must be 2D")
        _feature_names(features, NECK_STATE_KEY, 2)
        if _feature_shape(features, NECK_ACTION_KEY) != (2,):
            raise ValueError(f"Recording schema feature {NECK_ACTION_KEY!r} must be 2D")
        _feature_names(features, NECK_ACTION_KEY, 2)
    elif NECK_STATE_KEY in features or NECK_ACTION_KEY in features:
        raise ValueError(f"Recording schema neck_type={neck_type!r} must not define neck features")

    hdf5_keys = [FRAME_INDEX_KEY, TIMESTAMP_KEY, STATE_KEY, MODE_KEY, ACTION_KEY]
    if has_hand_action:
        hdf5_keys.extend((HAND_STATE_KEY, HAND_ACTION_KEY))
    if has_neck_action:
        hdf5_keys.extend((NECK_STATE_KEY, NECK_ACTION_KEY))

    manifest_path = root / "episodes.jsonl"
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Recording manifest not found or unreadable: {manifest_path}") from exc

    episodes: list[RecordingEpisode] = []
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            entry = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {manifest_path}:{line_number}") from exc
        if not isinstance(entry, dict):
            raise ValueError(f"Recording manifest entry at line {line_number} must be an object")

        expected_index = len(episodes)
        episode_index = entry.get("episode_index")
        if episode_index != expected_index:
            raise ValueError(
                f"Recording episode indices must be contiguous from 0; line {line_number} "
                f"expected {expected_index}, got {episode_index!r}"
            )
        frames = entry.get("frames")
        if not isinstance(frames, int) or frames <= 0:
            raise ValueError(
                f"Recording manifest entry {episode_index} frames must be positive, got {frames!r}"
            )
        task = entry.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"Recording manifest entry {episode_index} task must not be empty")

        videos = entry.get("videos")
        if not isinstance(videos, dict) or image_key not in videos:
            raise ValueError(
                f"Recording manifest entry {episode_index} is missing video path for {image_key!r}"
            )
        data_path = _resolve_recording_path(
            root,
            entry.get("data"),
            label=f"episode {episode_index} data",
        )
        video_path = _resolve_recording_path(
            root,
            videos[image_key],
            label=f"episode {episode_index} video {image_key}",
        )
        _validate_hdf5_episode(
            data_path,
            frames=frames,
            features=features,
            keys=tuple(hdf5_keys),
        )
        episodes.append(
            RecordingEpisode(
                episode_index=episode_index,
                frames=frames,
                task=task.strip(),
                data_path=data_path,
                video_path=video_path,
            )
        )

    if not episodes:
        raise ValueError(f"Recording manifest has no saved episodes: {manifest_path}")

    return RecordingDataset(
        root=root,
        schema=schema,
        features=features,
        fps=fps,
        image_key=image_key,
        image_shape=image_shape,
        mode_names=mode_names,
        joint_names=joint_names,
        hand_names=hand_names,
        has_hand_action=has_hand_action,
        has_neck_action=has_neck_action,
        episodes=tuple(episodes),
    )


def _normalized_quaternions(values: np.ndarray, *, label: str) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms < 1e-6):
        bad_frame = int(np.flatnonzero(norms[:, 0] < 1e-6)[0])
        raise ValueError(f"{label} contains a zero quaternion at frame {bad_frame}")
    return values / norms


def load_episode_review_data(
    dataset: RecordingDataset,
    episode: RecordingEpisode,
) -> EpisodeReviewData:
    """Load one episode and compute synchronized tracking metrics."""

    keys = [FRAME_INDEX_KEY, TIMESTAMP_KEY, STATE_KEY, MODE_KEY, ACTION_KEY]
    if dataset.has_hand_action:
        keys.extend((HAND_STATE_KEY, HAND_ACTION_KEY))
    if dataset.has_neck_action:
        keys.extend((NECK_STATE_KEY, NECK_ACTION_KEY))
    with h5py.File(episode.data_path, "r") as h5:
        arrays = {key: np.asarray(h5[key]) for key in keys}

    frame_index = arrays[FRAME_INDEX_KEY]
    timestamps = arrays[TIMESTAMP_KEY].astype(np.float64, copy=False)
    state = arrays[STATE_KEY].astype(np.float64, copy=False)
    mode = arrays[MODE_KEY].astype(np.int64, copy=False)
    action = arrays[ACTION_KEY].astype(np.float64, copy=False)
    hand_state = (
        arrays[HAND_STATE_KEY].astype(np.float64, copy=False)
        if dataset.has_hand_action
        else None
    )
    hand_action = (
        arrays[HAND_ACTION_KEY].astype(np.float64, copy=False)
        if dataset.has_hand_action
        else None
    )
    neck_state = (
        arrays[NECK_STATE_KEY].astype(np.float64, copy=False)
        if dataset.has_neck_action
        else None
    )
    neck_action = (
        arrays[NECK_ACTION_KEY].astype(np.float64, copy=False)
        if dataset.has_neck_action
        else None
    )

    expected_frame_index = np.arange(episode.frames, dtype=frame_index.dtype)
    if not np.array_equal(frame_index, expected_frame_index):
        raise ValueError(
            f"Recording episode {episode.episode_index} frame_index must be contiguous from 0"
        )
    numeric_arrays = {
        TIMESTAMP_KEY: timestamps,
        STATE_KEY: state,
        ACTION_KEY: action,
    }
    if hand_state is not None:
        numeric_arrays[HAND_STATE_KEY] = hand_state
    if hand_action is not None:
        numeric_arrays[HAND_ACTION_KEY] = hand_action
    if neck_state is not None:
        numeric_arrays[NECK_STATE_KEY] = neck_state
    if neck_action is not None:
        numeric_arrays[NECK_ACTION_KEY] = neck_action
    for key, values in numeric_arrays.items():
        if not np.isfinite(values).all():
            raise ValueError(
                f"Recording episode {episode.episode_index} dataset {key!r} contains NaN or Inf"
            )
    if episode.frames > 1 and np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(
            f"Recording episode {episode.episode_index} timestamps must be strictly increasing"
        )
    invalid_modes = sorted(set(int(value) for value in np.unique(mode)) - set(dataset.mode_names))
    if invalid_modes:
        raise ValueError(
            f"Recording episode {episode.episode_index} contains unknown mode codes {invalid_modes}"
        )

    actual_joint_pos = state[:, :NUM_JOINTS]
    reference_joint_pos = action[:, 7:]
    joint_error = actual_joint_pos - reference_joint_pos
    group_error = {
        name: np.sqrt(np.mean(np.square(joint_error[:, indices]), axis=1))
        for name, indices in JOINT_GROUPS
    }

    actual_quat = _normalized_quaternions(
        state[:, 58:62],
        label=f"episode {episode.episode_index} observation base quaternion",
    )
    reference_quat = _normalized_quaternions(
        action[:, 3:7],
        label=f"episode {episode.episode_index} reference root quaternion",
    )
    quat_dot = np.abs(np.sum(actual_quat * reference_quat, axis=1))
    root_orientation_error_rad = 2.0 * np.arccos(np.clip(quat_dot, 0.0, 1.0))

    max_frame, max_joint = np.unravel_index(
        int(np.argmax(np.abs(joint_error))),
        joint_error.shape,
    )
    return EpisodeReviewData(
        episode=episode,
        frame_index=frame_index,
        timestamps=timestamps,
        state=state,
        mode=mode,
        action=action,
        hand_state=hand_state,
        hand_action=hand_action,
        neck_state=neck_state,
        neck_action=neck_action,
        joint_error=joint_error,
        group_error=group_error,
        root_orientation_error_rad=root_orientation_error_rad,
        joint_rmse_rad=float(np.sqrt(np.mean(np.square(joint_error)))),
        root_orientation_rmse_rad=float(
            np.sqrt(np.mean(np.square(root_orientation_error_rad)))
        ),
        max_joint_error_rad=float(abs(joint_error[max_frame, max_joint])),
        max_joint_error_frame=int(max_frame),
        max_joint_error_name=dataset.joint_names[int(max_joint)],
    )


def aligned_qpos_pair(data: EpisodeReviewData, frame: int) -> tuple[np.ndarray, np.ndarray]:
    """Return actual/reference qpos with actual root position aligned to reference."""

    if frame < 0 or frame >= data.episode.frames:
        raise IndexError(f"Frame {frame} outside episode range [0, {data.episode.frames - 1}]")
    reference_qpos = data.action[frame].copy()
    reference_qpos[3:7] = _normalized_quaternions(
        reference_qpos[None, 3:7],
        label="reference root quaternion",
    )[0]
    actual_qpos = reference_qpos.copy()
    actual_qpos[3:7] = _normalized_quaternions(
        data.state[frame : frame + 1, 58:62],
        label="observation base quaternion",
    )[0]
    actual_qpos[7:] = data.state[frame, :NUM_JOINTS]
    return actual_qpos, reference_qpos


class RecordingVideoReader:
    def __init__(self, dataset: RecordingDataset, episode: RecordingEpisode) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "Recording review requires OpenCV; install with pip install -e '.[review]'"
            ) from exc
        self._cv2 = cv2
        self._episode = episode
        self._expected_shape = dataset.image_shape
        self._capture = cv2.VideoCapture(str(episode.video_path))
        if not self._capture.isOpened():
            raise RuntimeError(f"Cannot open recording video: {episode.video_path}")
        reported_frames = int(round(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        reported_fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        if reported_frames != episode.frames:
            self.close()
            raise ValueError(
                f"Recording episode {episode.episode_index} MP4 frame count {reported_frames} "
                f"!= manifest/HDF5 frame count {episode.frames}"
            )
        if abs(reported_fps - dataset.fps) > 0.1:
            self.close()
            raise ValueError(
                f"Recording episode {episode.episode_index} MP4 fps {reported_fps:g} "
                f"!= schema fps {dataset.fps}"
            )
        self._next_frame = 0

    def read_frame(self, frame: int) -> np.ndarray:
        if frame != self._next_frame:
            self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = self._capture.read()
        if not ok or bgr is None:
            raise RuntimeError(
                f"Failed to decode episode {self._episode.episode_index} video frame {frame}"
            )
        self._next_frame = frame + 1
        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
        if tuple(rgb.shape) != self._expected_shape:
            raise ValueError(
                f"Episode {self._episode.episode_index} video frame shape {rgb.shape} "
                f"!= schema shape {self._expected_shape}"
            )
        return rgb

    def close(self) -> None:
        if getattr(self, "_capture", None) is not None:
            self._capture.release()


class _PrefixedSceneApi:
    """Prefix Viser node names so two MuJoCo scenes can share one server."""

    def __init__(self, scene: Any, prefix: str) -> None:
        self._scene = scene
        self._prefix = prefix

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._scene, name)
        if callable(attribute) and name.startswith("add_"):
            return lambda path, *args, **kwargs: attribute(
                self._prefix + str(path),
                *args,
                **kwargs,
            )
        return attribute


class _PrefixedViserServer:
    def __init__(self, server: Any, prefix: str) -> None:
        self._server = server
        self.scene = _PrefixedSceneApi(server.scene, prefix)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._server, name)


class RobotOverlayScene:
    """Show observed G1 geometry with a translucent green reference overlay."""

    def __init__(self, server: Any, xml_path: Path) -> None:
        try:
            import mujoco
            from mjviser import ViserMujocoScene
        except ImportError as exc:
            raise RuntimeError(
                "Recording review requires mjviser; install with pip install -e '.[review]'"
            ) from exc

        self._mujoco = mujoco
        self._actual_model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._reference_model = mujoco.MjModel.from_xml_path(str(xml_path))
        for model in (self._actual_model, self._reference_model):
            if model.nq != FULL_QPOS_DIM:
                raise ValueError(
                    f"Recording reviewer robot XML nq={model.nq} != action dim "
                    f"{FULL_QPOS_DIM}: {xml_path}"
                )

        reference_color = np.array([0.1, 0.95, 0.2], dtype=np.float32)
        self._reference_model.geom_rgba[:, :3] = reference_color
        visible_geoms = self._reference_model.geom_rgba[:, 3] > 0.0
        self._reference_model.geom_rgba[visible_geoms, 3] = 0.38
        world_geoms = self._reference_model.geom_bodyid == 0
        self._reference_model.geom_rgba[world_geoms, 3] = 0.0
        if self._reference_model.nmat > 0:
            self._reference_model.mat_rgba[:, :3] = reference_color
            visible_materials = self._reference_model.mat_rgba[:, 3] > 0.0
            self._reference_model.mat_rgba[visible_materials, 3] = 0.38
            world_materials = np.unique(self._reference_model.geom_matid[world_geoms])
            world_materials = world_materials[world_materials >= 0]
            self._reference_model.mat_rgba[world_materials, 3] = 0.0

        self._actual_root = server.scene.add_frame("/actual", show_axes=False)
        self._reference_root = server.scene.add_frame("/reference", show_axes=False)
        self._actual_scene = ViserMujocoScene(
            _PrefixedViserServer(server, "/actual"),
            self._actual_model,
            num_envs=1,
        )
        self._reference_scene = ViserMujocoScene(
            _PrefixedViserServer(server, "/reference"),
            self._reference_model,
            num_envs=1,
        )
        self._actual_data = mujoco.MjData(self._actual_model)
        self._reference_data = mujoco.MjData(self._reference_model)

    def update(
        self,
        actual_qpos: np.ndarray,
        reference_qpos: np.ndarray,
        *,
        show_reference: bool,
    ) -> None:
        self._actual_data.qpos[:] = actual_qpos
        self._reference_data.qpos[:] = reference_qpos
        self._mujoco.mj_forward(self._actual_model, self._actual_data)
        self._mujoco.mj_forward(self._reference_model, self._reference_data)
        self._actual_scene.update_from_mjdata(self._actual_data)
        self._reference_root.visible = show_reference
        if show_reference:
            self._reference_scene.update_from_mjdata(self._reference_data)

    def close(self) -> None:
        self._actual_root.remove()
        self._reference_root.remove()


class RecordingReviewerApp:
    def __init__(
        self,
        *,
        dataset: RecordingDataset,
        xml_path: Path,
        port: int,
        initial_episode: int,
    ) -> None:
        try:
            import viser
        except ImportError as exc:
            raise RuntimeError(
                "Recording review requires Viser; install with pip install -e '.[review]'"
            ) from exc

        self._dataset = dataset
        self._episode_position = initial_episode
        self._data = load_episode_review_data(dataset, dataset.episodes[initial_episode])
        self._video = RecordingVideoReader(dataset, self._data.episode)
        self._server = viser.ViserServer(port=port, label="Recording Reviewer")
        self._server.gui.configure_theme(
            control_layout="fixed",
            control_width="large",
            dark_mode=True,
            show_share_button=False,
            brand_color=(34, 197, 94),
        )
        self._server.scene.world_axes.visible = False
        self._server.initial_camera.position = (2.4, -2.4, 1.8)
        self._server.initial_camera.look_at = (0.0, 0.0, 0.8)
        self._server.initial_camera.up = (0.0, 0.0, 1.0)
        self._server.initial_camera.fov = 45.0
        try:
            self._robot_scene = RobotOverlayScene(self._server, xml_path)
        except Exception:
            self._video.close()
            self._server.stop()
            raise

        self._playing = False
        self._speed = 1.0
        self._frame_accumulator = 0.0
        self._current_frame = -1
        self._show_reference = True
        self._pending_actions: list[str] = []
        self._pending_episode: int | None = None
        self._pending_scrub: int | None = None
        self._pending_joint: str | None = None
        self._pending_hand: str | None = None

        self._joint_chart: Any | None = None
        self._group_chart: Any | None = None
        self._mode_chart: Any | None = None
        self._hand_chart: Any | None = None
        self._neck_chart: Any | None = None
        self._setup_gui(self._video.read_frame(0))
        self._set_frame(0, force=True)

    def _setup_gui(self, first_camera_frame: np.ndarray) -> None:
        gui = self._server.gui
        episode_labels = [episode.label(self._dataset.fps) for episode in self._dataset.episodes]

        with gui.add_folder("Camera", order=0):
            self._camera_image = gui.add_image(
                first_camera_frame,
                label="D435i RGB",
                format="jpeg",
                jpeg_quality=82,
            )
            gui.add_markdown(
                "The main view is interactive 3D: the recorded robot state uses its normal "
                "appearance and the motion-tracker reference is translucent green."
            )
            self._current_html = gui.add_html("")

        with gui.add_folder("Episode", order=1):
            self._episode_dropdown = gui.add_dropdown(
                "Episode",
                options=episode_labels,
                initial_value=episode_labels[self._episode_position],
            )
            self._summary_html = gui.add_html("")

            @self._episode_dropdown.on_update
            def _(_) -> None:
                selected = episode_labels.index(self._episode_dropdown.value)
                if selected != self._episode_position:
                    self._pending_episode = selected

        with gui.add_folder("Playback", order=2):
            self._play_button = gui.add_button("Play", color="green")
            self._frame_slider = gui.add_slider(
                "Frame",
                min=0,
                max=max(0, self._data.episode.frames - 1),
                step=1,
                initial_value=0,
            )
            self._speed_group = gui.add_button_group(
                "Speed",
                options=["0.25x", "0.5x", "1x", "2x"],
            )
            self._speed_group.value = "1x"
            self._restart_button = gui.add_button("Restart")
            self._prev_button = gui.add_button("Previous episode")
            self._next_button = gui.add_button("Next episode")
            self._reference_checkbox = gui.add_checkbox(
                "Show green reference",
                initial_value=True,
            )
            @self._play_button.on_click
            def _(_) -> None:
                self._pending_actions.append("toggle_play")

            @self._frame_slider.on_update
            def _(_) -> None:
                requested = int(self._frame_slider.value)
                if requested != self._current_frame:
                    self._pending_scrub = requested

            @self._speed_group.on_click
            def _(event) -> None:
                self._speed = {
                    "0.25x": 0.25,
                    "0.5x": 0.5,
                    "1x": 1.0,
                    "2x": 2.0,
                }.get(str(event.target.value), 1.0)

            @self._restart_button.on_click
            def _(_) -> None:
                self._pending_actions.append("restart")

            @self._prev_button.on_click
            def _(_) -> None:
                self._pending_actions.append("previous")

            @self._next_button.on_click
            def _(_) -> None:
                self._pending_actions.append("next")

            @self._reference_checkbox.on_update
            def _(_) -> None:
                self._show_reference = bool(self._reference_checkbox.value)
                self._pending_actions.append("refresh")

        self._tracking_folder = gui.add_folder("Tracking", order=3)
        with self._tracking_folder:
            self._joint_dropdown = gui.add_dropdown(
                "Joint",
                options=list(self._dataset.joint_names),
                initial_value=self._dataset.joint_names[0],
            )
            gui.add_markdown(
                "The observed robot uses the reference root position because the recording "
                "does not contain measured root XYZ. Joint and root-orientation comparisons remain valid."
            )

            @self._joint_dropdown.on_update
            def _(_) -> None:
                self._pending_joint = str(self._joint_dropdown.value)

        self._signals_folder = gui.add_folder("Recorded signals", order=4, expand_by_default=False)
        with self._signals_folder:
            gui.add_markdown("Mode: `0 standing`, `1 mocap`, `2 arms`, `3 pause`.")
            if self._dataset.has_hand_action:
                self._hand_dropdown = gui.add_dropdown(
                    "Hand channel",
                    options=list(self._dataset.hand_names),
                    initial_value=self._dataset.hand_names[0],
                )

                @self._hand_dropdown.on_update
                def _(_) -> None:
                    self._pending_hand = str(self._hand_dropdown.value)
            else:
                self._hand_dropdown = None
                gui.add_markdown("This dataset does not contain `action.hand`.")
            if not self._dataset.has_neck_action:
                gui.add_markdown("This dataset does not contain `action.neck`.")

        self._refresh_summary()
        self._refresh_charts()

    @staticmethod
    def _chart_axes(y_label: str) -> tuple[dict[str, Any], dict[str, Any]]:
        return (
            {"label": "time (s)", "stroke": "#9ca3af"},
            {"label": y_label, "stroke": "#9ca3af"},
        )

    def _add_chart(
        self,
        folder: Any,
        *,
        data: tuple[np.ndarray, ...],
        series: tuple[dict[str, Any], ...],
        title: str,
        y_label: str,
        order: float,
    ) -> Any:
        with folder:
            return self._server.gui.add_uplot(
                data=data,
                series=series,
                title=title,
                axes=self._chart_axes(y_label),
                legend={"show": True, "live": True},
                cursor={"show": True, "x": True, "y": True},
                height=220,
                order=order,
            )

    def _refresh_charts(self) -> None:
        for handle in (
            self._joint_chart,
            self._group_chart,
            self._mode_chart,
            self._hand_chart,
            self._neck_chart,
        ):
            if handle is not None:
                handle.remove()

        timestamps = self._data.timestamps.astype(np.float64, copy=False)
        selected_joint = str(self._joint_dropdown.value)
        joint_index = self._dataset.joint_names.index(selected_joint)
        actual = self._data.state[:, joint_index]
        reference = self._data.action[:, 7 + joint_index]
        error = self._data.joint_error[:, joint_index]
        self._joint_chart = self._add_chart(
            self._tracking_folder,
            data=(timestamps, actual, reference, error),
            series=(
                {"label": "time"},
                {"label": "actual", "stroke": "#3b82f6", "width": 2.0},
                {"label": "reference", "stroke": "#22c55e", "width": 2.0},
                {"label": "error", "stroke": "#ef4444", "width": 1.5},
            ),
            title=f"Joint: {selected_joint}",
            y_label="rad",
            order=1,
        )
        group_values = tuple(self._data.group_error[name] for name, _ in JOINT_GROUPS)
        group_series: tuple[dict[str, Any], ...] = (
            {"label": "time"},
            *tuple(
                {"label": name, "stroke": color, "width": 1.6}
                for (name, _), color in zip(JOINT_GROUPS, GROUP_COLORS, strict=True)
            ),
        )
        self._group_chart = self._add_chart(
            self._tracking_folder,
            data=(timestamps, *group_values),
            series=group_series,
            title="Instantaneous group joint RMSE",
            y_label="rad",
            order=2,
        )
        self._mode_chart = self._add_chart(
            self._signals_folder,
            data=(timestamps, self._data.mode.astype(np.float64)),
            series=(
                {"label": "time"},
                {"label": "mode", "stroke": "#f59e0b", "width": 2.0},
            ),
            title="Mode timeline",
            y_label="mode code",
            order=1,
        )

        if (
            self._data.hand_state is not None
            and self._data.hand_action is not None
            and self._hand_dropdown is not None
        ):
            selected_hand = str(self._hand_dropdown.value)
            hand_index = self._dataset.hand_names.index(selected_hand)
            self._hand_chart = self._add_chart(
                self._signals_folder,
                data=(
                    timestamps,
                    self._data.hand_state[:, hand_index],
                    self._data.hand_action[:, hand_index],
                ),
                series=(
                    {"label": "time"},
                    {"label": "state", "stroke": "#3b82f6", "width": 2.0},
                    {"label": "target", "stroke": "#8b5cf6", "width": 2.0},
                ),
                title=f"LinkerHand state vs target: {selected_hand}",
                y_label="SDK pose",
                order=2,
            )
        else:
            self._hand_chart = None

        if self._data.neck_state is not None and self._data.neck_action is not None:
            self._neck_chart = self._add_chart(
                self._signals_folder,
                data=(
                    timestamps,
                    self._data.neck_state[:, 0],
                    self._data.neck_action[:, 0],
                    self._data.neck_state[:, 1],
                    self._data.neck_action[:, 1],
                ),
                series=(
                    {"label": "time"},
                    {"label": "yaw state", "stroke": "#3b82f6", "width": 2.0},
                    {"label": "yaw target", "stroke": "#06b6d4", "width": 2.0},
                    {"label": "pitch state", "stroke": "#f59e0b", "width": 2.0},
                    {"label": "pitch target", "stroke": "#ec4899", "width": 2.0},
                ),
                title="OpenNeck state vs target",
                y_label="degrees",
                order=3,
            )
        else:
            self._neck_chart = None

    def _mode_name(self, frame: int) -> str:
        return self._dataset.mode_names[int(self._data.mode[frame])]

    def _set_frame(self, frame: int, *, force: bool = False) -> None:
        frame = max(0, min(int(frame), self._data.episode.frames - 1))
        if not force and frame == self._current_frame:
            return
        self._camera_image.image = self._video.read_frame(frame)
        actual_qpos, reference_qpos = aligned_qpos_pair(self._data, frame)
        self._robot_scene.update(
            actual_qpos,
            reference_qpos,
            show_reference=self._show_reference,
        )
        self._current_frame = frame
        self._frame_slider.value = frame
        instant_rmse = float(np.sqrt(np.mean(np.square(self._data.joint_error[frame]))))
        selected_joint = str(self._joint_dropdown.value)
        joint_index = self._dataset.joint_names.index(selected_joint)
        selected_error = float(self._data.joint_error[frame, joint_index])
        self._current_html.content = (
            "<div style='line-height:1.45'>"
            f"<strong>Frame:</strong> {frame}/{self._data.episode.frames - 1}<br/>"
            f"<strong>Time:</strong> {self._data.timestamps[frame]:.2f}s<br/>"
            f"<strong>Mode:</strong> {html.escape(self._mode_name(frame))}<br/>"
            f"<strong>Instant joint RMSE:</strong> {instant_rmse:.3f} rad<br/>"
            f"<strong>{html.escape(selected_joint)} error:</strong> {selected_error:+.3f} rad"
            "</div>"
        )

    def _refresh_summary(self) -> None:
        episode = self._data.episode
        mode_counts = [
            f"{html.escape(name)}={int(np.count_nonzero(self._data.mode == code))}"
            for code, name in sorted(self._dataset.mode_names.items())
            if np.any(self._data.mode == code)
        ]
        self._summary_html.content = (
            "<div style='line-height:1.45'>"
            f"<strong>Task:</strong> {html.escape(episode.task)}<br/>"
            f"<strong>Frames:</strong> {episode.frames} @ {self._dataset.fps} FPS "
            f"({episode.frames / self._dataset.fps:.2f}s)<br/>"
            f"<strong>Modes:</strong> {', '.join(mode_counts)}<br/>"
            f"<strong>Joint RMSE:</strong> {self._data.joint_rmse_rad:.3f} rad "
            f"({np.degrees(self._data.joint_rmse_rad):.2f}°)<br/>"
            f"<strong>Root orientation RMSE:</strong> "
            f"{np.degrees(self._data.root_orientation_rmse_rad):.2f}°<br/>"
            f"<strong>Max joint error:</strong> {self._data.max_joint_error_rad:.3f} rad "
            f"at frame {self._data.max_joint_error_frame} "
            f"({html.escape(self._data.max_joint_error_name)})"
            "</div>"
        )

    def _set_playing(self, playing: bool) -> None:
        self._playing = playing
        self._frame_accumulator = 0.0
        self._play_button.label = "Pause" if playing else "Play"
        self._play_button.color = "red" if playing else "green"

    def _load_episode(self, position: int) -> None:
        position = max(0, min(position, len(self._dataset.episodes) - 1))
        if position == self._episode_position:
            return
        self._set_playing(False)
        new_data = load_episode_review_data(
            self._dataset,
            self._dataset.episodes[position],
        )
        new_video = RecordingVideoReader(self._dataset, new_data.episode)
        old_video = self._video
        self._data = new_data
        self._video = new_video
        self._episode_position = position
        old_video.close()
        self._current_frame = -1
        self._frame_slider.max = max(0, new_data.episode.frames - 1)
        self._episode_dropdown.value = new_data.episode.label(self._dataset.fps)
        self._refresh_summary()
        self._refresh_charts()
        self._set_frame(0, force=True)

    def _process_pending(self) -> None:
        if self._pending_episode is not None:
            position = self._pending_episode
            self._pending_episode = None
            self._load_episode(position)

        if self._pending_joint is not None:
            self._pending_joint = None
            self._refresh_charts()
            self._set_frame(self._current_frame, force=True)
        if self._pending_hand is not None:
            self._pending_hand = None
            self._refresh_charts()

        if self._pending_scrub is not None:
            frame = self._pending_scrub
            self._pending_scrub = None
            self._set_playing(False)
            self._set_frame(frame)

        while self._pending_actions:
            action = self._pending_actions.pop(0)
            if action == "toggle_play":
                if self._current_frame >= self._data.episode.frames - 1:
                    self._set_frame(0)
                self._set_playing(not self._playing)
            elif action == "restart":
                self._set_playing(False)
                self._set_frame(0, force=True)
            elif action == "previous":
                self._load_episode(self._episode_position - 1)
            elif action == "next":
                self._load_episode(self._episode_position + 1)
            elif action == "refresh":
                self._set_frame(self._current_frame, force=True)

    def run(self) -> None:
        print(f"\nRecording reviewer ready at http://localhost:{self._server.get_port()}")
        print(f"Dataset: {self._dataset.root}")
        print("Press Ctrl+C to exit.\n")
        previous_time = time.monotonic()
        try:
            while True:
                now = time.monotonic()
                elapsed = now - previous_time
                previous_time = now
                self._process_pending()
                if self._playing:
                    self._frame_accumulator += elapsed * self._dataset.fps * self._speed
                    advance = int(self._frame_accumulator)
                    if advance > 0:
                        self._frame_accumulator -= advance
                        next_frame = self._current_frame + advance
                        if next_frame >= self._data.episode.frames - 1:
                            self._set_frame(self._data.episode.frames - 1)
                            self._set_playing(False)
                        else:
                            self._set_frame(next_frame)
                time.sleep(1.0 / 60.0)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.close()

    def close(self) -> None:
        self._video.close()
        self._robot_scene.close()
        self._server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only synchronized reviewer for Teleopit sim2real recordings"
    )
    parser.add_argument(
        "--recording",
        type=str,
        default=str(DEFAULT_RECORDING_ROOT),
        help="Recording dataset root containing schema.json and episodes.jsonl",
    )
    parser.add_argument("--xml", type=str, default=None, help="Canonical G1 MuJoCo XML path")
    parser.add_argument("--episode", type=int, default=0, help="Initial episode index")
    parser.add_argument("--port", type=int, default=8013, help="Viser server port")
    args = parser.parse_args()

    recording_root = Path(args.recording)
    if not recording_root.is_absolute():
        recording_root = (PROJECT_ROOT / recording_root).resolve()
    try:
        dataset = load_recording_dataset(recording_root)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    episode_positions = {
        episode.episode_index: position for position, episode in enumerate(dataset.episodes)
    }
    if args.episode not in episode_positions:
        valid = [episode.episode_index for episode in dataset.episodes]
        print(f"ERROR: episode {args.episode} not found; available indices: {valid}", file=sys.stderr)
        raise SystemExit(1)

    xml_path = Path(args.xml).expanduser() if args.xml else DEFAULT_XML
    if not xml_path.is_absolute():
        xml_path = (PROJECT_ROOT / xml_path).resolve()
    if not xml_path.is_file():
        print(
            "ERROR: " + missing_gmr_assets_message(xml_path, label="Robot XML"),
            file=sys.stderr,
        )
        raise SystemExit(1)

    try:
        app = RecordingReviewerApp(
            dataset=dataset,
            xml_path=xml_path,
            port=args.port,
            initial_episode=episode_positions[args.episode],
        )
    except (ImportError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    app.run()


if __name__ == "__main__":
    main()
