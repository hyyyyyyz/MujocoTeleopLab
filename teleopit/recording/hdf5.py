"""Editable HDF5 dataset writer for Teleopit sim2real recording."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
from typing import Any

import h5py
import numpy as np

from teleopit.constants import FULL_QPOS_DIM, G1_JOINT_NAMES, NUM_JOINTS
from teleopit.controllers.observation import _quat_rotate_np
from teleopit.math_utils import quat_inv_np
from teleopit.runtime.common import cfg_get
from teleopit.sim2real.hands.linkerhand_l6 import L6_SDK_JOINT_ORDER
from teleopit.sim2real.hands.linkerhand_o6 import O6_SDK_JOINT_ORDER


logger = logging.getLogger(__name__)

IMAGE_KEY = "observation.images.d435i_rgb"
STATE_KEY = "observation.state"
HAND_STATE_KEY = "observation.state.hand"
NECK_STATE_KEY = "observation.state.neck"
MODE_KEY = "observation.mode"
ACTION_KEY = "action"
HAND_ACTION_KEY = "action.hand"
NECK_ACTION_KEY = "action.neck"
FRAME_INDEX_KEY = "frame_index"
TIMESTAMP_KEY = "timestamp"
STATE_DIM = 68
HAND_STATE_DIM = 12
NECK_STATE_DIM = 2
ACTION_DIM = FULL_QPOS_DIM
HAND_ACTION_DIM = 12
NECK_ACTION_DIM = 2
DEFAULT_IMAGE_SHAPE = (480, 640, 3)
HDF5_RECORDING_FORMAT = "teleopit_hdf5"
HDF5_RECORDING_VERSION = 4
DEFAULT_ROBOT_TYPE = "unitree_g1_29dof"
NO_HAND_TYPE = "none"
SUPPORTED_HAND_TYPES = (NO_HAND_TYPE, "linkerhand_l6", "linkerhand_o6")
NO_NECK_TYPE = "none"
SUPPORTED_NECK_TYPES = (NO_NECK_TYPE, "openneck")
MODE_CODES = {
    "standing": 0,
    "mocap": 1,
    "arms": 2,
    "pause": 3,
}
_EPISODE_HDF5_FILENAME = re.compile(r"episode_\d{6,}\.h5")
_EPISODE_MP4_FILENAME = re.compile(r"episode_\d{6,}\.mp4")


@dataclass(frozen=True)
class RecordingSchema:
    fps: int
    robot_type: str
    hand_type: str
    image_key: str
    image_shape: tuple[int, int, int]
    neck_type: str = NO_NECK_TYPE
    state_key: str = STATE_KEY
    state_dim: int = STATE_DIM
    hand_state_key: str = HAND_STATE_KEY
    hand_state_dim: int = HAND_STATE_DIM
    neck_state_key: str = NECK_STATE_KEY
    neck_state_dim: int = NECK_STATE_DIM
    mode_key: str = MODE_KEY
    action_key: str = ACTION_KEY
    action_dim: int = ACTION_DIM
    hand_action_key: str = HAND_ACTION_KEY
    hand_action_dim: int = HAND_ACTION_DIM
    neck_action_key: str = NECK_ACTION_KEY
    neck_action_dim: int = NECK_ACTION_DIM

    @property
    def has_hand_action(self) -> bool:
        return self.hand_type != NO_HAND_TYPE

    @property
    def has_neck_action(self) -> bool:
        return self.neck_type != NO_NECK_TYPE


@dataclass(frozen=True)
class MP4VideoConfig:
    codec: str = "libx264"
    quality: int = 8
    pixelformat: str = "yuv420p"


def build_recording_schema(
    camera_cfg: Any,
    *,
    fps: int = 30,
    robot_type: str = DEFAULT_ROBOT_TYPE,
    hand_type: str = NO_HAND_TYPE,
    neck_type: str = NO_NECK_TYPE,
) -> RecordingSchema:
    key = str(cfg_get(camera_cfg, "key", IMAGE_KEY)).strip()
    width = int(cfg_get(camera_cfg, "width", DEFAULT_IMAGE_SHAPE[1]))
    height = int(cfg_get(camera_cfg, "height", DEFAULT_IMAGE_SHAPE[0]))
    parsed_fps = int(fps)
    parsed_robot_type = str(robot_type).strip().lower()
    parsed_hand_type = str(hand_type).strip().lower()
    parsed_neck_type = str(neck_type).strip().lower()
    if not key:
        raise ValueError("recording.camera.key must not be empty")
    if width <= 0 or height <= 0:
        raise ValueError("recording.camera.width and recording.camera.height must be positive")
    if parsed_fps <= 0:
        raise ValueError("recording.fps must be positive")
    if parsed_robot_type != DEFAULT_ROBOT_TYPE:
        raise ValueError(
            f"Unsupported recording robot_type={parsed_robot_type!r}; expected {DEFAULT_ROBOT_TYPE!r}"
        )
    if parsed_hand_type not in SUPPORTED_HAND_TYPES:
        raise ValueError(
            f"Unsupported recording hand_type={parsed_hand_type!r}; expected one of {SUPPORTED_HAND_TYPES}"
        )
    if parsed_neck_type not in SUPPORTED_NECK_TYPES:
        raise ValueError(
            f"Unsupported recording neck_type={parsed_neck_type!r}; expected one of {SUPPORTED_NECK_TYPES}"
        )
    return RecordingSchema(
        fps=parsed_fps,
        robot_type=parsed_robot_type,
        hand_type=parsed_hand_type,
        neck_type=parsed_neck_type,
        image_key=key,
        image_shape=(height, width, 3),
    )


def build_mp4_video_config(video_cfg: Any) -> MP4VideoConfig:
    quality = int(cfg_get(video_cfg, "quality", 8))
    if quality < 0 or quality > 10:
        raise ValueError("recording.video.quality must be in [0, 10]")
    return MP4VideoConfig(
        codec=str(cfg_get(video_cfg, "codec", "libx264")),
        quality=quality,
        pixelformat=str(cfg_get(video_cfg, "pixelformat", "yuv420p")),
    )


def hdf5_schema(schema: RecordingSchema) -> dict[str, object]:
    features: dict[str, object] = {
        FRAME_INDEX_KEY: {
            "dtype": "int64",
            "shape": [],
        },
        TIMESTAMP_KEY: {
            "dtype": "float64",
            "shape": [],
            "units": "seconds",
        },
        schema.state_key: {
            "dtype": "float32",
            "shape": [schema.state_dim],
            "names": _state_names(),
            "groups": {
                "joint_pos": [0, 29],
                "joint_vel": [29, 58],
                "base_quat_wxyz": [58, 62],
                "base_ang_vel": [62, 65],
                "projected_gravity": [65, 68],
            },
        },
        schema.mode_key: {
            "dtype": "int8",
            "shape": [],
            "values": MODE_CODES,
        },
        schema.action_key: {
            "dtype": "float32",
            "shape": [schema.action_dim],
            "names": _reference_action_names(),
            "groups": {
                "root_pos": [0, 3],
                "root_quat_wxyz": [3, 7],
                "reference_joint_pos": [7, 36],
            },
        },
    }
    if schema.has_hand_action:
        features[schema.hand_state_key] = {
            "dtype": "float32",
            "shape": [schema.hand_state_dim],
            "names": _hand_action_names(schema.hand_type),
            "groups": {
                "left_hand_state": [0, 6],
                "right_hand_state": [6, 12],
            },
        }
        features[schema.hand_action_key] = {
            "dtype": "float32",
            "shape": [schema.hand_action_dim],
            "names": _hand_action_names(schema.hand_type),
            "groups": {
                "left_hand_target": [0, 6],
                "right_hand_target": [6, 12],
            },
        }
    if schema.has_neck_action:
        features[schema.neck_state_key] = {
            "dtype": "float32",
            "shape": [schema.neck_state_dim],
            "names": ["yaw_deg", "pitch_deg"],
            "units": "degrees",
        }
        features[schema.neck_action_key] = {
            "dtype": "float32",
            "shape": [schema.neck_action_dim],
            "names": ["yaw_deg", "pitch_deg"],
            "units": "degrees",
        }
    features[schema.image_key] = {
        "dtype": "video",
        "shape": list(schema.image_shape),
        "names": ["height", "width", "channel"],
    }
    return {
        "format": HDF5_RECORDING_FORMAT,
        "version": HDF5_RECORDING_VERSION,
        "fps": schema.fps,
        "robot_type": schema.robot_type,
        "hand_type": schema.hand_type,
        "neck_type": schema.neck_type,
        "features": features,
    }


def build_observation_state(robot_state: object) -> np.ndarray:
    joint_pos = np.asarray(getattr(robot_state, "qpos"), dtype=np.float32).reshape(-1)[:NUM_JOINTS]
    joint_vel = np.asarray(getattr(robot_state, "qvel"), dtype=np.float32).reshape(-1)[:NUM_JOINTS]
    base_quat = np.asarray(getattr(robot_state, "quat"), dtype=np.float32).reshape(-1)[:4]
    base_ang_vel = np.asarray(getattr(robot_state, "ang_vel"), dtype=np.float32).reshape(-1)[:3]
    if joint_pos.shape[0] != NUM_JOINTS:
        raise ValueError(f"robot_state.qpos must contain {NUM_JOINTS} joints, got {joint_pos.shape[0]}")
    if joint_vel.shape[0] != NUM_JOINTS:
        raise ValueError(f"robot_state.qvel must contain {NUM_JOINTS} joints, got {joint_vel.shape[0]}")
    if base_quat.shape[0] != 4:
        raise ValueError(f"robot_state.quat must be 4D (wxyz), got {base_quat.shape[0]}")
    if base_ang_vel.shape[0] != 3:
        raise ValueError(f"robot_state.ang_vel must be 3D, got {base_ang_vel.shape[0]}")
    gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    projected_gravity = _quat_rotate_np(quat_inv_np(base_quat), gravity_w)
    state = np.concatenate(
        [joint_pos, joint_vel, base_quat, base_ang_vel, projected_gravity],
        dtype=np.float32,
    )
    if state.shape[0] != STATE_DIM:
        raise ValueError(f"recording observation.state must be {STATE_DIM}D, got {state.shape[0]}")
    return state


def normalize_action_reference_qpos(reference_qpos: object) -> np.ndarray:
    action = np.asarray(reference_qpos, dtype=np.float32).reshape(-1)
    if action.shape[0] != ACTION_DIM:
        raise ValueError(f"recording action reference qpos must be {ACTION_DIM}D, got {action.shape[0]}")
    return action


def normalize_hand_action(left_pose: object, right_pose: object) -> np.ndarray:
    left = np.asarray(left_pose, dtype=np.float32).reshape(-1)
    right = np.asarray(right_pose, dtype=np.float32).reshape(-1)
    if left.shape[0] != 6:
        raise ValueError(f"recording left hand pose must be 6D, got {left.shape[0]}")
    if right.shape[0] != 6:
        raise ValueError(f"recording right hand pose must be 6D, got {right.shape[0]}")
    action = np.concatenate([left, right], dtype=np.float32)
    if action.shape[0] != HAND_ACTION_DIM:
        raise ValueError(f"recording action.hand must be {HAND_ACTION_DIM}D, got {action.shape[0]}")
    return action


def build_neck_action(yaw_deg: object, pitch_deg: object) -> np.ndarray:
    action = np.asarray([yaw_deg, pitch_deg], dtype=np.float32).reshape(-1)
    if action.shape[0] != NECK_ACTION_DIM:
        raise ValueError(f"recording action.neck must be {NECK_ACTION_DIM}D, got {action.shape[0]}")
    return action


def build_mode_observation(mode: str) -> np.int8:
    normalized = str(mode).strip().lower()
    if normalized not in MODE_CODES:
        raise ValueError(f"Unsupported recording mode {mode!r}; expected one of {sorted(MODE_CODES)}")
    return np.int8(MODE_CODES[normalized])


class TeleopitHDF5Recorder:
    """Writes editable per-episode HDF5 and MP4 files plus a JSONL manifest."""

    def __init__(
        self,
        *,
        output_dir: Path,
        task: str,
        schema: RecordingSchema,
        video_config: MP4VideoConfig | None = None,
    ) -> None:
        self._output_dir = output_dir
        self._task = str(task).strip()
        self._schema = schema
        self._fps = schema.fps
        self._video_config = video_config or MP4VideoConfig()
        self._active = False
        self._frames_in_episode = 0
        self._next_episode_index = 0
        self._active_episode_index: int | None = None
        self._h5: h5py.File | None = None
        self._tmp_path: Path | None = None
        self._episode_path: Path | None = None
        self._tmp_video_path: Path | None = None
        self._episode_video_path: Path | None = None
        self._data_rel_path: str | None = None
        self._video_rel_path: str | None = None
        self._video_writer: Any | None = None
        self._datasets: dict[str, h5py.Dataset] = {}
        if not self._task:
            raise ValueError("recording.task must not be empty")

    @classmethod
    def create(
        cls,
        *,
        output_dir: str | Path,
        task: str,
        schema: RecordingSchema,
        video_config: MP4VideoConfig | None = None,
    ) -> "TeleopitHDF5Recorder":
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        recorder = cls(
            output_dir=root,
            task=task,
            schema=schema,
            video_config=video_config,
        )
        recorder._initialize_dataset()
        return recorder

    def start_episode(self) -> None:
        if self._active:
            raise RuntimeError("Cannot start a new recording episode while one is active")
        episode_index = self._next_episode_index
        stem = f"episode_{episode_index:06d}"
        data_dir = self._output_dir / "data"
        video_storage_key = _video_storage_key(self._schema.image_key)
        video_dir = self._output_dir / "videos" / video_storage_key
        tmp_data_dir = self._output_dir / ".tmp" / "data"
        tmp_video_dir = self._output_dir / ".tmp" / "videos" / video_storage_key
        for path in (data_dir, video_dir, tmp_data_dir, tmp_video_dir):
            path.mkdir(parents=True, exist_ok=True)

        self._tmp_path = tmp_data_dir / f"{stem}.h5"
        self._episode_path = data_dir / f"{stem}.h5"
        self._tmp_video_path = tmp_video_dir / f"{stem}.mp4"
        self._episode_video_path = video_dir / f"{stem}.mp4"
        self._data_rel_path = self._episode_path.relative_to(self._output_dir).as_posix()
        self._video_rel_path = self._episode_video_path.relative_to(self._output_dir).as_posix()
        self._active_episode_index = episode_index
        if self._episode_path.exists() or self._episode_video_path.exists():
            self._reset_episode()
            raise FileExistsError(f"Recording episode {stem} already exists")

        try:
            self._h5 = h5py.File(self._tmp_path, "w")
            self._video_writer = self._create_video_writer(self._tmp_video_path)
            self._datasets = self._create_datasets(self._h5)
            self._active = True
            self._frames_in_episode = 0
        except Exception:
            self._cleanup_partial_episode()
            raise

    def add_frame(
        self,
        *,
        image: np.ndarray,
        state: np.ndarray,
        mode: object,
        action: np.ndarray,
        hand_state: np.ndarray | None = None,
        neck_state: np.ndarray | None = None,
        hand_action: np.ndarray | None = None,
        neck_action: np.ndarray | None = None,
    ) -> None:
        if not self._active or self._h5 is None:
            raise RuntimeError("Cannot add a recording frame without an active episode")
        image_arr = np.asarray(image, dtype=np.uint8)
        if tuple(image_arr.shape) != self._schema.image_shape:
            raise ValueError(f"{self._schema.image_key} frame shape {image_arr.shape} != {self._schema.image_shape}")
        state_arr = self._validate_vector(state, self._schema.state_key, self._schema.state_dim)
        mode_value = self._validate_mode(mode)
        action_arr = self._validate_vector(action, self._schema.action_key, self._schema.action_dim)
        optional_vectors: dict[str, np.ndarray] = {}
        for enabled, value, key, dim in (
            (self._schema.has_hand_action, hand_state, self._schema.hand_state_key, self._schema.hand_state_dim),
            (self._schema.has_hand_action, hand_action, self._schema.hand_action_key, self._schema.hand_action_dim),
            (self._schema.has_neck_action, neck_state, self._schema.neck_state_key, self._schema.neck_state_dim),
            (self._schema.has_neck_action, neck_action, self._schema.neck_action_key, self._schema.neck_action_dim),
        ):
            if enabled and value is None:
                raise ValueError(f"{key} is required when its device is enabled")
            if not enabled and value is not None:
                raise ValueError(f"{key} must be omitted when its device is disabled")
            if value is not None:
                optional_vectors[key] = self._validate_vector(value, key, dim)

        row = self._frames_in_episode
        for dataset in self._datasets.values():
            dataset.resize((row + 1, *dataset.shape[1:]))
        if self._video_writer is None:
            raise RuntimeError("MP4 recording writer is not open")
        self._video_writer.append_data(image_arr)
        self._datasets[FRAME_INDEX_KEY][row] = row
        self._datasets[TIMESTAMP_KEY][row] = float(row) / float(self._fps)
        self._datasets[self._schema.state_key][row] = state_arr
        self._datasets[self._schema.mode_key][row] = mode_value
        self._datasets[self._schema.action_key][row] = action_arr
        for key, value in optional_vectors.items():
            self._datasets[key][row] = value
        self._frames_in_episode += 1

    def save_episode(self) -> None:
        if not self._active:
            return
        tmp_path = self._require_tmp_path()
        episode_path = self._require_episode_path()
        tmp_video_path = self._tmp_video_path
        episode_video_path = self._episode_video_path
        episode_index = self._active_episode_index
        data_rel_path = self._data_rel_path
        video_rel_path = self._video_rel_path
        frames = self._frames_in_episode
        self._close_active_outputs()
        if (
            tmp_video_path is None
            or episode_video_path is None
            or episode_index is None
            or data_rel_path is None
            or video_rel_path is None
        ):
            raise RuntimeError("recording episode paths are incomplete")
        try:
            tmp_video_path.replace(episode_video_path)
            tmp_path.replace(episode_path)
            self._append_manifest_entry(
                {
                    "episode_index": episode_index,
                    "frames": frames,
                    "task": self._task,
                    "data": data_rel_path,
                    "videos": {self._schema.image_key: video_rel_path},
                }
            )
        except Exception:
            for path in (tmp_path, tmp_video_path, episode_path, episode_video_path):
                if path.exists():
                    path.unlink()
            self._reset_episode()
            raise
        self._next_episode_index += 1
        self._reset_episode()

    def discard_episode(self) -> None:
        if not self._active:
            return
        tmp_path = self._tmp_path
        tmp_video_path = self._tmp_video_path
        self._close_active_outputs()
        for path in (tmp_path, tmp_video_path):
            if path is not None and path.exists():
                path.unlink()
        self._reset_episode()

    def finalize(self) -> None:
        if self._active:
            self.discard_episode()

    def _initialize_dataset(self) -> None:
        schema_path = self._output_dir / "schema.json"
        expected_schema = self._schema_dict()
        if schema_path.exists():
            try:
                existing_schema = json.loads(schema_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid recording schema: {schema_path}") from exc
            if existing_schema != expected_schema:
                raise ValueError(
                    f"Recording schema mismatch at {schema_path}; use an empty output_dir for the new dataset"
                )
        else:
            if self._has_recorded_payload():
                raise ValueError(
                    f"Recording output {self._output_dir} contains data without schema.json; use an empty output_dir"
                )
            schema_path.write_text(
                json.dumps(expected_schema, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        manifest_path = self._manifest_path
        if not manifest_path.exists():
            if self._has_recorded_payload():
                raise ValueError(
                    f"Recording output {self._output_dir} contains data without episodes.jsonl; use an empty output_dir"
                )
            manifest_path.touch()
        entries = self._read_manifest_entries()
        self._discard_uncommitted_episode_files(entries)
        self._next_episode_index = len(entries)

    @property
    def _manifest_path(self) -> Path:
        return self._output_dir / "episodes.jsonl"

    def _has_recorded_payload(self) -> bool:
        for dirname in ("data", "videos", "episodes"):
            path = self._output_dir / dirname
            if path.exists() and any(item.is_file() for item in path.rglob("*")):
                return True
        return False

    def _read_manifest_entries(self) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        for line_number, raw_line in enumerate(
            self._manifest_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {self._manifest_path}:{line_number}") from exc
            if not isinstance(entry, dict):
                raise ValueError(f"Episode entry in {self._manifest_path}:{line_number} must be an object")
            expected_index = len(entries)
            if entry.get("episode_index") != expected_index:
                raise ValueError(
                    f"Episode indices in {self._manifest_path} must be contiguous from 0; "
                    f"line {line_number} expected {expected_index}, got {entry.get('episode_index')!r}"
                )
            data_path = entry.get("data")
            videos = entry.get("videos")
            if not isinstance(data_path, str) or not isinstance(videos, dict):
                raise ValueError(f"Episode entry in {self._manifest_path}:{line_number} has invalid paths")
            referenced_paths = [data_path, *[str(path) for path in videos.values()]]
            for relative_path in referenced_paths:
                if not (self._output_dir / relative_path).is_file():
                    raise ValueError(
                        f"Episode entry in {self._manifest_path}:{line_number} references missing file {relative_path!r}"
                    )
            entries.append(entry)
        return entries

    def _append_manifest_entry(self, entry: dict[str, object]) -> None:
        current = self._manifest_path.read_bytes()
        if current and not current.endswith(b"\n"):
            current += b"\n"
        encoded_entry = (
            json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        tmp_path = self._output_dir / ".tmp" / "episodes.jsonl"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tmp_path.open("wb") as handle:
                handle.write(current)
                handle.write(encoded_entry)
                handle.flush()
                os.fsync(handle.fileno())
            tmp_path.replace(self._manifest_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _discard_uncommitted_episode_files(self, entries: list[dict[str, object]]) -> None:
        committed_paths: set[Path] = set()
        for entry in entries:
            data_path = entry.get("data")
            videos = entry.get("videos")
            if isinstance(data_path, str):
                committed_paths.add((self._output_dir / data_path).resolve())
            if isinstance(videos, dict):
                committed_paths.update(
                    (self._output_dir / str(relative_path)).resolve()
                    for relative_path in videos.values()
                )

        video_storage_key = _video_storage_key(self._schema.image_key)
        final_candidates = [
            *self._matching_episode_files(
                self._output_dir / "data",
                _EPISODE_HDF5_FILENAME,
            ),
            *self._matching_episode_files(
                self._output_dir / "videos" / video_storage_key,
                _EPISODE_MP4_FILENAME,
            ),
        ]
        tmp_candidates = [
            *self._matching_episode_files(
                self._output_dir / ".tmp" / "data",
                _EPISODE_HDF5_FILENAME,
            ),
            *self._matching_episode_files(
                self._output_dir / ".tmp" / "videos" / video_storage_key,
                _EPISODE_MP4_FILENAME,
            ),
            self._output_dir / ".tmp" / "episodes.jsonl",
        ]
        for path in final_candidates:
            if path.resolve() not in committed_paths:
                self._discard_interrupted_artifact(path)
        for path in tmp_candidates:
            if path.is_file():
                self._discard_interrupted_artifact(path)

    @staticmethod
    def _matching_episode_files(directory: Path, pattern: re.Pattern[str]) -> list[Path]:
        if not directory.is_dir():
            return []
        return [
            path
            for path in directory.iterdir()
            if path.is_file() and pattern.fullmatch(path.name)
        ]

    @staticmethod
    def _discard_interrupted_artifact(path: Path) -> None:
        try:
            path.unlink()
        except OSError as exc:
            raise RuntimeError(f"Failed to discard interrupted recording artifact: {path}") from exc
        logger.warning("Discarded interrupted recording artifact: %s", path)

    def _create_datasets(self, h5: h5py.File) -> dict[str, h5py.Dataset]:
        datasets = {
            FRAME_INDEX_KEY: h5.create_dataset(
                FRAME_INDEX_KEY,
                shape=(0,),
                maxshape=(None,),
                chunks=(1024,),
                dtype=np.int64,
            ),
            TIMESTAMP_KEY: h5.create_dataset(
                TIMESTAMP_KEY,
                shape=(0,),
                maxshape=(None,),
                chunks=(1024,),
                dtype=np.float64,
            ),
            self._schema.state_key: self._create_vector_dataset(
                h5,
                self._schema.state_key,
                self._schema.state_dim,
            ),
            self._schema.mode_key: h5.create_dataset(
                self._schema.mode_key,
                shape=(0,),
                maxshape=(None,),
                chunks=(1024,),
                dtype=np.int8,
            ),
            self._schema.action_key: self._create_vector_dataset(
                h5,
                self._schema.action_key,
                self._schema.action_dim,
            ),
        }
        for enabled, key, dim in (
            (self._schema.has_hand_action, self._schema.hand_state_key, self._schema.hand_state_dim),
            (self._schema.has_hand_action, self._schema.hand_action_key, self._schema.hand_action_dim),
            (self._schema.has_neck_action, self._schema.neck_state_key, self._schema.neck_state_dim),
            (self._schema.has_neck_action, self._schema.neck_action_key, self._schema.neck_action_dim),
        ):
            if enabled:
                datasets[key] = self._create_vector_dataset(h5, key, dim)
        return datasets

    @staticmethod
    def _create_vector_dataset(h5: h5py.File, key: str, dim: int) -> h5py.Dataset:
        return h5.create_dataset(
            key,
            shape=(0, dim),
            maxshape=(None, dim),
            chunks=(1024, dim),
            dtype=np.float32,
        )

    @staticmethod
    def _validate_vector(value: object, key: str, dim: int) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.shape[0] != dim:
            raise ValueError(f"{key} must be {dim}D")
        return arr

    @staticmethod
    def _validate_mode(value: object) -> np.int8:
        arr = np.asarray(value).reshape(-1)
        if arr.shape[0] != 1:
            raise ValueError(f"{MODE_KEY} must be scalar")
        parsed = int(arr[0])
        if parsed not in MODE_CODES.values():
            raise ValueError(f"{MODE_KEY} must be one of {sorted(MODE_CODES.values())}, got {parsed}")
        return np.int8(parsed)

    def _schema_dict(self) -> dict[str, object]:
        return hdf5_schema(self._schema)

    def _create_video_writer(self, path: Path) -> Any:
        try:
            import imageio.v2 as imageio
        except Exception as exc:
            raise RuntimeError("MP4 recording requires imageio[ffmpeg]") from exc
        return imageio.get_writer(
            str(path),
            fps=self._fps,
            codec=self._video_config.codec,
            quality=self._video_config.quality,
            macro_block_size=1,
            pixelformat=self._video_config.pixelformat,
        )

    def _close_active_outputs(self) -> None:
        if self._video_writer is not None:
            self._video_writer.close()
            self._video_writer = None
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
        self._datasets = {}

    def _reset_episode(self) -> None:
        self._active = False
        self._frames_in_episode = 0
        self._active_episode_index = None
        self._h5 = None
        self._tmp_path = None
        self._episode_path = None
        self._tmp_video_path = None
        self._episode_video_path = None
        self._data_rel_path = None
        self._video_rel_path = None
        self._video_writer = None
        self._datasets = {}

    def _cleanup_partial_episode(self) -> None:
        if self._video_writer is not None:
            try:
                self._video_writer.close()
            except Exception:
                logger.exception("Failed to close partial MP4 recording writer")
            self._video_writer = None
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                logger.exception("Failed to close partial HDF5 recording file")
            self._h5 = None
        for path in (self._tmp_path, self._tmp_video_path):
            if path is not None and path.exists():
                try:
                    path.unlink()
                except Exception:
                    logger.exception("Failed to remove partial recording file: %s", path)
        self._reset_episode()

    def _require_tmp_path(self) -> Path:
        if self._tmp_path is None:
            raise RuntimeError("recording episode has no temporary path")
        return self._tmp_path

    def _require_episode_path(self) -> Path:
        if self._episode_path is None:
            raise RuntimeError("recording episode has no output path")
        return self._episode_path


def _state_names() -> list[str]:
    return [
        *[f"{name}.position" for name in G1_JOINT_NAMES],
        *[f"{name}.velocity" for name in G1_JOINT_NAMES],
        "base_quat.w",
        "base_quat.x",
        "base_quat.y",
        "base_quat.z",
        "base_ang_vel.x",
        "base_ang_vel.y",
        "base_ang_vel.z",
        "projected_gravity.x",
        "projected_gravity.y",
        "projected_gravity.z",
    ]


def _reference_action_names() -> list[str]:
    return [
        "root_pos.x",
        "root_pos.y",
        "root_pos.z",
        "root_quat.w",
        "root_quat.x",
        "root_quat.y",
        "root_quat.z",
        *G1_JOINT_NAMES,
    ]


def _hand_action_names(hand_type: str) -> list[str]:
    if hand_type == "linkerhand_l6":
        joint_order = L6_SDK_JOINT_ORDER
    elif hand_type == "linkerhand_o6":
        joint_order = O6_SDK_JOINT_ORDER
    else:
        raise ValueError(f"hand action names are unavailable for hand_type={hand_type!r}")
    return [
        *[f"left_{name}" for name in joint_order],
        *[f"right_{name}" for name in joint_order],
    ]


def _video_storage_key(value: str) -> str:
    leaf = value.rsplit(".", 1)[-1]
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", leaf).strip("._")
    return safe or "camera"
