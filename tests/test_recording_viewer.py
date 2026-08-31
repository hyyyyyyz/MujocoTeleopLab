from __future__ import annotations

import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.view.view_recording import (
    aligned_qpos_pair,
    load_episode_review_data,
    load_recording_dataset,
)
from teleopit.recording.hdf5 import (
    ACTION_KEY,
    FRAME_INDEX_KEY,
    HAND_ACTION_KEY,
    HAND_STATE_KEY,
    MODE_KEY,
    NECK_ACTION_KEY,
    NECK_STATE_KEY,
    RecordingSchema,
    STATE_KEY,
    TIMESTAMP_KEY,
    hdf5_schema,
)


def _write_recording(
    root: Path,
    *,
    frames: int = 4,
    manifest_frames: int | None = None,
    hand_type: str = "none",
    neck_type: str = "none",
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    schema = RecordingSchema(
        fps=30,
        robot_type="unitree_g1_29dof",
        hand_type=hand_type,
        neck_type=neck_type,
        image_key="observation.images.d435i_rgb",
        image_shape=(4, 6, 3),
    )
    (root / "schema.json").write_text(
        json.dumps(hdf5_schema(schema)),
        encoding="utf-8",
    )
    data_path = root / "data" / "episode_000000.h5"
    data_path.parent.mkdir()
    state = np.zeros((frames, 68), dtype=np.float32)
    state[:, 58] = 1.0
    action = np.zeros((frames, 36), dtype=np.float32)
    action[:, :3] = np.array([1.0, 2.0, 0.8], dtype=np.float32)
    action[:, 3] = 1.0
    action[:, 7:] = 0.1
    with h5py.File(data_path, "w") as h5:
        h5.create_dataset(FRAME_INDEX_KEY, data=np.arange(frames, dtype=np.int64))
        h5.create_dataset(TIMESTAMP_KEY, data=np.arange(frames, dtype=np.float64) / 30.0)
        h5.create_dataset(STATE_KEY, data=state)
        h5.create_dataset(MODE_KEY, data=np.ones(frames, dtype=np.int8))
        h5.create_dataset(ACTION_KEY, data=action)
        if hand_type != "none":
            h5.create_dataset(HAND_STATE_KEY, data=np.full((frames, 12), 20.0, dtype=np.float32))
            h5.create_dataset(HAND_ACTION_KEY, data=np.zeros((frames, 12), dtype=np.float32))
        if neck_type != "none":
            h5.create_dataset(
                NECK_STATE_KEY,
                data=np.tile(np.array([11.5, -7.5], dtype=np.float32), (frames, 1)),
            )
            h5.create_dataset(
                NECK_ACTION_KEY,
                data=np.tile(np.array([12.5, -8.0], dtype=np.float32), (frames, 1)),
            )

    video_path = root / "videos" / "d435i_rgb" / "episode_000000.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.touch()
    manifest = {
        "episode_index": 0,
        "frames": frames if manifest_frames is None else manifest_frames,
        "task": "test task",
        "data": "data/episode_000000.h5",
        "videos": {
            "observation.images.d435i_rgb": "videos/d435i_rgb/episode_000000.mp4"
        },
    }
    (root / "episodes.jsonl").write_text(json.dumps(manifest) + "\n", encoding="utf-8")


def test_recording_viewer_loads_schema_episode_and_tracking_metrics(tmp_path: Path) -> None:
    _write_recording(
        tmp_path,
        hand_type="linkerhand_o6",
        neck_type="openneck",
    )

    dataset = load_recording_dataset(tmp_path)
    data = load_episode_review_data(dataset, dataset.episodes[0])

    assert dataset.fps == 30
    assert dataset.image_shape == (4, 6, 3)
    assert dataset.has_hand_action is True
    assert dataset.has_neck_action is True
    assert data.hand_state is not None
    assert data.hand_action is not None
    assert data.neck_state is not None
    assert data.neck_action is not None
    np.testing.assert_allclose(data.hand_state[0], 20.0)
    np.testing.assert_allclose(data.neck_state[0], [11.5, -7.5])
    assert data.joint_rmse_rad == pytest.approx(0.1)
    assert data.root_orientation_rmse_rad == pytest.approx(0.0)
    assert data.max_joint_error_rad == pytest.approx(0.1)
    assert set(data.group_error) == {
        "left leg",
        "right leg",
        "waist",
        "left arm",
        "right arm",
    }
    assert all(np.allclose(values, 0.1) for values in data.group_error.values())


def test_recording_viewer_aligns_observed_root_position_to_reference(tmp_path: Path) -> None:
    _write_recording(tmp_path)
    dataset = load_recording_dataset(tmp_path)
    data = load_episode_review_data(dataset, dataset.episodes[0])

    actual_qpos, reference_qpos = aligned_qpos_pair(data, 0)

    np.testing.assert_allclose(actual_qpos[:3], [1.0, 2.0, 0.8])
    np.testing.assert_allclose(reference_qpos[:3], [1.0, 2.0, 0.8])
    np.testing.assert_allclose(actual_qpos[7:], 0.0)
    np.testing.assert_allclose(reference_qpos[7:], 0.1)


def test_recording_viewer_rejects_manifest_hdf5_frame_mismatch(tmp_path: Path) -> None:
    _write_recording(tmp_path, frames=4, manifest_frames=5)

    with pytest.raises(ValueError, match="shape .* != manifest/schema shape"):
        load_recording_dataset(tmp_path)


def test_recording_viewer_rejects_non_finite_episode_data(tmp_path: Path) -> None:
    _write_recording(tmp_path)
    data_path = tmp_path / "data" / "episode_000000.h5"
    with h5py.File(data_path, "r+") as h5:
        h5[STATE_KEY][2, 0] = np.nan

    dataset = load_recording_dataset(tmp_path)
    with pytest.raises(ValueError, match="contains NaN or Inf"):
        load_episode_review_data(dataset, dataset.episodes[0])


def test_recording_viewer_rejects_hdf5_dtype_mismatch(tmp_path: Path) -> None:
    _write_recording(tmp_path)
    data_path = tmp_path / "data" / "episode_000000.h5"
    with h5py.File(data_path, "r+") as h5:
        del h5[MODE_KEY]
        h5.create_dataset(MODE_KEY, data=np.full(4, 1.9, dtype=np.float32))

    with pytest.raises(ValueError, match="dtype float32 != schema dtype int8"):
        load_recording_dataset(tmp_path)
