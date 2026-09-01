from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from teleopit.scenes.controller import _HEADSET_TO_WORLD
from teleopit.scenes.view_state import SceneViewState


def _pose(quat: np.ndarray | None = None) -> list[float]:
    q = np.array([0.0, 0.0, 0.0, 1.0] if quat is None else quat, dtype=float)
    return [0.0, 0.0, 0.0, *q.tolist()]


def test_scene_view_state_b_is_edge_triggered_and_starts_stereo() -> None:
    state = SceneViewState()
    assert state.view_mode == "stereo"
    assert state.update_b_button(False) is None
    assert state.update_b_button(True) == "single"
    assert state.update_b_button(True) is None
    assert state.update_b_button(False) is None
    assert state.update_b_button(True) == "stereo"


@pytest.mark.parametrize("invalid", [1, 0, "true", None, np.array([True])])
def test_scene_view_state_rejects_non_boolean_b_samples(invalid: object) -> None:
    state = SceneViewState()
    with pytest.raises(ValueError, match="pressed must be a boolean"):
        state.update_b_button(invalid)


def test_scene_view_state_calibrates_head_and_returns_relative_rotation() -> None:
    state = SceneViewState()
    state.set_head_pose(_pose())
    assert np.allclose(state.head_rotation_delta(), np.eye(3))
    yaw = Rotation.from_euler("z", 0.4).as_quat()
    state.set_head_pose(_pose(yaw))
    expected = _HEADSET_TO_WORLD @ Rotation.from_quat(yaw).as_matrix() @ _HEADSET_TO_WORLD.T
    assert np.allclose(state.head_rotation_delta(), expected)
    state.reset_head_reference()
    assert state.head_rotation_delta() is None
    state.set_head_pose(_pose(yaw))
    assert np.allclose(state.head_rotation_delta(), np.eye(3))


def test_scene_view_state_reset_reanchors_the_current_head_pose() -> None:
    state = SceneViewState()
    initial_yaw = Rotation.from_euler("z", 0.4).as_quat()
    state.set_head_pose(_pose(initial_yaw))
    moved_yaw = Rotation.from_euler("z", 0.8).as_quat()
    state.set_head_pose(_pose(moved_yaw))
    assert not np.allclose(state.head_rotation_delta(), np.eye(3))

    # This is the operation used by the scene's double-grip reset callback;
    # the next sample, even if the operator is still looking aside, becomes
    # the new neutral reference.
    state.reset_head_reference()
    state.set_head_pose(_pose(moved_yaw))
    assert np.allclose(state.head_rotation_delta(), np.eye(3))


def test_scene_view_state_converts_native_yaw_and_pitch_to_mujoco_basis() -> None:
    state = SceneViewState()
    state.set_head_pose(_pose())

    native_yaw = Rotation.from_euler("y", 0.4).as_quat()
    state.set_head_pose(_pose(native_yaw))
    expected_yaw = _HEADSET_TO_WORLD @ Rotation.from_quat(native_yaw).as_matrix() @ _HEADSET_TO_WORLD.T
    assert np.allclose(state.head_rotation_delta(), expected_yaw)
    # PICO's native up axis is +Y, which maps to MuJoCo +Z.  A positive native
    # yaw therefore remains a positive z-up yaw after the basis conversion.
    assert np.isclose(Rotation.from_matrix(state.head_rotation_delta()).as_euler("xyz")[2], 0.4)

    state.reset_head_reference()
    state.set_head_pose(_pose())
    native_pitch = Rotation.from_euler("x", 0.3).as_quat()
    state.set_head_pose(_pose(native_pitch))
    expected_pitch = _HEADSET_TO_WORLD @ Rotation.from_quat(native_pitch).as_matrix() @ _HEADSET_TO_WORLD.T
    assert np.allclose(state.head_rotation_delta(), expected_pitch)
    assert np.isclose(Rotation.from_matrix(state.head_rotation_delta()).as_euler("xyz")[1], -0.3)


def test_scene_view_state_rejects_malformed_head_pose() -> None:
    state = SceneViewState()
    with pytest.raises(ValueError, match="seven finite"):
        state.set_head_pose([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="seven finite"):
        state.set_head_pose([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan])
    with pytest.raises(ValueError, match="seven finite"):
        state.set_head_pose(["0"] * 7)
    with pytest.raises(ValueError, match="seven finite"):
        state.set_head_pose([True] * 7)


def test_scene_view_state_does_not_calibrate_from_startup_zero_quaternion() -> None:
    """Tracking warm-up samples must not become the camera neutral pose."""
    state = SceneViewState()
    zero_quaternion_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    state.set_head_pose(zero_quaternion_pose)
    assert state.head_rotation_delta() is None

    actual = Rotation.from_euler("z", 0.65).as_quat()
    state.set_head_pose(_pose(actual))
    # The first non-zero orientation is the neutral reference, not a 0.65 rad
    # apparent turn away from the identity fallback.
    assert np.allclose(state.head_rotation_delta(), np.eye(3))


def test_scene_view_state_ignores_zero_quaternion_after_calibration() -> None:
    """A reconnect warm-up sample must not jerk an already calibrated view."""
    state = SceneViewState()
    initial = Rotation.from_euler("z", 0.2).as_quat()
    state.set_head_pose(_pose(initial))
    moved = Rotation.from_euler("z", 0.5).as_quat()
    state.set_head_pose(_pose(moved))
    expected = _HEADSET_TO_WORLD @ Rotation.from_quat(moved).as_matrix() @ Rotation.from_quat(initial).as_matrix().T @ _HEADSET_TO_WORLD.T
    assert np.allclose(state.head_rotation_delta(), expected)

    state.set_head_pose([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(state.head_rotation_delta(), expected)
