from __future__ import annotations

import numpy as np
import pytest

from teleopit.scenes.controller import SimpleSceneController
from teleopit.scenes.xr_packet import SceneXRPacket


def _packet(**overrides: object) -> SceneXRPacket:
    values: dict[str, object] = {
        "sequence": 1,
        "timestamp_s": 1.0,
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
    values.update(overrides)
    return SceneXRPacket.from_mapping(values)


def test_scene_xr_packet_round_trip() -> None:
    packet = _packet(left_axis=[0.25, -0.75], right_trigger=0.6)
    assert SceneXRPacket.from_wire(packet.to_wire()) == packet


def test_simple_mapping_keeps_b_free_and_maps_navigation_and_fingers() -> None:
    controller = SimpleSceneController()
    unlocked = controller.update(_packet(left_menu=True, left_trigger=1.0))
    assert unlocked.locomotion_toggled is True
    assert unlocked.locomotion_enabled is True
    assert unlocked.toggle_policy_action is True
    command = controller.update(
        _packet(
            sequence=2,
            timestamp_s=1.02,
            left_axis=[-1.0, 1.0],
            right_axis=[-1.0, 0.0],
            right_trigger=1.0,
            b=True,
        )
    )
    assert controller.active is False
    assert command.activation_toggled is False
    assert command.navigate_cmd.shape == (4,)
    assert np.allclose(command.navigate_cmd[:3], [0.5, 0.5, 1.0])
    # The fourth field is SIMPLE's integrated absolute target yaw.  With a
    # one-frame 50 Hz sample it advances by 0.02 rad at the 1 rad/s limit.
    assert np.isclose(command.navigate_cmd[3], 0.02)
    assert command.right_fingers["position"][9, 0, 3] == 1.0


def test_simple_start_and_reset_chords_are_edge_triggered() -> None:
    controller = SimpleSceneController()
    start = controller.update(_packet(left_menu=True, right_trigger=1.0))
    assert start.activation_toggled is True
    assert controller.active is True

    held_start = controller.update(
        _packet(sequence=2, timestamp_s=1.02, left_menu=True, right_trigger=1.0)
    )
    assert held_start.activation_toggled is False
    assert controller.active is True

    controller.update(_packet(sequence=3, timestamp_s=1.04))
    reset = controller.update(
        _packet(sequence=4, timestamp_s=1.06, left_grip=1.0, right_grip=1.0)
    )
    assert reset.reset_requested is True
    held_reset = controller.update(
        _packet(sequence=5, timestamp_s=1.08, left_grip=1.0, right_grip=1.0)
    )
    assert held_reset.reset_requested is False


def test_scene_reset_does_not_retrigger_while_grips_remain_held() -> None:
    controller = SimpleSceneController()
    reset_packet = _packet(left_grip=1.0, right_grip=1.0)
    first = controller.update(reset_packet)
    assert first.reset_requested is True

    # The runtime resets its mode state using the same sample that caused the
    # reset.  A continued physical grip hold must not generate another edge.
    controller.reset(packet=reset_packet)
    held = controller.update(
        _packet(sequence=2, timestamp_s=1.02, left_grip=1.0, right_grip=1.0)
    )
    assert held.reset_requested is False

    released = controller.update(_packet(sequence=3, timestamp_s=1.04))
    assert released.reset_requested is False
    pressed_again = controller.update(
        _packet(sequence=4, timestamp_s=1.06, left_grip=1.0, right_grip=1.0)
    )
    assert pressed_again.reset_requested is True


def test_simple_activation_and_reset_use_strict_half_threshold() -> None:
    """Match PicoStreamer: exactly 0.5 is below a pressed trigger/grip."""
    controller = SimpleSceneController()

    boundary = controller.update(
        _packet(
            left_menu=True,
            left_trigger=0.5,
            right_trigger=0.5,
            left_grip=0.5,
            right_grip=0.5,
        )
    )
    assert boundary.locomotion_toggled is False
    assert boundary.activation_toggled is False
    assert boundary.reset_requested is False
    assert controller.active is False

    pressed = controller.update(
        _packet(
            sequence=2,
            timestamp_s=1.02,
            left_menu=True,
            left_trigger=0.500001,
            right_trigger=0.500001,
            left_grip=0.500001,
            right_grip=0.500001,
        )
    )
    assert pressed.locomotion_toggled is True
    assert pressed.activation_toggled is True
    assert pressed.reset_requested is True


def test_navigation_is_continuous_and_left_chord_is_an_edge_pulse() -> None:
    controller = SimpleSceneController()
    command = controller.update(
        _packet(left_axis=[-1.0, 1.0], right_axis=[-1.0, 0.0])
    )
    # SIMPLE publishes joystick navigation continuously; the policy toggle is
    # a separate Menu+left-trigger event and must not gate the command stream.
    assert np.allclose(command.navigate_cmd[:3], [0.5, 0.5, 1.0])
    assert command.navigate_cmd.shape == (4,)
    assert np.isclose(command.navigate_cmd[3], 0.02)
    assert command.toggle_policy_action is False

    enabled = controller.update(
        _packet(sequence=2, timestamp_s=1.02, left_menu=True, left_trigger=1.0)
    )
    assert enabled.locomotion_toggled is True
    assert enabled.locomotion_enabled is True
    assert enabled.toggle_policy_action is True
    held = controller.update(
        _packet(sequence=3, timestamp_s=1.04, left_menu=True, left_trigger=1.0)
    )
    assert held.locomotion_toggled is False
    assert held.toggle_policy_action is False
    controller.update(_packet(sequence=3, timestamp_s=1.04))
    walking = controller.update(
        _packet(sequence=5, timestamp_s=1.08, left_axis=[-1.0, 1.0], right_axis=[-1.0, 0.0])
    )
    assert np.allclose(walking.navigate_cmd[:3], [0.5, 0.5, 1.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"activation_threshold": 0.0},
        {"reset_threshold": 1.1},
        {"dead_zone": 1.0},
        {"max_linear_velocity": float("nan")},
        {"max_yaw_velocity": -1.0},
        {"initial_base_height": 0.1},
        {"height_rate": -0.1},
        {"height_rate": "bad"},
    ],
)
def test_controller_rejects_invalid_configuration(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SimpleSceneController(**kwargs)  # type: ignore[arg-type]
