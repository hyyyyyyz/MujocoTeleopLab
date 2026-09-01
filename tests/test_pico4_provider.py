from __future__ import annotations

import threading
from collections import deque
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from teleopit.inputs.pico4_provider import BODY_JOINT_NAMES, Pico4InputProvider
from teleopit.inputs.realtime_frame_cache import RealtimeFrameCache
from teleopit.inputs.realtime_packet import ControlEventType


def _body_poses(offset: float) -> np.ndarray:
    body_poses = np.zeros((len(BODY_JOINT_NAMES), 7), dtype=np.float64)
    body_poses[:, 0] = offset
    body_poses[:, 6] = 1.0
    return body_poses


def _pico_frame(
    body_poses: np.ndarray,
    *,
    seq: int,
    timestamp: float,
    body_active: bool = True,
    right_primary: bool = False,
    right_secondary: bool = False,
    head_rotation_xyzw: np.ndarray | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        seq=seq,
        receive_time_s=timestamp,
        head=(
            None
            if head_rotation_xyzw is None
            else SimpleNamespace(rotation=np.asarray(head_rotation_xyzw, dtype=np.float64))
        ),
        body=SimpleNamespace(active=body_active, joints=body_poses),
        controllers=SimpleNamespace(
            left=SimpleNamespace(buttons={}),
            right=SimpleNamespace(buttons={"primaryButton": right_primary, "secondaryButton": right_secondary}),
        ),
    )


def _hand_state(*, active: bool, value: float) -> SimpleNamespace:
    joints = np.zeros((26, 7), dtype=np.float64)
    joints[:, 0:3] = value
    return SimpleNamespace(active=active, joints=joints)


def _make_provider() -> Pico4InputProvider:
    provider = object.__new__(Pico4InputProvider)
    provider._lock = threading.Lock()
    provider._frame_ready = threading.Event()
    provider._frame_cache = RealtimeFrameCache(buffer_size=8, fps_window=30)
    provider._timeout = 1.0
    provider._timestamp_gap_reset_s = 0.15
    provider._pending_control_events = deque()
    provider._pause_button = "A"
    provider._arms_button = "B"
    provider._pause_debounce_s = 0.0
    provider._arms_debounce_s = 0.0
    provider._pause_button_path = provider._resolve_button_path(provider._pause_button)
    provider._arms_button_path = provider._resolve_button_path(provider._arms_button)
    provider._last_pause_button_pressed = False
    provider._last_arms_button_pressed = False
    provider._last_pause_toggle_timestamp = None
    provider._last_arms_toggle_timestamp = None
    provider._last_raw_body_joints = None
    provider._last_frame_timestamp = None
    provider._last_source_seq = None
    provider._ground_alignment_offset = None
    provider._controller_snapshot = None
    provider._hand_snapshot = None
    provider._head_pose_snapshot = None
    provider._closed = False
    return provider


class _FakeBridge:
    instances: list["_FakeBridge"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.started = False
        self.closed = False
        _FakeBridge.instances.append(self)

    def start(self) -> None:
        self.started = True

    def wait_frame(self, timeout: float = 0.1, after_seq: int | None = None) -> Any:
        del timeout, after_seq
        raise TimeoutError

    def close(self) -> None:
        self.closed = True


class _LegacyBridge:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        discovery: bool,
        advertise_ip: str | None,
        video: str | None,
        history_size: int,
        start_timeout: float,
    ) -> None:
        del host, port, discovery, advertise_ip, video, history_size, start_timeout


def test_pico4_provider_starts_pico_bridge_receiver_with_config() -> None:
    _FakeBridge.instances.clear()

    provider = Pico4InputProvider(
        timeout=0.01,
        bridge_host="127.0.0.1",
        bridge_port=12345,
        bridge_discovery=False,
        bridge_advertise_ip="127.0.0.1",
        bridge_video="frames",
        bridge_video_enabled=True,
        bridge_start_timeout=1.5,
        bridge_history_size=7,
        bridge_cls=_FakeBridge,
    )

    try:
        bridge = _FakeBridge.instances[-1]
        assert bridge.started is True
        assert bridge.kwargs == {
            "host": "127.0.0.1",
            "port": 12345,
            "discovery": False,
            "advertise_ip": "127.0.0.1",
            "video": "frames",
            "video_enabled": True,
            "history_size": 7,
            "start_timeout": 1.5,
        }
    finally:
        provider.close()

    assert bridge.closed is True


def test_pico4_provider_requires_pico_bridge_0_2_1_signature() -> None:
    with pytest.raises(RuntimeError, match=r"pico_bridge >= 0\.2\.1"):
        Pico4InputProvider(timeout=0.01, bridge_cls=_LegacyBridge)


def test_pico4_provider_pushes_video_frame_to_bridge() -> None:
    _FakeBridge.instances.clear()

    def push_video_frame(self: _FakeBridge, frame: np.ndarray) -> int:
        self.video_frame = frame
        return 12

    _FakeBridge.push_video_frame = push_video_frame  # type: ignore[attr-defined]
    provider = Pico4InputProvider(timeout=0.01, bridge_video="frames", bridge_video_enabled=True, bridge_cls=_FakeBridge)
    try:
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        assert provider.push_video_frame(frame) == 12
        assert _FakeBridge.instances[-1].video_frame is frame
    finally:
        provider.close()
        delattr(_FakeBridge, "push_video_frame")


def test_pico4_provider_converts_pico_native_body_pose_convention() -> None:
    body_poses = _body_poses(0.0)
    body_poses[0] = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9]

    frame = Pico4InputProvider._convert_body_joints_to_frame(body_poses)
    np.testing.assert_allclose(frame["Pelvis"][0], [1.0, -3.0, 2.0], atol=1e-6)


def test_pico4_provider_applies_fixed_ground_alignment_from_first_real_frame() -> None:
    provider = _make_provider()
    body_poses = np.zeros((len(BODY_JOINT_NAMES), 7), dtype=np.float64)
    pelvis_idx = BODY_JOINT_NAMES.index("Pelvis")
    left_ankle_idx = BODY_JOINT_NAMES.index("Left_Ankle")
    right_ankle_idx = BODY_JOINT_NAMES.index("Right_Ankle")
    body_poses[pelvis_idx, 0:3] = [0.0, 0.8, 0.0]
    body_poses[left_ankle_idx, 0:3] = [0.1, -0.2, 0.0]
    body_poses[right_ankle_idx, 0:3] = [-0.1, 0.1, 0.0]
    body_poses[:, 6] = 1.0

    assert provider._accept_pico_frame(_pico_frame(body_poses, seq=1, timestamp=1.0)) is True
    first_frame, _, _ = provider._frame_cache.latest_packet()
    np.testing.assert_allclose(first_frame["Pelvis"][0][2], 0.8 + 0.2, atol=1e-6)
    np.testing.assert_allclose(first_frame["Left_Ankle"][0][2], 0.0, atol=1e-6)
    assert provider._ground_alignment_offset == pytest.approx(0.2)

    body_poses[:, 1] += 0.3
    assert provider._accept_pico_frame(_pico_frame(body_poses, seq=2, timestamp=1.1)) is True
    second_frame, _, _ = provider._frame_cache.latest_packet()
    np.testing.assert_allclose(second_frame["Pelvis"][0][2], first_frame["Pelvis"][0][2] + 0.3, atol=1e-6)
    np.testing.assert_allclose(second_frame["Left_Ankle"][0][2], 0.3, atol=1e-6)


def test_pico4_provider_aligns_floating_first_frame_down_to_ground() -> None:
    provider = _make_provider()
    body_poses = np.zeros((len(BODY_JOINT_NAMES), 7), dtype=np.float64)
    pelvis_idx = BODY_JOINT_NAMES.index("Pelvis")
    left_ankle_idx = BODY_JOINT_NAMES.index("Left_Ankle")
    right_ankle_idx = BODY_JOINT_NAMES.index("Right_Ankle")
    body_poses[:, 1] = 0.2
    body_poses[pelvis_idx, 0:3] = [0.0, 0.9, 0.0]
    body_poses[left_ankle_idx, 0:3] = [0.1, 0.2, 0.0]
    body_poses[right_ankle_idx, 0:3] = [-0.1, 0.4, 0.0]
    body_poses[:, 6] = 1.0

    assert provider._accept_pico_frame(_pico_frame(body_poses, seq=1, timestamp=1.0)) is True
    first_frame, _, _ = provider._frame_cache.latest_packet()
    np.testing.assert_allclose(first_frame["Left_Ankle"][0][2], 0.0, atol=1e-6)
    np.testing.assert_allclose(first_frame["Pelvis"][0][2], 0.7, atol=1e-6)
    assert provider._ground_alignment_offset == pytest.approx(-0.2)

    body_poses[:, 1] += 0.3
    assert provider._accept_pico_frame(_pico_frame(body_poses, seq=2, timestamp=1.1)) is True
    second_frame, _, _ = provider._frame_cache.latest_packet()
    np.testing.assert_allclose(second_frame["Left_Ankle"][0][2], 0.3, atol=1e-6)
    np.testing.assert_allclose(second_frame["Pelvis"][0][2], 1.0, atol=1e-6)


def test_pico4_provider_recomputes_ground_alignment_after_timestamp_gap_reset() -> None:
    provider = _make_provider()
    body_poses = np.zeros((len(BODY_JOINT_NAMES), 7), dtype=np.float64)
    pelvis_idx = BODY_JOINT_NAMES.index("Pelvis")
    left_ankle_idx = BODY_JOINT_NAMES.index("Left_Ankle")
    right_ankle_idx = BODY_JOINT_NAMES.index("Right_Ankle")
    body_poses[pelvis_idx, 0:3] = [0.0, 0.8, 0.0]
    body_poses[left_ankle_idx, 0:3] = [0.1, -0.2, 0.0]
    body_poses[right_ankle_idx, 0:3] = [-0.1, 0.1, 0.0]
    body_poses[:, 6] = 1.0

    assert provider._accept_pico_frame(_pico_frame(body_poses, seq=1, timestamp=1.0)) is True
    assert provider._ground_alignment_offset == pytest.approx(0.2)

    body_poses[pelvis_idx, 1] = 0.7
    body_poses[left_ankle_idx, 1] = -0.5
    body_poses[right_ankle_idx, 1] = 0.2
    assert provider._accept_pico_frame(_pico_frame(body_poses, seq=2, timestamp=1.3)) is True
    latest_frame, _, _ = provider._frame_cache.latest_packet()
    np.testing.assert_allclose(latest_frame["Left_Ankle"][0][2], 0.0, atol=1e-6)
    np.testing.assert_allclose(latest_frame["Pelvis"][0][2], 1.2, atol=1e-6)
    assert provider._ground_alignment_offset == pytest.approx(0.5)
    assert len(provider._frame_cache) == 1


def test_pico4_provider_drops_duplicate_raw_body_pose() -> None:
    provider = _make_provider()
    body_poses = _body_poses(1.0)

    assert provider._accept_pico_frame(_pico_frame(body_poses, seq=1, timestamp=1.0)) is True
    assert provider._accept_pico_frame(_pico_frame(body_poses.copy(), seq=2, timestamp=1.01)) is False
    assert len(provider._frame_cache) == 1


def test_pico4_provider_resets_interpolation_buffer_on_large_timestamp_gap() -> None:
    provider = _make_provider()

    assert provider._accept_pico_frame(_pico_frame(_body_poses(1.0), seq=1, timestamp=1.00)) is True
    assert provider._accept_pico_frame(_pico_frame(_body_poses(2.0), seq=2, timestamp=1.02)) is True
    assert len(provider._frame_cache) == 2

    assert provider._accept_pico_frame(_pico_frame(_body_poses(9.0), seq=3, timestamp=1.30)) is True
    assert len(provider._frame_cache) == 1
    latest_frame, latest_ts, latest_seq = provider._frame_cache.latest_packet()
    np.testing.assert_allclose(latest_frame["Pelvis"][0][0], 9.0, atol=1e-6)
    np.testing.assert_allclose(latest_ts, 1.30, atol=1e-6)
    assert latest_seq == 3


def test_pico4_provider_exposes_pause_control_events_once() -> None:
    provider = _make_provider()

    assert provider._accept_pico_frame(
        _pico_frame(_body_poses(1.0), seq=1, timestamp=1.0, right_primary=True)
    ) is True

    packet = provider.get_realtime_input_packet()
    assert [event.event_type for event in packet.control_events] == [ControlEventType.TOGGLE_PAUSE]

    packet = provider.get_realtime_input_packet()
    assert packet.control_events == ()


def test_pico4_provider_exposes_arms_control_events_once() -> None:
    provider = _make_provider()

    assert provider._accept_pico_frame(
        _pico_frame(_body_poses(1.0), seq=1, timestamp=1.0, right_secondary=True)
    ) is True

    packet = provider.get_realtime_input_packet()
    assert [event.event_type for event in packet.control_events] == [ControlEventType.TOGGLE_ARMS]

    packet = provider.get_realtime_input_packet()
    assert packet.control_events == ()


def test_pico4_provider_marks_controller_present_without_raw_field() -> None:
    provider = _make_provider()
    frame = _pico_frame(_body_poses(1.0), seq=1, timestamp=1.0)
    frame.controllers.left.axis = {"grip": 0.8, "trigger": 0.4}

    assert provider._accept_pico_frame(frame) is True

    snapshot = provider.get_controller_snapshot()
    assert snapshot is not None
    assert snapshot.left.present is True
    assert snapshot.left.raw is False
    assert snapshot.left.grip == pytest.approx(0.8)
    assert snapshot.left.trigger == pytest.approx(0.4)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, "0.5", True, [0.5]])
def test_pico4_provider_neutralizes_malformed_controller_axes(value: object) -> None:
    """Bad optional axis values must not discard a valid body frame."""
    provider = _make_provider()
    frame = _pico_frame(_body_poses(1.0), seq=1, timestamp=1.0)
    frame.controllers.left.axis = {"grip": value, "trigger": 0.25}

    assert provider._accept_pico_frame(frame) is True
    snapshot = provider.get_controller_snapshot()
    assert snapshot is not None
    assert snapshot.left.present is False
    assert snapshot.left.grip == 0.0
    assert snapshot.left.trigger == pytest.approx(0.25)


def test_pico4_provider_rejects_non_boolean_button_map_values() -> None:
    """A truthy string must never become a controller edge event."""
    provider = _make_provider()
    frame = _pico_frame(_body_poses(1.0), seq=1, timestamp=1.0)
    frame.controllers.right.buttons["primaryButton"] = "false"

    assert provider._accept_pico_frame(frame) is True
    assert provider.pop_control_events() == ()


def test_pico4_provider_reads_pause_control_events_when_body_inactive() -> None:
    """Safety/session controls remain usable before Full-body tracking."""
    provider = _make_provider()

    assert provider._accept_pico_frame(
        _pico_frame(_body_poses(1.0), seq=1, timestamp=1.0, body_active=False, right_primary=True)
    ) is False

    events = provider.pop_control_events()
    assert [event.event_type for event in events] == [ControlEventType.TOGGLE_PAUSE]
    assert provider._last_source_seq == 1


def test_pico4_provider_polls_pause_control_after_body_tracking_starts() -> None:
    """Once a valid body frame exists, a stationary body can still toggle."""
    provider = _make_provider()
    body_poses = _body_poses(1.0)

    assert provider._accept_pico_frame(
        _pico_frame(body_poses, seq=1, timestamp=1.0, right_primary=False)
    ) is True
    assert provider._accept_pico_frame(
        _pico_frame(body_poses.copy(), seq=2, timestamp=1.01, right_primary=True)
    ) is False

    events = provider.pop_control_events()
    assert [event.event_type for event in events] == [ControlEventType.TOGGLE_PAUSE]


def test_pico4_provider_does_not_consume_y_mode_before_body_tracking() -> None:
    """Y/X mode edges wait for a valid body frame, unlike A/B controls."""
    provider = _make_provider()
    inactive = _pico_frame(_body_poses(1.0), seq=1, timestamp=1.0, body_active=False)
    inactive.controllers.left.buttons = {"secondaryButton": True}
    assert provider._accept_pico_frame(inactive) is False
    assert provider.pop_control_events() == ()

    active = _pico_frame(_body_poses(1.0), seq=2, timestamp=1.1, body_active=True)
    active.controllers.left.buttons = {"secondaryButton": True}
    assert provider._accept_pico_frame(active) is True
    events = provider.pop_control_events()
    assert [event.event_type for event in events] == [ControlEventType.ENTER_MOCAP]


def test_pico4_provider_does_not_consume_y_mode_from_nonfinite_body() -> None:
    """A malformed shaped body packet must not arm a mode transition."""
    provider = _make_provider()
    malformed = _body_poses(1.0)
    malformed[0, 0] = np.nan
    frame = _pico_frame(malformed, seq=1, timestamp=1.0)
    frame.controllers.left.buttons = {"secondaryButton": True}

    assert provider._accept_pico_frame(frame) is False
    assert provider.pop_control_events() == ()
    assert provider.has_frame() is False


def test_pico4_poll_loop_survives_frame_adapter_exception() -> None:
    """One malformed bridge frame must not permanently kill input polling."""
    provider = _make_provider()
    provider._last_source_seq = None
    frames = [object(), object()]
    accepted: list[object] = []

    class _Bridge:
        def wait_frame(self, *, timeout: float = 0.1, after_seq: int | None = None) -> object:
            del timeout, after_seq
            if frames:
                return frames.pop(0)
            raise TimeoutError

    provider._bridge = _Bridge()

    def flaky_adapter(frame: object) -> None:
        accepted.append(frame)
        if len(accepted) == 1:
            raise ValueError("malformed optional controller field")
        provider._closed = True

    provider._accept_pico_frame = flaky_adapter  # type: ignore[method-assign]
    thread = threading.Thread(target=provider._poll_loop)
    thread.start()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert len(accepted) == 2


def test_pico4_provider_exposes_hand_snapshot_when_body_inactive() -> None:
    provider = _make_provider()
    frame = _pico_frame(_body_poses(1.0), seq=4, timestamp=2.0, body_active=False)
    frame.left_hand = _hand_state(active=True, value=1.5)
    frame.right_hand = _hand_state(active=False, value=2.5)

    assert provider._accept_pico_frame(frame) is False

    snapshot = provider.get_hand_snapshot()
    assert snapshot is not None
    assert snapshot.seq == 4
    assert snapshot.timestamp_s == pytest.approx(2.0)
    assert snapshot.left.present is True
    assert snapshot.left.active is True
    assert snapshot.right.present is True
    assert snapshot.right.active is False
    np.testing.assert_allclose(snapshot.left.joints[:, 0:3], 1.5)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_pico4_provider_rejects_nonfinite_hand_joints(value: float) -> None:
    provider = _make_provider()
    frame = _pico_frame(_body_poses(1.0), seq=4, timestamp=2.0, body_active=False)
    hand = _hand_state(active=True, value=0.0)
    hand.joints[0, 0] = value
    frame.left_hand = hand

    assert provider._accept_pico_frame(frame) is False
    snapshot = provider.get_hand_snapshot()
    assert snapshot is not None
    assert snapshot.left.present is True
    assert snapshot.left.active is False
    assert np.all(np.isfinite(snapshot.left.joints))


def test_pico4_provider_exposes_hmd_rotation_separately_from_skeleton_head() -> None:
    provider = _make_provider()
    body_poses = _body_poses(1.0)
    angle = np.deg2rad(30.0)
    hmd_rotation_xyzw = np.array(
        [0.0, np.sin(angle / 2.0), 0.0, np.cos(angle / 2.0)],
        dtype=np.float64,
    )

    assert provider._accept_pico_frame(
        _pico_frame(
            body_poses,
            seq=7,
            timestamp=3.0,
            head_rotation_xyzw=hmd_rotation_xyzw,
        )
    ) is True

    snapshot = provider.get_head_pose_snapshot()
    assert snapshot is not None
    assert snapshot.seq == 7
    assert snapshot.timestamp_s == pytest.approx(3.0)
    expected_body = body_poses.copy()
    expected_body[BODY_JOINT_NAMES.index("Head"), 3:7] = hmd_rotation_xyzw
    expected_frame = Pico4InputProvider._convert_body_joints_to_frame(expected_body)
    np.testing.assert_allclose(
        snapshot.hmd_rotation_wxyz,
        expected_frame["Head"][1],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        snapshot.spine3_rotation_wxyz,
        expected_frame["Spine3"][1],
        atol=1e-6,
    )
    skeleton_frame = Pico4InputProvider._convert_body_joints_to_frame(body_poses)
    assert not np.allclose(snapshot.hmd_rotation_wxyz, skeleton_frame["Head"][1])


def test_pico4_provider_updates_hmd_snapshot_when_duplicate_body_is_dropped() -> None:
    provider = _make_provider()
    body_poses = _body_poses(1.0)

    assert provider._accept_pico_frame(
        _pico_frame(
            body_poses,
            seq=1,
            timestamp=1.0,
            head_rotation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        )
    ) is True
    first = provider.get_head_pose_snapshot()
    assert first is not None

    angle = np.deg2rad(20.0)
    assert provider._accept_pico_frame(
        _pico_frame(
            body_poses.copy(),
            seq=2,
            timestamp=1.01,
            head_rotation_xyzw=np.array([np.sin(angle / 2.0), 0.0, 0.0, np.cos(angle / 2.0)]),
        )
    ) is False
    second = provider.get_head_pose_snapshot()

    assert second is not None
    assert second.seq == 2
    assert second.timestamp_s == pytest.approx(1.01)
    assert not np.allclose(second.hmd_rotation_wxyz, first.hmd_rotation_wxyz)


def test_pico4_provider_invalidates_spine3_when_body_tracking_is_inactive() -> None:
    provider = _make_provider()

    assert provider._accept_pico_frame(
        _pico_frame(
            _body_poses(1.0),
            seq=4,
            timestamp=2.0,
            body_active=False,
            head_rotation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        )
    ) is False

    snapshot = provider.get_head_pose_snapshot()
    assert snapshot is not None
    assert snapshot.hmd_rotation_wxyz is not None
    assert snapshot.spine3_rotation_wxyz is None
