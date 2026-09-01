from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from teleopit.inputs.pico4_provider import BODY_JOINT_NAMES, Pico4InputProvider
from teleopit.inputs.realtime_packet import ControlEventType
from teleopit.inputs.xrobotoolkit_provider import XRoboToolkitInputProvider
from teleopit.inputs.xrobotoolkit_utils import close_sdk_bounded


class _FakeSDK:
    def __init__(self) -> None:
        self.initialized = False
        self.closed = False
        self.timestamp_ns = 1
        self.a_pressed = False
        self.b_pressed = False
        self.x_pressed = False
        self.y_pressed = False
        self.left_menu_pressed = False
        self.left_grip = 0.25
        self.right_grip = 0.5
        self.left_trigger = 0.75
        self.right_trigger = 1.0
        self.body_available = True
        self.body = np.zeros((len(BODY_JOINT_NAMES), 7), dtype=np.float64)
        self.body[:, 0] = np.linspace(-0.2, 0.2, len(BODY_JOINT_NAMES))
        self.body[:, 1] = np.linspace(0.1, 1.0, len(BODY_JOINT_NAMES))
        self.body[:, 6] = 1.0

    def init(self) -> None:
        self.initialized = True

    def close(self) -> None:
        self.closed = True

    def is_body_data_available(self) -> bool:
        return self.body_available

    def get_body_joints_pose(self) -> np.ndarray:
        return self.body.copy()

    def get_body_timestamp_ns(self) -> int:
        return self.timestamp_ns

    def get_headset_pose(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    def get_left_grip(self) -> float:
        return self.left_grip

    def get_right_grip(self) -> float:
        return self.right_grip

    def get_left_trigger(self) -> float:
        return self.left_trigger

    def get_right_trigger(self) -> float:
        return self.right_trigger

    def get_left_menu_button(self) -> bool:
        return self.left_menu_pressed

    def get_A_button(self) -> bool:
        return self.a_pressed

    def get_B_button(self) -> bool:
        return self.b_pressed

    def get_X_button(self) -> bool:
        return self.x_pressed

    def get_Y_button(self) -> bool:
        return self.y_pressed


class _BlockingCloseSDK:
    """Minimal SDK whose close method models a disconnected service."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.close_started = threading.Event()

    def close(self) -> None:
        self.close_started.set()
        self.release.wait()


def test_xrobotoolkit_sdk_close_helper_is_bounded_for_blocking_binding(caplog) -> None:
    sdk = _BlockingCloseSDK()
    started = time.monotonic()
    assert close_sdk_bounded(sdk, timeout_s=0.03, context="test SDK") is False
    elapsed = time.monotonic() - started
    assert elapsed < 0.5
    assert sdk.close_started.wait(timeout=0.2)
    assert "did not return" in caplog.text
    # Let the daemon worker finish before the test process exits.  This also
    # avoids leaving a foreign-SDK call around for subsequent tests.
    sdk.release.set()


def test_xrobotoolkit_sdk_close_helper_handles_thread_start_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shutdown remains best-effort when no new thread can be created."""

    import teleopit.inputs.xrobotoolkit_utils as utils

    class _StartFailureThread:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def start(self) -> None:
            raise RuntimeError("thread quota exhausted")

    monkeypatch.setattr(utils.threading, "Thread", _StartFailureThread)
    class _Closable:
        def close(self) -> None:
            pass

    assert close_sdk_bounded(_Closable(), timeout_s=0.1, context="test SDK") is False


def test_xrobotoolkit_provider_shutdown_does_not_wait_forever_for_sdk() -> None:
    sdk = _BlockingCloseSDK()
    # Bypass __init__ so no SDK body methods are needed for this shutdown-only
    # regression test; close() should still invoke the configured binding.
    provider = object.__new__(XRoboToolkitInputProvider)
    provider._closed = True
    poll_thread = threading.Thread(target=lambda: None)
    poll_thread.start()
    poll_thread.join()
    provider._poll_thread = poll_thread
    provider._sdk = sdk
    provider._close_sdk = True
    provider._sdk_shutdown_settle_s = 0.0
    provider._sdk_started_at = time.monotonic()
    provider._sdk_close_timeout_s = 0.03

    started = time.monotonic()
    provider.close()
    assert time.monotonic() - started < 0.5
    assert sdk.close_started.wait(timeout=0.2)
    sdk.release.set()


def test_xrobotoolkit_provider_closes_sdk_when_poll_thread_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed Python worker startup must clean up a successful SDK init."""

    import teleopit.inputs.xrobotoolkit_provider as provider_module

    sdk = _BlockingCloseSDK()
    real_thread = threading.Thread

    class _StartFailureThread:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def start(self) -> None:
            raise RuntimeError("thread start failed")

    def thread_factory(*args: object, **kwargs: object):
        if kwargs.get("name") == "xrobotoolkit_input":
            return _StartFailureThread(*args, **kwargs)
        return real_thread(*args, **kwargs)

    # Replace only the provider module's Thread lookup.  The helper's own
    # threading module remains available so it can run the bounded close.
    monkeypatch.setattr(provider_module.threading, "Thread", thread_factory)
    sdk.init = lambda: None  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="thread start failed"):
        XRoboToolkitInputProvider(
            sdk=sdk,
            close_sdk=True,
            sdk_shutdown_settle_s=0.0,
            sdk_close_timeout_s=0.05,
        )

    # The blocking close worker is released after proving that the provider
    # attempted cleanup; this keeps the test process free of a daemon call.
    assert sdk.close_started.wait(timeout=0.2)
    sdk.release.set()


def test_xrobotoolkit_provider_closes_sdk_when_init_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native init error after partial startup still gets cleanup."""

    import teleopit.inputs.xrobotoolkit_provider as provider_module

    class _InitFailureSDK:
        def __init__(self) -> None:
            self.closed = False

        def init(self) -> None:
            raise RuntimeError("PXREAInit failed")

        def close(self) -> None:
            self.closed = True

    sdk = _InitFailureSDK()
    # Avoid a real settle delay and keep the test deterministic.
    with pytest.raises(RuntimeError, match="PXREAInit failed"):
        XRoboToolkitInputProvider(
            sdk=sdk,
            close_sdk=True,
            sdk_shutdown_settle_s=0.0,
            sdk_close_timeout_s=0.1,
        )
    assert sdk.closed is True


def test_xrobotoolkit_provider_adapts_body_frames_and_controls() -> None:
    sdk = _FakeSDK()
    provider = XRoboToolkitInputProvider(
        timeout=0.5,
        poll_hz=200.0,
        close_sdk=True,
        sdk_shutdown_settle_s=0.0,
        sdk=sdk,
    )
    try:
        frame, _, first_seq = provider.get_frame_packet()
        assert sdk.initialized is True
        assert set(frame) == set(BODY_JOINT_NAMES)
        controller_snapshot = provider.get_controller_snapshot()
        assert controller_snapshot is not None
        assert controller_snapshot.right.trigger == 1.0
        # Auxiliary snapshots are consumed by independent IPC workers.  Their
        # sequence must identify the same body sample, rather than the cache
        # sequence from the previous poll (which would add one frame of
        # apparent controller/HMD latency).
        assert controller_snapshot.seq == first_seq
        head_snapshot = provider.get_head_pose_snapshot()
        assert head_snapshot is not None
        assert head_snapshot.seq == first_seq

        sdk.left_menu_pressed = True
        sdk.timestamp_ns += 1
        sdk.body[0, 0] += 0.01
        deadline = time.monotonic() + 0.5
        packet = provider.get_realtime_input_packet()
        while (packet.seq <= first_seq or not packet.control_events) and time.monotonic() < deadline:
            time.sleep(0.01)
            packet = provider.get_realtime_input_packet()
        assert packet.seq > first_seq
        controller_snapshot = provider.get_controller_snapshot()
        assert controller_snapshot is not None
        assert controller_snapshot.seq == packet.seq
        assert [event.event_type for event in packet.control_events] == [ControlEventType.ENTER_MOCAP]
        assert packet.control_events[0].source == "xrobotoolkit:left_menu+right_trigger"

        sdk.left_menu_pressed = False
        sdk.left_grip = 1.0
        sdk.right_grip = 1.0
        sdk.timestamp_ns += 1
        sdk.body[0, 0] += 0.01
        deadline = time.monotonic() + 0.5
        packet = provider.get_realtime_input_packet()
        while ControlEventType.ENTER_STANDING not in [event.event_type for event in packet.control_events] and time.monotonic() < deadline:
            time.sleep(0.01)
            packet = provider.get_realtime_input_packet()
        assert ControlEventType.ENTER_STANDING in [event.event_type for event in packet.control_events]
        assert packet.control_events[0].source == "xrobotoolkit:left_grip+right_grip"
    finally:
        provider.close()
    assert sdk.closed is True


def test_xrobotoolkit_provider_ignores_chords_until_body_tracking_is_available() -> None:
    """Cached controller buttons must not activate a mode before Full-body."""
    sdk = _FakeSDK()
    sdk.body_available = False
    sdk.left_menu_pressed = True
    provider = XRoboToolkitInputProvider(
        timeout=0.1,
        poll_hz=200.0,
        close_sdk=True,
        sdk_shutdown_settle_s=0.0,
        sdk=sdk,
    )
    try:
        # Let several polling cycles run while only cached controller values
        # are available.  No event should be queued and no body frame should
        # become ready.
        time.sleep(0.03)
        assert provider.pop_control_events() == ()
        assert provider.has_frame() is False

        # Once a valid body sample arrives, the same physical chord is now
        # eligible and emits exactly one edge event.
        sdk.body_available = True
        sdk.timestamp_ns += 1
        sdk.body[0, 0] += 0.01
        deadline = time.monotonic() + 0.5
        events = ()
        while not events and time.monotonic() < deadline:
            events = provider.pop_control_events()
            if not events:
                time.sleep(0.01)
        assert [event.event_type for event in events] == [ControlEventType.ENTER_MOCAP]
    finally:
        provider.close()


def test_xrobotoolkit_provider_accepts_legacy_pause_arms_keywords(caplog) -> None:
    """Old integrations should fail soft while retaining SIMPLE mappings."""
    sdk = _FakeSDK()
    sdk.body_available = False
    provider = XRoboToolkitInputProvider(
        timeout=0.1,
        poll_hz=200.0,
        close_sdk=True,
        sdk_shutdown_settle_s=0.0,
        sdk=sdk,
        pause_button="A",
        pause_debounce_s=0.2,
        arms_button="B",
        arms_debounce_s=0.2,
    )
    try:
        assert "ignores legacy pause/arms" in caplog.text
    finally:
        provider.close()


def test_xrobotoolkit_provider_keeps_body_frame_when_auxiliary_scalar_is_malformed() -> None:
    """A transient SDK trigger failure must not freeze valid body tracking."""
    sdk = _FakeSDK()
    sdk.body_available = False
    sdk.left_menu_pressed = True
    provider = XRoboToolkitInputProvider(
        timeout=0.2,
        poll_hz=200.0,
        close_sdk=True,
        sdk_shutdown_settle_s=0.0,
        sdk=sdk,
    )
    try:
        # Keep the background worker out of the way and inject a malformed
        # right trigger into one otherwise valid body sample.
        time.sleep(0.02)
        sdk.body_available = True
        sdk.get_right_trigger = lambda: None  # type: ignore[method-assign]
        accepted = False
        for _ in range(5):
            sdk.timestamp_ns += 1
            sdk.body[0, 0] += 0.01
            accepted = provider._poll_once() or accepted
            if accepted:
                break
        assert accepted is True
        assert provider.has_frame() is True
        assert provider.pop_control_events() == ()
        snapshot = provider.get_controller_snapshot()
        assert snapshot is not None
        assert snapshot.right.present is False
        assert snapshot.right.trigger == 0.0
        assert provider.get_head_pose_snapshot() is not None
    finally:
        provider.close()


def test_xrobotoolkit_auxiliary_snapshot_sequence_matches_new_body_packet() -> None:
    """Controller/HMD snapshots use the sequence of the body frame they accompany."""
    sdk = _FakeSDK()
    # Keep the provider's polling thread from consuming the sample; this makes
    # the sequence relationship deterministic instead of relying on a later
    # duplicate SDK poll to refresh the snapshot.
    sdk.body_available = False
    provider = XRoboToolkitInputProvider(
        timeout=0.2,
        poll_hz=1000.0,
        close_sdk=False,
        sdk_shutdown_settle_s=0.0,
        sdk=sdk,
    )
    try:
        provider._closed = True
        provider._poll_thread.join(timeout=1.0)
        sdk.body_available = True

        assert provider._poll_once() is True
        _, _, first_seq = provider.get_frame_packet()
        controller_snapshot = provider.get_controller_snapshot()
        head_snapshot = provider.get_head_pose_snapshot()
        assert first_seq == 1
        assert controller_snapshot is not None and controller_snapshot.seq == first_seq
        assert head_snapshot is not None and head_snapshot.seq == first_seq

        sdk.timestamp_ns += 1
        sdk.body[0, 0] += 0.01
        assert provider._poll_once() is True
        _, _, second_seq = provider.get_frame_packet()
        controller_snapshot = provider.get_controller_snapshot()
        head_snapshot = provider.get_head_pose_snapshot()
        assert second_seq == 2
        assert controller_snapshot is not None and controller_snapshot.seq == second_seq
        assert head_snapshot is not None and head_snapshot.seq == second_seq
    finally:
        provider.close()


def test_xrobotoolkit_provider_accepts_changed_body_when_source_timestamp_is_coarse() -> None:
    """A coarse/stuck SDK timestamp must not suppress changed joint data."""
    sdk = _FakeSDK()
    sdk.body_available = False
    provider = XRoboToolkitInputProvider(
        timeout=0.2,
        poll_hz=1000.0,
        close_sdk=False,
        sdk_shutdown_settle_s=0.0,
        sdk=sdk,
    )
    try:
        provider._closed = True
        provider._poll_thread.join(timeout=1.0)
        sdk.body_available = True

        assert provider._poll_once() is True
        _, _, first_seq = provider.get_frame_packet()
        # Keep the SDK's body timestamp unchanged while moving one joint.  A
        # timestamp-only duplicate check would incorrectly discard this frame.
        sdk.body[0, 0] += 0.01
        assert provider._poll_once() is True
        _, _, second_seq = provider.get_frame_packet()
        assert second_seq > first_seq
    finally:
        provider.close()


def test_xrobotoolkit_provider_ignores_malformed_auxiliary_pose_without_losing_body() -> None:
    sdk = _FakeSDK()
    sdk.body_available = False
    provider = XRoboToolkitInputProvider(
        timeout=0.2,
        poll_hz=200.0,
        close_sdk=True,
        sdk_shutdown_settle_s=0.0,
        sdk=sdk,
    )
    try:
        time.sleep(0.02)
        sdk.body_available = True
        sdk.get_headset_pose = lambda: [float("nan")] * 7  # type: ignore[method-assign]
        accepted = False
        for _ in range(5):
            sdk.timestamp_ns += 1
            sdk.body[0, 1] += 0.01
            accepted = provider._poll_once() or accepted
            if accepted:
                break
        assert accepted is True
        assert provider.has_frame() is True
        head = provider.get_head_pose_snapshot()
        assert head is not None
        assert head.hmd_rotation_wxyz is None
    finally:
        provider.close()


def test_xrobotoolkit_provider_exposes_spine3_with_hmd_snapshot() -> None:
    """The neck auxiliary path receives same-frame Spine3, not skeleton Head."""
    sdk = _FakeSDK()
    sdk.body_available = False
    provider = XRoboToolkitInputProvider(
        timeout=0.2,
        poll_hz=1000.0,
        close_sdk=False,
        sdk_shutdown_settle_s=0.0,
        sdk=sdk,
    )
    try:
        provider._closed = True
        provider._poll_thread.join(timeout=1.0)
        sdk.body_available = True
        angle = np.deg2rad(25.0)
        sdk.body[BODY_JOINT_NAMES.index("Spine3"), 3:7] = [
            0.0,
            np.sin(angle / 2.0),
            0.0,
            np.cos(angle / 2.0),
        ]
        sdk.body[BODY_JOINT_NAMES.index("Head"), 3:7] = [0.0, 0.0, 0.0, 1.0]
        assert provider._poll_once() is True
        snapshot = provider.get_head_pose_snapshot()
        assert snapshot is not None
        expected = Pico4InputProvider._convert_body_joints_to_frame(sdk.body)
        np.testing.assert_allclose(
            snapshot.spine3_rotation_wxyz,
            expected["Spine3"][1],
            atol=1e-6,
        )
        assert snapshot.spine3_rotation_wxyz is not None
    finally:
        provider.close()


def test_xrobotoolkit_provider_rejects_nonfinite_body_sample() -> None:
    sdk = _FakeSDK()
    sdk.body_available = True
    sdk.body[0, 0] = np.nan
    provider = XRoboToolkitInputProvider(
        timeout=0.1,
        poll_hz=200.0,
        close_sdk=True,
        sdk_shutdown_settle_s=0.0,
        sdk=sdk,
    )
    try:
        time.sleep(0.02)
        assert provider._poll_once() is False
        assert provider.has_frame() is False
    finally:
        provider.close()
