from __future__ import annotations

import sys
import types

import pytest

from scripts.run.run_scene_xr_bridge import (
    DEFAULT_SOURCE_HEARTBEAT_INTERVAL_S,
    DEFAULT_SOURCE_STALE_TIMEOUT_S,
    _SourceFreshnessGate,
    _sample_signature,
    _source_timestamp_ns,
    _validate_sdk_api,
)
from teleopit.scenes.xr_packet import SceneXRPacket


class _TimestampSDK:
    def __init__(self, value: object) -> None:
        self.value = value

    def get_time_stamp_ns(self) -> object:
        return self.value


class _SpecificTimestampSDK:
    def __init__(self, value: object) -> None:
        self.value = value

    def get_motion_timestamp_ns(self) -> object:
        return self.value


class _ZeroGenericTimestampSDK:
    """A top-level timestamp API that has not produced a sample yet."""

    def __init__(self, value: object = 456) -> None:
        self.value = value

    def get_time_stamp_ns(self) -> int:
        return 0

    def get_motion_timestamp_ns(self) -> object:
        return self.value


class _MixedTimestampSDK:
    """Top-level XR timestamp must win over an unrelated body timestamp."""

    def get_time_stamp_ns(self) -> int:
        return 100

    def get_motion_timestamp_ns(self) -> int:
        return 900

    def get_body_timestamp_ns(self) -> int:
        return 800


class _StringTimestampSDK:
    def get_time_stamp_ns(self) -> str:
        return "123"


def test_scene_bridge_source_timestamp_gate_reads_positive_timestamp() -> None:
    assert _source_timestamp_ns(_TimestampSDK(123)) == (True, 123)


def test_scene_bridge_source_timestamp_gate_rejects_zero_or_malformed_values() -> None:
    assert _source_timestamp_ns(_TimestampSDK(0)) == (True, None)
    assert _source_timestamp_ns(_TimestampSDK("not-a-timestamp")) == (True, None)


def test_scene_bridge_source_timestamp_gate_supports_older_sdk_without_getter() -> None:
    assert _source_timestamp_ns(object()) == (False, None)


def test_scene_bridge_source_timestamp_gate_falls_back_to_specific_sdk_getter() -> None:
    assert _source_timestamp_ns(_SpecificTimestampSDK(456)) == (True, 456)


def test_scene_bridge_source_timestamp_gate_does_not_use_specific_stream_when_generic_is_zero() -> None:
    # The body/motion stream can continue publishing while controller/HMD
    # data is disconnected.  A present generic accessor is authoritative.
    assert _source_timestamp_ns(_ZeroGenericTimestampSDK()) == (True, None)


def test_scene_bridge_api_validation_reports_missing_control_surface() -> None:
    class _LifecycleOnlySDK:
        init = staticmethod(lambda: None)
        close = staticmethod(lambda: None)

    with pytest.raises(RuntimeError, match="missing required scene accessor") as exc_info:
        _validate_sdk_api(_LifecycleOnlySDK())
    assert "get_left_controller_pose" in str(exc_info.value)
    # Timestamp accessors are intentionally optional for old bindings; the
    # error should not claim that one is required.
    assert "get_time_stamp_ns" not in str(exc_info.value)


def test_scene_bridge_prefers_top_level_timestamp_over_independent_streams() -> None:
    assert _source_timestamp_ns(_MixedTimestampSDK()) == (True, 100)


def test_scene_bridge_source_timestamp_gate_waits_when_all_available_getters_are_zero() -> None:
    assert _source_timestamp_ns(_SpecificTimestampSDK(0)) == (True, None)


def test_scene_bridge_source_timestamp_gate_does_not_coerce_strings() -> None:
    assert _source_timestamp_ns(_StringTimestampSDK()) == (True, None)


def _packet(**overrides: object) -> SceneXRPacket:
    values: dict[str, object] = {
        "sequence": 0,
        "timestamp_s": 1.0,
        "left_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "right_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
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


def test_scene_bridge_sample_signature_ignores_transport_metadata() -> None:
    first = _packet(sequence=1, timestamp_s=1.0)
    same_source = _packet(sequence=99, timestamp_s=2.0)
    assert _sample_signature(first) == _sample_signature(same_source)


def test_scene_bridge_sample_signature_keeps_controller_changes_with_coarse_timestamp() -> None:
    first = _packet(right_trigger=0.0)
    changed = _packet(right_trigger=0.75)
    assert _sample_signature(first) != _sample_signature(changed)


def test_scene_bridge_source_gate_sends_bounded_heartbeats_for_stuck_timestamp() -> None:
    """A stationary coarse SDK timestamp must not immediately stale the scene."""
    gate = _SourceFreshnessGate(
        heartbeat_interval_s=DEFAULT_SOURCE_HEARTBEAT_INTERVAL_S,
        stale_timeout_s=DEFAULT_SOURCE_STALE_TIMEOUT_S,
    )
    signature = ("stationary",)

    assert gate.should_emit(now_s=0.0, timestamp_ns=10, signature=signature)
    gate.commit(now_s=0.0, timestamp_ns=10, signature=signature)
    assert not gate.should_emit(
        now_s=DEFAULT_SOURCE_HEARTBEAT_INTERVAL_S * 0.5,
        timestamp_ns=10,
        signature=signature,
    )
    assert gate.should_emit(
        now_s=DEFAULT_SOURCE_HEARTBEAT_INTERVAL_S,
        timestamp_ns=10,
        signature=signature,
    )
    gate.commit(
        now_s=DEFAULT_SOURCE_HEARTBEAT_INTERVAL_S,
        timestamp_ns=10,
        signature=signature,
    )
    # Once the bounded liveness grace has elapsed, cached SDK data is no
    # longer allowed to refresh the receiver's arrival timer.
    assert not gate.should_emit(
        now_s=DEFAULT_SOURCE_STALE_TIMEOUT_S + DEFAULT_SOURCE_HEARTBEAT_INTERVAL_S,
        timestamp_ns=10,
        signature=signature,
    )


def test_scene_bridge_source_gate_accepts_payload_change_with_coarse_timestamp() -> None:
    """A changed controller payload is fresh even when its timestamp is stuck."""
    gate = _SourceFreshnessGate(heartbeat_interval_s=0.1, stale_timeout_s=0.2)
    first = ("pose-a",)
    second = ("pose-b",)
    gate.commit(now_s=0.0, timestamp_ns=10, signature=first)
    assert gate.should_emit(now_s=1.0, timestamp_ns=10, signature=second)
    gate.commit(now_s=1.0, timestamp_ns=10, signature=second)
    assert not gate.should_emit(now_s=1.05, timestamp_ns=10, signature=second)


@pytest.mark.parametrize(
    ("heartbeat", "stale"),
    [(0.0, 1.0), (float("nan"), 1.0), (0.2, 0.1), (0.1, float("inf"))],
)
def test_scene_bridge_source_gate_rejects_invalid_bounds(heartbeat: float, stale: float) -> None:
    with pytest.raises(ValueError):
        _SourceFreshnessGate(heartbeat_interval_s=heartbeat, stale_timeout_s=stale)


def test_scene_bridge_closes_sdk_when_udp_socket_startup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A post-init socket error must not strand PXREA background workers."""

    import scripts.run.run_scene_xr_bridge as bridge

    sdk = types.ModuleType("xrobotoolkit_sdk")
    sdk.initialized = False
    sdk.closed = False

    def init() -> None:
        sdk.initialized = True

    def close() -> None:
        sdk.closed = True

    sdk.init = init  # type: ignore[attr-defined]
    sdk.close = close  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", sdk)

    def fail_socket(*args: object, **kwargs: object) -> object:
        raise OSError("too many open files")

    monkeypatch.setattr(bridge.socket, "socket", fail_socket)
    monkeypatch.setattr(sys, "argv", ["run_scene_xr_bridge.py", "--sdk-close-timeout", "0.1"])

    with pytest.raises(OSError, match="too many open files"):
        bridge.main()

    assert sdk.initialized is True
    assert sdk.closed is True


def test_scene_bridge_closes_sdk_when_init_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Partial native initialization must not strand SDK workers."""

    import scripts.run.run_scene_xr_bridge as bridge

    sdk = types.ModuleType("xrobotoolkit_sdk")
    sdk.closed = False

    def init() -> None:
        raise RuntimeError("PXREAInit failed after starting workers")

    def close() -> None:
        sdk.closed = True

    sdk.init = init  # type: ignore[attr-defined]
    sdk.close = close  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", sdk)
    monkeypatch.setattr(sys, "argv", ["run_scene_xr_bridge.py", "--sdk-close-timeout", "0.1"])

    with pytest.raises(RuntimeError, match="PXREAInit failed"):
        bridge.main()

    assert sdk.closed is True


def test_scene_bridge_closes_socket_and_sdk_when_signal_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding the bridge outside the main thread must clean up resources."""

    import scripts.run.run_scene_xr_bridge as bridge

    sdk = types.ModuleType("xrobotoolkit_sdk")
    sdk.closed = False
    sdk.init = lambda: None  # type: ignore[attr-defined]

    def close() -> None:
        sdk.closed = True

    sdk.close = close  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", sdk)

    class _Sender:
        closed = False

        def close(self) -> None:
            self.closed = True

    sender = _Sender()
    monkeypatch.setattr(bridge.socket, "socket", lambda *args, **kwargs: sender)

    def fail_signal(*args: object, **kwargs: object) -> object:
        raise ValueError("signal only works in main thread")

    monkeypatch.setattr(bridge.signal, "signal", fail_signal)
    monkeypatch.setattr(sys, "argv", ["run_scene_xr_bridge.py", "--sdk-close-timeout", "0.1"])

    with pytest.raises(ValueError, match="signal only works"):
        bridge.main()

    assert sender.closed is True
    assert sdk.closed is True
