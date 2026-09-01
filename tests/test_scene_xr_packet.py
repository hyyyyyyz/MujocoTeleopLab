from __future__ import annotations

import socket
import time

import pytest

from teleopit.scenes.xr_packet import SceneXRPacket, SceneXRReceiver


_SESSION_UNSET = object()


def _packet(sequence: int, *, session_id: object = _SESSION_UNSET) -> SceneXRPacket:
    values = {
        "sequence": sequence,
        "timestamp_s": float(sequence),
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
    if session_id is not _SESSION_UNSET:
        values["session_id"] = session_id
    return SceneXRPacket.from_mapping(values)


def _receiver_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    return port


def _wait_for_packets(receiver: SceneXRReceiver, count: int) -> None:
    deadline = time.monotonic() + 1.0
    while receiver.packet_count < count and time.monotonic() < deadline:
        receiver.poll()
        time.sleep(0.005)
    receiver.poll()
    assert receiver.packet_count == count


def test_scene_xr_receiver_counts_fresh_new_packets() -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    receiver = SceneXRReceiver("127.0.0.1", port)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sender.sendto(_packet(2).to_wire(), ("127.0.0.1", port))
        sender.sendto(_packet(1).to_wire(), ("127.0.0.1", port))
        deadline = time.monotonic() + 1.0
        while receiver.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert receiver.latest is not None
        assert receiver.latest.sequence == 2
        assert receiver.packet_count == 1
        assert receiver.age_s() is not None
        assert receiver.is_fresh(1.0)
    finally:
        sender.close()
        receiver.close()


def test_scene_xr_receiver_ignores_malformed_mapping_and_keeps_polling() -> None:
    """A bad UDP datagram must not take down the scene control loop."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    receiver = SceneXRReceiver("127.0.0.1", port)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sender.sendto(b'{"sequence":0}', ("127.0.0.1", port))
        sender.sendto(_packet(1).to_wire(), ("127.0.0.1", port))
        deadline = time.monotonic() + 1.0
        while receiver.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert receiver.latest is not None
        assert receiver.latest.sequence == 1
        assert receiver.packet_count == 1
    finally:
        sender.close()
        receiver.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("left_trigger", -0.01),
        ("right_trigger", 1.01),
        ("left_grip", float("nan")),
        ("right_grip", float("inf")),
    ],
)
def test_scene_xr_packet_rejects_invalid_unit_interval_values(field: str, value: float) -> None:
    payload = {
        "sequence": 1,
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
    payload[field] = value
    with pytest.raises(ValueError, match=field):
        SceneXRPacket.from_mapping(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("left_axis", [-1.01, 0.0]),
        ("right_axis", [0.0, 1.01]),
    ],
)
def test_scene_xr_packet_rejects_out_of_range_axis_values(field: str, value: list[float]) -> None:
    payload = {
        "sequence": 1,
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
    payload[field] = value
    with pytest.raises(ValueError, match=field):
        SceneXRPacket.from_mapping(payload)


def test_scene_xr_receiver_ignores_out_of_range_datagram() -> None:
    """Out-of-range analogue values must not replace the last valid sample."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    receiver = SceneXRReceiver("127.0.0.1", port)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sender.sendto(_packet(1).to_wire(), ("127.0.0.1", port))
        malformed = _packet(2).to_wire().replace(b'"right_trigger":0.0', b'"right_trigger":1.5')
        sender.sendto(malformed, ("127.0.0.1", port))
        deadline = time.monotonic() + 1.0
        while receiver.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert receiver.latest is not None
        assert receiver.latest.sequence == 1
        assert receiver.packet_count == 1
    finally:
        sender.close()
        receiver.close()


def test_scene_xr_receiver_ignores_out_of_range_axis_datagram() -> None:
    """Axis values outside [-1, 1] must not replace the last valid sample."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    receiver = SceneXRReceiver("127.0.0.1", port)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sender.sendto(_packet(1).to_wire(), ("127.0.0.1", port))
        malformed = _packet(2).to_wire().replace(b'"right_axis":[0.0,0.0]', b'"right_axis":[1.5,0.0]')
        sender.sendto(malformed, ("127.0.0.1", port))
        deadline = time.monotonic() + 1.0
        while receiver.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert receiver.latest is not None
        assert receiver.latest.sequence == 1
        assert receiver.packet_count == 1
    finally:
        sender.close()
        receiver.close()


def test_scene_xr_receiver_accepts_sequence_reset_after_bridge_restart() -> None:
    """A restarted bridge starts at sequence zero in a new generation."""
    port = _receiver_port()
    receiver = SceneXRReceiver("127.0.0.1", port)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sender.sendto(_packet(41, session_id="bridge-a").to_wire(), ("127.0.0.1", port))
        sender.sendto(_packet(0, session_id="bridge-b").to_wire(), ("127.0.0.1", port))
        _wait_for_packets(receiver, 2)
        assert receiver.latest is not None
        assert receiver.latest.session_id == "bridge-b"
        assert receiver.latest.sequence == 0
        assert receiver.session_id == "bridge-b"
    finally:
        sender.close()
        receiver.close()


def test_scene_xr_receiver_rejects_delayed_packets_from_retired_generation() -> None:
    """Old UDP datagrams must not roll a new bridge session back."""
    port = _receiver_port()
    receiver = SceneXRReceiver("127.0.0.1", port)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for packet in (
            _packet(10, session_id="bridge-a"),
            _packet(0, session_id="bridge-b"),
            _packet(11, session_id="bridge-a"),
            _packet(1, session_id="bridge-b"),
        ):
            sender.sendto(packet.to_wire(), ("127.0.0.1", port))
        _wait_for_packets(receiver, 3)
        assert receiver.latest is not None
        assert receiver.latest.session_id == "bridge-b"
        assert receiver.latest.sequence == 1
    finally:
        sender.close()
        receiver.close()


def test_scene_xr_receiver_close_is_idempotent_and_pollable() -> None:
    port = _receiver_port()
    receiver = SceneXRReceiver("127.0.0.1", port)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sender.sendto(_packet(1).to_wire(), ("127.0.0.1", port))
        _wait_for_packets(receiver, 1)
        latest = receiver.latest
        receiver.close()
        receiver.close()
        assert receiver.poll() is latest
        assert receiver.packet_count == 1
    finally:
        sender.close()
        # The explicit calls above are intentionally idempotent; this final
        # call also keeps the test safe if an assertion fails before close.
        receiver.close()


@pytest.mark.parametrize("host", ["", "   ", None, 123])
def test_scene_xr_receiver_rejects_empty_or_non_string_host(host: object) -> None:
    with pytest.raises(ValueError, match="host"):
        SceneXRReceiver(host=host, port=_receiver_port())  # type: ignore[arg-type]


def test_scene_xr_receiver_is_fresh_is_false_after_close() -> None:
    receiver = SceneXRReceiver("127.0.0.1", _receiver_port())
    try:
        assert receiver.closed is False
        receiver.close()
        assert receiver.closed is True
        assert receiver.is_fresh(1.0) is False
    finally:
        receiver.close()


@pytest.mark.parametrize("max_age_s", [-1.0, float("nan"), float("inf"), "1.0"])
def test_scene_xr_receiver_rejects_invalid_freshness_limit(max_age_s: object) -> None:
    receiver = SceneXRReceiver("127.0.0.1", _receiver_port())
    try:
        with pytest.raises(ValueError, match="max_age_s"):
            receiver.is_fresh(max_age_s)  # type: ignore[arg-type]
    finally:
        receiver.close()


def test_scene_xr_receiver_closes_socket_when_bind_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A constructor-time bind error must not leak the UDP descriptor."""

    class _Socket:
        def __init__(self) -> None:
            self.closed = False

        def setsockopt(self, *_args: object) -> None:
            pass

        def bind(self, _address: object) -> None:
            raise OSError("address already in use")

        def setblocking(self, _value: bool) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    receiver_socket = _Socket()
    monkeypatch.setattr("teleopit.scenes.xr_packet.socket.socket", lambda *args, **kwargs: receiver_socket)
    with pytest.raises(OSError, match="address already in use"):
        SceneXRReceiver("127.0.0.1", 17600)
    assert receiver_socket.closed is True


@pytest.mark.parametrize("sequence", [1.5, True, "1", -1, 2**63])
def test_scene_xr_packet_rejects_non_integer_or_negative_sequence(sequence: object) -> None:
    with pytest.raises(ValueError, match="sequence"):
        _packet(sequence)  # type: ignore[arg-type]


@pytest.mark.parametrize("session_id", ["", 1, None, "x" * 129, "\ud800"])
def test_scene_xr_packet_rejects_invalid_session_id(session_id: object) -> None:
    with pytest.raises(ValueError, match="session_id"):
        _packet(1, session_id=session_id)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("timestamp_s", "1.0"),
        ("left_trigger", "0.5"),
        ("right_grip", True),
        ("left_axis", ["0", 0.0]),
        ("right_pose", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "1.0"]),
    ],
)
def test_scene_xr_packet_rejects_coercible_non_numeric_wire_values(field: str, value: object) -> None:
    payload = {
        "sequence": 1,
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
    payload[field] = value
    with pytest.raises(ValueError, match=field):
        SceneXRPacket.from_mapping(payload)


@pytest.mark.parametrize("field", ["a", "b", "x", "y", "left_menu"])
@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, [], [False]])
def test_scene_xr_packet_rejects_non_boolean_button_values(field: str, value: object) -> None:
    payload = {
        "sequence": 1,
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
    payload[field] = value
    with pytest.raises(ValueError, match=field):
        SceneXRPacket.from_mapping(payload)
