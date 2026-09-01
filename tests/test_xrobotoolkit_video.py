from __future__ import annotations

import struct
import threading
import time

import numpy as np
import pytest

from teleopit.inputs.xrobotoolkit_video import (
    _CameraRequest,
    _XRoboToolkitVideoTransport,
    XRoboToolkitVideoRuntime,
    _encode_h264,
    _parse_camera_request,
    _parse_network_message,
)
from teleopit.inputs.pico_video import PicoVideoConfig


def test_xrobotoolkit_video_runtime_stops_transport_when_renderer_close_fails() -> None:
    """A renderer shutdown error must not leave the reconnect worker alive."""
    class _FailingProducer:
        def stop(self) -> None:
            raise RuntimeError("renderer close failed")

    class _Transport:
        def __init__(self) -> None:
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True

    transport = _Transport()
    runtime = XRoboToolkitVideoRuntime.__new__(XRoboToolkitVideoRuntime)
    runtime._producer = _FailingProducer()
    runtime._transport = transport

    with pytest.raises(RuntimeError, match="renderer close failed"):
        runtime.stop()
    assert transport.stopped is True


def test_xrobotoolkit_video_runtime_rolls_back_producer_when_start_fails() -> None:
    """A partially-created MuJoCo renderer is closed on startup failure."""

    class _FailingProducer:
        def __init__(self) -> None:
            self.stop_calls = 0

        def start(self) -> None:
            raise RuntimeError("renderer start failed")

        def stop(self) -> None:
            self.stop_calls += 1

    class _Transport:
        def __init__(self) -> None:
            self.started = False
            self.stopped = False

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.stopped = True

    producer = _FailingProducer()
    transport = _Transport()
    runtime = XRoboToolkitVideoRuntime.__new__(XRoboToolkitVideoRuntime)
    runtime._producer = producer
    runtime._transport = transport

    with pytest.raises(RuntimeError, match="renderer start failed"):
        runtime.start()

    assert transport.started is True
    assert transport.stopped is True
    assert producer.stop_calls == 1


def _camera_payload(*, host: str = "10.0.0.2", port: int = 12345) -> bytes:
    source = b"ZED"
    address = host.encode("utf-8")
    return (
        b"\xca\xfe\x01"
        + struct.pack("<7i", 64, 48, 30, 500_000, 0, 2, port)
        + bytes([len(source)])
        + source
        + bytes([len(address)])
        + address
    )


def test_xrobotoolkit_video_parses_open_camera_request() -> None:
    camera = _camera_payload()
    command = b"OPEN_CAMERA"
    message = struct.pack("<i", len(command)) + command + struct.pack("<i", len(camera)) + camera

    parsed_command, parsed_data = _parse_network_message(message)
    request = _parse_camera_request(parsed_data)

    assert parsed_command == "OPEN_CAMERA"
    assert request == _CameraRequest(width=64, height=48, fps=30, bitrate=500_000, host="10.0.0.2", port=12345)


def test_xrobotoolkit_video_accepts_nul_padded_control_command() -> None:
    """SIMPLE sends fixed-width command fields with terminal NUL padding."""

    camera = _camera_payload()
    command = b"OPEN_CAMERA\x00\x00"
    message = struct.pack("<i", len(command)) + command + struct.pack("<i", len(camera)) + camera

    parsed_command, parsed_data = _parse_network_message(message)

    assert parsed_command == "OPEN_CAMERA"
    assert _parse_camera_request(parsed_data).port == 12345


def test_xrobotoolkit_video_rejects_trailing_control_bytes() -> None:
    camera = _camera_payload()
    command = b"OPEN_CAMERA"
    message = (
        struct.pack("<i", len(command))
        + command
        + struct.pack("<i", len(camera))
        + camera
        + b"unexpected-trailer"
    )
    with pytest.raises(ConnectionError, match="payload length"):
        _parse_network_message(message)


def test_xrobotoolkit_video_rejects_invalid_utf8_control_fields() -> None:
    malformed_command = struct.pack("<i", 1) + b"\xff" + struct.pack("<i", 0)
    with pytest.raises(ConnectionError, match="command encoding"):
        _parse_network_message(malformed_command)

    # Keep the command valid and corrupt the source compact string.  The
    # parser must report a protocol error rather than leaking UnicodeDecodeError
    # out of the control thread.
    malformed_camera = (
        b"\xca\xfe\x01"
        + struct.pack("<7i", 64, 48, 30, 500_000, 0, 2, 12345)
        + b"\x01\xff"
        + b"\x00"
    )
    with pytest.raises(ConnectionError, match="compact string encoding"):
        _parse_camera_request(malformed_camera)


def test_xrobotoolkit_video_parsers_accept_bytes_like_payloads() -> None:
    camera = _camera_payload()
    command = b"OPEN_CAMERA"
    message = struct.pack("<i", len(command)) + command + struct.pack("<i", len(camera)) + camera

    parsed_command, parsed_data = _parse_network_message(memoryview(message))

    assert parsed_command == "OPEN_CAMERA"
    assert _parse_camera_request(bytearray(parsed_data)).host == "10.0.0.2"


def test_xrobotoolkit_video_encoder_emits_annex_b_h264() -> None:
    import av
    from fractions import Fraction

    encoder = av.CodecContext.create("libx264", "w")
    encoder.width = 64
    encoder.height = 48
    encoder.pix_fmt = "yuv420p"
    encoder.time_base = Fraction(1, 30)
    encoder.framerate = Fraction(30, 1)
    encoder.options = {"preset": "ultrafast", "tune": "zerolatency", "x264-params": "repeat-headers=1"}
    encoder.open()

    packets = _encode_h264(
        encoder,
        np.zeros((48, 64, 3), dtype=np.uint8),
        _CameraRequest(width=64, height=48, fps=30, bitrate=500_000, host="127.0.0.1", port=1),
    )

    assert packets
    assert packets[0].startswith(b"\x00\x00\x00\x01")


def test_xrobotoolkit_video_stop_closes_socket_during_blocked_send(monkeypatch) -> None:
    """A backpressured media write must not delay transport shutdown."""

    send_started = threading.Event()
    socket_closed = threading.Event()

    class BlockingSocket:
        def sendall(self, _payload: bytes) -> None:
            send_started.set()
            # ``stop`` closes this socket from the control thread. Model the
            # kernel waking a blocked send with an error once that happens.
            if not socket_closed.wait(1.0):
                raise AssertionError("test socket was not closed by stop()")
            raise OSError("socket closed")

        def close(self) -> None:
            socket_closed.set()

    transport = _XRoboToolkitVideoTransport()
    media = BlockingSocket()
    request = _CameraRequest(width=64, height=48, fps=30, bitrate=500_000, host="127.0.0.1", port=12345)
    with transport._media_lock:
        transport._media_socket = media  # type: ignore[assignment]
        transport._request = request
        transport._encoder = object()
    monkeypatch.setattr(
        "teleopit.inputs.xrobotoolkit_video._encode_h264",
        lambda _encoder, _frame, _request: [b"encoded"],
    )

    transport._stop.clear()
    worker = threading.Thread(target=transport._encode_frames)
    transport._encode_thread = worker
    worker.start()
    transport.publish_frame(np.zeros((48, 64, 3), dtype=np.uint8), time.monotonic())
    assert send_started.wait(1.0)

    started = time.monotonic()
    transport.stop()
    elapsed = time.monotonic() - started

    assert elapsed < 1.0
    assert not worker.is_alive()
    assert socket_closed.is_set()


def test_xrobotoolkit_video_stop_drains_frames_for_next_session() -> None:
    """A stopped legacy sender must never replay a queued old camera frame."""
    transport = _XRoboToolkitVideoTransport()
    transport.publish_frame(np.zeros((48, 64, 3), dtype=np.uint8), time.monotonic())
    assert not transport._frames.empty()

    transport.stop()
    assert transport._frames.empty()

    # The producer may race the stop flag; even a late publish is ignored and
    # a subsequent start begins with an empty queue.
    transport.publish_frame(np.ones((48, 64, 3), dtype=np.uint8), time.monotonic())
    assert transport._frames.empty()


def test_xrobotoolkit_video_restarts_after_encoder_worker_exits(monkeypatch) -> None:
    """A dead encoder handle must not permanently disable video restart."""
    request = _CameraRequest(
        width=128,
        height=48,
        fps=30,
        bitrate=500_000,
        host="127.0.0.1",
        port=12345,
    )
    transport = _XRoboToolkitVideoTransport(direct_request=request)
    # Keep this lifecycle test independent of a real Pico listener and make
    # both workers deterministic.
    monkeypatch.setattr(transport, "_ensure_direct_connection", lambda: None)
    monkeypatch.setattr(transport, "_maintain_direct_connection", lambda: None)
    monkeypatch.setattr(transport, "_encode_frames", lambda: None)

    transport.start()
    first_thread = transport._encode_thread
    assert first_thread is not None
    first_thread.join(timeout=1.0)
    assert not first_thread.is_alive()

    transport.start()
    second_thread = transport._encode_thread
    assert second_thread is not None
    assert second_thread is not first_thread
    transport.stop()


def test_xrobotoolkit_video_old_socket_failure_preserves_new_connection() -> None:
    """A stale send failure must not clear a socket installed by Listen."""

    class FakeSocket:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    transport = _XRoboToolkitVideoTransport()
    old_media = FakeSocket()
    new_media = FakeSocket()
    new_request = _CameraRequest(width=128, height=48, fps=30, bitrate=500_000, host="127.0.0.1", port=12346)
    with transport._media_lock:
        transport._media_socket = new_media  # type: ignore[assignment]
        transport._request = new_request
        transport._encoder = "new-encoder"

    transport._drop_media_if_current(old_media)  # type: ignore[arg-type]

    assert old_media.closed
    with transport._media_lock:
        assert transport._media_socket is new_media  # type: ignore[comparison-overlap]
        assert transport._request == new_request
        assert transport._encoder == "new-encoder"


def test_xrobotoolkit_video_encoder_duplicates_single_view_for_stereo_request() -> None:
    import av
    from fractions import Fraction

    encoder = av.CodecContext.create("libx264", "w")
    encoder.width = 128
    encoder.height = 48
    encoder.pix_fmt = "yuv420p"
    encoder.time_base = Fraction(1, 30)
    encoder.framerate = Fraction(30, 1)
    encoder.options = {"preset": "ultrafast", "tune": "zerolatency", "profile": "baseline", "x264-params": "annexb=1"}
    encoder.open()

    packets = _encode_h264(
        encoder,
        np.zeros((48, 64, 3), dtype=np.uint8),
        _CameraRequest(width=128, height=48, fps=30, bitrate=500_000, host="127.0.0.1", port=1),
    )

    assert packets
    assert packets[0].startswith(b"\x00\x00\x00\x01")


def test_xrobotoolkit_video_direct_mode_does_not_bind_legacy_control_port(monkeypatch) -> None:
    """A direct Pico listener must not claim the legacy TCP 13579 port."""

    request = _CameraRequest(
        width=128,
        height=48,
        fps=30,
        bitrate=500_000,
        host="127.0.0.1",
        port=12345,
    )
    transport = _XRoboToolkitVideoTransport(direct_request=request)
    listener_bind_attempts: list[tuple[object, object]] = []

    # If the implementation accidentally creates the legacy listener this
    # fake socket makes the test fail immediately, without depending on the
    # machine's current use of port 13579.
    class UnexpectedListener:
        def setsockopt(self, *_args: object) -> None:
            pass

        def bind(self, address: tuple[object, object]) -> None:
            listener_bind_attempts.append(address)

        def listen(self, *_args: object) -> None:
            pass

        def settimeout(self, *_args: object) -> None:
            pass

        def close(self) -> None:
            pass

    def fail_if_listener(*_args: object, **_kwargs: object) -> UnexpectedListener:
        return UnexpectedListener()

    monkeypatch.setattr("teleopit.inputs.xrobotoolkit_video.socket.socket", fail_if_listener)
    # Prevent the direct probe from making a real connection; the worker will
    # simply retry in the background until stop() is called.
    monkeypatch.setattr(
        "teleopit.inputs.xrobotoolkit_video.socket.create_connection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("offline")),
    )
    transport.start()
    try:
        assert transport._control_listener is None
        assert listener_bind_attempts == []
    finally:
        transport.stop()


def test_xrobotoolkit_video_closes_socket_when_connection_setup_fails(monkeypatch) -> None:
    """A socket option failure must not leak the newly-created media socket."""

    class FakeSocket:
        closed = False

        def settimeout(self, _value: float) -> None:
            raise ValueError("invalid timeout")

        def setsockopt(self, *_args: object) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    media = FakeSocket()
    monkeypatch.setattr(
        "teleopit.inputs.xrobotoolkit_video.socket.create_connection",
        lambda *_args, **_kwargs: media,
    )
    transport = _XRoboToolkitVideoTransport()
    request = _CameraRequest(width=64, height=48, fps=30, bitrate=500_000, host="127.0.0.1", port=12345)

    transport._connect_media(request)

    assert media.closed is True
    with transport._media_lock:
        assert transport._media_socket is None


@pytest.mark.parametrize(
    ("config_kwargs", "direct_port", "direct_host", "message"),
    [
        ({"width": 1280.5}, 12345, "127.0.0.1", "width must be an integer"),
        ({"height": 720.5}, 12345, "127.0.0.1", "height must be an integer"),
        ({}, 12345.5, "127.0.0.1", "port must be an integer"),
        ({}, 12345, 123, "target host must be a string"),
    ],
)
def test_xrobotoolkit_video_direct_config_rejects_implicit_coercion(
    monkeypatch: pytest.MonkeyPatch,
    config_kwargs: dict[str, object],
    direct_port: object,
    direct_host: object,
    message: str,
) -> None:
    class DummyProducer:
        def __init__(self, **_kwargs: object) -> None:
            pass

    monkeypatch.setattr(
        "teleopit.inputs.xrobotoolkit_video._MujocoCameraVideoProducer",
        DummyProducer,
    )
    config = PicoVideoConfig(enabled=True, source="mujoco", **config_kwargs)

    with pytest.raises(ValueError, match=message):
        XRoboToolkitVideoRuntime(
            config=config,
            robot=None,
            direct_host=direct_host,  # type: ignore[arg-type]
            direct_port=direct_port,  # type: ignore[arg-type]
        )
