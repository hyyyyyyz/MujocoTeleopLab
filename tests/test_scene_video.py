from __future__ import annotations

import queue
import socket
import struct
import threading
import time

import numpy as np
import pytest

import teleopit.scenes.video as scene_video
from teleopit.scenes.video import (
    DEFAULT_SCENE_VIDEO_FPS,
    DEFAULT_SCENE_VIDEO_HEIGHT,
    DEFAULT_SCENE_VIDEO_WIDTH,
)
from teleopit.scenes.xr_video_transport import DirectXRoboToolkitVideoTransport


def _recv_exact(peer: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = peer.recv(remaining)
        if not chunk:
            raise ConnectionError("scene video peer closed before a complete frame arrived")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def test_scene_video_transport_connects_to_direct_listener() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(2.0)
    port = int(listener.getsockname()[1])
    transport = DirectXRoboToolkitVideoTransport(
        host="127.0.0.1", port=port, width=128, height=48, fps=30
    )
    try:
        transport.start()
        peer, _ = listener.accept()
        try:
            deadline = time.monotonic() + 1.0
            while not transport.is_connected and time.monotonic() < deadline:
                time.sleep(0.01)
            assert transport.is_connected
            assert transport.last_connect_error is None
        finally:
            peer.close()
    finally:
        transport.stop()
        listener.close()


def test_scene_video_transport_sends_length_framed_h264() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(2.0)
    port = int(listener.getsockname()[1])
    transport = DirectXRoboToolkitVideoTransport(
        host="127.0.0.1", port=port, width=128, height=48, fps=30
    )
    try:
        transport.start()
        peer, _ = listener.accept()
        try:
            peer.settimeout(2.0)
            transport.publish_frame(np.zeros((48, 64, 3), dtype=np.uint8))
            header = _recv_exact(peer, 4)
            (size,) = struct.unpack(">I", header)
            payload = _recv_exact(peer, size)
            assert payload.startswith(b"\x00\x00\x00\x01")
            assert transport.frames_sent >= 1
            # A direct MediaDecoder session must receive one H.264 access unit
            # per newly rendered camera frame, not synthetic repeated frames.
            peer.settimeout(0.15)
            try:
                extra = peer.recv(1)
            except socket.timeout:
                extra = None
            assert extra is None
        finally:
            peer.close()
    finally:
        transport.stop()
        listener.close()


def test_scene_video_transport_repeats_headers_and_emits_short_recovery_idr() -> None:
    """A reconnecting Pico decoder must not wait seconds for a keyframe."""
    import av

    transport = DirectXRoboToolkitVideoTransport(
        host="127.0.0.1", port=12345, width=128, height=48, fps=60
    )
    encoder = av.CodecContext.create("libx264", "w")
    encoder.width = 128
    encoder.height = 48
    encoder.pix_fmt = "yuv420p"
    encoder.framerate = 60
    encoder.bit_rate = 4_000_000
    encoder.options = {
        "preset": "ultrafast",
        "tune": "zerolatency",
        "profile": "baseline",
        "x264-params": "annexb=1:repeat-headers=1:scenecut=0:keyint=15:min-keyint=15:bframes=0",
    }
    encoder.open()
    try:
        packets: list[bytes] = []
        for index in range(16):
            frame = np.full((48, 64, 3), index * 7, dtype=np.uint8)
            packets.extend(transport._encode(encoder, frame))
        # The first packet and the packet at the 15-frame recovery interval
        # both carry SPS/PPS (NAL types 7/8) before the IDR.  This mirrors
        # SIMPLE's insert-sps-pps=true, idrinterval=15 profile.
        assert packets

        def nal_types(payload: bytes) -> set[int]:
            starts: list[int] = []
            offset = 0
            while True:
                start = payload.find(b"\x00\x00\x00\x01", offset)
                if start < 0:
                    break
                starts.append(start + 4)
                offset = start + 4
            return {payload[start] & 0x1F for start in starts if start < len(payload)}

        assert {7, 8}.issubset(nal_types(packets[0]))
        assert {7, 8}.issubset(nal_types(packets[15]))
    finally:
        # PyAV codec contexts are released by reference counting; unlike a
        # renderer they do not expose ``close()`` on all supported versions.
        del encoder
        transport.stop()


def test_scene_video_transport_closes_connection_finished_after_stop(monkeypatch) -> None:
    """A connect completing during shutdown must not resurrect the stream."""

    entered = threading.Event()
    release = threading.Event()

    class FakeSocket:
        def __init__(self) -> None:
            self.closed = False

        def setsockopt(self, *_args: object) -> None:
            pass

        def settimeout(self, _value: float | None) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    media = FakeSocket()

    def delayed_connect(*_args: object, **_kwargs: object) -> FakeSocket:
        entered.set()
        assert release.wait(1.0)
        return media

    monkeypatch.setattr(
        "teleopit.scenes.xr_video_transport.socket.create_connection", delayed_connect
    )
    transport = DirectXRoboToolkitVideoTransport(
        host="127.0.0.1", port=12345, width=128, height=48, fps=30
    )
    connector = threading.Thread(target=transport._connect)
    connector.start()
    assert entered.wait(1.0)
    transport.stop()
    release.set()
    connector.join(timeout=1.0)

    assert not connector.is_alive()
    assert media.closed
    assert not transport.is_connected


def test_scene_video_transport_closes_socket_when_connection_setup_fails(monkeypatch) -> None:
    """A post-connect socket option failure must not leak the media socket."""

    class FakeSocket:
        def __init__(self) -> None:
            self.closed = False

        def setsockopt(self, *_args: object) -> None:
            raise OSError("TCP_NODELAY unavailable")

        def settimeout(self, _value: float | None) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    media = FakeSocket()
    monkeypatch.setattr(
        "teleopit.scenes.xr_video_transport.socket.create_connection",
        lambda *_args, **_kwargs: media,
    )
    transport = DirectXRoboToolkitVideoTransport(
        host="127.0.0.1", port=12345, width=128, height=48, fps=30
    )
    transport._connect()
    assert media.closed is True
    assert transport.is_connected is False
    assert transport.last_connect_error == "TCP_NODELAY unavailable"


def test_scene_video_transport_handles_non_oserror_connection_setup_failure(monkeypatch) -> None:
    """Unexpected socket-option errors still enter the reconnect path."""

    class FakeSocket:
        closed = False

        def setsockopt(self, *_args: object) -> None:
            raise ValueError("invalid socket state")

        def close(self) -> None:
            self.closed = True

    media = FakeSocket()
    monkeypatch.setattr(
        "teleopit.scenes.xr_video_transport.socket.create_connection",
        lambda *_args, **_kwargs: media,
    )
    transport = DirectXRoboToolkitVideoTransport(
        host="127.0.0.1", port=12345, width=128, height=48, fps=30
    )
    transport._connect()
    assert media.closed is True
    assert transport.last_connect_error == "invalid socket state"


def test_scene_video_transport_drops_frames_published_while_stopped() -> None:
    """A restarted sender must not replay a frame from the previous session."""

    transport = DirectXRoboToolkitVideoTransport(
        host="127.0.0.1", port=12345, width=128, height=48, fps=30
    )
    transport.publish_frame(np.zeros((48, 64, 3), dtype=np.uint8))
    transport.stop()
    assert transport._frames.empty()

    # ``publish_frame`` is frequently called from the render thread while the
    # control thread is shutting down.  The stopped guard must prevent a late
    # producer from repopulating the queue after the cleanup drain.
    transport.publish_frame(np.ones((48, 64, 3), dtype=np.uint8))
    assert transport._frames.empty()


def test_scene_remote_vision_start_rolls_back_transport_on_thread_failure(monkeypatch) -> None:
    """A failed renderer-thread start must not leave a live TCP worker behind."""

    class FakeTransport:
        endpoint = ("127.0.0.1", 12345)

        def __init__(self) -> None:
            self.started = False
            self.stopped = False

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.stopped = True

    class FailingThread:
        def __init__(self, **_: object) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("thread resources unavailable")

    transport = FakeTransport()
    vision = scene_video.SceneRemoteVision.__new__(scene_video.SceneRemoteVision)
    vision._render_thread = None
    vision._render_stop = threading.Event()
    vision._lifecycle_lock = threading.Lock()
    vision._render_error = None
    vision._next_frame_s = 99.0
    vision._transport = transport

    monkeypatch.setattr(scene_video.threading, "Thread", FailingThread)
    with pytest.raises(RuntimeError, match="thread resources"):
        vision.start()

    assert transport.started is True
    assert transport.stopped is True
    assert vision._render_thread is None
    assert vision._render_stop.is_set()


def test_scene_remote_vision_restarts_after_render_worker_exits(monkeypatch) -> None:
    """A dead renderer handle must not permanently disable Remote Vision."""

    class FakeTransport:
        endpoint = ("127.0.0.1", 12345)

        def __init__(self) -> None:
            self.start_calls = 0
            self.stop_calls = 0

        def start(self) -> None:
            self.start_calls += 1

        def stop(self) -> None:
            self.stop_calls += 1

    class FakeThread:
        def __init__(self, **_: object) -> None:
            self.started = False

        def start(self) -> None:
            self.started = True

        def is_alive(self) -> bool:
            return self.started

    transport = FakeTransport()
    vision = scene_video.SceneRemoteVision.__new__(scene_video.SceneRemoteVision)
    old_thread = FakeThread()
    vision._render_thread = old_thread
    vision._render_stop = threading.Event()
    vision._lifecycle_lock = threading.Lock()
    vision._render_error = RuntimeError("renderer failed")
    vision._next_frame_s = 99.0
    vision._snapshots = queue.Queue(maxsize=1)
    vision._snapshots.put_nowait(np.zeros(1, dtype=np.float64))
    vision._transport = transport

    monkeypatch.setattr(scene_video.threading, "Thread", FakeThread)
    vision.start()

    assert transport.stop_calls == 1
    assert transport.start_calls == 1
    assert vision._render_thread is not old_thread
    assert vision._render_thread is not None and vision._render_thread.is_alive()
    assert vision._render_stop.is_set() is False
    assert vision._render_error is None
    assert vision._snapshots.empty()


def test_scene_remote_vision_stop_retains_live_worker_after_join_timeout() -> None:
    """A stuck renderer must remain owned by the stopped generation.

    OpenGL/codec calls are outside Python's control and may outlive the
    bounded shutdown join.  Clearing the handle in that case would let a
    subsequent ``start`` clear the shared stop event and create a second
    renderer while the first still owns thread-affine state.
    """

    class FakeTransport:
        def __init__(self) -> None:
            self.stop_calls = 0

        def stop(self) -> None:
            self.stop_calls += 1

    class StuckThread:
        def __init__(self) -> None:
            self.join_calls: list[float] = []

        def join(self, timeout: float | None = None) -> None:
            self.join_calls.append(float(timeout) if timeout is not None else -1.0)

        def is_alive(self) -> bool:
            return True

    transport = FakeTransport()
    worker = StuckThread()
    vision = scene_video.SceneRemoteVision.__new__(scene_video.SceneRemoteVision)
    vision._render_thread = worker
    vision._render_stop = threading.Event()
    vision._lifecycle_lock = threading.Lock()
    vision._snapshots = queue.Queue(maxsize=1)
    vision._snapshots.put_nowait(np.zeros(1, dtype=np.float64))
    vision._transport = transport

    vision.stop()

    assert worker.join_calls == [2.0]
    assert vision._render_thread is worker
    assert vision._render_stop.is_set()
    assert transport.stop_calls == 1
    assert vision._snapshots.empty()


def test_scene_video_defaults_match_simple_zedmini_profile() -> None:
    """The scene's default is one 1280x720 eye at SIMPLE's 60 Hz profile."""

    assert (DEFAULT_SCENE_VIDEO_WIDTH, DEFAULT_SCENE_VIDEO_HEIGHT, DEFAULT_SCENE_VIDEO_FPS) == (
        1280,
        720,
        60,
    )
