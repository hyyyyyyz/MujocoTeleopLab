"""Direct XRoboToolkit Remote Vision sender for scene teleoperation.

Recent headset firmware exposes a TCP H.264 listener directly.  Keeping this
small sender in the scene package avoids importing Teleopit's full-body input
package into the isolated Python 3.10 WBC environment.
"""

from __future__ import annotations

import queue
import operator
import socket
import struct
import threading
import time

import numpy as np


def _validated_transport_int(value: object, name: str, *, even: bool = False) -> int:
    """Validate a direct-listener transport integer before worker startup."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"Remote Vision {name} must be a positive integer")
    try:
        integer = int(operator.index(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Remote Vision {name} must be a positive integer") from exc
    if integer <= 0:
        raise ValueError(f"Remote Vision {name} must be positive")
    if even and integer % 2:
        raise ValueError(f"Remote Vision {name} must be even for yuv420p")
    return integer


class DirectXRoboToolkitVideoTransport:
    """Best-effort, reconnecting H.264 sender for the headset's direct port.

    Each camera frame is encoded at most once.  The headset's TCP MediaDecoder
    has a small queue, so replaying the latest source frame to force a nominal
    FPS would eventually fill the queue and stall a healthy session.
    """

    def __init__(self, *, host: str, port: int, width: int, height: int, fps: int) -> None:
        self._host = str(host).strip()
        if not self._host:
            raise ValueError("Remote Vision host must not be empty")
        self._port = _validated_transport_int(port, "port")
        if self._port > 65535:
            raise ValueError("Remote Vision port must be in [1, 65535]")
        self._width = _validated_transport_int(width, "width", even=True)
        self._height = _validated_transport_int(height, "height", even=True)
        self._fps = _validated_transport_int(fps, "FPS")
        self._stop = threading.Event()
        # Keep only the newest encoded source frame.  If a Pico decoder or
        # Wi-Fi link backpressures ``sendall``, retaining a second stale frame
        # would add another full video period of avoidable view latency.
        self._frames: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        # ``_connect`` runs in the encoder thread while ``stop`` and the
        # status properties may run from the MuJoCo/control thread.  Protect
        # the socket hand-off so a connection that finishes while shutdown is
        # in progress cannot be installed after ``_disconnect`` has already
        # run (which would leak that socket and leave a stale connected state).
        self._media_lock = threading.Lock()
        self._media: socket.socket | None = None
        self._encoder: object | None = None
        self._next_connect_s = 0.0
        self._last_connect_error: str | None = None
        self._frames_sent = 0

    @property
    def is_connected(self) -> bool:
        """Whether the headset currently accepts the scene video stream."""
        with self._media_lock:
            return self._media is not None

    @property
    def endpoint(self) -> tuple[str, int]:
        """Configured Pico direct-listener endpoint."""
        return self._host, self._port

    @property
    def frames_sent(self) -> int:
        """Number of encoded video frames sent in the current process."""
        return int(self._frames_sent)

    @property
    def last_connect_error(self) -> str | None:
        """Most recent direct-listener connection error, if any."""
        return self._last_connect_error

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        # A worker can exit after an encoder/codec error while retaining its
        # handle.  Treat that as a stopped instance and make a subsequent
        # start deterministic instead of silently doing nothing.
        self._thread = None
        self._disconnect()
        self._drain_frames()
        self._stop.clear()
        with self._media_lock:
            self._next_connect_s = 0.0
            self._last_connect_error = None
        thread = threading.Thread(target=self._encode_loop, name="scene_xrobotoolkit_video", daemon=True)
        self._thread = thread
        try:
            thread.start()
        except BaseException:
            self._stop.set()
            self._thread = None
            self._disconnect()
            self._drain_frames()
            raise

    def publish_frame(self, frame: np.ndarray) -> None:
        if self._stop.is_set():
            # Do not enqueue frames after shutdown.  Without this guard a
            # producer racing ``stop()`` could refill the one-slot queue after
            # the sentinel was drained, and the next ``start()`` would send a
            # stale frame from the previous session.
            return
        image = np.ascontiguousarray(frame, dtype=np.uint8).copy()
        try:
            self._frames.put_nowait(image)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                return
            try:
                self._frames.put_nowait(image)
            except queue.Full:
                pass

    def stop(self) -> None:
        self._stop.set()
        # Close the media socket before joining the encoder.  A blocked TCP
        # ``sendall`` must be interrupted promptly; Remote Vision is a
        # best-effort side channel and must not delay scene shutdown.
        self._disconnect()
        try:
            self._frames.put_nowait(None)
        except queue.Full:
            pass
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
            # Do not discard a still-running handle.  A foreign codec/socket
            # implementation can ignore close() briefly; clearing the handle
            # here would let ``start()`` launch a second worker when it clears
            # the shared stop event, causing duplicate encoders and races on
            # the media socket.  A later start will observe the live handle
            # and wait until that generation exits.
            if not thread.is_alive():
                self._thread = None
        with self._media_lock:
            self._encoder = None
        # Remove both the shutdown sentinel and any frame published by a
        # producer immediately before the stop flag became visible.  A fresh
        # start must never transmit data captured by the prior session.
        self._drain_frames()

    def _drain_frames(self) -> None:
        """Discard all queued frames/sentinels without blocking."""

        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                return

    def _connect(self) -> None:
        now = time.monotonic()
        with self._media_lock:
            media_connected = self._media is not None
            next_connect_s = self._next_connect_s
        if media_connected or now < next_connect_s or self._stop.is_set():
            return
        with self._media_lock:
            # Another worker/status call may have won the connection race
            # since the first check above.  There is normally one encoder
            # worker, but keeping this check under the same lock makes the
            # lifecycle invariant explicit and protects direct test/use of
            # this helper.
            if self._media is not None or now < self._next_connect_s or self._stop.is_set():
                return
            self._next_connect_s = now + 1.0
        media: socket.socket | None = None
        try:
            media = socket.create_connection((self._host, self._port), timeout=1.0)
            media.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # ``create_connection`` leaves its timeout installed on the
            # connected socket.  It is appropriate for the initial Listen
            # probe but not for a continuous H.264 send: a Pico decoder can
            # briefly backpressure TCP for more than one second while keeping
            # the session healthy.  The encoder has its own thread and
            # ``stop()`` closes the socket, so an unbounded stream write
            # cannot stall the MuJoCo control loop or shutdown.
            media.settimeout(None)
        except Exception as exc:
            if media is not None:
                try:
                    media.close()
                except Exception:
                    pass
            with self._media_lock:
                self._last_connect_error = str(exc)
            return
        with self._media_lock:
            # ``stop`` can set the event and clear an earlier media socket
            # while ``create_connection`` is blocked.  Never publish a newly
            # connected socket after shutdown; close it on this side instead.
            if self._stop.is_set():
                close_after_connect = True
            else:
                self._media = media
                self._encoder = None
                self._last_connect_error = None
                close_after_connect = False
        if close_after_connect:
            try:
                media.close()
            except OSError:
                pass
            return
        print(f"Scene Remote Vision connected to {self._host}:{self._port} ({self._width}x{self._height}@{self._fps})")

    def _disconnect(self) -> None:
        with self._media_lock:
            media, self._media = self._media, None
            self._encoder = None
        if media is not None:
            try:
                media.close()
            except OSError:
                pass

    def _encode_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._frames.get(timeout=0.02)
            except queue.Empty:
                # Connect independently of rendering so Pico can open Listen
                # before its first MuJoCo camera frame is available.
                self._connect()
                continue
            if frame is None:
                continue
            self._connect()
            # Take a local reference before encoding/sending. ``stop()``
            # closes and clears ``self._media`` from the control thread; if
            # the worker read the attribute once per packet, a multi-packet
            # access unit could observe ``None`` halfway through shutdown
            # and terminate on an uncaught ``AttributeError``. A local
            # socket remains safe here: concurrent shutdown closes it, which
            # is reported as ``OSError`` and handled by the reconnect path.
            with self._media_lock:
                media = self._media
            if media is None:
                continue
            try:
                encoder = self._ensure_encoder()
                packets = self._encode(encoder, frame)
                # Publish the frame counter before the first blocking socket
                # write.  ``frames_sent`` is an operator-facing progress
                # indicator (and is read concurrently by the control/status
                # thread); updating it only after ``sendall`` creates a
                # narrow race where the receiver has already consumed a
                # complete access unit while the status still reports zero.
                # A failed write is handled by the reconnect path below, so
                # this counter intentionally tracks frames handed to the
                # transport rather than pretending it is an acknowledgement
                # from the headset.
                self._frames_sent += 1
                for packet in packets:
                    media.sendall(struct.pack(">I", len(packet)) + packet)
            except Exception as exc:
                # PyAV exposes codec failures as ``av.error.FFmpegError``
                # rather than one of the builtin RuntimeError/ValueError
                # classes.  This is an auxiliary side channel: no encoder or
                # socket exception should terminate the scene control worker.
                if not self._stop.is_set():
                    print(f"Scene Remote Vision reconnecting: {exc}")
                self._disconnect()

    def _ensure_encoder(self) -> object:
        if self._encoder is not None:
            return self._encoder
        try:
            import av
        except ImportError as exc:
            raise RuntimeError("PyAV is required for Scene Remote Vision") from exc
        encoder = av.CodecContext.create("libx264", "w")
        encoder.width = self._width
        encoder.height = self._height
        encoder.pix_fmt = "yuv420p"
        encoder.framerate = self._fps
        encoder.bit_rate = 4_000_000
        encoder.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "profile": "baseline",
            # Match SIMPLE's ZEDMINI sender: repeat parameter sets and emit a
            # recovery IDR every 15 frames.  A Pico listener can join after a
            # Wi-Fi reconnect; waiting four seconds for the old 120-frame
            # keyframe interval leaves its decoder on a blank frame.
            "x264-params": "annexb=1:repeat-headers=1:scenecut=0:keyint=15:min-keyint=15:bframes=0",
        }
        encoder.open()
        self._encoder = encoder
        return encoder

    def _encode(self, encoder: object, frame: np.ndarray) -> list[bytes]:
        image = np.asarray(frame, dtype=np.uint8)
        if image.shape == (self._height, self._width // 2, 3):
            image = np.concatenate((image, image), axis=1)
        if image.shape != (self._height, self._width, 3):
            raise ValueError(
                f"Scene camera must be {self._height}x{self._width // 2} RGB before stereo duplication, got {image.shape}"
            )
        import av

        bgr = av.VideoFrame.from_ndarray(np.ascontiguousarray(image[..., ::-1]), format="bgr24")
        yuv = bgr.reformat(width=self._width, height=self._height, format="yuv420p")
        return [bytes(packet) for packet in encoder.encode(yuv)]
