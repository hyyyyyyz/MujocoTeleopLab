"""XRoboToolkit Remote Vision sender for scene teleoperation.

Recent headset firmware exposes a TCP H.264 listener directly.  Keeping this
small sender in the scene package avoids importing Teleopit's full-body input
package into the isolated Python 3.10 WBC environment.

The headset app also runs a legacy operator-control handshake: it opens a TCP
connection to the PC on port 13579 and advertises its own media listener with
an ``OPEN_CAMERA`` request.  Without that control connection the headset's
Listen flow retries and never keeps its video listener open, so this sender
runs the direct-listener probe in parallel with a best-effort legacy control
listener (matching SIMPLE's dual-mode Pico camera setup).
"""

from __future__ import annotations

from dataclasses import dataclass
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
        raise ValueError(f"Remote Vision {name} must be a positive integer")
    if even and integer % 2:
        raise ValueError(f"Remote Vision {name} must be even for yuv420p")
    return integer


# XRoboToolkit's legacy operator-control channel.  The headset connects out to
# the PC here before it keeps its media listener open.
_CONTROL_PORT = 13579
_MAX_CONTROL_MESSAGE_BYTES = 2 * 1024 * 1024
# The headset control client drops an idle connection after ~5 s (its
# ``IdleTimeoutMs``), which triggers a Listen retry loop.  Send a PING every
# two seconds so a healthy session stays negotiated.
_CONTROL_PING_INTERVAL_S = 2.0


@dataclass(frozen=True)
class _SceneCameraRequest:
    """Parsed XRoboToolkit ``OPEN_CAMERA`` media-listener advertisement."""

    width: int
    height: int
    fps: int
    bitrate: int
    host: str
    port: int


def _strict_request_int(value: object, name: str) -> int:
    """Parse a wire integer without coercing floats or booleans."""

    if isinstance(value, (bool, np.bool_)):
        raise ConnectionError(f"XRoboToolkit camera {name} must be an integer")
    try:
        return int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConnectionError(f"XRoboToolkit camera {name} must be an integer") from exc


def _validate_negotiated_request(request: _SceneCameraRequest) -> _SceneCameraRequest:
    """Validate a legacy camera request before opening a media socket.

    The headset decoder uses ``yuv420p`` H.264, so frame dimensions must be
    even.  Validate all fields at the process boundary so malformed headset
    data cannot start a worker that repeatedly fails in the encoder thread.
    """

    width = _strict_request_int(request.width, "width")
    height = _strict_request_int(request.height, "height")
    fps = _strict_request_int(request.fps, "FPS")
    bitrate = _strict_request_int(request.bitrate, "bitrate")
    port = _strict_request_int(request.port, "port")
    if width <= 0 or height <= 0:
        raise ConnectionError(f"Invalid XRoboToolkit camera dimensions: {width}x{height}")
    if width % 2 or height % 2:
        raise ConnectionError("XRoboToolkit camera width and height must be even for yuv420p")
    if fps <= 0:
        raise ConnectionError(f"Invalid XRoboToolkit camera FPS: {fps}")
    if bitrate <= 0:
        raise ConnectionError(f"Invalid XRoboToolkit camera bitrate: {bitrate}")
    host = request.host.strip()
    if not host:
        raise ConnectionError("XRoboToolkit camera target host must not be empty")
    if port <= 0 or port > 65535:
        raise ConnectionError(f"Invalid XRoboToolkit camera target port: {port}")
    return _SceneCameraRequest(
        width=width,
        height=height,
        fps=fps,
        bitrate=bitrate,
        host=host,
        port=port,
    )


class DirectXRoboToolkitVideoTransport:
    """Best-effort, reconnecting H.264 sender for the headset's direct port.

    Each camera frame is encoded at most once.  The headset's TCP MediaDecoder
    has a small queue, so replaying the latest source frame to force a nominal
    FPS would eventually fill the queue and stall a healthy session.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        width: int,
        height: int,
        fps: int,
        control_port: int = _CONTROL_PORT,
    ) -> None:
        self._host = str(host).strip()
        if not self._host:
            raise ValueError("Remote Vision host must not be empty")
        self._port = _validated_transport_int(port, "port")
        if self._port > 65535:
            raise ValueError("Remote Vision port must be in [1, 65535]")
        self._width = _validated_transport_int(width, "width", even=True)
        self._height = _validated_transport_int(height, "height", even=True)
        self._fps = _validated_transport_int(fps, "FPS")
        # ``control_port=0`` asks the OS to auto-assign a free port.  This is
        # useful for tests and for machines where 13579 is already owned by
        # another operator service; ``_start_control_listener`` records the
        # actual bound port back on this attribute.
        if isinstance(control_port, (bool, np.bool_)):
            raise ValueError("Remote Vision control port must be an integer")
        try:
            control_port_i = int(operator.index(control_port))
        except (TypeError, ValueError) as exc:
            raise ValueError("Remote Vision control port must be an integer") from exc
        if control_port_i < 0 or control_port_i > 65535:
            raise ValueError("Remote Vision control port must be in [0, 65535]")
        self._control_port = control_port_i
        self._stop = threading.Event()
        # Keep only the newest encoded source frame.  If a Pico decoder or
        # Wi-Fi link backpressures ``sendall``, retaining a second stale frame
        # would add another full video period of avoidable view latency.
        self._frames: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        self._control_listener: socket.socket | None = None
        self._control_thread: threading.Thread | None = None
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
        # The media target starts at the configured direct endpoint.  A legacy
        # ``OPEN_CAMERA`` request (which advertises the headset's own listener)
        # is authoritative and replaces this target and the encoder profile.
        self._target_host = self._host
        self._target_port = self._port
        self._media_width = self._width
        self._media_height = self._height
        self._media_fps = self._fps

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
        self._start_control_listener()
        thread = threading.Thread(target=self._encode_loop, name="scene_xrobotoolkit_video", daemon=True)
        self._thread = thread
        try:
            thread.start()
        except BaseException:
            self._stop.set()
            self._thread = None
            self._disconnect()
            self._drain_frames()
            self._stop_control_listener()
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
        self._stop_control_listener()
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

    def _start_control_listener(self) -> None:
        """Best-effort legacy operator-control listener on ``_control_port``.

        The headset's Listen flow connects here before it keeps its media
        listener open.  Binding failure is not fatal: the direct listener
        probe remains a valid path for headset builds that never send the
        legacy request.
        """

        if self._control_thread is not None and self._control_thread.is_alive():
            return
        listener: socket.socket | None = None
        try:
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("0.0.0.0", self._control_port))
            listener.listen(1)
            listener.settimeout(0.5)
            if self._control_port == 0:
                self._control_port = int(listener.getsockname()[1])
        except OSError as exc:
            if listener is not None:
                try:
                    listener.close()
                except OSError:
                    pass
            print(
                "Scene Remote Vision legacy control listener on TCP "
                f"{self._control_port} unavailable: {exc}"
            )
            return
        self._control_listener = listener
        self._control_thread = threading.Thread(
            target=self._serve_control,
            name="scene_xrobotoolkit_control",
            daemon=True,
        )
        self._control_thread.start()
        print(f"Scene Remote Vision legacy control listener ready on TCP {self._control_port}")

    def _stop_control_listener(self) -> None:
        listener, self._control_listener = self._control_listener, None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        thread, self._control_thread = self._control_thread, None
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def _serve_control(self) -> None:
        while not self._stop.is_set():
            listener = self._control_listener
            if listener is None:
                return
            try:
                client, peer = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            print(f"Scene Remote Vision control client connected: {peer[0]}")
            try:
                with client:
                    self._handle_control_client(client)
            except (ConnectionError, OSError) as exc:
                if not self._stop.is_set():
                    print(f"Scene Remote Vision control client disconnected: {exc}")

    def _handle_control_client(self, client: socket.socket) -> None:
        next_ping_s = time.monotonic() + _CONTROL_PING_INTERVAL_S
        while not self._stop.is_set():
            now = time.monotonic()
            if now >= next_ping_s:
                try:
                    _send_control_ack(client, "PING")
                except OSError as exc:
                    raise ConnectionError(f"control keepalive failed: {exc}") from exc
                next_ping_s = now + _CONTROL_PING_INTERVAL_S
            # A short read timeout keeps the PING cadence regular while a
            # headset that never sends keeps the connection healthy.
            client.settimeout(0.5)
            try:
                header = _recv_exact(client, 4)
            except TimeoutError:
                continue
            if header is None:
                return
            (size,) = struct.unpack(">I", header)
            if size > _MAX_CONTROL_MESSAGE_BYTES:
                raise ConnectionError(f"XRoboToolkit control message too large: {size} bytes")
            payload = _recv_exact(client, size)
            if payload is None:
                return
            command, data = _parse_control_message(payload)
            if command == "OPEN_CAMERA":
                request = _parse_camera_request(data)
                self._connect_media(request)
                _send_control_ack(client, command)
            elif command == "CLOSE_CAMERA":
                self._disconnect()
                _send_control_ack(client, command)
            elif command == "PING":
                _send_control_ack(client, "PONG")
            else:
                print(f"Scene Remote Vision control command: {command}")

    def _connect_media(self, request: _SceneCameraRequest) -> None:
        with self._media_lock:
            self._target_host = request.host
            self._target_port = request.port
            self._media_width = request.width
            self._media_height = request.height
            self._media_fps = request.fps
            # The headset just restarted its media listener for this Listen
            # session.  Force a fresh encoder and probe immediately instead of
            # honoring a stale reconnect backoff from the direct path.  Close
            # any existing media socket under the lock so a legacy OPEN_CAMERA
            # is authoritative over a direct-mode connection.
            self._encoder = None
            self._next_connect_s = 0.0
            old_media, self._media = self._media, None
        if old_media is not None:
            try:
                old_media.close()
            except OSError:
                pass
        self._connect()

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
            host = self._target_host
            port = self._target_port
            width = self._media_width
            height = self._media_height
            fps = self._media_fps
        media: socket.socket | None = None
        try:
            media = socket.create_connection((host, port), timeout=1.0)
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
            # The direct probe and the legacy OPEN_CAMERA handler can both be
            # in flight: if a different worker installed a socket while this
            # ``connect()`` was blocked, keep the winner and drop the
            # duplicate probe.
            if self._stop.is_set() or self._media is not None:
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
        print(
            f"Scene Remote Vision connected to {host}:{port} "
            f"({width}x{height}@{fps})"
        )

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
            # Snapshot the encoder profile with the socket so a legacy
            # ``OPEN_CAMERA`` that changes geometry mid-stream cannot create
            # an access unit half-encoded at one profile and half at another.
            with self._media_lock:
                media = self._media
                width = self._media_width
                height = self._media_height
                fps = self._media_fps
            if media is None:
                continue
            try:
                encoder = self._ensure_encoder(width, height, fps)
                packets = self._encode(encoder, frame, width, height)
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

    def _ensure_encoder(self, width: int, height: int, fps: int) -> object:
        encoder = self._encoder
        if encoder is not None:
            if (
                getattr(encoder, "width", None) == width
                and getattr(encoder, "height", None) == height
                and getattr(encoder, "framerate", None) == fps
            ):
                return encoder
            # A legacy OPEN_CAMERA negotiated a different profile.  Drop the
            # stale encoder so the next access unit starts with a fresh
            # SPS/PPS for the new geometry.
            self._encoder = None
            encoder = None
        try:
            import av
        except ImportError as exc:
            raise RuntimeError("PyAV is required for Scene Remote Vision") from exc
        encoder = av.CodecContext.create("libx264", "w")
        encoder.width = width
        encoder.height = height
        encoder.pix_fmt = "yuv420p"
        encoder.framerate = fps
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

    def _encode(self, encoder: object, frame: np.ndarray, width: int, height: int) -> list[bytes]:
        image = np.asarray(frame, dtype=np.uint8)
        if image.shape == (height, width // 2, 3):
            image = np.concatenate((image, image), axis=1)
        if image.shape != (height, width, 3):
            raise ValueError(
                f"Scene camera must be {height}x{width // 2} RGB before stereo duplication, got {image.shape}"
            )
        import av

        bgr = av.VideoFrame.from_ndarray(np.ascontiguousarray(image[..., ::-1]), format="bgr24")
        yuv = bgr.reformat(width=width, height=height, format="yuv420p")
        return [bytes(packet) for packet in encoder.encode(yuv)]


def _recv_exact(sock: socket.socket, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        try:
            chunk = sock.recv(remaining)
        except socket.timeout as exc:
            raise TimeoutError from exc
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _parse_control_message(payload: bytes) -> tuple[str, bytes]:
    """Decode a NetworkDataProtocol body from the operator-control channel.

    Framing observed from XRoboToolkit: the control socket carries
    ``[4-byte BE body length][body]`` where ``body`` is the little-endian
    NetworkDataProtocol layout ``[4-byte LE command length][command]``
    ``[4-byte LE data length][data]``.  Trailing NUL bytes in the declared
    command string are padding, not part of the name.
    """

    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ConnectionError("Malformed XRoboToolkit control message")
    payload = bytes(payload)
    if len(payload) < 8:
        raise ConnectionError("Malformed XRoboToolkit control message")
    (command_size,) = struct.unpack_from("<i", payload, 0)
    if command_size < 0 or 4 + command_size + 4 > len(payload):
        raise ConnectionError("Malformed XRoboToolkit command length")
    offset = 4
    try:
        command = payload[offset : offset + command_size].decode("utf-8").rstrip("\x00")
    except UnicodeDecodeError as exc:
        raise ConnectionError("Malformed XRoboToolkit command encoding") from exc
    offset += command_size
    (data_size,) = struct.unpack_from("<i", payload, offset)
    offset += 4
    # The framed control message contains exactly one command and one data
    # payload.  Silently accepting trailing bytes makes a truncated or
    # concatenated packet ambiguous, so require an exact length match.
    if data_size < 0 or offset + data_size != len(payload):
        raise ConnectionError("Malformed XRoboToolkit control payload length")
    return command, payload[offset : offset + data_size]


def _send_control_ack(sock: socket.socket, command: str) -> None:
    command_bytes = command.encode("utf-8")
    body = struct.pack("<i", len(command_bytes)) + command_bytes + struct.pack("<i", 0)
    sock.sendall(struct.pack(">I", len(body)) + body)


def _parse_camera_request(payload: bytes) -> _SceneCameraRequest:
    """Parse the binary ``CameraRequestSerializer`` payload of OPEN_CAMERA."""

    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ConnectionError("Malformed XRoboToolkit OPEN_CAMERA request")
    payload = bytes(payload)
    if len(payload) < 31 or payload[:3] != b"\xca\xfe\x01":
        raise ConnectionError("Malformed XRoboToolkit OPEN_CAMERA request")
    width, height, fps, bitrate, _hevc, _render_mode, port = struct.unpack_from("<7i", payload, 3)
    offset = 31
    camera, offset = _read_compact_string(payload, offset)
    host, offset = _read_compact_string(payload, offset)
    if offset != len(payload) or not host.strip():
        raise ConnectionError("Malformed XRoboToolkit camera target")
    # The camera label is informational (ZEDMINI and friends).  Log it rather
    # than rejecting an otherwise well-formed request from a newer build.
    print(f"Scene Remote Vision camera source: {camera!r}")
    return _validate_negotiated_request(
        _SceneCameraRequest(
            width=width,
            height=height,
            fps=fps,
            bitrate=bitrate,
            host=host,
            port=port,
        )
    )


def _read_compact_string(payload: bytes, offset: int) -> tuple[str, int]:
    if offset >= len(payload):
        raise ConnectionError("Malformed XRoboToolkit compact string")
    size = payload[offset]
    offset += 1
    if offset + size > len(payload):
        raise ConnectionError("Truncated XRoboToolkit compact string")
    try:
        value = payload[offset : offset + size].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConnectionError("Malformed XRoboToolkit compact string encoding") from exc
    return value, offset + size
