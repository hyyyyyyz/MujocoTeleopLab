"""MuJoCo camera streaming for XRoboToolkit Remote Vision.

The XRoboToolkit headset app opens a TCP control connection to the operator on
port 13579.  Its ``OPEN_CAMERA`` request identifies a second TCP listener on
the headset.  This module renders the simulated G1 camera, encodes it as an
Annex-B H.264 elementary stream, and sends each encoded access unit with the
four-byte big-endian framing used by XRoboToolkit's reference sender.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import operator
import queue
import socket
import struct
import threading
import time
from typing import Any

import numpy as np

from teleopit.inputs.pico_video import PicoVideoConfig, _MujocoCameraVideoProducer

logger = logging.getLogger(__name__)

_CONTROL_PORT = 13579
_MAX_CONTROL_MESSAGE_BYTES = 2 * 1024 * 1024


def _strict_integer(value: object, name: str, *, error_type: type[Exception] = ConnectionError) -> int:
    """Return an integer protocol field without silently coercing values.

    ``int(1280.5)`` and ``int(True)`` are both legal Python operations, but
    neither is a valid camera setting.  The XRoboToolkit boundary accepts
    Python/NumPy integer scalars only; callers choose the exception type so
    configuration errors can use ``ValueError`` while malformed wire packets
    retain their ``ConnectionError`` classification.
    """

    if isinstance(value, (bool, np.bool_)):
        raise error_type(f"XRoboToolkit camera {name} must be an integer")
    try:
        return int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise error_type(f"XRoboToolkit camera {name} must be an integer") from exc


@dataclass(frozen=True)
class _CameraRequest:
    width: int
    height: int
    fps: int
    bitrate: int
    host: str
    port: int


def _validate_camera_request(request: _CameraRequest) -> _CameraRequest:
    """Validate a legacy/direct camera request before opening a socket.

    The receiver decodes ``yuv420p`` H.264, whose frame dimensions must be
    even.  Validate all fields at the process boundary so malformed headset
    data cannot start a worker that repeatedly fails in the encoder thread.
    """

    width = _strict_integer(request.width, "width")
    height = _strict_integer(request.height, "height")
    fps = _strict_integer(request.fps, "FPS")
    bitrate = _strict_integer(request.bitrate, "bitrate")
    port = _strict_integer(request.port, "port")
    if width <= 0 or height <= 0:
        raise ConnectionError(f"Invalid XRoboToolkit camera dimensions: {width}x{height}")
    if width % 2 or height % 2:
        raise ConnectionError("XRoboToolkit camera width and height must be even for yuv420p")
    if fps <= 0:
        raise ConnectionError(f"Invalid XRoboToolkit camera FPS: {fps}")
    if bitrate <= 0:
        raise ConnectionError(f"Invalid XRoboToolkit camera bitrate: {bitrate}")
    if not isinstance(request.host, str):
        raise ConnectionError("XRoboToolkit camera target host must be a string")
    host = request.host.strip()
    if not host:
        raise ConnectionError("XRoboToolkit camera target host must not be empty")
    if port <= 0 or port > 65535:
        raise ConnectionError(f"Invalid XRoboToolkit camera target port: {port}")
    return _CameraRequest(
        width=width,
        height=height,
        fps=fps,
        bitrate=bitrate,
        host=host,
        port=port,
    )


class XRoboToolkitVideoRuntime:
    """Publish the MuJoCo ``d435i_rgb`` camera through XRoboToolkit Remote Vision."""

    def __init__(
        self,
        *,
        config: PicoVideoConfig,
        robot: Any | None,
        direct_host: str | None = None,
        direct_port: int = 12345,
    ) -> None:
        if not config.enabled:
            raise ValueError("XRoboToolkitVideoRuntime requires input.video.enabled=true")
        if config.source != "mujoco":
            raise ValueError("XRoboToolkit Remote Vision supports input.video.source=mujoco only")
        endpoint: _CameraRequest | None = None
        if direct_host not in (None, ""):
            # Preserve the raw config values until the strict protocol
            # validator sees them.  Calling ``int`` here would silently turn
            # a typo such as ``1280.5`` into a different stream geometry.
            source_width = _strict_integer(config.width, "input.video.width", error_type=ValueError)
            # Recent XRoboToolkit headset builds expose this TCP listener
            # directly, rather than issuing the legacy OPEN_CAMERA request.
            # ZEDMINI's layout is a 2560x720 stereo frame; the single MuJoCo
            # RGB camera is duplicated for the two eyes below.
            endpoint = _CameraRequest(
                width=source_width * 2,
                height=config.height,
                # The direct XRoboToolkit sender fixes this ZEDMINI profile at
                # 2560x720@60.  Match it exactly; the headset initializes its
                # decoder texture from this profile before accepting packets.
                fps=60,
                bitrate=4_000_000,
                host=direct_host,
                port=direct_port,
            )
            # Fail during pipeline construction rather than from the
            # background reconnect worker when a local config contains an
            # unsupported odd dimension or malformed endpoint.
            try:
                endpoint = _validate_camera_request(endpoint)
            except ConnectionError as exc:
                raise ValueError(str(exc)) from exc
        self._transport = _XRoboToolkitVideoTransport(direct_request=endpoint)
        self._producer = _MujocoCameraVideoProducer(
            provider=None,
            config=config,
            robot=robot,
            frame_callback=self._transport.publish_frame,
        )

    @property
    def pushed_frames(self) -> int:
        return self._producer.pushed_frames

    def start(self) -> None:
        self._transport.start()
        try:
            self._producer.start()
        except Exception:
            # Renderer construction can allocate an OpenGL context before a
            # later validation step fails.  Tear it down as part of the same
            # transactional startup rollback; otherwise a retry leaks the
            # context even though the TCP transport was stopped.
            try:
                self._producer.stop()
            except Exception:
                logger.exception("Failed to stop XRoboToolkit video producer after startup error")
            finally:
                self._transport.stop()
            raise
        if self._transport.direct_request is None:
            logger.info(
                "XRoboToolkit Remote Vision legacy control listener ready on TCP port %d",
                _CONTROL_PORT,
            )
        else:
            request = self._transport.direct_request
            logger.info(
                "XRoboToolkit Remote Vision direct listener mode: connecting to %s:%d",
                request.host,
                request.port,
            )

    def tick(self) -> None:
        self._producer.tick()

    def stop(self) -> None:
        # The renderer is an auxiliary side channel.  Always tear down its
        # transport even if a MuJoCo/OpenGL close operation raises, otherwise
        # the reconnect worker can keep a headset socket (and daemon thread)
        # alive after the main simulation has already stopped.
        try:
            self._producer.stop()
        finally:
            self._transport.stop()


class _XRoboToolkitVideoTransport:
    """Small implementation of the public XRoboToolkit operator video protocol."""

    def __init__(self, *, direct_request: _CameraRequest | None = None) -> None:
        self._stop = threading.Event()
        self._control_listener: socket.socket | None = None
        self._control_thread: threading.Thread | None = None
        self._encode_thread: threading.Thread | None = None
        self._direct_thread: threading.Thread | None = None
        self._frames: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=2)
        self._media_lock = threading.Lock()
        self._media_socket: socket.socket | None = None
        self._request: _CameraRequest | None = None
        self._encoder: Any | None = None
        self._direct_request = direct_request
        self._next_direct_connect_s = 0.0

    @property
    def direct_request(self) -> _CameraRequest | None:
        """Configured direct endpoint, or ``None`` when legacy negotiation is used."""

        return self._direct_request

    def start(self) -> None:
        # A direct-listener session does not own the legacy control port.  Use
        # all worker handles as lifecycle sentinels (not just the encoder): a
        # reconnect worker can outlive an encoder that failed while a socket
        # operation was in flight.  Clearing the shared stop event and
        # starting a second generation in that window would create duplicate
        # reconnect loops and races on the media socket.
        existing_threads = tuple(
            thread
            for thread in (self._control_thread, self._encode_thread, self._direct_thread)
            if thread is not None
        )
        if any(thread.is_alive() for thread in existing_threads):
            return
        if existing_threads:
            # Workers have all exited after a codec/socket exception.  Tear
            # down the failed generation and drain its queue before retrying.
            self.stop()
        # A producer may have won the stop race immediately before the final
        # queue drain.  Clear any old frames/sentinels while the stop flag is
        # still set, then begin a clean generation below.
        self._drain_frames()
        self._stop.clear()
        with self._media_lock:
            # A previous generation may have reserved a reconnect retry just
            # before it was stopped.  Do not carry that backoff into a fresh
            # Listen session; probe the newly started endpoint immediately.
            self._next_direct_connect_s = 0.0
        # Recent XRoboToolkit builds open the media listener on the headset
        # directly (the local preset uses TCP 12345).  Binding the old
        # 13579 control port as well is unnecessary and can collide with a
        # concurrently running SIMPLE/XRoboToolkit service.  Keep the legacy
        # OPEN_CAMERA listener only for configurations that do not provide a
        # direct endpoint.
        if self._direct_request is None:
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(("0.0.0.0", _CONTROL_PORT))
                listener.listen(1)
                listener.settimeout(0.5)
            except Exception:
                listener.close()
                raise
            self._control_listener = listener
            self._control_thread = threading.Thread(
                target=self._serve_control,
                name="xrobotoolkit_video_control",
                daemon=True,
            )
        self._encode_thread = threading.Thread(target=self._encode_frames, name="xrobotoolkit_video_encode", daemon=True)
        try:
            if self._control_thread is not None:
                self._control_thread.start()
            self._encode_thread.start()
            self._ensure_direct_connection()
            if self._direct_request is not None:
                self._direct_thread = threading.Thread(
                    target=self._maintain_direct_connection,
                    name="xrobotoolkit_video_direct",
                    daemon=True,
                )
                self._direct_thread.start()
        except BaseException:
            # Roll back partially started workers/listeners so a caller can
            # retry setup without leaking a bound 13579 socket or reconnect
            # thread.  ``stop`` is bounded and all workers are daemons.
            self.stop()
            raise

    def stop(self) -> None:
        self._stop.set()
        listener, self._control_listener = self._control_listener, None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        self._disconnect_media()
        try:
            self._frames.put_nowait(None)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frames.put_nowait(None)
            except queue.Full:
                # A producer may have crossed the stopped flag check at the
                # same instant.  The worker is being joined below and the
                # final drain removes any residual frame; shutdown must not
                # fail merely because the one-slot sentinel could not be
                # inserted.
                pass
        for attr in ("_control_thread", "_encode_thread", "_direct_thread"):
            thread = getattr(self, attr)
            if thread is not None and thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=2.0)
            # Keep a still-running handle so a subsequent ``start`` cannot
            # clear ``_stop`` and race this generation.  Normally all workers
            # exit within the timeout; retaining only the exceptional live
            # handle makes the lifecycle safe without an unbounded join.
            if thread is None or not thread.is_alive():
                setattr(self, attr, None)
        with self._media_lock:
            self._encoder = None
        # Do this after workers have been given their sentinel and joined, and
        # again at the start of the next generation.  The second drain closes
        # the tiny race in which a producer passed the stopped check just
        # before ``_stop.set()`` and enqueued one final frame.
        self._drain_frames()

    def _drain_frames(self) -> None:
        """Discard queued frames and shutdown sentinels without blocking."""

        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                return

    def publish_frame(self, frame: np.ndarray, _: float) -> None:
        if self._stop.is_set():
            return
        # The renderer owns the source image; retain an independent contiguous
        # frame for the encoding worker without ever blocking the simulation.
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
            logger.info("XRoboToolkit Remote Vision control client connected: %s", peer[0])
            try:
                with client:
                    client.settimeout(1.0)
                    self._handle_control_client(client)
            except (ConnectionError, OSError) as exc:
                if not self._stop.is_set():
                    logger.info("XRoboToolkit Remote Vision control client disconnected: %s", exc)

    def _handle_control_client(self, client: socket.socket) -> None:
        while not self._stop.is_set():
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
            command, data = _parse_network_message(payload)
            if command == "OPEN_CAMERA":
                request = _parse_camera_request(data)
                self._connect_media(request)
                _send_control_ack(client, command)
            elif command == "CLOSE_CAMERA":
                self._disconnect_media()
                _send_control_ack(client, command)

    def _ensure_direct_connection(self) -> None:
        request = self._direct_request
        if request is None or self._stop.is_set():
            return
        now = time.monotonic()
        with self._media_lock:
            if self._media_socket is not None:
                return
            # Both the encoder and the reconnect worker may call this helper.
            # Reserve the next probe while holding the same lock used for the
            # socket hand-off so they cannot start duplicate connections.
            if now < self._next_direct_connect_s:
                return
            self._next_direct_connect_s = now + 1.0
        # A legacy OPEN_CAMERA request is authoritative and may replace an
        # existing stream.  A background direct probe, however, must not
        # replace a socket that was installed while its connect() call was in
        # flight.
        self._connect_media(request, only_if_empty=True)

    def _maintain_direct_connection(self) -> None:
        """Reconnect when the headset opens its direct video listener later.

        Live body input may not be available yet, so this cannot depend on the
        simulation thread reaching its normal per-step video tick.
        """
        while not self._stop.is_set():
            try:
                self._ensure_direct_connection()
            except ConnectionError as exc:
                # Keep this auxiliary worker alive if a configured request is
                # invalid or an SDK/config reload supplies malformed values.
                logger.warning("XRoboToolkit direct camera request invalid: %s", exc)
            self._stop.wait(0.25)

    def _connect_media(self, request: _CameraRequest, *, only_if_empty: bool = False) -> None:
        request = _validate_camera_request(request)
        media: socket.socket | None = None
        try:
            media = socket.create_connection((request.host, request.port), timeout=2.0)
            media.settimeout(0.25)
            media.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception as exc:
            if media is not None:
                try:
                    media.close()
                except Exception:
                    pass
            logger.warning("Unable to connect to XRoboToolkit headset media listener %s:%d: %s", request.host, request.port, exc)
            return
        with self._media_lock:
            # A connection can complete after ``stop`` or after another
            # control request has taken over the stream.  Do not resurrect a
            # transport that was already shut down; close the just-created
            # socket on this side instead.
            if self._stop.is_set():
                previous = None
                install = False
            elif only_if_empty and self._media_socket is not None:
                previous = None
                install = False
            else:
                previous = self._media_socket
                self._media_socket = media
                self._request = request
                self._encoder = None  # Emit parameter sets and an IDR for each Listen session.
                install = True
        if not install:
            try:
                media.close()  # type: ignore[union-attr]
            except Exception:
                pass
            return
        if previous is not None:
            try:
                previous.close()
            except Exception:
                pass
        logger.info(
            "XRoboToolkit Remote Vision streaming to %s:%d | %dx%d@%d",
            request.host,
            request.port,
            request.width,
            request.height,
            request.fps,
        )

    def _disconnect_media(self) -> None:
        with self._media_lock:
            media, self._media_socket = self._media_socket, None
            self._request = None
            self._encoder = None
        if media is not None:
            try:
                media.close()
            except OSError:
                pass

    def _encode_frames(self) -> None:
        last_frame: np.ndarray | None = None
        next_send_s = 0.0
        while not self._stop.is_set():
            try:
                frame = self._frames.get(timeout=0.02)
            except queue.Empty:
                frame = None
            if frame is None:
                if self._stop.is_set():
                    return
            else:
                last_frame = frame
            try:
                self._ensure_direct_connection()
            except ConnectionError as exc:
                # A malformed configured endpoint should not kill the
                # encoder worker; keep the side channel retryable while the
                # scene/control process continues normally.
                logger.warning("XRoboToolkit direct camera request invalid: %s", exc)
                self._stop.wait(0.25)
                continue
            # Snapshot the transport state under the lock, then do potentially
            # expensive H.264 work and blocking TCP writes without holding it.
            # ``stop()`` must be able to close a backpressured socket promptly.
            with self._media_lock:
                media = self._media_socket
                request = self._request
            if media is None or request is None or last_frame is None:
                continue
            now = time.monotonic()
            if now < next_send_s:
                continue
            try:
                encoder = self._ensure_encoder(request, media)
                packets = _encode_h264(encoder, last_frame, request)
                for packet in packets:
                    media.sendall(struct.pack(">I", len(packet)) + packet)
                next_send_s = now + 1.0 / float(request.fps)
            except Exception as exc:
                # PyAV codec errors use ``av.error.FFmpegError`` and are not
                # necessarily subclasses of the builtin RuntimeError.  Keep
                # this optional video worker retryable for every ordinary
                # encoder/transport exception instead of silently killing it.
                logger.warning("XRoboToolkit Remote Vision stream stopped: %s", exc)
                self._drop_media_if_current(media)

    def _drop_media_if_current(self, media: socket.socket) -> None:
        """Close ``media`` without clearing a newer connection.

        A control thread may replace the socket while an encoder send is in
        progress.  The failed send belongs to the old object and must not
        wipe the new endpoint or encoder state.
        """
        with self._media_lock:
            current = self._media_socket is media
            if current:
                self._media_socket = None
                self._request = None
                self._encoder = None
        try:
            media.close()
        except OSError:
            pass

    def _ensure_encoder(self, request: _CameraRequest, media: socket.socket) -> Any:
        # The encoder is only used by the encoding worker, but its pointer is
        # reset by connection/lifecycle threads.  Read and publish it under
        # the transport lock, and only install a newly-created encoder if the
        # same socket/request is still current.
        with self._media_lock:
            if self._media_socket is media and self._request == request and self._encoder is not None:
                return self._encoder
        try:
            import av
        except ImportError as exc:  # pragma: no cover - checked during setup/runtime
            raise RuntimeError("PyAV is required for XRoboToolkit MuJoCo video") from exc
        encoder = av.CodecContext.create("libx264", "w")
        encoder.width = request.width
        encoder.height = request.height
        encoder.pix_fmt = "yuv420p"
        encoder.framerate = request.fps
        encoder.bit_rate = max(request.bitrate, 250_000)
        encoder.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "profile": "baseline",
            # SIMPLE's reference GStreamer pipeline uses insert-sps-pps and
            # idrinterval=15.  Repeat SPS/PPS and keep a short IDR interval so
            # a listener that reconnects mid-stream can decode promptly.
            "x264-params": "annexb=1:repeat-headers=1:scenecut=0:keyint=15:min-keyint=15:bframes=0",
        }
        encoder.open()
        with self._media_lock:
            if self._media_socket is media and self._request == request and not self._stop.is_set():
                self._encoder = encoder
        return encoder


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


def _parse_network_message(payload: bytes) -> tuple[str, bytes]:
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
        # Some SIMPLE/XRoboToolkit builds serialize fixed-size command buffers
        # and include one or more trailing NUL bytes in the declared string.
        # They are padding, not part of the command name.  Keep embedded NULs
        # invalid/meaningful while accepting only terminal padding.
        command = payload[offset:offset + command_size].decode("utf-8").rstrip("\x00")
    except UnicodeDecodeError as exc:
        raise ConnectionError("Malformed XRoboToolkit command encoding") from exc
    offset += command_size
    (data_size,) = struct.unpack_from("<i", payload, offset)
    offset += 4
    # The framed control message contains exactly one command and one data
    # payload.  Silently accepting trailing bytes makes a truncated/concatenated
    # packet ambiguous and can desynchronize the next request on a persistent
    # control connection, so require an exact length match.
    if data_size < 0 or offset + data_size != len(payload):
        raise ConnectionError("Malformed XRoboToolkit control payload length")
    return command, payload[offset:offset + data_size]


def _send_control_ack(sock: socket.socket, command: str) -> None:
    command_bytes = command.encode("utf-8")
    body = struct.pack("<i", len(command_bytes)) + command_bytes + struct.pack("<i", 0)
    sock.sendall(struct.pack(">I", len(body)) + body)


def _parse_camera_request(payload: bytes) -> _CameraRequest:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ConnectionError("Malformed XRoboToolkit OPEN_CAMERA request")
    payload = bytes(payload)
    if len(payload) < 31 or payload[:3] != b"\xca\xfe\x01":
        raise ConnectionError("Malformed XRoboToolkit OPEN_CAMERA request")
    width, height, fps, bitrate, _hevc, _render_mode, port = struct.unpack_from("<7i", payload, 3)
    offset = 31
    camera, offset = _read_compact_string(payload, offset)  # camera source label
    if camera.strip().upper() not in {"ZED", "ZEDMINI"}:
        raise ConnectionError(f"Unsupported XRoboToolkit camera source: {camera!r}")
    host, offset = _read_compact_string(payload, offset)
    if offset != len(payload) or not host.strip():
        raise ConnectionError("Malformed XRoboToolkit camera target")
    return _validate_camera_request(
        _CameraRequest(width=width, height=height, fps=fps, bitrate=bitrate, host=host, port=port)
    )


def _read_compact_string(payload: bytes, offset: int) -> tuple[str, int]:
    if offset >= len(payload):
        raise ConnectionError("Malformed XRoboToolkit compact string")
    size = payload[offset]
    offset += 1
    if offset + size > len(payload):
        raise ConnectionError("Truncated XRoboToolkit compact string")
    try:
        value = payload[offset:offset + size].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConnectionError("Malformed XRoboToolkit compact string encoding") from exc
    return value, offset + size


def _encode_h264(encoder: Any, frame: np.ndarray, request: _CameraRequest) -> list[bytes]:
    image = np.asarray(frame, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"MuJoCo camera frame must be HxWx3 RGB, got {image.shape}")
    # A direct ZEDMINI stream is stereo side-by-side.  The simulator exposes
    # one first-person camera, so use the same view for both eyes rather than
    # distorting a 16:9 image to 32:9.
    if image.shape[0] == request.height and image.shape[1] * 2 == request.width:
        image = np.concatenate((image, image), axis=1)
    import av

    # Match the reference TCPVideoSender byte-for-byte in its color handling:
    # MuJoCo renders RGB, the reference turns it into BGR, then PyAV converts
    # BGR24 to YUV420P for libx264.  ``VideoFrame.reformat`` performs any
    # required scaling here; keeping it in PyAV avoids making the optional
    # OpenCV package a hidden dependency of the legacy Remote Vision path.
    bgr_frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image[..., ::-1]), format="bgr24")
    video_frame = bgr_frame.reformat(width=request.width, height=request.height, format="yuv420p")
    video_frame.pts = None
    return [bytes(packet) for packet in encoder.encode(video_frame)]
