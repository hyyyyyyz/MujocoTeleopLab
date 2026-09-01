"""Pico4 VR full-body motion capture input provider.

Uses the in-process ``pico_bridge`` receiver to collect PICO tracking frames.
The provider converts native PICO poses (meters, xyzw quaternions) into
Teleopit's realtime ``HumanFrame`` format.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import inspect
from importlib.metadata import PackageNotFoundError, version
import logging
import numbers
import threading
import time
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from teleopit.inputs.realtime_frame_cache import RealtimeFrameCache
from teleopit.inputs.realtime_packet import (
    ControlEvent,
    ControlEventType,
    HumanFrame,
    RealtimeInputPacket,
)
from teleopit.inputs.rot_utils import quat_mul_np
from teleopit.interfaces import RealtimeInputProvider
from teleopit.sim.reference_motion import interpolate_human_frames

logger = logging.getLogger(__name__)

# PICO native -> Teleopit retarget input space.
_INPUT_TO_TELEOPIT_MATRIX = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
_INPUT_TO_TELEOPIT_QUAT = R.from_matrix(_INPUT_TO_TELEOPIT_MATRIX).as_quat(scalar_first=True)

BODY_JOINT_NAMES = [
    "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee",
    "Spine2", "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot",
    "Neck", "Left_Collar", "Right_Collar", "Head", "Left_Shoulder", "Right_Shoulder",
    "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist", "Left_Hand", "Right_Hand",
]
BODY_JOINT_PARENTS = np.array(
    [
        -1,
        0, 0, 0, 1, 2,
        3, 4, 5, 6, 7, 8,
        9, 12, 12, 12, 13, 14,
        16, 17, 18, 19, 20, 21,
    ],
    dtype=np.int32,
)


@dataclass(frozen=True)
class PicoControllerState:
    """Latest per-controller input state exposed by pico_bridge."""

    raw: bool
    grip: float
    trigger: float
    present: bool = True


@dataclass(frozen=True)
class PicoControllerSnapshot:
    """Immutable snapshot of Pico controller inputs for auxiliary runtimes."""

    left: PicoControllerState
    right: PicoControllerState
    timestamp_s: float
    seq: int


@dataclass(frozen=True)
class PicoHandState:
    """Latest per-hand pose state exposed by pico_bridge."""

    active: bool
    joints: NDArray[np.float64]
    present: bool = True


@dataclass(frozen=True)
class PicoHandSnapshot:
    """Immutable snapshot of Pico hand poses for auxiliary runtimes."""

    left: PicoHandState
    right: PicoHandState
    timestamp_s: float
    seq: int


@dataclass(frozen=True)
class PicoHeadPoseSnapshot:
    """Synchronized HMD and torso orientations for active-neck control."""

    hmd_rotation_wxyz: NDArray[np.float64] | None
    spine3_rotation_wxyz: NDArray[np.float64] | None
    timestamp_s: float
    seq: int


_PAUSE_BUTTON_MAP: dict[str, tuple[str, str]] = {
    "A": ("right", "primaryButton"),
    "B": ("right", "secondaryButton"),
    "X": ("left", "primaryButton"),
    "Y": ("left", "secondaryButton"),
    "left_axis_click": ("left", "axisClick"),
    "right_axis_click": ("right", "axisClick"),
    "left_menu_button": ("left", "menuButton"),
    "right_menu_button": ("right", "menuButton"),
}

# Sentinel used when reading optional button-path attributes from lightweight
# provider shells made against an older Teleopit release.  ``None`` remains a
# meaningful value (explicitly disabled mapping), so it cannot serve as the
# missing marker.
_MISSING_BUTTON_PATH = object()


def _has_non_degenerate_positions(positions: NDArray[np.float64]) -> bool:
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if pos.size == 0:
        return False
    finite_mask = np.all(np.isfinite(pos), axis=1)
    valid_pos = pos[finite_mask]
    if valid_pos.shape[0] < 2:
        return False
    nonzero_pos = valid_pos[np.linalg.norm(valid_pos, axis=1) > 1e-9]
    if nonzero_pos.shape[0] < 2:
        return False
    extent = float(np.max(np.ptp(nonzero_pos, axis=0)))
    return extent > 1e-6


def _compute_ground_alignment_offset(positions: NDArray[np.float64]) -> float:
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if pos.size == 0:
        return 0.0
    finite_mask = np.all(np.isfinite(pos), axis=1)
    if not np.any(finite_mask):
        return 0.0
    min_z = float(np.min(pos[finite_mask, 2]))
    return -min_z


def _bridge_accepts_video_enabled(bridge_cls: type[Any]) -> bool:
    try:
        signature = inspect.signature(bridge_cls)
    except (TypeError, ValueError):
        return True
    parameters = signature.parameters
    if "video_enabled" in parameters:
        return True
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())


def _installed_pico_bridge_version() -> tuple[int, ...] | None:
    try:
        raw_version = version("pico-bridge")
    except PackageNotFoundError:
        return None
    release = raw_version.split("+", 1)[0].split("-", 1)[0]
    parts: list[int] = []
    for part in release.split("."):
        if not part.isdigit():
            break
        parts.append(int(part))
    return tuple(parts) if parts else None


def _coordinate_transform_input(body_pose_dict: dict[str, list]) -> dict[str, list]:
    """Transform provider-space poses into Teleopit's expected coordinates."""
    for body_name, value in body_pose_dict.items():
        x, y, z = value[0]
        qw, qx, qy, qz = value[1]

        orientation = quat_mul_np(
            _INPUT_TO_TELEOPIT_QUAT, np.array([qw, qx, qy, qz]), scalar_first=True
        )
        position = np.array([x, y, z]) @ _INPUT_TO_TELEOPIT_MATRIX.T

        body_pose_dict[body_name] = [position.tolist(), orientation.tolist()]

    return body_pose_dict


def _transform_pico_native_rotation(rotation_xyzw: Any) -> NDArray[np.float64] | None:
    """Convert one PICO-native xyzw orientation into Teleopit wxyz coordinates."""
    try:
        rotation = np.asarray(rotation_xyzw, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if rotation.shape != (4,) or not np.all(np.isfinite(rotation)):
        return None
    quat_wxyz = np.array(
        [rotation[3], rotation[0], rotation[1], rotation[2]],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(quat_wxyz))
    if norm <= 1e-9:
        return None
    transformed = quat_mul_np(
        _INPUT_TO_TELEOPIT_QUAT,
        quat_wxyz / norm,
        scalar_first=True,
    )
    transformed_norm = float(np.linalg.norm(transformed))
    if transformed_norm <= 1e-9 or not np.all(np.isfinite(transformed)):
        return None
    return np.asarray(transformed / transformed_norm, dtype=np.float64)


class Pico4InputProvider(RealtimeInputProvider):
    """Realtime input provider backed by the ``pico_bridge`` receiver."""

    def __init__(
        self,
        human_format: str = "pico_bridge",
        timeout: float = 60.0,
        buffer_size: int = 60,
        timestamp_gap_reset_s: float = 0.15,
        pause_button: str | None = "A",
        pause_debounce_s: float = 0.25,
        arms_button: str | None = "B",
        arms_debounce_s: float | None = None,
        bridge_host: str = "0.0.0.0",
        bridge_port: int = 63901,
        bridge_discovery: bool = True,
        bridge_advertise_ip: str | None = None,
        bridge_video: str | None = None,
        bridge_video_enabled: bool | None = None,
        bridge_start_timeout: float = 10.0,
        bridge_history_size: int = 120,
        bridge_cls: type[Any] | None = None,
    ) -> None:
        if bridge_cls is None:
            try:
                from pico_bridge import PicoBridge
            except ImportError as exc:
                raise ImportError(
                    "pico_bridge is required for Pico4 input. Install the receiver package, "
                    "for example: pip install -e '.[pico4]'"
                ) from exc
            installed_version = _installed_pico_bridge_version()
            if installed_version is None or installed_version < (0, 2, 1):
                raise RuntimeError(
                    "pico_bridge >= 0.2.1 is required for Pico4 input. Reinstall the Pico extra with "
                    "pip install -e '.[pico4]' so Teleopit receives pico_native tracking semantics."
                )
            bridge_cls = PicoBridge
        if not _bridge_accepts_video_enabled(bridge_cls):
            raise RuntimeError(
                "pico_bridge >= 0.2.1 is required for Pico4 input. Reinstall the Pico extra with "
                "pip install -e '.[pico4]' so PicoBridge accepts video_enabled and push_video_frame()."
            )

        self._human_format = human_format
        try:
            self._timeout = float(timeout)
            self._timestamp_gap_reset_s = float(timestamp_gap_reset_s)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("pico4 timeout and timestamp gap must be numeric") from exc
        if not np.isfinite(self._timeout) or self._timeout < 0.0:
            raise ValueError("pico4_timeout must be finite and non-negative")
        if not np.isfinite(self._timestamp_gap_reset_s) or self._timestamp_gap_reset_s < 0.0:
            raise ValueError("pico4_timestamp_gap_reset_s must be finite and non-negative")
        self._closed = False
        self._shutdown_complete = False
        self._frame_ready = threading.Event()
        self._lock = threading.Lock()
        self._frame_cache = RealtimeFrameCache[HumanFrame](buffer_size=buffer_size, fps_window=30)
        self._pending_control_events: deque[ControlEvent] = deque()
        self._pause_button = None if pause_button in (None, "", "null") else str(pause_button)
        self._arms_button = None if arms_button in (None, "", "null") else str(arms_button)
        try:
            self._pause_debounce_s = float(pause_debounce_s)
            self._arms_debounce_s = (
                self._pause_debounce_s
                if arms_debounce_s is None
                else float(arms_debounce_s)
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("pico4 button debounce values must be numeric") from exc
        if (
            not np.isfinite(self._pause_debounce_s)
            or self._pause_debounce_s < 0.0
            or not np.isfinite(self._arms_debounce_s)
            or self._arms_debounce_s < 0.0
        ):
            raise ValueError("pico4 button debounce values must be finite and non-negative")
        self._pause_button_path = self._resolve_button_path(self._pause_button)
        self._arms_button_path = self._resolve_button_path(self._arms_button)
        # Top-level Pico sim2sim mode controls follow the released controller
        # convention independently of the configurable pause/arms bindings:
        # Y enters MOCAP and X returns to STANDING.  They are harmless for
        # sim2real (which consumes only pause/arms events), but keeping the
        # mapping in the provider lets a non-interactive sim2sim process start
        # in STANDING and still be fully controller-driven.
        self._mocap_button_path = self._resolve_button_path("Y")
        self._standing_button_path = self._resolve_button_path("X")
        self._last_pause_button_pressed = False
        self._last_arms_button_pressed = False
        self._last_mocap_button_pressed = False
        self._last_standing_button_pressed = False
        self._last_pause_toggle_timestamp: float | None = None
        self._last_arms_toggle_timestamp: float | None = None
        self._last_mocap_toggle_timestamp: float | None = None
        self._last_standing_toggle_timestamp: float | None = None
        self._last_raw_body_joints: NDArray[np.float64] | None = None
        self._last_frame_timestamp: float | None = None
        self._last_source_seq: int | None = None
        self._controller_snapshot: PicoControllerSnapshot | None = None
        self._hand_snapshot: PicoHandSnapshot | None = None
        self._head_pose_snapshot: PicoHeadPoseSnapshot | None = None
        self._ground_alignment_offset: float | None = None
        bridge: Any | None = None
        try:
            bridge = bridge_cls(
                host=bridge_host,
                port=int(bridge_port),
                discovery=bool(bridge_discovery),
                advertise_ip=bridge_advertise_ip,
                video=bridge_video,
                video_enabled=bridge_video_enabled,
                history_size=int(bridge_history_size),
                start_timeout=float(bridge_start_timeout),
            )
            self._bridge = bridge
            bridge.start()
            self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True, name="pico4_input")
            self._poll_thread.start()
        except BaseException:
            # ``PicoBridge.start`` can create receiver/video workers before
            # reporting a connection error.  Roll those workers back when
            # provider construction fails; otherwise a failed pipeline setup
            # leaves a live native receiver behind with no owner to close it.
            self._closed = True
            if bridge is not None:
                close = getattr(bridge, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as exc:
                        logger.warning("Failed to close pico_bridge after startup error: %s", exc)
            raise
        if self._pause_button is not None and self._pause_button_path is None:
            logger.warning(
                "Pico4InputProvider pause button '%s' is unsupported by pico_bridge; pause events disabled",
                self._pause_button,
            )
        if self._arms_button is not None and self._arms_button_path is None:
            logger.warning(
                "Pico4InputProvider arms button '%s' is unsupported by pico_bridge; arms events disabled",
                self._arms_button,
            )
        logger.info("Pico4InputProvider initialized (pico_bridge)")

    @property
    def fps(self) -> float:
        with self._lock:
            return self._frame_cache.fps()

    @property
    def supports_mode_control_events(self) -> bool:
        """Whether Pico controller samples can drive the top-level sim mode.

        ``SimulationLoop`` uses this capability to start realtime runs in
        ``STANDING`` even when no interactive terminal is available.  The
        actual Y/X edges are emitted as ``ENTER_MOCAP``/
        ``ENTER_STANDING`` below; A/B remain the pause/arms events.
        """

        return self._mocap_button_path is not None or self._standing_button_path is not None

    @property
    def human_format(self) -> str:
        return self._human_format

    @property
    def bone_names(self) -> list[str]:
        return list(BODY_JOINT_NAMES)

    @property
    def bone_parents(self) -> NDArray[np.int32]:
        return BODY_JOINT_PARENTS.copy()

    def is_available(self) -> bool:
        return not self._closed and self._poll_thread.is_alive()

    def get_frame(self) -> HumanFrame:
        with self._lock:
            if len(self._frame_cache) > 0:
                return self._frame_cache.latest()
        if not self._frame_ready.wait(timeout=self._timeout):
            raise TimeoutError(f"No Pico4 body data received within {self._timeout:.1f}s timeout")
        with self._lock:
            if len(self._frame_cache) <= 0:
                raise RuntimeError("Pico4 frame buffer signaled ready without a latest frame")
            return self._frame_cache.latest()

    def get_frame_packet(self) -> tuple[HumanFrame, float, int]:
        with self._lock:
            if len(self._frame_cache) > 0:
                return self._frame_cache.latest_packet()
        if not self._frame_ready.wait(timeout=self._timeout):
            raise TimeoutError(f"No Pico4 body data received within {self._timeout:.1f}s timeout")
        with self._lock:
            if len(self._frame_cache) <= 0:
                raise RuntimeError("Pico4 frame buffer signaled ready without a latest frame")
            return self._frame_cache.latest_packet()

    def get_realtime_input_packet(self) -> RealtimeInputPacket[HumanFrame]:
        frame, timestamp_s, seq = self.get_frame_packet()
        with self._lock:
            control_events = tuple(self._pending_control_events)
            self._pending_control_events.clear()
        return RealtimeInputPacket(
            frame=frame,
            timestamp_s=timestamp_s,
            seq=seq,
            control_events=control_events,
        )

    def pop_control_events(self) -> tuple[ControlEvent, ...]:
        with self._lock:
            control_events = tuple(self._pending_control_events)
            self._pending_control_events.clear()
        return control_events

    def get_controller_snapshot(self) -> PicoControllerSnapshot | None:
        """Return the latest Pico controller-axis snapshot, if one has arrived."""
        with self._lock:
            return self._controller_snapshot

    def get_hand_snapshot(self) -> PicoHandSnapshot | None:
        """Return the latest Pico hand-pose snapshot, if one has arrived."""
        with self._lock:
            return self._hand_snapshot

    def get_head_pose_snapshot(self) -> PicoHeadPoseSnapshot | None:
        """Return the latest synchronized HMD/Spine3 orientation snapshot."""
        with self._lock:
            return self._head_pose_snapshot

    def push_video_frame(self, frame: NDArray[np.uint8]) -> int:
        """Push one RGB camera frame to pico-bridge 0.2.1 video output."""
        push_video_frame = getattr(self._bridge, "push_video_frame", None)
        if not callable(push_video_frame):
            raise RuntimeError("Installed pico_bridge does not expose push_video_frame(); use pico-bridge 0.2.1")
        return int(push_video_frame(frame))

    def has_frame(self) -> bool:
        with self._lock:
            return len(self._frame_cache) > 0

    def sample_frame(self, query_time_s: float, delay_s: float) -> HumanFrame:
        if not self._frame_ready.wait(timeout=self._timeout):
            raise TimeoutError(f"No Pico4 body data received within {self._timeout:.1f}s timeout")
        with self._lock:
            buf = self._frame_cache.snapshot()

        if not buf:
            raise RuntimeError("Pico4 frame buffer signaled ready without a latest frame")
        if len(buf) == 1:
            return buf[0][0]

        target_time = float(query_time_s - max(delay_s, 0.0))
        if target_time <= buf[0][1]:
            return buf[0][0]
        if target_time >= buf[-1][1]:
            return buf[-1][0]

        for i in range(1, len(buf)):
            older_frame, older_ts = buf[i - 1]
            newer_frame, newer_ts = buf[i]
            if target_time <= newer_ts:
                dt = newer_ts - older_ts
                if dt <= 1e-6:
                    return newer_frame
                alpha = float(np.clip((target_time - older_ts) / dt, 0.0, 1.0))
                return interpolate_human_frames(older_frame, newer_frame, alpha)

        return buf[-1][0]

    def close(self) -> None:
        # Cleanup is reached from both normal pipeline teardown and error
        # handlers.  PicoBridge's native close is not guaranteed idempotent,
        # so latch completion before joining/closing and make concurrent calls
        # harmless.  ``getattr`` keeps lightweight legacy provider shells
        # (used by downstream integrations/tests) compatible.
        if getattr(self, "_shutdown_complete", False):
            return
        self._shutdown_complete = True
        self._closed = True
        poll_thread = getattr(self, "_poll_thread", None)
        if poll_thread is not None and poll_thread is not threading.current_thread():
            poll_thread.join(timeout=3.0)
        close = getattr(getattr(self, "_bridge", None), "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                # A disconnected headset must not turn ordinary shutdown into
                # a failed simulation exit.  The native worker is daemonized
                # by pico_bridge, so log and continue if its deinitializer
                # reports a transient teardown error.
                logger.warning("Failed to close pico_bridge: %s", exc)
        logger.info("Pico4InputProvider closed")

    def _poll_loop(self) -> None:
        while not self._closed:
            try:
                frame = self._bridge.wait_frame(timeout=0.1, after_seq=self._last_source_seq)
            except TimeoutError:
                continue
            except Exception:
                if not self._closed:
                    logger.exception("Failed to read pico_bridge frame")
                    time.sleep(0.05)
                continue

            # A malformed optional field (for example a transient controller
            # axis value while the headset reconnects) must not terminate the
            # receiver thread.  ``_accept_pico_frame`` validates the body and
            # snapshots individually, but it intentionally stays a small
            # synchronous adapter; keep the process boundary resilient here
            # and continue waiting for the next bridge frame after an
            # unexpected SDK/schema exception.
            try:
                self._accept_pico_frame(frame)
            except Exception:
                if not self._closed:
                    logger.exception("Failed to adapt pico_bridge frame")
                    time.sleep(0.05)

    def _frame_timestamp(self, frame: Any) -> float:
        """Return a finite receive timestamp for one bridge frame.

        ``receive_time_s`` is metadata supplied by the Python bridge rather
        than part of the body payload.  During reconnects older bridge builds
        have briefly exposed ``None``/string/NaN values; those must not poison
        the interpolation cache or make its timestamp move backwards.  Fall
        back to the local monotonic clock while preserving valid values.
        """

        try:
            timestamp = float(getattr(frame, "receive_time_s"))
        except (AttributeError, TypeError, ValueError, OverflowError, RuntimeError):
            return time.monotonic()
        return timestamp if np.isfinite(timestamp) else time.monotonic()

    def _frame_sequence(self, frame: Any) -> int | None:
        """Read an optional non-negative source sequence without coercion."""

        try:
            value = getattr(frame, "seq")
        except (AttributeError, RuntimeError):
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
            return None
        sequence = int(value)
        return sequence if sequence >= 0 else None

    def _remember_source_sequence(self, frame: Any) -> None:
        sequence = self._frame_sequence(frame)
        if sequence is not None:
            self._last_source_seq = sequence

    def _accept_pico_frame(self, frame: Any) -> bool:
        timestamp = self._frame_timestamp(frame)
        source_sequence = self._frame_sequence(frame)
        self._accept_head_pose_snapshot(frame, timestamp=timestamp)
        self._accept_controller_snapshot(frame, timestamp=timestamp)
        self._accept_hand_snapshot(frame, timestamp=timestamp)

        # Pause/arms controls are safety/session controls and must remain
        # usable while Full-body tracking is temporarily inactive (for
        # example while the operator is recalibrating ankle trackers).  The
        # Y/X mode controls are handled below only after a valid body sample
        # exists, because entering MOCAP without a reference frame would
        # consume the edge while the session is still in STANDING.
        self._poll_legacy_control_events(frame, timestamp=timestamp)

        try:
            body = getattr(frame, "body", None)
        except Exception:
            body = None
        try:
            body_active = getattr(body, "active", False) if body is not None else False
        except Exception:
            body_active = False
        if body is None or not isinstance(body_active, (bool, np.bool_)) or not bool(body_active):
            if source_sequence is not None:
                self._last_source_seq = source_sequence
            return False

        try:
            body_joints = np.asarray(getattr(body, "joints"), dtype=np.float64)
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            logger.warning("Malformed pico_bridge body joint payload: %s", exc)
            if source_sequence is not None:
                self._last_source_seq = source_sequence
            return False
        if body_joints.shape != (len(BODY_JOINT_NAMES), 7):
            logger.warning("Unexpected pico_bridge body joint shape: %s", body_joints.shape)
            if source_sequence is not None:
                self._last_source_seq = source_sequence
            return False
        if not np.all(np.isfinite(body_joints)):
            # Do not consume a Y/X mode-button edge for a malformed body
            # packet.  PicoBridge can briefly publish a correctly shaped
            # array containing NaN/Inf while tracking is recovering; that
            # packet must not enter the retargeter or arm a new session.
            logger.warning("pico_bridge body joint sample contains non-finite values")
            if source_sequence is not None:
                self._last_source_seq = source_sequence
            return False

        # Mode-control chords are meaningful only with a valid full-body
        # sample.  PicoBridge can continue publishing controller/HMD frames
        # while body tracking is inactive; polling buttons before this check
        # would queue an activation pressed during that gap and apply it to
        # the first unrelated body frame after reconnect.  Keep this before
        # duplicate-body suppression so a stationary operator can still press
        # a mode button without moving the skeleton.
        self._poll_mode_control_events(frame, timestamp=timestamp)

        if self._last_raw_body_joints is not None and np.array_equal(body_joints, self._last_raw_body_joints):
            if source_sequence is not None:
                self._last_source_seq = source_sequence
            return False

        human_frame = self._convert_body_joints_to_frame(body_joints)
        with self._lock:
            if (
                self._last_frame_timestamp is not None
                and self._timestamp_gap_reset_s > 0.0
                and timestamp - self._last_frame_timestamp > self._timestamp_gap_reset_s
            ):
                self._frame_cache.clear()
                self._ground_alignment_offset = None
                logger.warning(
                    "Pico4InputProvider timestamp-gap reset | gap=%.4fs",
                    timestamp - self._last_frame_timestamp,
                )
            if self._last_frame_timestamp is not None and timestamp <= self._last_frame_timestamp + 1e-9:
                timestamp = self._last_frame_timestamp + 1e-6

            human_frame = self._apply_ground_alignment(human_frame)
            self._frame_cache.append(human_frame, timestamp, fps_timestamp=timestamp)
            self._last_raw_body_joints = body_joints.copy()
            self._last_frame_timestamp = timestamp
            if source_sequence is not None:
                self._last_source_seq = source_sequence

        self._frame_ready.set()
        return True

    def _accept_controller_snapshot(self, frame: Any, *, timestamp: float) -> None:
        seq = self._frame_sequence(frame)
        if seq is None:
            seq = self._last_source_seq if self._last_source_seq is not None else -1
        try:
            controllers = getattr(frame, "controllers", None)
            left_controller = None if controllers is None else getattr(controllers, "left", None)
            right_controller = None if controllers is None else getattr(controllers, "right", None)
        except Exception:
            controllers = None
            left_controller = right_controller = None
        snapshot = PicoControllerSnapshot(
            left=self._read_controller_state(left_controller),
            right=self._read_controller_state(right_controller),
            timestamp_s=float(timestamp),
            seq=seq,
        )
        with self._lock:
            self._controller_snapshot = snapshot

    def _accept_hand_snapshot(self, frame: Any, *, timestamp: float) -> None:
        seq = self._frame_sequence(frame)
        if seq is None:
            seq = self._last_source_seq if self._last_source_seq is not None else -1
        try:
            left_hand = getattr(frame, "left_hand", None)
            right_hand = getattr(frame, "right_hand", None)
        except Exception:
            left_hand = right_hand = None
        snapshot = PicoHandSnapshot(
            left=self._read_hand_state(left_hand),
            right=self._read_hand_state(right_hand),
            timestamp_s=float(timestamp),
            seq=seq,
        )
        with self._lock:
            self._hand_snapshot = snapshot

    def _accept_head_pose_snapshot(self, frame: Any, *, timestamp: float) -> None:
        """Capture HMD and Spine3 rotations from the same pico_bridge frame."""
        seq = self._frame_sequence(frame)
        if seq is None:
            seq = self._last_source_seq if self._last_source_seq is not None else -1

        try:
            head = getattr(frame, "head", None)
        except Exception:
            head = None
        hmd_rotation = _transform_pico_native_rotation(
            None if head is None else self._safe_optional_attr(head, "rotation")
        )

        spine3_rotation: NDArray[np.float64] | None = None
        try:
            body = getattr(frame, "body", None)
            body_active = getattr(body, "active", False) if body is not None else False
        except Exception:
            body = None
            body_active = False
        if body is not None and isinstance(body_active, (bool, np.bool_)) and bool(body_active):
            try:
                body_joints = np.asarray(getattr(body, "joints"), dtype=np.float64)
            except (AttributeError, TypeError, ValueError, OverflowError, RuntimeError):
                body_joints = np.empty((0, 0), dtype=np.float64)
            if body_joints.shape == (len(BODY_JOINT_NAMES), 7):
                spine3 = body_joints[BODY_JOINT_NAMES.index("Spine3")]
                spine3_rotation = _transform_pico_native_rotation(spine3[[3, 4, 5, 6]])

        snapshot = PicoHeadPoseSnapshot(
            hmd_rotation_wxyz=hmd_rotation,
            spine3_rotation_wxyz=spine3_rotation,
            timestamp_s=float(timestamp),
            seq=seq,
        )
        with self._lock:
            self._head_pose_snapshot = snapshot

    @staticmethod
    def _safe_optional_attr(value: Any, name: str, default: Any = None) -> Any:
        """Read an SDK proxy attribute without aborting frame adaptation."""

        try:
            return getattr(value, name, default)
        except Exception:
            return default

    def _poll_control_events(self, frame: Any, *, timestamp: float) -> bool:
        """Poll all controller controls.

        This compatibility wrapper retains the historical private helper
        surface.  Frame ingestion calls the two groups separately so legacy
        pause/arms events remain available before Full-body tracking starts,
        while Y/X mode events wait for a valid body sample.
        """
        emitted = self._poll_legacy_control_events(frame, timestamp=timestamp)
        return self._poll_mode_control_events(frame, timestamp=timestamp) or emitted

    def _poll_legacy_control_events(self, frame: Any, *, timestamp: float) -> bool:
        """Poll configurable pause/arms button edges (A/B by default)."""
        emitted = False
        emitted = self._poll_button_control_event(
            frame,
            timestamp=timestamp,
            button_path=self._pause_button_path,
            button_label=self._pause_button,
            event_type=ControlEventType.TOGGLE_PAUSE,
            last_pressed_attr="_last_pause_button_pressed",
            last_toggle_attr="_last_pause_toggle_timestamp",
            debounce_s=self._pause_debounce_s,
        ) or emitted
        emitted = self._poll_button_control_event(
            frame,
            timestamp=timestamp,
            button_path=self._arms_button_path,
            button_label=self._arms_button,
            event_type=ControlEventType.TOGGLE_ARMS,
            last_pressed_attr="_last_arms_button_pressed",
            last_toggle_attr="_last_arms_toggle_timestamp",
            debounce_s=self._arms_debounce_s,
        ) or emitted
        return emitted

    def _poll_mode_control_events(self, frame: Any, *, timestamp: float) -> bool:
        """Poll Y/X mode button edges after body shape/activity validation."""
        # Resolve optional attributes lazily for compatibility with provider
        # shells created by older integrations.  A real constructor always
        # stores these paths, including an explicit ``None`` when a binding
        # does not expose the requested button; do not use ``or`` here because
        # that would accidentally re-enable a deliberately disabled mapping.
        mocap_button_path = getattr(self, "_mocap_button_path", _MISSING_BUTTON_PATH)
        if mocap_button_path is _MISSING_BUTTON_PATH:
            mocap_button_path = self._resolve_button_path("Y")
        standing_button_path = getattr(self, "_standing_button_path", _MISSING_BUTTON_PATH)
        if standing_button_path is _MISSING_BUTTON_PATH:
            standing_button_path = self._resolve_button_path("X")
        emitted = False
        emitted = self._poll_button_control_event(
            frame,
            timestamp=timestamp,
            button_path=mocap_button_path,
            button_label="Y",
            event_type=ControlEventType.ENTER_MOCAP,
            last_pressed_attr="_last_mocap_button_pressed",
            last_toggle_attr="_last_mocap_toggle_timestamp",
            debounce_s=float(getattr(self, "_pause_debounce_s", 0.25)),
        ) or emitted
        emitted = self._poll_button_control_event(
            frame,
            timestamp=timestamp,
            button_path=standing_button_path,
            button_label="X",
            event_type=ControlEventType.ENTER_STANDING,
            last_pressed_attr="_last_standing_button_pressed",
            last_toggle_attr="_last_standing_toggle_timestamp",
            debounce_s=float(getattr(self, "_pause_debounce_s", 0.25)),
        ) or emitted
        return emitted

    def _poll_button_control_event(
        self,
        frame: Any,
        *,
        timestamp: float,
        button_path: tuple[str, str] | None,
        button_label: str | None,
        event_type: ControlEventType,
        last_pressed_attr: str,
        last_toggle_attr: str,
        debounce_s: float,
    ) -> bool:
        if button_path is None:
            return False

        side, button_name = button_path
        try:
            controllers = getattr(frame, "controllers", None)
            controller = None if controllers is None else getattr(controllers, side, None)
            raw_buttons = {} if controller is None else getattr(controller, "buttons", {})
        except Exception as exc:
            logger.debug("Malformed Pico controller button container: %s", exc)
            controllers = None
            controller = None
            raw_buttons = {}
        # ``buttons`` is normally a dict, but during an XR service reconnect
        # some SDK builds briefly expose ``None`` or a proxy object.  Treat a
        # malformed container as an unpressed sample instead of allowing it
        # to abort adaptation of an otherwise valid body frame.
        buttons = raw_buttons if hasattr(raw_buttons, "get") else {}
        try:
            raw_pressed = buttons.get(button_name, False)
        except Exception as exc:
            logger.debug("Malformed Pico controller button value: %s", exc)
            raw_pressed = False
        # Do not use ``bool(raw_pressed)`` here: values such as the string
        # ``"false"`` are truthy in Python and could spuriously trigger a
        # safety/session edge.  NumPy bool scalars are accepted because a few
        # bindings return them directly from their button map.
        pressed = bool(raw_pressed) if isinstance(raw_pressed, (bool, np.bool_)) else False
        # Keep this helper compatible with lightweight/legacy provider
        # instances constructed by downstream integrations (and older test
        # fixtures) before the Y/X mode-control latches were introduced.
        # Production instances initialize all of these attributes eagerly;
        # ``getattr`` here makes adding a new optional mode event non-breaking
        # for callers that subclass or deserialize a provider shell.
        last_pressed = bool(getattr(self, last_pressed_attr, False))
        emitted = False
        if pressed and not last_pressed:
            last_toggle = getattr(self, last_toggle_attr, None)
            if last_toggle is None or timestamp - float(last_toggle) >= debounce_s - 1e-9:
                with self._lock:
                    self._pending_control_events.append(
                        ControlEvent(
                            event_type=event_type,
                            source=f"pico4:{button_label}",
                            timestamp_s=float(timestamp),
                        )
                    )
                logger.info("Pico control event: %s from %s", event_type.value, button_label)
                setattr(self, last_toggle_attr, float(timestamp))
                emitted = True
        setattr(self, last_pressed_attr, pressed)
        return emitted

    @staticmethod
    def _resolve_button_path(button: str | None) -> tuple[str, str] | None:
        if button is None:
            return None
        return _PAUSE_BUTTON_MAP.get(button)

    @staticmethod
    def _read_controller_state(controller: Any) -> PicoControllerState:
        try:
            axis = {} if controller is None else getattr(controller, "axis", {})
        except Exception:
            axis = {}
        axis = axis if hasattr(axis, "get") else {}

        def read_axis(name: str) -> tuple[float, bool]:
            try:
                raw = axis.get(name, 0.0)
            except Exception:
                return 0.0, False
            # Keep malformed optional controller fields local to the snapshot;
            # body tracking and the other controller can continue normally.
            # Do not let ``float("0.5")`` turn a malformed wire/string value
            # into a valid trigger.  SDK values are numeric scalars; accepting
            # only ``numbers.Real`` also rejects arrays and arbitrary proxy
            # objects without raising out of the body-frame adapter.
            if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, numbers.Real):
                return 0.0, False
            try:
                value = float(raw)
            except (TypeError, ValueError, OverflowError):
                return 0.0, False
            if not np.isfinite(value) or value < 0.0 or value > 1.0:
                return 0.0, False
            return value, True

        grip, grip_valid = read_axis("grip")
        trigger, trigger_valid = read_axis("trigger")
        present = controller is not None and grip_valid and trigger_valid
        try:
            raw_value = False if controller is None else getattr(controller, "raw", False)
        except Exception:
            raw_value = False
        raw_flag = bool(raw_value) if isinstance(raw_value, (bool, np.bool_)) else False
        return PicoControllerState(
            raw=raw_flag,
            grip=grip,
            trigger=trigger,
            present=present,
        )

    @staticmethod
    def _read_hand_state(hand: Any) -> PicoHandState:
        joints = np.zeros((26, 7), dtype=np.float64)
        valid_shape = False
        if hand is None:
            return PicoHandState(active=False, joints=joints, present=False)
        try:
            raw_joints = np.asarray(getattr(hand, "joints"), dtype=np.float64)
            if raw_joints.shape == (26, 7) and np.all(np.isfinite(raw_joints)):
                joints = raw_joints.copy()
                valid_shape = True
        except (AttributeError, TypeError, ValueError, OverflowError):
            pass
        try:
            active_value = getattr(hand, "active", False)
        except Exception:
            active_value = False
        active = bool(active_value) if isinstance(active_value, (bool, np.bool_)) else False
        return PicoHandState(
            active=active and valid_shape,
            joints=joints,
            present=True,
        )

    @staticmethod
    def _convert_body_joints_to_frame(body_joints: NDArray[np.float64]) -> HumanFrame:
        body_pose_dict: dict[str, list] = {}
        for i, joint_name in enumerate(BODY_JOINT_NAMES):
            pos = [body_joints[i][0], body_joints[i][1], body_joints[i][2]]
            # pico_bridge 0.2.1 returns pico_native [x, y, z, qx, qy, qz, qw].
            rot = [body_joints[i][6], body_joints[i][3], body_joints[i][4], body_joints[i][5]]
            body_pose_dict[joint_name] = [pos, rot]

        body_pose_dict = _coordinate_transform_input(body_pose_dict)

        result: HumanFrame = {}
        for name, (pos, quat) in body_pose_dict.items():
            result[name] = (np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64))
        return result

    def _apply_ground_alignment(self, human_frame: HumanFrame) -> HumanFrame:
        """Apply one fixed Z offset so the initial Pico skeleton sits on the floor."""
        if self._ground_alignment_offset is None:
            positions = np.asarray([value[0] for value in human_frame.values()], dtype=np.float64)
            if _has_non_degenerate_positions(positions):
                self._ground_alignment_offset = _compute_ground_alignment_offset(positions)
            else:
                return human_frame

        offset = float(self._ground_alignment_offset)
        if abs(offset) <= 1e-12:
            return human_frame

        z_offset = np.array([0.0, 0.0, offset], dtype=np.float64)
        lifted: HumanFrame = {}
        for name, (pos, quat) in human_frame.items():
            lifted[name] = (np.asarray(pos, dtype=np.float64) + z_offset, np.asarray(quat, dtype=np.float64))
        return lifted
