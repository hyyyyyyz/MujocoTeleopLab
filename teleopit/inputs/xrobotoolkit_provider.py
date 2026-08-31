"""XRoboToolkit-backed Pico full-body input for Teleopit's MuJoCo runtime.

The native SDK owns the PC-side connection to the XRoboToolkit headset app.
This provider deliberately only adapts its 24-joint body stream to Teleopit's
realtime input contract; it does not start, stop, or otherwise manage the
system-wide RoboticsService process.
"""

from __future__ import annotations

from collections import deque
import logging
import threading
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from teleopit.inputs.pico4_provider import (
    BODY_JOINT_NAMES,
    BODY_JOINT_PARENTS,
    PicoControllerSnapshot,
    PicoControllerState,
    PicoHeadPoseSnapshot,
    _compute_ground_alignment_offset,
    _has_non_degenerate_positions,
    _transform_pico_native_rotation,
)
from teleopit.inputs.realtime_frame_cache import RealtimeFrameCache
from teleopit.inputs.realtime_packet import ControlEvent, ControlEventType, HumanFrame, RealtimeInputPacket
from teleopit.inputs.pico4_provider import Pico4InputProvider
from teleopit.interfaces import RealtimeInputProvider
from teleopit.sim.reference_motion import interpolate_human_frames

logger = logging.getLogger(__name__)


class XRoboToolkitInputProvider(RealtimeInputProvider):
    """Read Pico body tracking from ``xrobotoolkit_sdk`` on the local PC."""

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
        poll_hz: float = 120.0,
        close_sdk: bool = True,
        sdk_shutdown_settle_s: float = 1.0,
        sdk: Any | None = None,
    ) -> None:
        if poll_hz <= 0:
            raise ValueError("xrobotoolkit_poll_hz must be positive")
        if sdk is None:
            try:
                import xrobotoolkit_sdk as sdk_module
            except ImportError as exc:
                raise ImportError(
                    "xrobotoolkit_sdk is required for input.provider=xrobotoolkit. "
                    "Run scripts/setup/setup_xrobotoolkit.sh first."
                ) from exc
            sdk = sdk_module

        self._sdk = sdk
        self._human_format = str(human_format)
        self._timeout = float(timeout)
        self._timestamp_gap_reset_s = float(timestamp_gap_reset_s)
        self._poll_period_s = 1.0 / float(poll_hz)
        self._close_sdk = bool(close_sdk)
        self._sdk_shutdown_settle_s = max(float(sdk_shutdown_settle_s), 0.0)
        self._pause_button = None if pause_button in (None, "", "null") else str(pause_button)
        self._arms_button = None if arms_button in (None, "", "null") else str(arms_button)
        self._pause_debounce_s = max(float(pause_debounce_s), 0.0)
        self._arms_debounce_s = self._pause_debounce_s if arms_debounce_s is None else max(float(arms_debounce_s), 0.0)
        self._lock = threading.Lock()
        self._closed = False
        self._frame_ready = threading.Event()
        self._frame_cache = RealtimeFrameCache[HumanFrame](buffer_size=buffer_size, fps_window=30)
        self._pending_control_events: deque[ControlEvent] = deque()
        self._last_body_timestamp_ns: int | None = None
        self._last_raw_body_joints: NDArray[np.float64] | None = None
        self._last_frame_timestamp: float | None = None
        self._last_pause_pressed = False
        self._last_arms_pressed = False
        self._last_pause_toggle_timestamp: float | None = None
        self._last_arms_toggle_timestamp: float | None = None
        self._ground_alignment_offset: float | None = None
        self._controller_snapshot: PicoControllerSnapshot | None = None
        self._head_pose_snapshot: PicoHeadPoseSnapshot | None = None

        init = getattr(self._sdk, "init", None)
        if not callable(init):
            raise RuntimeError("xrobotoolkit_sdk does not expose init()")
        init()
        self._sdk_started_at = time.monotonic()
        self._poll_thread = threading.Thread(target=self._poll_loop, name="xrobotoolkit_input", daemon=True)
        self._poll_thread.start()
        logger.info("XRoboToolkitInputProvider initialized")

    @property
    def fps(self) -> float:
        with self._lock:
            return self._frame_cache.fps()

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

    def has_frame(self) -> bool:
        with self._lock:
            return len(self._frame_cache) > 0

    def get_frame(self) -> HumanFrame:
        return self.get_frame_packet()[0]

    def get_frame_packet(self) -> tuple[HumanFrame, float, int]:
        with self._lock:
            if len(self._frame_cache):
                return self._frame_cache.latest_packet()
        if not self._frame_ready.wait(timeout=self._timeout):
            raise TimeoutError(
                f"No XRoboToolkit body data received within {self._timeout:.1f}s. "
                "Start RoboticsService, connect the headset, and enable Full-body + Send in XRoboToolkit."
            )
        with self._lock:
            return self._frame_cache.latest_packet()

    def get_realtime_input_packet(self) -> RealtimeInputPacket[HumanFrame]:
        frame, timestamp_s, seq = self.get_frame_packet()
        with self._lock:
            events = tuple(self._pending_control_events)
            self._pending_control_events.clear()
        return RealtimeInputPacket(frame=frame, timestamp_s=timestamp_s, seq=seq, control_events=events)

    def pop_control_events(self) -> tuple[ControlEvent, ...]:
        with self._lock:
            events = tuple(self._pending_control_events)
            self._pending_control_events.clear()
        return events

    def get_controller_snapshot(self) -> PicoControllerSnapshot | None:
        with self._lock:
            return self._controller_snapshot

    def get_head_pose_snapshot(self) -> PicoHeadPoseSnapshot | None:
        with self._lock:
            return self._head_pose_snapshot

    def sample_frame(self, query_time_s: float, delay_s: float) -> HumanFrame:
        self.get_frame_packet()
        with self._lock:
            frames = self._frame_cache.snapshot()
        target = float(query_time_s - max(delay_s, 0.0))
        if len(frames) == 1 or target <= frames[0][1]:
            return frames[0][0]
        if target >= frames[-1][1]:
            return frames[-1][0]
        for index in range(1, len(frames)):
            older, older_ts = frames[index - 1]
            newer, newer_ts = frames[index]
            if target <= newer_ts:
                if newer_ts - older_ts <= 1e-6:
                    return newer
                return interpolate_human_frames(older, newer, float(np.clip((target - older_ts) / (newer_ts - older_ts), 0.0, 1.0)))
        return frames[-1][0]

    def close(self) -> None:
        self._closed = True
        self._poll_thread.join(timeout=3.0)
        close = getattr(self._sdk, "close", None)
        if self._close_sdk and callable(close):
            # SDK v1.0.2 has a startup race: close() can deadlock if its
            # feedback-stream thread has not installed its cancellation
            # context yet.  A short settle window lets the local PC service
            # accept the stream before teardown.
            remaining = self._sdk_shutdown_settle_s - (time.monotonic() - self._sdk_started_at)
            if remaining > 0.0:
                time.sleep(remaining)
            close()
        logger.info("XRoboToolkitInputProvider closed")

    def _poll_loop(self) -> None:
        while not self._closed:
            try:
                self._poll_once()
            except Exception:
                if not self._closed:
                    logger.exception("Failed to read XRoboToolkit state")
            time.sleep(self._poll_period_s)

    def _poll_once(self) -> bool:
        timestamp = time.monotonic()
        self._accept_auxiliary_state(timestamp)
        available = getattr(self._sdk, "is_body_data_available", None)
        if not callable(available) or not bool(available()):
            return False
        raw_joints = np.asarray(self._sdk.get_body_joints_pose(), dtype=np.float64)
        if raw_joints.shape != (len(BODY_JOINT_NAMES), 7):
            logger.warning("Unexpected XRoboToolkit body joint shape: %s", raw_joints.shape)
            return False
        source_ns = self._read_body_timestamp_ns()
        if source_ns is not None and source_ns == self._last_body_timestamp_ns:
            return False
        if source_ns is None and self._last_raw_body_joints is not None and np.array_equal(raw_joints, self._last_raw_body_joints):
            return False
        return self._accept_body_joints(raw_joints, timestamp=timestamp, source_ns=source_ns)

    def _read_body_timestamp_ns(self) -> int | None:
        getter = getattr(self._sdk, "get_body_timestamp_ns", None)
        if not callable(getter):
            return None
        try:
            value = int(getter())
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def _accept_body_joints(self, raw_joints: NDArray[np.float64], *, timestamp: float, source_ns: int | None) -> bool:
        frame = Pico4InputProvider._convert_body_joints_to_frame(raw_joints)
        with self._lock:
            if self._last_frame_timestamp is not None and timestamp - self._last_frame_timestamp > self._timestamp_gap_reset_s:
                self._frame_cache.clear()
                self._ground_alignment_offset = None
                logger.warning("XRoboToolkitInputProvider timestamp-gap reset | gap=%.4fs", timestamp - self._last_frame_timestamp)
            frame = self._apply_ground_alignment(frame)
            self._frame_cache.append(frame, timestamp, fps_timestamp=timestamp)
            self._last_frame_timestamp = timestamp
            self._last_raw_body_joints = raw_joints.copy()
            self._last_body_timestamp_ns = source_ns
        self._frame_ready.set()
        return True

    def _accept_auxiliary_state(self, timestamp: float) -> None:
        def value(name: str, default: float | bool = 0.0) -> float | bool:
            fn = getattr(self._sdk, name, None)
            return fn() if callable(fn) else default

        with self._lock:
            seq = self._frame_cache.frame_seq
        snapshot = PicoControllerSnapshot(
            left=PicoControllerState(raw=True, grip=float(value("get_left_grip")), trigger=float(value("get_left_trigger"))),
            right=PicoControllerState(raw=True, grip=float(value("get_right_grip")), trigger=float(value("get_right_trigger"))),
            timestamp_s=timestamp,
            seq=seq,
        )
        head = np.asarray(value("get_headset_pose", np.zeros(7)), dtype=np.float64)
        hmd_rotation = _transform_pico_native_rotation(head[3:7] if head.shape == (7,) else None)
        with self._lock:
            self._controller_snapshot = snapshot
            self._head_pose_snapshot = PicoHeadPoseSnapshot(hmd_rotation, None, timestamp, seq)
        self._poll_button("get_A_button", self._pause_button, ControlEventType.TOGGLE_PAUSE, timestamp)
        self._poll_button("get_B_button", self._arms_button, ControlEventType.TOGGLE_ARMS, timestamp)

    def _poll_button(self, sdk_name: str, configured_button: str | None, event_type: ControlEventType, timestamp: float) -> None:
        if configured_button is None:
            return
        expected = "A" if event_type is ControlEventType.TOGGLE_PAUSE else "B"
        if configured_button != expected:
            return
        getter = getattr(self._sdk, sdk_name, None)
        pressed = bool(getter()) if callable(getter) else False
        last_attr = "_last_pause_pressed" if event_type is ControlEventType.TOGGLE_PAUSE else "_last_arms_pressed"
        toggle_attr = "_last_pause_toggle_timestamp" if event_type is ControlEventType.TOGGLE_PAUSE else "_last_arms_toggle_timestamp"
        debounce = self._pause_debounce_s if event_type is ControlEventType.TOGGLE_PAUSE else self._arms_debounce_s
        last_pressed = bool(getattr(self, last_attr))
        last_toggle = getattr(self, toggle_attr)
        if pressed and not last_pressed and (last_toggle is None or timestamp - float(last_toggle) >= debounce):
            with self._lock:
                self._pending_control_events.append(ControlEvent(event_type, f"xrobotoolkit:{configured_button}", timestamp))
            setattr(self, toggle_attr, timestamp)
        setattr(self, last_attr, pressed)

    def _apply_ground_alignment(self, frame: HumanFrame) -> HumanFrame:
        if self._ground_alignment_offset is None:
            positions = np.asarray([pose[0] for pose in frame.values()], dtype=np.float64)
            if not _has_non_degenerate_positions(positions):
                return frame
            self._ground_alignment_offset = _compute_ground_alignment_offset(positions)
        offset = float(self._ground_alignment_offset)
        if abs(offset) <= 1e-12:
            return frame
        delta = np.array([0.0, 0.0, offset], dtype=np.float64)
        return {name: (pos + delta, quat) for name, (pos, quat) in frame.items()}
