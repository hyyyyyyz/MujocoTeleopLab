"""XRoboToolkit-backed Pico full-body input for Teleopit's MuJoCo runtime.

The native SDK owns the PC-side connection to the XRoboToolkit headset app.
This provider deliberately only adapts its 24-joint body stream to Teleopit's
realtime input contract; it does not start, stop, or otherwise manage the
system-wide RoboticsService process.
"""

from __future__ import annotations

from collections import deque
import logging
import numbers
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
from teleopit.inputs.xrobotoolkit_utils import (
    DEFAULT_SDK_CLOSE_TIMEOUT_S,
    close_sdk_bounded,
)
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
        activation_trigger_threshold: float = 0.5,
        reset_grip_threshold: float = 0.5,
        control_debounce_s: float = 0.25,
        poll_hz: float = 120.0,
        close_sdk: bool = True,
        sdk_shutdown_settle_s: float = 1.0,
        sdk_close_timeout_s: float = DEFAULT_SDK_CLOSE_TIMEOUT_S,
        sdk: Any | None = None,
        # Kept as ignored compatibility keywords for callers that used the
        # Pico4 provider's pre-XRoboToolkit constructor.  XRoboToolkit follows
        # SIMPLE's Menu/trigger and dual-grip chords, so silently mapping A/B
        # here would reintroduce the wrong controls.
        pause_button: str | None = None,
        pause_debounce_s: float | None = None,
        arms_button: str | None = None,
        arms_debounce_s: float | None = None,
    ) -> None:
        try:
            poll_hz_value = float(poll_hz)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("xrobotoolkit_poll_hz must be a finite positive number") from exc
        if not np.isfinite(poll_hz_value) or poll_hz_value <= 0.0:
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
        try:
            self._timeout = float(timeout)
            self._timestamp_gap_reset_s = float(timestamp_gap_reset_s)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("XRoboToolkit timeout and timestamp gap must be numeric") from exc
        if not np.isfinite(self._timeout) or self._timeout < 0.0:
            raise ValueError("xrobotoolkit_timeout must be a finite non-negative number")
        if not np.isfinite(self._timestamp_gap_reset_s) or self._timestamp_gap_reset_s < 0.0:
            raise ValueError("pico4_timestamp_gap_reset_s must be a finite non-negative number")
        self._poll_period_s = 1.0 / poll_hz_value
        self._close_sdk = bool(close_sdk)
        try:
            settle_s = float(sdk_shutdown_settle_s)
        except (TypeError, ValueError, OverflowError):
            settle_s = 0.0
        # A non-finite settle interval must never turn close() into an
        # unbounded sleep.  Treat malformed values as no settle delay; the
        # bounded SDK close helper still protects the subsequent teardown.
        self._sdk_shutdown_settle_s = settle_s if np.isfinite(settle_s) and settle_s > 0.0 else 0.0
        try:
            self._sdk_close_timeout_s = float(sdk_close_timeout_s)
        except (TypeError, ValueError, OverflowError):
            self._sdk_close_timeout_s = DEFAULT_SDK_CLOSE_TIMEOUT_S
        if not np.isfinite(self._sdk_close_timeout_s) or self._sdk_close_timeout_s < 0.0:
            self._sdk_close_timeout_s = DEFAULT_SDK_CLOSE_TIMEOUT_S
        try:
            self._activation_trigger_threshold = float(activation_trigger_threshold)
            self._reset_grip_threshold = float(reset_grip_threshold)
            self._control_debounce_s = float(control_debounce_s)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("XRoboToolkit control thresholds/debounce must be numeric") from exc
        if (
            not np.isfinite(self._activation_trigger_threshold)
            or not 0.0 < self._activation_trigger_threshold <= 1.0
        ):
            raise ValueError("activation_trigger_threshold must be finite and in (0, 1]")
        if not np.isfinite(self._reset_grip_threshold) or not 0.0 < self._reset_grip_threshold <= 1.0:
            raise ValueError("reset_grip_threshold must be finite and in (0, 1]")
        if not np.isfinite(self._control_debounce_s) or self._control_debounce_s < 0.0:
            raise ValueError("control_debounce_s must be finite and non-negative")
        if any(
            value not in (None, "", "null")
            for value in (pause_button, pause_debounce_s, arms_button, arms_debounce_s)
        ):
            logger.warning(
                "XRoboToolkitInputProvider ignores legacy pause/arms constructor options; "
                "use Menu+trigger and dual-grip SIMPLE chords instead."
            )
        if not 0.0 < self._activation_trigger_threshold <= 1.0:
            raise ValueError("activation_trigger_threshold must be in (0, 1]")
        if not 0.0 < self._reset_grip_threshold <= 1.0:
            raise ValueError("reset_grip_threshold must be in (0, 1]")
        self._lock = threading.Lock()
        self._closed = False
        self._shutdown_complete = False
        self._frame_ready = threading.Event()
        self._frame_cache = RealtimeFrameCache[HumanFrame](buffer_size=buffer_size, fps_window=30)
        self._pending_control_events: deque[ControlEvent] = deque()
        self._last_body_timestamp_ns: int | None = None
        self._last_raw_body_joints: NDArray[np.float64] | None = None
        self._last_frame_timestamp: float | None = None
        self._last_activation_chord_pressed = False
        self._last_reset_chord_pressed = False
        self._last_activation_timestamp: float | None = None
        self._last_reset_timestamp: float | None = None
        self._ground_alignment_offset: float | None = None
        self._controller_snapshot: PicoControllerSnapshot | None = None
        self._head_pose_snapshot: PicoHeadPoseSnapshot | None = None
        # Keep malformed optional SDK values from flooding the 120 Hz log.
        # The first occurrence is a warning (so a disconnected/mismatched SDK
        # is diagnosable); subsequent occurrences are debug-only.
        self._invalid_auxiliary_fields: set[str] = set()

        init = getattr(self._sdk, "init", None)
        if not callable(init):
            raise RuntimeError("xrobotoolkit_sdk does not expose init()")
        # ``init`` starts native SDK workers synchronously enough to return a
        # usable binding, so do not move it to a daemon thread (a late init
        # completion could race the first polling calls).  If Python fails
        # while creating/starting our poll worker after init succeeds, clean
        # up the already-initialized native client before propagating the
        # startup error; otherwise a failed provider construction would leave
        # PXREA's service-check/feedback threads running indefinitely.
        # Record the start before entering the native initializer.  A failed
        # PXREAInit can still have spawned service-check/feedback workers, so
        # the exception path below must be able to apply the same bounded
        # deinitializer even when ``init`` does not return successfully.
        self._sdk_started_at = time.monotonic()
        try:
            init()
            self._sdk_started_at = time.monotonic()
            self._poll_thread = threading.Thread(target=self._poll_loop, name="xrobotoolkit_input", daemon=True)
            self._poll_thread.start()
        except BaseException:
            self._closed = True
            if self._close_sdk:
                remaining = self._sdk_shutdown_settle_s - (time.monotonic() - self._sdk_started_at)
                if remaining > 0.0:
                    time.sleep(remaining)
                close_sdk_bounded(
                    self._sdk,
                    timeout_s=self._sdk_close_timeout_s,
                    logger=logger,
                    context="XRoboToolkitInputProvider SDK (startup)",
                )
            raise
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

    @property
    def supports_mode_control_events(self) -> bool:
        """XRoboToolkit provides SIMPLE-style controller mode controls."""
        return True

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
        # ``close`` is intentionally idempotent.  Pipeline error paths and
        # launcher cleanup can both call it; invoking PXREADeinit twice is
        # undefined in the native binding and may block while joining a
        # feedback thread that has already been torn down.
        if getattr(self, "_shutdown_complete", False):
            return
        # Set the completion latch before doing any potentially blocking
        # joins so a second cleanup caller returns immediately rather than
        # invoking the native deinitializer concurrently.
        self._shutdown_complete = True
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
            close_sdk_bounded(
                self._sdk,
                timeout_s=self._sdk_close_timeout_s,
                logger=logger,
                context="XRoboToolkitInputProvider SDK",
            )
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
        available = getattr(self._sdk, "is_body_data_available", None)
        if not callable(available):
            return False
        try:
            available_value = available()
        except Exception as exc:
            self._warn_invalid_auxiliary("is_body_data_available", str(exc))
            return False
        if not isinstance(available_value, (bool, np.bool_)) or not bool(available_value):
            return False
        getter = getattr(self._sdk, "get_body_joints_pose", None)
        if not callable(getter):
            return False
        try:
            raw_joints = np.asarray(getter(), dtype=np.float64)
        except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
            logger.warning("XRoboToolkit body joint sample is malformed: %s", exc)
            return False
        if raw_joints.shape != (len(BODY_JOINT_NAMES), 7):
            logger.warning("Unexpected XRoboToolkit body joint shape: %s", raw_joints.shape)
            return False
        if not np.all(np.isfinite(raw_joints)):
            logger.warning("XRoboToolkit body joint sample contains non-finite values")
            return False
        # The SDK may keep returning cached controller/HMD values while the
        # headset is disconnected or body tracking is disabled.  Do not turn
        # those cached button states into mode events before a valid full-body
        # sample exists.  Poll after shape validation but before duplicate
        # suppression so a stationary body can still generate a fresh button
        # edge.
        source_ns = self._read_body_timestamp_ns()
        # A source timestamp is useful for suppressing repeated SDK cache
        # reads, but a few bindings expose only millisecond/coarsely sampled
        # timestamps.  Do not drop a genuinely changed skeleton merely
        # because that coarse timestamp has not advanced yet.  Treat a sample
        # as duplicate only when both the source timestamp (if available) and
        # the raw joint payload are unchanged.  If the timestamp changes while
        # the payload is stationary, retain the sample so controller edges can
        # still be detected at a fixed pose.
        raw_body_duplicate = (
            self._last_raw_body_joints is not None
            and np.array_equal(raw_joints, self._last_raw_body_joints)
        )
        body_duplicate = raw_body_duplicate and (
            source_ns is None or source_ns == self._last_body_timestamp_ns
        )
        # Auxiliary controller/HMD getters are a best-effort side channel.
        # A transient ``None``/NaN or SDK exception must not discard an
        # otherwise valid body sample, because doing so would make the whole
        # retargeting pipeline appear frozen.  ``_accept_auxiliary_state``
        # validates each field independently and only emits a control edge
        # when all values needed by that chord are valid.
        # The snapshots and the body packet are consumed independently by the
        # process-isolated runtime.  When this sample is new, ``append`` below
        # will advance the cache sequence by one; publish the auxiliary state
        # with that *next* sequence so a hand/HMD update cannot appear to lag
        # its corresponding body frame.  For a duplicate body sample no
        # append occurs, so retain the current sequence while still polling
        # button edges (a stationary operator must be able to activate/reset).
        with self._lock:
            auxiliary_seq = self._frame_cache.frame_seq + (0 if body_duplicate else 1)
        self._accept_auxiliary_state(timestamp, seq=auxiliary_seq, raw_body_joints=raw_joints)
        if body_duplicate:
            return False
        return self._accept_body_joints(raw_joints, timestamp=timestamp, source_ns=source_ns)

    def _read_body_timestamp_ns(self) -> int | None:
        getter = getattr(self._sdk, "get_body_timestamp_ns", None)
        if not callable(getter):
            return None
        try:
            raw_value = getter()
            if isinstance(raw_value, (bool, np.bool_)) or not isinstance(raw_value, numbers.Integral):
                return None
            value = int(raw_value)
        except (TypeError, ValueError, OverflowError):
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

    def _accept_auxiliary_state(
        self,
        timestamp: float,
        *,
        seq: int | None = None,
        raw_body_joints: NDArray[np.float64] | None = None,
    ) -> None:
        if seq is None:
            with self._lock:
                seq = self._frame_cache.frame_seq

        left_grip = self._read_aux_scalar("get_left_grip")
        right_grip = self._read_aux_scalar("get_right_grip")
        left_trigger = self._read_aux_scalar("get_left_trigger")
        right_trigger = self._read_aux_scalar("get_right_trigger")
        left_menu = self._read_aux_button("get_left_menu_button")

        # Neutralize invalid values in the public snapshot and mark the side
        # unavailable.  This is safer than retaining a stale pressed trigger,
        # while still allowing body tracking to continue through a temporary
        # controller packet glitch.
        left_values_valid = left_grip is not None and left_trigger is not None
        right_values_valid = right_grip is not None and right_trigger is not None
        snapshot = PicoControllerSnapshot(
            left=PicoControllerState(
                raw=left_values_valid,
                grip=0.0 if left_grip is None else left_grip,
                trigger=0.0 if left_trigger is None else left_trigger,
                present=left_values_valid,
            ),
            right=PicoControllerState(
                raw=right_values_valid,
                grip=0.0 if right_grip is None else right_grip,
                trigger=0.0 if right_trigger is None else right_trigger,
                present=right_values_valid,
            ),
            timestamp_s=timestamp,
            seq=seq,
        )
        head = self._read_aux_pose("get_headset_pose")
        hmd_rotation = _transform_pico_native_rotation(head[3:7] if head is not None else None)
        # OpenNeck and other auxiliary consumers need the torso orientation
        # from the *same* body frame as the HMD sample.  Do not substitute the
        # skeleton Head joint: Pico's independent HMD pose is intentionally
        # decoupled from body tracking.  XRoboToolkit and pico-bridge expose
        # the same 24-joint, [xyz, qx,qy,qz,qw] layout, so reuse the canonical
        # Spine3 index and coordinate transform here.
        spine3_rotation: NDArray[np.float64] | None = None
        if raw_body_joints is not None:
            try:
                body_array = np.asarray(raw_body_joints, dtype=np.float64)
            except (TypeError, ValueError):
                body_array = np.empty((0, 0), dtype=np.float64)
            spine3_index = BODY_JOINT_NAMES.index("Spine3")
            if body_array.shape == (len(BODY_JOINT_NAMES), 7):
                spine3_rotation = _transform_pico_native_rotation(
                    body_array[spine3_index, 3:7]
                )
        with self._lock:
            self._controller_snapshot = snapshot
            self._head_pose_snapshot = PicoHeadPoseSnapshot(
                hmd_rotation,
                spine3_rotation,
                timestamp,
                seq,
            )
        # SIMPLE's PicoStreamer uses a strict ``> 0.5`` comparison for both
        # activation and reset chords.  Keep the comparison strict at the
        # configured boundary: XR samples are quantized and a value of exactly
        # 0.5 is still a half pull, not a pressed trigger/grip.
        if left_menu is not None and right_trigger is not None:
            activation_pressed = left_menu and right_trigger > self._activation_trigger_threshold
            self._poll_chord(
                pressed=activation_pressed,
                event_type=ControlEventType.ENTER_MOCAP,
                source="xrobotoolkit:left_menu+right_trigger",
                last_pressed_attr="_last_activation_chord_pressed",
                last_timestamp_attr="_last_activation_timestamp",
                timestamp=timestamp,
            )
        if left_grip is not None and right_grip is not None:
            reset_pressed = (
                left_grip > self._reset_grip_threshold
                and right_grip > self._reset_grip_threshold
            )
            self._poll_chord(
                pressed=reset_pressed,
                event_type=ControlEventType.ENTER_STANDING,
                source="xrobotoolkit:left_grip+right_grip",
                last_pressed_attr="_last_reset_chord_pressed",
                last_timestamp_attr="_last_reset_timestamp",
                timestamp=timestamp,
            )

    def _warn_invalid_auxiliary(self, name: str, detail: str) -> None:
        if name not in self._invalid_auxiliary_fields:
            self._invalid_auxiliary_fields.add(name)
            logger.warning("Invalid XRoboToolkit auxiliary field %s: %s", name, detail)
        else:
            logger.debug("Invalid XRoboToolkit auxiliary field %s: %s", name, detail)

    def _read_aux_scalar(self, name: str) -> float | None:
        """Read an optional controller scalar, returning ``None`` if invalid.

        XRoboToolkit exposes trigger/grip values in ``[0, 1]``.  Rejecting
        booleans, arrays, NaN/Inf, and out-of-range values prevents malformed
        SDK data from generating a false mode/reset edge.  Missing getters are
        represented as ``None`` (and therefore a neutral public value with
        ``present=False``) for compatibility with older bindings.
        """

        getter = getattr(self._sdk, name, None)
        if not callable(getter):
            # Missing optional getters mean that the controller side is not
            # observable.  Return ``None`` so the public snapshot advertises
            # ``present=False`` and, crucially, no activation/reset chord can
            # be synthesized from an assumed neutral value.
            return None
        try:
            raw = getter()
            if isinstance(raw, (bool, np.bool_)):
                raise ValueError("boolean is not a scalar trigger/grip value")
            array = np.asarray(raw)
            if array.size != 1:
                raise ValueError(f"expected one value, got shape {array.shape}")
            item = array.reshape(-1)[0]
            if not isinstance(item, numbers.Real):
                raise ValueError(f"expected a real number, got {type(item).__name__}")
            scalar = float(item)
        except Exception as exc:
            self._warn_invalid_auxiliary(name, str(exc))
            return None
        if not np.isfinite(scalar) or scalar < 0.0 or scalar > 1.0:
            self._warn_invalid_auxiliary(name, f"value {scalar!r} is outside [0, 1]")
            return None
        return scalar

    def _read_aux_button(self, name: str) -> bool | None:
        """Read an optional button without coercing arbitrary truthy values."""

        getter = getattr(self._sdk, name, None)
        if not callable(getter):
            return None
        try:
            raw = getter()
        except Exception as exc:
            self._warn_invalid_auxiliary(name, str(exc))
            return None
        if not isinstance(raw, (bool, np.bool_)):
            self._warn_invalid_auxiliary(name, f"expected boolean, got {type(raw).__name__}")
            return None
        return bool(raw)

    def _read_aux_pose(self, name: str) -> NDArray[np.float64] | None:
        """Read a finite seven-value ``[xyz, xyzw]`` HMD pose."""

        getter = getattr(self._sdk, name, None)
        if not callable(getter):
            return None
        try:
            raw = np.asarray(getter()).reshape(-1)
            if raw.shape != (7,):
                raise ValueError(f"expected seven finite values, got shape {raw.shape}")
            if any(
                isinstance(item, (bool, np.bool_)) or not isinstance(item, numbers.Real)
                for item in raw
            ):
                raise ValueError("pose contains a non-numeric value")
            pose = np.asarray(raw, dtype=np.float64)
        except Exception as exc:
            self._warn_invalid_auxiliary(name, str(exc))
            return None
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            self._warn_invalid_auxiliary(name, f"expected seven finite values, got shape {pose.shape}")
            return None
        return pose

    def _poll_chord(
        self,
        *,
        pressed: bool,
        event_type: ControlEventType,
        source: str,
        last_pressed_attr: str,
        last_timestamp_attr: str,
        timestamp: float,
    ) -> None:
        last_pressed = bool(getattr(self, last_pressed_attr))
        last_event_timestamp = getattr(self, last_timestamp_attr)
        if (
            pressed
            and not last_pressed
            and (
                last_event_timestamp is None
                or timestamp - float(last_event_timestamp) >= self._control_debounce_s
            )
        ):
            with self._lock:
                self._pending_control_events.append(ControlEvent(event_type, source, timestamp))
            setattr(self, last_timestamp_attr, timestamp)
        setattr(self, last_pressed_attr, pressed)

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
