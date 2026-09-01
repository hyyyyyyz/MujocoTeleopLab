#!/usr/bin/env python3
"""Publish XRoboToolkit controller samples for the scene WBC process.

Run this script with the project's Python 3.12 ``.venv``.  It needs only the
XRoboToolkit PC Service and paired Head/Controller devices; Full-body and Send
are not prerequisites for the table-top scene runtime.
"""

from __future__ import annotations

import argparse
import math
import numbers
import signal
import socket
import sys
import time
import uuid
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teleopit.scenes.xr_packet import (
    DEFAULT_XR_BRIDGE_HOST,
    DEFAULT_XR_BRIDGE_PORT,
    SceneXRPacket,
)
from teleopit.inputs.xrobotoolkit_utils import (
    DEFAULT_SDK_CLOSE_TIMEOUT_S,
    close_sdk_bounded,
)


# The native binding normally advances ``timeStampNs`` for every incoming
# XR sample.  A few service/firmware combinations only update it when a pose
# changes (or quantise it to a coarse interval), though.  In that case an
# unconditional duplicate-suppression gate would stop sending while the
# operator is still connected and the scene receiver would eventually mark
# the input stale.  Conversely, forwarding a cached SDK value forever would
# make a disconnected headset look alive.  The gate below permits a short,
# bounded heartbeat after the last source change and then deliberately lets
# the receiver time out until a new source value arrives.
DEFAULT_SOURCE_HEARTBEAT_INTERVAL_S = 0.10
DEFAULT_SOURCE_STALE_TIMEOUT_S = 1.00

# These accessors are the stable surface used by the scene bridge.  Timestamp
# accessors are intentionally *not* listed here: released SDK builds before
# the top-level ``get_time_stamp_ns`` addition remain usable through the
# best-effort stream-specific fallback in :func:`_source_timestamp_ns`.
_REQUIRED_SDK_GETTERS = (
    "get_left_controller_pose",
    "get_right_controller_pose",
    "get_headset_pose",
    "get_left_axis",
    "get_right_axis",
    "get_left_trigger",
    "get_right_trigger",
    "get_left_grip",
    "get_right_grip",
    "get_A_button",
    "get_B_button",
    "get_X_button",
    "get_Y_button",
    "get_left_menu_button",
)


class _SourceFreshnessGate:
    """Decide when a validated SDK sample should cross the UDP boundary.

    ``timestamp_ns`` and ``signature`` are source metadata/state, not the
    bridge sequence number.  A changed timestamp *or* payload is a fresh
    source sample.  An unchanged sample may be emitted at a bounded heartbeat
    cadence while it is within ``stale_timeout_s`` of the last source change.
    Once that grace period expires, no more duplicates are emitted; this is
    what lets ``SceneXRReceiver`` detect a disconnected/cached SDK stream.

    State is committed only after the caller successfully sends the packet.
    A transient ``sendto`` failure therefore does not consume the heartbeat
    or suppress a retry.
    """

    def __init__(
        self,
        *,
        heartbeat_interval_s: float = DEFAULT_SOURCE_HEARTBEAT_INTERVAL_S,
        stale_timeout_s: float = DEFAULT_SOURCE_STALE_TIMEOUT_S,
    ) -> None:
        try:
            heartbeat = float(heartbeat_interval_s)
            stale_timeout = float(stale_timeout_s)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("source heartbeat/stale timeouts must be finite numbers") from exc
        if not math.isfinite(heartbeat) or heartbeat <= 0.0:
            raise ValueError("source heartbeat interval must be finite and positive")
        if not math.isfinite(stale_timeout) or stale_timeout < heartbeat:
            raise ValueError(
                "source stale timeout must be finite and at least the heartbeat interval"
            )
        self.heartbeat_interval_s = heartbeat
        self.stale_timeout_s = stale_timeout
        self._last_timestamp_ns: int | None = None
        self._last_signature: tuple[object, ...] | None = None
        self._last_source_change_s: float | None = None
        self._last_emit_s: float | None = None

    def should_emit(
        self,
        *,
        now_s: float,
        timestamp_ns: int | None,
        signature: tuple[object, ...],
    ) -> bool:
        """Return whether this sample is fresh or due for a heartbeat.

        ``now_s`` is supplied by the bridge loop so tests and embedding
        callers can use a deterministic monotonic clock.  The method does not
        mutate state; call :meth:`commit` only after ``sendto`` succeeds.
        """

        now = float(now_s)
        if not math.isfinite(now):
            raise ValueError("source gate now_s must be finite")
        source_changed = (
            self._last_emit_s is None
            or timestamp_ns != self._last_timestamp_ns
            or signature != self._last_signature
        )
        if source_changed:
            return True
        assert self._last_source_change_s is not None
        assert self._last_emit_s is not None
        source_age = max(0.0, now - self._last_source_change_s)
        if source_age > self.stale_timeout_s:
            return False
        return max(0.0, now - self._last_emit_s) >= self.heartbeat_interval_s

    def commit(
        self,
        *,
        now_s: float,
        timestamp_ns: int | None,
        signature: tuple[object, ...],
    ) -> None:
        """Commit a successfully emitted source sample."""

        now = float(now_s)
        if not math.isfinite(now):
            raise ValueError("source gate now_s must be finite")
        source_changed = (
            self._last_emit_s is None
            or timestamp_ns != self._last_timestamp_ns
            or signature != self._last_signature
        )
        if source_changed:
            self._last_source_change_s = now
        self._last_timestamp_ns = timestamp_ns
        self._last_signature = signature
        self._last_emit_s = now


def _port(value: str) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("must be in [1, 65535]")
    return port


def _positive_finite(value: str) -> float:
    try:
        hz = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a finite number") from exc
    if not math.isfinite(hz) or hz <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return hz


def _nonnegative_finite(value: str) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a finite number") from exc
    if not math.isfinite(seconds) or seconds < 0.0:
        raise argparse.ArgumentTypeError("must be a finite number greater than or equal to zero")
    return seconds


def _destination_host(value: str) -> str:
    host = str(value).strip()
    if not host:
        raise argparse.ArgumentTypeError("must not be empty")
    return host


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="XRoboToolkit controller bridge for scene teleoperation")
    parser.add_argument("--host", type=_destination_host, default=DEFAULT_XR_BRIDGE_HOST)
    parser.add_argument("--port", type=_port, default=DEFAULT_XR_BRIDGE_PORT)
    parser.add_argument("--hz", type=_positive_finite, default=60.0)
    parser.add_argument(
        "--sdk-close-timeout",
        type=_nonnegative_finite,
        default=DEFAULT_SDK_CLOSE_TIMEOUT_S,
        help="maximum seconds to wait for XRoboToolkit close() during shutdown",
    )
    parser.add_argument(
        "--source-heartbeat-interval",
        type=_positive_finite,
        default=DEFAULT_SOURCE_HEARTBEAT_INTERVAL_S,
        help=(
            "seconds between bounded heartbeats when the SDK timestamp/payload "
            "is unchanged (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--source-stale-timeout",
        type=_positive_finite,
        default=DEFAULT_SOURCE_STALE_TIMEOUT_S,
        help=(
            "maximum seconds to forward an unchanged SDK sample before allowing "
            "the scene input timeout (default: %(default)s)"
        ),
    )
    return parser


def _value(sdk: object, name: str, fallback: object) -> object:
    getter = getattr(sdk, name, None)
    return getter() if callable(getter) else fallback


def _validate_sdk_api(sdk: object) -> None:
    """Fail clearly when the installed binding cannot drive scene controls.

    The bridge used to substitute identity poses/zero buttons for every
    missing accessor.  That made a mismatched or partially installed binding
    look connected while silently disabling teleoperation.  Validate the
    control surface before entering the polling loop.  Timestamp accessors are
    deliberately optional for compatibility with older binding revisions; they
    only affect stale-sample gating.
    """

    def has_callable(name: str) -> bool:
        try:
            return callable(getattr(sdk, name, None))
        except Exception:
            # A proxy/extension object may raise while resolving an optional
            # symbol.  Treat that exactly like a missing accessor and report
            # the complete API mismatch below instead of leaking an opaque
            # descriptor exception.
            return False

    missing = [name for name in _REQUIRED_SDK_GETTERS if not has_callable(name)]
    if not has_callable("init"):
        missing.insert(0, "init")
    if not has_callable("close"):
        missing.append("close")
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            "XRoboToolkit SDK is missing required scene accessor(s): "
            f"{names}. Install the supported PC Service/Python binding with "
            "scripts/setup/setup_xrobotoolkit.sh. Timestamp accessors are optional "
            "for older bindings."
        )


def _source_timestamp_ns(sdk: object) -> tuple[bool, int | None]:
    """Read the SDK's frame timestamp for stale-sample gating.

    ``get_time_stamp_ns`` is present in the released XRoboToolkit binding and
    advances when a new top-level controller/HMD sample arrives.  Treat that
    accessor as authoritative whenever it exists: a zero/invalid value means
    that the top-level stream has not produced a sample yet, and must not be
    replaced by an unrelated body/motion timestamp.  A body tracker can keep
    publishing while the controllers/headset are disconnected; using its
    timestamp as a fallback in that case would keep stale arm commands alive.
    A few older bindings expose only the more specific
    ``get_motion_timestamp_ns``/``get_body_timestamp_ns`` accessors, so use
    those only when the generic accessor is absent altogether.  A bridge
    process can stay alive while the headset/PC service is disconnected,
    however, and the SDK then continues returning the last controller pose.
    Sending that cached pose at 60 Hz would keep the scene receiver's
    *arrival* freshness timer alive and let an old arm command survive a
    disconnect.  The caller combines this source timestamp with a
    validated-payload signature: an unchanged timestamp *and* unchanged
    controls are treated as a cache replay, while a changed trigger/joystick
    is still forwarded even on a coarse timestamp.  Returning ``False`` keeps
    compatibility with bindings that lack every timestamp getter; those
    bindings retain the historical best-effort polling behavior.
    """

    def read(getter: object) -> int | None:
        if not callable(getter):
            return None
        try:
            raw_value = getter()
            if isinstance(raw_value, bool) or not isinstance(raw_value, numbers.Integral):
                return None
            value = int(raw_value)
        except Exception:
            return None
        return value if value > 0 else None

    # ``get_time_stamp_ns`` is the timestamp attached to the top-level XR
    # state packet (controller + HMD).  Prefer it whenever the binding exposes
    # a positive value: the body/motion timestamps belong to independent
    # streams, and taking their maximum can make a disconnected controller
    # look alive merely because body tracking keeps publishing cached poses.
    generic_getter = getattr(sdk, "get_time_stamp_ns", None)
    generic_available = callable(generic_getter)
    generic_timestamp = read(generic_getter)
    if generic_available:
        # Do not fall back when the authoritative top-level getter exists but
        # has not produced a positive value yet.  Body/motion timestamps are
        # independent streams and can otherwise make a disconnected headset
        # appear alive.
        return True, generic_timestamp

    # Older bindings may omit the generic accessor entirely.  Retain
    # compatibility by falling back to the stream-specific accessors only in
    # that case; they are never combined with a valid top-level timestamp.
    specific_available = False
    specific_values: list[int] = []
    for name in ("get_motion_timestamp_ns", "get_body_timestamp_ns"):
        getter = getattr(sdk, name, None)
        if not callable(getter):
            continue
        specific_available = True
        value = read(getter)
        if value is not None:
            specific_values.append(value)
    return generic_available or specific_available, (
        max(specific_values) if specific_values else None
    )


def _sample_signature(packet: SceneXRPacket) -> tuple[object, ...]:
    """Return the source fields used to detect a cached SDK sample.

    XRoboToolkit bindings do not all advance their timestamp at controller
    rate.  In particular, a coarse millisecond timestamp can remain constant
    while a trigger/joystick changes.  The bridge must therefore consider the
    complete validated payload as well as the timestamp when deciding whether
    a read is merely a disconnected-service cache replay.  ``sequence``, the
    bridge generation, and the local arrival timestamp are deliberately
    excluded: those are transport metadata rather than source state.
    """

    return (
        packet.left_pose,
        packet.right_pose,
        packet.head_pose,
        packet.left_axis,
        packet.right_axis,
        packet.left_trigger,
        packet.right_trigger,
        packet.left_grip,
        packet.right_grip,
        packet.a,
        packet.b,
        packet.x,
        packet.y,
        packet.left_menu,
    )


def main() -> int:
    args = _parser().parse_args()
    if args.source_stale_timeout < args.source_heartbeat_interval:
        raise ValueError(
            "--source-stale-timeout must be at least --source-heartbeat-interval"
        )
    try:
        import xrobotoolkit_sdk as xrt
    except ImportError as exc:
        raise RuntimeError("Install XRoboToolkit with scripts/setup/setup_xrobotoolkit.sh") from exc

    # Check the lifecycle entry point before starting native workers.  The
    # complete control-surface validation is performed below, after the
    # socket/signal guards are installed, so even a deliberately tiny test
    # double (or a partial SDK that fails during init) still exercises the
    # cleanup paths deterministically.
    if not callable(getattr(xrt, "init", None)):
        raise RuntimeError("XRoboToolkit SDK does not expose init(); reinstall the supported binding")

    # PXREAInit may start native workers before reporting an error.  Keep the
    # cleanup path active even when initialization itself fails; the bounded
    # helper is safe for both a fully and partially initialized binding.
    try:
        xrt.init()
    except BaseException:
        close_sdk_bounded(
            xrt,
            timeout_s=args.sdk_close_timeout,
            context="XRoboToolkit scene bridge SDK (init failure)",
        )
        raise
    destination = (str(args.host), int(args.port))
    # ``PXREAInit`` starts native background workers before returning.  If
    # creating the UDP socket fails (for example because the process hits its
    # file-descriptor limit), do not leave those workers running: the normal
    # loop ``finally`` below has not been entered yet, so explicitly perform
    # the same bounded SDK teardown on this startup-error path.
    try:
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except BaseException:
        close_sdk_bounded(
            xrt,
            timeout_s=args.sdk_close_timeout,
            context="XRoboToolkit scene bridge SDK (startup)",
        )
        raise
    try:
        running = True

        def stop(_: int, __: object) -> None:
            nonlocal running
            running = False

        # Signal registration is only legal in the main interpreter thread;
        # keeping it inside this guarded setup block prevents an embedding
        # caller (or a unit test) from leaking the socket/SDK when that
        # contract is violated.
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        # A missing controller accessor is a deterministic installation/API
        # mismatch, not a transient headset disconnect.  Validate it before
        # entering the polling loop while the guarded setup block can still
        # close both the UDP socket and any native SDK workers on failure.
        _validate_sdk_api(xrt)
        # ``sequence`` is intentionally process-local and starts at zero for a
        # clean restart.  Pairing it with a random generation ID lets the scene
        # receiver accept the restarted stream immediately instead of treating its
        # lower sequence numbers as stale samples.  The ID is generated once per
        # bridge invocation and remains stable for all packets in that invocation.
        session_id = uuid.uuid4().hex
        sequence = 0
        source_gate = _SourceFreshnessGate(
            heartbeat_interval_s=args.source_heartbeat_interval,
            stale_timeout_s=args.source_stale_timeout,
        )
        source_gate_available = False
        next_stale_status_s = 0.0
        period_s = 1.0 / float(args.hz)
        print(
            "XR scene bridge ready -> "
            f"udp://{destination[0]}:{destination[1]} | Head + Controller are sufficient; Full-body is optional."
        )
    except BaseException:
        try:
            sender.close()
        finally:
            close_sdk_bounded(
                xrt,
                timeout_s=args.sdk_close_timeout,
                context="XRoboToolkit scene bridge SDK (startup)",
            )
        raise
    try:
        while running:
            started = time.monotonic()
            try:
                source_gate_available, source_timestamp_ns = _source_timestamp_ns(xrt)
                if source_gate_available:
                    if source_timestamp_ns is None:
                        # The SDK has a timestamp API but no valid sample yet
                        # (normally startup or a lost RoboticsService link).
                        # Do not emit fallback poses that would look fresh to
                        # the isolated scene process.
                        if started >= next_stale_status_s:
                            print(
                                "XR scene bridge waiting for a fresh XRoboToolkit sample.",
                                file=sys.stderr,
                            )
                            next_stale_status_s = started + 5.0
                        remaining = period_s - (time.monotonic() - started)
                        if remaining > 0:
                            time.sleep(remaining)
                        continue
                packet = SceneXRPacket.from_mapping(
                    {
                        "sequence": sequence,
                        "session_id": session_id,
                        "timestamp_s": started,
                        "left_pose": _value(xrt, "get_left_controller_pose", [0, 0, 0, 0, 0, 0, 1]),
                        "right_pose": _value(xrt, "get_right_controller_pose", [0, 0, 0, 0, 0, 0, 1]),
                        "head_pose": _value(xrt, "get_headset_pose", [0, 0, 0, 0, 0, 0, 1]),
                        "left_axis": _value(xrt, "get_left_axis", [0, 0]),
                        "right_axis": _value(xrt, "get_right_axis", [0, 0]),
                        "left_trigger": _value(xrt, "get_left_trigger", 0.0),
                        "right_trigger": _value(xrt, "get_right_trigger", 0.0),
                        "left_grip": _value(xrt, "get_left_grip", 0.0),
                        "right_grip": _value(xrt, "get_right_grip", 0.0),
                        "a": _value(xrt, "get_A_button", False),
                        "b": _value(xrt, "get_B_button", False),
                        "x": _value(xrt, "get_X_button", False),
                        "y": _value(xrt, "get_Y_button", False),
                        "left_menu": _value(xrt, "get_left_menu_button", False),
                    }
                )
                sample_signature = _sample_signature(packet)
                # Controller getters are cache reads.  A duplicate source
                # sample receives a short, bounded heartbeat so coarse/stuck
                # timestamps do not make a stationary connected operator stale;
                # after the grace interval, suppression resumes and the scene
                # receiver can detect a disconnected/cached stream.
                if source_gate_available and not source_gate.should_emit(
                    now_s=started,
                    timestamp_ns=source_timestamp_ns,
                    signature=sample_signature,
                ):
                    remaining = period_s - (time.monotonic() - started)
                    if remaining > 0:
                        time.sleep(remaining)
                    continue
                sender.sendto(packet.to_wire(), destination)
                # Commit the source watermark only after the complete sample
                # has been validated and handed to the socket.  If a getter
                # returns a transient malformed value (or sendto fails), the
                # same source frame should be retried rather than being
                # permanently suppressed by a prematurely advanced watermark.
                if source_gate_available:
                    source_gate.commit(
                        now_s=started,
                        timestamp_ns=source_timestamp_ns,
                        signature=sample_signature,
                    )
                sequence += 1
            except Exception as exc:
                print(f"XR scene bridge sample failed: {exc}", file=sys.stderr)
            remaining = period_s - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        try:
            sender.close()
        finally:
            close_sdk_bounded(
                xrt,
                timeout_s=args.sdk_close_timeout,
                context="XRoboToolkit scene bridge SDK",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
