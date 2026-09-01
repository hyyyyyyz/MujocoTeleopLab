"""Small localhost protocol between XRoboToolkit and the scene runtime.

The XRoboToolkit SDK binary is presently built for the project's Python 3.12
environment, while the upstream decoupled-WBC stack uses Python 3.10.  Keeping
the SDK in a tiny bridge process avoids loading incompatible C++ runtimes into
one process.  Only controller/HMD data crosses this localhost boundary; no
body-tracking frame is required for scene teleoperation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import numbers
import operator
import socket
import time
from typing import Any

import numpy as np


DEFAULT_XR_BRIDGE_HOST = "127.0.0.1"
DEFAULT_XR_BRIDGE_PORT = 17600
_MAX_PACKET_BYTES = 16 * 1024
_MAX_SESSION_ID_BYTES = 128
_MAX_SEQUENCE = 2**63 - 1
_LEGACY_SESSION_ID = "legacy"


def _port(value: Any) -> int:
    """Validate a UDP port without silently coercing malformed values.

    ``int(value)`` accepts floats (and booleans) and silently truncates them;
    that is especially surprising at this process boundary because a typo
    such as ``17600.5`` would make the bridge and receiver bind different
    endpoints.  Configuration/CLI layers normally provide an ``int`` already,
    while ``operator.index`` also accepts NumPy integer scalars.
    """

    if isinstance(value, (bool, np.bool_)):
        raise ValueError("XR bridge port must be an integer in [1, 65535]")
    try:
        port = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("XR bridge port must be an integer in [1, 65535]") from exc
    if not 0 < port <= 65535:
        raise ValueError("XR bridge port must be in [1, 65535]")
    return port


def _pose(value: Any, name: str) -> tuple[float, ...]:
    try:
        raw = np.asarray(value).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be seven finite numeric values [x,y,z,qx,qy,qz,qw]") from exc
    if any(isinstance(item, (bool, np.bool_)) or not isinstance(item, numbers.Real) for item in raw):
        raise ValueError(f"{name} must be seven finite numeric values [x,y,z,qx,qy,qz,qw]")
    array = np.asarray(raw, dtype=np.float64)
    if array.shape != (7,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be seven finite values [x,y,z,qx,qy,qz,qw]")
    return tuple(float(item) for item in array)


def _axis(value: Any, name: str) -> tuple[float, float]:
    try:
        raw = np.asarray(value).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be two finite numeric values [x,y]") from exc
    if any(isinstance(item, (bool, np.bool_)) or not isinstance(item, numbers.Real) for item in raw):
        raise ValueError(f"{name} must be two finite numeric values [x,y]")
    array = np.asarray(raw, dtype=np.float64)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be two finite values [x,y]")
    if np.any(array < -1.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must be within [-1, 1]")
    return float(array[0]), float(array[1])


def _unit_interval(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite numeric scalar")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if not 0.0 <= scalar <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return scalar


def _finite_scalar(value: Any, name: str) -> float:
    """Parse a JSON numeric scalar without coercing strings or booleans."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite numeric scalar")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _boolean(value: Any, name: str) -> bool:
    """Validate a wire boolean without coercing arbitrary truthy values.

    ``bool(value)`` is deliberately not used at this process boundary: JSON
    senders occasionally encode a broken button field as ``"false"`` or
    ``1``, both of which would otherwise become ``True`` and could trigger a
    mode/reset edge.  ``json.loads`` produces the builtin :class:`bool` for a
    valid JSON boolean, so accepting that exact type is sufficient and keeps
    the protocol unambiguous.
    """

    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _session_id(value: Any, name: str = "session_id") -> str:
    """Validate the bridge generation identifier.

    A bridge sequence starts at zero for every process invocation.  The
    generation ID is therefore part of the protocol rather than an optional
    diagnostic field: receivers use it to distinguish a restarted bridge
    from an old, lower sequence number.  Keep the wire representation a short
    UTF-8 string so malformed datagrams cannot allocate unbounded memory.
    """

    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must contain valid UTF-8 characters") from exc
    if len(encoded) > _MAX_SESSION_ID_BYTES:
        raise ValueError(f"{name} is too long")
    return value


def _sequence(value: Any) -> int:
    """Validate a JSON integer sequence without silently truncating floats."""

    # ``int(1.5)`` would otherwise turn a malformed sequence into a duplicate
    # valid sample.  bool is an int subclass, so reject it explicitly too.
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("sequence must be a non-negative integer")
    sequence = int(value)
    if sequence < 0 or sequence > _MAX_SEQUENCE:
        raise ValueError("sequence must be a non-negative integer")
    return sequence


@dataclass(frozen=True)
class SceneXRPacket:
    """A validated XRoboToolkit controller/HMD sample."""

    sequence: int
    timestamp_s: float
    left_pose: tuple[float, ...]
    right_pose: tuple[float, ...]
    head_pose: tuple[float, ...]
    left_axis: tuple[float, float]
    right_axis: tuple[float, float]
    left_trigger: float
    right_trigger: float
    left_grip: float
    right_grip: float
    a: bool
    b: bool
    x: bool
    y: bool
    left_menu: bool
    # A fresh UUID is generated by the XRoboToolkit bridge for each process
    # invocation.  The default keeps direct construction and pre-session test
    # fixtures compatible with the original protocol; new bridge packets
    # always carry an explicit non-legacy ID.
    session_id: str = _LEGACY_SESSION_ID

    def to_wire(self) -> bytes:
        # Dataclass construction is intentionally lightweight for callers,
        # but packets can also be created directly (rather than through
        # ``from_mapping``).  Re-validate at serialization so an invalid
        # direct instance can never cross the process boundary or make
        # ``json.dumps`` emit a subtly coerced value.
        validated = type(self).from_mapping(
            {
                "sequence": self.sequence,
                "session_id": self.session_id,
                "timestamp_s": self.timestamp_s,
                "left_pose": self.left_pose,
                "right_pose": self.right_pose,
                "head_pose": self.head_pose,
                "left_axis": self.left_axis,
                "right_axis": self.right_axis,
                "left_trigger": self.left_trigger,
                "right_trigger": self.right_trigger,
                "left_grip": self.left_grip,
                "right_grip": self.right_grip,
                "a": self.a,
                "b": self.b,
                "x": self.x,
                "y": self.y,
                "left_menu": self.left_menu,
            }
        )
        payload = {
            "sequence": validated.sequence,
            "session_id": validated.session_id,
            "timestamp_s": validated.timestamp_s,
            "left_pose": validated.left_pose,
            "right_pose": validated.right_pose,
            "head_pose": validated.head_pose,
            "left_axis": validated.left_axis,
            "right_axis": validated.right_axis,
            "left_trigger": validated.left_trigger,
            "right_trigger": validated.right_trigger,
            "left_grip": validated.left_grip,
            "right_grip": validated.right_grip,
            "a": validated.a,
            "b": validated.b,
            "x": validated.x,
            "y": validated.y,
            "left_menu": validated.left_menu,
        }
        return json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "SceneXRPacket":
        if not isinstance(value, dict):
            raise ValueError("scene XR packet must be a mapping")
        try:
            sequence = _sequence(value["sequence"])
            # Packets produced before session IDs were introduced are accepted
            # as one legacy generation.  The XR bridge itself always sends a
            # UUID.
            session_id = _session_id(value.get("session_id", _LEGACY_SESSION_ID))
            timestamp_s = _finite_scalar(value["timestamp_s"], "timestamp_s")
            return cls(
                sequence=sequence,
                session_id=session_id,
                timestamp_s=timestamp_s,
                left_pose=_pose(value["left_pose"], "left_pose"),
                right_pose=_pose(value["right_pose"], "right_pose"),
                head_pose=_pose(value["head_pose"], "head_pose"),
                left_axis=_axis(value["left_axis"], "left_axis"),
                right_axis=_axis(value["right_axis"], "right_axis"),
                left_trigger=_unit_interval(value["left_trigger"], "left_trigger"),
                right_trigger=_unit_interval(value["right_trigger"], "right_trigger"),
                left_grip=_unit_interval(value["left_grip"], "left_grip"),
                right_grip=_unit_interval(value["right_grip"], "right_grip"),
                a=_boolean(value["a"], "a"),
                b=_boolean(value["b"], "b"),
                x=_boolean(value["x"], "x"),
                y=_boolean(value["y"], "y"),
                left_menu=_boolean(value["left_menu"], "left_menu"),
            )
        except KeyError as exc:
            raise ValueError(f"scene XR packet is missing field {exc.args[0]!r}") from exc

    @classmethod
    def from_wire(cls, payload: bytes) -> "SceneXRPacket":
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise ValueError("scene XR packet must be bytes")
        payload = bytes(payload)
        if not payload or len(payload) > _MAX_PACKET_BYTES:
            raise ValueError("invalid scene XR packet size")
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("invalid scene XR packet JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("scene XR packet must be a JSON object")
        return cls.from_mapping(value)


class SceneXRReceiver:
    """Non-blocking UDP receiver holding the most recent bridge packet."""

    def __init__(
        self,
        host: str = DEFAULT_XR_BRIDGE_HOST,
        port: int = DEFAULT_XR_BRIDGE_PORT,
    ) -> None:
        port = _port(port)
        # ``socket.bind`` accepts an empty hostname on some platforms (and
        # resolves it differently across libc implementations).  Reject it
        # before allocating a descriptor so a malformed launcher/config value
        # fails deterministically and cannot accidentally bind all interfaces.
        if not isinstance(host, str) or not host.strip():
            raise ValueError("XR bridge host must be a non-empty string")
        host = host.strip()
        receiver_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Do not set SO_REUSEADDR here.  On Linux, two UDP sockets that
            # both enable it can bind the same endpoint and the kernel then
            # delivers each datagram to an arbitrary process.  The scene
            # launcher already has a flock guard, but the receiver should be
            # safe when instantiated directly as well.
            receiver_socket.bind((host, port))
            receiver_socket.setblocking(False)
        except BaseException:
            # A failed bind (for example, because another scene instance owns
            # the endpoint) must release the descriptor immediately.  The
            # object has not finished construction, so ``close`` cannot be
            # relied on by the caller or by Python's eventual finalizer.
            try:
                receiver_socket.close()
            finally:
                raise
        self._socket = receiver_socket
        self._packet: SceneXRPacket | None = None
        self._session_id: str | None = None
        # Once a generation transition is observed, never accept a delayed
        # datagram from an older generation.  This matters when a bridge is
        # restarted while the previous process still has packets queued in the
        # kernel: simply comparing sequence numbers would let the old stream
        # take the receiver back to a stale pose.
        # Keep every generation seen during this receiver lifetime.  A
        # bounded LRU would eventually allow a very old delayed datagram to
        # become "new" again after enough bridge restarts and roll the scene
        # back to stale controller state.  Session IDs are short UUIDs, so
        # retaining this set is inexpensive compared with the safety benefit.
        self._retired_sessions: set[str] = set()
        self._received_at_s: float | None = None
        self._packet_count = 0
        self._closed = False

    def poll(self) -> SceneXRPacket | None:
        """Drain queued datagrams and return the newest valid sample."""
        if self._closed:
            return self._packet
        while True:
            try:
                payload, _ = self._socket.recvfrom(_MAX_PACKET_BYTES + 1)
            except BlockingIOError:
                return self._packet
            except OSError:
                # ``close()`` may race a final poll from the scene loop.  A
                # closed receiver is a normal terminal state; do not turn it
                # into an exception during shutdown.  Surface unrelated
                # socket errors while the receiver is live for diagnostics.
                if self._closed:
                    return self._packet
                raise
            try:
                packet = SceneXRPacket.from_wire(payload)
            except (
                UnicodeDecodeError,
                ValueError,
                TypeError,
                AttributeError,
                KeyError,
                OverflowError,
                RecursionError,
            ):
                # UDP is an untrusted process boundary.  A truncated JSON
                # object or a packet with missing/wrongly typed fields must
                # be ignored rather than terminating the 200 Hz scene loop.
                continue
            if self._session_id is None:
                self._session_id = packet.session_id
            elif packet.session_id == self._session_id:
                if self._packet is not None and packet.sequence <= self._packet.sequence:
                    continue
            elif packet.session_id in self._retired_sessions:
                # A delayed packet from a bridge generation that has already
                # been superseded must not switch the active stream back.
                continue
            else:
                if self._session_id is not None:
                    self._retired_sessions.add(self._session_id)
                self._session_id = packet.session_id
            self._packet = packet
            self._received_at_s = time.monotonic()
            self._packet_count += 1

    @property
    def latest(self) -> SceneXRPacket | None:
        return self._packet

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has been called on this receiver."""

        return self._closed

    @property
    def session_id(self) -> str | None:
        """Current bridge generation, or ``None`` before the first packet."""
        return self._session_id

    @property
    def packet_count(self) -> int:
        """Number of accepted bridge samples since this receiver was created."""
        return self._packet_count

    def age_s(self) -> float | None:
        """Age of the latest accepted packet, or ``None`` before the first one."""
        if self._received_at_s is None:
            return None
        return max(0.0, time.monotonic() - self._received_at_s)

    def is_fresh(self, max_age_s: float) -> bool:
        if isinstance(max_age_s, (bool, np.bool_)) or not isinstance(max_age_s, numbers.Real):
            raise ValueError("max_age_s must be a finite non-negative number")
        max_age = float(max_age_s)
        if not np.isfinite(max_age) or max_age < 0.0:
            raise ValueError("max_age_s must be a finite non-negative number")
        if self._closed:
            return False
        return self._received_at_s is not None and time.monotonic() - self._received_at_s <= max_age

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._socket.close()
