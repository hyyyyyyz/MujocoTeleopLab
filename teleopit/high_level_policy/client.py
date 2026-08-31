"""Synchronous ZeroMQ client used only from the isolated onboard worker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import zmq

from teleopit.high_level_policy.protocol import (
    ENDPOINTS,
    MAX_ACTION_HORIZON,
    MAX_IMAGE_BYTES,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    PolicyProtocolError,
    decode_float32_array,
    encode_float32_array,
    pack_message,
    unpack_message,
)


class PolicyTransportError(RuntimeError):
    """The host did not complete a REQ/REP exchange before the deadline."""


@dataclass(frozen=True)
class PolicyDescription:
    policy_type: str
    policy_id: str
    dataset_fps: int
    max_action_horizon: int


@dataclass(frozen=True)
class PolicyActionChunk:
    session_id: str
    source_sequence_id: int
    source_onboard_monotonic_timestamp_ns: int
    action_fps: int
    actions: np.ndarray
    policy_id: str
    server_inference_ms: float


class HighLevelPolicyClient:
    def __init__(
        self,
        endpoint: str,
        *,
        timeout_s: float,
        context: zmq.Context[Any] | None = None,
    ) -> None:
        if not str(endpoint).startswith(("tcp://", "inproc://")):
            raise ValueError("High-level policy endpoint must use tcp:// (or inproc:// in tests)")
        if not np.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("High-level policy timeout_s must be finite and > 0")
        self.endpoint = str(endpoint)
        self.timeout_s = float(timeout_s)
        self._own_context = context is None
        self._context = zmq.Context() if context is None else context
        self._socket: zmq.Socket[Any] | None = None
        self._open_socket()

    def close(self) -> None:
        self._close_socket()
        if self._own_context:
            self._context.term()

    def ping(self) -> bool:
        data = self._request("ping", {})
        if set(data) != {"ready"} or not isinstance(data["ready"], bool):
            raise PolicyProtocolError("invalid_response", "ping data must contain exactly boolean ready")
        return bool(data["ready"])

    def describe(self) -> PolicyDescription:
        data = self._request("describe", {})
        expected_fields = {
            "observation_schema",
            "observation_dim",
            "action_schema",
            "action_dim",
            "dataset_fps",
            "max_action_horizon",
            "policy_type",
            "policy_id",
            "ready",
        }
        if set(data) != expected_fields:
            raise PolicyProtocolError("invalid_response", "describe data contains unexpected fields")
        expected_schema = {
            "observation_schema": "teleopit-g1-joint-pos-dex-neck-state",
            "observation_dim": 43,
            "action_schema": "teleopit-g1-reference",
            "action_dim": 50,
        }
        for name, value in expected_schema.items():
            if data[name] != value:
                raise PolicyProtocolError(
                    "schema_mismatch", f"describe {name} must be {value!r}, got {data[name]!r}"
                )
        if data["dataset_fps"] != 30:
            raise PolicyProtocolError(
                "invalid_response", f"describe dataset_fps must be 30, got {data['dataset_fps']!r}"
            )
        if data["ready"] is not True:
            raise PolicyProtocolError("policy_not_ready", "Host policy reported ready=false")
        if not isinstance(data["policy_type"], str) or not data["policy_type"]:
            raise PolicyProtocolError("invalid_response", "describe policy_type must be non-empty")
        if not isinstance(data["policy_id"], str) or not data["policy_id"]:
            raise PolicyProtocolError("invalid_response", "describe policy_id must be non-empty")
        horizon = _int64(data["max_action_horizon"], name="max_action_horizon")
        if not 1 <= horizon <= MAX_ACTION_HORIZON:
            raise PolicyProtocolError("invalid_response", "describe max_action_horizon is outside limits")
        return PolicyDescription(
            policy_type=data["policy_type"],
            policy_id=data["policy_id"],
            dataset_fps=30,
            max_action_horizon=horizon,
        )

    def reset(self, session_id: str, task: str) -> None:
        data = self._request("reset", {"session_id": session_id, "task": task})
        if set(data) != {"session_id", "reset"} or data["session_id"] != session_id or data["reset"] is not True:
            raise PolicyProtocolError("invalid_response", "reset acknowledgement does not match the requested session")

    def get_action(
        self,
        *,
        session_id: str,
        sequence_id: int,
        onboard_monotonic_timestamp_ns: int,
        task: str,
        jpeg_image: bytes,
        body_joint_positions: object,
        dex_state: object,
        neck_state: object,
        source_reference_root_pose: object,
    ) -> PolicyActionChunk:
        if not isinstance(jpeg_image, bytes):
            raise PolicyProtocolError("invalid_image", "jpeg_image must be bytes")
        if not 4 <= len(jpeg_image) <= MAX_IMAGE_BYTES:
            raise PolicyProtocolError("invalid_image", "jpeg_image size is outside limits")
        if not jpeg_image.startswith(b"\xff\xd8") or not jpeg_image.endswith(b"\xff\xd9"):
            raise PolicyProtocolError("invalid_image", "jpeg_image is missing JPEG start/end markers")
        body_array = _finite_float32_vector(
            body_joint_positions,
            name="body_joint_positions",
            size=29,
        )
        dex_array = _finite_float32_vector(dex_state, name="dex_state", size=12)
        neck_array = _finite_float32_vector(neck_state, name="neck_state", size=2)
        source_pose_array = _finite_float32_vector(
            source_reference_root_pose,
            name="source_reference_root_pose",
            size=7,
        )
        quaternion_norm = float(np.linalg.norm(source_pose_array[3:7]))
        if abs(quaternion_norm - 1.0) > 1e-3:
            raise PolicyProtocolError(
                "invalid_reference_pose",
                "source_reference_root_pose quaternion norm must be near 1, "
                f"got {quaternion_norm:.6g}",
            )
        request_data = {
            "session_id": session_id,
            "sequence_id": _int64(sequence_id, name="sequence_id"),
            "onboard_monotonic_timestamp_ns": _int64(
                onboard_monotonic_timestamp_ns,
                name="onboard_monotonic_timestamp_ns",
            ),
            "task": task,
            "image_encoding": "jpeg",
            "image": jpeg_image,
            "body_joint_positions": encode_float32_array(body_array),
            "dex_state": encode_float32_array(dex_array),
            "neck_state": encode_float32_array(neck_array),
            "source_reference_root_pose": encode_float32_array(source_pose_array),
        }
        data = self._request("get_action", request_data)
        expected_fields = {
            "session_id",
            "source_sequence_id",
            "source_onboard_monotonic_timestamp_ns",
            "action_fps",
            "actions",
            "policy_id",
            "server_inference_ms",
        }
        if set(data) != expected_fields:
            raise PolicyProtocolError("invalid_response", "get_action data contains unexpected fields")
        if data["session_id"] != session_id:
            raise PolicyProtocolError("session_mismatch", "get_action response session_id does not match request")
        source_sequence = _int64(data["source_sequence_id"], name="source_sequence_id")
        source_timestamp = _int64(
            data["source_onboard_monotonic_timestamp_ns"],
            name="source_onboard_monotonic_timestamp_ns",
        )
        if source_sequence != sequence_id or source_timestamp != onboard_monotonic_timestamp_ns:
            raise PolicyProtocolError("stale_response", "get_action response does not echo the source observation")
        action_fps = _int64(data["action_fps"], name="action_fps")
        if action_fps != 30:
            raise PolicyProtocolError("invalid_response", f"action_fps must be 30, got {action_fps}")
        actions = decode_float32_array(data["actions"], name="actions", expected_shape=(None, 50))
        if not 1 <= len(actions) <= MAX_ACTION_HORIZON:
            raise PolicyProtocolError("invalid_response", f"actions horizon is invalid: {len(actions)}")
        if not isinstance(data["policy_id"], str) or not data["policy_id"]:
            raise PolicyProtocolError("invalid_response", "policy_id must be non-empty")
        inference_ms = float(data["server_inference_ms"])
        if not np.isfinite(inference_ms) or inference_ms < 0.0:
            raise PolicyProtocolError("invalid_response", "server_inference_ms must be finite and >= 0")
        return PolicyActionChunk(
            session_id=session_id,
            source_sequence_id=source_sequence,
            source_onboard_monotonic_timestamp_ns=source_timestamp,
            action_fps=action_fps,
            actions=actions,
            policy_id=data["policy_id"],
            server_inference_ms=inference_ms,
        )

    def _request(self, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
        if endpoint not in ENDPOINTS:
            raise PolicyProtocolError("unknown_endpoint", f"Unsupported endpoint {endpoint!r}")
        request = {"endpoint": endpoint, "data": data}
        payload = pack_message(request, max_bytes=MAX_REQUEST_BYTES)
        socket = self._socket
        if socket is None:
            raise PolicyTransportError("High-level policy client is closed")
        try:
            socket.send(payload)
            reply_payload = socket.recv()
        except zmq.Again as exc:
            self._recreate_socket()
            raise PolicyTransportError(
                f"High-level policy {endpoint} timed out after {self.timeout_s:.3f}s"
            ) from exc
        except zmq.ZMQError as exc:
            self._recreate_socket()
            raise PolicyTransportError(
                f"High-level policy {endpoint} transport failed: {exc}"
            ) from exc
        reply = unpack_message(reply_payload, max_bytes=MAX_RESPONSE_BYTES)
        return self._parse_reply(reply, endpoint=endpoint)

    def _parse_reply(self, reply: Mapping[str, Any], *, endpoint: str) -> dict[str, Any]:
        base = {"endpoint", "ok"}
        if not base.issubset(reply):
            raise PolicyProtocolError("invalid_response", "Response is missing envelope fields")
        if reply["endpoint"] != endpoint:
            raise PolicyProtocolError("invalid_response", f"Response endpoint {reply['endpoint']!r} != {endpoint!r}")
        if not isinstance(reply["ok"], bool):
            raise PolicyProtocolError("invalid_response", "Response ok must be boolean")
        if reply["ok"]:
            if set(reply) != base | {"data"} or not isinstance(reply["data"], dict):
                raise PolicyProtocolError("invalid_response", "Successful response must contain exactly data")
            return dict(reply["data"])
        if set(reply) != base | {"error"} or not isinstance(reply["error"], Mapping):
            raise PolicyProtocolError("invalid_response", "Failed response must contain exactly error")
        error = reply["error"]
        if set(error) != {"code", "message"} or not all(isinstance(error[key], str) for key in error):
            raise PolicyProtocolError("invalid_response", "Response error must contain string code/message")
        raise PolicyProtocolError(error["code"], error["message"])

    def _open_socket(self) -> None:
        socket = self._context.socket(zmq.REQ)
        timeout_ms = max(1, int(round(self.timeout_s * 1000.0)))
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVHWM, 1)
        socket.setsockopt(zmq.SNDHWM, 1)
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        socket.setsockopt(zmq.MAXMSGSIZE, MAX_RESPONSE_BYTES)
        socket.connect(self.endpoint)
        self._socket = socket

    def _close_socket(self) -> None:
        socket = self._socket
        self._socket = None
        if socket is not None:
            socket.close(linger=0)

    def _recreate_socket(self) -> None:
        self._close_socket()
        self._open_socket()


def _int64(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 2**63 - 1:
        raise PolicyProtocolError("invalid_value", f"{name} must be an int64 in [0, 2^63-1]")
    return int(value)


def _finite_float32_vector(value: object, *, name: str, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise PolicyProtocolError(
            "invalid_observation",
            f"{name} must be finite float32[{size}], got {array.shape}",
        )
    return np.ascontiguousarray(array, dtype=np.float32)
