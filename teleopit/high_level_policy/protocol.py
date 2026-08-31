"""Serialization helpers and fixed limits for high-level-policy messages."""

from __future__ import annotations

from typing import Any, Mapping

import msgpack
import numpy as np


ENDPOINTS = frozenset({"ping", "describe", "reset", "get_action"})
MAX_REQUEST_BYTES = 2_097_152
MAX_RESPONSE_BYTES = 262_144
MAX_IMAGE_BYTES = 1_572_864
MAX_TASK_UTF8_BYTES = 1_024
MAX_SESSION_ID_UTF8_BYTES = 128
MAX_ACTION_HORIZON = 50


class PolicyProtocolError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)


def encode_float32_array(values: object) -> dict[str, object]:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number):
        raise PolicyProtocolError("invalid_array", f"Array dtype must be numeric, got {array.dtype}")
    encoded = np.ascontiguousarray(array, dtype="<f4")
    if not np.all(np.isfinite(encoded)):
        raise PolicyProtocolError("nonfinite_array", "Array contains non-finite values")
    return {"dtype": "<f4", "shape": list(encoded.shape), "data": encoded.tobytes(order="C")}


def decode_float32_array(
    value: object,
    *,
    name: str,
    expected_shape: tuple[int | None, ...],
) -> np.ndarray:
    if not isinstance(value, Mapping) or set(value) != {"dtype", "shape", "data"}:
        raise PolicyProtocolError("invalid_array", f"{name} must contain exactly dtype, shape, and data")
    if value["dtype"] != "<f4":
        raise PolicyProtocolError("invalid_array", f"{name}.dtype must be '<f4'")
    shape = value["shape"]
    if not isinstance(shape, list) or len(shape) != len(expected_shape):
        raise PolicyProtocolError("invalid_array", f"{name}.shape must have rank {len(expected_shape)}")
    parsed: list[int] = []
    for index, (actual, expected) in enumerate(zip(shape, expected_shape, strict=True)):
        if not isinstance(actual, int) or isinstance(actual, bool) or actual < 0:
            raise PolicyProtocolError("invalid_array", f"{name}.shape[{index}] must be a non-negative int")
        if expected is not None and actual != expected:
            raise PolicyProtocolError(
                "invalid_array", f"{name}.shape[{index}] must be {expected}, got {actual}"
            )
        parsed.append(actual)
    raw = value["data"]
    if not isinstance(raw, bytes):
        raise PolicyProtocolError("invalid_array", f"{name}.data must be msgpack binary")
    expected_bytes = int(np.prod(parsed, dtype=np.int64)) * np.dtype("<f4").itemsize
    if len(raw) != expected_bytes:
        raise PolicyProtocolError(
            "invalid_array", f"{name}.data has {len(raw)} bytes, expected {expected_bytes}"
        )
    decoded = np.frombuffer(raw, dtype="<f4").reshape(tuple(parsed)).copy()
    if not np.all(np.isfinite(decoded)):
        raise PolicyProtocolError("nonfinite_array", f"{name} contains non-finite values")
    return decoded


def pack_message(message: Mapping[str, Any], *, max_bytes: int) -> bytes:
    try:
        payload = msgpack.packb(dict(message), use_bin_type=True, strict_types=True)
    except (TypeError, ValueError) as exc:
        raise PolicyProtocolError("serialization_error", f"Cannot serialize message: {exc}") from exc
    if len(payload) > max_bytes:
        raise PolicyProtocolError(
            "message_too_large", f"Serialized message has {len(payload)} bytes; limit is {max_bytes}"
        )
    return payload


def unpack_message(payload: object, *, max_bytes: int) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        raise PolicyProtocolError("invalid_message", "Protocol payload must be bytes")
    if len(payload) > max_bytes:
        raise PolicyProtocolError("message_too_large", f"Message has {len(payload)} bytes; limit is {max_bytes}")
    try:
        message = msgpack.unpackb(
            payload,
            raw=False,
            strict_map_key=True,
            max_bin_len=max_bytes,
            max_str_len=max_bytes,
            max_array_len=128,
            max_map_len=64,
            max_ext_len=0,
        )
    except (msgpack.ExtraData, msgpack.FormatError, msgpack.StackError, ValueError) as exc:
        raise PolicyProtocolError("invalid_msgpack", f"Cannot decode message: {exc}") from exc
    if not isinstance(message, dict):
        raise PolicyProtocolError("invalid_message", "Protocol message must be a map")
    return message
