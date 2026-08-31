from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from teleopit.sim2real.neck.config import NeckConfig


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class NeckCommand:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


class HmdPoseMapper:
    """Map synchronized Pico HMD/Spine3 orientations to OpenNeck angles."""

    def __init__(self, config: NeckConfig) -> None:
        self._cfg = config

    def map_pose(
        self,
        *,
        hmd_rotation_wxyz: FloatArray | None,
        spine3_rotation_wxyz: FloatArray | None,
    ) -> NeckCommand | None:
        q_hmd = _normalized_quat(hmd_rotation_wxyz)
        if q_hmd is None:
            return None
        q_body = _normalized_quat(spine3_rotation_wxyz)
        if q_body is None:
            return None
        # The HMD and Spine3 share the same neutral orientation in the supported
        # PICO convention, so their relative identity is the fixed zero pose.
        # Deliberately do not read the full-body tracker's skeleton Head joint:
        # its model constraints can under-report extreme head pitch.
        q_cmd = _qmul(_qconj(q_body), q_hmd)
        yaw_deg, pitch_deg, roll_deg = _openneck_yaw_pitch_roll_deg(q_cmd)
        # Convert the supported PICO convention to OpenNeck's physical command
        # convention: positive yaw turns left and positive pitch looks up.
        yaw_deg = -yaw_deg
        pitch_deg = -pitch_deg
        if abs(yaw_deg) < self._cfg.dead_zone_deg:
            yaw_deg = 0.0
        if abs(pitch_deg) < self._cfg.dead_zone_deg:
            pitch_deg = 0.0
        else:
            pitch_deg *= self._cfg.pitch_gain

        return NeckCommand(
            yaw_deg=float(yaw_deg),
            pitch_deg=float(pitch_deg),
            roll_deg=float(roll_deg),
        )


def _normalized_quat(value: FloatArray | None) -> FloatArray | None:
    if value is None:
        return None
    quat = np.asarray(value, dtype=np.float64).reshape(-1)
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        return None
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-9:
        return None
    return quat / norm


def _qconj(q: FloatArray) -> FloatArray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _qmul(a: FloatArray, b: FloatArray) -> FloatArray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _openneck_yaw_pitch_roll_deg(q_wxyz: FloatArray) -> tuple[float, float, float]:
    w, x, y, z = q_wxyz
    yaw = math.degrees(math.atan2(2.0 * (x * z + w * y), 1.0 - 2.0 * (y * y + z * z)))
    pitch = math.degrees(math.asin(float(np.clip(-2.0 * (y * z - w * x), -1.0, 1.0))))
    roll = math.degrees(math.atan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z)))
    return yaw, pitch, roll
