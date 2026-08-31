"""Session-local frame conversion and latency-aware 30 Hz action scheduling."""

from __future__ import annotations

from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from teleopit.high_level_policy.client import PolicyActionChunk
from teleopit.high_level_policy.config import HighLevelPolicySafetyConfig
from teleopit.high_level_policy.hand_calibration import HandCalibration
from teleopit.high_level_policy.protocol import MAX_ACTION_HORIZON
from teleopit.math_utils import quat_inv_np, quat_mul_np
from teleopit.sim.reference_motion import interpolate_retarget_qpos


ACTION_DIM = 50
BODY_ACTION_DIM = 36
ROOT_QUATERNION = slice(3, 7)
REFERENCE_HISTORY_SIZE = 256
MAX_REFERENCE_INTERPOLATION_GAP_PERIODS = 2.5


def _normalized_quaternion(value: object, *, name: str) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float32).reshape(-1)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError(f"{name} must be a finite wxyz quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        raise ValueError(f"{name} has a near-zero norm")
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(f"{name} norm must be near 1, got {norm:.6g}")
    return quaternion / np.float32(norm)


def _yaw_from_quaternion(value: object) -> float:
    w, x, y, z = _normalized_quaternion(value, name="anchor quaternion")
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _yaw_quaternion(yaw_rad: float) -> np.ndarray:
    half = 0.5 * float(yaw_rad)
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


@dataclass(frozen=True)
class PolicyFrameTransform:
    origin_xy: tuple[float, float]
    yaw_rad: float

    @classmethod
    def from_robot_pose(cls, root_xy: object, quaternion_wxyz: object) -> "PolicyFrameTransform":
        xy = np.asarray(root_xy, dtype=np.float64).reshape(-1)
        if xy.shape[0] < 2 or not np.all(np.isfinite(xy[:2])):
            raise ValueError("Policy session root_xy must contain two finite values")
        return cls(
            origin_xy=(float(xy[0]), float(xy[1])),
            yaw_rad=_yaw_from_quaternion(quaternion_wxyz),
        )

    def localize_body_action(self, action: object) -> np.ndarray:
        body = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        if body.shape != (BODY_ACTION_DIM,) or not np.all(np.isfinite(body)):
            raise ValueError(f"High-level body action must be finite float32[{BODY_ACTION_DIM}]")
        world_delta = body[:2].astype(np.float64) - np.asarray(self.origin_xy, dtype=np.float64)
        cosine = math.cos(self.yaw_rad)
        sine = math.sin(self.yaw_rad)
        body[0] = cosine * world_delta[0] + sine * world_delta[1]
        body[1] = -sine * world_delta[0] + cosine * world_delta[1]
        world_quaternion = _normalized_quaternion(body[ROOT_QUATERNION], name="action root quaternion")
        local_quaternion = quat_mul_np(quat_inv_np(_yaw_quaternion(self.yaw_rad)), world_quaternion)
        body[ROOT_QUATERNION] = _normalized_quaternion(
            local_quaternion, name="localized action root quaternion"
        )
        return body

    def delocalize_body_action(self, action: object) -> np.ndarray:
        body = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        if body.shape != (BODY_ACTION_DIM,) or not np.all(np.isfinite(body)):
            raise ValueError(f"High-level body action must be finite float32[{BODY_ACTION_DIM}]")
        local_xy = body[:2].astype(np.float64)
        cosine = math.cos(self.yaw_rad)
        sine = math.sin(self.yaw_rad)
        body[0] = cosine * local_xy[0] - sine * local_xy[1] + self.origin_xy[0]
        body[1] = sine * local_xy[0] + cosine * local_xy[1] + self.origin_xy[1]
        local_quaternion = _normalized_quaternion(body[ROOT_QUATERNION], name="action root quaternion")
        world_quaternion = quat_mul_np(_yaw_quaternion(self.yaw_rad), local_quaternion)
        body[ROOT_QUATERNION] = _normalized_quaternion(
            world_quaternion, name="delocalized action root quaternion"
        )
        return body


class HighLevelPolicyScheduler:
    def __init__(
        self,
        *,
        hold_s: float = 3.0,
        safety: HighLevelPolicySafetyConfig | None = None,
        output_hz: float = 50.0,
    ) -> None:
        if not np.isfinite(hold_s) or hold_s < 0.0:
            raise ValueError("high_level_policy.hold_s must be finite and >= 0")
        if not np.isfinite(output_hz) or output_hz <= 0.0:
            raise ValueError("High-level policy scheduler output_hz must be finite and > 0")
        self.hold_s = float(hold_s)
        self.safety = safety
        self.output_hz = float(output_hz)
        self._session_id: str | None = None
        self._chunk: PolicyActionChunk | None = None
        self._last_source_sequence_id = -1
        self._last_source_timestamp_ns = -1
        self._paused_at_s: float | None = None
        self._timestamp_shift_s = 0.0
        self._last_output_action: np.ndarray | None = None
        self._reference_history: deque[tuple[float, np.ndarray]] = deque(
            maxlen=REFERENCE_HISTORY_SIZE
        )

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def has_chunk(self) -> bool:
        return self._chunk is not None

    @property
    def paused(self) -> bool:
        return self._paused_at_s is not None

    def reset(
        self,
        session_id: str,
        *,
        initial_action: object | None = None,
        initial_reference: object | None = None,
        initial_timestamp_s: float | None = None,
    ) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("High-level policy session_id must be non-empty")
        if (initial_reference is None) != (initial_timestamp_s is None):
            raise ValueError(
                "High-level policy initial reference and initial_timestamp_s must be provided together"
            )
        self._session_id = session_id
        self._chunk = None
        self._last_source_sequence_id = -1
        self._last_source_timestamp_ns = -1
        self._paused_at_s = None
        self._timestamp_shift_s = 0.0
        initial_output = (
            None
            if initial_action is None
            else self._validate_single_action(initial_action, name="initial_action")
        )
        self._last_output_action = (
            None if initial_output is None else initial_output.copy()
        )
        self._reference_history.clear()
        if initial_reference is not None:
            assert initial_timestamp_s is not None
            self._record_reference(initial_timestamp_s, initial_reference)

    def clear(self) -> None:
        self._session_id = None
        self._chunk = None
        self._last_source_sequence_id = -1
        self._last_source_timestamp_ns = -1
        self._paused_at_s = None
        self._timestamp_shift_s = 0.0
        self._last_output_action = None
        self._reference_history.clear()

    def accept(self, chunk: PolicyActionChunk, *, now_s: float) -> None:
        self._accept(chunk, now_s=now_s)

    def _accept(
        self,
        chunk: PolicyActionChunk,
        *,
        now_s: float,
    ) -> None:
        if not np.isfinite(now_s):
            raise ValueError("High-level policy scheduler now_s must be finite")
        if self._session_id is None or chunk.session_id != self._session_id:
            raise ValueError(
                f"High-level policy action session mismatch: active={self._session_id!r}, "
                f"received={chunk.session_id!r}"
            )
        if (
            not isinstance(chunk.source_sequence_id, int)
            or isinstance(chunk.source_sequence_id, bool)
            or chunk.source_sequence_id < 0
        ):
            raise ValueError("High-level policy source sequence must be a non-negative integer")
        if chunk.source_sequence_id <= self._last_source_sequence_id:
            raise ValueError(
                "High-level policy source sequence must increase: "
                f"last={self._last_source_sequence_id}, received={chunk.source_sequence_id}"
            )
        if not isinstance(chunk.action_fps, int) or isinstance(chunk.action_fps, bool) or chunk.action_fps != 30:
            raise ValueError(f"High-level policy action_fps must be 30, got {chunk.action_fps}")
        if (
            not isinstance(chunk.source_onboard_monotonic_timestamp_ns, int)
            or isinstance(chunk.source_onboard_monotonic_timestamp_ns, bool)
            or not 0 <= chunk.source_onboard_monotonic_timestamp_ns <= 2**63 - 1
        ):
            raise ValueError("High-level policy source timestamp must be a non-negative int64")
        if chunk.source_onboard_monotonic_timestamp_ns <= self._last_source_timestamp_ns:
            raise ValueError(
                "High-level policy source timestamp must increase: "
                f"last={self._last_source_timestamp_ns}, "
                f"received={chunk.source_onboard_monotonic_timestamp_ns}"
            )
        if not isinstance(chunk.policy_id, str) or not chunk.policy_id:
            raise ValueError("High-level policy policy_id must be non-empty")
        if not np.isfinite(chunk.server_inference_ms) or chunk.server_inference_ms < 0.0:
            raise ValueError("High-level policy server_inference_ms must be finite and >= 0")
        source_s = chunk.source_onboard_monotonic_timestamp_ns * 1e-9
        if source_s > float(now_s) + 0.001:
            raise ValueError(
                "High-level policy source timestamp is in the future: "
                f"source={source_s:.9f}s now={float(now_s):.9f}s"
            )
        actions = self._validate_actions(chunk.actions)
        valid_until_s = source_s + len(actions) / float(chunk.action_fps) + self.hold_s
        if float(now_s) > valid_until_s:
            raise ValueError(
                "High-level policy action chunk is already expired: "
                f"age={float(now_s) - source_s:.3f}s horizon={len(actions) / chunk.action_fps:.3f}s"
            )
        self._chunk = PolicyActionChunk(
            session_id=chunk.session_id,
            source_sequence_id=chunk.source_sequence_id,
            source_onboard_monotonic_timestamp_ns=chunk.source_onboard_monotonic_timestamp_ns,
            action_fps=chunk.action_fps,
            actions=actions,
            policy_id=chunk.policy_id,
            server_inference_ms=chunk.server_inference_ms,
        )
        self._last_source_sequence_id = chunk.source_sequence_id
        self._last_source_timestamp_ns = chunk.source_onboard_monotonic_timestamp_ns
        self._timestamp_shift_s = 0.0
        if self._paused_at_s is not None:
            self._paused_at_s = float(now_s)

    def pause(self, now_s: float) -> None:
        if self._paused_at_s is None:
            timestamp_s = self._validated_timestamp(now_s, name="pause now_s")
            self._paused_at_s = timestamp_s
            self._record_last_output_reference(timestamp_s)

    def resume(self, now_s: float) -> None:
        if self._paused_at_s is None:
            return
        timestamp_s = self._validated_timestamp(now_s, name="resume now_s")
        self._timestamp_shift_s += max(0.0, timestamp_s - self._paused_at_s)
        self._paused_at_s = None
        self._record_last_output_reference(timestamp_s)

    def sample(self, now_s: float) -> np.ndarray | None:
        timestamp_s = self._validated_timestamp(now_s, name="sample now_s")
        desired = self._sample_unlimited(timestamp_s)
        if desired is None:
            return None
        previous = self._last_output_action
        safety = self.safety
        if previous is not None and safety is not None:
            desired = self._rate_limit_output(previous, desired, safety=safety)
        self._last_output_action = desired.copy()
        self._record_reference(timestamp_s, desired)
        return desired

    def reference_root_pose_at(self, timestamp_s: float) -> np.ndarray | None:
        """Return the active session-local reference root at a camera timestamp."""

        query_s = self._validated_timestamp(timestamp_s, name="reference timestamp_s")
        history = tuple(self._reference_history)
        if not history:
            return None
        if query_s <= history[0][0]:
            return history[0][1].copy()
        if query_s >= history[-1][0]:
            return history[-1][1].copy()
        timestamps = [sample[0] for sample in history]
        right_index = bisect_right(timestamps, query_s)
        left_timestamp, left_pose = history[right_index - 1]
        right_timestamp, right_pose = history[right_index]
        if (
            right_timestamp - left_timestamp
            > MAX_REFERENCE_INTERPOLATION_GAP_PERIODS / self.output_hz
        ):
            return left_pose.copy()
        alpha = (query_s - left_timestamp) / (right_timestamp - left_timestamp)
        return np.asarray(
            interpolate_retarget_qpos(left_pose, right_pose, alpha),
            dtype=np.float32,
        )

    def _record_last_output_reference(self, timestamp_s: float) -> None:
        if self._last_output_action is not None:
            self._record_reference(timestamp_s, self._last_output_action)

    def _record_reference(self, timestamp_s: float, action: object) -> None:
        timestamp = self._validated_timestamp(
            timestamp_s,
            name="reference history timestamp_s",
        )
        validated = self._validate_single_action(action, name="reference history action")
        root_pose = validated[:7].copy()
        if self._reference_history:
            latest_timestamp = self._reference_history[-1][0]
            if timestamp < latest_timestamp:
                raise ValueError(
                    "High-level policy reference history timestamp must not decrease: "
                    f"last={latest_timestamp:.9f}s received={timestamp:.9f}s"
                )
            if timestamp == latest_timestamp:
                self._reference_history[-1] = (timestamp, root_pose)
                return
        self._reference_history.append((timestamp, root_pose))

    @staticmethod
    def _validated_timestamp(value: object, *, name: str) -> float:
        timestamp = float(value)
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError(f"High-level policy {name} must be finite and >= 0")
        return timestamp

    def _sample_unlimited(self, now_s: float) -> np.ndarray | None:
        chunk = self._chunk
        if chunk is None:
            return None
        effective_now_s = self._paused_at_s if self._paused_at_s is not None else float(now_s)
        source_s = chunk.source_onboard_monotonic_timestamp_ns * 1e-9 + self._timestamp_shift_s
        frame_f = (effective_now_s - source_s) * float(chunk.action_fps)
        if frame_f <= 0.0:
            return chunk.actions[0].copy()
        last_index = len(chunk.actions) - 1
        if frame_f >= float(last_index):
            valid_until_s = source_s + len(chunk.actions) / float(chunk.action_fps) + self.hold_s
            if effective_now_s > valid_until_s:
                return None
            return chunk.actions[last_index].copy()
        index0 = int(math.floor(frame_f))
        index1 = min(index0 + 1, last_index)
        alpha = float(frame_f - index0)
        interpolated = interpolate_retarget_qpos(
            np.asarray(chunk.actions[index0], dtype=np.float64),
            np.asarray(chunk.actions[index1], dtype=np.float64),
            alpha,
        )
        return np.asarray(interpolated, dtype=np.float32)

    def _rate_limit_output(
        self,
        previous: np.ndarray,
        desired: np.ndarray,
        *,
        safety: HighLevelPolicySafetyConfig,
    ) -> np.ndarray:
        output = np.asarray(desired, dtype=np.float32).copy()
        previous = self._validate_single_action(previous, name="previous output action")

        root_delta = output[0:3].astype(np.float64) - previous[0:3].astype(np.float64)
        max_root_delta = safety.max_root_displacement_m * 30.0 / self.output_hz
        root_distance = float(np.linalg.norm(root_delta))
        if root_distance > max_root_delta:
            root_delta *= max_root_delta / root_distance
        max_xy_delta = safety.max_root_xy_speed_m_s / self.output_hz
        xy_distance = float(np.linalg.norm(root_delta[:2]))
        if xy_distance > max_xy_delta:
            root_delta[:2] *= max_xy_delta / xy_distance
        output[0:3] = previous[0:3] + root_delta.astype(np.float32)

        previous_yaw = _yaw_from_quaternion(previous[ROOT_QUATERNION])
        desired_quaternion = _normalized_quaternion(
            output[ROOT_QUATERNION], name="desired output root quaternion"
        )
        desired_yaw = _yaw_from_quaternion(desired_quaternion)
        yaw_delta = math.atan2(
            math.sin(desired_yaw - previous_yaw),
            math.cos(desired_yaw - previous_yaw),
        )
        max_yaw_delta = safety.max_yaw_rate_rad_s / self.output_hz
        limited_yaw = previous_yaw + float(np.clip(yaw_delta, -max_yaw_delta, max_yaw_delta))
        desired_tilt = quat_mul_np(
            quat_inv_np(_yaw_quaternion(desired_yaw)),
            desired_quaternion,
        )
        limited_quaternion = _normalized_quaternion(
            quat_mul_np(_yaw_quaternion(limited_yaw), desired_tilt),
            name="rate-limited output root quaternion",
        )
        if float(np.dot(previous[ROOT_QUATERNION], limited_quaternion)) < 0.0:
            limited_quaternion = -limited_quaternion
        output[ROOT_QUATERNION] = limited_quaternion

        max_joint_delta = safety.max_joint_rate_rad_s / self.output_hz
        output[7:36] = previous[7:36] + np.clip(
            output[7:36] - previous[7:36],
            -max_joint_delta,
            max_joint_delta,
        )
        return output

    def _validate_actions(
        self,
        values: object,
    ) -> np.ndarray:
        actions = np.asarray(values)
        if (
            actions.ndim != 2
            or actions.shape[1] != ACTION_DIM
            or not 1 <= len(actions) <= MAX_ACTION_HORIZON
        ):
            raise ValueError(
                f"High-level policy actions must have shape [T, {ACTION_DIM}] "
                f"with T in [1, {MAX_ACTION_HORIZON}]"
            )
        if not np.issubdtype(actions.dtype, np.number) or not np.all(np.isfinite(actions)):
            raise ValueError("High-level policy actions must be finite numeric values")
        validated = np.ascontiguousarray(actions, dtype=np.float32)
        previous: np.ndarray | None = None
        for index in range(len(validated)):
            quaternion = _normalized_quaternion(
                validated[index, ROOT_QUATERNION], name=f"action[{index}] root quaternion"
            )
            if previous is not None and float(np.dot(previous, quaternion)) < 0.0:
                quaternion = -quaternion
            validated[index, ROOT_QUATERNION] = quaternion
            previous = quaternion
        hand = validated[:, 36:48]
        if float(np.min(hand)) < 0.0 or float(np.max(hand)) > 1.0:
            raise ValueError("High-level policy LinkerHand closure must be within [0, 1]")
        safety = self.safety
        if safety is not None:
            joints = validated[:, 7:36]
            projected = np.clip(
                joints,
                np.asarray(safety.joint_pos_lower, dtype=np.float32),
                np.asarray(safety.joint_pos_upper, dtype=np.float32),
            )
            correction = np.abs(projected - joints)
            violations = np.argwhere(correction > safety.max_joint_projection_rad)
            if len(violations):
                frame, joint = (int(value) for value in violations[0])
                raise ValueError(
                    "High-level policy joint projection correction exceeds "
                    f"{safety.max_joint_projection_rad:.6g} rad: "
                    f"action[{frame}, {7 + joint}] correction="
                    f"{float(correction[frame, joint]):.6g} rad"
                )
            validated[:, 7:36] = projected
            validated[:, 48] = np.clip(
                validated[:, 48],
                safety.neck_yaw_min_deg,
                safety.neck_yaw_max_deg,
            )
            validated[:, 49] = np.clip(
                validated[:, 49],
                safety.neck_pitch_min_deg,
                safety.neck_pitch_max_deg,
            )
            self._validate_safety_limits(validated, safety=safety)
        return validated

    @staticmethod
    def _validate_single_action(values: object, *, name: str) -> np.ndarray:
        action = np.asarray(values)
        if action.shape != (ACTION_DIM,) or not np.issubdtype(action.dtype, np.number):
            raise ValueError(f"{name} must be a numeric float32[{ACTION_DIM}]")
        action = np.ascontiguousarray(action, dtype=np.float32)
        if not np.all(np.isfinite(action)):
            raise ValueError(f"{name} must contain only finite values")
        action[ROOT_QUATERNION] = _normalized_quaternion(
            action[ROOT_QUATERNION], name=f"{name} root quaternion"
        )
        return action

    @staticmethod
    def _validate_safety_limits(
        actions: np.ndarray,
        *,
        safety: HighLevelPolicySafetyConfig,
    ) -> None:
        root_height = actions[:, 2]
        if (
            float(np.min(root_height)) < safety.root_height_min_m
            or float(np.max(root_height)) > safety.root_height_max_m
        ):
            raise ValueError(
                "High-level policy root height is outside "
                f"[{safety.root_height_min_m}, {safety.root_height_max_m}] m"
            )

        joints = actions[:, 7:36]
        lower = np.asarray(safety.joint_pos_lower, dtype=np.float32)
        upper = np.asarray(safety.joint_pos_upper, dtype=np.float32)
        violations = np.argwhere((joints < lower[None, :]) | (joints > upper[None, :]))
        if len(violations):
            frame, joint = (int(value) for value in violations[0])
            raise ValueError(
                "High-level policy joint position exceeds real_robot limits: "
                f"action[{frame}, {7 + joint}]={float(joints[frame, joint]):.6g}, "
                f"range=[{float(lower[joint]):.6g}, {float(upper[joint]):.6g}]"
            )


def closure_to_o6_pose(
    closure: object,
    calibration: HandCalibration | None = None,
) -> tuple[int, ...]:
    calibration = calibration or HandCalibration.load()
    values = np.asarray(closure, dtype=np.float32).reshape(-1)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("LinkerHand O6 closure must contain six finite values")
    if float(np.min(values)) < 0.0 or float(np.max(values)) > 1.0:
        raise ValueError("LinkerHand O6 closure must be within [0, 1]")
    opened = np.asarray(calibration.open_raw, dtype=np.float32)
    closed = np.asarray(calibration.close_raw, dtype=np.float32)
    raw = np.rint(opened - values * (opened - closed)).astype(np.int64)
    return tuple(int(value) for value in raw)
