from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from teleopit.high_level_policy.protocol import MAX_ACTION_HORIZON
from teleopit.runtime.common import cfg_get


@dataclass(frozen=True)
class HighLevelPolicyConfig:
    endpoint: str
    task: str
    timeout_s: float
    reconnect_backoff_s: float
    replan_steps: int
    jpeg_quality: int
    max_observation_age_s: float
    max_result_age_s: float
    entry_timeout_s: float
    hold_s: float


@dataclass(frozen=True)
class HighLevelPolicyCameraConfig:
    source: str
    width: int
    height: int
    fps: int
    device: str | None


@dataclass(frozen=True)
class HighLevelPolicySafetyConfig:
    root_height_min_m: float
    root_height_max_m: float
    max_root_xy_speed_m_s: float
    max_root_displacement_m: float
    max_yaw_rate_rad_s: float
    max_joint_rate_rad_s: float
    max_joint_projection_rad: float
    joint_pos_lower: tuple[float, ...]
    joint_pos_upper: tuple[float, ...]
    neck_yaw_min_deg: float
    neck_yaw_max_deg: float
    neck_pitch_min_deg: float
    neck_pitch_max_deg: float


def parse_high_level_policy_config(cfg: Any) -> HighLevelPolicyConfig:
    policy_cfg = cfg_get(cfg, "high_level_policy", {}) or {}
    endpoint = str(cfg_get(policy_cfg, "endpoint", "tcp://127.0.0.1:5555")).strip()
    if not endpoint.startswith("tcp://"):
        raise ValueError("high_level_policy.endpoint must be a tcp:// endpoint")
    task = str(cfg_get(policy_cfg, "task", "")).strip()
    if not task:
        raise ValueError("high_level_policy.task must be a non-empty prompt")
    if len(task.encode("utf-8")) > 1024:
        raise ValueError("high_level_policy.task exceeds the protocol 1024-byte UTF-8 limit")
    timeout_s = _positive_float(cfg_get(policy_cfg, "timeout_s", 1.0), "timeout_s")
    reconnect_backoff_s = _positive_float(
        cfg_get(policy_cfg, "reconnect_backoff_s", 1.0), "reconnect_backoff_s"
    )
    replan_steps = int(cfg_get(policy_cfg, "replan_steps", 3))
    if not 1 <= replan_steps <= MAX_ACTION_HORIZON:
        raise ValueError(
            "high_level_policy.replan_steps must be in "
            f"[1, {MAX_ACTION_HORIZON}]"
        )
    jpeg_quality = int(cfg_get(policy_cfg, "jpeg_quality", 90))
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("high_level_policy.jpeg_quality must be in [1, 100]")
    max_observation_age_s = _positive_float(
        cfg_get(policy_cfg, "max_observation_age_s", 0.15), "max_observation_age_s"
    )
    max_result_age_s = _positive_float(
        cfg_get(policy_cfg, "max_result_age_s", 0.1), "max_result_age_s"
    )
    entry_timeout_s = _positive_float(
        cfg_get(policy_cfg, "entry_timeout_s", 5.0), "entry_timeout_s"
    )
    hold_s = float(cfg_get(policy_cfg, "hold_s", 3.0))
    if not math.isfinite(hold_s) or hold_s < 0.0:
        raise ValueError("high_level_policy.hold_s must be finite and >= 0")
    return HighLevelPolicyConfig(
        endpoint=endpoint,
        task=task,
        timeout_s=timeout_s,
        reconnect_backoff_s=reconnect_backoff_s,
        replan_steps=replan_steps,
        jpeg_quality=jpeg_quality,
        max_observation_age_s=max_observation_age_s,
        max_result_age_s=max_result_age_s,
        entry_timeout_s=entry_timeout_s,
        hold_s=hold_s,
    )


def parse_high_level_policy_camera_config(cfg: Any) -> HighLevelPolicyCameraConfig:
    camera_cfg = cfg_get(cfg, "camera", {}) or {}
    source = str(cfg_get(camera_cfg, "source", "realsense")).strip().lower()
    if source not in ("realsense", "test-pattern"):
        raise ValueError("camera.source must be realsense or test-pattern")
    width = int(cfg_get(camera_cfg, "width", 640))
    height = int(cfg_get(camera_cfg, "height", 480))
    fps = int(cfg_get(camera_cfg, "fps", 30))
    if (width, height, fps) != (640, 480, 30):
        raise ValueError(
            "High-level policy camera must be exactly width=640, height=480, fps=30"
        )
    device = cfg_get(camera_cfg, "device", None)
    return HighLevelPolicyCameraConfig(
        source=source,
        width=width,
        height=height,
        fps=fps,
        device=None if device in (None, "", "null") else str(device),
    )


def parse_high_level_policy_safety_config(cfg: Any) -> HighLevelPolicySafetyConfig:
    policy_cfg = cfg_get(cfg, "high_level_policy", {}) or {}
    safety_cfg = cfg_get(policy_cfg, "safety", {}) or {}
    real_cfg = cfg_get(cfg, "real_robot", {}) or {}

    root_height_min_m = _finite_float(
        cfg_get(safety_cfg, "root_height_min_m", 0.55),
        "safety.root_height_min_m",
    )
    root_height_max_m = _finite_float(
        cfg_get(safety_cfg, "root_height_max_m", 1.05),
        "safety.root_height_max_m",
    )
    if root_height_min_m >= root_height_max_m:
        raise ValueError(
            "high_level_policy.safety.root_height_min_m must be less than root_height_max_m"
        )

    joint_pos_lower = _joint_limit_vector(
        cfg_get(real_cfg, "joint_pos_lower", None),
        "real_robot.joint_pos_lower",
    )
    joint_pos_upper = _joint_limit_vector(
        cfg_get(real_cfg, "joint_pos_upper", None),
        "real_robot.joint_pos_upper",
    )
    if np.any(np.asarray(joint_pos_lower) >= np.asarray(joint_pos_upper)):
        raise ValueError("real_robot joint position lower limits must be below upper limits")

    neck_yaw_min_deg = _finite_float(
        cfg_get(safety_cfg, "neck_yaw_min_deg", -45.0),
        "safety.neck_yaw_min_deg",
    )
    neck_yaw_max_deg = _finite_float(
        cfg_get(safety_cfg, "neck_yaw_max_deg", 45.0),
        "safety.neck_yaw_max_deg",
    )
    neck_pitch_min_deg = _finite_float(
        cfg_get(safety_cfg, "neck_pitch_min_deg", -40.0),
        "safety.neck_pitch_min_deg",
    )
    neck_pitch_max_deg = _finite_float(
        cfg_get(safety_cfg, "neck_pitch_max_deg", 40.0),
        "safety.neck_pitch_max_deg",
    )
    if neck_yaw_min_deg >= neck_yaw_max_deg:
        raise ValueError(
            "high_level_policy.safety.neck_yaw_min_deg must be less than neck_yaw_max_deg"
        )
    if neck_pitch_min_deg >= neck_pitch_max_deg:
        raise ValueError(
            "high_level_policy.safety.neck_pitch_min_deg must be less than neck_pitch_max_deg"
        )

    return HighLevelPolicySafetyConfig(
        root_height_min_m=root_height_min_m,
        root_height_max_m=root_height_max_m,
        max_root_xy_speed_m_s=_positive_float(
            cfg_get(safety_cfg, "max_root_xy_speed_m_s", 2.5),
            "safety.max_root_xy_speed_m_s",
        ),
        max_root_displacement_m=_positive_float(
            cfg_get(safety_cfg, "max_root_displacement_m", 0.1),
            "safety.max_root_displacement_m",
        ),
        max_yaw_rate_rad_s=_positive_float(
            cfg_get(safety_cfg, "max_yaw_rate_rad_s", 2.5),
            "safety.max_yaw_rate_rad_s",
        ),
        max_joint_rate_rad_s=_positive_float(
            cfg_get(safety_cfg, "max_joint_rate_rad_s", 10.0),
            "safety.max_joint_rate_rad_s",
        ),
        max_joint_projection_rad=_positive_float(
            cfg_get(safety_cfg, "max_joint_projection_rad", 0.1),
            "safety.max_joint_projection_rad",
        ),
        joint_pos_lower=joint_pos_lower,
        joint_pos_upper=joint_pos_upper,
        neck_yaw_min_deg=neck_yaw_min_deg,
        neck_yaw_max_deg=neck_yaw_max_deg,
        neck_pitch_min_deg=neck_pitch_min_deg,
        neck_pitch_max_deg=neck_pitch_max_deg,
    )


def _positive_float(value: object, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"high_level_policy.{name} must be finite and > 0")
    return parsed


def _finite_float(value: object, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"high_level_policy.{name} must be finite")
    return parsed


def _joint_limit_vector(value: object, name: str) -> tuple[float, ...]:
    if value is None:
        raise ValueError(
            f"{name} is required for high-level-policy action validation"
        )
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (29,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain 29 finite values")
    return tuple(float(item) for item in array)
