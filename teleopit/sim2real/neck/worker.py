from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from teleopit.sim2real.neck.config import NeckConfig, parse_neck_config
from teleopit.sim2real.neck.mapper import HmdPoseMapper, NeckCommand
from teleopit.sim2real.neck.openneck import NeckDevice, build_neck_device

logger = logging.getLogger(__name__)
FloatArray = NDArray[np.float64]


class NeckRuntime:
    def __init__(self, config: NeckConfig, device: NeckDevice | None = None) -> None:
        self._cfg = config
        self._device = device or build_neck_device(config)
        self._mapper = HmdPoseMapper(config)

    def start(self) -> None:
        self._device.connect()
        if self._cfg.center_on_start:
            self._device.center()

    def read_deg(self) -> tuple[float, float]:
        return self._device.read_deg()

    def tick(
        self,
        *,
        hmd_rotation_wxyz: FloatArray | None,
        spine3_rotation_wxyz: FloatArray | None,
        pose_timestamp_s: float | None,
        active: bool,
        now_s: float | None = None,
    ) -> NeckCommand | None:
        now = time.monotonic() if now_s is None else float(now_s)
        if not active or pose_timestamp_s is None:
            return None
        if now - float(pose_timestamp_s) > self._cfg.frame_timeout_s:
            return None
        command = self._mapper.map_pose(
            hmd_rotation_wxyz=hmd_rotation_wxyz,
            spine3_rotation_wxyz=spine3_rotation_wxyz,
        )
        if command is None:
            return None
        applied_yaw_deg, applied_pitch_deg = self._device.move_deg(
            command.yaw_deg,
            command.pitch_deg,
        )
        return NeckCommand(
            yaw_deg=applied_yaw_deg,
            pitch_deg=applied_pitch_deg,
            roll_deg=command.roll_deg,
        )

    def close(self) -> None:
        try:
            if self._cfg.center_on_shutdown:
                try:
                    self._device.center()
                except Exception:
                    logger.exception("Failed to center OpenNeck on shutdown; closing device")
            if self._cfg.release_on_shutdown:
                try:
                    self._device.release_torque()
                except Exception:
                    logger.exception("Failed to release OpenNeck torque on shutdown; closing device")
        finally:
            self._device.close()


class DisabledNeckRuntime:
    def start(self) -> None:
        return None

    def read_deg(self) -> tuple[float, float]:
        raise RuntimeError("OpenNeck control is disabled")

    def tick(
        self,
        *,
        hmd_rotation_wxyz: FloatArray | None,
        spine3_rotation_wxyz: FloatArray | None,
        pose_timestamp_s: float | None,
        active: bool,
        now_s: float | None = None,
    ) -> None:
        del hmd_rotation_wxyz, spine3_rotation_wxyz, pose_timestamp_s, active, now_s
        return None

    def close(self) -> None:
        return None


def build_neck_runtime(cfg: Any | NeckConfig, device: NeckDevice | None = None) -> NeckRuntime | DisabledNeckRuntime:
    neck_cfg = cfg if isinstance(cfg, NeckConfig) else parse_neck_config(cfg)
    if not neck_cfg.enabled:
        return DisabledNeckRuntime()
    return NeckRuntime(neck_cfg, device=device)


def mode_packet_active(mode_packet: object | None, config: NeckConfig) -> bool:
    if mode_packet is None:
        return False
    mode = "pause" if bool(getattr(mode_packet, "mocap_paused", False)) else str(getattr(mode_packet, "mode", "")).strip().lower()
    return mode in config.active_modes


def head_pose_packet(
    packet: object | None,
) -> tuple[FloatArray | None, FloatArray | None, float | None, int]:
    if packet is None or not all(hasattr(packet, attr) for attr in ("snapshot", "timestamp_s", "seq")):
        return None, None, None, -1
    snapshot = getattr(packet, "snapshot")
    if snapshot is None or not all(
        hasattr(snapshot, attr)
        for attr in ("hmd_rotation_wxyz", "spine3_rotation_wxyz", "timestamp_s", "seq")
    ):
        return None, None, None, -1
    try:
        timestamp_s = float(getattr(packet, "timestamp_s"))
        seq = int(getattr(packet, "seq"))
        if int(getattr(snapshot, "seq")) != seq:
            return None, None, None, -1
        hmd_rotation = _optional_quat(getattr(snapshot, "hmd_rotation_wxyz"))
        spine3_rotation = _optional_quat(getattr(snapshot, "spine3_rotation_wxyz"))
        return hmd_rotation, spine3_rotation, timestamp_s, seq
    except (TypeError, ValueError):
        return None, None, None, -1


def _optional_quat(value: object | None) -> FloatArray | None:
    if value is None:
        return None
    try:
        return np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
