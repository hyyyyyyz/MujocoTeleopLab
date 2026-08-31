"""Pickle-serializable IPC message contracts for multiprocess sim2real."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from teleopit.inputs.realtime_packet import ControlEvent, HumanFrame
from teleopit.sim.reference_timeline import ReferenceWindow


Float64Array = NDArray[np.float64]
Float32Array = NDArray[np.float32]


@dataclass(frozen=True)
class BodyFramePacket:
    frame: HumanFrame
    timestamp_s: float
    seq: int


@dataclass(frozen=True)
class ReferencePacket:
    qpos: Float64Array
    timestamp_s: float
    seq: int
    source_timestamp_s: float
    source_seq: int
    frame_valid: bool = True
    reference_window: ReferenceWindow | None = None
    retarget_elapsed_s: float = 0.0
    playback_paused: bool = False
    playback_finished: bool = False


@dataclass(frozen=True)
class ControlEventsPacket:
    events: tuple[ControlEvent, ...]
    timestamp_s: float
    seq: int


@dataclass(frozen=True)
class SnapshotPacket:
    snapshot: Any
    timestamp_s: float
    seq: int


@dataclass(frozen=True)
class ModeStatePacket:
    mode: str
    mocap_active: bool
    mocap_paused: bool
    timestamp_s: float
    seq: int
    policy_paused: bool = False
    policy_session_id: str | None = None


@dataclass(frozen=True)
class RecordStepPacket:
    timestamp_s: float
    mode: str
    mocap_active: bool
    recordable: bool
    observation_state: Float64Array
    observation_mode: int
    action_reference_qpos: Float64Array
    seq: int


@dataclass(frozen=True)
class HandCommandPacket:
    timestamp_s: float
    driver: str
    mode: str
    active: bool
    left_pose: Float64Array
    right_pose: Float64Array
    seq: int
    left_state: Float64Array | None = None
    right_state: Float64Array | None = None


@dataclass(frozen=True)
class NeckCommandPacket:
    timestamp_s: float
    driver: str
    active: bool
    yaw_deg: float
    pitch_deg: float
    seq: int
    state_yaw_deg: float | None = None
    state_pitch_deg: float | None = None


@dataclass(frozen=True)
class HealthPacket:
    worker: str
    timestamp_s: float
    status: str = "ok"
    metrics: dict[str, float | int | str] = field(default_factory=dict)


@dataclass(frozen=True)
class CommandPacket:
    command: str
    timestamp_s: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SharedFrameDescriptor:
    shm_name: str
    slot: int
    seq: int
    timestamp_s: float
    shape: tuple[int, ...]
    dtype: str
    slots: int


@dataclass(frozen=True)
class HighLevelPolicySessionPacket:
    session_id: str
    task: str
    command: str
    timestamp_s: float
    seq: int


@dataclass(frozen=True)
class HighLevelPolicyObservationPacket:
    session_id: str
    sequence_id: int
    onboard_monotonic_timestamp_ns: int
    body_joint_positions: Float32Array
    dex_state: Float32Array
    neck_state: Float32Array
    source_reference_root_pose: Float32Array
    frame: SharedFrameDescriptor
    timestamp_s: float


@dataclass(frozen=True)
class HighLevelPolicyActionPacket:
    session_id: str
    source_sequence_id: int
    source_onboard_monotonic_timestamp_ns: int
    action_fps: int
    actions: Float32Array
    policy_id: str
    server_inference_ms: float
    received_timestamp_s: float


@dataclass(frozen=True)
class HighLevelPolicyStatusPacket:
    session_id: str | None
    status: str
    detail: str
    timestamp_s: float
    seq: int
    policy_type: str | None = None
    policy_id: str | None = None


@dataclass(frozen=True)
class HighLevelPolicyTargetPacket:
    session_id: str
    action: Float32Array
    timestamp_s: float
    seq: int
