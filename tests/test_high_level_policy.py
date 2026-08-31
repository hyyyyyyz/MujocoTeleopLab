from __future__ import annotations

import math
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import zmq

from teleopit.high_level_policy.client import HighLevelPolicyClient, PolicyActionChunk
from teleopit.high_level_policy.config import (
    HighLevelPolicySafetyConfig,
    parse_high_level_policy_config,
)
from teleopit.high_level_policy.hand_calibration import HandCalibration
from teleopit.high_level_policy.protocol import (
    MAX_ACTION_HORIZON,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    PolicyProtocolError,
    decode_float32_array,
    encode_float32_array,
    pack_message,
    unpack_message,
)
from teleopit.high_level_policy.scheduler import (
    HighLevelPolicyScheduler,
    PolicyFrameTransform,
    closure_to_o6_pose,
)
from teleopit.sim2real.mp.high_level_policy_runtime import (
    HighLevelPolicySim2RealRuntime,
    _apply_policy_neck_target,
    _policy_target_is_current,
    _stop_and_hardware_reset_realsense,
    _test_pattern,
    _validate_high_level_policy_runtime_config,
)
from teleopit.sim2real.mp.high_level_policy_worker import HighLevelPolicyWorker
from teleopit.sim2real.mp.messages import (
    HandCommandPacket,
    HighLevelPolicyActionPacket,
    HighLevelPolicyObservationPacket,
    HighLevelPolicySessionPacket,
    HighLevelPolicyStatusPacket,
    HighLevelPolicyTargetPacket,
    ModeStatePacket,
    NeckCommandPacket,
    SharedFrameDescriptor,
)
from teleopit.sim2real.mp.runtime import (
    RobotMode,
    Sim2RealRuntime,
    _RobotControlWorker,
)
from teleopit.runtime.mocap_session import MocapSessionState


def _chunk(*, source_s: float, sequence: int = 0, frames: int = 3) -> PolicyActionChunk:
    actions = np.zeros((frames, 50), dtype=np.float32)
    actions[:, 2] = 0.78
    actions[:, 3] = 1.0
    actions[:, 0] = np.arange(frames, dtype=np.float32)
    actions[:, 36:48] = 0.5
    actions[:, 48] = np.arange(frames, dtype=np.float32) * 10.0
    return PolicyActionChunk(
        session_id="session-1",
        source_sequence_id=sequence,
        source_onboard_monotonic_timestamp_ns=int(round(source_s * 1e9)),
        action_fps=30,
        actions=actions,
        policy_id="test",
        server_inference_ms=1.0,
    )


def _safe_actions(frames: int = 3) -> np.ndarray:
    actions = np.zeros((frames, 50), dtype=np.float32)
    actions[:, 0] = np.arange(frames, dtype=np.float32) * 0.02
    actions[:, 2] = 0.76
    actions[:, 3] = 1.0
    actions[:, 36:48] = 0.5
    return actions


def _safety_config() -> HighLevelPolicySafetyConfig:
    return HighLevelPolicySafetyConfig(
        root_height_min_m=0.55,
        root_height_max_m=1.05,
        max_root_xy_speed_m_s=2.5,
        max_root_displacement_m=0.1,
        max_yaw_rate_rad_s=2.5,
        max_joint_rate_rad_s=10.0,
        max_joint_projection_rad=0.1,
        joint_pos_lower=(-3.0,) * 29,
        joint_pos_upper=(3.0,) * 29,
        neck_yaw_min_deg=-45.0,
        neck_yaw_max_deg=45.0,
        neck_pitch_min_deg=-40.0,
        neck_pitch_max_deg=40.0,
    )


def _safe_chunk(actions: np.ndarray, *, source_s: float = 1.0, sequence: int = 0) -> PolicyActionChunk:
    return PolicyActionChunk(
        session_id="session-1",
        source_sequence_id=sequence,
        source_onboard_monotonic_timestamp_ns=int(round(source_s * 1e9)),
        action_fps=30,
        actions=actions,
        policy_id="test",
        server_inference_ms=1.0,
    )


def test_high_level_policy_default_hold_covers_inference_and_transport_jitter() -> None:
    config = parse_high_level_policy_config({"high_level_policy": {"task": "demo"}})

    assert config.hold_s == pytest.approx(3.0)


def test_high_level_policy_replan_steps_uses_protocol_horizon_limit() -> None:
    config = parse_high_level_policy_config(
        {
            "high_level_policy": {
                "task": "demo",
                "replan_steps": MAX_ACTION_HORIZON,
            }
        }
    )

    assert config.replan_steps == MAX_ACTION_HORIZON

    with pytest.raises(ValueError, match=rf"\[1, {MAX_ACTION_HORIZON}\]"):
        parse_high_level_policy_config(
            {
                "high_level_policy": {
                    "task": "demo",
                    "replan_steps": MAX_ACTION_HORIZON + 1,
                }
            }
        )


def test_packaged_hand_calibration_loads() -> None:
    calibration = HandCalibration.load()

    assert calibration.open_raw == (250.0, 250.0, 250.0, 250.0, 250.0, 250.0)
    assert calibration.close_raw == (86.0, 73.0, 118.0, 111.0, 110.0, 111.0)
    assert calibration.range_tolerance == pytest.approx(0.0001)


def test_msgpack_float32_array_roundtrip_is_little_endian() -> None:
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    message = {"array": encode_float32_array(values)}
    payload = pack_message(message, max_bytes=4096)
    decoded_message = unpack_message(payload, max_bytes=4096)
    decoded = decode_float32_array(
        decoded_message["array"],
        name="array",
        expected_shape=(3, 4),
    )

    assert decoded.dtype == np.dtype("float32")
    np.testing.assert_allclose(decoded, values.astype(np.float32))


def test_policy_frame_transform_localizes_and_delocalizes_action() -> None:
    yaw = math.pi / 2.0
    yaw_quaternion = np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=np.float32)
    transform = PolicyFrameTransform.from_robot_pose([2.0, 3.0], yaw_quaternion)

    body = np.zeros(36, dtype=np.float32)
    body[0] = 1.0
    body[2] = 0.78
    body[3] = 1.0
    world = transform.delocalize_body_action(body)
    np.testing.assert_allclose(world[:3], [2.0, 4.0, 0.78], atol=1e-6)
    np.testing.assert_allclose(world[3:7], yaw_quaternion, atol=1e-6)
    np.testing.assert_allclose(transform.localize_body_action(world), body, atol=1e-6)


def test_scheduler_uses_source_timestamp_and_interpolates_at_30hz() -> None:
    scheduler = HighLevelPolicyScheduler(hold_s=0.1)
    scheduler.reset("session-1")
    scheduler.accept(_chunk(source_s=10.0), now_s=10.01)

    halfway = scheduler.sample(10.0 + 0.5 / 30.0)
    assert halfway is not None
    assert halfway[0] == pytest.approx(0.5)
    assert halfway[48] == pytest.approx(5.0)


def test_scheduler_accepts_protocol_max_action_horizon() -> None:
    scheduler = HighLevelPolicyScheduler(hold_s=0.1)
    scheduler.reset("session-1")
    scheduler.accept(_chunk(source_s=10.0, frames=MAX_ACTION_HORIZON), now_s=10.01)

    assert scheduler.has_chunk


def test_scheduler_replaces_active_plan_using_new_source_timestamp() -> None:
    scheduler = HighLevelPolicyScheduler(hold_s=0.1)
    scheduler.reset("session-1")
    scheduler.accept(_chunk(source_s=20.0), now_s=20.0)

    replacement = _chunk(source_s=20.05, sequence=1)
    replacement.actions[:, 0] += 10.0
    scheduler.accept(replacement, now_s=20.06)

    scheduled = scheduler.sample(20.05 + 0.5 / 30.0)
    assert scheduled is not None
    assert scheduled[0] == pytest.approx(10.5)


def test_scheduler_pause_freezes_and_resume_shifts_plan_time() -> None:
    scheduler = HighLevelPolicyScheduler(hold_s=0.1)
    scheduler.reset("session-1")
    scheduler.accept(_chunk(source_s=20.0), now_s=20.0)
    scheduler.pause(20.02)

    paused = scheduler.sample(25.0)
    assert paused is not None
    scheduler.resume(25.0)
    resumed = scheduler.sample(25.0)
    assert resumed is not None
    np.testing.assert_allclose(resumed, paused)


def test_scheduler_interpolates_active_reference_history_at_camera_timestamp() -> None:
    scheduler = HighLevelPolicyScheduler(hold_s=0.1)
    initial = _safe_actions(1)[0]
    scheduler.reset("session-1", initial_action=initial)
    assert scheduler.reference_root_pose_at(1.0) is None
    scheduler.reset(
        "session-1",
        initial_reference=initial,
        initial_timestamp_s=1.0,
    )
    actions = _safe_actions(2)
    actions[1, 0] = 1.0
    yaw = math.pi / 2.0
    actions[1, 3:7] = [
        math.cos(yaw / 2.0),
        0.0,
        0.0,
        math.sin(yaw / 2.0),
    ]
    scheduler.accept(_safe_chunk(actions), now_s=1.0)
    scheduler.sample(1.0 + 1.0 / 30.0)

    source_pose = scheduler.reference_root_pose_at(1.0 + 0.5 / 30.0)

    assert source_pose is not None
    assert source_pose[0] == pytest.approx(0.5)
    source_yaw = 2.0 * math.atan2(float(source_pose[6]), float(source_pose[3]))
    assert source_yaw == pytest.approx(math.pi / 4.0)


def test_scheduler_rejects_wrong_session_and_expired_chunk() -> None:
    scheduler = HighLevelPolicyScheduler(hold_s=0.0)
    scheduler.reset("other")
    with pytest.raises(ValueError, match="session mismatch"):
        scheduler.accept(_chunk(source_s=1.0), now_s=1.0)

    scheduler.reset("session-1")
    with pytest.raises(ValueError, match="already expired"):
        scheduler.accept(_chunk(source_s=1.0), now_s=2.0)

    with pytest.raises(ValueError, match="in the future"):
        scheduler.accept(_chunk(source_s=3.0), now_s=2.0)


def test_scheduler_rejects_nonincreasing_source_timestamp() -> None:
    scheduler = HighLevelPolicyScheduler(hold_s=0.1)
    scheduler.reset("session-1")
    scheduler.accept(_chunk(source_s=1.0), now_s=1.0)

    with pytest.raises(ValueError, match="source timestamp must increase"):
        scheduler.accept(_chunk(source_s=1.0, sequence=1), now_s=1.01)


def test_linkerhand_closure_uses_hand_calibration() -> None:
    assert closure_to_o6_pose(np.zeros(6, dtype=np.float32)) == (250, 250, 250, 250, 250, 250)
    assert closure_to_o6_pose(np.ones(6, dtype=np.float32)) == (86, 73, 118, 111, 110, 111)


def test_scheduler_validates_complete_chunk_against_onboard_safety_limits() -> None:
    scheduler = HighLevelPolicyScheduler(hold_s=0.1, safety=_safety_config())
    initial = _safe_actions(1)[0]
    scheduler.reset("session-1", initial_action=initial)
    scheduler.accept(_safe_chunk(_safe_actions()), now_s=1.01)

    assert scheduler.has_chunk


def test_scheduler_accepts_internal_reference_discontinuities() -> None:
    scheduler = HighLevelPolicyScheduler(hold_s=0.1, safety=_safety_config())
    initial = _safe_actions(1)[0]
    initial[7] = 0.8
    actions = _safe_actions()
    actions[0, 0] = 0.2
    actions[0, 7] = -0.08
    actions[1, 0] = -0.2
    actions[1, 7] = 0.5
    yaw = 0.2
    actions[1, 3:7] = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
    scheduler.reset("session-1", initial_action=initial)

    scheduler.accept(_safe_chunk(actions), now_s=1.01)

    assert scheduler.has_chunk


def test_scheduler_clips_joint_positions_to_onboard_limits() -> None:
    scheduler = HighLevelPolicyScheduler(hold_s=0.1, safety=_safety_config())
    scheduler.reset(
        "session-1",
        initial_action=_safe_actions(1)[0],
    )
    actions = _safe_actions(1)
    actions[0, 7] = -3.08
    actions[0, 8] = 3.08

    scheduler.accept(_safe_chunk(actions), now_s=1.01)
    scheduled = None
    for _ in range(20):
        scheduled = scheduler.sample(1.0)

    assert scheduled is not None
    assert scheduled[7] == pytest.approx(-3.0)
    assert scheduled[8] == pytest.approx(3.0)


def test_scheduler_clips_openneck_angles_to_onboard_limits() -> None:
    scheduler = HighLevelPolicyScheduler(hold_s=0.1, safety=_safety_config())
    scheduler.reset(
        "session-1",
        initial_action=_safe_actions(1)[0],
    )
    actions = _safe_actions()
    actions[:, 48] = [-46.0, 0.0, 46.0]
    actions[:, 49] = [41.0, 0.0, -41.0]

    scheduler.accept(_safe_chunk(actions), now_s=1.01)
    first_action = scheduler.sample(1.0)
    final_action = scheduler.sample(1.0 + 2.0 / 30.0)

    assert first_action is not None
    assert first_action[48] == pytest.approx(-45.0)
    assert first_action[49] == pytest.approx(40.0)
    assert final_action is not None
    assert final_action[48] == pytest.approx(45.0)
    assert final_action[49] == pytest.approx(-40.0)


def test_scheduler_rejects_joint_projection_above_limit() -> None:
    scheduler = HighLevelPolicyScheduler(hold_s=0.1, safety=_safety_config())
    scheduler.reset(
        "session-1",
        initial_action=_safe_actions(1)[0],
    )
    actions = _safe_actions(1)
    actions[0, 7] = -3.11

    with pytest.raises(ValueError, match="joint projection correction exceeds"):
        scheduler.accept(_safe_chunk(actions), now_s=1.01)
    assert not scheduler.has_chunk


def test_scheduler_rejects_entire_unsafe_non_joint_chunk() -> None:
    scheduler = HighLevelPolicyScheduler(hold_s=0.1, safety=_safety_config())
    scheduler.reset(
        "session-1",
        initial_action=_safe_actions(1)[0],
    )
    actions = _safe_actions()
    actions[1, 2] = 0.4

    with pytest.raises(ValueError, match="root height"):
        scheduler.accept(_safe_chunk(actions), now_s=1.01)
    assert not scheduler.has_chunk


def test_scheduler_accepts_discontinuous_plan_and_rate_limits_output_at_50hz() -> None:
    scheduler = HighLevelPolicyScheduler(
        hold_s=0.1,
        safety=_safety_config(),
        output_hz=50.0,
    )
    initial = _safe_actions(1)[0]
    scheduler.reset("session-1", initial_action=initial)
    actions = _safe_actions(2)
    actions[1, 0] = 0.2
    actions[1, 7] = 0.5
    yaw = 0.2
    actions[1, 3:7] = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
    scheduler.accept(_safe_chunk(actions), now_s=1.01)

    output = scheduler.sample(1.0 + 1.0 / 30.0)

    assert output is not None
    assert output[0] == pytest.approx(2.5 / 50.0)
    assert output[7] == pytest.approx(10.0 / 50.0)
    output_yaw = 2.0 * math.atan2(float(output[6]), float(output[3]))
    assert output_yaw == pytest.approx(2.5 / 50.0, abs=1e-6)


def test_policy_client_roundtrip_matches_current_messages() -> None:
    context = zmq.Context()
    endpoint = "inproc://teleopit-policy-client-roundtrip"
    server = context.socket(zmq.REP)
    server.bind(endpoint)
    requests: list[dict[str, object]] = []

    def serve() -> None:
        for _ in range(3):
            request = unpack_message(server.recv(), max_bytes=MAX_REQUEST_BYTES)
            requests.append(request)
            name = request["endpoint"]
            if name == "describe":
                data = {
                    "observation_schema": "teleopit-g1-joint-pos-dex-neck-state",
                    "observation_dim": 43,
                    "action_schema": "teleopit-g1-reference",
                    "action_dim": 50,
                    "dataset_fps": 30,
                    "max_action_horizon": 3,
                    "policy_type": "replay",
                    "policy_id": "test-policy",
                    "ready": True,
                }
            elif name == "reset":
                data = {"session_id": "session-1", "reset": True}
            else:
                observation = request["data"]
                data = {
                    "session_id": "session-1",
                    "source_sequence_id": observation["sequence_id"],
                    "source_onboard_monotonic_timestamp_ns": observation[
                        "onboard_monotonic_timestamp_ns"
                    ],
                    "action_fps": 30,
                    "actions": encode_float32_array(_safe_actions()),
                    "policy_id": "test-policy",
                    "server_inference_ms": 1.0,
                }
            server.send(
                pack_message(
                    {
                        "endpoint": name,
                        "ok": True,
                        "data": data,
                    },
                    max_bytes=MAX_RESPONSE_BYTES,
                )
            )

    thread = threading.Thread(target=serve)
    thread.start()
    client = HighLevelPolicyClient(endpoint, timeout_s=0.2, context=context)
    try:
        description = client.describe()
        client.reset("session-1", "demo")
        body_joint_positions = np.linspace(-0.2, 0.2, 29, dtype=np.float32)
        dex_state = np.arange(12, dtype=np.float32) + 100.0
        neck_state = np.array([5.0, -7.0], dtype=np.float32)
        source_reference_root_pose = np.array(
            [1.0, 2.0, 0.76, 0.9995, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        chunk = client.get_action(
            session_id="session-1",
            sequence_id=4,
            onboard_monotonic_timestamp_ns=123,
            task="demo",
            jpeg_image=b"\xff\xd8test\xff\xd9",
            body_joint_positions=body_joint_positions,
            dex_state=dex_state,
            neck_state=neck_state,
            source_reference_root_pose=source_reference_root_pose,
        )
        assert description.policy_id == "test-policy"
        np.testing.assert_allclose(chunk.actions, _safe_actions())
        assert all(set(request) == {"endpoint", "data"} for request in requests)
        get_action_data = requests[2]["data"]
        assert isinstance(get_action_data, dict)
        assert set(get_action_data) == {
            "session_id",
            "sequence_id",
            "onboard_monotonic_timestamp_ns",
            "task",
            "image_encoding",
            "image",
            "body_joint_positions",
            "dex_state",
            "neck_state",
            "source_reference_root_pose",
        }
        np.testing.assert_array_equal(
            decode_float32_array(
                get_action_data["body_joint_positions"],
                name="body_joint_positions",
                expected_shape=(29,),
            ),
            body_joint_positions,
        )
        np.testing.assert_array_equal(
            decode_float32_array(
                get_action_data["dex_state"],
                name="dex_state",
                expected_shape=(12,),
            ),
            dex_state,
        )
        np.testing.assert_array_equal(
            decode_float32_array(
                get_action_data["neck_state"],
                name="neck_state",
                expected_shape=(2,),
            ),
            neck_state,
        )
        np.testing.assert_array_equal(
            decode_float32_array(
                get_action_data["source_reference_root_pose"],
                name="source_reference_root_pose",
                expected_shape=(7,),
            ),
            source_reference_root_pose,
        )
    finally:
        client.close()
        thread.join(timeout=1.0)
        server.close(linger=0)
        context.term()


def test_policy_client_rejects_extra_response_envelope_fields() -> None:
    client = object.__new__(HighLevelPolicyClient)

    with pytest.raises(PolicyProtocolError, match="exactly data"):
        client._parse_reply(
            {
                "endpoint": "describe",
                "ok": True,
                "data": {},
                "extra": "not allowed",
            },
            endpoint="describe",
        )


def test_high_level_policy_runtime_is_independent_from_pico_and_gmr() -> None:
    started: list[str] = []

    class FakeProcess:
        def __init__(self, *, name: str, target, args) -> None:  # type: ignore[no-untyped-def]
            del target, args
            self.name = name

        def start(self) -> None:
            started.append(self.name)

    runtime = object.__new__(HighLevelPolicySim2RealRuntime)
    runtime.cfg = {
        "hands": {"enabled": True},
        "neck": {"enabled": True, "driver": "openneck"},
    }
    runtime._ctx = SimpleNamespace(Process=FakeProcess)
    runtime._endpoints = SimpleNamespace()
    runtime._stop_event = SimpleNamespace()
    runtime._processes = []

    runtime._start_processes()

    assert started == ["camera", "high_level_policy", "robot_control", "policy_hand", "policy_neck"]
    assert all("pico" not in name and "reference" not in name and "retarget" not in name for name in started)


def test_high_level_policy_runtime_config_requires_safety_joint_limits() -> None:
    cfg = {
        "input": {"provider": "high_level_policy"},
        "camera": {"source": "test-pattern", "width": 640, "height": 480, "fps": 30},
        "high_level_policy": {"enabled": True, "task": "demo"},
        "reference_steps": [0],
        "recording": {"enabled": False},
        "hands": {"enabled": True, "driver": "linkerhand_o6", "sides": ["left", "right"]},
        "neck": {"enabled": True, "driver": "openneck"},
        "real_robot": {},
    }

    with pytest.raises(ValueError, match="joint_pos_lower"):
        _validate_high_level_policy_runtime_config(cfg)


def test_standard_pico_runtime_rejects_high_level_policy_flag() -> None:
    cfg = {
        "input": {"provider": "pico4"},
        "high_level_policy": {"enabled": True},
    }

    with pytest.raises(ValueError, match="independent.*run_high_level_policy_sim2real.py"):
        Sim2RealRuntime(cfg)


def test_high_level_policy_test_camera_is_exact_protocol_shape() -> None:
    frame = _test_pattern(480, 640, 7)
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8
    assert np.all(frame[:, :, 2] == 7)


def test_realsense_recovery_stops_pipeline_before_hardware_reset() -> None:
    calls: list[str] = []
    pipeline = SimpleNamespace(stop=lambda: calls.append("stop"))
    device = SimpleNamespace(hardware_reset=lambda: calls.append("hardware_reset"))

    _stop_and_hardware_reset_realsense(pipeline, device)

    assert calls == ["stop", "hardware_reset"]


def test_openneck_policy_target_is_sent_directly_in_physical_degrees() -> None:
    calls: list[tuple[float, float]] = []
    device = SimpleNamespace(move_deg=lambda yaw, pitch: calls.append((yaw, pitch)))
    action = _safe_actions(1)[0]
    action[48:50] = [12.5, -7.25]
    target = HighLevelPolicyTargetPacket(
        session_id="session-1",
        action=action,
        timestamp_s=1.0,
        seq=1,
    )

    _apply_policy_neck_target(device, target)

    assert calls == [(12.5, -7.25)]


def test_policy_hardware_target_requires_current_session_and_timestamp() -> None:
    action = _safe_actions(1)[0]
    target = HighLevelPolicyTargetPacket(
        session_id="session-1",
        action=action,
        timestamp_s=10.0,
        seq=5,
    )
    mode = ModeStatePacket(
        mode="policy",
        mocap_active=False,
        mocap_paused=False,
        timestamp_s=10.0,
        seq=1,
        policy_session_id="session-1",
    )

    assert _policy_target_is_current(
        target,
        mode,
        last_target_seq=4,
        max_age_s=0.2,
        now_s=10.1,
    )
    assert not _policy_target_is_current(
        target,
        mode,
        last_target_seq=5,
        max_age_s=0.2,
        now_s=10.1,
    )
    assert not _policy_target_is_current(
        HighLevelPolicyTargetPacket(
            session_id="old-session",
            action=action,
            timestamp_s=10.0,
            seq=6,
        ),
        mode,
        last_target_seq=4,
        max_age_s=0.2,
        now_s=10.1,
    )
    assert not _policy_target_is_current(
        target,
        mode,
        last_target_seq=4,
        max_age_s=0.2,
        now_s=10.3,
    )
    assert not _policy_target_is_current(
        target,
        ModeStatePacket(
            mode="policy",
            mocap_active=False,
            mocap_paused=False,
            timestamp_s=10.0,
            seq=2,
            policy_paused=True,
            policy_session_id="session-1",
        ),
        last_target_seq=4,
        max_age_s=0.2,
        now_s=10.1,
    )


def _remote(*, a: bool = False, b: bool = False, x: bool = False, y: bool = False):  # type: ignore[no-untyped-def]
    button = lambda pressed=False: SimpleNamespace(on_pressed=pressed, pressed=pressed)
    return SimpleNamespace(
        A=button(a),
        B=button(b),
        X=button(x),
        Y=button(y),
        start=button(False),
    )


def test_high_level_policy_y_requests_takeover_without_starting_mode_state() -> None:
    worker = object.__new__(_RobotControlWorker)
    worker.mode = RobotMode.STANDING
    worker.remote = _remote(y=True)
    worker._policy_entry_pending = False
    worker._policy_paused = False
    requests: list[str] = []

    def begin() -> None:
        requests.append("begin")
        worker._policy_entry_pending = True

    worker._begin_high_level_policy_entry = begin
    worker._publish_high_level_policy_session = lambda *_args, **_kwargs: None
    worker._policy_entry_deadline_s = None

    worker._handle_high_level_policy_transitions()

    assert requests == ["begin"]
    assert worker.mode == RobotMode.STANDING


def test_policy_transition_after_first_chunk_does_not_start_kp_ramp() -> None:
    worker = object.__new__(_RobotControlWorker)
    worker.mode = RobotMode.STANDING
    worker.robot = SimpleNamespace(get_state=lambda: SimpleNamespace())
    resume_qpos = np.zeros(36, dtype=np.float64)
    resume_qpos[3] = 1.0
    worker._build_robot_state_qpos = lambda _state: resume_qpos.copy()
    resets: list[str] = []
    worker._reset_policy_state = lambda: resets.append("reset")
    worker._last_retarget_qpos = np.ones(36, dtype=np.float64)
    worker._last_commanded_motion_qpos = None
    worker._policy_hold_qpos = None
    worker._policy_entry_pending = True
    worker._policy_entry_deadline_s = 2.0
    worker._policy_paused = True
    worker._policy_resume_pending = False
    worker._policy_resume_deadline_s = None
    worker._policy_resume_source_timestamp_ns = None
    worker._standing_return_ramp_duration = 0.5
    worker._standing_return_kp_ramp_floor_ratio = 0.5
    worker._safety = SimpleNamespace(
        start_kp_ramp=lambda **_kwargs: pytest.fail(
            "POLICY transition must not start an entry Kp ramp"
        )
    )

    worker._transition_to_high_level_policy()

    assert worker.mode == RobotMode.POLICY
    assert resets == ["reset"]
    assert not worker._policy_entry_pending
    assert not worker._policy_paused
    np.testing.assert_array_equal(worker._policy_hold_qpos, resume_qpos)


def test_policy_entry_rejects_action_received_after_deadline() -> None:
    worker = object.__new__(_RobotControlWorker)
    now_s = time.monotonic()
    worker.mode = RobotMode.STANDING
    worker._policy_video_sub = SimpleNamespace(recv_latest=lambda: None)
    worker._policy_status_sub = SimpleNamespace(recv_latest=lambda: None)
    worker._policy_action_sub = SimpleNamespace(
        recv_latest=lambda: HighLevelPolicyActionPacket(
            session_id="session-1",
            source_sequence_id=1,
            source_onboard_monotonic_timestamp_ns=int(round(now_s * 1e9)),
            action_fps=30,
            actions=_safe_actions(1),
            policy_id="test",
            server_inference_ms=1.0,
            received_timestamp_s=now_s,
        )
    )
    worker._last_policy_video_seq = -1
    worker._last_policy_status_seq = -1
    worker._policy_session_id = "session-1"
    worker._policy_paused = False
    worker._policy_resume_pending = False
    worker._policy_resume_source_timestamp_ns = None
    worker._policy_entry_pending = True
    worker._policy_entry_deadline_s = now_s - 0.01
    accepted: list[object] = []
    worker._high_level_policy_scheduler = SimpleNamespace(
        accept=lambda *args, **kwargs: accepted.append((args, kwargs))
    )
    worker._high_level_policy_cfg = SimpleNamespace(max_result_age_s=1.0)
    standing: list[str] = []
    worker._enter_standing = lambda: standing.append("standing")
    worker._transition_to_high_level_policy = lambda: pytest.fail(
        "expired entry action must not enter POLICY"
    )

    worker._drain_high_level_policy_ipc()

    assert standing == ["standing"]
    assert accepted == []


def test_policy_entry_first_chunk_uses_measured_reference_boundary() -> None:
    worker = object.__new__(_RobotControlWorker)
    now_s = time.monotonic()
    worker.mode = RobotMode.STANDING
    worker._policy_video_sub = SimpleNamespace(recv_latest=lambda: None)
    worker._policy_status_sub = SimpleNamespace(recv_latest=lambda: None)
    worker._policy_action_sub = SimpleNamespace(
        recv_latest=lambda: HighLevelPolicyActionPacket(
            session_id="session-1",
            source_sequence_id=1,
            source_onboard_monotonic_timestamp_ns=int(round(now_s * 1e9)),
            action_fps=30,
            actions=_safe_actions(1),
            policy_id="test",
            server_inference_ms=1.0,
            received_timestamp_s=now_s,
        )
    )
    worker._last_policy_video_seq = -1
    worker._last_policy_status_seq = -1
    worker._policy_session_id = "session-1"
    worker._policy_paused = False
    worker._policy_resume_pending = False
    worker._policy_resume_source_timestamp_ns = None
    worker._policy_entry_pending = True
    worker._policy_entry_deadline_s = now_s + 1.0
    worker._policy_frame_transform = PolicyFrameTransform.from_robot_pose(
        [0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    )
    current_qpos = np.zeros(36, dtype=np.float64)
    current_qpos[2] = 0.76
    current_qpos[3] = 1.0
    current_qpos[7] = 0.8
    state = SimpleNamespace()
    worker.robot = SimpleNamespace(get_state=lambda: state)
    worker._build_robot_state_qpos = lambda _state: current_qpos.copy()
    scheduler = HighLevelPolicyScheduler(hold_s=0.1, safety=_safety_config())
    boundary_action = worker._build_high_level_policy_boundary_action(state)
    scheduler.reset(
        "session-1",
        initial_action=boundary_action,
    )
    worker._high_level_policy_scheduler = scheduler
    worker._high_level_policy_cfg = SimpleNamespace(max_result_age_s=1.0)
    worker._enter_standing = lambda: pytest.fail(
        "valid first chunk must enter POLICY"
    )
    transitions: list[str] = []
    worker._transition_to_high_level_policy = lambda: transitions.append("policy")

    worker._drain_high_level_policy_ipc()

    scheduled = scheduler.sample(now_s)
    assert boundary_action[7] == pytest.approx(0.8)
    assert current_qpos[7] == pytest.approx(0.8)
    assert transitions == ["policy"]
    assert scheduler.has_chunk
    assert scheduled is not None
    assert scheduled[7] == pytest.approx(0.6)


def test_policy_session_seeds_source_history_from_active_reference() -> None:
    worker = object.__new__(_RobotControlWorker)
    state = SimpleNamespace(
        qpos=np.linspace(-0.2, 0.2, 29, dtype=np.float32),
        quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        base_pos=np.array([10.0, 20.0, 0.76], dtype=np.float64),
    )
    worker.robot = SimpleNamespace(get_state=lambda: state)
    worker.num_actions = 29
    worker._default_root_pos = np.array([0.0, 0.0, 0.76], dtype=np.float64)
    worker._high_level_policy_cfg = SimpleNamespace(entry_timeout_s=5.0)
    scheduler = HighLevelPolicyScheduler()
    worker._high_level_policy_scheduler = scheduler
    worker._policy_session_id = None
    worker._latest_policy_video = None
    worker._last_commanded_motion_qpos = np.zeros(36, dtype=np.float64)
    worker._last_commanded_motion_qpos[:7] = [
        12.0,
        24.0,
        0.82,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    worker._standing_qpos = np.zeros(36, dtype=np.float64)
    worker._standing_qpos[3] = 1.0
    worker._publish_high_level_policy_session = lambda *_args, **_kwargs: None

    worker._start_high_level_policy_entry_session()

    source_pose = scheduler.reference_root_pose_at(time.monotonic())
    assert source_pose is not None
    np.testing.assert_allclose(source_pose[:3], [2.0, 4.0, 0.82], atol=1e-6)
    np.testing.assert_array_equal(
        worker._policy_hold_qpos,
        worker._last_commanded_motion_qpos,
    )


def test_policy_observation_uses_active_reference_and_measured_hardware_state() -> None:
    worker = object.__new__(_RobotControlWorker)
    now_s = time.monotonic()
    frame = SharedFrameDescriptor(
        shm_name="camera",
        slot=0,
        seq=7,
        timestamp_s=now_s,
        shape=(480, 640, 3),
        dtype="uint8",
        slots=3,
    )
    worker._policy_entry_pending = True
    worker.mode = RobotMode.STANDING
    worker._policy_paused = False
    worker._policy_resume_pending = False
    worker._policy_session_id = "session-1"
    worker._high_level_policy_cfg = SimpleNamespace(max_observation_age_s=0.15)
    worker._latest_policy_video = frame
    worker._last_policy_video_seq = -1
    worker._policy_observation_seq = 0
    worker._policy_hand_state_sub = SimpleNamespace(recv_latest=lambda: None)
    worker._policy_neck_state_sub = SimpleNamespace(recv_latest=lambda: None)
    worker._latest_policy_hand_state = HandCommandPacket(
        timestamp_s=now_s,
        driver="linkerhand_o6",
        mode="policy",
        active=False,
        left_pose=np.full(6, 250.0, dtype=np.float32),
        right_pose=np.full(6, 250.0, dtype=np.float32),
        seq=1,
        left_state=np.arange(6, dtype=np.float32) + 10.0,
        right_state=np.arange(6, dtype=np.float32) + 20.0,
    )
    worker._latest_policy_neck_state = NeckCommandPacket(
        timestamp_s=now_s,
        driver="openneck",
        active=False,
        yaw_deg=0.0,
        pitch_deg=0.0,
        seq=1,
        state_yaw_deg=12.0,
        state_pitch_deg=-8.0,
    )
    scheduler = HighLevelPolicyScheduler()
    initial_action = _safe_actions(1)[0]
    active_reference = initial_action.copy()
    active_reference[:7] = [4.0, 5.0, 0.82, 1.0, 0.0, 0.0, 0.0]
    scheduler.reset(
        "session-1",
        initial_action=initial_action,
        initial_reference=active_reference,
        initial_timestamp_s=now_s,
    )
    worker._high_level_policy_scheduler = scheduler
    published: list[tuple[str, object]] = []
    worker._policy_control_pub = SimpleNamespace(
        publish=lambda topic, packet: published.append((topic, packet))
    )
    robot_state = SimpleNamespace(
        qpos=np.linspace(-0.2, 0.2, 29, dtype=np.float32),
        qvel=np.zeros(29, dtype=np.float32),
        quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        ang_vel=np.zeros(3, dtype=np.float32),
        base_pos=np.array([99.0, 98.0, 97.0], dtype=np.float32),
    )

    worker._publish_high_level_policy_observation(robot_state)

    assert len(published) == 1
    packet = published[0][1]
    assert isinstance(packet, HighLevelPolicyObservationPacket)
    np.testing.assert_array_equal(packet.body_joint_positions, robot_state.qpos)
    np.testing.assert_array_equal(
        packet.dex_state,
        np.concatenate(
            (
                worker._latest_policy_hand_state.left_state,
                worker._latest_policy_hand_state.right_state,
            )
        ),
    )
    np.testing.assert_array_equal(packet.neck_state, [12.0, -8.0])
    np.testing.assert_array_equal(
        packet.source_reference_root_pose,
        active_reference[:7],
    )
    assert packet.source_reference_root_pose[0] != robot_state.base_pos[0]


def test_policy_entry_stale_result_aborts_current_session() -> None:
    worker = object.__new__(_RobotControlWorker)
    now_s = time.monotonic()
    worker.mode = RobotMode.STANDING
    worker._policy_video_sub = SimpleNamespace(recv_latest=lambda: None)
    worker._policy_status_sub = SimpleNamespace(recv_latest=lambda: None)
    worker._policy_action_sub = SimpleNamespace(
        recv_latest=lambda: HighLevelPolicyActionPacket(
            session_id="session-1",
            source_sequence_id=1,
            source_onboard_monotonic_timestamp_ns=int(round(now_s * 1e9)),
            action_fps=30,
            actions=_safe_actions(1),
            policy_id="test",
            server_inference_ms=1.0,
            received_timestamp_s=now_s - 1.0,
        )
    )
    worker._last_policy_video_seq = -1
    worker._last_policy_status_seq = -1
    worker._policy_session_id = "session-1"
    worker._policy_paused = False
    worker._policy_resume_pending = False
    worker._policy_resume_source_timestamp_ns = None
    worker._policy_entry_pending = True
    worker._policy_entry_deadline_s = now_s + 1.0
    worker._high_level_policy_scheduler = SimpleNamespace()
    worker._high_level_policy_cfg = SimpleNamespace(max_result_age_s=0.1)
    standing: list[str] = []
    worker._enter_standing = lambda: standing.append("standing")

    worker._drain_high_level_policy_ipc()

    assert standing == ["standing"]


def test_policy_entry_cancel_from_standing_only_stops_session() -> None:
    worker = object.__new__(_RobotControlWorker)
    worker.high_level_policy_enabled = True
    worker.mode = RobotMode.STANDING
    worker._policy_entry_pending = True
    worker._mocap_entry_requested = False
    stops: list[str] = []

    def stop_session() -> None:
        stops.append("stop")
        worker._policy_entry_pending = False

    worker._stop_high_level_policy_session = stop_session
    worker._disarm_mocap_reference_if_needed = lambda: None
    worker._clear_reference_gate = lambda: None
    worker.robot = SimpleNamespace(
        get_state=lambda: pytest.fail("entry cancel must not rebuild STANDING"),
        lock_all_joints=lambda: pytest.fail(
            "entry cancel must not lock joints or block the control loop"
        ),
    )

    worker._enter_standing()

    assert stops == ["stop"]
    assert worker.mode == RobotMode.STANDING


def test_high_level_policy_body_action_uses_existing_tracker_without_second_alignment() -> None:
    worker = object.__new__(_RobotControlWorker)
    action = _safe_actions(1)[0]
    worker._high_level_policy_scheduler = SimpleNamespace(sample=lambda _now: action.copy())
    worker._policy_frame_transform = SimpleNamespace(
        delocalize_body_action=lambda body: np.asarray(body, dtype=np.float32) + np.float32(1.0)
    )
    worker._policy_session_id = "session-1"
    worker._policy_paused = False
    worker._policy_resume_pending = False
    worker.robot = SimpleNamespace(get_state=lambda: SimpleNamespace())
    worker._publish_high_level_policy_observation = lambda _state: None
    worker._policy_control_pub = None
    worker._policy_hold_qpos = None
    calls: list[tuple[np.ndarray, dict[str, object]]] = []

    def execute(reference, _state, **kwargs) -> None:  # type: ignore[no-untyped-def]
        calls.append((np.asarray(reference), kwargs))

    worker._execute_reference_pipeline = execute

    worker._high_level_policy_step()

    assert len(calls) == 1
    np.testing.assert_allclose(calls[0][0], action[:36] + 1.0)
    assert calls[0][1] == {
        "reference_window": None,
        "align_reference": False,
        "compose_arms": False,
    }


def test_policy_and_pico_remote_b_both_toggle_pause() -> None:
    policy_worker = object.__new__(_RobotControlWorker)
    policy_worker.mode = RobotMode.POLICY
    policy_worker.remote = _remote(b=True)
    policy_toggles: list[str] = []
    policy_worker._toggle_high_level_policy_pause = lambda: policy_toggles.append("policy")
    policy_worker._handle_high_level_policy_transitions()

    pico_worker = object.__new__(_RobotControlWorker)
    pico_worker.mode = RobotMode.MOCAP
    pico_worker.provider_kind = "pico4"
    pico_worker.remote = _remote(b=True)
    pico_worker._mocap_session = SimpleNamespace(state=MocapSessionState.ACTIVE)
    pico_commands: list[str] = []
    pico_worker._send_reference_command = pico_commands.append
    pico_worker._pause_active_mocap = lambda: pico_commands.append("paused")
    pico_worker._handle_transitions()

    assert policy_toggles == ["policy"]
    assert pico_commands == ["pause_mocap", "paused"]


def test_policy_worker_pause_resume_retransmission_is_idempotent() -> None:
    worker = object.__new__(HighLevelPolicyWorker)
    worker._last_session_seq = -1
    worker._active_session = None
    worker._ready = False
    worker._paused = False
    worker._last_observation_seq = -1
    worker._last_request_timestamp_ns = None
    worker._next_connect_time_s = 0.0
    worker._new_session_required = False
    statuses: list[str] = []
    worker._publish_status = lambda status, _detail: statuses.append(status)

    def packet(command: str, seq: int) -> HighLevelPolicySessionPacket:
        return HighLevelPolicySessionPacket(
            session_id="session-1",
            task="demo",
            command=command,
            timestamp_s=1.0,
            seq=seq,
        )

    worker._handle_session(packet("start", 1))
    worker._ready = True
    worker._handle_session(packet("pause", 2))
    worker._handle_session(packet("pause", 3))
    worker._handle_session(packet("resume", 4))
    worker._handle_session(packet("resume", 5))

    assert statuses == ["connecting", "paused", "ready"]


def test_policy_worker_resume_reconnects_faulted_current_session() -> None:
    worker = object.__new__(HighLevelPolicyWorker)
    worker._last_session_seq = -1
    worker._active_session = None
    worker._ready = False
    worker._paused = False
    worker._last_observation_seq = -1
    worker._last_request_timestamp_ns = None
    worker._next_connect_time_s = 0.0
    worker._new_session_required = False
    statuses: list[str] = []
    worker._publish_status = lambda status, _detail: statuses.append(status)

    def packet(command: str, seq: int) -> HighLevelPolicySessionPacket:
        return HighLevelPolicySessionPacket(
            session_id="session-1",
            task="demo",
            command=command,
            timestamp_s=1.0,
            seq=seq,
        )

    worker._handle_session(packet("start", 1))
    worker._new_session_required = True
    worker._handle_session(packet("pause", 2))
    worker._handle_session(packet("resume", 3))

    assert statuses == ["connecting", "paused", "connecting"]
    assert not worker._paused
    assert not worker._new_session_required


def test_policy_worker_replans_on_configured_source_frame_stride(monkeypatch) -> None:
    worker = object.__new__(HighLevelPolicyWorker)
    worker._active_session = HighLevelPolicySessionPacket(
        session_id="session-1",
        task="demo",
        command="start",
        timestamp_s=time.monotonic(),
        seq=1,
    )
    worker._ready = True
    worker._paused = False
    worker._last_observation_seq = -1
    worker._last_request_timestamp_ns = 1_000_000_000
    worker.policy_cfg = SimpleNamespace(
        replan_steps=3,
        max_observation_age_s=0.15,
        jpeg_quality=90,
    )
    requests: list[dict[str, object]] = []

    def get_action(**kwargs):  # type: ignore[no-untyped-def]
        requests.append(kwargs)
        return _chunk(
            source_s=int(kwargs["onboard_monotonic_timestamp_ns"]) * 1e-9,
            sequence=int(kwargs["sequence_id"]),
        )

    worker._client = SimpleNamespace(get_action=get_action)
    worker._policy_id = "test"
    worker._frame_reader = SimpleNamespace(
        read=lambda _descriptor, copy: np.zeros((480, 640, 3), dtype=np.uint8)
    )
    published: list[object] = []
    worker._result_pub = SimpleNamespace(
        publish=lambda _topic, packet: published.append(packet)
    )
    monkeypatch.setattr(
        "teleopit.sim2real.mp.high_level_policy_worker.encode_policy_jpeg",
        lambda _frame, quality: b"jpeg",
    )

    def observation(sequence_id: int, timestamp_ns: int) -> HighLevelPolicyObservationPacket:
        return HighLevelPolicyObservationPacket(
            session_id="session-1",
            sequence_id=sequence_id,
            onboard_monotonic_timestamp_ns=timestamp_ns,
            body_joint_positions=np.arange(29, dtype=np.float32),
            dex_state=np.arange(12, dtype=np.float32) + 100.0,
            neck_state=np.array([5.0, -2.0], dtype=np.float32),
            source_reference_root_pose=np.array(
                [1.0, 2.0, 0.76, 1.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            frame=object(),  # type: ignore[arg-type]
            timestamp_s=time.monotonic(),
        )

    worker._handle_observation(observation(1, 1_099_999_999))
    worker._handle_observation(observation(2, 1_100_000_000))

    assert len(requests) == 1
    assert requests[0]["sequence_id"] == 2
    np.testing.assert_array_equal(
        requests[0]["body_joint_positions"],
        np.arange(29, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        requests[0]["dex_state"],
        np.arange(12, dtype=np.float32) + 100.0,
    )
    np.testing.assert_array_equal(
        requests[0]["neck_state"],
        np.array([5.0, -2.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        requests[0]["source_reference_root_pose"],
        np.array([1.0, 2.0, 0.76, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert worker._last_request_timestamp_ns == 1_100_000_000
    assert len(published) == 1
    assert published[0].source_onboard_monotonic_timestamp_ns == 1_100_000_000


def test_policy_fault_uses_normal_pause_without_entering_standing() -> None:
    worker = object.__new__(_RobotControlWorker)
    worker.high_level_policy_enabled = True
    worker.mode = RobotMode.POLICY
    worker._policy_paused = False
    worker._policy_resume_pending = False
    worker._policy_resume_deadline_s = None
    worker._policy_resume_source_timestamp_ns = None
    paused: list[float] = []
    worker._high_level_policy_scheduler = SimpleNamespace(
        pause=lambda now_s: paused.append(float(now_s))
    )
    hold_qpos = np.arange(36, dtype=np.float64)
    worker._resolve_mocap_hold_qpos = lambda: hold_qpos.copy()
    published: list[str] = []
    worker._publish_high_level_policy_session = published.append
    worker._enter_standing = lambda: pytest.fail("fault must not enter STANDING")

    worker._handle_high_level_policy_fault("network timeout")

    assert worker.mode == RobotMode.POLICY
    assert worker._policy_paused
    assert not worker._policy_resume_pending
    assert len(paused) == 1
    assert published == ["pause"]
    np.testing.assert_array_equal(worker._policy_hold_qpos, hold_qpos)


def test_policy_watchdog_pauses_and_holds_last_reference() -> None:
    worker = object.__new__(_RobotControlWorker)
    worker.high_level_policy_enabled = True
    worker.mode = RobotMode.POLICY
    worker._policy_paused = False
    worker._policy_resume_pending = False
    worker._policy_resume_deadline_s = None
    worker._policy_resume_source_timestamp_ns = None
    paused: list[float] = []
    worker._high_level_policy_scheduler = SimpleNamespace(
        sample=lambda _now_s: None,
        pause=lambda now_s: paused.append(float(now_s)),
    )
    worker._policy_frame_transform = SimpleNamespace()
    worker._policy_session_id = "session-1"
    worker.robot = SimpleNamespace(get_state=lambda: SimpleNamespace())
    worker._publish_high_level_policy_observation = lambda _state: None
    hold_qpos = np.arange(36, dtype=np.float64)
    worker._last_commanded_motion_qpos = hold_qpos.copy()
    worker._last_retarget_qpos = None
    worker._publish_high_level_policy_session = lambda _command: None
    held: list[np.ndarray] = []
    worker._run_static_mocap_step = lambda qpos: held.append(np.asarray(qpos).copy())

    worker._high_level_policy_step()

    assert worker.mode == RobotMode.POLICY
    assert worker._policy_paused
    assert not worker._policy_resume_pending
    assert len(paused) == 1
    assert len(held) == 1
    np.testing.assert_array_equal(held[0], hold_qpos)


def test_current_policy_fault_status_is_handled_even_while_paused() -> None:
    worker = object.__new__(_RobotControlWorker)
    worker._policy_video_sub = SimpleNamespace(recv_latest=lambda: None)
    worker._policy_action_sub = SimpleNamespace(recv_latest=lambda: None)
    worker._policy_status_sub = SimpleNamespace(
        recv_latest=lambda: HighLevelPolicyStatusPacket(
            session_id="session-1",
            status="fault",
            detail="network timeout",
            timestamp_s=1.0,
            seq=1,
        )
    )
    worker._last_policy_video_seq = -1
    worker._last_policy_status_seq = -1
    worker._policy_session_id = "session-1"
    worker._policy_paused = True
    worker.mode = RobotMode.POLICY
    handled: list[str] = []
    worker._handle_high_level_policy_fault = handled.append

    worker._drain_high_level_policy_ipc()

    assert handled == ["network timeout"]


def test_policy_pause_resume_waits_for_fresh_chunk_then_resumes() -> None:
    worker = object.__new__(_RobotControlWorker)
    worker.mode = RobotMode.POLICY
    worker._policy_session_id = "session-1"
    worker._policy_paused = True
    worker._policy_resume_pending = False
    worker._policy_resume_deadline_s = None
    worker._policy_resume_source_timestamp_ns = None
    resumed: list[float] = []
    accepted: list[object] = []
    worker._high_level_policy_scheduler = SimpleNamespace(
        accept=lambda *args, **kwargs: accepted.append((args, kwargs)),
        resume=lambda now_s: resumed.append(float(now_s)),
    )
    worker._high_level_policy_cfg = SimpleNamespace(
        entry_timeout_s=1.0,
        max_result_age_s=0.1,
    )
    session_commands: list[str] = []
    worker._publish_high_level_policy_session = session_commands.append

    worker._toggle_high_level_policy_pause()

    assert worker._policy_paused
    assert worker._policy_resume_pending
    assert session_commands == ["resume"]

    source_timestamp_ns = worker._policy_resume_source_timestamp_ns
    assert source_timestamp_ns is not None
    worker._policy_video_sub = SimpleNamespace(recv_latest=lambda: None)
    worker._policy_status_sub = SimpleNamespace(recv_latest=lambda: None)
    worker._policy_action_sub = SimpleNamespace(
        recv_latest=lambda: HighLevelPolicyActionPacket(
            session_id="session-1",
            source_sequence_id=1,
            source_onboard_monotonic_timestamp_ns=source_timestamp_ns,
            action_fps=30,
            actions=_safe_actions(1),
            policy_id="test",
            server_inference_ms=1.0,
            received_timestamp_s=time.monotonic(),
        )
    )
    worker._last_policy_video_seq = -1
    worker._last_policy_status_seq = -1

    worker._drain_high_level_policy_ipc()

    assert len(accepted) == 1
    assert len(resumed) == 1
    assert not worker._policy_paused
    assert not worker._policy_resume_pending


def test_policy_resume_rejects_action_received_after_deadline() -> None:
    worker = object.__new__(_RobotControlWorker)
    now_s = time.monotonic()
    worker.mode = RobotMode.POLICY
    worker._policy_video_sub = SimpleNamespace(recv_latest=lambda: None)
    worker._policy_status_sub = SimpleNamespace(recv_latest=lambda: None)
    worker._policy_action_sub = SimpleNamespace(
        recv_latest=lambda: HighLevelPolicyActionPacket(
            session_id="session-1",
            source_sequence_id=1,
            source_onboard_monotonic_timestamp_ns=int(round(now_s * 1e9)),
            action_fps=30,
            actions=_safe_actions(1),
            policy_id="test",
            server_inference_ms=1.0,
            received_timestamp_s=now_s,
        )
    )
    worker._last_policy_video_seq = -1
    worker._last_policy_status_seq = -1
    worker._policy_session_id = "session-1"
    worker._policy_paused = True
    worker._policy_resume_pending = True
    worker._policy_resume_deadline_s = now_s - 0.01
    worker._policy_resume_source_timestamp_ns = int(round((now_s - 0.1) * 1e9))
    accepted: list[object] = []
    worker._high_level_policy_scheduler = SimpleNamespace(
        accept=lambda *args, **kwargs: accepted.append((args, kwargs)),
        resume=lambda _now_s: pytest.fail("expired resume must not resume scheduler"),
    )
    worker._high_level_policy_cfg = SimpleNamespace(max_result_age_s=1.0)
    faults: list[str] = []
    worker._handle_high_level_policy_fault = faults.append

    worker._drain_high_level_policy_ipc()

    assert faults == ["resume timed out waiting for a fresh action chunk"]
    assert accepted == []


def test_paused_robot_worker_discards_inflight_policy_result() -> None:
    worker = object.__new__(_RobotControlWorker)
    worker._policy_video_sub = SimpleNamespace(recv_latest=lambda: None)
    worker._policy_status_sub = SimpleNamespace(recv_latest=lambda: None)
    action = _safe_actions(1)
    worker._policy_action_sub = SimpleNamespace(
        recv_latest=lambda: HighLevelPolicyActionPacket(
            session_id="session-1",
            source_sequence_id=1,
            source_onboard_monotonic_timestamp_ns=1,
            action_fps=30,
            actions=action,
            policy_id="test",
            server_inference_ms=1.0,
            received_timestamp_s=1.0,
        )
    )
    worker._last_policy_video_seq = -1
    worker._last_policy_status_seq = -1
    worker._policy_session_id = "session-1"
    worker._policy_paused = True
    worker._policy_resume_pending = False
    accepted: list[object] = []
    worker._high_level_policy_scheduler = SimpleNamespace(
        accept=lambda *args, **kwargs: accepted.append((args, kwargs))
    )
    worker._high_level_policy_cfg = SimpleNamespace(max_result_age_s=0.1)

    worker._drain_high_level_policy_ipc()

    assert accepted == []
