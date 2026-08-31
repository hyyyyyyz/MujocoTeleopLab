from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from teleopit.inputs.pico4_provider import PicoControllerSnapshot, PicoControllerState
from teleopit.sim2real.hands.linkerhand_l6 import (
    GripperMapper,
    LinkerHandL6Device,
    RetargetPoseMapper,
    SomehandL6Mapper,
    parse_linkerhand_l6_config,
    trigger_to_pose,
)
from teleopit.sim2real.hands.base import HandPoseCommand
from teleopit.sim2real.hands.linkerhand_o6 import (
    CLOSE_POSE as O6_CLOSE_POSE,
    DEFAULT_SOMEHAND_CONFIG as O6_DEFAULT_SOMEHAND_CONFIG,
    LinkerHandO6Device,
    O6_SDK_JOINT_ORDER,
    SomehandO6Mapper,
    parse_linkerhand_o6_config,
)
from teleopit.sim2real.hands.pico_landmarks import pico_hand_to_landmarks
from teleopit.sim2real.hands.worker import HandRuntime


class FakeInnerHand:
    def __init__(self) -> None:
        self.close_calls = 0
        self.close_can_interface_calls = 0

    def close(self) -> None:
        self.close_calls += 1

    def close_can_interface(self) -> None:
        self.close_can_interface_calls += 1


class FakeLinkerHandApi:
    instances: list["FakeLinkerHandApi"] = []

    def __init__(self, *, hand_joint: str, hand_type: str, modbus: str, can: str) -> None:
        self.hand_joint = hand_joint
        self.hand_type = hand_type
        self.modbus = modbus
        self.can = can
        self.hand = FakeInnerHand()
        self.speed: list[int] | None = None
        self.poses: list[list[int]] = []
        self.state = [1, 2, 3, 4, 5, 6] if hand_type == "left" else [11, 12, 13, 14, 15, 16]
        self.close_can_calls = 0
        FakeLinkerHandApi.instances.append(self)

    def set_speed(self, speed: list[int]) -> None:
        self.speed = list(speed)

    def finger_move(self, pose: list[int]) -> None:
        self.poses.append(list(pose))

    def get_state(self) -> list[int]:
        return list(self.state)

    def close_can(self) -> None:
        self.close_can_calls += 1


def _cfg(mode: str = "gripper") -> dict[str, object]:
    return {
        "input": {"provider": "pico4"},
        "hands": {
            "enabled": True,
            "driver": "linkerhand_l6",
            "mode": mode,
            "sides": ["left", "right"],
            "rate_hz": 30.0,
            "frame_timeout_s": 0.3,
            "linkerhand_l6": {
                "left_can": "can0",
                "right_can": "can1",
                "modbus": "None",
                "trigger_deadzone": 0.05,
                "deadman_threshold": 0.5,
                "open_pose": [250, 10, 250, 250, 250, 250],
                "close_pose": [79, 10, 0, 0, 0, 0],
            },
            "somehand": {
                "rate_hz": 60.0,
                "max_iterations": 12,
                "temporal_filter_alpha": 1.0,
                "output_alpha": 1.0,
            },
        },
    }


def _o6_cfg(mode: str = "gripper") -> dict[str, object]:
    return {
        "input": {"provider": "pico4"},
        "hands": {
            "enabled": True,
            "driver": "linkerhand_o6",
            "mode": mode,
            "sides": ["left", "right"],
            "rate_hz": 30.0,
            "frame_timeout_s": 0.3,
            "linkerhand_o6": {
                "left_can": "can0",
                "right_can": "can1",
                "modbus": "None",
                "trigger_deadzone": 0.05,
                "deadman_threshold": 0.5,
            },
            "somehand": {
                "rate_hz": 60.0,
                "max_iterations": 12,
                "temporal_filter_alpha": 1.0,
                "output_alpha": 1.0,
            },
        },
    }


def test_pico_hand_to_landmarks_uses_teleopit_adapter() -> None:
    joints = np.zeros((26, 7), dtype=np.float64)
    joints[:, 0] = np.arange(26)
    joints[:, 1] = np.arange(26) + 100
    joints[:, 2] = np.arange(26) + 200

    landmarks = pico_hand_to_landmarks(joints)

    assert landmarks.shape == (21, 3)
    np.testing.assert_allclose(landmarks[0], [1.0, -201.0, 101.0])
    np.testing.assert_allclose(landmarks[-1], [25.0, -225.0, 125.0])


def test_gripper_mapper_maps_trigger_and_deadman() -> None:
    cfg = parse_linkerhand_l6_config(_cfg())
    mapper = GripperMapper(cfg)
    snapshot = PicoControllerSnapshot(
        left=PicoControllerState(raw=True, grip=1.0, trigger=1.0, present=True),
        right=PicoControllerState(raw=True, grip=0.1, trigger=1.0, present=True),
        timestamp_s=10.0,
        seq=1,
    )

    commands = mapper.map(controller_snapshot=snapshot, hand_snapshot=None, active=True, now_s=10.0)

    assert commands[0].side == "left"
    assert commands[0].pose == cfg.close_pose
    assert commands[1].side == "right"
    assert commands[1].pose == cfg.open_pose


def test_hand_mappers_force_open_once_when_inactive() -> None:
    cfg = parse_linkerhand_l6_config(_cfg())
    snapshot = PicoControllerSnapshot(
        left=PicoControllerState(raw=True, grip=1.0, trigger=1.0, present=True),
        right=PicoControllerState(raw=True, grip=1.0, trigger=1.0, present=True),
        timestamp_s=10.0,
        seq=1,
    )

    gripper = GripperMapper(cfg)
    assert gripper.map(controller_snapshot=None, hand_snapshot=None, active=False, now_s=9.0) == ()
    assert gripper.map(controller_snapshot=snapshot, hand_snapshot=None, active=True, now_s=10.0)
    first_inactive = gripper.map(controller_snapshot=snapshot, hand_snapshot=None, active=False, now_s=10.1)
    assert [command.force for command in first_inactive] == [True, True]
    assert gripper.map(controller_snapshot=snapshot, hand_snapshot=None, active=False, now_s=10.2) == ()

    somehand = SomehandL6Mapper(cfg)
    assert somehand.map(controller_snapshot=None, hand_snapshot=None, active=False, now_s=9.0) == ()
    somehand._active = True
    first_inactive = somehand.map(controller_snapshot=None, hand_snapshot=None, active=False, now_s=10.0)
    assert [command.force for command in first_inactive] == [True, True]
    assert somehand.map(controller_snapshot=None, hand_snapshot=None, active=False, now_s=10.1) == ()


def test_trigger_to_pose_applies_deadzone_and_fixed_thumb_yaw() -> None:
    assert trigger_to_pose(
        0.5,
        open_pose=[250, 10, 250, 250, 250, 250],
        close_pose=[79, 10, 0, 0, 0, 0],
        deadzone=0.05,
        thumb_yaw_default=10,
    ) == [164, 10, 125, 125, 125, 125]


def test_trigger_to_pose_can_interpolate_thumb_yaw_for_o6() -> None:
    assert trigger_to_pose(
        1.0,
        open_pose=[250, 250, 250, 250, 250, 250],
        close_pose=[86, 73, 118, 111, 110, 111],
        deadzone=0.05,
    ) == list(O6_CLOSE_POSE)


def test_linkerhand_l6_device_starts_sdk(monkeypatch) -> None:
    FakeLinkerHandApi.instances = []
    monkeypatch.setitem(
        sys.modules,
        "LinkerHand.linker_hand_api",
        SimpleNamespace(LinkerHandApi=FakeLinkerHandApi),
    )
    cfg = parse_linkerhand_l6_config(_cfg())
    device = LinkerHandL6Device(cfg)

    device.connect()
    device.send_pose("left", cfg.close_pose)
    state = device.get_state("left")
    device.close()

    assert state == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert [hand.can for hand in FakeLinkerHandApi.instances] == ["can0", "can1"]
    assert FakeLinkerHandApi.instances[0].speed == [50, 50, 50, 50, 50, 50]
    assert FakeLinkerHandApi.instances[0].poses[-2] == list(cfg.close_pose)
    assert [hand.hand.close_calls for hand in FakeLinkerHandApi.instances] == [1, 1]


def test_linkerhand_o6_gripper_defaults_to_reference_grasp_pose() -> None:
    cfg = parse_linkerhand_o6_config(_o6_cfg())
    mapper = GripperMapper(cfg)
    snapshot = PicoControllerSnapshot(
        left=PicoControllerState(raw=True, grip=1.0, trigger=1.0, present=True),
        right=PicoControllerState(raw=True, grip=0.1, trigger=1.0, present=True),
        timestamp_s=10.0,
        seq=1,
    )

    commands = mapper.map(controller_snapshot=snapshot, hand_snapshot=None, active=True, now_s=10.0)

    assert cfg.close_pose == O6_CLOSE_POSE
    assert commands[0].pose == O6_CLOSE_POSE
    assert commands[1].pose == cfg.open_pose


def test_linkerhand_o6_device_starts_sdk(monkeypatch) -> None:
    FakeLinkerHandApi.instances = []
    monkeypatch.setitem(
        sys.modules,
        "LinkerHand.linker_hand_api",
        SimpleNamespace(LinkerHandApi=FakeLinkerHandApi),
    )
    cfg = parse_linkerhand_o6_config(_o6_cfg())
    device = LinkerHandO6Device(cfg)

    device.connect()
    device.send_pose("left", cfg.close_pose)
    state = device.get_state("right")
    device.close()

    assert state == (11.0, 12.0, 13.0, 14.0, 15.0, 16.0)
    assert [hand.hand_joint for hand in FakeLinkerHandApi.instances] == ["O6", "O6"]
    assert [hand.can for hand in FakeLinkerHandApi.instances] == ["can0", "can1"]
    assert FakeLinkerHandApi.instances[0].speed == [255, 255, 255, 255, 255, 255]
    assert FakeLinkerHandApi.instances[0].poses[-2] == list(O6_CLOSE_POSE)
    assert [hand.close_can_calls for hand in FakeLinkerHandApi.instances] == [0, 0]
    assert [hand.hand.close_can_interface_calls for hand in FakeLinkerHandApi.instances] == [1, 1]
    assert [hand.hand.close_calls for hand in FakeLinkerHandApi.instances] == [0, 0]


def test_linkerhand_o6_accepts_vr_hand_pose() -> None:
    cfg = parse_linkerhand_o6_config(_o6_cfg(mode="vr_hand_pose"))

    assert cfg.mode == "vr_hand_pose"
    assert cfg.speed == (255, 255, 255, 255, 255, 255)
    assert cfg.somehand_config_path == O6_DEFAULT_SOMEHAND_CONFIG

    mapper = SomehandO6Mapper(cfg)
    assert mapper.map(controller_snapshot=None, hand_snapshot=None, active=False, now_s=10.0) == ()
    mapper._active = True
    first_inactive = mapper.map(controller_snapshot=None, hand_snapshot=None, active=False, now_s=10.1)
    assert [command.force for command in first_inactive] == [True, True]


def test_somehand_mapper_loads_only_configured_side(monkeypatch) -> None:
    class FakeHandModel:
        def get_joint_name_to_qpos_index(self) -> dict[str, int]:
            return {
                "lh_thumb_cmc_pitch": 0,
                "lh_thumb_cmc_roll": 1,
                "lh_index_mcp_pitch": 2,
                "lh_middle_mcp_pitch": 3,
                "lh_ring_mcp_pitch": 4,
                "lh_pinky_mcp_pitch": 5,
            }

    class FakeRetargetingEngine:
        def __init__(self, cfg: object) -> None:
            self.cfg = cfg
            self.hand_model = FakeHandModel()

    loaded_paths: list[str] = []
    somehand_api = ModuleType("somehand.api")
    somehand_api.HandFrame = object
    somehand_api.RetargetingEngine = FakeRetargetingEngine
    somehand_api.load_bihand_config = lambda path: SimpleNamespace(
        left_config_path="left-only.yaml",
        right_config_path="right-should-not-load.yaml",
    )

    def load_retargeting_config(path: str):
        loaded_paths.append(path)
        if path != "left-only.yaml":
            raise AssertionError(f"unexpected path loaded: {path}")
        return SimpleNamespace(
            solver=SimpleNamespace(max_iterations=30, output_alpha=0.7),
            preprocess=SimpleNamespace(temporal_filter_alpha=0.35),
        )

    somehand_api.load_retargeting_config = load_retargeting_config
    somehand_pkg = ModuleType("somehand")
    somehand_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "somehand", somehand_pkg)
    monkeypatch.setitem(sys.modules, "somehand.api", somehand_api)
    monkeypatch.setattr("teleopit.sim2real.hands.linkerhand_l6.version", lambda name: "0.3.0")
    monkeypatch.setattr("teleopit.sim2real.hands.linkerhand_l6._resolve_project_path", lambda path: SimpleNamespace(exists=lambda: True))
    monkeypatch.setattr(
        "teleopit.sim2real.hands.linkerhand_l6._load_linkerhand_mapping_module",
        lambda: SimpleNamespace(
            l6_l_min=[0.0, -0.087266, 0.0, 0.0, 0.0, 0.0],
            l6_l_max=[0.837758, 1.256637, 1.134464, 1.134464, 1.134464, 1.134464],
            l6_l_derict=[-1, -1, -1, -1, -1, -1],
        ),
    )
    config_dict = _cfg(mode="vr_hand_pose")
    config_dict["hands"]["sides"] = ["left"]  # type: ignore[index]
    mapper = SomehandL6Mapper(parse_linkerhand_l6_config(config_dict))

    mapper.start()

    assert loaded_paths == ["left-only.yaml"]


def test_o6_retarget_pose_mapper_uses_o6_thumb_yaw_and_mapping(monkeypatch) -> None:
    class FakeHandModel:
        def get_joint_name_to_qpos_index(self) -> dict[str, int]:
            return {
                "lh_thumb_cmc_pitch": 0,
                "lh_thumb_cmc_yaw": 1,
                "lh_index_mcp_pitch": 2,
                "lh_middle_mcp_pitch": 3,
                "lh_ring_mcp_pitch": 4,
                "lh_pinky_mcp_pitch": 5,
            }

    mapping = SimpleNamespace(
        o6_l_min=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        o6_l_max=[0.58, 1.36, 1.6, 1.6, 1.6, 1.6],
        o6_l_derict=[-1, -1, -1, -1, -1, -1],
        is_within_range=lambda value, lower, upper: max(lower, min(upper, value)),
        scale_value=lambda value, in_min, in_max, out_min, out_max: out_min
        + (value - in_min) * (out_max - out_min) / (in_max - in_min),
    )
    monkeypatch.setattr("teleopit.sim2real.hands.linkerhand_l6._load_linkerhand_mapping_module", lambda: mapping)

    mapper = RetargetPoseMapper(FakeHandModel(), side="left", family="O6", joint_order=O6_SDK_JOINT_ORDER)

    assert mapper.qpos_to_pose(np.asarray([0.58, 0.0, 0.8, 1.6, 0.0, 1.6])) == [0, 255, 128, 0, 255, 0]


def test_hand_runtime_closes_device_when_mapper_start_fails() -> None:
    calls: list[str] = []

    class FakeDevice:
        def connect(self) -> None:
            calls.append("connect")

        def send_pose(self, *args, **kwargs) -> None:
            raise AssertionError("send_pose should not be called")

        def open_all(self, *args, **kwargs) -> None:
            calls.append("open_all")

        def close(self) -> None:
            calls.append("close")

    class FailingMapper:
        def start(self) -> None:
            calls.append("mapper_start")
            raise RuntimeError("mapper failed")

        def map(self, *args, **kwargs):
            return ()

        def close(self) -> None:
            calls.append("mapper_close")

    runtime = HandRuntime(FakeDevice(), FailingMapper())

    with pytest.raises(RuntimeError, match="mapper failed"):
        runtime.start()

    assert calls == ["connect", "mapper_start", "close"]


def test_hand_runtime_reports_actual_open_commands() -> None:
    calls: list[tuple[str, object, object]] = []
    open_commands = (
        HandPoseCommand("left", (250, 10, 250, 250, 250, 250), True, "open"),
        HandPoseCommand("right", (250, 10, 250, 250, 250, 250), True, "open"),
    )

    class FakeDevice:
        def connect(self) -> None:
            calls.append(("connect", None, None))

        def get_state(self, side: str) -> tuple[float, ...]:
            start = 1.0 if side == "left" else 11.0
            return tuple(start + index for index in range(6))

        def send_pose(self, side, pose, *, force=False, reason="") -> None:
            calls.append((side, tuple(pose), reason))

        def open_all(self, *, force=False, reason="") -> None:
            calls.append(("open_all", force, reason))

        def close(self) -> None:
            calls.append(("close", None, None))

    class Mapper:
        def __init__(self) -> None:
            self.fail = False

        def start(self) -> None:
            calls.append(("mapper_start", None, None))

        def map(self, *args, **kwargs):
            if self.fail:
                raise RuntimeError("tick failed")
            return (HandPoseCommand("left", (1, 2, 3, 4, 5, 6), False, "mapped"),)

        def close(self) -> None:
            calls.append(("mapper_close", None, None))

    mapper = Mapper()
    runtime = HandRuntime(FakeDevice(), mapper, open_commands=open_commands)

    startup = runtime.start()
    assert runtime.get_state("right") == (11.0, 12.0, 13.0, 14.0, 15.0, 16.0)
    ticked = runtime.tick(controller_snapshot=None, hand_snapshot=None, active=True, now_s=1.0)
    mapper.fail = True
    failure = runtime.tick(controller_snapshot=None, hand_snapshot=None, active=True, now_s=2.0)
    shutdown = runtime.close()

    assert [command.reason for command in startup] == ["startup", "startup"]
    assert ticked[0].pose == (1, 2, 3, 4, 5, 6)
    assert [command.reason for command in failure] == ["failure", "failure"]
    assert [command.reason for command in shutdown] == ["shutdown", "shutdown"]
    assert ("open_all", True, "failure") in calls
    assert ("close", None, None) in calls


def test_linkerhand_l6_device_wraps_sdk_system_exit_and_cleans_up(monkeypatch) -> None:
    created_hands = []

    class ExitingLinkerHandApi:
        def __init__(self, *, hand_joint: str, hand_type: str, modbus: str, can: str) -> None:
            del hand_joint, modbus, can
            if hand_type == "right":
                raise SystemExit(1)
            self.hand = FakeInnerHand()
            created_hands.append(self)

        def set_speed(self, speed: list[int]) -> None:
            self.speed = list(speed)

        def finger_move(self, pose: list[int]) -> None:
            self.pose = list(pose)

    monkeypatch.setitem(
        sys.modules,
        "LinkerHand.linker_hand_api",
        SimpleNamespace(LinkerHandApi=ExitingLinkerHandApi),
    )
    cfg = parse_linkerhand_l6_config(_cfg())
    device = LinkerHandL6Device(cfg)

    with pytest.raises(RuntimeError, match="LinkerHand SDK exited during startup"):
        device.connect()

    assert len(created_hands) == 1
    assert created_hands[0].hand.close_calls == 1
