from __future__ import annotations

import math
from types import ModuleType, SimpleNamespace

import numpy as np

from teleopit.inputs.pico4_provider import PicoHeadPoseSnapshot
from teleopit.sim2real.neck.config import NeckConfig, parse_neck_config
from teleopit.sim2real.neck.mapper import HmdPoseMapper
from teleopit.sim2real.neck.openneck import DryRunNeckDevice, OpenNeckDevice
from teleopit.sim2real.neck.worker import NeckRuntime, head_pose_packet
from teleopit.sim2real.mp.messages import SnapshotPacket


def _quat_y(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    return np.array([math.cos(rad / 2.0), 0.0, math.sin(rad / 2.0), 0.0], dtype=np.float64)


def _quat_x(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    return np.array([math.cos(rad / 2.0), math.sin(rad / 2.0), 0.0, 0.0], dtype=np.float64)


class FakeDevice:
    def __init__(self) -> None:
        self.moves: list[tuple[float, float]] = []
        self.center_calls = 0
        self.released = False
        self.closed = False

    def connect(self) -> None:
        return None

    def center(self) -> None:
        self.center_calls += 1

    def release_torque(self) -> None:
        self.released = True

    def move_deg(self, yaw_deg: float, pitch_deg: float) -> tuple[float, float]:
        self.moves.append((yaw_deg, pitch_deg))
        return max(-20.0, min(20.0, yaw_deg)), max(-10.0, min(10.0, pitch_deg))

    def read_deg(self) -> tuple[float, float]:
        return -18.5, 9.5

    def close(self) -> None:
        self.closed = True


def test_hmd_pose_mapper_applies_pitch_gain_to_openneck_degrees() -> None:
    mapper = HmdPoseMapper(
        NeckConfig(enabled=True, dead_zone_deg=0.0, pitch_gain=1.4)
    )

    command = mapper.map_pose(
        hmd_rotation_wxyz=_quat_y(30.0),
        spine3_rotation_wxyz=_quat_y(0.0),
    )
    assert command is not None
    assert command.yaw_deg == pytest_approx(-30.0)

    command = mapper.map_pose(
        hmd_rotation_wxyz=_quat_y(0.0),
        spine3_rotation_wxyz=_quat_y(0.0),
    )
    assert command is not None
    assert command.yaw_deg == pytest_approx(0.0)

    command = mapper.map_pose(
        hmd_rotation_wxyz=_quat_x(15.0),
        spine3_rotation_wxyz=_quat_x(0.0),
    )
    assert command is not None
    assert command.pitch_deg == pytest_approx(-21.0)


def test_hmd_pose_mapper_applies_dead_zone_before_pitch_gain() -> None:
    mapper = HmdPoseMapper(
        NeckConfig(enabled=True, dead_zone_deg=0.5, pitch_gain=2.0)
    )

    inside_dead_zone = mapper.map_pose(
        hmd_rotation_wxyz=_quat_x(0.4),
        spine3_rotation_wxyz=_quat_x(0.0),
    )
    outside_dead_zone = mapper.map_pose(
        hmd_rotation_wxyz=_quat_x(10.0),
        spine3_rotation_wxyz=_quat_x(0.0),
    )

    assert inside_dead_zone is not None
    assert inside_dead_zone.pitch_deg == pytest_approx(0.0)
    assert outside_dead_zone is not None
    assert outside_dead_zone.pitch_deg == pytest_approx(-20.0)


def test_hmd_pose_mapper_uses_body_relative_orientation() -> None:
    mapper = HmdPoseMapper(NeckConfig(enabled=True, dead_zone_deg=0.0))

    command = mapper.map_pose(
        hmd_rotation_wxyz=_quat_y(40.0),
        spine3_rotation_wxyz=_quat_y(10.0),
    )

    assert command is not None
    assert command.yaw_deg == pytest_approx(-30.0)


def test_hmd_pose_mapper_requires_hmd_and_spine3_orientations() -> None:
    mapper = HmdPoseMapper(NeckConfig(enabled=True))

    assert mapper.map_pose(
        hmd_rotation_wxyz=_quat_y(30.0),
        spine3_rotation_wxyz=None,
    ) is None
    assert mapper.map_pose(
        hmd_rotation_wxyz=None,
        spine3_rotation_wxyz=_quat_y(0.0),
    ) is None


def test_neck_runtime_sends_degrees_and_returns_applied_target() -> None:
    device = FakeDevice()
    cfg = NeckConfig(
        enabled=True,
        dead_zone_deg=0.0,
        center_on_start=True,
        center_on_shutdown=True,
    )
    runtime = NeckRuntime(cfg, device=device)

    runtime.start()
    assert runtime.read_deg() == (-18.5, 9.5)
    command = runtime.tick(
        hmd_rotation_wxyz=_quat_y(30.0),
        spine3_rotation_wxyz=_quat_y(0.0),
        pose_timestamp_s=1.0,
        active=True,
        now_s=1.01,
    )
    neutral_command = runtime.tick(
        hmd_rotation_wxyz=_quat_y(0.0),
        spine3_rotation_wxyz=_quat_y(0.0),
        pose_timestamp_s=1.02,
        active=True,
        now_s=1.03,
    )
    runtime.close()

    assert command is not None
    assert command.yaw_deg == pytest_approx(-20.0)
    assert command.pitch_deg == pytest_approx(0.0)
    assert neutral_command is not None
    assert neutral_command.yaw_deg == pytest_approx(0.0)
    np.testing.assert_allclose(device.moves, [(-30.0, 0.0), (0.0, 0.0)], atol=1e-6)
    assert device.center_calls == 2
    assert device.closed is True


def test_neck_runtime_releases_torque_on_shutdown_when_enabled() -> None:
    device = FakeDevice()
    runtime = NeckRuntime(
        NeckConfig(enabled=True, center_on_start=False, release_on_shutdown=True),
        device=device,
    )

    runtime.close()

    assert device.released is True
    assert device.closed is True


def test_neck_shutdown_defaults_to_close_only() -> None:
    device = FakeDevice()
    runtime = NeckRuntime(NeckConfig(enabled=True, center_on_start=False), device=device)

    runtime.close()

    assert device.center_calls == 0
    assert device.released is False
    assert device.closed is True


def test_neck_runtime_closes_after_shutdown_center_failure() -> None:
    class CenterFailingDevice(FakeDevice):
        def center(self) -> None:
            raise RuntimeError("neck center failed")

    device = CenterFailingDevice()
    runtime = NeckRuntime(
        NeckConfig(enabled=True, center_on_start=False, center_on_shutdown=True),
        device=device,
    )

    runtime.close()

    assert device.closed is True


def test_head_pose_packet_extracts_synchronized_snapshot() -> None:
    snapshot = PicoHeadPoseSnapshot(
        hmd_rotation_wxyz=_quat_y(20.0),
        spine3_rotation_wxyz=_quat_y(5.0),
        timestamp_s=1.0,
        seq=4,
    )

    hmd_rotation, spine3_rotation, timestamp_s, seq = head_pose_packet(
        SnapshotPacket(snapshot=snapshot, timestamp_s=1.0, seq=4)
    )

    np.testing.assert_allclose(hmd_rotation, _quat_y(20.0))
    np.testing.assert_allclose(spine3_rotation, _quat_y(5.0))
    assert timestamp_s == 1.0
    assert seq == 4


def test_head_pose_packet_ignores_incomplete_or_mismatched_packets() -> None:
    assert head_pose_packet(None) == (None, None, None, -1)
    assert head_pose_packet(SimpleNamespace(snapshot=object(), timestamp_s=1.0, seq=1)) == (
        None,
        None,
        None,
        -1,
    )
    snapshot = PicoHeadPoseSnapshot(
        hmd_rotation_wxyz=_quat_y(0.0),
        spine3_rotation_wxyz=_quat_y(0.0),
        timestamp_s=1.0,
        seq=2,
    )
    assert head_pose_packet(SnapshotPacket(snapshot=snapshot, timestamp_s=1.0, seq=3)) == (
        None,
        None,
        None,
        -1,
    )


def test_openneck_device_uses_angle_api_and_returns_applied_target(monkeypatch) -> None:
    calls: list[str] = []

    class FakeOpenNeckController:
        port = "/dev/fake"

        def __init__(self, *, config: object, port: object) -> None:
            calls.append(f"init-{config}-{port}")

        def connect(self) -> None:
            calls.append("connect")

        def center(self) -> None:
            calls.append("center")

        def move_deg(self, yaw_deg: float, pitch_deg: float) -> SimpleNamespace:
            calls.append(f"move-{yaw_deg}-{pitch_deg}")
            return SimpleNamespace(yaw_deg=-20.0, pitch_deg=10.0)

        def read_deg(self) -> SimpleNamespace:
            calls.append("read")
            return SimpleNamespace(yaw_deg=-18.5, pitch_deg=9.5)

        def release_torque(self) -> None:
            calls.append("release-torque")

        def close(self) -> None:
            calls.append("close")

    module = ModuleType("openneck")
    module.OpenNeckController = FakeOpenNeckController  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "openneck", module)

    device = OpenNeckDevice(
        NeckConfig(enabled=True, config_path="neck.json", port="/dev/ttyACM0")
    )
    device.connect()
    device.center()
    applied = device.move_deg(-25.0, 15.0)
    state = device.read_deg()
    device.release_torque()
    device.close()

    assert applied == (-20.0, 10.0)
    assert state == (-18.5, 9.5)
    assert calls == [
        "init-neck.json-/dev/ttyACM0",
        "connect",
        "center",
        "move--25.0-15.0",
        "read",
        "release-torque",
        "close",
    ]


def test_dry_run_neck_device_reuses_openneck_calibration_clamp(monkeypatch) -> None:
    calls: list[str] = []

    class FakeOpenNeckController:
        def __init__(self, *, config: object, port: object) -> None:
            calls.append(f"init-{config}-{port}")

        def move_deg(self, yaw_deg: float, pitch_deg: float) -> None:
            del yaw_deg, pitch_deg
            raise AssertionError("dry-run must not send a hardware command")

        def read_deg(self) -> None:
            raise AssertionError("dry-run must not read hardware state")

        def _angle_to_step(self, axis: str, angle_deg: float) -> int:
            calls.append(f"angle-to-step-{axis}-{angle_deg}")
            low, high = (-20.0, 20.0) if axis == "yaw" else (-10.0, 10.0)
            return round(max(low, min(high, angle_deg)))

        def _step_to_angle(self, axis: str, step: int) -> float:
            calls.append(f"step-to-angle-{axis}-{step}")
            return float(step)

    module = ModuleType("openneck")
    module.OpenNeckController = FakeOpenNeckController  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "openneck", module)

    device = DryRunNeckDevice(
        NeckConfig(
            enabled=True,
            config_path="neck.json",
            port="/dev/ttyACM0",
            dry_run=True,
        )
    )
    device.connect()
    applied = device.move_deg(25.0, -15.0)
    try:
        device.read_deg()
    except RuntimeError as exc:
        assert "no hardware state" in str(exc)
    else:
        raise AssertionError("expected dry-run state read to fail")
    device.close()

    assert applied == (20.0, -10.0)
    assert calls == [
        "init-neck.json-/dev/ttyACM0",
        "angle-to-step-yaw-25.0",
        "angle-to-step-pitch--15.0",
        "step-to-angle-yaw-20",
        "step-to-angle-pitch--10",
    ]


def test_parse_neck_config_validates_rate() -> None:
    try:
        parse_neck_config({"neck": {"enabled": True, "rate_hz": 0}})
    except ValueError as exc:
        assert "neck.rate_hz" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_parse_neck_config_accepts_scalar_active_mode() -> None:
    cfg = parse_neck_config({"neck": {"enabled": True, "active_modes": "mocap"}})

    assert cfg.active_modes == ("mocap",)


def test_parse_neck_config_accepts_pitch_gain() -> None:
    cfg = parse_neck_config({"neck": {"enabled": True, "pitch_gain": 1.6}})

    assert cfg.pitch_gain == pytest_approx(1.6)


def test_parse_neck_config_rejects_invalid_pitch_gain() -> None:
    for value in (0.0, -1.0, float("nan"), float("inf")):
        try:
            parse_neck_config({"neck": {"enabled": True, "pitch_gain": value}})
        except ValueError as exc:
            assert "neck.pitch_gain" in str(exc)
        else:
            raise AssertionError(f"expected ValueError for pitch_gain={value!r}")


def test_parse_neck_config_rejects_unknown_active_mode() -> None:
    try:
        parse_neck_config({"neck": {"enabled": True, "active_modes": ["mocap", "idle"]}})
    except ValueError as exc:
        assert "neck.active_modes" in str(exc)
        assert "idle" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_parse_neck_config_rejects_removed_normalized_fields() -> None:
    try:
        parse_neck_config({"neck": {"enabled": True, "yaw_range_deg": 90.0}})
    except ValueError as exc:
        assert "Removed normalized OpenNeck config" in str(exc)
        assert "angles in degrees" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value, abs=1e-6)
