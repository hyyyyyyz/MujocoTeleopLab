from __future__ import annotations

import time

import numpy as np

from teleopit.inputs.pico4_provider import BODY_JOINT_NAMES
from teleopit.inputs.realtime_packet import ControlEventType
from teleopit.inputs.xrobotoolkit_provider import XRoboToolkitInputProvider


class _FakeSDK:
    def __init__(self) -> None:
        self.initialized = False
        self.closed = False
        self.timestamp_ns = 1
        self.a_pressed = False
        self.b_pressed = False
        self.body = np.zeros((len(BODY_JOINT_NAMES), 7), dtype=np.float64)
        self.body[:, 0] = np.linspace(-0.2, 0.2, len(BODY_JOINT_NAMES))
        self.body[:, 1] = np.linspace(0.1, 1.0, len(BODY_JOINT_NAMES))
        self.body[:, 6] = 1.0

    def init(self) -> None:
        self.initialized = True

    def close(self) -> None:
        self.closed = True

    def is_body_data_available(self) -> bool:
        return True

    def get_body_joints_pose(self) -> np.ndarray:
        return self.body.copy()

    def get_body_timestamp_ns(self) -> int:
        return self.timestamp_ns

    def get_headset_pose(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    def get_left_grip(self) -> float:
        return 0.25

    def get_right_grip(self) -> float:
        return 0.5

    def get_left_trigger(self) -> float:
        return 0.75

    def get_right_trigger(self) -> float:
        return 1.0

    def get_A_button(self) -> bool:
        return self.a_pressed

    def get_B_button(self) -> bool:
        return self.b_pressed


def test_xrobotoolkit_provider_adapts_body_frames_and_controls() -> None:
    sdk = _FakeSDK()
    provider = XRoboToolkitInputProvider(
        timeout=0.5,
        poll_hz=200.0,
        close_sdk=True,
        sdk_shutdown_settle_s=0.0,
        sdk=sdk,
    )
    try:
        frame, _, first_seq = provider.get_frame_packet()
        assert sdk.initialized is True
        assert set(frame) == set(BODY_JOINT_NAMES)
        assert provider.get_controller_snapshot() is not None
        assert provider.get_controller_snapshot().right.trigger == 1.0  # type: ignore[union-attr]

        sdk.a_pressed = True
        sdk.timestamp_ns += 1
        sdk.body[0, 0] += 0.01
        deadline = time.monotonic() + 0.5
        packet = provider.get_realtime_input_packet()
        while (packet.seq <= first_seq or not packet.control_events) and time.monotonic() < deadline:
            time.sleep(0.01)
            packet = provider.get_realtime_input_packet()
        assert packet.seq > first_seq
        assert [event.event_type for event in packet.control_events] == [ControlEventType.TOGGLE_PAUSE]
    finally:
        provider.close()
    assert sdk.closed is True
