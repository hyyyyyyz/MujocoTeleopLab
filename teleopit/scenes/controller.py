"""SIMPLE-compatible controller mapping for table-top scene teleoperation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .xr_packet import SceneXRPacket


_HEADSET_TO_WORLD = np.array(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


@dataclass(frozen=True)
class SceneControlCommand:
    """One normalized command consumed by the decoupled-WBC scene runtime."""

    left_wrist: np.ndarray
    right_wrist: np.ndarray
    left_fingers: dict[str, np.ndarray]
    right_fingers: dict[str, np.ndarray]
    navigate_cmd: np.ndarray
    base_height_command: float
    locomotion_enabled: bool = False
    locomotion_toggled: bool = False
    # One-frame edge pulse matching SIMPLE/PicoStreamer's
    # ``control_data["toggle_policy_action"]``.  Keep this separate from the
    # level state above so the runtime can forward the event to the WBC policy
    # without guessing whether the button is still held.
    toggle_policy_action: bool = False
    activation_toggled: bool = False
    reset_requested: bool = False


class SimpleSceneController:
    """Direct implementation of SIMPLE's Pico controller conventions.

    Menu + left index toggles the locomotion-input lock and Menu + right index
    toggles arm/hand teleoperation, matching SIMPLE's two activation chords.
    The WBC balance policy stays armed while locomotion is locked, so the
    table-top robot cannot drop merely because walking is paused.  Both grips
    request a full scene reset.  B deliberately remains untouched so
    XRoboToolkit Remote Vision can retain its single-/stereo-view control.
    """

    def __init__(
        self,
        *,
        activation_threshold: float = 0.5,
        reset_threshold: float = 0.5,
        dead_zone: float = 0.1,
        max_linear_velocity: float = 0.5,
        max_yaw_velocity: float = 1.0,
        initial_base_height: float = 0.74,
        height_rate: float = 0.5,
    ) -> None:
        values = {
            "activation_threshold": activation_threshold,
            "reset_threshold": reset_threshold,
            "dead_zone": dead_zone,
            "max_linear_velocity": max_linear_velocity,
            "max_yaw_velocity": max_yaw_velocity,
            "initial_base_height": initial_base_height,
            "height_rate": height_rate,
        }
        try:
            converted = {name: float(value) for name, value in values.items()}
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("scene controller options must be numeric") from exc
        if any(isinstance(value, (bool, np.bool_)) for value in values.values()):
            raise ValueError("scene controller options must be numeric")
        if not all(math.isfinite(value) for value in converted.values()):
            raise ValueError("scene controller options must be finite")
        if not 0.0 < converted["activation_threshold"] <= 1.0:
            raise ValueError("activation_threshold must be in (0, 1]")
        if not 0.0 < converted["reset_threshold"] <= 1.0:
            raise ValueError("reset_threshold must be in (0, 1]")
        if not 0.0 <= converted["dead_zone"] < 1.0:
            raise ValueError("dead_zone must be in [0, 1)")
        if converted["max_linear_velocity"] < 0.0 or converted["max_yaw_velocity"] < 0.0:
            raise ValueError("maximum velocities must be non-negative")
        if converted["initial_base_height"] < 0.2:
            raise ValueError("initial_base_height must be at least 0.2 m")
        if converted["height_rate"] < 0.0:
            raise ValueError("height_rate must be non-negative")
        self._activation_threshold = converted["activation_threshold"]
        self._reset_threshold = converted["reset_threshold"]
        self._dead_zone = converted["dead_zone"]
        self._max_linear_velocity = converted["max_linear_velocity"]
        self._max_yaw_velocity = converted["max_yaw_velocity"]
        self._initial_base_height = converted["initial_base_height"]
        self._height_rate = converted["height_rate"]
        self.reset()

    @property
    def active(self) -> bool:
        return self._active

    @property
    def locomotion_enabled(self) -> bool:
        """Whether joystick navigation is currently unlocked."""
        return self._locomotion_enabled

    def reset(self, *, packet: SceneXRPacket | None = None) -> None:
        """Reset the scene mode state.

        When a reset is requested by holding both side grips, the runtime
        resets the controller while the same XR sample is still physically
        held.  Preserve the edge-latches for that sample so the next packet
        does not immediately retrigger reset (or another held Menu chord).
        The operator must release a chord and press it again for a new edge.
        """
        self._active = False
        self._locomotion_enabled = False
        self._base_height = self._initial_base_height
        # The decoupled-WBC navigation contract contains both an instantaneous
        # yaw-rate flag and an integrated absolute target yaw.  Keep the
        # latter in the controller (rather than asking the WBC layer to infer
        # it from a rate) so a released joystick stops turning at the current
        # heading and re-engaging locomotion does not lose the operator's
        # chosen orientation.
        self._target_yaw = 0.0
        self._last_timestamp_s = None if packet is None else float(packet.timestamp_s)
        if packet is None:
            self._locomotion_was_pressed = False
            self._activation_was_pressed = False
            self._reset_was_pressed = False
        else:
            self._locomotion_was_pressed = bool(
                packet.left_menu and packet.left_trigger > self._activation_threshold
            )
            self._activation_was_pressed = bool(
                packet.left_menu and packet.right_trigger > self._activation_threshold
            )
            self._reset_was_pressed = bool(
                packet.left_grip > self._reset_threshold
                and packet.right_grip > self._reset_threshold
            )

    def update(self, packet: SceneXRPacket) -> SceneControlCommand:
        dt = self._input_dt(packet.timestamp_s)
        # Match ``PicoStreamer._generate_unified_raw_data`` exactly: the
        # upstream chord is considered pressed only once the analogue trigger
        # is *above* 0.5.  Keeping the strict inequality matters at the
        # boundary (XR SDK samples may quantize to exactly 0.5) and prevents a
        # half-pull from unexpectedly toggling a mode.
        locomotion_pressed = packet.left_menu and packet.left_trigger > self._activation_threshold
        activation_pressed = packet.left_menu and packet.right_trigger > self._activation_threshold
        reset_pressed = (
            packet.left_grip > self._reset_threshold and packet.right_grip > self._reset_threshold
        )
        locomotion_toggled = locomotion_pressed and not self._locomotion_was_pressed
        activation_toggled = activation_pressed and not self._activation_was_pressed
        reset_requested = reset_pressed and not self._reset_was_pressed
        self._locomotion_was_pressed = locomotion_pressed
        self._activation_was_pressed = activation_pressed
        self._reset_was_pressed = reset_pressed
        if locomotion_toggled:
            self._locomotion_enabled = not self._locomotion_enabled
        if activation_toggled:
            self._active = not self._active

        # SIMPLE's PicoStreamer always publishes the latest joystick command.
        # The Menu+left-trigger edge is a lower-body policy toggle, not a
        # transport gate: G1GearWbcPolicy decides whether to apply the walking
        # policy, while retaining the command lets it switch immediately when
        # the policy is re-enabled.
        forward = self._dead_zoned(packet.left_axis[1]) * self._max_linear_velocity
        strafe = -self._dead_zoned(packet.left_axis[0]) * self._max_linear_velocity
        yaw_rate = -self._dead_zoned(packet.right_axis[0]) * self._max_yaw_velocity
        self._target_yaw += yaw_rate * dt
        # Keep the integrated target bounded.  Besides matching SIMPLE's
        # streamer, wrapping avoids an ever-growing interpolation value during
        # long tabletop sessions and handles the ±pi discontinuity explicitly.
        self._target_yaw = float(np.arctan2(np.sin(self._target_yaw), np.cos(self._target_yaw)))
        if packet.y:
            self._base_height += self._height_rate * dt
        elif packet.x:
            self._base_height -= self._height_rate * dt
        self._base_height = float(np.clip(self._base_height, 0.2, self._initial_base_height))

        return SceneControlCommand(
            left_wrist=self._head_aligned_pose(packet.left_pose, packet.head_pose),
            right_wrist=self._head_aligned_pose(packet.right_pose, packet.head_pose),
            left_fingers=self._finger_targets(packet, "left"),
            right_fingers=self._finger_targets(packet, "right"),
            # decoupled_wbc's G1GearWbcPolicy consumes SIMPLE's four-value
            # navigation contract ``[vx, vy, vyaw, target_yaw]``.  The third
            # value is the instantaneous turning flag; the fourth is the
            # integrated absolute heading used to close the yaw loop.
            navigate_cmd=np.array([forward, strafe, yaw_rate, self._target_yaw], dtype=np.float64),
            base_height_command=self._base_height,
            locomotion_enabled=self._locomotion_enabled,
            locomotion_toggled=locomotion_toggled,
            toggle_policy_action=locomotion_toggled,
            activation_toggled=activation_toggled,
            reset_requested=reset_requested,
        )

    def _input_dt(self, timestamp_s: float) -> float:
        last = self._last_timestamp_s
        self._last_timestamp_s = float(timestamp_s)
        if last is None:
            return 1.0 / 50.0
        return float(np.clip(timestamp_s - last, 1.0 / 240.0, 0.1))

    def _dead_zoned(self, value: float) -> float:
        value = float(np.clip(value, -1.0, 1.0))
        if abs(value) < self._dead_zone:
            return 0.0
        return math.copysign((abs(value) - self._dead_zone) / (1.0 - self._dead_zone), value)

    @staticmethod
    def _head_aligned_pose(controller_pose: tuple[float, ...], headset_pose: tuple[float, ...]) -> np.ndarray:
        controller = np.asarray(controller_pose, dtype=np.float64)
        headset = np.asarray(headset_pose, dtype=np.float64)
        controller_quat = controller[3:7]
        headset_quat = headset[3:7]
        if np.linalg.norm(controller_quat) <= 1e-8:
            controller_quat = np.array([0.0, 0.0, 0.0, 1.0])
        if np.linalg.norm(headset_quat) <= 1e-8:
            headset_quat = np.array([0.0, 0.0, 0.0, 1.0])
        controller_pos = _HEADSET_TO_WORLD @ controller[:3]
        headset_pos = _HEADSET_TO_WORLD @ headset[:3]
        controller_rot = _HEADSET_TO_WORLD @ Rotation.from_quat(controller_quat).as_matrix() @ _HEADSET_TO_WORLD.T
        headset_rot = _HEADSET_TO_WORLD @ Rotation.from_quat(headset_quat).as_matrix() @ _HEADSET_TO_WORLD.T
        yaw = Rotation.from_matrix(headset_rot).as_euler("xyz")[2]
        inverse_yaw = Rotation.from_euler("z", -yaw).as_matrix()
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = inverse_yaw @ controller_rot
        result[:3, 3] = inverse_yaw @ (controller_pos - headset_pos)
        return result

    @staticmethod
    def _finger_targets(packet: SceneXRPacket, hand: str) -> dict[str, np.ndarray]:
        """Match the upstream PicoStreamer trigger/grip-to-Dex3 gesture mapping."""
        fingertips = np.zeros((25, 4, 4), dtype=np.float64)
        fingertips[4, 0, 3] = 1.0  # thumb
        trigger = packet.left_trigger if hand == "left" else packet.right_trigger
        grip = packet.left_grip if hand == "left" else packet.right_grip
        if not packet.left_menu:
            if trigger > 0.5 and grip <= 0.5:
                fingertips[9, 0, 3] = 1.0  # index pinch
            elif trigger > 0.5 and grip > 0.5:
                fingertips[14, 0, 3] = 1.0  # middle-finger power grip
            elif trigger <= 0.5 and grip > 0.5:
                fingertips[19, 0, 3] = 1.0  # ring gesture
        return {"position": fingertips}
