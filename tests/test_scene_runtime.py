from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from teleopit.scenes.controller import SimpleSceneController
from teleopit.scenes.runtime import SceneTeleopRuntime
from teleopit.scenes.xr_packet import SceneXRPacket


def _packet(**overrides: object) -> SceneXRPacket:
    values: dict[str, object] = {
        "sequence": 1,
        "timestamp_s": 1.0,
        "left_pose": [-0.2, 0.0, -0.3, 0.0, 0.0, 0.0, 1.0],
        "right_pose": [0.2, 0.0, -0.3, 0.0, 0.0, 0.0, 1.0],
        "head_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "left_axis": [-1.0, 1.0],
        "right_axis": [-1.0, 0.0],
        "left_trigger": 1.0,
        "right_trigger": 0.0,
        "left_grip": 0.0,
        "right_grip": 0.0,
        "a": False,
        "b": False,
        "x": False,
        "y": False,
        "left_menu": True,
    }
    values.update(overrides)
    return SceneXRPacket.from_mapping(values)


def _ground_check_runtime(*, root_z: float, foot_z: float, contact_dist: float | None = None):
    """Build a lightweight runtime shell for reset-ground unit tests."""

    class FakeContact:
        geom1 = 0
        geom2 = 1

        def __init__(self, distance: float) -> None:
            self.dist = distance

    class FakeData:
        def __init__(self) -> None:
            self.qpos = np.array([0.0, 0.0, root_z, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            self.qvel = np.zeros(6, dtype=np.float64)
            self.geom_xmat = np.zeros((1, 9), dtype=np.float64)
            self.geom_xmat[0, 8] = 1.0
            self.geom_xpos = np.zeros((1, 3), dtype=np.float64)
            self.xpos = np.array([[0.0, 0.0, foot_z]], dtype=np.float64)
            self.contact = [] if contact_dist is None else [FakeContact(contact_dist)]
            self.ncon = len(self.contact)

    class FakeModel:
        geom_bodyid = np.array([0, 1], dtype=np.int32)

    class FakeMujoco:
        @staticmethod
        def mj_forward(_model: object, _data: object) -> None:
            pass

    runtime = SceneTeleopRuntime.__new__(SceneTeleopRuntime)
    runtime._mujoco = FakeMujoco()
    runtime.model = FakeModel()
    runtime.data = FakeData()
    runtime._root_qpos_adr = 0
    runtime._root_qvel_adr = 0
    runtime._ground_plane_ids = (0,)
    runtime._robot_body_ids = frozenset({1})
    runtime._foot_body_ids = (0,)
    return runtime


def test_reset_ground_check_leaves_valid_released_pose_unchanged() -> None:
    runtime = _ground_check_runtime(root_z=0.793, foot_z=0.036136)
    before = runtime.data.qpos.copy()

    assert runtime._ensure_reset_ground_clearance() == 0.0
    assert np.array_equal(runtime.data.qpos, before)
    assert runtime.data.qpos[2] > 0.5
    assert runtime.data.xpos[0, 2] > 0.02


def test_reset_ground_check_lifts_zeroed_root_and_preserves_xy_pose() -> None:
    runtime = _ground_check_runtime(root_z=0.0, foot_z=-0.7568, contact_dist=-0.7918)
    runtime.data.qpos[0:2] = [0.37, -0.21]

    delta = runtime._ensure_reset_ground_clearance()

    assert delta > 0.79
    assert runtime.data.qpos[0:2].tolist() == [0.37, -0.21]
    assert runtime.data.qpos[2] >= 0.5


def test_released_scene_reset_has_finite_root_and_clear_ankles() -> None:
    """Exercise the guard against the real released MuJoCo geometry when present."""
    mujoco = pytest.importorskip("mujoco")
    xml = (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "decoupled_wbc"
        / "control"
        / "robot_model"
        / "model_data"
        / "g1"
        / "pnp_cube_43dof.xml"
    )
    if not xml.is_file():
        pytest.skip("released decoupled-WBC scene assets are not installed")

    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    runtime = SceneTeleopRuntime.__new__(SceneTeleopRuntime)
    runtime._mujoco = mujoco
    runtime.model = model
    runtime.data = data
    runtime._torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    runtime._robot_root_body_id = runtime._find_robot_root_body()
    runtime._robot_body_ids = runtime._descendant_body_ids(runtime._robot_root_body_id)
    runtime._root_qpos_adr, runtime._root_qvel_adr = runtime._find_root_addresses(
        runtime._robot_root_body_id
    )
    runtime._foot_body_ids = tuple(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("left_ankle_roll_link", "right_ankle_roll_link")
    )
    runtime._ground_plane_ids = tuple(
        int(geom_id)
        for geom_id in range(model.ngeom)
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_PLANE
    )

    before = data.qpos.copy()
    assert runtime._ensure_reset_ground_clearance() == 0.0
    assert np.array_equal(data.qpos, before)
    assert np.isfinite(data.qpos[runtime._root_qpos_adr + 2])
    assert data.qpos[runtime._root_qpos_adr + 2] > 0.5
    assert runtime._foot_body_lowest_z() is not None
    assert runtime._foot_body_lowest_z() >= 0.02

    # Reproduce the historical shared-viewer failure (a zeroed floating root)
    # and prove that the same guard restores both root and ankle clearance.
    data.qpos[runtime._root_qpos_adr + 2] = 0.0
    mujoco.mj_forward(model, data)
    assert runtime._ensure_reset_ground_clearance() > 0.0
    assert data.qpos[runtime._root_qpos_adr + 2] >= 0.5
    assert runtime._foot_body_lowest_z() is not None
    assert runtime._foot_body_lowest_z() >= 0.02


def test_runtime_forwards_locomotion_toggle_and_keeps_balance_armed() -> None:
    """The SIMPLE toggle is forwarded without allowing a passive WBC step."""

    class FakeLowerBody:
        def __init__(self) -> None:
            self.use_policy_action = False

    class FakeWbc:
        def __init__(self) -> None:
            self.lower_body_policy = FakeLowerBody()
            self.goal: dict[str, object] | None = None

        def set_observation(self, _: dict[str, object]) -> None:
            pass

        def set_goal(self, goal: dict[str, object]) -> None:
            self.goal = goal

        def activate_policy(self) -> None:
            self.lower_body_policy.use_policy_action = True

        def get_action(self, *, time: float) -> dict[str, np.ndarray]:
            del time
            return {"q": np.zeros(4, dtype=np.float64)}

    class FakeRobotModel:
        def get_body_actuated_joints(self, _: np.ndarray) -> np.ndarray:
            return np.zeros(1, dtype=np.float64)

        def get_hand_actuated_joints(self, _: np.ndarray, *, side: str) -> np.ndarray:
            del side
            return np.zeros(1, dtype=np.float64)

    runtime = SceneTeleopRuntime.__new__(SceneTeleopRuntime)
    runtime._control_hz = 50.0
    runtime._have_first_policy_target = False
    runtime._body_joint_names = ("body",)
    runtime._left_hand_joint_names = ("left",)
    runtime._right_hand_joint_names = ("right",)
    runtime._target_by_joint = {}
    runtime._wbc_policy = FakeWbc()
    runtime._robot_model = FakeRobotModel()
    runtime._wbc_observation = lambda: {}
    runtime._upper_body_target = lambda command, active: np.zeros(1)  # noqa: ARG005

    controller = SimpleSceneController()
    # First sample is locked (no Menu chord), so the runtime's explicit
    # locomotion safety gate can be asserted independently of the pulse.
    locked = controller.update(_packet(left_menu=False, left_trigger=0.0))
    assert locked.locomotion_enabled is False
    runtime.update_policy(locked, active=True)
    assert runtime._wbc_policy.goal is not None
    assert np.allclose(np.asarray(runtime._wbc_policy.goal["navigate_cmd"])[:3], 0.0)

    # A later Menu+left-trigger sample toggles the lock and is forwarded as a
    # one-frame WBC event; navigation is then allowed through.
    command = controller.update(_packet(sequence=2, timestamp_s=1.02))
    runtime.update_policy(command, active=True)

    assert runtime._wbc_policy.goal is not None
    assert runtime._wbc_policy.goal["toggle_policy_action"] is True
    assert np.allclose(np.asarray(runtime._wbc_policy.goal["navigate_cmd"])[:3], [0.5, 0.5, 1.0])
    assert runtime._wbc_policy.lower_body_policy.use_policy_action is True


def test_wbc_observation_uses_resolved_floating_root_addresses() -> None:
    """A free object declared before G1 must not become its policy pose."""

    class FakeData:
        # Put an unrelated free joint at qpos/qvel zero and the robot root at
        # non-zero addresses.  This is valid MuJoCo ordering for a custom
        # scene and catches accidental ``qpos[:7]``/``qvel[:6]`` slicing.
        qpos = np.arange(24, dtype=np.float64) + 10.0
        qvel = np.arange(20, dtype=np.float64) + 100.0
        qacc = np.arange(20, dtype=np.float64) + 200.0
        xquat = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)

    class FakeModel:
        pass

    class FakeMujoco:
        mjtObj = type("Obj", (), {"mjOBJ_BODY": 1})

        @staticmethod
        def mj_objectVelocity(_model: object, _data: object, _obj: object, _body: int, out: np.ndarray, _local: int) -> None:
            out[:] = np.arange(6, dtype=np.float64) + 300.0

    class FakeRobotModel:
        def get_configuration_from_actuated_joints(self, **kwargs: np.ndarray) -> np.ndarray:
            return np.concatenate(tuple(kwargs.values()))

    runtime = SceneTeleopRuntime.__new__(SceneTeleopRuntime)
    runtime.data = FakeData()
    runtime.model = FakeModel()
    runtime._mujoco = FakeMujoco()
    runtime._torso_id = 0
    runtime._root_qpos_adr = 7
    runtime._root_qvel_adr = 6
    runtime._qpos_adr = {"body": 12, "left": 13, "right": 14}
    runtime._qvel_adr = {"body": 12, "left": 13, "right": 14}
    runtime._body_joint_names = ("body",)
    runtime._left_hand_joint_names = ("left",)
    runtime._right_hand_joint_names = ("right",)
    runtime._robot_model = FakeRobotModel()

    observation = runtime._wbc_observation()

    assert np.array_equal(observation["floating_base_pose"], FakeData.qpos[7:14])
    assert np.array_equal(observation["floating_base_vel"], FakeData.qvel[6:12])
    assert np.array_equal(observation["floating_base_acc"], FakeData.qacc[6:12])


def test_find_robot_root_body_climbs_to_free_joint_ancestor_for_custom_scene() -> None:
    """Fixed grouping bodies above ``torso_link`` must not hide the root."""

    class FakeModel:
        nbody = 3
        # world <- robot_group (free joint) <- torso_link (hinge joint)
        body_parentid = np.array([0, 0, 1], dtype=np.int32)
        body_jntadr = np.array([0, 0, 1], dtype=np.int32)
        body_jntnum = np.array([0, 1, 1], dtype=np.int32)
        jnt_type = np.array([0, 1], dtype=np.int32)

    class FakeMujoco:
        mjtObj = type("Obj", (), {"mjOBJ_BODY": 1})
        mjtJoint = type("Joint", (), {"mjJNT_FREE": 0})

        @staticmethod
        def mj_name2id(_model: object, _kind: object, name: str) -> int:
            return -1 if name == "pelvis" else 2

    runtime = SceneTeleopRuntime.__new__(SceneTeleopRuntime)
    runtime.model = FakeModel()
    runtime._mujoco = FakeMujoco()
    runtime._torso_id = 2

    assert runtime._find_robot_root_body() == 1


def test_scene_runtime_defers_arm_calibration_for_zero_quaternion_tracking_warmup() -> None:
    """A fresh but orientation-invalid XR packet must not calibrate the wrists."""

    class FakeRobotModel:
        def get_initial_upper_body_pose(self) -> np.ndarray:
            return np.array([0.25, -0.5], dtype=np.float64)

    class FakeIk:
        def reset(self) -> None:
            raise AssertionError("IK reset is not expected while merely holding warm-up input")

    runtime = SceneTeleopRuntime.__new__(SceneTeleopRuntime)
    runtime._robot_model = FakeRobotModel()
    runtime._retargeting_ik = FakeIk()
    runtime._wrist_preprocessor = None
    runtime._wrist_tracking_valid = False
    runtime._last_upper_body_target = None

    command = SceneTeleopRuntime._standby_command()
    target = runtime._upper_body_target(command, active=True)

    assert np.array_equal(target, [0.25, -0.5])
    assert runtime._wrist_preprocessor is None


def test_scene_runtime_holds_last_arm_target_during_transient_invalid_orientation() -> None:
    """A reconnect zero-quaternion frame must not jerk an active arm target."""

    class FakeRobotModel:
        def get_initial_upper_body_pose(self) -> np.ndarray:
            return np.array([0.0, 0.0], dtype=np.float64)

    class FakeIk:
        def set_goal(self, _goal: object) -> None:
            raise AssertionError("IK must not receive an invalid wrist sample")

        def get_action(self) -> np.ndarray:
            raise AssertionError("IK must not run for an invalid wrist sample")

    runtime = SceneTeleopRuntime.__new__(SceneTeleopRuntime)
    runtime._robot_model = FakeRobotModel()
    runtime._retargeting_ik = FakeIk()
    runtime._wrist_preprocessor = object()
    runtime._wrist_tracking_valid = False
    runtime._last_upper_body_target = np.array([0.7, -0.2], dtype=np.float64)

    command = SceneTeleopRuntime._standby_command()
    target = runtime._upper_body_target(command, active=True)

    assert np.array_equal(target, [0.7, -0.2])
    # Returning a copy keeps the caller from mutating the safety-held target.
    assert target is not runtime._last_upper_body_target


def test_runtime_input_callback_keeps_reset_edge_after_standby_replacement() -> None:
    """Auxiliary consumers must observe the reset sample, not standby data."""

    class FakeReceiver:
        packet_count = 0
        latest = None

        def __init__(self, packet: SceneXRPacket) -> None:
            self._packet = packet
            self._sent = False

        def poll(self) -> SceneXRPacket | None:
            if self._sent:
                return None
            self._sent = True
            self.packet_count = 1
            self.latest = self._packet
            return self._packet

        def is_fresh(self, _: float) -> bool:
            return True

    packet = _packet(left_grip=1.0, right_grip=1.0)
    receiver = FakeReceiver(packet)
    controller = SimpleSceneController()
    runtime = SceneTeleopRuntime.__new__(SceneTeleopRuntime)
    runtime._control_decimation = 1
    runtime._input_timeout_s = 1.0
    runtime._last_packet_sequence = -1
    runtime._mujoco = type("FakeMujoco", (), {"mj_step": staticmethod(lambda *_: None)})()
    runtime.model = object()
    runtime.data = object()
    runtime.reset_calls = 0
    runtime.update_calls = 0

    def reset() -> None:
        runtime.reset_calls += 1

    def update_policy(command: object, *, active: bool) -> None:
        del command, active
        runtime.update_calls += 1

    runtime.reset = reset
    runtime.update_policy = update_policy
    runtime._apply_pd = lambda: None
    seen: list[object] = []

    runtime.run(
        receiver=receiver,  # type: ignore[arg-type]
        controller=controller,
        onscreen=False,
        duration_s=0.001,
        realtime=False,
        input_tick=lambda _packet, command: seen.append(command),
    )

    assert runtime.reset_calls == 1
    assert seen and getattr(seen[0], "reset_requested") is True


def test_runtime_clears_arm_calibration_once_when_input_goes_stale() -> None:
    """A reconnect must recalibrate wrists instead of reusing old offsets."""

    class FakeReceiver:
        packet_count = 0
        latest = None

        def __init__(self, packet: SceneXRPacket) -> None:
            self._packet = packet
            self._polls = 0

        def poll(self) -> SceneXRPacket | None:
            self._polls += 1
            if self._polls == 1:
                self.packet_count = 1
                self.latest = self._packet
                return self._packet
            return None

        def is_fresh(self, _: float) -> bool:
            return self._polls == 1

    packet = _packet()
    receiver = FakeReceiver(packet)
    controller = SimpleSceneController()
    # Keep the controller in an already-active session without generating the
    # Menu+right-trigger activation edge; the test is about stale-input cleanup
    # rather than wrist calibration itself.
    controller._active = True
    # Bypass heavy MuJoCo/WBC construction and exercise only the lifecycle
    # transition in ``run``.
    runtime = SceneTeleopRuntime.__new__(SceneTeleopRuntime)
    runtime._control_decimation = 1
    runtime._input_timeout_s = 0.001
    runtime._last_packet_sequence = -1
    runtime._mujoco = type("FakeMujoco", (), {"mj_step": staticmethod(lambda *_: None)})()
    runtime.model = object()
    runtime.data = object()
    runtime._reset_count = 0
    runtime._wrist_preprocessor = object()
    runtime._retargeting_ik = type(
        "FakeIk", (), {"reset": lambda self: setattr(self, "reset_count", getattr(self, "reset_count", 0) + 1)}
    )()
    runtime._apply_pd = lambda: None
    runtime.update_policy = lambda *_args, **_kwargs: None
    runtime._standby_command = staticmethod(SceneTeleopRuntime._standby_command)

    # Stop after the stale transition is observed; a tiny duration keeps this
    # deterministic without opening a viewer or requiring actual MuJoCo data.
    runtime.run(
        receiver=receiver,  # type: ignore[arg-type]
        controller=controller,
        onscreen=False,
        duration_s=0.005,
        realtime=False,
    )
    assert runtime._wrist_preprocessor is None
    assert runtime._retargeting_ik.reset_count == 1


def test_runtime_disables_failed_auxiliary_callbacks_without_stopping_control() -> None:
    """Remote Vision/diagnostic hooks must not terminate the physics loop."""

    class FakeReceiver:
        packet_count = 0
        latest = None

        def __init__(self, packet: SceneXRPacket) -> None:
            self._packet = packet
            self._polls = 0

        def poll(self) -> SceneXRPacket | None:
            self._polls += 1
            if self._polls == 1:
                self.packet_count = 1
                self.latest = self._packet
                return self._packet
            return None

        def is_fresh(self, _: float) -> bool:
            return True

    runtime = SceneTeleopRuntime.__new__(SceneTeleopRuntime)
    runtime._control_decimation = 1
    runtime._input_timeout_s = 1.0
    runtime._last_packet_sequence = -1
    runtime._mujoco = type("FakeMujoco", (), {"mj_step": staticmethod(lambda *_: None)})()
    runtime.model = object()
    runtime.data = object()
    runtime.update_count = 0
    runtime._apply_pd = lambda: None

    def update_policy(_command: object, *, active: bool) -> None:
        del active
        runtime.update_count += 1

    runtime.update_policy = update_policy
    packet = _packet()
    receiver = FakeReceiver(packet)
    callback_calls = {"frame": 0, "input": 0}

    def broken_frame_tick() -> None:
        callback_calls["frame"] += 1
        raise RuntimeError("renderer unavailable")

    def broken_input_tick(_packet: object, _command: object) -> None:
        callback_calls["input"] += 1
        raise RuntimeError("diagnostic sink unavailable")

    runtime.run(
        receiver=receiver,  # type: ignore[arg-type]
        controller=SimpleSceneController(),
        onscreen=False,
        duration_s=0.005,
        realtime=False,
        frame_tick=broken_frame_tick,
        input_tick=broken_input_tick,
    )

    assert runtime.update_count > 1
    assert callback_calls == {"frame": 1, "input": 1}


@pytest.mark.parametrize("duration_s", [-1.0, float("nan"), float("inf"), "not-a-number"])
def test_runtime_rejects_invalid_duration_before_opening_viewer(duration_s: object) -> None:
    """An invalid duration must fail fast instead of silently running forever."""
    runtime = SceneTeleopRuntime.__new__(SceneTeleopRuntime)
    with pytest.raises(ValueError, match="duration_s"):
        runtime.run(
            receiver=object(),  # type: ignore[arg-type]
            controller=object(),  # type: ignore[arg-type]
            onscreen=False,
            duration_s=duration_s,  # type: ignore[arg-type]
        )


def test_runtime_stops_when_receiver_is_closed() -> None:
    """Supervision can close the receiver to request a prompt runtime exit."""

    class ClosedReceiver:
        packet_count = 0
        latest = None
        closed = True

        def poll(self) -> None:
            raise AssertionError("closed receiver must be checked before polling")

    runtime = SceneTeleopRuntime.__new__(SceneTeleopRuntime)
    # No MuJoCo/WBC state is needed because the closed check happens before
    # the first physics step.
    runtime.run(
        receiver=ClosedReceiver(),  # type: ignore[arg-type]
        controller=object(),  # type: ignore[arg-type]
        onscreen=False,
        duration_s=1.0,
        realtime=False,
    )


@pytest.mark.parametrize(
    ("control_hz", "sim_hz", "timeout"),
    [
        ("not-a-number", 200.0, 0.35),
        (float("nan"), 200.0, 0.35),
        (50.0, "not-a-number", 0.35),
        (50.0, float("inf"), 0.35),
        (50.0, 200.0, "not-a-number"),
        (50.0, 200.0, float("nan")),
        (True, 200.0, 0.35),
    ],
)
def test_runtime_constructor_rejects_malformed_numeric_options(
    control_hz: object, sim_hz: object, timeout: object
) -> None:
    """Bad YAML/CLI values fail before scene asset loading or native imports."""
    with pytest.raises(ValueError, match="control_hz|sim_hz|input_timeout_s"):
        SceneTeleopRuntime(
            scene_xml="/does/not/exist.xml",
            control_hz=control_hz,  # type: ignore[arg-type]
            sim_hz=sim_hz,  # type: ignore[arg-type]
            input_timeout_s=timeout,  # type: ignore[arg-type]
        )
