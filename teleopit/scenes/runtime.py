"""43-DOF G1/Dex3 MuJoCo runtime for SIMPLE-style scene teleoperation."""

from __future__ import annotations

import logging
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .controller import SceneControlCommand, SimpleSceneController
from .xr_packet import SceneXRReceiver


_SCENE_NAMES = {
    "cube": "pnp_cube_43dof.xml",
    "bottle": "pnp_bottle_43dof.xml",
    "box": "lift_box_43dof.xml",
    "robosuite-can": "outputs/scenes/robosuite_can_43dof.xml",
    "robosuite-lemon": "outputs/scenes/robosuite_lemon_43dof.xml",
    "robosuite-bottle": "outputs/scenes/robosuite_bottle_43dof.xml",
}

_logger = logging.getLogger(__name__)


# The released scene XMLs put the floor at z=0 and leave roughly 3.6 cm
# between the ankle-link origins and that plane.  These values are deliberately
# conservative and are only used while restoring a scene reset.  They are not
# a controller limit: once the simulation is running, contact dynamics and the
# WBC policy remain authoritative.
_RESET_GROUND_CLEARANCE_M = 1.0e-3
_RESET_FOOT_BODY_CLEARANCE_M = 2.0e-2
_RESET_MIN_ROOT_HEIGHT_M = 5.0e-1
_RESET_PENETRATION_TOLERANCE_M = 1.0e-5

# XRoboToolkit emits an all-zero quaternion while the headset/controller
# tracking service is warming up (and occasionally for one reconnect frame).
# ``SimpleSceneController`` deliberately keeps accepting such packets so the
# transport can recover, but they must never be used as the neutral wrist
# calibration pose.  Keep the threshold in this runtime alongside the safety
# gate so the arm IK and camera code use the same notion of an orientation
# being usable.
_TRACKING_QUATERNION_EPS = 1.0e-8


def _pose_orientation_is_valid(pose: Any) -> bool:
    """Return whether one XR pose contains a usable finite quaternion."""

    try:
        values = np.asarray(pose, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return False
    return bool(
        values.shape == (7,)
        and np.all(np.isfinite(values))
        and np.linalg.norm(values[3:7]) > _TRACKING_QUATERNION_EPS
    )


def _packet_wrist_tracking_is_valid(packet: Any) -> bool:
    """Require HMD and both controller orientations before arm calibration.

    Wrist poses are first transformed into the HMD frame by the SIMPLE-style
    controller.  A valid wrist quaternion with an invalid HMD quaternion is
    therefore just as unsafe as an invalid wrist itself: the controller has to
    substitute an identity HMD orientation and would calibrate an arbitrary
    frame.  Position/axis/button fields are validated by ``SceneXRPacket``;
    this helper intentionally only gates the orientation-dependent arm path.
    """

    return all(
        _pose_orientation_is_valid(getattr(packet, name, None))
        for name in ("left_pose", "right_pose", "head_pose")
    )


def scene_xml_path(scene: str) -> Path:
    """Resolve one of the released 43-DOF manipulation scenes."""
    try:
        filename = _SCENE_NAMES[scene]
    except KeyError as exc:
        supported = ", ".join(sorted(_SCENE_NAMES))
        raise ValueError(f"Unknown scene '{scene}'. Supported scenes: {supported}") from exc
    root = Path(__file__).resolve().parents[2]
    if filename.startswith("outputs/"):
        path = root / filename
    else:
        path = root / "third_party" / "decoupled_wbc" / "control" / "robot_model" / "model_data" / "g1" / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Scene asset is missing: {path}. Initialize third_party/decoupled_wbc before running scene teleop."
        )
    return path


class SceneTeleopRuntime:
    """Run one physical G1/Dex3 manipulation scene.

    The control topology is deliberately the same as SIMPLE's Pico decoupled
    agent: controller poses → retargeting IK → decoupled balance/walk WBC →
    MuJoCo joint PD.  It stays separate from Teleopit's learned 29-DOF motion
    tracker because that policy cannot command Dex3 hands.
    """

    def __init__(
        self,
        *,
        scene_xml: str | Path,
        control_hz: float = 50.0,
        sim_hz: float = 200.0,
        input_timeout_s: float = 0.35,
    ) -> None:
        # Configuration values often arrive from YAML/CLI as strings.  Cast
        # once and validate the converted values so malformed input raises a
        # deterministic ValueError instead of leaking TypeError from
        # ``math.isfinite`` or a surprising implicit truncation later on.
        try:
            control_hz_value = float(control_hz)
            sim_hz_value = float(sim_hz)
            input_timeout_value = float(input_timeout_s)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "control_hz, sim_hz, and input_timeout_s must be numeric"
            ) from exc
        if any(
            isinstance(value, (bool, np.bool_))
            for value in (control_hz, sim_hz, input_timeout_s)
        ):
            raise ValueError("control_hz, sim_hz, and input_timeout_s must be numeric")
        if (
            not math.isfinite(control_hz_value)
            or not math.isfinite(sim_hz_value)
            or control_hz_value <= 0.0
            or sim_hz_value <= 0.0
        ):
            raise ValueError("control_hz and sim_hz must be positive")
        decimation = sim_hz_value / control_hz_value
        if abs(round(decimation) - decimation) > 1e-8:
            raise ValueError("sim_hz must be an integer multiple of control_hz")
        self._control_hz = control_hz_value
        self._sim_hz = sim_hz_value
        self._control_decimation = int(round(decimation))
        if not math.isfinite(input_timeout_value) or input_timeout_value < 0.0:
            raise ValueError("input_timeout_s must be a finite non-negative value")
        self._input_timeout_s = input_timeout_value
        self._scene_xml = Path(scene_xml).resolve()
        if not self._scene_xml.is_file():
            raise FileNotFoundError(f"MuJoCo scene does not exist: {self._scene_xml}")

        import mujoco

        self._mujoco = mujoco
        self.model = self._compile_scene_model()
        self.data = mujoco.MjData(self.model)
        if self.model.nu != 43:
            raise ValueError(
                f"Scene must expose 43 actuators (29 body + 14 Dex3 hand), found {self.model.nu}: {self._scene_xml}"
            )
        self.model.opt.timestep = 1.0 / self._sim_hz
        self._actuator_names = tuple(str(self.model.actuator(index).name) for index in range(self.model.nu))
        if any(not name for name in self._actuator_names):
            raise ValueError("Scene contains unnamed actuators; 43-DOF joint mapping cannot be validated")
        self._actuator_index = {name: index for index, name in enumerate(self._actuator_names)}
        if len(self._actuator_index) != self.model.nu:
            raise ValueError("Scene contains duplicate actuator names")
        self._qpos_adr = {
            name: int(self.model.jnt_qposadr[self.model.actuator_trnid[index, 0]])
            for index, name in enumerate(self._actuator_names)
        }
        self._qvel_adr = {
            name: int(self.model.jnt_dofadr[self.model.actuator_trnid[index, 0]])
            for index, name in enumerate(self._actuator_names)
        }
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        if torso_id < 0:
            raise ValueError("Scene is missing required torso_link body")
        self._torso_id = torso_id
        # Cache the robot subtree and its free-joint address once.  Reset
        # safety must move only the robot's floating root; object/table qpos
        # entries in the same scene are intentionally left untouched.
        self._robot_root_body_id = self._find_robot_root_body()
        self._robot_body_ids = self._descendant_body_ids(self._robot_root_body_id)
        self._root_qpos_adr, self._root_qvel_adr = self._find_root_addresses(
            self._robot_root_body_id
        )
        self._foot_body_ids = tuple(
            body_id
            for name in (
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_ankle_pitch_link",
                "right_ankle_pitch_link",
                "left_foot",
                "right_foot",
            )
            if (body_id := mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)) >= 0
        )
        plane_type = mujoco.mjtGeom.mjGEOM_PLANE
        self._ground_plane_ids = tuple(
            int(geom_id)
            for geom_id in range(self.model.ngeom)
            if self.model.geom_type[geom_id] == plane_type
        )

        self._setup_wbc()
        self._target_by_joint = self._current_joint_positions()
        self._have_first_policy_target = False
        # Packet sequence numbers are scoped to one XR bridge process.  Keep
        # the generation alongside the sequence so a bridge restart (which
        # intentionally starts its counter at zero) is accepted immediately.
        self._last_packet_session_id: str | None = None
        self._last_packet_sequence = -1
        self._last_command: SceneControlCommand | None = None
        self._last_control_time_s: float | None = None
        self._last_policy_action: dict[str, Any] | None = None
        # Packet freshness and tracking validity are separate concerns.  The
        # XR bridge may deliver a fresh warm-up sample whose quaternions are
        # all zero; such a sample must not become the arm's neutral pose.
        # Direct callers (for example the deterministic scene smoke test) may
        # provide already-normalized poses without going through ``run``.  In
        # the normal bridge path this value is overwritten for every packet
        # before any activation edge is handled.
        self._wrist_tracking_valid = True
        self._last_upper_body_target: np.ndarray | None = None
        self._tracking_invalid_reported = False
        self.reset()

    def _find_robot_root_body(self) -> int:
        """Return the robot root body (normally ``pelvis``).

        Released scenes always expose ``pelvis``.  For custom 43-DOF sources,
        walking up from ``torso_link`` keeps reset protection useful when the
        source uses a different pelvis name while still avoiding table/object
        bodies in the contact scan.  Do not stop at the first body whose
        parent is the world: a custom hierarchy may place the free joint on a
        higher ancestor while retaining one or more fixed grouping bodies
        below it.
        """
        pelvis_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )
        body_id = int(pelvis_id if pelvis_id >= 0 else self._torso_id)
        parent = np.asarray(self.model.body_parentid, dtype=np.int64)
        nbody = int(getattr(self.model, "nbody", len(parent)))
        visited: set[int] = set()
        last_valid_body = body_id
        while 0 <= body_id < nbody and body_id not in visited:
            visited.add(body_id)
            last_valid_body = body_id
            body_jntadr = int(self.model.body_jntadr[body_id])
            body_jntnum = int(self.model.body_jntnum[body_id])
            for joint_id in range(body_jntadr, body_jntadr + body_jntnum):
                if self.model.jnt_type[joint_id] == self._mujoco.mjtJoint.mjJNT_FREE:
                    return body_id
            if body_id == 0:
                break
            next_body = int(parent[body_id])
            if next_body == body_id or next_body < 0 or next_body >= nbody:
                break
            body_id = next_body

        # Keep a deterministic error from ``_find_root_addresses`` for a
        # malformed scene with no free joint, while avoiding a potentially
        # invalid index if a custom model's parent array is corrupt.
        return int(last_valid_body)

    def _descendant_body_ids(self, root_body_id: int) -> frozenset[int]:
        parent = np.asarray(self.model.body_parentid, dtype=np.int64)
        descendants: set[int] = set()
        for body_id in range(int(self.model.nbody)):
            current = body_id
            while current > 0 and current != root_body_id:
                current = int(parent[current])
            if current == root_body_id:
                descendants.add(body_id)
        return frozenset(descendants)

    def _find_root_addresses(self, root_body_id: int) -> tuple[int, int]:
        """Find qpos/qvel addresses for the robot floating-base joint."""
        body_jntadr = int(self.model.body_jntadr[root_body_id])
        body_jntnum = int(self.model.body_jntnum[root_body_id])
        for joint_id in range(body_jntadr, body_jntadr + body_jntnum):
            if self.model.jnt_type[joint_id] == self._mujoco.mjtJoint.mjJNT_FREE:
                return (
                    int(self.model.jnt_qposadr[joint_id]),
                    int(self.model.jnt_dofadr[joint_id]),
                )
        # A released G1 has a free pelvis joint.  Failing explicitly for a
        # malformed custom scene is safer than accidentally moving an object
        # qpos at index 2.
        raise ValueError(
            "Scene robot root must contain a floating free joint for reset ground correction"
        )

    def _compile_scene_model(self) -> Any:
        """Compile the source scene and add the camera used by Remote Vision.

        Adding it through ``MjSpec`` keeps the upstream 43-DOF XML read-only,
        so scene assets can be updated independently of Teleopit.
        """
        spec = self._mujoco.MjSpec.from_file(str(self._scene_xml))
        # Remote Vision renders one 1280×720 eye before duplicating it into
        # XRoboToolkit's 2560×720 stereo transport.
        spec.visual.global_.offwidth = 1280
        spec.visual.global_.offheight = 720
        torso = spec.body("torso_link")
        if torso is None:
            raise ValueError("Scene is missing required torso_link body")
        camera = torso.add_camera()
        camera.name = "scene_head_camera"
        camera.pos = np.array([0.10, 0.0, 0.28], dtype=np.float64)
        # MuJoCo cameras look along local -Z.  This convention looks along the
        # robot's positive-X direction (towards the default table) with a small
        # downward component, matching the released scene's ego-view camera.
        x_axis = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        y_axis = np.array([0.767, 0.001, 0.641], dtype=np.float64)
        x_axis /= np.linalg.norm(x_axis)
        y_axis -= x_axis * np.dot(x_axis, y_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        xyzw = Rotation.from_matrix(np.column_stack((x_axis, y_axis, z_axis))).as_quat()
        camera.quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
        camera.fovy = 90.0
        return spec.compile()

    @property
    def scene_xml(self) -> Path:
        return self._scene_xml

    @property
    def last_policy_action(self) -> dict[str, Any] | None:
        return self._last_policy_action

    def _setup_wbc(self) -> None:
        """Instantiate the released decoupled WBC and hand/arm IK stack."""
        try:
            from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig
            from decoupled_wbc.control.policy.wbc_policy_factory import get_wbc_policy
            from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
            from decoupled_wbc.control.teleop.solver.hand.instantiation.g1_hand_ik_instantiation import (
                instantiate_g1_hand_ik_solver,
            )
            from decoupled_wbc.control.teleop.teleop_retargeting_ik import TeleopRetargetingIK
            from decoupled_wbc.control.teleop.pre_processor.wrists.wrists import WristsPreProcessor
        except ImportError as exc:
            raise RuntimeError(
                "The scene runtime requires the decoupled-WBC environment. "
                "Use scripts/setup/setup_scene_teleop.sh before launching it."
            ) from exc

        self._robot_model = instantiate_g1_robot_model()
        wbc_config = ControlLoopConfig().load_wbc_yaml()
        self._wbc_config = wbc_config
        self._wbc_policy = get_wbc_policy(
            "g1", self._robot_model, wbc_config, init_time=time.monotonic()
        )
        # The released lower-body policy starts in a passive hold mode.  This
        # standalone scene runtime has no upstream keyboard dispatcher to send
        # its ``]`` activation key, so arm it explicitly before physics starts
        # and after every scene reset.
        self._activate_wbc_policy()
        left_hand_ik, right_hand_ik = instantiate_g1_hand_ik_solver()
        self._retargeting_ik = TeleopRetargetingIK(
            robot_model=self._robot_model,
            left_hand_ik_solver=left_hand_ik,
            right_hand_ik_solver=right_hand_ik,
            body_active_joint_groups=["upper_body"],
        )
        self._wrist_preprocessor_type = WristsPreProcessor
        self._wrist_preprocessor: Any | None = None
        info = self._robot_model.supplemental_info
        self._body_joint_names = tuple(info.body_actuated_joints)
        self._left_hand_joint_names = tuple(info.left_hand_actuated_joints)
        self._right_hand_joint_names = tuple(info.right_hand_actuated_joints)
        expected = set(self._body_joint_names + self._left_hand_joint_names + self._right_hand_joint_names)
        actual = set(self._actuator_names)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise ValueError(
                "43-DOF scene actuator names do not match the released WBC model "
                f"(missing={missing}, unexpected={unexpected})"
            )

        body_kp = np.asarray(wbc_config["MOTOR_KP"], dtype=np.float64)
        body_kd = np.asarray(wbc_config["MOTOR_KD"], dtype=np.float64)
        if body_kp.shape != (len(self._body_joint_names),) or body_kd.shape != body_kp.shape:
            raise ValueError("decoupled-WBC body PD gains do not match its 29-DOF joint declaration")
        hand_kp = np.array([5.0, 5.0, 5.0, 2.5, 2.5, 2.5, 2.5], dtype=np.float64)
        self._kp_by_joint = dict(zip(self._body_joint_names, body_kp, strict=True))
        self._kd_by_joint = dict(zip(self._body_joint_names, body_kd, strict=True))
        for names in (self._left_hand_joint_names, self._right_hand_joint_names):
            self._kp_by_joint.update(dict(zip(names, hand_kp, strict=True)))
            self._kd_by_joint.update(dict.fromkeys(names, 1.0))

    def _activate_wbc_policy(self) -> None:
        """Keep the balance policy armed across resets.

        ``decoupled_wbc`` deliberately keeps this state across its own reset,
        but an initial policy instance is passive.  Calling the public helper
        is idempotent for the released implementation and prevents a reset
        scene from dropping before the first Pico packet arrives.
        """
        lower_body_policy = self._wbc_policy.lower_body_policy
        if not bool(getattr(lower_body_policy, "use_policy_action", False)):
            self._wbc_policy.activate_policy()

    def _ground_height(self) -> float:
        """Return the lowest horizontal ground-plane height in the scene.

        The released scenes use one world ``plane`` at z=0.  Looking at the
        compiled pose rather than assuming that value also keeps custom scene
        sources (with a translated floor) usable.  Non-horizontal planes are
        ignored because a single root-z translation cannot safely correct
        against them.
        """
        heights: list[float] = []
        for geom_id in getattr(self, "_ground_plane_ids", ()):
            try:
                # The third column of xmat is the plane normal in world frame.
                normal_z = float(self.data.geom_xmat[geom_id][8])
                height = float(self.data.geom_xpos[geom_id][2])
            except (AttributeError, IndexError, TypeError, ValueError):
                continue
            if math.isfinite(height) and math.isfinite(normal_z) and abs(normal_z) >= 0.9:
                heights.append(height)
        return min(heights, default=0.0)

    def _ground_penetration(self) -> float:
        """Measure deepest robot/floor contact penetration after reset.

        MuJoCo's contact ``dist`` is signed; a negative value means that the
        colliding geometry overlaps the floor.  Restricting the other geometry
        to the pelvis subtree avoids treating a table or a free object below
        the robot as evidence that the robot itself needs to be teleported.
        """
        plane_ids = set(getattr(self, "_ground_plane_ids", ()))
        robot_body_ids = set(getattr(self, "_robot_body_ids", ()))
        if not plane_ids or not robot_body_ids:
            return 0.0
        deepest = 0.0
        try:
            ncon = int(self.data.ncon)
        except (AttributeError, TypeError, ValueError):
            return 0.0
        for contact_index in range(max(ncon, 0)):
            try:
                contact = self.data.contact[contact_index]
                geom1 = int(contact.geom1)
                geom2 = int(contact.geom2)
                if geom1 in plane_ids:
                    other_geom = geom2
                elif geom2 in plane_ids:
                    other_geom = geom1
                else:
                    continue
                other_body = int(self.model.geom_bodyid[other_geom])
                if other_body not in robot_body_ids:
                    continue
                distance = float(contact.dist)
            except (AttributeError, IndexError, TypeError, ValueError):
                continue
            if math.isfinite(distance):
                deepest = max(deepest, -distance)
        return deepest

    def _foot_body_lowest_z(self) -> float | None:
        """Return the lowest cached ankle/foot body origin, if available."""
        values: list[float] = []
        for body_id in getattr(self, "_foot_body_ids", ()):
            try:
                value = float(self.data.xpos[body_id][2])
            except (AttributeError, IndexError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        return min(values) if values else None

    def _ensure_reset_ground_clearance(self) -> float:
        """Lift the robot root only when reset data is visibly unsafe.

        ``mj_resetData`` correctly restores the released XML's qpos, but a
        custom scene (or a stale shared qpos from a viewer process) can leave
        the free root at zero.  In that case MuJoCo immediately reports deeply
        penetrating feet and the first rendered frame looks as if the robot is
        buried.  We correct only the root's z coordinate, preserve every
        object/table coordinate, and leave already-valid poses untouched.

        Returns the applied z offset in metres, primarily for diagnostics and
        deterministic tests.
        """
        root_adr_value = getattr(self, "_root_qpos_adr", None)
        if root_adr_value is None:
            raise RuntimeError("Scene runtime has not resolved the robot floating-root qpos address")
        root_adr = int(root_adr_value)
        try:
            root_z = float(self.data.qpos[root_adr + 2])
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("Scene reset data does not expose a floating-root z coordinate") from exc
        if not math.isfinite(root_z):
            raise ValueError(f"Scene reset produced a non-finite robot root height: {root_z!r}")

        ground_z = self._ground_height()
        required_delta = 0.0

        # Contact distance is the most faithful estimate because it accounts
        # for mesh geometry and foot orientation.  Ignore solver-scale noise
        # around zero so a normal XML reset is byte-for-byte unchanged.
        penetration = self._ground_penetration()
        if penetration > _RESET_PENETRATION_TOLERANCE_M:
            required_delta = max(required_delta, penetration + _RESET_GROUND_CLEARANCE_M)

        # Some custom scenes disable floor collisions.  The ankle origins are
        # still a useful conservative fallback in that case.  The released
        # pose has ~36 mm ankle clearance, so a 20 mm origin threshold does not
        # alter it while catching a zeroed/free-root reset.
        foot_z = self._foot_body_lowest_z()
        if foot_z is not None:
            foot_shortfall = ground_z + _RESET_FOOT_BODY_CLEARANCE_M - foot_z
            if foot_shortfall > 0.0 and math.isfinite(foot_shortfall):
                required_delta = max(required_delta, foot_shortfall)

        # A G1 root below this height cannot be a valid upright reset even if a
        # custom scene omitted ankle collision geoms.  Keep the threshold well
        # below the released 0.793 m pose so standard assets are unaffected.
        root_shortfall = ground_z + _RESET_MIN_ROOT_HEIGHT_M - (root_z + required_delta)
        if root_shortfall > 0.0 and math.isfinite(root_shortfall):
            required_delta = max(required_delta, root_shortfall)

        if required_delta <= 0.0:
            return 0.0

        self.data.qpos[root_adr + 2] = root_z + required_delta
        # A reset must never carry stale translational velocity from a prior
        # episode into the corrected pose.  ``mj_resetData`` already zeros
        # qvel, but this assignment also covers callers that invoke the helper
        # after manually perturbing qpos in a test or recovery path.
        root_vel_adr = int(getattr(self, "_root_qvel_adr", 0))
        try:
            self.data.qvel[root_vel_adr : root_vel_adr + 6] = 0.0
        except (AttributeError, IndexError, TypeError, ValueError):
            pass
        self._mujoco.mj_forward(self.model, self.data)

        corrected_root_z = float(self.data.qpos[root_adr + 2])
        if not math.isfinite(corrected_root_z):
            raise RuntimeError("Scene reset ground correction produced a non-finite root height")
        _logger.warning(
            "Scene reset root was below the ground-safe pose; lifted robot by %.4f m (root z %.4f -> %.4f)",
            required_delta,
            root_z,
            corrected_root_z,
        )
        return required_delta

    def reset(self) -> None:
        self._mujoco.mj_resetData(self.model, self.data)
        self.data.qvel[:] = 0.0
        self._mujoco.mj_forward(self.model, self.data)
        self._ensure_reset_ground_clearance()
        self._target_by_joint = self._current_joint_positions()
        self._wbc_policy.reset(init_time=time.monotonic())
        self._activate_wbc_policy()
        self._retargeting_ik.reset()
        self._wrist_preprocessor = None
        self._have_first_policy_target = False
        self._last_policy_action = None
        self._last_upper_body_target = None
        # Keep direct, packet-independent uses of ``_start_teleoperation``
        # backwards compatible.  The live ``run`` loop sets this flag from
        # each packet (and sets it false on a stale transition) before arm IK
        # can be evaluated.
        self._wrist_tracking_valid = True
        self._tracking_invalid_reported = False

    def _current_joint_positions(self) -> dict[str, float]:
        return {name: float(self.data.qpos[self._qpos_adr[name]]) for name in self._actuator_names}

    def _actuated_values(self, names: tuple[str, ...], quantity: str) -> np.ndarray:
        addresses = self._qpos_adr if quantity == "qpos" else self._qvel_adr
        return np.array([self.data.__getattribute__(quantity)[addresses[name]] for name in names], dtype=np.float64)

    def _wbc_observation(self) -> dict[str, np.ndarray]:
        body_q = self._actuated_values(self._body_joint_names, "qpos")
        left_hand_q = self._actuated_values(self._left_hand_joint_names, "qpos")
        right_hand_q = self._actuated_values(self._right_hand_joint_names, "qpos")
        body_dq = self._actuated_values(self._body_joint_names, "qvel")
        left_hand_dq = self._actuated_values(self._left_hand_joint_names, "qvel")
        right_hand_dq = self._actuated_values(self._right_hand_joint_names, "qvel")
        q = self._robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=body_q,
            left_hand_actuated_joint_values=left_hand_q,
            right_hand_actuated_joint_values=right_hand_q,
        )
        dq = self._robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=body_dq,
            left_hand_actuated_joint_values=left_hand_dq,
            right_hand_actuated_joint_values=right_hand_dq,
        )
        velocity = np.zeros(6, dtype=np.float64)
        self._mujoco.mj_objectVelocity(
            self.model,
            self.data,
            self._mujoco.mjtObj.mjOBJ_BODY,
            self._torso_id,
            velocity,
            1,
        )
        # MuJoCo reports [angular, linear]; the upstream WBC observation uses
        # [linear, angular] and later takes its final three angular entries.
        torso_velocity = np.concatenate((velocity[3:6], velocity[0:3]))
        # The robot free joint is resolved during model construction rather
        # than assumed to be the first qpos/qvel entry.  Released scenes happen
        # to place the G1 root at address zero, but a valid custom tabletop
        # scene may declare a free object before the robot.  Feeding that
        # object's pose to the balance policy makes gravity/yaw observations
        # nonsensical and can destabilize the robot as soon as it moves.
        root_qpos_adr = int(self._root_qpos_adr)
        root_qvel_adr = int(self._root_qvel_adr)
        floating_base_pose = np.asarray(
            self.data.qpos[root_qpos_adr : root_qpos_adr + 7], dtype=np.float64
        ).copy()
        floating_base_vel = np.asarray(
            self.data.qvel[root_qvel_adr : root_qvel_adr + 6], dtype=np.float64
        ).copy()
        floating_base_acc = np.asarray(
            self.data.qacc[root_qvel_adr : root_qvel_adr + 6], dtype=np.float64
        ).copy()
        if floating_base_pose.shape != (7,) or floating_base_vel.shape != (6,) or floating_base_acc.shape != (6,):
            raise RuntimeError(
                "Scene robot floating-base state has an unexpected shape; "
                f"qpos={floating_base_pose.shape}, qvel={floating_base_vel.shape}, "
                f"qacc={floating_base_acc.shape}"
            )
        return {
            "q": q,
            "dq": dq,
            "ddq": np.zeros_like(q),
            "tau_est": np.zeros_like(q),
            "floating_base_pose": floating_base_pose,
            "floating_base_vel": floating_base_vel,
            "floating_base_acc": floating_base_acc,
            "torso_quat": self.data.xquat[self._torso_id].copy(),
            "torso_ang_vel": torso_velocity[3:6].copy(),
        }

    @staticmethod
    def _standby_command() -> SceneControlCommand:
        # Keep the two hand payloads independent.  ``SceneControlCommand`` is
        # frozen, but its numpy members remain mutable; sharing one array here
        # means an auxiliary consumer that annotates either hand's standby
        # gesture can accidentally modify the other hand as well.
        left_fingers = {"position": np.zeros((25, 4, 4), dtype=np.float64)}
        right_fingers = {"position": np.zeros((25, 4, 4), dtype=np.float64)}
        return SceneControlCommand(
            left_wrist=np.eye(4, dtype=np.float64),
            right_wrist=np.eye(4, dtype=np.float64),
            left_fingers=left_fingers,
            right_fingers=right_fingers,
            # Keep the same four-dimensional shape as the WBC interpolation
            # waypoints, even before the first XR packet arrives.
            navigate_cmd=np.zeros(4, dtype=np.float64),
            base_height_command=0.74,
        )

    def _start_teleoperation(self, command: SceneControlCommand) -> bool:
        if not bool(getattr(self, "_wrist_tracking_valid", True)):
            # Keep the controller's active state latched, but defer creating a
            # wrist preprocessor until a complete HMD+controller orientation
            # arrives.  ``SceneXRPacket`` intentionally accepts zero
            # quaternions during tracking warm-up; using one here would bake
            # the identity fallback into the neutral arm reference.
            return False
        raw = {"left_wrist": command.left_wrist, "right_wrist": command.right_wrist}
        self._retargeting_ik.reset()
        wrist_preprocessor = self._wrist_preprocessor_type(
            motion_scale=self._robot_model.supplemental_info.teleop_upper_body_motion_scale
        )
        wrist_preprocessor.register(self._robot_model)
        wrist_preprocessor.calibrate(raw, "pico")
        self._wrist_preprocessor = wrist_preprocessor
        print("Scene teleoperation activated and arm reference calibrated.")
        return True

    def _upper_body_target(self, command: SceneControlCommand, active: bool) -> np.ndarray:
        initial_target = self._robot_model.get_initial_upper_body_pose()
        if not active:
            return initial_target
        if not bool(getattr(self, "_wrist_tracking_valid", True)):
            # Hold the last valid IK target through a transient tracking gap.
            # Returning the initial pose here would visibly jerk the arms and
            # could push an object while the headset is reconnecting.
            last_target = getattr(self, "_last_upper_body_target", None)
            if last_target is not None:
                return np.asarray(last_target, dtype=np.float64).copy()
            return initial_target
        if self._wrist_preprocessor is None:
            self._start_teleoperation(command)
        # ``_start_teleoperation`` can defer calibration when this method is
        # called directly (outside ``run``) during a tracking warm-up.  Keep
        # the same safe fallback instead of asserting on a missing processor.
        if self._wrist_preprocessor is None:
            return initial_target
        assert self._wrist_preprocessor is not None
        body_data = self._wrist_preprocessor(
            {"left_wrist": command.left_wrist, "right_wrist": command.right_wrist}
        )
        self._retargeting_ik.set_goal(
            {
                "body_data": body_data,
                "left_hand_data": command.left_fingers,
                "right_hand_data": command.right_fingers,
            }
        )
        target = np.asarray(self._retargeting_ik.get_action(), dtype=np.float64)
        if target.ndim != 1 or not np.all(np.isfinite(target)):
            raise ValueError(
                "Scene arm IK returned a non-finite or non-vector upper-body target"
            )
        self._last_upper_body_target = target.copy()
        return target

    def update_policy(self, command: SceneControlCommand, *, active: bool) -> None:
        now = time.monotonic()
        observation = self._wbc_observation()
        self._wbc_policy.set_observation(observation)
        # Keep the raw SIMPLE joystick mapping in ``SceneControlCommand`` but
        # retain a deliberate scene safety lock: until the operator presses
        # Menu+left-trigger, translational/turning input is held at zero.  The
        # balance policy itself stays armed, so a locked robot continues to
        # stabilize rather than falling into the upstream passive hold mode.
        navigate_cmd = np.asarray(command.navigate_cmd, dtype=np.float64).reshape(-1)
        # The released G1GearWbcPolicy (and its interpolation layer) consumes
        # SIMPLE's four-value navigation contract
        # ``[vx, vy, vyaw, target_yaw]``.  The third value is the instantaneous
        # turning flag and the fourth is an integrated absolute heading.  A
        # three-value command would make the first interpolation waypoint
        # incompatible with the WBC's standing default and fail mid-step.
        if navigate_cmd.shape != (4,) or not np.all(np.isfinite(navigate_cmd)):
            raise ValueError(
                "Scene navigation command must be four finite values "
                "[vx, vy, vyaw, target_yaw], "
                f"got shape {navigate_cmd.shape}"
            )
        navigate_cmd = navigate_cmd.copy()
        if not command.locomotion_enabled:
            navigate_cmd[:3] = 0.0
        goal = {
            "target_upper_body_pose": self._upper_body_target(command, active),
            "navigate_cmd": navigate_cmd,
            # SIMPLE's left Menu + left-trigger chord is an edge-triggered
            # lower-body policy toggle.  Forward the one-frame pulse through
            # G1DecoupledWholeBodyPolicy to G1GearWbcPolicy.
            "toggle_policy_action": bool(command.toggle_policy_action),
            "base_height_command": np.asarray([command.base_height_command], dtype=np.float64),
            "target_time": now + (2.0 if not self._have_first_policy_target else 1.0 / self._control_hz),
            "interpolation_garbage_collection_time": now - 2.0 / self._control_hz,
        }
        self._wbc_policy.set_goal(goal)
        # ``_activate_wbc_policy`` arms the lower-body balance policy before
        # the first physics step and after resets.  The released WBC uses the
        # same ``toggle_policy_action`` bit for its locomotion-policy switch;
        # forwarding the SIMPLE pulse is useful for compatibility, but must
        # not turn off this safety-critical balance policy in the scene path.
        # Re-arm immediately after the WBC consumes the pulse.  The lock above
        # still prevents walking until ``locomotion_enabled`` is true.
        self._activate_wbc_policy()
        action = self._wbc_policy.get_action(time=now)
        self._last_policy_action = action
        self._have_first_policy_target = True
        body = self._robot_model.get_body_actuated_joints(action["q"])
        left_hand = self._robot_model.get_hand_actuated_joints(action["q"], side="left")
        right_hand = self._robot_model.get_hand_actuated_joints(action["q"], side="right")
        self._target_by_joint.update(dict(zip(self._body_joint_names, body, strict=True)))
        self._target_by_joint.update(dict(zip(self._left_hand_joint_names, left_hand, strict=True)))
        self._target_by_joint.update(dict(zip(self._right_hand_joint_names, right_hand, strict=True)))

    def _apply_pd(self) -> None:
        for index, name in enumerate(self._actuator_names):
            q = float(self.data.qpos[self._qpos_adr[name]])
            dq = float(self.data.qvel[self._qvel_adr[name]])
            torque = self._kp_by_joint[name] * (self._target_by_joint[name] - q) - self._kd_by_joint[name] * dq
            # The released G1 scene XML puts its effort limits on the joints
            # (``actuatorfrcrange``), while the corresponding ``motor``
            # actuators do not set ``forcelimited`` themselves.  Looking only
            # at ``actuator_forcelimited`` therefore leaves all 43 PD commands
            # unclipped.  Prefer an actuator-level limit when present and
            # otherwise honor the limit attached to its transmission joint.
            if self.model.actuator_forcelimited[index]:
                lower, upper = self.model.actuator_forcerange[index]
                torque = float(np.clip(torque, lower, upper))
            else:
                joint_id = int(self.model.actuator_trnid[index, 0])
                if self.model.jnt_actfrclimited[joint_id]:
                    lower, upper = self.model.jnt_actfrcrange[joint_id]
                    torque = float(np.clip(torque, lower, upper))
            self.data.ctrl[index] = torque

    def run(
        self,
        *,
        receiver: SceneXRReceiver,
        controller: SimpleSceneController,
        onscreen: bool = True,
        duration_s: float = 0.0,
        realtime: bool = True,
        frame_tick: Any | None = None,
        input_tick: Any | None = None,
    ) -> None:
        """Run until the viewer closes, Ctrl-C is pressed, or duration expires.

        ``input_tick`` is an optional non-blocking callback invoked once for
        every newly accepted XR packet, after the controller has produced its
        command.  It is intended for auxiliary consumers such as Remote
        Vision (HMD pose and B-button view state); it must not mutate the
        MuJoCo control state or perform blocking I/O.
        """
        try:
            duration = float(duration_s)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("duration_s must be a finite non-negative number") from exc
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError("duration_s must be a finite non-negative number")
        viewer = None
        if onscreen:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(self.model, self.data)
        started = time.monotonic()
        next_step = started
        next_input_status = started + 3.0
        last_input_status_s = started
        last_input_packet_count = receiver.packet_count
        step_index = 0
        command = self._standby_command()
        # ``frame_tick`` and ``input_tick`` are deliberately auxiliary hooks
        # (the scene control loop must remain useful when Remote Vision or a
        # diagnostic consumer fails).  Keep local copies so a broken callback
        # can be disabled without changing the public arguments or touching
        # the 200 Hz control path.  Catch ``Exception`` only: a user Ctrl-C
        # must still propagate to the normal shutdown/finally path.
        active_frame_tick = frame_tick
        active_input_tick = input_tick

        def invoke_auxiliary_callback(
            callback_kind: str,
            callback: Any | None,
            *args: Any,
        ) -> Any | None:
            if callback is None:
                return None
            try:
                callback(*args)
            except Exception as exc:
                _logger.warning(
                    "Scene %s callback failed; disabling this auxiliary hook: %s",
                    callback_kind,
                    exc,
                    exc_info=True,
                )
                return None
            return callback
        # A transport gap must not let the pre-disconnect wrist calibration
        # survive into a newly connected XR session.  The bridge can stop
        # publishing for several hundred milliseconds while the headset or PC
        # service reconnects; when packets resume the operator's hands may be
        # in a completely different pose.  Keep this local transition latch so
        # a brief gap only emits one diagnostic/reset and the first fresh
        # packet re-calibrates through ``_upper_body_target``.
        input_was_fresh: bool | None = None
        try:
            while True:
                now = time.monotonic()
                # A receiver can be closed by launcher cleanup or another
                # supervising thread.  ``poll`` intentionally remains
                # idempotent after close, so check the lifecycle flag here to
                # stop the physics loop instead of stepping forever with a
                # permanently stale packet.
                if bool(getattr(receiver, "closed", False)):
                    return
                if duration > 0.0 and now - started >= duration:
                    return
                if viewer is not None and not viewer.is_running():
                    return
                packet = receiver.poll()
                if packet is not None:
                    last_packet_session_id = getattr(self, "_last_packet_session_id", None)
                    if packet.session_id != last_packet_session_id:
                        # A restarted bridge has a new generation ID and may
                        # restart its sequence at zero.  Do not let the old
                        # process-local counter suppress the first packet of
                        # the new stream.  Treat a generation handoff as a
                        # transport reconnect as well: if arm teleoperation
                        # was active, force a fresh wrist calibration so a
                        # newly connected operator pose cannot inherit stale
                        # offsets even when the gap was shorter than the
                        # normal freshness timeout.
                        previous_session_id = last_packet_session_id
                        self._last_packet_session_id = packet.session_id
                        self._last_packet_sequence = -1
                        if previous_session_id is not None and controller.active:
                            retargeting_ik = getattr(self, "_retargeting_ik", None)
                            if retargeting_ik is not None:
                                retargeting_ik.reset()
                            self._wrist_preprocessor = None
                            self._last_upper_body_target = None
                            print("Scene XR bridge session changed; arm reference will recalibrate.")
                    if packet.sequence <= self._last_packet_sequence:
                        packet = None
                if packet is not None:
                    self._last_packet_sequence = packet.sequence
                    # Keep this gate independent from ``receiver.is_fresh``:
                    # a warm-up frame can arrive on time yet contain zero
                    # HMD/controller quaternions.  The controller still
                    # consumes its buttons/joysticks, while arm calibration
                    # and wrist IK wait for a complete valid orientation.
                    tracking_valid = _packet_wrist_tracking_is_valid(packet)
                    self._wrist_tracking_valid = tracking_valid
                    if not tracking_valid:
                        if not bool(getattr(self, "_tracking_invalid_reported", False)):
                            print(
                                "Scene XR tracking orientation unavailable; "
                                "arm calibration will wait for a valid HMD/controller pose."
                            )
                            self._tracking_invalid_reported = True
                    else:
                        self._tracking_invalid_reported = False
                    command = controller.update(packet)
                    # Keep the edge-triggered command for auxiliary consumers
                    # (notably Remote Vision) even if the scene reset below
                    # replaces the control command with a safe standby pose.
                    input_command = command
                    if command.reset_requested:
                        self.reset()
                        # The reset sample may still have both grips held.
                        # Preserve its edge latches so one physical hold maps
                        # to one reset event until the operator releases it.
                        controller.reset(packet=packet)
                        command = self._standby_command()
                        print("Scene and WBC state reset (both Pico grips).")
                    elif command.activation_toggled:
                        if controller.active:
                            if not self._start_teleoperation(command):
                                print(
                                    "Scene teleoperation armed; waiting for a valid "
                                    "HMD/controller orientation before calibrating arms."
                                )
                        else:
                            self._retargeting_ik.reset()
                            self._wrist_preprocessor = None
                            self._last_upper_body_target = None
                            print("Scene teleoperation paused.")
                    if command.locomotion_toggled:
                        status = "enabled" if command.locomotion_enabled else "locked"
                        print(f"Scene locomotion input {status}; balance remains armed.")
                    if active_input_tick is not None:
                        active_input_tick = invoke_auxiliary_callback(
                            "input_tick", active_input_tick, packet, input_command
                        )
                fresh = receiver.is_fresh(self._input_timeout_s)
                if input_was_fresh is True and not fresh and controller.active:
                    # Drop the old wrist reference as soon as input becomes
                    # stale.  ``active`` remains latched for the operator, so
                    # a fresh packet below can re-enter teleop without
                    # requiring another Menu+right-trigger chord, while the
                    # arm target is held at the safe initial pose in the gap.
                    self._retargeting_ik.reset()
                    self._wrist_preprocessor = None
                    self._last_upper_body_target = None
                    self._wrist_tracking_valid = False
                    print("Scene XR input stale; arm reference will recalibrate on reconnect.")
                input_was_fresh = fresh
                if now >= next_input_status:
                    current_packet = receiver.latest
                    current_count = receiver.packet_count
                    elapsed = max(now - last_input_status_s, 1e-6)
                    packet_rate = (current_count - last_input_packet_count) / elapsed
                    if current_packet is None:
                        print("Scene XR input: waiting for bridge packets.")
                    else:
                        print(
                            "Scene XR input: "
                            f"{packet_rate:.1f} Hz, age={receiver.age_s() or 0.0:.3f}s, "
                            f"Menu={current_packet.left_menu}, "
                            f"L/R trigger={current_packet.left_trigger:.2f}/{current_packet.right_trigger:.2f}, "
                            f"L/R axis={current_packet.left_axis[0]:.2f},{current_packet.left_axis[1]:.2f}/"
                            f"{current_packet.right_axis[0]:.2f},{current_packet.right_axis[1]:.2f}, "
                            f"active={controller.active}, locomotion={controller.locomotion_enabled}."
                        )
                    last_input_status_s = now
                    last_input_packet_count = current_count
                    next_input_status = now + 5.0
                effective_command = command if fresh else self._standby_command()
                active = controller.active and fresh
                if step_index % self._control_decimation == 0:
                    self.update_policy(effective_command, active=active)
                self._apply_pd()
                self._mujoco.mj_step(self.model, self.data)
                if active_frame_tick is not None:
                    active_frame_tick = invoke_auxiliary_callback(
                        "frame_tick", active_frame_tick
                    )
                if viewer is not None and step_index % self._control_decimation == 0:
                    viewer.sync()
                step_index += 1
                if realtime:
                    next_step += 1.0 / self._sim_hz
                    sleep_s = next_step - time.monotonic()
                    if sleep_s > 0.0:
                        time.sleep(sleep_s)
                    elif -sleep_s > 0.25:
                        next_step = time.monotonic()
        finally:
            if viewer is not None:
                viewer.close()
