"""Deterministic tabletop trajectory generation for VLA dataset bootstrapping.

The first planner is deliberately dependency-light: it produces a validated
pick/lift/place wrist script that exercises the same 43-DOF scene runtime as
PICO teleoperation.  The planner protocol is kept separate so a CuRobo-backed
implementation can replace it without changing the recorder format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class WristWaypoint:
    """One controller-space right wrist target and gripper state."""

    pose: tuple[float, float, float, float, float, float, float]
    trigger: float = 0.0
    grip: float = 0.0
    duration_s: float = 1.0


@dataclass(frozen=True)
class JointWaypoint:
    """One collision-planner joint sample and optional gripper command."""

    positions: dict[str, float]
    wrist_pose: tuple[float, float, float, float, float, float, float]
    trigger: float = 0.0
    grip: float = 0.0
    duration_s: float = 0.05
    grasp: bool = False
    phase: str = "move"
    right_hand_positions: tuple[float, ...] | None = None


class KinematicObjectAttachment:
    """Track a grasp using MuJoCo contacts without changing object state.

    This class used to overwrite the free body's qpos every frame, which made
    the object a kinematic child of the wrist and removed gravity, friction,
    contact impulses, and slip.  SIMPLE's motion-planning data path keeps the
    object as a normal dynamic body and only closes the hand actuators.  The
    recorder follows the same rule: this helper is now a contact monitor and
    never writes object qpos/qvel.
    """

    def __init__(self, runtime: Any, object_name: str, *, hand_body: str = "right_wrist_yaw_link") -> None:
        mujoco = runtime._mujoco
        model = runtime.model
        joint_name = "cube_joint" if object_name == "cube" else f"robosuite_{object_name}_free"
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, hand_body)
        if joint_id < 0 or body_id < 0:
            raise ValueError(f"scene is missing object joint {joint_name!r} or hand body {hand_body!r}")
        self.runtime = runtime
        self._joint_qpos = int(model.jnt_qposadr[joint_id])
        self._joint_qvel = int(model.jnt_dofadr[joint_id])
        self._hand_body_id = int(body_id)
        self._finger_body_ids = tuple(
            int(candidate)
            for name in (
                "left_hand_index_1_link",
                "left_hand_middle_1_link",
                "left_hand_thumb_2_link",
                "right_hand_index_1_link",
                "right_hand_middle_1_link",
                "right_hand_thumb_2_link",
            )
            if (candidate := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)) >= 0
        )
        self._object_geom_ids = tuple(
            int(geom_id)
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) == int(model.jnt_bodyid[joint_id])
        )
        self._finger_geom_ids = tuple(
            int(geom_id)
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in self._finger_body_ids
        )
        self._attached = False
        self._ever_contacted = False

    @property
    def attached(self) -> bool:
        return self._attached

    @property
    def ever_contacted(self) -> bool:
        return self._ever_contacted

    def _has_finger_contact(self) -> bool:
        """Return whether a finger collision geom currently touches object."""
        data = self.runtime.data
        object_geoms = set(self._object_geom_ids)
        finger_geoms = set(self._finger_geom_ids)
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            if (int(contact.geom1) in object_geoms and int(contact.geom2) in finger_geoms) or (
                int(contact.geom2) in object_geoms and int(contact.geom1) in finger_geoms
            ):
                # A non-positive distance is a real overlap/touch in MuJoCo;
                # positive distances are merely broad-phase candidate pairs.
                if float(contact.dist) <= 1e-4:
                    return True
        return False

    def _hand_pose(self) -> tuple[np.ndarray, Rotation]:
        data = self.runtime.data
        position = np.asarray(data.xpos[self._hand_body_id], dtype=np.float64).copy()
        rotation = Rotation.from_matrix(np.asarray(data.xmat[self._hand_body_id], dtype=np.float64).reshape(3, 3))
        return position, rotation

    def _object_pose(self) -> tuple[np.ndarray, Rotation]:
        data = self.runtime.data
        position = np.asarray(data.qpos[self._joint_qpos : self._joint_qpos + 3], dtype=np.float64).copy()
        quat = np.asarray(data.qpos[self._joint_qpos + 3 : self._joint_qpos + 7], dtype=np.float64)
        rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        return position, rotation

    def try_attach(self, *, max_distance_m: float = 0.16, force: bool = False) -> bool:
        # ``max_distance_m`` is retained for API compatibility, but distance
        # alone is not a grasp.  Require an actual finger/object collision so a
        # planner cannot silently teleport or weld an object into the hand.
        del max_distance_m, force
        contacted = self._has_finger_contact()
        if contacted:
            self._attached = True
            self._ever_contacted = True
        return contacted

    def update(self) -> None:
        # Object motion is integrated exclusively by ``mj_step``.  Do not
        # call mj_forward here and never overwrite the free-joint state.
        if self._attached and not self._has_finger_contact():
            self._attached = False

    def release(self) -> None:
        self._attached = False


def place_object_on_table(
    runtime: Any,
    object_name: str,
    *,
    offset_xy: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Place one free object onto the compiled scene's tabletop."""
    mujoco = runtime._mujoco
    model, data = runtime.model, runtime.data
    joint_name = "cube_joint" if object_name == "cube" else f"robosuite_{object_name}_free"
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"scene is missing object free joint {joint_name!r}")
    object_body = int(model.jnt_bodyid[joint_id])
    object_geom_ids = [geom_id for geom_id in range(model.ngeom) if int(model.geom_bodyid[geom_id]) == object_body]
    collision_geom = next(
        (geom_id for geom_id in object_geom_ids if str(model.geom(geom_id).name).endswith("_collision")),
        object_geom_ids[0] if object_geom_ids else None,
    )
    table_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table_body")
    table_geom = next(
        (geom_id for geom_id in range(model.ngeom) if int(model.geom_bodyid[geom_id]) == table_body and str(model.geom(geom_id).name) == "table_top"),
        None,
    )
    if collision_geom is None or table_geom is None:
        raise ValueError("scene must contain object geometry and table_top")
    object_qpos = int(model.jnt_qposadr[joint_id])
    object_qvel = int(model.jnt_dofadr[joint_id])
    table_height = float(data.geom_xpos[table_geom][2] + model.geom_size[table_geom][2])
    object_position = np.asarray(data.qpos[object_qpos : object_qpos + 3], dtype=np.float64)
    table_center = np.asarray(data.geom_xpos[table_geom], dtype=np.float64)
    table_half_extents = np.asarray(model.geom_size[table_geom], dtype=np.float64)
    object_half_extents = np.asarray(model.geom_size[collision_geom], dtype=np.float64)
    margin = 0.005
    requested_xy = object_position[:2] + np.asarray(offset_xy, dtype=np.float64)
    object_position[:2] = np.clip(
        requested_xy,
        table_center[:2] - table_half_extents[:2] + object_half_extents[:2] + margin,
        table_center[:2] + table_half_extents[:2] - object_half_extents[:2] - margin,
    )
    data.qpos[object_qpos : object_qpos + 2] = object_position[:2]
    data.qpos[object_qpos + 2] = table_height + float(model.geom_size[collision_geom][2]) + 0.002
    data.qvel[object_qvel : object_qvel + 6] = 0.0
    mujoco.mj_forward(model, data)


class SceneTrajectoryPlanner:
    """Protocol-like base class for automatic scene trajectory planners."""

    def plan(self, *, object_name: str, episode_index: int) -> tuple[WristWaypoint, ...]:
        raise NotImplementedError


class CuroboSceneTrajectoryPlanner(SceneTrajectoryPlanner):
    """CuRobo MotionGen planner for collision-aware tabletop trajectories.

    CuRobo is imported lazily because it is a CUDA-only optional dependency.
    The planner consumes the live MuJoCo scene to build a collision world from
    non-robot geoms, plans approach/grasp/lift/place waypoints, and returns
    joint-space samples that the same 43-DOF PD loop executes.  No scripted
    interpolation is used by this backend.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        urdf_path: str,
        base_link: str = "pelvis",
        ee_link: str = "right_hand_palm_link",
        device: str = "cuda",
        interpolation_dt: float = 0.05,
        collision_activation_distance: float = 0.01,
    ) -> None:
        try:
            import torch
            # CuRobo releases before Warp 1.17 access the bridge as
            # ``wp.torch.device_from_torch``.  Warp 1.17 exposes these
            # functions at the top level instead, so provide the compatible
            # module alias before MotionGen constructs its collision checker.
            import warp as _warp
            if not hasattr(_warp, "torch"):
                _warp.torch = _warp
            from curobo.geom.types import Cuboid, WorldConfig
            from curobo.types.base import TensorDeviceType
            from curobo.types.math import Pose
            from curobo.types.robot import RobotConfig
            from curobo.types.state import JointState
            from curobo.wrap.reacher.motion_gen import (
                MotionGen,
                MotionGenConfig,
                MotionGenPlanConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "The --planner curobo backend requires CuRobo and a CUDA "
                "runtime. Install the optional CuRobo environment first."
            ) from exc
        if device != "cuda":
            raise ValueError("CuroboSceneTrajectoryPlanner currently requires device='cuda'")
        if not torch.cuda.is_available():
            raise RuntimeError("CuRobo scene planning requires torch.cuda.is_available() == True")
        if interpolation_dt <= 0.0 or not np.isfinite(interpolation_dt):
            raise ValueError("interpolation_dt must be positive and finite")
        self.runtime = runtime
        self._torch = torch
        self._Cuboid = Cuboid
        self._WorldConfig = WorldConfig
        self._Pose = Pose
        self._JointState = JointState
        self._TensorDeviceType = TensorDeviceType
        self._interpolation_dt = float(interpolation_dt)
        self._ee_link = ee_link
        # CuRobo's TensorDeviceType defaults to the active CUDA device and is
        # more stable across the pinned and upstream releases than passing a
        # version-specific constructor signature.
        tensor_args = TensorDeviceType()
        robot_cfg = RobotConfig.from_basic(
            str(urdf_path), base_link, ee_link, tensor_args
        )
        motion_cfg = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_model=WorldConfig(),
            interpolation_dt=self._interpolation_dt,
            interpolation_steps=1000,
            collision_cache={"obb": 64, "mesh": 16},
            collision_activation_distance=torch.tensor(
                [collision_activation_distance], device=device, dtype=torch.float32
            ),
            use_cuda_graph=False,
        )
        self._motion_gen = MotionGen(motion_cfg)
        warmup = getattr(self._motion_gen, "warmup", None)
        if callable(warmup):
            try:
                warmup(enable_graph=False)
            except TypeError:
                warmup()

    @staticmethod
    def _quat_inverse_rotate(quat_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        q = np.asarray(quat_wxyz, dtype=np.float64)
        rotation = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        return rotation.inv().apply(np.asarray(vector, dtype=np.float64))

    def _world(self, *, ignored_object_name: str | None = None) -> Any:
        """Convert MuJoCo non-robot collision geoms to CuRobo cuboids."""
        mujoco = self.runtime._mujoco
        model, data = self.runtime.model, self.runtime.data
        world = self._WorldConfig()
        robot_bodies = set(self.runtime._robot_body_ids)
        ignored_bodies: set[int] = set()
        if ignored_object_name is not None:
            joint_name = "cube_joint" if ignored_object_name == "cube" else f"robosuite_{ignored_object_name}_free"
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                object_body = int(model.jnt_bodyid[joint_id])
                ignored_bodies.update(self.runtime._descendant_body_ids(object_body))
        for geom_id in range(model.ngeom):
            body_id = int(model.geom_bodyid[geom_id])
            if body_id in robot_bodies or body_id in ignored_bodies:
                continue
            geom_type = int(model.geom_type[geom_id])
            if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
                dims = [2.0, 2.0, 0.02]
            else:
                # CuRobo's cuboid obstacle is a conservative AABB fallback
                # for MuJoCo boxes/cylinders/ellipsoids/meshes. It deliberately
                # over-approximates curved geometry so planned paths retain a
                # safety margin when the source scene has no CuRobo mesh asset.
                size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
                if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
                    mesh_id = int(model.geom_dataid[geom_id])
                    if mesh_id >= 0:
                        size = np.maximum(size, np.asarray(model.mesh_scale[mesh_id], dtype=np.float64))
                dims = np.maximum(2.0 * size, 0.01).tolist()
            position = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
            # MuJoCo exposes geom world orientation as a rotation matrix
            # (`geom_xmat`) rather than a quaternion on MjData.  Convert to
            # CuRobo's wxyz convention here so custom rotated obstacles are
            # represented correctly as well.
            from scipy.spatial.transform import Rotation

            geom_rotation = Rotation.from_matrix(
                np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
            )
            geom_xyzw = geom_rotation.as_quat()
            quaternion = np.array(
                [geom_xyzw[3], geom_xyzw[0], geom_xyzw[1], geom_xyzw[2]],
                dtype=np.float64,
            )
            position, quaternion = self._relative_pose(position, quaternion)
            world.add_obstacle(
                self._Cuboid(
                    name=f"mujoco_geom_{geom_id}",
                    pose=np.concatenate((position, quaternion)).tolist(),
                    dims=dims,
                )
            )
        return world

    def _relative_pose(self, world_position: np.ndarray, world_quat_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        root = np.asarray(self.runtime.data.qpos[self.runtime._root_qpos_adr : self.runtime._root_qpos_adr + 7], dtype=np.float64)
        position = self._quat_inverse_rotate(root[3:7], np.asarray(world_position) - root[:3])
        # Current released scenes keep the root orientation at identity during
        # tabletop generation. Preserve the full quaternion for custom scenes.
        from scipy.spatial.transform import Rotation

        root_rot = Rotation.from_quat([root[4], root[5], root[6], root[3]])
        obj_rot = Rotation.from_quat([world_quat_wxyz[1], world_quat_wxyz[2], world_quat_wxyz[3], world_quat_wxyz[0]])
        rel = root_rot.inv() * obj_rot
        xyzw = rel.as_quat()
        return position, np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)

    def _plan_segment(self, current: dict[str, float], goal_position: np.ndarray, goal_quat: np.ndarray, world: Any) -> tuple[np.ndarray, tuple[str, ...]]:
        torch = self._torch
        names = tuple(self._motion_gen.joint_names)
        missing = [name for name in names if name not in current]
        if missing:
            raise RuntimeError(f"CuRobo joint list is not present in the 43-DOF scene: {missing[:5]}")
        init = torch.tensor([current[name] for name in names], device="cuda", dtype=torch.float32).view(1, -1)
        # CuRobo's JointState.clone() calls ``joint_names.copy()``; pass a
        # list rather than the tuple exposed by MotionGen for compatibility
        # with the pinned release.
        state = self._JointState.from_position(
            self._TensorDeviceType().to_device(init), joint_names=list(names)
        )
        goal = self._Pose(
            position=torch.tensor(goal_position, device="cuda", dtype=torch.float32).view(1, 3),
            quaternion=torch.tensor(goal_quat, device="cuda", dtype=torch.float32).view(1, 4),
        )
        self._motion_gen.update_world(world)
        result = self._motion_gen.plan_single(
            state,
            goal,
            plan_config=__import__("curobo.wrap.reacher.motion_gen", fromlist=["MotionGenPlanConfig"]).MotionGenPlanConfig(
                enable_finetune_trajopt=True, num_trajopt_seeds=8, max_attempts=20
            ),
        )
        if not bool(result.success.item()):
            raise RuntimeError(f"CuRobo failed to plan segment: {result.status}")
        last_tstep = result.path_buffer_last_tstep
        if hasattr(last_tstep, "item"):
            end = int(last_tstep.item())
        else:
            first_tstep = last_tstep[0]
            end = int(first_tstep.item()) if hasattr(first_tstep, "item") else int(first_tstep)
        position = result.interpolated_plan.trim_trajectory(0, end).position
        if position.ndim == 3:
            position = position[0]
        trajectory = position.detach().cpu().numpy()
        return trajectory, names

    @staticmethod
    def episode_variation(episode_index: int) -> dict[str, float]:
        """Return bounded deterministic task variation for one episode.

        The sequence is reproducible from the episode index, but uses a
        low-discrepancy irrational phase rather than a five-value repeating
        pattern.  All offsets stay comfortably inside the released tabletop.
        """

        phase = (float(episode_index) * 0.6180339887498949) % 1.0
        phase2 = (float(episode_index) * 0.4142135623730950 + 0.17) % 1.0
        return {
            "object_dx": (phase - 0.5) * 0.12,
            "object_dy": (phase2 - 0.5) * 0.08,
            "lift": 0.08 + 0.10 * ((phase * 1.7) % 1.0),
            "place_dx": 0.12 + 0.12 * ((phase2 * 1.3) % 1.0),
            "place_dy": (phase * 0.8 % 1.0 - 0.4) * 0.12,
        }

    def plan(self, *, object_name: str, episode_index: int, runtime: Any | None = None) -> tuple[JointWaypoint, ...]:
        runtime = runtime or self.runtime
        mujoco = runtime._mujoco
        joint_name = "cube_joint" if object_name == "cube" else f"robosuite_{object_name}_free"
        joint_id = mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise RuntimeError(f"scene is missing object joint {joint_name!r}")
        object_qpos = int(runtime.model.jnt_qposadr[joint_id])
        object_pose = np.asarray(runtime.data.qpos[object_qpos : object_qpos + 7], dtype=np.float64)
        variation = self.episode_variation(episode_index)
        # Keep the object's stable orientation for the palm target.  The
        # activated-finger G1 model's wrist frame already includes the
        # tabletop rotation.
        object_rotation = Rotation.from_quat(
            [object_pose[4], object_pose[5], object_pose[6], object_pose[3]]
        )
        grasp_rotation = object_rotation
        grasp_xyzw = grasp_rotation.as_quat()
        grasp_quat = np.array(
            [grasp_xyzw[3], grasp_xyzw[0], grasp_xyzw[1], grasp_xyzw[2]],
            dtype=np.float64,
        )
        grasp = object_pose[:3].copy()
        target = grasp.copy()
        target[2] += variation["lift"]
        place = target.copy()
        place[0] += variation["place_dx"]
        place[1] += variation["place_dy"]
        # The manipulated object is deliberately excluded from the static
        # collision world.  Keeping it as an obstacle makes the grasp pose
        # itself invalid, so CuRobo falls back to a nearby contact/push path;
        # MuJoCo still supplies the real object contacts during execution.
        world = self._world(ignored_object_name=object_name)
        current = runtime._current_joint_positions()
        segments: list[tuple[np.ndarray, tuple[str, ...], float, float]] = []
        for position, trigger, grip in (
            (target, 0.0, 0.0),
            (grasp, 1.0, 1.0),
            (target + np.array([0, 0, 0.15]), 1.0, 1.0),
            (place, 1.0, 1.0),
            (place, 0.0, 0.0),
        ):
            relative_position, relative_quat = self._relative_pose(position, grasp_quat)
            trajectory, names = self._plan_segment(current, relative_position, relative_quat, world)
            segments.append((trajectory, names, trigger, grip))
            current.update(dict(zip(names, trajectory[-1], strict=True)))
        waypoints: list[JointWaypoint] = []
        # SIMPLE does not hold one binary close pose.  After reaching the
        # grasp it adds a 20-sample Dex3 squeeze stroke, then preserves that
        # stronger posture through lift and transport.  Use the same released
        # direction convention in this scene's joint order.
        close_hand = np.asarray(
            [0.02331954, -0.02398408, -0.22170663, 0.25662386, 1.3371105, 0.3085137, 0.9805285],
            dtype=np.float64,
        )
        squeeze_direction = np.asarray([0.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        squeeze_hand = close_hand + 0.2 * squeeze_direction
        phase_names = ("pregrasp", "grasp", "lift", "transport", "release")
        for segment_index, (trajectory, names, trigger, grip) in enumerate(segments):
            for row in trajectory:
                hand_target = None
                if segment_index == 1:
                    hand_target = tuple(float(value) for value in close_hand)
                elif segment_index in (2, 3):
                    hand_target = tuple(float(value) for value in squeeze_hand)
                waypoints.append(
                    JointWaypoint(
                        dict(zip(names, row, strict=True)),
                        (0.0, 0.0, -0.3, 0.0, 0.0, 0.0, 1.0),
                        trigger,
                        grip,
                        self._interpolation_dt,
                        grasp=segment_index in (1, 2, 3),
                        phase=phase_names[segment_index],
                        right_hand_positions=hand_target,
                    )
                )
            # SIMPLE inserts an explicit close/squeeze phase between the
            # approach and lift plans (ten control actions before continuing).
            # CuRobo returns only arm samples, so without this dwell the Dex3
            # fingers are still closing when the lift segment starts and the
            # dynamic object slips.  Hold the grasp target for ~0.5 s while
            # MuJoCo resolves finger-object contact and friction.
            if segment_index == 1 and trajectory.size:
                final_row = trajectory[-1]
                for squeeze_index in range(25):
                    ratio = min(1.0, float(squeeze_index + 1) / 20.0)
                    hand_target = close_hand + ratio * 0.2 * squeeze_direction
                    waypoints.append(
                        JointWaypoint(
                            dict(zip(names, final_row, strict=True)),
                            (0.0, 0.0, -0.3, 0.0, 0.0, 0.0, 1.0),
                            trigger,
                            grip,
                            self._interpolation_dt,
                            grasp=True,
                            phase="squeeze",
                            right_hand_positions=tuple(float(value) for value in hand_target),
                        )
                    )
        return tuple(waypoints)


class ScriptedPickPlacePlanner(SceneTrajectoryPlanner):
    """Generate a repeatable pick/lift/place trajectory in controller space.

    This is not a learned policy and does not claim collision-free planning for
    arbitrary layouts.  It is a safe baseline for dataset plumbing and smoke
    tests; object-specific dimensions can later be replaced by a CuRobo plan.
    """

    @staticmethod
    def episode_variation(episode_index: int) -> dict[str, float]:
        """Return bounded, repeatable placement variation.

        The scripted backend deliberately keeps the pre-grasp corridor fixed:
        unlike CuRobo it has no online IK/collision solve, so perturbing the
        wrist approach independently can miss the object and produce a push.
        Variation is therefore applied only to the post-lift placement target.
        """

        phase = (float(episode_index) * 0.6180339887498949) % 1.0
        phase2 = (float(episode_index) * 0.4142135623730950 + 0.17) % 1.0
        return {
            "place_dx": 0.12 + 0.08 * phase,
            "place_dy": (phase2 - 0.5) * 0.08,
        }

    def plan(self, *, object_name: str, episode_index: int) -> tuple[WristWaypoint, ...]:
        del object_name
        variation = self.episode_variation(episode_index)
        # Keep the approach/contact corridor identical to the validated SIMPLE
        # baseline.  This avoids turning harmless dataset variation into an
        # unvalidated open-loop grasp offset.
        x = 0.05
        y = -0.10
        # The released SIMPLE-compatible scene uses this controller-space pose
        # for a reliable approach/contact configuration.
        approach = (x, y, -0.38, 0.0, 0.0, 0.0, 1.0)
        # SIMPLE's Pico/controller frame has inverted Z: more negative input
        # raises the wrist.  The old -0.25 value therefore lowered the hand
        # after contact and only pushed the object across the tabletop.
        lift = (x, y, -1.15, 0.0, 0.0, 0.0, 1.0)
        place = (
            x + variation["place_dx"],
            y + variation["place_dy"],
            -1.15,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        return (
            WristWaypoint(approach, duration_s=2.0),
            WristWaypoint(approach, trigger=1.0, grip=1.0, duration_s=2.0),
            WristWaypoint(lift, trigger=1.0, grip=1.0, duration_s=1.5),
            WristWaypoint(place, trigger=1.0, grip=1.0, duration_s=1.5),
            WristWaypoint(place, duration_s=1.0),
        )


def interpolate_waypoints(
    waypoints: tuple[WristWaypoint, ...], *, hz: float
) -> Iterator[tuple[np.ndarray, float, float]]:
    """Yield fixed-rate controller poses with linear position interpolation."""

    if hz <= 0.0 or not np.isfinite(hz):
        raise ValueError("hz must be positive and finite")
    for index, waypoint in enumerate(waypoints):
        steps = max(1, int(round(waypoint.duration_s * hz)))
        start = np.asarray(waypoints[index - 1].pose if index else waypoint.pose, dtype=np.float64)
        end = np.asarray(waypoint.pose, dtype=np.float64)
        for ratio in np.linspace(0.0, 1.0, steps, endpoint=False):
            pose = start * (1.0 - ratio) + end * ratio
            pose[3:7] = end[3:7]
            yield pose, waypoint.trigger, waypoint.grip
    final = waypoints[-1]
    yield np.asarray(final.pose, dtype=np.float64), final.trigger, final.grip
