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

    def _world(self) -> Any:
        """Convert MuJoCo non-robot collision geoms to CuRobo cuboids."""
        mujoco = self.runtime._mujoco
        model, data = self.runtime.model, self.runtime.data
        world = self._WorldConfig()
        robot_bodies = set(self.runtime._robot_body_ids)
        for geom_id in range(model.ngeom):
            body_id = int(model.geom_bodyid[geom_id])
            if body_id in robot_bodies:
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

    def plan(self, *, object_name: str, episode_index: int, runtime: Any | None = None) -> tuple[JointWaypoint, ...]:
        runtime = runtime or self.runtime
        mujoco = runtime._mujoco
        joint_name = "cube_joint" if object_name == "cube" else f"robosuite_{object_name}_free"
        joint_id = mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise RuntimeError(f"scene is missing object joint {joint_name!r}")
        object_qpos = int(runtime.model.jnt_qposadr[joint_id])
        object_pose = np.asarray(runtime.data.qpos[object_qpos : object_qpos + 7], dtype=np.float64)
        target = object_pose[:3].copy()
        target[2] += 0.10
        place = target.copy()
        place[0] += 0.16 + 0.015 * ((episode_index % 5) - 2)
        world = self._world()
        current = runtime._current_joint_positions()
        segments: list[tuple[np.ndarray, tuple[str, ...], float, float]] = []
        for position, trigger, grip in ((target, 0.0, 0.0), (object_pose[:3], 1.0, 1.0), (target + np.array([0, 0, 0.15]), 1.0, 1.0), (place, 1.0, 1.0), (place, 0.0, 0.0)):
            relative_position, relative_quat = self._relative_pose(position, object_pose[3:7])
            trajectory, names = self._plan_segment(current, relative_position, relative_quat, world)
            segments.append((trajectory, names, trigger, grip))
            current.update(dict(zip(names, trajectory[-1], strict=True)))
        waypoints: list[JointWaypoint] = []
        for trajectory, names, trigger, grip in segments:
            for row in trajectory:
                waypoints.append(JointWaypoint(dict(zip(names, row, strict=True)), (0.0, 0.0, -0.3, 0.0, 0.0, 0.0, 1.0), trigger, grip, self._interpolation_dt))
        return tuple(waypoints)


class ScriptedPickPlacePlanner(SceneTrajectoryPlanner):
    """Generate a repeatable pick/lift/place trajectory in controller space.

    This is not a learned policy and does not claim collision-free planning for
    arbitrary layouts.  It is a safe baseline for dataset plumbing and smoke
    tests; object-specific dimensions can later be replaced by a CuRobo plan.
    """

    def plan(self, *, object_name: str, episode_index: int) -> tuple[WristWaypoint, ...]:
        # Small deterministic XY variation makes generated episodes useful for
        # initial VLA training while remaining inside the released table area.
        offset = ((episode_index % 5) - 2) * 0.015
        x = 0.05 + offset
        y = -0.10
        # The released SIMPLE-compatible scene uses this controller-space pose
        # for a reliable approach/contact configuration.
        approach = (x, y, -0.38, 0.0, 0.0, 0.0, 1.0)
        lift = (x, y, -0.25, 0.0, 0.0, 0.0, 1.0)
        place = (x + 0.16, y, -0.25, 0.0, 0.0, 0.0, 1.0)
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
