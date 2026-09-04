"""Adapters for external dexterous-grasp assets.

The project does not redistribute SIMPLE/Bodex/GraspNet files.  This module
only defines a small, versioned interchange format and a strict reader for
files that a user has obtained under the upstream terms.  Keeping the reader
separate from the planner lets the CuRobo/MuJoCo runtime consume the exact
``pregrasp -> grasp -> squeeze -> lift`` states used by SIMPLE without copying
its asset tree or importing its Python runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class DexGraspRecord:
    """One validated dexterous grasp in a named-joint coordinate space."""

    pregrasp: dict[str, float]
    grasp: dict[str, float]
    squeeze: dict[str, float]
    lift: dict[str, float]
    source_path: Path
    source_license: str = "external-user-supplied"


def _as_mapping(value: object, *, path: Path) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"grasp file must contain a mapping, got {type(value).__name__}: {path}")
    return value


def _phase_vector(payload: Mapping[str, object], phase: str, *, path: Path) -> np.ndarray:
    if phase not in payload:
        raise ValueError(f"grasp file is missing phase {phase!r}: {path}")
    vector = np.asarray(payload[phase], dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"grasp phase {phase!r} must be a finite non-empty vector: {path}")
    return vector


def _load_payload(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"grasp asset does not exist: {path}")
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    if path.suffix.lower() != ".npy":
        raise ValueError(f"expected a .npy or .npz grasp asset, got: {path}")
    # SIMPLE's Bodex cache is a pickled Python dict inside .npy.  Loading it is
    # intentionally opt-in and must only be used on files obtained from a
    # trusted upstream source; arbitrary pickle execution is possible.
    value = np.load(path, allow_pickle=True)
    try:
        return _as_mapping(value.item(), path=path)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"could not decode SIMPLE/Bodex .npy mapping: {path}") from exc


def _named(vector: np.ndarray, names: Sequence[str], *, phase: str, path: Path) -> dict[str, float]:
    if len(names) != vector.size:
        raise ValueError(
            f"{phase} vector has {vector.size} values but source_joint_names has {len(names)}: {path}"
        )
    if len(set(names)) != len(names) or any(not str(name) for name in names):
        raise ValueError("source_joint_names must contain unique non-empty names")
    return {str(name): float(value) for name, value in zip(names, vector, strict=True)}


def load_simple_bodex(
    path: str | Path,
    *,
    source_joint_names: Sequence[str] | None = None,
    target_joint_names: Sequence[str] | None = None,
    hand_permutation: Sequence[int] | None = None,
) -> DexGraspRecord:
    """Load SIMPLE's ``robot_pose`` phase states into named joint maps.

    SIMPLE stores the final three rows of ``robot_pose`` as grasp, squeeze and
    lift states.  Some Dex3 exports use a different hand-joint order; pass an
    explicit ``hand_permutation`` only after checking the upstream hand model.
    The permutation applies to the complete vector and is validated strictly.
    """

    asset_path = Path(path).expanduser().resolve()
    payload = _load_payload(asset_path)
    robot_pose = payload.get("robot_pose")
    if robot_pose is not None:
        poses = np.asarray(robot_pose, dtype=np.float64)
        if poses.ndim < 2 or poses.shape[0] == 0:
            raise ValueError(f"robot_pose must be a non-empty pose array, got {poses.shape}: {asset_path}")
        # SIMPLE exports a leading batch dimension, commonly [1, T, D].
        # Collapse all leading dimensions while retaining the time and joint
        # axes, so both [T, D] and [1, T, D] caches are accepted.
        if poses.ndim > 2:
            poses = poses.reshape((-1, poses.shape[-2], poses.shape[-1]))[0]
        if source_joint_names is None:
            inferred = payload.get("joint_names")
            if not isinstance(inferred, (list, tuple)):
                raise ValueError(
                    "source_joint_names is required when the Bodex file has no joint_names list: "
                    f"{asset_path}"
                )
            source_joint_names = tuple(str(name) for name in inferred[: poses.shape[-1]])
        if poses.shape[-1] != len(source_joint_names):
            raise ValueError(
                f"robot_pose must have shape [N, {len(source_joint_names)}], got {poses.shape}: {asset_path}"
            )
        if poses.shape[0] < 4:
            raise ValueError(f"robot_pose must contain at least pregrasp, grasp, squeeze and lift rows: {asset_path}")
        vectors = {
            "pregrasp": poses[0],
            "grasp": poses[-3],
            "squeeze": poses[-2],
            "lift": poses[-1],
        }
    else:
        vectors = {phase: _phase_vector(payload, phase, path=asset_path) for phase in ("pregrasp", "grasp", "squeeze", "lift")}

    if hand_permutation is not None:
        permutation = np.asarray(hand_permutation, dtype=np.int64).reshape(-1)
        if permutation.size != len(source_joint_names) or sorted(permutation.tolist()) != list(range(permutation.size)):
            raise ValueError("hand_permutation must be a permutation of all source joint indices")
        vectors = {phase: vector[permutation] for phase, vector in vectors.items()}

    if source_joint_names is None:
        raise ValueError("source_joint_names could not be inferred")
    names = tuple(str(name) for name in source_joint_names)
    records = {phase: _named(vector, names, phase=phase, path=asset_path) for phase, vector in vectors.items()}
    if target_joint_names is not None:
        target = tuple(str(name) for name in target_joint_names)
        missing = sorted(set(target).difference(names))
        if missing:
            raise ValueError(f"grasp asset does not contain target joints: {missing[:8]}")
        records = {phase: {name: records[phase][name] for name in target} for phase in records}
    return DexGraspRecord(source_path=asset_path, **records)
