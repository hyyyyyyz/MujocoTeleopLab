#!/usr/bin/env python3
"""Build a 43-DOF scene around a user-supplied MuJoCo mesh.

The mesh is intentionally never copied into the repository.  This helper is
for locally mounted assets such as SIMPLE/Bodex GraspNet objects.  The output
XML records the absolute mesh path, so it should be regenerated on each
machine rather than committed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def build_scene(
    root: Path,
    *,
    mesh: Path,
    object_name: str,
    output: Path,
    collision_dir: Path | None = None,
    object_z_offset: float = 0.0,
) -> Path:
    import mujoco

    base = root / "third_party/decoupled_wbc/control/robot_model/model_data/g1/pnp_cube_43dof.xml"
    if not base.is_file():
        raise FileNotFoundError(base)
    mesh = mesh.expanduser().resolve()
    if not mesh.is_file():
        raise FileNotFoundError(mesh)
    if not object_name or not object_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("object-name must contain only letters, numbers, '_' or '-'")

    spec = mujoco.MjSpec.from_file(str(base))
    # ``to_xml`` resolves the included G1 mesh files relative to the emitted
    # XML unless these directories are carried over explicitly.
    spec.modelfiledir = str(base.parent)
    spec.meshdir = str(base.parent / "meshes")
    # Remove the cube body from the template; retaining it would add a second
    # free object and confuse both the planner and replay validator.
    cube = next((body for body in spec.worldbody.bodies if body.name == "cube_body"), None)
    if cube is None:
        raise ValueError("base scene is missing cube_body")
    spec.worldbody.bodies.remove(cube)
    mesh_name = f"robosuite_{object_name}_visual"
    spec.add_mesh(name=mesh_name, file=str(mesh))
    if collision_dir is None:
        candidate = mesh.parent
        collision_dir = candidate if any(candidate.glob("convex_piece_*.obj")) else None
    collision_meshes: list[str] = []
    if collision_dir is not None:
        for index, collision_mesh in enumerate(sorted(collision_dir.glob("convex_piece_*.obj"))):
            name = f"{object_name}_collision_{index:03d}"
            spec.add_mesh(name=name, file=str(collision_mesh))
            collision_meshes.append(name)

    table_body = next((body for body in spec.worldbody.bodies if body.name == "table_body"), None)
    if table_body is None:
        raise ValueError("base scene is missing table_body")
    table_top = next((geom for geom in table_body.geoms if geom.name == "table_top"), None)
    if table_top is None:
        raise ValueError("base scene is missing table_top")
    table_top_z = float(table_body.pos[2]) + float(table_top.pos[2]) + float(table_top.size[2])

    # SIMPLE's world_cfg pose is expressed with the table's top at z=0.  The
    # external object is translated by this scene's tabletop height while its
    # xy and wxyz orientation are retained exactly.
    body = spec.worldbody.add_body(
        name=f"robosuite_{object_name}_body",
        # SIMPLE's world_cfg uses [0.4, 0, 0.017...] for this object.  The
        # tabletop is wider than its source cuboid, so keep the source x
        # coordinate instead of recentering on this template's table body.
        # SIMPLE's world_cfg uses table-relative z=0.01700369.  Optional
        # calibration is explicit and kept outside the committed asset.
        pos=[0.4, 0.0, table_top_z + 0.01700369 + float(object_z_offset)],
        quat=[0.98979837, -0.04731221, 0.13403188, -0.00980837],
    )
    free_joint = body.add_freejoint(name=f"robosuite_{object_name}_free")
    free_joint.damping = 0.0005
    body.add_geom(
        name=f"robosuite_{object_name}_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh_name,
        rgba=[0.72, 0.52, 0.22, 1.0],
        contype=0,
        conaffinity=0,
        density=0.0,
    )
    if collision_meshes:
        for index, name in enumerate(collision_meshes):
            body.add_geom(
                name=f"robosuite_{object_name}_collision_{index:03d}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=name,
                density=120.0,
                # SIMPLE's dex grasp assumes a high-friction rubberized
                # fingertip/object contact.  Keep this physical (no weld or
                # qpos override), but expose tangential and torsional friction
                # on every convex piece so the hand can carry the object.
                friction=[1.5, 0.8, 0.25],
                condim=4,
                margin=0.001,
                solimp=[0.998, 0.998, 0.001, 0.5, 2.0],
                solref=[0.001, 1.0],
            )
    else:
        body.add_geom(
            name=f"robosuite_{object_name}_collision",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            density=120.0,
            friction=[1.5, 0.8, 0.25],
            condim=4,
            margin=0.001,
            solimp=[0.998, 0.998, 0.001, 0.5, 2.0],
            solref=[0.001, 1.0],
        )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    xml = spec.to_xml()
    output.write_text(xml, encoding="utf-8")
    mujoco.MjModel.from_xml_string(xml)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--collision-dir", type=Path, default=None)
    parser.add_argument("--object-z-offset", type=float, default=0.0)
    parser.add_argument("--object-name", default="toy_rhinocero")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        path = build_scene(
            args.root.resolve(), mesh=args.mesh, object_name=args.object_name,
            output=args.output, collision_dir=args.collision_dir,
            object_z_offset=args.object_z_offset,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"External-mesh scene build failed: {exc}", file=sys.stderr)
        return 1
    print(f"Built scene: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
