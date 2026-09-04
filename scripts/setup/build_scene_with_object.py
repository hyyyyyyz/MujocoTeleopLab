#!/usr/bin/env python3
"""Build a local 43-DOF scene by injecting a downloaded robosuite object."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


OBJECTS = {
    "bottle": ("cylinder", (0.03, 0.075, 0.075), (0.35, 0.75, 0.95, 0.8)),
    "can": ("cylinder", (0.033, 0.05, 0.05), (0.8, 0.1, 0.1, 1.0)),
    "lemon": ("ellipsoid", (0.038, 0.025, 0.03), (0.95, 0.8, 0.1, 1.0)),
}


def build_scene(root: Path, object_name: str, *, output: Path) -> Path:
    import mujoco

    base = root / "third_party/decoupled_wbc/control/robot_model/model_data/g1/pnp_cube_43dof.xml"
    if object_name not in OBJECTS:
        raise ValueError(f"unknown object {object_name!r}; choose from {', '.join(OBJECTS)}")
    collision_type, collision_size, rgba = OBJECTS[object_name]
    asset_dir = root / "assets/objects/robosuite" / object_name / "v1"
    spec = mujoco.MjSpec.from_file(str(base))
    spec.modelfiledir = str(base.parent)
    spec.meshdir = str(base.parent / "meshes")
    spec.add_mesh(name=f"robosuite_{object_name}_visual", file=str(asset_dir / "objects" / "meshes" / f"{object_name}.stl"))
    # Place the object on the tabletop instead of leaving it suspended in
    # mid-air.  The base scene's tabletop top is derived from its body/geom
    # transforms, and collision_size[2] is the object's half-height for all
    # supported primitive collision shapes.
    table_body = next((item for item in spec.worldbody.bodies if item.name == "table_body"), None)
    if table_body is None:
        raise ValueError("base scene is missing table_body")
    table_geom = next((item for item in table_body.geoms if item.name == "table_top"), None)
    if table_geom is None:
        raise ValueError("base scene is missing table_top")
    table_top_z = float(table_body.pos[2]) + float(table_geom.pos[2]) + float(table_geom.size[2])
    object_center_z = table_top_z + float(collision_size[2]) + 0.002
    body = spec.worldbody.add_body(name=f"robosuite_{object_name}_body", pos=[0.35, 0.0, object_center_z])
    body.add_freejoint(name=f"robosuite_{object_name}_free")
    body.add_geom(
        name=f"robosuite_{object_name}_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=f"robosuite_{object_name}_visual",
        contype=0,
        conaffinity=0,
        group=1,
        rgba=list(rgba),
    )
    body.add_geom(
        name=f"robosuite_{object_name}_collision",
        type=getattr(mujoco.mjtGeom, {"cylinder": "mjGEOM_CYLINDER", "ellipsoid": "mjGEOM_ELLIPSOID"}[collision_type]),
        size=list(collision_size),
        density=50.0,
        friction=[0.95, 0.3, 0.1],
        solimp=[0.998, 0.998, 0.001, 0.5, 2.0],
        solref=[0.001, 1.0],
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    xml = spec.to_xml()
    output.write_text(xml, encoding="utf-8")
    mujoco.MjModel.from_xml_string(xml)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("object", choices=tuple(OBJECTS), nargs="?", default="bottle")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args(argv)
    try:
        output = args.output or Path(f"outputs/scenes/robosuite_{args.object}_43dof.xml")
        output = build_scene(args.root.resolve(), args.object, output=args.root / output)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Scene build failed: {exc}", file=sys.stderr)
        return 1
    print(f"Built scene: {output}")
    print(f"Run: .venv_scene/bin/python scripts/run/run_scene_teleop.py --scene-xml {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
