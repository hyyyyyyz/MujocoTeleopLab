#!/usr/bin/env python3
"""Compile a downloaded object fragment with a minimal MuJoCo wrapper."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


OBJECTS = ("bottle", "can", "lemon")


def validate_object(root: Path, object_name: str) -> None:
    import mujoco

    asset_dir = root / "assets/objects/robosuite" / object_name / "v1"
    assets = (asset_dir / "fragment_assets.xml").read_text(encoding="utf-8")
    body = (asset_dir / f"{object_name}_fragment.xml").read_text(encoding="utf-8")
    xml = f"""<mujoco model="teleopit_asset_check">
  {assets}
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.05"/>
    {body}
  </worldbody>
</mujoco>
"""
    spec = mujoco.MjSpec.from_string(xml)
    spec.modelfiledir = str(asset_dir)
    spec.meshdir = str(asset_dir)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if model.njnt != 1 or model.ngeom != 3:
        raise RuntimeError(f"unexpected generated {object_name} dimensions: njnt={model.njnt}, ngeom={model.ngeom}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("object", choices=OBJECTS, nargs="?", default="bottle")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    try:
        validate_object(args.root.resolve(), args.object)
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        print(f"Scene object validation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Robosuite {args.object} fragment loads in MuJoCo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
