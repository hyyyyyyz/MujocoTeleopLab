#!/usr/bin/env python3
"""Download licensed tabletop objects and emit MuJoCo body fragments.

The downloaded files are intentionally placed below the ignored
``assets/objects`` directory.  The manifest is the reviewable source of truth;
this script never copies a complete upstream simulator into Teleopit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROBOSUITE_REVISION = "824ac14cefcfb7ec125fe5eb2e0bad7364466154"
ROBOSUITE_BASE = (
    "https://raw.githubusercontent.com/ARISE-Initiative/robosuite/"
    f"{ROBOSUITE_REVISION}"
)


@dataclass(frozen=True)
class DownloadFile:
    relative_path: str
    sha256: str


_LICENSE_SHA256 = "177978cbece0a4c454c2aaec5b3f145b39270814874c43109da9e829c39d9cba"


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    files: tuple[DownloadFile, ...]
    mesh_file: str
    collision_type: str
    collision_size: tuple[float, ...]
    rgba: tuple[float, float, float, float]
    sites: tuple[tuple[str, tuple[float, float, float]], ...]


_BOTTLE_FILES = (
    DownloadFile("LICENSE", "177978cbece0a4c454c2aaec5b3f145b39270814874c43109da9e829c39d9cba"),
    DownloadFile("objects/bottle.xml", "aa724e4080db8bf0bfbf917d047fad952f8bcf50f205310d4bbf5eed66b06d70"),
    DownloadFile("objects/meshes/bottle.stl", "f2b6ca097b2d0d43b255fe3d605c8628117459b1d7369220ddf62ced5e84d962"),
    DownloadFile("objects/meshes/bottle.mtl", "e427f17d7d05eb45adaf1e2bd7fd249bc3fe56437ba8e348cbdf147cd384b996"),
    DownloadFile("objects/meshes/bottle.obj", "92353bf012eaea33c1e0d78d391fa61e083d53c57325e68d3410b31649c43da9"),
    DownloadFile("textures/glass.png", "9e12c4e1ed663ba3b690a701626e80189be02228f6d29c985273f15da4de423a"),
)

_CAN_FILES = (
    DownloadFile("LICENSE", _LICENSE_SHA256),
    DownloadFile("objects/can.xml", "c9401545f5e9b621d957e75432b0fc551ab94de96cd41833604080d0c16889a1"),
    DownloadFile("objects/meshes/can.stl", "cb459d8f0dcfc855aadc91815462d69ba58b4f5bcc289631043f17eccbf937d3"),
    DownloadFile("objects/meshes/can.mtl", "292e64f3385432e11d55dae641f29b6855e4fe4e2d1d48baf7e7ea218a5e9d85"),
    DownloadFile("objects/meshes/can.obj", "1904183501b49346131ed46bd1ee6fbf2c14f394c233ef9df048758c9a4b5118"),
    DownloadFile("textures/soda.png", "2774d29c6c16e1a3b3156e6679897a1695127ec59463d2de1a7ffd7f11b41b4d"),
)

_LEMON_FILES = (
    DownloadFile("LICENSE", _LICENSE_SHA256),
    DownloadFile("objects/lemon.xml", "1c377709b2e2efd185b9cf7cad5e410d255d73d545fc841aad16c363575da374"),
    DownloadFile("objects/meshes/lemon.stl", "bcaa216283267fff9621f524d609b59aad279971dbb52cb263680f03fc1f79db"),
    DownloadFile("objects/meshes/lemon.mtl", "e4457f44c4aacbe345e57c7e73ed87656f06b3dfba225e32ed427d2b79c79d34"),
    DownloadFile("objects/meshes/lemon.obj", "97cbc7113211f9525427cfd9087b891fb32eef55a5f691b1227c4f498e5bd99e"),
    DownloadFile("textures/lemon.png", "93a21938e5fe8d62a042c5591b13b9bec75d94791f7d08f375beb60fce523610"),
)

OBJECTS = {
    "bottle": ObjectSpec("bottle", _BOTTLE_FILES, "objects/meshes/bottle.stl", "cylinder", (0.03, 0.075, 0.075), (0.35, 0.75, 0.95, 0.8), (("bottom", (0.0, 0.0, -0.082)), ("top", (0.0, 0.0, 0.075)))),
    "can": ObjectSpec("can", _CAN_FILES, "objects/meshes/can.stl", "cylinder", (0.033, 0.05, 0.05), (0.8, 0.1, 0.1, 1.0), (("bottom", (0.0, 0.0, -0.06)), ("top", (0.0, 0.0, 0.04)))),
    "lemon": ObjectSpec("lemon", _LEMON_FILES, "objects/meshes/lemon.stl", "ellipsoid", (0.038, 0.025, 0.03), (0.95, 0.8, 0.1, 1.0), (("bottom", (0.0, 0.0, -0.035)), ("top", (0.0, 0.0, 0.02)))),
}


def _upstream_path(relative_path: str) -> str:
    if relative_path == "LICENSE":
        return "LICENSE"
    if relative_path.startswith("objects/"):
        return f"robosuite/models/assets/{relative_path}"
    if relative_path.startswith("textures/"):
        return f"robosuite/models/assets/{relative_path}"
    raise ValueError(f"unsupported upstream asset path: {relative_path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "Teleopit-asset-downloader/1"})
    with urlopen(request, timeout=30) as response, destination.open("wb") as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)


def _ensure_file(url: str, destination: Path, expected_sha256: str) -> None:
    if destination.is_file() and _sha256(destination) == expected_sha256:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        _download(url, temporary_path)
        actual = _sha256(temporary_path)
        if actual != expected_sha256:
            raise RuntimeError(f"SHA256 mismatch for {url}: expected {expected_sha256}, got {actual}")
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_fragment(root: Path, spec: ObjectSpec) -> Path:
    fragment = root / f"{spec.name}_fragment.xml"
    size = " ".join(f"{value:g}" for value in spec.collision_size)
    sites = "\n".join(
        f'  <site name="robosuite_{spec.name}_{suffix}" pos="{" ".join(f"{value:g}" for value in position)}" size="0.005" rgba="0 0 0 0"/>'
        for suffix, position in spec.sites
    )
    fragment.write_text(
        f'''<body name="robosuite_{spec.name}_body" pos="0 0 0">
  <joint name="robosuite_{spec.name}_free" type="free" damping="0.0005"/>
  <geom name="robosuite_{spec.name}_visual" type="mesh" mesh="{spec.name}_visual"
        contype="0" conaffinity="0" group="1" rgba="{' '.join(f'{value:g}' for value in spec.rgba)}"/>
  <geom name="robosuite_{spec.name}_collision" type="{spec.collision_type}" size="{size}"
        density="50" friction="0.95 0.3 0.1" solimp="0.998 0.998 0.001"
        solref="0.001 1"/>
{sites}
</body>
''',
        encoding="utf-8",
    )
    return fragment


def install_object(root: Path, object_name: str) -> Path:
    try:
        object_spec = OBJECTS[object_name]
    except KeyError as exc:
        raise ValueError(f"unknown object {object_name!r}; choose from {', '.join(sorted(OBJECTS))}") from exc
    destination_root = root / "assets" / "objects" / "robosuite" / object_spec.name / "v1"
    for entry in object_spec.files:
        source_path = entry.relative_path
        if source_path == "LICENSE":
            target = destination_root / "LICENSE"
        else:
            target = destination_root / source_path
        _ensure_file(f"{ROBOSUITE_BASE}/{_upstream_path(source_path)}", target, entry.sha256)
    xml_dir = destination_root
    (xml_dir / "fragment_assets.xml").write_text(
        f"""<asset>
  <mesh name="{object_spec.name}_visual" file="{object_spec.mesh_file}"/>
</asset>
""",
        encoding="utf-8",
    )
    return _write_fragment(destination_root, object_spec)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("object", choices=tuple(sorted(OBJECTS)), nargs="?", default="bottle")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    try:
        fragment = install_object(args.root.resolve(), args.object)
    except (OSError, HTTPError, URLError, RuntimeError) as exc:
        print(f"Scene object download failed: {exc}", file=sys.stderr)
        return 1
    print(f"Robosuite {args.object} installed under {fragment.parent}")
    print(f"MuJoCo body fragment: {fragment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
