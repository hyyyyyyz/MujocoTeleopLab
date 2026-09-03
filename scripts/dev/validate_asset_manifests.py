#!/usr/bin/env python3
"""Validate Teleopit's lightweight, licensing-aware asset manifests."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Any

import yaml


REQUIRED_ASSET_KEYS = {"id", "display_name", "version", "source", "license", "tracked"}
REQUIRED_SOURCE_KEYS = {"type", "uri", "revision"}
REQUIRED_LICENSE_KEYS = {"spdx", "redistribution"}


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"{path}: top-level document must be a mapping")
    return document


def _asset_file(asset: dict[str, Any]) -> str | None:
    """Return the primary downloaded file represented by an entry."""

    paths = asset.get("paths")
    if isinstance(paths, dict):
        for key in ("model", "scene_xml", "source_xml"):
            value = paths.get(key)
            if isinstance(value, str):
                return value
    runtime = asset.get("runtime")
    if isinstance(runtime, dict) and isinstance(runtime.get("xml"), str):
        return runtime["xml"]
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(path: Path, *, root: Path, check_files: bool = False) -> list[str]:
    """Return human-readable validation errors for one manifest."""

    errors: list[str] = []
    try:
        document = load_manifest(path)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        return [str(exc)]

    if document.get("schema_version") != 1:
        errors.append(f"{path}: schema_version must be 1")
    kind = document.get("kind")
    if not isinstance(kind, str) or not kind:
        errors.append(f"{path}: kind must be a non-empty string")
    assets = document.get("assets")
    if not isinstance(assets, list) or not assets:
        return errors + [f"{path}: assets must be a non-empty list"]

    ids: set[str] = set()
    for index, asset in enumerate(assets):
        label = f"{path} assets[{index}]"
        if not isinstance(asset, dict):
            errors.append(f"{label}: entry must be a mapping")
            continue
        missing = REQUIRED_ASSET_KEYS - asset.keys()
        errors.extend(f"{label}: missing {key}" for key in sorted(missing))
        asset_id = asset.get("id")
        if not isinstance(asset_id, str) or not asset_id or " " in asset_id:
            errors.append(f"{label}: id must be a non-empty space-free string")
        elif asset_id in ids:
            errors.append(f"{label}: duplicate id {asset_id!r}")
        else:
            ids.add(asset_id)
        if not isinstance(asset.get("version"), int) or asset["version"] < 1:
            errors.append(f"{label}: version must be a positive integer")
        if not isinstance(asset.get("tracked"), bool):
            errors.append(f"{label}: tracked must be boolean")

        source = asset.get("source")
        if not isinstance(source, dict):
            errors.append(f"{label}: source must be a mapping")
        else:
            errors.extend(f"{label}: source missing {key}" for key in sorted(REQUIRED_SOURCE_KEYS - source.keys()))
            for key in REQUIRED_SOURCE_KEYS:
                if key in source and (not isinstance(source[key], str) or not source[key]):
                    errors.append(f"{label}: source.{key} must be a non-empty string")

        license_info = asset.get("license")
        if not isinstance(license_info, dict):
            errors.append(f"{label}: license must be a mapping")
        else:
            errors.extend(f"{label}: license missing {key}" for key in sorted(REQUIRED_LICENSE_KEYS - license_info.keys()))
            spdx = license_info.get("spdx")
            if not isinstance(spdx, str) or not spdx:
                errors.append(f"{label}: license.spdx must be a non-empty string")
            redistribution = license_info.get("redistribution")
            if not isinstance(redistribution, str) or not redistribution:
                errors.append(f"{label}: license.redistribution must be a non-empty string")

        if check_files:
            paths = asset.get("paths")
            if isinstance(paths, dict):
                for name, value in paths.items():
                    if isinstance(value, str) and not (root / value).is_file():
                        errors.append(f"{label}: paths.{name} does not exist: {value}")
            runtime = asset.get("runtime")
            if isinstance(runtime, dict) and isinstance(runtime.get("xml"), str):
                xml = runtime["xml"]
                # Generated scene XML is a reproducible, ignored build
                # product. It is checked when present, but is optional on a
                # clean checkout until the scene builder is run.
                if not (root / xml).is_file() and not runtime.get("generated", False):
                    errors.append(f"{label}: runtime.xml does not exist: {xml}")
            integrity = asset.get("integrity")
            if isinstance(integrity, dict) and isinstance(integrity.get("sha256"), str):
                digest = integrity["sha256"]
                if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
                    errors.append(f"{label}: integrity.sha256 must be a 64-character hexadecimal digest")
                else:
                    primary_file = _asset_file(asset)
                    if primary_file is not None:
                        primary_path = root / primary_file
                        if primary_path.is_file() and _sha256(primary_path) != digest.lower():
                            errors.append(f"{label}: SHA256 mismatch for {primary_file}")
            license_info = asset.get("license")
            if isinstance(license_info, dict) and isinstance(license_info.get("notice_file"), str):
                notice = license_info["notice_file"]
                if not (root / notice).is_file():
                    errors.append(f"{label}: license.notice_file does not exist: {notice}")

    return errors


def validate_catalog(root: Path, *, check_files: bool = False) -> list[str]:
    manifest_dir = root / "assets" / "manifests"
    errors: list[str] = []
    manifest_paths = sorted(manifest_dir.glob("*.yaml"))
    documents: dict[str, dict[str, Any]] = {}
    all_ids: dict[str, Path] = {}
    for path in manifest_paths:
        if path.name == "README.md":
            continue
        errors.extend(validate_manifest(path, root=root, check_files=check_files))
        try:
            document = load_manifest(path)
        except (OSError, yaml.YAMLError, ValueError):
            continue
        documents[path.name] = document
        for asset in document.get("assets", []):
            if not isinstance(asset, dict) or not isinstance(asset.get("id"), str):
                continue
            asset_id = asset["id"]
            if asset_id in all_ids:
                errors.append(f"{path}: duplicate id {asset_id!r} (already in {all_ids[asset_id]})")
            else:
                all_ids[asset_id] = path

    scenes = documents.get("scenes.yaml", {})
    objects = documents.get("objects.yaml", {})
    object_ids = {
        item.get("id")
        for item in objects.get("assets", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    for index, scene in enumerate(scenes.get("assets", [])):
        if not isinstance(scene, dict):
            continue
        scene_objects = scene.get("objects", [])
        if isinstance(scene_objects, list):
            for object_id in scene_objects:
                if object_id not in object_ids:
                    errors.append(f"{manifest_dir / 'scenes.yaml'} assets[{index}]: unknown object {object_id!r}")

    if not errors and not manifest_paths:
        errors.append(f"No YAML manifests found in {manifest_dir}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--check-files", action="store_true", help="also require downloaded paths to exist")
    args = parser.parse_args(argv)
    errors = validate_catalog(args.root.resolve(), check_files=args.check_files)
    if errors:
        print("Asset manifest validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"Asset manifests OK ({args.root.resolve() / 'assets' / 'manifests'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
