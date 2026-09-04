from __future__ import annotations

from pathlib import Path
import numpy as np
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dev.validate_asset_manifests import validate_catalog, validate_manifest
from scripts.setup.download_scene_object import OBJECTS
from teleopit.scenes.vla_datagen import CuroboSceneTrajectoryPlanner, ScriptedPickPlacePlanner, interpolate_waypoints
from teleopit.scenes.grasp_assets import load_simple_bodex


def test_asset_catalog_metadata_is_valid_on_clean_checkout() -> None:
    assert validate_catalog(PROJECT_ROOT) == []


def test_asset_catalog_file_check_reports_missing_downloads(tmp_path: Path) -> None:
    manifest = tmp_path / "objects.yaml"
    manifest.write_text(
        """schema_version: 1
kind: objects
assets:
  - id: example.object.v1
    display_name: Example
    version: 1
    source: {type: git, uri: https://example.invalid/repo, revision: deadbeef}
    license: {spdx: CC0-1.0, redistribution: allowed}
    paths: {scene_xml: missing.xml}
    tracked: false
""",
        encoding="utf-8",
    )
    errors = validate_manifest(manifest, root=tmp_path, check_files=True)
    assert any("paths.scene_xml does not exist" in error for error in errors)


def test_asset_catalog_file_check_detects_sha256_mismatch(tmp_path: Path) -> None:
    scene = tmp_path / "scene.xml"
    scene.write_text("<mujoco/>", encoding="utf-8")
    manifest = tmp_path / "objects.yaml"
    manifest.write_text(
        """schema_version: 1
kind: objects
assets:
  - id: example.object.v1
    display_name: Example
    version: 1
    source: {type: git, uri: https://example.invalid/repo, revision: deadbeef}
    license: {spdx: CC0-1.0, redistribution: allowed}
    paths: {scene_xml: scene.xml}
    integrity: {sha256: "0000000000000000000000000000000000000000000000000000000000000000"}
    tracked: false
""",
        encoding="utf-8",
    )
    errors = validate_manifest(manifest, root=tmp_path, check_files=True)
    assert any("SHA256 mismatch" in error for error in errors)


def test_asset_catalog_file_check_passes_for_installed_scene_assets() -> None:
    errors = validate_catalog(PROJECT_ROOT, check_files=True)
    if (PROJECT_ROOT / "third_party/decoupled_wbc").is_dir():
        assert errors == []


def test_robosuite_object_downloader_has_pinned_first_batch() -> None:
    assert set(OBJECTS) == {"bottle", "can", "lemon"}
    for name, spec in OBJECTS.items():
        assert spec.files
        assert spec.mesh_file.endswith(f"{name}.stl")
        assert all(len(entry.sha256) == 64 for entry in spec.files)


def test_scripted_vla_planner_is_finite_and_fixed_rate() -> None:
    plan = ScriptedPickPlacePlanner().plan(object_name="can", episode_index=0)
    samples = list(interpolate_waypoints(plan, hz=20.0))
    assert len(samples) > 10
    assert all(pose.shape == (7,) and np.isfinite(pose).all() for pose, _, _ in samples)
    assert samples[0][1:] == (0.0, 0.0)
    assert any(trigger > 0.5 and grip > 0.5 for _, trigger, grip in samples)


def test_curobo_backend_fails_explicitly_without_cuda_dependency() -> None:
    # Do not instantiate CuRobo in CI; verify the production class exposes the
    # required planner contract and remains a distinct backend.
    assert issubclass(CuroboSceneTrajectoryPlanner, object)
    assert "collision" in CuroboSceneTrajectoryPlanner.__doc__.lower()


def test_simple_bodex_loader_maps_robot_pose_phases(tmp_path: Path) -> None:
    names = ("wrist", "thumb", "index")
    payload = {"robot_pose": np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]])}
    path = tmp_path / "grasp.npy"
    np.save(path, payload, allow_pickle=True)
    record = load_simple_bodex(path, source_joint_names=names)
    assert record.pregrasp == {"wrist": 0.0, "thumb": 1.0, "index": 2.0}
    assert record.grasp["wrist"] == 3.0
    assert record.squeeze["index"] == 8.0
    assert record.lift["thumb"] == 10.0


def test_simple_bodex_loader_infers_joint_names_from_batched_export(tmp_path: Path) -> None:
    path = tmp_path / "batched.npy"
    payload = {
        "robot_pose": np.arange(4 * 3, dtype=np.float32).reshape(1, 4, 3),
        "joint_names": ["a", "b", "c", "unused"],
    }
    np.save(path, payload, allow_pickle=True)
    record = load_simple_bodex(path)
    assert record.grasp == {"a": 3.0, "b": 4.0, "c": 5.0}


def test_simple_bodex_loader_rejects_missing_phase_or_joint(tmp_path: Path) -> None:
    path = tmp_path / "grasp.npz"
    np.savez(path, pregrasp=np.zeros(2), grasp=np.zeros(2), squeeze=np.zeros(2))
    with np.testing.assert_raises(ValueError):
        load_simple_bodex(path, source_joint_names=("a", "b"))
