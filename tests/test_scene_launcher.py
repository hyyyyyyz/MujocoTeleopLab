from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


def _run_launcher(project_root: Path, *args: str, **env_overrides: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(env_overrides)
    return subprocess.run(
        ["bash", str(project_root / "scripts/run/start_scene_teleop.sh"), *args],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("port", ["0", "65536", "not-a-port"])
def test_scene_launcher_validates_video_port_when_video_disabled(project_root: Path, port: str) -> None:
    """Wrapper-consumed video options must not bypass validation with --no-video."""
    result = _run_launcher(
        project_root,
        "--no-video",
        "--no-bridge",
        "--headless",
        "--video-port",
        port,
    )

    assert result.returncode == 2
    assert "video-port" in result.stderr


def test_scene_launcher_rejects_explicit_empty_video_host_when_disabled(project_root: Path) -> None:
    result = _run_launcher(
        project_root,
        "--no-video",
        "--no-bridge",
        "--headless",
        "--video-host=   ",
    )

    assert result.returncode == 2
    assert "video-host" in result.stderr


def test_scene_launcher_rejects_whitespace_bridge_host(project_root: Path) -> None:
    result = _run_launcher(
        project_root,
        "--no-video",
        "--no-bridge",
        "--headless",
        SCENE_BRIDGE_HOST=" \t\n",
    )

    assert result.returncode == 2
    assert "bridge host" in result.stderr


def test_scene_launcher_rejects_whitespace_video_host_from_environment(project_root: Path) -> None:
    result = _run_launcher(
        project_root,
        "--no-video",
        "--no-bridge",
        "--headless",
        PICO_VIDEO_HOST=" \t",
    )

    assert result.returncode == 2
    assert "video-host" in result.stderr
