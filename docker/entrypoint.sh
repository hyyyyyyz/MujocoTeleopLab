#!/usr/bin/env bash
set -euo pipefail

cd /workspace

ensure_scene_env() {
  # The scene stack is intentionally isolated from the main package, but the
  # environment is created inside the container (the repository is mounted at
  # /workspace, so the resulting venv is persistent across runs).  This keeps
  # scene/replay commands independent of any host-side Python installation.
  if [[ ! -x /workspace/.venv_scene/bin/python ]] || [[ ! -d /workspace/third_party/decoupled_wbc/.git ]]; then
    echo "Preparing the Python 3.10 scene environment inside the container..."
    SCENE_TORCH_PREINSTALLED=1 bash scripts/setup/setup_scene_teleop.sh
  fi
}

case "${1:-shell}" in
  shell)
    shift || true
    exec bash "$@"
    ;;
  check)
    python3.10 - <<'PY'
import sys
import torch
import mujoco
import curobo

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("NVIDIA GPU is not visible. Check the host driver and nvidia-container-toolkit.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"MuJoCo: {mujoco.__version__}")
print(f"CuRobo: {curobo.__file__}")
PY
    ;;
  vla)
    shift || true
    exec python3.10 scripts/run/generate_vla_scene_data.py --planner curobo "$@"
    ;;
  replay)
    shift || true
    ensure_scene_env
    exec .venv_scene/bin/python scripts/run/replay_vla_scene_data.py "$@"
    ;;
  scene)
    shift || true
    ensure_scene_env
    exec .venv_scene/bin/python scripts/run/run_scene_teleop.py "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
