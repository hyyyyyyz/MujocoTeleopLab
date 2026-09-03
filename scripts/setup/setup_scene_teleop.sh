#!/usr/bin/env bash
# Set up the isolated 43-DOF G1/Dex3 scene-teleoperation environment.
#
# This environment intentionally contains only the decoupled-WBC scene stack.
# The scene launchers add the repository root to PYTHONPATH themselves, so the
# main Teleopit package must stay in the regular .venv (Python 3.12 bridge)
# rather than being installed here with incompatible dependency metadata.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
scene_venv="$project_root/.venv_scene"
dwbc_root="$project_root/third_party/decoupled_wbc"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to create the Python 3.10 scene environment." >&2
  exit 1
fi

if [[ -e "$dwbc_root" && ! -d "$dwbc_root/.git" ]]; then
  echo "third_party/decoupled_wbc exists but is not a Git checkout: $dwbc_root" >&2
  echo "Move it aside or remove that incomplete directory deliberately, then rerun this script." >&2
  exit 1
fi
if [[ ! -d "$dwbc_root/.git" ]]; then
  git clone --depth 1 https://github.com/songlin/decoupled_wbc.git "$dwbc_root"
fi

# The WBC repository can be present while a checkout is incomplete (for
# example, when a clone was interrupted).  Fail here with the exact missing
# paths instead of waiting for MuJoCo/ONNX imports to fail deep in the runtime.
required_dwbc_files=(
  "$dwbc_root/control/robot_model/model_data/g1/g1_29dof_with_hand.urdf"
  # All released tabletop scenes include this activated-finger model rather
  # than the older seven-DOF hand XML.  Check the include explicitly so a
  # shallow/partial checkout fails during setup with a useful path instead of
  # producing a confusing MuJoCo XML include error at launch time.
  "$dwbc_root/control/robot_model/model_data/g1/g1_29dof_with_hand_rev_1_0_activatedfinger.xml"
  "$dwbc_root/control/robot_model/model_data/g1/pnp_cube_43dof.xml"
  "$dwbc_root/control/robot_model/model_data/g1/pnp_bottle_43dof.xml"
  "$dwbc_root/control/robot_model/model_data/g1/lift_box_43dof.xml"
  "$dwbc_root/control/main/teleop/configs/g1_29dof_gear_wbc.yaml"
  "$dwbc_root/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-Balance.onnx"
  "$dwbc_root/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-Walk.onnx"
)
missing_dwbc_files=()
for required_file in "${required_dwbc_files[@]}"; do
  if [[ ! -f "$required_file" ]]; then
    missing_dwbc_files+=("$required_file")
  fi
done
if (( ${#missing_dwbc_files[@]} > 0 )); then
  echo "decoupled_wbc checkout is missing required scene assets:" >&2
  printf '  %s\n' "${missing_dwbc_files[@]}" >&2
  echo "Fetch a complete checkout or move it aside and rerun this setup script." >&2
  exit 1
fi

if [[ ! -x "$scene_venv/bin/python" ]]; then
  venv_args=(--python 3.10)
  # CUDA Docker images keep the pinned torch/CuRobo installation in the
  # image's system interpreter.  Inherit those packages into the scene venv
  # instead of downloading a second (possibly CPU-only) torch wheel.
  if [[ "${SCENE_TORCH_PREINSTALLED:-0}" == "1" ]]; then
    venv_args+=(--system-site-packages)
  fi
  uv venv "${venv_args[@]}" "$scene_venv"
fi
scene_python_version="$("$scene_venv/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$scene_python_version" != "3.10" ]]; then
  echo "Scene environment uses Python $scene_python_version, but Python 3.10 is required: $scene_venv" >&2
  echo "Remove that environment deliberately and rerun this setup script, or choose a different path." >&2
  exit 1
fi
scene_packages=(
  'numpy==1.26.4' 'scipy==1.15.3' 'mujoco==3.3.4'
  onnxruntime pin pin-pink 'qpsolvers[osqp]' gymnasium
  pyyaml meshcat meshcat-shapes av
)
# The CUDA image already contains a pinned PyTorch/CuRobo pair.  Re-resolving
# ``torch`` from the default index here can silently replace it with a CPU or
# incompatible wheel, so Docker sets this opt-out explicitly.  Native setup
# keeps the historical behavior and installs torch as before.
if [[ "${SCENE_TORCH_PREINSTALLED:-0}" != "1" ]]; then
  scene_packages+=(torch)
fi
uv pip install --python "$scene_venv/bin/python" "${scene_packages[@]}"
uv pip install --python "$scene_venv/bin/python" --no-deps -e "$dwbc_root"

# Check the import surface used by the scene launcher immediately after the
# install.  This catches a broken wheel, an incompatible Python environment,
# or a partial editable install before the operator starts waiting for Pico.
if ! "$scene_venv/bin/python" - <<'PY'
import importlib

required_modules = (
    "numpy",
    "scipy",
    "mujoco",
    "onnxruntime",
    "torch",
    "pinocchio",
    "pink",
    "qpsolvers",
    "meshcat_shapes",
    "av",
    "decoupled_wbc",
)
for module_name in required_modules:
    importlib.import_module(module_name)
print("Scene dependency imports OK")
PY
then
  echo "Scene dependency import check failed in $scene_venv/bin/python." >&2
  echo "Re-run this setup script after correcting the reported package/import error." >&2
  exit 1
fi

echo
echo "Scene environment ready: $scene_venv"
echo "The main Teleopit package remains in .venv; scene launchers import it from the repository root."
echo "Run: PICO_VIDEO_HOST=<Pico IPv4> bash scripts/run/start_scene_teleop.sh --scene cube"
