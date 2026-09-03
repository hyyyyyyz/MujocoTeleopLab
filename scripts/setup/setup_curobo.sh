#!/usr/bin/env bash
set -euo pipefail

# CuRobo is an optional CUDA dependency and is not installed by the default
# scene setup. This helper prepares the isolated scene environment.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
venv_python="${CUROBO_PYTHON:-$project_root/.venv_scene/bin/python}"
curobo_root="${CUROBO_ROOT:-$project_root/third_party/curobo}"
curobo_revision="${CUROBO_REVISION:-a56e0f06db9efb99232586db18b18b323cb22c47}"

if [[ ! -x "$venv_python" ]]; then
  echo "Missing Python environment: $venv_python" >&2
  echo "Create the scene environment first with scripts/setup/setup_scene_teleop.sh." >&2
  exit 1
fi

if [[ ! -f "$curobo_root/pyproject.toml" ]]; then
  mkdir -p "$(dirname "$curobo_root")"
  git clone https://github.com/NVlabs/curobo.git "$curobo_root"
fi
git -C "$curobo_root" fetch --depth 1 origin "$curobo_revision" || true
git -C "$curobo_root" checkout --detach "$curobo_revision"

"$venv_python" -m pip install --no-build-isolation -e "$curobo_root"
"$venv_python" - <<'PY'
import torch
import curobo
if not torch.cuda.is_available():
    raise SystemExit("CuRobo installed, but torch.cuda.is_available() is false")
print("CuRobo scene-planning backend is ready")
PY
