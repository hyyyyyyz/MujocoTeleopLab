#!/usr/bin/env bash
# Install XRoboToolkit PC Service and its Python binding locally, without sudo.
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-"$PROJECT_ROOT/.venv/bin/python"}
TOOLS_DIR="$PROJECT_ROOT/.tools/xrobotoolkit"
SERVICE_DEB="$TOOLS_DIR/XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb"
SERVICE_DIR="$TOOLS_DIR/service/opt/apps/roboticsservice"
BINDING_DIR="$TOOLS_DIR/bindings"
SERVICE_URL="https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb"
BINDING_REPO="https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind.git"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  echo "Create Teleopit's .venv first, then rerun this script." >&2
  exit 1
fi
if ! command -v dpkg-deb >/dev/null; then
  echo "dpkg-deb is required to unpack the official PC Service package." >&2
  exit 1
fi
BUILD_ENV=("PATH=$PROJECT_ROOT/.venv/bin:$PATH")
if ! command -v c++ >/dev/null; then
  TOOLCHAIN_BIN="$PROJECT_ROOT/.toolchain/usr/bin"
  TOOLCHAIN_LIB="$PROJECT_ROOT/.toolchain/usr/lib/x86_64-linux-gnu"
  if [[ ! -x "$TOOLCHAIN_BIN/x86_64-linux-gnu-g++-13" ]]; then
    echo "A C++ compiler is required for xrobotoolkit_sdk (install build-essential), then rerun." >&2
    exit 1
  fi
  BUILD_BIN="$TOOLS_DIR/build-bin"
  mkdir -p "$BUILD_BIN"
  ln -sfn "$TOOLCHAIN_BIN/x86_64-linux-gnu-as" "$BUILD_BIN/as"
  ln -sfn "$TOOLCHAIN_BIN/x86_64-linux-gnu-ld.bfd" "$BUILD_BIN/ld"
  ln -sfn "$TOOLCHAIN_BIN/x86_64-linux-gnu-ar" "$BUILD_BIN/ar"
  ln -sfn "$TOOLCHAIN_BIN/x86_64-linux-gnu-ranlib" "$BUILD_BIN/ranlib"
  BUILD_ENV=(
    "PATH=$BUILD_BIN:$TOOLCHAIN_BIN:$PROJECT_ROOT/.venv/bin:$PATH"
    "LD_LIBRARY_PATH=$TOOLCHAIN_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    "CC=$TOOLCHAIN_BIN/x86_64-linux-gnu-gcc-13"
    "CXX=$TOOLCHAIN_BIN/x86_64-linux-gnu-g++-13"
  )
fi

mkdir -p "$TOOLS_DIR"
if [[ ! -f "$SERVICE_DEB" ]]; then
  curl -fL --retry 3 -o "$SERVICE_DEB" "$SERVICE_URL"
fi
if [[ ! -x "$SERVICE_DIR/RoboticsServiceProcess" ]]; then
  dpkg-deb -x "$SERVICE_DEB" "$TOOLS_DIR/service"
fi
if [[ ! -d "$BINDING_DIR/.git" ]]; then
  git clone --depth 1 "$BINDING_REPO" "$BINDING_DIR"
fi

mkdir -p "$BINDING_DIR/lib"
cp "$SERVICE_DIR/SDK/include/PXREARobotSDK.h" "$BINDING_DIR/include/PXREARobotSDK.h"
cp "$SERVICE_DIR/SDK/x64/libPXREARobotSDK.so" "$BINDING_DIR/lib/libPXREARobotSDK.so"
if ! rg -q 'BUILD_RPATH.*ORIGIN' "$BINDING_DIR/CMakeLists.txt"; then
  printf '%s\n' 'set_target_properties(xrobotoolkit_sdk PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")' >> "$BINDING_DIR/CMakeLists.txt"
fi

"$PYTHON_BIN" -m pip install cmake ninja pybind11
PYBIND_CMAKE=$(
  "$PYTHON_BIN" -c 'import pybind11; print(pybind11.get_cmake_dir())'
)
env "${BUILD_ENV[@]}" CMAKE_GENERATOR=Ninja CMAKE_PREFIX_PATH="$PYBIND_CMAKE" \
  "$PYTHON_BIN" -m pip install --force-reinstall --no-build-isolation "$BINDING_DIR"
SITE_PACKAGES=$(
  "$PYTHON_BIN" -c 'import site; print(site.getsitepackages()[0])'
)
cp "$BINDING_DIR/lib/libPXREARobotSDK.so" "$SITE_PACKAGES/"

"$PYTHON_BIN" -c 'import xrobotoolkit_sdk; print("xrobotoolkit_sdk OK")'
echo
echo "XRoboToolkit is installed locally. Start the PC service with:"
echo "  $SERVICE_DIR/runService.sh"
