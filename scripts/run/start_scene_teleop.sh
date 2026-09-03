#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
bridge_pid=""
scene_pid=""
bridge_python="$project_root/.venv/bin/python"
scene_python="$project_root/.venv_scene/bin/python"

# Help is intentionally handled before any runtime/environment validation.
# Operators should be able to inspect the available options on a fresh
# checkout (or while Pico is still charging) without setting PICO_VIDEO_HOST,
# installing XRoboToolkit, or starting either child process.  Delegate to the
# Python parser when the isolated environment exists so this wrapper and the
# scene entry point keep one source of truth for option text.
if (($# > 0)); then
  for launch_arg in "$@"; do
    if [[ "$launch_arg" == "-h" || "$launch_arg" == "--help" ]]; then
      if [[ -x "$scene_python" ]]; then
        exec "$scene_python" "$project_root/scripts/run/run_scene_teleop.py" "$@"
      fi
      cat <<'EOF'
usage: start_scene_teleop.sh [scene options]

Starts the XRoboToolkit bridge and the 43-DOF MuJoCo scene runtime.
Run scripts/run/run_scene_teleop.py --help for the complete option list.
The scene environment is not installed yet; run scripts/setup/setup_scene_teleop.sh first.
EOF
      exit 0
    fi
  done
fi

# The XRoboToolkit SDK lives in the Python 3.12 bridge environment while the
# MuJoCo/WBC scene runs in Python 3.10.  Keep one source of truth for the
# localhost UDP endpoint and pass it to both processes.  Previously the
# wrapper forwarded ``--bridge-host/--bridge-port`` only to the scene process,
# leaving the bridge silently sending to its defaults (127.0.0.1:17600).
bridge_host="${SCENE_BRIDGE_HOST:-127.0.0.1}"
bridge_port="${SCENE_BRIDGE_PORT:-17600}"
bridge_host_explicit=0
bridge_port_explicit=0
# Keep bridge-only tuning options in the Python 3.12 process.  Without this
# small routing layer an operator passing (for example)
# ``--source-stale-timeout`` to the wrapper would have it forwarded to the
# Python 3.10 scene parser, which exits before either child starts.  The
# environment variables are useful for service files where adding command-line
# arguments is inconvenient; explicit flags below always take precedence.
bridge_hz="${SCENE_BRIDGE_HZ:-60}"
sdk_close_timeout="${SCENE_SDK_CLOSE_TIMEOUT:-2.0}"
source_heartbeat_interval="${SCENE_SOURCE_HEARTBEAT_INTERVAL:-0.10}"
source_stale_timeout="${SCENE_SOURCE_STALE_TIMEOUT:-1.00}"
bridge_disabled="${SCENE_NO_BRIDGE:-0}"
video_host="${PICO_VIDEO_HOST:-}"
video_port="${PICO_VIDEO_PORT:-12345}"
video_host_explicit=0
no_video_arg=0

if [[ "$bridge_disabled" != "0" && "$bridge_disabled" != "1" ]]; then
  echo "SCENE_NO_BRIDGE must be 0 or 1: $bridge_disabled" >&2
  exit 2
fi

scene_args=()
bridge_args=()
launch_args=("$@")
arg_index=0
while (( arg_index < ${#launch_args[@]} )); do
  launch_arg="${launch_args[$arg_index]}"
  case "$launch_arg" in
    --bridge-host)
      if (( arg_index + 1 >= ${#launch_args[@]} )); then
        echo "--bridge-host requires a value" >&2
        exit 2
      fi
      bridge_host="${launch_args[$((arg_index + 1))]}"
      bridge_host_explicit=1
      scene_args+=("$launch_arg" "$bridge_host")
      ((arg_index += 2))
      continue
      ;;
    --bridge-host=*)
      bridge_host="${launch_arg#*=}"
      bridge_host_explicit=1
      scene_args+=("$launch_arg")
      ((arg_index += 1))
      continue
      ;;
    --bridge-port)
      if (( arg_index + 1 >= ${#launch_args[@]} )); then
        echo "--bridge-port requires a value" >&2
        exit 2
      fi
      bridge_port="${launch_args[$((arg_index + 1))]}"
      bridge_port_explicit=1
      scene_args+=("$launch_arg" "$bridge_port")
      ((arg_index += 2))
      continue
      ;;
    --bridge-port=*)
      bridge_port="${launch_arg#*=}"
      bridge_port_explicit=1
      scene_args+=("$launch_arg")
      ((arg_index += 1))
      continue
      ;;
    --hz|--bridge-hz)
      if (( arg_index + 1 >= ${#launch_args[@]} )); then
        echo "$launch_arg requires a value" >&2
        exit 2
      fi
      bridge_hz="${launch_args[$((arg_index + 1))]}"
      ((arg_index += 2))
      continue
      ;;
    --hz=*|--bridge-hz=*)
      bridge_hz="${launch_arg#*=}"
      ((arg_index += 1))
      continue
      ;;
    --sdk-close-timeout)
      if (( arg_index + 1 >= ${#launch_args[@]} )); then
        echo "--sdk-close-timeout requires a value" >&2
        exit 2
      fi
      sdk_close_timeout="${launch_args[$((arg_index + 1))]}"
      ((arg_index += 2))
      continue
      ;;
    --sdk-close-timeout=*)
      sdk_close_timeout="${launch_arg#*=}"
      ((arg_index += 1))
      continue
      ;;
    --source-heartbeat-interval)
      if (( arg_index + 1 >= ${#launch_args[@]} )); then
        echo "--source-heartbeat-interval requires a value" >&2
        exit 2
      fi
      source_heartbeat_interval="${launch_args[$((arg_index + 1))]}"
      ((arg_index += 2))
      continue
      ;;
    --source-heartbeat-interval=*)
      source_heartbeat_interval="${launch_arg#*=}"
      ((arg_index += 1))
      continue
      ;;
    --source-stale-timeout)
      if (( arg_index + 1 >= ${#launch_args[@]} )); then
        echo "--source-stale-timeout requires a value" >&2
        exit 2
      fi
      source_stale_timeout="${launch_args[$((arg_index + 1))]}"
      ((arg_index += 2))
      continue
      ;;
    --source-stale-timeout=*)
      source_stale_timeout="${launch_arg#*=}"
      ((arg_index += 1))
      continue
      ;;
    --no-bridge)
      # Keep the isolated scene useful while Pico/XRoboToolkit is offline.
      # The scene process still binds the receiver port (and therefore keeps
      # its normal startup path), but no Python 3.12 SDK bridge is launched.
      bridge_disabled=1
      ((arg_index += 1))
      continue
      ;;
    --video-host)
      if (( arg_index + 1 >= ${#launch_args[@]} )); then
        echo "--video-host requires a value" >&2
        exit 2
      fi
      video_host="${launch_args[$((arg_index + 1))]}"
      video_host_explicit=1
      ((arg_index += 2))
      continue
      ;;
    --video-host=*)
      video_host="${launch_arg#*=}"
      video_host_explicit=1
      ((arg_index += 1))
      continue
      ;;
    --video-port)
      if (( arg_index + 1 >= ${#launch_args[@]} )); then
        echo "--video-port requires a value" >&2
        exit 2
      fi
      video_port="${launch_args[$((arg_index + 1))]}"
      ((arg_index += 2))
      continue
      ;;
    --video-port=*)
      video_port="${launch_arg#*=}"
      ((arg_index += 1))
      continue
      ;;
    --no-video)
      # Add this once through video_args below; keeping it out of
      # scene_args avoids passing duplicate flags to the Python entry point.
      no_video_arg=1
      ((arg_index += 1))
      continue
      ;;
    *)
      scene_args+=("$launch_arg")
      ((arg_index += 1))
      ;;
  esac
done

if [[ -z "${bridge_host//[[:space:]]/}" ]]; then
  echo "XR scene bridge host must not be empty" >&2
  exit 2
fi
if [[ ! "$bridge_port" =~ ^[0-9]+$ ]] || (( 10#$bridge_port < 1 || 10#$bridge_port > 65535 )); then
  echo "XR scene bridge port must be an integer in [1, 65535]: $bridge_port" >&2
  exit 2
fi

# If the endpoint came from an environment variable, make the same value
# explicit to the scene receiver.  Explicit command-line values are already
# present in ``scene_args`` and are preserved in their original position.
if (( ! bridge_host_explicit )); then
  scene_args+=(--bridge-host "$bridge_host")
fi
if (( ! bridge_port_explicit )); then
  scene_args+=(--bridge-port "$bridge_port")
fi

# A second wrapper would otherwise be able to start another scene process and
# compete for the same XR samples.  The receiver deliberately does not reuse
# the UDP address, and the explicit lock also covers a direct/older launcher
# that might bind a different endpoint.  flock releases this lock automatically
# when the wrapper exits, including after an interrupted launch.
lock_root="${XDG_RUNTIME_DIR:-/tmp}"
lock_file="$lock_root/teleopit-scene-teleop.lock"
exec 9>"$lock_file"
if ! flock -n 9; then
  echo "A scene teleoperation instance is already running (lock: $lock_file)." >&2
  echo "Stop it before starting another instance." >&2
  exit 1
fi

# Never silently reuse a stale headset address.  The Pico address can change
# after reconnecting to Wi-Fi, and a stale default makes Remote Vision remain
# in SYN-SENT while the local scene appears healthy.  Video can be disabled
# only through an explicit opt-out for headless/local-viewer runs.
video_disabled="${SCENE_NO_VIDEO:-0}"
if (( no_video_arg )); then
  video_disabled=1
fi
if [[ "$video_disabled" != "0" && "$video_disabled" != "1" ]]; then
  echo "SCENE_NO_VIDEO must be 0 or 1: $video_disabled" >&2
  exit 2
fi
# Validate the endpoint options even when video is explicitly disabled.  The
# wrapper consumes these flags before invoking Python, so postponing this
# check to the enabled branch would silently accept a typo such as
# ``--video-port 0`` and make a later re-enable unexpectedly fail.
if [[ ! "$video_port" =~ ^[0-9]+$ ]] || (( 10#$video_port < 1 || 10#$video_port > 65535 )); then
  echo "PICO_VIDEO_PORT/--video-port must be an integer in [1, 65535]: $video_port" >&2
  exit 2
fi
if [[ -n "$video_host" && -z "${video_host//[[:space:]]/}" ]]; then
  echo "--video-host requires a non-empty value" >&2
  exit 2
fi
if [[ "$video_disabled" == "1" ]]; then
  video_args=(--no-video)
elif [[ -n "$video_host" ]]; then
  video_args=(
    --video-host "$video_host"
    --video-port "$video_port"
  )
else
  echo "PICO_VIDEO_HOST or --video-host is required for scene Remote Vision." >&2
  echo "Use: PICO_VIDEO_HOST=<Pico IPv4> bash scripts/run/start_scene_teleop.sh --scene cube" >&2
  echo "Or:  bash scripts/run/start_scene_teleop.sh --video-host <Pico IPv4> --scene cube" >&2
  echo "For a local/headless run only: SCENE_NO_VIDEO=1 bash scripts/run/start_scene_teleop.sh --scene cube" >&2
  exit 2
fi

# Also catch a scene started directly (or by an older wrapper that predates
# the lock above).  Avoid killing it implicitly: report the exact PID and let
# the operator stop it deliberately.  Restrict the match to the interpreter's
# command line so this script's own validation commands cannot match it.
existing_scene_pids="$(pgrep -f -- "$scene_python[[:space:]].*scripts/run/run_scene_teleop\.py" || true)"
if [[ -n "$existing_scene_pids" ]]; then
  echo "A scene runtime is already running (PID(s): $existing_scene_pids)." >&2
  echo "Stop it before starting another instance." >&2
  exit 1
fi
if (( ! bridge_disabled )); then
  existing_bridge_pids="$(pgrep -f -- "$bridge_python[[:space:]].*scripts/run/run_scene_xr_bridge\.py" || true)"
  if [[ -n "$existing_bridge_pids" ]]; then
    echo "An XRoboToolkit scene bridge is already running (PID(s): $existing_bridge_pids)." >&2
    echo "Stop it before starting another instance." >&2
    exit 1
  fi
fi

# These options belong to the bridge process, not the isolated scene parser.
# Route environment defaults as well as explicit command-line values through
# one argument vector.  The bridge's argparse validators remain the single
# source of truth for finite/range checks (and the explicit options are still
# validated even when they came from a service environment).
if (( ! bridge_disabled )); then
  bridge_args=(
    --host "$bridge_host"
    --port "$bridge_port"
    --hz "$bridge_hz"
    --sdk-close-timeout "$sdk_close_timeout"
    --source-heartbeat-interval "$source_heartbeat_interval"
    --source-stale-timeout "$source_stale_timeout"
  )
fi

# Fail before opening MuJoCo when the Python 3.12 bridge dependency is absent.
# Without this check the bridge would exit immediately on
# ``ModuleNotFoundError: xrobotoolkit_sdk`` while the isolated scene process
# remained alive forever waiting for UDP packets, which looks like a frozen
# simulator to the operator.
if (( ! bridge_disabled )); then
  if [[ ! -x "$bridge_python" ]]; then
    echo "XRoboToolkit bridge environment is missing: $bridge_python" >&2
    echo "Run: bash scripts/setup/setup_xrobotoolkit.sh" >&2
    exit 1
  fi
  if ! "$bridge_python" -c 'import xrobotoolkit_sdk' >/dev/null 2>&1; then
    echo "xrobotoolkit_sdk is not installed in $bridge_python" >&2
    echo "Run: bash scripts/setup/setup_xrobotoolkit.sh" >&2
    exit 1
  fi
fi
if [[ ! -x "$scene_python" ]]; then
  echo "Scene environment is missing: $scene_python" >&2
  echo "Run: bash scripts/setup/setup_scene_teleop.sh" >&2
  exit 1
fi

stop_child() {
  local pid="$1"
  local label="$2"
  [[ -n "$pid" ]] || return 0
  # ``kill -0`` also succeeds for a child that has exited but is still a
  # zombie waiting to be reaped.  Treat that state as finished so shutdown
  # does not needlessly sleep through the full grace period before ``wait``.
  child_running() {
    if ! kill -0 "$1" 2>/dev/null; then
      return 1
    fi
    if [[ -r "/proc/$1/stat" ]]; then
      local process_state
      process_state="$(awk '{print $3}' "/proc/$1/stat" 2>/dev/null || true)"
      [[ "$process_state" != "Z" ]]
    else
      return 0
    fi
  }

  if child_running "$pid"; then
    # Both children have signal handlers for a graceful shutdown.  Give the
    # handler a short window to release MuJoCo/SDK resources, then use TERM
    # as a last resort so a stuck video socket cannot keep the wrapper alive.
    kill -INT "$pid" 2>/dev/null || true
    for ((wait_tick = 0; wait_tick < 20; wait_tick++)); do
      if ! child_running "$pid"; then
        break
      fi
      sleep 0.05
    done
    if child_running "$pid"; then
      echo "Stopping scene $label (PID $pid) did not finish after SIGINT; sending SIGTERM." >&2
      kill -TERM "$pid" 2>/dev/null || true
      for ((wait_tick = 0; wait_tick < 20; wait_tick++)); do
        if ! child_running "$pid"; then
          break
        fi
        sleep 0.05
      done
      if child_running "$pid"; then
        echo "Stopping scene $label (PID $pid) did not finish after SIGTERM; sending SIGKILL." >&2
        kill -KILL "$pid" 2>/dev/null || true
      fi
    fi
  fi
  wait "$pid" 2>/dev/null || true
}

cleanup_done=0
cleanup() {
  if (( cleanup_done )); then
    return 0
  fi
  cleanup_done=1
  # Stop the scene first so it cannot continue stepping while its XR input
  # bridge is being torn down.  ``stop_child`` is idempotent and also reaps a
  # child that already exited, preventing zombies when one side fails.
  stop_child "$scene_pid" "runtime"
  stop_child "$bridge_pid" "bridge"
}

on_signal() {
  local signal_status="$1"
  cleanup
  # EXIT will run after this explicit exit; the guard above keeps cleanup from
  # sending a second signal to an already-reaped child.
  exit "$signal_status"
}

trap cleanup EXIT
trap 'on_signal 130' INT
trap 'on_signal 143' TERM

if (( ! bridge_disabled )); then
  "$bridge_python" "$project_root/scripts/run/run_scene_xr_bridge.py" \
    "${bridge_args[@]}" &
  bridge_pid="$!"
else
  echo "XRoboToolkit bridge disabled; running the local scene without Pico input." >&2
fi
scene_threads="${SCENE_NUM_THREADS:-1}"
# The WBC process combines Pinocchio/OSQP, ONNX Runtime and MuJoCo.  Their
# default worker pools can oversubscribe the host (and starve the 200 Hz loop),
# so use one native worker per library by default.  Each value remains
# configurable for machines where a larger pool benchmarks better.
scene_env=(
  env
  # Keep the Remote Vision offscreen renderer on EGL.  With the default GLFW
  # backend, MuJoCo's Renderer and the onscreen viewer create GLFW contexts
  # from different threads; Wayland/GLFW can then abort in
  # _glfwPlatformCreateMutex.  The viewer still uses GLFW independently, while
  # EGL handles the camera renderer safely in its worker thread.
  "MUJOCO_GL=${SCENE_MUJOCO_GL:-egl}"
  "OMP_NUM_THREADS=${SCENE_OMP_NUM_THREADS:-$scene_threads}"
  "MKL_NUM_THREADS=${SCENE_MKL_NUM_THREADS:-$scene_threads}"
  "OPENBLAS_NUM_THREADS=${SCENE_OPENBLAS_NUM_THREADS:-$scene_threads}"
  "NUMEXPR_NUM_THREADS=${SCENE_NUMEXPR_NUM_THREADS:-$scene_threads}"
  "BLIS_NUM_THREADS=${SCENE_BLIS_NUM_THREADS:-$scene_threads}"
  PYTHONUNBUFFERED=1
)
"${scene_env[@]}" \
  "$scene_python" "$project_root/scripts/run/run_scene_teleop.py" \
  "${video_args[@]}" \
  "${scene_args[@]}" &
scene_pid="$!"

# Supervise both children.  Keeping the scene in the background is deliberate:
# Bash cannot reliably deliver a TERM/INT trap while synchronously waiting for
# a foreground child, which previously left the MuJoCo process running after
# the launcher terminal was closed.  If either process exits, tear down its
# sibling and return the failing/closing process status to the caller.
finished_pid=""
if (( bridge_disabled )); then
  if wait "$scene_pid"; then
    finished_status=0
  else
    finished_status="$?"
  fi
  echo "Scene runtime exited (status $finished_status); XRoboToolkit bridge was disabled." >&2
else
  if wait -n -p finished_pid "$scene_pid" "$bridge_pid"; then
    finished_status=0
  else
    finished_status="$?"
  fi
  if [[ "$finished_pid" == "$bridge_pid" ]]; then
    echo "XRoboToolkit scene bridge exited (status $finished_status); stopping scene runtime." >&2
  elif [[ "$finished_pid" == "$scene_pid" ]]; then
    echo "Scene runtime exited (status $finished_status); stopping XRoboToolkit bridge." >&2
  else
    echo "A scene child exited (status $finished_status); stopping remaining children." >&2
  fi
fi
exit "$finished_status"
