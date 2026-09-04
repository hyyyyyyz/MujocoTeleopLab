#!/usr/bin/env bash
set -euo pipefail

service_root=/opt/xrobotoolkit/service/opt/apps/roboticsservice
service_log=/tmp/xrobotoolkit-pc-service.log
service_pid=""

cleanup() {
  if [[ -n "$service_pid" ]] && kill -0 "$service_pid" 2>/dev/null; then
    kill "$service_pid" 2>/dev/null || true
    wait "$service_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ ! -x "$service_root/runService.sh" ]]; then
  echo "XRoboToolkit PC Service is missing: $service_root/runService.sh" >&2
  exit 1
fi

echo "Starting XRoboToolkit PC Service..."
(
  cd "$service_root"
  exec ./runService.sh
) >"$service_log" 2>&1 &
service_pid=$!

# The service initializes asynchronously.  Give it a short bounded warm-up;
# the bridge itself will report SDK/service errors if initialization fails.
for _ in {1..20}; do
  if ! kill -0 "$service_pid" 2>/dev/null; then
    echo "XRoboToolkit PC Service exited during startup:" >&2
    sed -n '1,120p' "$service_log" >&2 || true
    exit 1
  fi
  sleep 0.25
done

set +e
/opt/xrbridge-venv/bin/python /workspace/scripts/run/run_scene_xr_bridge.py "$@"
bridge_status=$?
set -e
exit "$bridge_status"
