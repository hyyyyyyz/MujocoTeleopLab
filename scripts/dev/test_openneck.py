#!/usr/bin/env python3
"""Exercise optional OpenNeck active-vision control."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from teleopit.inputs.pico4_provider import Pico4InputProvider  # noqa: E402
from teleopit.sim2real.neck.config import NeckConfig  # noqa: E402
from teleopit.sim2real.neck.openneck import build_neck_device  # noqa: E402
from teleopit.sim2real.neck.worker import NeckRuntime  # noqa: E402


DEFAULT_RATE_HZ = 60.0
DEFAULT_FRAME_TIMEOUT_S = 0.3
DEFAULT_PITCH_GAIN = 1.4
DEFAULT_TEST_ANGLE_DEG = 5.0
DEFAULT_HOLD_S = 0.8
DEFAULT_PICO_TIMEOUT_S = 60.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test OpenNeck active-vision control")
    parser.add_argument(
        "--mode",
        choices=["direct", "pico"],
        default="direct",
        help=(
            "direct sends a conservative fixed motion pattern to OpenNeck; "
            "pico drives OpenNeck from live Pico HMD rotation relative to Spine3."
        ),
    )
    parser.add_argument("--port", default=None, help="Optional OpenNeck serial port, for example /dev/ttyACM0")
    parser.add_argument("--config", dest="config_path", default=None, help="Optional OpenNeck calibration config path")
    parser.add_argument("--dry-run", action="store_true", help="Compute/log commands without opening OpenNeck hardware")
    parser.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ)
    parser.add_argument("--frame-timeout-s", type=float, default=DEFAULT_FRAME_TIMEOUT_S)
    parser.add_argument(
        "--angle-deg",
        type=float,
        default=DEFAULT_TEST_ANGLE_DEG,
        help="Direct-test angle magnitude in degrees. Keep this conservative.",
    )
    parser.add_argument("--hold-s", type=float, default=DEFAULT_HOLD_S, help="Seconds to hold each direct-test command")
    parser.add_argument("--duration-s", type=float, default=0.0, help="Pico mode duration; 0 means until Ctrl-C")
    parser.add_argument("--no-center-on-start", action="store_true")
    parser.add_argument("--no-center-on-shutdown", action="store_true")
    parser.add_argument("--release-on-shutdown", action="store_true")
    parser.add_argument("--dead-zone-deg", type=float, default=0.5)
    parser.add_argument("--pitch-gain", type=float, default=DEFAULT_PITCH_GAIN)
    parser.add_argument("--bridge-host", default="0.0.0.0")
    parser.add_argument("--bridge-port", type=int, default=63901)
    parser.add_argument("--bridge-discovery", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bridge-advertise-ip", default=None)
    args = parser.parse_args()
    if args.rate_hz <= 0:
        raise SystemExit("--rate-hz must be > 0")
    if args.frame_timeout_s <= 0:
        raise SystemExit("--frame-timeout-s must be > 0")
    if args.hold_s <= 0:
        raise SystemExit("--hold-s must be > 0")
    if args.duration_s < 0:
        raise SystemExit("--duration-s must be >= 0")
    if args.angle_deg <= 0.0:
        raise SystemExit("--angle-deg must be > 0")
    if not math.isfinite(args.pitch_gain) or args.pitch_gain <= 0.0:
        raise SystemExit("--pitch-gain must be finite and > 0")
    return args


def make_neck_config(args: argparse.Namespace) -> NeckConfig:
    return NeckConfig(
        enabled=True,
        driver="openneck",
        config_path=args.config_path,
        port=args.port,
        rate_hz=args.rate_hz,
        frame_timeout_s=args.frame_timeout_s,
        active_modes=("mocap",),
        dead_zone_deg=args.dead_zone_deg,
        pitch_gain=args.pitch_gain,
        center_on_start=not bool(args.no_center_on_start),
        center_on_shutdown=not bool(args.no_center_on_shutdown),
        release_on_shutdown=bool(args.release_on_shutdown),
        dry_run=bool(args.dry_run),
    )


def make_pico_provider(args: argparse.Namespace) -> Pico4InputProvider:
    return Pico4InputProvider(
        timeout=DEFAULT_PICO_TIMEOUT_S,
        pause_button=None,
        arms_button=None,
        bridge_host=args.bridge_host,
        bridge_port=args.bridge_port,
        bridge_discovery=bool(args.bridge_discovery),
        bridge_advertise_ip=args.bridge_advertise_ip,
        bridge_video=None,
        bridge_video_enabled=False,
    )


def run_direct(args: argparse.Namespace) -> None:
    cfg = make_neck_config(args)
    device = build_neck_device(cfg)
    angle_deg = float(args.angle_deg)
    pattern = [
        ("center", 0.0, 0.0),
        ("yaw left", angle_deg, 0.0),
        ("center", 0.0, 0.0),
        ("yaw right", -angle_deg, 0.0),
        ("center", 0.0, 0.0),
        ("pitch up", 0.0, angle_deg),
        ("center", 0.0, 0.0),
        ("pitch down", 0.0, -angle_deg),
        ("center", 0.0, 0.0),
    ]

    print(
        f"Testing OpenNeck direct pattern | port={args.port} dry_run={args.dry_run} "
        f"angle={angle_deg:.2f}deg",
        flush=True,
    )
    try:
        device.connect()
        if cfg.center_on_start:
            device.center()
        for label, yaw_deg, pitch_deg in pattern:
            print(f"{label}: yaw={yaw_deg:.2f}deg pitch={pitch_deg:.2f}deg", flush=True)
            device.move_deg(yaw_deg, pitch_deg)
            time.sleep(float(args.hold_s))
    except KeyboardInterrupt:
        print("Interrupted; shutting down OpenNeck", flush=True)
    finally:
        try:
            if cfg.center_on_shutdown:
                device.center()
            if cfg.release_on_shutdown:
                device.release_torque()
        finally:
            device.close()


def run_pico(args: argparse.Namespace) -> None:
    cfg = make_neck_config(args)
    provider = make_pico_provider(args)
    runtime = NeckRuntime(cfg)
    sleep_s = 1.0 / max(float(args.rate_hz), 1.0)
    deadline = time.monotonic() + float(args.duration_s) if args.duration_s > 0.0 else None
    last_seq = -1
    command_count = 0

    print(
        "Testing OpenNeck active vision from the live Pico HMD rotation relative to Spine3; "
        "press Ctrl-C to stop.",
        flush=True,
    )
    try:
        runtime.start()
        while deadline is None or time.monotonic() < deadline:
            now_s = time.monotonic()
            snapshot = provider.get_head_pose_snapshot()
            if snapshot is not None and int(snapshot.seq) != last_seq:
                command = runtime.tick(
                    hmd_rotation_wxyz=snapshot.hmd_rotation_wxyz,
                    spine3_rotation_wxyz=snapshot.spine3_rotation_wxyz,
                    pose_timestamp_s=snapshot.timestamp_s,
                    active=True,
                    now_s=now_s,
                )
                moved = command is not None
                if moved:
                    command_count += 1
                last_seq = int(snapshot.seq)
                age_ms = max((now_s - float(snapshot.timestamp_s)) * 1000.0, 0.0)
                print(
                    f"pico seq={snapshot.seq} age={age_ms:.1f}ms "
                    f"moved={moved} commands={command_count}",
                    flush=True,
                )
            elif snapshot is None:
                runtime.tick(
                    hmd_rotation_wxyz=None,
                    spine3_rotation_wxyz=None,
                    pose_timestamp_s=None,
                    active=True,
                    now_s=now_s,
                )
            time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("Interrupted; shutting down OpenNeck", flush=True)
    finally:
        runtime.close()
        provider.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    if args.mode == "direct":
        run_direct(args)
    elif args.mode == "pico":
        run_pico(args)
    else:
        raise AssertionError(f"Unhandled mode: {args.mode}")


if __name__ == "__main__":
    main()
