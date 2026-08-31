#!/usr/bin/env python3
"""Pico Bridge mocap/video diagnostic entry point."""

from __future__ import annotations

import argparse
from collections import Counter
import logging
import os
from pathlib import Path
import signal
import sys
import time
import threading
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from teleopit.inputs.human_frame_validation import (  # noqa: E402
    HumanFrameValidationResult,
    validate_human_frame,
)
from teleopit.inputs.pico4_provider import Pico4InputProvider  # noqa: E402
from teleopit.inputs.pico_video import (  # noqa: E402
    PicoVideoConfig,
    PicoVideoRuntime,
    bridge_video_source,
)


logger = logging.getLogger("teleopit.tools.test_pico_bridge")

DEFAULT_BRIDGE_HOST = "0.0.0.0"
DEFAULT_BRIDGE_PORT = 63901
DEFAULT_VIDEO_SOURCE = "realsense"
DEFAULT_VIDEO_WIDTH = 1280
DEFAULT_VIDEO_HEIGHT = 720
DEFAULT_VIDEO_FPS = 30
DEFAULT_POLL_HZ = 120.0
DEFAULT_SUMMARY_INTERVAL_S = 1.0


def _fmt_vec(values: tuple[float, ...] | None) -> str:
    if values is None:
        return "None"
    return "[" + ", ".join(f"{value:.4f}" for value in values) + "]"


def _frame_stats(frame: dict[str, Any]) -> dict[str, Any]:
    positions = []
    quat_norms = []
    pelvis_pos = None
    for name, value in frame.items():
        try:
            pos, quat = value
        except Exception:
            continue
        try:
            pos_arr = np.asarray(pos, dtype=np.float64).reshape(-1)
            quat_arr = np.asarray(quat, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if pos_arr.shape[0] >= 3 and np.all(np.isfinite(pos_arr[:3])):
            positions.append(pos_arr[:3])
            if str(name) == "Pelvis":
                pelvis_pos = pos_arr[:3].copy()
        if quat_arr.size > 0 and np.all(np.isfinite(quat_arr)):
            quat_norms.append(float(np.linalg.norm(quat_arr)))

    if not positions:
        return {}

    pos = np.asarray(positions, dtype=np.float64)
    return {
        "pelvis_pos": pelvis_pos,
        "min_pos": np.min(pos, axis=0),
        "max_pos": np.max(pos, axis=0),
        "extent": np.ptp(pos, axis=0),
        "max_abs_pos": float(np.max(np.abs(pos))),
        "quat_norm_min": min(quat_norms) if quat_norms else None,
        "quat_norm_max": max(quat_norms) if quat_norms else None,
    }


def _log_invalid(seq: int, age_ms: float, result: HumanFrameValidationResult) -> None:
    logger.warning(
        "Invalid Pico body frame | seq=%s age_ms=%.1f reason=%s joint=%s "
        "max_abs_pos=%s pos=%s quat=%s detail=%s",
        seq,
        age_ms,
        result.reason,
        result.joint_name,
        f"{result.max_abs_pos:.4f}" if result.max_abs_pos is not None else "None",
        _fmt_vec(result.pos),
        _fmt_vec(result.quat),
        result.detail,
    )


def _log_summary(
    *,
    window_s: float,
    total: int,
    valid: int,
    invalid_reasons: Counter[str],
    provider_fps: float,
    last_seq: int | None,
    last_age_ms: float | None,
    last_stats: dict[str, Any],
    pushed_video_frames: int,
) -> None:
    if total <= 0:
        logger.info(
            "Pico Bridge summary | window=%.1fs samples=0 provider_fps=%.1f "
            "last_seq=%s video_frames=%d",
            window_s,
            provider_fps,
            last_seq,
            pushed_video_frames,
        )
        return

    invalid = total - valid
    reason_text = ",".join(f"{reason}:{count}" for reason, count in invalid_reasons.most_common()) or "none"
    pelvis = last_stats.get("pelvis_pos")
    extent = last_stats.get("extent")
    min_pos = last_stats.get("min_pos")
    max_pos = last_stats.get("max_pos")
    logger.info(
        "Pico Bridge summary | window=%.1fs samples=%d valid=%d invalid=%d reasons=%s "
        "provider_fps=%.1f last_seq=%s last_age_ms=%s video_frames=%d "
        "max_abs_pos=%s pelvis=%s extent=%s min=%s max=%s quat_norm=[%s,%s]",
        window_s,
        total,
        valid,
        invalid,
        reason_text,
        provider_fps,
        last_seq,
        f"{last_age_ms:.1f}" if last_age_ms is not None else "None",
        pushed_video_frames,
        f"{last_stats.get('max_abs_pos'):.4f}" if "max_abs_pos" in last_stats else "None",
        _fmt_np_vec(pelvis),
        _fmt_np_vec(extent),
        _fmt_np_vec(min_pos),
        _fmt_np_vec(max_pos),
        _fmt_float(last_stats.get("quat_norm_min")),
        _fmt_float(last_stats.get("quat_norm_max")),
    )


def _fmt_np_vec(values: Any) -> str:
    if values is None:
        return "None"
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{float(value):.4f}" for value in arr) + "]"


def _fmt_float(value: Any) -> str:
    if value is None:
        return "None"
    return f"{float(value):.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Pico Bridge body tracking and RealSense video streaming",
    )
    parser.add_argument("--bridge-host", default=DEFAULT_BRIDGE_HOST)
    parser.add_argument("--bridge-port", type=int, default=DEFAULT_BRIDGE_PORT)
    parser.add_argument(
        "--bridge-discovery",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--bridge-advertise-ip", default=None)
    parser.add_argument(
        "--video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream video to Pico; enabled by default. Use --no-video to disable it.",
    )
    parser.add_argument(
        "--video-source",
        choices=["realsense", "test-pattern"],
        default=DEFAULT_VIDEO_SOURCE,
    )
    parser.add_argument("--video-width", type=int, default=DEFAULT_VIDEO_WIDTH)
    parser.add_argument("--video-height", type=int, default=DEFAULT_VIDEO_HEIGHT)
    parser.add_argument("--video-fps", type=int, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--video-device", default=None)
    parser.add_argument(
        "--duration-s",
        type=float,
        default=0.0,
        help="Diagnostic duration; 0 means until Ctrl-C.",
    )
    args = parser.parse_args()

    if not 1 <= args.bridge_port <= 65535:
        parser.error("--bridge-port must be in [1, 65535]")
    if args.video and (args.video_width <= 0 or args.video_height <= 0 or args.video_fps <= 0):
        parser.error("--video-width, --video-height, and --video-fps must be > 0")
    if args.duration_s < 0.0:
        parser.error("--duration-s must be >= 0")
    return args


def _make_video_config(args: argparse.Namespace) -> PicoVideoConfig:
    return PicoVideoConfig(
        enabled=bool(args.video),
        source=str(args.video_source) if args.video else None,
        width=int(args.video_width),
        height=int(args.video_height),
        fps=int(args.video_fps),
        device=None if args.video_device in (None, "", "null") else str(args.video_device),
    )


def _build_provider(args: argparse.Namespace, video_cfg: PicoVideoConfig) -> Pico4InputProvider:
    return Pico4InputProvider(
        human_format="pico_bridge",
        pause_button=None,
        arms_button=None,
        bridge_host=str(args.bridge_host),
        bridge_port=int(args.bridge_port),
        bridge_discovery=bool(args.bridge_discovery),
        bridge_advertise_ip=args.bridge_advertise_ip,
        bridge_video=bridge_video_source(video_cfg),
        bridge_video_enabled=video_cfg.enabled,
    )


def _install_signal_handlers(stop_event: threading.Event) -> None:
    def _handle_signal(signum: int, _frame: Any) -> None:
        if stop_event.is_set():
            os._exit(130)
        logger.info("Received signal %s -- shutting down", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


def _start_video_runtime_async(video_runtime: PicoVideoRuntime) -> threading.Event:
    done = threading.Event()

    def _run() -> None:
        try:
            video_runtime.start()
        finally:
            done.set()

    thread = threading.Thread(target=_run, name="pico_video_start", daemon=True)
    thread.start()
    return done


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    video_cfg = _make_video_config(args)

    logger.info("Starting Pico Bridge diagnostic")
    logger.info(
        "Pico bridge | host=%s port=%s discovery=%s advertise_ip=%s",
        args.bridge_host,
        args.bridge_port,
        args.bridge_discovery,
        args.bridge_advertise_ip,
    )
    logger.info(
        "Signal check | validation=finite_values poll_hz=%.1f summary_interval_s=%.1f "
        "duration_s=%s video_enabled=%s video_source=%s",
        DEFAULT_POLL_HZ,
        DEFAULT_SUMMARY_INTERVAL_S,
        f"{args.duration_s:.1f}" if args.duration_s > 0.0 else "until Ctrl-C",
        video_cfg.enabled,
        video_cfg.source,
    )

    stop_event = threading.Event()
    _install_signal_handlers(stop_event)
    provider = _build_provider(args, video_cfg)
    video_runtime = PicoVideoRuntime(provider=provider, config=video_cfg)
    total = 0
    valid = 0
    invalid_reasons: Counter[str] = Counter()
    last_seq: int | None = None
    last_age_ms: float | None = None
    last_stats: dict[str, Any] = {}
    window_start_s = time.monotonic()
    start_s = window_start_s
    sleep_s = 1.0 / DEFAULT_POLL_HZ
    video_start_done: threading.Event | None = None

    try:
        if video_cfg.enabled:
            logger.info("Starting Pico video backend asynchronously")
            video_start_done = _start_video_runtime_async(video_runtime)
        while not stop_event.is_set():
            now = time.monotonic()
            if args.duration_s > 0.0 and now - start_s >= args.duration_s:
                break

            video_runtime.tick()
            if provider.has_frame():
                try:
                    frame, timestamp_s, seq = provider.get_frame_packet()
                except Exception:
                    logger.exception("Failed to read Pico body frame packet")
                else:
                    seq = int(seq)
                    if seq != last_seq:
                        last_seq = seq
                        last_age_ms = max((time.monotonic() - float(timestamp_s)) * 1000.0, 0.0)
                        last_stats = _frame_stats(frame)
                        result = validate_human_frame(frame)
                        total += 1
                        if result.valid:
                            valid += 1
                        else:
                            invalid_reasons[result.reason] += 1
                            _log_invalid(seq, last_age_ms, result)

            now = time.monotonic()
            if now - window_start_s >= DEFAULT_SUMMARY_INTERVAL_S:
                _log_summary(
                    window_s=now - window_start_s,
                    total=total,
                    valid=valid,
                    invalid_reasons=invalid_reasons,
                    provider_fps=float(provider.fps),
                    last_seq=last_seq,
                    last_age_ms=last_age_ms,
                    last_stats=last_stats,
                    pushed_video_frames=video_runtime.pushed_frames,
                )
                total = 0
                valid = 0
                invalid_reasons.clear()
                window_start_s = now

            if video_start_done is not None and not video_start_done.is_set() and now - start_s >= 5.0:
                logger.info("Waiting for Pico video backend to become ready in the background")
                video_start_done = None
            stop_event.wait(timeout=sleep_s)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt -- stopping Pico Bridge diagnostic")
    finally:
        video_runtime.stop()
        provider.close()


if __name__ == "__main__":
    main()
