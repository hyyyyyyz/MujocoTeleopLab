#!/usr/bin/env python3
"""Launch a 43-DOF SIMPLE-style table-top teleoperation scene."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from teleopit.scenes.controller import SimpleSceneController
from teleopit.scenes.runtime import SceneTeleopRuntime, scene_xml_path
from teleopit.scenes.video import (
    DEFAULT_SCENE_VIDEO_FPS,
    DEFAULT_SCENE_VIDEO_HEIGHT,
    DEFAULT_SCENE_VIDEO_WIDTH,
    SceneRemoteVision,
)
from teleopit.scenes.view_state import SceneViewState
from teleopit.scenes.xr_packet import DEFAULT_XR_BRIDGE_HOST, DEFAULT_XR_BRIDGE_PORT, SceneXRReceiver


def _port(value: str) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("must be in [1, 65535]")
    return port


def _nonnegative_finite(value: str) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a finite number") from exc
    if not math.isfinite(seconds) or seconds < 0.0:
        raise argparse.ArgumentTypeError("must be a finite number greater than or equal to zero")
    return seconds


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def _host(value: str) -> str:
    host = str(value).strip()
    if not host:
        raise argparse.ArgumentTypeError("must not be empty")
    return host


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SIMPLE-style Pico/XRoboToolkit G1 scene teleoperation")
    parser.add_argument("--scene", choices=("cube", "bottle", "box"), default="cube")
    parser.add_argument("--scene-xml", type=Path, help="Custom 43-DOF MuJoCo scene XML")
    parser.add_argument("--bridge-host", type=_host, default=DEFAULT_XR_BRIDGE_HOST)
    parser.add_argument("--bridge-port", type=_port, default=DEFAULT_XR_BRIDGE_PORT)
    parser.add_argument(
        "--no-bridge",
        action="store_true",
        help="Run only the local scene process; the shell launcher does not start XRoboToolkit input",
    )
    parser.add_argument("--headless", action="store_true", help="Do not open the local MuJoCo viewer")
    parser.add_argument("--seconds", type=_nonnegative_finite, default=0.0, help="Stop after N seconds; 0 means run until closed")
    parser.add_argument("--no-realtime", action="store_true", help="Run as fast as possible (headless smoke tests)")
    # ``None`` is important here: argparse applies a declared ``type`` to a
    # string default as well, and an empty-string default would therefore be
    # rejected by ``_host`` before the no-video/local-viewer path can start.
    parser.add_argument("--video-host", type=_host, default=None, help="Pico Remote Vision listener IPv4 address (for example 10.0.91.42)")
    parser.add_argument("--video-port", type=_port, default=12345)
    parser.add_argument(
        "--video-width",
        type=_positive_int,
        default=DEFAULT_SCENE_VIDEO_WIDTH,
        help="Rendered eye width; the transport duplicates it for the 2560-pixel stereo profile",
    )
    parser.add_argument(
        "--video-height",
        type=_positive_int,
        default=DEFAULT_SCENE_VIDEO_HEIGHT,
        help="Rendered eye height (default: 720)",
    )
    parser.add_argument(
        "--video-fps",
        type=_positive_int,
        default=DEFAULT_SCENE_VIDEO_FPS,
        help="Remote Vision frame rate (SIMPLE/ZEDMINI default: 60)",
    )
    parser.add_argument("--no-video", action="store_true", help="Do not publish the scene camera to Pico")
    parser.add_argument(
        "--input-timeout",
        type=_nonnegative_finite,
        default=0.35,
        help=(
            "seconds without a bridge packet before input is considered stale "
            "(default: 0.35)"
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    xml = args.scene_xml.resolve() if args.scene_xml else scene_xml_path(args.scene)
    runtime = SceneTeleopRuntime(scene_xml=xml, input_timeout_s=args.input_timeout)
    receiver = SceneXRReceiver(args.bridge_host, args.bridge_port)
    controller = SimpleSceneController()
    video = None
    view_state = SceneViewState()
    # Receiver/video setup can fail before the main ``runtime.run`` try block
    # (for example, a port collision or an unavailable PyAV/GL backend).  Keep
    # the constructor path transactional so a failed launch never strands the
    # UDP descriptor or a partially started Remote Vision worker.
    try:
        if args.video_host and not args.no_video:
            video = SceneRemoteVision(
                model=runtime.model,
                data=runtime.data,
                host=args.video_host,
                port=args.video_port,
                width=args.video_width,
                height=args.video_height,
                fps=args.video_fps,
                view_state=view_state,
            )
            video.start()
    except BaseException:
        if video is not None:
            try:
                video.stop()
            finally:
                receiver.close()
        else:
            receiver.close()
        raise
    print(f"Scene: {xml.name}")
    print("Controls: Left Menu + left index = walk input lock | Left Menu + right index = arm/hand teleop")
    print("          left stick = walk/strafe | right stick = turn")
    print("          triggers = hand gesture | X = squat | Y = stand | both grips = reset | B = Pico Remote Vision")

    reset_grips_were_pressed = False
    last_input_session_id: str | None = None
    last_input_timestamp_s: float | None = None

    def on_input(packet, command) -> None:
        # Keep HMD tracking independent from arm activation.  Remote Vision
        # consumes the latest pose from the thread-safe state in its renderer;
        # no camera/network work is done in this callback.
        nonlocal last_input_session_id, last_input_timestamp_s, reset_grips_were_pressed
        # ``SceneXRReceiver`` accepts a fresh session when the Python 3.12
        # bridge restarts.  Controller/wrist calibration is reset by the
        # scene runtime at that hand-off; re-anchor the visual neutral pose as
        # well, otherwise the first headset sample from the new bridge would
        # be interpreted relative to the old process and could jerk Remote
        # Vision by the operator's entire reconnect orientation.
        if last_input_session_id is not None and packet.session_id != last_input_session_id:
            view_state.reset_head_reference()
        last_input_session_id = packet.session_id
        # ``SceneTeleopRuntime`` drops arm calibration after this same packet
        # gap.  Re-anchor Remote Vision at the boundary too: otherwise moving
        # the headset while the bridge/PC service is disconnected would make
        # the first recovered frame jump by the entire accumulated head turn.
        if (
            last_input_timestamp_s is not None
            and packet.timestamp_s - last_input_timestamp_s > args.input_timeout
        ):
            view_state.reset_head_reference()
        last_input_timestamp_s = packet.timestamp_s
        reset_grips_pressed = packet.left_grip > 0.5 and packet.right_grip > 0.5
        if command.reset_requested and not reset_grips_were_pressed:
            # A scene reset also starts a fresh visual episode.  Re-anchor the
            # HMD before storing this same sample so a reset made while the
            # operator is looking aside does not leave the next episode with
            # a stale camera offset.  Use an edge here because the controller
            # intentionally reports only one reset event while both grips are
            # held; repeatedly calibrating every frame would freeze head-look
            # until the grips were released.
            view_state.reset_head_reference()
        reset_grips_were_pressed = reset_grips_pressed
        view_state.set_head_pose(packet.head_pose)
        mode = view_state.update_b_button(packet.b)
        if mode is not None:
            # SIMPLE's B mapping is performed by the Pico Remote Vision app.
            # The host keeps sending the fixed side-by-side ZEDMINI frame and
            # logs the edge so operators can verify that the press arrived.
            print(f"Scene Remote Vision view toggle requested: {mode} (Pico app applies layout)")

    try:
        runtime.run(
            receiver=receiver,
            controller=controller,
            onscreen=not args.headless,
            duration_s=args.seconds,
            realtime=not args.no_realtime,
            frame_tick=video.tick if video is not None else None,
            input_tick=on_input,
        )
    except KeyboardInterrupt:
        pass
    finally:
        receiver.close()
        if video is not None:
            video.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
