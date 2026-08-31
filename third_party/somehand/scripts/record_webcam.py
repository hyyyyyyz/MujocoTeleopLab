"""Automatically record a fixed-length webcam clip."""

import argparse
from datetime import datetime
from pathlib import Path
import time

import cv2


def default_output_path(output: str | None, output_dir: str, suffix: str) -> Path:
    if output:
        path = Path(output)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(output_dir) / f"webcam_{stamp}.{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def open_camera(camera_index: int, width: int | None, height: int | None, requested_fps: float | None) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {camera_index}")

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if requested_fps is not None:
        cap.set(cv2.CAP_PROP_FPS, requested_fps)
    return cap


def get_writer(cap: cv2.VideoCapture, output_path: Path, codec: str, fallback_fps: float) -> tuple[cv2.VideoWriter, tuple[int, int], float]:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 1e-3 or fps > 240.0:
        fps = fallback_fps

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for: {output_path}")
    return writer, (width, height), fps


def draw_overlay(frame, lines: list[str], color=(0, 255, 0)):
    y = 32
    for line in lines:
        cv2.putText(
            frame,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 32


def main():
    parser = argparse.ArgumentParser(description="Automatically record a webcam clip")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    parser.add_argument("--output", default=None, help="Output video path, default is recordings/webcam_<timestamp>.mp4")
    parser.add_argument("--output-dir", default="recordings", help="Directory used when --output is not provided")
    parser.add_argument("--duration", type=float, default=10.0, help="Recording duration in seconds")
    parser.add_argument("--countdown", type=float, default=3.0, help="Countdown before recording starts")
    parser.add_argument("--warmup", type=float, default=1.0, help="Camera warmup time before countdown")
    parser.add_argument("--width", type=int, default=None, help="Requested capture width")
    parser.add_argument("--height", type=int, default=None, help="Requested capture height")
    parser.add_argument("--fps", type=float, default=30.0, help="Fallback FPS for the output file")
    parser.add_argument("--codec", default="mp4v", help="FourCC codec, e.g. mp4v or XVID")
    parser.add_argument("--mirror", action="store_true", help="Mirror the saved frames horizontally")
    parser.add_argument("--no-preview", action="store_true", help="Disable preview window")
    args = parser.parse_args()

    suffix = "avi" if args.codec.upper() == "XVID" else "mp4"
    output_path = default_output_path(args.output, args.output_dir, suffix)

    cap = open_camera(args.camera, args.width, args.height, args.fps)
    writer = None
    preview_name = "Auto Webcam Recorder"

    try:
        warmup_deadline = time.monotonic() + max(args.warmup, 0.0)
        countdown_deadline = warmup_deadline + max(args.countdown, 0.0)
        recording_start = countdown_deadline
        recording_end = recording_start + max(args.duration, 0.0)
        frames_written = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read frame from camera")

            now = time.monotonic()
            save_frame = cv2.flip(frame, 1) if args.mirror else frame
            preview_frame = save_frame.copy()

            if writer is None:
                writer, (width, height), actual_fps = get_writer(cap, output_path, args.codec, args.fps)
                print(f"Output: {output_path}")
                print(f"Resolution: {width}x{height} | FPS: {actual_fps:.2f} | Codec: {args.codec}")
                print(f"Warmup: {args.warmup:.1f}s | Countdown: {args.countdown:.1f}s | Duration: {args.duration:.1f}s")
                print("Press 'q' to cancel early.")

            if now < warmup_deadline:
                draw_overlay(preview_frame, ["Warming up camera..."], color=(0, 255, 255))
            elif now < countdown_deadline:
                remaining = max(0.0, countdown_deadline - now)
                draw_overlay(
                    preview_frame,
                    ["Get ready", f"Recording starts in {remaining:.1f}s"],
                    color=(0, 255, 255),
                )
            elif now < recording_end:
                writer.write(save_frame)
                frames_written += 1
                elapsed = now - recording_start
                remaining = max(0.0, recording_end - now)
                draw_overlay(
                    preview_frame,
                    [
                        "REC",
                        f"Elapsed: {elapsed:.1f}s / {args.duration:.1f}s",
                        f"Remaining: {remaining:.1f}s",
                        f"Frames: {frames_written}",
                    ],
                    color=(0, 64, 255),
                )
            else:
                draw_overlay(
                    preview_frame,
                    ["Recording complete", f"Saved: {output_path.name}", f"Frames: {frames_written}"],
                    color=(0, 255, 0),
                )
                if not args.no_preview:
                    cv2.imshow(preview_name, preview_frame)
                    cv2.waitKey(800)
                break

            if not args.no_preview:
                cv2.imshow(preview_name, preview_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Recording cancelled by user.")
                    break

        print(f"Saved {frames_written} frames to {output_path}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
