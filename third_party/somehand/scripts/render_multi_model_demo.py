#!/usr/bin/env python3
"""Render a split-screen multi-model demo video from existing full renders."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RENDERS_ROOT = ROOT_DIR / "recordings" / "full_renders"
DEFAULT_OUTPUTS = {
    "zh": RENDERS_ROOT / "demo" / "multi_model_demo_preview.mp4",
    "en": RENDERS_ROOT / "demo" / "multi_model_demo_preview_en.mp4",
}
FONT_PATH = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
LATIN_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

PANEL_WIDTH = 720
PANEL_HEIGHT = 810
OUTPUT_WIDTH = PANEL_WIDTH * 2
OUTPUT_HEIGHT = PANEL_HEIGHT
OUTPUT_FPS = 30

SECTION_WINDOWS = {
    "left": (10.0, 50.0),
    "right": (0.0, 25.79),
    "bihand": (10.0, 80.0),
}

SECTION_LABELS = {
    "zh": {
        "left": "左手",
        "right": "右手",
        "bihand": "双手",
    },
    "en": {
        "left": "Left Hand",
        "right": "Right Hand",
        "bihand": "Bi-Hand",
    },
}

LANDMARK_PATHS = {
    "left": RENDERS_ROOT / "landmarks" / "left" / "pico_left_landmarks.mp4",
    "right": RENDERS_ROOT / "landmarks" / "right" / "pico_right_landmarks.mp4",
    "bihand": RENDERS_ROOT / "landmarks" / "bihand" / "pico_bihand_landmarks.mp4",
}


@dataclass(frozen=True)
class ModelLabel:
    company: str
    model: str


@dataclass(frozen=True)
class SegmentSpec:
    section: str
    code: str
    start: float
    end: float
    language: str

    @property
    def robot_path(self) -> Path:
        return RENDERS_ROOT / self.section / f"{self.code}_{self.section}.mp4"

    @property
    def landmarks_path(self) -> Path:
        return LANDMARK_PATHS[self.section]

    @property
    def section_label(self) -> str:
        return SECTION_LABELS[self.language][self.section]

    @property
    def company_text(self) -> str:
        return MODEL_LABELS[self.language][self.code].company

    @property
    def model_line_text(self) -> str:
        return MODEL_LABELS[self.language][self.code].model


MODEL_LABELS = {
    "zh": {
        "linkerhand_l6": ModelLabel(company="灵心巧手", model="L6"),
        "linkerhand_l10": ModelLabel(company="灵心巧手", model="L10"),
        "linkerhand_l20": ModelLabel(company="灵心巧手", model="L20"),
        "linkerhand_l20pro": ModelLabel(company="灵心巧手", model="L20 Pro"),
        "linkerhand_l21": ModelLabel(company="灵心巧手", model="L21"),
        "linkerhand_l25": ModelLabel(company="灵心巧手", model="L25"),
        "linkerhand_l30": ModelLabel(company="灵心巧手", model="L30"),
        "linkerhand_lhg20": ModelLabel(company="灵心巧手", model="LHG20"),
        "linkerhand_o6": ModelLabel(company="灵心巧手", model="O6"),
        "linkerhand_o7": ModelLabel(company="灵心巧手", model="O7"),
        "linkerhand_t12": ModelLabel(company="灵心巧手", model="T12"),
        "dexhand021": ModelLabel(company="灵巧智能", model="DexHand021"),
        "dex5": ModelLabel(company="宇树科技", model="Dex5"),
        "inspire_dfq": ModelLabel(company="因时机器人", model="DFQ"),
        "inspire_ftp": ModelLabel(company="因时机器人", model="FTP"),
        "omnihand": ModelLabel(company="智元", model="OmniHand"),
        "revo2": ModelLabel(company="强脑科技", model="Revo2"),
        "rohand": ModelLabel(company="傲意科技", model="RoHand"),
        "sharpa_wave": ModelLabel(company="Sharpa", model="Wave 01"),
        "wujihand": ModelLabel(company="舞肌科技", model="Wuji Hand"),
    },
    "en": {
        "linkerhand_l6": ModelLabel(company="LinkerHand", model="L6"),
        "linkerhand_l10": ModelLabel(company="LinkerHand", model="L10"),
        "linkerhand_l20": ModelLabel(company="LinkerHand", model="L20"),
        "linkerhand_l20pro": ModelLabel(company="LinkerHand", model="L20 Pro"),
        "linkerhand_l21": ModelLabel(company="LinkerHand", model="L21"),
        "linkerhand_l25": ModelLabel(company="LinkerHand", model="L25"),
        "linkerhand_l30": ModelLabel(company="LinkerHand", model="L30"),
        "linkerhand_lhg20": ModelLabel(company="LinkerHand", model="LHG20"),
        "linkerhand_o6": ModelLabel(company="LinkerHand", model="O6"),
        "linkerhand_o7": ModelLabel(company="LinkerHand", model="O7"),
        "linkerhand_t12": ModelLabel(company="LinkerHand", model="T12"),
        "dexhand021": ModelLabel(company="DexRobot", model="DexHand021"),
        "dex5": ModelLabel(company="Unitree", model="Dex5"),
        "inspire_dfq": ModelLabel(company="Inspire", model="DFQ"),
        "inspire_ftp": ModelLabel(company="Inspire", model="FTP"),
        "omnihand": ModelLabel(company="OmniHand", model="OmniHand"),
        "revo2": ModelLabel(company="BrainCo", model="Revo2"),
        "rohand": ModelLabel(company="RoHand", model="RoHand"),
        "sharpa_wave": ModelLabel(company="Sharpa", model="Wave 01"),
        "wujihand": ModelLabel(company="Wuji", model="Wuji Hand"),
    },
}

SECTION_CODES = {
    "left": [
        "linkerhand_l20",
        "dexhand021",
        "dex5",
        "inspire_dfq",
        "omnihand",
        "revo2",
        "rohand",
        "wujihand",
    ],
    "right": [
        "linkerhand_o6",
        "inspire_ftp",
        "sharpa_wave",
        "linkerhand_o7",
        "linkerhand_l25",
    ],
    "bihand": [
        "linkerhand_l6",
        "linkerhand_l10",
        "linkerhand_l20",
        "linkerhand_l21",
        "linkerhand_l25",
        "linkerhand_o6",
        "linkerhand_o7",
        "dexhand021",
        "dex5",
        "inspire_dfq",
        "inspire_ftp",
        "omnihand",
        "revo2",
        "rohand",
        "sharpa_wave",
        "wujihand",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a split-screen multi-model demo preview")
    parser.add_argument(
        "--language",
        choices=["zh", "en"],
        default="zh",
        help="Output label language. Default: zh",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output MP4 path. Defaults depend on --language",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep intermediate segment files next to the output video",
    )
    return parser.parse_args()


def build_segments(*, language: str) -> list[SegmentSpec]:
    segments: list[SegmentSpec] = []
    for section, codes in SECTION_CODES.items():
        start_time, end_time = SECTION_WINDOWS[section]
        duration = (end_time - start_time) / len(codes)
        for index, code in enumerate(codes):
            segment_start = start_time + index * duration
            segment_end = start_time + (index + 1) * duration
            segments.append(
                SegmentSpec(
                    section=section,
                    code=code,
                    start=segment_start,
                    end=segment_end,
                    language=language,
                )
            )
    return segments


def ensure_requirements(segments: list[SegmentSpec]) -> None:
    missing_paths: list[Path] = []
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not available")
    for font_path in (FONT_PATH, LATIN_FONT_PATH):
        if not font_path.exists():
            raise RuntimeError(f"Missing font file: {font_path}")
    for path in LANDMARK_PATHS.values():
        if not path.exists():
            missing_paths.append(path)
    for segment in segments:
        if not segment.robot_path.exists():
            missing_paths.append(segment.robot_path)
    if missing_paths:
        joined = "\n".join(str(path) for path in missing_paths)
        raise RuntimeError(f"Missing input files:\n{joined}")


def single_robot_filter() -> str:
    return (
        "crop=320:320:(iw-320)/2:(ih-320)/2,"
        f"scale={PANEL_WIDTH}:{PANEL_HEIGHT}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={PANEL_WIDTH}:{PANEL_HEIGHT}"
    )


def bihand_robot_filter() -> str:
    return (
        "crop=390:430:(iw-390)/2:(ih-430)/2,"
        f"scale={PANEL_WIDTH}:{PANEL_HEIGHT}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={PANEL_WIDTH}:{PANEL_HEIGHT}"
    )


def landmark_filter() -> str:
    return (
        f"scale={PANEL_WIDTH}:{PANEL_HEIGHT}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={PANEL_WIDTH}:{PANEL_HEIGHT}"
    )


def render_segment(segment: SegmentSpec, index: int, temp_dir: Path) -> Path:
    segment_path = temp_dir / f"segment_{index:02d}.mp4"
    company_text_path = temp_dir / f"company_{index:02d}.txt"
    company_text_path.write_text(segment.company_text, encoding="utf-8")
    model_text_path = temp_dir / f"model_{index:02d}.txt"
    model_text_path.write_text(segment.model_line_text, encoding="utf-8")
    company_font_path = FONT_PATH if not segment.company_text.isascii() else LATIN_FONT_PATH

    robot_filter = single_robot_filter() if segment.section != "bihand" else bihand_robot_filter()
    filter_complex = (
        f"[0:v]{landmark_filter()}[landmarks];"
        f"[1:v]{robot_filter}[robot];"
        "[landmarks][robot]hstack=inputs=2[stack];"
        f"[stack]fps={OUTPUT_FPS},"
        f"drawtext=fontfile={FONT_PATH if not segment.section_label.isascii() else LATIN_FONT_PATH}:text='{segment.section_label}':fontcolor=white:bordercolor=black:borderw=2:fontsize=28:x=40:y=32,"
        f"drawtext=fontfile={company_font_path}:textfile={company_text_path}:fontcolor=white:bordercolor=black:borderw=2:fontsize=32:x=w-text_w-40:y={OUTPUT_HEIGHT - 92},"
        f"drawtext=fontfile={LATIN_FONT_PATH}:textfile={model_text_path}:fontcolor=white:bordercolor=black:borderw=2:fontsize=30:x=w-text_w-40:y={OUTPUT_HEIGHT - 50}"
        "[video]"
    )

    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{segment.start:.3f}",
        "-to",
        f"{segment.end:.3f}",
        "-i",
        str(segment.landmarks_path),
        "-ss",
        f"{segment.start:.3f}",
        "-to",
        f"{segment.end:.3f}",
        "-i",
        str(segment.robot_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[video]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(segment_path),
    ]
    subprocess.run(command, check=True)
    return segment_path


def concat_segments(segment_paths: list[Path], output_path: Path, temp_dir: Path) -> None:
    concat_list_path = temp_dir / "segments.txt"
    concat_list_path.write_text(
        "".join(f"file '{path.resolve()}'\n" for path in segment_paths),
        encoding="utf-8",
    )
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list_path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    output_target = args.output or str(DEFAULT_OUTPUTS[args.language])
    output_path = Path(output_target).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    segments = build_segments(language=args.language)
    ensure_requirements(segments)

    if args.keep_temp:
        temp_dir = output_path.parent / f"{output_path.stem}_segments"
        temp_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="multi_model_demo_", dir=output_path.parent)
        temp_dir = Path(temp_dir_obj.name)
        cleanup = True

    try:
        rendered_segments: list[Path] = []
        for index, segment in enumerate(segments):
            print(
                f"[{index + 1}/{len(segments)}] render {segment.section} "
                f"{segment.code} {segment.start:.2f}-{segment.end:.2f}"
            )
            rendered_segments.append(render_segment(segment, index, temp_dir))

        print(f"[concat] {len(rendered_segments)} segments -> {output_path}")
        concat_segments(rendered_segments, output_path, temp_dir)
    finally:
        if cleanup:
            temp_dir_obj.cleanup()

    print(
        f"Saved demo preview to {output_path} "
        f"({OUTPUT_WIDTH}x{OUTPUT_HEIGHT}, {OUTPUT_FPS} fps)"
    )


if __name__ == "__main__":
    main()
