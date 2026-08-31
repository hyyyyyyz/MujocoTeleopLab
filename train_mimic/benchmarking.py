"""OmniXtreme-style benchmark helpers for motion tracking policies."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np

from train_mimic.data.dataset_lib import (
    compute_clip_sample_ranges,
    find_precomputed_motion_shards,
    parse_window_steps,
)


@dataclass(frozen=True)
class ClipSpec:
    clip_id: int
    shard_path: str
    shard_clip_index: int
    frame_offset: int
    num_frames: int
    fps: float
    sample_start_s: float
    sample_end_s: float


@dataclass(frozen=True)
class BenchmarkJob:
    job_id: int
    clip_id: int
    rollout_index: int
    start_time_s: float


@dataclass(frozen=True)
class BenchmarkPlan:
    clip_seconds: float
    control_steps: int
    step_dt: float
    eligible_clips: tuple[ClipSpec, ...]
    skipped_short_clips: tuple[ClipSpec, ...]
    jobs: tuple[BenchmarkJob, ...]


@dataclass(frozen=True)
class RolloutResult:
    job_id: int
    clip_id: int
    rollout_index: int
    success: bool
    steps: int
    failure_step: int | None
    failure_reason: str | None
    mpjpe_m: float
    root_pos_error_m: float
    root_rot_error_rad: float
    root_vel_error_m_s: float


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def load_clip_specs(
    motion_file: str | Path,
    *,
    window_steps: Sequence[int] = (0,),
) -> tuple[ClipSpec, ...]:
    """Load benchmark clip specs in the same order MotionLib assigns clip ids."""
    specs: list[ClipSpec] = []
    next_clip_id = 0
    steps = parse_window_steps(window_steps)
    max_future = max((step for step in steps if step > 0), default=0)
    max_history = -min((step for step in steps if step < 0), default=0)
    min_clip_length = max_history + 1 + max_future + 1
    for shard_path in find_precomputed_motion_shards(Path(motion_file)):
        with h5py.File(shard_path, "r") as h5:
            starts = np.asarray(h5["clip_starts"], dtype=np.int64)
            lengths = np.asarray(h5["clip_lengths"], dtype=np.int64)
            fps = np.asarray(h5["clip_fps"], dtype=np.float32)
            valid_mask = lengths >= min_clip_length
            if not np.any(valid_mask):
                continue
            valid_starts = starts[valid_mask]
            valid_lengths = lengths[valid_mask]
            valid_fps = fps[valid_mask]
            valid_shard_clip_indices = np.nonzero(valid_mask)[0]
            sample_starts, sample_ends = compute_clip_sample_ranges(
                valid_lengths,
                window_steps=steps,
            )
            for shard_clip_index, start, length, cur_fps, sample_start, sample_end in zip(
                valid_shard_clip_indices,
                valid_starts,
                valid_lengths,
                valid_fps,
                sample_starts,
                sample_ends,
                strict=True,
            ):
                specs.append(
                    ClipSpec(
                        clip_id=next_clip_id,
                        shard_path=str(shard_path),
                        shard_clip_index=int(shard_clip_index),
                        frame_offset=int(start),
                        num_frames=int(length),
                        fps=float(cur_fps),
                        sample_start_s=float(sample_start) / float(cur_fps),
                        sample_end_s=float(sample_end) / float(cur_fps),
                    )
                )
                next_clip_id += 1
    return tuple(specs)


def build_benchmark_plan(
    clips: Sequence[ClipSpec],
    *,
    clip_seconds: float,
    step_dt: float,
) -> BenchmarkPlan:
    if clip_seconds <= 0.0:
        raise ValueError(f"clip_seconds must be > 0, got {clip_seconds}")
    if step_dt <= 0.0:
        raise ValueError(f"step_dt must be > 0, got {step_dt}")

    control_steps_f = clip_seconds / step_dt
    control_steps = int(round(control_steps_f))
    if not np.isclose(control_steps_f, control_steps, atol=1e-6):
        raise ValueError(
            f"clip_seconds={clip_seconds} is not an integer number of control steps "
            f"for step_dt={step_dt}"
        )

    eligible: list[ClipSpec] = []
    skipped: list[ClipSpec] = []
    for clip in clips:
        duration_s = clip.sample_end_s - clip.sample_start_s
        if duration_s + 1e-9 >= clip_seconds:
            eligible.append(clip)
        else:
            skipped.append(clip)
    if not eligible:
        raise ValueError(
            f"No clips are at least {clip_seconds:.3f}s long after applying valid sample ranges."
        )

    jobs: list[BenchmarkJob] = []
    for clip in eligible:
        jobs.append(
            BenchmarkJob(
                job_id=len(jobs),
                clip_id=clip.clip_id,
                rollout_index=0,
                start_time_s=clip.sample_start_s,
            )
        )
    return BenchmarkPlan(
        clip_seconds=clip_seconds,
        control_steps=control_steps,
        step_dt=step_dt,
        eligible_clips=tuple(eligible),
        skipped_short_clips=tuple(skipped),
        jobs=tuple(jobs),
    )


def compute_tracking_metrics(
    aligned_ref_pos: np.ndarray,
    aligned_robot_pos: np.ndarray,
    root_pos_error_m: np.ndarray,
    root_rot_error_rad: np.ndarray,
    root_vel_error_m_s: np.ndarray,
) -> dict[str, float]:
    """Compute benchmark tracking metrics from aligned key bodies and root errors.

    Key-body inputs are ``(T, B, 3)`` arrays in root/anchor coordinates. Root
    error inputs are per-frame values computed with the same anchor metrics used
    by ``MotionCommand``.
    """
    ref = np.asarray(aligned_ref_pos, dtype=np.float64)
    robot = np.asarray(aligned_robot_pos, dtype=np.float64)
    if ref.shape != robot.shape:
        raise ValueError(f"aligned position shape mismatch: {ref.shape} vs {robot.shape}")
    if ref.ndim != 3 or ref.shape[-1] != 3:
        raise ValueError(f"aligned positions must be (T,B,3), got {ref.shape}")
    if ref.shape[0] == 0 or ref.shape[1] == 0:
        raise ValueError(
            f"aligned positions must have non-empty T and B dimensions, got {ref.shape}"
        )

    root_pos = np.asarray(root_pos_error_m, dtype=np.float64)
    root_rot = np.asarray(root_rot_error_rad, dtype=np.float64)
    root_vel = np.asarray(root_vel_error_m_s, dtype=np.float64)
    for name, values in (
        ("root_pos_error_m", root_pos),
        ("root_rot_error_rad", root_rot),
        ("root_vel_error_m_s", root_vel),
    ):
        if values.size == 0:
            raise ValueError(f"{name} must be non-empty")

    pos_error = np.linalg.norm(ref - robot, axis=-1)
    mpjpe = float(pos_error.mean())

    return {
        "mpjpe_m": mpjpe,
        "root_pos_error_m": float(root_pos.mean()),
        "root_rot_error_rad": float(root_rot.mean()),
        "root_vel_error_m_s": float(root_vel.mean()),
    }


def summarize_rollouts(results: Sequence[RolloutResult]) -> dict[str, Any]:
    if not results:
        raise ValueError("Cannot summarize an empty benchmark result set")

    def finite_mean(values: Iterable[float]) -> float:
        arr = np.asarray(list(values), dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan")
        return float(arr.mean())

    clip_ids = sorted({result.clip_id for result in results})
    per_clip: list[dict[str, Any]] = []
    for clip_id in clip_ids:
        clip_results = [result for result in results if result.clip_id == clip_id]
        per_clip.append(
            {
                "clip_id": clip_id,
                "rollouts": len(clip_results),
                "success_rate": 100.0
                * sum(1 for result in clip_results if result.success)
                / len(clip_results),
                "mpjpe_m": finite_mean(result.mpjpe_m for result in clip_results),
                "root_pos_error_m": finite_mean(
                    result.root_pos_error_m for result in clip_results
                ),
                "root_rot_error_rad": finite_mean(
                    result.root_rot_error_rad for result in clip_results
                ),
                "root_vel_error_m_s": finite_mean(
                    result.root_vel_error_m_s for result in clip_results
                ),
            }
        )

    return {
        "global": {
            "clips": len(clip_ids),
            "rollouts": len(results),
            "success_rate": 100.0
            * sum(1 for result in results if result.success)
            / len(results),
            "mpjpe_m": finite_mean(result.mpjpe_m for result in results),
            "root_pos_error_m": finite_mean(
                result.root_pos_error_m for result in results
            ),
            "root_rot_error_rad": finite_mean(
                result.root_rot_error_rad for result in results
            ),
            "root_vel_error_m_s": finite_mean(
                result.root_vel_error_m_s for result in results
            ),
        },
        "per_clip": per_clip,
    }


def write_benchmark_outputs(
    output_dir: str | Path,
    *,
    text_stem: str,
    metadata: dict[str, Any],
    plan: BenchmarkPlan,
    results: Sequence[RolloutResult],
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary = summarize_rollouts(results)

    txt_path = output_path / f"{text_stem}.txt"
    json_path = output_path / f"{text_stem}.json"
    per_clip_path = output_path / f"{text_stem}-per_clip.csv"
    per_rollout_path = output_path / f"{text_stem}-per_rollout.csv"

    global_summary = summary["global"]
    lines = [
        "OmniXtreme-style Benchmark Results",
        f"checkpoint: {metadata['checkpoint']}",
        f"motion_file: {metadata['motion_file']}",
        f"clip_seconds: {plan.clip_seconds:.6f}",
        f"control_steps: {plan.control_steps}",
        f"eligible_clips: {len(plan.eligible_clips)}",
        f"skipped_short_clips: {len(plan.skipped_short_clips)}",
        "",
        f"MPJPE(m): {global_summary['mpjpe_m']:.6f}",
        f"root_pos_error(m): {global_summary['root_pos_error_m']:.6f}",
        f"root_rot_error(rad): {global_summary['root_rot_error_rad']:.6f}",
        f"root_vel_error(m/s): {global_summary['root_vel_error_m_s']:.6f}",
        f"success_rate(%): {global_summary['success_rate']:.6f}",
    ]
    txt_path.write_text("\n".join(lines) + "\n")

    report = {
        "metadata": metadata,
        "protocol": {
            "clip_seconds": plan.clip_seconds,
            "control_steps": plan.control_steps,
            "step_dt": plan.step_dt,
        },
        "global": global_summary,
        "per_clip": summary["per_clip"],
        "per_rollout": [asdict(result) for result in results],
        "eligible_clips": [asdict(clip) for clip in plan.eligible_clips],
        "skipped_short_clips": [asdict(clip) for clip in plan.skipped_short_clips],
    }
    json_path.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False))

    with per_clip_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clip_id",
                "rollouts",
                "success_rate",
                "mpjpe_m",
                "root_pos_error_m",
                "root_rot_error_rad",
                "root_vel_error_m_s",
            ],
        )
        writer.writeheader()
        writer.writerows(summary["per_clip"])

    with per_rollout_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    return {
        "summary_txt": txt_path,
        "summary_json": json_path,
        "per_clip_csv": per_clip_path,
        "per_rollout_csv": per_rollout_path,
    }
