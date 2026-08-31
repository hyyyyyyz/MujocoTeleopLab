#!/usr/bin/env python3
"""Benchmark a G1 motion tracking policy with an OmniXtreme-style protocol.

Default protocol:
    * 10 second clips at the policy control rate (500 steps at 50 Hz)
    * One deterministic rollout per eligible motion clip
    * MPJPE, root tracking errors, and success rate

Usage:
    python train_mimic/scripts/benchmark.py \
        --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
        --motion_file data/datasets_precomputed \
        --num_envs 32
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from tensordict import TensorDict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_mimic.app import (
    DEFAULT_TASK,
    build_runner_cfg_dict,
    import_training_stack,
    load_task_components,
    resolve_device,
    validate_checkpoint_path,
    validate_motion_file,
)
from train_mimic.benchmarking import (
    BenchmarkJob,
    RolloutResult,
    build_benchmark_plan,
    compute_tracking_metrics,
    load_clip_specs,
    summarize_rollouts,
    write_benchmark_outputs,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OmniXtreme-style benchmark for the G1 tracking policy."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--motion_file",
        type=str,
        required=True,
        help="Precomputed training dataset root or shard produced by precompute_dataset.py",
    )
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip_seconds", type=float, default=10.0)
    parser.add_argument("--output_dir", type=str, default="benchmark_results")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        help="Task id to benchmark (default: %(default)s)",
    )
    return parser.parse_args(argv)


def _chunked(seq: Sequence[BenchmarkJob], size: int) -> list[Sequence[BenchmarkJob]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def _obs_tensordict(obs_dict: object, num_envs: int) -> TensorDict:
    return TensorDict(obs_dict, batch_size=[num_envs])


def _configure_benchmark_env_cfg(
    base_cfg: object,
    *,
    motion_file: str,
    clip_seconds: float,
) -> object:
    cfg = copy.deepcopy(base_cfg)
    cfg.commands["motion"].motion_file = motion_file
    cfg.commands["motion"].sampling_mode = "start"
    cfg.commands["motion"].resample_on_clip_end = False
    cfg.commands["motion"].pose_range = {}
    cfg.commands["motion"].velocity_range = {}
    cfg.commands["motion"].joint_position_range = (0.0, 0.0)
    cfg.events = {}
    cfg.episode_length_s = clip_seconds
    cfg.auto_reset = False
    return cfg


def _reset_to_jobs(
    env: object,
    jobs: Sequence[BenchmarkJob],
    torch_module: object,
) -> TensorDict:
    obs_dict, _extras = env.reset()
    cmd = env.command_manager.get_term("motion")
    env_ids = torch_module.arange(len(jobs), dtype=torch_module.long, device=env.device)
    motion_ids = torch_module.tensor(
        [job.clip_id for job in jobs],
        dtype=torch_module.long,
        device=env.device,
    )
    motion_times = torch_module.tensor(
        [job.start_time_s for job in jobs],
        dtype=torch_module.float32,
        device=env.device,
    )
    cmd.reset_to_motion(env_ids, motion_ids, motion_times)
    env.observation_manager.reset(env_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.command_manager.compute(dt=0.0)
    env.sim.sense()
    obs_dict = env.observation_manager.compute(update_history=True)
    env.obs_buf = obs_dict
    return _obs_tensordict(obs_dict, len(jobs))


def _aligned_keybody_positions(cmd: object) -> tuple[np.ndarray, np.ndarray]:
    from mjlab.utils.lab_api.math import quat_apply, quat_inv

    ref_anchor_pos = cmd.anchor_pos_w[:, None, :]
    robot_anchor_pos = cmd.robot_anchor_pos_w[:, None, :]
    num_bodies = cmd.body_pos_w.shape[1]

    ref_anchor_inv = quat_inv(cmd.anchor_quat_w)[:, None, :].expand(-1, num_bodies, -1)
    robot_anchor_inv = quat_inv(cmd.robot_anchor_quat_w)[:, None, :].expand(
        -1, num_bodies, -1
    )
    ref_aligned = quat_apply(ref_anchor_inv, cmd.body_pos_w - ref_anchor_pos)
    robot_aligned = quat_apply(robot_anchor_inv, cmd.robot_body_pos_w - robot_anchor_pos)
    return (
        ref_aligned.detach().cpu().numpy().astype(np.float32, copy=False),
        robot_aligned.detach().cpu().numpy().astype(np.float32, copy=False),
    )


def _root_tracking_errors(cmd: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from mjlab.utils.lab_api.math import quat_error_magnitude
    import torch

    root_pos_error = torch.norm(cmd.anchor_pos_w - cmd.robot_anchor_pos_w, dim=-1)
    root_rot_error = quat_error_magnitude(cmd.anchor_quat_w, cmd.robot_anchor_quat_w)
    root_vel_error = torch.norm(
        cmd.anchor_lin_vel_w - cmd.robot_anchor_lin_vel_w,
        dim=-1,
    )
    return (
        root_pos_error.detach().cpu().numpy().astype(np.float32, copy=False),
        root_rot_error.detach().cpu().numpy().astype(np.float32, copy=False),
        root_vel_error.detach().cpu().numpy().astype(np.float32, copy=False),
    )


def _failure_reason(env: object, env_index: int) -> str:
    manager = env.termination_manager
    for term_name in manager.active_terms:
        term_cfg = manager.get_term_cfg(term_name)
        if term_cfg.time_out:
            continue
        if bool(manager.get_term(term_name)[env_index].item()):
            return term_name
    for term_name in manager.active_terms:
        term_cfg = manager.get_term_cfg(term_name)
        if term_cfg.time_out and bool(manager.get_term(term_name)[env_index].item()):
            return term_name
    return "done"


def _run_batch(
    *,
    batch_index: int,
    jobs: Sequence[BenchmarkJob],
    base_env_cfg: object,
    agent_cfg: object,
    runner_cls: object,
    fallback_runner_cls: object,
    checkpoint: str,
    log_dir: str,
    device: str,
    torch_module: object,
    ManagerBasedRlEnv: object,
    RslRlVecEnvWrapper: object,
    clip_seconds: float,
    control_steps: int,
    seed: int,
) -> list[RolloutResult]:
    env_cfg = _configure_benchmark_env_cfg(
        base_env_cfg,
        motion_file=base_env_cfg.commands["motion"].motion_file,
        clip_seconds=clip_seconds,
    )
    env_cfg.seed = seed + batch_index
    env_cfg.scene.num_envs = len(jobs)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    agent_dict = build_runner_cfg_dict(agent_cfg, force_tensorboard=True)
    RunnerCls = runner_cls or fallback_runner_cls
    runner = RunnerCls(wrapped_env, agent_dict, log_dir=log_dir, device=device)
    runner.load(checkpoint, map_location=device)
    policy = runner.get_inference_policy(device=device)

    aligned_ref_by_env: list[list[np.ndarray]] = [[] for _ in jobs]
    aligned_robot_by_env: list[list[np.ndarray]] = [[] for _ in jobs]
    root_pos_error_by_env: list[list[float]] = [[] for _ in jobs]
    root_rot_error_by_env: list[list[float]] = [[] for _ in jobs]
    root_vel_error_by_env: list[list[float]] = [[] for _ in jobs]
    active = np.ones(len(jobs), dtype=bool)
    finished: dict[int, tuple[bool, int, int | None, str | None]] = {}

    try:
        obs = _reset_to_jobs(env, jobs, torch_module)
        cmd = env.command_manager.get_term("motion")
        for step in range(control_steps):
            ref_aligned, robot_aligned = _aligned_keybody_positions(cmd)
            root_pos_error, root_rot_error, root_vel_error = _root_tracking_errors(cmd)
            for env_index, is_active in enumerate(active):
                if is_active:
                    aligned_ref_by_env[env_index].append(ref_aligned[env_index])
                    aligned_robot_by_env[env_index].append(robot_aligned[env_index])
                    root_pos_error_by_env[env_index].append(
                        float(root_pos_error[env_index])
                    )
                    root_rot_error_by_env[env_index].append(
                        float(root_rot_error[env_index])
                    )
                    root_vel_error_by_env[env_index].append(
                        float(root_vel_error[env_index])
                    )

            with torch_module.no_grad():
                actions = policy(obs)
            if agent_cfg.clip_actions is not None:
                actions = torch_module.clamp(
                    actions,
                    -agent_cfg.clip_actions,
                    agent_cfg.clip_actions,
                )

            obs_dict, _rewards, terminated, truncated, _extras = env.step(actions)
            obs = _obs_tensordict(obs_dict, len(jobs))

            done = (terminated | truncated).detach().cpu().numpy().astype(bool)
            terminated_np = terminated.detach().cpu().numpy().astype(bool)
            truncated_np = truncated.detach().cpu().numpy().astype(bool)
            done_envs = [int(env_index) for env_index, is_done in enumerate(done) if is_done]
            for env_index, is_done in enumerate(done):
                if not active[env_index] or not is_done:
                    continue
                reached_horizon = step + 1 >= control_steps
                success = bool(
                    truncated_np[env_index]
                    and reached_horizon
                    and not terminated_np[env_index]
                )
                failure_step = None if success else step + 1
                failure_reason = None if success else _failure_reason(env, env_index)
                finished[env_index] = (success, step + 1, failure_step, failure_reason)
                active[env_index] = False

            if step + 1 < control_steps and done_envs:
                env_ids = torch_module.tensor(
                    done_envs,
                    dtype=torch_module.long,
                    device=env.device,
                )
                obs_dict, _extras = env.reset(env_ids=env_ids)
                obs = _obs_tensordict(obs_dict, len(jobs))

            if not active.any():
                break
    finally:
        env.close()

    results: list[RolloutResult] = []
    for env_index, job in enumerate(jobs):
        success, steps, failure_step, failure_reason = finished.get(
            env_index,
            (True, control_steps, None, None),
        )
        metrics = compute_tracking_metrics(
            np.stack(aligned_ref_by_env[env_index], axis=0),
            np.stack(aligned_robot_by_env[env_index], axis=0),
            np.asarray(root_pos_error_by_env[env_index], dtype=np.float32),
            np.asarray(root_rot_error_by_env[env_index], dtype=np.float32),
            np.asarray(root_vel_error_by_env[env_index], dtype=np.float32),
        )
        if not success:
            metrics = {
                "mpjpe_m": float("nan"),
                "root_pos_error_m": float("nan"),
                "root_rot_error_rad": float("nan"),
                "root_vel_error_m_s": float("nan"),
            }
        results.append(
            RolloutResult(
                job_id=job.job_id,
                clip_id=job.clip_id,
                rollout_index=job.rollout_index,
                success=success,
                steps=steps,
                failure_step=failure_step,
                failure_reason=failure_reason,
                mpjpe_m=metrics["mpjpe_m"],
                root_pos_error_m=metrics["root_pos_error_m"],
                root_rot_error_rad=metrics["root_rot_error_rad"],
                root_vel_error_m_s=metrics["root_vel_error_m_s"],
            )
        )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.num_envs <= 0:
        raise ValueError("--num_envs must be > 0")
    if args.clip_seconds <= 0.0:
        raise ValueError("--clip_seconds must be > 0")

    validate_motion_file(args.motion_file)
    try:
        validate_checkpoint_path(args.checkpoint)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1

    (
        torch,
        ManagerBasedRlEnv,
        RslRlVecEnvWrapper,
        MjlabOnPolicyRunner,
        _load_env_cfg,
        _load_rl_cfg,
        _load_runner_cls,
        configure_torch_backends,
    ) = import_training_stack()
    configure_torch_backends()

    task_name, base_env_cfg, agent_cfg, runner_cls = load_task_components(
        args.task,
        play=True,
        load_env_cfg=_load_env_cfg,
        load_rl_cfg=_load_rl_cfg,
        load_runner_cls=_load_runner_cls,
    )
    base_env_cfg.commands["motion"].motion_file = args.motion_file
    benchmark_env_cfg = _configure_benchmark_env_cfg(
        base_env_cfg,
        motion_file=args.motion_file,
        clip_seconds=args.clip_seconds,
    )
    step_dt = float(benchmark_env_cfg.decimation) * float(
        benchmark_env_cfg.sim.mujoco.timestep
    )
    clips = load_clip_specs(
        args.motion_file,
        window_steps=benchmark_env_cfg.commands["motion"].window_steps,
    )
    plan = build_benchmark_plan(
        clips,
        clip_seconds=args.clip_seconds,
        step_dt=step_dt,
    )

    device = resolve_device(args.device, torch)
    batches = _chunked(plan.jobs, args.num_envs)
    log_dir = os.path.dirname(args.checkpoint)
    results: list[RolloutResult] = []

    print(
        "Running OmniXtreme-style benchmark: "
        f"{len(plan.eligible_clips)} clips, {len(plan.jobs)} rollouts, "
        f"{plan.control_steps} steps/rollout, batch size {args.num_envs}."
    )
    if plan.skipped_short_clips:
        print(
            f"Skipping {len(plan.skipped_short_clips)} clips shorter than "
            f"{args.clip_seconds:.2f}s."
        )

    for batch_index, batch_jobs in enumerate(batches):
        print(
            f"[{batch_index + 1}/{len(batches)}] "
            f"rollouts {batch_jobs[0].job_id}..{batch_jobs[-1].job_id}"
        )
        results.extend(
            _run_batch(
                batch_index=batch_index,
                jobs=batch_jobs,
                base_env_cfg=benchmark_env_cfg,
                agent_cfg=agent_cfg,
                runner_cls=runner_cls,
                fallback_runner_cls=MjlabOnPolicyRunner,
                checkpoint=args.checkpoint,
                log_dir=log_dir,
                device=device,
                torch_module=torch,
                ManagerBasedRlEnv=ManagerBasedRlEnv,
                RslRlVecEnvWrapper=RslRlVecEnvWrapper,
                clip_seconds=args.clip_seconds,
                control_steps=plan.control_steps,
                seed=args.seed,
            )
        )

    output_stem = f"{task_name}-{Path(args.checkpoint).stem}-omnixtreme"
    paths = write_benchmark_outputs(
        args.output_dir,
        text_stem=output_stem,
        metadata={
            "task": task_name,
            "checkpoint": args.checkpoint,
            "motion_file": args.motion_file,
            "seed": args.seed,
            "num_envs": args.num_envs,
        },
        plan=plan,
        results=results,
    )

    summary = summarize_rollouts(results)["global"]
    print("\nBenchmark Results:")
    print(f"  MPJPE(m): {summary['mpjpe_m']:.4f}")
    print(f"  root_pos_error(m): {summary['root_pos_error_m']:.4f}")
    print(f"  root_rot_error(rad): {summary['root_rot_error_rad']:.4f}")
    print(f"  root_vel_error(m/s): {summary['root_vel_error_m_s']:.4f}")
    print(f"  success_rate(%): {summary['success_rate']:.2f}")
    for label, path in paths.items():
        print(f"Saved {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
