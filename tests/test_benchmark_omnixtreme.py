from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import numpy as np
import torch

from train_mimic.benchmarking import (
    BenchmarkJob,
    ClipSpec,
    RolloutResult,
    build_benchmark_plan,
    compute_tracking_metrics,
    summarize_rollouts,
    write_benchmark_outputs,
)
from train_mimic.tasks.tracking.mdp.commands import MotionCommand
from train_mimic.scripts.benchmark import _configure_benchmark_env_cfg, _run_batch, parse_args


@dataclass
class _FakeAgentCfg:
    clip_actions: float | None = None


def _clip(clip_id: int, duration_s: float) -> ClipSpec:
    return ClipSpec(
        clip_id=clip_id,
        shard_path="shard.h5",
        shard_clip_index=clip_id,
        frame_offset=clip_id * 1000,
        num_frames=int(duration_s * 30) + 1,
        fps=30.0,
        sample_start_s=0.0,
        sample_end_s=duration_s,
    )


def test_build_benchmark_plan_uses_all_eligible_clips() -> None:
    plan = build_benchmark_plan(
        [_clip(0, 10.0), _clip(1, 9.9), _clip(2, 12.0)],
        clip_seconds=10.0,
        step_dt=0.02,
    )

    assert plan.control_steps == 500
    assert [clip.clip_id for clip in plan.eligible_clips] == [0, 2]
    assert [clip.clip_id for clip in plan.skipped_short_clips] == [1]
    assert plan.jobs == (
        BenchmarkJob(job_id=0, clip_id=0, rollout_index=0, start_time_s=0.0),
        BenchmarkJob(job_id=1, clip_id=2, rollout_index=0, start_time_s=0.0),
    )


def test_build_benchmark_plan_requires_integer_control_steps() -> None:
    with pytest.raises(ValueError, match="integer number of control steps"):
        build_benchmark_plan(
            [_clip(0, 10.0)],
            clip_seconds=10.0,
            step_dt=0.03,
        )


def test_compute_tracking_metrics() -> None:
    ref = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    robot = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[1.1, 0.0, 0.0]],
            [[3.4, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    metrics = compute_tracking_metrics(
        ref,
        robot,
        root_pos_error_m=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        root_rot_error_rad=np.array([0.01, 0.02, 0.03], dtype=np.float32),
        root_vel_error_m_s=np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )

    assert metrics["mpjpe_m"] == pytest.approx((0.0 + 0.1 + 0.4) / 3.0)
    assert metrics["root_pos_error_m"] == pytest.approx(0.2)
    assert metrics["root_rot_error_rad"] == pytest.approx(0.02)
    assert metrics["root_vel_error_m_s"] == pytest.approx(2.0)


def test_summarize_rollouts_aggregates_success_and_metrics() -> None:
    results = [
        RolloutResult(0, 0, 0, True, 500, None, None, 0.01, 0.1, 0.01, 1.0),
        RolloutResult(
            1,
            0,
            1,
            False,
            120,
            120,
            "anchor_pos",
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        ),
        RolloutResult(2, 1, 0, True, 500, None, None, 0.03, 0.3, 0.03, 3.0),
    ]

    summary = summarize_rollouts(results)

    assert summary["global"]["success_rate"] == pytest.approx(200.0 / 3.0)
    assert summary["global"]["mpjpe_m"] == pytest.approx(0.02)
    assert summary["global"]["root_pos_error_m"] == pytest.approx(0.2)
    assert summary["global"]["root_rot_error_rad"] == pytest.approx(0.02)
    assert summary["global"]["root_vel_error_m_s"] == pytest.approx(2.0)
    assert summary["per_clip"][0]["success_rate"] == pytest.approx(50.0)
    assert summary["per_clip"][0]["mpjpe_m"] == pytest.approx(0.01)
    assert summary["per_clip"][1]["success_rate"] == pytest.approx(100.0)


def test_parse_args_rejects_removed_legacy_flags() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--checkpoint",
                "model.pt",
                "--motion_file",
                "data/datasets_precomputed",
                "--num_eval_steps",
                "2000",
            ]
        )


def test_parse_args_rejects_rollouts_per_clip_flag() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--checkpoint",
                "model.pt",
                "--motion_file",
                "data/datasets_precomputed",
                "--rollouts_per_clip",
                "2",
            ]
        )


def test_benchmark_env_cfg_disables_noise_events_and_clip_resample() -> None:
    motion = SimpleNamespace(
        motion_file="old",
        sampling_mode="rewind",
        resample_on_clip_end=True,
        pose_range={"x": (-1.0, 1.0)},
        velocity_range={"x": (-1.0, 1.0)},
        joint_position_range=(-0.1, 0.1),
    )
    cfg = SimpleNamespace(
        commands={"motion": motion},
        events={"base_com": object(), "add_joint_default_pos": object()},
        episode_length_s=1.0,
        auto_reset=True,
    )

    out = _configure_benchmark_env_cfg(
        cfg,
        motion_file="data/datasets_precomputed",
        clip_seconds=10.0,
    )

    assert out.commands["motion"].motion_file == "data/datasets_precomputed"
    assert out.commands["motion"].sampling_mode == "start"
    assert out.commands["motion"].resample_on_clip_end is False
    assert out.commands["motion"].pose_range == {}
    assert out.commands["motion"].velocity_range == {}
    assert out.commands["motion"].joint_position_range == (0.0, 0.0)
    assert out.events == {}
    assert out.episode_length_s == 10.0
    assert out.auto_reset is False


def test_reset_to_motion_rejects_sample_end_time() -> None:
    cmd = SimpleNamespace()
    cmd.device = "cpu"
    cmd.motion_times = torch.zeros(1, dtype=torch.float32)
    cmd.motion_ids = torch.zeros(1, dtype=torch.long)
    cmd.time_left = torch.zeros(1, dtype=torch.float32)
    cmd.motion = SimpleNamespace(
        num_clips=1,
        clip_sample_start_s=torch.tensor([0.0], dtype=torch.float32),
        clip_sample_end_s=torch.tensor([10.0], dtype=torch.float32),
    )

    with pytest.raises(ValueError, match=r"range=\[0\.000000, 10\.000000\)"):
        MotionCommand.reset_to_motion(
            cmd,
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([10.0]),
        )


def test_write_benchmark_outputs_serializes_failed_metrics_as_null(tmp_path) -> None:
    plan = build_benchmark_plan(
        [_clip(0, 10.0)],
        clip_seconds=10.0,
        step_dt=0.02,
    )
    result = RolloutResult(
        job_id=0,
        clip_id=0,
        rollout_index=0,
        success=False,
        steps=120,
        failure_step=120,
        failure_reason="anchor_pos",
        mpjpe_m=float("nan"),
        root_pos_error_m=float("nan"),
        root_rot_error_rad=float("nan"),
        root_vel_error_m_s=float("nan"),
    )

    paths = write_benchmark_outputs(
        tmp_path,
        text_stem="benchmark",
        metadata={
            "task": "General-Tracking-G1",
            "checkpoint": "model.pt",
            "motion_file": "dataset",
        },
        plan=plan,
        results=[result],
    )

    data = paths["summary_json"].read_text()
    assert "NaN" not in data
    report = __import__("json").loads(data)
    assert report["global"]["mpjpe_m"] is None
    assert report["global"]["root_pos_error_m"] is None
    assert report["per_rollout"][0]["mpjpe_m"] is None


def test_run_batch_resets_inactive_done_envs_and_excludes_failed_metrics(monkeypatch) -> None:
    import train_mimic.scripts.benchmark as benchmark_script

    class FakeTensor:
        def __init__(self, values):
            self.values = np.asarray(values)

        def __or__(self, other):
            return FakeTensor(self.values | other.values)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

        def item(self):
            return bool(self.values)

    class FakeTorch:
        long = "long"
        float32 = "float32"

        class no_grad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        @staticmethod
        def tensor(values, dtype=None, device=None):
            return list(values)

        @staticmethod
        def arange(n, dtype=None, device=None):
            return list(range(n))

    class FakeCmd:
        def reset_to_motion(self, env_ids, motion_ids, motion_times):
            return None

    class FakeCommandManager:
        def __init__(self):
            self.cmd = FakeCmd()

        def get_term(self, name):
            return self.cmd

        def compute(self, dt):
            return None

    class FakeScene:
        def write_data_to_sim(self):
            return None

    class FakeSim:
        def forward(self):
            return None

        def sense(self):
            return None

    class FakeObservationManager:
        def __init__(self):
            self.reset_calls = []

        def reset(self, env_ids):
            self.reset_calls.append(list(env_ids))
            return {}

        def compute(self, update_history):
            return {"actor": np.zeros((2, 1), dtype=np.float32)}

    class FakeTermCfg:
        time_out = False

    class FakeTerminationManager:
        active_terms = ("failure",)

        def get_term_cfg(self, term_name):
            return FakeTermCfg()

        def get_term(self, term_name):
            return [FakeTensor(False), FakeTensor(True)]

    class FakeEnv:
        instances = []

        def __init__(self, cfg, device, render_mode):
            self.cfg = cfg
            self.device = device
            self.scene = FakeScene()
            self.sim = FakeSim()
            self.command_manager = FakeCommandManager()
            self.observation_manager = FakeObservationManager()
            self.termination_manager = FakeTerminationManager()
            self.reset_calls = []
            self.step_index = 0
            FakeEnv.instances.append(self)

        def reset(self, env_ids=None):
            self.reset_calls.append(None if env_ids is None else list(env_ids))
            return {"actor": np.zeros((2, 1), dtype=np.float32)}, {}

        def step(self, actions):
            self.step_index += 1
            if self.step_index == 1:
                terminated = FakeTensor([False, True])
                truncated = FakeTensor([False, False])
            else:
                terminated = FakeTensor([False, True])
                truncated = FakeTensor([True, False])
            return (
                {"actor": np.zeros((2, 1), dtype=np.float32)},
                None,
                terminated,
                truncated,
                {},
            )

        def close(self):
            return None

    class FakeWrapper:
        def __init__(self, env, clip_actions):
            self.env = env

    class FakeRunner:
        def __init__(self, wrapped_env, agent_dict, log_dir, device):
            return None

        def load(self, checkpoint, map_location):
            return None

        def get_inference_policy(self, device):
            return lambda obs: np.zeros((2, 1), dtype=np.float32)

    def fake_aligned(_cmd):
        ref = np.zeros((2, 1, 3), dtype=np.float32)
        robot = np.zeros((2, 1, 3), dtype=np.float32)
        return ref, robot

    def fake_root_errors(_cmd):
        return (
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
        )

    monkeypatch.setattr(benchmark_script, "_aligned_keybody_positions", fake_aligned)
    monkeypatch.setattr(benchmark_script, "_root_tracking_errors", fake_root_errors)

    motion = SimpleNamespace(motion_file="dataset")
    base_env_cfg = SimpleNamespace(
        commands={"motion": motion},
        events={},
        episode_length_s=1.0,
        auto_reset=True,
        scene=SimpleNamespace(num_envs=0),
    )
    agent_cfg = _FakeAgentCfg()
    jobs = [
        BenchmarkJob(0, 0, 0, 0.0),
        BenchmarkJob(1, 1, 0, 0.0),
    ]

    results = _run_batch(
        batch_index=0,
        jobs=jobs,
        base_env_cfg=base_env_cfg,
        agent_cfg=agent_cfg,
        runner_cls=FakeRunner,
        fallback_runner_cls=FakeRunner,
        checkpoint="model.pt",
        log_dir="logs",
        device="cpu",
        torch_module=FakeTorch,
        ManagerBasedRlEnv=FakeEnv,
        RslRlVecEnvWrapper=FakeWrapper,
        clip_seconds=10.0,
        control_steps=2,
        seed=42,
    )

    env = FakeEnv.instances[-1]
    assert env.reset_calls == [None, [1]]
    assert env.observation_manager.reset_calls == [[0, 1]]
    assert results[0].success is True
    assert results[1].success is False
    assert np.isnan(results[1].mpjpe_m)
