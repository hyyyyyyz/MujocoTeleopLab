"""Run host high-level-policy control through Teleopit's onboard motion tracker."""

from __future__ import annotations

import inspect

import hydra
from omegaconf import DictConfig

from teleopit.high_level_policy.config import parse_high_level_policy_config
from teleopit.runtime.cli import validate_policy_path
from teleopit.runtime.console import (
    PlainConsole,
    configure_runtime_logging,
    high_level_policy_operator_controls,
)
from teleopit.sim2real.mp import HighLevelPolicySim2RealRuntime


@hydra.main(
    version_base=None,
    config_path="../../teleopit/configs",
    config_name="high_level_policy_sim2real",
)
def main(cfg: DictConfig) -> None:
    _run_high_level_policy_sim2real(cfg)


def _run_high_level_policy_sim2real(cfg: DictConfig) -> None:
    configure_runtime_logging(cfg, force=True)
    validate_policy_path(cfg, "run_high_level_policy_sim2real.py")
    policy_cfg = parse_high_level_policy_config(cfg)
    console = PlainConsole(title="Teleopit high-level policy sim2real")
    runtime_params = inspect.signature(HighLevelPolicySim2RealRuntime).parameters
    runtime = (
        HighLevelPolicySim2RealRuntime(cfg, console=console)
        if "console" in runtime_params
        else HighLevelPolicySim2RealRuntime(cfg)
    )
    console.start(
        status=(
            ("State", "IDLE"),
            ("Runtime", "high-level policy"),
            ("Host", policy_cfg.endpoint),
            ("Task", policy_cfg.task),
        ),
        controls=high_level_policy_operator_controls(),
        events=("Start enters STANDING; Remote Y requests host-policy takeover",),
        control_section="Controls",
        show_help_key=False,
    )
    try:
        runtime.run()
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    main()
