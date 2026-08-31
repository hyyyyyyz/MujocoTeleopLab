"""Runtime-only validation for filesystem-backed configuration fields."""

from __future__ import annotations

from pathlib import Path

from somehand.domain.config import BiHandRetargetingConfig, RetargetingConfig
from somehand.external_assets import build_missing_asset_message


def validate_runtime_retargeting_config(config: RetargetingConfig) -> None:
    if not Path(config.hand.mjcf_path).exists():
        raise FileNotFoundError(
            build_missing_asset_message(
                config.hand.mjcf_path,
                label="MJCF file",
            )
        )


def validate_runtime_bihand_config(config: BiHandRetargetingConfig) -> None:
    if not Path(config.left_config_path).exists():
        raise FileNotFoundError(f"Left-hand config not found: {config.left_config_path}")
    if not Path(config.right_config_path).exists():
        raise FileNotFoundError(f"Right-hand config not found: {config.right_config_path}")
