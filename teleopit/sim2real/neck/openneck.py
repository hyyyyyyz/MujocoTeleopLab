from __future__ import annotations

import logging
from typing import Protocol

from teleopit.sim2real.neck.config import NeckConfig

logger = logging.getLogger(__name__)


def _load_openneck_controller() -> type:
    try:
        from openneck import OpenNeckController
    except ModuleNotFoundError as exc:
        raise ImportError(
            "OpenNeck 0.2.0 is required for neck.driver=openneck. "
            "Install with: pip install -e '.[openneck]'"
        ) from exc
    if not all(callable(getattr(OpenNeckController, name, None)) for name in ("move_deg", "read_deg")):
        raise ImportError(
            "OpenNeck 0.2.0 move_deg/read_deg angle API is required; reinstall with: "
            "pip install --force-reinstall --no-deps "
            "'openneck @ git+https://github.com/BotRunner64/OpenNeck.git'"
        )
    return OpenNeckController


class NeckDevice(Protocol):
    def connect(self) -> None: ...

    def center(self) -> None: ...

    def release_torque(self) -> None: ...

    def move_deg(self, yaw_deg: float, pitch_deg: float) -> tuple[float, float]: ...

    def read_deg(self) -> tuple[float, float]: ...

    def close(self) -> None: ...


class OpenNeckDevice:
    def __init__(self, config: NeckConfig) -> None:
        self._cfg = config
        self._controller = None

    def connect(self) -> None:
        OpenNeckController = _load_openneck_controller()
        controller = OpenNeckController(
            config=self._cfg.config_path,
            port=self._cfg.port,
        )
        controller.connect()
        self._controller = controller
        logger.info("OpenNeck connected on port %s", getattr(self._controller, "port", self._cfg.port))

    def center(self) -> None:
        if self._controller is not None:
            self._controller.center()

    def move_deg(self, yaw_deg: float, pitch_deg: float) -> tuple[float, float]:
        if self._controller is None:
            raise RuntimeError("OpenNeck is not connected")
        applied = self._controller.move_deg(float(yaw_deg), float(pitch_deg))
        return float(applied.yaw_deg), float(applied.pitch_deg)

    def read_deg(self) -> tuple[float, float]:
        if self._controller is None:
            raise RuntimeError("OpenNeck is not connected")
        state = self._controller.read_deg()
        return float(state.yaw_deg), float(state.pitch_deg)

    def release_torque(self) -> None:
        if self._controller is not None:
            self._controller.release_torque()

    def close(self) -> None:
        controller = self._controller
        self._controller = None
        if controller is not None:
            controller.close()


class DryRunNeckDevice:
    def __init__(self, config: NeckConfig) -> None:
        self._cfg = config
        self._controller = None

    def connect(self) -> None:
        OpenNeckController = _load_openneck_controller()
        self._controller = OpenNeckController(
            config=self._cfg.config_path,
            port=self._cfg.port,
        )
        logger.info("OpenNeck dry-run device active")

    def center(self) -> None:
        logger.info("OpenNeck dry-run center")

    def move_deg(self, yaw_deg: float, pitch_deg: float) -> tuple[float, float]:
        controller = self._controller
        if controller is None:
            raise RuntimeError("OpenNeck dry-run device is not connected")
        # Reuse OpenNeck's calibration conversion without writing to
        # its servo driver, so dry-run reports the same clamped target.
        yaw_step = controller._angle_to_step("yaw", float(yaw_deg))
        pitch_step = controller._angle_to_step("pitch", float(pitch_deg))
        applied_yaw_deg = float(controller._step_to_angle("yaw", yaw_step))
        applied_pitch_deg = float(controller._step_to_angle("pitch", pitch_step))
        logger.debug(
            "OpenNeck dry-run command yaw=%.3fdeg pitch=%.3fdeg applied_yaw=%.3fdeg applied_pitch=%.3fdeg",
            yaw_deg,
            pitch_deg,
            applied_yaw_deg,
            applied_pitch_deg,
        )
        return applied_yaw_deg, applied_pitch_deg

    def read_deg(self) -> tuple[float, float]:
        raise RuntimeError("OpenNeck dry-run has no hardware state to read")

    def release_torque(self) -> None:
        logger.info("OpenNeck dry-run release")

    def close(self) -> None:
        self._controller = None
        logger.info("OpenNeck dry-run closed")


def build_neck_device(config: NeckConfig) -> NeckDevice:
    if config.driver != "openneck":
        raise ValueError("Unsupported neck.driver={!r}; supported drivers: openneck".format(config.driver))
    if config.dry_run:
        return DryRunNeckDevice(config)
    return OpenNeckDevice(config)
