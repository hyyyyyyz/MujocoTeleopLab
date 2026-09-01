from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig

from teleopit.bus.in_process import InProcessBus
from teleopit.controllers.observation import VelCmdObservationBuilder
from teleopit.controllers.rl_policy import RLPolicyController
from teleopit.inputs import BVHInputProvider, Pico4InputProvider, XRoboToolkitInputProvider
from teleopit.inputs.pico_video import PicoVideoRuntime, parse_pico_video_config
from teleopit.inputs.xrobotoolkit_video import XRoboToolkitVideoRuntime
from teleopit.retargeting.core import RetargetingModule
from teleopit.robots.mujoco_robot import MuJoCoRobot
from teleopit.runtime.common import cfg_get
from teleopit.runtime.console import PlainConsole
from teleopit.runtime.factory import build_inference_components
from teleopit.sim.loop import SimulationLoop


class TeleopPipeline:
    def __init__(self, cfg: DictConfig | dict[str, Any], *, console: PlainConsole | None = None) -> None:
        self.cfg = cfg
        self._project_root = Path(__file__).resolve().parent.parent
        components = build_inference_components(
            cfg,
            self._project_root,
            robot_cls=MuJoCoRobot,
            controller_cls=RLPolicyController,
            obs_builder_cls=VelCmdObservationBuilder,
            bvh_input_cls=BVHInputProvider,
            pico4_input_cls=Pico4InputProvider,
            xrobotoolkit_input_cls=XRoboToolkitInputProvider,
            retargeter_cls=RetargetingModule,
        )

        self.robot = components.robot
        self.controller = components.controller
        self.obs_builder = components.obs_builder
        self.input_provider = components.input_provider
        self.retargeter = components.retargeter
        self.bus = InProcessBus()

        # Provider construction starts the Pico/XR worker immediately.  Keep
        # the remainder of pipeline assembly transactional: malformed video
        # configuration or a SimulationLoop constructor failure must not leave
        # a native SDK thread alive with no owning pipeline to close it.
        try:
            input_cfg = cfg_get(self.cfg, "input", {})
            video_cfg = parse_pico_video_config(input_cfg)
            if str(cfg_get(input_cfg, "provider", "")).lower() == "xrobotoolkit" and video_cfg.enabled:
                raw_video_cfg = cfg_get(input_cfg, "video", {}) or {}
                self.video_runtime = XRoboToolkitVideoRuntime(
                    config=video_cfg,
                    robot=self.robot,
                    direct_host=cfg_get(raw_video_cfg, "direct_host", None),
                    # Keep the raw value so XRoboToolkitVideoRuntime can
                    # reject floats/bools instead of silently truncating
                    # malformed endpoint configuration with ``int(...)``.
                    direct_port=cfg_get(raw_video_cfg, "direct_port", 12345),
                )
            else:
                self.video_runtime = PicoVideoRuntime(
                    provider=self.input_provider,
                    config=video_cfg,
                    robot=self.robot,
                )
            self.loop = SimulationLoop(
                cast(Any, self.robot),
                cast(Any, self.controller),
                cast(Any, self.obs_builder),
                cast(Any, self.bus),
                components.sim_cfg,
                viewers=components.viewers,
                video_runtime=self.video_runtime,
                console=console,
            )
        except BaseException:
            close = getattr(self.input_provider, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    # Preserve the original construction error; provider
                    # shutdown is best effort (and XR close itself is
                    # bounded) during rollback.
                    pass
            raise

    def run(self, num_steps: int) -> dict[str, float | int | str]:
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative (0 = infinite)")

        self.controller.reset()
        try:
            return dict(self.loop.run(cast(Any, self.input_provider), cast(Any, self.retargeter), num_steps=num_steps))
        finally:
            close = getattr(self.input_provider, "close", None)
            if callable(close):
                close()
