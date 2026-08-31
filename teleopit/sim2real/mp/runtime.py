"""Multiprocess sim2real runtime using ZMQ and shared memory."""

from __future__ import annotations

import logging
import multiprocessing as mp
from multiprocessing.synchronize import Event as MpEvent
from enum import Enum
from pathlib import Path
import sys
import time
import uuid
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from teleopit.constants import FULL_QPOS_DIM, NUM_JOINTS, ROOT_DIM
from teleopit.high_level_policy.client import PolicyActionChunk
from teleopit.high_level_policy.config import (
    parse_high_level_policy_config,
    parse_high_level_policy_safety_config,
)
from teleopit.high_level_policy.scheduler import HighLevelPolicyScheduler, PolicyFrameTransform
from teleopit.controllers.observation import VelCmdObservationBuilder, align_motion_qpos_yaw
from teleopit.controllers.rl_policy import RLPolicyController
from teleopit.inputs.bvh_provider import BVHInputProvider
from teleopit.inputs.human_frame_validation import validate_human_frame
from teleopit.inputs.pico4_provider import Pico4InputProvider
from teleopit.inputs.pico_video import PicoVideoRuntime, bridge_video_source, parse_pico_video_config
from teleopit.inputs.realtime_packet import ControlEvent, ControlEventType
from teleopit.retargeting.core import RetargetingModule
from teleopit.runtime.offline_playback import OfflinePlaybackController
from teleopit.runtime.common import cfg_get, parse_viewers, require_section
from teleopit.runtime.console import (
    OPERATOR_LOGGER_NAME,
    PlainConsole,
    configure_runtime_logging,
    console_show_timing,
    console_timing_interval_s,
    sim2real_keyboard_controls,
)
from teleopit.runtime.factory import _build_policy_components, build_simulation_cfg
from teleopit.runtime.arm_mocap import (
    compose_arm_reference,
    compose_arm_reference_window,
    parse_arm_joint_indices,
)
from teleopit.runtime.mocap_session import MocapSessionManager, MocapSessionState
from teleopit.runtime.reference_config import parse_reference_config
from teleopit.runtime.terminal_keyboard import TerminalKeyboardReader
from teleopit.recording.hdf5 import (
    DEFAULT_ROBOT_TYPE,
    NO_HAND_TYPE,
    NO_NECK_TYPE,
    build_mode_observation,
    build_mp4_video_config,
    build_observation_state,
    build_recording_schema,
    normalize_action_reference_qpos,
    normalize_hand_action,
    build_neck_action,
)
from teleopit.sim.reference_motion import OfflineReferenceMotion
from teleopit.sim.reference_timeline import ReferenceTimeline, ReferenceWindow, ReferenceWindowBuilder
from teleopit.sim.reference_utils import (
    build_offline_reference_window,
    build_static_reference_window,
    obs_builder_requires_reference_window,
)
from teleopit.sim.realtime_utils import RealtimeReferenceManager
from teleopit.sim.viewer_subprocess import start_robot_viewer
from teleopit.sim2real.hands.worker import build_hand_runtime
from teleopit.sim2real.hands.base import HandPoseCommand
from teleopit.sim2real.hands.linkerhand_l6 import parse_linkerhand_l6_config
from teleopit.sim2real.hands.linkerhand_o6 import parse_linkerhand_o6_config
from teleopit.sim2real.neck.config import parse_neck_config
from teleopit.sim2real.neck.worker import build_neck_runtime, head_pose_packet, mode_packet_active
from teleopit.sim2real.mp.ipc import (
    BODY_TOPIC,
    COMMAND_TOPIC,
    CONTROL_EVENTS_TOPIC,
    CONTROLLER_TOPIC,
    HAND_COMMAND_TOPIC,
    HEAD_POSE_TOPIC,
    HAND_TOPIC,
    HEALTH_TOPIC,
    HIGH_LEVEL_POLICY_ACTION_TOPIC,
    HIGH_LEVEL_POLICY_OBSERVATION_TOPIC,
    HIGH_LEVEL_POLICY_SESSION_TOPIC,
    HIGH_LEVEL_POLICY_STATUS_TOPIC,
    HIGH_LEVEL_POLICY_TARGET_TOPIC,
    MODE_TOPIC,
    NECK_COMMAND_TOPIC,
    RECORD_TOPIC,
    REFERENCE_TOPIC,
    VIDEO_TOPIC,
    LatestSubscriber,
    Sim2RealIpcEndpoints,
    ZmqPublisher,
    default_endpoints,
)
from teleopit.sim2real.mp.messages import (
    BodyFramePacket,
    CommandPacket,
    ControlEventsPacket,
    HandCommandPacket,
    HealthPacket,
    HighLevelPolicyActionPacket,
    HighLevelPolicyObservationPacket,
    HighLevelPolicySessionPacket,
    HighLevelPolicyStatusPacket,
    HighLevelPolicyTargetPacket,
    ModeStatePacket,
    NeckCommandPacket,
    ReferencePacket,
    RecordStepPacket,
    SnapshotPacket,
    SharedFrameDescriptor,
)
from teleopit.sim2real.mp.shm import SharedFrameRingReader, SharedFrameRingWriter
from teleopit.sim2real.reference_processor import Sim2RealReferenceProcessor
from teleopit.sim2real.remote import UnitreeRemote
from teleopit.sim2real.safety import Sim2RealSafetyManager
from teleopit.sim2real.unitree_g1 import UnitreeG1Robot

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - OmegaConf is a project dependency.
    OmegaConf = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)
operator_logger = logging.getLogger(OPERATOR_LOGGER_NAME)

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARM_MOCAP_REFERENCE_COMMAND = "arm_mocap_reference"
DISARM_MOCAP_REFERENCE_COMMAND = "disarm_mocap_reference"
HIGH_LEVEL_POLICY_FAULT_COMMAND = "high_level_policy_fault"


class RobotMode(Enum):
    IDLE = "idle"
    STANDING = "standing"
    MOCAP = "mocap"
    ARMS = "arms"
    POLICY = "policy"
    DAMPING = "damping"


class _LoopTimingReporter:
    def __init__(
        self,
        *,
        target_period_s: float,
        log_interval_s: float = 1.0,
        deadline_miss_tolerance_s: float = 0.001,
        enabled: bool = True,
    ) -> None:
        self._target_period_s = float(target_period_s)
        self._log_interval_s = float(log_interval_s)
        self._deadline_miss_tolerance_s = float(deadline_miss_tolerance_s)
        self._enabled = bool(enabled)
        self._window_start_s: float | None = None
        self._loop_ms: list[float] = []
        self._late_ms: list[float] = []
        self._work_ms: list[float] = []
        self._pico_age_ms: list[float] = []
        self._deadline_miss_count = 0
        self._work_overrun_count = 0

    def record(self, *, loop_start_s: float, work_elapsed_s: float, cycle_elapsed_s: float, pico_age_s: float | None) -> None:
        if self._window_start_s is None:
            self._window_start_s = float(loop_start_s)
        self._loop_ms.append(float(cycle_elapsed_s) * 1000.0)
        self._late_ms.append(max(0.0, float(cycle_elapsed_s) - self._target_period_s) * 1000.0)
        self._work_ms.append(float(work_elapsed_s) * 1000.0)
        if pico_age_s is not None:
            self._pico_age_ms.append(float(pico_age_s) * 1000.0)
        if cycle_elapsed_s > self._target_period_s + self._deadline_miss_tolerance_s:
            self._deadline_miss_count += 1
        if work_elapsed_s > self._target_period_s + 1e-9:
            self._work_overrun_count += 1
        if loop_start_s - self._window_start_s >= self._log_interval_s:
            self._emit(loop_start_s)

    def _emit(self, end_s: float) -> None:
        sample_count = len(self._loop_ms)
        if sample_count <= 0:
            self._reset(end_s)
            return
        if not self._enabled:
            self._reset(end_s)
            return
        loop_summary = self._summarize(self._loop_ms)
        late_summary = self._summarize(self._late_ms)
        work_summary = self._summarize(self._work_ms)
        message = (
            "Timing stats | samples=%d window=%.1fs | "
            "loop_ms p50=%.2f p95=%.2f p99=%.2f max=%.2f | "
            "late_ms p50=%.2f p95=%.2f p99=%.2f max=%.2f deadline_miss(>%.2fms)=%d/%d | "
            "work_ms p50=%.2f p95=%.2f p99=%.2f max=%.2f work_overrun=%d/%d"
        )
        args: list[object] = [
            sample_count,
            end_s - float(self._window_start_s),
            *loop_summary,
            *late_summary,
            self._deadline_miss_tolerance_s * 1000.0,
            self._deadline_miss_count,
            sample_count,
            *work_summary,
            self._work_overrun_count,
            sample_count,
        ]
        if self._pico_age_ms:
            message += " | reference_age_ms p50=%.2f p95=%.2f p99=%.2f max=%.2f"
            args.extend(self._summarize(self._pico_age_ms))
        operator_logger.info(message, *args)
        self._reset(end_s)

    def _reset(self, window_start_s: float) -> None:
        self._window_start_s = float(window_start_s)
        self._loop_ms.clear()
        self._late_ms.clear()
        self._work_ms.clear()
        self._pico_age_ms.clear()
        self._deadline_miss_count = 0
        self._work_overrun_count = 0

    @staticmethod
    def _summarize(samples: list[float]) -> tuple[float, float, float, float]:
        values = np.asarray(samples, dtype=np.float64)
        if values.size <= 0:
            return 0.0, 0.0, 0.0, 0.0
        p50, p95, p99 = np.percentile(values, [50.0, 95.0, 99.0])
        return float(p50), float(p95), float(p99), float(np.max(values))


def _parse_sim2real_viewers(cfg: Any) -> set[str]:
    viewers = parse_viewers(cfg)
    unsupported = viewers.difference({"retarget"})
    if unsupported:
        raise ValueError(
            f"Sim2real supports only the optional 'retarget' viewer; got unsupported viewers {sorted(unsupported)}. "
            "Use viewers=retarget or viewers=none."
        )
    return viewers


class _Sim2RealRetargetViewer:
    def __init__(self, *, xml_path: str | None, enabled: bool) -> None:
        self._entry: tuple[Any, Any, Any, Any] | None = None
        if not enabled:
            return
        if not xml_path:
            raise ValueError("Sim2real retarget viewer requires robot.xml_path to be set.")
        self._entry = start_robot_viewer(xml_path, FULL_QPOS_DIM, True, "Retarget", 900, 50)

    def write(self, qpos: Float64Array) -> None:
        if self._entry is None:
            return
        _, arr, alive, _ = self._entry
        if not alive.value:
            return
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qpos.shape[0] < FULL_QPOS_DIM:
            return
        with arr.get_lock():
            arr[:FULL_QPOS_DIM] = qpos[:FULL_QPOS_DIM].tolist()

    def shutdown(self) -> None:
        if self._entry is None:
            return
        proc, _, _, shutdown = self._entry
        shutdown.set()
        proc.join(timeout=3)
        if proc.is_alive():
            proc.terminate()
        self._entry = None


def _plain_cfg(cfg: Any) -> dict[str, Any]:
    if isinstance(cfg, dict):
        return dict(cfg)
    if OmegaConf is not None and OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    raise TypeError(f"Unsupported sim2real cfg type for multiprocessing: {type(cfg)!r}")


def _mp_cfg(cfg: Any) -> Any:
    return cfg_get(cfg, "runtime", {}) or {}


def _input_provider_kind(cfg: Any) -> str:
    return str(cfg_get(cfg_get(cfg, "input", {}) or {}, "provider", "bvh")).strip().lower()


def _high_level_policy_enabled(cfg: Any) -> bool:
    policy_cfg = cfg_get(cfg, "high_level_policy", {}) or {}
    return bool(cfg_get(policy_cfg, "enabled", False))


def _recording_cfg(cfg: Any) -> Any:
    return cfg_get(cfg, "recording", {}) or {}


def _recording_enabled(cfg: Any) -> bool:
    return bool(cfg_get(_recording_cfg(cfg), "enabled", False))


def _recording_camera_cfg(cfg: Any) -> Any:
    return cfg_get(_recording_cfg(cfg), "camera", {}) or {}


def _recording_hardware_types(cfg: Any) -> tuple[str, str, str]:
    robot_cfg = cfg_get(cfg, "robot", {}) or {}
    robot_type = str(cfg_get(robot_cfg, "type", DEFAULT_ROBOT_TYPE)).strip().lower()
    hands_cfg = cfg_get(cfg, "hands", {}) or {}
    hand_type = (
        str(cfg_get(hands_cfg, "driver", "linkerhand_l6")).strip().lower()
        if bool(cfg_get(hands_cfg, "enabled", False))
        else NO_HAND_TYPE
    )
    neck_cfg = cfg_get(cfg, "neck", {}) or {}
    neck_type = (
        str(cfg_get(neck_cfg, "driver", "openneck")).strip().lower()
        if bool(cfg_get(neck_cfg, "enabled", False))
        else NO_NECK_TYPE
    )
    return robot_type, hand_type, neck_type


def _configured_open_hand_pose(cfg: Any) -> tuple[np.ndarray, np.ndarray]:
    hands_cfg = cfg_get(cfg, "hands", {}) or {}
    driver = str(cfg_get(hands_cfg, "driver", "linkerhand_l6")).strip().lower()
    if bool(cfg_get(hands_cfg, "enabled", False)):
        if driver == "linkerhand_o6":
            hand_cfg = parse_linkerhand_o6_config(cfg)
        else:
            hand_cfg = parse_linkerhand_l6_config(cfg)
        pose = np.asarray(hand_cfg.open_pose, dtype=np.float32).reshape(-1)
    elif driver == "linkerhand_o6":
        pose = np.array([250, 250, 250, 250, 250, 250], dtype=np.float32)
    else:
        driver_cfg = cfg_get(hands_cfg, "linkerhand_l6", {}) or {}
        thumb_yaw = int(cfg_get(driver_cfg, "thumb_yaw_center", 10))
        pose = np.array([250, thumb_yaw, 250, 250, 250, 250], dtype=np.float32)
    if pose.shape[0] != 6:
        raise ValueError(f"hands.{driver}.open_pose must contain 6 values")
    return pose.copy(), pose.copy()


def _validate_new_runtime_config(cfg: Any) -> None:
    legacy_keys = [key for key in ("sim2real_runtime", "multiprocess", "dexterous_hand") if cfg_get(cfg, key, None) is not None]
    if legacy_keys:
        raise ValueError(
            "Legacy sim2real config keys are no longer supported: "
            f"{', '.join(legacy_keys)}. Use input.provider, runtime, and hands instead."
        )
    if _high_level_policy_enabled(cfg):
        raise ValueError(
            "high_level_policy.enabled=true requires the independent "
            "scripts/run/run_high_level_policy_sim2real.py entry point"
        )
    provider = _input_provider_kind(cfg)
    if provider not in ("pico4", "bvh"):
        raise ValueError(f"sim2real input.provider must be pico4 or bvh, got {provider!r}")
    hands_cfg = cfg_get(cfg, "hands", {}) or {}
    if bool(cfg_get(hands_cfg, "enabled", False)) and provider != "pico4":
        raise ValueError("hands.enabled=true requires input.provider=pico4")
    neck_cfg = parse_neck_config(cfg)
    if neck_cfg.enabled:
        if provider != "pico4":
            raise ValueError("neck.enabled=true requires input.provider=pico4")
        if neck_cfg.driver != "openneck":
            raise ValueError(f"Unsupported neck.driver={neck_cfg.driver!r}; supported drivers: openneck")
    if _recording_enabled(cfg):
        if provider != "pico4":
            raise ValueError("recording.enabled=true requires input.provider=pico4")
        if bool(cfg_get(hands_cfg, "enabled", False)):
            hand_sides = {str(side).strip().lower() for side in cfg_get(hands_cfg, "sides", ("left", "right"))}
            if hand_sides != {"left", "right"}:
                raise ValueError("hand-state recording requires hands.sides=[left, right]")
        if neck_cfg.enabled and neck_cfg.dry_run:
            raise ValueError("neck-state recording requires neck.dry_run=false")
        rec_cfg = _recording_cfg(cfg)
        if str(cfg_get(rec_cfg, "format", "hdf5")) != "hdf5":
            raise ValueError("Only recording.format=hdf5 is supported")
        if str(cfg_get(rec_cfg, "control", "terminal")) != "terminal":
            raise ValueError("Only recording.control=terminal is supported")
        build_mp4_video_config(cfg_get(rec_cfg, "video", {}) or {})
        camera_cfg = _recording_camera_cfg(cfg)
        if not bool(cfg_get(camera_cfg, "enabled", True)):
            raise ValueError("recording.camera.enabled=false is not supported for HDF5 recording")
        if str(cfg_get(camera_cfg, "source", "realsense")).lower() != "realsense":
            raise ValueError("recording.camera.source must be realsense")
        if int(cfg_get(rec_cfg, "fps", 30)) != int(cfg_get(camera_cfg, "fps", 30)):
            raise ValueError("recording.fps must match recording.camera.fps")
        robot_type, hand_type, neck_type = _recording_hardware_types(cfg)
        build_recording_schema(
            camera_cfg,
            fps=int(cfg_get(rec_cfg, "fps", 30)),
            robot_type=robot_type,
            hand_type=hand_type,
            neck_type=neck_type,
        )
        input_video = parse_pico_video_config(cfg_get(cfg, "input", {}) or {})
        if not input_video.enabled:
            raise ValueError("recording.enabled=true requires input.video.enabled=true")
        if input_video.source != "realsense":
            raise ValueError("recording.enabled=true requires input.video.source=realsense")
        if int(input_video.width) != int(cfg_get(camera_cfg, "width", 640)):
            raise ValueError("recording.camera.width must match input.video.width")
        if int(input_video.height) != int(cfg_get(camera_cfg, "height", 480)):
            raise ValueError("recording.camera.height must match input.video.height")
        if int(input_video.fps) != int(cfg_get(camera_cfg, "fps", 30)):
            raise ValueError("recording.camera.fps must match input.video.fps")
        input_device = input_video.device
        camera_device = cfg_get(camera_cfg, "device", None)
        camera_device = None if camera_device in (None, "", "null") else str(camera_device)
        if input_device != camera_device:
            raise ValueError("recording.camera.device must match input.video.device")


def _require_recording_dependencies() -> None:
    try:
        from teleopit.recording.hdf5 import TeleopitHDF5Recorder

        TeleopitHDF5Recorder.create
    except Exception as exc:
        raise RuntimeError("HDF5 recording adapter is unavailable") from exc


def _worker_loop(name: str, cfg: dict[str, Any], fn: Callable[[], None]) -> None:
    configure_runtime_logging(cfg, force=True)
    try:
        fn()
    except KeyboardInterrupt:
        pass
    except BaseException:
        logger.exception("%s worker crashed", name)
        raise


def _human_frame_is_valid(frame: object) -> bool:
    return validate_human_frame(frame).valid


class Sim2RealRuntime:
    """Supervisor facade for the process-isolated sim2real runtime."""

    def __init__(self, cfg: Any, *, console: PlainConsole | None = None) -> None:
        self.cfg = _plain_cfg(cfg)
        _validate_new_runtime_config(self.cfg)

        mp_cfg = _mp_cfg(self.cfg)
        video_cfg = parse_pico_video_config(cfg_get(self.cfg, "input", {}))
        if video_cfg.enabled and video_cfg.source not in ("realsense", "test-pattern"):
            raise ValueError(
                "Sim2RealRuntime only supports input.video.source=realsense or test-pattern"
            )
        self._ctx = mp.get_context(str(cfg_get(mp_cfg, "start_method", "spawn")))
        self._stop_event = self._ctx.Event()
        self._processes: list[mp.Process] = []
        self._shutdown_timeout_s = float(cfg_get(mp_cfg, "shutdown_timeout_s", 3.0))
        self._endpoints = default_endpoints(
            host=str(cfg_get(mp_cfg, "host", "127.0.0.1")),
            base_port=int(cfg_get(mp_cfg, "base_port", 39700)),
        )
        self._command_pub: ZmqPublisher | None = None
        self._keyboard: TerminalKeyboardReader | None = None
        self._console = console or PlainConsole(title="Teleopit sim2real", enabled=False)
        self._console_controls = sim2real_keyboard_controls(self.cfg)
        if _recording_enabled(self.cfg):
            _require_recording_dependencies()
            if not sys.stdin.isatty():
                raise RuntimeError("recording.enabled=true requires an interactive TTY for terminal controls")
            if console is None:
                self._console = PlainConsole(title="Teleopit sim2real")

    def run(self) -> None:
        operator_logger.info("runtime starting")
        try:
            self._start_processes()
            if _recording_enabled(self.cfg):
                self._command_pub = ZmqPublisher(self._endpoints.command_pub)
                self._keyboard = TerminalKeyboardReader()
                operator_logger.info("keyboard recording controls active: R start, S save, D discard, Q shutdown, H help")
            reported_noncritical_dead: set[str] = set()
            while not self._stop_event.is_set():
                self._poll_terminal_recording_controls()
                time.sleep(0.2)
                critical_names = {"robot_control", "reference"}
                critical_dead = [
                    process.name
                    for process in self._processes
                    if not process.is_alive()
                    and process.exitcode not in (None, 0)
                    and process.name in critical_names
                ]
                if critical_dead:
                    operator_logger.error("critical worker exited: %s", ", ".join(critical_dead))
                    self._stop_event.set()
                    break
                noncritical_dead = [
                    process.name
                    for process in self._processes
                    if not process.is_alive()
                    and process.exitcode not in (None, 0)
                    and process.name not in critical_names
                    and process.name not in reported_noncritical_dead
                ]
                if noncritical_dead:
                    operator_logger.warning(
                        "non-critical worker exited: %s; G1 control remains active",
                        ", ".join(noncritical_dead),
                    )
                    reported_noncritical_dead.update(noncritical_dead)
        except KeyboardInterrupt:
            operator_logger.info("keyboard interrupt -> shutting down")
            self._stop_event.set()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._command_pub is not None:
            self._command_pub.publish(COMMAND_TOPIC, CommandPacket(command="shutdown", timestamp_s=time.monotonic()))
        for process in self._processes:
            process.join(timeout=self._shutdown_timeout_s)
        for process in self._processes:
            if process.is_alive():
                operator_logger.warning("terminating worker %s", process.name)
                process.terminate()
                process.join(timeout=1.0)
        self._processes.clear()
        if self._keyboard is not None:
            self._keyboard.close()
            self._keyboard = None
        if self._command_pub is not None:
            self._command_pub.close()
            self._command_pub = None

    def _start_processes(self) -> None:
        if self._processes:
            return

        specs: list[tuple[str, Callable[..., None]]] = []
        if _input_provider_kind(self.cfg) == "pico4":
            specs.append(("pico_input", _run_pico_io_worker))
        specs.extend(
            [
                ("reference", _run_reference_worker),
                ("robot_control", _run_robot_control_worker),
            ]
        )
        hands_cfg = cfg_get(self.cfg, "hands", {}) or {}
        if bool(cfg_get(hands_cfg, "enabled", False)):
            specs.append(("hand_worker", _run_hand_worker))
        neck_cfg = parse_neck_config(self.cfg)
        if neck_cfg.enabled:
            specs.append(("neck_worker", _run_neck_worker))
        if _recording_enabled(self.cfg):
            specs.append(("recording_worker", _run_recording_worker))
        video_cfg = parse_pico_video_config(cfg_get(self.cfg, "input", {}))
        if video_cfg.enabled:
            logger.info("Pico video runs inside pico_input so frames are pushed directly to PicoBridge")

        for name, target in specs:
            process = self._ctx.Process(
                name=name,
                target=target,
                args=(self.cfg, self._endpoints, self._stop_event),
            )
            process.start()
            self._processes.append(process)

    def _poll_terminal_recording_controls(self) -> None:
        if self._keyboard is None or self._command_pub is None:
            return
        events = self._keyboard.poll()
        if not events:
            return
        for event in events:
            normalized = str(event.key).strip().lower()
            if normalized == "h":
                self._console.help(self._console_controls)
                continue
            command = map_recording_key_to_command(event.key)
            if command is None:
                continue
            self._command_pub.publish(COMMAND_TOPIC, CommandPacket(command=command, timestamp_s=time.monotonic()))
            self._console.key_feedback(str(event.key).upper(), _recording_command_label(command))
            if command == "shutdown":
                self._stop_event.set()


def map_recording_key_to_command(key: str) -> str | None:
    normalized = str(key).strip().lower()
    if normalized == "r":
        return "record_start"
    if normalized == "s":
        return "record_save"
    if normalized == "d":
        return "record_discard"
    if normalized == "q":
        return "shutdown"
    return None


def _recording_command_label(command: str) -> str:
    if command == "record_start":
        return "start recording"
    if command == "record_save":
        return "save recording"
    if command == "record_discard":
        return "discard recording"
    if command == "shutdown":
        return "shutdown"
    return command


def _run_pico_io_worker(
    cfg: dict[str, Any],
    endpoints: Sim2RealIpcEndpoints,
    stop_event: MpEvent,
) -> None:
    def _main() -> None:
        input_cfg = cfg_get(cfg, "input", {}) or {}
        video_cfg = parse_pico_video_config(input_cfg)
        provider = Pico4InputProvider(
            human_format=str(cfg_get(input_cfg, "human_format", "pico_bridge")),
            timeout=float(cfg_get(input_cfg, "pico4_timeout", 60.0)),
            buffer_size=int(cfg_get(input_cfg, "pico4_buffer_size", 60)),
            timestamp_gap_reset_s=float(cfg_get(input_cfg, "pico4_timestamp_gap_reset_s", 0.15)),
            pause_button=cfg_get(input_cfg, "pause_button", "A"),
            pause_debounce_s=float(cfg_get(input_cfg, "pause_debounce_s", 0.25)),
            arms_button=cfg_get(input_cfg, "arms_button", "B"),
            arms_debounce_s=float(cfg_get(input_cfg, "arms_debounce_s", cfg_get(input_cfg, "pause_debounce_s", 0.25))),
            bridge_host=str(cfg_get(input_cfg, "bridge_host", "0.0.0.0")),
            bridge_port=int(cfg_get(input_cfg, "bridge_port", 63901)),
            bridge_discovery=bool(cfg_get(input_cfg, "bridge_discovery", True)),
            bridge_advertise_ip=cfg_get(input_cfg, "bridge_advertise_ip", None),
            bridge_video=bridge_video_source(video_cfg),
            bridge_video_enabled=video_cfg.enabled,
            bridge_start_timeout=float(cfg_get(input_cfg, "bridge_start_timeout", 10.0)),
            bridge_history_size=int(cfg_get(input_cfg, "bridge_history_size", 120)),
        )

        body_pub = ZmqPublisher(endpoints.body_pub)
        head_pose_pub: ZmqPublisher | None = None

        def _disable_head_pose_publisher(message: str) -> None:
            nonlocal head_pose_pub
            logger.exception(message)
            failed_publisher = head_pose_pub
            head_pose_pub = None
            if failed_publisher is None:
                return
            try:
                failed_publisher.close()
            except Exception:
                logger.exception("Failed to close disabled OpenNeck head-pose publisher")

        if parse_neck_config(cfg).enabled:
            try:
                head_pose_pub = ZmqPublisher(endpoints.head_pose_pub)
            except Exception:
                _disable_head_pose_publisher(
                    "OpenNeck head-pose IPC setup failed; neck control is disabled "
                    "while pico_input continues"
                )
        hand_pub = ZmqPublisher(endpoints.hand_pub)
        controller_pub = ZmqPublisher(endpoints.controller_pub)
        events_pub = ZmqPublisher(endpoints.control_events_pub)
        health_pub = ZmqPublisher(endpoints.health_pub)
        video_pub = ZmqPublisher(endpoints.video_pub) if _recording_enabled(cfg) else None
        command_sub = LatestSubscriber(endpoints.command_pub, COMMAND_TOPIC)
        frame_writer: SharedFrameRingWriter | None = None

        def _publish_recording_frame(frame: NDArray[np.generic], timestamp_s: float) -> None:
            nonlocal frame_writer
            if video_pub is None:
                return
            if frame_writer is None:
                frame_writer = SharedFrameRingWriter(
                    shape=tuple(np.asarray(frame).shape),
                    dtype=np.uint8,
                    slots=int(cfg_get(_mp_cfg(cfg), "video_slots", 3)),
                )
            descriptor = frame_writer.write(np.asarray(frame, dtype=np.uint8), timestamp_s=float(timestamp_s))
            video_pub.publish(VIDEO_TOPIC, descriptor)

        video_runtime = PicoVideoRuntime(
            provider=provider,
            config=video_cfg,
            frame_callback=_publish_recording_frame if _recording_enabled(cfg) else None,
        )

        hz = float(cfg_get(_mp_cfg(cfg), "pico_input_hz", 120.0))
        sleep_s = 1.0 / max(hz, 1.0)
        last_body_seq = -1
        last_head_pose_seq = -1
        last_hand_seq = -1
        last_controller_seq = -1
        last_video_seq = -1
        last_health_s = 0.0
        try:
            try:
                video_runtime.start()
            except Exception:
                logger.exception(
                    "Pico video startup failed; video is disabled while pico_input and robot control continue"
                )
            while not stop_event.is_set():
                try:
                    video_runtime.tick()
                except Exception:
                    logger.exception(
                        "Pico video runtime failed; video is disabled while pico_input and robot control continue"
                    )
                command = command_sub.recv_latest()
                if isinstance(command, CommandPacket) and command.command == "shutdown":
                    stop_event.set()
                    break

                now = time.monotonic()
                if head_pose_pub is not None:
                    try:
                        head_pose_snapshot = provider.get_head_pose_snapshot()
                        if (
                            head_pose_snapshot is not None
                            and int(head_pose_snapshot.seq) != last_head_pose_seq
                        ):
                            head_pose_pub.publish(
                                HEAD_POSE_TOPIC,
                                SnapshotPacket(
                                    snapshot=head_pose_snapshot,
                                    timestamp_s=float(head_pose_snapshot.timestamp_s),
                                    seq=int(head_pose_snapshot.seq),
                                ),
                            )
                            last_head_pose_seq = int(head_pose_snapshot.seq)
                    except Exception:
                        _disable_head_pose_publisher(
                            "OpenNeck head-pose stream failed; neck control is disabled "
                            "while pico_input continues"
                        )

                if callable(getattr(provider, "has_frame", None)) and provider.has_frame():
                    try:
                        frame, timestamp_s, seq = provider.get_frame_packet()
                    except Exception:
                        logger.exception("pico_input failed to read body frame")
                    else:
                        if int(seq) != last_body_seq:
                            body_pub.publish(
                                BODY_TOPIC,
                                BodyFramePacket(frame=frame, timestamp_s=float(timestamp_s), seq=int(seq)),
                            )
                            last_body_seq = int(seq)

                events = provider.pop_control_events()
                if events:
                    events_pub.publish(
                        CONTROL_EVENTS_TOPIC,
                        ControlEventsPacket(events=tuple(events), timestamp_s=now, seq=last_body_seq),
                    )

                controller_snapshot = provider.get_controller_snapshot()
                if controller_snapshot is not None and int(controller_snapshot.seq) != last_controller_seq:
                    controller_pub.publish(
                        CONTROLLER_TOPIC,
                        SnapshotPacket(
                            snapshot=controller_snapshot,
                            timestamp_s=float(controller_snapshot.timestamp_s),
                            seq=int(controller_snapshot.seq),
                        ),
                    )
                    last_controller_seq = int(controller_snapshot.seq)

                hand_snapshot = provider.get_hand_snapshot()
                if hand_snapshot is not None and int(hand_snapshot.seq) != last_hand_seq:
                    hand_pub.publish(
                        HAND_TOPIC,
                        SnapshotPacket(
                            snapshot=hand_snapshot,
                            timestamp_s=float(hand_snapshot.timestamp_s),
                            seq=int(hand_snapshot.seq),
                        ),
                    )
                    last_hand_seq = int(hand_snapshot.seq)

                if video_cfg.enabled:
                    last_video_seq = int(video_runtime.pushed_frames)

                if now - last_health_s >= 1.0:
                    health_pub.publish(
                        HEALTH_TOPIC,
                        HealthPacket(
                            worker="pico_input",
                            timestamp_s=now,
                            metrics={
                                "body_seq": last_body_seq,
                                "body_fps": float(provider.fps),
                                "head_pose_seq": last_head_pose_seq,
                                "hand_seq": last_hand_seq,
                                "controller_seq": last_controller_seq,
                                "video_seq": last_video_seq,
                            },
                        ),
                    )
                    last_health_s = now
                time.sleep(sleep_s)
        finally:
            try:
                video_runtime.stop()
            except Exception:
                logger.exception("Failed to stop Pico video runtime during pico_input cleanup")
            if frame_writer is not None:
                frame_writer.close(unlink=True)
            command_sub.close()
            if head_pose_pub is not None:
                try:
                    head_pose_pub.close()
                except Exception:
                    logger.exception("Failed to close OpenNeck head-pose publisher")
            for publisher in (
                body_pub,
                hand_pub,
                controller_pub,
                events_pub,
                health_pub,
                video_pub,
            ):
                if publisher is not None:
                    publisher.close()
            provider.close()

    _worker_loop("pico_input", cfg, _main)


def _run_reference_worker(
    cfg: dict[str, Any],
    endpoints: Sim2RealIpcEndpoints,
    stop_event: MpEvent,
) -> None:
    if _input_provider_kind(cfg) == "bvh":
        _run_bvh_reference_worker(cfg, endpoints, stop_event)
        return
    _run_pico_reference_worker(cfg, endpoints, stop_event)


def _run_pico_reference_worker(
    cfg: dict[str, Any],
    endpoints: Sim2RealIpcEndpoints,
    stop_event: MpEvent,
) -> None:
    def _main() -> None:
        input_cfg = cfg_get(cfg, "input", {}) or {}
        policy_hz = float(cfg_get(cfg, "policy_hz", 50.0))
        ref_cfg = parse_reference_config(cfg, provider_fps=None)
        reference_window_builder = ReferenceWindowBuilder(
            policy_dt_s=1.0 / policy_hz,
            reference_steps=cfg_get(cfg, "reference_steps", [0]),
        )
        if ref_cfg.retarget_buffer_enabled and ref_cfg.reference_delay_s is not None:
            reference_window_builder.validate_runtime_support(
                delay_s=float(ref_cfg.reference_delay_s or 0.0),
                window_s=ref_cfg.retarget_buffer_window_s,
                config_label="Multiprocess sim2real reference timeline",
            )
        timeline = ReferenceTimeline(window_s=ref_cfg.retarget_buffer_window_s) if ref_cfg.retarget_buffer_enabled else None
        reference_manager = (
            RealtimeReferenceManager(
                reference_window_builder=reference_window_builder,
                warmup_steps=ref_cfg.realtime_buffer_warmup_steps,
            )
            if timeline is not None
            else None
        )

        retargeter = RetargetingModule(
            robot_name=str(cfg_get(input_cfg, "robot_name", "unitree_g1")),
            human_format=str(cfg_get(input_cfg, "human_format", "pico_bridge")),
            actual_human_height=float(cfg_get(input_cfg, "human_height", 1.75)),
        )
        body_sub = LatestSubscriber(endpoints.body_pub, BODY_TOPIC)
        health_sub = LatestSubscriber(endpoints.health_pub, HEALTH_TOPIC)
        command_sub = LatestSubscriber(endpoints.command_pub, COMMAND_TOPIC)
        reference_command_sub = LatestSubscriber(endpoints.reference_command_pub, COMMAND_TOPIC)
        ref_pub = ZmqPublisher(endpoints.reference_pub)
        idle_sleep_s = float(cfg_get(_mp_cfg(cfg), "retarget_idle_sleep_s", 0.001))
        mocap_armed = False
        last_body_seq = -1
        last_body_timestamp_s: float | None = None
        body_dt_s_ema: float | None = None
        latest_body_fps: float | None = None
        resolved_reference_delay_s = (
            float(ref_cfg.reference_delay_s) if ref_cfg.reference_delay_s is not None else None
        )
        runtime_support_validated = ref_cfg.reference_delay_s is not None or not reference_window_builder.requires_timeline
        last_valid_qpos: Float64Array | None = None

        def _reset_realtime_reference_state(*, reset_retargeter: bool) -> None:
            nonlocal last_body_timestamp_s
            nonlocal body_dt_s_ema
            nonlocal resolved_reference_delay_s
            nonlocal runtime_support_validated
            nonlocal last_valid_qpos
            if timeline is not None:
                timeline.clear()
            if reference_manager is not None:
                reference_manager.reset()
            last_body_timestamp_s = None
            body_dt_s_ema = None
            resolved_reference_delay_s = (
                float(ref_cfg.reference_delay_s) if ref_cfg.reference_delay_s is not None else None
            )
            runtime_support_validated = (
                ref_cfg.reference_delay_s is not None or not reference_window_builder.requires_timeline
            )
            last_valid_qpos = None
            if reset_retargeter:
                retargeter.reset()

        def _handle_reference_command(command: CommandPacket | None) -> None:
            nonlocal mocap_armed
            if not isinstance(command, CommandPacket):
                return
            if command.command == "shutdown":
                stop_event.set()
                return
            if command.command == ARM_MOCAP_REFERENCE_COMMAND:
                if mocap_armed:
                    return
                logger.info("reference worker armed for Pico MOCAP")
                mocap_armed = True
                _reset_realtime_reference_state(reset_retargeter=True)
                return
            if command.command == DISARM_MOCAP_REFERENCE_COMMAND:
                if mocap_armed:
                    logger.info("reference worker disarmed for Pico STANDING")
                mocap_armed = False
                _reset_realtime_reference_state(reset_retargeter=True)

        def _publish_invalid_reference(packet: BodyFramePacket, *, elapsed_s: float) -> None:
            qpos = np.zeros(FULL_QPOS_DIM, dtype=np.float64)
            qpos[3] = 1.0
            if last_valid_qpos is not None:
                qpos = np.asarray(last_valid_qpos, dtype=np.float64).copy()
            ref_pub.publish(
                REFERENCE_TOPIC,
                ReferencePacket(
                    qpos=qpos,
                    timestamp_s=time.monotonic(),
                    seq=int(packet.seq),
                    source_timestamp_s=float(packet.timestamp_s),
                    source_seq=int(packet.seq),
                    frame_valid=False,
                    retarget_elapsed_s=elapsed_s,
                ),
            )

        try:
            while not stop_event.is_set():
                _handle_reference_command(command_sub.recv_latest())
                _handle_reference_command(reference_command_sub.recv_latest())
                if stop_event.is_set():
                    break

                health_packet = health_sub.recv_latest()
                if isinstance(health_packet, HealthPacket) and health_packet.worker == "pico_input":
                    metric_fps = health_packet.metrics.get("body_fps")
                    if isinstance(metric_fps, (int, float)) and float(metric_fps) > 0.0:
                        latest_body_fps = float(metric_fps)

                packet = body_sub.recv_latest()
                if packet is None:
                    time.sleep(idle_sleep_s)
                    continue
                if not isinstance(packet, BodyFramePacket) or int(packet.seq) == last_body_seq:
                    continue
                if not mocap_armed:
                    last_body_seq = int(packet.seq)
                    continue
                start_s = time.monotonic()
                frame_valid = _human_frame_is_valid(packet.frame)
                if not frame_valid:
                    last_body_seq = int(packet.seq)
                    last_body_timestamp_s = None
                    body_dt_s_ema = None
                    _publish_invalid_reference(packet, elapsed_s=time.monotonic() - start_s)
                    logger.warning("reference worker dropped invalid body frame seq=%s", packet.seq)
                    continue

                try:
                    retargeted = retargeter.retarget(packet.frame)
                    qpos = np.asarray(retargeted, dtype=np.float64).reshape(-1)
                    reference_window: ReferenceWindow | None = None
                    if timeline is not None:
                        timeline.append(qpos, float(packet.timestamp_s))
                        if reference_manager is not None:
                            reference_manager.note_realtime_frame()
                        if reference_manager is None or not reference_manager.warmup_done:
                            last_body_timestamp_s = float(packet.timestamp_s)
                            last_body_seq = int(packet.seq)
                            continue
                        if last_body_timestamp_s is not None:
                            dt_s = float(packet.timestamp_s) - float(last_body_timestamp_s)
                            if dt_s > 1e-6:
                                body_dt_s_ema = dt_s if body_dt_s_ema is None else 0.9 * body_dt_s_ema + 0.1 * dt_s
                        last_body_timestamp_s = float(packet.timestamp_s)
                        if resolved_reference_delay_s is None:
                            if latest_body_fps is not None and latest_body_fps > 1e-6:
                                resolved_reference_delay_s = 1.0 / latest_body_fps
                            elif body_dt_s_ema is not None and body_dt_s_ema > 1e-6:
                                resolved_reference_delay_s = float(body_dt_s_ema)
                            elif reference_window_builder.requires_timeline:
                                last_body_seq = int(packet.seq)
                                continue
                            else:
                                resolved_reference_delay_s = 0.0
                        if not runtime_support_validated:
                            reference_window_builder.validate_runtime_support(
                                delay_s=float(resolved_reference_delay_s),
                                window_s=ref_cfg.retarget_buffer_window_s,
                                config_label="Multiprocess sim2real reference timeline",
                            )
                            runtime_support_validated = True
                        reference_window, _diag = reference_manager.sample(
                            timeline,
                            time.monotonic() - float(resolved_reference_delay_s),
                        )
                        qpos = reference_window.current_sample().qpos
                    last_valid_qpos = np.asarray(qpos, dtype=np.float64).copy()
                    ref_pub.publish(
                        REFERENCE_TOPIC,
                        ReferencePacket(
                            qpos=np.asarray(qpos, dtype=np.float64).copy(),
                            timestamp_s=time.monotonic(),
                            seq=int(packet.seq),
                            source_timestamp_s=float(packet.timestamp_s),
                            source_seq=int(packet.seq),
                            frame_valid=True,
                            reference_window=reference_window,
                            retarget_elapsed_s=time.monotonic() - start_s,
                        ),
                    )
                    last_body_seq = int(packet.seq)
                except Exception:
                    logger.exception("reference worker failed to retarget body seq=%s", getattr(packet, "seq", None))
        finally:
            body_sub.close()
            health_sub.close()
            command_sub.close()
            reference_command_sub.close()
            ref_pub.close()

    _worker_loop("reference", cfg, _main)


def _run_bvh_reference_worker(
    cfg: dict[str, Any],
    endpoints: Sim2RealIpcEndpoints,
    stop_event: MpEvent,
) -> None:
    def _main() -> None:
        input_cfg = cfg_get(cfg, "input", {}) or {}
        policy_hz = float(cfg_get(cfg, "policy_hz", 50.0))
        provider = BVHInputProvider(
            str(cfg_get(input_cfg, "bvh_file", "")),
            human_format=str(cfg_get(input_cfg, "bvh_format", cfg_get(input_cfg, "human_format", "lafan1"))),
        )
        retargeter = RetargetingModule(
            robot_name=str(cfg_get(input_cfg, "robot_name", "unitree_g1")),
            human_format=str(cfg_get(input_cfg, "human_format", cfg_get(input_cfg, "bvh_format", "lafan1"))),
            actual_human_height=float(cfg_get(input_cfg, "human_height", provider.human_height)),
        )
        offline_reference = OfflineReferenceMotion(provider, retargeter)
        playback_cfg = cfg_get(cfg, "playback", {}) or {}
        playback = OfflinePlaybackController(
            duration_s=offline_reference.duration_s,
            step_dt_s=1.0 / policy_hz,
            pause_on_end=bool(cfg_get(playback_cfg, "pause_on_end", True)),
        )
        reference_window_builder = ReferenceWindowBuilder(
            policy_dt_s=1.0 / policy_hz,
            reference_steps=cfg_get(cfg, "reference_steps", [0]),
        )
        ref_pub = ZmqPublisher(endpoints.reference_pub)
        command_sub = LatestSubscriber(endpoints.command_pub, COMMAND_TOPIC)
        reference_command_sub = LatestSubscriber(endpoints.reference_command_pub, COMMAND_TOPIC)
        mode_sub = LatestSubscriber(endpoints.mode_pub, MODE_TOPIC)
        health_pub = ZmqPublisher(endpoints.health_pub)
        tick_s = 1.0 / policy_hz
        seq = 0
        last_health_s = 0.0
        mocap_active = False

        def _publish(sample_time_s: float, *, frame_valid: bool = True) -> Float64Array | None:
            nonlocal seq
            start_s = time.monotonic()
            sampled = offline_reference.sample(sample_time_s)
            if sampled is None:
                return None
            reference_window = None
            if reference_window_builder.requires_timeline:
                reference_window = build_offline_reference_window(
                    offline_reference,
                    sample_time_s,
                    reference_window_builder,
                    policy_hz,
                )
            qpos = np.asarray(sampled.qpos, dtype=np.float64).copy()
            ref_pub.publish(
                REFERENCE_TOPIC,
                ReferencePacket(
                    qpos=qpos,
                    timestamp_s=time.monotonic(),
                    seq=seq,
                    source_timestamp_s=float(sample_time_s),
                    source_seq=int(sampled.frame_idx0),
                    frame_valid=frame_valid,
                    reference_window=reference_window,
                    retarget_elapsed_s=time.monotonic() - start_s,
                    playback_paused=playback.paused,
                    playback_finished=playback.finished,
                ),
            )
            seq += 1
            return qpos

        try:
            while not stop_event.is_set():
                t0 = time.monotonic()
                command = command_sub.recv_latest()
                if isinstance(command, CommandPacket):
                    if command.command == "shutdown":
                        stop_event.set()
                        break
                reference_command = reference_command_sub.recv_latest()
                if isinstance(reference_command, CommandPacket):
                    command = reference_command
                    if command.command == "pause_mocap":
                        playback.pause()
                    elif command.command == "resume_mocap":
                        if not playback.finished:
                            playback.resume()
                    elif command.command == "replay_mocap":
                        playback.replay()
                mode_packet = mode_sub.recv_latest()
                if isinstance(mode_packet, ModeStatePacket):
                    mocap_active = bool(mode_packet.mocap_active)

                qpos = _publish(playback.current_time_s)
                if qpos is None:
                    playback.finish()
                    _publish(playback.current_time_s)
                elif mocap_active:
                    playback.advance()

                now = time.monotonic()
                if now - last_health_s >= 1.0:
                    health_pub.publish(
                        HEALTH_TOPIC,
                        HealthPacket(
                            worker="reference",
                            timestamp_s=now,
                            metrics={
                                "source": "bvh",
                                "seq": seq,
                                "playback_time_s": float(playback.current_time_s),
                                "paused": int(playback.paused),
                                "finished": int(playback.finished),
                            },
                        ),
                    )
                    last_health_s = now
                elapsed = time.monotonic() - t0
                if elapsed < tick_s:
                    time.sleep(tick_s - elapsed)
        finally:
            command_sub.close()
            reference_command_sub.close()
            mode_sub.close()
            ref_pub.close()
            health_pub.close()

    _worker_loop("reference", cfg, _main)


class _RobotControlWorker:
    def __init__(
        self,
        cfg: dict[str, Any],
        endpoints: Sim2RealIpcEndpoints,
        stop_event: MpEvent,
    ) -> None:
        self.cfg = cfg
        self.endpoints = endpoints
        self.stop_event = stop_event
        self.provider_kind = _input_provider_kind(cfg)
        self.high_level_policy_enabled = _high_level_policy_enabled(cfg)
        self.mode = RobotMode.IDLE
        self.policy_hz = float(cfg_get(cfg, "policy_hz", 50.0))
        self.dt = 1.0 / self.policy_hz

        self.robot = UnitreeG1Robot(cfg_get(cfg, "real_robot"))
        self.remote = UnitreeRemote()
        self.policy, self.obs_builder = self._build_policy_and_obs()

        robot_cfg = cfg_get(cfg, "robot")
        self.default_angles = np.asarray(cfg_get(robot_cfg, "default_angles"), dtype=np.float32)
        default_root_qpos = np.asarray(
            cfg_get(robot_cfg, "mujoco_default_qpos", [0.0, 0.0, 0.0]), dtype=np.float64
        ).reshape(-1)
        self._default_root_pos = np.zeros(3, dtype=np.float64)
        if default_root_qpos.shape[0] >= 3:
            self._default_root_pos[:] = default_root_qpos[:3]
        self.num_actions = int(cfg_get(robot_cfg, "num_actions", NUM_JOINTS))
        self._safety = Sim2RealSafetyManager(cfg, self.robot, self.policy_hz, self.num_actions)
        self._standing_return_ramp_duration = float(cfg_get(cfg, "standing_return_ramp_duration", 0.5))
        self._standing_return_kp_ramp_floor_ratio = float(
            cfg_get(cfg, "standing_return_kp_ramp_floor_ratio", 0.5)
        )
        self._arm_joint_indices = parse_arm_joint_indices(cfg, num_actions=self.num_actions)

        self._ref_cfg = parse_reference_config(cfg, provider_fps=None)
        self._reference_window_builder = ReferenceWindowBuilder(
            policy_dt_s=self.dt,
            reference_steps=cfg_get(cfg, "reference_steps", [0]),
        )
        if self.high_level_policy_enabled and self._reference_window_builder.reference_steps != (0,):
            raise ValueError("High-level policy sim2real currently requires reference_steps=[0]")
        self._ref_proc = Sim2RealReferenceProcessor(
            obs_builder=self.obs_builder,
            policy=self.policy,
            policy_hz=self.policy_hz,
            num_actions=self.num_actions,
            reference_velocity_smoothing_alpha=self._ref_cfg.reference_velocity_smoothing_alpha,
            reference_anchor_velocity_smoothing_alpha=self._ref_cfg.reference_anchor_velocity_smoothing_alpha,
        )

        self._standing_qpos = np.zeros(FULL_QPOS_DIM, dtype=np.float64)
        self._standing_qpos[3] = 1.0
        self._standing_qpos[ROOT_DIM:FULL_QPOS_DIM] = self.default_angles.astype(np.float64)
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)
        self._last_retarget_qpos: Float64Array | None = None
        self._last_commanded_motion_qpos: Float64Array | None = None
        self._last_mocap_hold_reason: str | None = None
        self._mocap_reentry_armed = False
        self._mocap_entry_requested = False
        self._mocap_reference_armed = False
        self._mocap_reference_arm_time_s: float | None = None
        self._mocap_reference_arm_retry_s = float(cfg_get(_mp_cfg(cfg), "mocap_reference_arm_retry_s", 0.1))
        self._mocap_session = MocapSessionManager()

        self._high_level_policy_cfg = (
            parse_high_level_policy_config(cfg) if self.high_level_policy_enabled else None
        )
        self._high_level_policy_safety_cfg = (
            parse_high_level_policy_safety_config(cfg)
            if self.high_level_policy_enabled
            else None
        )
        self._high_level_policy_scheduler = (
            HighLevelPolicyScheduler(
                hold_s=self._high_level_policy_cfg.hold_s,
                safety=self._high_level_policy_safety_cfg,
                output_hz=self.policy_hz,
            )
            if self._high_level_policy_cfg is not None
            else None
        )
        self._policy_entry_pending = False
        self._policy_entry_deadline_s: float | None = None
        self._policy_session_id: str | None = None
        self._policy_frame_transform: PolicyFrameTransform | None = None
        self._policy_paused = False
        self._policy_resume_pending = False
        self._policy_resume_deadline_s: float | None = None
        self._policy_resume_source_timestamp_ns: int | None = None
        self._policy_hold_qpos: Float64Array | None = None
        self._policy_session_seq = 0
        self._policy_observation_seq = 0
        self._policy_target_seq = 0
        self._last_policy_session_publish_s = 0.0
        self._last_policy_video_seq = -1
        self._latest_policy_video: SharedFrameDescriptor | None = None
        self._latest_policy_status: HighLevelPolicyStatusPacket | None = None
        self._last_policy_status_seq = -1

        self._latest_reference: ReferencePacket | None = None
        mp_cfg = _mp_cfg(cfg)
        self._max_reference_age_s = float(cfg_get(mp_cfg, "max_reference_age_s", 0.25))
        self._stale_reference_hold_s = float(cfg_get(mp_cfg, "stale_reference_hold_s", 0.08))
        mocap_sw = cfg_get(cfg, "mocap_switch", {}) or {}
        self._check_frames = int(cfg_get(mocap_sw, "check_frames", 10))
        self._last_reference_seq = -1
        self._consecutive_valid_references = 0

        self._reference_sub = (
            None
            if self.high_level_policy_enabled
            else LatestSubscriber(endpoints.reference_pub, REFERENCE_TOPIC)
        )
        self._events_sub = (
            None
            if self.high_level_policy_enabled
            else LatestSubscriber(endpoints.control_events_pub, CONTROL_EVENTS_TOPIC)
        )
        self._command_sub = LatestSubscriber(endpoints.command_pub, COMMAND_TOPIC)
        self._reference_command_pub = (
            None
            if self.high_level_policy_enabled
            else ZmqPublisher(endpoints.reference_command_pub)
        )
        self._policy_video_sub = (
            LatestSubscriber(endpoints.video_pub, VIDEO_TOPIC)
            if self.high_level_policy_enabled
            else None
        )
        self._policy_hand_state_sub = (
            LatestSubscriber(endpoints.hand_command_pub, HAND_COMMAND_TOPIC)
            if self.high_level_policy_enabled
            else None
        )
        self._policy_neck_state_sub = (
            LatestSubscriber(endpoints.neck_command_pub, NECK_COMMAND_TOPIC)
            if self.high_level_policy_enabled
            else None
        )
        self._policy_action_sub = (
            LatestSubscriber(endpoints.high_level_policy_result_pub, HIGH_LEVEL_POLICY_ACTION_TOPIC)
            if self.high_level_policy_enabled
            else None
        )
        self._policy_status_sub = (
            LatestSubscriber(endpoints.high_level_policy_result_pub, HIGH_LEVEL_POLICY_STATUS_TOPIC)
            if self.high_level_policy_enabled
            else None
        )
        self._policy_control_pub = (
            ZmqPublisher(endpoints.high_level_policy_control_pub)
            if self.high_level_policy_enabled
            else None
        )
        self._mode_pub = ZmqPublisher(endpoints.mode_pub)
        self._record_pub = ZmqPublisher(endpoints.record_pub) if _recording_enabled(cfg) else None
        self._latest_policy_hand_state: HandCommandPacket | None = None
        self._latest_policy_neck_state: NeckCommandPacket | None = None

        viewers = _parse_sim2real_viewers(cfg)
        self._retarget_viewer = _Sim2RealRetargetViewer(
            xml_path=str(cfg_get(robot_cfg, "xml_path", "")) if "retarget" in viewers else None,
            enabled="retarget" in viewers,
        )
        self._mode_seq = 0

    def run(self) -> None:
        operator_logger.info("robot control ready | mode=IDLE | policy_hz=%.0f", self.policy_hz)
        timing = _LoopTimingReporter(
            target_period_s=self.dt,
            log_interval_s=console_timing_interval_s(self.cfg),
            enabled=console_show_timing(self.cfg),
        )
        try:
            while not self.stop_event.is_set():
                t0 = time.monotonic()
                self._drain_ipc()

                remote_bytes = self.robot.get_wireless_remote()
                self.remote.update(remote_bytes)
                if self.remote.LB.pressed and self.remote.RB.pressed:
                    if self.mode != RobotMode.DAMPING:
                        logger.warning("EMERGENCY STOP (L1+R1)")
                        operator_logger.warning("DAMPING requested by emergency stop")
                        self._enter_damping()
                else:
                    self._handle_transitions()
                    if self.mode == RobotMode.STANDING:
                        self._standing_step()
                    elif self.mode in (RobotMode.MOCAP, RobotMode.ARMS):
                        self._mocap_step()
                    elif self.mode == RobotMode.POLICY:
                        self._high_level_policy_step()

                self._publish_mode_state()
                work_elapsed_s = time.monotonic() - t0
                cycle_elapsed_s = self._sleep_until(t0, self.dt)
                timing.record(
                    loop_start_s=t0,
                    work_elapsed_s=work_elapsed_s,
                    cycle_elapsed_s=cycle_elapsed_s,
                    pico_age_s=None if self.high_level_policy_enabled else self._reference_age_s(),
                )
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self.high_level_policy_enabled and self._policy_session_id is not None:
            self._stop_high_level_policy_session()
        if self.mode in (RobotMode.STANDING, RobotMode.MOCAP, RobotMode.ARMS, RobotMode.POLICY):
            try:
                self.robot.set_damping()
                time.sleep(0.5)
            except Exception:
                logger.exception("Failed to send damping during robot_control shutdown")
            try:
                self.robot.exit_debug_mode()
            except Exception:
                logger.exception("Failed to exit debug mode during robot_control shutdown")
        self._retarget_viewer.shutdown()
        for subscriber in (
            self._reference_sub,
            self._events_sub,
            self._policy_video_sub,
            self._policy_hand_state_sub,
            self._policy_neck_state_sub,
            self._policy_action_sub,
            self._policy_status_sub,
        ):
            if subscriber is not None:
                subscriber.close()
        self._command_sub.close()
        if self._reference_command_pub is not None:
            self._reference_command_pub.close()
        if self._policy_control_pub is not None:
            self._policy_control_pub.close()
        self._mode_pub.close()
        if self._record_pub is not None:
            self._record_pub.close()
        self.robot.close()

    def _build_policy_and_obs(self) -> tuple[Any, Any]:
        robot_cfg = require_section(self.cfg, "robot")
        controller_cfg = require_section(self.cfg, "controller")
        sim_cfg = build_simulation_cfg(self.cfg)
        policy, obs_builder = _build_policy_components(
            robot_cfg=robot_cfg,
            controller_cfg=controller_cfg,
            sim_cfg=sim_cfg,
            project_root=PROJECT_ROOT,
            controller_cls=RLPolicyController,
        )
        if not bool(getattr(policy, "_multi_input", False)):
            raise ValueError("Sim2real requires an ONNX policy with dual inputs ('obs' and 'obs_history').")
        return policy, obs_builder

    def _drain_ipc(self) -> None:
        command = self._command_sub.recv_latest()
        if isinstance(command, CommandPacket) and command.command == "shutdown":
            self.stop_event.set()
            return
        if isinstance(command, CommandPacket) and command.command == HIGH_LEVEL_POLICY_FAULT_COMMAND:
            detail = str(
                command.payload.get(
                    "detail",
                    "required high-level-policy input worker exited",
                )
            )
            self._handle_high_level_policy_fault(detail)
            return
        if bool(getattr(self, "high_level_policy_enabled", False)):
            self._drain_high_level_policy_ipc()
            return
        if self._reference_sub is None or self._events_sub is None:
            raise RuntimeError("Teleoperation robot worker is missing reference/event subscribers")
        reference = self._reference_sub.recv_latest()
        if isinstance(reference, ReferencePacket):
            self._note_reference_packet(reference)
        events = self._events_sub.recv_latest()
        if isinstance(events, ControlEventsPacket):
            self._handle_mocap_control_events(events.events)

    def _handle_transitions(self) -> None:
        if bool(getattr(self, "high_level_policy_enabled", False)):
            self._handle_high_level_policy_transitions()
            return
        if self.mode == RobotMode.IDLE:
            if self.remote.start.on_pressed:
                operator_logger.info("Start -> STANDING")
                self._enter_standing()
        elif self.mode == RobotMode.STANDING:
            reentry_request = self._mocap_reentry_armed and self.remote.Y.pressed
            if self.remote.Y.on_pressed or reentry_request:
                self._mocap_entry_requested = True
            if self._mocap_entry_requested:
                self._arm_mocap_reference_if_needed()
                if self._can_switch_to_mocap():
                    operator_logger.info("Y -> MOCAP")
                    self._transition_to_mocap()
                elif self.remote.Y.on_pressed or reentry_request:
                    operator_logger.warning("Y -> waiting for fresh retarget reference")
        elif self.mode in (RobotMode.MOCAP, RobotMode.ARMS):
            if self.provider_kind == "bvh" and self.remote.B.on_pressed:
                operator_logger.info("B -> replay BVH from frame 0")
                self._send_reference_command("replay_mocap")
                self._resume_paused_mocap_if_needed()
                return
            pause_pressed = (
                self.remote.B.on_pressed if self.provider_kind == "pico4" else self.remote.A.on_pressed
            )
            if pause_pressed:
                button = "B" if self.provider_kind == "pico4" else "A"
                if self._mocap_session.state == MocapSessionState.PAUSED:
                    operator_logger.info("%s -> resume playback", button)
                    self._send_reference_command("resume_mocap")
                    self._resume_paused_mocap()
                else:
                    operator_logger.info("%s -> pause playback", button)
                    self._send_reference_command("pause_mocap")
                    self._pause_active_mocap()
                return
            if self.remote.X.on_pressed:
                operator_logger.info("X -> STANDING")
                self._enter_standing()
        elif self.mode == RobotMode.DAMPING:
            if self.remote.start.on_pressed:
                operator_logger.info("Start -> STANDING")
                self._enter_standing()

    def _drain_high_level_policy_ipc(self) -> None:
        if self._policy_video_sub is None or self._policy_action_sub is None or self._policy_status_sub is None:
            raise RuntimeError("High-level policy robot worker is missing IPC subscribers")
        video = self._policy_video_sub.recv_latest()
        if isinstance(video, SharedFrameDescriptor) and int(video.seq) > self._last_policy_video_seq:
            self._latest_policy_video = video
        status = self._policy_status_sub.recv_latest()
        if isinstance(status, HighLevelPolicyStatusPacket) and int(status.seq) > self._last_policy_status_seq:
            self._latest_policy_status = status
            self._last_policy_status_seq = int(status.seq)
            if status.status in ("fault", "unavailable"):
                logger.warning("High-level policy host status=%s: %s", status.status, status.detail)
            current_session = status.session_id == self._policy_session_id
            terminal_fault = status.status == "fault" or (
                status.status == "unavailable" and self.mode == RobotMode.POLICY
            )
            if current_session and terminal_fault:
                self._handle_high_level_policy_fault(status.detail)
                return
        packet = self._policy_action_sub.recv_latest()
        if not isinstance(packet, HighLevelPolicyActionPacket):
            return
        if packet.session_id != self._policy_session_id:
            logger.debug(
                "Discarded high-level policy action for inactive session: active=%r received=%r",
                self._policy_session_id,
                packet.session_id,
            )
            return
        # A request may already be in flight when the operator pauses. Drain
        # its result without replacing the reference frozen at the B press.
        if self._policy_paused and not self._policy_resume_pending:
            return
        scheduler = self._high_level_policy_scheduler
        policy_cfg = self._high_level_policy_cfg
        if scheduler is None or policy_cfg is None:
            return
        now_s = time.monotonic()
        if self.mode == RobotMode.STANDING and self._policy_entry_pending:
            deadline_s = self._policy_entry_deadline_s
            if deadline_s is not None and now_s > deadline_s:
                operator_logger.warning(
                    "High-level policy entry timed out; remaining in STANDING"
                )
                self._enter_standing()
                return
        if self.mode == RobotMode.POLICY and self._policy_resume_pending:
            deadline_s = self._policy_resume_deadline_s
            if deadline_s is not None and now_s > deadline_s:
                self._handle_high_level_policy_fault(
                    "resume timed out waiting for a fresh action chunk"
                )
                return
        result_age_s = now_s - float(packet.received_timestamp_s)
        if (
            not np.isfinite(result_age_s)
            or result_age_s < 0.0
            or result_age_s > policy_cfg.max_result_age_s
        ):
            logger.warning(
                "Rejected stale high-level policy result: age=%.3fs limit=%.3fs",
                result_age_s,
                policy_cfg.max_result_age_s,
            )
            if self.mode == RobotMode.STANDING and self._policy_entry_pending:
                operator_logger.warning(
                    "High-level policy entry failed; received a stale action result"
                )
                self._enter_standing()
            return
        minimum_source_timestamp_ns = self._policy_resume_source_timestamp_ns
        if (
            self._policy_resume_pending
            and minimum_source_timestamp_ns is not None
            and int(packet.source_onboard_monotonic_timestamp_ns)
            < minimum_source_timestamp_ns
        ):
            logger.warning("Discarded pre-resume high-level policy action chunk")
            return
        if self.mode == RobotMode.STANDING and not self._policy_entry_pending:
            return
        chunk = PolicyActionChunk(
            session_id=packet.session_id,
            source_sequence_id=int(packet.source_sequence_id),
            source_onboard_monotonic_timestamp_ns=int(
                packet.source_onboard_monotonic_timestamp_ns
            ),
            action_fps=int(packet.action_fps),
            actions=np.asarray(packet.actions, dtype=np.float32),
            policy_id=str(packet.policy_id),
            server_inference_ms=float(packet.server_inference_ms),
        )
        if self.mode == RobotMode.STANDING:
            try:
                scheduler.accept(chunk, now_s=now_s)
            except ValueError as exc:
                logger.warning("Rejected high-level policy entry chunk: %s", exc)
                operator_logger.warning(
                    "High-level policy entry failed; remaining in STANDING"
                )
                self._enter_standing()
                return
            self._transition_to_high_level_policy()
            return
        try:
            scheduler.accept(chunk, now_s=now_s)
        except ValueError as exc:
            logger.warning("Rejected high-level policy action chunk: %s", exc)
            return
        if self._policy_resume_pending:
            scheduler.resume(now_s)
            self._policy_paused = False
            self._policy_resume_pending = False
            self._policy_resume_deadline_s = None
            self._policy_resume_source_timestamp_ns = None
            operator_logger.info("fresh action chunk -> resume POLICY")

    def _handle_high_level_policy_transitions(self) -> None:
        if self.mode == RobotMode.IDLE:
            if self.remote.start.on_pressed:
                operator_logger.info("Start -> STANDING")
                self._enter_standing()
            return
        if self.mode == RobotMode.STANDING:
            if self.remote.X.on_pressed and self._policy_entry_pending:
                operator_logger.info("X -> cancel high-level policy entry")
                self._enter_standing()
                return
            if self.remote.Y.on_pressed and not self._policy_entry_pending:
                operator_logger.info("Y -> request high-level policy")
                self._begin_high_level_policy_entry()
            if self._policy_entry_pending:
                self._publish_high_level_policy_session(
                    "start",
                    repeat=True,
                )
                deadline_s = self._policy_entry_deadline_s
                if deadline_s is not None and time.monotonic() > deadline_s:
                    operator_logger.warning("High-level policy entry timed out; remaining in STANDING")
                    self._enter_standing()
            return
        if self.mode == RobotMode.POLICY:
            if self.remote.X.on_pressed:
                operator_logger.info("X -> STANDING")
                self._enter_standing()
                return
            if self.remote.B.on_pressed:
                self._toggle_high_level_policy_pause()
                return
            self._publish_high_level_policy_session(
                "resume"
                if self._policy_resume_pending or not self._policy_paused
                else "pause",
                repeat=True,
            )
            return
        if self.mode == RobotMode.DAMPING and self.remote.start.on_pressed:
            operator_logger.info("Start -> STANDING")
            self._enter_standing()

    def _begin_high_level_policy_entry(self) -> None:
        self._start_high_level_policy_entry_session()

    def _build_high_level_policy_boundary_action(self, state: object) -> np.ndarray:
        transform = self._policy_frame_transform
        if transform is None:
            raise RuntimeError("High-level policy entry is missing its frame transform")
        initial_action = np.zeros(50, dtype=np.float32)
        initial_action[:36] = transform.localize_body_action(
            self._build_robot_state_qpos(state)
        )
        return initial_action

    def _build_high_level_policy_reference_action(self, reference_qpos: object) -> np.ndarray:
        transform = self._policy_frame_transform
        if transform is None:
            raise RuntimeError("High-level policy entry is missing its frame transform")
        initial_reference = np.zeros(50, dtype=np.float32)
        initial_reference[:36] = transform.localize_body_action(reference_qpos)
        return initial_reference

    def _start_high_level_policy_entry_session(self) -> None:
        policy_cfg = self._high_level_policy_cfg
        scheduler = self._high_level_policy_scheduler
        if policy_cfg is None or scheduler is None:
            raise RuntimeError("High-level policy runtime is not configured")
        if self._policy_session_id is not None:
            self._publish_high_level_policy_session("stop")
        state = self.robot.get_state()
        root_pos = self._resolve_base_pos(state)
        self._policy_frame_transform = PolicyFrameTransform.from_robot_pose(
            root_pos[:2],
            getattr(state, "quat"),
        )
        active_reference = (
            np.asarray(self._last_commanded_motion_qpos, dtype=np.float64).copy()
            if self._last_commanded_motion_qpos is not None
            else self._standing_qpos.copy()
        )
        session_started_s = time.monotonic()
        self._policy_session_id = uuid.uuid4().hex
        scheduler.reset(
            self._policy_session_id,
            initial_action=self._build_high_level_policy_boundary_action(state),
            initial_reference=self._build_high_level_policy_reference_action(
                active_reference
            ),
            initial_timestamp_s=session_started_s,
        )
        self._policy_entry_pending = True
        self._policy_entry_deadline_s = session_started_s + policy_cfg.entry_timeout_s
        self._policy_paused = False
        self._policy_resume_pending = False
        self._policy_resume_deadline_s = None
        self._policy_resume_source_timestamp_ns = None
        self._policy_hold_qpos = active_reference.copy()
        self._policy_observation_seq = 0
        self._last_policy_video_seq = (
            -1
            if self._latest_policy_video is None
            else int(self._latest_policy_video.seq)
        )
        self._last_policy_session_publish_s = 0.0
        self._publish_high_level_policy_session("start", repeat=False)

    def _publish_high_level_policy_session(self, command: str, *, repeat: bool = False) -> None:
        publisher = self._policy_control_pub
        session_id = self._policy_session_id
        policy_cfg = self._high_level_policy_cfg
        if publisher is None or session_id is None or policy_cfg is None:
            return
        now_s = time.monotonic()
        if repeat and now_s - self._last_policy_session_publish_s < 0.2:
            return
        self._policy_session_seq += 1
        publisher.publish(
            HIGH_LEVEL_POLICY_SESSION_TOPIC,
            HighLevelPolicySessionPacket(
                session_id=session_id,
                task=policy_cfg.task,
                command=str(command),
                timestamp_s=now_s,
                seq=self._policy_session_seq,
            ),
        )
        self._last_policy_session_publish_s = now_s

    def _publish_high_level_policy_observation(self, robot_state: object) -> None:
        if not (self._policy_entry_pending or self.mode == RobotMode.POLICY):
            return
        if self._policy_paused and not self._policy_resume_pending:
            return
        publisher = self._policy_control_pub
        frame = self._latest_policy_video
        scheduler = self._high_level_policy_scheduler
        session_id = self._policy_session_id
        policy_cfg = self._high_level_policy_cfg
        if (
            publisher is None
            or frame is None
            or scheduler is None
            or session_id is None
            or policy_cfg is None
        ):
            return
        if int(frame.seq) <= self._last_policy_video_seq:
            return
        now_s = time.monotonic()
        if abs(now_s - float(frame.timestamp_s)) > policy_cfg.max_observation_age_s:
            return
        self._drain_high_level_policy_hardware_state()
        hardware_state = self._high_level_policy_hardware_state(
            now_s=now_s,
            max_age_s=policy_cfg.max_observation_age_s,
        )
        if hardware_state is None:
            return
        dex_state, neck_state = hardware_state
        source_reference_root_pose = scheduler.reference_root_pose_at(
            float(frame.timestamp_s)
        )
        if source_reference_root_pose is None:
            return
        body_joint_positions = build_observation_state(robot_state)[:NUM_JOINTS]
        sequence_id = self._policy_observation_seq
        publisher.publish(
            HIGH_LEVEL_POLICY_OBSERVATION_TOPIC,
            HighLevelPolicyObservationPacket(
                session_id=session_id,
                sequence_id=sequence_id,
                onboard_monotonic_timestamp_ns=int(round(float(frame.timestamp_s) * 1e9)),
                body_joint_positions=body_joint_positions.astype(
                    np.float32,
                    copy=True,
                ),
                dex_state=dex_state,
                neck_state=neck_state,
                source_reference_root_pose=source_reference_root_pose,
                frame=frame,
                timestamp_s=now_s,
            ),
        )
        self._policy_observation_seq += 1
        self._last_policy_video_seq = int(frame.seq)

    def _drain_high_level_policy_hardware_state(self) -> None:
        hand_subscriber = self._policy_hand_state_sub
        neck_subscriber = self._policy_neck_state_sub
        if hand_subscriber is None or neck_subscriber is None:
            return
        hand_state = hand_subscriber.recv_latest()
        if isinstance(hand_state, HandCommandPacket):
            self._latest_policy_hand_state = hand_state
        neck_state = neck_subscriber.recv_latest()
        if isinstance(neck_state, NeckCommandPacket):
            self._latest_policy_neck_state = neck_state

    def _high_level_policy_hardware_state(
        self,
        *,
        now_s: float,
        max_age_s: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        hand = self._latest_policy_hand_state
        neck = self._latest_policy_neck_state
        if (
            hand is None
            or hand.left_state is None
            or hand.right_state is None
            or neck is None
            or neck.state_yaw_deg is None
            or neck.state_pitch_deg is None
        ):
            return None
        hand_age_s = float(now_s) - float(hand.timestamp_s)
        neck_age_s = float(now_s) - float(neck.timestamp_s)
        if not (
            np.isfinite(hand_age_s)
            and 0.0 <= hand_age_s <= float(max_age_s)
            and np.isfinite(neck_age_s)
            and 0.0 <= neck_age_s <= float(max_age_s)
        ):
            return None
        left_state = np.asarray(hand.left_state, dtype=np.float32).reshape(-1)
        right_state = np.asarray(hand.right_state, dtype=np.float32).reshape(-1)
        dex_state = np.concatenate((left_state, right_state), dtype=np.float32)
        neck_state = np.asarray(
            [neck.state_yaw_deg, neck.state_pitch_deg],
            dtype=np.float32,
        )
        if dex_state.shape != (12,) or not np.all(np.isfinite(dex_state)):
            return None
        if neck_state.shape != (2,) or not np.all(np.isfinite(neck_state)):
            return None
        return dex_state.copy(), neck_state.copy()

    def _transition_to_high_level_policy(self) -> None:
        state = self.robot.get_state()
        resume_qpos = self._build_robot_state_qpos(state)
        self._reset_policy_state()
        self._last_retarget_qpos = None
        self._last_commanded_motion_qpos = resume_qpos.copy()
        self._policy_hold_qpos = resume_qpos.copy()
        self._policy_entry_pending = False
        self._policy_entry_deadline_s = None
        self._policy_paused = False
        self._policy_resume_pending = False
        self._policy_resume_deadline_s = None
        self._policy_resume_source_timestamp_ns = None
        self.mode = RobotMode.POLICY
        operator_logger.info("mode -> POLICY")

    def _toggle_high_level_policy_pause(self) -> None:
        scheduler = self._high_level_policy_scheduler
        if scheduler is None:
            return
        now_s = time.monotonic()
        if self._policy_paused:
            if self._policy_resume_pending:
                return
            policy_cfg = self._high_level_policy_cfg
            if policy_cfg is None:
                return
            self._policy_resume_pending = True
            self._policy_resume_deadline_s = now_s + policy_cfg.entry_timeout_s
            self._policy_resume_source_timestamp_ns = int(round(now_s * 1e9))
            self._publish_high_level_policy_session("resume")
            operator_logger.info("B -> resume POLICY; waiting for a fresh action chunk")
        else:
            scheduler.pause(now_s)
            self._policy_paused = True
            self._policy_resume_pending = False
            self._policy_resume_deadline_s = None
            self._policy_resume_source_timestamp_ns = None
            self._policy_hold_qpos = self._resolve_mocap_hold_qpos()
            self._publish_high_level_policy_session("pause")
            operator_logger.info("B -> pause POLICY")

    def _stop_high_level_policy_session(self) -> None:
        if self._policy_session_id is not None:
            self._publish_high_level_policy_session("stop")
        scheduler = self._high_level_policy_scheduler
        if scheduler is not None:
            scheduler.clear()
        self._policy_entry_pending = False
        self._policy_entry_deadline_s = None
        self._policy_session_id = None
        self._policy_frame_transform = None
        self._policy_paused = False
        self._policy_resume_pending = False
        self._policy_resume_deadline_s = None
        self._policy_resume_source_timestamp_ns = None
        self._policy_hold_qpos = None
        self._latest_policy_status = None

    def _handle_high_level_policy_fault(self, detail: str) -> None:
        if not bool(getattr(self, "high_level_policy_enabled", False)):
            return
        if self.mode == RobotMode.POLICY:
            if self._policy_paused and not self._policy_resume_pending:
                return
            scheduler = self._high_level_policy_scheduler
            if scheduler is not None:
                scheduler.pause(time.monotonic())
            self._policy_paused = True
            self._policy_resume_pending = False
            self._policy_resume_deadline_s = None
            self._policy_resume_source_timestamp_ns = None
            self._policy_hold_qpos = self._resolve_mocap_hold_qpos()
            self._publish_high_level_policy_session("pause")
            operator_logger.warning(
                "High-level policy fault -> pause POLICY: %s",
                detail,
            )
            return
        if self._policy_entry_pending:
            operator_logger.warning(
                "High-level policy entry failed; remaining in STANDING: %s",
                detail,
            )
            self._enter_standing()

    def _standing_step(self) -> None:
        robot_state = self.robot.get_state()
        qpos = self._standing_qpos.copy()
        motion_joint_vel = np.zeros(self.num_actions, dtype=np.float32)
        motion_qpos = np.asarray(qpos[:7 + self.num_actions], dtype=np.float32)
        reference_window = None
        if obs_builder_requires_reference_window(self.obs_builder):
            reference_window = build_static_reference_window(qpos, self._reference_window_builder, self.policy_hz)
        obs = self._ref_proc.build_observation(
            robot_state=robot_state,
            motion_qpos=motion_qpos,
            motion_joint_vel=motion_joint_vel,
            last_action=self._last_action,
            anchor_lin_vel_w=np.zeros(3, dtype=np.float32),
            anchor_ang_vel_w=np.zeros(3, dtype=np.float32),
            reference_window=reference_window,
        )
        obs = self._ref_proc.validate_observation(obs)
        action = self.policy.compute_action(obs)
        target_dof_pos = self._safety.clip_to_joint_limits(self.policy.get_target_dof_pos(action))
        self._safety.send_positions(target_dof_pos)
        self._last_action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._last_retarget_qpos = qpos.copy()
        self._last_commanded_motion_qpos = qpos.copy()
        self._publish_high_level_policy_observation(robot_state)
        self._publish_record_step(robot_state=robot_state, reference_qpos=qpos)
        self._write_retarget_viewer(qpos)

    def _mocap_step(self) -> None:
        if self._mocap_session.state == MocapSessionState.PAUSED:
            self._paused_mocap_step()
            return

        reference = self._latest_reference
        age_s = self._reference_age_s()
        if reference is None or age_s is None:
            self._hold_mocap_reference("no retarget reference")
            return
        if not reference.frame_valid:
            self._hold_mocap_reference("invalid retarget reference")
            return
        if age_s > self._stale_reference_hold_s:
            self._hold_mocap_reference(
                "delayed retarget reference",
                detail=f"age={age_s:.3f}s",
            )
            return

        robot_state = self.robot.get_state()
        self._execute_mocap_pipeline(reference.qpos, robot_state, reference.reference_window)

    def _high_level_policy_step(self) -> None:
        scheduler = self._high_level_policy_scheduler
        transform = self._policy_frame_transform
        session_id = self._policy_session_id
        if self._policy_resume_pending:
            robot_state = self.robot.get_state()
            self._publish_high_level_policy_observation(robot_state)
            deadline_s = self._policy_resume_deadline_s
            if deadline_s is not None and time.monotonic() > deadline_s:
                self._handle_high_level_policy_fault("resume timed out waiting for a fresh action chunk")
            hold_qpos = self._policy_hold_qpos
            if hold_qpos is None:
                hold_qpos = self._resolve_mocap_hold_qpos()
                self._policy_hold_qpos = hold_qpos.copy()
            self._run_static_mocap_step(hold_qpos)
            return
        if self._policy_paused:
            hold_qpos = self._policy_hold_qpos
            if hold_qpos is None:
                hold_qpos = self._resolve_mocap_hold_qpos()
                self._policy_hold_qpos = hold_qpos.copy()
            self._run_static_mocap_step(hold_qpos)
            return
        if scheduler is None or transform is None or session_id is None:
            detail = "POLICY mode is missing its scheduler/session transform"
            logger.error(detail)
            self._handle_high_level_policy_fault(detail)
            hold_qpos = self._policy_hold_qpos
            if hold_qpos is None:
                hold_qpos = self._resolve_mocap_hold_qpos()
                self._policy_hold_qpos = hold_qpos.copy()
            self._run_static_mocap_step(hold_qpos)
            return

        robot_state = self.robot.get_state()
        self._publish_high_level_policy_observation(robot_state)
        scheduled = scheduler.sample(time.monotonic())
        if scheduled is None:
            self._handle_high_level_policy_fault("action watchdog expired")
            hold_qpos = self._policy_hold_qpos
            if hold_qpos is None:
                hold_qpos = self._resolve_mocap_hold_qpos()
                self._policy_hold_qpos = hold_qpos.copy()
            self._run_static_mocap_step(hold_qpos)
            return
        reference_qpos = transform.delocalize_body_action(scheduled[:36]).astype(np.float64)
        self._execute_reference_pipeline(
            reference_qpos,
            robot_state,
            reference_window=None,
            align_reference=False,
            compose_arms=False,
        )
        self._policy_hold_qpos = reference_qpos.copy()
        publisher = self._policy_control_pub
        if publisher is not None:
            self._policy_target_seq += 1
            publisher.publish(
                HIGH_LEVEL_POLICY_TARGET_TOPIC,
                HighLevelPolicyTargetPacket(
                    session_id=session_id,
                    action=np.asarray(scheduled, dtype=np.float32).copy(),
                    timestamp_s=time.monotonic(),
                    seq=self._policy_target_seq,
                ),
            )

    def _execute_mocap_pipeline(
        self,
        reference_qpos: Float64Array,
        robot_state: object,
        reference_window: ReferenceWindow | None,
    ) -> None:
        self._execute_reference_pipeline(
            reference_qpos,
            robot_state,
            reference_window=reference_window,
            align_reference=True,
            compose_arms=self.mode == RobotMode.ARMS,
        )

    def _execute_reference_pipeline(
        self,
        reference_qpos: Float64Array,
        robot_state: object,
        *,
        reference_window: ReferenceWindow | None,
        align_reference: bool,
        compose_arms: bool,
    ) -> None:
        reference_window_aligned = False
        if align_reference:
            reference_qpos = self._ref_proc.align_reference_yaw(
                reference_qpos,
                robot_state=robot_state,
            )
        else:
            reference_qpos = np.asarray(reference_qpos, dtype=np.float64).copy()
            reference_window_aligned = True
        if compose_arms:
            reference_qpos = self._compose_arm_reference(reference_qpos)
            aligned_window = self._ref_proc.align_reference_window(reference_window, robot_state)
            reference_window = self._compose_arm_reference_window(aligned_window)
            reference_window_aligned = True
        qpos = reference_qpos.copy()
        if qpos.shape[0] < 7 + self.num_actions:
            raise ValueError(f"Retargeted qpos too short: {qpos.shape[0]} (need >= {7 + self.num_actions})")
        motion_joint_pos = np.asarray(qpos[7:7 + self.num_actions], dtype=np.float32)
        if self._last_retarget_qpos is None:
            raw_motion_joint_vel = np.zeros((self.num_actions,), dtype=np.float32)
        else:
            prev_joint_pos = np.asarray(self._last_retarget_qpos[7:7 + self.num_actions], dtype=np.float32)
            raw_motion_joint_vel = (motion_joint_pos - prev_joint_pos) * np.float32(self.policy_hz)
        motion_joint_vel = self._ref_proc.apply_joint_vel_smoothing(raw_motion_joint_vel)

        anchor_lin_vel_w = np.zeros(3, dtype=np.float32)
        anchor_ang_vel_w = np.zeros(3, dtype=np.float32)
        if not obs_builder_requires_reference_window(self.obs_builder):
            raw_lin, raw_ang = self._ref_proc.compute_anchor_velocities(reference_qpos)
            anchor_lin_vel_w, anchor_ang_vel_w = self._ref_proc.apply_anchor_vel_smoothing(raw_lin, raw_ang)

        motion_qpos = np.asarray(qpos[:7 + self.num_actions], dtype=np.float32)
        obs = self._ref_proc.build_observation(
            robot_state=robot_state,
            motion_qpos=motion_qpos,
            motion_joint_vel=motion_joint_vel,
            last_action=self._last_action,
            anchor_lin_vel_w=anchor_lin_vel_w,
            anchor_ang_vel_w=anchor_ang_vel_w,
            reference_window=reference_window,
            reference_window_aligned=reference_window_aligned,
        )
        obs = self._ref_proc.validate_observation(obs)
        action = self.policy.compute_action(obs)
        target_dof_pos = self._safety.clip_to_joint_limits(self.policy.get_target_dof_pos(action))
        self._safety.send_positions(target_dof_pos)
        self._last_action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._last_retarget_qpos = qpos.copy()
        self._ref_proc.last_reference_qpos = reference_qpos.copy()
        self._last_commanded_motion_qpos = qpos.copy()
        self._last_mocap_hold_reason = None
        self._publish_record_step(robot_state=robot_state, reference_qpos=qpos)
        self._write_retarget_viewer(qpos)

    def _compose_arm_reference(self, retarget_qpos: Float64Array) -> Float64Array:
        return compose_arm_reference(
            standing_qpos=self._standing_qpos,
            retarget_qpos=retarget_qpos,
            arm_joint_indices=self._arm_joint_indices,
            num_actions=self.num_actions,
        )

    def _compose_arm_reference_window(self, reference_window: ReferenceWindow | None) -> ReferenceWindow | None:
        return compose_arm_reference_window(
            reference_window,
            standing_qpos=self._standing_qpos,
            arm_joint_indices=self._arm_joint_indices,
            num_actions=self.num_actions,
        )

    def _enter_standing(self) -> None:
        prev_mode = self.mode
        if bool(getattr(self, "high_level_policy_enabled", False)) and (
            prev_mode == RobotMode.POLICY or self._policy_entry_pending
        ):
            self._stop_high_level_policy_session()
        self._disarm_mocap_reference_if_needed()
        self._clear_reference_gate()
        self._mocap_entry_requested = False
        if prev_mode == RobotMode.STANDING:
            return
        already_in_debug = self.mode in (
            RobotMode.STANDING,
            RobotMode.MOCAP,
            RobotMode.ARMS,
            RobotMode.POLICY,
        )
        if not already_in_debug:
            logger.info("Entering debug mode...")
            ok = self.robot.enter_debug_mode()
            if not ok:
                logger.error("Failed to enter debug mode -- staying in %s", self.mode.value)
                return
            time.sleep(0.5)

        state = self.robot.get_state()
        if prev_mode not in (
            RobotMode.STANDING,
            RobotMode.MOCAP,
            RobotMode.ARMS,
            RobotMode.POLICY,
        ):
            logger.info("Locking joints to current position...")
            self.robot.lock_all_joints()
            time.sleep(0.3)

        init_qpos = self._build_robot_state_qpos(state)
        self._last_retarget_qpos = init_qpos
        self._ref_proc.last_reference_qpos = None
        self._mocap_session.reset()
        self._last_commanded_motion_qpos = None
        self._set_default_standing_reference(state)
        self._reset_policy_state()
        if prev_mode in (
            RobotMode.MOCAP,
            RobotMode.ARMS,
            RobotMode.POLICY,
        ):
            self._safety.start_kp_ramp(
                duration_s=self._standing_return_ramp_duration,
                floor_ratio=self._standing_return_kp_ramp_floor_ratio,
            )
        else:
            self._safety.start_kp_ramp()
        self._mocap_reentry_armed = prev_mode in (RobotMode.MOCAP, RobotMode.ARMS)
        self.mode = RobotMode.STANDING
        operator_logger.info("mode -> STANDING")

    def _can_switch_to_mocap(self) -> bool:
        if self.provider_kind == "pico4" and not self._mocap_reference_armed:
            return False
        age_s = self._reference_age_s()
        if self._latest_reference is None or age_s is None:
            return False
        if not self._latest_reference.frame_valid:
            return False
        if self.provider_kind == "bvh":
            return True
        if age_s > self._max_reference_age_s:
            return False
        if self._consecutive_valid_references < self._check_frames:
            logger.warning(
                "Mocap check: only %d/%d valid references",
                self._consecutive_valid_references,
                self._check_frames,
            )
            return False
        return True

    def _transition_to_mocap(self) -> None:
        state = self.robot.get_state()
        last_commanded = getattr(self, "_last_commanded_motion_qpos", None)
        hold_qpos = last_commanded if last_commanded is not None else self._standing_qpos
        resume_qpos = self._build_resume_alignment_qpos(hold_qpos, state)
        self._mocap_reentry_armed = False
        self._reset_policy_state()
        self._last_retarget_qpos = None
        self._last_commanded_motion_qpos = resume_qpos.copy()
        self._ref_proc.reset_alignment(target_qpos=resume_qpos)
        if self.provider_kind == "bvh":
            self._send_reference_command("replay_mocap")
        self.mode = RobotMode.MOCAP
        self._mocap_entry_requested = False
        operator_logger.info("mode -> MOCAP")

    def _toggle_arms_mode(self) -> None:
        if self.provider_kind != "pico4" or self.mode not in (RobotMode.MOCAP, RobotMode.ARMS):
            return
        if self._mocap_session.state == MocapSessionState.PAUSED:
            logger.info("Ignoring Pico B mode toggle while mocap session is paused")
            return

        state = self.robot.get_state()
        resume_qpos = self._build_resume_alignment_qpos(self._last_commanded_motion_qpos, state)
        next_mode = RobotMode.ARMS if self.mode == RobotMode.MOCAP else RobotMode.MOCAP
        if next_mode == RobotMode.ARMS:
            self._set_default_standing_reference(state)
        self._reset_policy_state()
        self._last_retarget_qpos = None
        self._last_commanded_motion_qpos = resume_qpos.copy()
        self._ref_proc.reset_alignment(target_qpos=resume_qpos)
        self._safety.start_kp_ramp(
            duration_s=self._standing_return_ramp_duration,
            floor_ratio=self._standing_return_kp_ramp_floor_ratio,
        )
        self.mode = next_mode
        operator_logger.info("mode -> %s", next_mode.value.upper())

    def _resume_paused_mocap_if_needed(self) -> None:
        if self._mocap_session.state == MocapSessionState.PAUSED:
            self._resume_paused_mocap()

    def _enter_damping(self) -> None:
        if bool(getattr(self, "high_level_policy_enabled", False)) and (
            self.mode == RobotMode.POLICY or self._policy_entry_pending
        ):
            self._stop_high_level_policy_session()
        self._disarm_mocap_reference_if_needed()
        self._clear_reference_gate()
        self._mocap_entry_requested = False
        if self.mode in (RobotMode.STANDING, RobotMode.MOCAP, RobotMode.ARMS, RobotMode.POLICY):
            logger.info("DAMPING: sending LowCmd damping...")
            self.robot.set_damping()
            time.sleep(0.5)
            logger.info("DAMPING: exiting debug mode...")
        self.robot.exit_debug_mode()
        self.mode = RobotMode.DAMPING
        self._publish_damping_record_step()
        self._ref_proc.last_reference_qpos = None
        self._mocap_reentry_armed = False
        self._mocap_session.reset()
        self._last_commanded_motion_qpos = None
        self._last_mocap_hold_reason = None
        operator_logger.warning("mode -> DAMPING")

    def _reset_policy_state(self) -> None:
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)
        self._ref_proc.reset_smoothers()
        self._ref_proc.reset_alignment()
        self._mocap_session.reset()
        self._last_commanded_motion_qpos = None
        self._last_mocap_hold_reason = None
        self.policy.reset()
        self.obs_builder.reset()

    def _reset_policy_reference_state(self) -> None:
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)
        self._ref_proc.reset_smoothers()
        self._ref_proc.reset_alignment()
        self._mocap_session.reset()
        self._last_commanded_motion_qpos = None
        self._last_mocap_hold_reason = None
        self.policy.reset()
        self.obs_builder.reset()

    def _build_robot_state_qpos(self, state: object) -> Float64Array:
        qpos = np.zeros(FULL_QPOS_DIM, dtype=np.float64)
        qpos[0:3] = self._resolve_base_pos(state)
        qpos[3:7] = np.asarray(getattr(state, "quat"), dtype=np.float64).reshape(-1)[:4]
        qpos[ROOT_DIM:FULL_QPOS_DIM] = np.asarray(getattr(state, "qpos"), dtype=np.float64).reshape(-1)[
            : self.num_actions
        ]
        return qpos

    def _set_default_standing_reference(self, state: object) -> None:
        self._standing_qpos[:] = 0.0
        self._standing_qpos[0:3] = self._resolve_base_pos(state)
        self._standing_qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        align_motion_qpos_yaw(np.asarray(getattr(state, "quat"), dtype=np.float32), self._standing_qpos)
        self._standing_qpos[ROOT_DIM:FULL_QPOS_DIM] = self.default_angles.astype(np.float64)

    def _resolve_base_pos(self, state: object) -> Float64Array:
        base_pos = getattr(state, "base_pos", None)
        if base_pos is None:
            return self._default_root_pos.copy()
        resolved = self._default_root_pos.copy()
        live = np.asarray(base_pos, dtype=np.float64).reshape(-1)
        resolved[: min(3, live.shape[0])] = live[:3]
        return resolved

    def _build_resume_alignment_qpos(self, hold_qpos: Float64Array | None, state: object) -> Float64Array:
        qpos = self._build_robot_state_qpos(state)
        if hold_qpos is not None:
            qpos[0:2] = np.asarray(hold_qpos, dtype=np.float64).reshape(-1)[0:2]
        base_pos = getattr(state, "base_pos", None)
        if base_pos is not None:
            qpos[0:2] = np.asarray(base_pos, dtype=np.float64).reshape(-1)[0:2]
        return qpos

    def _handle_mocap_control_events(self, control_events: tuple[ControlEvent, ...]) -> None:
        for event in control_events:
            if event.event_type == ControlEventType.TOGGLE_ARMS:
                self._toggle_arms_mode()
                continue
            if event.event_type == ControlEventType.TOGGLE_PAUSE:
                if self.mode not in (RobotMode.MOCAP, RobotMode.ARMS):
                    continue
                if self._mocap_session.state == MocapSessionState.PAUSED:
                    self._resume_paused_mocap()
                else:
                    self._pause_active_mocap()

    def _pause_active_mocap(self) -> None:
        hold_qpos = self._resolve_mocap_hold_qpos()
        self._last_retarget_qpos = hold_qpos.copy()
        self._ref_proc.last_reference_qpos = hold_qpos.copy()
        self._last_commanded_motion_qpos = hold_qpos.copy()
        self._reset_policy_reference_state()
        self._mocap_session.pause(hold_qpos)
        logger.info("Mocap session -> PAUSED (multiprocess episode-reset)")

    def _resume_paused_mocap(self) -> None:
        hold_qpos = self._mocap_session.hold_qpos
        if hold_qpos is None:
            raise RuntimeError("Cannot resume mocap without a paused hold qpos")
        state = self.robot.get_state()
        resume_qpos = self._build_resume_alignment_qpos(hold_qpos, state)
        self._last_commanded_motion_qpos = resume_qpos.copy()
        self._reset_policy_reference_state()
        self._last_retarget_qpos = None
        self._last_commanded_motion_qpos = resume_qpos.copy()
        self._ref_proc.reset_alignment(target_qpos=resume_qpos)
        if self.mode == RobotMode.ARMS:
            self._set_default_standing_reference(state)
            self._safety.start_kp_ramp(
                duration_s=self._standing_return_ramp_duration,
                floor_ratio=self._standing_return_kp_ramp_floor_ratio,
            )
        logger.info("Mocap session -> ACTIVE (multiprocess episode-reset + reference realignment)")

    def _send_reference_command(self, command: str) -> None:
        if self._reference_command_pub is None:
            return
        self._reference_command_pub.publish(
            COMMAND_TOPIC,
            CommandPacket(command=command, timestamp_s=time.monotonic()),
        )

    def _arm_mocap_reference_if_needed(self) -> None:
        if getattr(self, "provider_kind", None) != "pico4":
            return
        now_s = time.monotonic()
        if not bool(getattr(self, "_mocap_reference_armed", False)):
            self._clear_reference_gate()
            self._mocap_reference_armed = True
            self._mocap_reference_arm_time_s = now_s
        elif getattr(self, "_latest_reference", None) is not None:
            return
        else:
            last_arm_s = getattr(self, "_mocap_reference_arm_time_s", None)
            retry_s = float(getattr(self, "_mocap_reference_arm_retry_s", 0.1))
            if last_arm_s is not None and now_s - float(last_arm_s) < retry_s:
                return
        self._send_reference_command(ARM_MOCAP_REFERENCE_COMMAND)

    def _disarm_mocap_reference_if_needed(self) -> None:
        if getattr(self, "provider_kind", None) != "pico4" or not bool(getattr(self, "_mocap_reference_armed", False)):
            return
        self._send_reference_command(DISARM_MOCAP_REFERENCE_COMMAND)
        self._mocap_reference_armed = False
        self._mocap_reference_arm_time_s = None

    def _clear_reference_gate(self) -> None:
        self._latest_reference = None
        self._last_reference_seq = -1
        self._consecutive_valid_references = 0

    def _resolve_mocap_hold_qpos(self) -> Float64Array:
        if self._last_commanded_motion_qpos is not None:
            return self._last_commanded_motion_qpos.copy()
        if self._last_retarget_qpos is not None:
            return np.asarray(self._last_retarget_qpos, dtype=np.float64).copy()
        state = self.robot.get_state()
        hold_qpos = np.zeros(FULL_QPOS_DIM, dtype=np.float64)
        hold_qpos[3:7] = np.asarray(state.quat, dtype=np.float64)
        hold_qpos[ROOT_DIM:FULL_QPOS_DIM] = np.asarray(state.qpos, dtype=np.float64)
        return hold_qpos

    def _paused_mocap_step(self) -> None:
        hold_qpos = self._mocap_session.hold_qpos
        if hold_qpos is None:
            raise RuntimeError("Paused mocap session is missing a hold_qpos")
        self._run_static_mocap_step(hold_qpos)

    def _run_static_mocap_step(self, hold_qpos: Float64Array) -> object:
        robot_state = self.robot.get_state()
        qpos = np.asarray(hold_qpos, dtype=np.float64).copy()
        motion_joint_vel = np.zeros(self.num_actions, dtype=np.float32)
        motion_qpos = np.asarray(qpos[:7 + self.num_actions], dtype=np.float32)
        reference_window = None
        if obs_builder_requires_reference_window(self.obs_builder):
            reference_window = build_static_reference_window(qpos, self._reference_window_builder, self.policy_hz)
        obs = self._ref_proc.build_observation(
            robot_state=robot_state,
            motion_qpos=motion_qpos,
            motion_joint_vel=motion_joint_vel,
            last_action=self._last_action,
            anchor_lin_vel_w=np.zeros(3, dtype=np.float32),
            anchor_ang_vel_w=np.zeros(3, dtype=np.float32),
            reference_window=reference_window,
        )
        obs = self._ref_proc.validate_observation(obs)
        action = self.policy.compute_action(obs)
        target_dof_pos = self._safety.clip_to_joint_limits(self.policy.get_target_dof_pos(action))
        self._safety.send_positions(target_dof_pos)
        self._last_action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._last_retarget_qpos = qpos.copy()
        self._ref_proc.last_reference_qpos = qpos.copy()
        self._last_commanded_motion_qpos = qpos.copy()
        self._publish_record_step(robot_state=robot_state, reference_qpos=qpos)
        self._write_retarget_viewer(qpos)
        return robot_state

    def _hold_mocap_reference(self, reason: str, *, detail: str | None = None) -> None:
        if self._last_mocap_hold_reason != reason:
            suffix = f" ({detail})" if detail else ""
            logger.warning("Mocap reference not fresh: %s%s -- holding command", reason, suffix)
            self._last_mocap_hold_reason = reason
        hold_qpos = self._resolve_mocap_hold_qpos()
        self._run_static_mocap_step(hold_qpos)

    def _publish_mode_state(self) -> None:
        self._mode_seq += 1
        mocap_like = self.mode in (RobotMode.MOCAP, RobotMode.ARMS)
        active = mocap_like and self._mocap_session.state == MocapSessionState.ACTIVE
        paused = mocap_like and self._mocap_session.state == MocapSessionState.PAUSED
        self._mode_pub.publish(
            MODE_TOPIC,
            ModeStatePacket(
                mode=self.mode.value,
                mocap_active=active,
                mocap_paused=paused,
                timestamp_s=time.monotonic(),
                seq=self._mode_seq,
                policy_paused=self.mode == RobotMode.POLICY and self._policy_paused,
                policy_session_id=(
                    self._policy_session_id if self.mode == RobotMode.POLICY else None
                ),
            ),
        )

    def _publish_record_step(self, *, robot_state: object, reference_qpos: Float64Array) -> None:
        if self._record_pub is None:
            return
        record_mode = self._recording_mode_label()
        mocap_like = self.mode in (RobotMode.MOCAP, RobotMode.ARMS)
        active = mocap_like and self._mocap_session.state == MocapSessionState.ACTIVE
        recordable = self.mode != RobotMode.DAMPING
        try:
            self._record_pub.publish(
                RECORD_TOPIC,
                RecordStepPacket(
                    timestamp_s=time.monotonic(),
                    mode=record_mode,
                    mocap_active=active,
                    recordable=recordable,
                    observation_state=build_observation_state(robot_state).astype(np.float32, copy=True),
                    observation_mode=int(build_mode_observation(record_mode)),
                    action_reference_qpos=normalize_action_reference_qpos(reference_qpos).astype(np.float32, copy=True),
                    seq=self._mode_seq,
                ),
            )
        except Exception:
            logger.exception("Failed to publish sim2real recording step")

    def _publish_damping_record_step(self) -> None:
        if self._record_pub is None:
            return
        try:
            robot_state = self.robot.get_state()
            reference_qpos = self._build_robot_state_qpos(robot_state)
            self._record_pub.publish(
                RECORD_TOPIC,
                RecordStepPacket(
                    timestamp_s=time.monotonic(),
                    mode=RobotMode.DAMPING.value,
                    mocap_active=False,
                    recordable=False,
                    observation_state=build_observation_state(robot_state).astype(np.float32, copy=True),
                    observation_mode=-1,
                    action_reference_qpos=normalize_action_reference_qpos(reference_qpos).astype(np.float32, copy=True),
                    seq=self._mode_seq,
                ),
            )
        except Exception:
            logger.exception("Failed to publish sim2real damping recording state")

    def _recording_mode_label(self) -> str:
        if (
            self.mode in (RobotMode.MOCAP, RobotMode.ARMS)
            and self._mocap_session.state == MocapSessionState.PAUSED
        ):
            return "pause"
        return self.mode.value

    def _write_retarget_viewer(self, qpos: Float64Array) -> None:
        try:
            self._retarget_viewer.write(qpos)
        except Exception:
            logger.exception("Sim2real retarget viewer update failed; control continues")

    def _reference_age_s(self) -> float | None:
        if self._latest_reference is None:
            return None
        return max(0.0, time.monotonic() - float(self._latest_reference.timestamp_s))

    def _note_reference_packet(self, reference: ReferencePacket) -> None:
        if int(reference.seq) <= self._last_reference_seq:
            return
        arm_time_s = getattr(self, "_mocap_reference_arm_time_s", None)
        if (
            self.provider_kind == "pico4"
            and bool(getattr(self, "_mocap_reference_armed", False))
            and arm_time_s is not None
            and float(reference.timestamp_s) < float(arm_time_s)
        ):
            return
        self._last_reference_seq = int(reference.seq)
        self._latest_reference = reference
        if (
            self.provider_kind == "bvh"
            and bool(getattr(reference, "playback_paused", False))
            and self.mode in (RobotMode.MOCAP, RobotMode.ARMS)
            and self._mocap_session.state == MocapSessionState.ACTIVE
        ):
            self._pause_active_mocap()
        if not reference.frame_valid:
            self._consecutive_valid_references = 0
            return
        self._consecutive_valid_references += 1

    @staticmethod
    def _sleep_until(t0: float, dt: float) -> float:
        elapsed = time.monotonic() - t0
        remaining = dt - elapsed
        if remaining > 0:
            time.sleep(remaining)
        return time.monotonic() - t0


def _run_robot_control_worker(
    cfg: dict[str, Any],
    endpoints: Sim2RealIpcEndpoints,
    stop_event: MpEvent,
) -> None:
    def _main() -> None:
        worker = _RobotControlWorker(cfg, endpoints, stop_event)
        worker.run()

    _worker_loop("robot_control", cfg, _main)


class _RecordingWorker:
    _CAMERA_TIMEOUT_S = 1.0

    def __init__(
        self,
        cfg: dict[str, Any],
        endpoints: Sim2RealIpcEndpoints,
        stop_event: MpEvent,
        *,
        recorder_factory: Callable[..., Any] | None = None,
        frame_reader: SharedFrameRingReader | None = None,
    ) -> None:
        self.cfg = cfg
        self.endpoints = endpoints
        self.stop_event = stop_event
        self.rec_cfg = _recording_cfg(cfg)
        self.camera_cfg = _recording_camera_cfg(cfg)
        self.record_modes = {
            str(mode).lower()
            for mode in cfg_get(self.rec_cfg, "record_modes", ["standing", "mocap", "arms", "pause"])
        }
        self.min_episode_seconds = float(cfg_get(self.rec_cfg, "min_episode_seconds", 1.0))
        self.discard_on_shutdown = bool(cfg_get(self.rec_cfg, "discard_on_shutdown", True))
        self.task = str(cfg_get(self.rec_cfg, "task", "demo"))
        self.fps = int(cfg_get(self.rec_cfg, "fps", 30))
        self._record_sub = LatestSubscriber(endpoints.record_pub, RECORD_TOPIC)
        self._video_sub = LatestSubscriber(endpoints.video_pub, VIDEO_TOPIC)
        self._hand_command_sub = LatestSubscriber(endpoints.hand_command_pub, HAND_COMMAND_TOPIC)
        self._neck_command_sub = LatestSubscriber(endpoints.neck_command_pub, NECK_COMMAND_TOPIC)
        self._command_sub = LatestSubscriber(endpoints.command_pub, COMMAND_TOPIC)
        self._frame_reader = frame_reader or SharedFrameRingReader()
        self._latest_record: RecordStepPacket | None = None
        left_open, right_open = _configured_open_hand_pose(cfg)
        self._latest_hand_command = HandCommandPacket(
            timestamp_s=0.0,
            driver=str(cfg_get(cfg_get(cfg, "hands", {}) or {}, "driver", "linkerhand_l6")).strip().lower(),
            mode=str(cfg_get(cfg_get(cfg, "hands", {}) or {}, "mode", "gripper")).strip().lower(),
            active=False,
            left_pose=left_open.astype(np.float32, copy=True),
            right_pose=right_open.astype(np.float32, copy=True),
            seq=0,
        )
        self._latest_neck_command = NeckCommandPacket(
            timestamp_s=0.0,
            driver=str(cfg_get(cfg_get(cfg, "neck", {}) or {}, "driver", "openneck")).strip().lower(),
            active=False,
            yaw_deg=0.0,
            pitch_deg=0.0,
            seq=0,
        )
        self._latest_video_seq = -1
        self._latest_video_received_s: float | None = None
        self._active = False
        self._episode_started_s = 0.0
        self._episode_frames = 0

        from teleopit.recording.hdf5 import TeleopitHDF5Recorder

        robot_type, hand_type, neck_type = _recording_hardware_types(cfg)
        self._schema = build_recording_schema(
            self.camera_cfg,
            fps=self.fps,
            robot_type=robot_type,
            hand_type=hand_type,
            neck_type=neck_type,
        )
        self._video_config = build_mp4_video_config(cfg_get(self.rec_cfg, "video", {}) or {})
        factory = recorder_factory or TeleopitHDF5Recorder.create
        self._recorder = factory(
            output_dir=cfg_get(self.rec_cfg, "output_dir", "data/recordings/sim2real_hdf5"),
            task=self.task,
            schema=self._schema,
            video_config=self._video_config,
        )

    def run(self) -> None:
        operator_logger.info("recording worker ready | fps=%d | modes=%s", self.fps, sorted(self.record_modes))
        idle_sleep_s = 1.0 / max(float(self.fps) * 4.0, 1.0)
        try:
            while not self.stop_event.is_set():
                command = self._command_sub.recv_latest()
                if isinstance(command, CommandPacket):
                    if self._handle_command(command):
                        break

                record = self._record_sub.recv_latest()
                if isinstance(record, RecordStepPacket):
                    self._latest_record = record

                hand_command = self._hand_command_sub.recv_latest()
                if isinstance(hand_command, HandCommandPacket):
                    self._latest_hand_command = hand_command

                neck_command = self._neck_command_sub.recv_latest()
                if isinstance(neck_command, NeckCommandPacket):
                    self._latest_neck_command = neck_command

                video = self._video_sub.recv_latest()
                if isinstance(video, SharedFrameDescriptor):
                    self._handle_video(video)
                self._discard_if_camera_stale()

                time.sleep(idle_sleep_s)
        finally:
            if self._active:
                if self.discard_on_shutdown:
                    self._discard_episode("shutdown")
                else:
                    self._save_episode()
            try:
                self._recorder.finalize()
            finally:
                self._record_sub.close()
                self._video_sub.close()
                self._hand_command_sub.close()
                self._neck_command_sub.close()
                self._command_sub.close()
                self._frame_reader.close()

    def _handle_command(self, command: CommandPacket) -> bool:
        name = command.command
        if name == "shutdown":
            self.stop_event.set()
            return True
        if name == "record_start":
            self._start_episode()
        elif name == "record_save":
            self._save_episode()
        elif name == "record_discard":
            self._discard_episode("manual discard")
        return False

    def _start_episode(self) -> None:
        if self._active:
            logger.warning("Recording episode already active; ignoring R")
            return
        record = self._latest_record
        if record is None:
            logger.warning("Cannot start recording: no robot record packet yet")
            return
        mode = str(record.mode).lower()
        if mode not in self.record_modes or not bool(record.recordable):
            logger.warning(
                "Cannot start recording: mode=%s recordable=%s",
                record.mode,
                record.recordable,
            )
            return
        if not self._camera_is_fresh():
            operator_logger.warning("cannot start recording: no fresh RealSense frame")
            return
        self._recorder.start_episode()
        self._active = True
        self._episode_started_s = time.monotonic()
        self._episode_frames = 0
        operator_logger.info("recording episode started")

    def _save_episode(self) -> None:
        if not self._active:
            operator_logger.info("no active recording episode to save")
            return
        if not self._camera_is_fresh():
            self._discard_episode("camera stream timeout")
            return
        duration_s = time.monotonic() - self._episode_started_s
        if self._episode_frames <= 0:
            self._discard_episode("empty episode")
            return
        if duration_s < self.min_episode_seconds:
            self._discard_episode(f"short episode ({duration_s:.2f}s < {self.min_episode_seconds:.2f}s)")
            return
        self._recorder.save_episode()
        operator_logger.info("recording episode saved | frames=%d duration=%.2fs", self._episode_frames, duration_s)
        self._active = False
        self._episode_frames = 0

    def _discard_episode(self, reason: str) -> None:
        if not self._active:
            operator_logger.info("no active recording episode to discard")
            return
        self._recorder.discard_episode()
        operator_logger.info("recording episode discarded | reason=%s | frames=%d", reason, self._episode_frames)
        self._active = False
        self._episode_frames = 0

    def _handle_video(self, descriptor: SharedFrameDescriptor) -> None:
        if int(descriptor.seq) == self._latest_video_seq:
            return
        self._latest_video_seq = int(descriptor.seq)
        self._latest_video_received_s = time.monotonic()
        if not self._active:
            return
        record = self._latest_record
        if record is None:
            return
        mode = str(record.mode).lower()
        if mode not in self.record_modes or not bool(record.recordable):
            logger.warning("Recording stopped because mode is no longer recordable: %s", record.mode)
            self._discard_episode("mode not recordable")
            return
        if self._schema.has_hand_action and (
            self._latest_hand_command.left_state is None
            or self._latest_hand_command.right_state is None
        ):
            return
        if self._schema.has_neck_action and (
            self._latest_neck_command.state_yaw_deg is None
            or self._latest_neck_command.state_pitch_deg is None
        ):
            return
        image = self._frame_reader.read(descriptor, copy=True)
        hand_state = (
            normalize_hand_action(
                self._latest_hand_command.left_state,
                self._latest_hand_command.right_state,
            )
            if self._schema.has_hand_action
            else None
        )
        hand_action = (
            normalize_hand_action(
                self._latest_hand_command.left_pose,
                self._latest_hand_command.right_pose,
            )
            if self._schema.has_hand_action
            else None
        )
        neck_state = (
            build_neck_action(
                self._latest_neck_command.state_yaw_deg,
                self._latest_neck_command.state_pitch_deg,
            )
            if self._schema.has_neck_action
            else None
        )
        neck_action = (
            build_neck_action(
                self._latest_neck_command.yaw_deg,
                self._latest_neck_command.pitch_deg,
            )
            if self._schema.has_neck_action
            else None
        )
        frame_kwargs = {
            "image": np.asarray(image, dtype=np.uint8),
            "state": np.asarray(record.observation_state, dtype=np.float32),
            "mode": record.observation_mode,
            "action": np.asarray(record.action_reference_qpos, dtype=np.float32),
            "hand_state": hand_state,
            "hand_action": hand_action,
        }
        if neck_state is not None:
            frame_kwargs["neck_state"] = neck_state
        if neck_action is not None:
            frame_kwargs["neck_action"] = neck_action
        self._recorder.add_frame(**frame_kwargs)
        self._episode_frames += 1

    def _camera_is_fresh(self, *, now_s: float | None = None) -> bool:
        if self._latest_video_received_s is None:
            return False
        now = time.monotonic() if now_s is None else float(now_s)
        return now - self._latest_video_received_s <= self._CAMERA_TIMEOUT_S

    def _discard_if_camera_stale(self, *, now_s: float | None = None) -> bool:
        if not self._active or self._camera_is_fresh(now_s=now_s):
            return False
        self._discard_episode("camera stream timeout")
        return True


def _run_recording_worker(
    cfg: dict[str, Any],
    endpoints: Sim2RealIpcEndpoints,
    stop_event: MpEvent,
) -> None:
    def _main() -> None:
        worker = _RecordingWorker(cfg, endpoints, stop_event)
        worker.run()

    _worker_loop("recording_worker", cfg, _main)


def _run_neck_worker(
    cfg: dict[str, Any],
    endpoints: Sim2RealIpcEndpoints,
    stop_event: MpEvent,
) -> None:
    def _main() -> None:
        neck_cfg = parse_neck_config(cfg)
        runtime = build_neck_runtime(neck_cfg)
        head_pose_sub = LatestSubscriber(endpoints.head_pose_pub, HEAD_POSE_TOPIC)
        mode_sub = LatestSubscriber(endpoints.mode_pub, MODE_TOPIC)
        command_sub = LatestSubscriber(endpoints.command_pub, COMMAND_TOPIC)
        neck_command_pub = (
            ZmqPublisher(endpoints.neck_command_pub)
            if _recording_enabled(cfg)
            else None
        )
        latest_hmd_rotation: Float64Array | None = None
        latest_spine3_rotation: Float64Array | None = None
        latest_pose_timestamp_s: float | None = None
        latest_pose_seq = -1
        latest_mode: ModeStatePacket | None = None
        command_count = 0
        command_seq = 0
        sleep_s = 1.0 / max(float(neck_cfg.rate_hz), 1.0)
        last_status_s = 0.0

        def _publish_neck_command(
            *,
            timestamp_s: float,
            active: bool,
            yaw_deg: float,
            pitch_deg: float,
        ) -> None:
            nonlocal command_seq
            if neck_command_pub is None:
                return
            try:
                state_yaw_deg, state_pitch_deg = runtime.read_deg()
            except Exception:
                logger.exception("OpenNeck state read failed")
                state_yaw_deg = state_pitch_deg = None
            command_seq += 1
            neck_command_pub.publish(
                NECK_COMMAND_TOPIC,
                NeckCommandPacket(
                    timestamp_s=float(timestamp_s),
                    driver=neck_cfg.driver,
                    active=bool(active),
                    yaw_deg=float(yaw_deg),
                    pitch_deg=float(pitch_deg),
                    seq=command_seq,
                    state_yaw_deg=state_yaw_deg,
                    state_pitch_deg=state_pitch_deg,
                ),
            )

        try:
            runtime.start()
            if neck_cfg.center_on_start or neck_command_pub is not None:
                _publish_neck_command(
                    timestamp_s=time.monotonic(),
                    active=False,
                    yaw_deg=0.0,
                    pitch_deg=0.0,
                )
            while not stop_event.is_set():
                runtime_command = command_sub.recv_latest()
                if isinstance(runtime_command, CommandPacket) and runtime_command.command == "shutdown":
                    stop_event.set()
                    break
                pose_packet = head_pose_sub.recv_latest()
                hmd_rotation, spine3_rotation, pose_timestamp_s, pose_seq = head_pose_packet(pose_packet)
                if pose_seq >= 0:
                    latest_hmd_rotation = hmd_rotation
                    latest_spine3_rotation = spine3_rotation
                    latest_pose_timestamp_s = pose_timestamp_s
                    latest_pose_seq = pose_seq
                mode_packet = mode_sub.recv_latest()
                if isinstance(mode_packet, ModeStatePacket):
                    latest_mode = mode_packet
                now_s = time.monotonic()
                active = mode_packet_active(latest_mode, neck_cfg)
                try:
                    neck_command = runtime.tick(
                        hmd_rotation_wxyz=latest_hmd_rotation,
                        spine3_rotation_wxyz=latest_spine3_rotation,
                        pose_timestamp_s=latest_pose_timestamp_s,
                        active=active,
                        now_s=now_s,
                    )
                    if neck_command is not None:
                        command_count += 1
                        _publish_neck_command(
                            timestamp_s=now_s,
                            active=active,
                            yaw_deg=neck_command.yaw_deg,
                            pitch_deg=neck_command.pitch_deg,
                        )
                except Exception:
                    logger.exception("OpenNeck worker tick failed; neck control continues")
                if now_s - last_status_s >= 5.0:
                    logger.debug(
                        "OpenNeck worker status | head_pose_seq=%s commands=%s active=%s",
                        latest_pose_seq,
                        command_count,
                        active,
                    )
                    last_status_s = now_s
                time.sleep(sleep_s)
        finally:
            try:
                runtime.close()
            finally:
                head_pose_sub.close()
                mode_sub.close()
                command_sub.close()
                if neck_command_pub is not None:
                    neck_command_pub.close()

    _worker_loop("neck_worker", cfg, _main)


class _HandSnapshotProxy:
    def __init__(self) -> None:
        self.hand_snapshot: Any | None = None
        self.controller_snapshot: Any | None = None

    def get_hand_snapshot(self) -> Any | None:
        return self.hand_snapshot

    def get_controller_snapshot(self) -> Any | None:
        return self.controller_snapshot


def _hand_worker_active_for_mode(mode_packet: ModeStatePacket) -> bool:
    del mode_packet
    return True


def _run_hand_worker(
    cfg: dict[str, Any],
    endpoints: Sim2RealIpcEndpoints,
    stop_event: MpEvent,
) -> None:
    def _main() -> None:
        proxy = _HandSnapshotProxy()
        runtime = build_hand_runtime(cfg)
        hand_sub = LatestSubscriber(endpoints.hand_pub, HAND_TOPIC)
        controller_sub = LatestSubscriber(endpoints.controller_pub, CONTROLLER_TOPIC)
        mode_sub = LatestSubscriber(endpoints.mode_pub, MODE_TOPIC)
        command_sub = LatestSubscriber(endpoints.command_pub, COMMAND_TOPIC)
        hand_command_pub = ZmqPublisher(endpoints.hand_command_pub)
        active = False
        hz = float(cfg_get(_mp_cfg(cfg), "hand_worker_hz", 120.0))
        sleep_s = 1.0 / max(hz, 1.0)
        hands_cfg = cfg_get(cfg, "hands", {}) or {}
        driver = str(cfg_get(hands_cfg, "driver", "linkerhand_l6")).strip().lower()
        hand_mode = str(cfg_get(hands_cfg, "mode", "gripper")).strip().lower()
        left_pose, right_pose = _configured_open_hand_pose(cfg)
        command_seq = 0
        recording_enabled = _recording_enabled(cfg)
        state_interval_s = (
            1.0 / float(cfg_get(_recording_cfg(cfg), "fps", 30))
            if recording_enabled else 0.0
        )

        def _apply_hand_commands(commands: tuple[HandPoseCommand, ...]) -> bool:
            nonlocal left_pose, right_pose
            changed = False
            for hand_command in commands:
                pose = np.asarray(hand_command.pose, dtype=np.float32).reshape(-1)
                if pose.shape[0] != 6:
                    logger.warning("Ignoring %s hand command with invalid pose shape %s", hand_command.side, pose.shape)
                    continue
                if hand_command.side == "left":
                    left_pose = pose.copy()
                    changed = True
                elif hand_command.side == "right":
                    right_pose = pose.copy()
                    changed = True
                else:
                    logger.warning("Ignoring hand command with unsupported side %r", hand_command.side)
            return changed

        def _publish_hand_command(
            *,
            timestamp_s: float,
            active_state: bool,
            read_state: bool = True,
        ) -> None:
            nonlocal command_seq
            left_state = right_state = None
            if recording_enabled and read_state:
                try:
                    left_state = np.asarray(runtime.get_state("left"), dtype=np.float32)
                    right_state = np.asarray(runtime.get_state("right"), dtype=np.float32)
                except Exception:
                    logger.exception("LinkerHand state read failed")
            command_seq += 1
            hand_command_pub.publish(
                HAND_COMMAND_TOPIC,
                HandCommandPacket(
                    timestamp_s=float(timestamp_s),
                    driver=driver,
                    mode=hand_mode,
                    active=bool(active_state),
                    left_pose=np.asarray(left_pose, dtype=np.float32).copy(),
                    right_pose=np.asarray(right_pose, dtype=np.float32).copy(),
                    seq=command_seq,
                    left_state=None if left_state is None else left_state.copy(),
                    right_state=None if right_state is None else right_state.copy(),
                ),
            )

        try:
            startup_commands = runtime.start()
            startup_s = time.monotonic()
            _apply_hand_commands(startup_commands)
            _publish_hand_command(timestamp_s=startup_s, active_state=False)
            last_state_s = startup_s
            while not stop_event.is_set():
                command = command_sub.recv_latest()
                if isinstance(command, CommandPacket) and command.command == "shutdown":
                    stop_event.set()
                    break
                hand_packet = hand_sub.recv_latest()
                if isinstance(hand_packet, SnapshotPacket):
                    proxy.hand_snapshot = hand_packet.snapshot
                controller_packet = controller_sub.recv_latest()
                if isinstance(controller_packet, SnapshotPacket):
                    proxy.controller_snapshot = controller_packet.snapshot
                mode_packet = mode_sub.recv_latest()
                if isinstance(mode_packet, ModeStatePacket):
                    active = _hand_worker_active_for_mode(mode_packet)
                try:
                    now_s = time.monotonic()
                    commands = runtime.tick(
                        controller_snapshot=proxy.controller_snapshot,
                        hand_snapshot=proxy.hand_snapshot,
                        active=active,
                        now_s=now_s,
                    )
                    commands_changed = bool(commands) and _apply_hand_commands(commands)
                    state_due = recording_enabled and now_s - last_state_s >= state_interval_s
                    if commands_changed or state_due:
                        _publish_hand_command(timestamp_s=now_s, active_state=active)
                        last_state_s = now_s
                except Exception:
                    logger.exception("Dexterous hand worker tick failed; hand control continues")
                time.sleep(sleep_s)
        finally:
            try:
                shutdown_commands = runtime.close()
                shutdown_s = time.monotonic()
                if _apply_hand_commands(shutdown_commands):
                    _publish_hand_command(timestamp_s=shutdown_s, active_state=False, read_state=False)
            finally:
                hand_sub.close()
                controller_sub.close()
                mode_sub.close()
                command_sub.close()
                hand_command_pub.close()

    _worker_loop("hand_worker", cfg, _main)
