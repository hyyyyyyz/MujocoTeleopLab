"""Isolated client worker for asynchronous receding-horizon policy inference."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import numpy as np

from teleopit.high_level_policy.client import (
    HighLevelPolicyClient,
    PolicyTransportError,
)
from teleopit.high_level_policy.config import parse_high_level_policy_config
from teleopit.high_level_policy.protocol import PolicyProtocolError
from teleopit.sim2real.mp.ipc import (
    COMMAND_TOPIC,
    HIGH_LEVEL_POLICY_ACTION_TOPIC,
    HIGH_LEVEL_POLICY_OBSERVATION_TOPIC,
    HIGH_LEVEL_POLICY_SESSION_TOPIC,
    HIGH_LEVEL_POLICY_STATUS_TOPIC,
    LatestSubscriber,
    Sim2RealIpcEndpoints,
    ZmqPublisher,
)
from teleopit.sim2real.mp.messages import (
    CommandPacket,
    HighLevelPolicyActionPacket,
    HighLevelPolicyObservationPacket,
    HighLevelPolicySessionPacket,
    HighLevelPolicyStatusPacket,
)
from teleopit.sim2real.mp.shm import SharedFrameRingReader


logger = logging.getLogger(__name__)


def encode_policy_jpeg(frame: object, *, quality: int) -> bytes:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "High-level policy image encoding requires OpenCV; install teleopit[sim2real]"
        ) from exc
    rgb = np.asarray(frame)
    if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
        raise ValueError(
            f"High-level policy camera frame must be uint8[480,640,3], got {rgb.dtype}{rgb.shape}"
        )
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("OpenCV failed to encode the high-level policy JPEG")
    payload = encoded.tobytes()
    if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
        raise RuntimeError("OpenCV returned an invalid JPEG payload")
    return payload


class HighLevelPolicyWorker:
    def __init__(
        self,
        cfg: dict[str, Any],
        endpoints: Sim2RealIpcEndpoints,
        stop_event: Any,
        *,
        client_factory: Callable[..., HighLevelPolicyClient] = HighLevelPolicyClient,
        frame_reader: SharedFrameRingReader | None = None,
    ) -> None:
        self.cfg = cfg
        self.endpoints = endpoints
        self.stop_event = stop_event
        self.policy_cfg = parse_high_level_policy_config(cfg)
        self._client_factory = client_factory
        self._frame_reader = frame_reader or SharedFrameRingReader()
        self._session_sub = LatestSubscriber(
            endpoints.high_level_policy_control_pub,
            HIGH_LEVEL_POLICY_SESSION_TOPIC,
        )
        self._observation_sub = LatestSubscriber(
            endpoints.high_level_policy_control_pub,
            HIGH_LEVEL_POLICY_OBSERVATION_TOPIC,
        )
        self._command_sub = LatestSubscriber(endpoints.command_pub, COMMAND_TOPIC)
        self._result_pub = ZmqPublisher(endpoints.high_level_policy_result_pub)
        self._client: HighLevelPolicyClient | None = None
        self._active_session: HighLevelPolicySessionPacket | None = None
        self._ready = False
        self._paused = False
        self._last_session_seq = -1
        self._last_observation_seq = -1
        self._last_request_timestamp_ns: int | None = None
        self._next_connect_time_s = 0.0
        self._status_seq = 0
        self._policy_type: str | None = None
        self._policy_id: str | None = None
        self._new_session_required = False

    def run(self) -> None:
        try:
            while not self.stop_event.is_set():
                command = self._command_sub.recv_latest()
                if isinstance(command, CommandPacket) and command.command == "shutdown":
                    break
                session = self._session_sub.recv_latest()
                if isinstance(session, HighLevelPolicySessionPacket):
                    self._handle_session(session)
                if self._active_session is not None and not self._ready:
                    self._connect_if_due()
                observation = self._observation_sub.recv_latest()
                if isinstance(observation, HighLevelPolicyObservationPacket):
                    self._handle_observation(observation)
                time.sleep(0.001)
        finally:
            self.close()

    def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            client.close()
        self._frame_reader.close()
        self._session_sub.close()
        self._observation_sub.close()
        self._command_sub.close()
        self._result_pub.close()

    def _handle_session(self, packet: HighLevelPolicySessionPacket) -> None:
        if int(packet.seq) <= self._last_session_seq:
            return
        self._last_session_seq = int(packet.seq)
        command = str(packet.command).strip().lower()
        if command == "start":
            if self._active_session is None or packet.session_id != self._active_session.session_id:
                self._active_session = packet
                self._ready = False
                self._paused = False
                self._last_observation_seq = -1
                self._last_request_timestamp_ns = None
                self._next_connect_time_s = 0.0
                self._new_session_required = False
                self._policy_type = None
                self._policy_id = None
                self._publish_status("connecting", "connecting to host policy")
            return
        if self._active_session is None or packet.session_id != self._active_session.session_id:
            return
        if command == "pause":
            if not self._paused:
                self._paused = True
                self._publish_status("paused", "policy requests paused")
        elif command == "resume":
            if self._paused:
                self._paused = False
                self._last_request_timestamp_ns = None
                if self._ready:
                    self._publish_status("ready", "policy requests resumed")
                else:
                    self._new_session_required = False
                    self._next_connect_time_s = 0.0
                    self._policy_type = None
                    self._policy_id = None
                    self._publish_status(
                        "connecting",
                        "reconnecting the paused policy session",
                    )
        elif command == "stop":
            self._publish_status("stopped", "policy session stopped")
            self._active_session = None
            self._ready = False
            self._paused = False

    def _connect_if_due(self) -> None:
        session = self._active_session
        now_s = time.monotonic()
        if session is None or self._new_session_required or now_s < self._next_connect_time_s:
            return
        try:
            if self._client is None:
                self._client = self._client_factory(
                    self.policy_cfg.endpoint,
                    timeout_s=self.policy_cfg.timeout_s,
                )
            description = self._client.describe()
            if self.policy_cfg.replan_steps > description.max_action_horizon:
                raise ValueError(
                    "high_level_policy.replan_steps exceeds host max_action_horizon: "
                    f"{self.policy_cfg.replan_steps} > {description.max_action_horizon}"
                )
            self._client.reset(session.session_id, session.task)
            self._policy_type = description.policy_type
            self._policy_id = description.policy_id
            self._ready = True
            self._publish_status("ready", "host policy session reset")
        except (PolicyProtocolError, PolicyTransportError, ValueError, RuntimeError) as exc:
            self._ready = False
            self._next_connect_time_s = now_s + self.policy_cfg.reconnect_backoff_s
            self._publish_status("unavailable", str(exc))

    def _handle_observation(self, packet: HighLevelPolicyObservationPacket) -> None:
        session = self._active_session
        if session is None or not self._ready or self._paused:
            return
        if packet.session_id != session.session_id or packet.sequence_id <= self._last_observation_seq:
            return
        now_s = time.monotonic()
        if now_s - float(packet.timestamp_s) > self.policy_cfg.max_observation_age_s:
            return
        minimum_interval_ns = int(round(self.policy_cfg.replan_steps / 30.0 * 1e9))
        if (
            self._last_request_timestamp_ns is not None
            and packet.onboard_monotonic_timestamp_ns - self._last_request_timestamp_ns
            < minimum_interval_ns
        ):
            return
        client = self._client
        if client is None:
            return
        try:
            frame = self._frame_reader.read(packet.frame, copy=True)
            jpeg = encode_policy_jpeg(frame, quality=self.policy_cfg.jpeg_quality)
            chunk = client.get_action(
                session_id=session.session_id,
                sequence_id=int(packet.sequence_id),
                onboard_monotonic_timestamp_ns=int(packet.onboard_monotonic_timestamp_ns),
                task=session.task,
                jpeg_image=jpeg,
                body_joint_positions=packet.body_joint_positions,
                dex_state=packet.dex_state,
                neck_state=packet.neck_state,
                source_reference_root_pose=packet.source_reference_root_pose,
            )
            if self._policy_id is None or chunk.policy_id != self._policy_id:
                raise PolicyProtocolError(
                    "policy_mismatch",
                    "get_action policy_id does not match the preceding describe response",
                )
            self._result_pub.publish(
                HIGH_LEVEL_POLICY_ACTION_TOPIC,
                HighLevelPolicyActionPacket(
                    session_id=chunk.session_id,
                    source_sequence_id=chunk.source_sequence_id,
                    source_onboard_monotonic_timestamp_ns=chunk.source_onboard_monotonic_timestamp_ns,
                    action_fps=chunk.action_fps,
                    actions=np.asarray(chunk.actions, dtype=np.float32).copy(),
                    policy_id=chunk.policy_id,
                    server_inference_ms=chunk.server_inference_ms,
                    received_timestamp_s=time.monotonic(),
                ),
            )
            self._last_observation_seq = int(packet.sequence_id)
            self._last_request_timestamp_ns = int(packet.onboard_monotonic_timestamp_ns)
        except (PolicyProtocolError, PolicyTransportError, ValueError, RuntimeError) as exc:
            logger.warning("High-level policy request failed: %s", exc)
            self._ready = False
            self._paused = True
            self._new_session_required = True
            self._publish_status(
                "fault",
                f"{exc}; POLICY paused and can be resumed with B after recovery",
            )

    def _publish_status(self, status: str, detail: str) -> None:
        self._status_seq += 1
        session_id = None if self._active_session is None else self._active_session.session_id
        self._result_pub.publish(
            HIGH_LEVEL_POLICY_STATUS_TOPIC,
            HighLevelPolicyStatusPacket(
                session_id=session_id,
                status=str(status),
                detail=str(detail),
                timestamp_s=time.monotonic(),
                seq=self._status_seq,
                policy_type=self._policy_type,
                policy_id=self._policy_id,
            ),
        )


def run_high_level_policy_worker(
    cfg: dict[str, Any],
    endpoints: Sim2RealIpcEndpoints,
    stop_event: Any,
) -> None:
    worker = HighLevelPolicyWorker(cfg, endpoints, stop_event)
    worker.run()
