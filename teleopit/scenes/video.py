"""XRoboToolkit Remote Vision support for the 43-DOF scene runtime."""

from __future__ import annotations

import queue
import copy
import operator
import sys
import threading
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .xr_video_transport import DirectXRoboToolkitVideoTransport
from .view_state import SceneViewState


# XRoboToolkit's ZEDMINI receiver expects the released SIMPLE profile: one
# 1280x720 rendered eye duplicated into a 2560x720 side-by-side frame at
# 60 Hz.  Keep these defaults in the scene sender (rather than inheriting the
# generic 30 Hz RealSense/video defaults used by the learned sim2sim path).
DEFAULT_SCENE_VIDEO_WIDTH = 1280
DEFAULT_SCENE_VIDEO_HEIGHT = 720
DEFAULT_SCENE_VIDEO_FPS = 60


def _validated_video_int(value: Any, name: str, *, even: bool = False) -> int:
    """Validate an integer Remote Vision parameter without coercion.

    H.264's default ``yuv420p`` pixel format requires even frame dimensions;
    accepting a float via ``int(value)`` or an odd height would otherwise let
    the renderer start and fail asynchronously in the encoder thread.
    """

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"Remote Vision {name} must be a positive integer")
    try:
        integer = int(operator.index(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Remote Vision {name} must be a positive integer") from exc
    if integer <= 0:
        raise ValueError(f"Remote Vision {name} must be positive")
    if even and integer % 2:
        raise ValueError(f"Remote Vision {name} must be even for yuv420p")
    return integer


class SceneRemoteVision:
    """Render ``scene_head_camera`` and publish it through XRoboToolkit TCP.

    The tested XRoboToolkit transport already lives in Teleopit's regular Pico
    input path.  The 43-DOF runtime only provides the renderer, keeping the
    headset protocol identical across both simulation modes.
    """

    def __init__(
        self,
        *,
        model: Any,
        data: Any,
        host: str,
        port: int = 12345,
        width: int = DEFAULT_SCENE_VIDEO_WIDTH,
        height: int = DEFAULT_SCENE_VIDEO_HEIGHT,
        fps: int = DEFAULT_SCENE_VIDEO_FPS,
        view_state: SceneViewState | None = None,
        control_port: int = 13579,
    ) -> None:
        host = str(host).strip()
        if not host:
            raise ValueError("Remote Vision host must not be empty")
        width_i = _validated_video_int(width, "width", even=True)
        height_i = _validated_video_int(height, "height", even=True)
        fps_i = _validated_video_int(fps, "FPS")
        port_i = _validated_video_int(port, "port")
        if port_i > 65535:
            raise ValueError("Remote Vision port must be in [1, 65535]")
        control_port_i = _validated_video_int(control_port, "control port")
        if control_port_i > 65535:
            raise ValueError("Remote Vision control port must be in [1, 65535]")
        import mujoco
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "scene_head_camera")
        if camera_id < 0:
            raise ValueError("Scene model does not contain the required scene_head_camera")
        self._mujoco = mujoco
        self._model = model
        self._data = data
        self._view_state = view_state
        self._width = width_i
        self._height = height_i
        # Renderer/OpenGL state is thread-affine on several MuJoCo builds.
        # Keep construction and rendering in one worker and pass only a qpos
        # snapshot from the 200 Hz control loop.  A one-slot queue is
        # intentional: stale camera poses are less useful than the newest one
        # for a teleoperation view, and the control loop must never wait for
        # H.264 or GPU work.
        self._snapshots: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self._render_stop = threading.Event()
        # Serialize start/stop so a shutdown racing renderer startup cannot
        # clear the thread handle and then have the in-flight start install a
        # transport after the caller believes the runtime is stopped.
        self._lifecycle_lock = threading.Lock()
        self._render_thread: threading.Thread | None = None
        self._render_error: BaseException | None = None
        self._renderer: Any | None = None
        self._camera = "scene_head_camera"
        self._camera_id = int(camera_id)
        # ``model.cam_quat`` is immutable for the control loop, but changing
        # it in the render worker would race with the viewer/control thread.
        # Keep the compiled neutral camera orientation and apply HMD deltas to
        # the worker's private model copy only (created lazily below).
        self._base_camera_quat = np.asarray(model.cam_quat[self._camera_id], dtype=np.float64).copy()
        self._fps = int(fps)
        self._next_frame_s = 0.0
        self._next_status_s = 0.0
        self._transport = DirectXRoboToolkitVideoTransport(
            host=host,
            port=port_i,
            width=width_i * 2,
            height=height_i,
            # Keep the direct ZEDMINI geometry but encode only source camera
            # frames. Repeating a 30 Hz frame at a nominal 60 Hz floods the
            # headset's TCP decoder queue; SIMPLE's reference sender likewise
            # sends one access unit per newly rendered frame.
            fps=fps_i,
            control_port=control_port_i,
        )

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._render_thread is not None:
                if self._render_thread.is_alive():
                    return
                # A renderer can fail after its thread has started (for
                # example when EGL/GLFW is unavailable).  The old handle is
                # then non-None but dead, and blindly returning here leaves a
                # stopped Remote Vision transport with no way to recover.
                # Tear down that failed generation before creating a fresh
                # worker; ``DirectXRoboToolkitVideoTransport.stop`` is
                # idempotent and also drops any queued stale frame.
                self._render_stop.set()
                self._transport.stop()
                self._drain_snapshots()
                self._render_thread = None
            self._render_stop.clear()
            self._render_error = None
            self._next_frame_s = 0.0
            render_thread = threading.Thread(
                target=self._render_loop,
                name="scene_mujoco_render",
                daemon=True,
            )
            # Publish the thread handle before starting the transport so a
            # concurrent ``stop`` can always see the in-flight startup.  If
            # either worker startup fails, roll back both sides: otherwise a
            # failed ``thread.start()`` would leave a reconnecting TCP
            # transport alive and a non-None handle that prevents a later
            # retry.
            self._render_thread = render_thread
            try:
                self._transport.start()
                render_thread.start()
            except BaseException:
                self._render_stop.set()
                try:
                    self._transport.stop()
                finally:
                    self._render_thread = None
                raise
        host, port = self._transport.endpoint
        print(
            "Scene Remote Vision waiting for Pico listener at "
            f"{host}:{port} (choose ZEDMINI, then Listen)."
        )

    @property
    def is_connected(self) -> bool:
        return self._transport.is_connected

    @property
    def view_mode(self) -> str:
        """Advisory Pico Remote Vision layout state (stereo/single)."""
        return self._view_state.view_mode if self._view_state is not None else "stereo"

    def set_head_pose(self, pose: Any, *, calibrate_if_needed: bool = True) -> None:
        """Update the HMD pose consumed by the background camera renderer."""
        if self._view_state is not None:
            self._view_state.set_head_pose(pose, calibrate_if_needed=calibrate_if_needed)

    def handle_b_button(self, pressed: bool) -> str | None:
        """Forward a B-button sample and return the mode on a press edge."""
        if self._view_state is None:
            return None
        return self._view_state.update_b_button(pressed)

    @property
    def frames_sent(self) -> int:
        return self._transport.frames_sent

    def tick(self) -> None:
        # ``tick`` may race the scene's finally/stop path.  Do not enqueue a
        # snapshot after shutdown: a subsequent start must not render a frame
        # from the previous lifecycle generation.
        if self._render_stop.is_set():
            return
        now = time.monotonic()
        if now < self._next_frame_s:
            return
        self._next_frame_s = now + 1.0 / self._fps
        # ``tick`` runs in the MuJoCo control loop.  Only copying qpos and
        # replacing the single pending snapshot is done synchronously here;
        # rendering and encoding happen in background workers.
        snapshot = np.asarray(self._data.qpos, dtype=np.float64).copy()
        try:
            self._snapshots.put_nowait(snapshot)
        except queue.Full:
            try:
                self._snapshots.get_nowait()
            except queue.Empty:
                return
            try:
                self._snapshots.put_nowait(snapshot)
            except queue.Full:
                pass
        if now >= self._next_status_s:
            if self._transport.is_connected:
                print(f"Scene Remote Vision live ({self._transport.frames_sent} frames sent).")
            else:
                error = self._transport.last_connect_error or "listener not yet available"
                print(f"Scene Remote Vision waiting for Pico Listen: {error}")
            self._next_status_s = now + 5.0

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._render_stop.set()
            try:
                self._snapshots.put_nowait(None)
            except queue.Full:
                try:
                    self._snapshots.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._snapshots.put_nowait(None)
                except queue.Full:
                    pass
            if self._render_thread is not None:
                render_thread = self._render_thread
                render_thread.join(timeout=2.0)
                # A foreign OpenGL/codec call can ignore the stop event for
                # longer than the bounded join.  Keep the live handle in that
                # case: clearing it would let a subsequent ``start()`` clear
                # the shared stop event and launch a second renderer while
                # the first one still owns thread-affine MuJoCo state.
                if not render_thread.is_alive():
                    self._render_thread = None
            self._transport.stop()
            self._drain_snapshots()

    def _drain_snapshots(self) -> None:
        """Discard pending camera snapshots without blocking."""
        while True:
            try:
                self._snapshots.get_nowait()
            except queue.Empty:
                return

    def _render_loop(self) -> None:
        """Render snapshots on a thread with its own MuJoCo ``MjData``.

        MuJoCo's model is immutable after compilation and safe to read from
        both threads.  ``MjData`` and the OpenGL renderer are deliberately
        private to this worker, so no MuJoCo data object crosses the control
        and rendering threads.
        """
        renderer: Any | None = None
        try:
            # MuJoCo's model arrays are independent under ``copy.copy`` while
            # the compiled assets remain shared/read-only.  This private model
            # lets HMD camera orientation updates stay out of the control and
            # onscreen-viewer model, which may be used concurrently.
            render_model = copy.copy(self._model)
            render_data = self._mujoco.MjData(render_model)
            renderer = self._mujoco.Renderer(
                render_model,
                height=self._height,
                width=self._width,
            )
            self._renderer = renderer
            while not self._render_stop.is_set():
                try:
                    snapshot = self._snapshots.get(timeout=0.05)
                except queue.Empty:
                    continue
                if snapshot is None:
                    continue
                if snapshot.shape != render_data.qpos.shape:
                    raise ValueError(
                        "Remote Vision qpos snapshot shape changed: "
                        f"expected {render_data.qpos.shape}, got {snapshot.shape}"
                    )
                # Restore the compiled neutral orientation before forwarding
                # this snapshot.  ``render_model`` is reused across frames;
                # without this reset, applying a head delta against the
                # previous frame would make the view drift cumulatively.
                render_model.cam_quat[self._camera_id] = self._base_camera_quat
                render_data.qpos[:] = snapshot
                self._mujoco.mj_forward(render_model, render_data)
                # HMD deltas are applied only to a private camera model/data
                # pair.  Use ``cam_quat`` (wxyz) and run forward once more so
                # the renderer receives a coherent camera transform.
                view_delta = self._view_state.head_rotation_delta() if self._view_state is not None else None
                if view_delta is not None:
                    # SceneViewState returns an active HMD turn in MuJoCo's
                    # z-up world basis.  Apply it to the neutral *world*
                    # camera orientation, then convert back to the camera's
                    # body-local quaternion.  This matters once the robot's
                    # torso turns: writing a world-space delta directly into
                    # ``cam_quat`` would otherwise rotate around torso axes.
                    base_camera_rot = Rotation.from_quat(
                        np.array(
                            [
                                self._base_camera_quat[1],
                                self._base_camera_quat[2],
                                self._base_camera_quat[3],
                                self._base_camera_quat[0],
                            ],
                            dtype=np.float64,
                        )
                    ).as_matrix()
                    camera_body_id = int(render_model.cam_bodyid[self._camera_id])
                    body_rot = render_data.xmat[camera_body_id].reshape(3, 3)
                    # ``cam_xmat`` is the camera-to-world matrix after the
                    # neutral forward pass.  Recompute it from the body and
                    # local neutral orientation rather than relying on a
                    # possibly model-version-specific layout.
                    neutral_world_rot = body_rot @ base_camera_rot
                    desired_world_rot = view_delta @ neutral_world_rot
                    camera_local_rot = body_rot.T @ desired_world_rot
                    camera_xyzw = Rotation.from_matrix(camera_local_rot).as_quat()
                    render_model.cam_quat[self._camera_id] = np.array(
                        [camera_xyzw[3], camera_xyzw[0], camera_xyzw[1], camera_xyzw[2]],
                        dtype=np.float64,
                    )
                    self._mujoco.mj_forward(render_model, render_data)
                renderer.update_scene(render_data, camera=self._camera)
                frame = np.ascontiguousarray(renderer.render(), dtype=np.uint8)
                self._transport.publish_frame(frame)
        except BaseException as exc:
            self._render_error = exc
            was_stopping = self._render_stop.is_set()
            # A renderer failure must not leave the TCP reconnect worker alive
            # indefinitely after the producer thread has exited.  Mark this
            # lifecycle generation stopped; ``start`` can later detect the
            # dead handle and create a clean generation.
            self._render_stop.set()
            try:
                self._transport.stop()
            except Exception:
                pass
            if not was_stopping:
                print(f"Scene Remote Vision renderer stopped: {exc}", file=sys.stderr)
        finally:
            self._renderer = None
            if renderer is not None:
                try:
                    renderer.close()
                except Exception:
                    pass
