"""Thread-safe state shared by the scene input and Remote Vision workers.

The XRoboToolkit headset reports its HMD pose in its native ``[x, y, z,
qx, qy, qz, qw]`` convention.  The scene camera is mounted to the MuJoCo
torso, so only the *change* from the first pose is applied to the camera's
configured neutral orientation.  This keeps the initial camera framing
stable while allowing the operator to look around by moving their head.

``B`` is intentionally represented here even though the Pico app performs
the actual stereo/single-eye layout switch locally.  The host always sends
the fixed side-by-side ZEDMINI frame required by the headset decoder; keeping
the edge-triggered state on the host makes the input semantics observable and
leaves a clean hook for future headset protocols.
"""

from __future__ import annotations

import threading
import numbers
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .controller import _HEADSET_TO_WORLD


_VIEW_MODES = ("stereo", "single")


def _validated_pose(value: Any) -> tuple[np.ndarray, bool]:
    """Return a finite pose, normalized quaternion, and orientation validity.

    XRoboToolkit occasionally reports an all-zero quaternion while tracking
    is warming up.  The returned boolean marks that fallback so callers can
    avoid using the artificial identity as a calibration sample.  Other
    malformed/non-finite values are rejected so a broken SDK sample cannot
    silently steer the camera.
    """

    try:
        raw = np.asarray(value).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("head_pose must be seven finite values [x,y,z,qx,qy,qz,qw]") from exc
    if any(
        isinstance(item, (bool, np.bool_)) or not isinstance(item, numbers.Real)
        for item in raw
    ):
        raise ValueError("head_pose must be seven finite values [x,y,z,qx,qy,qz,qw]")
    try:
        pose = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("head_pose must be seven finite values [x,y,z,qx,qy,qz,qw]") from exc
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError("head_pose must be seven finite values [x,y,z,qx,qy,qz,qw]")
    quat = pose[3:7].copy()
    norm = float(np.linalg.norm(quat))
    has_orientation = norm > 1e-8
    if norm <= 1e-8:
        quat[:] = (0.0, 0.0, 0.0, 1.0)
    else:
        quat /= norm
    pose[3:7] = quat
    return pose, has_orientation


class SceneViewState:
    """Synchronize HMD camera orientation and Remote Vision view mode.

    All returned arrays are copies.  The scene control loop can therefore
    update the state at XR input rate while the independent MuJoCo renderer
    consumes a coherent snapshot without sharing mutable numpy storage.
    """

    def __init__(self, *, view_mode: str = "stereo") -> None:
        mode = str(view_mode).strip().lower()
        if mode not in _VIEW_MODES:
            raise ValueError(f"view_mode must be one of {_VIEW_MODES}, got {view_mode!r}")
        self._lock = threading.Lock()
        self._head_pose: np.ndarray | None = None
        self._head_reference: np.ndarray | None = None
        self._b_pressed = False
        self._view_mode = mode

    @property
    def view_mode(self) -> str:
        """Current edge-triggered B-button view mode (``stereo``/``single``)."""

        with self._lock:
            return self._view_mode

    def set_view_mode(self, mode: str) -> str:
        """Set the advisory view mode and return its normalized value."""

        normalized = str(mode).strip().lower()
        if normalized not in _VIEW_MODES:
            raise ValueError(f"view_mode must be one of {_VIEW_MODES}, got {mode!r}")
        with self._lock:
            self._view_mode = normalized
            return self._view_mode

    def update_b_button(self, pressed: bool) -> str | None:
        """Apply a right-controller B sample and return a mode on a press edge.

        Returning ``None`` for a held button is important: the XR bridge sends
        samples at 60 Hz, while SIMPLE toggles once per physical press.
        """

        # Do not coerce arbitrary truthy values at this boundary.  A malformed
        # SDK value such as ``"false"`` or ``1`` would otherwise become a
        # press edge and toggle the Remote Vision layout unexpectedly.  The
        # XR packet validator emits a builtin ``bool``; direct callers should
        # use the same unambiguous representation.
        if not isinstance(pressed, (bool, np.bool_)):
            raise ValueError(f"pressed must be a boolean, got {type(pressed).__name__}")
        is_pressed = bool(pressed)
        with self._lock:
            toggled = is_pressed and not self._b_pressed
            self._b_pressed = is_pressed
            if not toggled:
                return None
            self._view_mode = "single" if self._view_mode == "stereo" else "stereo"
            return self._view_mode

    def set_head_pose(self, pose: Any, *, calibrate_if_needed: bool = True) -> None:
        """Store one native HMD pose, optionally calibrating its neutral pose."""

        validated, has_orientation = _validated_pose(pose)
        with self._lock:
            # A zero quaternion is emitted by XRoboToolkit while HMD tracking
            # is warming up (and occasionally during a reconnect).  Keep the
            # last known-good sample in that case rather than replacing it
            # with an artificial identity pose.  In particular, never use the
            # fallback identity as the neutral reference: doing so would make
            # the first real HMD orientation appear as a large camera jump.
            if has_orientation or self._head_pose is None:
                self._head_pose = validated
            if calibrate_if_needed and has_orientation and self._head_reference is None:
                self._head_reference = validated.copy()

    def reset_head_reference(self) -> None:
        """Forget the neutral HMD orientation; the next pose becomes neutral."""

        with self._lock:
            self._head_reference = None

    @property
    def head_pose(self) -> np.ndarray | None:
        """Latest native HMD pose, or ``None`` before the first sample."""

        with self._lock:
            return None if self._head_pose is None else self._head_pose.copy()

    def head_rotation_delta(self) -> np.ndarray | None:
        """Return the calibrated HMD turn in MuJoCo's z-up world basis.

        XRoboToolkit reports poses in its native y-up basis.  SIMPLE converts
        controller and headset rotations with ``R_HEADSET_TO_WORLD @ R @
        R_HEADSET_TO_WORLD.T`` before using them in the z-up WBC.  Apply the
        same basis change here so Remote Vision does not mix native PICO axes
        with MuJoCo camera axes.  The returned matrix is an *active* world
        rotation (``R_current @ R_reference.T``), so a renderer should apply
        it on the left of its neutral camera orientation:
        ``R_camera = head_rotation_delta() @ R_neutral``.
        """

        with self._lock:
            if self._head_pose is None or self._head_reference is None:
                return None
            reference = self._head_reference[3:7].copy()
            current = self._head_pose[3:7].copy()
        # scipy consumes quaternions in x,y,z,w order.  ``R_current @
        # R_reference.T`` is the active rotation taking the calibrated HMD
        # frame to the current frame.  Convert that relative rotation from
        # XRoboToolkit's native basis into MuJoCo's z-up basis, exactly as the
        # SIMPLE controller does for absolute poses.
        reference_rotation = Rotation.from_quat(reference).as_matrix()
        current_rotation = Rotation.from_quat(current).as_matrix()
        relative_native = current_rotation @ reference_rotation.T
        return _HEADSET_TO_WORLD @ relative_native @ _HEADSET_TO_WORLD.T


__all__ = ["SceneViewState"]
