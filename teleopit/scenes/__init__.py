"""SIMPLE-style XRoboToolkit scene teleoperation runtime.

The scene runtime is intentionally separate from Teleopit's 29-DOF motion
tracking pipeline.  It uses the 43-DOF G1/Dex3 model and decoupled WBC assets
needed for physical table-top manipulation.
"""

from .xr_packet import SceneXRPacket, SceneXRReceiver
from .view_state import SceneViewState

__all__ = ["SceneXRPacket", "SceneXRReceiver", "SceneViewState"]
