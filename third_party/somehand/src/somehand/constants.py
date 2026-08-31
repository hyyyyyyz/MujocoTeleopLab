"""MediaPipe hand landmark indices and default retargeting vector pairs."""

# MediaPipe hand landmark indices (21 landmarks total)
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
LITTLE_MCP = 17
LITTLE_PIP = 18
LITTLE_DIP = 19
LITTLE_TIP = 20

NUM_LANDMARKS = 21

# Finger names
FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]

# Default vector pairs for retargeting: (origin_landmark_idx, target_landmark_idx)
# Each pair defines a directional vector from origin to target.
DEFAULT_VECTOR_PAIRS = [
    # Wrist to finger MCP (palm structure, 4 vectors)
    (WRIST, INDEX_MCP),
    (WRIST, MIDDLE_MCP),
    (WRIST, RING_MCP),
    (WRIST, LITTLE_MCP),
    # Thumb chain (3 vectors)
    (THUMB_CMC, THUMB_MCP),
    (THUMB_MCP, THUMB_IP),
    (THUMB_IP, THUMB_TIP),
    # Index chain (3 vectors)
    (INDEX_MCP, INDEX_PIP),
    (INDEX_PIP, INDEX_DIP),
    (INDEX_DIP, INDEX_TIP),
    # Middle chain (3 vectors)
    (MIDDLE_MCP, MIDDLE_PIP),
    (MIDDLE_PIP, MIDDLE_DIP),
    (MIDDLE_DIP, MIDDLE_TIP),
    # Ring chain (3 vectors)
    (RING_MCP, RING_PIP),
    (RING_PIP, RING_DIP),
    (RING_DIP, RING_TIP),
    # Little chain (3 vectors)
    (LITTLE_MCP, LITTLE_PIP),
    (LITTLE_PIP, LITTLE_DIP),
    (LITTLE_DIP, LITTLE_TIP),
]
