"""Robot-specific dimension constants for the Unitree G1 pipeline."""

ROOT_POS_DIM = 3
ROOT_QUAT_DIM = 4
ROOT_DIM = ROOT_POS_DIM + ROOT_QUAT_DIM  # 7: pos(3) + quat_wxyz(4)
NUM_JOINTS = 29  # G1 actuated joints
FULL_QPOS_DIM = ROOT_DIM + NUM_JOINTS  # 36: root + joints

# Canonical actuator order used by the downloaded g1_29dof.xml, policy output,
# Unitree command path, and sim2real recording schema.
G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
