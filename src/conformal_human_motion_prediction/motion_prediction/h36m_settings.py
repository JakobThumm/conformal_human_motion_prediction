N_JOINTS = 13
INPUT_HORIZON_LENGTH = 50
PREDICTION_HORIZON_LENGTH = 10
REDUCED_TIMESTEP = 4  # Predict only timestep 4
REDUCED_JOINT_INDICES = [0, 5, 6]  # Predict only joints: Head, Left Hand, Right Hand
# Only used in (get_h36m_motion_dataset_with_uncertainty)
FAKE_INPUT_UNCERTAINTY = 0.01
OOD_THRESHOLD = 5e5
# Number of recent non-ood 3D poses required to accept the current motion prediction.
# This prevents an infinite feedback loop of predicted poses.
N_CORRECT_POSES_REQUIRED = 3

# Covariance calibration for motion prediction
COV_CALIBRATION_CT = 4.0
COV_CALIBRATION_IT = 0.2
COV_CALIBRATION_FACTORS = [1.0, 1.0, 1.0, 1.0, 1.7,
                           1.7, 1.7, 1.0, 1.0, 1.0,
                           1.0, 1.5, 1.5]
# Likelihood boundary for the predicted set
SET_LIKELIHOOD = 0.995

SARA_MEASUREMENT_UNCERTAINTY = 0.005

# Per-joint spherical occupancy radius in meters, ordered like JOINT_NAMES_13:
#   ['Nose', 'LShoulder', 'RShoulder', 'LElbow', 'RElbow', 'LWrist', 'RWrist',
#    'LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle']
# Each value is the largest half-thickness (radius = thickness / 2) of the body
# segments the joint belongs to, so the joint sphere envelops the thickest
# connected limb. Thicknesses from DIN 33402-2:2020-12 (95th percentile) plus the
# head/hand approximations:
#   Head 0.25, Torso (shoulder width) 0.434, UpperArm 0.2, Hand 0.206,
#   UpperLeg 0.203, LowerArm/LowerLeg 0.132.
HUMAN_RADIUS = [
    0.125,   # Nose      -> Head (0.25)
    0.217,   # LShoulder -> Torso_L (0.434)
    0.217,   # RShoulder -> Torso_R (0.434)
    0.100,   # LElbow    -> L_UpperArm (0.2)
    0.100,   # RElbow    -> R_UpperArm (0.2)
    0.103,   # LWrist    -> L_Hand (0.206)
    0.103,   # RWrist    -> R_Hand (0.206)
    0.217,   # LHip      -> Torso_L (0.434)
    0.217,   # RHip      -> Torso_R (0.434)
    0.1015,  # LKnee     -> L_UpperLeg (0.203)
    0.1015,  # RKnee     -> R_UpperLeg (0.203)
    0.066,   # LAnkle    -> L_LowerLeg (0.132)
    0.066,   # RAnkle    -> R_LowerLeg (0.132)
]

# Hand speed value according to DIN EN ISO 13855 2025-10-00 EN
V_HUMAN_ISO = 2.0

# Maximal speed of collaborative robot according to ISO TS 15066
V_ROBOT_ISO = 0.25

# Measurement uncertainty of the motion capture system [mm]
MOCAP_MEASUREMENT_UNCERTAINTY = 0.0
