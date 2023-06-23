import numpy as np
import sys

sys.path.append("..")
from configs.common import EYE_POSITION, EYE_POSITION_BOUND, MONITOR_BOUND


## TIME UNITs
TIME_UNIT = {
    1: 0.001,
    2: 0.002,
    5: 0.005,
    10: 0.01,
    50: 0.05,
    100: 0.1,
    300: 0.3,
    1000: 1.0,
    2000: 2.0
}


### FPS Simulation environment setting
TARGET_ASPEED_MIN = 0.0
TARGET_ASPEED_MAX = 45.0     # In experiment, this is 38.8656

TARGET_RADIUS_MIN = 1.5e-3
TARGET_RADIUS_MAX = 13e-3
TARGET_RADIUS_BASE = 4.5e-3


### Simulated player setting
USER_PARAM_MEAN = dict(
    sensi = 1000,         # == 1 deg/mm

    # Cognitive-physical parameters
    theta_s=0.139,         # Visual perception noise (Stocker's model)
    theta_p=0.110,         # Peripheral vision inaccuracy
    theta_m=0.165,         # Motor noise constant (parallel)
    theta_q=18.24,         # Main Seq.: Tangent btw amplitude and peak velocity

    # Additional constants (Fixed)
    theta_c=0.056,       # Precision of internal clock
    va=37.9,             # Main Seq.: Peak vel. at amplitude
    cmu=0.33,            # Implicit aim point
    nu=20.52398,         # Drift rate
    delta=0.411,         # Visual encoding precision limit 

    # Reward weights
    hit = 20,
    miss = -4, 
    time = -41,
    motor = -22,
)

USER_PARAM_STD = dict(
    theta_s=0.061, 
    theta_p=0.031,
    theta_m=0.073,
    theta_q=2.1,
    # theta_c=0.037,

    # hit = 20,
    miss = -4,
    time = -41,
    motor = -22,
)

USER_PARAM_MAX = dict(
    theta_s=0.4,
    theta_p=0.3,
    theta_m=0.5,
    theta_q=27.00,

    miss = -1,
    time = -1,
    motor = -1
)

USER_PARAM_MIN = dict(
    theta_s=0.02,
    theta_p=0.02,
    theta_m=0.05,
    theta_q=12.00,

    miss = -40,
    time = -80,
    motor = -40
)

MAX_SIGMA_SCALE = 3

COG_SPACE = [f"theta_{p}" for p in "spmq"]
REW_SPACE = ["miss", "time", "motor"]   # Fix hit reward
ALL_SPACE = COG_SPACE + REW_SPACE
PARAM_SPACE = dict(
    cog = COG_SPACE,
    rew = REW_SPACE,
    all = COG_SPACE + REW_SPACE
)


# EPISODE SETTING
BUMP_INTERVAL = 0.1                 # Unit: sec
MUSCLE_INTERVAL = 0.05              # Unit: sec
INTERP_INTERVAL = 0.002             # Unit: sec

MAXIMUM_PREDICTION_HORIZON = 2.0    # Unit: sec
MAXIMUM_EPISODE_LENGTH = 3.0        # Unit: sec
MAXIMUM_HAND_SPEED = 3.0            # Unit: m/s
MAXIMUM_TARGET_ELEVATION = 83.0     # Unit: degree
MAXIMUM_TARGET_MSPEED = 1.0         # Unit: m/s, depends on fps condition

THRESHOLD_SHOOT = 0.9
THRESHOLD_AMPLIFY_PENALTY_DIST = 0.04   # Unit: m
FORCED_TERMINATION_PENALTY_COEFF = 20

OBS_MIN = np.array(
    [
        *(-MONITOR_BOUND),          # Target position
        -MAXIMUM_TARGET_MSPEED,     
        -MAXIMUM_TARGET_MSPEED,     # Target monitor velocity (by orbit)
        TARGET_RADIUS_MIN,          # Target radius
        -MAXIMUM_HAND_SPEED,
        -MAXIMUM_HAND_SPEED,        # Hand velocity
        *(-MONITOR_BOUND),          # Gaze position
        *(EYE_POSITION - EYE_POSITION_BOUND)    # EYE 3D POSITION
    ]
)

OBS_MAX = np.array(
    [
        *(MONITOR_BOUND),           # Target position
        MAXIMUM_TARGET_MSPEED,     
        MAXIMUM_TARGET_MSPEED,      # Target monitor velocity (by orbit)
        TARGET_RADIUS_MAX,          # Target radius
        MAXIMUM_HAND_SPEED,
        MAXIMUM_HAND_SPEED,         # Hand velocity
        *(MONITOR_BOUND),           # Gaze position
        *(EYE_POSITION + EYE_POSITION_BOUND)    # EYE 3D POSITION
    ]
)

ACT_MIN = np.array(
    [
        BUMP_INTERVAL,  # Prediction Horizon
        0,              # Shoot attention
        0               # Gaze damping (crosshair)
    ]
)

ACT_MAX = np.array(
    [
        MAXIMUM_PREDICTION_HORIZON,     # Prediction Horizon
        1,                              # Shoot attention
        1                               # Gaze damping (target)
    ]
)

FIXED_Z_PRESET = np.array([
    [0, 0, 0, 0],
    [-0.3, 0.7, 0.2, -0.3],
    [0.4, -0.2, 0.7, 0.04],
    [-0.5, 0.2, 0, 0.6],
    [0.7, -0.7, 0.7, -0.2],
    [-0.6, 0.5, -0.2, 0.3],
    [0.1, 0.8, 0, 0.4],
    [-0.3, 0, -1, 0.8]
])