import numpy as np

X, Y, Z = 0, 1, 2
UP = np.array([0, 1, 0])

# TO_RADIAN = np.pi / 180.0
# TO_DEGREE = 180.0 / np.pi

MM = 1e6
INCH_TO_METER = 0.0254

# FPS setting
RESOLUTION = np.array([1920, 1080])
MONITOR = np.array([531.3, 298.8]) / 1000
MONITOR_BOUND = MONITOR / 2     # Unit: m
CAMERA_ANGLE_BOUND = 90
CROSSHAIR = np.zeros(2)     # Center
FOV = np.radians([103, 70.533])    # Field of view, Unit: degree -> radian

TARGET_SPAWN_BD_HOR = [2.0, 37.0]
TARGET_SPAWN_BD_VER = [1.0, 21.0]

TARGET_RADIUS_LARGE = 12e-3
TARGET_RADIUS_SMALL = 4.5e-3
TARGET_RADIUS_VERY_SMALL = 1.5e-3
TARGET_RADIUS_MINIMUM = 1.2e-3
TARGET_RADIUS_MAXIMUM = 13e-3

TARGET_ASPEED_STATIC = 0
TARGET_ASPEED_FAST = 38.8656     # deg/s
TARGET_ASPEED_MAX = 40

MAX_TCT = 3
MAX_NORM_SE = 5
MAX_GD = 24

VALID_INITIAL_GAZE_RADIUS = 0.035      # 3.5cm
HEAD_POSITION = np.array([0., 0.0779, 0.575])
HEAD_POSITION_STD = np.array([
    0.0165648,
    0.0189342,
    0.0240023
])
HEAD_POSITION_BOUND = np.array([0.065, 0.120, 0.167])
SACCADE_VEL_TH = 38
SACCADE_AMP_TH = 1

MEAN_GAZE_REACTION = 0.1490799318
MEAN_HAND_REACTION = 0.1609365982

# METRIC NAME
M_TCT = "trial_completion_time"
M_ACC = "result"
M_SE_NORM = "shoot_error_normalized"
M_SE = "shoot_error"
M_GD = "glancing_distance"
M_CTL = "cam_traverse_length"
M_CGE = "cam_geometric_entropy"
M_CSM = "cam_max_speed"

METRICS = [
    M_ACC,
    M_TCT,
    M_SE,
    M_SE_NORM,
    M_GD,
    "hand_reaction",
    "gaze_reaction",
    M_CGE,
    M_CSM,
    M_CTL
]

METRICS_ABBR = [
    "acc",
    "tct",
    "se",
    "se_norm",
    "gd",
    "hrt",
    "grt",
    "cge",
    "csm",
    "ctl"
]