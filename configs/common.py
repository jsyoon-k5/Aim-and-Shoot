import numpy as np

X, Y, Z = 0, 1, 2
UP = np.array([0, 1, 0])

TO_RADIAN = np.pi / 180.0
TO_DEGREE = 180.0 / np.pi
MM = 1e6
INCH_TO_METER = 0.0254

# FPS setting
RESOLUTION = np.array([1920, 1080])
MONITOR = np.array([53.13, 29.88])*0.01
MONITOR_BOUND = MONITOR / 2     # Unit: meter
CROSSHAIR = np.zeros(2)     # Center
FOV = np.array([103, 70.533])*TO_RADIAN     # Field of view, Unit: degree -> radian
TARGET_SPAWN_BD_HOR = [2.0, 37.0]
TARGET_SPAWN_BD_VER = [1.0, 21.0]

DIST_EYE_TO_MONITOR = 0.63      # Unit: meter
EYE_POSITION = np.array([0., 0.0779, 0.575])    # Unit: meter
EYE_POSITION_STD = np.array([
    0.0165648,
    0.0189342,
    0.0240023
])
EYE_POSITION_BOUND = np.array([0.065, 0.12, 0.167])

# METRIC NAME
M_TCT = "trial_completion_time"
M_SE = "shoot_error"
M_GD = "glancing_distance"
M_FSD = "first_saccade_duration"