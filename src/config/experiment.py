import numpy as np
from box import Box

class IJHCS:
    # Task environment
    MONITOR = np.array([0.5313, 0.2988])
    MONITOR_QT = np.array([0.5313, 0.2988]) / 2
    FOV = np.array([103, 70.533])
    RESOLUTION = np.array([1920, 1080])
    CROSSHAIR = np.zeros(2)

    # Group name
    TIER = ["pro", "ama"]
    PLAYER_NUM = dict(pro=10, ama=10)
    SENSI_MODE = ["habitual", "default"]

    # Data information
    PRESET_CAM_FRONT = np.zeros(2)      # Camera default front vector
    PRESET_CAM_POS = np.array([2.147898, 6.780012, 2.5])

    TRIAL_PER_BLOCK = 60
    BLOCK_PER_SESSION = 2

    # Used in processing
    THRESHOLD = Box(dict(
        VALID_GAZE_RATIO = 0.6,
        VALID_SHOOT_ERROR = 0.05,
        VALID_GAZE_DIST = 0.035,    # 3.5cm
        SACCADE_VEL = 15,
    ))

    SESSION_LIST = ["ssw", "fsw", "slw", "flw", "svsw", "ssg", "fsg", "slg", "flg"]
    SESSION_FULLNAME = {
        "ssw": "static_small_white", 
        "fsw": "fast_small_white", 
        "slw": "static_large_white", 
        "flw": "fast_large_white",
        "svsw": "static_very_small_white", 
        "ssg": "static_small_gray",  
        "fsg": "fast_small_gray", 
        "slg": "static_large_gray",  
        "flg": "fast_large_gray"
    }

    TASK_CONDITION = Box(dict(
        TARGET = dict(
            RADIUS = dict(
                verysmall = 0.0015,
                small = 0.0045,
                large = 0.012,
            ),
            SPEED = dict(
                stationary = 0,
                fast = 38.8656
            ),
            COLOR = dict(
                white = "#FFFFFF",
                gray = "#1E1E1E"
            )
        ),
        SENSI = dict(
            custom = "habitual",
            default = "default"
        )
    ))

    GP3 = Box(dict(
        LATENCY = 52_000,   # 52 millisecond
        ANGLE = 23.5,       # degree
        YPOS_OFFSET = 0.03  # meter
    ))

    F_CALIB = ['CALX', 'CALY', 'LX', 'LY', 'LV', 'RX', 'RY', 'RV']
    F_GAZE = [
        'CNT', 'TIME', 'TIME_TICK',
        'FPOGX', 'FPOGY', 'FPOGS', 'FPOGD', 'FPOGID', 'FPOGV',
        'LPOGX', 'LPOGY', 'LPOGV', 'RPOGX', 'RPOGY', 'RPOGV', 'BPOGX', 'BPOGY', 'BPOGV',
        'LPCX', 'LPCY', 'LPD', 'LPS', 'LPV', 'RPCX', 'RPCY', 'RPD', 'RPS', 'RPV',
        'LEYEX', 'LEYEY', 'LEYEZ', 'LPUPILD', 'LPUPILV',
        'REYEX', 'REYEY', 'REYEZ', 'RPUPILD', 'RPUPILV',
        'CX', 'CY', 'CS', 'BKID', 'BKDUR', 'BKPMIN'
    ]
    F_GAZE_TYPE = [
        int, float, int,
        float, float, float, float, int, int,
        float, float, int, float, float, int, float, float, int,
        float, float, float, float, int, float, float, float, float, int,
        float, float, float, float, int,
        float, float, float, float, int,
        float, float, int, int, float, int
    ]
    N_CALIB_PT = 9

    VERIF_SESS_NAME = "calib"
    VERIF_TARGET_POS = np.array([
        [ 0.00000000e+00,  3.80280598e-06],
        [-1.05653776e-01,  3.80280598e-06],
        [-1.69046041e-01,  8.45106024e-02],
        [-5.63493847e-02,  8.45106024e-02],
        [ 5.63472716e-02,  8.45106024e-02],
        [ 1.69046041e-01,  8.45106024e-02],
        [ 1.05653776e-01,  3.80280598e-06],
        [ 1.69046041e-01, -8.45029968e-02],
        [ 5.63472716e-02, -8.45029968e-02],
        [-5.63493847e-02, -8.45029968e-02],
        [-1.69046041e-01, -8.45029968e-02]
    ])
    VERIF_CORRECTION_WEIGHT = np.array([7, 3, 1, 3, 3, 1, 3, 1, 3, 3, 1])
    VERIF_TIMESTAMP = np.array([       # Stays for 1.7s and move in 0.1s
        [1.7*i for i in range(11)], 
        [1.7*(i+1)-0.1 for i in range(11)]
    ]).T
    VERIF_POS_ORDER = np.array([5, 4, 7, 8, 9, 10, 6, 3, 2, 1, 0])     # Index: exp. order, Element: pos. order (Up -> Bottom, Left -> Right)
    VERIF_POS_FIT = np.array([0, 1, 3, 4, 6, 8, 9])      # Points to use for POG correction



class T1AData:

    PLAYER = [
        "yeslab",
        "adb7199"
    ]

    # Task environment
    MONITOR = np.array([0.5313, 0.2988])
    MONITOR_QT = np.array([0.5313, 0.2988]) / 2
    FOV = np.array([103, 70.533])
    RESOLUTION = np.array([1920, 1080])
    CROSSHAIR = np.zeros(2)

    N_SESSION = 15
    SESSION_LIST = ["smallstat", "smallmove", "largestat", "largemove"]
    N_BLOCK = 4
    N_TRIAL = 60

    TARGET_NAME = [f"s{j}v{i}" for i in range(2) for j in range(2)]
    TARGET_RAD = [0.00451488, 0.01203968]
    TARGET_SPD = [0., 38.8656]

    PRESET_CAM_FRONT = np.zeros(2)      # Camera default front vector
    PRESET_CAM_POS = np.array([2.147898, 6.780012, 2.5])

    # Used in processing
    THRESHOLD = Box(dict(
        VALID_GAZE_RATIO = 0.6,
        VALID_SHOOT_ERROR = 0.05,
        VALID_GAZE_DIST = 0.035,    # 3.5cm
        SACCADE_VEL = 15,
    ))

    GP3 = Box(dict(
        LATENCY = 83_000,   # 83 millisecond
        ANGLE = 28,       # degree
        YPOS_OFFSET = 0.03  # meter
    ))