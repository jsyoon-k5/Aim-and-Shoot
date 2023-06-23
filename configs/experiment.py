import numpy as np

TIER = ["AMA", "PRO"]
PLAYER = {
    "AMA": [
        "2also0", #
        "mlp", #
        "sos4252", #
        "KKW", #
        "Tigrise", #
        "kth", # 
        "YEC", 
        "KEH",
        "HJI", #
        "pdh", #
    ],
    "PRO": [
        "Agatha0514",
        "what2d0",
        "sjrla",
        "F0verqz",
        "C9INEkr",
        "dispect",
        "Richard",
        "thjska",
        "khanis0",
        "Yoonjin04",
    ]
}
PLAYERS = PLAYER["AMA"] + PLAYER["PRO"]
MODE = ["custom", "default"]
COLOR = ["white", "gray"]

TARGET_RADIUS_LARGE = 12e-3
TARGET_RADIUS_SMALL = 4.5e-3
TARGET_RADIUS_VERY_SMALL = 1.5e-3
TARGET_ASPEED_STATIC = 0
TARGET_ASPEED_FAST = 38.8656
TARGET_DIST_MEDIAN = 0.08399429194751959

LEARNING_EFFECT_IGNORANCE = 10      # Remove first few trials considering potential learning effect
TIME_STR_FORMAT = '%Y-%m-%d %H:%M:%S.%f'    # Time string format in FPSci data table

PRESET_CAM_FRONT = np.zeros(2)      # Camera default front vector
PRESET_CAM_POS = np.array([2.147898, 6.780012, 2.5])

MINIMUM_VALID_RATIO_OF_GAZE = 0.4
VALID_INITIAL_GAZE_RADIUS = 0.035   # 3.5cm
SACCADE_MIN_VEL = 15                # deg/s

TRIAL_NUM_PER_BLOCK = 60
SES_NAME_ABBR = [
    "ssw", 
    "fsw", 
    "slw", 
    "flw", 
    "svsw", 
    "ssg", 
    "fsg", 
    "slg", 
    "flg"
]
SES_NAME_FULL = [
    "static_small_white", 
    "fast_small_white", 
    "static_large_white", 
    "fast_large_white",
    "static_very_small_white", 
    "static_small_gray",  
    "fast_small_gray", 
    "static_large_gray",  
    "fast_large_gray"
]

GP3_LATENCY = 52_000      # 1_000_000 is 1 sec
GP3_ANGLE = 23.5          # Unit: degree
GP3_YPOS_OFFSET = 0.03    # Unit: meter


# Experiment - verification session settings
VERIF_SESS_NAME = "calib"
# VERIF_TARGET_POS = np.array([
#     [3.149545, 6.78003, 2.5],
#     [3.149545, 6.78003, 2.0],
#     [3.149545, 7.18003, 1.7],
#     [3.149545, 7.18003, 2.23333],
#     [3.149545, 7.18003, 2.76666],
#     [3.149545, 7.18003, 3.3],
#     [3.149545, 6.78003, 3.0],
#     [3.149545, 6.38003, 3.3],
#     [3.149545, 6.38003, 2.76666],
#     [3.149545, 6.38003, 2.23333],
#     [3.149545, 6.38003, 1.7]
# ])
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
    [1.7*i for i in range(VERIF_TARGET_POS.shape[0])], 
    [1.7*(i+1)-0.1 for i in range(VERIF_TARGET_POS.shape[0])]
]).T
VERIF_POS_ORDER = np.array([5, 4, 7, 8, 9, 10, 6, 3, 2, 1, 0])     # Index: exp. order, Element: pos. order (Up -> Bottom, Left -> Right)
VERIF_POS_FIT = np.array([0, 1, 3, 4, 6, 8, 9])      # Points to use for POG correction

METRICS = [
    "result",
    "trial_completion_time",
    "shoot_error",
    "glancing_distance",
    # "first_saccade_duration",
    "hand_reaction",
    "gaze_reaction",
    "hand_geometric_entropy"
]

METRICS_ABBR = [
    "acc",
    "tct",
    "se",
    "gd",
    # "fsd",
    "hrt",
    "grt",
    "hgent"
]





F_CALIB = ['CALX', 'CALY', 'LX', 'LY', 'LV', 'RX', 'RY', 'RV']
F_GAZE = [
    'CNT', 'TIME', 'TIME_TICK',
    'FPOGX', 'FPOGY', 'FPOGS', 'FPOGD', 'FPOGID', 'FPOGV', \
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