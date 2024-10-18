import numpy as np
import cv2
from box import Box

MILLION = 1000000
FPSCI_TIME_STR_FORMAT = '%Y-%m-%d %H:%M:%S.%f'    # Time string format in FPSci data table

# This constant is only for the function default argument value.
# Strongly recommended to replace them in the runtime
MONITOR_QT = np.array([0.5313, 0.2988]) / 2

AXIS = Box(dict(
    X = 0,
    Y = 1,
    Z = 2,

    A = 0,
    E = 1,
))


VECTOR = Box(dict(
    HEAD = np.array([0., 0.0779, 0.575]),
    UP = np.array([0., 1., 0.]),
))


PARAM_SYMBOL = dict(
    theta_m=r"$\theta_m$",
    theta_p=r"$\theta_p$",
    theta_s=r"$\theta_s$", 
    theta_c=r"$\theta_c$", 

    rew_succ=r"$r_{succ}$",
    rew_fail=r"$r_{fail}$",
    decay_succ=r"$\lambda_{succ}$",
    decay_fail=r"$\lambda_{fail}$",
)
    

FOLDER = Box(dict(
    DATA_RL_MODEL = "ans_model",
    MODEL_CHECKPT = "checkpoint",
    MODEL_BEST = "best",
    RL_TENSORBOARD = "tb_log",

    VISUALIZATION = "visualization",
    VIDEO = "video",

    AMORTIZER = "amortizer",
))
    


METRIC = Box(dict(
    # Aim-and-Shoot performance
    SUMMARY = dict(
        TCT = "trial_completion_time",
        ACC = "accuracy",
        SHE = "shoot_error",
        NSE = "normalized_shoot_error",
        ENDPT = dict(
            X = "shoot_endpoint_x",
            Y = "shoot_endpoint_y",
            NX = "norm_shoot_endpoint_x",
            NY = "norm_shoot_endpoint_y",
        ),
        SCD = "saccadic_deviation",
        CTL = "camera_traverse_length",
        CGE = "camera_geometric_entropy",
        CSM = "camera_max_speed",
        GRT = "gaze_reaction_time",
        MRT = "mouse_reaction_time",
    ),
    SERIES = dict(
        SCCA = "saccadic_amplitude",
        SCCV = "saccadic_peak_velocity",
        SCCD = "saccadic_duration",
    )
))
    
FIELD = Box(dict(
    # Task condition and flags
    TRIAL = dict(
        TIER = "tier",
        PLAYER = "player",
        EXPCODE = "experiment_date",
        SENSI_MODE = "sensitivity_mode",     # Habitual, Default
        SENSI = "sensitivity",
        MDPI = "mouse_dpi",
        TARGET = "target_name",
        BIDX = "block_index",
        TIDX = "trial_index",
        TRID = "trial_id",
        SESSION_ID = "session_id",
        START_T = "start_time",
    ),

    TARGET = dict(
        SPD_LVL = "target_speed_level",
        RAD_LVL = "target_radius_level",
        CLR_LVL = "target_color_level",
        SPD = "target_speed",
        RAD = "target_radius",
        CLR = "target_color",
        POS = dict(
            GAME = dict(
                NAME0 = "target_init_pos_game",
                A = "target_pos_game_azim",
                E = "target_pos_game_elev",
                A0 = "target_init_pos_game_azim",
                E0 = "target_init_pos_game_elev"
            ),
            MONITOR = dict(
                X = "target_pos_monitor_x",
                Y = "target_pos_monitor_y",
                X0 = "target_init_pos_monitor_x",
                Y0 = "target_init_pos_monitor_y",
                D0 = "target_init_pos_distance",
                ID = "index_of_diff",
            ),
        ),
        ORBIT = dict(
            NAME = "target_orbit_axis",
            A = "target_orbit_axis_azim",
            E = "target_orbit_axis_elev"
        ),
        DIRECTION = "target_direction_angle"
    ),
        
    PLAYER = dict(
        CAM = dict(
            A = "player_cam_az",
            E = "player_cam_el",
            A0 = "player_init_cam_az",
            E0 = "player_init_cam_el",
        ),
        GAZE = dict(
            NAME = "gaze_position",
            X = "player_gaze_x",
            Y = "player_gaze_y",
            X0 = "player_init_gaze_x",
            Y0 = "player_init_gaze_y",
            D0 = 'player_init_gaze_dist',
            A = "player_gaze_azim",
            E = "player_gaze_elev",
            VALIDITY = "player_gaze_valid",
        ),
        HEAD = dict(
            NAME = "head_position",     # This is for the agent
            X = "player_head_x",
            Y = "player_head_y",
            Z = "player_head_z",
            X0 = "player_init_head_x",
            Y0 = "player_init_head_y",
            Z0 = "player_init_head_z",
        )   
    ),

    VALIDITY = dict(
        GAZE_INIT_POS = "close_gaze_init_pos",
        ENOUGH_GAZE = "sufficient_gaze"
    ),

    OUTLIER = dict(
        TCT = "outlier_tct_flag",
        SHE = "outlier_shoot_error",
    ),

    ETC = dict(
        TRUNC = "truncated",
        TIMESTAMP = "timestamp",
    )
))
    

VIDEO = Box(dict(
    CODEC = cv2.VideoWriter_fourcc(*'X264'),
    FONT = cv2.FONT_HERSHEY_SIMPLEX
))