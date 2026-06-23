class TINTERVAL:
    BUMP = 100
    MUSCLE = 50
    INTERP10 = 10
    INTERP1 = 1


class AGENT_STATE_ALIAS:
    TARGET_BUMP_LATER_POS_MONITOR_MM = 0
    TARGET_BUMP_TH_LATER_POS_MONITOR_MM = 1
    TARGET_VEL_HAT_BY_ORBIT_MM_S = 2
    CLOCK_NOISE = 3
    HTRAJ_IDEAL_P = 4
    HTRAJ_IDEAL_V = 5
    HTRAJ_ACTUAL_P = 6
    HTRAJ_ACTUAL_V = 7
    SHOOT_ERROR_MM = 8
    SHOOT_RESULT = 9
    SHOOT_ERROR_DEG = 10
    SHOOT_MOMENT_MS = 11
    SHOOT_ENDPOINT_MM = 12
    DONE = 13
    TRUNC = 14
    ACTION = 15
    REWARD = 16
    ELAPSED_TIME_MS = 17
    HTRAJ_SEG_SHOOT_POS = 18
    HTRAJ_SEG_SHOOT_VEL = 19
    HTRAJ_SEG_SHOOT_TS = 20
    MAX_HAND_SPEED = 21
    HAND_ACCEL_SUM = 22
    FINAL_HSPD_MM_S = 23


class FEATURES:
    TCT = "completion_time_ms"
    ACC = "accuracy"
    RES = "shoot_result"
    ERR = "shoot_error_mm"
    ERR_DEG = "shoot_error_deg"
    NORM_ERR = "normalized_shoot_error"
    MAX_HSPD = "max_hand_speed_mm_s"
    FINAL_HSPD = "final_hand_speed_mm_s"
    FINAL_HSPD = "final_hand_speed_mm_s"
    HAND_ACCEL_SUM = "hand_accel_sum_m_s2"
    TAU = "tau"
    REW = "reward"
    TRUNC = "truncated"
    # Derived / computed features
    TARGET_RADIUS_MM = "target_radius_mm"          # angular radius → mm (pinhole, at screen center)
    MOVEMENT_TIME = "movement_time_ms"              # TCT − hand_reaction_time
    HRT = "hand_reaction_time"                      # hand reaction time (ms); time from target appear to first mouse movement
    FITTS_IOD = "target_index_of_difficulty"        # log2(D / (2W) + 1), D and W in mm

    HTRAJP = "hand_traj_pos"
    HTRAJV = "hand_traj_vel"
    HTRAJT = "hand_traj_timestamp"
    GTRAJP = "gaze_traj_pos_mm"
    GTRAJ_AZEL = "gaze_traj_azel_deg"
    GTRAJT = "gaze_traj_timestamp"

    CAM_TRAJ_AZEL  = "camera_traj_azel"       # (T, 2) az/el deg – BUMP-resolution camera trajectory
    TGT_TRAJ_AZEL  = "target_traj_azel"       # (T, 2) az/el deg – BUMP-resolution target trajectory
    COARSE_TRAJ_TS = "coarse_traj_timestamps"  # (T,)   ms        – timestamps shared by cam/tgt

    TARGET_ASPEED = "target_angular_speed_deg_s"
    TARGET_RADIUS = "target_radius_deg"
    TARGET_POS_WORLD_AZ = "target_pos_world_az"
    TARGET_POS_WORLD_EL = "target_pos_world_el"
    TARGET_POS_MONITOR_X = "target_pos_monitor_x_mm"
    TARGET_POS_MONITOR_Y = "target_pos_monitor_y_mm"
    TARGET_POS_DISTANCE = "target_distance_mm"
    TARGET_ORBIT_AXIS_AZ = "target_orbit_axis_az_deg"
    TARGET_ORBIT_AXIS_EL = "target_orbit_axis_el_deg"
    TARGET_MOVEMENT_MONITOR_DIRECTION = "target_movement_monitor_direction_deg"
    TARGET_INIT_X = "target_init_x"                 # world-space initial position (game units)
    TARGET_INIT_Y = "target_init_y"
    TARGET_INIT_Z = "target_init_z"

    CAM_INIT_ANGLE_AZ = "camera_init_angle_az_deg"
    CAM_INIT_ANGLE_EL = "camera_init_angle_el_deg"
    CAM_SHOT_ANGLE_AZ = "shot_camera_az"             # camera az/el at the moment of the shot
    CAM_SHOT_ANGLE_EL = "shot_camera_el"

    TRIAL_START_UTC = "start_time_utc"              # UTC timestamp when active target appeared
    TRIAL_END_UTC = "end_time_utc"                  # UTC timestamp when trial ended

    GRT = "gaze_reaction_time"
    SCD_DEG = "saccadic_deviation_deg"
    SCD_MM = "saccadic_deviation_mm"
    GAZE_INIT_X = "gaze_init_x_mm"
    GAZE_INIT_Y = "gaze_init_y_mm"
    GAZE_INIT_DISTANCE = "gaze_init_distance_mm"
    GAZE_INIT_AZ = "gaze_init_az_deg"
    GAZE_INIT_EL = "gaze_init_el_deg"

    HEAD_POS_X = "head_position_x_mm"
    HEAD_POS_Y = "head_position_y_mm"
    HEAD_POS_Z = "head_position_z_mm"


class FEATURES_ALIAS:
    completion_time_ms = "TCT"
    accuracy = "ACC"
    shoot_result = "RES"
    shoot_error_mm = "ERR_mm"
    shoot_error_deg = "ERR_deg"
    normalized_shoot_error = "NERR"
    max_hand_speed_mm_s = "MAX_HSPD"
    hand_accel_sum_m_s2 = "HAND_ACCEL_SUM"
    final_hand_speed_mm_s = "FINAL_HSPD"
    saccadic_deviation_deg = "SCD_deg"
    saccadic_deviation_mm = "SCD_mm"


class FOLDERS:
    DATA = "data"
    RL_MODEL = "ans_agent_models"
    RL_TB = "tensorboard"
    MODEL_CHECKPT = "checkpoints"
    MODEL_BEST = "best"
    PM_TEACHER_DATA = "teacher_dataset"
    PM_BOARD = "board"      # TensorBoard logs per manifold run
    PM_PTS = "pts"          # .pt checkpoints per manifold run
    SIMUL = "simulations"
    MODEL_PROXY = "proxy"               # <RL_MODEL>/<model_name>/proxy/<split>
    PROXY_ANALYSIS = "analysis_fig"     # <RL_MODEL>/<model_name>/proxy/train/analysis_fig
    VIDEO = "videos"
    PRESETS = "presets"
    SUMMARY_DF = "summary_df"
    VISUALIZATION = "visualization"
    TRENDS = "trends"
    BRANCH_RL_PACKED_SOURCES = "packed_sources"
    AMORTIZER = "amortized_inference"    # data/amortized_inference/
    AMT_DATASETS = "datasets"            # …/datasets/<config_preset>/train|valid
    AMT_MODELS = "models"                # …/models/<run_name>/pts, board, etc.
    ANALYSIS = "analysis"                # data/analysis/
    POLICY_CKPT = "policy_ckpt"          # data/analysis/policy_ckpt/
    EMPIRICAL = "empirical"              # human experiment data root
    EMPIRICAL_RAW = "empirical/raw"      # raw SQLite databases from FPS experiment
    EMPIRICAL_PROCESSED = "empirical/processed"  # processed CSV/YAML outputs
    EMPIRICAL_VIS = "empirical/visualization"    # visualization outputs per user


# ── Column mapping: DB column → FEATURES constant name ────────────────
# session_trial_info columns that have a direct FEATURES equivalent.
# Used by the data-processing pipeline to produce DataFrames compatible
# with the RL simulator's ``get_summarized_result()`` output.
DB_TO_FEATURES = {
    "completion_time_ms": FEATURES.TCT,              # "completion_time_ms"
    "hit":                FEATURES.ACC,               # "accuracy"
    "shoot_error_deg":    FEATURES.ERR_DEG,           # "shoot_error_deg"
    "target_radius_deg":  FEATURES.TARGET_RADIUS,     # "target_radius_deg"
    "target_speed_deg_s": FEATURES.TARGET_ASPEED,     # "target_angular_speed_deg_s"
    "target_motion_dir_deg": FEATURES.TARGET_MOVEMENT_MONITOR_DIRECTION,
    "target_orbit_axis_az_deg": FEATURES.TARGET_ORBIT_AXIS_AZ,
    "target_orbit_axis_el_deg": FEATURES.TARGET_ORBIT_AXIS_EL,
    "target_pos_az_deg":  FEATURES.TARGET_POS_WORLD_AZ,
    "target_pos_el_deg":  FEATURES.TARGET_POS_WORLD_EL,
    "ref_click_camera_az": FEATURES.CAM_INIT_ANGLE_AZ,
    "ref_click_camera_el": FEATURES.CAM_INIT_ANGLE_EL,
}

# Columns in the summary CSV that do NOT come from DB_TO_FEATURES
# (experiment-specific metadata prepended to every row).
EMP_META_COLUMNS = ["user_name", "session_name", "trial_idx"]


class SYMBOLS:
    param_motor_noise = rf"\theta_m"
    param_position_noise = rf"\theta_p"
    param_speed_noise = rf"\theta_s"
    param_clock_noise = rf"\theta_c"
    param_succ_reward = rf"r_h"
    param_fail_penalty = rf"r_m"
    param_reward_decay = rf"\lambda_h"
    param_penalty_decay = rf"\lambda_m"


MAXIMUM_ELEV_ANGLE = 89.999999
TRUNCATE_ELEV_ANGLE = 89.0
