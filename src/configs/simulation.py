import numpy as np
import sys

sys.path.append("..")
from configs.common import *


### FPS Simulation environment setting
# TARGET_ASPEED_MIN = 0.0
# TARGET_ASPEED_MAX = 45.0     # In experiment, this is 38.8656

# TARGET_RADIUS_MIN = 1.5e-3
# TARGET_RADIUS_MAX = 13e-3
TARGET_RADIUS_BASE = 4.5e-3

DEFAULT_HEAD_POS = np.array([0, 0, 0.58])

MIN_GAZE_REACTION = 50e-3      # unit: sec
MAX_GAZE_REACTION = 300e-3     # unit: sec
MIN_HAND_REACTION = 100e-3     # unit: sec
MAX_HAND_REACTION = 300e-3     # unit: sec

########
# DIRECTLY USED IN OTG
# CAREFUL WITH USING IN OTHER PARTS
BUMP_INTERVAL = 100                 # Unit: msec
MUSCLE_INTERVAL = 50                # Unit: msec
INTERP_INTERVAL = 1                 # Unit: msec
INTERP_INTERVAL_5 = 5               # Unit: msec
INTERP_INTERVAL_10 = 10
########

MAXIMUM_HAND_SPEED = 1.5            # Unit: m/s
MAXIMUM_CAM_SPEED = 300             # Unit: deg/s
MAXIMUM_TARGET_ELEVATION = 83.0     # Unit: degree
MAXIMUM_TARGET_MSPEED = 2.0         # Unit: m/s, depends on fps condition, ORBIT ONLY

THRESHOLD_SHOOT = 0.5
PENALTY_LARGE = -100


PARAM_SYMBOL = dict(
    theta_m=r"$\theta_m$",
    theta_p=r"$\theta_p$", 
    theta_s=r"$\theta_s$", 
    theta_c=r"$\theta_c$", 
    theta_n=r"$\theta_n$",
    theta_e=r"$\theta_e$",

    hit_decay=r"$\lambda_{h}$",
    miss_decay=r"$\lambda_{m}$",
    time=r"$\mathbf{r}_t$",
    time_h=r"$\lambda_{h}$",
    time_m=r"$\lambda_{m}$",

    hit=r"$\mathbf{r}_h$",
    miss=r"$\mathbf{r}_m$",
    motor=r"$\mathbf{r}_{motor}$",
)

PARAM_NAME = dict(
    theta_m="motor noise",
    theta_p="position noise",
    theta_s="speed noise",
    theta_c="clock noise",
    theta_n="motor noise (ppd)",
    theta_e="eff. copy",

    hit="hit reward",
    miss="miss penalty",
    time="time penalty",

    hit_decay="hit decay over time",
    miss_decay="miss decay over time",
    time_h="hit retention rate",
    time_m="miss retention rate",
    motor="motor effort",
)


### Simulated player setting

USER_CONFIG_1 = dict(
    params_mean = dict(
        # Cognitive-physical parameters
        theta_m=0.165,         # Motor noise constant (parallel)
        theta_p=0.110,         # Peripheral vision inaccuracy
        theta_s=0.139,         # Visual perception noise (Stocker's model)
        # theta_f=0.0,          # Foveal effect on speed perception
        theta_c=0.056,         # Click noise
        
        theta_q=42.8,        # Main Seq.: Tangent btw amplitude and peak velocity

        # Reward weights
        hit=24,
        miss=16,
        hit_decay=0.7,
        miss_decay=0.4,

        # Behavior constraint
        max_episode_length = 3.0,    # Second
        penalty_large = -100
    ),

    params_modulate = [*(f"theta_{p}" for p in "mpsc"), "hit", "miss", "hit_decay", "miss_decay"],
    params_max = dict(
        theta_m=0.5,
        theta_p=0.5,
        theta_s=0.5,
        theta_c=1.0,
        
        hit=64,
        miss=64,
        hit_decay=0.95,
        miss_decay=0.95
    ),
    params_min = dict(
        theta_m=0.0,
        theta_p=0.0,
        theta_s=0.0,
        theta_c=0.0,
        
        hit=1,
        miss=1,
        hit_decay=0.05,
        miss_decay=0.05
    ),
    param_log_scale = 1,

    obs_min = np.array(
        [
            *(-MONITOR_BOUND),          # Target position
            -MAXIMUM_TARGET_MSPEED,     
            -MAXIMUM_TARGET_MSPEED,     # Target monitor velocity (orbit)
            -TARGET_RADIUS_LARGE,       # Target radius
            -MAXIMUM_HAND_SPEED,
            -MAXIMUM_HAND_SPEED,        # Hand velocity
            *(-MONITOR_BOUND),          # Gaze position
            *(HEAD_POSITION - HEAD_POSITION_BOUND)    # EYE 3D POSITION
        ], dtype=np.float32
    ),
    obs_max = np.array(
        [
            *(MONITOR_BOUND),          # Target position
            MAXIMUM_TARGET_MSPEED,     
            MAXIMUM_TARGET_MSPEED,     # Target monitor velocity (orbit)
            TARGET_RADIUS_LARGE,       # Target radius
            MAXIMUM_HAND_SPEED,
            MAXIMUM_HAND_SPEED,        # Hand velocity
            *(MONITOR_BOUND),          # Gaze position
            *(HEAD_POSITION + HEAD_POSITION_BOUND)    # EYE 3D POSITION
        ], dtype=np.float32
    ),

    act_min = np.array([0.1, 0, 0, 0], dtype=np.float32), # th, kc, tc, kg
    act_max = np.array([2.0, 1, 1, 1], dtype=np.float32)
)


USER_CONFIG_1_BASE = dict(
    params_mean = dict(
        # Cognitive-physical parameters
        theta_m=0.165,         # Motor noise constant (parallel)
        theta_p=0.0,         # Peripheral vision inaccuracy
        theta_s=0.139,         # Visual perception noise (Stocker's model)
        theta_c=0.056,         # Click noise
        
        theta_q=42.8,        # Main Seq.: Tangent btw amplitude and peak velocity

        # Reward weights
        hit=24,
        miss=16,
        hit_decay=0.7,
        miss_decay=0.4,

        # Behavior constraint
        max_episode_length = 3.0,    # Second
        penalty_large = -100
    ),

    params_modulate = [*(f"theta_{p}" for p in "msc"), "hit", "miss", "hit_decay", "miss_decay"],
    params_max = dict(
        theta_m=0.5,
        # theta_p=0.35,
        theta_s=0.5,
        theta_c=1.0,
        
        hit=64,
        miss=64,
        hit_decay=0.95,
        miss_decay=0.95
    ),
    params_min = dict(
        theta_m=0.0,
        # theta_p=0.0,
        theta_s=0.0,
        theta_c=0.0,
        
        hit=1,
        miss=1,
        hit_decay=0.05,
        miss_decay=0.05
    ),
    param_log_scale = 1,

    obs_min = np.array(
        [
            *(-MONITOR_BOUND),          # Target position
            -MAXIMUM_TARGET_MSPEED,     
            -MAXIMUM_TARGET_MSPEED,     # Target monitor velocity (orbit)
            -TARGET_RADIUS_LARGE,       # Target radius
            -MAXIMUM_HAND_SPEED,
            -MAXIMUM_HAND_SPEED,        # Hand velocity
            # *(-MONITOR_BOUND),          # Gaze position
            # *(HEAD_POSITION - HEAD_POSITION_BOUND)    # EYE 3D POSITION
        ], dtype=np.float32
    ),
    obs_max = np.array(
        [
            *(MONITOR_BOUND),          # Target position
            MAXIMUM_TARGET_MSPEED,     
            MAXIMUM_TARGET_MSPEED,     # Target monitor velocity (orbit)
            TARGET_RADIUS_LARGE,       # Target radius
            MAXIMUM_HAND_SPEED,
            MAXIMUM_HAND_SPEED,        # Hand velocity
            # *(MONITOR_BOUND),          # Gaze position
            # *(HEAD_POSITION + HEAD_POSITION_BOUND)    # EYE 3D POSITION
        ], dtype=np.float32
    ),

    act_min = np.array([0.1, 0, 0], dtype=np.float32), # th, kc, tc
    act_max = np.array([2.0, 1, 1], dtype=np.float32)
)

USER_CONFIG_2 = dict(
    params_mean = dict(
        # Cognitive-physical parameters
        theta_m=0.165,         # Motor noise constant (parallel)
        theta_p=0.110,         # Peripheral vision inaccuracy
        theta_s=0.139,         # Visual perception noise (Stocker's model)
        theta_c=0.056,         # Click noise
        
        theta_q=42.8,        # Main Seq.: Tangent btw amplitude and peak velocity

        # Reward weights
        # hit=24,
        miss=16,
        hit_decay=0.7,
        miss_decay=0.4,

        # Behavior constraint
        max_episode_length = 3.0,    # Second
        penalty_large = -100
    ),

    params_modulate = [*(f"theta_{p}" for p in "mpsc"), "hit_max", "hit_peak", "hit_std", "miss", "hit_decay", "miss_decay"],
    params_max = dict(
        theta_m=0.5,
        theta_p=0.5,
        theta_s=0.5,
        theta_c=1,
        
        hit_max=100,
        hit_peak=1,
        hit_std=5,

        miss=100,
        hit_decay=0.95,
        miss_decay=0.95
    ),
    params_min = dict(
        theta_m=0.0,
        theta_p=0.0,
        theta_s=0.0,
        theta_c=0.0,
        
        hit_max=4,
        hit_peak=0,
        hit_std=0,

        miss=4,
        hit_decay=0.05,
        miss_decay=0.05
    ),
    param_log_scale = 1,

    obs_min = np.array(
        [
            *(-MONITOR_BOUND),          # Target position
            -MAXIMUM_TARGET_MSPEED,     
            -MAXIMUM_TARGET_MSPEED,     # Target monitor velocity (orbit)
            -TARGET_RADIUS_LARGE,       # Target radius
            -MAXIMUM_HAND_SPEED,
            -MAXIMUM_HAND_SPEED,        # Hand velocity
            *(-MONITOR_BOUND),          # Gaze position
            *(HEAD_POSITION - HEAD_POSITION_BOUND)    # EYE 3D POSITION
        ], dtype=np.float32
    ),
    obs_max = np.array(
        [
            *(MONITOR_BOUND),          # Target position
            MAXIMUM_TARGET_MSPEED,     
            MAXIMUM_TARGET_MSPEED,     # Target monitor velocity (orbit)
            TARGET_RADIUS_LARGE,       # Target radius
            MAXIMUM_HAND_SPEED,
            MAXIMUM_HAND_SPEED,        # Hand velocity
            *(MONITOR_BOUND),          # Gaze position
            *(HEAD_POSITION + HEAD_POSITION_BOUND)    # EYE 3D POSITION
        ], dtype=np.float32
    ),

    act_min = np.array([0.1, 0, 0, 0], dtype=np.float32), # th, kc, tc, kg
    act_max = np.array([2.0, 1, 1, 1], dtype=np.float32)
)


USER_CONFIG_2_BASE = dict(
    params_mean = dict(
        # Cognitive-physical parameters
        theta_m=0.165,         # Motor noise constant (parallel)
        theta_p=0.0,         # Peripheral vision inaccuracy
        theta_s=0.139,         # Visual perception noise (Stocker's model)
        theta_c=0.056,         # Click noise
        
        theta_q=42.8,        # Main Seq.: Tangent btw amplitude and peak velocity

        # Reward weights
        hit=24,
        miss=16,
        hit_decay=0.7,
        miss_decay=0.4,

        # Behavior constraint
        max_episode_length = 3.0,    # Second
        penalty_large = -100
    ),

    params_modulate = [*(f"theta_{p}" for p in "mpsc"), "hit_max", "hit_peak", "hit_std", "miss", "hit_decay", "miss_decay"],
    params_max = dict(
        theta_m=0.5,
        # theta_p=0.5,
        theta_s=0.5,
        theta_c=1,
        
        hit_max=100,
        hit_peak=1,
        hit_std=5,

        miss=100,
        hit_decay=0.95,
        miss_decay=0.95
    ),
    params_min = dict(
        theta_m=0.0,
        # theta_p=0.0,
        theta_s=0.0,
        theta_c=0.0,
        
        hit_max=4,
        hit_peak=0,
        hit_std=0,

        miss=4,
        hit_decay=0.05,
        miss_decay=0.05
    ),
    param_log_scale = 1,

    obs_min = np.array(
        [
            *(-MONITOR_BOUND),          # Target position
            -MAXIMUM_TARGET_MSPEED,     
            -MAXIMUM_TARGET_MSPEED,     # Target monitor velocity (orbit)
            -TARGET_RADIUS_LARGE,       # Target radius
            -MAXIMUM_HAND_SPEED,
            -MAXIMUM_HAND_SPEED,        # Hand velocity
            # *(-MONITOR_BOUND),          # Gaze position
            # *(HEAD_POSITION - HEAD_POSITION_BOUND)    # EYE 3D POSITION
        ], dtype=np.float32
    ),
    obs_max = np.array(
        [
            *(MONITOR_BOUND),          # Target position
            MAXIMUM_TARGET_MSPEED,     
            MAXIMUM_TARGET_MSPEED,     # Target monitor velocity (orbit)
            TARGET_RADIUS_LARGE,       # Target radius
            MAXIMUM_HAND_SPEED,
            MAXIMUM_HAND_SPEED,        # Hand velocity
            # *(MONITOR_BOUND),          # Gaze position
            # *(HEAD_POSITION + HEAD_POSITION_BOUND)    # EYE 3D POSITION
        ], dtype=np.float32
    ),

    act_min = np.array([0.1, 0, 0], dtype=np.float32), # th, kc, tc
    act_max = np.array([2.0, 1, 1], dtype=np.float32)
)


USER_CONFIG_3 = dict(
    params_mean = dict(
        # Cognitive-physical parameters
        theta_m=0.165,         # Motor noise constant (parallel)
        theta_p=0.110,         # Peripheral vision inaccuracy
        theta_s=0.139,         # Visual perception noise (Stocker's model)
        # theta_f=0.0,          # Foveal effect on speed perception
        theta_c=0.056,         # Click noise
        
        theta_q=42.8,        # Main Seq.: Tangent btw amplitude and peak velocity

        # Reward weights
        hit=24,
        miss=16,
        hit_decay=0.7,
        miss_decay=0.4,

        # Behavior constraint
        max_episode_length = 3.0,    # Second
        penalty_large = -100
    ),

    params_modulate = [*(f"theta_{p}" for p in "mpsc"), "hit", "miss", "hit_decay", "miss_decay"],
    params_max = dict(
        theta_m=0.5,
        theta_p=0.5,
        theta_s=0.5,
        theta_c=0.5,
        
        hit=64,
        miss=64,
        hit_decay=0.95,
        miss_decay=0.95
    ),
    params_min = dict(
        theta_m=0.0,
        theta_p=0.0,
        theta_s=0.0,
        theta_c=0.0,
        
        hit=1,
        miss=1,
        hit_decay=0.05,
        miss_decay=0.05
    ),
    param_log_scale = 1,

    obs_min = np.array(
        [
            *(-MONITOR_BOUND),          # Target position
            -MAXIMUM_TARGET_MSPEED,     
            -MAXIMUM_TARGET_MSPEED,     # Target monitor velocity (orbit)
            -TARGET_RADIUS_LARGE,       # Target radius
            -MAXIMUM_HAND_SPEED,
            -MAXIMUM_HAND_SPEED,        # Hand velocity
            *(-MONITOR_BOUND),          # Gaze position
            *(HEAD_POSITION - HEAD_POSITION_BOUND)    # EYE 3D POSITION
        ], dtype=np.float32
    ),
    obs_max = np.array(
        [
            *(MONITOR_BOUND),          # Target position
            MAXIMUM_TARGET_MSPEED,     
            MAXIMUM_TARGET_MSPEED,     # Target monitor velocity (orbit)
            TARGET_RADIUS_LARGE,       # Target radius
            MAXIMUM_HAND_SPEED,
            MAXIMUM_HAND_SPEED,        # Hand velocity
            *(MONITOR_BOUND),          # Gaze position
            *(HEAD_POSITION + HEAD_POSITION_BOUND)    # EYE 3D POSITION
        ], dtype=np.float32
    ),

    act_min = np.array([0.1, 0, 0, 0], dtype=np.float32), # th, kc, tc, kg
    act_max = np.array([2.0, 1, 1, 1], dtype=np.float32)
)


USER_CONFIG_3_BASE = dict(
    params_mean = dict(
        # Cognitive-physical parameters
        theta_m=0.165,         # Motor noise constant (parallel)
        theta_p=0.0,         # Peripheral vision inaccuracy
        theta_s=0.139,         # Visual perception noise (Stocker's model)
        theta_c=0.056,         # Click noise
        
        theta_q=42.8,        # Main Seq.: Tangent btw amplitude and peak velocity

        # Reward weights
        hit=24,
        miss=16,
        hit_decay=0.7,
        miss_decay=0.4,

        # Behavior constraint
        max_episode_length = 3.0,    # Second
        penalty_large = -100
    ),

    params_modulate = [*(f"theta_{p}" for p in "msc"), "hit", "miss", "hit_decay", "miss_decay"],
    params_max = dict(
        theta_m=0.5,
        theta_s=0.5,
        theta_c=0.5,
        
        hit=64,
        miss=64,
        hit_decay=0.95,
        miss_decay=0.95
    ),
    params_min = dict(
        theta_m=0.0,
        theta_s=0.0,
        theta_c=0.0,
        
        hit=1,
        miss=1,
        hit_decay=0.05,
        miss_decay=0.05
    ),
    param_log_scale = 1,

    obs_min = np.array(
        [
            *(-MONITOR_BOUND),          # Target position
            -MAXIMUM_TARGET_MSPEED,     
            -MAXIMUM_TARGET_MSPEED,     # Target monitor velocity (orbit)
            -TARGET_RADIUS_LARGE,       # Target radius
            -MAXIMUM_HAND_SPEED,
            -MAXIMUM_HAND_SPEED,        # Hand velocity
            # *(-MONITOR_BOUND),          # Gaze position
            # *(HEAD_POSITION - HEAD_POSITION_BOUND)    # EYE 3D POSITION
        ], dtype=np.float32
    ),
    obs_max = np.array(
        [
            *(MONITOR_BOUND),          # Target position
            MAXIMUM_TARGET_MSPEED,     
            MAXIMUM_TARGET_MSPEED,     # Target monitor velocity (orbit)
            TARGET_RADIUS_LARGE,       # Target radius
            MAXIMUM_HAND_SPEED,
            MAXIMUM_HAND_SPEED,        # Hand velocity
            # *(MONITOR_BOUND),          # Gaze position
            # *(HEAD_POSITION + HEAD_POSITION_BOUND)    # EYE 3D POSITION
        ], dtype=np.float32
    ),

    act_min = np.array([0.1, 0, 0], dtype=np.float32), # th, kc, tc
    act_max = np.array([2.0, 1, 1], dtype=np.float32)
)

USER_CONFIG_BASE_FINAL = dict(
    params_mean = dict(
        # Cognitive-physical parameters
        theta_m=0.165,         # Motor noise constant (parallel)
        theta_p=0.0,         # Peripheral vision inaccuracy
        theta_s=0.139,         # Visual perception noise (Stocker's model)
        theta_c=0.056,         # Click noise
        
        theta_q=42.8,        # Main Seq.: Tangent btw amplitude and peak velocity

        cmu = 0.185,
        nu = 19.931,
        delta = 0.399,

        # Reward weights
        hit=24,
        miss=16,
        hit_decay=0.7,
        miss_decay=0.4,

        # Behavior constraint
        max_episode_length = 3.0,    # Second
        penalty_large = -100
    ),

    params_modulate = [*(f"theta_{p}" for p in "msc"), "hit", "miss", "hit_decay", "miss_decay"],
    params_max = dict(
        theta_m=0.5,
        theta_s=0.5,
        theta_c=0.5,
        
        hit=64,
        miss=64,
        hit_decay=0.95,
        miss_decay=0.95
    ),
    params_min = dict(
        theta_m=0.0,
        theta_s=0.0,
        theta_c=0.0,
        
        hit=1,
        miss=1,
        hit_decay=0.05,
        miss_decay=0.05
    ),
    param_log_scale = 1,

    obs_min = np.array(
        [
            *(-MONITOR_BOUND),          # Target position
            -MAXIMUM_TARGET_MSPEED,     
            -MAXIMUM_TARGET_MSPEED,     # Target monitor velocity (orbit)
            -TARGET_RADIUS_LARGE,       # Target radius
            -MAXIMUM_HAND_SPEED,
            -MAXIMUM_HAND_SPEED,        # Hand velocity
            # *(-MONITOR_BOUND),          # Gaze position
            # *(HEAD_POSITION - HEAD_POSITION_BOUND)    # EYE 3D POSITION
        ], dtype=np.float32
    ),
    obs_max = np.array(
        [
            *(MONITOR_BOUND),          # Target position
            MAXIMUM_TARGET_MSPEED,     
            MAXIMUM_TARGET_MSPEED,     # Target monitor velocity (orbit)
            TARGET_RADIUS_LARGE,       # Target radius
            MAXIMUM_HAND_SPEED,
            MAXIMUM_HAND_SPEED,        # Hand velocity
            # *(MONITOR_BOUND),          # Gaze position
            # *(HEAD_POSITION + HEAD_POSITION_BOUND)    # EYE 3D POSITION
        ], dtype=np.float32
    ),

    act_min = np.array([0.1, 0], dtype=np.float32), # th, kc
    act_max = np.array([2.0, 1], dtype=np.float32)
)