import numpy as np
import copy

import sys
sys.path.append('..')

from configs.common import *
from configs.simulation import *

VALUE_RANGE = {
    M_TCT: [-MAX_TCT, MAX_TCT],
    M_ACC: [-1, 1],
    M_SE: [-0.02, 0.02],
    M_SE_NORM: [-MAX_NORM_SE, MAX_NORM_SE],
    M_GD: [-MAX_GD, MAX_GD],
    M_CTL: [-0.5, 0.5],
    M_CSM: [-400, 400],
    M_CGE: [-80, 80],

    "endpoint_x": [-0.02, 0.02],
    "endpoint_y": [-0.02, 0.02],
    "endpoint_x_norm": [-0.02, 0.02],
    "endpoint_y_norm": [-0.02, 0.02],

    "tmpos0_x": [-MONITOR_BOUND[X], MONITOR_BOUND[X]], 
    "tmpos0_y": [-MONITOR_BOUND[Y], MONITOR_BOUND[Y]],
    "toax_az": [-180, 180], 
    "toax_el": [-90, 90],
    "target_speed": [-TARGET_ASPEED_FAST, TARGET_ASPEED_FAST], 
    "target_radius": [-TARGET_RADIUS_LARGE, TARGET_RADIUS_LARGE],
    "gaze_reaction": [-0.3, 0.3], 
    "hand_reaction": [-0.3, 0.3],
    "eye_y": [HEAD_POSITION[Y]-HEAD_POSITION_BOUND[Y], HEAD_POSITION[Y]+HEAD_POSITION_BOUND[Y]], 
    "eye_z": [HEAD_POSITION[Z]-HEAD_POSITION_BOUND[Z], HEAD_POSITION[Z]+HEAD_POSITION_BOUND[Z]]
}

COMPLETE_SUMMARY = [
    M_TCT, M_ACC, M_SE, M_SE_NORM, M_GD, M_CTL, M_CSM, M_CGE,
    "endpoint_x", "endpoint_y", "endpoint_x_norm", "endpoint_y_norm",
    "tmpos0_x", "tmpos0_y", "toax_az", "toax_el", "target_speed", "target_radius", 
    "gaze_reaction", "hand_reaction", "eye_y", "eye_z"
]

SUMMARY_DATA = dict(
    full = [
        M_TCT,
        M_SE_NORM,
        M_GD,

        "tmpos0_x", "tmpos0_y",
        "toax_az", "toax_el",
        "target_speed", "target_radius",
        "gaze_reaction", "hand_reaction",
        "eye_y", "eye_z"
    ],
    base = [
        M_TCT,
        M_SE_NORM,

        "tmpos0_x", "tmpos0_y",
        "toax_az", "toax_el",
        "target_speed", "target_radius",
    ],
    abl1 = [
        M_TCT,
        M_SE_NORM,

        "tmpos0_x", "tmpos0_y",
        "toax_az", "toax_el",
        "target_speed", "target_radius",
        "hand_reaction",
    ],
    abl2 = [
        M_TCT,
        M_SE_NORM,

        "tmpos0_x", "tmpos0_y",
        "toax_az", "toax_el",
        "target_speed", "target_radius",
        # "hand_reaction",
    ],
    abl3 = [
        M_TCT,
        M_SE_NORM,
        M_GD,

        "tmpos0_x", "tmpos0_y",
        "toax_az", "toax_el",
        "target_speed", "target_radius",
        "gaze_reaction", "hand_reaction",
        "eye_y", "eye_z"
    ],
)

COMPLETE_TRAJECTORY = ["timestamp", "target_x", "target_y", "gaze_x", "gaze_y", "camera_az", "camera_el"]

TRAJECTORY_DATA = dict(
    full = [
        "timestamp",
        "target_x",
        "target_y",
        # "gaze_x",
        # "gaze_y",
        "camera_az",
        "camera_el"
    ],
    base = [
        "timestamp",
        "target_x",
        "target_y",
        "camera_az",
        "camera_el"
    ],
    abl1 = [
        "timestamp",
        "target_x",
        "target_y",
        "camera_az",
        "camera_el"
    ],
    abl2 = [],
    abl3 = [],
)

# TARGET_METRIC = dict(
#     full = [M_TCT, M_ACC, M_GD],
#     base = [M_TCT, M_ACC],
#     abl1 = [M_TCT, M_ACC, M_GD],
#     abl2 = [M_TCT, M_ACC, M_GD],
# )

SIMULATOR_CONFIG = dict(
    full = dict(
        seed = None,
        policy = "full_231219_165740_73",
        checkpt = 20000000,
        targeted_params = [*(f"theta_{p}" for p in "mpsc"), "hit", "miss", "hit_decay", "miss_decay"],
        prior = "log-uniform",
    ),
    base = dict(
        seed = None,
        policy = "base_240104_164252_68",
        checkpt = 20000000,
        targeted_params = [*(f"theta_{p}" for p in "msc"), "hit", "miss", "hit_decay", "miss_decay"],
        prior = "log-uniform",
    ),
    abl1 = dict(
        seed = None,
        policy = "full_231219_165740_73",
        checkpt = 20000000,
        targeted_params = [*(f"theta_{p}" for p in "mpsc"), "hit", "miss", "hit_decay", "miss_decay"],
        prior = "log-uniform",
    ),
    abl2 = dict(
        seed = None,
        policy = "full_231219_165740_73",
        checkpt = 20000000,
        targeted_params = [*(f"theta_{p}" for p in "mpsc"), "hit", "miss", "hit_decay", "miss_decay"],
        prior = "log-uniform",
    ),
    abl3 = dict(
        seed = None,
        policy = "full_231219_165740_73",
        checkpt = 20000000,
        targeted_params = [*(f"theta_{p}" for p in "mpsc"), "hit", "miss", "hit_decay", "miss_decay"],
        prior = "log-uniform",
    ),
)

DEFAULT_TRAIN_CONFIG = dict(
    learning_rate = 0.0001,
    lr_gamma = 0.9,
    clipping = 0.5,
    point_estimation = False,
)

DEFAULT_ENCODER = dict(     # Generate static size latent vector from simul.
    batch_norm = True,
    traj_encoder_type = "conv_rnn",  # ["transformer", "conv_rnn"]
    transformer = dict( # Trajectory
        num_latents = 4,
        n_block = 2,
        query_sz = 8,   # 8~16
        out_sz = 8,     # 8~16
        head_sz = 8,
        n_head = 4,
        attn_dropout = 0.4, # 0.2~0.4 if low, training may be unstable
        res_dropout = 0.4,
        max_freq = 10,
        n_freq_bands = 2,
        max_step = 204,
    ),
    conv1d = [],
    rnn = dict(     # Trajectory
        type = "LSTM",
        bidirectional = True,
        dropout = 0.2,
        feat_sz = 8,     ### Experiment
        depth = 2,        ### Optional: 2 or 3
    ),
    mlp = dict(     # Static
        feat_sz = 64,   # 64~128
        out_sz = 24,    # mlp out_sz : encoder out_sz -> reliability weight of data (stat, traj)
        depth = 2,
    ),
)

DEFAULT_TRIAL_ENCODER = dict(       # Generate static size l.v. from stack of latent vector
    attention = dict(
        num_latents = 4,
        n_block = 2,
        query_sz = 32,
        out_sz = 32,
        head_sz = 8,
        n_head = 4,
        attn_dropout = 0.4,
        res_dropout = 0.4,
    )
)

DEFAULT_INN_CONFIG = dict(
    n_block = 5,
    act_norm = True,
    invert_conv = True,
    batch_norm = False,
    block = dict(
        permutation = False,
        head_depth = 2,
        head_sz = 32,
        cond_sz = 32,
        feat_sz = 32,   # 16, 32, 64
        depth = 2,
    )
)

DEFAULT_PEST_LINEAR = dict(
    in_sz = 32,
    hidden_sz = 256,
    hidden_depth = 2,
    batch_norm = False,
    activation = "relu",
)


default_ans_config = {
    mode: dict(
        name = mode,
        **DEFAULT_TRAIN_CONFIG,
        amortizer = dict(
            device = None,
            trial_encoder_type = "attention",   # ["attention", None]
            encoder = dict(
                traj_sz = len(TRAJECTORY_DATA[mode]),
                stat_sz = len(SUMMARY_DATA[mode]),
                **DEFAULT_ENCODER
            ),
            trial_encoder = copy.deepcopy(DEFAULT_TRIAL_ENCODER),
            invertible = dict(
                param_sz = len(SIMULATOR_CONFIG[mode]["targeted_params"]),       # No. of target parameters
                **DEFAULT_INN_CONFIG
            ),
            linear = dict(
                out_sz = len(SIMULATOR_CONFIG[mode]["targeted_params"]),
                **DEFAULT_PEST_LINEAR,
            )
        ),
        simulator = SIMULATOR_CONFIG[mode]
    ) for mode in [
        "full", 
        "base",
        "abl1", 
        "abl3"
    ]
}