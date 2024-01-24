import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
from matplotlib.lines import Line2D
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append('..')

from agent.agents import *
from agent.agent_simulation import *
from experiment.data_process import *
from configs.path import *
from utils.mymath import *
from utils.plots import fig_save

model_name = "full_231207_010501_40"

def save_fl(model, ckpt):
    print(f"Saving FL+TH info for {model_name} ckpt {ckpt}")
    simul = Simulator(model_name=model, checkpt=ckpt)
    simul.update_parameter(
        param_w=dict(
            theta_m=0.1,
            theta_p=0.1,
            theta_s=0.1,
            theta_c=0.1,
            hit=30,
            miss=16,
            hit_decay=0.7,
            miss_decay=0.3,
        )
    )

    load_cond = dict(
        player='thjska',
        # tier='AMA',
        mode='default',
        # target_speed=0,
        block_index=1,
        target_color='white',
        # session_id='ssw',
    )

    cond = simul._load_experiment_cond(**load_cond)
    exp = load_experiment(**load_cond)

    simul.run_simulation_with_cond(cond, verbose=True)
    df = simul.export_result_df()
    tid = df["t_iod"].to_numpy()
    tct = df[M_TCT].to_numpy()
    acc = df[M_ACC].to_numpy()
    se = df[M_SE].to_numpy()
    gd = df[M_GD].to_numpy()
    cge = df[M_CGE].to_numpy()
    csm = df[M_CSM].to_numpy()
    ctl = df[M_CTL].to_numpy()

    fig, ax = plt.subplots(3, 3, figsize=(9, 9), constrained_layout=True)

    # TCT
    rng = (
        min(tct.min(), exp[M_TCT].min()),
        max(tct.max(), exp[M_TCT].max())
    )
    ax[1][0].set_xlabel("Trial completion time")
    ax[1][0].hist(exp[M_TCT].to_numpy(), range=rng, bins=15, density=True, histtype='step', color='red', label='p')
    ax[1][0].axvline(exp[M_TCT].to_numpy().mean(), linewidth=1.5, color='red')
    ax[1][0].hist(tct, range=rng, bins=15, density=True, histtype='step', color='blue', label='s')
    ax[1][0].axvline(tct.mean(), linewidth=1.5, color='blue')
    ax[1][0].legend()


    res = np_groupby(np.vstack((tid, tct, acc)).T, key_pos=0, lvl=5)
    tid = res[:,0]
    tct = res[:,1]
    acc = res[:,2]

    r2 = get_r_squared(tid, tct)

    label = f"$R^2={r2:.2f}$"
    
    ### Fitts' law by ID - TCT
    sns.regplot(
        x=tid,
        y=tct,
        scatter_kws={"color": "black", "alpha": 0.3},
        line_kws={"color": "red", "lw": 2.5},
        ax=ax[0][0]
    )
    ax[0][0].set_title(f"{model_name}_{ckpt}")
    ax[0][0].set_xlabel("Index of Difficulty (bit)")
    ax[0][0].set_ylabel("Completion time (s)")
    
    ax[0][0].set_xlim([0, 6])
    ax[0][0].legend([Line2D([0], [0], color="red", lw=2.5)], [label,], fontsize=8, loc="lower right")
    ax[0][0].grid(linestyle="--", linewidth=0.5)

    ### ID - ACC
    sns.regplot(
        x=tid,
        y=acc,
        scatter_kws={"color": "black", "alpha": 0.3},
        line_kws={"color": "red", "lw": 2.5},
        ax=ax[0][1]
    )
    ax[0][1].set_xlabel("Index of Difficulty (bit)")
    ax[0][1].set_ylabel("Accuracy")
    
    ax[0][1].set_xlim([0, 6])
    ax[0][1].set_ylim([0, 1.1])
    ax[0][1].grid(linestyle="--", linewidth=0.5)

    # TH
    th_time, th_value = simul.m_prediction_horizon()
    ax[0][2].plot(th_time, th_value, linewidth=1, color='k', label='mean')

    for isn, sn in enumerate(["svsw", "ssw", "slw"]):
        select_trial = df[df["session_id"] == sn].index.to_list()
        th_time, th_value = simul.m_prediction_horizon(rec_indices=select_trial)
        ax[0][2].plot(th_time, th_value, linewidth=0.6, label=["3mm","9mm","24mm"][isn])
    # for isn, sn in enumerate(["fsw", "flw"]):
    #     select_trial = df[df["session_id"] == sn].index.to_list()
    #     th_time, th_value = simul.m_prediction_horizon(rec_indices=select_trial)
    #     ax[0][2].plot(th_time, th_value, linewidth=0.6, label=["9mm","24mm"][isn])
    ax[0][2].plot(*simul.m_shoot_attention(), linewidth=1, color='red', label='shoot')
    ax[0][2].plot(*simul.m_implicit_aim_point(), linewidth=1, color='green', label='aim')
    ax[0][2].set_xlim([0, 1])
    ax[0][2].set_ylim([0, 2.0])
    ax[0][2].legend()
    
    ax[0][2].set_xlabel("Normalized time")
    ax[0][2].set_ylabel("Prediction Horizon")

    # ### Fitts' law by target cond - tct
    # tid2 = []
    # tct2 = []
    # acc2 = []
    # tct_pts = []
    # acc_pts = []
    # for i, sn in enumerate(["svsw", "ssw", "slw"]):
    #     _df = df[df["session_id"] == sn]
    #     _tid = _df["t_iod"].to_numpy()
    #     _tct = _df[M_TCT].to_numpy()
    #     _acc = _df[M_ACC].to_numpy()
    #     res = np_groupby(np.vstack((_tid, _tct, _acc)).T, key_pos=0, lvl=2)
    #     _tid = res[:,0]; tid2.append(_tid)
    #     _tct = res[:,1]; tct2.append(_tct)
    #     _acc = res[:,2]; acc2.append(_acc)

    #     tct_pt = ax[1][0].scatter(
    #         _tid, _tct, marker='osD'[i], s=40, color='k', alpha=0.5
    #     )
    #     acc_pt = ax[1][1].scatter(
    #         _tid, _acc, marker='osD'[i], s=40, color='k', alpha=0.5
    #     )
    #     tct_pts.append(tct_pt)
    #     acc_pts.append(acc_pt)

    # tid2 = np.concatenate(tid2)
    # tct2 = np.concatenate(tct2)
    # acc2 = np.concatenate(acc2)
    # r2 = get_r_squared(tid2, tct2)
    # label = f"$R^2={r2:.2f}$"

    # sns.regplot(
    #     x=tid2,
    #     y=tct2,
    #     scatter_kws={"color": "black", "alpha": 0},
    #     line_kws={"color": "red", "lw": 2.5},
    #     ax=ax[1][0]
    # )
    
    # ax[1][0].set_title(f"{model_name}_{ckpt}")
    # ax[1][0].set_xlabel("Index of Difficulty (bit)")
    # ax[1][0].set_ylabel("Completion time (s)")
    
    # ax[1][0].set_xlim([0, 6])
    # ax[1][0].legend([Line2D([0], [0], color="red", lw=2.5), *tct_pts], [label, '3mm', '9mm', '24mm'], fontsize=8, loc="lower right")
    # ax[1][0].grid(linestyle="--", linewidth=0.5)


    # sns.regplot(
    #     x=tid2,
    #     y=acc2,
    #     scatter_kws={"color": "black", "alpha": 0},
    #     line_kws={"color": "red", "lw": 2.5},
    #     ax=ax[1][1]
    # )
    # ax[1][1].set_xlabel("Index of Difficulty (bit)")
    # ax[1][1].set_ylabel("Accuracy")
    
    # ax[1][1].set_xlim([0, 6])
    # ax[1][1].set_ylim([0, 1.1])
    # ax[1][1].grid(linestyle="--", linewidth=0.5)

    # SE
    rng = (
        min(se.min(), exp[M_SE].min()),
        max(se.max(), exp[M_SE].max())
    )
    ax[1][1].set_xlabel("Shoot error")
    ax[1][1].hist(exp[M_SE].to_numpy(), range=rng, bins=15, density=True, histtype='step', color='red', label='p')
    ax[1][1].hist(se, range=rng, bins=15, density=True, histtype='step', color='blue', label='s')
    ax[1][1].legend()

    # GD
    rng = (
        min(gd.min(), exp[M_GD].min()),
        max(gd.max(), exp[M_GD].max())
    )
    ax[1][2].set_xlabel("Glancing distance")
    ax[1][2].hist(exp[M_GD].to_numpy(), range=rng, bins=15, density=True, histtype='step', color='red', label='p')
    ax[1][2].axvline(exp[M_GD].to_numpy().mean(), linewidth=1.5, color='red')
    ax[1][2].hist(gd, range=rng, bins=15, density=True, histtype='step', color='blue', label='s')
    ax[1][2].axvline(gd.mean(), linewidth=1.5, color='blue')
    ax[1][2].legend()

    # CGE
    rng = (
        min(cge.min(), exp[M_CGE].min()),
        max(cge.max(), exp[M_CGE].max())
    )
    ax[2][0].set_xlabel("Camera geometric entp.")
    ax[2][0].hist(exp[M_CGE].to_numpy(), range=rng, bins=15, density=True, histtype='step', color='red', label='p')
    ax[2][0].axvline(exp[M_CGE].to_numpy().mean(), linewidth=1.5, color='red')
    ax[2][0].hist(cge, range=rng, bins=15, density=True, histtype='step', color='blue', label='s')
    ax[2][0].axvline(cge.mean(), linewidth=1.5, color='blue')
    ax[2][0].legend()

    # CSM
    rng = (
        min(csm.min(), exp[M_CSM].min()),
        max(csm.max(), exp[M_CSM].max())
    )
    ax[2][1].set_xlabel("Camera max spd.")
    ax[2][1].hist(exp[M_CSM].to_numpy(), range=rng, bins=15, density=True, histtype='step', color='red', label='p')
    ax[2][1].axvline(exp[M_CSM].to_numpy().mean(), linewidth=1.5, color='red')
    ax[2][1].hist(csm, range=rng, bins=15, density=True, histtype='step', color='blue', label='s')
    ax[2][1].axvline(csm.mean(), linewidth=1.5, color='blue')
    ax[2][1].legend()

    # CGE
    rng = (
        min(ctl.min(), exp[M_CTL].min()),
        max(ctl.max(), exp[M_CTL].max())
    )
    ax[2][2].set_xlabel("Camera trav. len.")
    ax[2][2].hist(exp[M_CTL].to_numpy(), range=rng, bins=15, density=True, histtype='step', color='red', label='p')
    ax[2][2].axvline(exp[M_CTL].to_numpy().mean(), linewidth=1.5, color='red')
    ax[2][2].hist(ctl, range=rng, bins=15, density=True, histtype='step', color='blue', label='s')
    ax[2][2].axvline(ctl.mean(), linewidth=1.5, color='blue')
    ax[2][2].legend()


    fig_save(PATH_RL_LOG % model_name, f"fl_{ckpt}")
    plt.close(fig)


# save_fl(model_name, 'best')
check_ckpt = [int(c.split('_')[-2]) for c in glob.glob(f"{PATH_RL_CHECKPT % model_name}rl_model_*_steps.zip")]
check_ckpt.sort(reverse=True)
for c in check_ckpt:
    save_fl(model_name, c)




# checkpoints = [int(c.split('_')[-2]) for c in glob.glob(f"{PATH_RL_CHECKPT % model_name}rl_model_*_steps.zip")]
# Parallel(n_jobs=6)(delayed(save_fl)(model_name, ckpt) for ckpt in tqdm(checkpoints))

