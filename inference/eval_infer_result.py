import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys, os, glob
sys.path.append("..")

from configs.common import *
from configs.experiment import *
from configs.path import *
from utilities.plots import fig_save
from utilities.mymath import compute_r2


def load_result(player, color, mode='default', metrics=[M_TCT, M_SE, M_GD]):
    # Ground truth
    gt = pd.read_csv(f"{PATH_INFER_RESULT}/gt_{player}.csv" % (mode, color))
    grouping = gt["group_index"].to_numpy()
    exper_result = gt[metrics].to_numpy()

    # Simulation
    fl = glob.glob(f"{PATH_INFER_RESULT}/mf_{player}_*.csv" % (mode, color))
    fl.sort()
    simul_result = pd.read_csv(fl[-1])
    simul_result = simul_result.iloc[-1].to_numpy()
    simul_result = simul_result[4:-2*len(metrics)]
    simul_result = simul_result.reshape((simul_result.size // len(metrics), len(metrics)))

    return grouping, exper_result, simul_result


def plot_player_mean(color, mode='default', metrics=[M_TCT, M_SE, M_GD]):
    exper_result = {}
    simul_result = {}
    for tier in PLAYER.keys():
        sr = []
        er = []
        for player in PLAYER[tier]:
            _, _er, _sr = load_result(player, color, mode=mode, metrics=metrics)
            sr.append(np.mean(_sr, axis=0))
            er.append(np.mean(_er, axis=0))
        exper_result[tier] = np.array(er)
        simul_result[tier] = np.array(sr)

    ### PLOT SETTING
    titles=[r"Trial completion time (s)", r"Shooting error (mm)", r"Glancing distance ($^\circ$)"]
    lims = [[0.4, 0.86], [0.001, 0.013], [1.2, 6.5]]
    ticks = [
        ([0.4, 0.5, 0.6, 0.7, 0.8], [0.4, 0.5, 0.6, 0.7, 0.8]),
        ([0.002, 0.005, 0.008, 0.011], [2, 5, 8, 11]),
        ([2, 3, 4, 5, 6], [2, 3, 4, 5, 6])
    ]
    ###############

    fig, axs = plt.subplots(1, 3, figsize=(11, 3), gridspec_kw={"width_ratios":[1, 1, 1]})

    for i in range(3):
        # Get R2
        e = np.concatenate((exper_result["AMA"][:,i], exper_result["PRO"][:,i]))
        s = np.concatenate((simul_result["AMA"][:,i], simul_result["PRO"][:,i]))
        func, r2 = compute_r2(e, s)

        min_x, max_x = e.min(), e.max()
        dist_val = (max_x - min_x) * 0.1

        # Plot
        axs[i].set_title(titles[i])
        if i == 0: axs[i].set_ylabel("Performance of Simulator")
        if i == 1: axs[i].set_xlabel("Performance of Human Player")

        # y = x line
        axs[i].plot(lims[i], lims[i], color='gray', linestyle='--', linewidth=0.5, zorder=10)
        # Fitting line
        axs[i].plot(
            [min_x - dist_val, max_x + dist_val],
            func([min_x - dist_val, max_x + dist_val]),
            color='r', linestyle='--', linewidth=0.8, zorder=11
        )
        # Scatterplot
        axs[i].scatter(
            exper_result["AMA"][:,i], simul_result["AMA"][:,i],
            marker='s', color='b', label='Ama.', s=20, zorder=12
        )
        axs[i].scatter(
            exper_result["PRO"][:,i], simul_result["PRO"][:,i],
            marker='^', color='k', label='Pro.', s=20, zorder=12
        )
        axs[i].set_xlim(*lims[i])
        axs[i].set_ylim(*lims[i])
        axs[i].set_xticks(*ticks[i])
        axs[i].set_yticks(*ticks[i])
        axs[i].legend(loc="upper left")
        axs[i].grid(True)
    
    fig_save(PATH_VIS_INFER_RESULT, "per_player_comp")






plot_player_mean('white')
    