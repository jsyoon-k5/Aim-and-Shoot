import enum
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib import gridspec, colors
import pandas as pd
import numpy as np
import seaborn as sns
import os, sys

from scipy.stats import gaussian_kde, pearsonr, ttest_ind, f_oneway
from scipy.optimize import minimize_scalar
from time import time
from tqdm import tqdm

from sklearn.metrics import r2_score
from matplotlib.ticker import StrMethodFormatter

sys.path.append("..")
from configs.common import *
from configs.path import *
from configs.experiment import SES_NAME_ABBR_SYMM, TIER, PLAYER, PLAYERS, COLOR
from utils.mymath import cl_95_intv, discrete_labeling, find_divisors, get_r_squared
from utils.distrib_distance import hist_kld

def fig_save(dir, fn, DPI=100, save_svg=False):
    os.makedirs(dir, exist_ok=True)
    if dir[-1] == '/':
        fullpath = f"{dir}{fn}"
    else:
        fullpath = f"{dir}/{fn}"
    plt.savefig(f"{fullpath}.png", dpi=DPI, bbox_inches='tight', pad_inches=0)
    if save_svg:
        while True:
            try:
                plt.savefig(f"{fullpath}.pdf", dpi=DPI, bbox_inches='tight', pad_inches=0)
                break
            except PermissionError:
                _ = input("The file seems to be opened... close and press enter to proceed.")
    plt.close('all')


def fig_axes_setting(axs, zoom=1, num_ticks=3, grid=True):
    """Monitor plot axes setting"""
    axs.set_xlim(-MONITOR_BOUND[X] / zoom, MONITOR_BOUND[X] / zoom)
    axs.set_ylim(-MONITOR_BOUND[Y] / zoom, MONITOR_BOUND[Y] / zoom)
    xlin = np.linspace(-MONITOR_BOUND[X] / zoom, MONITOR_BOUND[X] / zoom, num_ticks+2)
    ylin = np.linspace(-MONITOR_BOUND[Y] / zoom, MONITOR_BOUND[Y] / zoom, num_ticks+2)
    axs.set_xticks(xlin, ["%2.1f" % (ff*100) for ff in xlin])
    axs.set_yticks(ylin, ["%2.1f" % (ff*100) for ff in ylin])
    axs.set_xlabel("Monitor Width (cm)")
    axs.set_ylabel("Monitor Height (cm)")
    if grid: axs.grid(grid, zorder=0)
    else: axs.grid(grid)


def set_tick_and_range(axs, unit, tick_scale, max_value, min_value=0, axis='x', omit_tick=0, erase_tick=False):
    r_min = unit * int(np.floor(min_value / unit))
    r_max = unit * int(np.ceil(max_value / unit))
    rng = (r_min, r_max)
    tick_v = np.linspace(r_min, r_max, int((r_max - r_min) / unit) + 1)
    tick_n = list(map(int, tick_v * tick_scale))
    if omit_tick > 1:
        for i in range(1, len(tick_n), omit_tick):
            tick_n[i] = ''

    if axis == 'x':
        axs.set_xlim(*rng)
        if not erase_tick:
            axs.set_xticks(tick_v, tick_n)
        else:
            axs.set_xticks(tick_v, [''] * len(tick_v))
    elif axis == 'y':
        axs.set_ylim(*rng)
        if not erase_tick:
            axs.set_yticks(tick_v, tick_n)
        else:
            axs.set_yticks(tick_v, [''] * len(tick_v))


def auto_set_tick_and_range(max_v, min_v, tick_scale, unit, offset=(0.1, 0.23), fmt='%.1f'):
    # Deprecate this function later.
    d = max_v - min_v
    rng = (min_v - d * offset[0], max_v + d * offset[1])

    ticks = np.arange(rng[1] // unit + 1) * unit
    ticks = ticks[ticks >= rng[0]]
    ticks = ticks[ticks <= rng[1]]

    dtype = np.rint if fmt == '%d' else float
    ticks_lb = [fmt % dtype(v) for v in (ticks * tick_scale)]

    return rng, ticks, ticks_lb



def plt_trajectory(timestamp, target, gaze, gv, trad, username, tag):
    """Draw trajectory and distance plot"""
    fig, axs = plt.subplots(2, 1, figsize=(8, 8.5), gridspec_kw={"height_ratios":[1, 1]})

    axs[0].set_title("Traj and Dist (%s, %s)" % (username, tag))
    axs[0].scatter(*target.T, marker='.', s=0.5, color='k', label="target", zorder=15)
    if len(gaze) > 0:
        axs[0].scatter(*(gaze[gv==1]).T, marker='.', s=0.5, color='r', label="gaze", zorder=16)
        axs[0].scatter(*(gaze[gv==0]).T, marker='.', s=0.5, color='b', label="gaze(interp)", zorder=16)
        axs[0].scatter(*gaze[0], color='r', marker='o', s=10, label="gaze(start)", zorder=17)
        axs[0].scatter(*gaze[-1], color='r', marker='x', s=10, label="gaze(end)", zorder=17)
    target_circle = plt.Circle(target[-1], trad, fill=False, linewidth=0.5, color='gray')
    axs[0].add_patch(target_circle)
    axs[0].legend()
    fig_axes_setting(axs[0])

    t2x = np.linalg.norm(target - CROSSHAIR, axis=1)

    axs[1].plot(timestamp, t2x, color='k', linewidth=0.8, label="Target-Crosshair", zorder=15)
    if len(gaze) > 0:
        # axs[1].plot(timestamp, np.linalg.norm(target - gaze, axis=1), color='g', linewidth=0.8, label="Target-Gaze", zorder=15)
        g2x = np.linalg.norm(gaze - gaze[0], axis=1)
        axs[1].scatter(timestamp[gv==1], g2x[gv==1], s=0.5, color='r', label="Gaze-Init.Gp", zorder=15)
        axs[1].scatter(timestamp[gv==0], g2x[gv==0], s=0.5, color='b', label="Gaze(Interp)", zorder=14)
        maxd = max(np.max(t2x), np.max(g2x))
    else: maxd = np.max(t2x)
    axs[1].axhline(trad, linestyle='--', linewidth=0.5, color='grey', label="Target radius", zorder=0)
    axs[1].legend()
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_ylabel("Distance (cm)")
    set_tick_and_range(axs[1], 0.1, 1000, timestamp[-1], axis='x')
    set_tick_and_range(axs[1], 0.01, 100, maxd, axis='y')
    axs[1].grid(True, zorder=0)

    fig_save(PATH_VIS_EXP_TRAJ % username, tag, save_svg=False)
    plt.cla()
    plt.clf()
    plt.close('all')


def plt_model_fitting_r2(
    labels, 
    d_true, d_hat, group_index, 
    dir, fn, 
    marker_size=30, marker='x', a=1, 
    save_svg=True
):
    """Draw inferring process"""
    score = []

    # Groupyby mean
    mean_t = np.array([np.mean(d_true[group_index == gi], axis=0) for gi in np.unique(group_index)])
    mean_h = np.array([np.mean(d_hat[group_index == gi], axis=0) for gi in np.unique(group_index)])

    plt.figure(figsize=(1+4.2*len(labels), 3))
    for i in range(len(labels)):
        x = mean_t[:,i]
        y = mean_h[:,i]
        fit = np.polyfit(x, y, 1)
        func = np.poly1d(fit)
        r2 = r2_score(y, func(x))
        score.append(r2)

        plt.subplot(1, len(labels), i+1)
        plt.title(labels[i])
        plt.xlabel("Human")
        if i == 0: plt.ylabel("Simulator")

        min_x, max_x = x.min(), x.max()
        dist_val = (max_x - min_x) * 0.1

        # Baseline
        plt.plot(
            [min_x - dist_val, max_x + dist_val],
            [min_x - dist_val, max_x + dist_val],
            color="gray",
            linestyle="--", linewidth=0.5, zorder=0
        )
        plt.plot(
            [min_x - dist_val, max_x + dist_val],
            func([min_x - dist_val, max_x + dist_val]),
            color='r', linewidth=0.8, zorder=1
        )
        plt.scatter(x, y, marker=marker, s=marker_size, alpha=a, label=f'$R^2$={r2:.2f}', zorder=2)

        plt.xlim(min_x - dist_val, max_x + dist_val)
        #plt.ylim(min_x - dist_val, max_x + dist_val)
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
        plt.legend(loc="lower right")
    
    fig_save(dir, fn, DPI=70, save_svg=save_svg)
    return score



def plot_parameter_recovery(
    p_true,     # np.ndarray with shape (batch_sz, param_sz)
    p_pred,     # np.ndarray with shape (batch_sz, param_sz)
    fname,
    param_labels=None,
    fpath=None,
    label_prefix=['True', 'Inferred']
):
    """
    ##### https://github.com/hsmoon121/amortized-inference-hci
    Plot the comparison between true and inferred parameter values, and return R2 values

    Code modification: now all R2 plots of parameters are drawn in a single figure, and R2 uses scipy library
    """
    plt.rcParams["font.family"] = "sans-serif"
    # plt.rcParams["font.size"] = 18
    # plt.rcParams["axes.linewidth"] = 2
    sns.set_style("white")
    r2_list = []

    # Grid configuration
    _r, _c = find_divisors(p_true.shape[1])
    fig, axs = plt.subplots(_r, _c, figsize=np.array([_c,_r])*3, constrained_layout=True)
    
    for i in range(p_true.shape[1]):
        y_true = p_true[:, i]
        y_pred = p_pred[:, i]
        y_fit = np.polyfit(y_true, y_pred, 1)
        y_func = np.poly1d(y_fit)
        r_squared = get_r_squared(y_true, y_pred)
        r2_list.append(r_squared)

        max_val = max(max(y_true), max(y_pred))
        min_val = min(min(y_true), min(y_pred))
        dist_val = (max_val - min_val) * 0.1

        label = f"$({y_fit[0]:.2f})x + ({y_fit[1]:.2f}), R^2={r_squared:.2f}$"

        r, c = i // _c, i % _c
        ax = axs[r][c] if _r > 1 else axs[c]

        sns.regplot(
            x=y_true,
            y=y_pred,
            scatter_kws={"color": "black", "alpha": 0.3},
            line_kws={"color": "red", "lw": 2.5},
            ax=ax
        )
        ax.plot(
            [min_val - dist_val, max_val + dist_val],
            [min_val - dist_val, max_val + dist_val],
            color="gray",
            linestyle="--"
        )
        ax.set_ylabel(f"{label_prefix[1]} {param_labels[i]}")
        ax.set_xlabel(f"{label_prefix[0]} {param_labels[i]}")
        
        ax.set_xlim([min_val - dist_val, max_val + dist_val])
        ax.set_ylim([min_val - dist_val, max_val + dist_val])
        ax.legend([Line2D([0], [0], color="red", lw=2.5)], [label,], fontsize=8, loc="lower right")
        ax.grid(linestyle="--", linewidth=0.5)
        ax.set_aspect("equal")

    fig_save(fpath, fname, DPI=300, save_svg=True)
    plt.close(fig)

    return r2_list


def plot_recovery_r2(
    n_trial,
    r2_record,
    fname,
    param_labels=None,
    fpath=None
):
    plt.figure(figsize=(6, 6))
    for i in range(r2_record.shape[1]):
        plt.plot(n_trial, r2_record[:,i], label=param_labels[i], marker='o', markersize=2)
    plt.ylim(0, 1)
    plt.xlabel("No. of observed trials")
    plt.ylabel(f"$R^2$")
    plt.legend()
    fig_save(fpath, fname, DPI=300, save_svg=True)
    plt.close('all')



def plot_stat_comparison(
    stat, fname,
    metric, 
    maximum_values=None,
    metric_units=None,
    metric_scales=None,
    metric_labels=None, 
    metric_descriptions=None,
    fpath=None
):
    df = pd.DataFrame(dict(zip(
        [f"p_{metric_labels[m]}" for m in metric] + \
            [f"s_{metric_labels[m]}" for m in metric] + \
                ["binning_p", "binning_tiod", "binning_tc", "binning_ts"],
        stat.T
    )))
    # Plot 1. Scatterplot: binned by player
    r2_player_list = list()
    fig, axs = plt.subplots(1, len(metric), figsize=np.array([len(metric),1])*3, constrained_layout=True)
    df_mean = df.groupby(["binning_p"], as_index=False).mean()
    for i, m in enumerate(metric):
        y_true = df_mean[f"p_{metric_labels[m]}"].to_numpy()
        y_pred = df_mean[f"s_{metric_labels[m]}"].to_numpy()
        r_squared = get_r_squared(y_true, y_pred)
        r2_player_list.append(r_squared)

        max_val = max(max(y_true), max(y_pred))
        min_val = min(min(y_true), min(y_pred))
        dist_val = (max_val - min_val) * 0.1

        label = f"$R^2={r_squared:.2f}$"

        sns.regplot(
            x=y_true,
            y=y_pred,
            scatter_kws={"color": "black", "alpha": 0},
            line_kws={"color": "red", "lw": 2.5},
            ax=axs[i]
        )
        # AMA - PRO order
        ama_points = axs[i].scatter(y_true[:10], y_pred[:10], marker='s', color='blue', s=40, alpha=0.5, zorder=100)
        pro_points = axs[i].scatter(y_true[10:], y_pred[10:], marker='D', color='green', s=40, alpha=0.5, zorder=100)

        rng, ticks, ticks_lb = auto_set_tick_and_range(
            max_val, min_val, metric_scales[m], metric_units[m],
            offset=(0.1, 0.1), fmt='%d'
        )
        axs[i].plot(
            [min_val - dist_val, max_val + dist_val],
            [min_val - dist_val, max_val + dist_val],
            color="gray",
            linestyle="--"
        )
        axs[i].set_title(metric_descriptions[m])
        if i == 0:
            axs[i].set_ylabel("Simulator")
            axs[i].set_xlabel("Player")
        
        axs[i].set_xlim(*rng)
        axs[i].set_ylim(*rng)
        axs[i].set_xticks(ticks, ticks_lb); axs[i].set_yticks(ticks, ticks_lb)
        if i == len(metric) - 1:
            axs[i].legend(
                [pro_points, ama_points, Line2D([0], [0], color="red", lw=2.5)], 
                ["Pro.", "Ama.", label], 
                fontsize=8, loc="lower right"
            )
        else:
            axs[i].legend(
                [Line2D([0], [0], color="red", lw=2.5)], 
                [label,], 
                fontsize=8, loc="lower right"
            )
        axs[i].grid(linestyle="--", linewidth=0.5)
        axs[i].set_aspect("equal")
    fig_save(fpath, f"{fname}_scatter_player", DPI=300, save_svg=True)
    plt.close(fig)


    # Plot 2. Scatterplot: binned by target
    r2_target_list = list()
    fig, axs = plt.subplots(1, len(metric), figsize=np.array([len(metric),1])*3, constrained_layout=True)
    df["binning_tiod_disc"] = discrete_labeling(df["binning_tiod"].to_numpy(), lvl=5)
    df_mean = df.groupby(
        ["binning_tiod_disc", "binning_tc", "binning_ts"], 
        as_index=False
    ).mean()
    tc_cond = df_mean["binning_tc"].to_numpy()
    ts_cond = df_mean["binning_ts"].to_numpy()
    for i, m in enumerate(metric):
        y_true = df_mean[f"p_{metric_labels[m]}"].to_numpy()
        y_pred = df_mean[f"s_{metric_labels[m]}"].to_numpy()
        r_squared = get_r_squared(y_true, y_pred)
        r2_target_list.append(r_squared)

        max_val = max(max(y_true), max(y_pred))
        min_val = min(min(y_true), min(y_pred))
        dist_val = (max_val - min_val) * 0.1

        label = f"$R^2={r_squared:.2f}$"

        sns.regplot(
            x=y_true,
            y=y_pred,
            scatter_kws={"color": "black", "alpha": 0},
            line_kws={"color": "red", "lw": 2.5},
            ax=axs[i]
        )
        # Use different marker/color by target condition
        ws_points = axs[i].scatter(
            y_true[(tc_cond == 0) & (ts_cond == 0)], 
            y_pred[(tc_cond == 0) & (ts_cond == 0)], 
            marker='o', edgecolor='black', linewidth=1, facecolor='white', s=40, alpha=0.5, zorder=100
        )
        gs_points = axs[i].scatter(
            y_true[(tc_cond == 1) & (ts_cond == 0)], 
            y_pred[(tc_cond == 1) & (ts_cond == 0)], 
            marker='o', edgecolor='black', linewidth=1, facecolor='gray', s=40, alpha=0.5, zorder=100
        )
        wm_points = axs[i].scatter(
            y_true[(tc_cond == 0) & (ts_cond == 1)], 
            y_pred[(tc_cond == 0) & (ts_cond == 1)], 
            marker='D', edgecolor='black', linewidth=1, facecolor='white', s=40, alpha=0.5, zorder=100
        )
        gm_points = axs[i].scatter(
            y_true[(tc_cond == 1) & (ts_cond == 1)], 
            y_pred[(tc_cond == 1) & (ts_cond == 1)], 
            marker='D', edgecolor='black', linewidth=1, facecolor='gray', s=40, alpha=0.5, zorder=100
        )

        rng, ticks, ticks_lb = auto_set_tick_and_range(
            max_val, min_val, metric_scales[m], metric_units[m],
            offset=(0.1, 0.1), fmt='%d'
        )
        axs[i].plot(
            [min_val - dist_val, max_val + dist_val],
            [min_val - dist_val, max_val + dist_val],
            color="gray",
            linestyle="--"
        )
        axs[i].set_title(metric_descriptions[m])
        if i == 0:
            axs[i].set_ylabel("Simulator")
            axs[i].set_xlabel("Player")
        
        axs[i].set_xlim(*rng)
        axs[i].set_ylim(*rng)
        axs[i].set_xticks(ticks, ticks_lb); axs[i].set_yticks(ticks, ticks_lb)
        if i == len(metric) - 1:
            axs[i].legend(
                [ws_points, gs_points, wm_points, gm_points, Line2D([0], [0], color="red", lw=2.5)], 
                ["white-stat.", "gray-stat.", "white-moving", "gray-moving", label], 
                fontsize=8, loc="lower right"
            )
        else:
            axs[i].legend(
                [Line2D([0], [0], color="red", lw=2.5)], 
                [label,], 
                fontsize=8, loc="lower right"
            )
        axs[i].grid(linestyle="--", linewidth=0.5)
        axs[i].set_aspect("equal")
    fig_save(fpath, f"{fname}_scatter_target", DPI=300, save_svg=True)
    plt.close(fig)
    

    # Plot 3: distribution
    kl_list = list()
    fig, axs = plt.subplots(1, len(metric), figsize=np.array([len(metric), 0.7])*3, constrained_layout=True)
    for i, m in enumerate(metric):
        y_true = df[f"p_{metric_labels[m]}"].to_numpy()
        y_pred = df[f"s_{metric_labels[m]}"].to_numpy()
        # max_val = max(y_true.max(), y_pred.max(), maximum_values[m])
        max_val = maximum_values[m]
        dist_val = max_val * 0.1

        axs[i].hist(
            y_true, range=(0, max_val), bins=15, 
            density=True, histtype='step', linewidth=1.5,
            color='blue', label='Experiment', zorder=10
        )
        axs[i].hist(
            y_pred, range=(0, max_val), bins=15, 
            density=True, histtype='step', linewidth=1.5,
            color='red', label='Simulation', zorder=9
        )
        rng, ticks, ticks_lb = auto_set_tick_and_range(
            max_val, 0, metric_scales[m], metric_units[m],
            offset=(0.1, 0.1), fmt='%d'
        )
        if len(ticks) > 5:
            ticks = ticks[::int(len(ticks)/5)]
            ticks_lb = ticks_lb[::int(len(ticks_lb)/5)]
        axs[i].set_xlim(*rng)
        axs[i].set_xticks(ticks, ticks_lb)
        axs[i].set_xlabel(metric_descriptions[m])
        if i == len(metric)-1: axs[i].legend()

        kl_list.append(hist_kld(y_true, y_pred))
    
    fig_save(fpath, f"{fname}_hist", DPI=300, save_svg=True)
    plt.close(fig)


    # Plot 4: Fitts' Law
    # There are 20 players
    # Upper 2 rows are Professional
    # Bottom 2 rows are amateurs
    fig, axs = plt.subplots(4, 5, figsize=np.array([5, 4])*3, constrained_layout=True)
    df["binning_tiod_disc"] = discrete_labeling(df["binning_tiod"].to_numpy(), lvl=7)
    for tn, tier in enumerate(TIER):
        for pn, player in enumerate(PLAYER[tier]):
            _df = df[
                (df["binning_p"] == PLAYERS.index(player)) &
                (df["binning_ts"] == 0)
            ]
            r = 2*tn + (pn // 5)
            c = pn % 5
            ax = axs[r][c]

            ax.set_title(["Professional", "Amateur"][tn] + f" {pn+1:02d}")
            if r == 3 and c == 0:
                ax.set_xlabel("Index of Difficulty (bit)")
                ax.set_ylabel("Trial completion time (ms)")

            fl_r2 = list()

            for ib, b in enumerate("ps"):  # player or simulation
                b_df = _df[[f"{b}_tct", f"{b}_acc", "binning_tiod", "binning_tiod_disc", "binning_tc"]]
                b_df = b_df[b_df[f"{b}_acc"] == 1]
                for tc in range(2):     # target color
                    _b_df = b_df[b_df["binning_tc"] == tc]
                    _b_df = _b_df.groupby(["binning_tiod_disc"], as_index=False).mean()

                    iod = _b_df["binning_tiod"].to_numpy()
                    tct = _b_df[f"{b}_tct"].to_numpy()
                    r_squared = get_r_squared(iod, tct)
                    fl_r2.append(r_squared)

                    sns.regplot(
                        x=iod,
                        y=tct,
                        scatter_kws={"color": "black", "alpha": 0},
                        line_kws={"color": [["coral","royalblue"],["sienna","steelblue"]][ib][tc], "lw": 1.5},
                        ax=ax
                    )
                    ax.scatter(
                        iod, tct, 
                        marker='o^'[ib], edgecolor='black',
                        linewidth=1, facecolor=['white', 'gray'][tc],
                        s=25, alpha=0.5, zorder=100
                    )
                    ax.set_xlim(1, 5.5)
                    ax.set_ylim(0.3, 1.0)
                    ax.set_yticks([0.1*t for t in range(3, 11)], [100*t for t in range(3, 11)])
                    ax.set_xticks([1, 2, 3, 4, 5])

            if r == 3 and c == 0:
                ax.legend(
                    [
                        Line2D([0], [0], color="coral", marker='o', markersize=7, markeredgecolor='black', markerfacecolor=colors.to_rgba('white', 0.5), lw=1.5),
                        Line2D([0], [0], color="royalblue", marker='o', markersize=7, markeredgecolor='black', markerfacecolor=colors.to_rgba('gray', 0.5), lw=1.5),
                        Line2D([0], [0], color="sienna", marker='^', markersize=7, markeredgecolor='black', markerfacecolor=colors.to_rgba('white', 0.5), lw=1.5),
                        Line2D([0], [0], color="steelblue", marker='^', markersize=7, markeredgecolor='black', markerfacecolor=colors.to_rgba('gray', 0.5), lw=1.5),
                    ], 
                    [
                        f"player-white ($R^2$={fl_r2[0]:.2f})",
                        f"player-gray ($R^2$={fl_r2[1]:.2f})",
                        f"simul.-white ($R^2$={fl_r2[2]:.2f})",
                        f"simul.-gray ($R^2$={fl_r2[3]:.2f})",
                    ], 
                    fontsize=7, loc="lower right"
                )
            else:
                ax.legend(
                    [
                        Line2D([0], [0], color="coral", marker='o', markersize=7, markeredgecolor='black', markerfacecolor=colors.to_rgba('white', 0.5), lw=1.5),
                        Line2D([0], [0], color="royalblue", marker='o', markersize=7, markeredgecolor='black', markerfacecolor=colors.to_rgba('gray', 0.5), lw=1.5),
                        Line2D([0], [0], color="sienna", marker='^', markersize=7, markeredgecolor='black', markerfacecolor=colors.to_rgba('white', 0.5), lw=1.5),
                        Line2D([0], [0], color="steelblue", marker='^', markersize=7, markeredgecolor='black', markerfacecolor=colors.to_rgba('gray', 0.5), lw=1.5),
                    ], 
                    [
                        f"$R^2$={fl_r2[0]:.2f}",
                        f"$R^2$={fl_r2[1]:.2f}",
                        f"$R^2$={fl_r2[2]:.2f}",
                        f"$R^2$={fl_r2[3]:.2f}",
                    ], 
                    fontsize=7, loc="lower right"
                )
        
    fig_save(fpath, f"{fname}_fitts", DPI=300, save_svg=True)
    plt.close(fig)

    return r2_player_list, r2_target_list, kl_list


def plot_distance_comparison(
    exp_dist,
    sim_dist,
    fname,
    fpath=None
):
    # There are 20 players
    # Upper 2 rows are Professional
    # Bottom 2 rows are amateurs
    fig, axs = plt.subplots(4, 5, figsize=np.array([5, 2.8])*3, constrained_layout=True)

    for tn, tier in enumerate(TIER):
        for pn, player in enumerate(PLAYER[tier]):
            p_ts, p_tx, p_gx = (
                exp_dist[player]["default"][1]["ts"],
                exp_dist[player]["default"][1]["tx"],
                exp_dist[player]["default"][1]["gx"]
            )
            s_ts, s_tx, s_gx = (
                sim_dist[player]["ts"],
                sim_dist[player]["tx"],
                sim_dist[player]["gx"]
            )

            r = 2*tn + (pn // 5)
            c = pn % 5

            axs[r][c].set_title(["Professional", "Amateur"][tn] + f" {pn+1:02d}")
            axs[r][c].plot(p_ts, p_tx, linestyle='-', linewidth=0.8, color='k', label='T-X (Exp.)', zorder=30)
            axs[r][c].plot(p_ts, p_gx, linestyle='--', linewidth=0.8, color='k', label='G-X (Exp.)', zorder=30)
            axs[r][c].plot(s_ts, s_tx, linestyle='-', linewidth=0.8, color='r', label='T-X (Sim.)', zorder=30)
            axs[r][c].plot(s_ts, s_gx, linestyle='--', linewidth=0.8, color='r', label='G-X (Sim.)', zorder=30)
            set_tick_and_range(axs[r][c], 0.1, 1000, max_value=1, axis='x', omit_tick=2)
            set_tick_and_range(axs[r][c], 0.02, 100, max_value=0.1, axis='y')
            axs[r][c].grid()
            if r == 3 and c == 0:
                axs[r][c].set_xlabel("Time (ms)")
                axs[r][c].set_ylabel("Distance (cm)")
                axs[r][c].legend()

    fig_save(fpath, fname, DPI=300, save_svg=True)
    plt.close(fig)


def plot_compare_inferred_param_by_tier(
    inferred_param,
    fname,
    param_labels=None,
    fpath=None
):
    F_value, p_value = list(), list()

    params = {tier: None for tier in TIER}
    for tier in TIER:
        _params = list()
        for player in PLAYER[tier]:
            for color in COLOR:
                _params.append(inferred_param[player][color])
        params[tier] = np.array(_params)
    
    _r, _c = find_divisors(params["PRO"].shape[1])
    fig, axs = plt.subplots(_r, _c, figsize=np.array([_c/2,_r])*3, constrained_layout=True)

    for i in range(params["PRO"].shape[1]):
        _F, _p = f_oneway(params["PRO"][:,i], params["AMA"][:,i])
        F_value.append(_F)
        p_value.append(_p)

        max_v, min_v = -999, 999

        r, c = i // _c, i % _c
        ax = axs[r][c] if _r > 1 else axs[c]

        for j, tier in enumerate(TIER):
            data = params[tier][:, i]
            _m, _er = np.mean(data), cl_95_intv(data)
            ax.bar(
                -0.5 + j,
                _m,
                yerr=_er,
                color='gray',
                capsize=5,
                zorder=10
            )
            if _m + _er > max_v: max_v = _m + _er
            if _m - _er < min_v: min_v = _m - _er

        ax.set_xticks([-0.5, 0.5], TIER)
        ax.set_xlim(-1, 1)

        d_val = (max_v - min_v) * 0.1
        # ax.set_ylim(max(0, min_v - d_val), max_v + d_val)
        ax.set_ylim(min_v - d_val, max_v + d_val)
        ax.set_title(f"{param_labels[i]}, p={_p:.3f}")
        ax.grid(axis='y')

    fig_save(fpath, fname, DPI=300, save_svg=True)
    plt.close(fig)

    return p_value


def plot_compare_inferred_param_by_color(
    inferred_param,
    fname,
    param_labels=None,
    fpath=None
):
    F_value, p_value = list(), list()

    params = {color: None for color in COLOR}
    for color in COLOR:
        _params = list()
        for player in PLAYERS:
            _params.append(inferred_param[player][color])
        params[color] = np.array(_params)
    
    _r, _c = find_divisors(params["white"].shape[1])
    fig, axs = plt.subplots(_r, _c, figsize=np.array([_c/2,_r])*3, constrained_layout=True)

    for i in range(params["white"].shape[1]):
        _F, _p = f_oneway(params["white"][:,i], params["gray"][:,i])
        F_value.append(_F)
        p_value.append(_p)

        max_v, min_v = -999, 999

        r, c = i // _c, i % _c
        ax = axs[r][c] if _r > 1 else axs[c]

        for j, color in enumerate(COLOR):
            data = params[color][:, i]
            _m, _er = np.mean(data), cl_95_intv(data)
            ax.bar(
                -0.5 + j,
                _m,
                yerr=_er,
                color='gray',
                capsize=5,
                zorder=10
            )
            if _m + _er > max_v: max_v = _m + _er
            if _m - _er < min_v: min_v = _m - _er

        ax.set_xticks([-0.5, 0.5], COLOR)
        ax.set_xlim(-1, 1)

        d_val = (max_v - min_v) * 0.1
        # ax.set_ylim(max(0, min_v - d_val), max_v + d_val)
        ax.set_ylim(min_v - d_val, max_v + d_val)
        ax.set_title(f"{param_labels[i]}, p={_p:.3f}")
        ax.grid(axis='y')

    fig_save(fpath, fname, DPI=300, save_svg=True)
    plt.close(fig)

    return p_value



if __name__ == "__main__":
    pass