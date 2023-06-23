import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import os, sys

from sklearn.metrics import r2_score
from matplotlib.ticker import StrMethodFormatter

sys.path.append("..")
from configs.common import *
from configs.path import *

def fig_save(dir, fn, DPI=100, save_svg=False):
    os.makedirs(dir, exist_ok=True)
    if dir[-1] == '/':
        fullpath = f"{dir}{fn}"
    else:
        fullpath = f"{dir}/{fn}"
    ext = 'svg' if save_svg else 'png'
    plt.savefig(f"{fullpath}.{ext}", dpi=DPI, bbox_inches='tight', pad_inches=0)
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


def set_tick_and_range(axs, unit, tick_scale, max_value, min_value=0, axis='x'):
    r_min = unit * int(np.floor(min_value / unit))
    r_max = unit * int(np.ceil(max_value / unit))
    rng = (r_min, r_max)
    tick_v = np.linspace(r_min, r_max, int((r_max - r_min) / unit) + 1)
    tick_n = list(map(int, tick_v * tick_scale))

    if axis == 'x':
        axs.set_xlim(*rng)
        axs.set_xticks(tick_v, tick_n)
    elif axis == 'y':
        axs.set_ylim(*rng)
        axs.set_yticks(tick_v, tick_n)


def plt_trajectory(timestamp, target, gaze, trad, username, tag):
    """Draw trajectory and distance plot"""
    fig, axs = plt.subplots(2, 1, figsize=(8, 8.5), gridspec_kw={"height_ratios":[1, 1]})

    axs[0].set_title("Traj and Dist (%s, %s)" % (username, tag))
    axs[0].scatter(*target.T, marker='.', s=0.5, color='k', label="target", zorder=15)
    if len(gaze) > 0:
        axs[0].scatter(*gaze.T, marker='.', s=0.5, color='r', label="gaze", zorder=16)
        axs[0].scatter(*gaze[0], color='r', marker='o', s=10, label="gaze(start)", zorder=17)
        axs[0].scatter(*gaze[-1], color='r', marker='x', s=10, label="gaze(end)", zorder=17)
    target_circle = plt.Circle(target[-1], trad, fill=False, linewidth=0.5, color='gray')
    axs[0].add_patch(target_circle)
    axs[0].legend()
    fig_axes_setting(axs[0])

    axs[1].plot(timestamp, np.linalg.norm(target - CROSSHAIR, axis=1), color='k', linewidth=0.8, label="Target-Crosshair", zorder=15)
    if len(gaze) > 0:
        axs[1].plot(timestamp, np.linalg.norm(target - gaze, axis=1), color='g', linewidth=0.8, label="Target-Gaze", zorder=15)
        axs[1].plot(timestamp, np.linalg.norm(gaze - gaze[0], axis=1), color='r', linewidth=0.8, label="Gaze-Crosshair", zorder=15)
    axs[1].axhline(trad, linestyle='--', linewidth=0.5, color='grey', label="Target radius", zorder=0)
    axs[1].legend()
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_ylabel("Distance (cm)")
    set_tick_and_range(axs[1], 0.1, 1000, timestamp[-1], axis='x')
    set_tick_and_range(axs[1], 0.01, 100, np.linalg.norm(target - CROSSHAIR, axis=1).max(), axis='y')
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



if __name__ == "__main__":
    pass