import numpy as np
import pandas as pd

import copy
import seaborn as sns
from torch._C import _get_float32_matmul_precision
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import f_oneway
from joblib import Parallel, delayed

import sys, os

sys.path.append('..')

from experiment.data_process import *
from configs.path import *
from configs.experiment import *
from configs.simulation import *
from configs.amort import *
from utils.mymath import *
from utils.utils import *
from utils.plots import *

#### infer split: task cond

block_list = dict(
    all = [0, 1],
    half = [1]
)
ses_list = dict(
    all = SES_NAME_ABBR,
    sub = SES_NAME_ABBR_SYMM
)

# Experiment data load
try:
    exp = pickle_load(f"{PATH_DATA_SUMMARY}experiment_result_dict.pkl")
except:
    exp = {
        key_block: {
            key_sess: {
                player: {
                    mode: {
                        color: load_experiment(
                            player=player,
                            mode=mode,
                            target_color=color,
                            block_index=block_list[key_block],
                            session_id=ses_list[key_sess]
                        ) for color in COLOR
                    } for mode in MODE
                } for player in tqdm(PLAYERS)
            } for key_sess in ["all", "sub"]
        } for key_block in ["all", "half"]
    }
    pickle_save(f"{PATH_DATA_SUMMARY}experiment_result_dict.pkl", exp)

exp_dist = player_distance_info_amort_eval()


metric_list = dict(
    full = [M_TCT, M_ACC, M_GD],
    base = [M_TCT, M_ACC],
    abl0 = [M_TCT, M_ACC, M_GD],
    abl1 = [M_TCT, M_ACC, M_GD],
    abl2 = [M_TCT, M_ACC, M_GD],
    abl3 = [M_TCT, M_ACC, M_GD],
)
metric_unit = {
    M_TCT : 0.1,
    M_ACC : 0.1,
    M_GD : 2,
    M_SE_NORM : 1,
}
metric_unit2 = {
    M_TCT : 0.15,
    M_ACC : 0.2,
    M_GD : 2,
    M_SE_NORM : 1,
}
metric_scale = {
    M_TCT : 1000,
    M_ACC : 100,
    M_GD : 1,
    M_SE_NORM : 1,
}
metric_description = {
    M_TCT : "Trial completion time (ms)",
    M_ACC : "Accuracy (%)",
    M_GD : f"Glancing distance ($^\circ$)",
    M_SE_NORM : "Normalized shoot error"
}
metric_unit_str = {
    M_TCT : "(ms)",
    M_ACC : "(%)",
    M_GD : f"($^\circ$)",
    M_SE_NORM : ""
}
metric_description2 = {
    M_TCT : "Trial completion time of %s",
    M_ACC : "Accuracy of %s",
    M_GD : "Glancing distance of %s",
    M_SE_NORM : "Normalized shoot error of %s"
}
metric_max = {
    M_TCT : 1.5,
    M_ACC : 1,
    M_GD : 20,
    M_SE_NORM : 5,
}
metric_max2 = {
    M_TCT : 1.15,
    M_ACC : 1,
    M_GD : 19,
    M_SE_NORM : 3.9,
}
metric_hist_unit = {
    M_TCT : 0.3,
    M_ACC : 1,
    M_GD : 5,
    M_SE_NORM : 1,
}

prefix_block = {
    "all": "fullblock",
    "half": "latterblock",
}
prefix_session = {
    "all": "fullsession",
    "sub": "subsession"
}


plt.rcParams["font.family"] = "sans-serif"
sns.set_style("white")


def load_param_list(model:str):
    mode = model.split('_')[0]
    return default_ans_config[mode]["simulator"]["targeted_params"]


def load_param(model, session, iter, block, target_sess):
    return pd.read_csv(f"{PATH_AMORT_EVAL % (model, session, iter)}/{prefix_block[block]}_{prefix_session[target_sess]}/infer_w.csv")


def simul_data(model, session, iter, block, target_sess):
    return {
        player: pickle_load(f"{PATH_AMORT_EVAL % (model, session, iter)}/{prefix_block[block]}_{prefix_session[target_sess]}/best_simul/{player}.pkl") for player in PLAYERS
    }


def anova_tier(model, session, iter, block, target_sess):
    param_list = load_param_list(model)
    param = load_param(model, session, iter, block, target_sess)
    symbol = [PARAM_SYMBOL[v] for v in param_list]
    names = [PARAM_NAME[v] for v in param_list]

    _r, _c = find_divisors(len(param_list))

    # Draw ANOVA bar graph - tier
    fig, axs = plt.subplots(_r, _c, figsize=np.array([_c*1.4, _r*2.1]), constrained_layout=True)
    for i, theta in enumerate(param_list):
        data = param[[f"{mode[0]}_{color}_{theta}" for mode in MODE for color in COLOR]].to_numpy()
        data = {
            "P": [np.mean(data[:10]), np.std(data[:10]), 1.96 * np.std(data[:10]) / np.sqrt(40)],
            "A": [np.mean(data[10:]), np.std(data[10:]), 1.96 * np.std(data[10:]) / np.sqrt(40)]
        }

        max_v = max(data["P"][0] + data["P"][2], data["A"][0] + data["A"][2])
        min_v = min(data["P"][0] - data["P"][2], data["A"][0] - data["A"][2])
        d_val = (max_v - min_v)

        ax = axs[i//_c][i%_c] if _r > 1 else axs[i]
        ax.scatter([-1], [data["P"][0]], color='k', marker='s', s=15, zorder=20)
        ax.errorbar([-1], [data["P"][0]], yerr=data["P"][2], color='gray', capsize=3.5, zorder=10)
        ax.scatter([1], [data["A"][0]], color='k', marker='s', s=15, zorder=20)
        ax.errorbar([1], [data["A"][0]], yerr=data["A"][2], color='gray', capsize=3.5, zorder=10)

        ax.set_xlim(-1.75, 1.75)
        ax.set_xticks([-1, 1], ["Pro.", "Ama."])
        ax.set_ylim(min_v - d_val * 0.1, max_v + d_val * 0.2)
        
        ax.set_ylabel(f"{names[i]} ({symbol[i]})")
        ax.grid(axis='y', linestyle='--')

    fig_save(f"{PATH_VIS_AMORT_RESULT}{model}/{session}/{iter}/{prefix_block[block]}_{prefix_session[target_sess]}", "parameter_tier", DPI=300, save_svg=True)


def anova_sensitivity(model, session, iter, block, target_sess):
    param_list = load_param_list(model)
    param = load_param(model, session, iter, block, target_sess)
    symbol = [PARAM_SYMBOL[v] for v in param_list]
    names = [PARAM_NAME[v] for v in param_list]

    _r, _c = find_divisors(len(param_list))

    # Draw ANOVA bar graph - csensitivity
    fig, axs = plt.subplots(_r, _c, figsize=np.array([_c*1.4, _r*2.1]), constrained_layout=True)
    for i, theta in enumerate(param_list):
        data = param[[f"{mode[0]}_{color}_{theta}" for mode in MODE for color in COLOR]].to_numpy()
        data = {
            "C": [np.mean(data[:,0:2]), np.std(data[:,0:2]), 1.96 * np.std(data[:,0:2]) / np.sqrt(40)],
            "D": [np.mean(data[:,2:]), np.std(data[:,2:]), 1.96 * np.std(data[:,2:]) / np.sqrt(40)]
        }

        max_v = max(data["C"][0] + data["C"][2], data["D"][0] + data["D"][2])
        min_v = min(data["C"][0] - data["C"][2], data["D"][0] - data["D"][2])
        d_val = (max_v - min_v)

        ax = axs[i//_c][i%_c] if _r > 1 else axs[i]
        ax.scatter([-1], [data["C"][0]], color='k', marker='s', s=15, zorder=20)
        ax.errorbar([-1], [data["C"][0]], yerr=data["C"][2], color='gray', capsize=3.5, zorder=10)
        ax.scatter([1], [data["D"][0]], color='k', marker='s', s=15, zorder=20)
        ax.errorbar([1], [data["D"][0]], yerr=data["D"][2], color='gray', capsize=3.5, zorder=10)

        ax.set_xlim(-1.75, 1.75)
        ax.set_xticks([-1, 1], ["Hab.", "Def."])
        ax.set_ylim(min_v - d_val * 0.1, max_v + d_val * 0.2)
        
        ax.set_ylabel(f"{names[i]} ({symbol[i]})")
        ax.grid(axis='y', linestyle='--')

    fig_save(f"{PATH_VIS_AMORT_RESULT}{model}/{session}/{iter}/{prefix_block[block]}_{prefix_session[target_sess]}", "parameter_sensi", DPI=300, save_svg=True)


def anova_color(model, session, iter, block, target_sess):
    param_list = load_param_list(model)
    param = load_param(model, session, iter, block, target_sess)
    symbol = [PARAM_SYMBOL[v] for v in param_list]
    names = [PARAM_NAME[v] for v in param_list]

    _r, _c = find_divisors(len(param_list))

    # Draw ANOVA bar graph - color
    fig, axs = plt.subplots(_r, _c, figsize=np.array([_c*1.4, _r*2.1]), constrained_layout=True)
    for i, theta in enumerate(param_list):
        data = param[[f"{mode[0]}_{color}_{theta}" for color in COLOR for mode in MODE]].to_numpy()
        data = {
            "W": [np.mean(data[:,0:2]), np.std(data[:,0:2]), 1.96 * np.std(data[:,0:2]) / np.sqrt(40)],
            "G": [np.mean(data[:,2:]), np.std(data[:,2:]), 1.96 * np.std(data[:,2:]) / np.sqrt(40)]
        }

        max_v = max(data["W"][0] + data["W"][2], data["G"][0] + data["G"][2])
        min_v = min(data["W"][0] - data["W"][2], data["G"][0] - data["G"][2])
        d_val = (max_v - min_v)

        ax = axs[i//_c][i%_c] if _r > 1 else axs[i]
        ax.scatter([-1], [data["W"][0]], color='k', marker='s', s=15, zorder=20)
        ax.errorbar([-1], [data["W"][0]], yerr=data["W"][2], color='gray', capsize=3.5, zorder=10)
        ax.scatter([1], [data["G"][0]], color='k', marker='s', s=15, zorder=20)
        ax.errorbar([1], [data["G"][0]], yerr=data["G"][2], color='gray', capsize=3.5, zorder=10)

        ax.set_xlim(-1.75, 1.75)
        ax.set_xticks([-1, 1], ["White", "Gray"])
        ax.set_ylim(min_v - d_val * 0.1, max_v + d_val * 0.2)
        
        ax.set_ylabel(f"{names[i]} ({symbol[i]})")
        ax.grid(axis='y', linestyle='--')

    fig_save(f"{PATH_VIS_AMORT_RESULT}{model}/{session}/{iter}/{prefix_block[block]}_{prefix_session[target_sess]}", "parameter_color", DPI=300, save_svg=True)




def metric_comp_binning_player(model, session, iter, block, target_sess):

    regcolor = "darkgreen"

    amort_mode = model.split('_')[0]
    metrics = metric_list[amort_mode]

    sim = simul_data(model, session, iter, block, target_sess)

    # Draw scatterplot of metric fitting
    exp_data = []
    sim_data = []
    for tier in TIER:
        for player in PLAYER[tier]:
            temp_exp = []
            temp_sim = []
            for mode in MODE:
                for color in COLOR:
                    temp_exp.append(exp[block][target_sess][player][mode][color][metrics].to_numpy())
                    temp_sim.append(sim[player][mode][color]["df"][metrics].to_numpy())
            exp_data.append(np.concatenate(temp_exp, axis=0).mean(axis=0))
            sim_data.append(np.concatenate(temp_sim, axis=0).mean(axis=0))

    exp_data = np.array(exp_data)
    sim_data = np.array(sim_data)

    fig, axs = plt.subplots(1, len(metrics), figsize=np.array([3*len(metrics), 3]), constrained_layout=True)
    for i, m in enumerate(metrics):
        r_squared = get_r_squared(exp_data[:,i], sim_data[:,i])
        max_v = max(max(exp_data[:,i]), max(sim_data[:,i]))
        min_v = min(min(exp_data[:,i]), min(sim_data[:,i]))
        d_val = (max_v - min_v) * 0.1
        label = f"$R^2={r_squared:.2f}$"

        sns.regplot(
            x=exp_data[:,i],
            y=sim_data[:,i],
            scatter_kws={"color": "black", "alpha": 0},
            line_kws={"color": regcolor, "lw": 2, "alpha": 0.6},
            ax=axs[i],
            ci=None
        )
        axs[i].plot(
            [min_v - d_val, max_v + d_val],
            [min_v - d_val, max_v + d_val],
            color="gray",
            linestyle="--"
        )
        p_pts = axs[i].scatter(
            exp_data[:,i][:10], 
            sim_data[:,i][:10], 
            marker='s', 
            color='royalblue', 
            s=40,
            alpha=0.8, 
            zorder=100
        )
        a_pts = axs[i].scatter(
            exp_data[:,i][10:], 
            sim_data[:,i][10:], 
            marker='D', 
            color='lightcoral', 
            s=40, 
            alpha=0.8, 
            zorder=100
        )
        rng, ticks, ticks_lb = auto_set_tick_and_range(
            max_v, min_v, metric_scale[m], metric_unit[m], offset=(0.1, 0.1), fmt='%d'
        )
        axs[i].set_title(metric_description[m])
        axs[i].set_xlim(*rng)
        axs[i].set_ylim(*rng)
        axs[i].set_xticks(ticks, ticks_lb)
        axs[i].set_yticks(ticks, ticks_lb)

        if i == 0:
            if amort_mode != "base":
                axs[i].set_ylabel("AnS model")
            else:
                axs[i].set_ylabel("Baseline model")
            
            axs[i].set_xlabel("Player")
            axs[i].legend(
                [p_pts, a_pts, Line2D([0], [0], color=regcolor, lw=2, alpha=0.6)],
                ["Professional", "Amateur", label],
                fontsize=8, loc='lower right'
            )
        else:
            axs[i].legend(
                [Line2D([0], [0], color=regcolor, lw=2, alpha=0.6)],
                [label],
                fontsize=8, loc='lower right'
            )
        
        axs[i].grid(linestyle='--', linewidth=0.5)
        axs[i].set_aspect('equal')

    fig_save(
        f"{PATH_VIS_AMORT_RESULT}{model}/{session}/{iter}/{prefix_block[block]}_{prefix_session[target_sess]}",
        "metric_comparison", 
        DPI=300, 
        save_svg=True
    )


def draw_dist(model, session, iter, block, target_sess):
    amort_mode = model.split('_')[0]
    
    sim = simul_data(model, session, iter, block, target_sess)
    sim_dist = {player: dict(ts=None, tx=None, gx=None) for player in PLAYERS}

    for player in PLAYERS:
        ts_list, tx_list, gx_list = list(), list(), list()
        mtct = []
        for mode in MODE:
            for color in COLOR:
                ts_list += sim[player][mode][color]["ts"]
                tx_list += sim[player][mode][color]["tx"]
                gx_list += sim[player][mode][color]["gx"]
                mtct.append(sim[player][mode][color]["df"][M_TCT].to_numpy())
        mtct = np.concatenate(mtct).mean()
        sts, stx, sgx = mean_distance(ts_list, tx_list, gx_list, mtct)
        sim_dist[player]["ts"] = sts
        sim_dist[player]["tx"] = stx
        sim_dist[player]["gx"] = sgx

    fig, axs = plt.subplots(4, 5, figsize=(10, 7), constrained_layout=True)

    for tn, tier in enumerate(TIER):
        for pn, player in enumerate(PLAYER[tier]):
            p_ts, p_tx, p_gx = (
                exp_dist[block][target_sess][player]["ts"],
                exp_dist[block][target_sess][player]["tx"],
                exp_dist[block][target_sess][player]["gx"]
            )
            s_ts, s_tx, s_gx = (
                sim_dist[player]["ts"],
                sim_dist[player]["tx"],
                sim_dist[player]["gx"]
            )

            r = 2*tn + (pn // 5)
            c = pn % 5

            axs[r][c].set_title(["Professional", "Amateur"][tn] + f" {pn+1:02d}")
            axs[r][c].plot(
                p_ts, p_tx, linestyle='-', linewidth=1.0, 
                color=['royalblue', 'lightcoral'][tn], 
                # label='T-X (Exp.)',
                zorder=30
            )
            axs[r][c].plot(
                s_ts, s_tx, linestyle='-', linewidth=1.0, 
                color='k', 
                # label='T-X (Sim.)', 
                zorder=30
            )
            if amort_mode != 'base':
                axs[r][c].plot(p_ts, p_gx, linestyle='--', linewidth=1.0, color=['royalblue', 'lightcoral'][tn], zorder=30)
                axs[r][c].plot(s_ts, s_gx, linestyle='--', linewidth=1.0, color='k', zorder=30)
            set_tick_and_range(axs[r][c], 0.1, 1000, max_value=0.7, axis='x', omit_tick=2, erase_tick=(r!=3))
            set_tick_and_range(axs[r][c], 0.02, 100, max_value=0.1, axis='y', erase_tick=(c!=0))
            axs[r][c].grid(alpha=0.5)
            if r == 3 and c == 0:
                axs[r][c].set_xlabel("Time (ms)")
                axs[r][c].set_ylabel("Distance (cm)")

                model_name = "AnS model" if amort_mode != 'base' else "Baseline model"

                axs[r][c].legend(
                    [
                        Line2D([0], [0], color="royalblue", lw=1.0), 
                        Line2D([0], [0], color="lightcoral", lw=1.0),
                        Line2D([0], [0], color="k", lw=1.0),
                    ], 
                    [
                        "Professional", 
                        "Amateur",
                        model_name,
                    ], 
                    fontsize=8, loc="upper right"
                )

    fig_save(f"{PATH_VIS_AMORT_RESULT}{model}/{session}/{iter}/{prefix_block[block]}_{prefix_session[target_sess]}", f"distance_comp", DPI=300, save_svg=True)


def compute_traj_diff(model, session, iter, block, target_sess):

    dtw_table = {'player': PLAYER["PRO"] + PLAYER["AMA"] + ["M", "SD"], "ttj": [], "gtj": [], "ctj": []}

    sim = simul_data(model, session, iter, block, target_sess)
    dtw_list = dict(ttj=[], gtj=[], ctj=[])

    def get_player_dtw(player):
        player_dtw = dict(ttj=[], gtj=[], ctj=[])
        for mode in MODE:
            for color in COLOR:
                exp_traj = player_trajectory_info(
                    player=player, 
                    mode=mode, 
                    target_color=color, 
                    block_index=block_list[block],
                    session_id=ses_list[target_sess]
                )
                for mt in ['ttj', 'gtj', 'ctj']:
                    for stj, etj in zip(sim[player][mode][color][mt], exp_traj[mt]):
                        player_dtw[mt].append(dtw(stj, etj))
        return player_dtw
    
    eps = Parallel(n_jobs=12)(delayed(get_player_dtw)(player) for player in tqdm(PLAYERS))
    for i, player in enumerate(PLAYERS):
        for mt in ['ttj', 'gtj', 'ctj']:
            dtw_table[mt].append(np.mean(eps[i][mt]))
    
    for mt in ['ttj', 'gtj', 'ctj']:
        dtw_list = np.concatenate([eps[i][mt] for i in range(len(PLAYERS))])
        dtw_table[mt].append(np.mean(dtw_list))
        dtw_table[mt].append(np.std(dtw_list))
    
    pd.DataFrame(dtw_table).to_csv(f"{PATH_VIS_AMORT_RESULT}{model}/{session}/{iter}/{prefix_block[block]}_{prefix_session[target_sess]}/traj_dtw.csv", index=False)



def metric_mean_comp_per_player(model, session, iter, block, target_sess, lvl=4):
    amort_mode = model.split('_')[0]
    metrics = copy.deepcopy(metric_list[amort_mode])
    metrics.append(M_SE_NORM)

    sim = simul_data(model, session, iter, block, target_sess)

    r2_table = {'player': PLAYER["PRO"] + PLAYER["AMA"] + ["M", "SD"]}
    mae_table = {'player': PLAYER["PRO"] + PLAYER["AMA"] + ["M", "SD"]}

    for mt in metrics:
        r2_table[mt] = []
        mae_table[mt] = []

    for i, mt in enumerate(metrics):

        fig, axs = plt.subplots(4, 5, figsize=np.array([5, 4])*2.3, constrained_layout=True)

        max_v = -99999
        min_v = 99999
        mae_list = []

        for tn, tier in enumerate(TIER):
            for pn, player in enumerate(PLAYER[tier]):

                regcolor = ['royalblue', 'lightcoral'][tn]

                r = 2*tn + (pn // 5)
                c = pn % 5

                ax = axs[r][c]

                exp_data = pd.concat([exp[block][target_sess][player][mode][color] for mode in MODE for color in COLOR])
                sim_data = pd.concat([sim[player][mode][color]["df"] for mode in MODE for color in COLOR])

                data = pd.DataFrame(
                    dict(zip(
                        ["session_id", "exp", "sim"],
                        [
                            exp_data["session_id"].to_list(),
                            exp_data[mt].to_numpy(),
                            sim_data[mt].to_numpy()
                        ]
                    ))
                )
                mae_raw = np.abs(data["exp"] - data["sim"])
                mae_list.append(mae_raw)
                mae = mae_raw.mean()

                data = data.groupby(["session_id"], as_index=False).mean()
                
                session_label = data["session_id"].to_list()
                data = data[["exp", "sim"]].to_numpy()

                r_squared = get_r_squared(data[:,0], data[:,1])
                # mae = np.abs(data[:,0] - data[:,1]).mean()

                max_v = max(max(data[:,0]), max(data[:,1]), max_v)
                min_v = min(min(data[:,0]), min(data[:,1]), min_v)
                
                label = f"$R^2={r_squared:.2f}$"

                r2_table[mt].append(r_squared)
                mae_table[mt].append(mae)
                
                sns.regplot(
                    x=data[:,0],
                    y=data[:,1],
                    scatter_kws={"color": "black", "alpha": 0.0},
                    line_kws={"color": regcolor, "lw": 1.5, "zorder": 150},
                    ax=ax,
                    ci=None
                )
                for i, sn in enumerate(session_label):
                    # Speed
                    if sn[0] == 'f': marker = 'D'
                    if sn[0] == 's': marker = 'o'
                    # Color
                    if sn[-1] == 'w': facecolor = 'white'
                    if sn[-1] == 'g': facecolor = 'gray'
                    # Size
                    if sn[1] == 'l': msize = 85
                    if sn[1] == 's': msize = 45
                    if sn[1] == 'v': msize = 20

                    ax.scatter(
                        data[:,0][i],
                        data[:,1][i],
                        marker=marker,
                        edgecolor='black',
                        linewidth=1,
                        facecolor=facecolor,
                        s=msize,
                        alpha=0.8,
                        zorder=100,
                    )

                ax.set_title(["Professional", "Amateur"][tn] + f" {pn+1:02d}")
                ax.legend(
                    [Line2D([0], [0], color=regcolor, lw=1.5)],
                    [label],
                    fontsize=8, loc='lower right'
                )
                ax.grid(linestyle='--', alpha=0.5)
                ax.set_aspect('equal')

        for axx in range(5):
            for axy in range(4):
                ax = axs[axy][axx]
                d_val = (max_v - min_v) * 0.1
                ax.plot(
                    [min_v - d_val, max_v + d_val],
                    [min_v - d_val, max_v + d_val],
                    color="gray",
                    linestyle="--"
                )

                rng, ticks, ticks_lb = auto_set_tick_and_range(
                    max_v, min_v, metric_scale[mt], metric_unit2[mt], offset=(0.1, 0.1), fmt='%d'
                )
                ax.set_xlim(*rng)
                ax.set_ylim(*rng)
                ax.set_xticks(ticks, ticks_lb)
                ax.set_yticks(ticks, ticks_lb)

        model_name = "AnS model" if amort_mode != 'base' else "Baseline model"

        axs[3][2].set_xlabel(metric_description2[mt] % "player" + f" {metric_unit_str[mt]}")
        axs[1][0].set_ylabel(metric_description2[mt] % model_name + f" {metric_unit_str[mt]}")
        fig_save(f"{PATH_VIS_AMORT_RESULT}{model}/{session}/{iter}/{prefix_block[block]}_{prefix_session[target_sess]}", f"comp_binning_{mt}", DPI=300, save_svg=True)

        r2_table[mt].append(np.mean(r2_table[mt]))
        mae_table[mt].append(np.concatenate(mae_list).mean())

        r2_table[mt].append(np.std(r2_table[mt][:20]))
        mae_table[mt].append(np.concatenate(mae_list).std())
    
    pd.DataFrame(r2_table).to_csv(f"{PATH_VIS_AMORT_RESULT}{model}/{session}/{iter}/{prefix_block[block]}_{prefix_session[target_sess]}/comp_binning_r2.csv", index=False)
    pd.DataFrame(mae_table).to_csv(f"{PATH_VIS_AMORT_RESULT}{model}/{session}/{iter}/{prefix_block[block]}_{prefix_session[target_sess]}/comp_mae.csv", index=False)



def metric_mean_comp_per_player_collected(model, session, iter, block, target_sess):

    regcolor = "darkgreen"

    amort_mode = model.split('_')[0]
    metrics = copy.deepcopy(metric_list[amort_mode])

    sim = simul_data(model, session, iter, block, target_sess)

    fig, axs = plt.subplots(1, len(metrics), figsize=np.array([3*len(metrics), 3]), constrained_layout=True)
    for i, mt in enumerate(metrics):

        max_v = -99999
        min_v = 99999

        data_merge = []

        for tn, tier in enumerate(TIER):
            for pn, player in enumerate(PLAYER[tier]):

                markeredgecolor = ['royalblue', 'lightcoral'][tn]

                exp_data = pd.concat([exp[block][target_sess][player][mode][color] for mode in MODE for color in COLOR])
                sim_data = pd.concat([sim[player][mode][color]["df"] for mode in MODE for color in COLOR])

                data = pd.DataFrame(
                    dict(zip(
                        ["session_id", "exp", "sim"],
                        [
                            exp_data["session_id"].to_list(),
                            exp_data[mt].to_numpy(),
                            sim_data[mt].to_numpy()
                        ]
                    ))
                ).groupby(["session_id"], as_index=False).mean()
                
                session_label = data["session_id"].to_list()
                data = data[["exp", "sim"]].to_numpy()

                data_merge.append(data)

                max_v = max(max(data[:,0]), max(data[:,1]), max_v)
                min_v = min(min(data[:,0]), min(data[:,1]), min_v)

                for j, sn in enumerate(session_label):
                    # Speed
                    if sn[0] == 'f': marker = 'D'
                    if sn[0] == 's': marker = 'o'
                    # Color
                    if sn[-1] == 'w': facecolor = 'white'
                    if sn[-1] == 'g': facecolor = 'gray'
                    # Size
                    if sn[1] == 'l': msize = 85
                    if sn[1] == 's': msize = 45
                    if sn[1] == 'v': msize = 20

                    axs[i].scatter(
                        data[:,0][j],
                        data[:,1][j],
                        marker=marker,
                        edgecolor=markeredgecolor,
                        linewidth=1,
                        facecolor=facecolor,
                        s=msize,
                        alpha=0.8,
                        zorder=100,
                    )

        data_merge = np.vstack(data_merge)
        r_squared = get_r_squared(data_merge[:,0], data_merge[:,1])
        d_val = (max_v - min_v) * 0.1
        label = f"$R^2={r_squared:.2f}$"

        sns.regplot(
            x=data_merge[:,0],
            y=data_merge[:,1],
            scatter_kws={"color": "black", "alpha": 0},
            line_kws={"color": regcolor, "lw": 2, "alpha": 0.6},
            ax=axs[i],
            ci=None
        )
        axs[i].plot(
            [min_v - d_val, max_v + d_val],
            [min_v - d_val, max_v + d_val],
            color="gray",
            linestyle="--"
        )
        rng, ticks, ticks_lb = auto_set_tick_and_range(
            max_v, min_v, metric_scale[mt], metric_unit2[mt], offset=(0.1, 0.1), fmt='%d'
        )
        axs[i].set_title(metric_description[mt])
        axs[i].set_xlim(*rng)
        axs[i].set_ylim(*rng)
        axs[i].set_xticks(ticks, ticks_lb)
        axs[i].set_yticks(ticks, ticks_lb)

        if i == 0:
            if amort_mode != "base":
                axs[i].set_ylabel("AnS model")
            else:
                axs[i].set_ylabel("Baseline model")
            
            axs[i].set_xlabel("Player")

        axs[i].legend(
            [Line2D([0], [0], color=regcolor, lw=2, alpha=0.6)],
            [label],
            fontsize=8, loc='lower right'
        )

    fig_save(
        f"{PATH_VIS_AMORT_RESULT}{model}/{session}/{iter}/{prefix_block[block]}_{prefix_session[target_sess]}", 
        f"comp_binning_target", DPI=300, 
        save_svg=True
    )



def metric_hist_comp_per_player(model, session, iter, block, target_sess, bins=18):
    amort_mode = model.split('_')[0]
    metrics = copy.deepcopy(metric_list[amort_mode])
    metrics[metrics.index(M_ACC)] = M_SE_NORM

    sim = simul_data(model, session, iter, block, target_sess)

    kld_table = {'player': PLAYER["PRO"] + PLAYER["AMA"] + ["M", "SD"]}

    for mt in metrics:
        kld_table[mt] = []

    for i, mt in enumerate(metrics):

        fig, axs = plt.subplots(4, 5, figsize=np.array([5, 2.5])*2.5, constrained_layout=True)

        for tn, tier in enumerate(TIER):
            for pn, player in enumerate(PLAYER[tier]):
                r = 2*tn + (pn // 5)
                c = pn % 5

                ax = axs[r][c]

                exp_data = pd.concat([exp[block][target_sess][player][mode][color] for mode in MODE for color in COLOR])
                sim_data = pd.concat([sim[player][mode][color]["df"] for mode in MODE for color in COLOR])
                exp_data[M_SE_NORM] = exp_data[M_SE] / exp_data["target_radius"]
                sim_data[M_SE_NORM] = sim_data[M_SE] / sim_data["target_radius"]

                exp_data = exp_data[mt].to_numpy()
                sim_data = sim_data[mt].to_numpy()

                kld_table[mt].append(kl_divergence(exp_data, sim_data))
                
                w_e, *_ = ax.hist(exp_data, bins=bins, range=(0, metric_max2[mt]), density=True, color=['royalblue', 'lightcoral'][tn], linewidth=1.5, histtype='step', zorder=80)
                w_s, *_ = ax.hist(sim_data, bins=bins, range=(0, metric_max2[mt]), density=True, color='k', linewidth=1.5, histtype='step', alpha=0.8, zorder=100)
                # sns.kdeplot(
                #     data=exp_data,
                #     color='red',
                #     linewidth=2,
                #     fill=False,
                #     ax=ax
                # )
                # sns.kdeplot(
                #     data=sim_data,
                #     color='blue',
                #     linewidth=2,
                #     fill=False,
                #     ax=ax
                # )
                ax.axvline(np.mean(exp_data), linewidth=2, color=['royalblue', 'lightcoral'][tn], linestyle='--', zorder=110)
                ax.axvline(np.mean(sim_data), linewidth=2, color='k', linestyle='--', alpha=0.8, zorder=120)

                max_d = max(np.max(w_e), np.max(w_s))
                axs[r][c].set_ylim(0, 1.1*max_d)
                if c == 0: axs[r][c].set_yticks([0, max_d/2, max_d], [0, 0.5, 1])
                else: axs[r][c].set_yticks([0, max_d/2, max_d], ['', '', ''])

                if r != 3 or c != 0:
                    ax.set_xlabel("")
                    ax.set_ylabel("")

                axs[r][c].set_title(["Professional", "Amateur"][tn] + f" {pn+1:02d}")
                
        model_name = "AnS model" if amort_mode != 'base' else "Baseline model"

        axs[3][2].set_xlabel(metric_description[mt])
        axs[1][0].set_ylabel("Normalized probability")
        axs[3][0].legend(
            [
                Line2D([0], [0], color="royalblue", lw=1.5), 
                Line2D([0], [0], color="lightcoral", lw=1.5),
                Line2D([0], [0], color="k", lw=1.5, alpha=0.8),
            ], 
            [
                "Professional", 
                "Amateur",
                model_name,
            ], 
            fontsize=8, loc="upper right"
        )
        for r in range(4):
            for c in range(5):
                # axs[r][c].set_ylim(0, max_d * 1.1)
                xrng, xticks, xticks_lb = auto_set_tick_and_range(
                    metric_max2[mt], 0, metric_scale[mt], 
                    metric_hist_unit[mt], 
                    offset=(0, 0), 
                    fmt='%d'
                )
                axs[r][c].set_xlim(*xrng)
                if r == 3: axs[r][c].set_xticks(xticks, xticks_lb)
                else: axs[r][c].set_xticks(xticks, [None] * len(xticks))

                # yrng, yticks, yticks_lb = auto_set_tick_and_range(
                #     max_d, 0, 1, 
                #     max_d / 2, 
                #     offset=(0, 0.05),
                # )
                # axs[r][c].set_ylim(*yrng)
                # if c == 0: axs[r][c].set_yticks(yticks, yticks_lb)
                # else: axs[r][c].set_yticks(yticks, [None] * len(yticks))

        fig_save(f"{PATH_VIS_AMORT_RESULT}{model}/{session}/{iter}/{prefix_block[block]}_{prefix_session[target_sess]}", f"comp_hist_{mt}", DPI=300, save_svg=True)

        

        kld_table[mt].append(np.mean(kld_table[mt]))
        kld_table[mt].append(np.std(kld_table[mt][:20]))
    
    pd.DataFrame(kld_table).to_csv(f"{PATH_VIS_AMORT_RESULT}{model}/{session}/{iter}/{prefix_block[block]}_{prefix_session[target_sess]}/comp_kld.csv", index=False)


def draw_fitts_law(model, session, iter, block, target_sess):
    sim = simul_data(model, session, iter, block, target_sess)
    amort_mode = model.split('_')[0]

    sim_tid = {player: {color: np.array([]) for color in COLOR} for player in PLAYERS}
    sim_tct = {player: {color: np.array([]) for color in COLOR} for player in PLAYERS}
    exp_tid = {player: {color: np.array([]) for color in COLOR} for player in PLAYERS}
    exp_tct = {player: {color: np.array([]) for color in COLOR} for player in PLAYERS}

    for player in PLAYERS:
        for color in COLOR:
            for mode in MODE:
                _sim = sim[player][mode][color]["df"].copy()
                sim_tid[player][color] = np.append(
                    sim_tid[player][color],
                    _sim[_sim["session_id"].str.endswith(color[0])]["t_iod"].to_numpy()
                )
                sim_tct[player][color] = np.append(
                    sim_tct[player][color],
                    _sim[_sim["session_id"].str.endswith(color[0])][M_TCT].to_numpy()
                )
                _exp = exp[block][target_sess][player][mode][color].copy()
                exp_tid[player][color] = np.append(
                    exp_tid[player][color],
                    _exp[_exp["session_id"].str.endswith(color[0])]["t_iod"].to_numpy()
                )
                exp_tct[player][color] = np.append(
                    exp_tct[player][color],
                    _exp[_exp["session_id"].str.endswith(color[0])][M_TCT].to_numpy()
                )
    
    r2_table = {'player': PLAYER["PRO"] + PLAYER["AMA"] + ["M", "SD"]}
    
    for color in COLOR:

        fig, axs = plt.subplots(4, 5, figsize=np.array([5, 4])*2.5, constrained_layout=True)
        r2_table[f"{color}_R2"] = []
        r2_table[f"{color}_delta_a"] = []
        r2_table[f"{color}_delta_b"] = []

        min_diff = 9999
        min_tct = 9999
        max_diff = -9999
        max_tct = -9999

        for tn, tier in enumerate(TIER):
            for pn, player in enumerate(PLAYER[tier]):
                
                sim_data = np.vstack((sim_tid[player][color], sim_tct[player][color])).T
                exp_data = np.vstack((exp_tid[player][color], exp_tct[player][color])).T
                sres = np_groupby(sim_data, key_pos=0, lvl=7)
                eres = np_groupby(exp_data, key_pos=0, lvl=7)

                r = 2*tn + (pn // 5)
                c = pn % 5
                ax = axs[r][c]

                max_diff = max(np.max(sres[:,0]), np.max(eres[:,0]), max_diff)
                max_tct = max(np.max(sres[:,1]), np.max(eres[:,1]), max_tct)
                min_diff = min(np.min(sres[:,0]), np.min(eres[:,0]), min_diff)
                min_tct = min(np.min(sres[:,1]), np.min(eres[:,1]), min_tct)

                sns.regplot(
                    x=sres[:,0],
                    y=sres[:,1],
                    scatter_kws={"color": "black", "alpha": 0},
                    line_kws={"color": "k", "lw": 2.0, "alpha": 0.8, "zorder": 100},
                    ax=ax,
                    ci=None
                )
                ax.scatter(
                    sres[:,0], sres[:,1], 
                    marker='o', edgecolor='black', linewidth=1, 
                    facecolor=color, 
                    s=40, alpha=0.8, zorder=100
                )
                sns.regplot(
                    x=eres[:,0],
                    y=eres[:,1],
                    scatter_kws={"color": "black", "alpha": 0},
                    line_kws={"color": ['royalblue', 'lightcoral'][tn], "lw": 2.0, "zorder": 60},
                    ax=ax,
                    ci=None
                )
                ax.scatter(
                    eres[:,0], eres[:,1], 
                    marker='o', edgecolor=['royalblue', 'lightcoral'][tn], linewidth=1, 
                    facecolor=color, 
                    s=40, alpha=1, zorder=60
                )

                fl_sim = np.polyfit(sres[:,0], sres[:,1], 1)
                fl_exp = np.polyfit(eres[:,0], eres[:,1], 1)

                a_delta = abs(fl_sim[0] - fl_exp[0])
                b_delta = abs(fl_sim[1] - fl_exp[1])

                r2_sim = get_r_squared(np.poly1d(fl_sim)(sres[:,0]), sres[:,1])
                r2_exp = get_r_squared(np.poly1d(fl_exp)(eres[:,0]), eres[:,1])

                # model_name = "AnS" if amort_mode != 'base' else "Baseline"

                ax.legend(
                    [
                        Line2D([0], [0], color=['royalblue', 'lightcoral'][tn], lw=2.0), 
                        Line2D([0], [0], color="k", lw=2.0, alpha=0.8)
                    ], 
                    [
                        # f"Player: $y={fl_exp[0]:.2f}x+{fl_exp[1]:.2f}$ $(R^2={r2_exp:.2f})$", 
                        # f"{model_name}: $y={fl_sim[0]:.2f}x+{fl_sim[1]:.2f}$ $(R^2={r2_sim:.2f})$",
                        f"$R^2={r2_exp:.2f}$", 
                        f"$R^2={r2_sim:.2f}$"
                    ], 
                    fontsize=7.5, loc="lower right"
                )

                r2_table[f"{color}_R2"].append(r2_sim)
                r2_table[f"{color}_delta_a"].append(a_delta)
                r2_table[f"{color}_delta_b"].append(b_delta)
        
        xrng, xticks, xticks_lb = auto_set_tick_and_range(
            max_diff, min_diff, 1, 1, offset=(0.1, 0.1), fmt='%d'
        )
        yrng, yticks, yticks_lb = auto_set_tick_and_range(
            max_tct, min_tct, 1, 0.2, offset=(0.1, 0.1), fmt='%.1f'
        )

        for r in range(4):
            for c in range(5):
                axs[r][c].set_xlim(*xrng)
                axs[r][c].set_ylim(*yrng)
                if r == 3: axs[r][c].set_xticks(xticks, xticks_lb)
                else: axs[r][c].set_xticks(xticks, None)
                if c == 0: axs[r][c].set_yticks(yticks, yticks_lb)
                else: axs[r][c].set_yticks(yticks, None)

        fig_save(f"{PATH_VIS_AMORT_RESULT}{model}/{session}/{iter}/{prefix_block[block]}_{prefix_session[target_sess]}", f"fitts_law_{color}", DPI=300, save_svg=True)
        r2_table[f"{color}_R2"].append(np.mean(r2_table[f"{color}_R2"]))
        r2_table[f"{color}_R2"].append(np.std(r2_table[f"{color}_R2"][:20]))
        r2_table[f"{color}_delta_a"].append(np.mean(r2_table[f"{color}_delta_a"]))
        r2_table[f"{color}_delta_a"].append(np.std(r2_table[f"{color}_delta_a"][:20]))
        r2_table[f"{color}_delta_b"].append(np.mean(r2_table[f"{color}_delta_b"]))
        r2_table[f"{color}_delta_b"].append(np.std(r2_table[f"{color}_delta_b"][:20]))
    
    pd.DataFrame(r2_table).to_csv(f"{PATH_VIS_AMORT_RESULT}{model}/{session}/{iter}/{prefix_block[block]}_{prefix_session[target_sess]}/FL_r2.csv", index=False)


def draw_endpoint_normalized(model, session, iter, block, target_sess):
    os.makedirs(f"{PATH_VIS_AMORT_RESULT}{model}/{session}/{iter}/{prefix_block[block]}_{prefix_session[target_sess]}/endpoint/", exist_ok=True)

    sim = simul_data(model, session, iter, block, target_sess)

    exp_data = pd.concat([exp[block][target_sess][player][mode][color] for player in PLAYERS for mode in MODE for color in COLOR])
    sim_data = pd.concat([sim[player][mode][color]["df"] for player in PLAYERS for mode in MODE for color in COLOR])

    for sn in SES_NAME_ABBR:
        e = exp_data[exp_data["session_id"] == sn]
        s = sim_data[sim_data["session_id"] == sn]

        exp_ep_norm = e[["endpoint_x_norm", "endpoint_y_norm"]].to_numpy()
        sim_ep_norm = s[["endpoint_x_norm", "endpoint_y_norm"]].to_numpy()

        fig, axs = plt.subplots(1, 2, figsize=(6.4, 1.8), constrained_layout=True)

        sns.kdeplot(
            x=exp_ep_norm[:,0],
            y=exp_ep_norm[:,1],
            cmap='Reds',
            fill=True,
            bw_adjust=0.5,
            ax=axs[0]
        )
        axs[0].set_xlim(-MONITOR_BOUND[X]/8, MONITOR_BOUND[X]/8)
        axs[0].set_ylim(-MONITOR_BOUND[Y]/8, MONITOR_BOUND[Y]/8)
        axs[0].set_title(f'Experiment ({sn})')

        sns.kdeplot(
            x=sim_ep_norm[:,0],
            y=sim_ep_norm[:,1],
            cmap='Reds',
            fill=True,
            bw_adjust=0.5,
            ax=axs[1]
        )
        axs[1].set_xlim(-MONITOR_BOUND[X]/8, MONITOR_BOUND[X]/8)
        axs[1].set_ylim(-MONITOR_BOUND[Y]/8, MONITOR_BOUND[Y]/8)
        axs[1].set_title('Simulation')
        
        fig_save(f"{PATH_VIS_AMORT_RESULT}{model}/{session}/{iter}/{prefix_block[block]}_{prefix_session[target_sess]}/endpoint/", f"{sn}", DPI=300, save_svg=False)


model_list = {
    "full_tf_q16_o4_do0.4-mlp_f128_o64-pte_128x2-tr_it2048_b64_n64_full_231219_165740_73_20000000" : {
        "0107_192443": [100],
    },
    "base_tf_q16_o4_do0.4-mlp_f128_o64-pte_128x2-tr_it2048_b64_n64_base_240104_164252_68_20000000" : {
        "0117_144253": [100],
    },
    # "abl1_tf_q16_o4_do0.4-mlp_f128_o64-pte_128x2-tr_it2048_b64_n64_full_231219_165740_73_20000000": {
    #     "0108_145616": [100],
    # },
    # "abl3_tf_q16_o4_do0.4-mlp_f128_o64-pte_128x2-tr_it2048_b64_n64_full_231219_165740_73_20000000": {
    #     "0108_174146": [100]
    # }
}

run_list = []
for model in model_list.keys():
    for session in model_list[model]:
        for iter in model_list[model][session]:
            for block_use in ['half']:
                for session_use in ['all', 'sub']:
                    run_list.append((model, session, iter, block_use, session_use))


def run_visualization(model, session, iter, block_use, session_use):
    os.makedirs(f"{PATH_VIS_AMORT_RESULT}{model}/{session}/{iter}/{prefix_block[block_use]}_{prefix_session[session_use]}", exist_ok=True)
    # anova_tier(model, session, iter, block_use, session_use)
    # anova_color(model, session, iter, block_use, session_use)
    # anova_sensitivity(model, session, iter, block_use, session_use)
    # metric_comp_binning_player(model, session, iter, block_use, session_use)
    # draw_dist(model, session, iter, block_use, session_use)
    # metric_mean_comp_per_player(model, session, iter, block_use, session_use)
    # metric_mean_comp_per_player_collected(model, session, iter, block_use, session_use)
    # metric_hist_comp_per_player(model, session, iter, block_use, session_use)
    draw_fitts_law(model, session, iter, block_use, session_use)
    # draw_endpoint_normalized(model, session, iter, block_use, session_use)
    # compute_traj_diff(model, session, iter, block_use, session_use)


Parallel(n_jobs=12)(delayed(run_visualization)(*args) for args in tqdm(run_list))