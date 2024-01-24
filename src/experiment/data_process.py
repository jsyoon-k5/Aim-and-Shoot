from pickle import load
import pandas as pd
import seaborn as sns

import sys, os, argparse

from tqdm import tqdm
from joblib import Parallel, delayed

sys.path.append("..")
from experiment.raw_data_organizer import ExperimentDataRaw
from experiment.data_organizer import ExperimentData
from utils.mymath import gaussian

from configs.common import *
from configs.experiment import *
from configs.path import *
from utils.dataloader import parse_tier_by_playername
from utils.mymath import *
from utils.plots import fig_save, set_tick_and_range
from utils.utils import pickle_load, pickle_save

import matplotlib.pyplot as plt

def generate_traj_pickles():
    """Create basic experiment datatables and save trial trajectory data"""
    def save_trajs(p, m):
        d = ExperimentDataRaw(p, m, msg=False)
        d.load_existing_tables(msg=False)
        for sn in SES_NAME_ABBR:
            for bn in range(2):
                d.save_trial_trajectories(sn, bn, plot=False)

    Parallel(n_jobs=12)(delayed(save_trajs)(p, m) for p, m in zip(tqdm(PLAYERS + PLAYERS), ["custom"]*20+["default"]*20))


def generate_individual_expdata(plot=False):
    def make_csv(p, m):
        ExperimentData(p, m).export_to_csv(plot_mseq=plot)
    Parallel(n_jobs=12)(delayed(make_csv)(p, m) for p, m in zip(tqdm(PLAYERS + PLAYERS), ["custom"]*20+["default"]*20))



def merge_individual_expdata(fill_empty_reaction=True):
    if not fill_empty_reaction:
        data = pd.concat([
            pd.read_csv(PATH_DATA_SUMMARY_PLAYER % (player, mode)) for mode in MODE for player in PLAYERS
        ])
    else:
        data = []
        for player in PLAYERS:
            for mode in MODE:
                d = pd.read_csv(PATH_DATA_SUMMARY_PLAYER % (player, mode))
                g = d.groupby(["session_id"])["gaze_reaction"].mean()
                fdict = {k:v for k, v in zip(g.index, g.to_numpy())}
                d["gaze_reaction"] = d["gaze_reaction"].fillna(d["session_id"].map(fdict))
                g = d.groupby(["session_id"])["hand_reaction"].mean()
                fdict = {k:v for k, v in zip(g.index, g.to_numpy())}
                d["hand_reaction"] = d["hand_reaction"].fillna(d["session_id"].map(fdict))
                data.append(d)
        data = pd.concat(data)
    
    # data["include_analysis"] = data["learning_effect"] + data["invalid_tct"] + data["not_enough_gaze"] + data["init_gaze_faraway"] == 0
    data.to_csv(PATH_DATA_SUMMARY_MERGED, index=False)

    data = pd.concat([
        pd.read_csv(PATH_DATA_SUMMARY_PLAYER % (player, f"{mode}-mseq")) for mode in MODE for player in PLAYERS
    ])
    data.to_csv(PATH_DATA_SUMMARY_MERGED_MSEQ, index=False)
    


def organize_anova_format():
    if not os.path.exists(PATH_DATA_SUMMARY_MERGED):
        print("Merged csv table required first.")
        return
    
    data = load_experiment(block_index=1)
    data = data.groupby(
        ["tier", "player", "mode", "session_id"], 
        as_index=False
    )[METRICS].mean(numeric_only=False)

    ses_list = SES_NAME_ABBR[:]
    ses_list.remove('svsw')
    ses_list.append('svsw') # Re-ordering
    
    dict_data = dict(
        tier = [],
        player = []
    )
    dict_data.update(
        {
            f"{ma}_{mode[0]}_{sn}": [] for mode in MODE for ma in METRICS_ABBR for sn in ses_list
        }
    )

    for player in PLAYERS:
        for mode in MODE:
            for sn in ses_list:
                values = data[
                    (data.player == player) &
                    (data["mode"] == mode) &
                    (data["session_id"] == sn)
                ][METRICS].to_numpy()[0]
                for i in range(len(METRICS)):
                    dict_data[f"{METRICS_ABBR[i]}_{mode[0]}_{sn}"].append(values[i])
        dict_data["tier"].append(parse_tier_by_playername(player))
        dict_data["player"].append(player)  
    
    dict_data = pd.DataFrame(dict_data)
    dict_data.to_csv(PATH_DATA_SUMMARY_ANOVA, index=False)


def organize_learning_effect_table():
    if not os.path.exists(PATH_DATA_SUMMARY_MERGED):
        print("Merged csv table required first.")
        return
    
    data = load_experiment()
    data = data.groupby(
        ["player", "session_id", "block_index"], 
        as_index=False
    )[METRICS].mean(numeric_only=False)

    dict_data = dict(
        tier = []
    )
    dict_data.update({
        f"{ma}_{sn}_{bn}": [] for ma in METRICS_ABBR for sn in SES_NAME_ABBR for bn in range(2)
    })

    for player in PLAYERS:
        for sn in SES_NAME_ABBR:
            for bn in range(2):
                values = data[
                    (data["player"] == player) &
                    (data["session_id"] == sn) &
                    (data["block_index"] == bn)
                ][METRICS].to_numpy()[0]
                for i in range(len(METRICS)):
                    dict_data[f"{METRICS_ABBR[i]}_{sn}_{bn}"].append(values[i])
        dict_data["tier"].append(parse_tier_by_playername(player))

    dict_data = pd.DataFrame(dict_data)
    dict_data.to_csv(f"{PATH_DATA_SUMMARY}/learning_effect_block.csv", index=False)


def add_outlier_flag(metrics=[M_TCT, M_SE, M_CGE, M_CSM, M_CTL]):
    exp = pd.read_csv(PATH_DATA_SUMMARY_MERGED)

    for mt in metrics:
        if f"outlier_{METRICS_ABBR[METRICS.index(mt)]}" in exp.columns:
            del exp[f"outlier_{METRICS_ABBR[METRICS.index(mt)]}"]

    lowest = {mt: [] for mt in metrics}
    highest = {mt: [] for mt in metrics}

    flag = {mt: [] for mt in metrics}

    for mt in metrics:
        for sn in SES_NAME_ABBR:
            data = exp[exp["session_id"] == sn][mt].to_numpy()
            data = np.sort(data)

            quantile_25 = np.percentile(data, 25)
            quantile_75 = np.percentile(data, 75)
            IQR = quantile_75 - quantile_25
            
            lowest[mt].append(quantile_25 - 1.5 * IQR)
            highest[mt].append(quantile_75 + 1.5 * IQR)

    for mt in metrics:
        for sn, val in zip(exp["session_id"].to_list(), exp[mt].to_numpy()):
            l, h = lowest[mt][SES_NAME_ABBR.index(sn)], highest[mt][SES_NAME_ABBR.index(sn)]
            if val < l or val > h:
                flag[mt].append(1)
            else:
                flag[mt].append(0)
    
    for mt in metrics:
        exp[f"outlier_{METRICS_ABBR[METRICS.index(mt)]}"] = flag[mt]

    exp.to_csv(PATH_DATA_SUMMARY_MERGED, index=False)


def load_experiment(
    exclude_invalid=True, 
    exclude_outlier=True, 
    outlier_metrics=[M_TCT, M_SE],
    **kwargs
):
    exp = pd.read_csv(PATH_DATA_SUMMARY_MERGED)

    if exclude_invalid:
        exp = exp[
            (exp["not_enough_gaze"] == 0) &
            (exp["init_gaze_faraway"] == 0)
        ]
    if exclude_outlier:
        for mt in outlier_metrics:
            exp = exp[exp[f"outlier_{METRICS_ABBR[METRICS.index(mt)]}"] == 0]

    for key, value in kwargs.items():
        if type(value) is list: exp = exp[exp[key].isin(value)]
        else: exp = exp[exp[key] == value]
    
    return exp


def load_trajectory(player, mode, sn, bn, tn):
    """Assume that parameters are explicitly set valid when using this function"""
    (
        timestamp, 
        traj_t,
        traj_g,
        _,
        _,
        _,
        traj_h,
        traj_c,
        *_
    ) = pickle_load(f"{PATH_DATA_EXP_TRAJ_PKL % player}{mode}-{sn}-{bn}-{tn:02d}.pkl")
    return timestamp, traj_t, traj_g, traj_h, traj_c


def player_distance_info(exclude_invalid=True, return_error=False, verbose=False, **conds):
    exp = load_experiment(exclude_invalid=exclude_invalid, **conds)

    timestamp, dist_tx, dist_gx = [], [], []
    m_tct = list()
    for p, m, sn, bn, tn in zip(
        tqdm(exp["player"]) if verbose else exp["player"],
        exp["mode"],
        exp["session_id"], 
        exp["block_index"], 
        exp["trial_index"]
    ):
        ts, t_traj, g_traj, *_ = load_trajectory(p, m, sn, bn, tn)
        t_dist = np.linalg.norm(t_traj - CROSSHAIR, axis=1)
        g_dist = np.linalg.norm(g_traj - g_traj[0], axis=1)
        
        timestamp.append(ts)
        dist_tx.append(t_dist)
        dist_gx.append(g_dist)
        m_tct.append(ts[-1])
    
    return mean_distance(timestamp, dist_tx, dist_gx, np.mean(m_tct), return_error=return_error)


def save_player_distance_info():
    pickle_save(
        f"{PATH_DATA_SUMMARY}dist_tier.pkl", 
        {
            tier: dict(zip(
                ["ts", "tx", "gx"], 
                list(player_distance_info(tier=tier))
            )) for tier in TIER
        }
    )
    pickle_save(
        f"{PATH_DATA_SUMMARY}dist_player.pkl", 
        {
            player: dict(zip(
                ["ts", "tx", "gx"], 
                list(player_distance_info(player=player))
            )) for player in PLAYERS
        }
    )
    pickle_save(
        f"{PATH_DATA_SUMMARY}dist_player_mode_block.pkl", 
        {
            p: {
                mode: {
                    block: dict(zip(
                        ["ts", "tx", "gx"], 
                        list(player_distance_info(player=p, mode=mode, block_index=block))
                    )) for block in range(2)
                } for mode in MODE
            } for p in PLAYERS
        }
    )
    pickle_save(
        f"{PATH_DATA_SUMMARY}dist_player_mode_color_block.pkl", 
        {
            p: {
                mode: {
                    color: {
                        block: dict(zip(
                            ["ts", "tx", "gx"], 
                            list(player_distance_info(player=p, mode=mode, block_index=block, target_color=color))
                        )) for block in range(2)
                    } for color in COLOR
                } for mode in MODE
            } for p in PLAYERS
        }
    )


def player_distance_info_amort_eval():
    # This function is made for amortizer evaluation
    if os.path.exists(f"{PATH_DATA_SUMMARY}dist_player_eval.pkl"):
        return pickle_load(f"{PATH_DATA_SUMMARY}dist_player_eval.pkl")
    
    block_list = dict(
        all = [0, 1],
        half = [1]
    )
    ses_list = dict(
        all = SES_NAME_ABBR,
        sub = SES_NAME_ABBR_SYMM
    )
    
    dist_info = {}
    for key_block in ["all", "half"]:
        dist_info[key_block] = {}
        for key_sess in ["all", "sub"]:
            dist_info[key_block][key_sess] = {
                player: dict(zip(
                    ["ts", "tx", "gx"], 
                    list(player_distance_info(player=player, block_index=block_list[key_block], session_id=ses_list[key_sess]))
                )) for player in tqdm(PLAYERS)
            }
    pickle_save(
        f"{PATH_DATA_SUMMARY}dist_player_eval.pkl",
        dist_info
    )

    return pickle_load(f"{PATH_DATA_SUMMARY}dist_player_eval.pkl")



def player_trajectory_info(exclude_invalid=True, return_error=False, **conds):
    exp = load_experiment(exclude_invalid=exclude_invalid, **conds)

    timestamp, ttj, gtj, htj, ctj = [], [], [], [], []
    for p, m, sn, bn, tn in zip(
        exp["player"],
        exp["mode"],
        exp["session_id"], 
        exp["block_index"], 
        exp["trial_index"]
    ):
        ts, t_traj, g_traj, h_traj, c_traj = load_trajectory(p, m, sn, bn, tn)
        
        timestamp.append(ts)
        ttj.append(t_traj)
        gtj.append(g_traj)
        htj.append(h_traj)
        ctj.append(c_traj)
    
    return dict(
        ts=timestamp, 
        ttj=ttj, 
        gtj=gtj,
        ctj=ctj
    )




def fitts_law(exclude_invalid=True, fname='default', title=None, **conds):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    exp = load_experiment(exclude_invalid=exclude_invalid, **conds)
    tiod = exp["t_iod"].to_numpy()
    tct = exp["trial_completion_time"].to_numpy()
    acc = exp["result"].to_numpy()

    data = np.vstack((tiod, tct, acc)).T
    data = np_groupby(data, key_pos=0, lvl=6)

    y_true = data[:,0]
    y_pred = data[:,1]
    y_fit = np.polyfit(y_true, y_pred, 1)
    y_func = np.poly1d(y_fit)
    r_squared = get_r_squared(y_true, y_pred)

    label = f"$({y_fit[0]:.2f})x + ({y_fit[1]:.2f}), R^2={r_squared:.2f}$"

    fig, ax = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
    sns.regplot(
        x=y_true,
        y=y_pred,
        scatter_kws={"color": "black", "alpha": 0.3},
        line_kws={"color": "red", "lw": 2.5},
        ax=ax[0]
    )
    if title is not None: ax.set_title(title)
    ax[0].set_xlabel("Index of Difficulty (bit)")
    ax[0].set_ylabel("Completion time (s)")
    
    ax[0].set_xlim([0, 6])
    ax[0].set_ylim([0.1, 1])
    ax[0].legend([Line2D([0], [0], color="red", lw=2.5)], [label,], fontsize=8, loc="lower right")
    ax[0].grid(linestyle="--", linewidth=0.5)

    sns.regplot(
        x=y_true,
        y=data[:,2],
        scatter_kws={"color": "black", "alpha": 0.3},
        line_kws={"color": "red", "lw": 2.5},
        ax=ax[1]
    )
    ax[1].set_xlabel("Index of Difficulty (bit)")
    ax[1].set_ylabel("Accuracy")
    
    ax[1].set_xlim([0, 6])
    ax[1].set_ylim([0, 1])
    ax[1].grid(linestyle="--", linewidth=0.5)

    fig_save(PATH_VIS_FITTS_EXP, fname)
    plt.close(fig)


def initial_gazepoint(exclude_invalid=True):
    g = []
    for player in PLAYERS:
        for mode in MODE:
            exp_loader = ExperimentData(player, mode)
            exp = load_experiment(
                exclude_invalid=exclude_invalid,
                player=player, 
                mode=mode, 
            )
            for sn, bn, tn in zip(exp["session_id"], exp["block_index"], exp["trial_index"]):
                _, _, g_traj, *_ = exp_loader.trial_trajectory(sn, bn, tn)
                g.append(g_traj[0])
    
    g = np.array(g)

    return g




def hand_gaze_reaction_correlation():
    data = load_experiment()

    hrt = data["hand_reaction"].to_numpy()
    grt = data["gaze_reaction"].to_numpy()

    mask1 = ~np.isnan(hrt)
    mask2 = ~np.isnan(grt)

    hrt = hrt[mask1 & mask2]
    grt = grt[mask1 & mask2]

    plt.scatter(hrt, grt, s=0.1, alpha=0.2)
    # plt.hist2d(hrt, grt, bins=35)
    plt.xlim(0, 0.4)
    plt.ylim(0, 0.4)
    plt.xlabel("Hand reaction")
    plt.ylabel("Gaze reaction")
    plt.show()




def init_gaze_pos_gaussian_fit():
    data = load_experiment()
    data = data[["gaze0_x", "gaze0_y"]].to_numpy()

    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    h, xedges, yedges, _ = plt.hist2d(*data.T, bins=100, density=True)
    h = h.ravel()
    x = (xedges[1:] + xedges[:-1]) / 2
    y = (yedges[1:] + yedges[:-1]) / 2
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x**2+y**2)
    r[49:] *= -1
    r = r.ravel()
    popt, pcov = curve_fit(gaussian, r, h, p0=[1, 1], bounds=(0, np.inf))
    plt.close()

    plt.scatter(r, h, s=1, color='b')
    plt.scatter(r, gaussian(r, *popt), s=1, color='r')
    plt.show()

    print(popt)
    

    # y, x_edge, _ = plt.hist(data[:,Y], bins=100, density=True)
    # plt.show()
    # plt.close()
    # x = (x_edge[1:] + x_edge[:-1]) / 2
    # popt, pcov = curve_fit(gaussian, x, y, p0=[1,1], bounds=(0, np.inf))
    
    # plt.scatter(x, y, color='r', s=1)
    # plt.plot(x, gaussian(x, *popt), color='k')
    # plt.show()


def plot_eye_position():
    data = load_experiment()
    data = data[[f"eye_{a}" for a in 'xyz']].to_numpy()

    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    for a in [X, Y, Z]:
        y, x_edge, _ = plt.hist(data[:,a], bins=100, density=True, histtype='step', label='XYZ'[a])
        x = (x_edge[1:] + x_edge[:-1]) / 2
        def func(x, A, sigma):
            return gaussian(x - HEAD_POSITION[a], A, sigma)
        popt, pcov = curve_fit(func, x, y, p0=[20,1], bounds=((15, 0), (35, 10)))
        plt.plot(x, func(x, *popt))
        print(f"{a}: {popt}")
    plt.legend()
    plt.show()

    print(np.max(data, axis=0) - HEAD_POSITION)
    print(np.min(data, axis=0) - HEAD_POSITION)



def merge_individual_mseq():
    data = pd.concat([
        pd.read_csv(PATH_DATA_SUMMARY_PLAYER % (player, "mseq")) for player in PLAYERS
    ])
    data.to_csv(PATH_DATA_SUMMARY_MERGED_MSEQ, index=False)


def visualize_main_sequence(reaction_max=0.5):
    data = pd.read_csv(PATH_DATA_SUMMARY_MERGED_MSEQ)
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    # Reaction time
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    for tier in TIER:
        r = data[data["tier"] == tier]["saccade_reaction"].to_numpy()
        r = r[~np.isnan(r)]
        r = r[r < reaction_max]
        axs.hist(r, bins=100, density=True, histtype='step', label=tier)
    axs.legend()
    set_tick_and_range(axs, 0.1, 1000, reaction_max)
    fig_save(PATH_VIS_EXP_MSEQ, "reaction")

    data = data.dropna()

    amp = {
        tier: data[data["tier"] == tier]["saccade_amplitude"].to_numpy() for tier in TIER
    }
    pve = {
        tier: data[data["tier"] == tier]["saccade_peak_velocity"].to_numpy() for tier in TIER
    }

    plt.scatter(amp["AMA"], pve["AMA"], s=0.1, alpha=0.05, color='r')
    plt.scatter(amp["PRO"], pve["PRO"], s=0.1, alpha=0.05, color='b')
    plt.xlim(0, 25)
    plt.ylim(0, 800)
    plt.show()



def gaze_reaction_time_fitting():
    data = pd.read_csv(PATH_DATA_SUMMARY_MERGED_MSEQ)
    import matplotlib.pyplot as plt
    from scipy import stats

    r = data["saccade_reaction"].to_numpy() / 1000
    r = r[~np.isnan(r)]
    r = r[r < 0.5]

    a, loc, scale = stats.skewnorm.fit(r)
    sample = stats.skewnorm(a, loc, scale).rvs(len(r))

    plt.hist(r, bins=50, density=True, histtype='step', color='b')
    plt.hist(sample, bins=50, density=True, histtype='step', color='r')
    plt.show()

    print(a, loc, scale)


def hand_reaction_time_fitting():
    data = load_experiment()
    data = data["hand_reaction"].to_numpy() / 1000

    import matplotlib.pyplot as plt
    from scipy import stats

    a, loc, scale = stats.skewnorm.fit(data)
    sample = stats.skewnorm(a, loc, scale).rvs(len(data))

    plt.hist(data, bins=20, density=True, histtype='step', color='b')
    plt.hist(sample, bins=20, density=True, histtype='step', color='r')
    plt.show()

    print(a, loc, scale)


def hand_speed_stat():
    exp = load_experiment(mode='default')
    hs = exp["hand_max_speed"]

    import matplotlib.pyplot as plt

    plt.hist(hs, bins=100)
    plt.show()


def show_shoot_error():
    exp = load_experiment(
        mode='default', 
        target_speed=0, 
        block_index=1, 
        # result=1
    )

    se = [exp[exp["target_radius"] == r]["endpoint_x"].to_numpy() / r for r in [
        TARGET_RADIUS_VERY_SMALL, TARGET_RADIUS_SMALL, TARGET_RADIUS_LARGE
    ]]

    for i, s in enumerate(se):
        plt.hist(s, bins=100, histtype='step', density=True, label=i)
    plt.legend()
    plt.show()

    # import matplotlib.pyplot as plt
    # from scipy import stats

    # loc, scale = list(), list()
    # for i, _s in enumerate(se):
    #     _l, _s = stats.norm.fit(_s)
    #     loc.append(_l)
    #     scale.append(_s)
    
    # return loc, scale



def main_sequence_fitting(player=None):
    data = pd.read_csv(PATH_DATA_SUMMARY_MERGED_MSEQ)
    if player is not None:
        data = data[data["player"] == player]
    # data = data.dropna()
    amp = data["saccade_amplitude"].to_numpy()
    pve = data["saccade_peak_velocity"].to_numpy()

    va = np.mean(pve[amp < 1.05])
    ath = np.mean(amp[amp < 1.05])

    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    def func(x, theta):
        return 38 + theta * (x - 1)

    popt, pcov = curve_fit(func, amp, pve, p0=1, bounds=(0, np.inf))

    x = np.linspace(amp.min(), amp.max(), 1000)

    plt.figure(figsize=(5,5))
    plt.title(f"Main Sequence - {player}")
    plt.scatter(amp, pve, s=0.1, color='b', alpha=0.5)
    # plt.scatter(amp[data["target_color"] == "white"], pve[data["target_color"] == "white"], s=0.1, color='b')
    # plt.scatter(amp[data["target_color"] == "gray"], pve[data["target_color"] == "gray"], s=0.1, color='r')
    plt.plot(x, func(x, *popt), linestyle='--', color='k', label=fr"$y=38+{popt[0]:.2f}\cdot(x-1)$")
    plt.legend()
    plt.xlim(0, 25)
    plt.ylim(0, 800)
    plt.xlabel("Amplitude (deg)")
    plt.ylabel("Peak Velocity (deg/s)")
    if player is not None:
        fig_save(PATH_VIS_EXP_MSEQ, f"peak_vel_{player}")
    else:
        fig_save(PATH_VIS_EXP_MSEQ, f"peak_vel_all")

    return popt[0]


def analyze_doubleline_main_sequence():
    data = pd.read_csv(PATH_DATA_SUMMARY_MERGED_MSEQ)
    data = data[~np.isnan(data["saccade_reaction"].to_numpy())]

    # data = data[data["target_speed"] > 0]
    amp = data["saccade_amplitude"].to_numpy()
    pve = data["saccade_peak_velocity"].to_numpy()

    tt = 0.053

    amp = data[data["saccade_duration"] < tt]["saccade_amplitude"].to_numpy()
    pve = data[data["saccade_duration"] < tt]["saccade_peak_velocity"].to_numpy() 
    plt.scatter(amp, pve, s=0.1, alpha=0.1, color='red', label=f'dur<{tt:.3f}')

    amp = data[data["saccade_duration"] >= tt]["saccade_amplitude"].to_numpy()
    pve = data[data["saccade_duration"] >= tt]["saccade_peak_velocity"].to_numpy() 
    plt.scatter(amp, pve, s=0.1, alpha=0.1, color='blue', label=f'dur>{tt:.3f}')

    plt.legend()
    plt.xlabel("Amplitude")
    plt.ylabel("Peak Velocity")
    plt.show()
    plt.close('all')



def render_all_replay_videos():
    def make_vid(p, m):
        d = ExperimentData(p, m)
        for sn in SES_NAME_ABBR:
            for bn in range(2):
                d.export_replay(
                    d.get_list_of_trial(sn, bn),
                    framerate=144,
                    res=1080,
                    videoname=f"{m}_{sn}_{bn}"
                )
    Parallel(n_jobs=6)(delayed(make_vid)(p, m) for m in MODE for p in PLAYERS)


def observe_data_distribution():
    exp = load_experiment()

    ranges = [
        [0.15, 1.2],
        [0, 0.025],
        [0, 16],
        [0, 0.3],
        [0.1, 0.3],
        [0, 1],
        [9, 780],
        [0, 120],
    ]

    for im, mt in enumerate([M_TCT, M_SE, M_GD, "gaze_reaction", "hand_reaction", M_CGE, M_CSM, M_CTL]):
        fig, axs = plt.subplots(10, 2, figsize=(10, 10), constrained_layout=True)

        for i, player in enumerate(PLAYER["PRO"] + PLAYER["AMA"]):
            data = exp[exp["player"] == player][mt].to_numpy()

            ax = axs[i%10][i//10]

            ax.hist(data, bins=100, range=ranges[im], histtype='step')
            ax.set_xlim(*ranges[im])
            ax.set_title(player)
        
        fig_save(PATH_VIS_ROOT, f'dist_{mt}')


            


def check_outlier():
    import matplotlib.pyplot as plt

    data = load_experiment(mode='default')
    tct = data[M_TCT].to_numpy()
    tct = np.sort(tct)

    q1 = tct[tct.size // 4]
    q3 = tct[tct.size // 4 * 3]
    iqr = q3 - q1

    q_min_1 = q1 - iqr
    q_max_1 = q3 + iqr

    q_min_2 = q1 - 1.5 * iqr
    q_max_2 = q3 + 1.5 * iqr

    plt.hist(tct, bins=100, histtype='step')
    plt.axvline(q1, color='r', label='Q1, Q3')
    plt.axvline(q3, color='r')
    plt.axvline(q_min_1, color='g', label='1 IQR')
    plt.axvline(q_max_1, color='g')
    plt.axvline(q_min_2, color='k', label='1.5 IQR')
    plt.axvline(q_max_2, color='k')
    plt.legend()
    plt.show()


def check_outlier_by_task():
    import matplotlib.pyplot as plt

    data = load_experiment(mode='default')

    fig, axs = plt.subplots(9, 1, figsize=(9, 16), constrained_layout=True)

    for i, sn in enumerate(SES_NAME_ABBR):
        data = load_experiment(exclude_invalid=False, exclude_outlier=False, mode='default', session_id=sn)
        tct = np.linalg.norm(data[["gaze0_x", "gaze0_y"]].to_numpy(), axis=1)
        tct = np.sort(tct)

        q1 = tct[tct.size // 4]
        q3 = tct[tct.size // 4 * 3]
        iqr = q3 - q1

        q_min_1 = q1 - iqr
        q_max_1 = q3 + iqr

        q_min_2 = q1 - 1.5 * iqr
        q_max_2 = q3 + 1.5 * iqr

        axs[i].hist(tct, bins=50, histtype='step', color='k')
        axs[i].axvline(q1, color='r', label='Q1, Q3')
        axs[i].axvline(q3, color='r')
        axs[i].axvline(q_min_1, color='g', label='1 IQR')
        axs[i].axvline(q_max_1, color='g')
        axs[i].axvline(q_min_2, color='k', label='1.5 IQR')
        axs[i].axvline(q_max_2, color='k')
        if i == 0:
            axs[i].legend()
        axs[i].set_xlim(0, 0.06)
        axs[i].set_ylabel(SES_NAME_FULL[i])
    
    fig_save(PATH_VIS_ROOT, 'ep_task')


def collect_pcam():
    import matplotlib.pyplot as plt

    pcam = []
    for player in PLAYERS:
        for mode in MODE:
            d = ExperimentData(player, mode)
            task_list = load_experiment(player=player, mode=mode)
            for sn, bn, tn in zip(task_list["session_id"], task_list["block_index"], task_list["trial_index"]):
                *_, _pcam, _, _ = d.trial_trajectory(sn, bn, tn)
                pcam.append(_pcam)
    
    pcam = np.concatenate(pcam, axis=0)

    plt.scatter(*pcam.T, s=0.1, alpha=0.1)
    plt.xlim(-90, 90)
    plt.ylim(-90, 90)
    plt.show()


def difficulty_dist():
    import matplotlib.pyplot as plt

    exp = load_experiment(target_speed = 0)
    tid = exp["t_iod"].to_numpy()

    sns.kdeplot(tid)
    for p in [25, 50, 75]:
        plt.axvline(np.percentile(tid, p))
    plt.show()


def normalized_shoot_error():
    import matplotlib.pyplot as plt

    for tier in TIER:
        exp = load_experiment(tier=tier)
        trad = exp["target_radius"].to_numpy()
        se = exp["shoot_error"].to_numpy()
        norm_acc = se / trad
        tct = exp["trial_completion_time"].to_numpy()
        # plt.hist(norm_acc, bins=100, histtype='step')
        sns.kdeplot(tct, label=tier)
    plt.legend()

    plt.show()



def _temp_add_hand_reaction():
    exp = pd.read_csv(PATH_DATA_SUMMARY_MERGED)
    exp["hand_reaction"] += 0.02
    exp.to_csv(PATH_DATA_SUMMARY_MERGED)


def get_max_acc_of_cam():
    # ts = []
    # ctj = []
    # for player in tqdm(PLAYERS):
    #     data = player_trajectory_info(player=player, block_index=1)
    #     ts += data["ts"]
    #     ctj += data["ctj"]
    
    # pickle_save("temp.pkl", (ts, ctj))
    ts, ctj = pickle_load('temp.pkl')
    
    speed = []
    accel = []
    for t, c in zip(tqdm(ts), ctj):
        p = apply_gaussian_filter_2d(c, sigma=13)
        p = np.linalg.norm(p, axis=1)
        v = np.abs(np.diff(p)) / np.diff(t)
        a = derivative_central(t[1:], v)
        speed.append(v)
        accel.append(a)
    
    speed = np.concatenate(speed)
    accel = np.concatenate(accel)
    print(speed.min(), speed.max(), np.percentile(speed, 99.9))
    plt.hist(speed, bins=500)
    plt.show()
    print(accel.min(), accel.max())
    plt.hist(accel, bins=500)
    plt.show()


def contour_endpoint_norm(ratio=8):
    data = load_experiment()
    # for sn in SES_NAME_ABBR:
    #     d = data[data["session_id"] == sn][["endpoint_x_norm", "endpoint_y_norm"]].to_numpy()
    #     fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    #     sns.kdeplot(
    #         x=d[:,0],
    #         y=d[:,1],
    #         cmap='Reds',
    #         fill=True,
    #         bw_adjust=0.5,
    #         ax=ax
    #     )
    #     ax.set_xlim(-MONITOR_BOUND[X]/ratio, MONITOR_BOUND[X]/ratio)
    #     ax.set_ylim(-MONITOR_BOUND[X]/ratio, MONITOR_BOUND[X]/ratio)
    #     ax.set_title(f'Experiment ({sn})')
    #     ax.set_aspect('equal')

    #     fig_save(PATH_VIS_ROOT, f'endpoint_norm_{sn}')
    
    d = data[["endpoint_x_norm", "endpoint_y_norm"]].to_numpy()
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    sns.kdeplot(
        x=d[:,0],
        y=d[:,1],
        cmap='Reds',
        fill=True,
        bw_adjust=0.5,
        ax=ax
    )
    ax.set_xlim(-MONITOR_BOUND[X]/ratio, MONITOR_BOUND[X]/ratio)
    ax.set_ylim(-MONITOR_BOUND[X]/ratio, MONITOR_BOUND[X]/ratio)
    ax.set_title(f'Experiment (all)')
    ax.set_aspect('equal')

    fig_save(PATH_VIS_ROOT, f'endpoint_norm_all')





if __name__ == "__main__":

    # generate_traj_pickles()
    # generate_individual_expdata(plot=False)
    # merge_individual_expdata(fill_empty_reaction=True)
    # add_outlier_flag()
    # organize_anova_format()
    # save_player_distance_info()
    # observe_data_distribution()

    # gaze_reaction_time_fitting()
    # hand_reaction_time_fitting()

    # for player in PLAYERS:
    #     for mode in MODE:
    #         fitts_law(fname=player, player=player, mode=mode)

    # difficulty_dist()
    # normalized_shoot_error()

    # timestamp, ttj, gtj, htj, ctj = player_trajectory_info(player='KKW', session_id='ssw', mode='default', block_index=1)

    # _temp_add_hand_reaction()

    # get_max_acc_of_cam()

    # contour_endpoint_norm()

    # organize_learning_effect_table()

    player_distance_info_amort_eval()

    pass
