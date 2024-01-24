import numpy as np
import pandas as pd

import torch
import copy
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import f_oneway
from joblib import Parallel, delayed
from typing import Union

import sys, os

sys.path.append('..')

from nets.amortizer import AmortizerForTrialData, RegressionForTrialData
from amortizer.ans_dataset import AnSPlayerDataset
from amortizer.ans_simulator import AnSSimulator
from agent.agent_simulation import Simulator
from experiment.data_process import *
from configs.amort import default_ans_config
from configs.path import *
from configs.experiment import *
from configs.simulation import *
from utils.mymath import *
from utils.utils import *
from utils.plots import *

#### infer split: task condition

"""
exp_dist := dict[key_block][key_sess][player][ts,tx,gx]
"""
exp_dist = player_distance_info_amort_eval()

discrepancy_metric = dict(
    full = [M_TCT, M_SE_NORM, M_GD],
    base = [M_TCT, M_SE_NORM],
    abl0 = [M_TCT, M_SE_NORM, M_GD],
    abl1 = [M_TCT, M_SE_NORM, M_GD],
    abl2 = [M_TCT, M_SE_NORM, M_GD],
    abl3 = [M_TCT, M_SE_NORM, M_GD],
)

discrepancy_scaler = dict(
    full = np.array([5.17651777, 1, 1.14167245]),
    base = np.array([5.17651777, 1]),
    abl0 = np.array([5.17651777, 1, 1.14167245]),
    abl1 = np.array([5.17651777, 1, 1.14167245]),
    abl2 = np.array([5.17651777, 1, 1.14167245]),
    abl3 = np.array([5.17651777, 1, 1.14167245]),
)


def get_amortizer_and_simulator(model_name):
    setting = model_name.replace('-','_').split('_')
    mode = setting[0]
    cfg = copy.deepcopy(default_ans_config[mode])["amortizer"]
    sim = copy.deepcopy(default_ans_config[mode])["simulator"]

    # Encoder type
    if setting[1] == "rn":
        cfg["encoder"]["traj_encoder_type"] = "conv_rnn"
        cfg["encoder"]["rnn"]["feat_sz"] = int(setting[2][1:])
        cfg["encoder"]["rnn"]["depth"] = int(setting[3][1:])
        cfg["encoder"]["mlp"]["feat_sz"] = int(setting[5][1:])
        cfg["encoder"]["mlp"]["out_sz"] = int(setting[6][1:])

    elif setting[1] == "tf":
        cfg["encoder"]["traj_encoder_type"] = "transformer"
        cfg["encoder"]["transformer"]["query_sz"] = int(setting[2][1:])
        cfg["encoder"]["transformer"]["out_sz"] = int(setting[3][1:])
        cfg["encoder"]["transformer"]["attn_dropout"] = float(setting[4][2:])
        cfg["encoder"]["mlp"]["feat_sz"] = int(setting[6][1:])
        cfg["encoder"]["mlp"]["out_sz"] = int(setting[7][1:])

    elif setting[1] == 'so':
        cfg["encoder"]["traj_sz"] = 0
        cfg["encoder"]["traj_encoder_type"] = None
        cfg["encoder"]["mlp"]["feat_sz"] = int(setting[3][1:])
        cfg["encoder"]["mlp"]["out_sz"] = int(setting[4][1:])


    # Inference type
    if 'inn' in setting:
        idx = setting.index('inn')
        amortizer_fn = AmortizerForTrialData
        cfg["invertible"]["block"]["feat_sz"] = int(setting[idx+1][1:])

    elif 'pte' in setting:
        idx = setting.index('pte')
        amortizer_fn = RegressionForTrialData
        net_arch = setting[idx+1].split('x')
        cfg["linear"]["hidden_sz"] = int(net_arch[0])
        cfg["linear"]["hidden_depth"] = int(net_arch[1])
    

    return (
        amortizer_fn(config=cfg), 
        Simulator(
            model_name=sim["policy"],
            checkpt=sim["checkpt"]
        ),
        AnSPlayerDataset(mode=mode),
        AnSSimulator(mode=mode),
        mode
    )


def set_exp_env(
    amortizer:Union[AmortizerForTrialData, RegressionForTrialData],
    model_name:str,
    session:str,
    iter_num:int
):
    amort = copy.deepcopy(amortizer)
    amort.load_state_dict(
        torch.load(
            f"{PATH_AMORT_MODEL}/{model_name}/{session}/iter{iter_num:03d}.pt"
        )["model_state_dict"]
    )
    amort.eval()

    os.makedirs(PATH_AMORT_EVAL % (model_name, session, iter_num), exist_ok=True)
    return amort


def find_best_parameter(
    amortizer:Union[AmortizerForTrialData, RegressionForTrialData], 
    player_data:AnSPlayerDataset,
    param_list:list,
    convertor:AnSSimulator,
    block_use='all',
    session_use='all',
    fpath=''
):

    assert block_use in ['all', 'half']
    assert session_use in ['all', 'sub']

    result_z = {mode: {color: [] for color in COLOR} for mode in MODE}
    result_w = {mode: {color: [] for color in COLOR} for mode in MODE}

    postfix = 'fullblock' if block_use == 'all' else 'latterblock'
    postfix += '_fullsession' if session_use == 'all' else '_subsession'

    block_list = dict(all = [0, 1], half = [1])
    ses_list = dict(all = SES_NAME_ABBR, sub = SES_NAME_ABBR_SYMM)

    for mode in MODE:
        for color in COLOR:
            for tier in TIER:
                for player in PLAYER[tier]:
                    stat, traj, _ = player_data.sample(
                        player=player, 
                        mode=mode, 
                        target_color=color,
                        block_index=block_list[block_use],
                        session_id=ses_list[session_use],
                        random_sample=True
                    )
                    inferred_param = amortizer.infer(
                        stat,
                        traj_data=traj,
                        n_sample=300,
                        type='mean'
                    )
                    inferred_param = np.clip(inferred_param, -1, 1)
                    w = convertor.convert_from_output(inferred_param)[0]

                    result_z[mode][color].append(inferred_param)
                    result_w[mode][color].append(w)
            
            result_z[mode][color] = np.array(result_z[mode][color])
            result_w[mode][color] = np.array(result_w[mode][color])
    
    os.makedirs(f"{fpath}/{postfix}", exist_ok=True)
    
    pd.DataFrame(dict(zip(
        ["tier", "player", *(f"{mode[0]}_{color}_{param}" for mode in MODE for color in COLOR for param in param_list)],
        [
            ["PRO"]*10+["AMA"]*10, 
            PLAYER["PRO"] + PLAYER["AMA"], 
            *result_z['custom']['white'].T,
            *result_z['custom']['gray'].T,
            *result_z['default']['white'].T,
            *result_z['default']['gray'].T
        ]
    ))).to_csv(f"{fpath}/{postfix}/infer_z.csv", index=False)

    pd.DataFrame(dict(zip(
        ["tier", "player", *(f"{mode[0]}_{color}_{param}" for mode in MODE for color in COLOR for param in param_list)],
        [
            ["PRO"]*10+["AMA"]*10, 
            PLAYER["PRO"] + PLAYER["AMA"], 
            *result_w['custom']['white'].T,
            *result_w['custom']['gray'].T,
            *result_w['default']['white'].T,
            *result_w['default']['gray'].T
        ]
    ))).to_csv(f"{fpath}/{postfix}/infer_w.csv", index=False)



def find_best_simulation(
    player,
    param_path,
    amort_mode,
    simulator:Simulator,
    block_use='all',
    session_use='all',
    fpath='',
    repeat=5,
):

    assert block_use in ['all', 'half']
    assert session_use in ['all', 'sub']

    postfix = 'fullblock' if block_use == 'all' else 'latterblock'
    postfix += '_fullsession' if session_use == 'all' else '_subsession'

    block_list = dict(all = [0, 1], half = [1])
    ses_list = dict(all = SES_NAME_ABBR, sub = SES_NAME_ABBR_SYMM)

    # Check replication & fitts law
    # ssw 0 - ssw 1 - fsw 0 - fsw 1 - ...
    conds = {
        mode: {
            color: simulator._load_experiment_cond(
                fix_head_position=(amort_mode in ['base', 'abl1', 'abl2']),
                fix_gaze_reaction=(amort_mode in ['abl1', 'abl2']),
                fix_hand_reaction=(amort_mode in ["base"]),
                player=player,
                mode=mode,
                target_color=color,
                block_index=block_list[block_use],
                session_id=ses_list[session_use],
            ) for color in COLOR
        } for mode in MODE
    }
    exp = {
        mode: {
            color: load_experiment(
                player=player,
                mode=mode,
                target_color=color,
                block_index=block_list[block_use],
                session_id=ses_list[session_use],
            ) for color in COLOR
        } for mode in MODE
    }

    param_csv = pd.read_csv(f"{param_path}/{postfix}/infer_z.csv")
    param_csv = param_csv[param_csv["player"] == player]

    os.makedirs(f"{fpath}/{postfix}/best_simul/", exist_ok=True)
    
    sim_result = {
        mode: {
            color: dict(
                df=None,
                ts=None,
                tx=None,
                gx=None,
                ttj=None,
                gtj=None,
                ctj=None
            ) for color in COLOR
        } for mode in MODE
    }
    
    for mode in MODE:
        for color in COLOR:
            param_z = param_csv[[f"{mode[0]}_{color}_{p}" for p in simulator.env.variables]].to_numpy()[0]
            simulator.update_parameter(param_z=dict(zip(simulator.env.variables, param_z)))

            lowest_discrepancy_score = np.inf

            for _ in range(repeat):
                simulator.run_simulation_with_cond(conds[mode][color], verbose=False, overwrite_existing_simul=True)
                sim_df = simulator.export_result_df()
                ts_l, _, ttj_l, gtj_l, ctj_l, tx_l, gx_l, _, _, _ = simulator.collect_distance_info()

                discrepancy = np.sum(np.mean(
                    np.abs(
                        sim_df[discrepancy_metric[amort_mode]].to_numpy() - exp[mode][color][discrepancy_metric[amort_mode]].to_numpy()
                    ), 
                    axis=0
                ) * discrepancy_scaler[amort_mode])

                if discrepancy < lowest_discrepancy_score:
                    lowest_discrepancy_score = discrepancy
                    sim_result[mode][color]["df"] = sim_df
                    sim_result[mode][color]["ts"] = ts_l
                    sim_result[mode][color]["tx"] = tx_l
                    sim_result[mode][color]["gx"] = gx_l
                    sim_result[mode][color]["ttj"] = ttj_l
                    sim_result[mode][color]["gtj"] = gtj_l
                    sim_result[mode][color]["ctj"] = ctj_l


    ### DRAWING SESSION
    fig, axs = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)

    # Fitts law - white and gray
    r2s = []
    for i in range(2):
        if i == 0:
            sim_data = sim_result[mode]['white']["df"].copy()
            sim_data = sim_data[sim_data["target_speed"] == 0]
            exp_data = exp[mode]['white'].copy()
            exp_data = exp_data[exp_data["target_speed"] == 0]
        else:
            sim_data = sim_result[mode]['gray']["df"].copy()
            sim_data = sim_data[sim_data["target_speed"] == 0]
            exp_data = exp[mode]['gray'].copy()
            exp_data = exp_data[exp_data["target_speed"] == 0]

        stid = sim_data["t_iod"].to_numpy()
        stct = sim_data[M_TCT].to_numpy()
        sres = np_groupby(np.vstack((stid, stct)).T, key_pos=0, lvl=6)
        r2s.append(get_r_squared(sres[:,0], sres[:,1]))

        etid = exp_data["t_iod"].to_numpy()
        etct = exp_data[M_TCT].to_numpy()
        eres = np_groupby(np.vstack((etid, etct)).T, key_pos=0, lvl=6)

        sns.regplot(
            x=sres[:,0], y=sres[:,1],
            scatter_kws={"color": "black", "alpha": 0},
            line_kws={"color": ["red", 'darkred'][i], "lw": 2.5},
            ax=axs[0][0]
        )
        axs[0][0].scatter(
            sres[:,0], sres[:,1], 
            marker='o', edgecolor='black', linewidth=1, 
            facecolor=['white', 'gray'][i], 
            s=40, alpha=0.5, zorder=100
        )
        sns.regplot(
            x=eres[:,0], y=eres[:,1],
            scatter_kws={"color": "black", "alpha": 0},
            line_kws={"color": ["red", 'darkred'][i], "lw": 1.5, "alpha": 0.35},
            ax=axs[0][0]
        )
        axs[0][0].scatter(
            eres[:,0], eres[:,1],
            marker='D', edgecolor='black', linewidth=1, 
            facecolor=['white', 'gray'][i], 
            s=40, alpha=0.35, zorder=60
        )
    axs[0][0].legend(
        [Line2D([0], [0], color="red", lw=2.5), Line2D([0], [0], color="darkred", lw=2.5)], 
        [f"white $R^2={r2s[0]:.2f}$", f"gray $R^2={r2s[1]:.2f}$"], 
        fontsize=8, loc="lower right"
    )


    # Distance
    ets, etx, egx = (
        exp_dist[block_use][session_use][player]["ts"],
        exp_dist[block_use][session_use][player]["tx"],
        exp_dist[block_use][session_use][player]["gx"]
    )
    ts_list, tx_list, gx_list = list(), list(), list()
    mtct = []
    for mode in MODE:
        for color in COLOR:
            ts_list += sim_result[mode][color]["ts"]
            tx_list += sim_result[mode][color]["tx"]
            gx_list += sim_result[mode][color]["gx"]
            mtct.append(sim_result[mode][color]["df"][M_TCT].to_numpy())
    mtct = np.concatenate(mtct).mean()
    sts, stx, sgx = mean_distance(ts_list, tx_list, gx_list, mtct)

    axs[1][0].plot(ets, etx, linestyle='-', linewidth=0.8, color='k', label='T-X (Exp.)', zorder=30)
    axs[1][0].plot(ets, egx, linestyle='--', linewidth=0.8, color='k', label='G-X (Exp.)', zorder=30)
    axs[1][0].plot(sts, stx, linestyle='-', linewidth=0.8, color='r', label='T-X (Sim.)', zorder=30)
    axs[1][0].plot(sts, sgx, linestyle='--', linewidth=0.8, color='r', label='G-X (Sim.)', zorder=30)
    set_tick_and_range(axs[1][0], 0.1, 1000, max_value=1, axis='x', omit_tick=2)
    set_tick_and_range(axs[1][0], 0.02, 100, max_value=0.12, axis='y')
    axs[1][0].grid()
    axs[1][0].legend()
    axs[1][0].set_xlabel("Time (ms)")
    axs[1][0].set_ylabel("Distance (cm)")

    sim_merged = pd.concat([sim_result[mode][color]["df"] for mode in MODE for color in COLOR])
    exp_merged = pd.concat([exp[mode][color] for mode in MODE for color in COLOR])
    tid = exp_merged["t_iod"].to_numpy()
    tid_label = discrete_labeling(tid, lvl=4)

    for i, mt in enumerate([M_TCT, M_ACC, M_GD]):
        ax = axs[0][i+1]

        data = pd.DataFrame(
            dict(zip(
                ["target_color", "target_speed", "t_iod", "exp", "sim"],
                [
                    exp_merged["target_color"].to_list(),
                    exp_merged["target_speed"].to_numpy(),
                    tid_label,
                    exp_merged[mt].to_numpy(),
                    sim_merged[mt].to_numpy()
                ]
            ))
        ).groupby(["target_color", "target_speed", "t_iod"], as_index=False).mean()
        data = data[["exp", "sim"]].to_numpy()

        r_squared = get_r_squared(data[:,0], data[:,1])
        max_v = max(max(data[:,0]), max(data[:,1]))
        min_v = min(min(data[:,0]), min(data[:,1]))
        d_val = (max_v - min_v) * 0.1
        label = f"$R^2={r_squared:.2f}$"

        sns.regplot(
            x=data[:,0],
            y=data[:,1],
            scatter_kws={"color": "black", "alpha": 0.3},
            line_kws={"color": "red", "lw": 2.5},
            ax=ax
        )
        ax.plot(
            [min_v - d_val, max_v + d_val],
            [min_v - d_val, max_v + d_val],
            color="gray",
            linestyle="--"
        )

        rng, ticks, ticks_lb = auto_set_tick_and_range(
            max_v, min_v, [1000, 100, 1][i], [0.1, 0.1, 1][i], offset=(0.1, 0.1), fmt='%d'
        )
        ax.set_title(mt)
        ax.set_xlim(*rng)
        ax.set_ylim(*rng)
        ax.set_xticks(ticks, ticks_lb)
        ax.set_yticks(ticks, ticks_lb)

        ax.set_xlabel("Player")
        ax.set_ylabel("Simulator")
        ax.legend(
            [Line2D([0], [0], color="red", lw=2.5)],
            [label],
            fontsize=8, loc='lower right'
        )
        ax.grid(linestyle='--', linewidth=0.5)
        ax.set_aspect('equal')
    
    for i, mt in enumerate([M_TCT, M_SE_NORM, M_GD]):
        ax = axs[1][i+1]

        sim_data = sim_merged[mt].to_numpy()
        exp_data = exp_merged[mt].to_numpy()

        sns.kdeplot(
            data=exp_data,
            color='red',
            linewidth=2,
            fill=False,
            ax=ax
        )
        sns.kdeplot(
            data=sim_data,
            color='blue',
            linewidth=2,
            fill=False,
            ax=ax
        )
        ax.axvline(np.mean(exp_data), linewidth=2, color='red', linestyle='--')
        ax.axvline(np.mean(sim_data), linewidth=2, color='blue', linestyle='--')
        ax.set_xlim(0, [1.5, 5, 20][i])
        ax.legend(
            [Line2D([0], [0], color="red", lw=2), Line2D([0], [0], color="blue", lw=2)],
            ["Player", "Simulator"],
            fontsize=8, loc='upper right'
        )
        
    fig_save(f"{fpath}/{postfix}/best_simul", f"{player}", DPI=300, save_svg=False)
    pickle_save(f"{fpath}/{postfix}/best_simul/{player}.pkl", sim_result)



def cross_validation_inference(
    amortizer:Union[AmortizerForTrialData, RegressionForTrialData], 
    player_data:AnSPlayerDataset,
    param_list:list,
    convertor:AnSSimulator,
    n_validate=10,
    fpath=''
):
    os.makedirs(f"{fpath}cross_valid/", exist_ok=True)

    result_z = {player: {mode: {color: [] for color in COLOR} for mode in MODE} for player in PLAYERS}
    result_w = {player: {mode: {color: [] for color in COLOR} for mode in MODE} for player in PLAYERS}

    for _ in range(n_validate):
        # Inference
        for mode in MODE:
            for color in COLOR:
                for tier in TIER:
                    for player in PLAYER[tier]:
                        stat, traj, _, tr_idx, vd_idx = player_data.sample(
                            cross_valid=True, player=player, mode=mode, target_color=color
                        )
                        inferred_param = amortizer.infer(
                            stat[tr_idx],
                            traj_data=traj[tr_idx],
                            n_sample=300,
                            type='mode'
                        )
                        inferred_param = np.clip(inferred_param, -1, 1)

                        result_z[player][mode][color].append((dict(zip(param_list, inferred_param)), vd_idx))

                        w = convertor.convert_from_output(inferred_param)[0]

                        result_w[player][mode][color].append(w)

    
    pickle_save(f"{fpath}cross_valid/cross_valid_inference.pkl", (result_z, result_w))


def cross_validation_simulation(
    simulator:Simulator,
    player,
    fpath=''
):

    os.makedirs(f"{fpath}cross_valid/{player}", exist_ok=True)
    
    result_z, _ = pickle_load(f"{fpath}cross_valid/cross_valid_inference.pkl")
    result_z = result_z[player]

    n_validate = len(result_z['custom']['white'])

    for vtrial in range(n_validate):
        sim_result = {
            mode: {
                color: dict(
                    df=None,
                    ts=None,
                    tx=None,
                    gx=None,
                    ttj=None,
                    gtj=None,
                    ctj=None,
                    trial=None
                ) for color in COLOR
            } for mode in MODE
        }

        for mode in MODE:
            for color in COLOR:
                param_z, trial_idx = result_z[mode][color][vtrial]
                simulator.update_parameter(param_z=param_z)
                conds = simulator._load_experiment_cond(
                    player=player,
                    mode=mode,
                    target_color=color
                )
                conds = list(np.array(conds, dtype=object)[trial_idx])
                simulator.run_simulation_with_cond(conds, verbose=False, overwrite_existing_simul=True)
                sim_df = simulator.export_result_df().to_numpy()

                sim_result[mode][color]["df"] = sim_df
                ts_l, _, ttj_l, gtj_l, ctj_l, tx_l, gx_l, _, _, _ = simulator.collect_distance_info()
                sim_result[mode][color]["ts"] = ts_l
                sim_result[mode][color]["tx"] = tx_l
                sim_result[mode][color]["gx"] = gx_l
                sim_result[mode][color]["ttj"] = ttj_l
                sim_result[mode][color]["gtj"] = gtj_l
                sim_result[mode][color]["ctj"] = ctj_l
                sim_result[mode][color]["trial"] = trial_idx
        
        pickle_save(f"{fpath}cross_valid/{player}/simulation{(vtrial+1):02d}.pkl", sim_result)

        # Draw - Binning, histogram, 




model_list = {
    "full_tf_q16_o4_do0.4-mlp_f128_o64-pte_128x2-tr_it2048_b64_n64_full_231219_165740_73_20000000" : {
        "0107_192443": [100],
    },
    "base_tf_q16_o4_do0.4-mlp_f128_o64-pte_128x2-tr_it2048_b64_n64_base_240104_164252_68_20000000" : {
        "0117_144253": [100],
    },
    "abl1_tf_q16_o4_do0.4-mlp_f128_o64-pte_128x2-tr_it2048_b64_n64_full_231219_165740_73_20000000": {
        "0108_145616": [100],
    },
    "abl2_tf_q16_o4_do0.4-mlp_f128_o64-pte_128x2-tr_it2048_b64_n64_full_231219_165740_73_20000000": {
        "0108_145704": [100],
    },
    "abl3_tf_q16_o4_do0.4-mlp_f128_o64-pte_128x2-tr_it2048_b64_n64_full_231219_165740_73_20000000": {
        "0108_174146": [100]
    }
}


def evaluate_model(skip_inference=True, cpu=12, repeat=5, block_list=["all", "half"], session_list=["all", "sub"]):
    for model in model_list.keys():
        amortizer, simulator, player_data, convertor, infer_mode = get_amortizer_and_simulator(model)
        for session in model_list[model].keys():
            for iter in model_list[model][session]:
                amortizer = set_exp_env(amortizer, model, session, iter)
                for block_use in block_list:
                    for session_use in session_list:
                        if not skip_inference:
                            find_best_parameter(
                                amortizer, 
                                player_data, 
                                simulator.env.variables, 
                                convertor,
                                block_use=block_use,
                                session_use=session_use,
                                fpath=f"{PATH_AMORT_EVAL % (model, session, iter)}"
                            )
                        def best_simul(player):
                            find_best_simulation(
                                player,
                                amort_mode=infer_mode,
                                simulator=simulator,
                                block_use=block_use,
                                session_use=session_use,
                                param_path=PATH_AMORT_EVAL % (model, session, iter),
                                fpath=PATH_AMORT_EVAL % (model, session, iter),
                                repeat=repeat
                            )
                        Parallel(n_jobs=cpu)(delayed(best_simul)(player) for player in tqdm(PLAYERS))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Fitting')

    parser.add_argument('--skip_infer', type=bool, default=False)
    parser.add_argument('--simul_repeat', type=int, default=5)
    parser.add_argument('--cpu', type=int, default=12)

    parser.add_argument('--block_run', type=int, default=2)
    parser.add_argument('--ses_run', type=int, default=0)

    args = parser.parse_args()

    if args.block_run == 0: block_list = ["all", "half"]
    elif args.block_run == 1: block_list = ["all"]
    elif args.block_run == 2: block_list = ["half"]

    if args.ses_run == 0: session_list = ["all", "sub"]
    elif args.ses_run == 1: session_list = ["all"]
    elif args.ses_run == 2: session_list = ["sub"]

    evaluate_model(
        skip_inference=args.skip_infer, 
        repeat=args.simul_repeat, 
        cpu=args.cpu,
        block_list=block_list, 
        session_list=session_list
    )