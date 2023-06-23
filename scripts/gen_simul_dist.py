import numpy as np

import sys, argparse, glob
sys.path.append("..")

from utilities.utils import pickle_load, now_to_string
from utilities.plots import *
from configs.path import PATH_TEMP
from agent.ans_simulation import Simulator


# parser = argparse.ArgumentParser(description='Gaze speed')
# parser.add_argument('--gaze', type=str, default="fast")
# args = parser.parse_args()


def gen_simul_pkl():
    simulator = Simulator(
        model_name="cog_230530_202928_90",
        modulated=True,
        param_z=dict(
            theta_s=0, theta_p=0, theta_m=0, theta_q=0
        )
    )

    conds = simulator._load_experiment_cond(
        player='KKW', mode='default', target_color='white', block_index=0, exclude_invalid=False
    )

    for i in range(125):
        print(f"{i:03d}")
        fixed_z = np.random.uniform(-1, 1, size=3)

        # Fast
        simulator.update_parameter(
            param_z=dict(
                theta_s=fixed_z[0], 
                theta_p=fixed_z[1], 
                theta_m=fixed_z[2],
                theta_q=1
            )
        )
        simulator.run_simulation_with_cond(conds)
        simulator.collect_distance_info(save_pkl=f"{PATH_TEMP}fast_{now_to_string()}.pkl")
        simulator.clear_record()

        # Slow
        simulator.update_parameter(
            param_z=dict(
                theta_s=fixed_z[0], 
                theta_p=fixed_z[1], 
                theta_m=fixed_z[2],
                theta_q=-1
            )
        )
        simulator.run_simulation_with_cond(conds)
        simulator.collect_distance_info(save_pkl=f"{PATH_TEMP}slow_{now_to_string()}.pkl")
        simulator.clear_record()


def draw_traj(g):
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    axs.set_title(g)

    files = glob.glob(f"{PATH_TEMP}{g}_*.pkl")
    maxtime = 0
    maxd = 0
    for f in files:
        _, _, _, _, _, _, ts, tx, gx = pickle_load(f)
        if ts[-1] > maxtime: maxtime = ts[-1]
        if np.max(tx) > maxd: maxd = np.max(tx)
        if np.max(gx) > maxd: maxd = np.max(gx)

        axs.plot(ts, tx, color='b', linewidth=0.5, alpha=0.05)
        axs.plot(ts, gx, color='r', linewidth=0.5, alpha=0.05)

    set_tick_and_range(axs, 0.1, 1000, 0.9, axis='x')
    set_tick_and_range(axs, 0.01, 100, maxd, axis='y')
    
    axs.set_xlabel("Time (millisec)")
    axs.set_ylabel("Distance (cm)")

    plt.show()


# gen_simul_pkl()
draw_traj('fast')