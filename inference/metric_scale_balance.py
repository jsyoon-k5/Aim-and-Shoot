import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from typing import List

import sys, glob
sys.path.append("..")

from configs.simulation import COG_SPACE
from configs.common import (
    M_TCT, M_GD, M_SE
)
from configs.path import PATH_INFER_RANDOM
from utilities.utils import now_to_string
from agent.ans_simulation import *
from expdata.data_process import load_experiment


def sample_difference(
    model: str = "cog_230601_194710_66",
    variables: List[str] = COG_SPACE,
    metrics: List[str] = [M_TCT, M_SE, M_GD],
):
    
    exp_data = load_experiment(
        tier="AMA",
        # player="KKW", 
        mode="default", 
        target_color="white",
        block_index=0,
        # session_id ="ssw"
    )

    # Get all target conditions
    cond_list = [
        dict(
            pcam=_pcam,
            tgpos=sp2ct(*_tgpos),
            toax=_toax,
            gpos=_gpos,
            session=_sess
        ) for _pcam, _tgpos, _toax, _gpos, _sess in zip(
            exp_data[["pcam0_az", "pcam0_el"]].to_numpy(), 
            exp_data[["tgpos0_az", "tgpos0_el"]].to_numpy(), 
            exp_data[["toax_az", "toax_el"]].to_numpy(), 
            exp_data[["gaze0_x", "gaze0_y"]].to_numpy(), 
            exp_data["session_id"].to_list()
        )
    ]

    simulator = Simulator(
        model_name=model,
        modulated=True,
        param_z={v:0 for v in variables}
    )
    simulator.set_random_z_sample()
    simulator.run_simulation_with_cond(cond_list, output_msg=True)
    res = simulator.export_result_dict(keys=["session"])

    exp_result = exp_data[metrics].to_numpy()
    sim_result = pd.DataFrame(res)[metrics].to_numpy()
    diff = exp_result - sim_result

    data = {}
    for i, m in enumerate(metrics):
        data[m] = diff[:,i]
    
    pd.DataFrame(data).to_csv(PATH_INFER_RANDOM % (model, now_to_string()), index=False)


def get_balancing(
    model: str = "cog_230601_194710_66",
    metrics: List[str] = [M_TCT, M_SE, M_GD]
):
    sample_files = glob.glob(PATH_INFER_RANDOM % (model, "*"))
    data = pd.concat([pd.read_csv(f) for f in sample_files])
    data = np.abs(data[metrics].to_numpy())
    mm = np.min(data, axis=0)
    data -= np.min(data, axis=0) - 1e-6
    diff_mean = np.mean(data, axis=0)
    mean_scaler = max(diff_mean) / diff_mean
    data *= mean_scaler

    _, lmbda = sp.stats.boxcox(data.ravel())
    data_bc = (data ** lmbda - 1) / lmbda

    for i, m in enumerate(metrics):
        plt.hist(data_bc[:,i], bins=100, density=True, histtype='step', label=m)
    plt.legend()
    plt.show()

    return mean_scaler, lmbda


if __name__ == "__main__":
    # sample_difference()
    ms, lmbda = get_balancing()
    print(ms)
    print(lmbda)