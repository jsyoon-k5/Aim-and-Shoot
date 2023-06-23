"""
Parameter inference through Pattern Search algorithm
"""

import numpy as np
import pandas as pd
from typing import List

from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

import sys, time
sys.path.append("..")

from configs.common import (
    M_TCT, M_GD, M_SE
)
from configs.simulation import COG_SPACE
from configs.experiment import *
from configs.path import PATH_INFER_RESULT
from utilities.utils import now_to_string, list2str
from utilities.plots import plt_model_fitting_r2
from agent.ans_simulation import *
from expdata.data_process import load_experiment

def ans_infer(
    player,
    mode: str = "default",
    tcolor: str = "white",
    model: str = "cog_230601_194710_66",
    variables: List[str] = COG_SPACE,
    metrics: List[str] = [M_TCT, M_SE, M_GD],
    scale_balancing: np.ndarray = np.array([19.892, 561.701, 1.0]),     # use metric_scale_balance.py
    lmbda: float = 0.28167,
    weight: np.ndarray = np.array([1.0, 1.0, 1.2])
):
    assert len(metrics) == len(scale_balancing) and len(metrics) == len(weight)
    assert tcolor in COLOR

    tag = now_to_string()

    # Load corresponding experiment data
    exp_data = load_experiment(
        player=player,
        mode=mode,
        target_color=tcolor,
        
        target_speed = 0
    )

    # Target conditions to simulate
    cond_list = [
        dict(
            tmpos=_tmpos,
            toax=_toax,
            session=_sess
        ) for _tmpos, _toax, _sess in zip(
            exp_data[["tmpos0_x", "tmpos0_y"]].to_numpy(), 
            exp_data[["toax_az", "toax_el"]].to_numpy(), 
            exp_data["session_id"].to_list()
        )
    ]

    # Assign group number for binning
    # Session * target initial distance
    gindex_ses = np.array([SES_NAME_ABBR.index(sn) for sn in exp_data["session_id"].to_list()])
    gindex_td0 = np.array(exp_data["tmpos0_dist"].to_numpy() > TARGET_DIST_MEDIAN).astype(int)
    gindex = 2 * gindex_ses + gindex_td0

    exp_result = exp_data[metrics].to_numpy()

    # Make dir
    file_path = PATH_INFER_RESULT % (mode, tcolor)
    os.makedirs(file_path, exist_ok=True)

    # Save condition and ground truth
    gt = dict(
        session_id = exp_data["session_id"].to_list(),
        tmpos0_x = exp_data["tmpos0_x"].to_numpy(),
        tmpos0_y = exp_data["tmpos0_y"].to_numpy(),
        toax_az = exp_data["toax_az"].to_numpy(),
        toax_el = exp_data["toax_el"].to_numpy(),
        group_index = gindex
    )
    for m in metrics:
        gt[m] = exp_data[m].to_numpy()
    pd.DataFrame(gt).to_csv(f"{file_path}/gt_{player}.csv", index=False)

    # Prepare Model Fitting log file
    filename = f"{file_path}/mf_{player}_{tag}.csv"
    with open(filename, 'w') as f:
        f.write(",".join(variables))
        for tn in range(gindex.size):
            for m in metrics:
                f.write(f",{m}_{tn}")
        for m in metrics:
            f.write(f",{m}_R2")
        for m in metrics:
            f.write(f",{m}_loss")
        f.write("\n")

    
    # Prepare simulator
    simulator = Simulator(
        model_name=model,
        modulated=True,
        param_z={v:0 for v in variables}
    )

    # Begin timer
    start_t = time.time()


    # Define discrepancy
    def loss_func(sr, er, lmbda=lmbda):
        """
        Loss value of simulation result and experiment result
        Both are 2D np array with shape (N, len(metrics)).
                             mt1    mt2    mt3  ...
        trials (400~500)      v1     v2     v3  ...
        """

        diff = np.abs(sr - er) * scale_balancing
        diff = (diff ** lmbda - 1) / lmbda
        loss = np.sum(diff, axis=0) * weight

        return loss
    
    def discrepency(z):
        simulator.update_parameter(param_z={_v:_z for _v, _z in zip(variables, z)})
        simulator.run_simulation_with_cond(
            cond_list,
            limit_num_of_exclusion=50
        )
        res = simulator.export_result_dict(keys=["session"])
        simulator.clear_record()
        simul_result = np.array([res[m] for m in metrics]).T

        R2s = plt_model_fitting_r2(
            metrics, exp_result, simul_result, gindex, 
            file_path, f"plot_running_{player}_{tag}", save_svg=False
        )
        loss = loss_func(simul_result, exp_result)

        with open(filename, 'a') as f:
            f.write(list2str(z))
            f.write(f',{list2str(simul_result.reshape(gindex.size * len(metrics)))},{list2str(R2s)},{list2str(loss)}\n')
        
        print(f"Iter {discrepency.counter: 3d} - time: {int(time.time() - start_t): 5d}s", end=' | ')
        print(("%+.3f "*len(variables)) % tuple(z) + ("%.1f " * len(metrics)) % tuple(loss), end=' ... ')
        print(f"loss: {np.mean(loss):+.3f}")
        discrepency.counter += 1
        
        return np.sum(loss)
    
    discrepency.counter = 0


    # Pattern Search algorithm
    class SimulationFitting(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=len(variables), n_obj=1, xl=-1, xu=1)
        
        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = discrepency(np.array(x))

    print(f"Optimization for player: {player} ({mode}, {tcolor})")

    problem = SimulationFitting()
    algorithm = PatternSearch(
        n_sample_points=250,
        #n_sample_points =1,
        init_delta=0.25,
        init_rho=0.85,
        step_size=1.0
    )
    #termination = get_termination("n_eval", 2)

    print("Initialization Complete. Running optimization ...")
    res = minimize(
        problem, 
        algorithm, 
        #termination, 
        seed=1, 
        verbose=False
    )
    x_opt = res.X
    fx_opt = res.F[0]

    print(f"Opt Params: {x_opt}")
    print(f"Opt Results: {fx_opt:.6f}")
    print(f"Elapsed Time: {(time.time() - start_t):.2f} (s)\n")


    # Run final simulation
    with open(filename, 'a') as f:
        f.write(",".join(list(map(str, x_opt))))

        simulator.update_parameter(param_z={_v:_z for _v, _z in zip(variables, x_opt)})
        simulator.run_simulation_with_cond(cond_list)
        res = simulator.export_result_dict(keys=["session"])
        simul_result = np.array([res[mt] for mt in metrics]).T

        f.write(',' + list2str(simul_result.reshape(gindex.size * len(metrics))))

        r2s = plt_model_fitting_r2(
            metrics, exp_result, simul_result, gindex, 
            file_path, f"plot_optimal_{player}_{tag}",
            save_svg=False
        )
        f.write(f",{list2str(r2s)},{fx_opt}\n")
    
    # Save trajectory info
    traj_data = simulator.collect_trial_trajectory()
    dist_data = simulator.collect_distance_info()
    with open(f"{file_path}/traj_{player}_{tag}.pkl", "wb") as fp:
        pickle.dump(
            (
                traj_data,
                dist_data
            ), fp
        )
