"""
Aim-and-Shoot simulator for amortized inference
Orignial code structure written by Hee-seung Moon (https://github.com/hsmoon121/amortized-inference-hci)

Code modified by June-Seop Yoon
"""

import enum
import os, sys
from time import time
import numpy as np
from copy import deepcopy
from tqdm import tqdm

sys.path.append("..")

from configs.simulation import *
from configs.amort import *
from utils.mymath import linear_denormalize, linear_normalize, log_denormalize
from agent.agent_simulation import Simulator
from agent.fps_task import GameState

class AnSSimulator(object):
    """
    A simulator of aim-and-shoot behavior
    Code wrapped to run amortized inference
    """
    def __init__(self, config=None, mode='full'):
        assert mode in ["full", "base", "abl0", "abl1", "abl2", "abl3"]

        self.mode = mode

        if config is None:
            config = deepcopy(default_ans_config[self.mode]["simulator"])
        
        self.config = config
        
        self.simulator = Simulator(
            model_name=config["policy"],
            checkpt=config["checkpt"]
        )
        self.simulator.clear_record()
        self.game = GameState()
        
        self.targeted_params = deepcopy(config["targeted_params"])
        assert set(self.targeted_params) == set(self.simulator.env.variables)

        # self.target_data_list = BEHAVIOR_DATA[self.mode]
        # self.value_range = VALUE_RANGE[self.mode]


    def simulate(
        self,
        n_param=1,
        sim_per_param=1,    # Number of simulation per (param, task)
        fixed_params=None,
        verbose=False
    ):
        """
        Simulates human behavior based on given (or sampled) free parameters

        Arguments (Inputs):
        - n_param: no. of parameter sets to sample to simulate (only used when fixed_params is not given)
        - task_per_param: no. of task condition sets to sample to simulate
        - sim_per_param: no. of simulation per parameter set
        - fixed_params: dict:numpy.ndarray of free parameter (see below) sets to simulate (z-value)
        =======
        Free params in aim-and-shoot model
        1) theta_m: log-uniform (min=0, max=0.5)
        2) theta_p: log-uniform (min=0, max=0.5)
        3) theta_s: log-uniform (min=0, max=0.5)
        4) theta_c: log-uniform (min=0, max=0.5)
        
        5) hit:      log-uniform (min=1, max=64)
        6) miss:     log-uniform (min=1, max=64)
        6) hit_decay:    log-uniform (min=0.05, max=0.95)
        7) miss_decay:   log-uniform (min=0.05, max=0.95)
        =======
        - fixed_initial_cond: fix task environment

        Outputs:
        - lognorm_params: free parameter sets (with log-normalized values) used for simulation
            > ndarray with size ((n_param), (dim. of free parameters))
        - stats: static (fixed-size) behavioral outputs for every trial (see below)
            > ndarray with size ((n_param), (sim_per_param), (dim. of static behavior))
        =======
        Static behavioral output (normalized)
        1) trial completion time
        2) normalized shoot error
        3) glancing distance                -> excluded in baseline

        4) initial target position (2D)
        5) target orbit axis (2D)
        6) target angular speed
        7) target radius
        8) gaze reaction time               -> excluded in baseline
        9) hand reaction time
        10) eye position (2D)               -> excluded in baseline
        =======
        - trajs: trajectory (variable-size) for every trial (see below)
        =======
        Data for each timstep in trajectory data (normalized)
        1) time difference from prev. timestep
        2) target position (2D)
        3) gaze position (2D)               -> excluded in baseline
        4) camera angle (2D)
        =======
        """
        if fixed_params is not None:
            lognorm_params = fixed_params
            n_param = lognorm_params.shape[0]
        else:
            lognorm_params = np.random.uniform(low=-1, high=1, size=(n_param, len(self.targeted_params)))

        stats, trajs = [], []
        for i in (tqdm(range(n_param)) if verbose else range(n_param)):
            _z = lognorm_params[i]
            self.simulator.update_parameter(param_z_raw=_z)
            self.simulator.run_simulation_with_cond(
                game_cond=self.game.sample_task_condition(
                    sim_per_param,
                    fix_to_experiment_session=True,     # 5 levels of speed * radius
                    head_pos = HEAD_POSITION if self.mode == 'base' else None,
                    hrt = 0.2 if self.mode == 'base' else None
                )
            )
            stat_result = self.simulator.export_result_df(include_target_cond=True)
            traj_result = self.simulator.collect_trial_trajectory_downsample(
                downsample=40, normalize=True,
                # include_gaze=False if self.mode == 'base' else True
            )
            self.simulator.clear_record()
            stat_result = linear_normalize(
                stat_result[COMPLETE_SUMMARY].to_numpy(), 
                *np.array([VALUE_RANGE[metric] for metric in COMPLETE_SUMMARY]).T
            )

            if sim_per_param == 1:
                stats.append(stat_result[0])
                trajs.append(traj_result[0])
            else:
                stats.append(stat_result)
                trajs.append(traj_result)

        return lognorm_params, np.array(stats, dtype=np.float32), trajs


    def convert_from_output(self, outputs):
        """
        outputs : (batch, param_sz) or (param_sz,) output results (log-normalized free params)
        """
        param_in = deepcopy(outputs)
        if len(np.array(param_in).shape) == 1:
            param_in = np.expand_dims(param_in, axis=0)

        param_out = np.zeros_like((param_in))
        for i, v in enumerate(self.targeted_params):
            param_out[:,i] = log_denormalize(
                param_in[:,i], 
                self.simulator.env_config["params_min"][v], 
                self.simulator.env_config["params_max"][v]
            )
        return param_out  


# class AnSSimulatorParameterMasking(object):
#     """
#     This simulator will treat Cognitive parameters as task initial condition
#     """
#     def __init__(self, config=None, mode='full', masking='cog'):
#         assert mode in ["full", "base", "abl1", "abl2"]
#         assert masking in ["cog", "rew"]

#         self.mode = mode

#         if config is None:
#             config = deepcopy(default_ans_config[self.mode]["simulator"])
        
#         self.config = config
        
#         self.simulator = Simulator(
#             model_name=config["policy"],
#             checkpt=config["checkpt"]
#         )
#         self.simulator.clear_record()
#         self.game = GameState()
        
#         if masking == 'cog':
#             self.targeted_params = deepcopy(config["rew_params"])
#             self.masked_params = deepcopy(config["cog_params"])
#         elif masking == 'rew':
#             self.targeted_params = deepcopy(config["cog_params"])
#             self.masked_params = deepcopy(config["rew_params"])


#     def simulate(
#         self,
#         n_param=1,
#         sim_per_param=1,    # Number of simulation per (param, task)
#         verbose=False
#     ):
#         lognorm_params = np.random.uniform(low=-1, high=1, size=(n_param, len(self.targeted_params)))

#         stats, trajs = [], []
#         for i in (tqdm(range(n_param)) if verbose else range(n_param)):
#             target_z = lognorm_params[i]
#             mask_z = np.random.uniform(low=-1, high=1, size=(sim_per_param, len(self.masked_params)))
#             params = [
#                 dict(zip(self.targeted_params + self.masked_params, np.concatenate((target_z, mask_z[j])))) \
#                     for j in range(sim_per_param)
#             ]
#             conds = self.game.sample_task_condition(
#                 sim_per_param,
#                 fix_to_experiment_session=True,
#                 head_pos = HEAD_POSITION if self.mode == 'base' else None
#             )
#             self.simulator.run_simulation_with_cond(
#                 game_cond=conds,
#                 param_list=params
#             )
#             stat_result = self.simulator.export_result_df(include_target_cond=True)
#             traj_result = self.simulator.collect_trial_trajectory_downsample(
#                 downsample=40, normalize=True,
#                 # include_gaze=False if self.mode == 'base' else True
#             )
#             self.simulator.clear_record()
#             stat_result = linear_normalize(
#                 stat_result[COMPLETE_SUMMARY].to_numpy(), 
#                 *np.array([VALUE_RANGE[metric] for metric in COMPLETE_SUMMARY]).T
#             )

#             if sim_per_param == 1:
#                 stats.append(np.concatenate((stat_result[0], mask_z[0])))
#                 trajs.append(traj_result[0])
#             else:
#                 stats.append(np.concatenate((stat_result, mask_z), axis=1))
#                 trajs.append(traj_result)

#         return lognorm_params, np.array(stats, dtype=np.float32), trajs


#     def convert_from_output(self, outputs):
#         """
#         outputs : (batch, param_sz) or (param_sz,) output results (log-normalized free params)
#         """
#         param_in = deepcopy(outputs)
#         if len(np.array(param_in).shape) == 1:
#             param_in = np.expand_dims(param_in, axis=0)

#         param_out = np.zeros_like((param_in))
#         for i, v in enumerate(self.targeted_params):
#             param_out[:,i] = log_denormalize(
#                 param_in[:,i], 
#                 self.simulator.env_config["params_min"][v], 
#                 self.simulator.env_config["params_max"][v]
#             )
#         return param_out  
    



if __name__ == "__main__":
    x = AnSSimulator()

    params = np.random.uniform(-1, 1, (50, 7))

    from time import time

    start_t = time()
    p, s, t = x.simulate(50, 16, fixed_params=params, verbose=True)
    print(time() - start_t)

    x = AnSSimulator()

    from time import time

    start_t = time()
    p, s, t = x.simulate(50, 16, fixed_params=params, verbose=True)
    print(time() - start_t)

    pass