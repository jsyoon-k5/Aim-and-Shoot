"""
Dataset generator for amortized inference
Orignial code written by Hee-seung Moon (https://github.com/hsmoon121/amortized-inference-hci)

Code modified by June-Seop Yoon
"""

import os, sys, glob, random
from copy import deepcopy
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from datetime import datetime
from time import time

sys.path.append("..")

from amortizer.ans_simulator import AnSSimulator
from experiment.data_process import load_experiment, load_trajectory
from utils.mymath import np_interp_nd, linear_normalize, linear_denormalize
from utils.utils import pickle_load, pickle_save, now_to_string
from configs.amort import *
from configs.common import *
from configs.experiment import *
from configs.path import PATH_AMORT_SIM_DATASET, PATH_DATA_SUMMARY


class AnSPlayerDataset(object):
    """
    The class 'AnSPlayerDataset' handles the creation and retrieval of an empirical dataset for Aim-and-Shoot tasks.
    See expdata folder to view raw experiment data
    """
    def __init__(self, mode='full', downsample_rate=40):
        assert mode in ["full", "base", "abl0", "abl1", "abl2", "abl3"]
        self.mode = mode
        fpath = f"{PATH_DATA_SUMMARY}amort_player_data.pkl"

        self.downsample = downsample_rate

        if not os.path.exists(fpath):
            expdata = load_experiment()
            self.task_data = expdata[[
                "tier", 
                "player",
                "mode",
                "session_id",
                "block_index",
                "trial_index",
                "target_speed", 
                "target_radius",
                "target_color"
            ]].reset_index(drop=True)
            self.stat_data = list()     # list of np.ndarray; np.ndarray := [trials][stat_info]
            self.traj_data = list()     # list of (list of np.ndarray)
            
            self._get_stat_data(expdata)
            self._get_traj_data(expdata)

            pickle_save(fpath, (self.task_data, self.stat_data, self.traj_data))
        else:
            (self.task_data, self.stat_data, self.traj_data) = pickle_load(fpath)
        
        # Processing
        self.stat_data = self.stat_data[:,[COMPLETE_SUMMARY.index(metric) for metric in SUMMARY_DATA[self.mode]]]
        for i in range(len(self.traj_data)):
            self.traj_data[i] = self.traj_data[i][:, [COMPLETE_TRAJECTORY.index(metric) for metric in TRAJECTORY_DATA[self.mode]]]

        print(f"[ player dataset ] loaded ({self.mode})")


    def _get_stat_data(self, expdata:pd.DataFrame):
        """
        Read static (fixed-size) behavioral outputs for every trial from experiment CSV file
        It removes the first block and normlizes the data.

        1) trial completion time
        2) shoot result
        3) glancing distance

        4) relative target position (2D, angular)
        5) target orbit axis (2D, angular)
        6) target speed (angular)
        7) target radius
        8) gaze reaction time
        9) hand reaction time
        10) eye position (2D, eye_x excluded)
        """
        self.stat_data = linear_normalize(
            expdata[COMPLETE_SUMMARY].to_numpy(), 
            *np.array([VALUE_RANGE[metric] for metric in COMPLETE_SUMMARY]).T
        )

    
    def _get_traj_data(self, expdata:pd.DataFrame, verbose=True):
        """
        Read trajectory (variable-size) behavioral outputs for every trial from pickle file
        Data for each timstep in trajectory data (normalized)
        1) time difference from prev. timestep
        2) target position (2D)
        3) gaze position (2D)
        4) camera angle (2D)
        """
        for p, m, sn, bn, tn in zip(
            tqdm(expdata["player"].to_list()) if verbose else expdata["player"].to_list(),
            expdata["mode"].to_list(),
            expdata["session_id"].to_list(), 
            expdata["block_index"].to_list(), 
            expdata["trial_index"].to_list()
        ):
            ts, traj_t, traj_g, _, traj_c = load_trajectory(p, m, sn, bn, tn)
            # Normalization
            traj_t = np.clip(traj_t / MONITOR_BOUND, a_min=-1, a_max=1)
            traj_g = np.clip(traj_g / MONITOR_BOUND, a_min=-1, a_max=1)
            traj_c = np.clip(traj_c / CAMERA_ANGLE_BOUND, a_min=-1, a_max=1)

            # Downsampling
            ts_ds = np.linspace(ts[0], ts[-1], int(ts[-1] * self.downsample))
            traj_t = np_interp_nd(ts_ds, ts, traj_t)
            traj_g = np_interp_nd(ts_ds, ts, traj_g)
            traj_c = np_interp_nd(ts_ds, ts, traj_c)
            ts_ds = np.insert(np.diff(ts_ds), 0, 0)
            traj_data = np.vstack((ts_ds, traj_t.T, traj_g.T, traj_c.T)).T
            self.traj_data.append(traj_data.astype(np.float32))
        self.traj_data = np.array(self.traj_data, dtype=object)
    

    def _sample_trial(self, n_trial=None, random_sample=False, **cond):
        """
        Sample the specified number of trials for a given condition (tier, player, session ...).
        If the requested number of trials is larger than the available data,
        it repeats the data until the required number of trials is reached.
        """
        texp = deepcopy(self.task_data)
        for k, v in cond.items():
            if type(v) is list: texp = texp[texp[k].isin(v)]
            else: texp = texp[texp[k] == v]
            
        indices = texp.index.to_numpy(dtype=np.int32)
        if n_trial is not None:
            if not random_sample:
                if indices.size < n_trial:
                    indices = np.tile(indices, int(np.ceil(n_trial / indices.size)))[:n_trial]
                else:
                    indices = indices[:n_trial]
            else:
                if indices.size < n_trial:
                    indices = np.concatenate((indices, np.random.choice(indices, n_trial - indices.size)))
                else:
                    indices = np.random.choice(indices, n_trial, replace=False)
        elif random_sample:
            np.random.shuffle(indices)

        return self.stat_data[indices], self.traj_data[indices], texp.loc[indices]


    def sample(self, n_trial=None, random_sample=False, cross_valid=False, **cond):
        """
        Sample data from the dataset with a specified number of trials and task conditions.
        If only cross_valid is set, it splits the data into two groups,
        with half of the data used for training and the other half for validation.
        """
        stat, traj, task = self._sample_trial(n_trial, random_sample=random_sample, **cond)
        
        if cross_valid:
            index_order = np.arange(len(stat))
            np.random.shuffle(index_order)

            if index_order.size % 2 == 0:
                train_indices = index_order[:index_order.size//2]
                valid_indices = index_order[index_order.size//2:]
            else:
                train_indices = index_order[:index_order.size//2+1]
                valid_indices = index_order[index_order.size//2+1:]

            return stat, traj, task, np.sort(train_indices), np.sort(valid_indices)
        else:
            return stat, traj, task



class AnSTrainDataset(object):
    """
    The class 'AnSTrainDataset' handles the creation and retrieval of a training dataset with aim-and-shoot simulator.
    It allows you to sample data from the dataset and supports various configurations.
    """
    def __init__(self, n_ep=64, sim_config=None, mode='full', load_existing_data=True):
        """
        Initialize the dataset object with a specified number of total simulations, episodes,
        and a simulation configuration.
        """
        assert mode in ["full", "base", "abl0", "abl1", "abl2", "abl3"]

        self.mode = mode
        self.data_prefix = 'full' if 'base' != mode else mode

        if sim_config is None:
            self.sim_config = deepcopy(default_ans_config[self.mode]["simulator"])
        else:
            self.sim_config = deepcopy(sim_config)
        self.sim_config["seed"] = 100

        self.n_ep = n_ep
        self.policy = f"{self.sim_config['policy']}_{self.sim_config['checkpt']}"

        os.makedirs(f"{PATH_AMORT_SIM_DATASET}{self.policy}", exist_ok=True)
        if load_existing_data: self._get_dataset()
    

    def _get_dataset(self):
        """Load an existing dataset from file."""
        self.datafilelist = glob.glob(f"{PATH_AMORT_SIM_DATASET}{self.policy}/train_{self.data_prefix}_*_step_{self.n_ep}ep.pkl")
        
        # No data found
        if len(self.datafilelist) == 0:
            print("WARNING!!! NO DATASET EXIST!!!")
            return

        self.datafilelist.sort()
        # self.n_subdataset = len(self.datafilelist)

        self.total_sim = 0
        for f in self.datafilelist:
            self.total_sim += int(f.split('_')[-3])

        # self.subdataset_order = []
        # self.select_sub_dataset()
        # print(f"[ simulated dataset ({self.mode}) ] {len(self.datafilelist)} sub-dataset in queue, {self.total_sim} simulations, currently selected: {self.subdataset_index}")

        self.load_all_dataset()
        print(f"[ simulated dataset ({self.mode}) ] {self.total_sim} simulations")


    def _generate_dataset(self, total_sim=2**23, save_sim=2**21, num_cpu=12):
        """Generate simulation dataset"""
        self.simulator = AnSSimulator(self.sim_config, mode=self.data_prefix)

        save_freq = int(np.ceil(total_sim / save_sim))

        def get_simul_res(simulator, i):
            np.random.seed(datetime.now().microsecond + i)
            args = simulator.simulate(
                n_param=1,
                sim_per_param=self.n_ep,
                verbose=False
            )
            return args

        for _ in range(save_freq):
            n_param = (total_sim // self.n_ep) // save_freq
            sub_sim = total_sim // save_freq

            eps = Parallel(n_jobs=num_cpu)(
                delayed(get_simul_res)(self.simulator, i) for i in tqdm(range(n_param))
            )

            params_arr = np.concatenate([eps[i][0] for i in range(n_param)], axis=0, dtype=np.float32)
            stats_arr = np.concatenate([eps[i][1] for i in range(n_param)], axis=0, dtype=np.float32)
            trajs_arr = []
            for i in range(n_param): trajs_arr += eps[i][2]
            trajs_arr = np.array(trajs_arr, dtype=object)

            fname = f"{PATH_AMORT_SIM_DATASET}{self.policy}/train_{self.data_prefix}_{now_to_string(omit_year=True)}_{sub_sim:08d}_step_{self.n_ep}ep.pkl"
            
            pickle_save(
                fname, 
                dict(
                    params=params_arr,         # np.array (n_param, param_sz)
                    stat_data=stats_arr,          # np.array (n_param, n_ep, stat_sz)
                    traj_data=trajs_arr           # np.array (n_param, n_ep) of (T, traj_sz)
                )
            )

    
    def load_all_dataset(self, verbose=True):
        param = []
        stat = []
        traj = []
        if verbose: print(f"Loading train dataset - {self.mode}")
        for f in tqdm(self.datafilelist) if verbose else self.datafilelist:
            dataset = pickle_load(f)
            param.append(dataset["params"])
            stat.append(dataset["stat_data"][:,:,[COMPLETE_SUMMARY.index(metric) for metric in SUMMARY_DATA[self.mode]]])
            _traj = dataset["traj_data"]
            for r in range(_traj.shape[0]):
                for c in range(_traj.shape[1]):
                    _traj[r][c] = _traj[r][c][:,[COMPLETE_TRAJECTORY.index(metric) for metric in TRAJECTORY_DATA[self.mode]]]
            traj.append(_traj)
        
        self.dataset = dict(
            params = np.concatenate(param, dtype=np.float32),
            stat_data = np.concatenate(stat, dtype=np.float32),
            traj_data = np.concatenate(traj, dtype=object)
        )
        self.n_param = self.dataset["params"].shape[0]

    def sample(self, batch_sz, sim_per_param=1, change_subdataset=False):
        """
        Returns a random sample from the dataset with
        the specified number of parameter sets (batch size),
        the number of unique initial task conditions (n_unique_cond),
        and number of simulated trials per parameter-condition pair (sim_per_param_cond).
        """
        # Dataset update
        # if change_subdataset: self.select_sub_dataset()

        # Parameter selection
        indices = np.random.choice(self.n_param, batch_sz, replace=False)
        ep_indices = np.random.choice(self.n_ep, sim_per_param)
        rows = np.repeat(indices, sim_per_param).reshape((-1, sim_per_param))
        cols = np.tile(ep_indices, (batch_sz, 1))
        if sim_per_param == 1:
            return (
                self.dataset["params"][indices],
                self.dataset["stat_data"][rows, cols].squeeze(1),
                self.dataset["traj_data"][rows, cols].squeeze(1),
            )
        else:
            return (
                self.dataset["params"][indices],
                self.dataset["stat_data"][rows, cols],
                self.dataset["traj_data"][rows, cols],
            )
    
    # def select_sub_dataset(self):
    #     # To save memory, split dataset in several files and select among them
    #     if len(self.subdataset_order) == 0:
    #         self.subdataset_order = list(range(self.n_subdataset))
    #         random.shuffle(self.subdataset_order)

    #     self.subdataset_index = self.subdataset_order.pop(0)
    #     self.dataset = pickle_load(self.datafilelist[self.subdataset_index])

    #     ### EXCLUSIONS
    #     if self.mode == 'full':
    #         self.dataset["stat_data"] = np.delete(self.dataset["stat_data"], [3, 4, 5], axis=2)
    #     if self.mode == 'abl1':
    #         self.dataset["stat_data"] = np.delete(self.dataset["stat_data"], [2, 9, 11, 12], axis=2)
    #         for r in range(self.dataset["traj_data"].shape[0]):
    #             for c in range(self.dataset["traj_data"].shape[1]):
    #                 self.dataset["traj_data"][r][c] = np.delete(self.dataset["traj_data"][r][c], [3, 4], axis=1)

    #     elif self.mode == 'abl2':
    #         self.dataset["stat_data"] = np.delete(self.dataset["stat_data"], [2, 9, 10, 11, 12], axis=2)

    #     self.n_param = self.dataset["params"].shape[0]




class AnSValidDataset(object):
    """
    The class 'AnSValidDataset' handles the creation and retrieval of a validation dataset with aim-and-shoot simulator.
    """
    def __init__(self, total_user=100, trial_per_user=600, sim_config=None, mode='full', load_existing_data=True):
        """
        Initialize the dataset object with a specified number of total user (different parameter sets), episodes,
        and a simulation configuration.
        """
        assert mode in ["full", "base", "abl0", "abl1", "abl2", "abl3"]

        self.mode = mode
        self.data_prefix = 'full' if 'base' != mode else mode

        self.total_user = total_user
        self.trial_per_user = trial_per_user
        if sim_config is None:
            self.sim_config = deepcopy(default_ans_config[self.mode]["simulator"])
        else:
            self.sim_config = deepcopy(sim_config)
        self.sim_config["seed"] = 121
        
        self.total_user = total_user
        self.trial_per_user = trial_per_user
        self.policy = f"{self.sim_config['policy']}_{self.sim_config['checkpt']}"

        os.makedirs(f"{PATH_AMORT_SIM_DATASET}{self.policy}", exist_ok=True)
        self.fpath = f"{PATH_AMORT_SIM_DATASET}{self.policy}/valid_{self.data_prefix}_{total_user}_param_{trial_per_user}ep.pkl"

        if load_existing_data: self._get_dataset()
        

    def _get_dataset(self):
        if not os.path.exists(self.fpath):
            print("WARNING!!! NO DATASET EXIST!!!")
            return
        self.dataset = pickle_load(self.fpath)

        self.dataset["stat_data"] = self.dataset["stat_data"][:,:,[COMPLETE_SUMMARY.index(metric) for metric in SUMMARY_DATA[self.mode]]]
        for r in range(self.dataset["traj_data"].shape[0]):
            for c in range(self.dataset["traj_data"].shape[1]):
                self.dataset["traj_data"][r][c] = \
                    self.dataset["traj_data"][r][c][:,[COMPLETE_TRAJECTORY.index(metric) for metric in TRAJECTORY_DATA[self.mode]]]

        self.n_param = self.dataset["params"].shape[0]



    def _generate_dataset(self, num_cpu=12):
        self.simulator = AnSSimulator(self.sim_config, mode=self.data_prefix)

        def get_simul_res(simulator, i):
            np.random.seed(datetime.now().microsecond + i)
            args = simulator.simulate(
                n_param=1,
                sim_per_param=self.trial_per_user,
                verbose=False
            )
            return args

        # Parallelize the creation of the dataset.
        eps = Parallel(n_jobs=num_cpu)(
            delayed(get_simul_res)(self.simulator, i) for i in tqdm(range(self.total_user))
        )
        params_arr = np.concatenate([eps[i][0] for i in range(self.total_user)], axis=0, dtype=np.float32)
        stats_arr = np.concatenate([eps[i][1] for i in range(self.total_user)], axis=0, dtype=np.float32)
        trajs_arr = []
        for i in range(self.total_user): trajs_arr += eps[i][2]
        trajs_arr = np.array(trajs_arr, dtype=object)

        pickle_save(
            self.fpath,
            dict(
                params=params_arr,         # np.array (n_param, param_sz)
                stat_data=stats_arr,          # np.array (n_param, n_ep, stat_sz)
                traj_data=trajs_arr           # np.array (n_param, n_ep) of (T, traj_sz)
            )
        )


    def sample(self, n_trial, n_user=None, indiv_user=False, cross_valid=False):
        """
        Sample the dataset for the given number of trials and users.
        It returns the sampled data based on the specified options (indiv_user and cross_valid).
        If indiv_user is set to True, the data for each user is returned individually.
        If cross_valid is set to True, the data is divided into two parts for cross-validation purposes.
        """
        if n_user is None:
            n_user = self.total_user
        user_data, valid_user_data = list(), list()
        params = list()
        for user in range(n_user):
            params.append(self.dataset["params"][user])
            stats = self.dataset["stat_data"][user][:n_trial]
            trajs = self.dataset["traj_data"][user][:n_trial]

            if indiv_user:
                if cross_valid:
                    half = n_trial // 2
                    user_data.append([stats[:half], trajs[:half]])
                    valid_user_data.append([stats[half:], trajs[half:]])
                else:
                    user_data.append([stats, trajs])
            else:
                if user >= n_user // 2 and cross_valid:
                    valid_user_data.append([stats, trajs])
                else:
                    user_data.append([stats, trajs])

        if cross_valid:
            return np.array(params, dtype=np.float32), user_data, valid_user_data
        else:
            return np.array(params, dtype=np.float32), user_data



if __name__ == "__main__":

    np.seterr(all='ignore')

    import argparse
    parser = argparse.ArgumentParser(description='Data synthesis')

    parser.add_argument('--user', type=bool, default=False)
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--valid', type=bool, default=False)

    parser.add_argument('--full', type=bool, default=False)
    parser.add_argument('--base', type=bool, default=False)

    parser.add_argument('--cpu', type=int, default=16)
    parser.add_argument('--exp', type=int, default=21)
    parser.add_argument('--mul', type=int, default=1)

    args = parser.parse_args()

    if args.user: AnSPlayerDataset()

    if args.full:
        if args.train: 
            AnSTrainDataset(mode='full', load_existing_data=False)._generate_dataset(total_sim=2**args.exp * args.mul, num_cpu=args.cpu)
        if args.valid:
            AnSValidDataset(mode='full', load_existing_data=False)._generate_dataset(num_cpu=args.cpu)
        
    if args.base:
        if args.train: 
            AnSTrainDataset(mode='base', load_existing_data=False)._generate_dataset(total_sim=2**args.exp * args.mul, num_cpu=args.cpu)
        if args.valid:
            AnSValidDataset(mode='base', load_existing_data=False)._generate_dataset(num_cpu=args.cpu)