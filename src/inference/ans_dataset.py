"""
Dataset generator/loader for amortized inference
Orignial code written by Hee-seung Moon (https://github.com/hsmoon121/amortized-inference-hci)

Code modified by June-Seop Yoon
"""

import os
from pathlib import Path
from joblib import Parallel, delayed
import psutil
import numpy as np
import pandas as pd
from box import Box
from tqdm import tqdm
import copy
from datetime import datetime
import glob

from ..config.config import INF
from ..config.constant import FOLDER
from ..agent.ans_simulator import AnSSimulator
from ..utils.myutils import get_timebase_session_name, npz_load, npz_save, save_dict_to_yaml, load_config
from ..utils.mymath import linear_denormalize, linear_normalize, log_denormalize

DIR_TO_DATA = Path(__file__).parent.parent.parent

class AnSSimulatorForAmortizer:
    def __init__(self, model_name, ckpt, downsample_rate, task_cond=dict(), user_stat_cond=dict()):
        self.simulator = AnSSimulator(model_name=model_name, ckpt=ckpt)
        self.target_param, self.param_range = self.simulator.modulated_param_list()

        # Trajectory data downsampling rate
        self.downsample_rate = downsample_rate
        self.task_cond = task_cond
        self.user_stat_cond = user_stat_cond
        self.game = copy.deepcopy(self.simulator.env.game_env)
    

    def simulate(
        self,
        n_param=1,
        sim_per_param=1,    # No. of simulation per (parameter, task) condition,
        verbose=False
    ):
        param_z = np.random.uniform(low=-1, high=1, size=(n_param, len(self.target_param)))

        stats, trajs = list(), list()
        for i in (tqdm(range(n_param)) if verbose else range(n_param)):
            z = param_z[i]
            self.simulator.fix_user_param(param_z=dict(zip(self.target_param, z)))
            task_list = self.game.sample_task_condition(sim_per_param, **self.task_cond)
            user_stat_list = self.simulator.sample_user_stat_conditions(sim_per_param, **self.user_stat_cond)
            self.simulator.simulate(task_list=task_list, user_stat_list=user_stat_list, verbose=False)
            stat_result, traj_result = self.simulator.get_simulation_result(return_traj=True, downsample=self.downsample_rate)
            self.simulator.clear_simulation()

            stats.append(stat_result)   # pd.DataFrame
            trajs.append(traj_result)   # list of pd.DataFrame
        
        return (
            param_z, # np.ndarray with shape (n_param, num of target param)
            stats,   # [pd.df (sim_per_param x n_beh_feat), pd.df, ...] 1D list
            trajs    # [[pd.df traj 1, ... pd.df traj sim_per_param], [ ... ], ..., [ ... ]] 2D nested list
        )

    
    def convert_param_z_to_w(self, outputs):
        param_in = copy.deepcopy(outputs)
        if len(np.array(param_in).shape) == 1:
            param_in = np.expand_dims(param_in, axis=0)

        param_out = np.zeros_like((param_in))
        for i, p in enumerate(self.target_param):
            if self.param_range[p].type == 'uniform':
                param_out[:,i] = linear_denormalize(
                    param_in[:,i], 
                    self.param_range[p].min, 
                    self.param_range[p].max
                )
            elif self.param_range[p].type == 'loguniform':
                param_out[:,i] = log_denormalize(
                    param_in[:,i], 
                    self.param_range[p].min, 
                    self.param_range[p].max,
                    scale=self.param_range[p].scale
                )
        return param_out 
    


class AnSAmortizerTrainingDataset(object):
    def __init__(
        self, 
        simulator_config="default",
        normalize_config="default",
        num_of_epsiode_per_param=64,
        load_existing_data=True,
        verbose=True
    ):
        self.simulator = AnSSimulatorForAmortizer(
            model_name=INF["simulator"][simulator_config]["model_name"], 
            ckpt=INF["simulator"][simulator_config]["ckpt"],
            downsample_rate=INF["simulator"][simulator_config]["traj_downsample"]
        )
        self.sim_config = INF["simulator"][simulator_config]
        self.norm_config = INF["normalize"][normalize_config]
        self.num_of_epsiode_per_param = num_of_epsiode_per_param
        self.data_path = os.path.join(
            DIR_TO_DATA, 
            f"data/{FOLDER.AMORTIZER}/dataset/{self.sim_config.model_name}_{self.sim_config.ckpt}_{self.sim_config.traj_downsample}"
        )
        os.makedirs(self.data_path, exist_ok=True)

        self.simul_model = INF["simulator"][simulator_config]["model_name"]
        self.simul_model_ckpt = INF["simulator"][simulator_config]["ckpt"]
        self.simul_model_traj_downsample = INF["simulator"][simulator_config]["traj_downsample"]

        if load_existing_data:
            self._get_dataset(verbose)
    

    def _get_dataset(self, verbose=False):
        datafilelist = glob.glob(os.path.join(
            DIR_TO_DATA, 
            f"{self.data_path}/tr_{self.num_of_epsiode_per_param}epp_*.npz"
        ))

        if not len(datafilelist):
            raise FileNotFoundError("No data file found ... generate the dataset first.")

        n_param = 0
        n_simul = 0

        param, stat, traj = list(), list(), list()

        datafilelist.sort()
        for f in datafilelist if not verbose else tqdm(datafilelist, desc="Processing training data simulations ... "):
            ff = os.path.basename(f).split('_')
            n_param += int(ff[2][:-1])
            n_simul += int(ff[3][:-2])

            data = npz_load(f)
            param.append(data["params"])

            # Feature order
            data_feat_order = load_config(f.replace('.npz', '.yaml'))
            stat_target_indices = [data_feat_order.stat.index(key) for key in self.norm_config.stat.list]
            traj_target_indices = [data_feat_order.traj.index(key) for key in self.norm_config.traj.list]

            stat_value_range = np.array([
                [self.norm_config.stat.range[key].min, 
                self.norm_config.stat.range[key].max] for key in self.norm_config.stat.list]
            )
            traj_value_range = np.array([
                [self.norm_config.traj.range[key].min, 
                self.norm_config.traj.range[key].max] for key in self.norm_config.traj.list]
            )

            # Normalize and re-organize
            def norm_stat(stat):    # stat := (n_trial x n_feature) np.ndarray
                return linear_normalize(stat[:,stat_target_indices], *stat_value_range.T, dtype=np.float32)
            
            def norm_traj(traj):    # traj := (n_length x n_feature) np.ndarray
                if "timestamp" in data_feat_order.traj:
                    arr = traj[:,data_feat_order.traj.index("timestamp")]
                    arr = np.insert(np.diff(arr), 0, 0)
                    traj[:,data_feat_order.traj.index("timestamp")] = arr
                
                return linear_normalize(traj[:,traj_target_indices], *traj_value_range.T, dtype=np.float32)

            # Normalization
            for s in data["stat_data"]:
                stat.append(norm_stat(s))
            for traj_list in data["traj_data"]:
                new_traj_list = list()
                for t in traj_list:
                    new_traj_list.append(norm_traj(t))
                traj.append(new_traj_list)

        self.n_param = n_param

        params = np.concatenate(param, dtype=np.float32)
        stats = np.array(stat, dtype=np.float32)
        trajs = np.array(traj, dtype=object)

        self.dataset = dict(
            params = params,
            stat_data = stats,
            traj_data = trajs
        )

        if verbose:
            print(f"[ simulated AnS dataset ({self.simul_model}-{self.simul_model_ckpt}-{self.simul_model_traj_downsample}) loaded for training ] {n_simul} simulations in total ({self.num_of_epsiode_per_param} eps per param, {self.n_param} params)")


    def _generate_raw_dataset(self, num_of_total_sim=2**23, freq_of_save=2**20, num_cpu=None):
        # Settings
        num_cpu = psutil.cpu_count(logical=False) if num_cpu is None else min(num_cpu, psutil.cpu_count(logical=False))
        save_freq = int(np.ceil(num_of_total_sim / freq_of_save))

        def get_simul_res(simulator: AnSSimulatorForAmortizer, i):
            np.random.seed(datetime.now().microsecond + i)
            result = simulator.simulate(
                n_param=1,
                sim_per_param=self.num_of_epsiode_per_param,
                verbose=False
            )
            return result   # Normalized parameter, summary (nested list of pd.DataFrame), trajectory (nested list of pd.DataFrame)

        for _ in range(save_freq):
            n_param = (num_of_total_sim // self.num_of_epsiode_per_param) // save_freq
            num_of_subset_sim = num_of_total_sim // save_freq

            filepath = os.path.join(
                DIR_TO_DATA, 
                f"{self.data_path}/tr_{self.num_of_epsiode_per_param}epp_{n_param}p_{num_of_subset_sim}ep_{get_timebase_session_name()}"
            )

            eps = Parallel(n_jobs=num_cpu)(
                delayed(get_simul_res)(self.simulator, i) for i in tqdm(range(n_param))
            )

            # Save column
            column_info = dict(
                stat = list(eps[0][1][0].columns),
                traj = list(eps[0][2][0][0].columns)
            )
            save_dict_to_yaml(column_info, f"{filepath}.yaml")

            params_arr = np.concatenate([eps[i][0] for i in range(n_param)], axis=0, dtype=np.float32)
            stat_data = np.array([pd_beh.to_numpy() for i in range(n_param) for pd_beh in eps[i][1]], dtype=np.float32)
            traj_data = np.array([[df.to_numpy(dtype=np.float32) for df in traj_list] for i in range(n_param) for traj_list in eps[i][2]], dtype=object)

            npz_save(
                filename=f"{filepath}.npz",
                params=params_arr,
                stat_data=stat_data,
                traj_data=traj_data
            )

    
    def sample(self, batch_sz, sim_per_param=1):
        """
        Returns a random sample from the dataset with
        the specified number of parameter sets (batch size),
        and number of simulated trials per parameter-condition pair (sim_per_param).
        """
        indices = np.random.choice(self.n_param, batch_sz, replace=False)
        ep_indices = np.random.choice(self.num_of_epsiode_per_param, sim_per_param)
        rows = np.repeat(indices, sim_per_param).reshape((-1, sim_per_param))
        cols = np.tile(ep_indices, (batch_sz, 1))
        if self.dataset["traj_data"] is not None:
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
        else:
            if sim_per_param == 1:
                return (
                    self.dataset["params"][indices],
                    self.dataset["stat_data"][rows, cols].squeeze(1),
                    None,
                )
            else:
                return (
                    self.dataset["params"][indices],
                    self.dataset["stat_data"][rows, cols],
                    None,
                )


class AnSAmortizerValidationDataset(object):
    def __init__(
        self, 
        simulator_config="default",
        normalize_config="default",
        total_user=100,
        trial_per_user=600,
        load_existing_data=True,
        verbose=True
    ):
        self.simulator = AnSSimulatorForAmortizer(
            model_name=INF["simulator"][simulator_config]["model_name"], 
            ckpt=INF["simulator"][simulator_config]["ckpt"],
            downsample_rate=INF["simulator"][simulator_config]["traj_downsample"]
        )
        self.total_user = total_user
        self.trial_per_user = trial_per_user
        self.sim_config = INF["simulator"][simulator_config]
        self.norm_config = INF["normalize"][normalize_config]
        self.data_path = os.path.join(
            DIR_TO_DATA, 
            f"data/{FOLDER.AMORTIZER}/dataset/{self.sim_config.model_name}_{self.sim_config.ckpt}_{self.sim_config.traj_downsample}/vd_{self.total_user}users_{self.trial_per_user}trials.npz"
        )
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)

        self.simul_model = INF["simulator"][simulator_config]["model_name"]
        self.simul_model_ckpt = INF["simulator"][simulator_config]["ckpt"]
        self.simul_model_traj_downsample = INF["simulator"][simulator_config]["traj_downsample"]

        if load_existing_data:
            self._get_dataset(verbose)
    
    def _get_dataset(self, verbose=False):
        try:
            data = npz_load(self.data_path)
        except FileNotFoundError:
            raise FileNotFoundError("No data file found for validation ... generate the dataset first.")

        # Feature order
        data_feat_order = load_config(self.data_path.replace('.npz', '.yaml'))
        stat_target_indices = [data_feat_order.stat.index(key) for key in self.norm_config.stat.list]
        traj_target_indices = [data_feat_order.traj.index(key) for key in self.norm_config.traj.list]

        stat_value_range = np.array([
            [self.norm_config.stat.range[key].min, 
            self.norm_config.stat.range[key].max] for key in self.norm_config.stat.list]
        )
        traj_value_range = np.array([
            [self.norm_config.traj.range[key].min, 
            self.norm_config.traj.range[key].max] for key in self.norm_config.traj.list]
        )

        # Normalize and re-organize
        def norm_stat(stat):    # stat := (n_trial x n_feature) np.ndarray
            return linear_normalize(stat[:,stat_target_indices], *stat_value_range.T, dtype=np.float32)
        
        def norm_traj(traj):    # traj := (n_length x n_feature) np.ndarray
            if "timestamp" in data_feat_order.traj:
                arr = traj[:,data_feat_order.traj.index("timestamp")]
                arr = np.insert(np.diff(arr), 0, 0)
                traj[:,data_feat_order.traj.index("timestamp")] = arr
            
            return linear_normalize(traj[:,traj_target_indices], *traj_value_range.T, dtype=np.float32)

        param = data["params"]
        stats = np.array([norm_stat(s) for s in data["stat_data"]], dtype=np.float32)
        trajs = np.array([[norm_traj(t) for t in traj_list] for traj_list in data["traj_data"]], dtype=object)

        self.dataset = dict(
            params = param,
            stat_data = stats,
            traj_data = trajs
        )

        if verbose:
            print(f"[ simulated AnS dataset ({self.simul_model}-{self.simul_model_ckpt}-{self.simul_model_traj_downsample}) loaded for validation ] {self.total_user} users x {self.trial_per_user} trials")


    def _generate_raw_dataset(self, num_cpu=None):
        # Settings
        num_cpu = psutil.cpu_count(logical=False) if num_cpu is None else min(num_cpu, psutil.cpu_count(logical=False))

        def get_simul_res(simulator: AnSSimulatorForAmortizer, i):
            np.random.seed(datetime.now().microsecond + i)
            result = simulator.simulate(
                n_param=1,
                sim_per_param=self.trial_per_user,
                verbose=False
            )
            return result   # Normalized parameter, summary (nested list of pd.DataFrame), trajectory (nested list of pd.DataFrame)

        eps = Parallel(n_jobs=num_cpu)(
            delayed(get_simul_res)(self.simulator, i) for i in tqdm(range(self.total_user))
        )

        # Save column
        column_info = dict(
            stat = list(eps[0][1][0].columns),
            traj = list(eps[0][2][0][0].columns)
        )
        save_dict_to_yaml(column_info, self.data_path.replace('.npz', '.yaml'))

        params_arr = np.concatenate([eps[i][0] for i in range(self.total_user)], axis=0, dtype=np.float32)
        stat_data = np.array([pd_beh.to_numpy() for i in range(self.total_user) for pd_beh in eps[i][1]], dtype=np.float32)
        traj_data = np.array([[df.to_numpy(dtype=np.float32) for df in traj_list] for i in range(self.total_user) for traj_list in eps[i][2]], dtype=object)

        npz_save(
            filename=self.data_path,
            params=params_arr,
            stat_data=stat_data,
            traj_data=traj_data
        )
    

    def sample(self, n_trial, n_user=None, indiv_user=False, cross_valid=False, random_sample=False):
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
            if not random_sample:
                stats = self.dataset["stat_data"][user][:n_trial]
                trajs = self.dataset["traj_data"][user][:n_trial] if self.dataset["traj_data"] is not None else [None] * n_trial
            else:
                sample_indices = np.random.choice(np.arange(self.trial_per_user), size=n_trial, replace=False)
                stats = self.dataset["stat_data"][user][sample_indices]
                trajs = self.dataset["traj_data"][user][sample_indices] if self.dataset["traj_data"] is not None else [None] * n_trial

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
    import argparse
    parser = argparse.ArgumentParser(description='Data synthesis')

    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--valid', type=bool, default=False)

    parser.add_argument('--cpu', type=int, default=16)
    parser.add_argument('--exp', type=int, default=21)
    parser.add_argument('--mul', type=int, default=1)

    args = parser.parse_args()

    if args.train: 
        data = AnSAmortizerTrainingDataset(load_existing_data=False)
        data._generate_raw_dataset(2**args.exp * args.mul, num_cpu=args.cpu)
    if args.valid:
        data = AnSAmortizerValidationDataset(load_existing_data=False)
        data._generate_raw_dataset(num_cpu=args.cpu)