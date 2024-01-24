"""
Trainer for amortized inference
Orignial code written by Hee-seung Moon (https://github.com/hsmoon121/amortized-inference-hci)

Code modified by June-Seop Yoon
"""

import os, sys
from time import time
from copy import deepcopy
from pathlib import Path
from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score
import torch
import torch.nn.functional as F
from experiment.data_process import load_experiment

sys.path.append("..")

from nets.amortizer import AmortizerForTrialData, RegressionForTrialData
from amortizer.ans_simulator import AnSSimulator
from amortizer.ans_dataset import AnSPlayerDataset, AnSTrainDataset, AnSValidDataset
from utils.schedulers import CosAnnealWR
from configs.amort import default_ans_config
from configs.path import *
from configs.common import *
from configs.experiment import *
from configs.simulation import PARAM_SYMBOL
from utils.loggers import Logger
from utils.mymath import *
from utils.plots import *
from utils.utils import now_to_string, pickle_load, pickle_save


class AnSTrainer(ABC):
    def __init__(self, config=None, mode='full'):
        assert mode in ["full", "base", "abl0", "abl1", "abl2", "abl3"]
        self.mode = mode

        if config is None:
            self.config = deepcopy(default_ans_config[self.mode])
        else:
            self.config = config
        self.iter = 0
        self.name = self.config["name"]

        # Initialize the amortizer, simulator, and datasets
        self.point_estimation = self.config["point_estimation"]
        amortizer_fn = RegressionForTrialData if self.point_estimation else AmortizerForTrialData
        self.amortizer = amortizer_fn(config=self.config["amortizer"]) 
        self.user_dataset = AnSPlayerDataset(mode=mode)

        self.simulator = AnSSimulator(config=self.config["simulator"], mode=mode)
        self.valid_dataset = AnSValidDataset(sim_config=self.config["simulator"], mode=mode)
        self.train_dataset = AnSTrainDataset(sim_config=self.config["simulator"], mode=mode)

        self.targeted_params = deepcopy(self.simulator.targeted_params)
        self.param_symbol = [PARAM_SYMBOL[p] for p in self.targeted_params]

        if self.mode != "base":
            self.obs_list = [M_TCT, M_ACC, M_GD]
            self.obs_max = dict(zip(self.obs_list, [MAX_TCT, 1, MAX_GD]))
            self.obs_label = dict(zip(self.obs_list, ["tct", "acc", "gd"]))
            self.obs_unit = dict(zip(self.obs_list, [0.1, 0.1, 2]))
            self.obs_scale = dict(zip(self.obs_list, [1000, 100, 1]))
            self.obs_description = dict(zip(self.obs_list, [
                "Trial completion time (ms)",
                "Accuracy (%)",
                r"Glancing distance ($^\circ$)"
            ]))
        else:   # Baseline
            self.obs_list = [M_TCT, M_ACC]
            self.obs_max = dict(zip(self.obs_list, [MAX_TCT, 1]))
            self.obs_label = dict(zip(self.obs_list, ["tct", "acc"]))
            self.obs_unit = dict(zip(self.obs_list, [0.1, 0.1]))
            self.obs_scale = dict(zip(self.obs_list, [1000, 100]))
            self.obs_description = dict(zip(self.obs_list, [
                "Trial completion time (ms)",
                "Accuracy (%)"
            ]))
        

        # Initialize the optimizer and scheduler
        self.lr = self.config["learning_rate"]
        self.lr_gamma = self.config["lr_gamma"]
        self.clipping = self.config["clipping"]
        self.optimizer = torch.optim.Adam(self.amortizer.parameters(), lr=1e-9)
        self.scheduler = CosAnnealWR(self.optimizer, T_0=10, T_mult=1, eta_max=self.lr, T_up=1, gamma=self.lr_gamma)

        self.datetime_str = now_to_string(omit_year=True, omit_ms=True)
        self.model_path = PATH_AMORT_MODEL
        self.board_path = PATH_AMORT_BOARD
        self.result_path = PATH_AMORT_RESULT
        self.clipping = float("Inf")


    def train(
        self,
        n_iter=500,     # 20~, 100~200
        step_per_iter=2048,
        batch_sz=64,    # As maximum as possible (memory & speed)
        n_trial=64,     # What would be the minimum number of trials to properly do inference
        board=True,
        save_freq=10,
    ):
        """
        Training loop

        n_iter (int): Number of training iterations
        step_per_iter (int): Number of training steps per iteration
        batch_sz (int): Batch size
        n_trial (int): Number of trials for each user data (default: 1)
        board (bool): Whether to use tensorboard (default: True)
        """
        iter = self.iter
        last_step = self.iter * step_per_iter

        self.logger = Logger(
            self.name, 
            self.datetime_str,
            last_step=last_step, 
            board=board, 
            board_path=self.board_path
        )

        # Training iterations
        losses = dict()
        print(f"\n[ Training - {self.name} ]")
        for iter in range(self.iter + 1, n_iter + 1):
            losses[iter] = []

            # Training loop
            with tqdm(total=step_per_iter, desc=f" Iter {iter}") as progress:
                for step in range(step_per_iter):
                    batch_args = self.train_dataset.sample(
                        batch_sz=batch_sz, 
                        sim_per_param=n_trial
                    )

                    # Training step
                    loss = self._train_step(*batch_args)
                    losses[iter].append(loss)

                    # Logging
                    if step % 10 == 0:
                        self.logger.write_scalar(train_loss=loss, lr=self.scheduler.get_last_lr()[0])
                    progress.set_postfix_str(f"Avg.Loss: {np.mean(losses[iter]):.3f}")
                    progress.update(1)
                    self.logger.step()
                    self.scheduler.step((iter-1) + step/step_per_iter)
                    
                    if np.isnan(loss):
                        raise RuntimeError("Nan loss computed.")

            # Save model
            if iter % save_freq == 0:
                self.save(iter)
                valid_res = self.valid()
                self.logger.write_scalar(**valid_res)
            self.iter = iter

        print("\n[ Training Done ]")
        if iter in losses:
            print(f"  Training Loss: {np.mean(losses[iter])}\n")
        


    def _train_step(self, params, stat_data, traj_data=None):
        """
        Training step

        params (ndarray): [n_sim, n_param] array of parameters
        stat_data (list): [n_sim] list of static data
        traj_data (list): [n_sim] list of trajectories (default: None)
        """
        self.amortizer.train()
        if self.point_estimation:
            params_tensor = torch.FloatTensor(params).to(self.amortizer.device)
            loss = F.mse_loss(self.amortizer(stat_data, traj_data), params_tensor)
        else:
            z, log_det_J = self.amortizer(params, stat_data, traj_data)
            loss = torch.mean(0.5 * torch.square(torch.norm(z, dim=-1)) - log_det_J)
        return self._optim_step(loss)
    

    def _optim_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.amortizer.parameters():
            param.grad.data.clamp_(-self.clipping, self.clipping)
        self.optimizer.step()
        return loss.item()


    def save(self, iter, path=None):
        """
        Save model, optimizer, and scheduler with iteration number
        """
        if path is None:
            os.makedirs(f"{self.model_path}/{self.name}/{self.datetime_str}", exist_ok=True)
            ckpt_path = f"{self.model_path}/{self.name}/{self.datetime_str}/iter{iter:03d}.pt"
        else:
            os.makedirs(path, exist_ok=True)
            ckpt_path = path + f"iter{iter:03d}.pt"
        torch.save({
            "iteration": iter,
            "model_state_dict": self.amortizer.state_dict(),
            "optim_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, ckpt_path)
        

    def load(self, model_name, model_session):
        """
        Load model, optimizer, and scheduler from the latest checkpoint
        """
        import glob
        ckpt_paths = glob.glob(f"{self.model_path}/{model_name}/{model_session}/iter*.pt")
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]

        self.name = model_name
        self.datetime_str = model_session

        ckpt = torch.load(ckpt_path, map_location=self.amortizer.device.type)
        self.amortizer.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optim_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        iter = ckpt["iteration"]
        self.scheduler.step(iter)

        print(f"[ amortizer - loaded checkpoint ]\n\t{model_name} - {model_session} - {iter}")

        self.iter = iter


    def valid(self,
        n_trial=np.array([10, 20, 50, 100, 200, 500]),
        n_sample=300,   # Sample for distribution estimation
        verbose=True
    ):
        self.amortizer.eval()
        valid_res = dict()
        os.makedirs(f"{self.result_path}/{self.name}/{self.datetime_str}/iter{self.iter:03d}/", exist_ok=True)

        ### 1) Parameter recovery from simulated
        type_list = ["mode", "mean", "median"] if type(self.amortizer) is AmortizerForTrialData else ["mode"]
        start_t = time()
        
        for infer_type in type_list:
            rsq = list()
            for n in n_trial:
                sim_gt_params, sim_valid_data = self.valid_dataset.sample(n)
                r2s = self.parameter_recovery(
                    valid_res,
                    sim_gt_params,
                    sim_valid_data,
                    n_sample,
                    n,
                    infer_type,
                    surfix="_sim",
                )
                rsq.append(r2s)
            rsq = np.array(rsq)

            pd.DataFrame(dict(zip(
                ["n", *self.targeted_params],
                [n_trial, *rsq.T]
            ))).to_csv(f"{self.result_path}/{self.name}/{self.datetime_str}/iter{self.iter:03d}/recovery_{infer_type}/recovery_r2.csv", index=False)
            
            plot_recovery_r2(
                n_trial,
                rsq,
                fname=f"r2_params_flow",
                param_labels=self.param_symbol,
                fpath=f"{self.result_path}/{self.name}/{self.datetime_str}/iter{self.iter:03d}/recovery_{infer_type}/"
            )

        if verbose:
            print(f"- parameter recovery (simulated) ({time() - start_t:.3f}s)")
        
        ### 2) Compare simulation and human players
        start_t = time()
        self.player_versus_simulator(
            valid_res,
            n_sample, 
            infer_type='mode'
        )
        if verbose:
            print(f"- player simulation ({time() - start_t:.3f}s)")

        return valid_res


    def parameter_recovery(
        self,
        res,
        gt_params,
        valid_data,
        n_sample,
        n_datasize,
        infer_type,
        surfix="",
    ):
        os.makedirs(f"{self.result_path}/{self.name}/{self.datetime_str}/iter{self.iter:03d}/recovery_{infer_type}/", exist_ok=True)

        # Note: all parameters are normalized: -1 ~ 1
        n_param = gt_params.shape[0]
        inferred_params = list()

        for param_i in range(n_param):
            stat_i, traj_i = valid_data[param_i] # valid_data := list[(stat, traj)]
            lognorm_param = self.amortizer.infer(
                stat_i, traj_i, 
                n_sample=n_sample, 
                type=infer_type
            )
            lognorm_param = self._clip_params(lognorm_param)
            p = self.simulator.convert_from_output(lognorm_param)[0]
            inferred_params.append(p)
        inferred_params = np.array(inferred_params)
                
        gt_params = self.simulator.convert_from_output(gt_params)
        
        pickle_save(
            f"{self.result_path}/{self.name}/{self.datetime_str}/iter{self.iter:03d}/recovery_{infer_type}/recovery_{n_datasize:03d}.pkl",
            (gt_params, inferred_params)
        )
        
        r_squared = plot_parameter_recovery(
            gt_params,
            inferred_params,
            fname=f"r2_params_{infer_type}_{n_datasize:03d}",
            param_labels=self.param_symbol,
            fpath=f"{self.result_path}/{self.name}/{self.datetime_str}/iter{self.iter:03d}/recovery_{infer_type}/"
        )

        for i, l in enumerate(self.targeted_params):
            res[f"Parameter_Recovery/{infer_type}/r2_{n_datasize:03d}_" + l + surfix] = r_squared[i]

        return r_squared


    def player_versus_simulator(self, res, n_sample, infer_type):
        ### Run simulation using inferred parameters
        ### To save time, only infer mode=default and block_index=1
        inferred_param = {p: {color: None for color in COLOR} for p in PLAYERS}

        stat_merged = list()

        exp_dist = pickle_load(f"{PATH_DATA_SUMMARY}dist_player_mode_block.pkl")
        sim_dist = {p: dict(ts=None, tx=None, gx=None) for p in PLAYERS}
        
        with tqdm(total=len(PLAYERS)*2, desc=f" Valid(vs)") as progress:
            for p in PLAYERS:
                self.simulator.simulator.clear_record()
                stat_player = list()
                for color in COLOR:
                    stat, traj, _ = self.user_dataset.sample(
                        player=p, 
                        mode='default', 
                        target_color=color, 
                        block_index=1
                    )
                    lognorm_param = self.amortizer.infer(stat, traj, n_sample=n_sample, type=infer_type)
                    lognorm_param = self._clip_params(lognorm_param)

                    raw_param = self.simulator.convert_from_output(lognorm_param)[0]
                    inferred_param[p][color] = raw_param

                    exp_cond = self.simulator.simulator._load_experiment_cond(
                        fix_head_position=(self.mode in ['base', 'abl1', 'abl2']),
                        fix_gaze_reaction=(self.mode in ['base', 'abl1', 'abl2']),
                        fix_hand_reaction=(self.mode in ['base']),
                        player=p,
                        mode='default',
                        target_color=color,
                        block_index=1
                    )
                    self.simulator.simulator.update_parameter(
                        param_z=dict(zip(self.targeted_params, lognorm_param))
                    )
                    self.simulator.simulator.run_simulation_with_cond(exp_cond, overwrite_existing_simul=False)

                    exp_df = self.simulator.simulator.export_result_df(include_target_cond=True)

                    # Comparison: stat
                    stat_player.append(load_experiment(
                        player=p,
                        mode='default',
                        target_color=color,
                        block_index=1
                    )[self.obs_list].to_numpy())
                    progress.update(1)

                stat_player = np.concatenate(stat_player, axis=0)
                stat_simulator = exp_df[self.obs_list].to_numpy()   # Simulation records are accumulated
        
                binning_player = np.array([PLAYERS.index(p)] * len(exp_df))
                binning_tiod = exp_df["t_iod"].to_numpy()
                binning_tcolor = (np.array(list(map(SES_NAME_ABBR.index, exp_df["session_id"].to_list()))) > 4).astype(int) # 0-white, 1-gray
                binning_tspeed = np.array(exp_df["target_speed"].to_numpy() > 1, dtype=int) # 0-stat, 1-moving
                    
                stat_merged.append(
                    np.hstack((
                        stat_player, 
                        stat_simulator, 
                        np.expand_dims(binning_player, axis=0).T,
                        np.expand_dims(binning_tiod, axis=0).T,
                        np.expand_dims(binning_tcolor, axis=0).T,
                        np.expand_dims(binning_tspeed, axis=0).T
                    ))
                )
                # Comparison: traj
                ts, tx, gx = self.simulator.simulator.collect_distance_info(return_mean_only=True)
                sim_dist[p]["ts"] = ts
                sim_dist[p]["tx"] = tx
                sim_dist[p]["gx"] = gx
                
        
        ### Static data comparison
        stat_merged = np.concatenate(stat_merged, axis=0)
        r2_player, r2_target, kl_stat = plot_stat_comparison(
            stat_merged,
            metric=self.obs_list, 
            fname="r2_stats",
            maximum_values=self.obs_max,
            metric_units=self.obs_unit,
            metric_scales=self.obs_scale,
            metric_labels=self.obs_label, 
            metric_descriptions=self.obs_description,
            fpath=f"{self.result_path}/{self.name}/{self.datetime_str}/iter{self.iter:03d}/"
        )
        for i, m in enumerate(self.obs_label):
            res[f"Stat_Recovery/r2_player_{m}"] = r2_player[i]
            res[f"Stat_Recovery/r2_target_{m}"] = r2_target[i]
            res[f"Stat_Recovery/kld_{m}"] = kl_stat[i]
        

        ### Trajectory comparison
        if self.mode != "base":
            plot_distance_comparison(
                exp_dist, sim_dist,
                fname="traj_dist",
                fpath=f"{self.result_path}/{self.name}/{self.datetime_str}/iter{self.iter:03d}/"
            )

        ### Inferred parameters comparison
        p_value = plot_compare_inferred_param_by_tier(
            inferred_param,
            fname="param_comp_tier",
            param_labels=self.param_symbol,
            fpath=f"{self.result_path}/{self.name}/{self.datetime_str}/iter{self.iter:03d}/"
        )
        for i, l in enumerate(self.targeted_params):
            res["Parameter_Compare/tier_p_" + l] = p_value[i]
        
        p_value = plot_compare_inferred_param_by_color(
            inferred_param,
            fname="param_comp_color",
            param_labels=self.param_symbol,
            fpath=f"{self.result_path}/{self.name}/{self.datetime_str}/iter{self.iter:03d}/"
        )
        for i, l in enumerate(self.targeted_params):
            res["Parameter_Compare/color_p_" + l] = p_value[i]




    def _clip_params(self, params):
        return np.clip(
            params,
            np.array([-1.] * len(self.targeted_params)),
            np.array([1.] * len(self.targeted_params))
        )

    
if __name__ == "__main__":
    trainer = AnSTrainer()
    trainer.train()