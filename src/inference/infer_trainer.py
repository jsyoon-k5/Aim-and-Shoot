"""
Code written by June-Seop Yoon

Trainer for amortized inference
Orignial code was written by Dr. Hee-seung Moon (https://github.com/hsmoon121/amortized-inference-hci)
"""

import os
from time import time
from copy import deepcopy
from pathlib import Path
from abc import ABC
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from ..config.config import INF
from ..config.constant import PARAM_SYMBOL
from ..inference.ans_dataset import (
    AnSAmortizerTrainingDataset,
    AnSAmortizerValidationDataset
)
from ..nets.amortizer import AmortizerForTrialData, RegressionForTrialData
from ..utils.myutils import get_timebase_session_name, save_dict_to_yaml, pickle_save
from ..utils.schedulers import CosAnnealWR
from ..utils.loggers import Logger
from ..utils.myplot import figure_grid, figure_save, draw_r2_plot


DIR_TO_DATA = Path(__file__).parent.parent.parent

class AnSAmortizerTrainer(ABC):
    def __init__(
        self,
        name=None,
        train_config='default',         # Training hyperparameter
        simul_config='default',         # Simulator model setting
        data_config='default',    # Data processing setting
    ):
        # Initialize simulator, amortizer, and dataset
        self.iter = 0
        self.train_dataset = AnSAmortizerTrainingDataset(
            simulator_config=simul_config, 
            normalize_config=data_config
        )
        self.valid_dataset = AnSAmortizerValidationDataset(
            simulator_config=simul_config, 
            normalize_config=data_config
        )
        self.simulator = deepcopy(self.train_dataset.simulator.simulator)

        # List of target parameters
        self.target_param = self.simulator.env.user.param_modul.list

        # Configuration setting
        self.train_config = INF.training[train_config]
        stat_size = len(INF.normalize[data_config].stat.list)
        if "stat_mean" in INF.normalize[data_config]:
            if not INF.normalize[data_config].stat_mean.ignore_statmean:
                stat_size += len(INF.normalize[data_config].stat_mean.list)
        self.train_config.amortizer.encoder.stat_sz = stat_size
        self.train_config.amortizer.encoder.traj_sz = len(INF.normalize[data_config].traj.list) \
            if not INF.normalize[data_config].traj.ignore_traj else 0
        if isinstance(simul_config, str):
            self.train_config.amortizer.encoder.transformer.max_step = \
                round(INF.simulator[simul_config].traj_downsample * self.simulator.env.user.truncate_time / 1000) + 5
        else:
            self.train_config.amortizer.encoder.transformer.max_step = \
                round(simul_config["traj_downsample"] * self.simulator.env.user.truncate_time / 1000) + 5
        self.train_config.amortizer.invertible.param_sz = len(self.target_param)
        self.train_config.amortizer.linear.out_sz = len(self.target_param)
        self.train_config.amortizer.linear.in_sz = self.train_config.amortizer.trial_encoder.attention.out_sz

        # Amortizer name
        self.name = name if name is not None else f"{'PTE' if self.train_config.point_estimation else 'INN'}_{get_timebase_session_name()}"

        # Warmup
        self.point_estimation = self.train_config.point_estimation
        amortizer_fn = RegressionForTrialData if self.point_estimation else AmortizerForTrialData
        self.amortizer = amortizer_fn(config=self.train_config["amortizer"])

        self.lr = self.train_config["learning_rate"]
        self.lr_gamma = self.train_config["lr_gamma"]
        self.clipping = self.train_config["clipping"]
        self.optimizer = torch.optim.Adam(self.amortizer.parameters(), lr=1e-9)
        self.scheduler = CosAnnealWR(self.optimizer, T_0=10, T_mult=1, eta_max=self.lr, T_up=1, gamma=self.lr_gamma)

        self.model_path = os.path.join(DIR_TO_DATA, f"data/amortizer/models/{self.name}/pts")
        self.board_path = os.path.join(DIR_TO_DATA, f"data/amortizer/models/{self.name}/board")
        self.result_path = os.path.join(DIR_TO_DATA, f"data/amortizer/models/{self.name}/results")

        save_dict_to_yaml(self.train_config, os.path.join(DIR_TO_DATA, f"data/amortizer/models/{self.name}/train_setting.yaml"))
        save_dict_to_yaml(self.train_dataset.norm_config, os.path.join(DIR_TO_DATA, f"data/amortizer/models/{self.name}/data_normalization_setting.yaml"))
        save_dict_to_yaml(self.train_dataset.sim_config, os.path.join(DIR_TO_DATA, f"data/amortizer/models/{self.name}/simulator_setting.yaml"))
        
        self.clipping = float("Inf")

    
    def train(
        self,
        n_iter=500,
        step_per_iter=2048,
        batch_sz=64,
        n_trial=64,
        board=True,
        save_freq=10,
    ):
        """
        Training loop

        n_iter (int): Number of training iterations
        step_per_iter (int): Number of training steps per iteration
        batch_sz (int): Batch size
        n_trial (int): Number of trials for each user data
        board (bool): Whether to use tensorboard (default: True)
        """
        iter = self.iter
        last_step = self.iter * step_per_iter

        self.logger = Logger(
            self.name,
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
                # self.logger.write_scalar(**valid_res)
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
            os.makedirs(self.model_path, exist_ok=True)
            ckpt_path = f"{self.model_path}/iter{iter:03d}.pt"
        else:
            os.makedirs(path, exist_ok=True)
            ckpt_path = path + f"iter{iter:03d}.pt"
        torch.save({
            "iteration": iter,
            "model_state_dict": self.amortizer.state_dict(),
            "optim_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, ckpt_path)
        

    def load(self, model_name):
        """
        Load model, optimizer, and scheduler from the latest checkpoint
        """
        assert model_name is not None, "You must specify the model name to be loaded."
        import glob
        os.path.join(DIR_TO_DATA, f"data/amortizer/models/{self.name}/pts")
        ckpt_paths = glob.glob(os.path.join(DIR_TO_DATA, f"data/amortizer/models/{model_name}/pts/iter*.pt"))
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]

        self.name = model_name
        self.model_path = os.path.join(DIR_TO_DATA, f"data/amortizer/models/{self.name}/pts")
        self.board_path = os.path.join(DIR_TO_DATA, f"data/amortizer/models/{self.name}/board")
        self.result_path = os.path.join(DIR_TO_DATA, f"data/amortizer/models/{self.name}/results")

        ckpt = torch.load(ckpt_path, map_location=self.amortizer.device.type)
        self.amortizer.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optim_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        iter = ckpt["iteration"]
        self.scheduler.step(iter)

        print(f"[ amortizer - loaded checkpoint ] Model {model_name} - Iteration {iter}")

        self.iter = iter
    

    def valid(self,
        n_trial=np.array([20, 50, 100, 200, 500]),
        n_sample=300,   # Sample for distribution estimation
        verbose=True
    ):
        self.amortizer.eval()
        valid_res = dict()

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
                    surfix="sim",
                )
                rsq.append(r2s)
            rsq = np.array(rsq)

            os.makedirs(f"{self.result_path}/iter{self.iter:03d}", exist_ok=True)
            pd.DataFrame(dict(zip(
                ["n", *self.target_param],
                [n_trial, *rsq.T]
            ))).to_csv(f"{self.result_path}/iter{self.iter:03d}/recovery_r2_{infer_type}.csv", index=False)

        if verbose:
            print(f"- parameter recovery (simulated) ({time() - start_t:.3f}s)")
        
        return valid_res
    

    def parameter_recovery(
        self,
        result_log,
        gt_params,
        valid_data,
        n_sample,
        n_datasize,
        infer_type,
        surfix="",
    ):
        os.makedirs(f"{self.result_path}/iter{self.iter:03d}/recovery_{infer_type}/", exist_ok=True)

        # Note: all parameters are normalized: -1 ~ 1
        n_param = gt_params.shape[0]
        inferred_params = list()

        for param_i in range(n_param):
            stat_i, traj_i = valid_data[param_i] # valid_data := list[(stat, traj)]
            param_z = self.amortizer.infer(
                stat_i, traj_i, 
                n_sample=n_sample, 
                type=infer_type
            )
            param_z = self._clip_params(param_z)
            p = self.convert_param_z_to_w(param_z)[0]
            inferred_params.append(p)
        inferred_params = np.array(inferred_params)
                
        gt_params = self.convert_param_z_to_w(gt_params)
        
        pickle_save(
            f"{self.result_path}/iter{self.iter:03d}/recovery_{infer_type}/trial{n_datasize:03d}.pkl",
            (gt_params, inferred_params)
        )
        
        fig, axs = figure_grid(1, len(self.target_param), size_ax=3)
        rsq_list = list()

        for i, p in enumerate(self.target_param):
            rsq = draw_r2_plot(
                ax=axs[i],
                xdata=gt_params[:,i],
                ydata=inferred_params[:,i],
                xlabel=f"True {PARAM_SYMBOL[p]}",
                ylabel=f"Inferred {PARAM_SYMBOL[p]}",
            )
            result_log[f"Parameter_Recovery/{infer_type}/r2_{n_datasize:03d}_{p}_{surfix}"] = rsq
            rsq_list.append(rsq)

        figure_save(fig, f"{self.result_path}/iter{self.iter:03d}/recovery_{infer_type}/trial{n_datasize:03d}.png")

        return rsq_list

        
    def _clip_params(self, params):
        return np.clip(
            params,
            np.array([-1.] * len(self.target_param)),
            np.array([1.] * len(self.target_param))
        )

    def convert_param_z_to_w(self, outputs):
        return self.train_dataset.simulator.convert_param_z_to_w(outputs)


