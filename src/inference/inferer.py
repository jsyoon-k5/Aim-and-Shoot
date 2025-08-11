import numpy as np
import pandas as pd
from pathlib import Path
import os
import glob
from box import Box
import torch
from typing import List, Optional

from ..config.constant import FIELD
from ..utils.myutils import load_config
from ..nets.amortizer import AmortizerForTrialData, RegressionForTrialData
from ..inference.ans_dataset import AnSSimulatorForAmortizer
from ..utils.mymath import linear_normalize


DIR_TO_DATA = Path(__file__).parent.parent.parent

class AnSInferer:
    def __init__(self, model_name, ckpt=None, verbose=True):
        # Configuration setting
        self.root_path = os.path.join(DIR_TO_DATA, f"data/amortizer/models/{model_name}")

        self.data_normalization = load_config(f"{self.root_path}/data_normalization_setting.yaml")
        self.simulator_config = load_config(f"{self.root_path}/simulator_setting.yaml")
        self.train_config = load_config(f"{self.root_path}/train_setting.yaml")
        self.amortizer_config = self.train_config.amortizer
        
        # Amortizer setting
        if ckpt is None:
            model_list = glob.glob(f"{self.root_path}/pts/iter*.pt")
            model_list.sort()
            model_path = model_list[-1]
            ckpt = int(model_path[-6:-3])
        else:
            model_path = f"{self.root_path}/pts/iter{ckpt:03d}.pt"
        
        amortizer_fn = RegressionForTrialData if self.train_config.point_estimation else AmortizerForTrialData

        self.amortizer = amortizer_fn(config=self.amortizer_config)
        ckpt = torch.load(model_path, map_location=self.amortizer.device.type)
        self.amortizer.load_state_dict(ckpt["model_state_dict"])
        self.amortizer.eval()
        iter = ckpt["iteration"]

        if verbose:
            print(f"[ amortizer - loaded model ] {model_name} - Iteration {iter}")
        
        # Simulator setting
        self.simulator = AnSSimulatorForAmortizer(
            model_name = self.simulator_config.model_name,
            ckpt = self.simulator_config.ckpt,
            downsample_rate = self.simulator_config.traj_downsample
        )

        # Data normalization setting
        self.stat_range = np.array([
            [self.data_normalization.stat.range[feat].min,
            self.data_normalization.stat.range[feat].max] for feat in self.data_normalization.stat.list
        ])
        self.traj_range = np.array([
            [self.data_normalization.traj.range[feat].min,
            self.data_normalization.traj.range[feat].max] for feat in self.data_normalization.traj.list
        ])

    
    def infer(self, stat, traj=None, n_sample=300, type='mean', denormalize=False):
        z = self.amortizer.infer(stat, traj_data=traj, n_sample=n_sample, type=type)
        z = np.clip(z, -1, 1)
        if not denormalize:
            return z
        return self.simulator.convert_param_z_to_w(z)[0]

    
    def process_data(self, stat:pd.DataFrame, traj:Optional[List[pd.DataFrame]]=None):
        """
        stat := pandas.DataFrame with shape (n_trial, n_feature)
        traj := list of pandas.DataFrame with each of the shape (n_tct_length, n_feature), list length n_trial
        Select target features & normalize, return numpy array
        """
        if traj is not None:
            assert stat.shape[0] == len(traj)
        
        full_stat_column = list(stat.columns)
        stat_target_indices = [full_stat_column.index(feat) for feat in self.data_normalization.stat.list]
        stat_norm = linear_normalize(
            stat.to_numpy()[:,stat_target_indices].astype(np.float32), *self.stat_range.T, dtype=np.float32
        )

        if traj is not None:
            assert FIELD.ETC.TIMESTAMP in list(traj[0].keys()), "Column 'timestamp' must be included in trajectory information."
            traj_norm = list()
            for _t in traj:
                # Downsampling
                old_timestamp = _t[FIELD.ETC.TIMESTAMP]
                new_timestamp = np.linspace(0, old_timestamp[-1], int(old_timestamp[-1] * self.simulator_config.traj_downsample))
                new_traj = [np.insert(np.diff(new_timestamp), 0, 0)]
                for key in self.data_normalization.traj.list:
                    if key != FIELD.ETC.TIMESTAMP:
                        new_traj.append(np.interp(new_timestamp, old_timestamp, _t[key]))
                new_traj = np.array(new_traj).T
                traj_norm.append(linear_normalize(new_traj, *self.traj_range.T, dtype=np.float32))

            traj_norm = np.array(traj_norm, dtype=object)
            return stat_norm, traj_norm
        
        return stat_norm