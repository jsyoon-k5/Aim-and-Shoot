import numpy as np
import pandas as pd
import os
from pathlib import Path

from ..config.constant import FIELD
from ..utils.myutils import pickle_load


class BaseDataLoader:
    def __init__(self):
        self.path_to_summary = None
        self.path_to_saccade = None

    
    def load_dataset(self, include_invalid=False, include_outlier=False, **kwargs):
        return (
            self.load_summary(include_invalid=include_invalid, include_outlier=include_outlier, **kwargs),
            self.load_trajectory(include_invalid=include_invalid, include_outlier=include_outlier, **kwargs)
        )

    
    def load_summary(self, include_invalid=False, include_outlier=False, **kwargs):
        data = pd.read_csv(self.path_to_summary)

        if not include_invalid:
            for key in FIELD.VALIDITY:
                data = data[data[FIELD.VALIDITY[key]] == 1]
        
        if not include_outlier:
            for key in FIELD.OUTLIER:
                data = data[data[FIELD.OUTLIER[key]] == 0]

        for key, value in kwargs.items():
            if type(value) is list: data = data[data[key].isin(value)]
            else: data = data[data[key] == value]
        
        return data

    def load_trajectory(self, include_invalid=False, include_outlier=False, **kwargs):
        return None


class ExperimentIJHCS(BaseDataLoader):
    def __init__(self):
        self.path_to_summary = os.path.join(
            Path(__file__).parent.parent.parent,
            f"data/exp_ijhcs/summary/general_summary.csv"
        )
        self.path_to_saccade = os.path.join(
            Path(__file__).parent.parent.parent,
            f"data/exp_ijhcs/summary/saccade_summary.csv"
        )
    
    def load_trajectory(self, include_invalid=False, include_outlier=False, **kwargs):
        data = self.load_summary(include_invalid=include_invalid, include_outlier=include_outlier, **kwargs)
        data = data[["player", "sensitivity_mode", "target_name", "block_index", "trial_index"]]

        traj_list = list()

        # Efficiently change the pickle load
        current_pkl_md = None  # (player, mode, sid, bid)
        traj_data = None

        for p, m, sid, bidx, tidx in zip(
            data["player"].to_list(), 
            data["sensitivity_mode"].to_list(), 
            data["target_name"].to_list(), 
            data["block_index"].to_list(), 
            data["trial_index"].to_list()
        ):
            if current_pkl_md is None or current_pkl_md != (p, m, sid, bidx):
                current_pkl_md = (p, m, sid, bidx)
                traj_data = self._get_traj_pickle_file(p, m, sid, bidx)
            
            traj_list.append(traj_data[tidx])
        
        return traj_list
    

    def _get_traj_pickle_file(self, player, mode, session_id, block_index):
        return pickle_load(
            os.path.join(
                Path(__file__).parent.parent.parent,
                f"data/exp_ijhcs/trajectory/{player}_{mode}_{session_id}_{block_index}.pkl"
            )
        )

