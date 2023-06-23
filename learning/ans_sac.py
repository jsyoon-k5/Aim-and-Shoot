"""
Soft Actor Critic
for Aim-and-Shoot task training

Both fixed parameter training and modulation are supported.

Code written by June-Seop Yoon
with help of Hee-Seung Moon
"""

import gym
import numpy as np
import pandas as pd
import os, sys, copy

sys.path.append("..")
from agent.ans_agent import *
from utilities.utils import now_to_string
from configs.path import *

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement

from learning.sac_policy import ModulatedSACPolicy

# tensorboard --logdir={tb log directory}

# Normal RL with fixed parameters
class SACTrain:
    def __init__(
        self, 
        model_name: str = None, 
        variable_mean: dict = USER_PARAM_MEAN,
        variable_std: dict = USER_PARAM_STD,
        variable_max: dict = USER_PARAM_MAX,
        variable_min: dict = USER_PARAM_MIN,
        z_scale_range: int = MAX_SIGMA_SCALE,
        param_scale_z: dict = None,
        param_scale_w: dict = None
    ):
        # Set model name
        if model_name is None:
            self.model_name = now_to_string()
        else: self.model_name = f"{model_name}_{now_to_string()}"

        # Environments
        self.env = Env(
            variable_mean=variable_mean,
            variable_std=variable_std,
            variable_max=variable_max,
            variable_min=variable_min,
            z_scale_range=z_scale_range,
            param_scale_z=param_scale_z,
            param_scale_w=param_scale_w
        )
        self.eval_env = copy.deepcopy(self.env)
        self.param_scale_z = param_scale_z

        self.cb_list = None     # Callback list

        self.param_info = copy.deepcopy(self.env.user_params)
        self.train_info = dict()

    
    def create_paths(self):
        # Set directories for log file, trained policy, checkpoints, etc.
        os.makedirs(PATH_RL_CHECKPT % self.model_name, exist_ok=True)
        os.makedirs(PATH_RL_BEST % self.model_name, exist_ok=True)
        os.makedirs(PATH_RL_LOG % self.model_name, exist_ok=True)
    

    def set_callbacks(
        self,
        save_freq=5e5,
        eval_freq=2e3,
        eval_ep=256,
        stop_if_no_improve_for=None
    ):
        # Callbacks
        checkpt_cb = CheckpointCallback(
            save_freq=save_freq,
            save_path=PATH_RL_CHECKPT % self.model_name
        )

        if stop_if_no_improve_for is not None:
            max_no_improvement = stop_if_no_improve_for // eval_freq
            min_eval_n = max_no_improvement * 2
            stop_train_cb = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=max_no_improvement, 
                min_evals=min_eval_n, 
                verbose=1
            )
        else:
            stop_train_cb = None

        eval_cb = EvalCallback(
                self.eval_env, 
                eval_freq=eval_freq, 
                best_model_save_path=PATH_RL_BEST % self.model_name,
                callback_after_eval=stop_train_cb, 
                n_eval_episodes=eval_ep,
                verbose=1
        )
        self.cb_list = CallbackList([checkpt_cb, eval_cb])
    

    def set_model(
        self,
        n_layer_unit=512,
        n_layer_depth=3,
        batch_size=1024,
        lr=1e-5,
        ent=0.1,
    ):
        self.model = SAC(
            "MlpPolicy",
            self.env,
            tensorboard_log=PATH_RL_LOG % self.model_name,
            verbose=1,
            batch_size=batch_size,
            ent_coef=ent,
            learning_rate=lr,
            policy_kwargs={
                "net_arch": [n_layer_unit] * n_layer_depth
            }
        )
        self.train_info["learning_rate"] = lr
        self.train_info["entropy"] = ent
        self.train_info["batch_size"] = batch_size
        self.train_info["n_layer_unit"] = n_layer_unit
        self.train_info["n_layer_depth"] = n_layer_depth

        self.__save_info()
    

    def run_train(
        self,
        train_steps=2e6,
        log_freq=1e2
    ):
        self.model.learn(
            train_steps,
            log_interval=log_freq,
            tb_log_name=self.model_name,
            callback=self.cb_list
        )
    

    def __save_info(self):
        ti = pd.DataFrame([self.train_info])
        ti.to_csv(PATH_RL_INFO % (self.model_name, "train"), index=False)
        pi = pd.DataFrame([self.param_info])
        pi.to_csv(PATH_RL_INFO % (self.model_name, "param"), index=False)



class ModulatedSACTrain(SACTrain):
    def __init__(
        self,
        model_name: str = None, 
        modul_space: str = "cog",
        variables: list = COG_SPACE, 
        variable_mean: dict = USER_PARAM_MEAN,
        variable_std: dict = USER_PARAM_STD,
        variable_max: dict = USER_PARAM_MAX,
        variable_min: dict = USER_PARAM_MIN,
        z_scale_range: int = MAX_SIGMA_SCALE,
        eval_preset_name: str = 'cog256',
        eval_cond_name: str = "task_cond_256",
        eval_rep: int = 2
    ):
        super().__init__(model_name=model_name)
        if modul_space is not None:
            assert modul_space in PARAM_SPACE.keys()
            variables = PARAM_SPACE[modul_space]
            eval_preset_name = f"{modul_space}256"

        self.env = VariableEnv(
            variables=variables,
            variable_mean=variable_mean,
            variable_std=variable_std,
            variable_max=variable_max,
            variable_min=variable_min,
            z_scale_range=z_scale_range
        )
        self.eval_env = EvalEnv(
            variables=variables,
            variable_mean=variable_mean,
            variable_std=variable_std,
            variable_max=variable_max,
            variable_min=variable_min,
            z_scale_range=z_scale_range,
            eval_preset_name=eval_preset_name,
            eval_cond_name=eval_cond_name
        )
        self.n_eval_rep = eval_rep

        self.param_info = dict()
        self.param_mean_info = dict()
        self.param_std_info = dict()
        for v in variables:
            self.param_mean_info[v] = self.env.var_mean[v]
            self.param_std_info[v] = self.env.var_std[v]

        for v in self.env.user_params.keys():
            if v not in variables:
                self.param_info[v] = self.env.user_params[v]
        self.param_info["z_scale_range"] = z_scale_range

        self.train_info = dict()


    def set_model(
        self,
        n_layer_unit=1024,
        n_layer_depth=3,
        batch_size=1024,
        concat_layers=[0, 1, 2, 3],
        embed_net_arch=None,
        lr=1.5e-6,
        ent=0.1,
    ):
        self.model = SAC(
            ModulatedSACPolicy,
            self.env,
            tensorboard_log=PATH_RL_LOG % self.model_name,
            verbose=1,
            batch_size=batch_size,
            ent_coef=ent,
            learning_rate=lr,
            policy_kwargs={
                "net_arch": [n_layer_unit] * n_layer_depth,
                "sim_param_dim": len(self.env.variables),
                "concat_layers": concat_layers,       # 0 - Observation, 1~: Hidden layer
                "embed_net_arch": embed_net_arch,     # Generate concat values through net
                # "optimizer_kwargs": {
                #     "betas": (0.9, 0.999)
                # }
            }
        )

        self.train_info["learning_rate"] = lr
        self.train_info["entropy"] = ent
        self.train_info["batch_size"] = batch_size
        self.train_info["n_layer_unit"] = n_layer_unit
        self.train_info["n_layer_depth"] = n_layer_depth
        self.train_info["concat_layers"] = concat_layers
        self.train_info["embed_net_arch"] = embed_net_arch

        self.__save_info()
    

    def set_callbacks(
        self,
        save_freq=5e5,
        eval_freq=2e3,
        stop_if_no_improve_for=None
    ):
        super().set_callbacks(
            save_freq=save_freq,
            eval_freq=eval_freq,
            eval_ep=self.eval_env.z_preset_num * self.n_eval_rep,
            stop_if_no_improve_for=stop_if_no_improve_for
        )

    
    def run_train(
        self,
        train_steps=1e7,
        log_freq=1e2,
    ):
        self.model.learn(
            train_steps,
            log_interval=log_freq,
            eval_env=self.eval_env,
            n_eval_episodes=self.eval_env.z_preset_num * self.n_eval_rep,
            tb_log_name=self.model_name,
            callback=self.cb_list
        )
    

    def __save_info(self):
        ti = pd.DataFrame([self.train_info])
        ti.to_csv(PATH_RL_INFO % (self.model_name, "train"), index=False)
        pi = pd.DataFrame([self.param_info])
        pi.to_csv(PATH_RL_INFO % (self.model_name, "param_fix"), index=False)
        pmi = pd.DataFrame([self.param_mean_info])
        pmi.to_csv(PATH_RL_INFO % (self.model_name, "param_mean"), index=False)
        psi = pd.DataFrame([self.param_std_info])
        psi.to_csv(PATH_RL_INFO % (self.model_name, "param_std"), index=False)



if __name__ == "__main__":
    # x = SACTrain(
    #     model_name="action_normalized"
    # )
    # x.create_paths()
    # x.set_callbacks()
    # x.set_model(lr=1e-4)
    # x.run_train(train_steps=8e6)

    x = ModulatedSACTrain(
        model_name = f"cog_{now_to_string()}"
    )
    x.create_paths()
    x.set_callbacks()
    x.set_model()
    x.run_train(train_steps=2e7)