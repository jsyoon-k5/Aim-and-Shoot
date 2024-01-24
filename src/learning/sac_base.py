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

from agent.agents import *
from utils.utils import now_to_string
from configs.path import *

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (CallbackList, EvalCallback, CheckpointCallback)

from learning.sac_policy import ModulatedSACPolicy
from learning.sac_logger import TensorboardStdCallback

# tensorboard --logdir={tb log directory}

# Normal RL with fixed parameters
class SACTrain:
    def __init__(
        self, 
        model_name=None,
        env_class=EnvDefault,
        env_setting=USER_CONFIG_1
    ):
        # Set model name
        model_name = "ind" if model_name is None else model_name
        self.model_name = f"{model_name}_{now_to_string()}"

        # Environments
        self.env = env_class(env_setting=env_setting)

        self.cb_list = None     # Callback list

        self.param_info = copy.deepcopy(self.env.user_params)
        self.train_info = dict()

        # Save configuration
        os.makedirs(PATH_RL_CHECKPT % self.model_name, exist_ok=True)
        os.makedirs(PATH_RL_BEST % self.model_name, exist_ok=True)
        os.makedirs(PATH_RL_LOG % self.model_name, exist_ok=True)

        pickle_save(PATH_RL_INFO % (self.model_name, "env_class.pkl"), env_class)
        pickle_save(PATH_RL_INFO % (self.model_name, "configuration.pkl"), env_setting)
    

    def set_callbacks(
        self,
        save_freq=5e5,
        eval_freq=1e5,
        eval_ep=2048,
    ):
        # Callbacks
        checkpt_cb = CheckpointCallback(
            save_freq=save_freq,
            save_path=PATH_RL_CHECKPT % self.model_name,
            save_replay_buffer=True
        )

        eval_cb = EvalCallback(
                eval_env=self.env, 
                eval_freq=eval_freq, 
                best_model_save_path=PATH_RL_BEST % self.model_name,
                callback_after_eval=None, 
                n_eval_episodes=eval_ep,
                verbose=1
        )

        std_cb = TensorboardStdCallback()

        self.cb_list = CallbackList([checkpt_cb, eval_cb, std_cb])
    

    def set_model(
        self,
        n_layer_unit=512,
        n_layer_depth=3,
        batch_size=1024,
        lr=1e-6,
        train_freq=1,
        gamma=1,
        ent='auto',
    ):
        self.model = SAC(
            "MlpPolicy",
            self.env,
            tensorboard_log=PATH_RL_LOG % self.model_name,
            verbose=1,
            batch_size=batch_size,
            ent_coef=ent,
            learning_rate=lr,
            gamma=gamma,
            train_freq=(train_freq, "step"),
            policy_kwargs={
                "net_arch": [n_layer_unit] * n_layer_depth
            }
        )
        self.train_info["learning_rate"] = lr
        self.train_info["entropy"] = ent
        self.train_info["batch_size"] = batch_size
        self.train_info["n_layer_unit"] = n_layer_unit
        self.train_info["n_layer_depth"] = n_layer_depth
        self.train_info["train_freq"] = train_freq
        self.train_info["gamma"] = gamma

        self.__save_info()
    

    def run_train(
        self,
        train_steps=1e7,
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
        ti.to_csv(PATH_RL_INFO % (self.model_name, "train.csv"), index=False)
        pi = pd.DataFrame([self.param_info])
        pi.to_csv(PATH_RL_INFO % (self.model_name, "param.csv"), index=False)



class MultiProcessingSACTrain(SACTrain):
    def __init__(
        self, 
        model_name=None,
        env_class=EnvDefault,
        env_setting=USER_CONFIG_1,
        num_cpu=8
    ):
        # Set model name
        model_name = "indpc" if model_name is None else model_name
        self.model_name = f"{model_name}_{now_to_string()}"

        # Create instances of env_class for each subprocess
        def make_class():
            class temp_class(env_class):
                def __init__(self):
                    super().__init__(env_setting=env_setting)

            return temp_class

        # Initialize SubprocVecEnv with the created environments
        self.env = SubprocVecEnv([make_class() for i in range(num_cpu)])
        sample_env = make_class()()

        self.cb_list = None     # Callback list

        self.param_info = copy.deepcopy(sample_env.user_params)
        self.train_info = dict()

        # Save configuration
        os.makedirs(PATH_RL_CHECKPT % self.model_name, exist_ok=True)
        os.makedirs(PATH_RL_BEST % self.model_name, exist_ok=True)
        os.makedirs(PATH_RL_LOG % self.model_name, exist_ok=True)

        pickle_save(PATH_RL_INFO % (self.model_name, "env_class.pkl"), env_class)
        pickle_save(PATH_RL_INFO % (self.model_name, "configuration.pkl"), env_setting)



class ModulatedSACTrain(SACTrain):
    def __init__(
        self, 
        model_name=None,
        env_class=VariableEnvDefault,
        env_setting=USER_CONFIG_1
    ):
        model_name = "mod" if model_name is None else model_name
        self.model_name = f"{model_name}_{now_to_string()}"

        self.env = env_class(env_setting=env_setting)

        self.param_fix_info = dict()
        self.param_range_info = dict()
        self.param_min_info = dict()
        for i, v in enumerate(self.env.variables):
            self.param_range_info[v] = [*self.env.variable_range[i]]

        for v in self.env.user_params.keys():
            if v not in self.env.variables:
                self.param_fix_info[v] = self.env.user_params[v]

        self.train_info = dict()

        # Save configuration
        os.makedirs(PATH_RL_CHECKPT % self.model_name, exist_ok=True)
        os.makedirs(PATH_RL_BEST % self.model_name, exist_ok=True)
        os.makedirs(PATH_RL_LOG % self.model_name, exist_ok=True)

        pickle_save(PATH_RL_INFO % (self.model_name, "env_class.pkl"), env_class)
        pickle_save(PATH_RL_INFO % (self.model_name, "configuration.pkl"), env_setting)


    def set_model(
        self,
        n_layer_unit=1024,
        n_layer_depth=3,
        batch_size=1024,
        concat_layers=[0, 1, 2, 3],
        embed_net_arch=None,
        lr=1e-6,
        ent='auto',
        train_freq=1,
        gamma=1,
    ):

        self.model = SAC(
            ModulatedSACPolicy,
            self.env,
            tensorboard_log=PATH_RL_LOG % self.model_name,
            verbose=1,
            batch_size=batch_size,
            ent_coef=ent,
            learning_rate=lr,
            gamma=gamma,
            train_freq=(train_freq, "step"),
            policy_kwargs={
                "net_arch": [n_layer_unit] * n_layer_depth,
                "sim_param_dim": self.env.z_size,
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
        self.train_info["train_freq"] = train_freq
        self.train_info["gamma"] = gamma
        self.train_info["embed_net_arch"] = embed_net_arch

        self.__save_info()
    

    def set_callbacks(
        self,
        save_freq=5e5,
        eval_freq=5e4,
        eval_ep=2048,
    ):
        super().set_callbacks(save_freq=save_freq, eval_freq=eval_freq, eval_ep=eval_ep)

    
    def load_model(
        self,
        model_name='',
        ckpt='best'
    ):
        if ckpt == 'best':
            path = f"{PATH_RL_BEST % model_name}best_model.zip"
        else:
            path = f"{PATH_RL_CHECKPT % model_name}rl_model_{ckpt}_steps.zip"

        self.model.load(path)
        self.model.set_env(self.env)

        info = pd.read_csv(PATH_RL_INFO % (self.model_name, "train"))
        info["loaded_model"] = [model_name]
        info["loaded_ckpt"] = [ckpt]
        info.to_csv(PATH_RL_INFO % (self.model_name, "train"), index=False)

    
    def run_train(
        self,
        train_steps=1e7,
        log_freq=1e2,
    ):
        self.model.learn(
            train_steps,
            log_interval=log_freq,
            # eval_env=self.eval_env,
            # n_eval_episodes=self.eval_env.z_preset_num * self.n_eval_rep,
            tb_log_name=self.model_name,
            callback=self.cb_list
        )
    

    def __save_info(self):
        ti = pd.DataFrame([self.train_info])
        ti.to_csv(PATH_RL_INFO % (self.model_name, "train.csv"), index=False)
        pfi = pd.DataFrame([self.param_fix_info])
        pfi.to_csv(PATH_RL_INFO % (self.model_name, "param_fix.csv"), index=False)
        pri = pd.DataFrame(self.param_range_info)
        pri.to_csv(PATH_RL_INFO % (self.model_name, "param_modulate.csv"), index=False)



class MultiProcessingModulatedSACTrain(ModulatedSACTrain):
    def __init__(
        self, 
        model_name=None,
        env_class=VariableEnvDefault,
        env_setting=USER_CONFIG_1,
        num_cpu=4
    ):
        model_name = "modpc" if model_name is None else model_name
        self.model_name = f"{model_name}_{now_to_string()}"

        # Create instances of env_class for each subprocess
        def make_class():
            class temp_class(env_class):
                def __init__(self):
                    super().__init__(env_setting=env_setting)

            return temp_class

        # Initialize SubprocVecEnv with the created environments
        self.env = SubprocVecEnv([make_class() for i in range(num_cpu)])
        sample_env = make_class()()

        self.param_fix_info = dict()
        self.param_range_info = dict()
        self.param_min_info = dict()
        for i, v in enumerate(sample_env.variables):
            self.param_range_info[v] = [*sample_env.variable_range[i]]

        for v in sample_env.user_params.keys():
            if v not in sample_env.variables:
                self.param_fix_info[v] = sample_env.user_params[v]

        self.train_info = dict()

        # Save configuration
        os.makedirs(PATH_RL_CHECKPT % self.model_name, exist_ok=True)
        os.makedirs(PATH_RL_BEST % self.model_name, exist_ok=True)
        os.makedirs(PATH_RL_LOG % self.model_name, exist_ok=True)

        pickle_save(PATH_RL_INFO % (self.model_name, "env_class.pkl"), env_class)
        pickle_save(PATH_RL_INFO % (self.model_name, "configuration.pkl"), env_setting)

        self.z_size = len(env_setting["params_modulate"])
        self.num_cpu = num_cpu


    def set_model(
        self,
        n_layer_unit=1024,
        n_layer_depth=3,
        batch_size=1024,
        concat_layers=[0, 1, 2, 3],
        embed_net_arch=None,
        lr=1e-4,
        ent='auto',
        train_freq=1,
        gamma=0.99,
        target_ent=0.08,
    ):
        self.model = SAC(
            ModulatedSACPolicy,
            self.env,
            tensorboard_log=PATH_RL_LOG % self.model_name,
            verbose=1,
            batch_size=batch_size,
            ent_coef=ent,
            learning_rate=lr,
            gamma=gamma,
            target_entropy=target_ent,
            train_freq=(train_freq, "step"),
            policy_kwargs={
                "net_arch": [n_layer_unit] * n_layer_depth,
                "sim_param_dim": self.z_size,
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
        self.train_info["train_freq"] = train_freq
        self.train_info["gamma"] = gamma
        self.train_info["target_entropy"] = target_ent
        self.train_info["embed_net_arch"] = embed_net_arch

        ti = pd.DataFrame([self.train_info])
        ti.to_csv(PATH_RL_INFO % (self.model_name, "train.csv"), index=False)
        pfi = pd.DataFrame([self.param_fix_info])
        pfi.to_csv(PATH_RL_INFO % (self.model_name, "param_fix.csv"), index=False)
        pri = pd.DataFrame(self.param_range_info)
        pri.to_csv(PATH_RL_INFO % (self.model_name, "param_modulate.csv"), index=False)
    

    def load_model(
        self,
        model_name='',
        ckpt='best'
    ):
        if ckpt == 'best':
            path = f"{PATH_RL_BEST % model_name}best_model.zip"
        else:
            path = f"{PATH_RL_CHECKPT % model_name}rl_model_{ckpt}_steps.zip"

        self.model.set_parameters(path)
        self.model.set_env(self.env)

        info = pd.read_csv(PATH_RL_INFO % (self.model_name, "train.csv"))
        info["loaded_model"] = [model_name]
        info["loaded_ckpt"] = [ckpt]
        info.to_csv(PATH_RL_INFO % (self.model_name, "train.csv"), index=False)


    def set_callbacks(
        self,
        save_freq=1e6,
        eval_freq=5e4,
        eval_ep=2048,
    ):
        super().set_callbacks(
            save_freq=save_freq//self.num_cpu, 
            eval_freq=eval_freq//self.num_cpu, 
            eval_ep=eval_ep//self.num_cpu
        )



if __name__ == "__main__":
    pass
    