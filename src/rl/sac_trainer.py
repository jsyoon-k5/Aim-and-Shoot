"""
Soft Actor Critic
for Point-and-Click agent training

Code written by June-Seop Yoon
"""
import os, psutil
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import (CallbackList, EvalCallback, CheckpointCallback)

from ..utils.myutils import get_timebase_session_name, save_dict_to_yaml
from ..agent.agent_manager import ANS_ENV
from ..config.config import RLC
from ..config.constant import FOLDER
from .sac_policy import ModulatedSACPolicy
from .sac_logger import TensorboardStdCallback

# tensorboard --logdir={tb log directory}
DIR_TO_DATA = Path(__file__).parent.parent.parent

# Normal RL with fixed parameters
class SACTrainer:
    def __init__(
        self,
        model_name_prefix=None,
        num_cpu=16,
        train_config="default",
        env_class="default",
        env_setting=dict(
            agent_name="default",
            game_env_name="default",
            interval_name="default",
        )
    ):
        # Set model name
        self.model_name = get_timebase_session_name() if model_name_prefix is None \
            else f"{model_name_prefix}_{get_timebase_session_name()}"
        
        # Configuration
        self.config = RLC[train_config]
    
        # Create instances of env_class for each subprocess
        def make_class():
            class ans_env_class(ANS_ENV[env_class]):
                def __init__(self):
                    super().__init__(**env_setting)
            return ans_env_class

        # Environments
        self.num_cpu = min(num_cpu, psutil.cpu_count(logical=False))
        self.env = SubprocVecEnv([make_class() for _ in range(num_cpu)])
        temp_env = make_class()()
        env_info = temp_env.env_setting_info
        env_info["env_class"] = env_class
        save_dict_to_yaml(
            temp_env.env_setting_info, 
            os.path.join(DIR_TO_DATA, f"data/{FOLDER.DATA_RL_MODEL}/{self.model_name}/env_info.yaml")
        )
        self.param_modul_length = len(temp_env.param_mod_z)
        
        self.cb_list = CallbackList([])

        # Some pre executions for directory
        os.makedirs(os.path.join(DIR_TO_DATA, f"data/{FOLDER.DATA_RL_MODEL}/{self.model_name}/{FOLDER.RL_TENSORBOARD}"), exist_ok=True)
        self.set_callbacks()
        self.set_model()

    
    def set_callbacks(self):
        # Callbacks
        checkpt_cb = CheckpointCallback(
            save_freq=self.config.callback.save_freq // self.num_cpu,
            save_path=os.path.join(DIR_TO_DATA, f"data/{FOLDER.DATA_RL_MODEL}/{self.model_name}/{FOLDER.MODEL_CHECKPT}"),
            save_replay_buffer=True
        )
        eval_cb = EvalCallback(
            eval_env=self.env, 
            eval_freq=self.config.callback.eval_freq // self.num_cpu, 
            best_model_save_path=os.path.join(DIR_TO_DATA, f"data/{FOLDER.DATA_RL_MODEL}/{self.model_name}/{FOLDER.MODEL_BEST}"),
            callback_after_eval=None, 
            n_eval_episodes=self.config.callback.eval_ep // self.num_cpu,
            verbose=1
        )
        logging_cb = TensorboardStdCallback()
        self.cb_list = CallbackList([checkpt_cb, eval_cb, logging_cb])
    

    def set_model(self):
        self.model = SAC(
            ModulatedSACPolicy,
            self.env,
            tensorboard_log=os.path.join(DIR_TO_DATA, f"data/{FOLDER.DATA_RL_MODEL}/{self.model_name}/{FOLDER.RL_TENSORBOARD}"),
            verbose=1,
            batch_size=self.config.model.batch,
            ent_coef=self.config.model.entropy,
            learning_rate=self.config.model.lr,
            gamma=self.config.model.gamma,
            target_entropy=self.config.model.target_entropy,
            train_freq=(1, "step"),
            policy_kwargs={
                "net_arch": self.config.model.mlp.arch,
                "sim_param_dim": self.param_modul_length,
                "concat_layers": self.config.model.mlp.concat,       # 0 - Observation, 1~: Hidden layer
                "embed_net_arch": self.config.model.mlp.embed,     # Generate concat values through net
            }
        )


    def update_config(self, keys, value):
        key = keys[0]
        if len(keys) == 1:
            self.config[key] = value
            return
        self.update_config(keys[1:], value)

    
    def update_trainer(self):
        # Run this function after configuration complete
        self.set_callbacks()
        self.set_model()


    def run(self):
        # Save configuration and start training
        save_dict_to_yaml(
            self.config, 
            os.path.join(DIR_TO_DATA, f"data/{FOLDER.DATA_RL_MODEL}/{self.model_name}/train_config.yaml")
        )
        self.model.learn(
            self.config.train.step,
            log_interval=self.config.train.log_freq,
            tb_log_name=self.model_name,
            callback=self.cb_list
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Training option')

    parser.add_argument('--num_cpu', type=int, default=psutil.cpu_count(logical=False))
    parser.add_argument('--agent_type', type=str, default="default")
    parser.add_argument('--ans_type', type=str, default="default")
    parser.add_argument('--env_class_type', type=str, default="default")
    parser.add_argument('--rl_setting', type=str, default="default")

    args = parser.parse_args()

    trainer = SACTrainer(
        num_cpu=args.num_cpu,
        train_config=args.rl_setting,
        env_class=args.env_class_type,
        env_setting=dict(
            agent_name=args.agent_type,
            game_env_name=args.ans_type,
        )
    )
    trainer.run()
    