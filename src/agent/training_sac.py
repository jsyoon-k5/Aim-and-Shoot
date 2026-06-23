"""
Soft Actor Critic
for Point-and-Click agent training

Code written by June-Seop Yoon
"""
import os
import math
from copy import deepcopy
from pathlib import Path

import psutil

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure as configure_sb3_logger

from ..agent import vplayer
from ..utils.myutils import get_compact_timestamp_str, save_dict_to_yaml
from ..configs.loader import CFG_AGENT_RL
from ..configs.constants import FOLDERS
from ..utils.sac_policy_modul import ModulatedSACPolicy
from ..utils.loggers import TensorboardStdCallback, TruncationLoggingEvalCallback



# tensorboard --logdir={tb log directory}
DIR_TO_DATA = Path(__file__).parent.parent.parent

# -----------------------------------------------------------------------------
# Temporary folder constants (edit freely later)
# -----------------------------------------------------------------------------
# DIRNAME_DATA = "data"
# DIRNAME_RL_MODEL = "ans_agent_models"
# DIRNAME_RL_TENSORBOARD = "tensorboard"
# DIRNAME_MODEL_CHECKPT = "checkpoints"
# DIRNAME_MODEL_BEST = "best"

# Normal RL with fixed parameters
class LatestReplayBufferCallback(BaseCallback):
    """Periodically save replay buffer to a fixed file path (overwrite)."""

    def __init__(self, save_freq_calls: int, save_path: Path):
        super().__init__(verbose=0)
        self.save_freq_calls = max(1, int(save_freq_calls))
        self.save_path = Path(save_path)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq_calls == 0:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save_replay_buffer(str(self.save_path))
        return True


class SACTrainer:
    def __init__(
        self,
        model_name_prefix=None,
        num_cpu=16,
        train_config_preset="default",
        agent_class="AnSPlayerAgentDefault",
        agent_config=None,
        agent_config_preset="default",
    ):
        # Set model name
        self.model_name = get_compact_timestamp_str() if model_name_prefix is None \
            else f"{model_name_prefix}_{get_compact_timestamp_str()}"
        
        # Configuration
        self.config = deepcopy(CFG_AGENT_RL[train_config_preset])
        self.config["model_name"] = self.model_name
        self.agent_class = agent_class

        # Directories
        self.model_dir = DIR_TO_DATA / FOLDERS.DATA / FOLDERS.RL_MODEL / self.model_name
        self.tb_dir = self.model_dir / FOLDERS.RL_TB
        self.checkpoint_dir = self.model_dir / FOLDERS.MODEL_CHECKPT
        self.best_model_dir = self.model_dir / FOLDERS.MODEL_BEST
        os.makedirs(self.tb_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)
    
        # Create callable env factory for subprocesses
        self._env_class = getattr(vplayer, self.agent_class)

        def make_env():
            def _init():
                return self._env_class(config_preset=agent_config_preset, config=agent_config)
            return _init

        # Environments
        physical_cpu = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
        self.num_cpu = max(1, min(int(num_cpu), int(physical_cpu)))
        self.env = SubprocVecEnv([make_env() for _ in range(self.num_cpu)])

        temp_env = self._env_class(config_preset=agent_config_preset, config=agent_config)
        agent_config = deepcopy(getattr(temp_env, "config", {}))
        agent_config["agent_class"] = agent_class
        agent_config["model_name"] = self.model_name
        save_dict_to_yaml(agent_config, str(self.model_dir / "agent_config.yaml"))
        self.param_modul_length = getattr(temp_env, "n_obs_conditioning_features", 0)
        del temp_env
        
        self.cb_list = CallbackList([])

        self.set_callbacks()
        self.set_model()


    def _global_step_to_callback_calls(self, global_step_freq: int) -> int:
        """Convert global-timestep frequency to callback-call frequency for VecEnv."""
        return max(1, int(math.ceil(float(global_step_freq) / float(self.num_cpu))))

    
    def set_callbacks(self):
        # Callbacks
        save_freq_calls = self._global_step_to_callback_calls(self.config["callback"]["save_freq"])
        eval_freq_calls = self._global_step_to_callback_calls(self.config["callback"]["eval_freq"])

        checkpt_cb = CheckpointCallback(
            save_freq=save_freq_calls,
            save_path=str(self.checkpoint_dir),
            save_replay_buffer=False,
        )
        eval_cb = TruncationLoggingEvalCallback(
            eval_env=self.env, 
            eval_freq=eval_freq_calls,
            best_model_save_path=str(self.best_model_dir),
            callback_after_eval=None, 
            n_eval_episodes=max(1, int(self.config["callback"]["eval_ep"] // self.num_cpu)),
            verbose=0,
        )
        replay_latest_cb = LatestReplayBufferCallback(
            save_freq_calls=save_freq_calls,
            save_path=self.checkpoint_dir / "replay_buffer_latest.pkl",
        )
        logging_cb = TensorboardStdCallback()
        self.cb_list = CallbackList([checkpt_cb, eval_cb, replay_latest_cb, logging_cb])
    

    def set_model(self):
        self.model = SAC(
            ModulatedSACPolicy,
            self.env,
            tensorboard_log=None,
            verbose=0,
            batch_size=self.config["model"]["batch"],
            ent_coef=self.config["model"]["entropy"],
            learning_rate=self.config["model"]["lr"],
            gamma=self.config["model"]["gamma"],
            target_entropy=self.config["model"]["target_entropy"],
            train_freq=(1, "step"),
            policy_kwargs={
                "net_arch": self.config["model"]["mlp"]["arch"],
                "sim_param_dim": self.param_modul_length,
                "concat_layers": self.config["model"]["mlp"]["concat"],  # 0 - Observation, 1~: Hidden layer
                "embed_net_arch": self.config["model"]["mlp"]["embed"],  # Generate concat values through net
            }
        )

        # Write TensorBoard events directly into tb_dir (without auto run-name suffix).
        custom_logger = configure_sb3_logger(folder=str(self.tb_dir), format_strings=["tensorboard"])
        self.model.set_logger(custom_logger)


    def update_config(self, keys, value):
        if isinstance(keys, str):
            keys = keys.split(".")
        if not keys:
            raise ValueError("keys must be a non-empty path")

        cursor = self.config
        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                raise KeyError(f"Invalid config path: {'.'.join(keys)}")
            cursor = cursor[key]
        cursor[keys[-1]] = value

    
    def update_trainer(self):
        # Run this function after configuration complete
        self.set_callbacks()
        self.set_model()


    def run(self):
        # Save configuration and start training
        save_dict_to_yaml(
            self.config, 
            str(self.model_dir / "train_config.yaml"),
        )
        self.model.learn(
            self.config["train"]["step"],
            log_interval=self.config["train"]["log_freq"],
            tb_log_name=self.model_name,
            callback=self.cb_list,
            progress_bar=True,
        )



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SAC training options")

    default_cpu = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
    parser.add_argument("--num_cpu", type=int, default=16)
    parser.add_argument("--model_name_prefix", type=str, default="ga_legacy")
    parser.add_argument("--train_config_preset", type=str, default="default")
    parser.add_argument("--agent_class", type=str, default="AnSPlayerAgentLegacy")
    parser.add_argument("--agent_config_preset", type=str, default="gaze_ablated")

    args = parser.parse_args()

    trainer = SACTrainer(
        model_name_prefix=args.model_name_prefix,
        num_cpu=args.num_cpu,
        train_config_preset=args.train_config_preset,
        agent_class=args.agent_class,
        agent_config_preset=args.agent_config_preset,
    )
    trainer.run()