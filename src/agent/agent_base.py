'''
Aim-and-Shoot AGENT - base class

Code written by June-Seop Yoon
with help of Seungwon Do, Hee-Seung Moon
'''

from copy import deepcopy
from collections import deque

import sys, os, pickle

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.envs.registration import EnvSpec, spec
gym.logger.set_level(40)

sys.path.append("..")

from agent.fps_task import GameState

from configs.simulation import *
from configs.common import *
from configs.experiment import *
from utils.mymath import *


class BaseEnv(gym.Env):
    def __init__(
        self,
        # seed=None,
        user_params: dict=dict(),
        spec_name: str='ans-default'
    ):
        # self.seed(seed=seed)
        self.viewer = None
        self.spec = EnvSpec(spec_name)
        self.env_name = "base_agent"

        # Observation and Action space
        self.observation_space = None
        self.action_space = None

        self.user_params = user_params.copy()

        self.game_setting = dict(
            pcam=None,
            tgpos=None,
            tmpos=None,
            toax=None,
            session=None,
            gpos=None,
            hrt=None,
            grt=None,
            head_pos=None,
            follow_exp_prior=False
        )
        self.game = GameState()


        # Cognitive state - this is what user perceives and remember
        self.cstate = dict()

        # Motor plans
        self.ongoing_mp_ideal = dict(
            p = np.zeros((BUMP_INTERVAL//MUSCLE_INTERVAL+1, 2)),
            v = np.zeros((BUMP_INTERVAL//MUSCLE_INTERVAL+1, 2)),
        )
        self.ongoing_mp_actual = dict(
            p = np.zeros((BUMP_INTERVAL//MUSCLE_INTERVAL+1, 2)),
            v = np.zeros((BUMP_INTERVAL//MUSCLE_INTERVAL+1, 2)),
        )
        self.shoot_mp_ideal = dict(
            p = None,
            v = None
        )
        self.shoot_mp_actual = dict(
            p = None,
            v = None
        )
        
        # Basic environment setting
        self.time = 0
        self.result = 0

        self.bump_plan_wait = 0
        self.shoot_timing = 999
        self.shoot_mp_generated = False
        self.delayed_time = 0
        self.gaze_cooldown = 0
        self.forced_termination = False

        self.time_mean = deque(maxlen=1000)
        self.error_rate = deque(maxlen=1000)

        # Episode history
        self.h_traj_p = []      # [P1, P2, ...]
        self.h_traj_v = []      # [V1, V2, ...]
        self.g_traj_p = []      # [P1, P2, ...]
        self.g_traj_t = []      # [T1, T2, ...] - interval btw pos
    
        self.initial_game_cond = dict()
    

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def update_game_setting(self, game_setting=dict()):
        self.game_setting.update(game_setting)
        return

    
    def episode_trajectory(self):
        return (
            self.h_traj_p,
            self.h_traj_v,
            self.g_traj_p,
            self.g_traj_t,
            self.delayed_time
        )


    def set_user_param(
        self, 
        param_scale_z:dict=None, 
        param_scale_w:dict=None
    ):
        pass


    def np_cstate(self):
        pass
    

    def step(self, action):
        """
        Serial process inside a single BUMP
        """
        # return self.np_cstate(), rew_s, done, info
        pass


    def reset(self, seed=None):
        # Reset game
        self.initial_game_cond = self.game.reset(**self.game_setting)

        # Reaction sync
        # Set hand reaction time to 400ms (4 BUMP INTERVAL)
        tp = BUMP_INTERVAL / 1000
        self.delayed_time = 4 * tp - self.initial_game_cond["hrt"]
        self.gaze_cooldown = self.initial_game_cond["grt"] + self.delayed_time
        self.bump_plan_wait = 3

        self.time = -self.delayed_time      # Reversed time
        self.game.orbit(-self.delayed_time)

        # Pass BUMPs where hand & gaze are both stationary
        num_of_stat = min(int(self.gaze_cooldown*1000), 3 * BUMP_INTERVAL) // BUMP_INTERVAL
        self.gaze_cooldown -= num_of_stat * tp
        self.bump_plan_wait -= num_of_stat
        self.time += num_of_stat * tp      # Elapsed time

        self.meffort = 0    # Hand movement effort
        self.result = 0     # Shoot result

        self.shoot_timing = 999          # Remaining shooting moment
        self.forced_termination = False     # Episode terminated due to cond. violation
        self.shoot_mp_generated = False

        self.h_traj_p = np.zeros((num_of_stat*BUMP_INTERVAL//MUSCLE_INTERVAL+1, 2))
        self.h_traj_v = np.zeros((num_of_stat*BUMP_INTERVAL//MUSCLE_INTERVAL+1, 2))
        self.g_traj_p = np.array([self.initial_game_cond["gpos"], self.initial_game_cond["gpos"]])
        self.g_traj_t = np.array([0, num_of_stat * tp]) - self.delayed_time

        self.ongoing_mp_ideal.update(
            dict(
                p = np.zeros((BUMP_INTERVAL//MUSCLE_INTERVAL+1, 2)),
                v = np.zeros((BUMP_INTERVAL//MUSCLE_INTERVAL+1, 2)),
            )
        )
        self.ongoing_mp_actual.update(
            dict(
                p = np.zeros((BUMP_INTERVAL//MUSCLE_INTERVAL+1, 2)),
                v = np.zeros((BUMP_INTERVAL//MUSCLE_INTERVAL+1, 2)),
            )
        )
        self.shoot_mp_ideal = dict(
            p = None,
            v = None
        )
        self.shoot_mp_actual = dict(
            p = None,
            v = None
        )

        # State and game update to passed BUMP
        self.game.orbit(num_of_stat * tp)