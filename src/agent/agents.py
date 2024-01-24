'''
Aim-and-Shoot AGENTs

Code written by June-Seop Yoon
with help of Seungwon Do and Hee-Seung Moon

SB3==2.1.0
'''

from copy import deepcopy
from collections import deque

import sys, os, pickle

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.envs.registration import EnvSpec, spec
from sklearn.utils.extmath import cartesian
gym.logger.set_level(40)

sys.path.append("..")

from agent import module_perceive as perception
from agent import module_aim as aim
from agent import module_gaze as gaze
from agent.fps_task import GameState

from configs.simulation import *
from configs.common import *
from configs.experiment import *
from utils.mymath import *

from agent.agent_base import BaseEnv
from agent.agents_extra import *


class EnvDefault(BaseEnv):
    """
    Observation:
        Type: Box
        Num Observation                     Min         Max         Unit
        0   Target Position X               -W/2        W/2         (m)
        1   Target Position Y               -H/2        H/2         (m)
        2   Target Velocity X (Orbit)       -LIMIT      LIMIT       (m/s)
        3   Target Velocity Y (Orbit)       -LIMIT      LIMIT       (m/s)
        4   Target Radius                   R_MIN       R_MAX       (m)
        5   Hand Velocity X                 -LIMIT      LIMIT       (m/s)
        6   Hand Velocity Y                 -LIMIT      LIMIT       (m/s)
        7   Gaze Position X                 -W/2        W/2         (m)
        8   Gaze Position Y                 -H/2        H/2         (m)
        9   Head position X                  -0.065      0.065       (m)
        10  Head position Y                  -0.0421     0.1979      (m)
        11  Head position Z                  0.408       0.742       (m)

    Actions:
        Type: Box
        0   th (Prediction Horizon)          0.1        2.0         (s)
        1   kc (Shoot Attention)              0          1          None
        2   tc (Shoot timing)                 0          1          None
        3   kg (Gaze damping)                 0          1          None

    Rewards:
        - Shot hit: 1 if the agent shot the target successfully, otherwise 0
        - Shot miss: 1 if the agent miss the shot, otherwise 0
        - Elapsed time: time interval between the decision-makings
        Aggregated reward is calculated by weighted summation of the objective terms
    
    Starting State:
        Initial player position is (0, 0, 0).
        Initial player camera direction is randomized within reference target.
        Target position is randomly sampled in player FOV.
        Target diameter is randomly sampled btw 0.001 and 0.03.
        Target (angular) speed is randomly sampled. Maximum speed depends on target radius.
        Target orbit axis is assigned a uniform random value in elevation angle (0~360)
        where it is perpendicular to vector player->target.
        Initial hand position is (0, 0).
        Initial hand velocity is (0, 0).
        Initial point of gaze is randomized by Gaussian distribution.
        Initial head position is randomized by Gaussian distribution.
    
    Episode Termination:
        - The agent shoots
        - Target elevation exceeds 83 degree
        - Episode length exceeds 12 (1.2 seconds)
        - Target position out of monitor bound
    """
    def __init__(
        self,
        env_setting: dict=USER_CONFIG_1,
        spec_name: str='ans-v1'
    ):
        super().__init__(
            # seed=seed,
            user_params=env_setting["params_mean"],
            spec_name=spec_name
        )
        self.env_name = "sparse-time-reward"

        self.observation_space = spaces.Box(
            -np.ones(env_setting["obs_min"].size),
            np.ones(env_setting["obs_max"].size),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            -np.ones(env_setting["act_min"].size),
            np.ones(env_setting["act_max"].size),
            dtype=np.float32
        )
        self.obs_range = np.vstack((
            env_setting["obs_min"], env_setting["obs_max"]
        )).T
        self.act_range = np.vstack((
            env_setting["act_min"], env_setting["act_max"]
        )).T

        self.cstate = dict(
            tpos = None,
            gpos = None,
            tvel = None,        # Orbit
            hvel = None,
            trad = None,
            head_pos = None
        )
    

    def np_cstate(self):
        obs = np.array([
            *self.cstate["tpos"],
            *self.cstate["tvel"],
            self.cstate["trad"],
            *self.cstate["hvel"],
            *self.cstate["gpos"],
            *self.cstate["head_pos"]
        ])
        return linear_normalize(obs, *self.obs_range.T)
    

    def reset(self, seed=None):
        super().reset(seed)

        # Initial perception
        tp = BUMP_INTERVAL / 1000

        tpos_0_hat = perception.position_perception(
            self.game.tmpos,
            self.game.gpos,
            self.user_params["theta_p"],
            head_pos=self.game.head_pos
        )
        tvel_true = self.game.tvel_by_orbit()
        speed_hat_ratio = perception.speed_perception(
            tvel_true,
            self.game.tmpos,
            self.user_params["theta_s"],
            head_pos=self.game.head_pos
        )
        tgspd_hat = self.game.tgspd * speed_hat_ratio
        tvel_hat = tvel_true * speed_hat_ratio
        tpos_1_hat = self.game.tmpos_if_hand_and_orbit(
            tpos_0_hat,
            dp=np.zeros(2),
            a=tgspd_hat * tp
        )

        self.cstate.update(
            dict(
                tpos = tpos_1_hat,
                gpos = self.game.gpos,
                tvel = tvel_hat,
                hvel = self.game.hvel,
                trad = self.game.trad,
                head_pos = self.game.head_pos
            )
        )

        info = dict(
            time = self.time,
            result = self.result,
            is_success = bool(self.result),
            forced_termination = self.forced_termination,
            overtime = False,
            shoot_error = None,
        )

        return self.np_cstate(), info


    def step(self, action):
        # Unpack action
        th, kc, tc, kg = self._unpack_action(action)
        tp = BUMP_INTERVAL/1000
        dist_to_target = -1

        ###################### PERCEPTION ######################
        tpos_0_hat, tpos_1_hat, tpos_2_hat, tvel_1_hat, cpos_0_hat = self._perception_and_prediction(th=th)
        ###################### AIM ######################
        ideal_plan_p, ideal_plan_v, noisy_plan_p, noisy_plan_v = self._plan_mouse_movement(kc, th, tpos_2_hat, tvel_1_hat, cpos_0_hat)
        ###################### GAZE ######################
        gaze_time, gaze_traj, gaze_dest_ideal, gaze_dest_noisy = self._plan_gaze_movement(kg, cpos_0_hat, tpos_1_hat)
        ###################### SHOOT ######################
        self._sample_click_timing(tc, th)
        
        # Shoot timing reached
        done = False
        target_hit = False
        if self.shoot_timing <= tp:
            # Interpolate ongoing motor plan for further calculations
            interp_plan_p, _ = aim.interpolate_plan(
                self.ongoing_mp_actual["p"], self.ongoing_mp_actual["v"],
                MUSCLE_INTERVAL, INTERP_INTERVAL
            )
            # shoot_index = int_floor(self.shoot_timing / (INTERP_INTERVAL/1000))
            shoot_index = int_floor(self.shoot_timing * 1000)       # INTERP INTERVAL is 1ms.
            rounded_shoot_moment = shoot_index / 1000
            mouse_displacement = interp_plan_p[shoot_index] - interp_plan_p[0]

            # Check hit
            self.game.orbit(rounded_shoot_moment)
            self.game.move_hand(mouse_displacement)
            target_hit, dist_to_target = self.game.crosshair_on_target()
            self.game.move_hand(-mouse_displacement)
            self.game.orbit(-rounded_shoot_moment)

            done = True


        # Update states to next BUMP
        self.shoot_timing -= tp
        self.game.orbit(tp)
        self.game.move_hand(
            self.ongoing_mp_actual["p"][-1] - self.ongoing_mp_actual["p"][0], 
            v=self.ongoing_mp_actual["v"][-1]
        )
        self.game.fixate(gaze_dest_noisy)

        self.h_traj_p = np.append(self.h_traj_p, self.ongoing_mp_actual["p"][1:], axis=0)
        self.h_traj_v = np.append(self.h_traj_v, self.ongoing_mp_actual["v"][1:], axis=0)
        self.g_traj_p = np.append(self.g_traj_p, gaze_traj[1:], axis=0)
        self.g_traj_t = np.append(self.g_traj_t, gaze_time[1:])


        # Observation Update
        self.cstate.update(
            dict(
                tpos = tpos_1_hat,
                gpos = gaze_dest_ideal,
                tvel = tvel_1_hat,
                hvel = self.ongoing_mp_ideal["v"][-1]
            )
        )
        # Update plan
        self.ongoing_mp_ideal.update(dict(p = ideal_plan_p, v = ideal_plan_v))
        self.ongoing_mp_actual.update(dict(p = noisy_plan_p, v = noisy_plan_v))

        # Compute reward
        if done:
            self.time += rounded_shoot_moment
            self.result = int(target_hit)

            self.error_rate.append(int(target_hit))
            self.time_mean.append(self.time)
        else:
            self.time += tp
            self.result = 0
        
        rew = self._reward_function(
            done, 
            target_hit, 
            self.time,
            rounded_shoot_moment if done else tp
        )

        # Forced termination
        overtime = False
        if ((
            abs(cart2sphr(*self.game.tgpos)[1]) > MAXIMUM_TARGET_ELEVATION or \
            (self.game.target_out_of_monitor() and self.time > 0)
        ) and not done) or (self.time >= self.user_params["max_episode_length"]):
            done = True
            self.forced_termination = True
            self.result = 0
            self.error_rate.append(0)
            rew = self.user_params["penalty_large"]
            if self.time >= self.user_params["max_episode_length"]:
                overtime = True
        

        info = dict(
            time = self.time,
            result = self.result,
            is_success = bool(self.result),
            forced_termination = self.forced_termination,
            overtime = overtime,
            shoot_error = None,
        )
        if done:
            info.update(dict(shoot_error=dist_to_target))
            
        return self.np_cstate(), rew, done, self.forced_termination, info
    
    

    def _unpack_action(self, action):
        [th, kc, tc, kg] = linear_denormalize(action, *self.act_range.T)

        # Processing on actions
        th = 0.05 * int(th / 0.05)      # Interval : 50 ms
        kg = np.sqrt(kg)

        return th, kc, tc, kg

    
    def _perception_and_prediction(self, th, tp=BUMP_INTERVAL/1000):
        # Target state perception on SA (t=t0)
        # Position perception on target and crosshair
        tpos_0_hat = perception.position_perception(
            self.game.tmpos,
            self.game.gpos,
            self.user_params["theta_p"],
            head_pos=self.game.head_pos
        )
        cpos_0_hat = perception.position_perception(
            CROSSHAIR,
            self.game.gpos,
            self.user_params["theta_p"],
            head_pos=self.game.head_pos
        )

        # Target state estimation on RP
        # If target position exceed the range of monitor,
        # the error caused by assumption (all movements are linear and constant)
        # rapidly increases. Therefore, convert monitor speed to orbit angular speed
        tvel_true = self.game.tvel_if_hand_and_orbit(tpos_0_hat, self.ongoing_mp_actual["v"][0])
        tvel_hat = tvel_true * perception.speed_perception(
            tvel_true,
            tpos_0_hat,
            self.user_params["theta_s"],
            head_pos=self.game.head_pos
        )
        # The agent predictis the target movement emerged by aiming implicitly
        # assuming the ideal execution of motor plan
        tvel_aim_hat = self.game.tvel_if_aim(self.ongoing_mp_ideal["v"][0])
        tgvel_hat = tvel_hat - tvel_aim_hat     # Target speed by orbit on monitor. Speed and direction are perceived
        # tgspd_hat = self.game.tgspd_if_tvel(tpos_0_hat, tgvel_hat)
        # toax_hat = self.game.toax_if_tmpos(tpos_0_hat, tgvel_hat)   # Percieved orbit direction

        # Estimated target position on t0 + tp and t0 + tp + th
        # Ideal motor execution + estimated target orbit
        # tpos_1_hat = self.game.tmpos_if_hand_and_orbit(
        #     tmpos=tpos_0_hat,
        #     dp=self.ongoing_mp_ideal["p"][-1] - self.ongoing_mp_ideal["p"][0],
        #     a=tgspd_hat * tp,
        #     toax=toax_hat
        #     # toax=self.game.toax
        # )
        # tpos_2_hat = self.game.tmpos_if_hand_and_orbit(
        #     tmpos=tpos_0_hat,
        #     dp=self.ongoing_mp_ideal["p"][-1] - self.ongoing_mp_ideal["p"][0],
        #     a=tgspd_hat * (tp + th),
        #     toax=toax_hat
        #     # toax=self.game.toax
        # )

        delta_tmpos_by_aim = self.game.delta_tmpos_if_hand_move(tpos_0_hat, self.ongoing_mp_ideal["p"][-1] - self.ongoing_mp_ideal["p"][0])
        tpos_1_hat = tpos_0_hat + delta_tmpos_by_aim + tp * tgvel_hat
        tpos_2_hat = tpos_0_hat + delta_tmpos_by_aim + (tp + th) * tgvel_hat

        return tpos_0_hat, tpos_1_hat, tpos_2_hat, tgvel_hat, cpos_0_hat

    
    def _plan_mouse_movement(self, kc, th, tpos_2_hat, tvel_1_hat, cpos_0_hat):
        # Branch - Check whether the mouse can move
        if self.bump_plan_wait == 0:
            # Plan hand movement for RE (RP of next BUMP)
            # 1. Aims to predicted target position
            # 2. Offset target velocity to zero
            # 3. Assume the ideal execution of ongoing motor plan
            # 4. If shoot is decided, generate full motor plan

            # ideal_plan_p, ideal_plan_v -> next ideal motor plan
            # noisy_plan_p, noisy_plan_v -> next noisy motor plan in queue

            # 1. Shoot motor plan not in queue -> Intermittent control
            if self.shoot_mp_actual["p"] is None:
                # Not decided to shoot: keep updating the motor plan
                if kc < THRESHOLD_SHOOT:
                    ideal_plan_p, ideal_plan_v = aim.plan_hand_movement(
                        self.ongoing_mp_actual["p"][-1],
                        self.ongoing_mp_actual["v"][-1],  # Start at the end of last "actual" motor execution
                        self.game.ppos,
                        self.game.cam_if_hand_move(self.ongoing_mp_ideal["p"][-1] - self.ongoing_mp_ideal["p"][0]),
                        tpos_2_hat,
                        tvel_1_hat,
                        cpos_0_hat,
                        self.game.sensi,
                        plan_duration=int(th * 1000),
                        execute_duration=BUMP_INTERVAL,
                        interval=MUSCLE_INTERVAL
                    )

                    noisy_plan_p, noisy_plan_v = aim.add_motor_noise(
                        ideal_plan_p[0],
                        ideal_plan_v,
                        self.user_params["theta_m"],
                        interval=MUSCLE_INTERVAL
                    )
                # Decided to shoot: fix the rest of motor plan with th
                else:
                    ideal_plan_p, ideal_plan_v = aim.plan_hand_movement(
                        self.ongoing_mp_actual["p"][-1],
                        self.ongoing_mp_actual["v"][-1],  # Start at the end of last "actual" motor execution
                        self.game.ppos,
                        self.game.cam_if_hand_move(self.ongoing_mp_ideal["p"][-1] - self.ongoing_mp_ideal["p"][0]),
                        tpos_2_hat,
                        tvel_1_hat,
                        cpos_0_hat,
                        self.game.sensi,
                        plan_duration=int(th * 1000),
                        execute_duration=int(th * 1000),
                        interval=MUSCLE_INTERVAL
                    )
                    # Sufficiently expand motor plan
                    expand_length = int((self.user_params["max_episode_length"] - max(self.time, 0)) / (MUSCLE_INTERVAL/1000)) + 1
                    ideal_plan_v = np.pad(ideal_plan_v, ((0, expand_length), (0, 0)), mode='constant', constant_values=((0, 0), (0, 0)))
                    
                    noisy_plan_p, noisy_plan_v = aim.add_motor_noise(
                        ideal_plan_p[0],
                        ideal_plan_v,
                        self.user_params["theta_m"],
                        interval=MUSCLE_INTERVAL
                    )
                    ideal_plan_p = np.copy(noisy_plan_p)

                    # Store motor plans on SHOOT MOVING
                    self.shoot_mp_ideal["p"] = ideal_plan_p[2:]
                    self.shoot_mp_ideal["v"] = ideal_plan_v[2:]
                    self.shoot_mp_actual["p"] = noisy_plan_p[2:]
                    self.shoot_mp_actual["v"] = noisy_plan_v[2:]
                    self.shoot_mp_generated = True

                    # Maintain "next motor plan length" as tp
                    ideal_plan_p = ideal_plan_p[:3]
                    ideal_plan_v = ideal_plan_v[:3]
                    noisy_plan_p = noisy_plan_p[:3]
                    noisy_plan_v = noisy_plan_v[:3]

            # 2. Shoot motor plan in queue
            # Pop from queued plan
            else:
                noisy_plan_p = self.shoot_mp_actual["p"][:3]
                noisy_plan_v = self.shoot_mp_actual["v"][:3]

                ideal_plan_p = self.shoot_mp_ideal["p"][:3]
                ideal_plan_p += (noisy_plan_p[0] - ideal_plan_p[0])
                ideal_plan_v = self.shoot_mp_ideal["v"][:3]

                self.shoot_mp_ideal["p"] = self.shoot_mp_ideal["p"][2:]
                self.shoot_mp_ideal["v"] = self.shoot_mp_ideal["v"][2:]
                self.shoot_mp_actual["p"] = self.shoot_mp_actual["p"][2:]
                self.shoot_mp_actual["v"] = self.shoot_mp_actual["v"][2:]

        # Still waiting for reaction time to end     
        else:
            self.bump_plan_wait -= 1
            (
                ideal_plan_p,
                ideal_plan_v,
                noisy_plan_p,
                noisy_plan_v
            ) = (
                np.zeros((BUMP_INTERVAL//MUSCLE_INTERVAL+1, 2)),
                np.zeros((BUMP_INTERVAL//MUSCLE_INTERVAL+1, 2)),
                np.zeros((BUMP_INTERVAL//MUSCLE_INTERVAL+1, 2)),
                np.zeros((BUMP_INTERVAL//MUSCLE_INTERVAL+1, 2))
            )

        return ideal_plan_p, ideal_plan_v, noisy_plan_p, noisy_plan_v

    
    def _plan_gaze_movement(self, kg, cpos_0_hat, tpos_1_hat, tp=BUMP_INTERVAL/1000):
        # Waiting for reaction time to end
        if self.gaze_cooldown > tp:
            self.gaze_cooldown -= tp
            gaze_time = np.array([self.time, self.time + tp])
            gaze_traj = np.array([self.game.gpos, self.game.gpos])
            gaze_dest_ideal = self.game.gpos
            gaze_dest_noisy = self.game.gpos

        # Saccade
        else:
            gaze_dest_ideal = cpos_0_hat * (1-kg) + tpos_1_hat * kg
            gaze_dest_noisy = gaze.gaze_landing_point(self.game.gpos, gaze_dest_ideal, head_pos=self.game.head_pos)
            gaze_time, gaze_traj = gaze.gaze_plan(
                self.game.gpos,
                gaze_dest_noisy,
                theta_q=self.user_params["theta_q"],
                delay=self.gaze_cooldown,
                exe_until=tp,
                head_pos=self.game.head_pos
            )
            self.gaze_cooldown = 0
            gaze_time += self.time  # Translate timestamp to current time
            gaze_dest_noisy = gaze_traj[-1]
        
        return gaze_time, gaze_traj, gaze_dest_ideal, gaze_dest_noisy

    
    def _sample_click_timing(self, tc, th, tp=BUMP_INTERVAL/1000):
        # Click decision made!
        # This branch will be executed only once
        if self.shoot_mp_generated:
            self.shoot_mp_generated = False 
            self.shoot_timing = np.clip(
                np.random.normal(
                    tp + th * tc,
                    (tp + th) * self.user_params["theta_c"]
                ),
                0.002,
                np.inf
            )

    def _reward_function(self, done, hit, time, delta_time):
        if done:
            if hit:
                return self.user_params["hit"] * ((1-self.user_params["hit_decay"]) ** time)
            else:
                return -self.user_params["miss"] * ((1-self.user_params["miss_decay"]) ** time)
        else: return 0



class VariableEnvDefault(EnvDefault):
    def __init__(
        self,
        # seed=None,
        env_setting: dict=USER_CONFIG_1,
        fixed_z: dict=None,
        fixed_w: dict=None,
        spec_name: str='ans-mod-v1'
    ):
        super().__init__(
            # seed=seed,
            env_setting=env_setting,
            spec_name=spec_name
        )
        self.env_name = "sparse-time-reward-modulation"

        assert set(env_setting["params_modulate"]) == set(env_setting["params_max"].keys())
        assert set(env_setting["params_modulate"]) == set(env_setting["params_min"].keys())

        self.variables = env_setting["params_modulate"]
        self.variable_range = np.array([
            [env_setting["params_min"][v] for v in self.variables],
            [env_setting["params_max"][v] for v in self.variables],
        ]).T
        self.variable_scale = env_setting["param_log_scale"]
        self.z_size = len(self.variables)

        # Extended observation space for parameter scale vector
        self.observation_space = spaces.Box(
            -np.ones(self.z_size + env_setting["obs_min"].size),
            np.ones(self.z_size + env_setting["obs_max"].size),
            dtype=np.float32
        )

        # Parameter modulation
        if fixed_z is not None:
            assert set(self.variables) == set(fixed_z.keys())
            self._sample_z = False
            self.z = np.array([np.clip(fixed_z[v], -1, 1) for v in self.variables])
            # self.w = np.array([log_denormalize(_z, *self.variable_range[i], scale=self.variable_scale) for i, _z in enumerate(self.z)])
        elif fixed_w is not None:
            assert set(self.variables) == set(fixed_w.keys())
            self._sample_z = False
            # self.w = np.array([np.clip(fixed_w[v], *self.variable_range[i]) for i, v in enumerate(self.variables)])
            self.z = np.array([log_normalize(_w, *self.variable_range[i], scale=self.variable_scale) for i, _w in enumerate(self.w)])
        else:
            self._sample_z = True
            self.z = self._z_sampler()
            # self.w = np.array([log_denormalize(_z, *self.variable_range[i], scale=self.variable_scale) for i, _z in enumerate(self.z)])

        self.update_params()

    

    def update_params(self):
        for i, v in enumerate(self.variables):
            self.user_params[v] = log_denormalize(self.z[i], *self.variable_range[i], scale=self.variable_scale)

    
    def step(self, action):
        s, r, d, t, info = super().step(action)
        return np.append(self.z, s, axis=0), r, d, t, info
    

    def reset(self, seed=None):
        if self._sample_z:
            self.z = self._z_sampler()
        self.update_params()
        state, info = super().reset(seed)
        return np.append(self.z, state, axis=0), info
    

    def set_param(self, raw_z:np.ndarray=None, z:dict=None, w:dict=None):
        self.fix_z()
        if raw_z is not None:
            self.z = raw_z
        elif z is not None:
            self.z = np.array([np.clip(z[v], -1, 1) for v in self.variables])
            # self.w = np.array([log_denormalize(z[v], *self.variable_range[i], scale=self.variable_scale) for i, v in enumerate(self.variables)])
        elif w is not None:
            # self.w = np.array([np.clip(w[v], *self.variable_range[i]) for i, v in enumerate(self.variables)])
            self.z = np.array([log_normalize(w[v], *self.variable_range[i], scale=self.variable_scale) for i, v in enumerate(self.variables)])
        else:
            raise ValueError("Either z or w must be a dictionary variable.")
        self.update_params()

    
    def fix_z(self):
        self._sample_z = False


    def unfix_z(self):
        self._sample_z = True

    def _z_sampler(self, boundary_p=0.0):
        return np.clip(np.random.uniform(-(1+boundary_p), 1, size=self.z_size), -1, 1)


class EnvBaseline(BaseEnv):
    """
    Observation:
        Type: Box
        Num Observation                     Min         Max         Unit
        0   Target Position X               -W/2        W/2         (m)
        1   Target Position Y               -H/2        H/2         (m)
        2   Target Velocity X (Orbit)       -LIMIT      LIMIT       (m/s)
        3   Target Velocity Y (Orbit)       -LIMIT      LIMIT       (m/s)
        4   Target Radius                   R_MIN       R_MAX       (m)
        5   Hand Velocity X                 -LIMIT      LIMIT       (m/s)
        6   Hand Velocity Y                 -LIMIT      LIMIT       (m/s)

    Actions:
        Type: Box
        0   th (Prediction Horizon)          0.1        2.0         (s)
        1   kc (Shoot Attention)              0          1          None
        2   tc (Shoot timing)                 0          1          None
    """
    def __init__(
        self,
        env_setting: dict=USER_CONFIG_1_BASE,
        spec_name: str='ans-v2'
    ):
        super().__init__(
            # seed=seed,
            user_params=env_setting["params_mean"],
            spec_name=spec_name
        )

        self.observation_space = spaces.Box(
            -np.ones(env_setting["obs_min"].size),
            np.ones(env_setting["obs_max"].size),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            -np.ones(env_setting["act_min"].size),
            np.ones(env_setting["act_max"].size),
            dtype=np.float32
        )
        self.obs_range = np.vstack((
            env_setting["obs_min"], env_setting["obs_max"]
        )).T
        self.act_range = np.vstack((
            env_setting["act_min"], env_setting["act_max"]
        )).T

        self.cstate = dict(
            tpos = None,
            # gpos = None,
            tvel = None,        # Orbit
            hvel = None,
            trad = None,
            # head_pos = None
        )
        self.game_setting["head_pos"] = HEAD_POSITION
    

    def np_cstate(self):
        obs = np.array([
            *self.cstate["tpos"],
            *self.cstate["tvel"],
            self.cstate["trad"],
            *self.cstate["hvel"],
            # *self.cstate["gpos"],
            # *self.cstate["head_pos"]
        ])
        return linear_normalize(obs, *self.obs_range.T)
    

    def reset(self, seed=None):
        super().reset(seed)

        # Initial perception
        tp = BUMP_INTERVAL / 1000

        tpos_0_hat = np.copy(self.game.tmpos)
        tvel_true = self.game.tvel_by_orbit()
        speed_hat_ratio = perception.speed_perception(
            tvel_true,
            self.game.tmpos,
            self.user_params["theta_s"],
            # head_pos=self.game.head_pos
        )
        tgspd_hat = self.game.tgspd * speed_hat_ratio
        tvel_hat = tvel_true * speed_hat_ratio
        tpos_1_hat = self.game.tmpos_if_hand_and_orbit(
            tpos_0_hat,
            dp=np.zeros(2),
            a=tgspd_hat * tp
        )

        self.cstate.update(
            dict(
                tpos = tpos_1_hat,
                # gpos = self.game.gpos,
                tvel = tvel_hat,
                hvel = self.game.hvel,
                trad = self.game.trad,
                # head_pos = self.game.head_pos
            )
        )

        info = dict(
            time = self.time,
            result = self.result,
            is_success = bool(self.result),
            forced_termination = self.forced_termination,
            overtime = False,
            shoot_error = None,
        )

        return self.np_cstate(), info


    def step(self, action):
        # Unpack action
        th, kc, tc = self._unpack_action(action)
        tp = BUMP_INTERVAL/1000
        dist_to_target = -1

        ###################### PERCEPTION ######################
        tpos_0_hat, tpos_1_hat, tpos_2_hat, tvel_1_hat, cpos_0_hat = self._perception_and_prediction(th=th)
        ###################### AIM ######################
        ideal_plan_p, ideal_plan_v, noisy_plan_p, noisy_plan_v = self._plan_mouse_movement(kc, th, tpos_2_hat, tvel_1_hat, cpos_0_hat)
        ###################### GAZE ######################
        gaze_time, gaze_traj, gaze_dest_ideal, gaze_dest_noisy = self._plan_gaze_movement(0, CROSSHAIR, CROSSHAIR)
        ###################### SHOOT ######################
        self._sample_click_timing(tc, th)
        
        # Shoot timing reached
        done = False
        target_hit = False
        if self.shoot_timing <= tp:
            # Interpolate ongoing motor plan for further calculations
            interp_plan_p, _ = aim.interpolate_plan(
                self.ongoing_mp_actual["p"], self.ongoing_mp_actual["v"],
                MUSCLE_INTERVAL, INTERP_INTERVAL
            )
            # shoot_index = int_floor(self.shoot_timing / (INTERP_INTERVAL/1000))
            shoot_index = int_floor(self.shoot_timing * 1000)       # INTERP INTERVAL is 1ms.
            rounded_shoot_moment = shoot_index / 1000
            mouse_displacement = interp_plan_p[shoot_index] - interp_plan_p[0]

            # Check hit
            self.game.orbit(rounded_shoot_moment)
            self.game.move_hand(mouse_displacement)
            target_hit, dist_to_target = self.game.crosshair_on_target()
            self.game.move_hand(-mouse_displacement)
            self.game.orbit(-rounded_shoot_moment)

            done = True


        # Update states to next BUMP
        self.shoot_timing -= tp
        self.game.orbit(tp)
        self.game.move_hand(
            self.ongoing_mp_actual["p"][-1] - self.ongoing_mp_actual["p"][0], 
            v=self.ongoing_mp_actual["v"][-1]
        )
        self.game.fixate(gaze_dest_noisy)

        self.h_traj_p = np.append(self.h_traj_p, self.ongoing_mp_actual["p"][1:], axis=0)
        self.h_traj_v = np.append(self.h_traj_v, self.ongoing_mp_actual["v"][1:], axis=0)
        self.g_traj_p = np.append(self.g_traj_p, gaze_traj[1:], axis=0)
        self.g_traj_t = np.append(self.g_traj_t, gaze_time[1:])


        # Observation Update
        self.cstate.update(
            dict(
                tpos = tpos_1_hat,
                # gpos = gaze_dest_ideal,
                tvel = tvel_1_hat,
                hvel = self.ongoing_mp_ideal["v"][-1]
            )
        )
        # Update plan
        self.ongoing_mp_ideal.update(dict(p = ideal_plan_p, v = ideal_plan_v))
        self.ongoing_mp_actual.update(dict(p = noisy_plan_p, v = noisy_plan_v))

        # Compute reward
        if done:
            self.time += rounded_shoot_moment
            self.result = int(target_hit)

            self.error_rate.append(int(target_hit))
            self.time_mean.append(self.time)
        else:
            self.time += tp
            self.result = 0
        
        rew = self._reward_function(
            done, 
            target_hit, 
            self.time,
            rounded_shoot_moment if done else tp
        )

        # Forced termination
        overtime = False
        if ((
            abs(cart2sphr(*self.game.tgpos)[1]) > MAXIMUM_TARGET_ELEVATION or \
            (self.game.target_out_of_monitor() and self.time > 0)
        ) and not done) or (self.time >= self.user_params["max_episode_length"]):
            done = True
            self.forced_termination = True
            self.result = 0
            self.error_rate.append(0)
            rew = self.user_params["penalty_large"]
            if self.time >= self.user_params["max_episode_length"]:
                overtime = True
        

        info = dict(
            time = self.time,
            result = self.result,
            is_success = bool(self.result),
            forced_termination = self.forced_termination,
            overtime = overtime,
            shoot_error = None,
        )
        if done:
            info.update(dict(shoot_error=dist_to_target))
            
        return self.np_cstate(), rew, done, self.forced_termination, info
    
    

    def _unpack_action(self, action):
        [th, kc, tc] = linear_denormalize(action, *self.act_range.T)

        th = 0.05 * int(th / 0.05)      # Interval : 50 ms
        # kg = np.sqrt(kg)

        return th, kc, tc

    
    def _perception_and_prediction(self, th, tp=BUMP_INTERVAL/1000):
        tpos_0_hat = np.copy(self.game.tmpos)
        cpos_0_hat = np.copy(CROSSHAIR)

        tvel_true = self.game.tvel_if_hand_and_orbit(tpos_0_hat, self.ongoing_mp_actual["v"][0])
        tvel_hat = tvel_true * perception.speed_perception(
            tvel_true,
            tpos_0_hat,
            self.user_params["theta_s"],
            head_pos=self.game.head_pos
        )
        tvel_aim_hat = self.game.tvel_if_aim(self.ongoing_mp_ideal["v"][0])
        tgvel_hat = tvel_hat - tvel_aim_hat
        # tgspd_hat = self.game.tgspd_if_tvel(tpos_0_hat, tgvel_hat)
        # toax_hat = self.game.toax_if_tmpos(tpos_0_hat, tgvel_hat)

        # tpos_1_hat = self.game.tmpos_if_hand_and_orbit(
        #     tmpos=tpos_0_hat,
        #     dp=self.ongoing_mp_ideal["p"][-1] - self.ongoing_mp_ideal["p"][0],
        #     a=tgspd_hat * tp,
        #     toax=toax_hat
        # )
        # tpos_2_hat = self.game.tmpos_if_hand_and_orbit(
        #     tmpos=tpos_0_hat,
        #     dp=self.ongoing_mp_ideal["p"][-1] - self.ongoing_mp_ideal["p"][0],
        #     a=tgspd_hat * (tp + th),
        #     toax=toax_hat
        # )

        delta_tmpos_by_aim = self.game.delta_tmpos_if_hand_move(tpos_0_hat, self.ongoing_mp_ideal["p"][-1] - self.ongoing_mp_ideal["p"][0])
        tpos_1_hat = tpos_0_hat + delta_tmpos_by_aim + tp * tgvel_hat
        tpos_2_hat = tpos_0_hat + delta_tmpos_by_aim + (tp + th) * tgvel_hat

        return tpos_0_hat, tpos_1_hat, tpos_2_hat, tgvel_hat, cpos_0_hat

    
    def _plan_mouse_movement(self, kc, th, tpos_2_hat, tvel_1_hat, cpos_0_hat):
        # Branch - Check whether the mouse can move
        if self.bump_plan_wait == 0:
            if self.shoot_mp_actual["p"] is None:
                # Not decided to shoot: keep updating the motor plan
                if kc < THRESHOLD_SHOOT:
                    ideal_plan_p, ideal_plan_v = aim.plan_hand_movement(
                        self.ongoing_mp_actual["p"][-1],
                        self.ongoing_mp_actual["v"][-1],  # Start at the end of last "actual" motor execution
                        self.game.ppos,
                        self.game.cam_if_hand_move(self.ongoing_mp_ideal["p"][-1] - self.ongoing_mp_ideal["p"][0]),
                        tpos_2_hat,
                        tvel_1_hat,
                        cpos_0_hat,
                        self.game.sensi,
                        plan_duration=int(th * 1000),
                        execute_duration=BUMP_INTERVAL,
                        interval=MUSCLE_INTERVAL
                    )

                    noisy_plan_p, noisy_plan_v = aim.add_motor_noise(
                        ideal_plan_p[0],
                        ideal_plan_v,
                        self.user_params["theta_m"],
                        interval=MUSCLE_INTERVAL
                    )
                # Decided to shoot: fix the rest of motor plan with th
                else:
                    ideal_plan_p, ideal_plan_v = aim.plan_hand_movement(
                        self.ongoing_mp_actual["p"][-1],
                        self.ongoing_mp_actual["v"][-1],  # Start at the end of last "actual" motor execution
                        self.game.ppos,
                        self.game.cam_if_hand_move(self.ongoing_mp_ideal["p"][-1] - self.ongoing_mp_ideal["p"][0]),
                        tpos_2_hat,
                        tvel_1_hat,
                        cpos_0_hat,
                        self.game.sensi,
                        plan_duration=int(th * 1000),
                        execute_duration=int(th * 1000),
                        interval=MUSCLE_INTERVAL
                    )
                    # Sufficiently expand motor plan
                    expand_length = int((self.user_params["max_episode_length"] - max(self.time, 0)) / (MUSCLE_INTERVAL/1000)) + 1
                    ideal_plan_v = np.pad(ideal_plan_v, ((0, expand_length), (0, 0)), mode='constant', constant_values=((0, 0), (0, 0)))
                    
                    noisy_plan_p, noisy_plan_v = aim.add_motor_noise(
                        ideal_plan_p[0],
                        ideal_plan_v,
                        self.user_params["theta_m"],
                        interval=MUSCLE_INTERVAL
                    )
                    ideal_plan_p = np.copy(noisy_plan_p)

                    # Store motor plans on SHOOT MOVING
                    self.shoot_mp_ideal["p"] = ideal_plan_p[2:]
                    self.shoot_mp_ideal["v"] = ideal_plan_v[2:]
                    self.shoot_mp_actual["p"] = noisy_plan_p[2:]
                    self.shoot_mp_actual["v"] = noisy_plan_v[2:]
                    self.shoot_mp_generated = True

                    # Maintain "next motor plan length" as tp
                    ideal_plan_p = ideal_plan_p[:3]
                    ideal_plan_v = ideal_plan_v[:3]
                    noisy_plan_p = noisy_plan_p[:3]
                    noisy_plan_v = noisy_plan_v[:3]

            # 2. Shoot motor plan in queue
            # Pop from queued plan
            else:
                noisy_plan_p = self.shoot_mp_actual["p"][:3]
                noisy_plan_v = self.shoot_mp_actual["v"][:3]

                ideal_plan_p = self.shoot_mp_ideal["p"][:3]
                ideal_plan_p += (noisy_plan_p[0] - ideal_plan_p[0])
                ideal_plan_v = self.shoot_mp_ideal["v"][:3]

                self.shoot_mp_ideal["p"] = self.shoot_mp_ideal["p"][2:]
                self.shoot_mp_ideal["v"] = self.shoot_mp_ideal["v"][2:]
                self.shoot_mp_actual["p"] = self.shoot_mp_actual["p"][2:]
                self.shoot_mp_actual["v"] = self.shoot_mp_actual["v"][2:]

        # Still waiting for reaction time to end     
        else:
            self.bump_plan_wait -= 1
            (
                ideal_plan_p,
                ideal_plan_v,
                noisy_plan_p,
                noisy_plan_v
            ) = (
                np.zeros((BUMP_INTERVAL//MUSCLE_INTERVAL+1, 2)),
                np.zeros((BUMP_INTERVAL//MUSCLE_INTERVAL+1, 2)),
                np.zeros((BUMP_INTERVAL//MUSCLE_INTERVAL+1, 2)),
                np.zeros((BUMP_INTERVAL//MUSCLE_INTERVAL+1, 2))
            )

        return ideal_plan_p, ideal_plan_v, noisy_plan_p, noisy_plan_v

    
    def _plan_gaze_movement(self, kg, cpos_0_hat, tpos_1_hat, tp=BUMP_INTERVAL/1000):
        # Waiting for reaction time to end
        if self.gaze_cooldown > tp:
            self.gaze_cooldown -= tp
            gaze_time = np.array([self.time, self.time + tp])
            gaze_traj = np.array([self.game.gpos, self.game.gpos])
            gaze_dest_ideal = self.game.gpos
            gaze_dest_noisy = self.game.gpos

        # Saccade
        else:
            gaze_dest_ideal = cpos_0_hat * (1-kg) + tpos_1_hat * kg
            gaze_dest_noisy = gaze.gaze_landing_point(self.game.gpos, gaze_dest_ideal, head_pos=self.game.head_pos)
            gaze_time, gaze_traj = gaze.gaze_plan(
                self.game.gpos,
                gaze_dest_noisy,
                theta_q=self.user_params["theta_q"],
                delay=self.gaze_cooldown,
                exe_until=tp,
                head_pos=self.game.head_pos
            )
            self.gaze_cooldown = 0
            gaze_time += self.time  # Translate timestamp to current time
            gaze_dest_noisy = gaze_traj[-1]
        
        return gaze_time, gaze_traj, gaze_dest_ideal, gaze_dest_noisy

    
    def _sample_click_timing(self, tc, th, tp=BUMP_INTERVAL/1000):
        # Click decision made!
        # This branch will be executed only once
        if self.shoot_mp_generated:
            self.shoot_mp_generated = False 
            self.shoot_timing = np.clip(
                np.random.normal(
                    tp + th * tc,
                    (tp + th) * self.user_params["theta_c"]
                ),
                0.002,
                np.inf
            )

    def _reward_function(self, done, hit, time, delta_time):
        if done:
            if hit:
                return self.user_params["hit"] * ((1-self.user_params["hit_decay"]) ** time)
            else:
                return -self.user_params["miss"] * ((1-self.user_params["miss_decay"]) ** time)
        else: return 0


class VariableEnvBaseline(EnvBaseline):
    def __init__(
        self,
        # seed=None,
        env_setting: dict=USER_CONFIG_1_BASE,
        fixed_z: dict=None,
        fixed_w: dict=None,
        spec_name: str='ans-mod-v2'
    ):
        super().__init__(
            # seed=seed,
            env_setting=env_setting,
            spec_name=spec_name
        )
        self.env_name = "sparse-time-reward-modulation"

        assert set(env_setting["params_modulate"]) == set(env_setting["params_max"].keys())
        assert set(env_setting["params_modulate"]) == set(env_setting["params_min"].keys())

        self.variables = env_setting["params_modulate"]
        self.variable_range = np.array([
            [env_setting["params_min"][v] for v in self.variables],
            [env_setting["params_max"][v] for v in self.variables],
        ]).T
        self.variable_scale = env_setting["param_log_scale"]
        self.z_size = len(self.variables)

        # Extended observation space for parameter scale vector
        self.observation_space = spaces.Box(
            -np.ones(self.z_size + env_setting["obs_min"].size),
            np.ones(self.z_size + env_setting["obs_max"].size),
            dtype=np.float32
        )

        # Parameter modulation
        if fixed_z is not None:
            assert set(self.variables) == set(fixed_z.keys())
            self._sample_z = False
            self.z = np.array([np.clip(fixed_z[v], -1, 1) for v in self.variables])
            # self.w = np.array([log_denormalize(_z, *self.variable_range[i], scale=self.variable_scale) for i, _z in enumerate(self.z)])
        elif fixed_w is not None:
            assert set(self.variables) == set(fixed_w.keys())
            self._sample_z = False
            # self.w = np.array([np.clip(fixed_w[v], *self.variable_range[i]) for i, v in enumerate(self.variables)])
            self.z = np.array([log_normalize(_w, *self.variable_range[i], scale=self.variable_scale) for i, _w in enumerate(self.w)])
        else:
            self._sample_z = True
            self.z = self._z_sampler()
            # self.w = np.array([log_denormalize(_z, *self.variable_range[i], scale=self.variable_scale) for i, _z in enumerate(self.z)])

        self.update_params()

    

    def update_params(self):
        for i, v in enumerate(self.variables):
            self.user_params[v] = log_denormalize(self.z[i], *self.variable_range[i], scale=self.variable_scale)

    
    def step(self, action):
        s, r, d, t, info = super().step(action)
        return np.append(self.z, s, axis=0), r, d, t, info
    

    def reset(self, seed=None):
        if self._sample_z:
            self.z = self._z_sampler()
        self.update_params()
        state, info = super().reset(seed)
        return np.append(self.z, state, axis=0), info
    

    def set_param(self, raw_z:np.ndarray=None, z:dict=None, w:dict=None):
        self.fix_z()
        if raw_z is not None:
            self.z = raw_z
        elif z is not None:
            self.z = np.array([np.clip(z[v], -1, 1) for v in self.variables])
            # self.w = np.array([log_denormalize(z[v], *self.variable_range[i], scale=self.variable_scale) for i, v in enumerate(self.variables)])
        elif w is not None:
            # self.w = np.array([np.clip(w[v], *self.variable_range[i]) for i, v in enumerate(self.variables)])
            self.z = np.array([log_normalize(w[v], *self.variable_range[i], scale=self.variable_scale) for i, v in enumerate(self.variables)])
        else:
            raise ValueError("Either z or w must be a dictionary variable.")
        self.update_params()

    
    def fix_z(self):
        self._sample_z = False


    def unfix_z(self):
        self._sample_z = True

    def _z_sampler(self, boundary_p=0.0):
        return np.clip(np.random.uniform(-(1+boundary_p), 1, size=self.z_size), -1, 1)
    


class VariableEnvExponentialTime(VariableEnvDefault):
    def __init__(
        self,
        env_setting: dict=USER_CONFIG_2,
        fixed_z: dict=None,
        fixed_w: dict=None,
        spec_name: str='ans-mod-v2'
    ):
        super().__init__(
            env_setting=env_setting,
            fixed_z=fixed_z,
            fixed_w=fixed_w,
            spec_name=spec_name
        )

    def _reward_function(self, done, hit, time, delta_time):
        if done:
            if hit:
                return self.user_params["hit"] * (self.user_params["time_h"] ** time)
            else:
                return -self.user_params["miss"] * (self.user_params["time_m"] ** time)
        else: return 0



class VariableEnvExponentialTimeBase(VariableEnvBaseline):
    def __init__(
        self,
        env_setting: dict=USER_CONFIG_2_BASE,
        fixed_z: dict=None,
        fixed_w: dict=None,
        spec_name: str='ans-mod-v2'
    ):
        super().__init__(
            env_setting=env_setting,
            fixed_z=fixed_z,
            fixed_w=fixed_w,
            spec_name=spec_name
        )

    def _reward_function(self, done, hit, time, delta_time):
        if done:
            if hit:
                return self.user_params["hit"] * (self.user_params["time_h"] ** time)
            else:
                return -self.user_params["miss"] * (self.user_params["time_m"] ** time)
        else: return 0


# class VariableEnvTaskDependentReward(VariableEnvDefault):
#     def __init__(
#         self,
#         env_setting: dict=USER_CONFIG_2,
#         fixed_z: dict=None,
#         fixed_w: dict=None,
#         spec_name: str='ans-mod-v2'
#     ):
#         super().__init__(
#             env_setting=env_setting,
#             fixed_z=fixed_z,
#             fixed_w=fixed_w,
#             spec_name=spec_name
#         )

#     def _reward_function(self, done, hit, time, delta_time):
#         if done:
#             if hit:
#                 return self._hit_reward_max() * ((1-self.user_params["hit_decay"]) ** time)
#             else:
#                 return -self.user_params["miss"] * ((1-self.user_params["miss_decay"]) ** time)
#         else: return 0
    
#     def _difficulty_index(self):
#         # These values are determined by sampling result in fps_task
#         dist = np.linalg.norm(self.initial_game_cond["tmpos"])
#         id_norm = np.clip(np.log2(dist / (2 * self.initial_game_cond["trad"]) + 1) / 6, 0, 1)
#         spd = self.initial_game_cond["tgspd"] / TARGET_ASPEED_MAX + 1
#         return np.clip(id_norm * spd / 1.4, 0, 1)

#     def _hit_reward_max(self):
#         diff = self._difficulty_index()
#         return self.user_params["hit_max"] * \
#             np.exp(-self.user_params["hit_std"] * (diff - self.user_params["hit_peak"]) ** 2)



# class VariableEnvTaskDependentRewardBase(VariableEnvBaseline):
#     def __init__(
#         self,
#         env_setting: dict=USER_CONFIG_2_BASE,
#         fixed_z: dict=None,
#         fixed_w: dict=None,
#         spec_name: str='ans-mod-v2'
#     ):
#         super().__init__(
#             env_setting=env_setting,
#             fixed_z=fixed_z,
#             fixed_w=fixed_w,
#             spec_name=spec_name
#         )

#     def _reward_function(self, done, hit, time, delta_time):
#         if done:
#             if hit:
#                 return self._hit_reward_max() * ((1-self.user_params["hit_decay"]) ** time)
#             else:
#                 return -self.user_params["miss"] * ((1-self.user_params["miss_decay"]) ** time)
#         else: return 0
    
#     def _difficulty_index(self):
#         # These values are determined by sampling result in fps_task
#         dist = np.linalg.norm(self.initial_game_cond["tmpos"])
#         id_norm = np.clip(np.log2(dist / (2 * self.initial_game_cond["trad"]) + 1) / 6, 0, 1)
#         spd = self.initial_game_cond["tgspd"] / TARGET_ASPEED_MAX + 1
#         return np.clip(id_norm * spd / 1.4, 0, 1)

#     def _hit_reward_max(self):
#         diff = self._difficulty_index()
#         return self.user_params["hit_max"] * \
#             np.exp(-self.user_params["hit_std"] * (diff - self.user_params["hit_peak"]) ** 2)




if __name__ == "__main__":

    x = VariableEnvDefault()
    x.set_param(raw_z=-np.ones(7))
    print(x.user_params)
    # x.reset()

    # import matplotlib.pyplot as plt

    # x = np.random.normal(0.1, 0.1*0.5, size=1000)
    # plt.hist(x, bins=20)
    # plt.show()

    pass





# GRAVEYARD



# class EnvWithMotorEffort(EnvDefault):
#     def __init__(
#         self,
#         seed=None,
#         env_setting: dict=USER_CONFIG_1_2,
#         spec_name: str='ans-v2'
#     ):
#         super().__init__(
#             seed=seed,
#             env_setting=env_setting,
#             spec_name=spec_name
#         )
#         self.meffort = 0
    

#     def reset(self):
#         npc = super().reset()
#         self.meffort = 0
#         return npc
    

#     def step(self, action):
#         # Unpack action
#         th, kc, tc, kg = self._unpack_action(action)
#         tp = BUMP_INTERVAL/1000
#         dist_to_target = -1

#         ###################### PERCEPTION ######################
#         tpos_0_hat, tpos_1_hat, tpos_2_hat, tvel_1_hat, cpos_0_hat = self._perception_and_prediction(th=th)
#         ###################### AIM ######################
#         ideal_plan_p, ideal_plan_v, noisy_plan_p, noisy_plan_v = self._plan_mouse_movement(kc, th, tpos_2_hat, tvel_1_hat, cpos_0_hat)
#         ###################### GAZE ######################
#         gaze_time, gaze_traj, gaze_dest_ideal, gaze_dest_noisy = self._plan_gaze_movement(kg, cpos_0_hat, tpos_1_hat)
#         ###################### SHOOT ######################
#         self._sample_click_timing(tc, th)
        
#         # Shoot timing reached
#         done = False
#         target_hit = False
#         if self.shoot_timing <= tp:
#             # Interpolate ongoing motor plan for further calculations
#             interp_plan_p, _ = aim.interpolate_plan(
#                 self.ongoing_mp_actual["p"], self.ongoing_mp_actual["v"],
#                 MUSCLE_INTERVAL, INTERP_INTERVAL
#             )
#             # shoot_index = int_floor(self.shoot_timing / (INTERP_INTERVAL/1000))
#             shoot_index = int_floor(self.shoot_timing * 1000)       # INTERP INTERVAL is 1ms.
#             rounded_shoot_moment = shoot_index / 1000
#             mouse_displacement = interp_plan_p[shoot_index] - interp_plan_p[0]

#             # Check hit
#             self.game.orbit(rounded_shoot_moment)
#             self.game.move_hand(mouse_displacement)
#             target_hit, dist_to_target = self.game.crosshair_on_target()
#             self.game.move_hand(-mouse_displacement)
#             self.game.orbit(-rounded_shoot_moment)

#             done = True


#         # Update states to next BUMP
#         self.shoot_timing -= tp
#         self.game.orbit(tp)
#         self.game.move_hand(
#             self.ongoing_mp_actual["p"][-1] - self.ongoing_mp_actual["p"][0], 
#             v=self.ongoing_mp_actual["v"][-1]
#         )
#         self.game.fixate(gaze_dest_noisy)

#         self.h_traj_p = np.append(self.h_traj_p, self.ongoing_mp_actual["p"][1:], axis=0)
#         self.h_traj_v = np.append(self.h_traj_v, self.ongoing_mp_actual["v"][1:], axis=0)
#         self.g_traj_p = np.append(self.g_traj_p, gaze_traj[1:], axis=0)
#         self.g_traj_t = np.append(self.g_traj_t, gaze_time[1:])

#         # Motor effort
#         _, interp_plan_v = aim.interpolate_plan(
#             self.ongoing_mp_actual["p"],
#             self.ongoing_mp_actual["v"],
#             MUSCLE_INTERVAL,
#             INTERP_INTERVAL
#         )
#         meffort = -aim.accel_sum(interp_plan_v)


#         # Observation Update
#         self.cstate.update(
#             dict(
#                 tpos = tpos_1_hat,
#                 gpos = gaze_dest_ideal,
#                 tvel = tvel_1_hat,
#                 hvel = self.ongoing_mp_ideal["v"][-1]
#             )
#         )
#         # Update plan
#         self.ongoing_mp_ideal.update(dict(p = ideal_plan_p, v = ideal_plan_v))
#         self.ongoing_mp_actual.update(dict(p = noisy_plan_p, v = noisy_plan_v))

#         # Compute reward
#         if done:
#             self.time += rounded_shoot_moment
#             self.result = int(target_hit)

#             self.error_rate.append(int(target_hit))
#             self.miss_decayean.append(self.time)
#         else:
#             self.time += tp
#             self.result = 0
#         self.meffort += meffort
        
#         rew = meffort * self.user_params["motor"] + self._reward_function(
#             done, 
#             target_hit, 
#             self.time,
#             rounded_shoot_moment if done else tp
#         )

#         # Forced termination
#         if ((
#             abs(cart2sphr(*self.game.tgpos)[1]) > MAXIMUM_TARGET_ELEVATION or \
#             (self.game.target_out_of_monitor() and self.time > 0)
#         ) and not done) or (self.time >= self.user_params["max_episode_length"]):
#             done = True
#             self.forced_termination = True
#             self.result = 0
#             self.error_rate.append(0)
#             rew = PENALTY_LARGE
        

#         info = dict(
#             time = self.time,
#             result = self.result,
#             meffort = self.meffort,
#             is_success = bool(self.result),
#             forced_termination = self.forced_termination
#         )
#         if done:
#             info.update(dict(shoot_error=dist_to_target))
            
#         return self.np_cstate(), rew, done, info


#     def _sample_click_timing(self, tc, th, tp=BUMP_INTERVAL/1000):
#         # Click decision made!
#         # This branch will be executed only once
#         if self.shoot_mp_generated:
#             self.shoot_mp_generated = False 
#             self.shoot_timing = np.clip(
#                 np.random.normal(
#                     tp + th * tc,
#                     (tp + th) * self.user_params["theta_c"]
#                 ),
#                 0.05,
#                 np.inf
#             )


# class VariableEnvWithMotorEffort(EnvWithMotorEffort):
#     def __init__(
#         self,
#         seed=None,
#         env_setting: dict=USER_CONFIG_1_2,
#         fixed_z: dict=None,
#         fixed_w: dict=None,
#         spec_name: str='ans-mod-v2'
#     ):
#         super().__init__(
#             seed=seed,
#             env_setting=env_setting,
#             spec_name=spec_name
#         )
#         self.env_name = "sparse-time-reward-modulation"

#         assert set(env_setting["params_modulate"]) == set(env_setting["params_max"].keys())
#         assert set(env_setting["params_modulate"]) == set(env_setting["params_min"].keys())

#         self.variables = env_setting["params_modulate"]
#         self.variable_range = np.array([
#             [env_setting["params_min"][v] for v in self.variables],
#             [env_setting["params_max"][v] for v in self.variables],
#         ]).T
#         self.variable_scale = env_setting["param_log_scale"]
#         self.z_size = len(self.variables)

#         # Extended observation space for parameter scale vector
#         self.observation_space = spaces.Box(
#             -np.ones(self.z_size + env_setting["obs_min"].size),
#             np.ones(self.z_size + env_setting["obs_max"].size)
#         )

#         # Parameter modulation
#         if fixed_z is not None:
#             assert set(self.variables) == set(fixed_z.keys())
#             self._sample_z = False
#             self.z = np.array([np.clip(fixed_z[v], -1, 1) for v in self.variables])
#             # self.w = np.array([log_denormalize(_z, *self.variable_range[i], scale=self.variable_scale) for i, _z in enumerate(self.z)])
#         elif fixed_w is not None:
#             assert set(self.variables) == set(fixed_w.keys())
#             self._sample_z = False
#             # self.w = np.array([np.clip(fixed_w[v], *self.variable_range[i]) for i, v in enumerate(self.variables)])
#             self.z = np.array([log_normalize(_w, *self.variable_range[i], scale=self.variable_scale) for i, _w in enumerate(self.w)])
#         else:
#             self._sample_z = True
#             self.z = np.random.uniform(-1, 1, size=self.z_size)
#             # self.w = np.array([log_denormalize(_z, *self.variable_range[i], scale=self.variable_scale) for i, _z in enumerate(self.z)])

#         self.update_params()

    

#     def update_params(self):
#         for i, v in enumerate(self.variables):
#             self.user_params[v] = log_denormalize(self.z[i], *self.variable_range[i], scale=self.variable_scale)

    
#     def step(self, action):
#         s, r, d, info = super().step(action)
#         return np.append(self.z, s, axis=0), r, d, info
    

#     def reset(self):
#         if self._sample_z:
#             self.z = 2 * np.random.random_sample(self.z_size) - 1
#         self.update_params()
#         state = super().reset()
#         return np.append(self.z, state, axis=0)
    

#     def set_param(self, raw_z:np.ndarray=None, z:dict=None, w:dict=None):
#         self.fix_z()
#         if raw_z is not None:
#             self.z = raw_z
#         elif z is not None:
#             self.z = np.array([np.clip(z[v], -1, 1) for v in self.variables])
#             # self.w = np.array([log_denormalize(z[v], *self.variable_range[i], scale=self.variable_scale) for i, v in enumerate(self.variables)])
#         elif w is not None:
#             # self.w = np.array([np.clip(w[v], *self.variable_range[i]) for i, v in enumerate(self.variables)])
#             self.z = np.array([log_normalize(w[v], *self.variable_range[i], scale=self.variable_scale) for i, v in enumerate(self.variables)])
#         else:
#             raise ValueError("Either z or w must be a dictionary variable.")
#         self.update_params()

    
#     def fix_z(self):
#         self._sample_z = False


#     def unfix_z(self):
#         self._sample_z = True
        



# ### Agent: Fixed miss penalty, modulated hit and time reward
# class EnvHitTime(EnvDefault):
#     def __init__(
#         self,
#         seed=None,
#         env_setting: dict=USER_CONFIG_2,
#         spec_name: str='ans-v2'
#     ):
#         super().__init__(
#             seed=seed,
#             env_setting=env_setting,
#             spec_name=spec_name
#         )
    
#     def _reward_function(self, done, hit, time, delta_time):
#         if done:
#             if hit:
#                 return self.user_params["hit"] / (1 + (time - 0.1) * (1/(1-self.user_params["time"]) - 1))
#             else:
#                 return self.user_params["miss"]
#         else: return 0


# class VariableEnvHitTime(VariableEnvDefault):
#     def __init__(
#         self,
#         seed=None,
#         env_setting: dict=USER_CONFIG_2,
#         fixed_z: dict=None,
#         fixed_w: dict=None,
#         spec_name: str='ans-mod-v2'
#     ):
#         super().__init__(
#             seed=seed,
#             env_setting=env_setting,
#             fixed_z=fixed_z,
#             fixed_w=fixed_w,
#             spec_name=spec_name
#         )

#     def _reward_function(self, done, hit, time, delta_time):
#         if done:
#             if hit:
#                 return self.user_params["hit"] / (1 + (time - 0.1) * (1/(1-self.user_params["time"]) - 1))
#             else:
#                 return self.user_params["miss"]
#         else: return 0
        


# ### Agent: tc is always 1 (end of prediction horizon)
# class EnvAutoClick(EnvDefault):
#     def __init__(
#         self,
#         seed=None,
#         env_setting: dict=USER_CONFIG_3,
#         spec_name: str='ans-v2'
#     ):
#         super().__init__(
#             seed=seed,
#             env_setting=env_setting,
#             spec_name=spec_name
#         )


#     def _unpack_action(self, action):
#         [th, kc, kg] = linear_denormalize(action, *self.act_range.T)

#         # Processing on actions
#         th = 0.05 * int(th / 0.05)      # Interval : 50 ms
#         kg = np.sqrt(kg)

#         return th, kc, 1, kg    # Fix tc to 1




# ### Agent: Same as Default agent, but reward is not sparse anymore.
# class EnvLinearTimePenalty(EnvDefault):
#     def __init__(
#         self,
#         seed=None,
#         env_setting: dict=USER_CONFIG_4,
#         spec_name: str='ans-v3'
#     ):
#         super().__init__(
#             seed=seed,
#             env_setting=env_setting,
#             spec_name=spec_name
#         )
    
#     def _reward_function(self, done, hit, time, delta_time):
#         if done:
#             if hit:
#                 return self.user_params["hit"] - self.user_params["time"] * delta_time
#             else:
#                 return self.user_params["miss"] - self.user_params["time"] * delta_time
#         else: return -self.user_params["time"] * delta_time
    
   

# class VariableEnvLinearTimePenalty(VariableEnvDefault):
#     def __init__(
#         self,
#         seed=None,
#         env_setting: dict=USER_CONFIG_4,
#         fixed_z: dict=None,
#         fixed_w: dict=None,
#         spec_name: str='ans-mod-v3'
#     ):
#         super().__init__(
#             seed=seed,
#             env_setting=env_setting,
#             fixed_z=fixed_z,
#             fixed_w=fixed_w,
#             spec_name=spec_name
#         )

#     def _reward_function(self, done, hit, time, delta_time):
#         if done:
#             if hit:
#                 return self.user_params["hit"] - self.user_params["time"] * delta_time
#             else:
#                 return self.user_params["miss"] - self.user_params["time"] * delta_time
#         else: return -self.user_params["time"] * delta_time