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
from agent import module_shoot as shoot
from agent.fps_task import GameState

from configs.simulation import *
from configs.common import *
from configs.experiment import *
from utils.mymath import *

from agent.agent_base import BaseEnv


class EnvTimeNoise(BaseEnv):
    def __init__(
        self,
        env_setting: dict=USER_CONFIG_3,
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
        tpos_0_hat, tpos_1_hat, tpos_2_hat, tvel_1_hat, cpos_0_hat, clock_noise = self._perception_and_prediction(th=th)
        ###################### AIM ######################
        ideal_plan_p, ideal_plan_v, noisy_plan_p, noisy_plan_v = self._plan_mouse_movement(kc, th, tpos_2_hat, tvel_1_hat, cpos_0_hat)
        ###################### GAZE ######################
        gaze_time, gaze_traj, gaze_dest_ideal, gaze_dest_noisy = self._plan_gaze_movement(kg, cpos_0_hat, tpos_1_hat)
        ###################### SHOOT ######################
        self._sample_click_timing(th, tc, clock_noise)
        
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
        delta_tmpos_by_aim = self.game.delta_tmpos_if_hand_move(tpos_0_hat, self.ongoing_mp_ideal["p"][-1] - self.ongoing_mp_ideal["p"][0])
        while True:
            clock_noise = np.random.normal(1, self.user_params["theta_c"])
            if 0.01 < clock_noise < 2: break
        tpos_1_hat = tpos_0_hat + delta_tmpos_by_aim + tp * clock_noise * tgvel_hat
        tpos_2_hat = tpos_0_hat + delta_tmpos_by_aim + (tp + th) * clock_noise * tgvel_hat

        return tpos_0_hat, tpos_1_hat, tpos_2_hat, tgvel_hat, cpos_0_hat, clock_noise

    
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
                    ideal_plan_v = np.pad(ideal_plan_v, ((0, expand_length), (0, 0)), mode='edge')
                    ideal_plan_p = np.concatenate((
                        ideal_plan_p, 
                        ideal_plan_p[-1] + (ideal_plan_v[-1][:, np.newaxis] * np.arange(1, expand_length+1)).T * (MUSCLE_INTERVAL/1000)
                    ))
                    
                    noisy_plan_p, noisy_plan_v = aim.add_motor_noise(
                        ideal_plan_p[0],
                        ideal_plan_v,
                        self.user_params["theta_m"],
                        interval=MUSCLE_INTERVAL
                    )

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

    
    def _sample_click_timing(self, th, tc, clock_noise, tp=BUMP_INTERVAL/1000):
        # Click decision made!
        # This branch will be executed only once
        if self.shoot_mp_generated:
            self.shoot_mp_generated = False
            self.shoot_timing = np.clip((tp + tc * th) * clock_noise, 0.001, np.inf)


    def _reward_function(self, done, hit, time, delta_time):
        if done:
            if hit:
                return self.user_params["hit"] * ((1-self.user_params["hit_decay"]) ** time)
            else:
                return -self.user_params["miss"] * ((1-self.user_params["miss_decay"]) ** time)
        else: return 0



class VariableEnvTimeNoise(EnvTimeNoise):
    def __init__(
        self,
        # seed=None,
        env_setting: dict=USER_CONFIG_3,
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
    

class EnvTimeNoiseBase(BaseEnv):
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
        tpos_0_hat, tpos_1_hat, tpos_2_hat, tvel_1_hat, cpos_0_hat, clock_noise = self._perception_and_prediction(th=th)
        ###################### AIM ######################
        ideal_plan_p, ideal_plan_v, noisy_plan_p, noisy_plan_v = self._plan_mouse_movement(kc, th, tpos_2_hat, tvel_1_hat, cpos_0_hat)
        ###################### GAZE ######################
        gaze_time, gaze_traj, gaze_dest_ideal, gaze_dest_noisy = self._plan_gaze_movement(0, CROSSHAIR, CROSSHAIR)
        ###################### SHOOT ######################
        self._sample_click_timing(th, tc, clock_noise)
        
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

        delta_tmpos_by_aim = self.game.delta_tmpos_if_hand_move(tpos_0_hat, self.ongoing_mp_ideal["p"][-1] - self.ongoing_mp_ideal["p"][0])
        while True:
            clock_noise = np.random.normal(1, self.user_params["theta_c"])
            if 0.01 < clock_noise < 2: break
        tpos_1_hat = tpos_0_hat + delta_tmpos_by_aim + tp * clock_noise * tgvel_hat
        tpos_2_hat = tpos_0_hat + delta_tmpos_by_aim + (tp + th) * clock_noise * tgvel_hat

        return tpos_0_hat, tpos_1_hat, tpos_2_hat, tgvel_hat, cpos_0_hat, clock_noise

    
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
                        tpos_2_hat, # + aim.implicit_aim_point(self.game.trad),
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
                        tpos_2_hat, # + aim.implicit_aim_point(self.game.trad),
                        tvel_1_hat,
                        cpos_0_hat,
                        self.game.sensi,
                        plan_duration=int(th * 1000),
                        execute_duration=int(th * 1000),
                        interval=MUSCLE_INTERVAL
                    )
                    # Sufficiently expand motor plan
                    expand_length = int((self.user_params["max_episode_length"] - max(self.time, 0)) / (MUSCLE_INTERVAL/1000)) + 1
                    ideal_plan_v = np.pad(ideal_plan_v, ((0, expand_length), (0, 0)), mode='edge')
                    ideal_plan_p = np.concatenate((
                        ideal_plan_p, 
                        ideal_plan_p[-1] + (ideal_plan_v[-1][:, np.newaxis] * np.arange(1, expand_length+1)).T * (MUSCLE_INTERVAL/1000)
                    ))
                    
                    noisy_plan_p, noisy_plan_v = aim.add_motor_noise(
                        ideal_plan_p[0],
                        ideal_plan_v,
                        self.user_params["theta_m"],
                        interval=MUSCLE_INTERVAL
                    )

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

    
    def _sample_click_timing(self, th, tc, clock_noise, tp=BUMP_INTERVAL/1000):
        if self.shoot_mp_generated:
            self.shoot_mp_generated = False
            self.shoot_timing = np.clip((tp + tc * th) * clock_noise, 0.001, np.inf)


    def _reward_function(self, done, hit, time, delta_time):
        if done:
            if hit:
                return self.user_params["hit"] * ((1-self.user_params["hit_decay"]) ** time)
            else:
                return -self.user_params["miss"] * ((1-self.user_params["miss_decay"]) ** time)
        else: return 0


class VariableEnvTimeNoiseBase(EnvTimeNoiseBase):
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


class EnvTimeNoiseBase2(BaseEnv):
    def __init__(
        self,
        env_setting: dict=USER_CONFIG_3,
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
        tpos_0_hat, tpos_1_hat, tpos_2_hat, tvel_1_hat, cpos_0_hat, clock_noise = self._perception_and_prediction(th=th)
        ###################### AIM ######################
        ideal_plan_p, ideal_plan_v, noisy_plan_p, noisy_plan_v = self._plan_mouse_movement(kc, th, tpos_2_hat, tvel_1_hat, cpos_0_hat)
        ###################### GAZE ######################
        gaze_time, gaze_traj, gaze_dest_ideal, gaze_dest_noisy = self._plan_gaze_movement(kg, cpos_0_hat, tpos_1_hat)
        ###################### SHOOT ######################
        self._sample_click_timing(th, tc, clock_noise)
        
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
        tgvel_true = self.game.tvel_by_orbit(dt=0.001)
        tgvel_hat = tgvel_true * perception.speed_perception(
            tgvel_true,
            tpos_0_hat,
            self.user_params["theta_s"],
            head_pos=self.game.head_pos
        )
        delta_tmpos_by_aim = self.game.delta_tmpos_if_hand_move(tpos_0_hat, self.ongoing_mp_ideal["p"][-1] - self.ongoing_mp_ideal["p"][0])
        while True:
            clock_noise = np.random.normal(1, self.user_params["theta_c"])
            if 0.01 < clock_noise < 2: break
        tpos_1_hat = tpos_0_hat + delta_tmpos_by_aim + tp * clock_noise * tgvel_hat
        tpos_2_hat = tpos_0_hat + delta_tmpos_by_aim + (tp + th) * clock_noise * tgvel_hat

        return tpos_0_hat, tpos_1_hat, tpos_2_hat, tgvel_hat, cpos_0_hat, clock_noise

    
    def _plan_mouse_movement(self, kc, th, tpos_2_hat, tvel_1_hat, cpos_0_hat):
        # Branch - Check whether the mouse can move
        if self.bump_plan_wait == 0:
            if self.shoot_mp_actual["p"] is None:
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
                    ideal_plan_v = np.pad(ideal_plan_v, ((0, expand_length), (0, 0)), mode='edge')
                    ideal_plan_p = np.concatenate((
                        ideal_plan_p, 
                        ideal_plan_p[-1] + (ideal_plan_v[-1][:, np.newaxis] * np.arange(1, expand_length+1)).T * (MUSCLE_INTERVAL/1000)
                    ))
                    
                    noisy_plan_p, noisy_plan_v = aim.add_motor_noise(
                        ideal_plan_p[0],
                        ideal_plan_v,
                        self.user_params["theta_m"],
                        interval=MUSCLE_INTERVAL
                    )

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

    
    def _sample_click_timing(self, th, tc, clock_noise, tp=BUMP_INTERVAL/1000):
        # Click decision made!
        # This branch will be executed only once
        if self.shoot_mp_generated:
            self.shoot_mp_generated = False
            self.shoot_timing = np.clip((tp + tc * th) * clock_noise, 0.001, np.inf)


    def _reward_function(self, done, hit, time, delta_time):
        if done:
            if hit:
                return self.user_params["hit"] * ((1-self.user_params["hit_decay"]) ** time)
            else:
                return -self.user_params["miss"] * ((1-self.user_params["miss_decay"]) ** time)
        else: return 0



class VariableEnvTimeNoiseBase2(EnvTimeNoiseBase2):
    def __init__(
        self,
        # seed=None,
        env_setting: dict=USER_CONFIG_3,
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


class EnvTimeNoiseBase3(BaseEnv):
    def __init__(
        self,
        env_setting: dict=USER_CONFIG_3_BASE,
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
        tpos_0_hat, tpos_1_hat, tpos_2_hat, tvel_1_hat, cpos_0_hat, clock_noise = self._perception_and_prediction(th=th)
        ###################### AIM ######################
        ideal_plan_p, ideal_plan_v, noisy_plan_p, noisy_plan_v = self._plan_mouse_movement(kc, th, tpos_2_hat, tvel_1_hat, cpos_0_hat)
        ###################### GAZE ######################
        gaze_time, gaze_traj, gaze_dest_ideal, gaze_dest_noisy = self._plan_gaze_movement(0, CROSSHAIR, CROSSHAIR)
        ###################### SHOOT ######################
        self._sample_click_timing(th, tc, clock_noise)
        
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

        tgvel_true = self.game.tvel_by_orbit(dt=0.001)
        tgvel_hat = tgvel_true * perception.speed_perception(
            tgvel_true,
            tpos_0_hat,
            self.user_params["theta_s"],
            head_pos=self.game.head_pos
        )

        delta_tmpos_by_aim = self.game.delta_tmpos_if_hand_move(tpos_0_hat, self.ongoing_mp_ideal["p"][-1] - self.ongoing_mp_ideal["p"][0])
        while True:
            clock_noise = np.random.normal(1, self.user_params["theta_c"])
            if 0.01 < clock_noise < 2: break
        tpos_1_hat = tpos_0_hat + delta_tmpos_by_aim + tp * clock_noise * tgvel_hat
        tpos_2_hat = tpos_0_hat + delta_tmpos_by_aim + (tp + th) * clock_noise * tgvel_hat

        return tpos_0_hat, tpos_1_hat, tpos_2_hat, tgvel_hat, cpos_0_hat, clock_noise

    
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
                        tpos_2_hat, # + aim.implicit_aim_point(self.game.trad),
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
                        tpos_2_hat, # + aim.implicit_aim_point(self.game.trad),
                        tvel_1_hat,
                        cpos_0_hat,
                        self.game.sensi,
                        plan_duration=int(th * 1000),
                        execute_duration=int(th * 1000),
                        interval=MUSCLE_INTERVAL
                    )
                    # Sufficiently expand motor plan
                    expand_length = int((self.user_params["max_episode_length"] - max(self.time, 0)) / (MUSCLE_INTERVAL/1000)) + 1
                    ideal_plan_v = np.pad(ideal_plan_v, ((0, expand_length), (0, 0)), mode='edge')
                    ideal_plan_p = np.concatenate((
                        ideal_plan_p, 
                        ideal_plan_p[-1] + (ideal_plan_v[-1][:, np.newaxis] * np.arange(1, expand_length+1)).T * (MUSCLE_INTERVAL/1000)
                    ))
                    
                    noisy_plan_p, noisy_plan_v = aim.add_motor_noise(
                        ideal_plan_p[0],
                        ideal_plan_v,
                        self.user_params["theta_m"],
                        interval=MUSCLE_INTERVAL
                    )

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

    
    def _sample_click_timing(self, th, tc, clock_noise, tp=BUMP_INTERVAL/1000):
        if self.shoot_mp_generated:
            self.shoot_mp_generated = False
            self.shoot_timing = np.clip((tp + tc * th) * clock_noise, 0.001, np.inf)


    def _reward_function(self, done, hit, time, delta_time):
        if done:
            if hit:
                return self.user_params["hit"] * ((1-self.user_params["hit_decay"]) ** time)
            else:
                return -self.user_params["miss"] * ((1-self.user_params["miss_decay"]) ** time)
        else: return 0


class VariableEnvTimeNoiseBase3(EnvTimeNoiseBase3):
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



class EnvTimeNoiseBaseFinal(BaseEnv):
    def __init__(
        self,
        env_setting: dict=USER_CONFIG_BASE_FINAL,
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
        th, kc = self._unpack_action(action)
        tp = BUMP_INTERVAL/1000
        dist_to_target = -1

        ###################### PERCEPTION ######################
        tpos_0_hat, tpos_1_hat, tpos_2_hat, tvel_1_hat, tavel_hat, cpos_0_hat = self._perception_and_prediction(th=th)
        ###################### AIM ######################
        ideal_plan_p, ideal_plan_v, noisy_plan_p, noisy_plan_v = self._plan_mouse_movement(kc, th, tpos_2_hat, tvel_1_hat, cpos_0_hat)
        ###################### GAZE ######################
        gaze_time, gaze_traj, gaze_dest_ideal, gaze_dest_noisy = self._plan_gaze_movement(0, CROSSHAIR, CROSSHAIR)
        ###################### SHOOT ######################
        self._sample_click_timing(tpos_0_hat, cpos_0_hat, tvel_1_hat, tavel_hat, self.game.trad)
        
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
        [th, kc] = linear_denormalize(action, *self.act_range.T)

        th = 0.05 * int(th / 0.05)      # Interval : 50 ms

        return th, kc

    
    def _perception_and_prediction(self, th, tp=BUMP_INTERVAL/1000):
        tpos_0_hat = np.copy(self.game.tmpos)
        cpos_0_hat = np.copy(CROSSHAIR)

        tgvel_true = self.game.tvel_by_orbit(dt=0.001)
        tgvel_hat = tgvel_true * perception.speed_perception(
            tgvel_true,
            tpos_0_hat,
            self.user_params["theta_s"],
            head_pos=self.game.head_pos
        )
        tavel_hat = self.game.tvel_if_aim(self.ongoing_mp_ideal["v"][0])

        delta_tmpos_by_aim = self.game.delta_tmpos_if_hand_move(tpos_0_hat, self.ongoing_mp_ideal["p"][-1] - self.ongoing_mp_ideal["p"][0])
        tpos_1_hat = tpos_0_hat + delta_tmpos_by_aim + tp * tgvel_hat
        tpos_2_hat = tpos_0_hat + delta_tmpos_by_aim + (tp + th) * tgvel_hat

        return tpos_0_hat, tpos_1_hat, tpos_2_hat, tgvel_hat, tavel_hat, cpos_0_hat

    
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
                        tpos_2_hat, # + aim.implicit_aim_point(self.game.trad),
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
                        tpos_2_hat, # + aim.implicit_aim_point(self.game.trad),
                        tvel_1_hat,
                        cpos_0_hat,
                        self.game.sensi,
                        plan_duration=int(th * 1000),
                        execute_duration=int(th * 1000),
                        interval=MUSCLE_INTERVAL
                    )
                    # Sufficiently expand motor plan
                    expand_length = int((self.user_params["max_episode_length"] - max(self.time, 0)) / (MUSCLE_INTERVAL/1000)) + 1
                    ideal_plan_v = np.pad(ideal_plan_v, ((0, expand_length), (0, 0)), mode='edge')
                    ideal_plan_p = np.concatenate((
                        ideal_plan_p, 
                        ideal_plan_p[-1] + (ideal_plan_v[-1][:, np.newaxis] * np.arange(1, expand_length+1)).T * (MUSCLE_INTERVAL/1000)
                    ))
                    
                    noisy_plan_p, noisy_plan_v = aim.add_motor_noise(
                        ideal_plan_p[0],
                        ideal_plan_v,
                        self.user_params["theta_m"],
                        interval=MUSCLE_INTERVAL
                    )

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

    
    def _sample_click_timing(self, tpos, cpos, tgvel, tavel, trad):
        if self.shoot_mp_generated:
            self.shoot_mp_generated = False
            self.shoot_timing = shoot.sample_shoot_timing(tpos, cpos, tgvel, tavel, trad, self.user_params)


    def _reward_function(self, done, hit, time, delta_time):
        if done:
            if hit:
                return self.user_params["hit"] * ((1-self.user_params["hit_decay"]) ** time)
            else:
                return -self.user_params["miss"] * ((1-self.user_params["miss_decay"]) ** time)
        else: return 0


class VariableEnvTimeNoiseBaseFinal(EnvTimeNoiseBaseFinal):
    def __init__(
        self,
        # seed=None,
        env_setting: dict=USER_CONFIG_BASE_FINAL,
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


class EnvFixedTh(BaseEnv):
    def __init__(
        self,
        env_setting: dict=USER_CONFIG_3,
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
        self.prediction_horizon = None

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

        if self.prediction_horizon is None or self.prediction_horizon < tp:
            self.prediction_horizon = th

        ###################### PERCEPTION ######################
        tpos_0_hat, tpos_1_hat, tpos_2_hat, tvel_1_hat, cpos_0_hat, clock_noise = self._perception_and_prediction(th=self.prediction_horizon)
        ###################### AIM ######################
        ideal_plan_p, ideal_plan_v, noisy_plan_p, noisy_plan_v = self._plan_mouse_movement(kc, self.prediction_horizon, tpos_2_hat, tvel_1_hat, cpos_0_hat)
        ###################### GAZE ######################
        gaze_time, gaze_traj, gaze_dest_ideal, gaze_dest_noisy = self._plan_gaze_movement(kg, cpos_0_hat, tpos_1_hat)
        ###################### SHOOT ######################
        self._sample_click_timing(self.prediction_horizon, tc, clock_noise)
        
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
        self.prediction_horizon -= tp
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
        delta_tmpos_by_aim = self.game.delta_tmpos_if_hand_move(tpos_0_hat, self.ongoing_mp_ideal["p"][-1] - self.ongoing_mp_ideal["p"][0])
        while True:
            clock_noise = np.random.normal(1, self.user_params["theta_c"])
            if 0.01 < clock_noise < 2: break
        tpos_1_hat = tpos_0_hat + delta_tmpos_by_aim + tp * clock_noise * tgvel_hat
        tpos_2_hat = tpos_0_hat + delta_tmpos_by_aim + (tp + th) * clock_noise * tgvel_hat

        return tpos_0_hat, tpos_1_hat, tpos_2_hat, tgvel_hat, cpos_0_hat, clock_noise

    
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
                    ideal_plan_v = np.pad(ideal_plan_v, ((0, expand_length), (0, 0)), mode='edge')
                    ideal_plan_p = np.concatenate((
                        ideal_plan_p, 
                        ideal_plan_p[-1] + (ideal_plan_v[-1][:, np.newaxis] * np.arange(1, expand_length+1)).T * (MUSCLE_INTERVAL/1000)
                    ))
                    
                    noisy_plan_p, noisy_plan_v = aim.add_motor_noise(
                        ideal_plan_p[0],
                        ideal_plan_v,
                        self.user_params["theta_m"],
                        interval=MUSCLE_INTERVAL
                    )

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

    
    def _sample_click_timing(self, th, tc, clock_noise, tp=BUMP_INTERVAL/1000):
        # Click decision made!
        # This branch will be executed only once
        if self.shoot_mp_generated:
            self.shoot_mp_generated = False
            self.shoot_timing = np.clip((tp + tc * th) * clock_noise, 0.001, np.inf)


    def _reward_function(self, done, hit, time, delta_time):
        if done:
            if hit:
                return self.user_params["hit"] * ((1-self.user_params["hit_decay"]) ** time)
            else:
                return -self.user_params["miss"] * ((1-self.user_params["miss_decay"]) ** time)
        else: return 0



class VariableEnvFixedTh(EnvFixedTh):
    def __init__(
        self,
        # seed=None,
        env_setting: dict=USER_CONFIG_3,
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