'''
Aim-and-Shoot AGENT

Code written by June-Seop Yoon
with help of Seungwon Do, Hee-Seung Moon
'''

from copy import deepcopy
from collections import deque

import sys, os, pickle

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import EnvSpec, spec
gym.logger.set_level(40)

sys.path.append("..")

from agent import _perception as perception
from agent import _aim as aim
from agent import _shoot as shoot
from agent import _gaze as gaze
from agent.fps_task import GameState

from configs.simulation import *
from configs.common import *
from configs.path import PATH_MAT
from utilities.mymath import (
    ct2sp, 
    ecc_dist, 
    int_floor,
    float_round, 
    logscale_mean_std,
    np_interp_2d, 
    v_normalize, 
    v_denormalize, 
    convert_w2z, 
    convert_z2w
)
from utilities.utils import pickle_load


class Env(gym.Env):
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
        9   Eye position X                  -0.065      0.065       (m)
        10  Eye position Y                  -0.0421     0.1979      (m)
        11  Eye position Z                  0.408       0.742       (m)

    Actions:
        Type: Box
        0   th (Prediction Horizon)          0.1        2.5         (s)
        1   kc (Shoot Attention)              0          1          None
        2   kg (Gaze damping)                 0          1          None

    Rewards:
        - Shot hit: 1 if the agent shot the target successfully, otherwise 0
        - Shot miss: 1 if the agent miss the shot, otherwise 0
        - Elapsed time: time interval between the decision-makings
        - Motor effort: sum of the absolute acceleration of the simulated hand
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
        Initial eye position is randomized by Gaussian distribution.
    
    Episode Termination:
        - The agent shoots
        - Target elevation exceeds 83 degree
        - Episode length exceeds 30 (3 seconds)
        - Target position out of monitor bound
    """

    def __init__(
        self,
        seed=None,
        variables=COG_SPACE,
        variable_mean=USER_PARAM_MEAN,
        variable_std=USER_PARAM_STD,
        variable_max=USER_PARAM_MAX,
        variable_min=USER_PARAM_MIN,
        z_scale_range=MAX_SIGMA_SCALE,
        param_scale_z: dict=None,
        param_scale_w: dict=None,
        game_setting: dict=None,
        spec_name: str='ans-v0'
    ):
        self.seed(seed=seed)
        self.viewer = None
        self.spec = EnvSpec(spec_name)

        self._options = dict(
            param_scale_z = param_scale_z,
            param_scale_w = param_scale_w
        )

        # Observation and Action space
        self.observation_space = spaces.Box(
            -np.ones(OBS_MIN.size),
            np.ones(OBS_MAX.size),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            -np.ones(ACT_MIN.size),
            np.ones(ACT_MAX.size),
            dtype=np.float32
        )

        # Simulated player constraint
        self.variables = variables
        assert variable_mean is not None
        self.var_mean = variable_mean
        self.zscale = z_scale_range
        if variable_std is not None:
            self.var_std = variable_std
        else:
            assert variable_max is not None and variable_min is not None
            self.var_std = dict()
            for v in self.variables:
                m, s = logscale_mean_std(
                    variable_max[v],
                    variable_min[v],
                    r=self.zscale
                )
                self.var_mean[v] = m
                self.var_std[v] = s
        

        self.user_params = deepcopy(variable_mean)
        self.set_user_param(param_scale_z=param_scale_z, param_scale_w=param_scale_w)

        
        # FPS task setting
        if game_setting is not None:
            self.game_setting = game_setting
        else:
            self.game_setting = dict(
                pcam=None,
                tgpos=None,
                tmpos=None,
                toax=None,
                session=None,
                gpos=None,
                hrt=None,
                grt=None,
                eye_pos=None
            )
        
        self._game = GameState(sensi=self.user_params["sensi"])
        self._game.reset(**self.game_setting)


        # Cognitive state - this is what user perceives and remember
        self.cstate = dict(
            tpos = self._game.tmpos,
            gpos = self._game.gpos,
            tovel = self._game.tvel_by_orbit(),
            hvel = self._game.hvel,
            trad = self._game.trad,
        )

        # Motor plans
        self.ongoing_mp_ideal = dict(
            p = np.zeros((int(BUMP_INTERVAL/MUSCLE_INTERVAL)+1, 2)),
            v = np.zeros((int(BUMP_INTERVAL/MUSCLE_INTERVAL)+1, 2)),
        )
        self.ongoing_mp_actual = dict(
            p = np.zeros((int(BUMP_INTERVAL/MUSCLE_INTERVAL)+1, 2)),
            v = np.zeros((int(BUMP_INTERVAL/MUSCLE_INTERVAL)+1, 2)),
        )
        

        # Basic environment setting
        self.time = 0
        self.meffort = 0
        self.result = 0

        self.shoot_decided = False
        self.shoot_timing_sampled = False
        self.shoot_timing = MAXIMUM_EPISODE_LENGTH
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


    def set_user_param(
        self, 
        param_scale_z:dict=None, 
        param_scale_w:dict=None
    ):
        if param_scale_z is not None:
            for v, z in param_scale_z.items():
                self.user_params[v] = convert_z2w(z, self.var_mean[v], self.var_std[v], r=self.zscale)
        elif param_scale_w is not None:
            for v, w in param_scale_w.items():
                self.user_params[v] = w


    def np_cstate(self):
        obs = np.array(
            [
                *self.cstate["tpos"],
                *self.cstate["tovel"],
                self.cstate["trad"],
                *self.cstate["hvel"],
                *self.cstate["gpos"],
                *self.cstate["eye_pos"]
            ]
        )
        return v_normalize(obs, OBS_MIN, OBS_MAX)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def update_game_setting(self, game_setting):
        self.game_setting = game_setting
        return

    
    def episode_trajectory(self):
        return (
            self.h_traj_p,
            self.h_traj_v,
            self.g_traj_p,
            self.g_traj_t,
            self.delayed_time
        )
    

    def step(self, action):
        """
        Serial process inside a single BUMP
        """
        # Unpack action - prediction horizon, shoot attention, gaze damping
        [th, kc, kg] = v_denormalize(action, ACT_MIN, ACT_MAX)

        # BUMP INTERVAL
        tp = BUMP_INTERVAL

        dist_to_target = -1
    
        ###################### AIM ######################
        # Target state perception on SA (t=t0)
        tpos_0_hat = perception.position_perception(
            self._game.tmpos,
            self._game.gpos,
            self.user_params["theta_p"],
            ep=self._game.eye_pos
        )
        cpos_0_hat = perception.position_perception(
            CROSSHAIR,
            self._game.gpos,
            self.user_params["theta_p"],
            ep=self._game.eye_pos
        )


        # Target state estimation on RP
        # Assume ideal motor execution
        ideal_plan_exe = self.ongoing_mp_ideal["p"][-1] - self.ongoing_mp_ideal["p"][0]
        tpos_delta_ideal = self._game.tmpos_if_hand_move(ideal_plan_exe) - self._game.tmpos
        
        tgvel_0_hat = perception.speed_perception(
            self._game.tvel_by_orbit(),
            tpos_0_hat,
            self.user_params["theta_s"],
            ep=self._game.eye_pos
        )
        
        # If target position exceed the range of monitor,
        # the error caused by assumption (all movements are linear and constant)
        # rapidly increases. Therefore, convert monitor speed to orbit angular speed
        tgspd_hat = self._game.tgspd_if_tvel(tgvel_0_hat)

        # Estimated target position on t0 + tp
        tpos_1_hat = tpos_0_hat + tpos_delta_ideal + self._game.tmpos_delta_if_orbit(
            tgspd_hat * tp
        )
        # Estimated target position on t0 + tp + th
        tpos_2_hat = tpos_0_hat + tpos_delta_ideal + self._game.tmpos_delta_if_orbit(
            tgspd_hat * (tp + th)
        )

        if self.bump_plan_wait == 0:
            # Plan hand movement for RE (RP of next BUMP)
            # 1. Aims to predicted target position
            # 2. Offset target velocity to zero
            # Assume the ideal execution of ongoing motor plan
            ideal_plan_p, ideal_plan_v = aim.plan_hand_movement(
                self.ongoing_mp_actual["p"][-1],
                self.ongoing_mp_actual["v"][-1],  # Start at the end of last "actual" motor execution
                self._game.ppos,
                self._game.cam_if_hand_move(ideal_plan_exe),
                tpos_2_hat,
                tgvel_0_hat,
                cpos_0_hat,
                self._game.sensi,
                th,
                tp,
                interval=MUSCLE_INTERVAL
            )

            noisy_plan_p, noisy_plan_v = aim.add_motor_noise(
                ideal_plan_p[0],
                ideal_plan_v,
                self.user_params["theta_m"],
                interval=MUSCLE_INTERVAL
            )
        else:
            # Still waiting to be ready... (hand reaction remaining)
            self.bump_plan_wait -= 1
            (
                ideal_plan_p,
                ideal_plan_v,
                noisy_plan_p,
                noisy_plan_v
            ) = (
                np.zeros((int(tp/MUSCLE_INTERVAL)+1, 2)),
                np.zeros((int(tp/MUSCLE_INTERVAL)+1, 2)),
                np.zeros((int(tp/MUSCLE_INTERVAL)+1, 2)),
                np.zeros((int(tp/MUSCLE_INTERVAL)+1, 2))
            )

        # Interpolate ongoing motor plan for further calculations
        interp_plan_p, interp_plan_v = aim.interpolate_plan(
            self.ongoing_mp_actual["p"], self.ongoing_mp_actual["v"],
            MUSCLE_INTERVAL, INTERP_INTERVAL
        )

        ###################### GAZE ######################
        if self.gaze_cooldown > tp:
            self.gaze_cooldown -= tp
            gaze_time = np.array([self.time, self.time + tp])
            gaze_traj = np.array([self._game.gpos, self._game.gpos])
            gaze_dest_ideal = self._game.gpos
            gaze_dest_noisy = self._game.gpos
        else:
            gaze_dest_ideal = cpos_0_hat * (1-kg) + tpos_1_hat * kg
            gaze_dest_noisy = gaze.gaze_landing_point(self._game.gpos, gaze_dest_ideal, ep=self._game.eye_pos)
            gaze_time, gaze_traj = gaze.gaze_plan(
                self._game.gpos,
                gaze_dest_noisy,
                self.user_params["va"],
                self.user_params["theta_q"],
                delay=self.gaze_cooldown,
                exe_until=tp-self.gaze_cooldown,
                ep=self._game.eye_pos
            )
            self.gaze_cooldown = 0
            gaze_time += self.time
            gaze_dest_noisy = gaze_traj[-1]
        


        ###################### SHOOT ######################
        # Decided to shoot from prev. BUMP
        if self.shoot_decided and not self.shoot_timing_sampled and self.time > 0:
            self.shoot_decided = False

            tavel_hat = perception.speed_perception(
                self._game.tvel_if_aim(self.ongoing_mp_actual["p"], MUSCLE_INTERVAL),
                tpos_0_hat,
                self.user_params["theta_s"],
                ep=self._game.eye_pos
            )

            # Sample shoot timing
            self.shoot_timing = shoot.sample_shoot_timing(
                tpos_0_hat,
                cpos_0_hat,
                tgvel_0_hat,
                tavel_hat,
                self._game.trad,
                0.497*self.user_params["theta_s"] + 0.053   # Reference: Speeding Up inference
                # self.user_params["theta_c"]
            )
            self.shoot_timing_sampled = True
        
        # Decided to sample shoot timing on current BUMP
        self.shoot_decided = (kc > THRESHOLD_SHOOT)

        # Shoot timing reached
        done = False
        if self.shoot_timing < tp:
            # State at the moment of shoot
            index_of_shoot = int_floor(self.shoot_timing / INTERP_INTERVAL)
            rounded_shoot_moment = float_round(self.shoot_timing, INTERP_INTERVAL)
            hpos_delta_on_shoot = interp_plan_p[index_of_shoot] - interp_plan_p[0]

            # CHECK HIT
            self._game.orbit(rounded_shoot_moment)
            self._game.move_hand(hpos_delta_on_shoot)
            target_hit, dist_to_target = self._game.crosshair_on_target()
            self._game.move_hand(-hpos_delta_on_shoot)
            self._game.orbit(-rounded_shoot_moment)
            
            # Episode done
            done = True


        # Update states to next BUMP
        self.shoot_timing -= tp
        self._game.orbit(tp)
        self._game.move_hand(
            self.ongoing_mp_actual["p"][-1] - self.ongoing_mp_actual["p"][0], 
            v=self.ongoing_mp_actual["v"][-1]
        )
        self._game.fixate(gaze_dest_noisy)

        self.h_traj_p = np.append(self.h_traj_p, self.ongoing_mp_actual["p"][1:], axis=0)
        self.h_traj_v = np.append(self.h_traj_v, self.ongoing_mp_actual["v"][1:], axis=0)
        self.g_traj_p = np.append(self.g_traj_p, gaze_traj[1:], axis=0)
        self.g_traj_t = np.append(self.g_traj_t, gaze_time[1:])

        self.cstate.update(
            dict(
                tpos = tpos_1_hat,
                gpos = gaze_dest_ideal,
                tovel = tgvel_0_hat,
                hvel = self.ongoing_mp_ideal["v"][-1]
            )
        )

        # Update plan
        self.ongoing_mp_ideal.update(
            dict(
                p = ideal_plan_p,
                v = ideal_plan_v
            )
        )
        self.ongoing_mp_actual.update(
            dict(
                p = noisy_plan_p,
                v = noisy_plan_v
            )
        )

        # Compute reward
        if done:
            me = aim.accel_sum(interp_plan_v[:index_of_shoot+1])

            self.time += rounded_shoot_moment
            self.meffort += me
            self.result = int(target_hit)

            rew_t = self.user_params["time"] * (rounded_shoot_moment - self.delayed_time)
            rew_e = self.user_params["motor"] * me

            if target_hit:
                rew_s = self.user_params["hit"]
            else:
                penalty_coeff = 1 + (100 * (
                    dist_to_target - THRESHOLD_AMPLIFY_PENALTY_DIST
                )) ** 2 if dist_to_target > THRESHOLD_AMPLIFY_PENALTY_DIST else 1
                rew_s = self.user_params["miss"] * penalty_coeff

            self.error_rate.append(int(target_hit))
            self.time_mean.append(self.time)
        else:
            me = aim.accel_sum(interp_plan_v)

            self.time += tp
            self.meffort += me
            self.result = 0

            rew_t = self.user_params["time"] * tp
            rew_e = self.user_params["motor"] * me
            rew_s = 0

        # Forced termination
        if (
            abs(ct2sp(*self._game.tgpos)[1]) > MAXIMUM_TARGET_ELEVATION or \
            self.time > MAXIMUM_EPISODE_LENGTH or \
            (self._game.target_out_of_monitor() and self.time > 0)
        ) and not done:
            done = True
            self.forced_termination = True
            self.result = 0
            self.error_rate.append(0)
            rew_s = self.user_params["miss"] * FORCED_TERMINATION_PENALTY_COEFF
        

        info = dict(
            time = self.time,
            meffort = self.meffort,
            result = self.result,
            is_success = bool(self.result),
            forced_termination = self.forced_termination
        )
        if done:
            info.update(dict(shoot_error=dist_to_target))

        return self.np_cstate(), rew_s + rew_e + rew_t, done, info


    def reset(self):
        # Reset game
        self.initial_game_cond = self._game.reset(**self.game_setting)

        # Reaction sync
        # Set hand reaction time to 400ms (4 BUMP INTERVAL)
        self.delayed_time = 4 * BUMP_INTERVAL - self.initial_game_cond["hrt"]
        self.gaze_cooldown = self.initial_game_cond["grt"] + self.delayed_time
        self.bump_plan_wait = 3

        self.time = -self.delayed_time      # Reversed time
        self._game.orbit(-self.delayed_time)

        # Pass BUMPs where hand & gaze are both stationary
        num_of_stat = int(min(self.gaze_cooldown, 3 * BUMP_INTERVAL) // BUMP_INTERVAL)
        self.gaze_cooldown -= num_of_stat * BUMP_INTERVAL
        self.bump_plan_wait -= num_of_stat
        self.time += num_of_stat * BUMP_INTERVAL      # Elapsed time
        self.meffort = 0                # Hand movement effort
        self.result = 0                 # Shoot result

        self.shoot_decided = False      # True if decided to plan shoot from prev. BUMP
        self.shoot_timing_sampled = False
        self.shoot_timing = 99          # Remaining shooting moment
        self.forced_termination = False # Episode terminated due to cond. violation

        self.h_traj_p = np.zeros((int(num_of_stat*BUMP_INTERVAL/MUSCLE_INTERVAL)+1, 2))
        self.h_traj_v = np.zeros((int(num_of_stat*BUMP_INTERVAL/MUSCLE_INTERVAL)+1, 2))
        self.g_traj_p = np.array([self.initial_game_cond["gpos"], self.initial_game_cond["gpos"]])
        self.g_traj_t = np.array([0, num_of_stat * BUMP_INTERVAL]) - self.delayed_time

        # SA -> perception and prediction
        tpos_0_hat = perception.position_perception(
            self._game.tmpos,
            self._game.gpos,
            self.user_params["theta_p"],
            ep=self._game.eye_pos
        )
        # cpos_0_hat = perception.position_perception(
        #     CROSSHAIR,
        #     self._game.gpos,
        #     self.user_params["theta_p"],
        #     ep=self._game.eye_pos
        # )

        tgvel_0_hat = perception.speed_perception(
            self._game.tvel_by_orbit(),
            tpos_0_hat,
            self.user_params["theta_s"],
            ep=self._game.eye_pos
        )
        tgspd_hat = self._game.tgspd_if_tvel(tgvel_0_hat)
        tpos_1_hat = tpos_0_hat + self._game.tmpos_delta_if_orbit(
            tgspd_hat * BUMP_INTERVAL
        )

        # State and game update to passed BUMP
        self._game.orbit(num_of_stat * BUMP_INTERVAL)
        self.cstate.update(
            dict(
                tpos = tpos_1_hat,
                gpos = self._game.gpos,
                # gspd = 0,
                tovel = tgvel_0_hat,
                hvel = self._game.hvel,
                trad = self._game.trad,
                eye_pos = self._game.eye_pos
            )
        )
        self.ongoing_mp_ideal.update(
            dict(
                p = np.zeros((int(BUMP_INTERVAL/MUSCLE_INTERVAL)+1, 2)),
                v = np.zeros((int(BUMP_INTERVAL/MUSCLE_INTERVAL)+1, 2)),
            )
        )
        self.ongoing_mp_actual.update(
            dict(
                p = np.zeros((int(BUMP_INTERVAL/MUSCLE_INTERVAL)+1, 2)),
                v = np.zeros((int(BUMP_INTERVAL/MUSCLE_INTERVAL)+1, 2)),
            )
        )

        return self.np_cstate()





### Modulated Env
class VariableEnv(Env):
    def __init__(
        self,
        variables=list(),
        variable_mean=USER_PARAM_MEAN,
        variable_std=USER_PARAM_STD,
        variable_max=USER_PARAM_MAX,
        variable_min=USER_PARAM_MIN,
        z_scale_range=MAX_SIGMA_SCALE,
        fixed_z=None,
        fixed_w=None,
        seed=None,
        game_setting=None,
        spec_name='ans-v1'
    ):
        super().__init__(
            seed=seed,
            variables=variables,
            variable_mean=variable_mean,
            variable_std=variable_std,
            variable_max=variable_max,
            variable_min=variable_min,
            z_scale_range=z_scale_range,
            game_setting=game_setting,
            spec_name=spec_name
        )
        # Extended observation space for parameter scale vector
        self.observation_space = spaces.Box(
            -np.ones(len(variables) + len(OBS_MIN)),
            np.ones(len(variables) + len(OBS_MAX)), 
            dtype=np.float32
        )
        self.z_size = len(self.variables)

        if fixed_z is not None:
            assert len(variables) == len(fixed_z)
            self._sample_z = False
            self.z = fixed_z
        elif fixed_w is not None:
            assert len(variables) == len(fixed_w)
            self._sample_z = False
            self.z = np.array(
                [
                    convert_w2z(
                        fixed_w[i],
                        self.var_mean[v],
                        self.var_std[v],
                        r = self.zscale
                    ) for i, v in enumerate(self.variables)
                ]
            )
        else:
            self._sample_z = True
            self.z = 2 * np.random.random_sample(self.z_size) - 1

        self.update_params()
    

    def update_params(self):
        # user parameter: multiplied by 1/scale ~ scale
        for i, v in enumerate(self.variables):
            self.user_params[v] = convert_z2w(
                self.z[i],
                self.var_mean[v],
                self.var_std[v],
                r = self.zscale
            )
    

    def step(self, action):
        s, r, d, info = super().step(action)
        return np.append(self.z, s, axis=0), r, d, info
    

    def reset(self):
        if self._sample_z:
            self.z = 2 * np.random.random_sample(self.z_size) - 1
        self.update_params()
        state = super().reset()
        return np.append(self.z, state, axis=0)
    

    def set_z(self, z):
        self._sample_z = False
        self.z = z


    def unfix_z(self):
        self._sample_z = True



### Evaluation Env
class EvalEnv(Env):
    def __init__(
        self,
        variables=list(),
        variable_mean=USER_PARAM_MEAN,
        variable_std=USER_PARAM_STD,
        variable_max=USER_PARAM_MAX,
        variable_min=USER_PARAM_MIN,
        z_scale_range=MAX_SIGMA_SCALE,
        seed=None,
        game_setting=None,
        eval_preset_name='cog128',
        eval_cond_name='task_cond_256',
        spec_name='ans-v2'
    ):
        super().__init__(
            seed=seed,
            variables=variables,
            variable_mean=variable_mean,
            variable_std=variable_std,
            variable_max=variable_max,
            variable_min=variable_min,
            z_scale_range=z_scale_range,
            game_setting=game_setting,
            spec_name=spec_name
        )
        # Extended observation space for parameter scale vector
        self.observation_space = spaces.Box(
            -np.ones(len(variables) + len(OBS_MIN)),
            np.ones(len(variables) + len(OBS_MAX)), 
            dtype=np.float32
        )

        # Parameter scale preset
        assert os.path.exists(f"{PATH_MAT}{eval_preset_name}.pkl")
        self.z_preset = pickle_load(f"{PATH_MAT}{eval_preset_name}.pkl")

        # Task condition preset
        assert os.path.exists(f"{PATH_MAT}{eval_cond_name}.pkl")
        self.cond_preset = pickle_load(f"{PATH_MAT}{eval_cond_name}.pkl")

        assert self.z_preset.shape[1] == len(self.variables)
        assert len(self.cond_preset) == self.z_preset.shape[0]

        self.z_preset_index = 0
        self.z_preset_num = self.z_preset.shape[0]

        self.update_params()
    

    def update_params(self):
        # user parameter: multiplied by 1/scale ~ scale
        for z, v in zip(self.z_preset[self.z_preset_index], self.variables):
            self.user_params[v] = convert_z2w(
                z,
                self.var_mean[v],
                self.var_std[v],
                r = self.zscale
            )
        self.update_game_setting(self.cond_preset[self.z_preset_index])


    def step(self, action):
        s, r, d, info = super().step(action)
        return np.append(self.z_preset[self.z_preset_index], s, axis=0), r, d, info
    

    def reset(self):
        self.z_preset_index += 1
        if self.z_preset_index == self.z_preset_num:
            self.z_preset_index = 0
        self.update_params()
        state = super().reset()
        return np.append(self.z_preset[self.z_preset_index], state, axis=0)






if __name__ == "__main__":
    a = Env()
    a.reset()
    t, y, u, i = a.step([0.1, 0, 0.2])
    # a.step([0.4, 0, 0.5])
    # a.step([1.0, 0, 1])
    # a.step([0.5, 1, 0])