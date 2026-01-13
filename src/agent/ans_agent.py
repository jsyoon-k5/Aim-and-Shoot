'''
Aim-and-Shoot agents

Code written by June-Seop Yoon
with help of Seungwon Do and Hee-Seung Moon

SB3==2.1.0
'''

import numpy as np
import scipy as sp
from box import Box
from collections import deque

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.envs.registration import EnvSpec
# gym.logger.set_level(40)


from ..config.config import SIM, AGN
from ..config.constant import FIELD, METRIC, AXIS, VECTOR
from ..agent.module_aim import Aim
from ..agent.module_gaze import Gaze
from ..agent.module_perceive import Perceive
from ..agent.ans_task import AnSGame
from ..utils.mymath import Convert, cos_sin_array, linear_normalize, linear_denormalize, log_denormalize, log_normalize


class AnSEnvDefault(gym.Env):
    def __init__(
        self,
        spec_name: str="ans-base",
        agent_name: str="default",
        game_env_name: str="default",
        interval_name: str="default",
        agent_cfg=None,
        game_env_cfg=None,
        interval_cfg=None,
    ):
        self.viewer = None
        self.spec = EnvSpec(spec_name)
        
        # Configuration settings
        self.intv = SIM.interval[interval_name] if interval_cfg is None else interval_cfg
        self.user = AGN[agent_name] if agent_cfg is None else agent_cfg

        # User parameter settings
        self.param_mod_z = {p: 0.0 for p in self.user.param_modul.list}
        self.user_param = dict()
        if self.user.param_const is not None:
            for v, w in self.user.param_const.items():
                self.user_param[v] = w
        self.sample_mod_z = True
        self.z_sample_min_prob = list()
        for p in self.user.param_modul.list:
            if "sample_min_prob" in self.user.param_modul[p]:
                self.z_sample_min_prob.append(self.user.param_modul[p].sample_min_prob)
            else:
                self.z_sample_min_prob.append(0)
        self.z_sample_min_prob = np.array(self.z_sample_min_prob)

        ### Observation and Action space
        self.observation_space = spaces.Box(    # Modulated parameter + environment state
            -np.ones(len(self.user.param_modul.list) + np.sum([len(self.user.observation[key].max) for key in self.user.observation.list])),
            np.ones( len(self.user.param_modul.list) + np.sum([len(self.user.observation[key].max) for key in self.user.observation.list])),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            -np.ones(len(self.user.action.list)),
            np.ones(len(self.user.action.list)),
            dtype=np.float32
        )
        self.observation_range = np.vstack((
            np.concatenate([self.user.observation[key].min for key in self.user.observation.list]),
            np.concatenate([self.user.observation[key].max for key in self.user.observation.list])
        )).T
        self.action_range = np.vstack((
            [self.user.action[key].min for key in self.user.action.list],
            [self.user.action[key].max for key in self.user.action.list]
        )).T


        # Point-and-Click environment
        self.game_env_setting = dict(
            cdir = None,    # Azim, Elev
            tmpos = None,   # X, Y
            tgpos = None,   # X, Y, Z
            torbit = None,  # Azim, Elev
            tmdir = None,   # Scalar (degree)
            tspd = None,    # Scalar (degree/s)
            trad = None,    # Scalar (m),
        )
        self.game_env = AnSGame(config_name=game_env_name, config=game_env_cfg)

        # User status that is non-cognitive parameter
        self.user_status_setting = {
            METRIC.SUMMARY.MRT: None,     # Random sample hand reaction time. To fix, specify value
            METRIC.SUMMARY.GRT: None,
            FIELD.PLAYER.HEAD.NAME: None,
            FIELD.PLAYER.GAZE.NAME: None,
        }
        self.user_current_status = Box(dict(
            gaze_reaction_time = self.intv.bump,
            hand_reaction_time = 2 * self.intv.bump,
            gaze_pos = np.zeros(2),
            head_pos = VECTOR.HEAD,
        ))
        

        # Cognitive state
        self.cstate = dict(
            target_pos_monitor = None,
            target_vel_orbit = None,
            target_rad = None,
            hand_pos = None,
            hand_vel = None,
            gaze_pos = np.zeros(2),
            head_pos = np.zeros(3)
        )
        ans_current_state = self.game_env.get_current_state()
        for st in self.cstate:
            if st in ans_current_state:
                self.cstate[st] = ans_current_state[st]

        # Motor plans
        self.ongoing_mp_ideal = dict(
            p = np.zeros((self.intv.bump // self.intv.muscle + 1, 2)),
            v = np.zeros((self.intv.bump // self.intv.muscle + 1, 2)),
        )
        self.ongoing_mp_actual = dict(
            p = np.zeros((self.intv.bump // self.intv.muscle + 1, 2)),
            v = np.zeros((self.intv.bump // self.intv.muscle + 1, 2)),
        )
        self.pending_mp_ideal = dict(p=None, v=None)
        self.pending_mp_actual = dict(p=None, v=None)

        self.shoot_mp_determined = False
        self.shoot_timing_determined = False

        self.shoot_timing = 100_000  # 100 s
        self.truncation = False
        self.delayed_time = 0
        self.gaze_cooldown = 0
        self.bump_plan_wait = 3

        self.timer = 0
        self.result = 0

        self.time_mean = deque(maxlen=1000)
        self.error_rate = deque(maxlen=1000)

        # Behavior history
        self.h_traj_p = list()
        self.h_traj_v = list()
        self.g_traj_p = list()      # [P1, P2, ...]
        self.g_traj_t = list()      # [T1, T2, ...] - interval btw pos
        self.initial_env_cond = dict()

        self.env_setting_info = dict(
            user_config = self.user.to_dict(),
            task_config = self.game_env.config.to_dict(),
            interval = self.intv.to_dict()
        )
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    

    def update_task_setting(self, ans_env_setting=dict()):
        self.game_env_setting.update(ans_env_setting)

    def update_user_status(self, user_stat_setting=dict()):
        self.user_status_setting.update(user_stat_setting)

    def np_cstate(self):
        param_z = np.array([self.param_mod_z[v] for v in self.user.param_modul.list])
        obs = np.concatenate([np.atleast_1d(self.cstate[key]) for key in self.user.observation.list])
        obs = linear_normalize(obs, *self.observation_range.T, clip=False)
        return np.concatenate((param_z, obs))


    def episode_trajectory(self):
        return (
            self.h_traj_p,
            self.h_traj_v,
            self.g_traj_p,
            self.g_traj_t,
            self.delayed_time
        )
    

    def set_user_param(self, param_z=dict(), param_w=dict()):
        if not param_z.keys():
            self.fix_z(param=param_z)
        elif not param_w.keys():
            for v, w in param_w.items():
                if self.user.param_modul[v].type == 'loguniform':
                    param_z[v] = log_normalize(
                        w, 
                        self.user.param_modul[v].min, 
                        self.user.param_modul[v].max, 
                        scale=self.user.param_modul[v].scale
                    )
                elif self.user.param_modul[v].type == 'uniform':
                    param_z[v] = linear_normalize(w, self.user.param_modul[v].min, self.user.param_modul[v].max)
            self.fix_z(param=param_z)
        else:
            raise ValueError("ValueError: either param_z of param_w must be non-empty dictionary.")


    def update_params(self):
        for v in self.user.param_modul.list:
            if self.user.param_modul[v].type == "uniform":
                self.user_param[v] = linear_denormalize(
                    self.param_mod_z[v],
                    self.user.param_modul[v].min,
                    self.user.param_modul[v].max
                )
            elif self.user.param_modul[v].type == "loguniform":
                self.user_param[v] = log_denormalize(
                    self.param_mod_z[v],
                    self.user.param_modul[v].min,
                    self.user.param_modul[v].max,
                    scale=self.user.param_modul[v].scale
                )


    def fix_z(self, param:dict=None):
        self.sample_mod_z = False
        if param is not None:
            for v, z in param.items():
                self.param_mod_z[v] = z
            self.update_params()
    

    def unfix_z(self):
        self.sample_mod_z = True


    def _z_sampler(self, sample_min_prob=0.0):
        # ext = sample_min_prob / (1 - sample_min_prob)
        # return np.clip(np.random.uniform(-(1+ext), 1, size=len(self.user.param_modul.list)), -1, 1)
        ext = self.z_sample_min_prob / (1 - self.z_sample_min_prob)
        return np.clip(np.random.uniform(-ext - 1, np.ones(len(self.user.param_modul.list))), -1, 1)


    def sample_z(self, sample_min_prob=0.0):
        new_z = self._z_sampler(sample_min_prob=sample_min_prob)
        for i, v in enumerate(self.param_mod_z):
            self.param_mod_z[v] = new_z[i]
    

    def np_z(self):
        return np.array([self.param_mod_z[v] for v in self.user.param_modul.list])

    
    def user_stat_sampler(self):
        stat = {
            METRIC.SUMMARY.MRT: None,
            METRIC.SUMMARY.GRT: None,
            FIELD.PLAYER.HEAD.NAME: None,
            FIELD.PLAYER.GAZE.NAME: None
        }
        # Hand reaction
        if self.user_status_setting[METRIC.SUMMARY.MRT] is not None:
            stat[METRIC.SUMMARY.MRT] = np.clip(
                self.user_status_setting[METRIC.SUMMARY.MRT],
                a_min=self.user.status_variable.reaction.hand.min,
                a_max=self.user.status_variable.reaction.hand.max,
                dtype=int
            )
        else:
            while True:
                # if self.user.status_variable.reaction.hand.type == 'skewnorm':
                hrt = round(sp.stats.skewnorm(
                    self.user.status_variable.reaction.hand.alpha,
                    self.user.status_variable.reaction.hand.mean,
                    self.user.status_variable.reaction.hand.std
                ).rvs(1)[0])
                # elif:
                if self.user.status_variable.reaction.hand.min <= hrt <= self.user.status_variable.reaction.hand.max:
                    break
            stat[METRIC.SUMMARY.MRT] = hrt

        # Gaze reaction
        if self.user_status_setting[METRIC.SUMMARY.GRT] is not None:
            stat[METRIC.SUMMARY.GRT] = np.clip(
                self.user_status_setting[METRIC.SUMMARY.GRT],
                a_min=self.user.status_variable.reaction.gaze.min,
                a_max=self.user.status_variable.reaction.gaze.max,
                dtype=int
            )
        else:
            while True:
                # if self.user.status_variable.reaction.hand.type == 'skewnorm':
                grt = round(sp.stats.skewnorm(
                    self.user.status_variable.reaction.gaze.alpha,
                    self.user.status_variable.reaction.gaze.mean,
                    self.user.status_variable.reaction.gaze.std
                ).rvs(1)[0])
                # elif:
                if self.user.status_variable.reaction.gaze.min <= grt <= self.user.status_variable.reaction.gaze.max:
                    break
            stat[METRIC.SUMMARY.GRT] = grt

        # Head position
        if self.user_status_setting[FIELD.PLAYER.HEAD.NAME] is not None:
            stat[FIELD.PLAYER.HEAD.NAME] = np.clip(
                self.user_status_setting[FIELD.PLAYER.HEAD.NAME],
                a_min=self.user.status_variable.head.min,
                a_max=self.user.status_variable.head.max
            )
        else:
            # if self.user.status_variable.reaction.head.type == 'norm':
            head_pos = np.random.normal(self.user.status_variable.head.mean, self.user.status_variable.head.std)
            # elif:
            head_pos = np.clip(
                head_pos, 
                a_min=self.user.status_variable.head.min,
                a_max=self.user.status_variable.head.max
            )
            stat[FIELD.PLAYER.HEAD.NAME] = head_pos
        
        # Gaze position
        if self.user_status_setting[FIELD.PLAYER.GAZE.NAME] is not None:
            ### TBU ###
            stat[FIELD.PLAYER.GAZE.NAME] = self.user_status_setting[FIELD.PLAYER.GAZE.NAME]
        else:
            gaze_angle = np.random.uniform(0, 360)
            while True:
                # if self.user.status_variable.reaction.head.type == 'uniform':
                gaze_dev = np.random.normal(self.user.status_variable.gaze.mean, self.user.status_variable.gaze.std)
                if self.user.status_variable.gaze.min <= gaze_dev <= self.user.status_variable.gaze.max:
                    break
                # elif:
            gaze_pos = cos_sin_array(gaze_angle) * gaze_dev
            stat[FIELD.PLAYER.GAZE.NAME] = gaze_pos

        return stat
    

    def reset(
        self, 
        seed=None, 
        sample_min_prob=0.0,
    ):
        if self.sample_mod_z:
            self.sample_z(sample_min_prob)
        self.update_params()

        # Aim-and-Shoot task condition reset
        self.initial_env_cond = self.game_env.reset(**self.game_env_setting)

        # Agent initial status reset
        current_user_stat = self.user_stat_sampler()
        self.user_current_status.hand_reaction_time = current_user_stat[METRIC.SUMMARY.MRT]
        self.user_current_status.gaze_reaction_time = current_user_stat[METRIC.SUMMARY.GRT]
        self.user_current_status.head_pos = current_user_stat[FIELD.PLAYER.HEAD.NAME]
        self.user_current_status.gaze_pos = current_user_stat[FIELD.PLAYER.GAZE.NAME]

        self.initial_env_cond[METRIC.SUMMARY.MRT] = self.user_current_status.hand_reaction_time
        self.initial_env_cond[METRIC.SUMMARY.GRT] = self.user_current_status.gaze_reaction_time
        self.initial_env_cond[FIELD.PLAYER.HEAD.NAME] = self.user_current_status.head_pos
        self.initial_env_cond[FIELD.PLAYER.GAZE.NAME] = self.user_current_status.gaze_pos

        # Set hand reaction time to 400 ms (== 4 BUMP interval)
        self.delayed_time = 4 * self.intv.bump - self.user_current_status.hand_reaction_time
        self.gaze_cooldown = self.user_current_status.gaze_reaction_time + self.delayed_time
        self.bump_plan_wait = 3
        self.timer = -self.delayed_time
        self.game_env.orbit_target(-self.delayed_time, unit='ms', inplace=True)

        # Skip BUMPs where both hand & gaze are stationary
        n_stationary_bump = min(self.gaze_cooldown, 3 * self.intv.bump) // self.intv.bump
        self.gaze_cooldown -= n_stationary_bump * self.intv.bump
        self.bump_plan_wait -= n_stationary_bump
        self.timer += n_stationary_bump * self.intv.bump      # Elapsed time
        self.game_env.orbit_target(n_stationary_bump * self.intv.bump, unit='ms', inplace=True)

        # Result attributes reset
        self.result = 0
        self.shoot_timing = 100_000
        self.truncation = False
        self.shoot_mp_determined = False
        self.shoot_timing_determined = False

        # Motor planning setting and history reset (considering BUMP skip)
        self.ongoing_mp_ideal = dict(
            p = np.zeros((self.intv.bump // self.intv.muscle + 1, 2)),
            v = np.zeros((self.intv.bump // self.intv.muscle + 1, 2)),
        )
        self.ongoing_mp_actual = dict(
            p = np.zeros((self.intv.bump // self.intv.muscle + 1, 2)),
            v = np.zeros((self.intv.bump // self.intv.muscle + 1, 2)),
        )
        self.pending_mp_ideal = dict(p=None, v=None)
        self.pending_mp_actual = dict(p=None, v=None)
        
        ans_current_state = self.game_env.get_current_state()
        for st in self.cstate:
            if st in ans_current_state:
                self.cstate[st] = ans_current_state[st]
        self.cstate["gaze_pos"] = self.user_current_status.gaze_pos
        self.cstate["head_pos"] = self.user_current_status.head_pos
        
        # self.h_traj_p = np.array([self.game_env.hand.pos.copy() for _ in range(n_stationary_bump * self.intv.bump // self.intv.muscle + 1)])
        # self.h_traj_v = np.array([self.game_env.hand.vel.copy() for _ in range(n_stationary_bump * self.intv.bump // self.intv.muscle + 1)])
        self.h_traj_p = np.zeros((n_stationary_bump * self.intv.bump // self.intv.muscle + 1, 2))
        self.h_traj_v = np.zeros((n_stationary_bump * self.intv.bump // self.intv.muscle + 1, 2))
        self.g_traj_p = np.array([self.user_current_status.gaze_pos, self.user_current_status.gaze_pos])
        self.g_traj_t = np.array([0, n_stationary_bump * self.intv.bump]) - self.delayed_time


        # Initial perception process
        target_pos_0_hat, t_sigma = Perceive.position_perception(
            self.game_env.target.pos.monitor,
            self.user_current_status.gaze_pos,
            self.user_param["theta_p"],
            head=self.user_current_status.head_pos,
            monitor_qt=self.game_env.window_qt,
            return_sigma=True
        )
        crosshair_pos_0_hat, c_sigma = Perceive.position_perception(
            self.game_env.crosshair,
            self.user_current_status.gaze_pos,
            self.user_param["theta_p"],
            head=self.user_current_status.head_pos,
            monitor_qt=self.game_env.window_qt,
            return_sigma=True
        )
        target_vel_by_orbit_true = self.game_env.target_monitor_velocity()
        target_vel_by_orbit_hat = Perceive.speed_perception(
            target_vel_by_orbit_true,
            # target_pos_0_hat,
            self.game_env.target.pos.monitor,
            self.user_param["theta_s"],
            head=self.user_current_status.head_pos
        )
        speed_error_ratio = np.linalg.norm(target_vel_by_orbit_hat) / np.linalg.norm(target_vel_by_orbit_true) \
            if np.linalg.norm(target_vel_by_orbit_true) > 0 else 1
        target_orbit_speed_hat = self.game_env.target.spd * speed_error_ratio
        target_pos_1_hat = self.game_env.target_monitor_position(
            initial_target_mpos=target_pos_0_hat,
            hand_displacement=np.zeros(2),
            orbit_angle=target_orbit_speed_hat * self.intv.bump / 1000
        )

        tpos_hat_error = target_pos_0_hat - self.game_env.target.pos.monitor
        tvel_hat_error = target_vel_by_orbit_hat - target_vel_by_orbit_true
        cpos_hat_error = crosshair_pos_0_hat - self.game_env.crosshair

        self.cstate.update(
            dict(
                target_pos_monitor = target_pos_1_hat,
                target_vel_orbit = target_vel_by_orbit_hat,
                target_rad = self.game_env.target.rad,
                hand_pos = self.game_env.hand.pos,
                hand_vel = self.game_env.hand.vel,
                # gaze_pos = self.user_current_status.gaze_pos,
                # head_pos = self.user_current_status.head_pos
            )
        )

        info = dict(
            time = self.timer,
            is_success = bool(self.result),
            belief = dict(
                t_sigma = t_sigma,
                c_sigma = c_sigma,
                tpos_hat_error = tpos_hat_error,
                tvel_hat_error = tvel_hat_error,
                cpos_hat_error = cpos_hat_error
            )
        )

        return self.np_cstate(), info
    

    def step(self, action_norm):
        action = self._unpack_action(action_norm)
        state = self._step_init_state()

        self._perceive_and_predict(state, action)   # Inplace state update
        self._plan_mouse_movement(state, action)
        self._plan_gaze_movement(state, action)
        self._plan_shoot_timing(state, action)
        self._check_shot_result(state, action)
        self._update_to_next_step(state, action)
        return self._terminate_or_truncate(state, action)
    

    def _unpack_action(self, action_norm):
        action_val = linear_denormalize(action_norm, *self.action_range.T)
        _action = {a: v for a, v in zip(self.user.action.list, action_val)}
        
        # Processing action values if required
        _action["th"] = (round(_action["th"]) // self.intv.muscle) * self.intv.muscle
        _action["th"] = min(max(_action["th"], int(round(self.user.action.th.min))), int(round(self.user.action.th.max)))
        _action["kc"] = int(_action["kc"] >= self.user.action.kc.threshold)
        _action["kg"] = np.sqrt(_action["kg"])

        return _action
    

    def _step_init_state(self):
        return Box(dict(
            tpos_0_hat = None,
            tpos_1_hat = None,
            tpos_2_hat = None,
            tvel_1_hat = None,
            cpos_0_hat = None,
            current_time = self.timer,
        ))
    

    def _perceive_and_predict(self, state, action):
        # Target state perception on SA (t=t0)
        # Position perception on target and crosshair
        tpos_0_hat, pos_sigma = Perceive.position_perception(
            self.game_env.target.pos.monitor,
            self.user_current_status.gaze_pos,
            self.user_param["theta_p"],
            head=self.user_current_status.head_pos,
            return_sigma=True
        )
        cpos_0_hat = Perceive.position_perception(
            self.game_env.crosshair,
            self.user_current_status.gaze_pos,
            self.user_param["theta_p"],
            head=self.user_current_status.head_pos
        )

        # Target state estimation on RP
        # If target position exceed the range of monitor,
        # the error caused by assumption (all movements are linear and constant)
        # rapidly increases. Therefore, convert monitor speed to orbit angular speed
        tvel_true = self.game_env.target_monitor_velocity(initial_target_mpos=tpos_0_hat, hand_vel=self.ongoing_mp_actual["v"][0])
        tvel_hat = Perceive.speed_perception(
            tvel_true,
            tpos_0_hat,
            self.user_param["theta_s"],
            head=self.user_current_status.head_pos
        )
        # The agent predictis the target movement emerged by aiming implicitly
        # assuming the ideal execution of motor plan
        tvel_aim_hat = self.game_env.target_monitor_velocity(
            hand_vel=self.ongoing_mp_ideal["v"][0], target_orbit_spd=0
        )
        tgvel_hat = tvel_hat - tvel_aim_hat     # Target speed by orbit on monitor. Speed and direction are perceived

        # Estimated target position on t0 + tp and t0 + tp + th
        # Ideal motor execution + estimated target orbit
        clock_noise = Perceive.timing_perception(1, self.user_param["theta_c"])
        tmpos_by_aim = self.game_env.target_monitor_position(
            initial_target_mpos=tpos_0_hat, 
            hand_displacement=self.ongoing_mp_ideal["p"][-1] - self.ongoing_mp_ideal["p"][0]
        )
        tpos_1_hat = tmpos_by_aim + (self.intv.bump / 1000) * tgvel_hat * clock_noise
        tpos_2_hat = tmpos_by_aim + (self.intv.bump + action["th"]) / 1000 * tgvel_hat * clock_noise

        state.tpos_0_hat = tpos_0_hat
        state.tpos_1_hat = tpos_1_hat
        state.tpos_2_hat = tpos_2_hat
        state.tvel_1_hat = tgvel_hat
        state.cpos_0_hat = cpos_0_hat
        state.clock_noise = clock_noise
        state.tpos_hat_error = tpos_0_hat - self.game_env.target.pos.monitor
        state.tpos_0_true = self.game_env.target.pos.monitor.copy()
        state.tpos_sigma = pos_sigma


    def _plan_mouse_movement(self, state, action):
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
            if self.pending_mp_actual["p"] is None:
                # Not decided to shoot: keep updating the motor plan
                if not action["kc"]:
                    ideal_plan_p, ideal_plan_v = Aim.plan_hand_movement(
                        self.ongoing_mp_actual["p"][-1],
                        self.ongoing_mp_actual["v"][-1],  # Start at the end of last "actual" motor execution
                        self.game_env.camera.pos,
                        self.game_env.camera.dir + self.game_env.hand.sensi * (self.ongoing_mp_ideal["p"][-1] - self.ongoing_mp_ideal["p"][0]),
                        state.cpos_0_hat,
                        state.tpos_2_hat,
                        state.tvel_1_hat,
                        self.game_env.hand.sensi,
                        self.game_env.camera.fov,
                        self.game_env.window_qt,
                        plan_duration=action["th"],
                        execute_duration=self.intv.bump,
                        interval=self.intv.muscle,
                    )

                    noisy_plan_p, noisy_plan_v = Aim.add_motor_noise(
                        ideal_plan_p[0],
                        ideal_plan_v,
                        self.user_param["theta_m"],
                        interval=self.intv.muscle
                    )
                # Decided to shoot: fix the rest of motor plan with th
                else:
                    ideal_plan_p, ideal_plan_v = Aim.plan_hand_movement(
                        self.ongoing_mp_actual["p"][-1],
                        self.ongoing_mp_actual["v"][-1],  # Start at the end of last "actual" motor execution
                        self.game_env.camera.pos,
                        self.game_env.camera.dir + self.game_env.hand.sensi * (self.ongoing_mp_ideal["p"][-1] - self.ongoing_mp_ideal["p"][0]),
                        state.cpos_0_hat,
                        state.tpos_2_hat,
                        state.tvel_1_hat,
                        self.game_env.hand.sensi,
                        self.game_env.camera.fov,
                        self.game_env.window_qt,
                        plan_duration=action["th"],
                        execute_duration=action["th"],
                        interval=self.intv.muscle,
                    )
                    # Sufficiently expand motor plan
                    expand_length = int((self.user.truncate_time - max(self.timer, 0)) // self.intv.muscle) + 2
                    ideal_plan_v = np.pad(ideal_plan_v, ((0, expand_length), (0, 0)), mode='edge')
                    ideal_plan_p = np.concatenate((
                        ideal_plan_p, 
                        ideal_plan_p[-1] + (ideal_plan_v[-1][:, np.newaxis] * np.arange(1, expand_length+1)).T * (self.intv.muscle/1000)
                    ))

                    noisy_plan_p, noisy_plan_v = Aim.add_motor_noise(
                        ideal_plan_p[0],
                        ideal_plan_v,
                        self.user_param["theta_m"],
                        interval=self.intv.muscle
                    )

                    # Store motor plans on SHOOT MOVING
                    self.pending_mp_ideal["p"] = ideal_plan_p[2:]
                    self.pending_mp_ideal["v"] = ideal_plan_v[2:]
                    self.pending_mp_actual["p"] = noisy_plan_p[2:]
                    self.pending_mp_actual["v"] = noisy_plan_v[2:]
                    self.shoot_mp_determined = True

                    # Maintain "next motor plan length" as tp
                    ideal_plan_p = ideal_plan_p[:3]
                    ideal_plan_v = ideal_plan_v[:3]
                    noisy_plan_p = noisy_plan_p[:3]
                    noisy_plan_v = noisy_plan_v[:3]

            # 2. Shoot motor plan in queue
            # Pop from queued plan
            else:
                noisy_plan_p = self.pending_mp_actual["p"][:3]
                noisy_plan_v = self.pending_mp_actual["v"][:3]

                ideal_plan_p = self.pending_mp_ideal["p"][:3]
                ideal_plan_p += (noisy_plan_p[0] - ideal_plan_p[0])
                ideal_plan_v = self.pending_mp_ideal["v"][:3]

                self.pending_mp_ideal["p"] = self.pending_mp_ideal["p"][2:]
                self.pending_mp_ideal["v"] = self.pending_mp_ideal["v"][2:]
                self.pending_mp_actual["p"] = self.pending_mp_actual["p"][2:]
                self.pending_mp_actual["v"] = self.pending_mp_actual["v"][2:]

        # Still waiting for reaction time to end     
        else:
            self.bump_plan_wait -= 1
            (
                ideal_plan_p,
                ideal_plan_v,
                noisy_plan_p,
                noisy_plan_v
            ) = (
                np.zeros((self.intv.bump // self.intv.muscle + 1, 2)),
                np.zeros((self.intv.bump // self.intv.muscle + 1, 2)),
                np.zeros((self.intv.bump // self.intv.muscle + 1, 2)),
                np.zeros((self.intv.bump // self.intv.muscle + 1, 2))
            )

        state.ideal_plan_p = ideal_plan_p
        state.ideal_plan_v = ideal_plan_v
        state.noisy_plan_p = noisy_plan_p
        state.noisy_plan_v = noisy_plan_v



    def _plan_gaze_movement(self, state, action):
        # Waiting for reaction time to end
        if self.gaze_cooldown > self.intv.bump:
            self.gaze_cooldown -= self.intv.bump
            gaze_time = np.array([self.timer, self.timer + self.intv.bump])
            gaze_traj = np.array([self.user_current_status.gaze_pos, self.user_current_status.gaze_pos])
            gaze_dest_ideal = self.user_current_status.gaze_pos
            gaze_dest_noisy = self.user_current_status.gaze_pos

        # Saccade
        else:
            gaze_dest_ideal = state.cpos_0_hat * (1-action["kg"]) + state.tpos_1_hat * action["kg"]
            gaze_dest_noisy = Gaze.gaze_landing_point(
                self.user_current_status.gaze_pos, 
                gaze_dest_ideal, 
                head=self.user_current_status.head_pos,
                monitor_qt=self.game_env.window_qt
            )
            gaze_time, gaze_traj = Gaze.gaze_plan(
                self.user_current_status.gaze_pos,
                gaze_dest_noisy,
                delay=self.gaze_cooldown,
                exe_until=self.intv.bump,
                head=self.user_current_status.head_pos
            )
            self.gaze_cooldown = 0
            gaze_time = self.timer + gaze_time  # Translate timestamp to current time
            gaze_dest_noisy = gaze_traj[-1]
        
        state.gaze_time = gaze_time
        state.gaze_traj = gaze_traj
        state.gaze_dest_ideal = gaze_dest_ideal
        state.gaze_dest_noisy = gaze_dest_noisy
    

    def _plan_shoot_timing(self, state, action):
        if self.shoot_mp_determined:
            self.shoot_mp_determined = False
            self.shoot_timing = max(round((self.intv.bump + action["tc"] * action["th"]) * state.clock_noise), 1)
    

    def _check_shot_result(self, state, action):
        done = False
        if self.shoot_timing <= self.intv.bump:
            # Interpolate ongoing motor plan for further calculations
            interp_plan_p, _ = Aim.interpolate_plan(
                self.ongoing_mp_actual["p"], self.ongoing_mp_actual["v"],
                self.intv.muscle, self.intv.interp
            )
            shoot_index = int(np.ceil(self.shoot_timing / self.intv.interp)) - 1
            hand_displacement = interp_plan_p[shoot_index] - interp_plan_p[0]
            ending_target = self.game_env.target_monitor_position(hand_displacement=hand_displacement,
                                                                  orbit_angle=self.intv.interp / 1000 * shoot_index * self.game_env.target.spd)
            state.shoot_distance = np.linalg.norm(ending_target - self.game_env.crosshair)
            state.shoot_result = state.shoot_distance <= self.game_env.target.rad
            state.shoot_moment = self.timer + self.shoot_timing
            state.shoot_endpoint = ending_target
            done = True

        state.done = done
        state.truncated = False
    

    def _update_to_next_step(self, state, action):
        self.shoot_timing -= self.intv.bump
        self.game_env.orbit_target(dt=self.intv.bump, unit='ms', inplace=True)
        self.game_env.move_hand(
            self.ongoing_mp_actual["p"][-1] - self.ongoing_mp_actual["p"][0], 
            self.ongoing_mp_actual["v"][-1]
        )
        self.user_current_status.gaze_pos = state.gaze_dest_noisy

        # self.game_env.output_current_status()

        self.h_traj_p = np.append(self.h_traj_p, self.ongoing_mp_actual["p"][1:], axis=0)
        self.h_traj_v = np.append(self.h_traj_v, self.ongoing_mp_actual["v"][1:], axis=0)
        self.g_traj_p = np.append(self.g_traj_p, state.gaze_traj[1:], axis=0)
        self.g_traj_t = np.append(self.g_traj_t, state.gaze_time[1:])


        # Observation Update
        self.cstate.update(
            dict(
                target_pos_monitor = state.tpos_1_hat,
                gaze_pos = state.gaze_dest_ideal,
                target_vel_orbit = state.tvel_1_hat,
                hand_vel = self.ongoing_mp_ideal["v"][-1]
            )
        )
        # Update plan
        self.ongoing_mp_ideal.update(dict(p = state.ideal_plan_p, v = state.ideal_plan_v))
        self.ongoing_mp_actual.update(dict(p = state.noisy_plan_p, v = state.noisy_plan_v))

        # For debugging in simulator ...
        state.camera_dir = self.game_env.camera.dir.copy()
    

    def _terminate_or_truncate(self, state, action):

        if state.done:
            self.timer = state.shoot_moment
            self.result = int(state.shoot_result)
            self.time_mean.append(self.timer)
            self.error_rate.append(int(state.shoot_result))
        else:
            self.timer += self.intv.bump
            self.result = 0

        state.action = action

        # Forced termination
        if ((
            abs(Convert.cart2sphr(*self.game_env.target.pos.game)[AXIS.E]) > self.user.truncate_target_elev or \
            (self.game_env.target_out_of_monitor() and self.timer > 0)
        ) and not state.done) or (self.timer >= self.user.truncate_time):
            state.done = True
            state.truncated = True
            state.shoot_result = False
            state.shoot_distance = self.game_env.target_crosshair_distance()
            state.shoot_endpoint = self.game_env.target.pos.monitor.copy()
            self.result = 0
            self.time_mean.append(self.timer)
            self.error_rate.append(0)
            # self.shoot_moment = self.timer
            if self.timer >= self.user.truncate_time:
                state.overtime = True
        
        reward = self._reward_function(state, action)
        info = dict(
            time = state.shoot_moment if (state.done and not state.truncated) else self.timer,
            result = state.shoot_result if (state.done and not state.truncated) else False,
            is_success = state.shoot_result if (state.done and not state.truncated) else False,
            step_state = state
        )

        return self.np_cstate(), reward, state.done, state.truncated, info


    def _reward_function(self, state, action):
        if state.truncated:
            rew = self.user.truncate_penalty
        elif state.done:

            if state.shoot_result:
                rew = self.user_param["rew_succ"] * ((1-self.user_param["decay_succ"]/100) ** (state.shoot_moment / 1000))
            else:
                rew = -self.user_param["rew_fail"] * ((1-self.user_param["decay_fail"]/100) ** (state.shoot_moment / 1000))
        else:
            rew = 0
        
        state.reward = rew
        return rew



class AnSBayesianUpdateVersion(AnSEnvDefault):
    def reset(self, seed=None, sample_min_prob=0.0):
        obs, info = super().reset(seed=seed, sample_min_prob=sample_min_prob)
        self.belief = info["belief"]
        return obs, info
    
    def _perceive_and_predict(self, state, action):
        tpos_0_hat, tpos_sigma = Perceive.position_perception(
            self.game_env.target.pos.monitor,
            self.user_current_status.gaze_pos,
            self.user_param["theta_p"],
            head=self.user_current_status.head_pos,
            return_sigma=True
        )
        cpos_0_hat = Perceive.position_perception(
            self.game_env.crosshair,
            self.user_current_status.gaze_pos,
            self.user_param["theta_p"],
            head=self.user_current_status.head_pos
        )
        ### ------------- MODIFIED SECTION
        tpos_0_hat_error = tpos_0_hat - self.game_env.target.pos.monitor
        cpos_0_hat_error = cpos_0_hat - self.game_env.crosshair

        # Target position belief update
        kalman_ft_tp_0 = self.belief["t_sigma"] ** 2 / (self.belief["t_sigma"] ** 2 + tpos_sigma ** 2)
        self.belief["t_sigma"] = np.sqrt(self.belief["t_sigma"] ** 2 - kalman_ft_tp_0 * self.belief["t_sigma"] ** 2)
        self.belief["tpos_hat_error"] = self.belief["tpos_hat_error"] + kalman_ft_tp_0 * (tpos_0_hat_error - self.belief["tpos_hat_error"])
        tpos_0_hat = self.game_env.target.pos.monitor + self.belief["tpos_hat_error"]

        # Crosshair position belief update
        kalman_ft_cp_0 = self.belief["c_sigma"] ** 2 / (self.belief["c_sigma"] ** 2 + tpos_sigma ** 2)
        self.belief["c_sigma"] = np.sqrt(self.belief["c_sigma"] ** 2 - kalman_ft_cp_0 * self.belief["c_sigma"] ** 2)
        self.belief["cpos_hat_error"] = self.belief["cpos_hat_error"] + kalman_ft_cp_0 * (cpos_0_hat_error - self.belief["cpos_hat_error"])
        cpos_0_hat = self.game_env.crosshair + self.belief["cpos_hat_error"]
        ### ------------- MODIFICATION ENDS

        tvel_true = self.game_env.target_monitor_velocity(initial_target_mpos=tpos_0_hat, hand_vel=self.ongoing_mp_actual["v"][0])
        tvel_hat = Perceive.speed_perception(
            tvel_true,
            tpos_0_hat,
            self.user_param["theta_s"],
            head=self.user_current_status.head_pos
        )
        tvel_aim_hat = self.game_env.target_monitor_velocity(
            hand_vel=self.ongoing_mp_ideal["v"][0], target_orbit_spd=0
        )
        tgvel_hat = tvel_hat - tvel_aim_hat

        clock_noise = Perceive.timing_perception(1, self.user_param["theta_c"])
        tmpos_by_aim = self.game_env.target_monitor_position(
            initial_target_mpos=tpos_0_hat, 
            hand_displacement=self.ongoing_mp_ideal["p"][-1] - self.ongoing_mp_ideal["p"][0]
        )
        tpos_1_hat = tmpos_by_aim + (self.intv.bump / 1000) * tgvel_hat
        tpos_2_hat = tmpos_by_aim + (self.intv.bump + action["th"]) / 1000 * tgvel_hat

        state.tpos_0_hat = tpos_0_hat
        state.tpos_1_hat = tpos_1_hat
        state.tpos_2_hat = tpos_2_hat
        state.tvel_1_hat = tgvel_hat
        state.cpos_0_hat = cpos_0_hat
        state.clock_noise = clock_noise

        ###
        state.tpos_hat_error = tpos_0_hat - self.game_env.target.pos.monitor
        state.tpos_0_true = self.game_env.target.pos.monitor.copy()
        state.tpos_sigma = tpos_sigma