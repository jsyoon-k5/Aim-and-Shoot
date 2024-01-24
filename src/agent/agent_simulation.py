"""
Aim-and-Shoot Simulator
This code implements the simulation queries.

Code written by June-Seop Yoon
"""
import numpy as np
from numpy.linalg.linalg import det
import pandas as pd
import cv2

from stable_baselines3 import SAC
from typing import List, Union, Optional

import sys, os, copy
from tqdm import tqdm
sys.path.append("..")

from configs.path import *
from configs.simulation import *
from configs.render import *
from configs.experiment import MAX_DEFAULT_MOUSE_DEV, SES_NAME_ABBR
from utils.otg import otg_2d
from utils.mymath import *
from utils.utils import *
from utils.render import *
from agent.agents import *
from experiment.data_process import load_experiment


class EpisodeRecord:
    def __init__(self, game_cond, params, z=None):
        self.game_cond = game_cond
        self.actions = []

        self.times = []
        self.results = []
        self.mefforts = []
        self.params = copy.deepcopy(params)
        if z is not None:
            self.z = copy.deepcopy(z)
        else:
            self.z = None


    def update_status(self, env, action, done, info):
        self.actions.append(action)
        if info is not None:
            self.times.append(info["time"])
            self.results.append(info["result"])
            if "meffort" in info.keys():
                self.mefforts.append(info["meffort"])
        
        if done:
            self.actions = np.array(self.actions)
            self.times = np.array(self.times)
            self.results = np.array(self.results)
            self.mefforts = np.array(self.mefforts)
            self.shoot_error = info["shoot_error"]
            # self.env = env

            (
                self._timestamp, 
                self._ending_index, 
                self._hpos, 
                self._tpos, 
                self._gpos,
                self._pcam      # Camera
            ) = self.interp_plan(env)
    
    
    def interp_plan(self, env, interval=5):
        hpp, hpv, gpp, gpt, delay = env.episode_trajectory()
        # 200 FPS - 5ms interval btw frame

        # Hand movement interpolation
        raw_timestamp = [[-delay]]
        hpos = [hpp[0:1]]
        for i in range(1, hpp.shape[0]):
            hp, _ = otg_2d(
                hpp[i-1], hpv[i-1],
                hpp[i], hpv[i],
                interval,       #ms
                MUSCLE_INTERVAL,
                MUSCLE_INTERVAL
            )
            hpos.append(hp[1:])
            raw_timestamp.append(
                np.linspace(
                    (i-1)*MUSCLE_INTERVAL/1000,
                    i*MUSCLE_INTERVAL/1000,
                    len(hp)
                )[1:] - delay
            )
        raw_timestamp = np.concatenate(raw_timestamp)
        hpos = np.concatenate(hpos)

        # Add time = 0
        timestamp = np.insert(raw_timestamp, np.searchsorted(raw_timestamp, 0), 0)
        timestamp = timestamp[timestamp >= 0]
        hpos = np_interp_nd(timestamp, raw_timestamp, hpos)
        hpos -= hpos[0]

        # Eye movement generation
        gpos = np_interp_nd(timestamp, gpt, gpp)

        # Target position
        game = GameState()
        game.reset(follow_exp_prior=False, **self.game_cond)
        tpos = [game.tmpos]
        pcam = [self.game_cond["pcam"]]
        for i in range(1, timestamp.size):
            hp_delta = hpos[i] - hpos[i-1]
            game.move_hand(hp_delta)
            pcam.append(game.pcam)
            game.orbit(timestamp[i] - timestamp[i-1])
            tpos.append(game.tmpos)
        tpos = np.array(tpos)
        pcam = np.array(pcam)

        # Ending moment
        ending_index = np.searchsorted(timestamp, self.times[-1]) - 1

        return timestamp, ending_index, hpos, tpos, gpos, pcam


    def trajectory(self, downsample=60, normalize=True, include_gaze=True, include_camera=True):
        ts, traj_t, traj_g, traj_c = (
            self._timestamp[:self._ending_index+1],
            self._tpos[:self._ending_index+1],
            self._gpos[:self._ending_index+1],
            self._pcam[:self._ending_index+1]
        )

        ts_ds = np.linspace(ts[0], ts[-1], max(2, int(ts[-1] * downsample)))
        traj_t = np_interp_nd(ts_ds, ts, traj_t)
        traj_g = np_interp_nd(ts_ds, ts, traj_g)
        traj_c = np_interp_nd(ts_ds, ts, traj_c)

        if normalize:
            traj_t = np.clip(traj_t / MONITOR_BOUND, a_min=-1, a_max=1)
            traj_g = np.clip(traj_g / MONITOR_BOUND, a_min=-1, a_max=1)
            traj_c = np.clip(traj_c / CAMERA_ANGLE_BOUND, a_min=-1, a_max=1)

        ts_diff = np.insert(np.diff(ts_ds), 0, 0)

        if include_gaze and include_camera:
            return np.vstack((ts_diff, traj_t.T, traj_g.T, traj_c.T), dtype=np.float32).T
        elif include_gaze and not include_camera:
            return np.vstack((ts_diff, traj_t.T, traj_g.T), dtype=np.float32).T
        elif not include_gaze and include_camera:
            return np.vstack((ts_diff, traj_t.T, traj_c.T), dtype=np.float32).T
        else:
            return np.vstack((ts_diff, traj_t.T), dtype=np.float32).T

    

    def m_trial_completion_time(self):
        return self.times[-1]
    

    def m_result(self):
        return self.results[-1]
    

    def m_shoot_error(self):
        return np.linalg.norm(self._tpos[self._ending_index] - CROSSHAIR)
    

    def m_shoot_error_norm(self):
        return self.m_shoot_error() / self.game_cond["trad"]
    

    def m_endpoint(self):
        return self._tpos[self._ending_index]

    def m_endpoint_norm(self):
        endpoint = self.m_endpoint()
        tdir_traj = np.array([
            target_monitor_position(np.zeros(3), _ptj, sphr2cart(*self._pcam[self._ending_index])) for _ptj in self._pcam[self._ending_index-10:self._ending_index]
        ])
        if len(tdir_traj) >= 2:
            tdir = np.mean(np.diff(tdir_traj, axis=0), axis=0)
            norm_den = np.linalg.norm(tdir)
            if norm_den > 0: tdir /= norm_den
            else: tdir = np.array([1.0, 0.0])
            tppd = rotate_point_2d(tdir, 90)
        else:
            tdir = np.array([1.0, 0.0])
            tppd = np.array([0.0, 1.0])

        return np.linalg.inv(np.array([tdir, tppd]).T) @ endpoint
    

    def m_glancing_distance(self, in_degree=True):
        dist_argmax = np.argmax(
            np.linalg.norm(self._gpos - CROSSHAIR, axis=1)
        )
        gp0 = np.array([*CROSSHAIR, 0])
        gpn = np.array([*self._gpos[dist_argmax], 0])
        return angle_between(
            gp0 - self.game_cond["head_pos"], 
            gpn - self.game_cond["head_pos"], 
            return_in_degree=in_degree
        )

    
    def m_cam_geometric_entropy(self):
        return geometric_entropy(self._pcam[:self._ending_index+1])
    

    def m_cam_max_speed(self):
        _, cs = self.cam_speed_profile()
        return np.max(cs)
    
    def m_cam_traverse_length(self):
        return np.sum(np.linalg.norm(np.diff(self._pcam[:self._ending_index+1], axis=0), axis=1))
    

    def m_prediction_horizon(self, min_th=0.1, max_th=2.0):
        return self.times/self.times[-1], linear_denormalize(self.actions[:,0], min_th, max_th)
    

    def m_shoot_attention(self):
        return self.times/self.times[-1], (linear_denormalize(self.actions[:,1], 0, 1) > THRESHOLD_SHOOT).astype(int)
    
    def m_implicit_aim_point(self):
        return self.times/self.times[-1], linear_denormalize(self.actions[:,2], 0, 1)
    

    def cam_speed_profile(self):
        ts = self._timestamp[:self._ending_index + 1]
        # cp = self._pcam[:self._ending_index + 1]
        
        # dt = ts[1:] - ts[:-1]
        # dp = apply_gaussian_filter(np.linalg.norm(cp[1:] - cp[:-1], axis=1))
        # cs = dp / dt

        return ts[:-1], 0
        


    def render_episode(
        self,
        videowriter,
        framerate:int,
        res:int,
        interval:int,   # Number of frame
        model_name:str, # Model name
        ep_i:int,
        ep_n:int,       # Number of current episode / total episode
        draw_gaze:bool=True,
        gray_target:bool=False,
        display_param:list=[*(f"theta_{p}" for p in "mpsc"), "hit", "miss", "hit_decay", "miss_decay"]
    ):
        game = GameState()
        game.reset(**self.game_cond)

        timestamp_frame = np.linspace(
            0, self._timestamp[self._ending_index],
            int_floor(self._timestamp[self._ending_index] * framerate) + 1
        )

        # Interp plan
        tpos = np_interp_nd(timestamp_frame, self._timestamp, self._tpos)
        hpos = np_interp_nd(timestamp_frame, self._timestamp, self._hpos)
        gpos = np_interp_nd(timestamp_frame, self._timestamp, self._gpos)
        pcam = np_interp_nd(timestamp_frame, self._timestamp, self._pcam)

        (
            hpos_pixel, 
            mp_range_meter, 
            mp_range_pixel
        ) = convert_hand_traj_meter_to_pixel(hpos, res)

        frame_num = timestamp_frame.size
        for i in range(frame_num):
            # Game
            scene = draw_game_scene(
                res=res,
                pcam=pcam[i],
                tmpos=tpos[i],
                trad=game.trad,
                gpos=gpos[i],
                draw_gaze=draw_gaze,
                gray_target=gray_target
            )
            # Mouse
            scene = draw_mouse_trajectory(
                scene,
                mp_range_pixel,
                mp_range_meter,
                hpos_pixel[:i+1]
            )

            # Text information
            current_time = i / framerate if i < frame_num - 1 else self.times[-1]
            scene = draw_messages(scene,
                [
                    [f'Model: {model_name}', C_WHITE],
                    [f'Time: {current_time:02.3f}', C_WHITE],
                    [f'Episode: {ep_i}/{ep_n}', C_WHITE],
                ]
            )
            scene = draw_messages(scene,
                [
                    [f'size = {self.game_cond["trad"]*2000:.2f} mm', C_WHITE],
                    [f'speed = {self.game_cond["tgspd"]:.2f} deg/s', C_WHITE],
                    [f'orbit = ({self.game_cond["toax"][0]:.2f}, {self.game_cond["toax"][1]:.2f})', C_WHITE],
                    [f'hand reaction = {self.game_cond["hrt"]*1000:.1f} ms', C_WHITE],
                    [f'gaze reaction = {self.game_cond["grt"]*1000:.1f} ms', C_WHITE],
                ], s=0.4, skip_row=6
            )
            scene = draw_messages(scene,
                [
                    [f'{p} = {self.params[p]:.3f}', C_SKYBLUE] for p in display_param
                ], s=0.3, skip_row=15
            )
            videowriter.write(scene)
        
        # Paused part
        if self.results[-1] == 1:
            scene = draw_messages(scene, [["Shot Hit", C_GREEN]], skip_row=3)
        else:
            scene = draw_messages(scene, [["Shot Miss", C_RED]], skip_row=3)
        for _ in range(interval): videowriter.write(scene)



class Simulator:
    def __init__(
        self, 
        model_name: str, 
        checkpt: Optional[int] | str = 'best'
    ):
        ### Load model
        self.model_name = model_name
        if checkpt == 'best':
            self.policy = PATH_RL_BEST % model_name + "best_model.zip"
            self.policy_name = f"{self.model_name} - best model"
        elif type(checkpt) is int:
            self.policy = PATH_RL_CHECKPT % model_name + f"rl_model_{checkpt}_steps.zip"
            self.policy_name = f"{self.model_name} - {checkpt} steps trained model"
        else:
            raise ValueError("Put valid checkpt parameter.")

        self.modulated = os.path.exists(PATH_RL_INFO % (self.model_name, "param_fix.csv"))
        self.model = SAC.load(self.policy)
        self.simulations = []

        class_env = pickle_load(PATH_RL_INFO % (self.model_name, "env_class.pkl"))
        self.env_config = pickle_load(PATH_RL_INFO % (self.model_name, "configuration.pkl"))
        self.env = class_env(env_setting=self.env_config)

        self.fixed_param_w = dict()
        self.fixed_param_z = dict()
        
        self.set_game_env(dict())
    
    
    def set_game_env(self, game_setting:dict):
        game_setting["follow_exp_prior"] = False
        self.env.update_game_setting(game_setting)
    

    def set_random_z_sample(self):
        if self.modulated: self.env.unfix_z()
    
    def fix_current_z(self):
        if self.modulated: self.env.fix_z()
    

    def fix_parameter(
        self,
        param_z: Optional[dict] = None,
        param_w: Optional[dict] = None
    ):
        if param_z is not None:
            self.fixed_param_z = param_z.copy()
            self.fixed_param_w = dict()
            for v in self.fixed_param_z.keys():
                self.fixed_param_w[v] = log_denormalize(
                    param_z[v],
                    *self.env.variable_range[self.env.variables.index(v)],
                    self.env.variable_scale
                )
        elif param_w is not None:
            self.fixed_param_w = param_w.copy()
            self.fixed_param_z = dict()
            for v in self.fixed_param_w.keys():
                self.fixed_param_z[v] = log_normalize(
                    param_w[v],
                    *self.env.variable_range[self.env.variables.index(v)],
                    self.env.variable_scale
                )

    
    def update_parameter(
        self,
        param_z_raw: Optional[np.ndarray] = None,
        param_z: Optional[dict] = None,
        param_w: Optional[dict] = None,
        fill_empty: str = "random"
    ):
        if param_z_raw is not None:
            self.env.set_param(raw_z=param_z_raw)
        if param_z is not None:
            param_z.update(self.fixed_param_z)
            empty_params = list(set(self.env.variables).difference(set(param_z.keys())))
            if fill_empty == 'random':
                for v in empty_params: param_z[v] = np.random.uniform(-1, 1)
            elif fill_empty == 'min':
                for v in empty_params: param_z[v] = -1
            elif fill_empty == 'mean':
                for v in empty_params: param_z[v] = 0
            self.env.set_param(z=param_z)
        elif param_w is not None:
            param_w.update(self.fixed_param_w)
            empty_params = list(set(self.env.variables).difference(set(param_w.keys())))
            for v in empty_params:
                param_w[v] = log_denormalize(-1, *self.env.variable_range[self.env.variables.index(v)], self.env.variable_scale)
            self.env.set_param(w=param_w)
    

    def simulate(
        self,
        num_of_simul: int,
        overwrite_existing_simul: bool = False,
        exclude_forced_termination: bool = True,
        exclude_dummy_result: bool = False,
        limit_num_of_exclusion: int = 20,
        verbose=True
    ):
        episode = 0
        num_exclude = 0

        if overwrite_existing_simul: self.clear_record()

        progress = tqdm(total=num_of_simul, desc="Simulations: ") if verbose else None

        while True:
            obs, _ = self.env.reset()
            if self.modulated:
                rec = EpisodeRecord(self.env.initial_game_cond, self.env.user_params, z=self.env.z)
            else:
                rec = EpisodeRecord(self.env.initial_game_cond, self.env.user_params)
            dones = False

            while not dones:
                action, state = self.model.predict(obs, deterministic=False)
                obs, rewards, dones, truncated, info = self.env.step(action)
                rec.update_status(self.env, action, dones, info)
                if dones: break
            
            # Simplify these branches ...
            if (
                (exclude_forced_termination and info["forced_termination"] and not info["overtime"]) or \
                (exclude_dummy_result and info["shoot_error"] > VALID_SHOOT_ERROR)
            ):
                num_exclude += 1
                if num_exclude >= limit_num_of_exclusion:
                    # print("Maximum exclusion reached... appending this trial:", self.env.z)
                    num_exclude = 0
                    self.simulations.append(rec)
                    episode += 1
                    if verbose: progress.update(1)
            else:
                self.simulations.append(rec)
                episode += 1
                if verbose: progress.update(1)

            if episode == num_of_simul:
                break
    

    def run_simulation_with_cond(
        self, 
        game_cond: List[dict], 
        param_list: Optional[List[dict]] = None,    # Must be specified as "z"
        rep: Optional[List[int]] | int = None,
        verbose: bool = False,
        overwrite_existing_simul: bool = False,
        exclude_forced_termination: bool = True,
        limit_num_of_exclusion: int = 20,
    ):
        '''Run simulation for given condition list'''
        if rep is None: rep = [1] * len(game_cond)
        elif type(rep) is int: rep = [rep] * len(game_cond)

        if overwrite_existing_simul: self.clear_record()

        for i in tqdm(range(len(game_cond))) if verbose else range(len(game_cond)):
            self.set_game_env(game_cond[i])
            if param_list is not None:
                self.update_parameter(param_z=param_list[i])
            self.simulate(
                rep[i],
                overwrite_existing_simul=False,
                exclude_forced_termination=exclude_forced_termination,
                limit_num_of_exclusion=limit_num_of_exclusion,
                verbose=False
            )
    

    def _load_experiment_cond(
        self, 
        exclude_invalid=True, 
        fix_head_position=False, 
        fix_gaze_reaction=False,
        fix_hand_reaction=False,
        **conditions
    ):
        exp_data = load_experiment(**conditions, exclude_invalid=exclude_invalid)

        if fix_head_position:
            for i, a in enumerate('xyz'):
                exp_data[f"eye_{a}"] = HEAD_POSITION[i]
        if fix_gaze_reaction:
            exp_data["gaze_reaction"] = MEAN_GAZE_REACTION
        if fix_hand_reaction:
            # exp_data["hand_reaction"] = MEAN_HAND_REACTION
            exp_data["hand_reaction"] = 0.2

        cond_list = [
            dict(
                pcam=_pcam,
                tgpos=sphr2cart(*_tgpos),
                toax=_toax,
                gpos=_gpos,
                hrt=_hrt,
                grt=_grt,
                head_pos=_ep,
                session=_sess
            ) for _pcam, _tgpos, _toax, _gpos, _ep, _hrt, _grt, _sess in zip(
                exp_data[["pcam0_az", "pcam0_el"]].to_numpy(), 
                exp_data[["tgpos0_az", "tgpos0_el"]].to_numpy(), 
                exp_data[["toax_az", "toax_el"]].to_numpy(), 
                exp_data[["gaze0_x", "gaze0_y"]].to_numpy(), 
                exp_data[["eye_x", "eye_y", "eye_z"]].to_numpy(),
                exp_data["hand_reaction"].to_numpy(),
                exp_data["gaze_reaction"].to_numpy(),
                exp_data["session_id"].to_list()
            )
        ]
            
        return cond_list


    def _reward_mapper(self, cond_list, cog_dict, reward_list, reward_map_dict):
        params = []
        for c in cond_list:
            tdist = np.linalg.norm(target_monitor_position(np.zeros(3), c["pcam"], c["tgpos"]))
            tspd = 0 if c["session"][0] == 's' else TARGET_ASPEED_FAST
            if c["session"][1] == 's': trad = TARGET_RADIUS_SMALL
            elif c["session"][1] == 'v': trad = TARGET_RADIUS_VERY_SMALL
            elif c["session"][1] == 'l': trad = TARGET_RADIUS_LARGE
            iod = linear_normalize(np.log2(tdist / (2*trad) + 1), -6, 6) # [0, 1]
            spd = linear_normalize(tspd, -TARGET_ASPEED_FAST, TARGET_ASPEED_FAST)            # [0, 1]
            diff = (linear_normalize(iod * (1 + spd), 0.0747, 1.489) + 1) / 2     # [0, 1]

            rew_z = []
            for rew in reward_list:
                map_a = reward_map_dict[f"{rew}_a"]
                map_b = reward_map_dict[f"{rew}_b"]
                map_b = (map_b + 1) / 2     # [-1, 1] -> [0, 1]
                map_rew = map_a * map_b * (diff-1) + map_b if map_a >= 0 else map_a * map_b * diff + map_b
                rew_z.append(map_rew * 2 - 1)   # [0, 1] -> [-1, 1]
            params.append({**cog_dict, **dict(zip(reward_list, rew_z))})

        return params


    def m_trial_completion_time(self):
        return np.array([rec.m_trial_completion_time() for rec in self.simulations])
    
    def m_result(self):
        return np.array([rec.m_result() for rec in self.simulations])

    def m_endpoint(self):
        return np.array([rec.m_endpoint() for rec in self.simulations])
    
    def m_endpoint_norm(self):
        return np.array([rec.m_endpoint_norm() for rec in self.simulations])

    def m_shoot_error(self):
        return np.array([rec.m_shoot_error() for rec in self.simulations])
    
    def m_shoot_error_norm(self):
        return np.array([rec.m_shoot_error_norm() for rec in self.simulations])

    def m_glancing_distance(self):
        return np.array([rec.m_glancing_distance() for rec in self.simulations])

    def m_cam_geometric_entropy(self):
        return np.array([rec.m_cam_geometric_entropy() for rec in self.simulations])
    
    def m_cam_max_speed(self):
        return np.array([rec.m_cam_max_speed() for rec in self.simulations])
    
    def m_cam_traverse_length(self):
        return np.array([rec.m_cam_traverse_length() for rec in self.simulations])
    
    def m_prediction_horizon(self, rec_indices=None):
        t = []
        th = []
        if rec_indices is None:
            for rec in self.simulations:
                _t, _th = rec.m_prediction_horizon()
                t.append(_t)
                th.append(_th)
        else:
            for ir in rec_indices:
                _t, _th = self.simulations[ir].m_prediction_horizon()
                t.append(_t)
                th.append(_th)
        t = np.concatenate(t)
        th = np.concatenate(th)
        data = np.vstack((t, th)).T
        data = data[data[:,0].argsort()]
        data = np_groupby(data, key_pos=0, lvl=20)
        return data[:,0], data[:,1]
    

    def m_shoot_attention(self, rec_indices=None):
        t = []
        sa = []
        if rec_indices is None:
            for rec in self.simulations:
                _t, _sa = rec.m_shoot_attention()
                t.append(_t)
                sa.append(_sa)
        else:
            for ir in rec_indices:
                _t, _sa = self.simulations[ir].m_shoot_attention()
                t.append(_t)
                sa.append(_sa)
        t = np.concatenate(t)
        sa = np.concatenate(sa)
        data = np.vstack((t, sa)).T
        data = data[data[:,0].argsort()]
        data = np_groupby(data, key_pos=0, lvl=20)
        return data[:,0], data[:,1]
    

    def m_implicit_aim_point(self, rec_indices=None):
        t = []
        sa = []
        if rec_indices is None:
            for rec in self.simulations:
                _t, _sa = rec.m_implicit_aim_point()
                t.append(_t)
                sa.append(_sa)
        else:
            for ir in rec_indices:
                _t, _sa = self.simulations[ir].m_implicit_aim_point()
                t.append(_t)
                sa.append(_sa)
        t = np.concatenate(t)
        sa = np.concatenate(sa)
        data = np.vstack((t, sa)).T
        data = data[data[:,0].argsort()]
        data = np_groupby(data, key_pos=0, lvl=20)
        return data[:,0], data[:,1]
        

    
    def target_cond_dict(self):
        """Return experiment_result like dictionary target condition info"""
        assert len(self.simulations) > 0
        # Task initial condition
        cond = dict(
            session_id=[],
            target_speed=[],
            target_radius=[],
            tgpos0_az=[],
            tgpos0_el=[],
            tmpos0_x=[],
            tmpos0_y=[],
            tmpos0_dist=[],
            t_iod=[],
            pcam0_az=[],
            pcam0_el=[],
            gaze0_x=[],
            gaze0_y=[],
            eye_x=[],
            eye_y=[],
            eye_z=[],
            toax_az=[],
            toax_el=[],
            hand_reaction=[],
            gaze_reaction=[]
        )
        # Modulation
        if self.modulated:
            for param in self.env.variables: cond[param] = []
        for rec in self.simulations:
            cond["session_id"].append(rec.game_cond["session"])
            cond["target_speed"].append(rec.game_cond["tgspd"])
            cond["target_radius"].append(rec.game_cond["trad"])
            tgpos = cart2sphr(*rec.game_cond["tgpos"])
            cond["tgpos0_az"].append(tgpos[X])
            cond["tgpos0_el"].append(tgpos[Y])
            cond["tmpos0_x"].append(rec.game_cond["tmpos"][X])
            cond["tmpos0_y"].append(rec.game_cond["tmpos"][Y])
            cond["tmpos0_dist"].append(np.linalg.norm(rec.game_cond["tmpos"]))
            cond["t_iod"].append(
                np.log2(np.linalg.norm(rec.game_cond["tmpos"])/(2*rec.game_cond["trad"]) + 1)
            )
            cond["pcam0_az"].append(rec.game_cond["pcam"][X])
            cond["pcam0_el"].append(rec.game_cond["pcam"][Y])
            cond["gaze0_x"].append(rec.game_cond["gpos"][Y])
            cond["gaze0_y"].append(rec.game_cond["gpos"][X])
            cond["eye_x"].append(rec.game_cond["head_pos"][X])
            cond["eye_y"].append(rec.game_cond["head_pos"][Y])
            cond["eye_z"].append(rec.game_cond["head_pos"][Z])
            cond["toax_az"].append(rec.game_cond["toax"][X])
            cond["toax_el"].append(rec.game_cond["toax"][Y])
            cond["hand_reaction"].append(rec.game_cond["hrt"])
            cond["gaze_reaction"].append(rec.game_cond["grt"])

            if self.modulated:
                for i, param in enumerate(self.env.variables):
                    cond[param].append(rec.z[i])

        return cond

    
    def export_result_df(self, include_target_cond=True):
        if include_target_cond:
            data = self.target_cond_dict()
        else:
            data = dict()

        data[M_ACC] = self.m_result()
        data[M_TCT] = self.m_trial_completion_time()
        endp = self.m_endpoint()
        endp_norm = self.m_endpoint_norm()
        data["endpoint_x"] = endp[:,X]; data["endpoint_y"] = endp[:,Y]
        data["endpoint_x_norm"] = endp_norm[:,X]; data["endpoint_y_norm"] = endp_norm[:,Y]
        data[M_SE] = self.m_shoot_error()
        data[M_SE_NORM] = self.m_shoot_error_norm()
        data[M_GD] = self.m_glancing_distance()
        data[M_CGE] = self.m_cam_geometric_entropy()
        data[M_CSM] = self.m_cam_max_speed()
        data[M_CTL] = self.m_cam_traverse_length()

        return pd.DataFrame(data)
    

    def collect_trial_trajectory_downsample(self, downsample=60, normalize=True, include_gaze=True, include_camera=True):
        return [
            rec.trajectory(
                downsample=downsample, 
                normalize=normalize,
                include_gaze=include_gaze,
                include_camera=include_camera
            ) for rec in self.simulations
        ]
        

    def collect_trial_trajectory(self):
        ts, hp, tp, gp, cp = [], [], [], [], []
        for rec in self.simulations:
            ts.append(rec._timestamp[:rec._ending_index+1])
            hp.append(rec._hpos[:rec._ending_index+1])
            tp.append(rec._tpos[:rec._ending_index+1])
            gp.append(rec._gpos[:rec._ending_index+1])
            cp.append(rec._pcam[:rec._ending_index+1])

        return ts, hp, tp, gp, cp
    

    # def collect_distance_info_normalized(self, return_mean_only=True):



    def collect_distance_info(self, save_pkl=None, return_mean_only=False):
        if save_pkl is not None:
            assert type(save_pkl is str)
        
        ts, hp, tp, gp, cp = self.collect_trial_trajectory()

        tct_mean = self.m_trial_completion_time().mean()

        dist_tx = [np.linalg.norm(tj - CROSSHAIR, axis=1) for tj in tp]
        dist_gx = [np.linalg.norm(gj - gj[0], axis=1) for gj in gp]
        
        new_ts, mean_tx, mean_gx = mean_distance(ts, dist_tx, dist_gx, tct_mean)

        data = (
            ts, hp, tp, gp, cp,
            dist_tx, dist_gx, 
            new_ts, mean_tx, mean_gx
        )

        if save_pkl is not None:
            pickle_save(save_pkl, data)

        if return_mean_only:
            return (new_ts, mean_tx, mean_gx)
        return data

    
    def export_simulation_video(
        self,
        framerate: int = 60,
        res: int = 720,
        interval: Union[float, List[float]] = 0.6,    # sec
        path: Optional[str] = None,
        videoname: Optional[str] = None,
        draw_gaze: bool = True,
        gray_target: bool = False,
        warning_threshold: int = 100,
        verbose=True
    ):
        if framerate > 200:
            print("Exceeded maximum frame rate - 200 fps")
            return

        num_simul = len(self.simulations)
        if num_simul > warning_threshold:
            print(f"Warning: there are {num_simul} episode saved.")
            print("Are you sure to render all of these?")
            ans = input("Type y to proceed: ")
            if ans != 'y' and ans != 'Y': return
        
        # Interval between episodes (frame)
        if type(interval) is list:
            interval = (np.array(interval) * framerate).astype(int)
            assert num_simul == interval.size
        else:
            interval = [int(interval * framerate)] * num_simul

        if path is None:
            path = PATH_VIS_VIDEO % self.model_name
        os.makedirs(path, exist_ok=True)

        if videoname is None: videoname = now_to_string()

        video = cv2.VideoWriter(
            f"{path}{videoname}.mp4",
            CODEC,
            framerate, img_resolution(res)
        )

        print(f"Rendering {num_simul} simulation(s) ...")

        for episode, rec in enumerate(tqdm(self.simulations)) if verbose else enumerate(self.simulations):
            rec.render_episode(
                video,
                framerate,
                res,
                interval[episode],    # Number of frame
                self.policy_name,  # Model name
                episode+1, num_simul,  # Number of current episode / total episode
                draw_gaze=draw_gaze,
                gray_target=gray_target
            )
        video.release()
    

    def clear_record(self):
        self.simulations = []



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    simulator = Simulator(model_name="base_240104_164252_68", checkpt=20000000)
    simulator.update_parameter(
        param_w=dict(
            theta_m=0.1,
            # theta_p=0.1,
            theta_s=0.1,
            theta_c=0.16,
            hit=30,
            miss=16,
            hit_decay=0.7,
            miss_decay=0.3,
        )
    )

    # cond = simulator._load_experiment_cond(**load_cond)
    # simulator.run_simulation_with_cond(cond, verbose=True)

    cond = simulator._load_experiment_cond(
        player='sjrla',
        # tier='PRO',
        mode='custom',
        session_id='ssw',
        block_index=1,
        fix_hand_reaction=True
        # target_speed=0,
        # target_color='white'
    )
    simulator.run_simulation_with_cond(cond, verbose=True)
    # m = np.array([r.mefforts[-1] for r in simulator.simulations])

    # plt.hist(m, bins=20)
    # plt.show()
    simulator.export_simulation_video(draw_gaze=True)

    # df = simulator.export_result_df()
    # df.to_csv("temp.csv", index=False)

    # ts, tx, gx = simulator.collect_distance_info(return_mean_only=True)

    # plt.plot(ts, tx)
    # plt.plot(ts, gx)
    # plt.show()