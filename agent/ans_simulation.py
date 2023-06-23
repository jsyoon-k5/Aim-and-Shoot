"""
Aim-and-Shoot Simulator
This code implements the simulation queries.

Code written by June-Seop Yoon
"""
import numpy as np
import pandas as pd
import cv2

from stable_baselines3 import SAC
from typing import List, Union, Optional

import sys, os, copy
sys.path.append("..")

from configs.path import *
from configs.simulation import *
from configs.render import *
from utilities.otg import otg_2d
from utilities.mymath import (
    angle_btw,
    apply_gaussian_filter,
    int_floor, 
    img_resolution, 
    np_interp_2d, 
    mean_unbalanced_data
)
from utilities.utils import now_to_string, pickle_save
from utilities.render import *
from agent.ans_agent import *
from expdata.data_process import load_experiment


class EpisodeRecord:
    def __init__(self, game_cond, params):
        self.game_cond = game_cond
        self.actions = []

        self.times = [0, BUMP_INTERVAL]
        self.mefforts = [0, 0]
        self.results = [0, 0]
        self.params = copy.deepcopy(params)


    def update_status(self, env, action, done, info):
        self.actions.append(action)
        if info is not None:
            self.times.append(info["time"])
            self.mefforts.append(info["meffort"])
            self.results.append(info["result"])
        
        if done:
            self.actions = np.array(self.actions)
            self.times = np.array(self.times)
            self.mefforts = np.array(self.mefforts)
            self.results = np.array(self.results)
            self.shoot_error = info["shoot_error"]

            (
                self._timestamp, 
                self._ending_index, 
                self._hpos, 
                self._tpos, 
                self._gpos
            ) = self.interp_plan(env)
    
    
    def interp_plan(self, env):
        hpp, hpv, gpp, gpt, delay = env.episode_trajectory()
        # 200 FPS - 5ms interval btw frame

        # Hand movement interpolation
        raw_timestamp = [[-delay]]
        hpos = [hpp[0:1]]
        for i in range(1, hpp.shape[0]):
            hp, _ = otg_2d(
                hpp[i-1], hpv[i-1],
                hpp[i], hpv[i],
                TIME_UNIT[5],
                MUSCLE_INTERVAL,
                MUSCLE_INTERVAL
            )
            hpos.append(hp[1:])
            raw_timestamp.append(
                np.linspace(
                    (i-1)*MUSCLE_INTERVAL,
                    i*MUSCLE_INTERVAL,
                    len(hp)
                )[1:] - delay
            )
        raw_timestamp = np.concatenate(raw_timestamp)
        hpos = np.concatenate(hpos)

        # Add time = 0
        timestamp = np.insert(raw_timestamp, np.searchsorted(raw_timestamp, 0), 0)
        timestamp = timestamp[timestamp >= 0]
        hpos = np_interp_2d(timestamp, raw_timestamp, hpos)

        # Eye movement generation
        gpos = np_interp_2d(
            # np.arange(hpos.shape[0]) * TIME_UNIT[5],
            timestamp,
            gpt,    # Timestamp
            gpp
        )

        # Timestamp
        # timestamp = np.arange(hpos.shape[0]) * TIME_UNIT[5]

        # Target position
        game = GameState()
        game.reset(**self.game_cond)
        tpos = [game.tmpos]
        for i in range(1, timestamp.size):
            hp_delta = hpos[i] - hpos[i-1]
            game.move_hand(hp_delta)
            game.orbit(timestamp[i] - timestamp[i-1])
            tpos.append(game.tmpos)
        tpos = np.array(tpos)

        # Ending moment
        ending_index = np.searchsorted(timestamp, self.times[-1])

        return timestamp, ending_index, hpos, tpos, gpos
    

    def m_trial_completion_time(self):
        return self.times[-1]
    

    def m_result(self):
        return self.results[-1]
    

    def m_shoot_error(self):
        """DEPRECATED"""
        return np.linalg.norm(self._tpos[self._ending_index] - CROSSHAIR)
    

    def m_glancing_distance(self, ep=EYE_POSITION, in_degree=True):
        dist_argmin = np.argmin(
            np.linalg.norm(self._gpos - self._tpos[0], axis=1)
        )
        gp0 = np.array([*self._gpos[0], 0])
        gpn = np.array([*self._gpos[dist_argmin], 0])
        return angle_btw(gp0-ep, gpn-ep, output_degree=in_degree)
    

    def hand_speed_profile(self):
        ts = self._timestamp[:self._ending_index + 1]
        hp = self._hpos[:self._ending_index + 1]
        
        dt = ts[1:] - ts[:-1]
        dp = apply_gaussian_filter(np.linalg.norm(hp[1:] - hp[:-1], axis=1))
        # dp = np.linalg.norm(hp[1:] - hp[:-1], axis=1)
        hs = dp / dt

        return ts[:-1], hs
        


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
        gray_target:bool=False
    ):
        game = GameState()
        game.reset(**self.game_cond)

        timestamp_frame = np.linspace(
            0, self._timestamp[self._ending_index],
            int_floor(self._timestamp[self._ending_index] * framerate) + 1
        )

        # Interp plan
        tpos = np_interp_2d(timestamp_frame, self._timestamp, self._tpos)
        hpos = np_interp_2d(timestamp_frame, self._timestamp, self._hpos)
        gpos = np_interp_2d(timestamp_frame, self._timestamp, self._gpos)

        (
            hpos_pixel, 
            mp_range_meter, 
            mp_range_pixel
        ) = convert_hand_traj_meter_to_pixel(hpos, res)

        frame_num = timestamp_frame.size
        for i in range(frame_num):
            # if i > 0:
                # hp_delta = hpos[i] - hpos[i-1]
                # game.move_hand(hp_delta)
                # game.orbit(1/framerate)
            # game.fixate(gpos[i])

            # Game
            scene = draw_game_scene(
                res=res,
                pcam=hpos[i] * game.sensi,
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
                    [f'{p} = {self.params[p]:.3f}', C_SKYBLUE] for p in COG_SPACE
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
        checkpt: Optional[int] = None,
        modulated: bool = False,
        param_z: Optional[dict] = None,
        param_w: Optional[dict] = None
    ):
        self.model_name = model_name
        if checkpt is None:
            self.policy = PATH_RL_BEST % model_name + "best_model.zip"
            self.policy_name = f"{self.model_name} - best model"
        else:
            self.policy = PATH_RL_CHECKPT % model_name + f"rl_model_{checkpt}_steps.zip"
            self.policy_name = f"{self.model_name} - {checkpt} steps trained model"

        self.model = SAC.load(self.policy)
        self.simulations = []
        self.env = None

        if not modulated:
            param_w = pd.read_csv(PATH_RL_INFO % (self.model_name, "param")).to_dict("records")[0]
            self.env = Env(
                param_scale_z=None,
                param_scale_w=param_w
            )
        else:
            assert param_z is not None or param_w is not None

            param_fix = pd.read_csv(PATH_RL_INFO % (self.model_name, "param_fix")).to_dict("records")[0]
            self.param_mean = pd.read_csv(PATH_RL_INFO % (self.model_name, "param_mean")).to_dict("records")[0]
            self.param_std = pd.read_csv(PATH_RL_INFO % (self.model_name, "param_std")).to_dict("records")[0]

            var = self.param_mean.keys()

            if param_z is not None:
                self.env = VariableEnv(
                    variables=var,
                    variable_mean={**param_fix, **self.param_mean},
                    variable_std=self.param_std,
                    z_scale_range=param_fix["z_scale_range"],
                    fixed_z=np.array([param_z[v] for v in param_z.keys()])
                )
            elif param_w is not None:
                self.env = VariableEnv(
                    variables=var,
                    variable_mean={**param_fix, **self.param_mean},
                    variable_std=self.param_std,
                    z_scale_range=param_fix["z_scale_range"],
                    fixed_w=np.array([param_w[v] for v in param_w.keys()])
                )
    
    
    def set_game_env(self, game_setting:dict):
        self.env.update_game_setting(game_setting)
    

    def set_random_z_sample(self):
        assert type(self.env) == VariableEnv
        self.env.unfix_z()

    
    def update_parameter(
        self,
        param_z: Optional[dict] = None,
        param_w: Optional[dict] = None
    ):
        if param_z is not None:
            self.env.set_z(
                np.array([
                    param_z[v] for v in self.env.variables
                ])
            )
        elif param_w is not None:
            self.env.set_z(
                np.array([
                    convert_w2z(
                        param_w[v],
                        self.param_mean[v],
                        self.param_std[v]
                    ) for v in self.env.variables
                ])
            )
    

    def simulate(
        self,
        num_of_simul: int,
        overwrite_existing_simul: bool = False,
        exclude_forced_termination: bool = True,
        limit_num_of_exclusion: int = 100,
    ):
        episode = 0
        num_exclude = 0

        if overwrite_existing_simul: self.clear_record()

        while True:
            obs = self.env.reset()
            rec = EpisodeRecord(self.env.initial_game_cond, self.env.user_params)
            dones = False

            while not dones:
                action, state = self.model.predict(obs)
                obs, rewards, dones, info = self.env.step(action)
                rec.update_status(self.env, action, dones, info)
                if dones: break
            
            if not exclude_forced_termination or (exclude_forced_termination and not info["forced_termination"]):
                self.simulations.append(rec)
                episode += 1
            elif info["forced_termination"]:
                if exclude_forced_termination:
                    num_exclude += 1
                    if num_exclude == limit_num_of_exclusion:
                        num_exclude = 0
                        self.simulations.append(rec)
                        episode += 1
                else:
                    self.simulations.append(rec)
                    episode += 1

            if episode == num_of_simul:
                break
    

    def run_simulation_with_cond(
        self, 
        game_cond: List[dict], 
        rep: Optional[List[int]] = None,
        output_msg: bool = False,
        overwrite_existing_simul: bool = False,
        exclude_forced_termination: bool = True,
        limit_num_of_exclusion: int = 100,
    ):
        '''Run simulation for given condition list'''
        if rep is None: rep = [1] * len(game_cond)
        if output_msg:
            print(f"Running {np.sum(rep)} simulations ", end='')
        for i, (gc, n) in enumerate(zip(game_cond, rep)):
            self.set_game_env(gc)
            self.simulate(
                n,
                overwrite_existing_simul=overwrite_existing_simul,
                exclude_forced_termination=exclude_forced_termination,
                limit_num_of_exclusion=limit_num_of_exclusion
            )
            if output_msg:
                if i % (len(game_cond) // 5) == 0: print('.', end='')
        if output_msg: print()
    

    def _load_experiment_cond(self, exclude_invalid=True, **conditions):
        exp_data = load_experiment(**conditions, exclude_invalid=exclude_invalid)
        cond_list = [
            dict(
                pcam=_pcam,
                tgpos=sp2ct(*_tgpos),
                toax=_toax,
                gpos=_gpos,
                hrt=_hrt,
                grt=_grt,
                eye_pos=_ep,
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


    def m_trial_completion_time(self):
        return np.array([rec.m_trial_completion_time() for rec in self.simulations])

    
    def m_result(self):
        return np.array([rec.m_result() for rec in self.simulations])
    

    def m_shoot_error(self):
        return np.array([rec.m_shoot_error() for rec in self.simulations])
    

    def m_glancing_distance(self):
        return np.array([rec.m_glancing_distance() for rec in self.simulations])

    
    def target_cond_list(
        self, 
        keys=[
            "pcam", 
            "tgpos", 
            "tmpos", 
            "toax", 
            "tgspd", 
            "trad", 
            "gpos", 
            "session"
        ]
    ):
        cond = {k:[] for k in keys}
        for rec in self.simulations:
            for k in keys:
                cond[k].append(rec.game_cond[k])
        return cond


    
    def export_result_dict(
        self, 
        keys=[
            "pcam", 
            "tgpos", 
            "tmpos", 
            "toax", 
            "tgspd", 
            "trad", 
            "gpos", 
            "session"
        ]
    ):
        data = self.target_cond_list(keys=keys)
        data["result"] = self.m_result()
        data[M_TCT] = self.m_trial_completion_time()
        data[M_SE] = self.m_shoot_error()
        data[M_GD] = self.m_glancing_distance()

        return data
        

    def collect_trial_trajectory(self):
        ts, hp, tp, gp = [], [], [], []
        for rec in self.simulations:
            ts.append(rec._timestamp[:rec._ending_index])
            hp.append(rec._hpos[:rec._ending_index])
            tp.append(rec._tpos[:rec._ending_index])
            gp.append(rec._gpos[:rec._ending_index])

        return ts, hp, tp, gp


    def collect_distance_info(self, save_pkl=None):
        if save_pkl is not None:
            assert type(save_pkl is str)
        
        ts, hp, tp, gp = self.collect_trial_trajectory()
        dist_tx = [
            np.linalg.norm(tj - CROSSHAIR, axis=1) for tj in tp
        ]
        dist_gx = [
            np.linalg.norm(gj - gj[0], axis=1) for gj in gp
        ]
        mean_ts = np.arange(
            int_floor(self.m_trial_completion_time().mean() / TIME_UNIT[5])
        ) * TIME_UNIT[5]
        mean_tx = mean_unbalanced_data(
            dist_tx, 
            min_valid_n=1
        )[:mean_ts.size]
        mean_gx = mean_unbalanced_data(
            dist_gx,
            min_valid_n=1
        )[:mean_ts.size]

        data = (
            ts,
            hp,
            tp,
            gp,
            dist_tx, 
            dist_gx, 
            mean_ts, 
            mean_tx, 
            mean_gx
        )

        if save_pkl is not None:
            pickle_save(save_pkl, data)

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
    ):
        if framerate > 1 / TIME_UNIT[5]:
            print("Exceeded maximum frame rate - 200 fps")
            return

        num_simul = len(self.simulations)
        if num_simul > warning_threshold:
            print(f"Warning: there are {num_simul} episode saved.")
            print("Are you sure to render all of these?")
            ans = int(input("Type y to proceed: "))
            if ans != 'y': return
        
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

        for episode, rec in enumerate(self.simulations):
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

    model_name = "cog_230601_194710_66"

    simulator = Simulator(
        model_name,
        modulated = True,
        param_z = dict(
            theta_s = 0,
            theta_p = 0,
            theta_m = 0,
            theta_q = 0.8
        )
    )

    conds = simulator._load_experiment_cond(
        player = 'Tigrise',
        session_id = 'fsw',
        mode = 'default',
        # target_color="white",
        block_index = 0,
        # trial_index = 0,
        exclude_invalid = False
    )

    simulator.run_simulation_with_cond(conds)

    # x = simulator.export_result_dict()

    # for sess in ["fsw"]:
    #     simulator.set_game_env(dict(session=sess))
    #     simulator.simulate(50)
    
    *_, ts, tx, gx = simulator.collect_distance_info()
    simulator.export_simulation_video()
    plt.plot(ts, tx)
    plt.plot(ts, gx)
    plt.show()