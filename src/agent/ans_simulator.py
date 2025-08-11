"""
Aim-and-Shoot Simulator
This code implements the simulation queries.

Code written by June-Seop Yoon
"""

import numpy as np
import pandas as pd
import cv2

import os
from tqdm import tqdm
from pathlib import Path
from box import Box

from stable_baselines3 import SAC
from typing import List, Union, Optional

from ..agent import ans_agent
from ..agent.module_aim import Aim
from ..agent.ans_task import AnSGame
from ..config.constant import FOLDER, VIDEO, AXIS, FIELD, METRIC
from ..utils.myutils import load_config, get_timebase_session_name
from ..utils.render import RENDER, BACKGROUND_REFERENCE, COLOR
from ..utils.mymath import (
    angle_between, linear_normalize, log_normalize, np_interp_nd, cos_sin_array, Convert, Rotate
)


class AnSEpisodeRecordDefault:
    def __init__(self, env):
        self.user = env.user
        self.intv = env.intv
        self.param = env.user_param.copy()
        self.initial_task_cond = env.initial_env_cond.copy()
        self.task_config = Box(env.env_setting_info["task_config"])

        self.monitor_qt = np.array([
            self.task_config.window.monitor.width,
            self.task_config.window.monitor.height,
        ], dtype=np.float32) / 2
        self.fov = np.array([
            self.task_config.window.fov.width,
            self.task_config.window.fov.height,
        ])

        self.actions = list()
        self.perceive = dict(
            tpos_0_hat = list(),
            tpos_1_hat = list(),
            tpos_2_hat = list(),
            tvel_1_hat = list(),
            cpos_0_hat = list(),
            tpos_hat_error = list(),
            tpos_0_true = list(),
            tpos_sigma = list(),
            current_time = list(),
            reward = list(),
            camera_dir = list()
        )
        self.result = Box(dict(
            success = False,
            time = 0,
            shoot_distance = 0,
            shoot_endpoint = None,
            truncated = False,
            reward = 0
        ))
    

    def record(self, env, action, done, truncated, info):
        self.actions.append(pd.DataFrame([info["step_state"].action]))
        # self.actions.append(env._unpack_action(action))
        for k in self.perceive:
            self.perceive[k].append(info["step_state"][k])

        if done or truncated:
            self.actions = pd.concat(self.actions)
            for k in self.perceive:
                self.perceive[k] = np.array(self.perceive[k])
            self.result.success = info["step_state"].shoot_result
            self.result.time = int(info["time"])
            self.result.shoot_distance = info["step_state"].shoot_distance
            self.result.truncated = info["step_state"].truncated
            self.result.shoot_endpoint = info["step_state"].shoot_endpoint
            self.result.reward = info["step_state"].reward

            self.interpolate_action(env)
    

    def interpolate_action(self, env):
        hp, hv, gp, gt, delay = env.episode_trajectory()
        hp, hv = Aim.interpolate_plan(hp, hv, self.intv.muscle, self.intv.interp)

        timestamp_old = np.arange(hp.shape[0]) * self.intv.interp - delay
        timestamp_new = np.linspace(0, self.result.time, int(self.result.time // self.intv.interp) + 1)
        self.gtraj_p = np_interp_nd(timestamp_new, gt, gp)
        self.htraj_p = np_interp_nd(timestamp_new, timestamp_old, hp)
        self.htraj_v = np_interp_nd(timestamp_new, timestamp_old, hv)
        self.replay = self.replay_and_save()
    

    def replay_and_save(self):
        ans_replay = AnSGame(config=self.task_config)

        # TBU - use AnSGame.replay_and_save built-in function later...
        # Currently, synethesize the replay manually

        cdir = ans_replay.hand.sensi * self.htraj_p + self.initial_task_cond["cdir"]
        tgpos = list()
        for i in range(len(self.htraj_p)):
            tgpos.append(Rotate.point_about_axis(
                self.initial_task_cond["tgpos"],
                self.intv.interp / 1000 * i * self.initial_task_cond["tspd"],
                *self.initial_task_cond["torbit"]
            ))
        tgpos = np.array(tgpos)
        tmpos = Convert.games2monitors(
            np.tile(ans_replay.camera.pos, (tgpos.shape[0], 1)),
            cdir,
            tgpos,
            ans_replay.camera.fov,
            ans_replay.window_qt
        )
        hand_pos = self.htraj_p.copy()

        return dict(
            target_pos_monitor = tmpos,
            hand_pos = hand_pos,
            camera_dir = cdir
        )

        # ans_replay.reset(**self.initial_task_cond)
        # return ans_replay.replay_and_save(self.htraj_p, self.htraj_v, 
        #                                   self.intv.interp, unit='ms', 
        #                                   keys=["target_pos_monitor", "hand_pos", "camera_dir"])


    def get_summary_result(self):

        # Convert cart to sphr of target game position
        tgpos_sphr = Convert.cart2sphr(*self.initial_task_cond["tgpos"])
        target_dist = np.linalg.norm(self.initial_task_cond["tmpos"])

        # Saccadic deviation
        # gaze_dist_argmax = np.argmax(np.linalg.norm(self.gtraj_p - self.gtraj_p[0], axis=1))
        gaze_dist_argmax = np.argmax(np.linalg.norm(self.gtraj_p - np.zeros(2), axis=1))
        m_scd = angle_between(
            np.array([*self.gtraj_p[0], 0]) - self.initial_task_cond[FIELD.PLAYER.HEAD.NAME],
            np.array([*self.gtraj_p[gaze_dist_argmax], 0]) - self.initial_task_cond[FIELD.PLAYER.HEAD.NAME],
            return_in_degree=True
        )

        # Normalized endpoint - TBU

        return pd.DataFrame([{
            # Task condition
            FIELD.TARGET.SPD: self.initial_task_cond["tspd"],
            FIELD.TARGET.RAD: self.initial_task_cond["trad"],
            FIELD.TARGET.POS.GAME.A0: tgpos_sphr[AXIS.A],
            FIELD.TARGET.POS.GAME.E0: tgpos_sphr[AXIS.E],
            FIELD.TARGET.POS.MONITOR.X0: self.initial_task_cond["tmpos"][AXIS.X],
            FIELD.TARGET.POS.MONITOR.Y0: self.initial_task_cond["tmpos"][AXIS.Y],
            FIELD.TARGET.POS.MONITOR.D0: target_dist,
            FIELD.TARGET.POS.MONITOR.ID: np.log2(target_dist / self.initial_task_cond["trad"] + 1),
            FIELD.TARGET.ORBIT.A: self.initial_task_cond["torbit"][AXIS.A],
            FIELD.TARGET.ORBIT.E: self.initial_task_cond["torbit"][AXIS.E],
            FIELD.TARGET.DIRECTION: self.initial_task_cond["tdir"],

            FIELD.PLAYER.CAM.A0: self.initial_task_cond["cdir"][AXIS.A],
            FIELD.PLAYER.CAM.E0: self.initial_task_cond["cdir"][AXIS.E],
            FIELD.PLAYER.GAZE.X0: self.initial_task_cond[FIELD.PLAYER.GAZE.NAME][AXIS.X],
            FIELD.PLAYER.GAZE.Y0: self.initial_task_cond[FIELD.PLAYER.GAZE.NAME][AXIS.Y],
            FIELD.PLAYER.HEAD.X0: self.initial_task_cond[FIELD.PLAYER.HEAD.NAME][AXIS.X],
            FIELD.PLAYER.HEAD.Y0: self.initial_task_cond[FIELD.PLAYER.HEAD.NAME][AXIS.Y],
            FIELD.PLAYER.HEAD.Z0: self.initial_task_cond[FIELD.PLAYER.HEAD.NAME][AXIS.Z],

            # Model parameter
            **self.param,

            # Behavioral result
            METRIC.SUMMARY.TCT: self.result.time / 1000,     # ms -> s
            METRIC.SUMMARY.ACC: int(self.result.success),
            METRIC.SUMMARY.SHE: self.result.shoot_distance,
            METRIC.SUMMARY.NSE: self.result.shoot_distance / self.initial_task_cond["trad"],
            METRIC.SUMMARY.ENDPT.X: self.result.shoot_endpoint[AXIS.X],
            METRIC.SUMMARY.ENDPT.Y: self.result.shoot_endpoint[AXIS.Y],
            # METRIC.SUMMARY.ENDPT.NX: 0,
            # METRIC.SUMMARY.ENDPT.NY: 0,
            METRIC.SUMMARY.SCD: m_scd,
            METRIC.SUMMARY.CTL: np.sum(np.linalg.norm(np.diff(self.replay["camera_dir"]))),
            METRIC.SUMMARY.MRT: self.initial_task_cond[METRIC.SUMMARY.MRT] / 1000,   # ms -> s
            METRIC.SUMMARY.GRT: self.initial_task_cond[METRIC.SUMMARY.GRT] / 1000,   # ms -> s

            # Simulation information
            FIELD.ETC.TRUNC: int(self.result.truncated),    # bool -> binary
        }])


    def get_trajectory_result(self, downsample=None):
        timestamp = np.arange(self.replay["target_pos_monitor"].shape[0]) * self.intv.interp / 1000
        target_pos = self.replay["target_pos_monitor"]
        camera_dir = self.replay["camera_dir"]
        gaze_pos = self.gtraj_p

        if downsample is not None:
            assert type(downsample) is int
            timestamp_new = np.linspace(0, timestamp[-1], round(timestamp[-1] * downsample) + 1)
            target_pos = np_interp_nd(timestamp_new, timestamp, target_pos)
            camera_dir = np_interp_nd(timestamp_new, timestamp, camera_dir)
            gaze_pos = np_interp_nd(timestamp_new, timestamp, gaze_pos)
            timestamp = timestamp_new

        return pd.DataFrame({
            FIELD.ETC.TIMESTAMP: timestamp,
            FIELD.TARGET.POS.MONITOR.X: target_pos[:, AXIS.X],
            FIELD.TARGET.POS.MONITOR.Y: target_pos[:, AXIS.Y],
            FIELD.PLAYER.CAM.A: camera_dir[:, AXIS.A],
            FIELD.PLAYER.CAM.E: camera_dir[:, AXIS.E],
            FIELD.PLAYER.GAZE.X: gaze_pos[:, AXIS.X],
            FIELD.PLAYER.GAZE.Y: gaze_pos[:, AXIS.Y],
        })


    def display_replay(self):
        import matplotlib.pyplot as plt
        
        plt.scatter(*self.replay["target_pos_monitor"][0], s=11, color='r', marker='s')
        plt.scatter(*self.replay["target_pos_monitor"][:int(self.result.time)].T, s=0.5, color='k')

        plt.xlim(-0.26565, 0.26565)
        plt.ylim(-0.1494, 0.1494)
        plt.gca().set_aspect('equal')
        plt.show()

        # dist = np.linalg.norm(self.replay["target_pos_monitor"][:int(self.result.time)], axis=1)
        # plt.plot(dist)
        # plt.show()

    
    def render_episode(
        self,
        videowriter: cv2.VideoWriter,
        framerate:int=60,
        resolution:int=1080,
        hold_interval:int=36,   # Frame
        draw_mouse_pad:bool=True,
        draw_hud:bool=True,
        **kwargs
    ):
        # Interpolate to draw in given framerate
        timestamp_original = np.arange(self.replay["target_pos_monitor"].shape[0]) * self.intv.interp
        timestamp = np.linspace(0, self.result.time, int(round(self.result.time * framerate / 1000)) + 1)

        traj_m = dict(
            target = np_interp_nd(timestamp, timestamp_original, self.replay["target_pos_monitor"]),
            gaze = np_interp_nd(timestamp, timestamp_original, self.gtraj_p),
            camera = np_interp_nd(timestamp, timestamp_original, self.replay["camera_dir"]),
            hand = np_interp_nd(timestamp, timestamp_original, self.replay["hand_pos"]),
        )

        frame_num = timestamp.size

        # Mouse on desk surface
        (
            hpos_pixel, 
            mp_range_meter, 
            mp_range_pixel
        ) = self.__convert_hand_traj_meter_to_pixel(resolution, traj_m["hand"])

        for i in range(frame_num):
            scene = self.__draw_task_scene(
                resolution,
                traj_m["camera"][i],
                traj_m["target"][i],
                self.initial_task_cond["trad"],
                traj_m["gaze"][i]
            )
            if draw_mouse_pad:
                scene = self.__draw_mouse_trajectory(
                    scene,
                    hpos_pixel[:i+1], mp_range_pixel, mp_range_meter
                )
        
            if draw_hud:
                current_time = i / framerate if i < frame_num - 1 else timestamp[-1] / 1000
                scene = RENDER.draw_messages(scene,
                    [
                        [f'Model: {kwargs["model_name"]}', COLOR.WHITE],
                        [f'Time: {current_time:02.3f}', COLOR.WHITE],
                        [f'Episode: {kwargs["current_ep"]}/{kwargs["total_ep"]}', COLOR.WHITE],
                    ]
                )
                scene = RENDER.draw_messages(scene,
                    [
                        [f'size = {self.initial_task_cond["trad"]*2000:.2f} mm', COLOR.WHITE],
                        [f'speed = {self.initial_task_cond["tspd"]:.2f} deg/s', COLOR.WHITE],
                        [f'hand reaction = {self.initial_task_cond[METRIC.SUMMARY.MRT]} ms', COLOR.WHITE],
                        [f'gaze reaction = {self.initial_task_cond[METRIC.SUMMARY.GRT]} ms', COLOR.WHITE],
                    ], s=0.4, skip_space=6
                )

            videowriter.write(scene)
        
        # Hold
        if draw_hud:
            scene = RENDER.draw_messages(
                scene,
                [[f"Shot Hit ({self.result.shoot_distance*1000:.2f} mm)", COLOR.GREEN]] \
                    if self.result.success else [[f"Shot Miss ({self.result.shoot_distance*1000:.2f} mm)", COLOR.RED]],
                skip_space=3
            )
        for _ in range(hold_interval):
            videowriter.write(scene)


    def __draw_task_scene(self, resolution, camera_dir, target_pos, target_rad, gaze_pos, gaze_rad=0.002, crosshair_rad=0.0015):
        # HUD not included in this method.
        scene = RENDER.empty_scene(resolution, COLOR.BLACK)

        # Draw background
        bg_targets = np.array([
            Convert.game2monitor(np.zeros(3), camera_dir, bgt, self.fov, self.monitor_qt) for bgt in BACKGROUND_REFERENCE
        ])
        bg_targets = RENDER.convert_meter_to_pixel(bg_targets, resolution, self.monitor_qt)
        for bgt in bg_targets:
            if (bgt <= np.array(RENDER.img_resolution(resolution))).all() and (bgt >= np.zeros(2)).all():
                    scene = RENDER.draw_circle(scene, bgt, int(3*resolution/720), COLOR.GRAY, -1)


        # Maximum window width = Image width
        tp_pixel = RENDER.convert_meter_to_pixel(target_pos, resolution, self.monitor_qt)
        tr_pixel = max(1, int(round(target_rad / (2*self.monitor_qt[AXIS.Y]) * resolution)))
        gp_pixel = RENDER.convert_meter_to_pixel(gaze_pos, resolution, self.monitor_qt)
        gr_pixel = max(1, int(round(gaze_rad / (2*self.monitor_qt[AXIS.Y]) * resolution)))
        cp_pixel = RENDER.convert_meter_to_pixel(np.zeros(2), resolution, self.monitor_qt)
        cr_pixel = max(1, int(round(resolution * crosshair_rad / (2*self.monitor_qt[AXIS.Y]))))

        # Draw both objects
        scene = RENDER.draw_circle(scene, tp_pixel, tr_pixel, COLOR.WHITE)
        scene = RENDER.draw_circle(scene, gp_pixel, gr_pixel, COLOR.RED, thick=2)
        scene = RENDER.draw_circle(scene, cp_pixel, cr_pixel, COLOR.GREEN)
        scene = RENDER.draw_circle(scene, cp_pixel, cr_pixel, COLOR.BLACK, thick=1)

        # Flip horizontally (opencv uses flipped starting position in Y-axis)
        scene = cv2.flip(scene, 0)

        return scene


    def __convert_hand_traj_meter_to_pixel(self, resolution, traj, pad_expand=1.2, pad_size=0.35):
        translated_htj = traj - np.min(traj, axis=0)
        mp_range_meter = np.max(translated_htj) * pad_expand
        trans_to_center = np.max(translated_htj, axis=0) / 2 - np.array([mp_range_meter] * 2) / 2
        mp_range_pixel = int(resolution * pad_size)
        hpos_pixel = ((translated_htj - trans_to_center) * mp_range_pixel / mp_range_meter).astype(int)
        return hpos_pixel, mp_range_meter, mp_range_pixel


    def __draw_mouse_trajectory(self, scene, hpos_pixel, mp_range_pixel, mp_range_meter, pad_offset=0.1):
        res = scene.shape[0]
        scale_ratio = res / 720
        ofs = int(mp_range_pixel * pad_offset)
        pad_area = scene[res-ofs-mp_range_pixel:res-ofs, ofs:ofs+mp_range_pixel]
        pad_opac = np.ones(pad_area.shape, dtype=np.uint8) * 255
        pad_area = cv2.addWeighted(pad_area, 0.7, pad_opac, 0.3, 0)
        scene[res-ofs-mp_range_pixel:res-ofs, ofs:ofs+mp_range_pixel] = pad_area

        # Trajectory
        x = hpos_pixel[:,0] + ofs
        y = res - (hpos_pixel[:,1] + ofs)
        if len(hpos_pixel) >= 2:
            for i in range(len(hpos_pixel)-1):
                scene = cv2.line(scene, (x[i], y[i]), (x[i+1], y[i+1]), (0, 255, 255), 1)
        scene = cv2.circle(scene, (x[-1], y[-1]), int(3*scale_ratio), (0, 255, 255), 1)

        # Text
        scene = RENDER.put_text_at(
            scene, 
            f"Mouse Trajectory (Box scale: {mp_range_meter*100:2.2f}cm)", 
            (ofs, res - (mp_range_pixel + ofs + int(10 * scale_ratio))), 
            0.4 * scale_ratio, COLOR.WHITE
        )
        return scene
        


class AnSSimulator:
    def __init__(self, model_name: str, ckpt: Optional[int] | str = 'best', record_type=AnSEpisodeRecordDefault):

        # Policy file
        if ckpt == 'best':
            self.policy = os.path.join(
                Path(__file__).parent.parent.parent, 
                f"data/{FOLDER.DATA_RL_MODEL}/{model_name}/{FOLDER.MODEL_BEST}/best_model.zip"
            )
        elif type(ckpt) is int:
            self.policy = os.path.join(
                Path(__file__).parent.parent.parent, 
                f"data/{FOLDER.DATA_RL_MODEL}/{model_name}/{FOLDER.MODEL_CHECKPT}/rl_model_{ckpt}_steps.zip"
            )
        else:
            raise TypeError("Argument ckpt must be either int or 'best'.")
        
        self.model = SAC.load(self.policy)
        
        # Configurations
        env_config = load_config(os.path.join(
            Path(__file__).parent.parent.parent, 
            f"data/{FOLDER.DATA_RL_MODEL}/{model_name}/env_info.yaml"
        ))

        self.model_name = f"{model_name}_{ckpt}"

        env_class = getattr(ans_agent, env_config.env_class)
        env_intv = env_config.interval
        env_task = env_config.task_config
        env_user = env_config.user_config

        # Load environment
        self.env = env_class(
            agent_cfg=env_user,
            game_env_cfg=env_task,
            interval_cfg=env_intv,
        )

        # List of simulated result
        self.simulation = list()

        self.param_fixed = False
        self.record_type = record_type
    

    def modulated_param_list(self):
        param_list = self.env.user.param_modul.list
        param_range = Box({p: self.env.user.param_modul[p] for p in param_list})
        return param_list, param_range

    
    def fix_user_param(self, param_z=None, param_w=None):
        assert not (param_z is None and param_w is None), "No parameter information given."
        if param_z is not None:
            self.env.fix_z(param=param_z)
        elif param_w is not None:
            z = dict()
            for v, w in param_w.items():
                if self.env.user.param_modul[v].type == 'uniform':
                    z[v] = linear_normalize(w, self.env.user.param_modul[v].min, self.env.user.param_modul[v].max)
                elif self.env.user.param_modul[v].type == 'loguniform':
                    z[v] = log_normalize(w, self.env.user.param_modul[v].min, self.env.user.param_modul[v].max, scale=self.env.user.param_modul[v].scale)
            self.env.fix_z(param=z)
        self.param_fixed = True
    
    def unfix_user_param(self):
        self.env.unfix_z()
        self.param_fixed = False
    
    def set_ans_task(self, setting:dict):
        self.env.update_task_setting(ans_env_setting=setting)
    
    def set_user_stat(self, setting:dict):
        self.env.update_user_status(user_stat_setting=setting)

    def sample_user_stat_conditions(self, n, **conds):
        return [self.env.user_stat_sampler(**conds) for _ in range(n)]

    def clear_simulation(self):
        self.simulation = list()
    

    def simulate(
        self,
        param_list: Optional[List[dict]] = None,
        task_list: Optional[List[dict]] = None,
        user_stat_list: Optional[List[dict]] = None,
        num_simul: Optional[int] = None,
        overwrite_existing_simul: bool=False,
        resimulate_max_num: int=0,
        deterministic=False,
        verbose=True
    ):
        if param_list is not None and task_list is not None:
            assert len(param_list) == len(task_list), "The number of parameter and task set should be the same."
        
        assert not (param_list is None and task_list is None and user_stat_list is None and num_simul is None), \
            "Must specify the simulation condition."

        # Priority: param_list = task_list > num_simul
        if num_simul is None:
            if param_list is not None:
                num_simul = len(param_list)
            elif task_list is not None:
                num_simul = len(task_list)
            elif user_stat_list is not None:
                num_simul = len(user_stat_list)

        if self.param_fixed and param_list is not None:
            print("Warning - The user parameters are currently fixed, but they will be overwritten by input param list...")

        n_ep = 0
        n_resim = 0

        if param_list is not None: self.fix_user_param(param_list[n_ep])
        if task_list is not None: self.set_ans_task(task_list[n_ep])
        if user_stat_list is not None: self.set_user_stat(user_stat_list[n_ep])

        if overwrite_existing_simul:
            self.clear_simulation()
        
        progress = tqdm(total=num_simul, desc="Simulations: ") if verbose else None
        while True:
            obs, _ = self.env.reset()
            ep_rec = self.record_type(self.env)
            done = False
            truncated = False

            # Run 1 simulation
            while True:
                action, state = self.model.predict(obs, deterministic=deterministic)
                obs, rew, done, truncated, info = self.env.step(action)
                ep_rec.record(self.env, action, done, truncated, info)
                if done or truncated:
                    break
            
            if not truncated or (n_resim == resimulate_max_num):
                n_ep += 1
                n_resim = 0
                self.simulation.append(ep_rec)
                if n_ep == num_simul:
                    break
                if param_list is not None: self.fix_user_param(param_list[n_ep])
                if task_list is not None: self.set_ans_task(task_list[n_ep])
                if user_stat_list is not None: self.set_user_stat(user_stat_list[n_ep])
                if verbose: progress.update(1)
            else:
                n_resim += 1
        if verbose: progress.update(1)
    

    def get_simulation_result(self, return_traj=False, downsample=None):
        summ, traj = list(), list()
        for rec in self.simulation:
            summ.append(rec.get_summary_result())
            if return_traj:
                traj.append(rec.get_trajectory_result(downsample=downsample))
        summ = pd.concat(summ)
        summ.reset_index()
        
        if return_traj:
            return summ, traj
        return summ
    

    def extract_task_and_user_stat(self, summary_data:pd.DataFrame):
        task, ustat = list(), list()
        for _, row in summary_data.iterrows():
            task.append(dict(
                cdir = np.array([row[FIELD.PLAYER.CAM.A0], row[FIELD.PLAYER.CAM.E0]]),
                tmpos = np.array([row[FIELD.TARGET.POS.MONITOR.X0], 
                                  row[FIELD.TARGET.POS.MONITOR.Y0]]),
                torbit = np.array([row[FIELD.TARGET.ORBIT.A], row[FIELD.TARGET.ORBIT.E]]),
                tspd = row[FIELD.TARGET.SPD],
                trad = row[FIELD.TARGET.RAD],
            ))
            ustat.append({
                METRIC.SUMMARY.MRT: round(row[METRIC.SUMMARY.MRT] * 1000),
                METRIC.SUMMARY.GRT: round(row[METRIC.SUMMARY.GRT] * 1000),
                FIELD.PLAYER.HEAD.NAME: np.array([row[FIELD.PLAYER.HEAD.X0], 
                                                  row[FIELD.PLAYER.HEAD.Y0],
                                                  row[FIELD.PLAYER.HEAD.Z0]]),
                FIELD.PLAYER.GAZE.NAME: np.array([row[FIELD.PLAYER.GAZE.X0],
                                                  row[FIELD.PLAYER.GAZE.Y0]])
            })
        return task, ustat


    def export_simulation_video(
        self,
        framerate: int = 60,
        resolution: int = 720,
        interval: Union[float, List[float]] = 0.6,    # sec
        video_path: Optional[str] = None,
        video_name: Optional[str] = None,
        warning_threshold: int = 100,
        verbose=True
    ):
        framerate = min(300, framerate)

        num_simul = len(self.simulation)
        if num_simul > warning_threshold:
            print(f"Warning: there are {num_simul} episode saved.")
            print("Are you sure to render all of these?")
            ans = input("Type y to proceed: ")
            if ans != 'y' and ans != 'Y':
                return
        
        # Interval between episodes (frame)
        if type(interval) is list:
            interval = (np.array(interval) * framerate).astype(int)
            assert num_simul == interval.size
        else:
            interval = [int(interval * framerate)] * num_simul

        if video_path is None:
            video_path = os.path.join(
                Path(__file__).parent.parent.parent, 
                f"data/{FOLDER.VISUALIZATION}/{FOLDER.VIDEO}/{self.model_name}"
            )
        os.makedirs(video_path, exist_ok=True)

        if video_name is None:
            video_name = get_timebase_session_name()

        video = cv2.VideoWriter(
            f"{video_path}/{video_name}.mp4",
            VIDEO.CODEC,
            framerate, 
            RENDER.img_resolution(resolution)
        )
    
        print(f"Rendering {num_simul} simulation(s) ...")

        for episode, rec in enumerate(tqdm(self.simulation)) if verbose else enumerate(self.simulation):
            rec.render_episode(
                video,
                framerate,
                resolution,
                interval[episode],    # Number of frame
                model_name=self.model_name,
                current_ep=episode+1,
                total_ep=len(self.simulation),
                
            )
        video.release()



if __name__ == "__main__":
    # Sample simulation
    simulator = AnSSimulator("default", 20000000)
    simulator.simulate(num_simul=10, verbose=True)
    simulator.export_simulation_video()
