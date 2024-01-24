"""
Experiment data class

Code written by June-Seop Yoon
"""

import sys, os, pickle
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pymovements as pm

from typing import List, Union, Optional

sys.path.append("..")
from configs.common import *
from configs.experiment import *
from configs.path import *
from utils.mymath import *
from utils.dataloader import *
from utils.plots import *
from utils.render import *
from utils.utils import now_to_string
from agent.fps_task import *

class ExperimentData:
    def __init__(self, player, mode, verbose=False):
        if verbose: print(f"======== User ID: {player} ({mode}) ========")
        self.player = player
        self.mode = mode
        
        self.tri = pd.read_csv(PATH_DATA_EXP_DB_TRI % (self.player, self.player, self.mode), low_memory=False)
        user_info = pd.read_csv(PATH_DATA_EXP_DB_INFO % (self.player, self.player, self.mode))
        self.sensitivity = user_info["sensitivity"][0]
        self.mouse_dpi = user_info["mouse_dpi"][0]

    
    def export_to_csv(self, plot_mseq=False):
        base = dict(
            tier = [],
            player = [],
            mode = [],
            sensitivity = [],
            mouse_dpi = [],
            session_id = [],
            block_index = [],
            trial_index = [],
            target_speed = [],
            target_radius = [],
            target_color = [],

            tgpos0_az = [],
            tgpos0_el = [],
            tmpos0_x = [],
            tmpos0_y = [],
            tmpos0_dist = [],
            t_iod = [],
            pcam0_az = [],
            pcam0_el = [],
            gaze0_x = [],
            gaze0_y = [],
            gaze0_dist = [],
            eye_x = [],
            eye_y = [],
            eye_z = [],
            toax_az = [],
            toax_el = [],

            # invalid_tct = [],
            not_enough_gaze = [],
            init_gaze_faraway = [],
            # shoot_error_large = [],
        )

        data1 = dict(
            hand_reaction = [],
            gaze_reaction = [],
            result = [],
            trial_completion_time = [],
            endpoint_x = [],
            endpoint_y = [],
            endpoint_x_norm = [],
            endpoint_y_norm = [],
            shoot_error = [],
            shoot_error_normalized = [],
            cam_traverse_length = [],
            cam_geometric_entropy = [],
            cam_max_speed = [],
            
            glancing_distance = []
        )

        data2 = dict(
            saccade_reaction = [],
            saccade_amplitude = [],
            saccade_duration = [],
            saccade_peak_velocity = []
        )

        data1 = copy.deepcopy({**base, **data1})
        data2 = copy.deepcopy({**base, **data2})   # Gaze metric included here (main sequence)

        tier = parse_tier_by_playername(self.player)

        for sn in SES_NAME_ABBR:
            for bn in range(2):
                for tn in range(TRIAL_NUM_PER_BLOCK):
                    if self.check_trial_exist(sn, bn, tn):
                        # Task information
                        data1["tier"].append(tier)
                        data1["player"].append(self.player)
                        data1["mode"].append(self.mode)
                        data1["sensitivity"].append(self.sensitivity)
                        data1["mouse_dpi"].append(self.mouse_dpi)
                        data1["session_id"].append(sn)
                        data1["block_index"].append(bn)
                        data1["trial_index"].append(tn)

                        # Task conditions
                        data1["target_speed"].append(tgspd_from_sess(sn))
                        data1["target_radius"].append(trad_from_sess(sn))
                        data1["target_color"].append(tcolor_from_sess(sn))

                        tgpos0, tmpos0 = self.target_initial_position(sn, bn, tn)
                        data1["tgpos0_az"].append(tgpos0[X])
                        data1["tgpos0_el"].append(tgpos0[Y])
                        data1["tmpos0_x"].append(tmpos0[X])
                        data1["tmpos0_y"].append(tmpos0[Y])
                        data1["tmpos0_dist"].append(np.linalg.norm(tmpos0))
                        data1["t_iod"].append(np.log2(1 + np.linalg.norm(tmpos0) / (2*trad_from_sess(sn))))
                        
                        toax = self.get_target_orbit_axis(sn, bn, tn)
                        data1["toax_az"].append(toax[0])
                        data1["toax_el"].append(toax[1])

                        pcam0 = self.camera_initial_view(sn, bn, tn)
                        data1["pcam0_az"].append(pcam0[X])
                        data1["pcam0_el"].append(pcam0[Y])
                        g0, eye_pos = self.gaze_initial_position(sn, bn, tn)
                        data1["gaze0_x"].append(g0[X])
                        data1["gaze0_y"].append(g0[Y])
                        data1["gaze0_dist"].append(np.linalg.norm(g0))
                        data1["eye_x"].append(eye_pos[X])
                        data1["eye_y"].append(eye_pos[Y])
                        data1["eye_z"].append(eye_pos[Z])

                        data1[M_ACC].append(self.trial_result(sn, bn, tn))
                        tct = self.trial_completion_time(sn, bn, tn)
                        data1[M_TCT].append(tct)

                        endpoint, endpoint_norm, shoot_error = self.trial_shoot_error(sn, bn, tn)
                        data1["endpoint_x"].append(endpoint[X])
                        data1["endpoint_y"].append(endpoint[Y])
                        data1["endpoint_x_norm"].append(endpoint_norm[X])
                        data1["endpoint_y_norm"].append(endpoint_norm[Y])
                        data1[M_SE].append(shoot_error)
                        data1[M_SE_NORM].append(shoot_error / trad_from_sess(sn))

                        hrt, *_ = self.trial_hand_metrics(sn, bn, tn)
                        data1["hand_reaction"].append(hrt)

                        cam_ge, cam_max_spd, cam_trav_len = self.trial_cam_metrics(sn, bn, tn)
                        data1[M_CTL].append(cam_trav_len)
                        data1[M_CGE].append(cam_ge)
                        data1[M_CSM].append(cam_max_spd)

                        grt, ms_amp, ms_dur, ms_pvel = self.parse_saccades(sn, bn, tn, plot=plot_mseq)
                        gd = self.trial_glancing_distance(sn, bn, tn)
                        data1["gaze_reaction"].append(grt)
                        data1[M_GD].append(gd)

                        # flag_tct = int(allowed_tct[0] > tct or allowed_tct[1] < tct)
                        flag_gazef = int(not self.check_gaze_validity(sn, bn, tn))
                        flag_gazep = int(np.linalg.norm(g0) > VALID_INITIAL_GAZE_RADIUS)
                        # flag_sef = int(shoot_error > VALID_SHOOT_ERROR)

                        # data1["invalid_tct"].append(flag_tct)
                        data1["not_enough_gaze"].append(flag_gazef)
                        data1["init_gaze_faraway"].append(flag_gazep)
                        # data1["shoot_error_large"].append(flag_sef)

                        # Main sequence
                        if len(ms_amp) > 0:
                            ms_grt = [grt] + [np.nan] * (len(ms_amp) - 1)
                            for msr, msa, msd, msp in zip(ms_grt, ms_amp, ms_dur, ms_pvel):
                                # Task information
                                data2["tier"].append(tier)
                                data2["player"].append(self.player)
                                data2["mode"].append(self.mode)
                                data2["sensitivity"].append(self.sensitivity)
                                data2["mouse_dpi"].append(self.mouse_dpi)
                                data2["session_id"].append(sn)
                                data2["block_index"].append(bn)
                                data2["trial_index"].append(tn)

                                # Task conditions
                                data2["target_speed"].append(tgspd_from_sess(sn))
                                data2["target_radius"].append(trad_from_sess(sn))
                                data2["target_color"].append(tcolor_from_sess(sn))

                                data2["tgpos0_az"].append(tgpos0[X])
                                data2["tgpos0_el"].append(tgpos0[Y])
                                data2["tmpos0_x"].append(tmpos0[X])
                                data2["tmpos0_y"].append(tmpos0[Y])
                                data2["tmpos0_dist"].append(np.linalg.norm(tmpos0))
                                data2["t_iod"].append(np.log2(1 + np.linalg.norm(tmpos0) / (2*trad_from_sess(sn))))
                                
                                data2["toax_az"].append(toax[0])
                                data2["toax_el"].append(toax[1])

                                data2["pcam0_az"].append(pcam0[X])
                                data2["pcam0_el"].append(pcam0[Y])
                                data2["gaze0_x"].append(g0[X])
                                data2["gaze0_y"].append(g0[Y])
                                data2["gaze0_dist"].append(np.linalg.norm(g0))
                                data2["eye_x"].append(eye_pos[X])
                                data2["eye_y"].append(eye_pos[Y])
                                data2["eye_z"].append(eye_pos[Z])

                                data2["saccade_reaction"].append(msr)
                                data2["saccade_amplitude"].append(msa)
                                data2["saccade_duration"].append(msd)
                                data2["saccade_peak_velocity"].append(msp)
                                
                                # data2["invalid_tct"].append(flag_tct)
                                data2["not_enough_gaze"].append(flag_gazef)
                                data2["init_gaze_faraway"].append(flag_gazep)
                                # data2["shoot_error_large"].append(flag_sef)

        data1 = pd.DataFrame(data1)
        data1.to_csv(PATH_DATA_SUMMARY_PLAYER % (self.player, self.mode), index=False)

        data2 = pd.DataFrame(data2)
        data2.to_csv(PATH_DATA_SUMMARY_PLAYER % (self.player, f"{self.mode}-mseq"), index=False)


    def export_replay(
        self,
        list_of_trials: Optional[dict],
        framerate: int = 60,
        res: int = 720,
        interval: Union[float, List[float]] = 0.6,    # sec
        path: Optional[str] = None,
        videoname: Optional[str] = None,
        draw_gaze: bool = True
    ):
        assert (
            "sn" in list_of_trials.keys() and
            "bn" in list_of_trials.keys() and
            "tn" in list_of_trials.keys()
        )

        if framerate > 500:
            print("Exceeded maximum frame rate - 500 fps")
            return
        
        if path is None:
            path = PATH_VIS_VIDEO % self.player
        os.makedirs(path, exist_ok=True)

        num_simul = len(list_of_trials["sn"])

        # Interval between episodes (frame)
        if type(interval) is list:
            interval = (np.array(interval) * framerate).astype(int)
            assert num_simul == interval.size
        else:
            interval = [int(interval * framerate)] * num_simul

        if videoname is None: videoname = now_to_string()

        video = cv2.VideoWriter(
            f"{path}{videoname}.mp4",
            CODEC,
            framerate, img_resolution(res)
        )

        print(f"Rendering {self.player}'s replay ...")

        for i, (sn, bn, tn) in enumerate(zip(
            list_of_trials["sn"],
            list_of_trials["bn"],
            list_of_trials["tn"]
        )):
            self.render_trial(
                sn, bn, tn,
                video,
                framerate,
                res,
                interval[i],
                draw_gaze=draw_gaze
            )
        video.release()


    def render_trial(
        self,
        sn, bn, tn,
        videowriter:int,
        framerate:int,
        res:int,
        interval:int,   # Number of frames to stop
        draw_gaze:bool=True,
        anonymous:bool=True
    ):
        trad = trad_from_sess(sn)
        tcol = tcolor_from_sess(sn)
        tct = self.trial_completion_time(sn, bn, tn)
        hit = self.trial_result(sn, bn, tn)

        (
            timestamp, 
            _tpos, 
            _gpos, _, _, _,
            _hpos,
            _pcam,
            tgpos0,
            toax
        ) = self.trial_trajectory(sn, bn, tn)

        timestamp_frame = np.linspace(
            0, timestamp[-1],
            int_floor(timestamp[-1] * framerate) + 1
        )

        # Interp plan
        tpos = np_interp_nd(timestamp_frame, timestamp, _tpos)
        hpos = np_interp_nd(timestamp_frame, timestamp, _hpos)
        gpos = np_interp_nd(timestamp_frame, timestamp, _gpos)
        pcam = np_interp_nd(timestamp_frame, timestamp, _pcam)

        (
            hpos_pixel, 
            mp_range_meter, 
            mp_range_pixel
        ) = convert_hand_traj_meter_to_pixel(hpos, res)

        game = GameState(sensi=self.sensitivity*1000) # This will be used only for background rendering
        game.reset(pcam=pcam[0], gpos=_gpos[0])

        frame_num = timestamp_frame.size
        for i in range(frame_num):
            game.set_cam_angle(pcam[i])
            game.fixate(gpos[i])

            # Game
            scene = draw_game_scene(
                res,
                game.pcam,
                tpos[i],
                trad,
                gpos[i],
                draw_gaze=draw_gaze, 
                gray_target=(tcol==COLOR[1])
            )
            # Mouse
            scene = draw_mouse_trajectory(
                scene,
                mp_range_pixel,
                mp_range_meter,
                hpos_pixel[:i+1]
            )

            # Text information
            current_time = i / framerate if i < frame_num - 1 else tct
            if not anonymous:
                player_text = [f'Player: {self.player} ({self.mode})', C_WHITE]
            else:
                tier = parse_tier_by_playername(self.player)
                player_text = [f'Player: {tier}{PLAYER[tier].index(self.player):02d} ({self.mode})', C_WHITE]
            scene = draw_messages(scene,
                [
                    player_text,
                    [f'Time: {current_time:02.3f}', C_WHITE],
                    [f'Target: {ses_name_abbr_to_full(sn)}', C_WHITE],
                    [f'Block {bn}, Trial {tn}', C_WHITE],
                    [f'Mouse Sensitivity: {self.sensitivity:.2f} deg/mm', C_WHITE]
                ]
            )
            videowriter.write(scene)
        
        # Paused part
        if hit == 1:
            scene = draw_messages(scene, [["Shot Hit", C_GREEN]], skip_row=5)
        else:
            scene = draw_messages(scene, [["Shot Miss", C_RED]], skip_row=5)
        for _ in range(interval): videowriter.write(scene)


    def get_list_of_trial(
        self,
        session_ids: Union[List[str], str],
        block_indices: Union[List[int], int],
        include_valid_only: bool = True
    ):
        if type(session_ids) is str:
            session_ids = [session_ids]
        if type(block_indices) is int:
            block_indices = [block_indices]

        lot = dict(
            sn = [],
            bn = [],
            tn = []
        )
        
        for sn in session_ids:
            for bn in block_indices:
                for tn in range(TRIAL_NUM_PER_BLOCK):
                    if self.check_trial_exist(sn, bn, tn) and (
                        not include_valid_only or (
                            include_valid_only and
                            tn >= LEARNING_EFFECT_IGNORANCE and
                            self.check_gaze_validity(sn, bn, tn)
                        )
                    ):
                        lot["sn"].append(sn)
                        lot["bn"].append(bn)
                        lot["tn"].append(tn)
        
        return lot

    
    def check_gaze_validity(self, sn, bn, tn, threshold=MINIMUM_VALID_RATIO_OF_GAZE):
        if not self.check_trial_exist(sn, bn, tn):
            return False
        *_, v, _, _, _, _ = self.trial_trajectory(sn, bn, tn)
        return len(v) > 0


    def check_trial_exist(self, sn, bn, tn):
        return len(self.tri[
            (self.tri.session_id == sn) &
            (self.tri.block_index == bn) &
            (self.tri.trial_index == tn)
        ]) > 0


    def trial_completion_time(self, sn, bn, tn):
        return self.tri[
            (self.tri["session_id"] == sn) &
            (self.tri["block_index"] == bn) &
            (self.tri["trial_index"] == tn)
        ]["task_execution_time"].to_numpy()[0]


    def trial_result(self, sn, bn, tn):
        return self.tri[
            (self.tri["session_id"] == sn) &
            (self.tri["block_index"] == bn) &
            (self.tri["trial_index"] == tn)
        ]["destroyed_targets"].to_numpy()[0]
    

    def trial_shoot_error(self, sn, bn, tn):
        _, ttj, *_, ptj, tgpos0, _ = self.trial_trajectory(sn, bn, tn)
        endpoint = ttj[-1]

        # Normalize endpoint
        tdir_traj = np.array([
            target_monitor_position(np.zeros(3), _ptj, sphr2cart(*ptj[-1])) for _ptj in ptj[-12:]
        ])
        tdir = np.mean(np.diff(tdir_traj, axis=0), axis=0)
        norm_den = np.linalg.norm(tdir)
        if norm_den > 0: tdir /= norm_den
        else: tdir = np.array([1, 0])
        tppd = rotate_point_2d(tdir, 90)

        endpoint_norm = np.linalg.inv(np.array([tdir, tppd]).T) @ endpoint

        return endpoint, endpoint_norm, np.linalg.norm(endpoint - CROSSHAIR)


    def trial_glancing_distance(self, sn, bn, tn,):
        _, _, gtj, gtj_a, eye_pos, gaze_flag, *_ = self.trial_trajectory(sn, bn, tn)
        if len(gtj_a) == 0: return np.nan

        gdist = np.linalg.norm(gtj - CROSSHAIR, axis=1)
        furthest_g = gtj[np.argmax(gdist)]
        glancing_dist = angle_between(
            np.array([*furthest_g, 0]) - eye_pos,
            np.array([*CROSSHAIR, 0]) - eye_pos,
            return_in_degree=True
        )
        
        return glancing_dist
    


    def parse_saccades(
        self, sn, bn, tn, plot=False,
        amp_th=1,
        pvel_th=38,
        react_th=0.1,
    ):
        data = self.trial_gaze_profile(sn, bn, tn)
        if data is None: return np.nan, [], [], []

        # First saccade's reaction time
        # Saccade amplitude / duration / peak velocity
        (ts, gmpos, gapos, gvel, ep, gaze_flag) = data
        # gspd = []
        # for t in range(1, len(ts)):
        #     gspd.append(angle_btw(
        #         np.array([*gmpos[t], 0]) - ep[t],
        #         np.array([*gmpos[t-1], 0]) - ep[t-1]
        #     ) / (ts[t] - ts[t-1]))
        # gspd = np.insert(gspd, 0, gspd[0])
        gspd = np.linalg.norm(gvel, axis=1)

        amp_list, dur_list, pvel_list = [], [], []
        react_time = np.nan

        try:
            events = pm.events.microsaccades(
                positions=gapos,
                velocities=gvel,
                # timesteps=ts,
                minimum_duration=6,
                threshold_factor=3.5,
                threshold='engbert2015'
            )
        except:
            return np.nan, [], [], []
        
        sacc_st = events.frame["onset"].to_numpy()
        sacc_et = events.frame["offset"].to_numpy()

        sacc_indices = []

        for ss, se in zip(sacc_st, sacc_et):
            amp = angle_between(
                np.array([*gmpos[ss], 0]) - ep, 
                np.array([*gmpos[se], 0]) - ep
            )
            # amp = np.linalg.norm(np.max(gmpos[ss:se+1], axis=0) - np.min(gmpos[ss:se+1], axis=0))
            dur = ts[se] - ts[ss]
            pvel = np.max(gspd[ss:se+1])

            if pvel >= pvel_th and amp >= amp_th and ts[se] >= react_th:
                amp_list.append(amp)
                dur_list.append(dur)
                pvel_list.append(pvel)
                if np.isnan(react_time): react_time = ts[ss]

                sacc_indices.append((ss, se))


        if plot:
            fig, axs = plt.subplots(2, 1, figsize=(8, 8.5), gridspec_kw={"height_ratios":[1, 1]})

            gdist = np.linalg.norm(gapos - gapos[0], axis=1)
            axs[0].plot(ts, gdist, linewidth=0.8, color='k', zorder=9)
            for (ss, se) in sacc_indices:
                axs[0].scatter(ts[ss], gdist[ss], marker='s', s=25, color='r', zorder=11)
                axs[0].scatter(ts[se], gdist[se], marker='s', s=25, color='b', zorder=11)
            axs[0].set_xlabel("Time (ms)")
            axs[0].set_ylabel("Deviation (deg)")
            set_tick_and_range(axs[0], 0.1, 1000, ts[-1], axis='x')
            set_tick_and_range(axs[0], 1, 1, np.max(gdist), axis='y')
            axs[0].grid(True, zorder=0)

            axs[1].plot(ts, gspd, linewidth=0.8, color='k', zorder=9)
            for (ss, se) in sacc_indices:
                axs[1].scatter(ts[ss], gspd[ss], marker='s', s=25, color='r', zorder=11)
                axs[1].scatter(ts[se], gspd[se], marker='s', s=25, color='b', zorder=11)
            axs[1].set_xlabel("Time (ms)")
            axs[1].set_ylabel("Speed (deg/s)")
            set_tick_and_range(axs[1], 0.1, 1000, ts[-1], axis='x')
            set_tick_and_range(axs[1], 50, 1, np.max(gspd), axis='y')
            axs[1].grid(True, zorder=0)

            fig_save(PATH_VIS_EXP_GAZE_PROF % self.player, f"{self.mode}-{sn}-{bn}-{tn:02d}")
        
        return react_time, amp_list, dur_list, pvel_list



    def trial_gaze_profile(self, sn, bn, tn):
        """Explicitly set (sn, bn, tn) as valid trials only"""
        ts, _, gtj_m, gtj_a, eye_pos, gaze_flag, *_ = self.trial_trajectory(sn, bn, tn)
        if len(gtj_m) == 0: return None

        try:
            gvel = pm.gaze.transforms.pos2vel(
                gtj_a,
                sampling_rate=240,
                method="smooth"
            )
        except:
            return None

        return ts, gtj_m, gtj_a, gvel, eye_pos, gaze_flag

    
    def trial_hand_metrics(self, sn, bn, tn, min_rt=0.1, max_rt=0.3, size=13, sigma=3, threshold=0.15, plot=False):
        # Hand reaction time & hand geometric entropy
        ts, hs, htj = self.trial_hand_profile(sn, bn, tn, size=size, sigma=sigma)
        hand_ge = geometric_entropy(htj)

        if ts[-1] <= min_rt:
            return np.nan, np.nan, np.nan

        # hs_target = hs[(ts >= min_rt) & (ts <= max_rt)]
        # move_threshold = np.max(hs_target) * threshold
        # if np.all(hs_target > move_threshold):
        #     react_time = min_rt
        # else:
        #     speed_reached = intersections(
        #         hs_target, move_threshold, consider_sign=False
        #     )
        #     if len(speed_reached) == 0:
        #         react_time = max_rt
        #     else:
        #         react_time = ts[speed_reached[0] + len(ts[ts <= min_rt])]

        hs_gf = apply_gaussian_filter(hs, size=25)
        ha = derivative_central(ts, hs_gf)
        convexness = derivative_central(ts, ha)
        hs0 = np.interp([min_rt], ts, hs_gf)[0]
        hsn = np.max(hs_gf[(ts >= min_rt) & (ts <= max_rt)])
        peak_moment = ts[np.argmax(hs_gf[(ts >= min_rt) & (ts <= max_rt)]) + len(ts[ts < min_rt])]
        react_acc_th = (hsn - hs0) / (peak_moment - min_rt)

        accelerated = intersections(
            ha[(ts >= min_rt) & (ts <= max_rt)], 
            react_acc_th, 
            consider_sign=False
        )
        react_time = None
        if len(accelerated) == 1:
            react_time = ts[accelerated[0] + len(ts[ts < min_rt])]
        elif len(accelerated) > 1:
            for acc in accelerated:
                if convexness[acc + len(ts[ts < min_rt])] >= 0:
                    react_time = ts[acc + len(ts[ts < min_rt])]
                    break
        else:
            react_time = min_rt

        if react_time is None:
            react_time = min_rt

        if plot:
            fig, axs = plt.subplots(1, 1, figsize=(8, 4))

            axs.plot(ts, hs, color='r', linewidth=0.8, zorder=9)
            axs.plot(ts, hs_gf, color='y', linewidth=0.7, zorder=8)
            axs.set_xlabel("Time (ms)")
            axs.set_ylabel("Velocity (cm/s)")
            set_tick_and_range(axs, 0.1, 1000, ts[-1], axis='x')
            set_tick_and_range(axs, 0.02, 100, np.max(hs), axis='y')
            # axs.axvline(react_time, linestyle='--', linewidth=1, color='b', zorder=10)
            # axs.axvline(0.3, linestyle='--', linewidth=1, color='b', zorder=10)
            axs.grid(True, zorder=0)

            fig_save(PATH_VIS_EXP_HAND_PROF % self.player, f"{self.mode}-{sn}-{bn}-{tn:02d}")
        
        return react_time, hand_ge, np.max(hs)


    def trial_hand_profile(self, sn, bn, tn, plot=False, size=25, sigma=3):
        ts, _, _, _, _, _, htj, *_ = self.trial_trajectory(sn, bn, tn)
        dt = ts[1:] - ts[:-1]
        dp = apply_gaussian_filter(np.linalg.norm(htj[1:] - htj[:-1], axis=1), size=size, sigma=sigma)
        hs = dp / dt
        hs = np.pad(hs, (1, 0), mode='edge')

        if plot:
            fig, axs = plt.subplots(1, 1, figsize=(8, 4))

            axs.plot(ts, hs, color='r', linewidth=0.8, zorder=9)
            axs.plot(ts, apply_gaussian_filter(hs), color='y', linewidth=0.7, zorder=8)
            axs.set_xlabel("Time (ms)")
            axs.set_ylabel("Velocity (cm/s)")
            set_tick_and_range(axs, 0.1, 1000, ts[-1], axis='x')
            set_tick_and_range(axs, 0.02, 100, np.max(hs), axis='y')
            axs.axvline(0.1, linestyle='--', linewidth=1, color='b', zorder=10)
            axs.axvline(0.3, linestyle='--', linewidth=1, color='b', zorder=10)
            axs.grid(True, zorder=0)

            fig_save(PATH_VIS_EXP_HAND_PROF % self.player, f"{self.mode}-{sn}-{bn}-{tn:02d}")

        return ts, hs, htj

    
    def trial_cam_profile(self, sn, bn, tn, plot=False, size=25, sigma=3):
        ts, *_, ctj, _, _ = self.trial_trajectory(sn, bn, tn)
        dt = ts[1:] - ts[:-1]
        dp = apply_gaussian_filter(np.linalg.norm(ctj[1:] - ctj[:-1], axis=1), size=size, sigma=sigma)
        cs = dp / dt
        cs = np.pad(cs, (1, 0), mode='edge')

        if plot:
            fig, axs = plt.subplots(1, 1, figsize=(8, 4))

            axs.plot(ts, cs, color='r', linewidth=0.8, zorder=9)
            axs.set_xlabel("Time (ms)")
            axs.set_ylabel(f"Velocity ($^\circ$/s)")
            set_tick_and_range(axs, 0.1, 1000, ts[-1], axis='x')
            set_tick_and_range(axs, 10, 1, np.max(cs), axis='y')
            axs.axvline(0.1, linestyle='--', linewidth=1, color='b', zorder=10)
            axs.axvline(0.3, linestyle='--', linewidth=1, color='b', zorder=10)
            axs.grid(True, zorder=0)

            fig_save(PATH_VIS_EXP_CAM_PROF % self.player, f"{self.mode}-{sn}-{bn}-{tn:02d}")

        return ts, cs, ctj


    def trial_cam_metrics(self, sn, bn, tn, plot=False, size=25, sigma=3):
        ts, cs, ctj = self.trial_cam_profile(sn, bn, tn, plot=plot, size=size, sigma=sigma)

        cam_ge = geometric_entropy(ctj)
        max_cs = np.max(cs)
        ctl = np.sum(np.linalg.norm(np.diff(ctj, axis=0), axis=1))

        return cam_ge, max_cs, ctl
    

    def trial_efficiency_of_aim(self, sn, bn, tn):
        pass


    def trial_distance(self, sn, bn, tn):
        (
            target_ts, 
            target_traj,
            gaze_traj_m,
            *_
        ) = self.trial_trajectory(sn, bn, tn)
        dist_tx = np.linalg.norm(target_traj - CROSSHAIR, axis=1)
        dist_gx = np.linalg.norm(gaze_traj_m - gaze_traj_m[0], axis=1)
        return target_ts, dist_tx, dist_gx

    
    def collect_trial_distance(self, trial_list, mean_tct):
        ts, tx, gx, mtx, mgx = [], [], [], [], []
        std_ts = np.linspace(0, mean_tct, int(mean_tct*240))

        for sn, bn, tn in zip(trial_list["session_id"], trial_list["block_index"], trial_list["trial_index"]):
            _ts, _tx, _gx = self.trial_distance(sn, bn, tn)
            ts.append(_ts)
            tx.append(
                sp.interpolate.interp1d(
                    _ts, _tx,
                    kind='quadratic',
                    bounds_error=False, 
                    fill_value=(_tx[0], _tx[-1])
                )(std_ts[std_ts <= _ts[-1]])
            )
            gx.append(
                sp.interpolate.interp1d(
                    _ts, _gx,
                    kind='quadratic',
                    bounds_error=False, 
                    fill_value=(_gx[0], _gx[-1])
                )(std_ts[std_ts <= _ts[-1]])
            )
        
        mtx = mean_unbalanced_data(tx, min_valid_n=1)
        mgx = mean_unbalanced_data(gx, min_valid_n=1)

        return ts, tx, gx, std_ts, mtx, mgx
        

    

    def trial_trajectory(self, sn, bn, tn, plot=False):
        filename = f"{self.mode}-{sn}-{bn}-{tn:02d}.pkl"
        (
            target_ts, 
            target_traj,
            gaze_traj_m,
            gaze_traj_a,
            eye_pos,
            gaze_flag,
            hand_traj,
            pcam_traj,
            tgpos0,
            toax
        ) = pickle_load(PATH_DATA_EXP_TRAJ_PKL % self.player + filename)
            
        if plot:
            plt_trajectory(
                target_ts, target_traj, gaze_traj_m, gaze_flag, trad_from_sess(sn),
                self.player, f"{self.mode}-{sn}-{bn}-{tn:02d}"
            )

        return (
            target_ts, 
            target_traj,
            gaze_traj_m,
            gaze_traj_a,
            eye_pos,
            gaze_flag,
            hand_traj,
            pcam_traj,
            tgpos0,
            toax
        )

    
    def camera_initial_view(self, sn, bn, tn):
        *_, pcam, _, _ = self.trial_trajectory(sn, bn, tn)
        return pcam[0]

    
    def gaze_initial_position(self, sn, bn, tn):
        _, _, gtj, _, eye_pos, *_ = self.trial_trajectory(sn, bn, tn)
        if len(gtj) > 0:
            return gtj[0], eye_pos
        return [np.nan, np.nan], [np.nan, np.nan, np.nan]
    

    def target_initial_position(self, sn, bn, tn):
        *_, pcam, tgpos, _ = self.trial_trajectory(sn, bn, tn)
        tmpos = target_monitor_position(np.zeros(3), pcam[0], sphr2cart(*tgpos))
        return tgpos, tmpos


    def get_target_orbit_axis(self, sn, bn, tn):
        *_, toax = self.trial_trajectory(sn, bn, tn)
        return toax





if __name__ == "__main__":
    d = ExperimentData('Tigrise', 'custom')
    # t, s = d.trial_hand_speed_profile('ssw', 0, 5)
    # plt.plot(t, s)
    # plt.show()
    d.export_replay(
        d.get_list_of_trial(
            "fsw", 0
        ),
        framerate=144,
        res=1080,
        videoname='fsw'
    )