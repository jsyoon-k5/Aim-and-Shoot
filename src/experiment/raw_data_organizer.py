"""
The purpose of this code is to merge FPSci database and
eyetracker, mouse input log with some sync. process.

Since each table uses slightly different timeline,
this code mainly focuses on selecting proper time frame for requested info.

Code written by June-Seop Yoon
"""

import sys, os, sqlite3
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

from typing import List, Union, Optional

sys.path.append("..")
from configs.common import *
from configs.experiment import *
from configs.path import *
from utils.mymath import *
from utils.dataloader import *
from utils.plots import *
from utils.render import *
from utils.utils import pickle_save
from agent.fps_task import *

class ExperimentDataRaw:
    def __init__(self, player, mode, msg=True):
        if msg: print(f"======== User ID: {player} ({mode}) ========")
        self.player = player
        self.mode = mode
    

    def generate_basic_tables(self, msg=True):
        if msg: print("Loading mouse, gaze data ...")
        self.data_m, self.data_g, self.data_c = load_raw_synced_mouse_gaze_data(self.player)
        db_conn = sqlite3.connect(
            PATH_DATA_EXP_RAW_DB % (self.player, self.mode)
        )
        time_range = queryDB(db_conn, "SELECT time FROM Frame_Info")["time"].to_numpy()
        self.data_m = data_in_timerange(
            self.data_m,
            "UTC_TIME",
            time_range[0] - MM,
            time_range[-1] + MM
        )
        self.data_g = data_in_timerange(
            self.data_g,
            "UTC_TIME",
            time_range[0] - MM,
            time_range[-1] + MM
        )

        # Interpolate gaze position information in advance
        full_ts = self.data_g["UTC_TIME"].to_numpy()
        full_ts = (full_ts - full_ts[0]) / MM
        # Interpolate eye position information in advance
        for D in "LR":
            eye_3d = self.data_g[[f"{D}EYEX", f"{D}EYEY", f"{D}EYEZ"]].to_numpy()
            eye_3d = sp_interp_nd(
                full_ts,
                full_ts[self.data_g[f"{D}PUPILV"].to_numpy()==1],
                eye_3d[self.data_g[f"{D}PUPILV"].to_numpy()==1], n=3
            )
            for i, A in enumerate("XYZ"): self.data_g[f"{D}EYE{A}"] = eye_3d[:,i]
        # Save the cropped mouse + gaze data
        self.data_m.to_csv(PATH_DATA_EXP_CROP_MOUSE % (self.player, self.player, self.mode), index=False)
        self.data_g.to_csv(PATH_DATA_EXP_CROP_GAZE % (self.player, self.player, self.mode), index=False)
        self.data_c.to_csv(PATH_DATA_EXP_RAW_CALIB % (self.player, self.player), index=False)

        if msg: print("Loading FPSci database ...")
        self.pac = queryDB(db_conn, "SELECT * FROM Player_Action")
        self.ses = queryDB(db_conn, "SELECT * FROM Sessions")
        self.ttj = queryDB(db_conn, "SELECT * FROM Target_Trajectory")
        self.tar = queryDB(db_conn, "SELECT * FROM Targets")
        self.tri = queryDB(db_conn, "SELECT * FROM Trials")
        self.__rework_table_session()
        self.__rework_table_trials()
        self.__rework_table_targets()
        self.__rework_table_player_action()
        self.__rework_table_target_trajectory()

        user_info = queryDB(db_conn, "SELECT * FROM Users")
        self.sensitivity = user_info["mouseDegPerMillimeter"].iloc[0]
        self.mouse_dpi = user_info["mouseDPI"].iloc[0]

        info_d = dict(
            sensitivity=[self.sensitivity],
            mouse_dpi=[self.mouse_dpi]
        )
        pd.DataFrame(info_d).to_csv(PATH_DATA_EXP_DB_INFO % (self.player, self.player, self.mode), index=False)
        self.ses_order = self.ses[self.ses.sessionID != "verif"]["sessionID"].to_list()
        self.gaze_correction = self._save_verif_gaze_correction_info()


    def load_existing_tables(self, msg=True):
        if msg: print("Loading mouse, gaze data ...")
        self.data_m = pd.read_csv(PATH_DATA_EXP_CROP_MOUSE % (self.player, self.player, self.mode), low_memory=False)
        self.data_g = pd.read_csv(PATH_DATA_EXP_CROP_GAZE % (self.player, self.player, self.mode), low_memory=False)
        self.data_c = pd.read_csv(PATH_DATA_EXP_RAW_CALIB % (self.player, self.player))

        if msg: print("Loading FPSci database ...")
        self.pac = pd.read_csv(PATH_DATA_EXP_DB_PAC % (self.player, self.player, self.mode), low_memory=False)
        self.ses = pd.read_csv(PATH_DATA_EXP_DB_SES % (self.player, self.player, self.mode), low_memory=False)
        self.ttj = pd.read_csv(PATH_DATA_EXP_DB_TTJ % (self.player, self.player, self.mode), low_memory=False)
        self.tar = pd.read_csv(PATH_DATA_EXP_DB_TAR % (self.player, self.player, self.mode), low_memory=False)
        self.tri = pd.read_csv(PATH_DATA_EXP_DB_TRI % (self.player, self.player, self.mode), low_memory=False)
        user_info = pd.read_csv(PATH_DATA_EXP_DB_INFO % (self.player, self.player, self.mode), low_memory=False)
        self.sensitivity = user_info["sensitivity"][0]
        self.mouse_dpi = user_info["mouse_dpi"][0]
        self.ses_order = self.ses[self.ses.sessionID != "verif"]["sessionID"].to_list()
        self.gaze_correction = pd.read_csv(PATH_DATA_EXP_GAZE_CORRECTION % (self.player, self.player, self.mode), low_memory=False)


    def save_trial_trajectories(self, sn, bn, plot=False):
        data = []
        for tn in range(TRIAL_NUM_PER_BLOCK):
            if self.check_trial_exist(sn, bn, tn):
                data.append(self.get_trial_trajectory(sn, bn, tn))
        
        # Correct gaze starting position
        gp0 = []
        for d in data:
            if len(d[2]) > 0:   # Gaze trajectory
                gp0.append(d[2][0])
        gp0 = np.array(gp0)
        gp0_mean = np.mean(gp0, axis=0)

        os.makedirs(PATH_DATA_EXP_TRAJ_PKL % self.player, exist_ok=True)
        for tn in range(TRIAL_NUM_PER_BLOCK):
            if not self.check_trial_exist(sn, bn, tn): continue
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
            ) = data[tn]
            if len(gaze_traj_m) > 0:
                gaze_traj_m -= gp0_mean
                # gaze_traj = apply_gaussian_filter_2d(gaze_traj, size=13)
            
            pickle_save(
                PATH_DATA_EXP_TRAJ_PKL % self.player + filename,
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
                )
            )

            if plot:
                plt_trajectory(
                    target_ts, target_traj, gaze_traj_m, gaze_flag, trad_from_sess(sn),
                    self.player, f"{self.mode}-{sn}-{bn}-{tn:02d}"
                )


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


    def check_gaze_validity(self, sn, bn, tn):
        if not self.check_trial_exist(sn, bn, tn):
            return False
        *_, v, _, _, _, _ = self.get_trial_trajectory(sn, bn, tn)
        return len(v) > 0


    def check_trial_exist(self, sn, bn, tn):
        return len(self.tri[
            (self.tri.session_id == sn) &
            (self.tri.block_index == bn) &
            (self.tri.trial_index == tn)
        ]) > 0 and \
        len(self.tar[
            (self.tar.session_id == sn) &
            (self.tar.block_index == bn) &
            (self.tar.trial_index == tn)
        ]) > 0


    def get_trial_trajectory(self, sn, bn, tn):
        # Target trajectory
        target_ts = data_in_timerange(
            self.pac, "time", 
            *self.__target_existed_time_range(sn, bn, tn)
        ).time.to_numpy()
        target_traj = self._target_monitor_trajectory(target_ts)

        start_mt = target_ts[0]
        target_ts_ft = (target_ts - start_mt) / MM

        # Gaze trajectory
        gaze_ts, gaze_traj, eye_pos, gaze_flag = self.load_gaze_data(sn, bn, tn, expand=0.5)
        if len(gaze_ts) > 0:
            gaze_flag_ft = np.array(np.interp(target_ts, gaze_ts, gaze_flag) >= 0.5, dtype=int)
            gaze_ts_ft = (gaze_ts - start_mt) / MM
            gaze_traj_m = sp_interp_nd(
                target_ts_ft,
                gaze_ts_ft,
                gaze_traj
            )
            # Interpolation
            vd = np.diff(np.append(np.insert(gaze_flag_ft, 0, 1), 1))
            onset, offset = np.where(vd<0)[0], np.where(vd>0)[0]
            for s, e in zip(onset, offset):
                if s == 0:
                    gaze_traj_m[s:e] = gaze_traj_m[e]
                elif e == len(gaze_flag_ft):
                    gaze_traj_m[s:e] = gaze_traj_m[s-1]
                elif np.sum(np.linalg.norm(np.diff(gaze_traj_m[s:e], axis=0), axis=1)) > 0.26:
                    gaze_traj_m[s:e] = np_interp_nd(
                        target_ts_ft[s:e],
                        target_ts_ft[gaze_flag_ft==1],
                        gaze_traj_m[gaze_flag_ft==1]
                    )
            gvec = np.insert(gaze_traj_m, 2, 0, axis=1) - eye_pos
            gaze_traj_a = cart2sphr(-gvec[:,2], gvec[:,1], gvec[:,0])
        else:
            gaze_flag_ft = np.array([])
            gaze_traj_m = np.array([])
            gaze_traj_a = np.array([])

        # Hand trajectory
        # hand_ts, hand_traj = self.load_mouse_data(sn, bn, tn)
        # hand_ts = (hand_ts - start_mt) / MM
        # hand_traj = np_interp_nd(target_ts_ft, hand_ts, hand_traj)
        # hand_traj -= hand_traj[0]

        # Save trial initial condition as dict together
        # pcam, tgpos_sp
        _, pcam_traj = self._camera_trajectory(target_ts)
        hand_traj = pcam_traj / self.sensitivity / 1000
        hand_traj -= hand_traj[0]
        # pcam0 = pcam_traj[0]
        tgpos0 = self.get_target_spawn_position(sn, bn, tn)
        toax = self.get_target_orbit_axis(sn, bn, tn)

        return (
            target_ts_ft, 
            target_traj,
            gaze_traj_m,
            gaze_traj_a,
            eye_pos,
            gaze_flag_ft,
            hand_traj,
            pcam_traj,
            tgpos0,
            toax
        )

    
    def target_initial_position(self, sn, bn, tn):
        timestamp = data_in_timerange(
            self.pac, "time", 
            *self.__target_existed_time_range(sn, bn, tn)
        ).time.to_numpy()
        tj_t = self._target_monitor_trajectory(timestamp)
        return tj_t[0]


    def get_target_orbit_axis(self, sn, bn, tn):
        """Return orbit axis in the form <az, el>"""
        if sn[0] == 's':    # Static
            return np.array([0.0, 0.0])
        
        tgpos = self.ttj[
            (self.ttj["session_id"] == sn) &
            (self.ttj["block_index"] == bn) &
            (self.ttj["trial_index"] == tn)
        ][["position_x", "position_y", "position_z"]].to_numpy()
        vec_p2t = tgpos - PRESET_CAM_POS
        orbit_axes = cart2sphr(*np.cross(vec_p2t[0], vec_p2t[1:]).T)
        return np.mean(orbit_axes, axis=0)

    
    def get_target_spawn_position(self, sn, bn, tn):
        """Return initial target position in <az, el>"""
        tgpos0 = self.tar[
            (self.tar["session_id"] == sn) &
            (self.tar["block_index"] == bn) &
            (self.tar["trial_index"] == tn)
        ][["spawn_ecc_h", "spawn_ecc_v"]].to_numpy()[0]
        return tgpos0
    

    def load_mouse_data(self, sn, bn, tn, expand=0.06):
        tt = self.__trial_time_range(sn, bn, tn)
        if tt is None: return None

        m = data_in_timerange(
            self.data_m, "UTC_TIME",
            tt[0] - expand * MM,
            tt[1] + expand * MM
        )
        # Compute cumulative -> dx, dy to mx, my
        mtraj = np.cumsum(
            m[["DX", "DY"]].to_numpy(), 
            axis=0
        ) / self.mouse_dpi * INCH_TO_METER
        mtraj -= mtraj[0]     # Translation: init pos to zero
        return m["UTC_TIME"].to_numpy(), mtraj


    def load_gaze_data(self, sn, bn, tn, expand=0.06, gaze_threshold=MINIMUM_VALID_RATIO_OF_GAZE):
        tt = self.__trial_time_range(sn, bn, tn)
        if tt is None: return np.array([]), np.array([]), np.array([]), np.array([])

        trans_mat = self.gaze_correction[
            (self.gaze_correction.session_id == sn) &
            (self.gaze_correction.block_id == bn) 
        ][["xa", "ya", "xb", "yb"]].to_numpy()[0].reshape((2, 2))

        pog = data_in_timerange(
            self.data_g, "UTC_TIME",
            tt[0] - expand * MM,
            tt[1] + expand * MM
        )
        gaze_traj = pog[["FPOGX", "FPOGY"]].to_numpy() * trans_mat[0] + trans_mat[1]
        gaze_flag = (
            pog["FPOGV"] *
            (pog["FPOGX"] < MONITOR_BOUND[X]) *
            (pog["FPOGX"] > -MONITOR_BOUND[X]) *
            (pog["FPOGY"] < MONITOR_BOUND[Y]) *
            (pog["FPOGY"] > -MONITOR_BOUND[Y])
        ).to_numpy()

        if np.mean(gaze_flag) < gaze_threshold:
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Distance from eye to monitor
        leye_3d = pog[["LEYEX", "LEYEY", "LEYEZ"]].to_numpy()
        reye_3d = pog[["REYEX", "REYEY", "REYEZ"]].to_numpy()
        eye_pos = np.mean((leye_3d + reye_3d) / 2, axis=0)
        
        # Compute eye position about monitor coordinate [z is depth here]
        eye_pos = np.array([
            eye_pos[X],
            eye_pos[Z]*np.sin(np.deg2rad(GP3_ANGLE)) - eye_pos[Y]*np.cos(np.deg2rad(GP3_ANGLE)) - (MONITOR[Y]/2 + GP3_YPOS_OFFSET),
            eye_pos[Z]*np.cos(np.deg2rad(GP3_ANGLE)) + eye_pos[Y]*np.sin(np.deg2rad(GP3_ANGLE))
        ]).T
        

        return pog["UTC_TIME"].to_numpy(), gaze_traj, eye_pos, gaze_flag
    

    def _target_monitor_trajectory(self, timestamp, expand=0.5):
        """Return target monitor position on given timestamps"""
        ppos, pcam = self._camera_trajectory(timestamp, expand=expand)
        tpos = self._target_game_trajectory(timestamp, expand=expand)
        return target_multi_pos_monitor(ppos, pcam, tpos)

    
    def _camera_trajectory(self, timestamp, expand=0.5):
        """Return camera position and angle on given timestamps"""
        _pa = data_in_timerange(
            self.pac, "time",
            timestamp.min() - expand * MM,
            timestamp.max() + expand * MM
        )
        ppos = np.array([
            np.interp(timestamp, _pa["time"].to_numpy(), _pa["position_"+axis].to_numpy())
            for axis in ["x", "y", "z"]
        ]).T
        pcam = np.array([
            np.interp(timestamp, _pa["time"].to_numpy(), _pa["position_"+axis].to_numpy())
            for axis in ["az", "el"]
        ]).T
        return ppos, pcam


    def _target_game_trajectory(self, timestamp, expand=0.5):
        """Return target game position on given timestamps"""
        _ttj = data_in_timerange(
            self.ttj, "time",
            timestamp.min() - expand * MM,
            timestamp.max() + expand * MM
        )
        tpos = np.array([
            np.interp(timestamp, _ttj["time"].to_numpy(), _ttj["position_"+axis].to_numpy())
            for axis in ["x", "y", "z"]
        ]).T
        return tpos

    
    def _save_verif_gaze_correction_info(self):
        info = dict(
            session_id = [],
            block_id = [],
            xa = [],
            ya = [],
            xb = [],
            yb = []
        )
        for sn in SES_NAME_ABBR:
            for bn in [0, 1]:
                verif_n = self.ses_order[bn*len(SES_NAME_ABBR):].index(sn) + bn*len(SES_NAME_ABBR)
                tm = self.__get_gaze_correction(verif_n)
                info["session_id"].append(sn)
                info["block_id"].append(bn)
                info["xa"].append(tm[1][0])
                info["ya"].append(tm[1][1])
                info["xb"].append(tm[0][0])
                info["yb"].append(tm[0][1])
        info = pd.DataFrame(info)
        info.to_csv(PATH_DATA_EXP_GAZE_CORRECTION % (self.player, self.player, self.mode), index=False)
        return info
    

    def __get_gaze_correction(
        self, 
        verif_n, 
        threshold_speed=0.00053,
        threshold_minimum_gaze_point=2,
        threshold_dist=0.045,
        plot=False
    ):
        """Get POG of all verification session in time order"""
        # Get gaze data of given verification session
        tt = self.tri[
            (self.tri.session_id == 'verif') & (self.tri.block_index == verif_n)
        ][["start_time", "end_time"]].to_numpy()[0]
        pog = data_in_timerange(self.data_g, "UTC_TIME", *tt)
        timestamp = pog["UTC_TIME"].to_numpy()

        # 1. Filter invalid flag
        gaze_traj = pog[["FPOGX", "FPOGY"]].to_numpy()
        validity_flag = ((pog.FPOGV == 1) & (pog.LPUPILV == 1) & (pog.RPUPILV == 1)).to_numpy()

        # 2. Filter saccade (speed > threshold)
        timestamp = (timestamp - timestamp[0]) / MM
        spd_gaze = np.linalg.norm(gaze_traj[1:] - gaze_traj[:-1], axis=1)
        spd_gaze = np.pad(spd_gaze, (0, 1), mode='edge')
        validity_flag[spd_gaze > threshold_speed] = False

        # Apply all filters
        timestamp = timestamp[validity_flag]
        gaze_traj = gaze_traj[validity_flag]

        # Cluster
        true_fixation = []
        gaze_fixation = []
        for c in range(VERIF_TARGET_POS.shape[0]):
            gaze_crop = gaze_traj[
                (timestamp >= VERIF_TIMESTAMP[c][0]) & 
                (timestamp <= VERIF_TIMESTAMP[c][1])
            ]
            if len(gaze_crop) < threshold_minimum_gaze_point: continue
            gaze_error = np.linalg.norm(
                gaze_crop - VERIF_TARGET_POS[c], axis=1
            )
            gaze_crop = gaze_crop[gaze_error < threshold_dist]
            if len(gaze_crop) < threshold_minimum_gaze_point: continue
            else:
                for _ in range(VERIF_CORRECTION_WEIGHT[c]):
                    true_fixation.append(VERIF_TARGET_POS[c])
                    gaze_fixation.append(np.mean(gaze_crop, axis=0))

                
        true_fixation = np.array(true_fixation)
        gaze_fixation = np.array(gaze_fixation)

        # Least Square method
        trans_mat = []
        for axis in [X, Y]:
            _x = np.vstack((
                np.ones(len(true_fixation)),
                gaze_fixation[:,axis]
            )).T
            _y = true_fixation[:, axis]
            trans_mat.append(np.linalg.inv(_x.T @ _x) @ _x.T @ _y)
        trans_mat = np.concatenate(trans_mat).reshape((2,2)).T

        if plot:
            plt.figure(figsize=(8,4.5))
            corrected_mg = gaze_fixation * trans_mat[1] + trans_mat[0]
            plt.scatter(
                *true_fixation.T, 
                s=15, zorder=10, color='k', marker='x'
            )
            plt.scatter(
                *corrected_mg.T, 
                s=15, zorder=15, color='r', marker='o'
            )
            for r, m in zip(true_fixation, corrected_mg):
                plt.plot(
                    *np.array([r, m]).T, 
                    zorder=10,
                    color='g',
                    linewidth=1
                )
            plt.xlim(-MONITOR_BOUND[X], MONITOR_BOUND[X])
            plt.ylim(-MONITOR_BOUND[Y], MONITOR_BOUND[Y])
            fig_save(PATH_VIS_EXP_VERIF % self.player, f"{self.mode}_{verif_n}")
        
        return trans_mat


    def __rework_table_session(self):
        # Remove unncessary columns
        del self.ses["frameRate"]
        del self.ses["frameDelay"]
        del self.ses["appendingDescription"]
        del self.ses["subjectID"]

        # Leave completed session only
        self.ses = self.ses[self.ses.complete == 'true']

        # Add index to verification session
        verifs = self.ses[self.ses.sessionID == "calib"].copy()
        verifs.insert(1, "block_index", np.arange(len(verifs))-1)
        verifs = verifs[verifs.block_index >= 0]
        verifs.sessionID = "verif"      # Name change: calib -> verif

        # Add index to game session
        rows = self.ses[(self.ses.sessionID != "calib") & (self.ses.sessionID != "prac")].copy()
        rows.insert(1, "block_index", [0] * len(SES_NAME_ABBR) + [1] * len(SES_NAME_ABBR))
        
        # Concat and replace
        self.ses = pd.concat([verifs, rows], axis=0).reset_index(drop=True)
        self.ses.complete = (self.ses.complete == "true")

        self.ses.to_csv(PATH_DATA_EXP_DB_SES % (self.player, self.player, self.mode), index=False)


    def __rework_table_trials(self):
        # Remove unnecessary columns
        del self.tri["trial_id"]
        del self.tri["block_id"]
        del self.tri["pretrial_duration"]
        del self.tri["total_targets"]

        # Add index to verification session
        verifs = self.tri[self.tri.session_id == "calib"].copy()
        verifs.insert(1, "block_index", np.arange(len(verifs))-1)
        verifs = verifs[verifs.block_index >= 0]
        verifs.session_id = "verif"      # Name change: calib -> verif
        verifs.trial_index = 0

        # Add index to each trials
        new_table = [verifs]
        half_time = self.__blocks_time_baseline()
        for sn in SES_NAME_ABBR:
            block0 = self.tri[(self.tri.session_id == sn) & (self.tri.end_time < half_time)].copy()
            block1 = self.tri[(self.tri.session_id == sn) & (self.tri.start_time > half_time)].copy()
            # Leave targets from complete session only
            block0 = block0.groupby("trial_index", as_index=False).last()
            block1 = block1.groupby("trial_index", as_index=False).last()
            block0.insert(1, "block_index", 0)
            block1.insert(1, "block_index", 1)
            new_table.append(pd.concat([block0, block1], axis=0))
        self.tri = pd.concat(new_table, axis=0).reset_index(drop=True)

        self.tri.to_csv(PATH_DATA_EXP_DB_TRI % (self.player, self.player, self.mode), index=False)

    
    def __rework_table_targets(self):
        """Split target name by session name and index (with block number)"""
        # Remove verification session & practice
        rows = self.tar[
            (self.tar.target_type_name != "practice_target") & 
            (self.tar.target_type_name != "calib_points")
        ]

        # Add columns to verification session
        verifs = self.tar[self.tar.target_type_name == "calib_points"].copy()
        verifs.insert(0, "session_id", "verif")
        verifs.insert(1, "block_index", np.arange(len(verifs))-1)
        verifs.insert(2, "trial_index", 0)
        verifs = verifs[verifs.block_index >= 0]

        # Parse trial index and block index for each target
        new_table = [verifs]
        half_time = self.__blocks_time_baseline()
        for sn, tn in zip(SES_NAME_ABBR, SES_NAME_FULL):
            block0 = rows[(rows.target_type_name == tn) & (rows.spawn_time < half_time)].copy()
            block1 = rows[(rows.target_type_name == tn) & (rows.spawn_time > half_time)].copy()
            # Leave targets from complete session only
            block0 = block0.groupby("name", as_index=False).last()
            block1 = block1.groupby("name", as_index=False).last()
            # Add new column: session_id and trial_index
            block0.insert(0, "session_id", sn)
            block1.insert(0, "session_id", sn)
            block0.insert(1, "block_index", 0)
            block1.insert(1, "block_index", 1)
            block0.insert(2, "trial_index", trial_index_from_names(block0.name.to_list()))
            block1.insert(2, "trial_index", trial_index_from_names(block1.name.to_list()))
            new_table.append(pd.concat([block0, block1], axis=0).sort_values(["block_index", "trial_index"]))
        self.tar = pd.concat(new_table, axis=0).reset_index(drop=True)

        # Remove unnecessary columns
        del self.tar["name"]
        del self.tar["target_type_name"]

        # Convert size to radius (monitor, (m))
        self.tar.insert(3, "radius", self.tar["size"] * 0.09406)
        del self.tar["size"]

        self.tar.to_csv(PATH_DATA_EXP_DB_TAR % (self.player, self.player, self.mode), index=False)


    def __rework_table_player_action(self):
        # Get rows with click events only (destroy or miss)
        click_events = self.pac[
            ((self.pac.event == "miss") | (self.pac.event == "destroy")) & 
            (self.pac.state == "trialTask") &
            ~(self.pac.target_id.str.contains("prac"))
        ].copy()
        # Prepare columns
        self.pac.insert(9, "session_id", np.nan)
        self.pac.insert(10, "block_index", np.nan)
        self.pac.insert(11, "trial_index", np.nan)
        # Fill in columns
        for sn in SES_NAME_ABBR:
            for bn in range(2):
                for tn in range(TRIAL_NUM_PER_BLOCK):
                    time_range = self.__trial_time_range(sn, bn, tn)
                    try:
                        click_time = click_events[
                            (click_events.time >= time_range[0]) & (click_events.time <= time_range[1])
                        ].index
                        self.pac.at[click_time[0], "session_id"] = sn
                        self.pac.at[click_time[0], "block_index"] = bn
                        self.pac.at[click_time[0], "trial_index"] = tn
                    except: continue # np.nan detected
        self.pac["position_az"] -= 90    # Let az=0, el=0 be <1, 0, 0>

        self.pac.to_csv(PATH_DATA_EXP_DB_PAC % (self.player, self.player, self.mode), index=False)


    def __rework_table_target_trajectory(self):
        # Add session id, trial index, block index
        self.ttj["session_id"] = sess_id_from_names(self.ttj.target_id.to_list())
        half_time = self.__blocks_time_baseline()
        def convert_to_block_index(t, sid):
            if sid in SES_NAME_ABBR: return int(half_time < t)
            else: return np.nan
        def convert_to_block_indices(t, sid):
            return list(map(convert_to_block_index, t, sid))
        self.ttj["block_index"] = convert_to_block_indices(self.ttj.time.to_list(), self.ttj.session_id.to_list())
        self.ttj["trial_index"] = trial_index_from_names(self.ttj.target_id.to_list())
        # # Monitor position
        # ttj_timestamp = self.ttj["time"].to_numpy()
        # ttj_tpos = self.ttj[["position_x", "position_y", "position_z"]].to_numpy()

        # pac_timestamp = self.pac["time"].to_numpy()
        # pac_camvec = self.pac[["position_az", "position_el"]].to_numpy()
        # pac_campos = self.pac[["position_x", "position_y", "position_z"]].to_numpy()
        # ttj_camvec = np_interp_2d(ttj_timestamp, pac_timestamp, pac_camvec)
        # ttj_campos = np_interp_3d(ttj_timestamp, pac_timestamp, pac_campos)

        # ttj_tmpos = target_multi_pos_monitor(ttj_campos, ttj_camvec, ttj_tpos)
        # self.ttj["position_mx"] = ttj_tmpos[:,X]
        # self.ttj["position_my"] = ttj_tmpos[:,Y]

        # self.ttj["cam_az"] = ttj_camvec[:,X]
        # self.ttj["cam_el"] = ttj_camvec[:,Y]

        # self.ttj["cam_x"] = ttj_campos[:,X]
        # self.ttj["cam_y"] = ttj_campos[:,Y]
        # self.ttj["cam_z"] = ttj_campos[:,Z]

        self.ttj.to_csv(PATH_DATA_EXP_DB_TTJ % (self.player, self.player, self.mode), index=False)

    
    def __blocks_time_baseline(self):
        """
        Return time between first block group and second block group
        Computed as end of 9th verification session
        """
        return self.__session_time_range("verif", len(SES_NAME_ABBR)-1)[1]
    

    def __session_time_range(self, sn, bn):
        """Return start time and end time of session sn, block bn"""
        try:
            return self.ses[
                (self.ses.sessionID == sn) & (self.ses.block_index == bn)
            ][["start_time", "end_time"]].to_numpy()[0]
        except:
            print(f"Error raised on loading time range of session: {sn} ({bn})")
            return None

    
    def __trial_time_range(self, sn, bn, tn):
        """Return start time and end time of trial sn-bn-tn, using trial table"""
        try:
            return self.tri[
                (self.tri.session_id == sn) & (self.tri.block_index == bn) & (self.tri.trial_index == tn)
            ][["start_time", "end_time"]].to_numpy()[0]
        except:
            # print("Error raised on loading time range of trial: %s, %d, %d" % (sn, bn, tn))
            return None
        

    def __target_existed_time_range(self, sn, bn, tn, expand=0.05):
        """Return start time and end time of trial sn-bn-tn, using target trajectory table"""
        try:
            spt = self.tar[
                (self.tar.session_id == sn) & 
                (self.tar.block_index == bn) & 
                (self.tar.trial_index == tn)
            ]["spawn_time"].to_numpy()[0]
            return self.ttj[
                (self.ttj.session_id == sn) & 
                (self.ttj.block_index == bn) & 
                (self.ttj.trial_index == tn) &
                (self.ttj.time >= spt - expand*MM)
            ]["time"].to_numpy()[[0, -1]]
        except:
            print("Error raised on loading time range of trial: %s, %d, %d" % (sn, bn, tn))
            return None
    

    def __exist_processed_table_files(self):
        return (
            os.path.exists(PATH_DATA_EXP_DB_PAC % (self.player, self.player, self.mode)) and
            os.path.exists(PATH_DATA_EXP_DB_SES % (self.player, self.player, self.mode)) and
            os.path.exists(PATH_DATA_EXP_DB_TTJ % (self.player, self.player, self.mode)) and
            os.path.exists(PATH_DATA_EXP_DB_TAR % (self.player, self.player, self.mode)) and
            os.path.exists(PATH_DATA_EXP_DB_TRI % (self.player, self.player, self.mode)) and
            os.path.exists(PATH_DATA_EXP_DB_INFO % (self.player, self.player, self.mode)) and
            os.path.exists(PATH_DATA_EXP_GAZE_CORRECTION % (self.player, self.player, self.mode)) 
        )



if __name__ == "__main__":

    import argparse
    from joblib import Parallel, delayed

    def save_trajs(p, m):
        d = ExperimentDataRaw(p, m, msg=False)
        d.load_existing_tables(msg=False)
        for sn in SES_NAME_ABBR:
            for bn in range(2):
                d.save_trial_trajectories(sn, bn, plot=False)

    Parallel(n_jobs=12)(delayed(save_trajs)(p, m) for p, m in zip(tqdm(PLAYERS + PLAYERS), ["custom"]*20+["default"]*20))