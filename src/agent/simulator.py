"""
Aim-and-Shoot Simulator
This code implements the simulation queries.

Code written by June-Seop Yoon
"""

import numpy as np
import pandas as pd
import psutil
from copy import deepcopy

# import os
from tqdm import tqdm
from pathlib import Path

from stable_baselines3 import SAC
from typing import List, Optional
from joblib import Parallel, delayed

from ..agent import vplayer
from ..nets.sac_policy_modul import ModulatedSACPolicy
# from ..agent.task import AimandShootSpiderShotTask as AnSTask
# from ..agent.module_aim import Aim
from ..configs.constants import FEATURES, TINTERVAL, FOLDERS
from ..configs.constants import AGENT_STATE_ALIAS as AS

from ..utils.mymath import Convert, angle_between_vectors, np_interp_nd
from ..utils.myutils import load_yaml_config, get_compact_timestamp_str, format_large_number
from ..utils.otg import otg_2d


# Module-level worker state — cached per worker process.
# loky reuses worker processes across Parallel() calls, so we must detect when
# the policy path changes (i.e. a new checkpoint) and reload accordingly.
_worker_model = None
_worker_agent = None
_worker_policy_path = None  # tracks which checkpoint is currently loaded

class AimandShootEpisodeRecorder:
    def __init__(
        self,
        agent_env: vplayer.AnSPlayerAgentDefault,
        save_trajectory: bool = True,
        save_coarse_trajectory: bool = False,
    ):
        # self.config = agent_env.config.copy()
        self.player_state = agent_env.player_state.copy()
        self.init_game_env = agent_env.init_game_env.copy()
        self.actions = list()
        self.result_report = dict()
        self.save_trajectory = save_trajectory
        self.save_coarse_trajectory = save_coarse_trajectory
        supports_gaze_metrics = getattr(agent_env, "supports_gaze_metrics", None)
        self.supports_gaze_metrics = bool(supports_gaze_metrics()) if callable(supports_gaze_metrics) else False


    def record_step(self, agent_env: vplayer.AnSPlayerAgentDefault, done, truncated, info):
        state = info["step_state"]
        self.actions.append(state[AS.ACTION])
        if done or truncated:
            self.actions = pd.DataFrame(self.actions)
            self.result_report[FEATURES.TCT] = state[AS.SHOOT_MOMENT_MS]
            self.result_report[FEATURES.RES] = state[AS.SHOOT_RESULT]
            self.result_report[FEATURES.ERR] = state[AS.SHOOT_ERROR_MM]
            self.result_report[FEATURES.ERR_DEG] = state[AS.SHOOT_ERROR_DEG]
            self.result_report[FEATURES.MAX_HSPD] = state[AS.MAX_HAND_SPEED]
            self.result_report[FEATURES.FINAL_HSPD] = state[AS.FINAL_HSPD_MM_S]
            self.result_report[FEATURES.HAND_ACCEL_SUM] = state[AS.HAND_ACCEL_SUM]
            self.result_report[FEATURES.REW] = state[AS.REWARD]
            self.result_report[FEATURES.TRUNC] = truncated

            if self.save_trajectory:
                hand_traj_pos, hand_traj_vel, hand_traj_timestamp = agent_env.get_hand_trajectory()
                self.result_report[FEATURES.HTRAJP] = hand_traj_pos
                self.result_report[FEATURES.HTRAJV] = hand_traj_vel
                self.result_report[FEATURES.HTRAJT] = hand_traj_timestamp

            if self.supports_gaze_metrics:
                self._record_gaze_metrics(agent_env)

            if self.save_coarse_trajectory:
                cam_azel, coarse_ts = agent_env.get_camera_trajectory_azel()
                tgt_azel, _         = agent_env.get_target_trajectory_azel()
                self.result_report[FEATURES.CAM_TRAJ_AZEL]  = cam_azel
                self.result_report[FEATURES.TGT_TRAJ_AZEL]  = tgt_azel
                self.result_report[FEATURES.COARSE_TRAJ_TS] = coarse_ts

    def _head_position(self):
        head_position = self.player_state.get("head_position", None)
        if head_position is None:
            return None
        head_position = np.asarray(head_position, dtype=float).reshape(-1)
        if head_position.shape[0] < 3 or not np.all(np.isfinite(head_position[:3])):
            return None
        return head_position[:3]

    def _record_gaze_metrics(self, agent_env: vplayer.AnSPlayerAgentDefault):
        try:
            gaze_traj_pos, gaze_traj_timestamp = agent_env.get_gaze_trajectory()
        except (AttributeError, ValueError):
            return

        gaze_traj_pos = np.asarray(gaze_traj_pos, dtype=float).reshape(-1, 2)
        gaze_traj_timestamp = np.asarray(gaze_traj_timestamp, dtype=float).reshape(-1)
        if gaze_traj_pos.shape[0] == 0 or gaze_traj_pos.shape[0] != gaze_traj_timestamp.shape[0]:
            return

        head_position = self._head_position()
        if head_position is not None:
            gaze_traj_azel = _monitor_points_to_visual_azel_deg(gaze_traj_pos, head_position)
            scd_deg, scd_mm = self._compute_saccadic_deviation(gaze_traj_pos, gaze_traj_timestamp)
            self.result_report[FEATURES.SCD_DEG] = scd_deg
            self.result_report[FEATURES.SCD_MM] = scd_mm
        else:
            gaze_traj_azel = np.full_like(gaze_traj_pos, np.nan, dtype=float)

        if self.save_trajectory:
            self.result_report[FEATURES.GTRAJP] = gaze_traj_pos.astype(np.float32)
            self.result_report[FEATURES.GTRAJ_AZEL] = gaze_traj_azel.astype(np.float32)
            self.result_report[FEATURES.GTRAJT] = gaze_traj_timestamp.astype(np.float32)

    def _compute_saccadic_deviation(self, gaze_traj_pos, gaze_traj_timestamp):
        head_position = self._head_position()
        if head_position is None:
            return np.nan, np.nan

        gaze_traj_pos = np.asarray(gaze_traj_pos, dtype=float).reshape(-1, 2)
        gaze_traj_timestamp = np.asarray(gaze_traj_timestamp, dtype=float).reshape(-1)
        if gaze_traj_pos.shape[0] == 0 or gaze_traj_pos.shape[0] != gaze_traj_timestamp.shape[0]:
            return np.nan, np.nan

        try:
            traj_pos, _ = _interpolate_time_series(
                gaze_traj_pos,
                gaze_traj_timestamp,
                end_ms=self.result_report.get(FEATURES.TCT, None),
            )
        except ValueError:
            return np.nan, np.nan

        if traj_pos.shape[0] == 0:
            return np.nan, np.nan

        peak_idx = int(np.argmax(np.linalg.norm(traj_pos, axis=1)))
        ray0 = np.array([traj_pos[0, 0], traj_pos[0, 1], 0.0], dtype=float) - head_position
        ray1 = np.array([traj_pos[peak_idx, 0], traj_pos[peak_idx, 1], 0.0], dtype=float) - head_position
        try:
            scd_deg = float(angle_between_vectors(ray0, ray1, return_in_degree=True))
        except ValueError:
            scd_deg = 0.0
        scd_mm = float(np.linalg.norm(traj_pos[peak_idx] - traj_pos[0]))
        return scd_deg, scd_mm

    def get_trajectory(self, interpolate: bool = True, episode_time: bool = True):
        """Return the hand trajectory keyframes, optionally OTG-interpolated to 1ms.

        Parameters
        ----------
        interpolate : bool, default True
            When *True* (original behaviour) each MUSCLE-interval segment is
            OTG-smoothed and all remaining gaps are linearly filled to 1ms
            resolution.  When *False* the raw MUSCLE-interval keyframes are
            returned directly — much faster for large-scale dataset generation.

        Returns
        -------
        p : (M, 2) float  — hand positions (episode-relative mm)
        v : (M, 2) float  — hand velocities (mm/s)
        t : (M,)   float  — timestamps (ms)
        """
        assert self.save_trajectory, "Trajectory data not saved in this recorder."
        p = self.result_report[FEATURES.HTRAJP].astype(float)
        v = self.result_report[FEATURES.HTRAJV].astype(float)
        t = self.result_report[FEATURES.HTRAJT].astype(float)

        if not interpolate:
            # Collapse the 1ms-interpolated shoot sub-segment to just its endpoint.
            # The shoot segment always appears at the tail of the raw trajectory as a
            # consecutive run of INTERP1 (≈1ms) intervals.  All pre-shoot segments
            # use MUSCLE (≈50ms) or reaction-time intervals, both >> 1ms.
            dts = np.diff(t)
            if len(dts) > 0:
                # Walk backwards: find the index in `t` where the 1ms run starts.
                shoot_start = len(t) - 1   # sentinel: no 1ms segment
                for i in range(len(dts) - 1, -1, -1):
                    if abs(dts[i] - TINTERVAL.INTERP1) < 0.5:
                        shoot_start = i    # t[shoot_start] is last pre-shoot keyframe
                    else:
                        break
                if shoot_start < len(t) - 1:
                    # Collapse 1ms run → single endpoint (shoot moment)
                    p = np.concatenate([p[:shoot_start + 1], p[-1:]])
                    v = np.concatenate([v[:shoot_start + 1], v[-1:]])
                    t = np.concatenate([t[:shoot_start + 1], t[-1:]])
            if episode_time:
                return self._resample_hand_to_episode_time(p, v, t, interval_ms=None)
            return p, v, t

        intervals = np.diff(t)
        p_segs, v_segs, t_segs = [], [], []

        for i, dt in enumerate(intervals):
            t0, p0, v0 = t[i], p[i], v[i]
            t1, p1, v1 = t[i + 1], p[i + 1], v[i + 1]

            if abs(dt - TINTERVAL.MUSCLE) < 0.5:
                # OTG smooth interpolation for MUSCLE-spaced keyframes
                seg_dur = int(round(dt))
                p_seg, v_seg = otg_2d(p0, v0, p1, v1,
                                      TINTERVAL.INTERP1, seg_dur, seg_dur)
                p_segs.append(p_seg[:-1])
                v_segs.append(v_seg[:-1])
                t_segs.append(t0 + np.arange(seg_dur) * TINTERVAL.INTERP1)
            else:
                # Linear interpolation (orbit gap or 1ms shoot segment)
                n_pts = max(1, int(round(dt / TINTERVAL.INTERP1)))
                alpha = np.linspace(0.0, 1.0, n_pts + 1)[:-1]
                p_segs.append(p0 + alpha[:, np.newaxis] * (p1 - p0))
                v_segs.append(v0 + alpha[:, np.newaxis] * (v1 - v0))
                t_segs.append(t0 + alpha * dt)

        p_segs.append(p[-1:])
        v_segs.append(v[-1:])
        t_segs.append(t[-1:])

        p_fine = np.concatenate(p_segs)
        v_fine = np.concatenate(v_segs)
        t_fine = np.concatenate(t_segs)

        if episode_time:
            return self._resample_hand_to_episode_time(
                p_fine,
                v_fine,
                t_fine,
                interval_ms=TINTERVAL.INTERP1,
            )

        return p_fine, v_fine, t_fine

    def _resample_hand_to_episode_time(self, p, v, t, interval_ms=None):
        tct_ms = float(self.result_report[FEATURES.TCT])
        p, t_clean = _clean_time_series(p, t)
        v, _ = _clean_time_series(v, t)

        if interval_ms is None:
            interior = t_clean[(t_clean > 0.0) & (t_clean < tct_ms)]
            query = np.unique(np.concatenate(([0.0], interior, [tct_ms]))).astype(float)
        else:
            n = int(np.floor(tct_ms / interval_ms)) + 1
            query = np.arange(n, dtype=float) * float(interval_ms)
            if query.size == 0 or query[-1] < tct_ms:
                query = np.append(query, tct_ms)

        return (
            np_interp_nd(query, t_clean, p),
            np_interp_nd(query, t_clean, v),
            query,
        )

    def has_gaze_metrics(self) -> bool:
        return FEATURES.GTRAJP in self.result_report and FEATURES.GTRAJT in self.result_report

    def get_gaze_trajectory(self, interpolate: bool = True):
        """Return gaze trajectory in both monitor-mm and visual-angle units."""
        assert self.has_gaze_metrics(), (
            "Gaze trajectory not saved. Use an agent that supports gaze metrics "
            "and construct the recorder with save_trajectory=True."
        )
        gaze_traj_pos = self.result_report[FEATURES.GTRAJP].astype(float)
        gaze_traj_timestamp = self.result_report[FEATURES.GTRAJT].astype(float)

        if interpolate:
            gaze_traj_pos, gaze_traj_timestamp = _interpolate_time_series(
                gaze_traj_pos,
                gaze_traj_timestamp,
                end_ms=self.result_report.get(FEATURES.TCT, None),
            )

        head_position = self._head_position()
        if head_position is None:
            gaze_traj_azel = np.full_like(gaze_traj_pos, np.nan, dtype=float)
        else:
            gaze_traj_azel = _monitor_points_to_visual_azel_deg(gaze_traj_pos, head_position)
        return gaze_traj_pos, gaze_traj_azel, gaze_traj_timestamp

    def get_gaze_trajectory_record(self, interpolate: bool = True) -> dict:
        """Return gaze trajectory as aligned 1-D arrays for DataFrame export."""
        gaze_pos, gaze_azel, ts = self.get_gaze_trajectory(interpolate=interpolate)
        ts = np.asarray(ts, dtype=np.float32)
        dt = np.zeros_like(ts)
        if len(ts) > 1:
            dt[1:] = np.diff(ts)

        return dict(
            timestamp_ms=ts,
            dt_ms=dt,
            gaze_x_mm=gaze_pos[:, 0].astype(np.float32),
            gaze_y_mm=gaze_pos[:, 1].astype(np.float32),
            gaze_az_deg=gaze_azel[:, 0].astype(np.float32),
            gaze_el_deg=gaze_azel[:, 1].astype(np.float32),
        )

    def get_camera_trajectory_azel(self) -> "tuple[np.ndarray, np.ndarray]":
        """Camera az/el trajectory at BUMP resolution (requires save_coarse_trajectory=True).

        Returns
        -------
        azel : (T, 2) float32  — [az_deg, el_deg] per recording point
        timestamps : (T,) float32  — ms; T = 1 + n_decision_steps
        """
        assert self.save_coarse_trajectory, (
            "Coarse trajectory not saved.  Construct with save_coarse_trajectory=True."
        )
        return (
            self.result_report[FEATURES.CAM_TRAJ_AZEL],
            self.result_report[FEATURES.COARSE_TRAJ_TS],
        )

    def get_target_trajectory_azel(self) -> "tuple[np.ndarray, np.ndarray]":
        """Target az/el trajectory at BUMP resolution (requires save_coarse_trajectory=True).

        Returns
        -------
        azel : (T, 2) float32  — [az_deg, el_deg] per recording point
        timestamps : (T,) float32  — ms; shares timestamps with camera trajectory
        """
        assert self.save_coarse_trajectory, (
            "Coarse trajectory not saved.  Construct with save_coarse_trajectory=True."
        )
        return (
            self.result_report[FEATURES.TGT_TRAJ_AZEL],
            self.result_report[FEATURES.COARSE_TRAJ_TS],
        )


    def get_coarse_trajectory_record(self) -> dict:
        """Return the coarse (BUMP-resolution) trajectory as a dict of 1-D
        arrays aligned to camera timestamps.

        Camera timestamps (BUMP = 100 ms) are a **subset** of hand timestamps
        (MUSCLE = 50 ms).  All returned arrays share the same length T and are
        directly convertible to a ``pandas.DataFrame``.

        Requires ``save_coarse_trajectory=True`` at construction time.

        Returns
        -------
        dict with keys:
            timestamp_ms    – (T,) camera-aligned timestamps in ms
            dt_ms           – (T,) Δt between consecutive timestamps (first = 0)
            camera_az_deg   – (T,)
            camera_el_deg   – (T,)
            target_az_deg   – (T,)
            target_el_deg   – (T,)
        """
        cam_azel, ts = self.get_camera_trajectory_azel()   # (T,2), (T,)
        tgt_azel, _  = self.get_target_trajectory_azel()   # (T,2), (T,)

        ts = np.asarray(ts, dtype=np.float32)
        dt = np.zeros_like(ts)
        if len(ts) > 1:
            dt[1:] = np.diff(ts)

        return dict(
            timestamp_ms=ts,
            dt_ms=dt,
            camera_az_deg=cam_azel[:, 0].astype(np.float32),
            camera_el_deg=cam_azel[:, 1].astype(np.float32),
            target_az_deg=tgt_azel[:, 0].astype(np.float32),
            target_el_deg=tgt_azel[:, 1].astype(np.float32),
        )

    
    def get_summarized_result(self):
        # Convert cart to sphr
        target_world_pos_azel = Convert.cart2sphr_scalar(*self.init_game_env["target_pos_world"])
        player_state_flat = {
            k: v
            for k, v in self.player_state.items()
            if k not in ("head_position", "gaze_position")
        }

        head_summary = {}
        head_position = self._head_position()
        if head_position is not None:
            head_summary = {
                FEATURES.HEAD_POS_X: float(head_position[0]),
                FEATURES.HEAD_POS_Y: float(head_position[1]),
                FEATURES.HEAD_POS_Z: float(head_position[2]),
            }

        gaze_summary = {}
        gaze_position = self.player_state.get("gaze_position", None)
        if gaze_position is not None:
            gaze_position = np.asarray(gaze_position, dtype=float).reshape(-1)
            if gaze_position.shape[0] >= 2:
                if head_position is None:
                    gaze_azel = np.array([np.nan, np.nan], dtype=float)
                else:
                    gaze_azel = _monitor_points_to_visual_azel_deg(gaze_position[:2], head_position)
                gaze_summary.update({
                    FEATURES.GAZE_INIT_X: float(gaze_position[0]),
                    FEATURES.GAZE_INIT_Y: float(gaze_position[1]),
                    FEATURES.GAZE_INIT_DISTANCE: float(np.linalg.norm(gaze_position[:2])),
                    FEATURES.GAZE_INIT_AZ: float(gaze_azel[0]),
                    FEATURES.GAZE_INIT_EL: float(gaze_azel[1]),
                })

        if "gaze_reaction_time" in self.player_state:
            gaze_summary[FEATURES.GRT] = float(self.player_state["gaze_reaction_time"])
        if FEATURES.SCD_DEG in self.result_report:
            gaze_summary[FEATURES.SCD_DEG] = float(self.result_report[FEATURES.SCD_DEG])
        if FEATURES.SCD_MM in self.result_report:
            gaze_summary[FEATURES.SCD_MM] = float(self.result_report[FEATURES.SCD_MM])

        return {
            FEATURES.TARGET_POS_MONITOR_X: self.init_game_env["target_pos_monitor_mm"][0],
            FEATURES.TARGET_POS_MONITOR_Y: self.init_game_env["target_pos_monitor_mm"][1],
            FEATURES.TARGET_POS_DISTANCE: float(np.linalg.norm(self.init_game_env["target_pos_monitor_mm"])),
            FEATURES.TARGET_POS_WORLD_AZ: target_world_pos_azel[0],
            FEATURES.TARGET_POS_WORLD_EL: target_world_pos_azel[1],
            FEATURES.TARGET_RADIUS: self.init_game_env["target_radius_deg"],
            FEATURES.TARGET_ASPEED: self.init_game_env["target_speed_deg_s"],
            FEATURES.TARGET_ORBIT_AXIS_AZ: self.init_game_env["target_orbit_axis_deg"][0],
            FEATURES.TARGET_ORBIT_AXIS_EL: self.init_game_env["target_orbit_axis_deg"][1],
            FEATURES.TARGET_MOVEMENT_MONITOR_DIRECTION: self.init_game_env["target_motion_dir_deg"],

            FEATURES.CAM_INIT_ANGLE_AZ: self.init_game_env["camera_azel_deg"][0],
            FEATURES.CAM_INIT_ANGLE_EL: self.init_game_env["camera_azel_deg"][1],

            **head_summary,
            **player_state_flat,
            **gaze_summary,

            FEATURES.TCT: self.result_report[FEATURES.TCT],
            FEATURES.ACC: int(self.result_report[FEATURES.RES]),
            FEATURES.ERR: self.result_report[FEATURES.ERR],
            FEATURES.ERR_DEG: self.result_report[FEATURES.ERR_DEG],
            FEATURES.NORM_ERR: (
                float(self.result_report[FEATURES.ERR])
                / float(self.init_game_env["target_radius_mm"])
            ),
            FEATURES.MAX_HSPD: self.result_report[FEATURES.MAX_HSPD],
            FEATURES.FINAL_HSPD: self.result_report[FEATURES.FINAL_HSPD],
            FEATURES.HAND_ACCEL_SUM: self.result_report[FEATURES.HAND_ACCEL_SUM],
            FEATURES.REW: self.result_report[FEATURES.REW],
            FEATURES.TRUNC: int(self.result_report[FEATURES.TRUNC]),
            # ── Derived features ────────────────────────────────────────────
            FEATURES.TARGET_RADIUS_MM: float(self.init_game_env["target_radius_mm"]),
            FEATURES.MOVEMENT_TIME: max(
                0.0,
                float(self.result_report[FEATURES.TCT])
                - float(player_state_flat.get("hand_reaction_time", 0.0))
            ),
            FEATURES.FITTS_IOD: float(np.log2(
                float(np.linalg.norm(self.init_game_env["target_pos_monitor_mm"]))
                / (2.0 * float(self.init_game_env["target_radius_mm"]))
                + 1.0
            )),
        }


class AimandShootSimulator:
    def __init__(self, model_name: str, ckpt: int | str = 'latest'):
        base_dir = Path(__file__).resolve().parent.parent.parent / FOLDERS.DATA / FOLDERS.RL_MODEL / model_name
        checkpoint_dir = base_dir / FOLDERS.MODEL_CHECKPT
        best_model_path = base_dir / FOLDERS.MODEL_BEST / "best_model.zip"
        
        if not base_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {base_dir}")

        resolved_ckpt = ckpt
        if ckpt == "best":
            policy_path = best_model_path
        elif isinstance(ckpt, int):
            policy_path = checkpoint_dir / f"rl_model_{ckpt}_steps.zip"
        elif ckpt == "latest":
            latest_ckpt = None
            latest_step = -1
            for p in checkpoint_dir.glob("rl_model_*_steps.zip"):
                stem = p.stem  # rl_model_<step>_steps
                parts = stem.split("_")
                if len(parts) >= 4 and parts[0] == "rl" and parts[1] == "model" and parts[-1] == "steps":
                    try:
                        step = int(parts[2])
                    except ValueError:
                        continue
                    if step > latest_step:
                        latest_step = step
                        latest_ckpt = p

            if latest_ckpt is not None:
                policy_path = latest_ckpt
                resolved_ckpt = latest_step
            elif best_model_path.exists():
                policy_path = best_model_path
                resolved_ckpt = "best"
            else:
                raise FileNotFoundError(
                    f"No checkpoint found in {checkpoint_dir} and no best model at {best_model_path}"
                )
        else:
            raise TypeError("Argument ckpt must be one of: int, 'best', or 'latest'.")

        if not policy_path.exists():
            raise FileNotFoundError(f"Policy file not found: {policy_path}")

        self.policy = str(policy_path)
        self.model = _load_sac_for_inference(self.policy)
        self._model_name_base = model_name
        self.model_name = f"{model_name}_{format_large_number(resolved_ckpt)}"

        # Load environment config saved by current training pipeline.
        agent_cfg_path = base_dir / "agent_config.yaml"
        if not agent_cfg_path.exists():
            raise FileNotFoundError(f"agent_config.yaml not found: {agent_cfg_path}")

        env_config = load_yaml_config(str(agent_cfg_path))
        env_class_name = env_config.get("agent_class", "AnSPlayerAgentDefault")
        env_class = getattr(vplayer, env_class_name)
        env_config = {k: v for k, v in env_config.items() if k != "agent_class"}
        self._env_class_name = env_class_name
        self._env_config = deepcopy(env_config)

        self.agent: vplayer.AnSPlayerAgentDefault = env_class(config=env_config)
        self.simulation_records: list[AimandShootEpisodeRecorder] = []


    def clear_records(self):
        self.simulation_records.clear()


    def simulate(
        self,
        env_preset_list: Optional[List[dict]] = None,
        num_simul: Optional[int] = None,
        num_cpu: int = 1,
        overwrite_existing_simul: bool = False,
        resimulate_max_num: int = 0,
        deterministic: bool = True,
        verbose: bool = True,
        save_trajectory: bool = True,
        save_coarse_trajectory: bool = False,
        player_state_preset: Optional[dict] = None,
        tqdm_position: int = 0,
    ):
        """Run Monte-Carlo simulation episodes."""
        if env_preset_list is None:
            if num_simul is None:
                raise ValueError("num_simul must be provided when env_preset_list is None.")
            preset_sequence = [None] * int(num_simul)
            num_simul = int(num_simul)
        else:
            # When explicit presets are provided, always use all of them.
            preset_sequence = list(env_preset_list)
            num_simul = len(preset_sequence)

        if player_state_preset:
            shared_preset = dict(player_state_preset)
            preset_sequence = [
                {**shared_preset, **(preset or {})}
                for preset in preset_sequence
            ]

        max_cpu = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
        num_cpu = max(1, min(int(num_cpu), int(max_cpu)))

        if overwrite_existing_simul:
            self.clear_records()

        if num_cpu == 1:
            progress = tqdm(
                total=num_simul,
                desc=f"Simulations [cpu={num_cpu}]",
                position=tqdm_position,
                leave=False,
            ) if verbose else None
            ep_idx = 0
            retry_count = 0

            while ep_idx < num_simul:
                preset = preset_sequence[ep_idx]
                if preset is None:
                    self.agent.update_env_presets()
                else:
                    self.agent.update_env_presets(**preset)
                obs, _ = self.agent.reset()
                ep_rec = AimandShootEpisodeRecorder(
                    self.agent,
                    save_trajectory=save_trajectory,
                    save_coarse_trajectory=save_coarse_trajectory,
                )

                done = False
                truncated = False
                while not (done or truncated):
                    action, _ = self.model.predict(obs, deterministic=deterministic)
                    obs, _, done, truncated, info = self.agent.step(action)
                    ep_rec.record_step(self.agent, done, truncated, info)

                if truncated and retry_count < resimulate_max_num:
                    retry_count += 1
                    continue

                self.simulation_records.append(ep_rec)
                ep_idx += 1
                retry_count = 0
                if progress is not None:
                    progress.update(1)

            if progress is not None:
                progress.close()
            return

        n_workers = min(num_cpu, num_simul)

        results_gen = Parallel(n_jobs=n_workers, return_as="generator")(
            delayed(_simulate_single_episode_joblib)(
                self.policy,
                self._env_class_name,
                self._env_config,
                preset,
                deterministic,
                resimulate_max_num,
                save_trajectory,
                save_coarse_trajectory,
            )
            for preset in preset_sequence
        )

        progress = tqdm(
            total=num_simul,
            desc=f"Simulations [cpu={n_workers}]",
            disable=not verbose,
            position=tqdm_position,
            leave=False,
        )
        for ep_rec in results_gen:
            self.simulation_records.append(ep_rec)
            progress.update(1)
        progress.close()


    def get_summarized_results(self):
        return pd.DataFrame([rec.get_summarized_result() for rec in self.simulation_records])


    def render_records_to_video(
        self,
        output_path: Optional[str] = None,
        render_config: Optional[dict] = None,
    ) -> str:
        """
        Render all stored simulation records as a single combined MP4 video.

        All episodes are concatenated in order; each episode ends with a
        freeze-frame hold (``freeze_ms`` in render_config, default 500 ms).

        Parameters
        ----------
        output_path : str, optional
            Full path for the output file.  When omitted, a timestamped name
            is created under <project_root>/data/video/<model_name>/.
        render_config : dict, optional
            Passed through to EpisodeVideoRenderer.
            Keys: width, height, fps, freeze_ms, target_color.

        Returns
        -------
        str : path of the written video file.
        """
        if not self.simulation_records:
            raise ValueError("No simulation records to render. Run simulate() first.")

        n = len(self.simulation_records)
        if n > 50:
            print(
                f"[Renderer] Warning: {n} episodes queued for rendering. "
                "This may take a very long time."
            )

        if output_path is None:
            out_dir = (
                Path(__file__).resolve().parent.parent.parent
                / FOLDERS.DATA
                / "video"
                / self.model_name
            )
            output_path = str(out_dir / f"{get_compact_timestamp_str()}_n{n}.mp4")

        from ..visualizer.renderer import EpisodeVideoRenderer  # lazy import — avoids circular dependency

        renderer = EpisodeVideoRenderer(task=self.agent.game_env, config=render_config or {})

        return renderer.render_all(self.simulation_records, output_path)
    

def _simulate_single_episode_joblib(
    policy_path: str,
    env_class_name: str,
    env_config: dict,
    preset,
    deterministic: bool,
    resimulate_max_num: int,
    save_trajectory: bool,
    save_coarse_trajectory: bool = False,
) -> "AimandShootEpisodeRecorder":
    global _worker_model, _worker_agent, _worker_policy_path

    # Reload when: (a) first episode in this worker, or (b) checkpoint path changed.
    # loky reuses worker processes across separate Parallel() calls, so without this
    # check every checkpoint after the first would silently reuse the initial model.
    if _worker_model is None or _worker_policy_path != policy_path:
        _worker_policy_path = policy_path
        _worker_model = None  # ensure clean reload
        _worker_agent = None
        import os
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        _worker_model = _load_sac_for_inference(policy_path)
        env_class = getattr(vplayer, env_class_name)
        _worker_agent = env_class(config=deepcopy(env_config))

    # Sync coarse-trajectory recording flag with caller's request.
    _worker_agent._record_coarse = save_coarse_trajectory

    retry_count = 0
    while True:
        if preset is None:
            _worker_agent.update_env_presets()
        else:
            _worker_agent.update_env_presets(**preset)

        obs, _ = _worker_agent.reset()
        ep_rec = AimandShootEpisodeRecorder(
            _worker_agent,
            save_trajectory=save_trajectory,
            save_coarse_trajectory=save_coarse_trajectory,
        )

        done = False
        truncated = False
        while not (done or truncated):
            action, _ = _worker_model.predict(obs, deterministic=deterministic)
            obs, _, done, truncated, info = _worker_agent.step(action)
            ep_rec.record_step(_worker_agent, done, truncated, info)

        if truncated and retry_count < resimulate_max_num:
            retry_count += 1
            continue

        return ep_rec


def _monitor_points_to_visual_azel_deg(points_mm, head_position_mm):
    points_mm = np.asarray(points_mm, dtype=float)
    head = np.asarray(head_position_mm, dtype=float).reshape(3)
    pts = np.atleast_2d(points_mm)

    def _raw_azel(point_xy):
        dx = float(point_xy[0] - head[0])
        dy = float(point_xy[1] - head[1])
        depth = float(head[2])
        az = np.degrees(np.arctan2(dx, depth))
        el = np.degrees(np.arctan2(dy, np.sqrt(depth * depth + dx * dx)))
        return np.array([az, el], dtype=float)

    center = _raw_azel(np.zeros(2, dtype=float))
    out = np.array([_raw_azel(p) - center for p in pts], dtype=float)
    return out[0] if points_mm.ndim == 1 else out


def _clean_time_series(values, timestamps):
    values = np.asarray(values, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
    if values.shape[0] != timestamps.shape[0]:
        raise ValueError("Trajectory values and timestamps must have the same first dimension.")

    order = np.argsort(timestamps, kind="stable")
    t_sorted = timestamps[order]
    v_sorted = values[order]

    unique_t = []
    unique_v = []
    for t, v in zip(t_sorted, v_sorted):
        if unique_t and abs(t - unique_t[-1]) < 1e-9:
            unique_v[-1] = v
        else:
            unique_t.append(t)
            unique_v.append(v)
    return np.asarray(unique_v, dtype=float), np.asarray(unique_t, dtype=float)


def _interpolate_time_series(values, timestamps, start_ms=0.0, end_ms=None, interval_ms=1.0):
    values, timestamps = _clean_time_series(values, timestamps)
    if end_ms is None:
        end_ms = float(timestamps[-1])
    start_ms = float(start_ms)
    end_ms = float(end_ms)
    if timestamps.size < 2 or end_ms <= start_ms:
        return values[:1], np.array([start_ms], dtype=float)

    n = int(np.floor((end_ms - start_ms) / interval_ms)) + 1
    query = start_ms + np.arange(n, dtype=float) * interval_ms
    if query[-1] < end_ms:
        query = np.append(query, end_ms)
    return np_interp_nd(query, timestamps, values), query


def _legacy_sac_custom_objects():
    return {
        # Prevent deserializing legacy cloudpickled schedule objects that can
        # fail across Python/SB3 versions.
        "lr_schedule": (lambda _: 0.0),
        # Some migrated old checkpoints were saved with
        # learning.sac_policy.ModulatedSACPolicy. Supplying the current class
        # avoids requiring a top-level learning compatibility package.
        "policy_class": ModulatedSACPolicy,
    }


def _load_sac_for_inference(policy_path: str):
    """Load a SAC checkpoint for predict-only simulation without replay-buffer bulk."""
    model = SAC.load(
        policy_path,
        custom_objects=_legacy_sac_custom_objects(),
        buffer_size=1,
        device="cpu",
    )
    # SAC.load() constructs a training replay buffer even when the model is used
    # only for predict().  Drop it so many simulator workers do not waste RAM.
    model.replay_buffer = None
    return model



if __name__ == "__main__":
    from ..datamanager.load_emp_data import IJHCSExpDataLoader
    from ..agent.preset_converter import ijhcs_summary_to_env_presets

    MODEL_NAME = "ijhcs_default"
    NUM_EPISODES = 10
    PROFESSIONAL_PLAYER_INDEX = 1
    RANDOM_SEED = 42
    CKPT = "latest"
    PARAM_KEYS = (
        "param_motor_noise",
        "param_position_noise",
        "param_speed_noise",
        "param_clock_noise",
        "param_succ_reward",
        "param_fail_penalty",
        "param_reward_decay",
        "param_penalty_decay",
    )

    project_root = Path(__file__).resolve().parent.parent.parent
    inference_path = project_root / "empirical" / "ijhcs_inference_results.csv"
    inference = pd.read_csv(inference_path)
    player_param_rows = inference[
        (inference["expertise_group"] == "professional")
        & (inference["player_index"].astype(int) == int(PROFESSIONAL_PLAYER_INDEX))
    ].copy()
    if player_param_rows.empty:
        raise ValueError(f"No IJHCS inference parameters found for professional player {PROFESSIONAL_PLAYER_INDEX}.")

    loader = IJHCSExpDataLoader(load_traj=False)
    professional_trials = loader.load(
        return_traj=False,
        expertise_group="professional",
        player_index=PROFESSIONAL_PLAYER_INDEX,
    )
    if professional_trials.empty:
        raise ValueError(f"No IJHCS trials found for professional player {PROFESSIONAL_PLAYER_INDEX}.")

    sampled_trials = professional_trials.sample(
        n=min(NUM_EPISODES, len(professional_trials)),
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)
    task_presets = ijhcs_summary_to_env_presets(sampled_trials)

    def _parameter_preset_for_trial(row) -> dict:
        matches = player_param_rows
        for key in ("sensitivity_mode", "target_color_level"):
            matches = matches[matches[key].astype(str) == str(row[key])]
        if matches.empty:
            raise ValueError(
                "No parameter row found for "
                f"professional player {PROFESSIONAL_PLAYER_INDEX}, "
                f"sensitivity_mode={row['sensitivity_mode']}, "
                f"target_color_level={row['target_color_level']}."
            )
        param_row = matches.iloc[0]
        return {key: float(param_row[key]) for key in PARAM_KEYS}

    env_preset_list = [
        {**task_preset, **_parameter_preset_for_trial(trial)}
        for task_preset, (_, trial) in zip(task_presets, sampled_trials.iterrows())
    ]

    print(
        f"Simulating {len(env_preset_list)} IJHCS trials from professional player "
        f"{PROFESSIONAL_PLAYER_INDEX} with model {MODEL_NAME} ({CKPT})."
    )
    print("Sampled trajectory keys:", ", ".join(sampled_trials["trajectory_key"].astype(str).tolist()))

    simulator = AimandShootSimulator(model_name=MODEL_NAME, ckpt=CKPT)
    simulator.simulate(
        env_preset_list=env_preset_list,
        num_cpu=1,
        resimulate_max_num=5,
        deterministic=True,
        save_trajectory=True,
        save_coarse_trajectory=False,
    )
    print(f"Simulated {len(simulator.simulation_records)} episodes.")

    video_path = simulator.render_records_to_video(
        output_path=str(
            project_root
            / FOLDERS.DATA
            / "video"
            / simulator.model_name
            / f"{get_compact_timestamp_str()}_professional{PROFESSIONAL_PLAYER_INDEX}_n{len(env_preset_list)}.mp4"
        ),
        render_config=dict(
            width=1920,
            height=1080,
            fps=60,
            freeze_ms=500,
        )
    )
    print(f"Rendered video: {video_path}")
