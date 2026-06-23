"""
Virtual Aim-and-Shoot Player Agents for a Gym-based agent environment

Code written by June-Seop Yoon
SB3 == 2.1.0+ is required for the custom policy to work.
"""

import math as _math
_math_log = _math.log

import numpy as np
import scipy as sp
from collections import deque

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.envs.registration import EnvSpec

# from ..config.config import SIM, AGN
# from ..config.constant import FIELD, METRIC, AXIS, VECTOR

from ..configs.loader import CFG_AGENT_PROFILE
from ..configs.constants import TINTERVAL, TRUNCATE_ELEV_ANGLE
from ..configs.constants import AGENT_STATE_ALIAS as AS

from ..agent.module_aim import Aim
from ..agent.module_gaze import Gaze
from ..agent.module_perceive import Perceive
from ..agent.module_shoot import Shoot
from ..agent.task import AimandShootSpiderShotTask
from ..utils.mymath import (
    Convert, 
    # cos_sin_array, 
    linear_normalize, 
    linear_denormalize, 
    log_denormalize, 
    log_normalize,
    monitor_mm_to_view_angle_deg,
    view_angle_deg_to_monitor_mm,
)


class AnSPlayerAgentDefault(gym.Env):
    def __init__(
        self,
        spec_name: str = "ans-default",
        config_preset: str = "default",
        config: dict = None,
    ):
        self.viewer = None
        self.spec = EnvSpec(spec_name)
        
        # Configuration settings
        self.config = CFG_AGENT_PROFILE[config_preset] if config is None else config

        # Virtual game environment (FPS Spider Shot Task)
        self.game_env = AimandShootSpiderShotTask(self.config["task"])
        self.game_env_preset = dict(
            camera_azel_deg=None,
            target_pos_monitor_mm=None,
            target_pos_world=None,
            target_orbit_axis_deg=None,
            target_motion_dir_deg=None,
            target_speed_deg_s=None,
            target_radius_deg=None
        )
        self.init_game_env = dict()

        # Player state (parameters and dynamic state)
        self.player_state = self._sample_player_initial_state()
        self.player_state_preset = {key: None for key in self.player_state.keys()}

        # Observation and action space
        self._setup_observation_space()
        self._setup_action_space()

        # Initialization for episode management
        self.perceived_obs_state = {key: None for key in self.obs_total_features}
        self._reset_episode_variables()

        # Set False to skip coarse-trajectory cart2sphr calls and list appends
        # (logic-neutral: trajectory data simply won't be recorded).
        self._record_coarse = True

    
    def _reset_episode_variables(self):
        self.ongoing_motor_planned_movement_pos = np.zeros((TINTERVAL.BUMP // TINTERVAL.MUSCLE + 1, 2), dtype=np.float32)
        self.ongoing_motor_planned_movement_vel = np.zeros((TINTERVAL.BUMP // TINTERVAL.MUSCLE + 1, 2), dtype=np.float32)

        self.ongoing_motor_actual_movement_pos = np.zeros((TINTERVAL.BUMP // TINTERVAL.MUSCLE + 1, 2), dtype=np.float32)
        self.ongoing_motor_actual_movement_vel = np.zeros((TINTERVAL.BUMP // TINTERVAL.MUSCLE + 1, 2), dtype=np.float32)

        self.waiting_motor_planned_movement_pos = None
        self.waiting_motor_planned_movement_vel = None

        self.waiting_motor_actual_movement_pos = None
        self.waiting_motor_actual_movement_vel = None

        self.hand_trajectory_pos = [np.zeros((1, 2), dtype=np.float32)]
        self.hand_trajectory_vel = [np.zeros((1, 2), dtype=np.float32)]
        self.hand_trajectory_timestamp = [np.array([0.0], dtype=np.float32)]

        self.shooting_motor_plan_determined = False
        self.shoot_timing = 2 * self.config["task"]["time_limit_ms"] * 2 # initialized with a large value to ensure the agent will not shoot before determining the timing
    
        self.task_timer = 0
        self.task_time_offset = 0
        self.shoot_result = 0   # 0: miss, 1 : hit (will be decided on shoot occurrence)
        self.episode_truncated = False
        self.episode_terminated = False

        # Coarse (BUMP-resolution) camera and target trajectories.
        # Populated by reset() (initial entry) and _update_to_next_step() (one per step).
        self.camera_trajectory_azel: list = []
        self.target_trajectory_azel: list = []
        self.coarse_trajectory_timestamps: list = []


    def update_env_presets(self, **kwargs):
        self.update_game_env_preset(**kwargs)
        self.update_player_state_preset(**kwargs)


    def update_game_env_preset(self, **kwargs):
        for key in self.game_env_preset.keys():
            self.game_env_preset[key] = kwargs[key] if key in kwargs else None
    

    def update_player_state_preset(self, **kwargs):
        for key in self.player_state_preset.keys():
            self.player_state_preset[key] = kwargs[key] if key in kwargs else None


    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(int(seed))

        self._reset_game_env()
        self._reset_player_state()
        self._reset_episode_variables()

        # Time offset setting to account for the initial hand reaction time before the agent starts moving
        # Since reset, 100 ms = stationary hand, and then starts moving hand
        self.task_time_offset = self.player_state["hand_reaction_time"] - TINTERVAL.BUMP
        self.task_timer = self.task_time_offset

        # Record t=0: initial game condition before the reaction-time orbit.
        # This captures where the target appears at the very start of the episode.
        if self._record_coarse:
            self.camera_trajectory_azel.append(self.game_env.camera_azel_deg.copy())
            self.target_trajectory_azel.append(
                np.array(Convert.cart2sphr_scalar(*self.game_env.target_pos_world)[:2], dtype=np.float32)
            )
            self.coarse_trajectory_timestamps.append(0.0)

        self.game_env.orbit_target(self.task_timer, unit='ms', inplace=True)

        # Record t=task_time_offset: state after reaction-time orbit, just before first action.
        if self._record_coarse:
            self.camera_trajectory_azel.append(self.game_env.camera_azel_deg.copy())
            self.target_trajectory_azel.append(
                np.array(Convert.cart2sphr_scalar(*self.game_env.target_pos_world)[:2], dtype=np.float32)
            )
            self.coarse_trajectory_timestamps.append(float(self.task_timer))

        self.hand_trajectory_pos.append(np.zeros((1, 2), dtype=np.float32))
        self.hand_trajectory_vel.append(np.zeros((1, 2), dtype=np.float32))
        self.hand_trajectory_timestamp.append(np.array([self.task_timer], dtype=np.float32))

        # Perception & prediction
        target_vel_true = self.game_env.target_monitor_velocity()
        target_vel_hat = Perceive.velocity(
            vel=target_vel_true,
            pos=self.game_env.target_monitor_position(clip_ratio=1.2),
            noise=self.player_state["param_speed_noise"],
            head=self.player_state["head_position"]
        )
        speed_perception_error_ratio = np.linalg.norm(target_vel_hat) / np.linalg.norm(target_vel_true) \
            if np.linalg.norm(target_vel_true) > 1e-6 else 1.0
        target_bump_later_pos_monitor_mm = self._future_target_pos_monitor_mm(
            elapsed_time_ms=TINTERVAL.BUMP,
            assumed_orbit_speed_deg_s=self.game_env.target_speed_deg_s * speed_perception_error_ratio
        )

        # Update perceived_obs_state with initial state
        for feature in self.perceived_obs_state.keys():
            if feature in self.player_state:
                self.perceived_obs_state[feature] = self.player_state[feature]
            elif feature == "elapsed_time_ms":
                self.perceived_obs_state[feature] = self.task_timer
            elif feature == "target_radius_deg":
                self.perceived_obs_state[feature] = self.game_env.target_radius_deg
            elif feature == "target_radius_mm":
                self.perceived_obs_state[feature] = self.game_env.target_radius_mm
            elif feature == "hand_vel_mm_per_s":
                self.perceived_obs_state[feature] = np.array([0.0, 0.0], dtype=np.float32)
            elif feature == "target_bump_later_pos_monitor_mm":
                self.perceived_obs_state[feature] = target_bump_later_pos_monitor_mm
            elif feature == "target_vel_by_orbit_mm_s":
                self.perceived_obs_state[feature] = target_vel_hat
            else:
                raise ValueError(f"Unsupported observation feature: {feature}")
            
        info = {
            "perceived_obs_state": self.perceived_obs_state.copy(),
            "task_timer": self.task_timer,
            "shoot_result": self.shoot_result,
            "action": None
        }
        
        return self.get_np_obs(), info
    

    def _future_target_pos_monitor_mm(
        self, 
        elapsed_time_ms, 
        assumed_orbit_speed_deg_s,
        hand_displacement_mm=np.zeros(2),
        assumed_target_pos_monitor_mm=None
    ):
        return self.game_env.target_monitor_position(
            target_orbit_angle_deg=assumed_orbit_speed_deg_s * (elapsed_time_ms / 1000),
            hand_displacement_mm=hand_displacement_mm,
            initial_target_pos_monitor_mm=assumed_target_pos_monitor_mm,
            clip_ratio=2.0
        )

    
    def get_np_obs(self):
        """Fast obs builder using preallocated buffer and precomputed plan."""
        state = self.perceived_obs_state
        buf   = self._obs_buf
        for feat, col, kind, center, span, extra in self._obs_plan:
            raw = state[feat]
            if kind == 1:                          # linear vector
                raw_arr = np.asarray(raw, dtype=np.float64).reshape(-1)
                z = (2.0 * raw_arr - center) / span
                buf[col:col + raw_arr.size] = np.clip(z, -1.0, 1.0)
            elif kind == 0:                        # linear scalar
                z = (2.0 * float(raw) - center) / span
                buf[col] = z if -1.0 <= z <= 1.0 else (-1.0 if z < -1.0 else 1.0)
            else:                                  # kind == 2: loguniform scalar
                z = (2.0 * float(raw) - center) / span
                if z < -1.0: z = -1.0
                elif z > 1.0: z = 1.0
                a_m1, log_a = extra
                z = 2.0 * _math_log(1.0 + a_m1 * ((z + 1.0) * 0.5)) / log_a - 1.0
                buf[col] = z if -1.0 <= z <= 1.0 else (-1.0 if z < -1.0 else 1.0)
        return buf.copy()

    
    def get_np_obs_raw(self):
        obs_chunks = []
        for feature in self.obs_total_features:
            if feature not in self.perceived_obs_state:
                raise KeyError(f"Feature '{feature}' is missing in perceived_obs_state")
            if self.perceived_obs_state[feature] is None:
                raise ValueError(f"Feature '{feature}' is None in perceived_obs_state")

            raw = np.array(self.perceived_obs_state[feature], dtype=np.float32).reshape(-1)
            obs_chunks.append(raw)

        return np.concatenate(obs_chunks, axis=0).astype(np.float32)
    

    def get_hand_trajectory(self):
        return (
            np.concatenate(self.hand_trajectory_pos),
            np.concatenate(self.hand_trajectory_vel),
            np.concatenate(self.hand_trajectory_timestamp).astype(np.float32)
        )

    def supports_gaze_metrics(self) -> bool:
        return False

    def get_camera_trajectory_azel(self) -> "tuple[np.ndarray, np.ndarray]":
        """Camera az/el trajectory at BUMP resolution.

        Returns
        -------
        azel : (T, 2) float32  — [az_deg, el_deg]; T = 1 + n_decision_steps.
               For the shoot step, camera position at shoot moment (accurate).
               For non-shoot steps, position at end of BUMP.
        timestamps : (T,) float32  — ms; t_0 then t_0+BUMP … then shoot_time.
        """
        return (
            np.array(self.camera_trajectory_azel, dtype=np.float32),
            np.array(self.coarse_trajectory_timestamps, dtype=np.float32),
        )

    def get_target_trajectory_azel(self) -> "tuple[np.ndarray, np.ndarray]":
        """Target az/el trajectory at BUMP resolution.

        Returns
        -------
        azel : (T, 2) float32  — [az_deg, el_deg]; T = 1 + n_decision_steps.
               For the shoot step, pre-orbit position (what the agent tracked).
               For non-shoot steps, position at end of BUMP (after orbit_target).
        timestamps : (T,) float32  — ms; shares timestamps with camera trajectory.
        """
        return (
            np.array(self.target_trajectory_azel, dtype=np.float32),
            np.array(self.coarse_trajectory_timestamps, dtype=np.float32),
        )


    def step(self, action_normalized):
        action = self._unpack_action(action_normalized)
        state = self._step_init_state()

        self._perceive_and_predict(state, action)   # Inplace state update
        self._plan_hand_movement(state, action)
        self._plan_shoot_timing(state, action)
        self._check_shot_result(state, action)
        self._update_to_next_step(state, action)
        return self._terminate_or_truncate(state, action)



    def _unpack_action(self, action_normalized):
        action_val = linear_denormalize(action_normalized, *self.action_range.T)
        action_dict = dict(zip(self.action_list, action_val))

        # Processing action values if necessary
        if "th" in action_dict:
            action_dict["th"] = (round(action_dict["th"] // TINTERVAL.MUSCLE) * TINTERVAL.MUSCLE)
            action_dict["th"] = min(max(action_dict["th"], self.config["action"]["th"]["min"]), 
                                    self.config["action"]["th"]["max"])
        if "kc" in action_dict:
            action_dict["kc"] = int(action_dict["kc"] >= self.config["action"]["kc"]["threshold"])

        return action_dict


    def _step_init_state(self):
        return {
            AS.TARGET_BUMP_LATER_POS_MONITOR_MM: None,
            AS.TARGET_BUMP_TH_LATER_POS_MONITOR_MM: None,
            AS.TARGET_VEL_HAT_BY_ORBIT_MM_S: None,
            AS.MAX_HAND_SPEED: 0.0,
            AS.HAND_ACCEL_SUM: 0.0,
        }
    
    
    def _perceive_and_predict(self, state, action):
        # Perception & prediction
        # Total apparent velocity = orbit contribution + camera motion from current actual hand velocity.
        # The aim contribution (ideal plan) is subtracted below to isolate the orbit estimate.
        target_vel_true = self.game_env.target_monitor_velocity(
            hand_vel_mm=self.ongoing_motor_actual_movement_vel[0]
        )
        target_vel_hat = Perceive.velocity(
            vel=target_vel_true,
            pos=self.game_env.target_monitor_position(clip_ratio=1.1),
            noise=self.player_state["param_speed_noise"],
            head=self.player_state["head_position"]
        )
        target_vel_hat_by_aim_mm_s = self.game_env.target_monitor_velocity(
            target_speed_deg_s = 0.0,
            hand_vel_mm = self.ongoing_motor_planned_movement_vel[0]
        )
        target_vel_hat_by_orbit_mm_s = target_vel_hat - target_vel_hat_by_aim_mm_s

        clock_noise = Perceive.timing(1.0, self.player_state["param_clock_noise"])
        target_pos_after_bump_aim_mm = self.game_env.target_monitor_position(
            hand_displacement_mm=self.ongoing_motor_planned_movement_pos[-1] - \
                self.ongoing_motor_planned_movement_pos[0],
            target_orbit_angle_deg=0.0,
            clip_ratio=1.1
        )

        target_bump_later_pos_monitor_mm = target_pos_after_bump_aim_mm + \
            (TINTERVAL.BUMP / 1000) * target_vel_hat_by_orbit_mm_s * clock_noise
        target_bump_th_later_pos_monitor_mm = target_pos_after_bump_aim_mm + \
            ((TINTERVAL.BUMP + action["th"]) / 1000) * target_vel_hat_by_orbit_mm_s * clock_noise


        state[AS.TARGET_VEL_HAT_BY_ORBIT_MM_S] = target_vel_hat_by_orbit_mm_s
        state[AS.TARGET_BUMP_LATER_POS_MONITOR_MM] = target_bump_later_pos_monitor_mm
        state[AS.TARGET_BUMP_TH_LATER_POS_MONITOR_MM] = target_bump_th_later_pos_monitor_mm
        state[AS.CLOCK_NOISE] = clock_noise

    
    def _plan_hand_movement(self, state, action):
        if self.waiting_motor_actual_movement_pos is None:
            ideal_motor_plan_pos, ideal_motor_plan_vel = Aim.plan_hand_movement(
                self.ongoing_motor_actual_movement_pos[-1],
                self.ongoing_motor_actual_movement_vel[-1],
                self.game_env.camera_pos_world,
                self.game_env.camera_azel_deg + self.game_env.hand_sensi_deg_per_mm * (
                    self.ongoing_motor_planned_movement_pos[-1] - self.ongoing_motor_planned_movement_pos[0]
                ),
                state.get("crosshair_pos_hat_mm", self.game_env.crosshair_pos_mm),
                state[AS.TARGET_BUMP_TH_LATER_POS_MONITOR_MM],
                state[AS.TARGET_VEL_HAT_BY_ORBIT_MM_S],
                self.game_env.hand_sensi_deg_per_mm,
                self.game_env.camera_fov_deg,
                self.game_env.monitor_half_size_mm,
                planning_duration_ms=action["th"],
                execution_duration_ms=TINTERVAL.BUMP if not action["kc"] else action["th"],
                interval_ms=TINTERVAL.MUSCLE,
                max_camera_speed_deg_s=self.config["player"].get(
                    "max_camera_speed_deg_s",
                    self.config["player"]["max_hand_speed_mm_s"] * self.game_env.hand_sensi_deg_per_mm
                )
            )

            if action["kc"]:
                # Sufficiently expand motor plan
                expand_length = int((self.config["task"]["time_limit_ms"] - max(self.task_timer, 0)) // TINTERVAL.MUSCLE) + 2
                ideal_motor_plan_vel = np.pad(ideal_motor_plan_vel, ((0, expand_length), (0, 0)), mode='edge')
                ideal_motor_plan_pos = np.concatenate((
                    ideal_motor_plan_pos, 
                    ideal_motor_plan_pos[-1] + \
                        (ideal_motor_plan_vel[-1][:, np.newaxis] * np.arange(1, expand_length+1)).T * (TINTERVAL.MUSCLE / 1000)
                ))

            noisy_motor_plan_pos, noisy_motor_plan_vel = Aim.add_motor_noise(
                ideal_motor_plan_pos[0], ideal_motor_plan_vel,
                self.player_state["param_motor_noise"],
                interval_ms=TINTERVAL.MUSCLE
            )

            if action["kc"]:
                # Store motor plans on SHOOT MOVING
                _index = TINTERVAL.BUMP // TINTERVAL.MUSCLE
                self.waiting_motor_planned_movement_pos = ideal_motor_plan_pos[_index:]
                self.waiting_motor_planned_movement_vel = ideal_motor_plan_vel[_index:]
                self.waiting_motor_actual_movement_pos = noisy_motor_plan_pos[_index:]
                self.waiting_motor_actual_movement_vel = noisy_motor_plan_vel[_index:]
                self.shooting_motor_plan_determined = True

                # Maintain "next motor plan length" as tp
                ideal_motor_plan_pos = ideal_motor_plan_pos[:_index + 1]
                ideal_motor_plan_vel = ideal_motor_plan_vel[:_index + 1]
                noisy_motor_plan_pos = noisy_motor_plan_pos[:_index + 1]
                noisy_motor_plan_vel = noisy_motor_plan_vel[:_index + 1]

        else:   # Shoot motor plan in queue. Pop from queued motor plan and execute
            _index = TINTERVAL.BUMP // TINTERVAL.MUSCLE
            noisy_motor_plan_pos = self.waiting_motor_actual_movement_pos[:_index + 1].copy()
            noisy_motor_plan_vel = self.waiting_motor_actual_movement_vel[:_index + 1].copy()

            ideal_motor_plan_pos = self.waiting_motor_planned_movement_pos[:_index + 1].copy()
            ideal_motor_plan_pos += (noisy_motor_plan_pos[0] - ideal_motor_plan_pos[0])
            ideal_motor_plan_vel = self.waiting_motor_planned_movement_vel[:_index + 1].copy()

            self.waiting_motor_planned_movement_pos = self.waiting_motor_planned_movement_pos[_index:]
            self.waiting_motor_planned_movement_vel = self.waiting_motor_planned_movement_vel[_index:]
            self.waiting_motor_actual_movement_pos = self.waiting_motor_actual_movement_pos[_index:]
            self.waiting_motor_actual_movement_vel = self.waiting_motor_actual_movement_vel[_index:]
        
        state[AS.HTRAJ_IDEAL_P] = ideal_motor_plan_pos
        state[AS.HTRAJ_IDEAL_V] = ideal_motor_plan_vel
        state[AS.HTRAJ_ACTUAL_P] = noisy_motor_plan_pos
        state[AS.HTRAJ_ACTUAL_V] = noisy_motor_plan_vel


    def _plan_shoot_timing(self, state, action):
        if self.shooting_motor_plan_determined:
            self.shooting_motor_plan_determined = False
            self.shoot_timing = max(round((TINTERVAL.BUMP + action["tc"] * action["th"]) * state[AS.CLOCK_NOISE]), 1)
    

    def _check_shot_result(self, state, action):
        done = False
        if self.shoot_timing <= TINTERVAL.BUMP:
            # Interpolate ongoing motor plan for further calculations
            interp_plan_p, inter_plan_v = Aim.interpolate_plan(
                self.ongoing_motor_actual_movement_pos, self.ongoing_motor_actual_movement_vel,
                TINTERVAL.MUSCLE, TINTERVAL.INTERP1
            )
            shoot_index = max(int(np.ceil(self.shoot_timing / TINTERVAL.INTERP1)) - 1, 0)
            hand_displacement = interp_plan_p[shoot_index] - interp_plan_p[0]
            ending_target = self.game_env.target_monitor_position(
                hand_displacement_mm=hand_displacement,
                target_orbit_angle_deg=TINTERVAL.INTERP1 / 1000 * shoot_index * self.game_env.target_speed_deg_s
            )
            state[AS.SHOOT_ERROR_MM] = self.game_env.target_monitor_distance_mm(ending_target)
            _, state[AS.SHOOT_ERROR_DEG] = self.game_env.crosshair_on_target(
                target_pos_monitor_mm=ending_target, 
                camera_azel_deg=self.game_env.camera_azel_deg + self.game_env.hand_sensi_deg_per_mm * hand_displacement,
                return_dist=True
            )
            state[AS.SHOOT_RESULT] = state[AS.SHOOT_ERROR_MM] <= self.game_env.target_radius_mm
            state[AS.SHOOT_MOMENT_MS] = self.task_timer + self.shoot_timing
            state[AS.SHOOT_ENDPOINT_MM] = ending_target
            state[AS.HTRAJ_SEG_SHOOT_POS] = interp_plan_p[:shoot_index + 1]
            state[AS.HTRAJ_SEG_SHOOT_VEL] = inter_plan_v[:shoot_index + 1]
            state[AS.HTRAJ_SEG_SHOOT_TS] = np.arange(0, shoot_index + 1) * TINTERVAL.INTERP1
            state[AS.HAND_ACCEL_SUM] += Aim.accel_sum(inter_plan_v[:shoot_index + 1], TINTERVAL.INTERP1)
            state[AS.FINAL_HSPD_MM_S] = np.linalg.norm(inter_plan_v[shoot_index])
            done = True

        state[AS.DONE] = done
        state[AS.TRUNC] = False

    
    def _update_to_next_step(self, state, action):
        self.shoot_timing -= TINTERVAL.BUMP

        # Capture pre-update positions for coarse trajectory recording.
        if self._record_coarse:
            _cam_pre = self.game_env.camera_azel_deg.copy()
            _tgt_pre = np.array(
                Convert.cart2sphr_scalar(*self.game_env.target_pos_world)[:2], dtype=np.float32
            )

        self.game_env.orbit_target(dt=TINTERVAL.BUMP, unit='ms', inplace=True)
        self.game_env.move_hand(
            self.ongoing_motor_actual_movement_pos[-1] - self.ongoing_motor_actual_movement_pos[0], 
            self.ongoing_motor_actual_movement_vel[-1]
        )
        state[AS.MAX_HAND_SPEED] = max(
            state[AS.MAX_HAND_SPEED],
            np.max(np.linalg.norm(self.ongoing_motor_actual_movement_vel, axis=1))
        )
        if state[AS.DONE]:
            self.hand_trajectory_pos.append(state[AS.HTRAJ_SEG_SHOOT_POS][1:])
            self.hand_trajectory_vel.append(state[AS.HTRAJ_SEG_SHOOT_VEL][1:])
            self.hand_trajectory_timestamp.append(
                self.task_timer + state[AS.HTRAJ_SEG_SHOOT_TS][1:]
            )
            self.task_timer = state[AS.SHOOT_MOMENT_MS]
            self.shoot_result = int(state[AS.SHOOT_RESULT])
            if self._record_coarse:
                # Camera at shoot moment (accurate via shoot displacement);
                # target at pre-orbit position (= what the agent was tracking).
                _shoot_disp = state[AS.HTRAJ_SEG_SHOOT_POS][-1] - state[AS.HTRAJ_SEG_SHOOT_POS][0]
                self.camera_trajectory_azel.append(
                    (_cam_pre + self.game_env.hand_sensi_deg_per_mm * _shoot_disp).astype(np.float32)
                )
                self.target_trajectory_azel.append(_tgt_pre)
                self.coarse_trajectory_timestamps.append(float(self.task_timer))
        else:
            self.hand_trajectory_pos.append(self.ongoing_motor_actual_movement_pos[1:])
            self.hand_trajectory_vel.append(self.ongoing_motor_actual_movement_vel[1:])
            self.hand_trajectory_timestamp.append(
                self.task_timer + np.arange(TINTERVAL.MUSCLE, TINTERVAL.BUMP + 1, TINTERVAL.MUSCLE)
            )
            self.task_timer += TINTERVAL.BUMP
            self.shoot_result = 0
            state[AS.HAND_ACCEL_SUM] += Aim.accel_sum(self.ongoing_motor_actual_movement_vel, TINTERVAL.MUSCLE)
            if self._record_coarse:
                # Camera and target at end of BUMP (= state for next observation).
                self.camera_trajectory_azel.append(self.game_env.camera_azel_deg.copy().astype(np.float32))
                self.target_trajectory_azel.append(
                    np.array(Convert.cart2sphr_scalar(*self.game_env.target_pos_world)[:2], dtype=np.float32)
                )
                self.coarse_trajectory_timestamps.append(float(self.task_timer))


        state[AS.ACTION] = action
        state[AS.ELAPSED_TIME_MS] = self.task_timer

        self.perceived_obs_state.update(
            dict(
                target_bump_later_pos_monitor_mm = state[AS.TARGET_BUMP_LATER_POS_MONITOR_MM],
                target_vel_by_orbit_mm_s = state[AS.TARGET_VEL_HAT_BY_ORBIT_MM_S],
                hand_vel_mm_per_s = self.ongoing_motor_planned_movement_vel[-1],
                elapsed_time_ms = self.task_timer
            )
        )
        # Update plan
        self.ongoing_motor_planned_movement_pos = state[AS.HTRAJ_IDEAL_P]
        self.ongoing_motor_planned_movement_vel = state[AS.HTRAJ_IDEAL_V]
        self.ongoing_motor_actual_movement_pos = state[AS.HTRAJ_ACTUAL_P]
        self.ongoing_motor_actual_movement_vel = state[AS.HTRAJ_ACTUAL_V]
    

    def _terminate_or_truncate(self, state, action):
        # Forced termination
        target_elev_deg = Convert.cart2sphr_scalar(*self.game_env.target_pos_world)[1]
        target_escaped = (
            abs(target_elev_deg) > TRUNCATE_ELEV_ANGLE or
            (self.game_env.target_out_of_monitor() and self.task_timer > 0)
        )
        if (target_escaped and not state[AS.DONE]) or self.task_timer >= self.config["task"]["time_limit_ms"]:
            state[AS.DONE] = True
            state[AS.TRUNC] = True
            state[AS.SHOOT_RESULT] = False
            state[AS.SHOOT_ERROR_MM] = self.game_env.target_crosshair_distance_mm()
            state[AS.SHOOT_ENDPOINT_MM] = self.game_env.target_monitor_position()
            state[AS.SHOOT_MOMENT_MS] = self.task_timer
            _, state[AS.SHOOT_ERROR_DEG] = self.game_env.crosshair_on_target(return_dist=True)
            state[AS.FINAL_HSPD_MM_S] = float(np.linalg.norm(self.ongoing_motor_actual_movement_vel[-1]))
            self.result = 0
        
        reward = self._reward_function(state, action)
        info = dict(
            time = state[AS.SHOOT_MOMENT_MS] if (state[AS.DONE] and not state[AS.TRUNC]) else self.task_timer,
            result = state[AS.SHOOT_RESULT] if (state[AS.DONE] and not state[AS.TRUNC]) else False,
            is_success = state[AS.SHOOT_RESULT] if (state[AS.DONE] and not state[AS.TRUNC]) else False,
            is_truncated = bool(state[AS.TRUNC]),
            step_state = state
        )

        return self.get_np_obs(), reward, state[AS.DONE], state[AS.TRUNC], info


    def _reward_function(self, state, action):
        if state[AS.TRUNC]:
            state[AS.REWARD] = self.config["player"]["truncate_penalty"]
            return state[AS.REWARD]
        
        time_penalty = -self.player_state["param_time_penalty_weight"] * (TINTERVAL.BUMP / 1000) if not state[AS.DONE] else \
            -self.player_state["param_time_penalty_weight"] * ((state[AS.SHOOT_MOMENT_MS] % TINTERVAL.BUMP) / 1000)
        shoot_rew = 0 if not state[AS.DONE] else (
            self.player_state["param_succ_reward"] if state[AS.SHOOT_RESULT] else \
                -self.player_state["param_fail_penalty"]
        )
        state[AS.REWARD] = shoot_rew + time_penalty
        return state[AS.REWARD]



    def _setup_observation_space(self):
        n_dim = 0
        self.obs_conditioning_features = self.config["observation"]["policy_conditioning_features"]
        self.n_obs_conditioning_features = 0
        self.obs_total_features = self.obs_conditioning_features + self.config["observation"]["state_features"]
        self.observation_range = []
        feature_ranges = self.config["observation"].get("feature_ranges", {})

        for feature in self.obs_total_features:
            if feature in feature_ranges and not str(feature).startswith("param_"):
                lo = np.asarray(feature_ranges[feature]["min"], dtype=float).reshape(-1)
                hi = np.asarray(feature_ranges[feature]["max"], dtype=float).reshape(-1)
                if lo.shape != hi.shape:
                    raise ValueError(f"Observation range shape mismatch for feature '{feature}'.")
                for _lo, _hi in zip(lo, hi):
                    self.observation_range.append([_lo, _hi])
                n_dim += lo.size
                if feature in self.obs_conditioning_features:
                    self.n_obs_conditioning_features += lo.size

            elif str(feature).startswith("param_"):
                self.observation_range.append([
                    self.config["player"][feature]["min"],
                    self.config["player"][feature]["max"]
                ])
                n_dim += 1
                if feature in self.obs_conditioning_features:
                    self.n_obs_conditioning_features += 1

            elif feature == "elapsed_time_ms":
                self.observation_range.append([
                    -self.config["task"]["time_limit_ms"],
                    self.config["task"]["time_limit_ms"]
                ])
                n_dim += 1
                if feature in self.obs_conditioning_features:
                    self.n_obs_conditioning_features += 1

            elif feature == "target_bump_later_pos_monitor_mm":
                self.observation_range.append([
                    -self.config["task"]["monitor_width_mm"] / 2,
                    self.config["task"]["monitor_width_mm"] / 2
                ])
                self.observation_range.append([
                    -self.config["task"]["monitor_height_mm"] / 2,
                    self.config["task"]["monitor_height_mm"] / 2
                ])
                n_dim += 2
                if feature in self.obs_conditioning_features:
                    self.n_obs_conditioning_features += 2

            elif feature == "target_vel_by_orbit_mm_s":
                max_speed_mm_s = self.config["task"]["target_aspeed_deg_s_range"]["max"] \
                    / 40 * 150 * self.config["observation"].get("helper_taspd_range_scaler", 1.5) # rough estimation based on max target speed in deg/s, converted to mm/s with some margin
                self.observation_range.append([-max_speed_mm_s, max_speed_mm_s])
                self.observation_range.append([-max_speed_mm_s, max_speed_mm_s])    # Twice, since it's a 2D velocity vector on the monitor plane
                n_dim += 2
                if feature in self.obs_conditioning_features:
                    self.n_obs_conditioning_features += 2

            elif feature == "target_radius_deg":
                if "target_radius_deg_range" in self.config["task"]:
                    radius_max_deg = self.config["task"]["target_radius_deg_range"]["max"]
                else:
                    radius_max_deg = float(monitor_mm_to_view_angle_deg(
                        self.config["task"]["target_radius_mm_range"]["max"],
                        self.config["task"]["monitor_width_mm"],
                        self.config["task"]["cam_fov_deg_width"],
                    ))
                self.observation_range.append([
                    -radius_max_deg,
                    radius_max_deg,
                ])
                n_dim += 1
                if feature in self.obs_conditioning_features:
                    self.n_obs_conditioning_features += 1

            elif feature == "target_radius_mm":
                if "target_radius_mm_range" in self.config["task"]:
                    radius_max_mm = self.config["task"]["target_radius_mm_range"]["max"]
                else:
                    radius_max_mm = float(view_angle_deg_to_monitor_mm(
                        self.config["task"]["target_radius_deg_range"]["max"],
                        self.config["task"]["monitor_width_mm"],
                        self.config["task"]["cam_fov_deg_width"],
                    ))
                self.observation_range.append([-radius_max_mm, radius_max_mm])
                n_dim += 1
                if feature in self.obs_conditioning_features:
                    self.n_obs_conditioning_features += 1

            elif feature == "hand_vel_mm_per_s":
                self.observation_range.append([
                    -self.config["player"]["max_hand_speed_mm_s"],
                    self.config["player"]["max_hand_speed_mm_s"]
                ])
                self.observation_range.append([
                    -self.config["player"]["max_hand_speed_mm_s"],
                    self.config["player"]["max_hand_speed_mm_s"]
                ])
                n_dim += 2
                if feature in self.obs_conditioning_features:
                    self.n_obs_conditioning_features += 2

            elif feature == "gaze_pos_mm":
                self.observation_range.append([
                    -self.config["task"]["monitor_width_mm"] / 2,
                    self.config["task"]["monitor_width_mm"] / 2
                ])
                self.observation_range.append([
                    -self.config["task"]["monitor_height_mm"] / 2,
                    self.config["task"]["monitor_height_mm"] / 2
                ])
                n_dim += 2
                if feature in self.obs_conditioning_features:
                    self.n_obs_conditioning_features += 2

            elif feature == "head_position":
                head_cfg = self.config["player"]["head_position"]
                if isinstance(head_cfg, dict):
                    lo = head_cfg["min"]
                    hi = head_cfg["max"]
                else:
                    head = np.array(head_cfg, dtype=float)
                    span = np.array([65.0, 120.0, 167.0], dtype=float)
                    lo = head - span
                    hi = head + span
                for _lo, _hi in zip(lo, hi):
                    self.observation_range.append([_lo, _hi])
                n_dim += 3
                if feature in self.obs_conditioning_features:
                    self.n_obs_conditioning_features += 3

            else:
                raise NotImplementedError(f"Unsupported observation feature: {feature}")

        self.observation_range = np.array(self.observation_range, dtype=np.float32).T
        self.observation_space = spaces.Box(-np.ones(n_dim, dtype=np.float32), 
                                            np.ones(n_dim, dtype=np.float32), 
                                            dtype=np.float32)

        # ── Precompute fast-obs plan ──────────────────────────────────────────
        # Each entry: (feature, buf_col, kind, center, span, extra)
        #   kind 0 = linear scalar   → extra = None
        #   kind 1 = linear 2-vector → center/span are length-2 float64 arrays
        #   kind 2 = loguniform      → extra = (a_minus_1, log_a)
        self._obs_buf  = np.zeros(n_dim, dtype=np.float32)
        _plan, _col = [], 0
        for _feat in self.obs_total_features:
            if _feat in ("target_bump_later_pos_monitor_mm",
                         "target_vel_by_orbit_mm_s", "hand_vel_mm_per_s",
                         "gaze_pos_mm", "head_position"):
                _width = 3 if _feat == "head_position" else 2
                _lo = self.observation_range[0, _col:_col+_width].astype(np.float64)
                _hi = self.observation_range[1, _col:_col+_width].astype(np.float64)
                _plan.append((_feat, _col, 1,
                              (_lo + _hi), (_hi - _lo), None))
                _col += _width
            else:
                _lo = float(self.observation_range[0, _col])
                _hi = float(self.observation_range[1, _col])
                _ctr, _spn = _lo + _hi, _hi - _lo
                if _feat.startswith("param_"):
                    _pcfg = self.config["player"][_feat]
                    if isinstance(_pcfg, dict) and _pcfg.get("type") == "loguniform":
                        _sc = float(_pcfg.get("scale", 1))
                        if _sc == 0.0:
                            _plan.append((_feat, _col, 0, _ctr, _spn, None))
                        else:
                            _a  = _math.e * _sc
                            _plan.append((_feat, _col, 2, _ctr, _spn,
                                          (_a - 1.0, _math_log(_a))))
                    else:
                        _plan.append((_feat, _col, 0, _ctr, _spn, None))
                else:
                    _plan.append((_feat, _col, 0, _ctr, _spn, None))
                _col += 1
        self._obs_plan = _plan



    def _setup_action_space(self):
        self.action_list = self.config["action"]["list"]
        self.action_space = spaces.Box(
            -np.ones(len(self.action_list), dtype=np.float32),
            np.ones(len(self.action_list), dtype=np.float32),
            dtype=np.float32
        )
        self.action_range = np.vstack((
            [self.config["action"][action]["min"] for action in self.action_list],
            [self.config["action"][action]["max"] for action in self.action_list]
        ), dtype=np.float32).T


    def _reset_game_env(self):
        # task.reset() already includes target_radius_mm in its return dict
        # (computed via the @property from target_radius_deg and the task config),
        # so init_game_env automatically contains it without any extra work here.
        self.init_game_env = self.game_env.reset(**self.game_env_preset)


    def _reset_player_state(self):
        if all(self.player_state_preset[key] is not None for key in self.player_state.keys()):
            self.player_state = {
                key: self.player_state_preset[key]
                for key in self.player_state.keys()
            }
            return

        player_state = self._sample_player_initial_state()
        for key in self.player_state.keys():
            if self.player_state_preset[key] is not None:
                player_state[key] = self.player_state_preset[key]
        self.player_state = player_state

    
    def get_initial_conditions(self):
        return {**self.init_game_env.copy(), **self.player_state.copy()}


    def _sample_player_initial_state(self):
        return dict(
            head_position = self._sample_head_position(),
            hand_reaction_time = self._sample_hand_reaction_time(),
            **self._sample_user_params()
        )
    

    def _sample_user_params(self):
        params = {}
        for key in self.config["player"]:
            if key.startswith("param_"):
                if isinstance(self.config["player"][key], dict):
                    if self.config["player"][key]["type"] == "uniform":
                        params[key] = np.random.uniform(
                            low=self.config["player"][key]["min"],
                            high=self.config["player"][key]["max"]
                        )
                    elif self.config["player"][key]["type"] == "loguniform":
                        params[key] = log_denormalize(
                            np.random.uniform(-1.0, 1.0),
                            self.config["player"][key]["min"],
                            self.config["player"][key]["max"],
                            self.config["player"][key]["scale"]
                        )
                    else:
                        raise NotImplementedError(f"Unsupported distribution type for {key}: {self.config['player'][key]['type']}")
                else:
                    params[key] = self.config["player"][key]
        return params


    def _sample_head_position(self):
        cfg = self.config["player"]["head_position"]
        if not isinstance(cfg, dict):
            return np.array(cfg, dtype=float) # mm, (x, y, z)

        cfg_type = cfg.get("type")
        lo = np.array(cfg["min"], dtype=float)
        hi = np.array(cfg["max"], dtype=float)
        if cfg_type == "norm":
            head = np.random.normal(
                np.array(cfg["mean"], dtype=float),
                np.array(cfg["std"], dtype=float),
            )
        elif cfg_type == "uniform":
            head = np.random.uniform(lo, hi)
        else:
            raise NotImplementedError(f"Unsupported head_position distribution type: {cfg_type}")
        return np.clip(head, lo, hi)


    def _sample_hand_reaction_time(self):
        if isinstance(self.config["player"]["hand_reaction_time"], dict):
            if self.config["player"]["hand_reaction_time"]["type"] == "skewnorm":
                while True:
                    a = self.config["player"]["hand_reaction_time"]["alpha"]
                    loc = self.config["player"]["hand_reaction_time"]["mean"]
                    scale = self.config["player"]["hand_reaction_time"]["std"]
                    hrt = round(sp.stats.skewnorm.rvs(a=a, loc=loc, scale=scale))
                    if self.config["player"]["hand_reaction_time"]["min"] <= hrt \
                        <= self.config["player"]["hand_reaction_time"]["max"]:
                        break
                return hrt
            else:
                raise NotImplementedError(f"Unsupported hand_reaction_time distribution type: {self.config['player']['hand_reaction_time']['type']}")
        else:
            return int(self.config["player"]["hand_reaction_time"]) # milliseconds



class AnSPlayerAgentLegacy(AnSPlayerAgentDefault):
    def __init__(
        self,
        spec_name: str = "ans-legacy",
        config_preset: str = "legacy",
        config: dict = None,
    ):
        super().__init__(spec_name, config_preset, config)

    
    def _reward_function(self, state, action):
        if state[AS.TRUNC]:
            state[AS.REWARD] = self.config["player"]["truncate_penalty"]
            return state[AS.REWARD]
        
        if state[AS.DONE]:
            if state[AS.SHOOT_RESULT]:
                state[AS.REWARD] = self.player_state["param_succ_reward"] * ((1-self.player_state["param_reward_decay"]/100) ** (state[AS.SHOOT_MOMENT_MS] / 1000))
            else:
                state[AS.REWARD] = -self.player_state["param_fail_penalty"] * ((1-self.player_state["param_penalty_decay"]/100) ** (state[AS.SHOOT_MOMENT_MS] / 1000))
        else:
            state[AS.REWARD] = 0.0
        
        return state[AS.REWARD]


class AnSPlayerOriginalAgent(AnSPlayerAgentLegacy):
    _STATE_TARGET_POS_0_HAT = "target_pos_0_hat_mm"
    _STATE_CROSSHAIR_POS_0_HAT = "crosshair_pos_hat_mm"
    _STATE_GAZE_TIME = "gaze_traj_timestamp"
    _STATE_GAZE_TRAJ = "gaze_traj_pos"
    _STATE_GAZE_DEST_IDEAL = "gaze_dest_ideal_mm"
    _STATE_GAZE_DEST_NOISY = "gaze_dest_noisy_mm"

    def __init__(
        self,
        spec_name: str = "ans-original",
        config_preset: str = "original",
        config: dict = None,
    ):
        super().__init__(spec_name, config_preset, config)

    
    def _reset_episode_variables(self):
        super()._reset_episode_variables()
        self.delayed_time = 0
        self.gaze_cooldown = 0
        self.bump_plan_wait = 0
        self.gaze_trajectory_pos = [np.zeros((1, 2), dtype=np.float32)]
        self.gaze_trajectory_timestamp = [np.array([0.0], dtype=np.float32)]


    def reset(self, seed=None, **kwargs):
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            np.random.seed(int(seed))

        self._reset_game_env()
        self._reset_player_state()
        self._reset_episode_variables()

        self.delayed_time = 4 * TINTERVAL.BUMP - self.player_state["hand_reaction_time"]
        self.gaze_cooldown = self.player_state["gaze_reaction_time"] + self.delayed_time
        self.bump_plan_wait = 3
        self.task_time_offset = -self.delayed_time
        self.task_timer = -self.delayed_time

        if self._record_coarse:
            self.camera_trajectory_azel.append(self.game_env.camera_azel_deg.copy())
            self.target_trajectory_azel.append(
                np.array(Convert.cart2sphr_scalar(*self.game_env.target_pos_world)[:2], dtype=np.float32)
            )
            self.coarse_trajectory_timestamps.append(0.0)

        self.game_env.orbit_target(-self.delayed_time, unit="ms", inplace=True)

        n_stationary_bump = int(min(self.gaze_cooldown, 3 * TINTERVAL.BUMP) // TINTERVAL.BUMP)
        skipped_ms = n_stationary_bump * TINTERVAL.BUMP
        self.gaze_cooldown -= skipped_ms
        self.bump_plan_wait -= n_stationary_bump
        self.task_timer += skipped_ms
        self.game_env.orbit_target(skipped_ms, unit="ms", inplace=True)

        if self._record_coarse:
            self.camera_trajectory_azel.append(self.game_env.camera_azel_deg.copy())
            self.target_trajectory_azel.append(
                np.array(Convert.cart2sphr_scalar(*self.game_env.target_pos_world)[:2], dtype=np.float32)
            )
            self.coarse_trajectory_timestamps.append(float(self.task_timer))

        stationary_hand_len = skipped_ms // TINTERVAL.MUSCLE + 1
        self.hand_trajectory_pos = [np.zeros((stationary_hand_len, 2), dtype=np.float32)]
        self.hand_trajectory_vel = [np.zeros((stationary_hand_len, 2), dtype=np.float32)]
        self.hand_trajectory_timestamp = [
            (np.arange(0, skipped_ms + 1, TINTERVAL.MUSCLE, dtype=np.float32) - self.delayed_time)
        ]
        gaze_initial = np.asarray(self.player_state["gaze_position"], dtype=np.float32)
        self.gaze_trajectory_pos = [
            np.array([gaze_initial, gaze_initial], dtype=np.float32)
        ]
        self.gaze_trajectory_timestamp = [
            (np.array([0.0, float(skipped_ms)], dtype=np.float32) - self.delayed_time)
        ]

        target_pos_0_hat, target_sigma = Perceive.position(
            self.game_env.target_pos_monitor_mm,
            self.player_state["gaze_position"],
            self.player_state["param_position_noise"],
            head=self.player_state["head_position"],
            monitor_qt=self.game_env.monitor_half_size_mm,
            return_sigma=True,
        )
        crosshair_pos_0_hat, crosshair_sigma = Perceive.position(
            self.game_env.crosshair_pos_mm,
            self.player_state["gaze_position"],
            self.player_state["param_position_noise"],
            head=self.player_state["head_position"],
            monitor_qt=self.game_env.monitor_half_size_mm,
            return_sigma=True,
        )
        target_vel_true = self.game_env.target_monitor_velocity()
        target_vel_hat = Perceive.velocity(
            vel=target_vel_true,
            pos=self.game_env.target_pos_monitor_mm,
            noise=self.player_state["param_speed_noise"],
            head=self.player_state["head_position"],
        )
        speed_error_ratio = (
            np.linalg.norm(target_vel_hat) / np.linalg.norm(target_vel_true)
            if np.linalg.norm(target_vel_true) > 1e-6 else 1.0
        )
        target_bump_later_pos_monitor_mm = self.game_env.target_monitor_position(
            initial_target_pos_monitor_mm=target_pos_0_hat,
            hand_displacement_mm=np.zeros(2),
            target_orbit_angle_deg=self.game_env.target_speed_deg_s * speed_error_ratio * TINTERVAL.BUMP / 1000,
            clip_ratio=2.0,
        )

        for feature in self.perceived_obs_state.keys():
            if feature in self.player_state:
                self.perceived_obs_state[feature] = self.player_state[feature]
            elif feature == "elapsed_time_ms":
                self.perceived_obs_state[feature] = self.task_timer
            elif feature == "target_radius_deg":
                self.perceived_obs_state[feature] = self.game_env.target_radius_deg
            elif feature == "target_radius_mm":
                self.perceived_obs_state[feature] = self.game_env.target_radius_mm
            elif feature == "hand_vel_mm_per_s":
                self.perceived_obs_state[feature] = np.array([0.0, 0.0], dtype=np.float32)
            elif feature == "target_bump_later_pos_monitor_mm":
                self.perceived_obs_state[feature] = target_bump_later_pos_monitor_mm
            elif feature == "target_vel_by_orbit_mm_s":
                self.perceived_obs_state[feature] = target_vel_hat
            elif feature == "gaze_pos_mm":
                self.perceived_obs_state[feature] = self.player_state["gaze_position"]
            else:
                raise ValueError(f"Unsupported observation feature: {feature}")

        info = {
            "perceived_obs_state": self.perceived_obs_state.copy(),
            "task_timer": self.task_timer,
            "shoot_result": self.shoot_result,
            "action": None,
            "belief": {
                "target_sigma": target_sigma,
                "crosshair_sigma": crosshair_sigma,
                "target_pos_hat_error": target_pos_0_hat - self.game_env.target_pos_monitor_mm,
                "target_vel_hat_error": target_vel_hat - target_vel_true,
                "crosshair_pos_hat_error": crosshair_pos_0_hat - self.game_env.crosshair_pos_mm,
            },
        }
        return self.get_np_obs(), info


    def step(self, action_normalized):
        action = self._unpack_action(action_normalized)
        state = self._step_init_state()

        self._perceive_and_predict(state, action)
        self._plan_hand_movement(state, action)
        self._plan_gaze_movement(state, action)
        self._plan_shoot_timing(state, action)
        self._check_shot_result(state, action)
        self._update_to_next_step(state, action)
        return self._terminate_or_truncate(state, action)


    def _unpack_action(self, action_normalized):
        action_val = linear_denormalize(action_normalized, *self.action_range.T)
        action_dict = dict(zip(self.action_list, action_val))

        action_dict["th"] = (round(action_dict["th"]) // TINTERVAL.MUSCLE) * TINTERVAL.MUSCLE
        action_dict["th"] = min(max(action_dict["th"], self.config["action"]["th"]["min"]),
                                self.config["action"]["th"]["max"])
        action_dict["kc"] = int(action_dict["kc"] >= self.config["action"]["kc"]["threshold"])
        action_dict["kg"] = float(np.sqrt(max(action_dict["kg"], 0.0)))
        return action_dict


    def _step_init_state(self):
        state = super()._step_init_state()
        state.update({
            self._STATE_TARGET_POS_0_HAT: None,
            self._STATE_CROSSHAIR_POS_0_HAT: None,
            self._STATE_GAZE_TIME: None,
            self._STATE_GAZE_TRAJ: None,
            self._STATE_GAZE_DEST_IDEAL: None,
            self._STATE_GAZE_DEST_NOISY: None,
        })
        return state


    def _perceive_and_predict(self, state, action):
        target_pos_0_hat, pos_sigma = Perceive.position(
            self.game_env.target_pos_monitor_mm,
            self.player_state["gaze_position"],
            self.player_state["param_position_noise"],
            head=self.player_state["head_position"],
            monitor_qt=self.game_env.monitor_half_size_mm,
            return_sigma=True,
        )
        crosshair_pos_0_hat = Perceive.position(
            self.game_env.crosshair_pos_mm,
            self.player_state["gaze_position"],
            self.player_state["param_position_noise"],
            head=self.player_state["head_position"],
            monitor_qt=self.game_env.monitor_half_size_mm,
        )

        target_vel_true = self.game_env.target_monitor_velocity(
            initial_target_pos_monitor_mm=target_pos_0_hat,
            hand_vel_mm=self.ongoing_motor_actual_movement_vel[0],
        )
        target_vel_hat = Perceive.velocity(
            vel=target_vel_true,
            pos=target_pos_0_hat,
            noise=self.player_state["param_speed_noise"],
            head=self.player_state["head_position"],
        )
        target_vel_hat_by_aim_mm_s = self.game_env.target_monitor_velocity(
            target_speed_deg_s=0.0,
            hand_vel_mm=self.ongoing_motor_planned_movement_vel[0],
        )
        target_vel_hat_by_orbit_mm_s = target_vel_hat - target_vel_hat_by_aim_mm_s

        clock_noise = Perceive.timing(1.0, self.player_state["param_clock_noise"])
        target_pos_after_bump_aim_mm = self.game_env.target_monitor_position(
            initial_target_pos_monitor_mm=target_pos_0_hat,
            hand_displacement_mm=self.ongoing_motor_planned_movement_pos[-1] -
                self.ongoing_motor_planned_movement_pos[0],
            target_orbit_angle_deg=0.0,
            clip_ratio=1.1,
        )
        target_bump_later_pos_monitor_mm = target_pos_after_bump_aim_mm + \
            (TINTERVAL.BUMP / 1000) * target_vel_hat_by_orbit_mm_s * clock_noise
        target_bump_th_later_pos_monitor_mm = target_pos_after_bump_aim_mm + \
            ((TINTERVAL.BUMP + action["th"]) / 1000) * target_vel_hat_by_orbit_mm_s * clock_noise

        state[AS.TARGET_VEL_HAT_BY_ORBIT_MM_S] = target_vel_hat_by_orbit_mm_s
        state[AS.TARGET_BUMP_LATER_POS_MONITOR_MM] = target_bump_later_pos_monitor_mm
        state[AS.TARGET_BUMP_TH_LATER_POS_MONITOR_MM] = target_bump_th_later_pos_monitor_mm
        state[AS.CLOCK_NOISE] = clock_noise
        state[self._STATE_TARGET_POS_0_HAT] = target_pos_0_hat
        state[self._STATE_CROSSHAIR_POS_0_HAT] = crosshair_pos_0_hat
        state["crosshair_pos_hat_mm"] = crosshair_pos_0_hat
        state["target_pos_hat_error_mm"] = target_pos_0_hat - self.game_env.target_pos_monitor_mm
        state["target_pos_sigma_deg"] = pos_sigma


    def _plan_hand_movement(self, state, action):
        if self.bump_plan_wait > 0:
            self.bump_plan_wait -= 1
            shape = (TINTERVAL.BUMP // TINTERVAL.MUSCLE + 1, 2)
            zeros = np.zeros(shape, dtype=np.float32)
            state[AS.HTRAJ_IDEAL_P] = zeros.copy()
            state[AS.HTRAJ_IDEAL_V] = zeros.copy()
            state[AS.HTRAJ_ACTUAL_P] = zeros.copy()
            state[AS.HTRAJ_ACTUAL_V] = zeros.copy()
            return
        super()._plan_hand_movement(state, action)


    def _plan_gaze_movement(self, state, action):
        gaze_pos = np.asarray(self.player_state["gaze_position"], dtype=float)
        if self.gaze_cooldown > TINTERVAL.BUMP:
            self.gaze_cooldown -= TINTERVAL.BUMP
            gaze_time = np.array([self.task_timer, self.task_timer + TINTERVAL.BUMP], dtype=np.float32)
            gaze_traj = np.array([gaze_pos, gaze_pos], dtype=np.float32)
            gaze_dest_ideal = gaze_pos
            gaze_dest_noisy = gaze_pos
        else:
            gaze_dest_ideal = (
                state[self._STATE_CROSSHAIR_POS_0_HAT] * (1.0 - action["kg"]) +
                state[AS.TARGET_BUMP_LATER_POS_MONITOR_MM] * action["kg"]
            )
            gaze_dest_noisy = Gaze.gaze_landing_point(
                gaze_pos,
                gaze_dest_ideal,
                head=self.player_state["head_position"],
                monitor_qt=self.game_env.monitor_half_size_mm,
            )
            gaze_time, gaze_traj = Gaze.gaze_plan(
                gaze_pos,
                gaze_dest_noisy,
                delay=self.gaze_cooldown,
                exe_until=TINTERVAL.BUMP,
                head=self.player_state["head_position"],
            )
            self.gaze_cooldown = 0
            gaze_time = self.task_timer + gaze_time
            gaze_dest_noisy = gaze_traj[-1]

        state[self._STATE_GAZE_TIME] = np.asarray(gaze_time, dtype=np.float32)
        state[self._STATE_GAZE_TRAJ] = np.asarray(gaze_traj, dtype=np.float32)
        state[self._STATE_GAZE_DEST_IDEAL] = np.asarray(gaze_dest_ideal, dtype=np.float32)
        state[self._STATE_GAZE_DEST_NOISY] = np.asarray(gaze_dest_noisy, dtype=np.float32)


    def _update_to_next_step(self, state, action):
        super()._update_to_next_step(state, action)
        self.player_state["gaze_position"] = np.asarray(state[self._STATE_GAZE_DEST_NOISY], dtype=float)
        self.gaze_trajectory_pos.append(state[self._STATE_GAZE_TRAJ][1:])
        self.gaze_trajectory_timestamp.append(state[self._STATE_GAZE_TIME][1:])
        self.perceived_obs_state.update(
            dict(
                gaze_pos_mm=state[self._STATE_GAZE_DEST_IDEAL],
                head_position=self.player_state["head_position"],
            )
        )


    def get_gaze_trajectory(self):
        return (
            np.concatenate(self.gaze_trajectory_pos),
            np.concatenate(self.gaze_trajectory_timestamp).astype(np.float32),
        )


    def supports_gaze_metrics(self) -> bool:
        return True


    def _sample_player_initial_state(self):
        return dict(
            head_position=self._sample_head_position(),
            gaze_position=self._sample_gaze_position(),
            hand_reaction_time=self._sample_hand_reaction_time(),
            gaze_reaction_time=self._sample_gaze_reaction_time(),
            **self._sample_user_params(),
        )


    def _sample_head_position(self):
        cfg = self.config["player"]["head_position"]
        if not isinstance(cfg, dict):
            return np.array(cfg, dtype=float)
        if cfg["type"] != "norm":
            raise NotImplementedError(f"Unsupported head_position distribution type: {cfg['type']}")
        head = np.random.normal(np.array(cfg["mean"], dtype=float), np.array(cfg["std"], dtype=float))
        return np.clip(head, np.array(cfg["min"], dtype=float), np.array(cfg["max"], dtype=float))


    def _sample_gaze_position(self):
        cfg = self.config["player"]["gaze_position"]
        if not isinstance(cfg, dict):
            return np.array(cfg, dtype=float)
        if cfg["type"] not in ("radial_normal", "uniform"):
            raise NotImplementedError(f"Unsupported gaze_position distribution type: {cfg['type']}")
        angle = np.random.uniform(0.0, 2.0 * np.pi)
        if cfg["type"] == "uniform":
            dev = np.random.uniform(cfg["min"], cfg["max"])
        else:
            while True:
                dev = np.random.normal(cfg["mean"], cfg["std"])
                if cfg["min"] <= dev <= cfg["max"]:
                    break
        return np.array([np.cos(angle), np.sin(angle)], dtype=float) * dev


    def _sample_gaze_reaction_time(self):
        cfg = self.config["player"]["gaze_reaction_time"]
        if isinstance(cfg, dict):
            if cfg["type"] == "skewnorm":
                while True:
                    grt = round(sp.stats.skewnorm.rvs(
                        a=cfg["alpha"],
                        loc=cfg["mean"],
                        scale=cfg["std"],
                    ))
                    if cfg["min"] <= grt <= cfg["max"]:
                        return int(grt)
            raise NotImplementedError(f"Unsupported gaze_reaction_time distribution type: {cfg['type']}")
        return int(cfg)


class AnSLegacyShootingModule(AnSPlayerOriginalAgent):
    _SHOOT_TIMING_DEFAULTS = {
        "cmu": 0.185,
        "nu": 19.931,
        "delta": 0.399,
    }

    def _plan_shoot_timing(self, state, action):
        if self.shooting_motor_plan_determined:
            self.shooting_motor_plan_determined = False
            tavel_hat = self.game_env.target_monitor_velocity(
                target_speed_deg_s=0.0,
                hand_vel_mm=self.ongoing_motor_planned_movement_vel[0],
            )
            shoot_timing_s = Shoot.sample_shoot_timing(
                tpos=state[self._STATE_TARGET_POS_0_HAT],
                cpos=state[self._STATE_CROSSHAIR_POS_0_HAT],
                tgvel=state[AS.TARGET_VEL_HAT_BY_ORBIT_MM_S],
                tavel=tavel_hat,
                trad=self.game_env.target_radius_mm,
                param_clock_noise=self._shoot_scalar("param_clock_noise", default=0.056),
                cmu=self._shoot_scalar("param_shoot_cmu", default=self._SHOOT_TIMING_DEFAULTS["cmu"]),
                nu=self._shoot_scalar("param_shoot_nu", default=self._SHOOT_TIMING_DEFAULTS["nu"]),
                delta=self._shoot_scalar("param_shoot_delta", default=self._SHOOT_TIMING_DEFAULTS["delta"]),
                interp_interval_ms=TINTERVAL.INTERP1,
            )
            self.shoot_timing = max(int(round(float(shoot_timing_s) * 1000.0)), int(TINTERVAL.INTERP1))


    def _shoot_scalar(self, name, default):
        player_cfg = self.config.get("player", {})
        if name in self.player_state:
            return float(np.asarray(self.player_state[name], dtype=float))
        cfg_value = player_cfg.get(name)
        if cfg_value is not None and not isinstance(cfg_value, dict):
            return float(cfg_value)
        return float(default)
    

class AnSAblatedEfCopy(AnSPlayerOriginalAgent):
    def _perceive_and_predict(self, state, action):
        target_pos_0_hat, pos_sigma = Perceive.position(
            self.game_env.target_pos_monitor_mm,
            self.player_state["gaze_position"],
            self.player_state["param_position_noise"],
            head=self.player_state["head_position"],
            monitor_qt=self.game_env.monitor_half_size_mm,
            return_sigma=True,
        )
        crosshair_pos_0_hat = Perceive.position(
            self.game_env.crosshair_pos_mm,
            self.player_state["gaze_position"],
            self.player_state["param_position_noise"],
            head=self.player_state["head_position"],
            monitor_qt=self.game_env.monitor_half_size_mm,
        )

        target_vel_true_by_orbit = self.game_env.target_monitor_velocity(
            initial_target_pos_monitor_mm=target_pos_0_hat,
            hand_vel_mm=np.zeros(2, dtype=np.float32)
        )
        target_vel_hat_by_orbit_mm_s = Perceive.velocity(
            vel=target_vel_true_by_orbit,
            pos=target_pos_0_hat,
            noise=self.player_state["param_speed_noise"],
            head=self.player_state["head_position"]
        )

        clock_noise = Perceive.timing(1.0, self.player_state["param_clock_noise"])
        target_pos_after_bump_aim_mm = self.game_env.target_monitor_position(
            initial_target_pos_monitor_mm=target_pos_0_hat,
            hand_displacement_mm=self.ongoing_motor_planned_movement_pos[-1] - \
                self.ongoing_motor_planned_movement_pos[0],
            target_orbit_angle_deg=0.0,
            clip_ratio=1.1
        )

        target_bump_later_pos_monitor_mm = target_pos_after_bump_aim_mm + \
            (TINTERVAL.BUMP / 1000) * target_vel_hat_by_orbit_mm_s * clock_noise
        target_bump_th_later_pos_monitor_mm = target_pos_after_bump_aim_mm + \
            ((TINTERVAL.BUMP + action["th"]) / 1000) * target_vel_hat_by_orbit_mm_s * clock_noise

        state[AS.TARGET_VEL_HAT_BY_ORBIT_MM_S] = target_vel_hat_by_orbit_mm_s
        state[AS.TARGET_BUMP_LATER_POS_MONITOR_MM] = target_bump_later_pos_monitor_mm
        state[AS.TARGET_BUMP_TH_LATER_POS_MONITOR_MM] = target_bump_th_later_pos_monitor_mm
        state[AS.CLOCK_NOISE] = clock_noise
        state[self._STATE_TARGET_POS_0_HAT] = target_pos_0_hat
        state[self._STATE_CROSSHAIR_POS_0_HAT] = crosshair_pos_0_hat
        state["crosshair_pos_hat_mm"] = crosshair_pos_0_hat
        state["target_pos_hat_error_mm"] = target_pos_0_hat - self.game_env.target_pos_monitor_mm
        state["target_pos_sigma_deg"] = pos_sigma


class AnSOriginalBaseline(AnSPlayerAgentLegacy):
    _STATE_TARGET_POS_0_HAT = "target_pos_0_hat_mm"
    _STATE_CROSSHAIR_POS_0_HAT = "crosshair_pos_hat_mm"
    _SHOOT_TIMING_DEFAULTS = AnSLegacyShootingModule._SHOOT_TIMING_DEFAULTS

    def __init__(
        self,
        spec_name: str = "ans-original-baseline",
        config_preset: str = "legacy",
        config: dict = None,
    ):
        super().__init__(spec_name, config_preset, config)


    def _perceive_and_predict(self, state, action):
        target_pos_0_hat = np.asarray(self.game_env.target_pos_monitor_mm, dtype=np.float32).copy()
        crosshair_pos_0_hat = np.asarray(self.game_env.crosshair_pos_mm, dtype=np.float32).copy()

        target_vel_true_by_orbit = self.game_env.target_monitor_velocity(
            initial_target_pos_monitor_mm=target_pos_0_hat,
            hand_vel_mm=np.zeros(2, dtype=np.float32),
        )
        target_vel_hat_by_orbit_mm_s = Perceive.velocity(
            vel=target_vel_true_by_orbit,
            pos=target_pos_0_hat,
            noise=self.player_state["param_speed_noise"],
            head=self.player_state["head_position"],
        )

        clock_noise = Perceive.timing(1.0, self.player_state["param_clock_noise"])
        target_pos_after_bump_aim_mm = self.game_env.target_monitor_position(
            initial_target_pos_monitor_mm=target_pos_0_hat,
            hand_displacement_mm=self.ongoing_motor_planned_movement_pos[-1] -
                self.ongoing_motor_planned_movement_pos[0],
            target_orbit_angle_deg=0.0,
            clip_ratio=1.1,
        )

        target_bump_later_pos_monitor_mm = target_pos_after_bump_aim_mm + \
            (TINTERVAL.BUMP / 1000) * target_vel_hat_by_orbit_mm_s * clock_noise
        target_bump_th_later_pos_monitor_mm = target_pos_after_bump_aim_mm + \
            ((TINTERVAL.BUMP + action["th"]) / 1000) * target_vel_hat_by_orbit_mm_s * clock_noise

        state[AS.TARGET_VEL_HAT_BY_ORBIT_MM_S] = target_vel_hat_by_orbit_mm_s
        state[AS.TARGET_BUMP_LATER_POS_MONITOR_MM] = target_bump_later_pos_monitor_mm
        state[AS.TARGET_BUMP_TH_LATER_POS_MONITOR_MM] = target_bump_th_later_pos_monitor_mm
        state[AS.CLOCK_NOISE] = clock_noise
        state[self._STATE_TARGET_POS_0_HAT] = target_pos_0_hat
        state[self._STATE_CROSSHAIR_POS_0_HAT] = crosshair_pos_0_hat
        state["crosshair_pos_hat_mm"] = crosshair_pos_0_hat


    def _plan_shoot_timing(self, state, action):
        if self.shooting_motor_plan_determined:
            self.shooting_motor_plan_determined = False
            tavel_hat = self.game_env.target_monitor_velocity(
                target_speed_deg_s=0.0,
                hand_vel_mm=self.ongoing_motor_planned_movement_vel[0],
            )
            shoot_timing_s = Shoot.sample_shoot_timing(
                tpos=state[self._STATE_TARGET_POS_0_HAT],
                cpos=state[self._STATE_CROSSHAIR_POS_0_HAT],
                tgvel=state[AS.TARGET_VEL_HAT_BY_ORBIT_MM_S],
                tavel=tavel_hat,
                trad=self.game_env.target_radius_mm,
                param_clock_noise=self._shoot_scalar("param_clock_noise", default=0.056),
                cmu=self._shoot_scalar("param_shoot_cmu", default=self._SHOOT_TIMING_DEFAULTS["cmu"]),
                nu=self._shoot_scalar("param_shoot_nu", default=self._SHOOT_TIMING_DEFAULTS["nu"]),
                delta=self._shoot_scalar("param_shoot_delta", default=self._SHOOT_TIMING_DEFAULTS["delta"]),
                interp_interval_ms=TINTERVAL.INTERP1,
            )
            self.shoot_timing = max(int(round(float(shoot_timing_s) * 1000.0)), int(TINTERVAL.INTERP1))


    def _shoot_scalar(self, name, default):
        player_cfg = self.config.get("player", {})
        if name in self.player_state:
            return float(np.asarray(self.player_state[name], dtype=float))
        cfg_value = player_cfg.get(name)
        if cfg_value is not None and not isinstance(cfg_value, dict):
            return float(cfg_value)
        return float(default)