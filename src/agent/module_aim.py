"""
1. Motor Control Module
Basic Unit of Motor Production (BUMP) model

2. Motor Execution
Signal-dependent motor noise

Motor Planning (OTG)

Reference:
1) "The BUMP model of response planning: Variable horizon predictive control accounts for the speed–accuracy tradeoffs and velocity profiles of aimed movement", Robin T. Bye and Peter D. Neilson
https://www.sciencedirect.com/science/article/pii/S0167945708000377
2) "The use of ballistic movement as an additional method to assess performance of computer mice"
https://www.sciencedirect.com/science/article/pii/S016981411400170X

Code modified by June-Seop Yoon
Original code was written by Seungwon Do (dodoseung)
"""

import numpy as np

from ..utils.mymath import Convert
from ..utils.otg import otg_2d


class Aim:
    def plan_hand_movement(
        hand_pos,
        hand_vel,
        camera_pos,
        camera_azel,
        crosshair_monitor_pos,
        target_monitor_pos,
        target_monitor_vel,
        mouse_gain_deg_per_mm,
        camera_fov_deg,
        monitor_half_size_mm,
        planning_duration_ms: int,
        execution_duration_ms: int,
        interval_ms: int,           # Duration/interval units: milliseconds
        max_camera_speed_deg_s=300,
    ):
        '''
        Return required ideal hand adjustment.
        Assume that the simulated user thinks hand direction should be
        identical to target direction, to replicate its velocity.
        In reality, the direction kept vary due to the nature of 3D camera projection

        Note that input target and crosshair position could be estimated values
        '''
        # Estimated 3d positions of target and crosshair
        target_world_pos = Convert.monitor2game(
            camera_pos,
            camera_azel,
            target_monitor_pos,
            camera_fov_deg,
            monitor_half_size_mm,
        )
        crosshair_world_pos = Convert.monitor2game(
            camera_pos,
            camera_azel,
            crosshair_monitor_pos,
            camera_fov_deg,
            monitor_half_size_mm,
        )

        # Compute required cam adjustment (delta azimuth and elevation) to hit the target
        cam_adjustment = (
            Convert.cart2sphr_scalar(
                target_world_pos[0], target_world_pos[1], target_world_pos[2], return_in_degree=True
            )
            - Convert.cart2sphr_scalar(
                crosshair_world_pos[0], crosshair_world_pos[1], crosshair_world_pos[2], return_in_degree=True
            )
        )

        # Convert to hand movement
        hand_adjustment = cam_adjustment / mouse_gain_deg_per_mm

        # Offset target velocity to zero by hand movement
        hvel_n = Aim._replicate_target_movement(
            camera_pos,
            camera_azel,
            target_monitor_pos,
            target_monitor_vel,
            mouse_gain_deg_per_mm,
            camera_fov_deg,
            monitor_half_size_mm,
        )

        hp, hv = otg_2d(
            hand_pos,
            hand_vel,
            hand_pos + hand_adjustment,
            hvel_n,
            interval_ms,
            planning_duration_ms,
            execution_duration_ms,
        )

        # Limit the maximum hand speed
        hs = np.linalg.norm(hv, axis=1)
        maximum_hspd = max_camera_speed_deg_s / mouse_gain_deg_per_mm
        # print(maximum_camera_speed)
        if np.any(hs >= maximum_hspd):
            hs_ratio = np.max([np.ones(hs.size), hs / maximum_hspd], axis=0)
            hs_ratio = np.reshape(hs_ratio, (hs_ratio.size, 1))
            hv_limited = hv / hs_ratio
            hp_limited = hand_pos + np.cumsum(
                (hv_limited[1:] + hv_limited[:-1]) / 2 * interval_ms / 1000,
                axis=0
            )
            hp_limited = np.insert(hp_limited, 0, hand_pos, axis=0)

            return hp_limited, hv_limited
        else:
            return hp, hv
    

    def _replicate_target_movement(
        camera_pos,
        camera_azel,
        target_monitor_pos,
        target_monitor_vel,
        mouse_gain_deg_per_mm,
        camera_fov_deg,
        monitor_half_size_mm,
        dt_s=0.001,
    ):
        monitor_pos_t0 = target_monitor_pos
        monitor_pos_t1 = target_monitor_pos + target_monitor_vel * dt_s
        target_world_t0 = Convert.monitor2game(
            camera_pos,
            camera_azel,
            monitor_pos_t0,
            camera_fov_deg,
            monitor_half_size_mm,
        )
        target_world_t1 = Convert.monitor2game(
            camera_pos,
            camera_azel,
            monitor_pos_t1,
            camera_fov_deg,
            monitor_half_size_mm,
        )
        cam_adjust = (
            Convert.cart2sphr_scalar(
                target_world_t1[0], target_world_t1[1], target_world_t1[2], return_in_degree=True
            )
            - Convert.cart2sphr_scalar(
                target_world_t0[0], target_world_t0[1], target_world_t0[2], return_in_degree=True
            )
        )
        hand_adjust = cam_adjust / mouse_gain_deg_per_mm
        return hand_adjust / dt_s


    def add_motor_noise(
        initial_pos,
        velocity_traj,
        noise_std,
        interval_ms,
        noise_std_perp=None,
        perp_noise_ratio=0.192,
    ):
        '''
        Add motor noise to hand motor plan
        theta_m is motor noise in parallel.
        '''
        noisy_velocity = np.copy(velocity_traj)
        nc = (
            np.array([1, perp_noise_ratio]) * noise_std
            if noise_std_perp is None
            else np.array([noise_std, noise_std_perp])
        )

        for i, vel in enumerate(velocity_traj[1:]):
            if np.linalg.norm(vel) < 1e-6:
                vel_dir = np.array([0, 0])
            else:
                vel_dir = vel / np.linalg.norm(vel)
            vel_perp = np.array([-vel_dir[1], vel_dir[0]])

            noise = nc * np.linalg.norm(vel) * np.random.normal(0, 1, 2)
            noisy_velocity[i + 1] += noise @ np.array([vel_dir, vel_perp])

        noisy_position = initial_pos + np.cumsum(
            (noisy_velocity[1:] + noisy_velocity[:-1]) / 2 * interval_ms / 1000,
            axis=0
        )
        noisy_position = np.insert(noisy_position, 0, initial_pos, axis=0)

        return noisy_position, noisy_velocity


    def interp_segment(start_pos, start_vel, end_pos, end_vel, segment_interval_ms: int, interp_interval_ms: int):
        return otg_2d(
            start_pos,
            start_vel,
            end_pos,
            end_vel,
            interp_interval_ms,
            segment_interval_ms,
            segment_interval_ms,
        )


    def interpolate_plan(position_traj, velocity_traj, original_interval_ms: int, interp_interval_ms: int):
        interp_position = [position_traj[0:1]]
        interp_velocity = [velocity_traj[0:1]]

        for i in range(position_traj.shape[0] - 1):
            _p, _v = Aim.interp_segment(
                position_traj[i],
                velocity_traj[i],
                position_traj[i + 1],
                velocity_traj[i + 1],
                original_interval_ms,
                interp_interval_ms,
            )
            interp_position.append(_p[1:])
            interp_velocity.append(_v[1:])
        
        return np.concatenate(interp_position), np.concatenate(interp_velocity)


    def accel_sum(velocity_traj, interval_ms: int):
        """Return cumulative acceleration magnitude in m/s² (velocity in mm/s)."""
        if len(velocity_traj) < 2: return 0
        dt = interval_ms / 1000.0
        acc = (velocity_traj[1:] - velocity_traj[:-1]) / dt  # mm/s²
        return np.sum(np.sqrt(np.sum(acc**2, axis=1))) / 1000.0  # → m/s²