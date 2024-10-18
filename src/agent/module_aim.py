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
        hpos,
        hvel,
        ppos,
        pcam,
        cpos,
        tmpos,
        tmvel,
        sensitivity,
        fov,
        monitor_qt,
        plan_duration:int,
        execute_duration:int,
        interval:int,           # Duration/Interval units: millisec
        maximum_camera_speed=np.inf
    ):
        '''
        Return required ideal hand adjustment.
        Assume that the simulated user thinks hand direction should be
        identical to target direction, to replicate its velocity.
        In reality, the direction kept vary due to the nature of 3D camera projection

        Note that input target and crosshair position could be estimated values
        '''
        # Estimated 3d positions of target and crosshair
        tgpos = Convert.monitor2game(ppos, pcam, tmpos, fov, monitor_qt)
        cgpos = Convert.monitor2game(ppos, pcam, cpos, fov, monitor_qt)

        # Compute required cam adjustment
        cam_adjustment = Convert.cart2sphr(*tgpos) - Convert.cart2sphr(*cgpos)

        # Convert to hand movement
        hand_adjustment = cam_adjustment / sensitivity

        # Offset target velocity to zero by hand movement
        hvel_n = Aim._replicate_target_movement(
            ppos, pcam, tmpos, tmvel, sensitivity, fov, monitor_qt
        )
        hp, hv = otg_2d(
            hpos, hvel, hpos + hand_adjustment, hvel_n,
            interval, plan_duration, execute_duration
        )

        # Limit the maximum hand speed
        hs = np.linalg.norm(hv, axis=1)
        maximum_hspd = maximum_camera_speed / sensitivity
        if np.any(hs >= maximum_hspd):
            hs_ratio = np.max([np.ones(hs.size), hs / maximum_hspd], axis=0)
            hs_ratio = np.reshape(hs_ratio, (hs_ratio.size, 1))
            hv_limited = hv / hs_ratio
            hp_limited = hpos + np.cumsum(
                (hv_limited[1:] + hv_limited[:-1]) / 2 * interval / 1000,
                axis=0
            )
            hp_limited = np.insert(hp_limited, 0, hpos, axis=0)

            return hp_limited, hv_limited
        else:
            return hp, hv
    

    def _replicate_target_movement(ppos, pcam, tmpos, tmvel, sensitivity, fov, monitor_qt, dt=0.001):
        tmpos_0 = tmpos
        tmpos_n = tmpos + tmvel * dt   # Slight moment after
        tgpos_0 = Convert.monitor2game(ppos, pcam, tmpos_0, fov, monitor_qt)
        tgpos_n = Convert.monitor2game(ppos, pcam, tmpos_n, fov, monitor_qt)
        cam_adjust = Convert.cart2sphr(*tgpos_n) - Convert.cart2sphr(*tgpos_0)
        hand_adjust = cam_adjust / sensitivity
        return hand_adjust / dt


    def add_motor_noise(
        p0,     # Initial position
        v,      # Velocity trajectory
        noise,
        interval,       # millisec
        noise_ppd=None,
        ppd_noise_ratio=0.192     # Perpendicular noise is propotional to parallel
    ):
        '''
        Add motor noise to hand motor plan
        theta_m is motor noise in parallel.
        '''
        v_noisy = np.copy(v)
        nc = np.array([1, ppd_noise_ratio]) * noise if noise_ppd is None else np.array([noise, noise_ppd])

        for i, v in enumerate(v[1:]):
            if np.linalg.norm(v) == 0:
                v_dir = np.array([0, 0])
            else:
                v_dir = v / np.linalg.norm(v)
            v_per = np.array([-v_dir[1], v_dir[0]])

            noise = nc * np.linalg.norm(v) * np.random.normal(0, 1, 2)
            v_noisy[i+1] += noise @ np.array([v_dir, v_per])
        
        p_noisy = p0 + np.cumsum(
            (v_noisy[1:] + v_noisy[:-1]) / 2 * interval / 1000, 
            axis=0
        )
        p_noisy = np.insert(p_noisy, 0, p0, axis=0)

        return p_noisy, v_noisy


    def interp_segment(p0, v0, p1, v1, full_interval:int, interp_interval:int):
        return otg_2d(
            p0, v0, p1, v1,
            interp_interval,
            full_interval,
            full_interval
        )


    def interpolate_plan(traj_p, traj_v, original_interval:int, interp_interval:int):
        interp_p = [traj_p[0:1]]
        interp_v = [traj_v[0:1]]

        for i in range(traj_p.shape[0] - 1):
            _p, _v = Aim.interp_segment(
                traj_p[i], traj_v[i], traj_p[i+1], traj_v[i+1],
                original_interval,
                interp_interval
            )
            interp_p.append(_p[1:])
            interp_v.append(_v[1:])
        
        return np.concatenate(interp_p), np.concatenate(interp_v)


    def accel_sum(v):
        if len(v) < 2: return 0
        acc = (v[1:] - v[:-1])
        return np.sum(np.sqrt(np.sum(acc**2, axis=1)))