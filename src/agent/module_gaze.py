"""
Gaze Control Module

1) Gaze dynamics based on main sequence (peak velocity, amplitude, duration)
2) Add noise to eye landing point

Reference:
1) "An integrated model of eye movements and visual encoding"
https://www.sciencedirect.com/science/article/pii/S1389041700000152
2) "The saccade main sequence revised A fast and repeatable tool for oculomotor analysis"
https://link.springer.com/article/10.3758/s13428-020-01388-2
3) "The Spectral Main Sequence of Human Saccades"
https://www.jneurosci.org/content/19/20/9098.short

This code was written by June-Seop Yoon
"""

import numpy as np

from ..utils.mymath import angle_between, normalize_vector, minimum_jerk, Rotate, Convert
from ..config.constant import MONITOR_QT, VECTOR, AXIS

class Gaze:
    ### EMMA model
    def gaze_landing_point(curr, dest, head=VECTOR.HEAD, deviation=0.1, monitor_qt=MONITOR_QT):
        ecc = Gaze._eccentricity(curr, dest, head)
        land_error = np.random.normal(0, deviation*ecc)

        # Sample the estimated position in visual angle space
        head_to_dest = normalize_vector(np.array([*dest, 0]) - head)
        head_to_dest_noisy = Rotate.point_about_axis(
            Rotate.point_about_axis(
                head_to_dest,
                land_error,
                *Convert.cart2sphr(*np.cross(head_to_dest, VECTOR.UP))
            ),
            np.random.uniform(0, 360),
            *Convert.cart2sphr(*head_to_dest),
            angle_is_degree=True
        )
        return np.clip(
            (head - head[AXIS.Z] / head_to_dest_noisy[AXIS.Z] * head_to_dest_noisy)[[AXIS.X, AXIS.Y]],
            -monitor_qt,
            monitor_qt
        )
    
    def _eccentricity(p0, p1, head):
        return angle_between(
            np.array([*p0, 0]) - head,
            np.array([*p1, 0]) - head,
            return_in_degree=True
        )


    def peak_velocity(amp, a_th=1, slope=35.6, va=78.7):
        return max(va, va + slope * (amp-a_th))

    def saccade_duration(amp, peak_vel):
        return 30 * amp / (16 * peak_vel)

    def saccade_speed(timestamp, delay, amp):
        dur = Gaze.saccade_duration(amp, Gaze.peak_velocity(amp))
        return minimum_jerk(timestamp, amp, delay, dur)


    def gaze_plan(
        g0, gn,
        delay:int=0,
        exe_until:int=100, 
        interp_intv:int=5, 
        head=VECTOR.HEAD
    ):
        eye2g0 = np.array([*g0, 0]) - head
        eye2gn = np.array([*gn, 0]) - head
        amp = angle_between(eye2g0, eye2gn, return_in_degree=True)
        timestamp = np.linspace(0, exe_until, exe_until // interp_intv + 1, dtype=int)  # Millisec
        sac_spd = Gaze.saccade_speed(timestamp/1000, delay/1000, amp)    # Pass second unit timestamp
        amp_cum = np.cumsum(interp_intv/1000 * (sac_spd[1:] + sac_spd[:-1])/2)
        amp_cum = np.insert(amp_cum, 0, 0)

        # return timestamp, sac_spd, amp_cum

        base_rot_axis = np.cross(eye2g0, eye2gn)
        norm_fix_vec = normalize_vector(eye2g0)
        rot_fix_vec = np.array([
            Rotate.point_about_axis(norm_fix_vec, a, *Convert.cart2sphr(*base_rot_axis)) for a in amp_cum
        ])
        
        gaze_traj = (-np.multiply(head[AXIS.Z] / rot_fix_vec[:,AXIS.Z], rot_fix_vec.T).T + head)[:,AXIS.X:AXIS.Z]


        return timestamp, gaze_traj
    
