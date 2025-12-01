"""
Visual Perception Module

1) Perceiving target position and velocity on monitor

Reference:
1) "Noise characteristics and prior expectations in human visual speed perception"
https://www.nature.com/articles/nn1669
2) "Estimation of cortical magnification from positional error
in normally sighted and amblyopic subjects"
https://jov.arvojournals.org/article.aspx?articleid=2213265

Code modified by June-Seop Yoon
"""

import numpy as np

from ..config.constant import AXIS, VECTOR, MONITOR_QT
from ..utils.mymath import angle_between, normalize_vector, Rotate, Convert


class Perceive:

    def position_perception(opos, gpos, noise, head=VECTOR.HEAD, monitor_qt=MONITOR_QT, return_sigma=False):
        """
        opos, gpos := objective/gaze position as 2D state on the monitor
        head := 3D state in the physical world
        """
        ecc_dist = angle_between(
            np.array([*opos, 0]) - head,
            np.array([*gpos, 0]) - head,
            return_in_degree=True
        )
        est_error = np.random.normal(0, noise * ecc_dist)

        # Sample the estimated position in visual angle space
        head_to_obj = normalize_vector(np.array([*opos, 0]) - head)
        head_to_obj_noisy = Rotate.point_about_axis(
            Rotate.point_about_axis(
                head_to_obj,
                est_error,
                *Convert.cart2sphr(*np.cross(head_to_obj, VECTOR.UP))
            ),
            np.random.uniform(0, 360),
            *Convert.cart2sphr(*head_to_obj),
            angle_is_degree=True
        )
        hat_pos = np.clip(
            (head - head[AXIS.Z] / head_to_obj_noisy[AXIS.Z] * head_to_obj_noisy)[[AXIS.X, AXIS.Y]],
            -monitor_qt,
            monitor_qt
        )
        if return_sigma:
            return hat_pos, noise * ecc_dist
        return hat_pos

    def speed_perception(vel, pos, noise, head=VECTOR.HEAD, s0=0.3, dt=0.01):
        """
        pos, vel := 2D state on the monitor
        """
        spd = np.linalg.norm(vel)
        if spd <= 0: return vel

        p0 = np.array([*pos, 0])
        p1 = np.array([*(pos + vel * dt), 0])
        
        # Angle between objective movement and head to objective
        aspd = angle_between(p0 - head, p1 - head) / dt
        if aspd <= 0:
            return vel
        
        aspd_hat = np.log(1 + aspd / s0)
        s_prime = np.random.lognormal(aspd_hat, noise)
        s_final = np.clip((s_prime - 1) * s0, 0, np.inf)

        return (s_final / aspd) * vel
    

    def timing_perception(t, noise):
        while True:
            clock_noise = np.random.normal(1, noise)
            if 0.01 < clock_noise < 2: break
        return t * clock_noise
    

