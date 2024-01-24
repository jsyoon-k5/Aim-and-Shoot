"""
Visual Perception Module

1) Perceiving target position and velocity on monitor
2) They are influenced by gazepoint

Reference:
1) "Noise characteristics and prior expectations in human visual speed perception"
https://www.nature.com/articles/nn1669
2) "Estimation of cortical magnification from positional error
in normally sighted and amblyopic subjects"
https://jov.arvojournals.org/article.aspx?articleid=2213265

Code modified by June-Seop Yoon
Original code was written by Seungwon Do (dodoseung)
"""

import numpy as np

import sys
sys.path.append("..")

from configs.common import *
from configs.simulation import MAXIMUM_TARGET_MSPEED
from utils.mymath import eccentricity_distance, angle_between, rotate_point_around_axis, cart2sphr, normalize_vector


def position_perception(tpos, gpos, theta_p, head_pos=HEAD_POSITION):
    """Sample perceived position"""
    ed = eccentricity_distance(tpos, gpos, head_pos=head_pos, return_in_degree=True)
    view_error = np.random.normal(0, theta_p * ed)
    eye2tar = normalize_vector(np.array([*tpos, 0]) - head_pos)
    rot = np.random.uniform(0, 360)
    base_rot_axis = np.cross(eye2tar, UP)
    base_land_vec = rotate_point_around_axis(eye2tar, view_error, *cart2sphr(*base_rot_axis))
    rot_land_vec = rotate_point_around_axis(base_land_vec, rot, *cart2sphr(*eye2tar))

    return np.clip(
        (head_pos - head_pos[Z]/rot_land_vec[Z] * rot_land_vec)[[X, Y]],
        -MONITOR_BOUND,
        MONITOR_BOUND
    )


def speed_perception(
    tgvel,
    tpos,
    theta_s,
    head_pos=HEAD_POSITION, 
    # max_speed=MAXIMUM_TARGET_MSPEED,
    s0=0.3,
    dt=0.01
):
    """Sample perceived speed"""
    spd = np.linalg.norm(tgvel)
    if spd <= 0: return 1

    ### Stocker's model
    # Convert to 3D point
    tmpos0 = np.array([*tpos, 0])
    tmpos1 = np.array([*(tpos + dt * tgvel), 0])
    # tmposmax = np.array([*(tpos + dt * max_speed * normalize_vector(tgvel)), 0])
    # Convert to visual angle speed
    aspd = angle_between(tmpos0 - head_pos, tmpos1 - head_pos) / dt    # deg/s
    if aspd <= 0: return 1
    aspd_hat = np.log(1 + aspd/s0)
    # aspd_max = angle_between(tmpos0 - head_pos, tmposmax - head_pos) / dt
    # Sample speed
    s_prime = np.random.lognormal(aspd_hat, theta_s)
    s_final = np.clip((s_prime - 1) * s0, 0, np.inf)

    return s_final / aspd



if __name__ == "__main__":

    # Position perception debugging
    # p = []
    # for _ in range(10000):
    #     p.append(
    #         position_perception(
    #             np.array([0.09, 0]),
    #             np.zeros(2),
    #             0.1
    #         )
    #     )
    # p = np.array(p)

    # import matplotlib.pyplot as plt
    # X, Y = 0, 1
    # # plt.hist2d(*p.T, bins=100)
    # # plt.hist(p[:,0], bins=100)
    # plt.scatter(*p.T, s=0.1, alpha=0.1)
    # plt.xlim(-MONITOR_BOUND[X], MONITOR_BOUND[X])
    # plt.xticks(np.linspace(-MONITOR_BOUND[X], MONITOR_BOUND[X], 5))
    # plt.ylim(-MONITOR_BOUND[Y], MONITOR_BOUND[Y])
    # plt.yticks(np.linspace(-MONITOR_BOUND[Y], MONITOR_BOUND[Y], 5))
    # plt.show()

    # Visual perception debugging
    v = []
    for _ in range(10000):
        v.append(
            speed_perception(
                np.array([0.5, 0]),
                np.array([0.2, 0]),
                0.15
            )
        )
    v = np.array(v)

    import matplotlib.pyplot as plt
    plt.hist(v, bins=100)
    # plt.axvline(x=0.25, color='r', zorder=100)
    plt.show()