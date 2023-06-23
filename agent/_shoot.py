"""
Shoot Action Module
Intermittent Click Planing (ICP) model

Reference:
1) "An Intermittent Click Planning Model", Eubnji Park and Byungjoo Lee
https://dl.acm.org/doi/abs/10.1145/3313831.3376725

Code modified by June-Seop Yoon
Original code was written by Seungwon Do (dodoseung)
"""

import numpy as np

import sys
sys.path.append("..")

from utilities.mymath import dist_point2line
from configs.simulation import INTERP_INTERVAL, MAXIMUM_EPISODE_LENGTH, USER_PARAM_MEAN

def icp(Wt, tc, theta_c, p=1):
    m = USER_PARAM_MEAN["cmu"] * Wt
    s_nom = theta_c * p
    s_denom = np.sqrt(
        1 + (
            p / (1/(np.exp(
                min(USER_PARAM_MEAN["nu"] * tc, 100)
            )-1) + USER_PARAM_MEAN["delta"])
        ) ** 2
    )
    s = s_nom / s_denom

    total_time = tc + np.random.normal(m, s)
    return np.clip(total_time, INTERP_INTERVAL, MAXIMUM_EPISODE_LENGTH)



def sample_shoot_timing(
    tpos, cpos, tgvel, tavel, trad, theta_c,
):
    dist_t2x = np.linalg.norm(tpos - cpos)
    tvel = tgvel + tavel
    tspd = np.linalg.norm(tvel)

    # 1. Target is moving
    if tspd > 0:
        dist_x2ttj = dist_point2line(cpos, tpos, tvel)
        # 1-1. Crosshair on target
        if dist_t2x <= trad:
            tc = 1e-6   # Sufficiently small cue-viewing time
            value = np.sqrt(trad**2 - dist_x2ttj**2)
            bot = np.sqrt(dist_t2x**2 - dist_x2ttj**2)
            # Compute the relative position of crosshair by perpendicular line
            if (-tvel).dot(cpos-tpos) > 0:
                inter_len = value - bot
            else:
                inter_len = value + bot
        # 1-2. Crosshair outside target
        else:
            if dist_x2ttj > trad: inter_len = 0
            else: inter_len = 2*np.sqrt(trad**2 - dist_x2ttj**2)
            
            tc = (
                np.sqrt(
                    dist_t2x**2 - dist_x2ttj**2
                ) - inter_len/2
            ) / tspd
        
        Wt = inter_len / tspd
        return icp(Wt, tc, theta_c)
    
    # 2. Target is stationary
    else:
        return icp(0, 1e-6, theta_c)


if __name__ == "__main__":

    tpos = np.array([0.3, 0.3])
    cpos = np.zeros(2)
    tgvel = np.array([-0.2, 0])
    tavel = np.array([0, -0.3])
    trad = 0.012
    theta = 0.05

    t = []
    for _ in range(10000):
        t.append(
            sample_shoot_timing(tpos, cpos, tgvel, tavel, trad, theta)
        )
    
    import matplotlib.pyplot as plt

    plt.hist(t, bins=100)
    plt.show()

    pass