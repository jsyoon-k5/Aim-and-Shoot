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

from utils.mymath import dist_point2line
from configs.simulation import INTERP_INTERVAL, BUMP_INTERVAL

def icp(Wt, t_touch, cmu, theta_c, nu=20.524, p=1, delta=0.411):
    m = max(cmu * Wt, INTERP_INTERVAL / 1000)
    tc = t_touch + m
    sigma_t = theta_c * tc
    # sigma_v = theta_c * (1 / (np.exp(np.clip(nu * tc, 1e-5, 10)) - 1) + delta)
    # s = np.sqrt(sigma_t ** 2 * sigma_v ** 2 / (sigma_t ** 2 + sigma_v ** 2)) if theta_c > 0 else 0

    while True:
        total_time = tc + np.random.normal(m, sigma_t)
        if total_time >= INTERP_INTERVAL / 1000: break
    return total_time


def icp2(Wt, tc, theta_c, cmu, max_timing):
    m = cmu * Wt
    s = theta_c * (m + tc)

    total_time = tc + np.random.normal(m, s)
    return np.clip(total_time, INTERP_INTERVAL / 1000, max_timing + INTERP_INTERVAL)


def sample_timing(ttraj, cpos, trad, th, interval, ta, theta_c, tp=BUMP_INTERVAL/1000):
    rel_pos = ttraj - cpos
    dist = np.linalg.norm(rel_pos, axis=1)
    if np.all(dist > trad):
        touch_index = np.argmin(dist)
    else:
        touch_index = np.where(dist <= trad)[0][0]
    touch_time = touch_index * interval

    Wt = tp + th - touch_time

    return icp(Wt, touch_time, ta, theta_c)



def internal_clock_noise(theta_c):
    while True:
        noise = np.random.normal(1, theta_c)
        if 0.001 < noise < 2:
            return noise


def icp_old(Wt, tc, params):
    m = Wt * params["cmu"]
    st = params["theta_c"] * tc
    sv = params["theta_c"] * (1 / (np.exp(np.clip(params["nu"] * tc, 1e-5, 10)) - 1) + params["delta"])
    s = np.sqrt(st ** 2 * sv ** 2 / (st ** 2 + sv ** 2)) if params["theta_c"] > 0 else 0

    total_time = np.clip(tc + np.random.normal(m, s), INTERP_INTERVAL/1000, np.inf)
    return total_time


def sample_shoot_timing(tpos, cpos, tgvel, tavel, trad, params):
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
        return icp_old(Wt, tc, params)
    
    # 2. Target is stationary
    else:
        return icp_old(0, 1e-6, params)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    t = [icp(0.1, 0.5, 0.33, 0.5) for _ in range(5000)]
    plt.hist(t, bins=50)
    plt.show()