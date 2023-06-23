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

import sys
sys.path.append("..")

from agent.fps_task import GameState
from configs.simulation import MAXIMUM_HAND_SPEED
from utilities.mymath import target_pos_game, ct2sp, replicate_target_movement
from utilities.otg import *


def plan_hand_movement(
    hpos,   # Current state
    hvel,   # Current state
    ppos,   # Current state
    pcam,   # Current state
    tmpos,  # Future state
    tvel,   # Future state
    cpos,   # Current state
    sensi,  # Current state
    plan_duration,
    execute_duration,
    interval=TIME_UNIT[50],             # Unit: sec
    maximum_hspd=MAXIMUM_HAND_SPEED     # Unit: m/s
):
    '''
    Return required ideal hand adjustment.
    Assume that the simulated user thinks hand direction should be
    identical to target direction, to replicate its velocity.
    In reality, the direction kept vary due to the nature of 3D camera projection

    Note that input target and crosshair position could be estimated values
    '''
    # Estimated 3d positions of target and crosshair
    tgpos = target_pos_game(ppos, pcam, tmpos)
    cgpos = target_pos_game(ppos, pcam, cpos)

    # Compute required cam adjustment
    cam_adjustment = ct2sp(*tgpos) - ct2sp(*cgpos)

    # Convert to hand movement
    hand_adjustment = cam_adjustment / sensi

    # Offset target velocity to zero by hand movement
    hvel_n = replicate_target_movement(
        ppos, pcam, tmpos, tvel, sensi
    )

    hp, hv = otg_2d(
        hpos, hvel, hpos + hand_adjustment, hvel_n,
        interval, plan_duration, execute_duration
    )

    # Limit the maximum hand speed
    hs = np.linalg.norm(hv, axis=1)
    if np.any(hs > maximum_hspd):
        hs_ratio = np.max([np.ones(hs.size), hs / maximum_hspd], axis=0)
        hs_ratio = np.reshape(hs_ratio, (hs_ratio.size, 1))
        hv_limited = hv / hs_ratio
        hp_limited = hpos + np.cumsum(
            (hv_limited[1:] + hv_limited[:-1]) / 2 * interval,
            axis=0
        )
        hp_limited = np.insert(hp_limited, 0, hpos, axis=0)

        return hp_limited, hv_limited
    else:
        return hp, hv


def add_motor_noise(
    p0,     # Initial position
    v,      # Velocity trajectory
    theta_m,
    interval=TIME_UNIT[50],
    ppd_noise_ratio=0.192     # Perpendicular noise is propotional to parallel
):
    '''
    Add motor noise to hand motor plan
    theta_m is motor noise in parallel.
    '''
    v_noisy = np.copy(v)
    nc = np.array([1, ppd_noise_ratio]) * theta_m

    for i, v in enumerate(v[1:]):
        if np.linalg.norm(v) == 0:
            v_dir = np.array([0, 0])
        else:
            v_dir = v / np.linalg.norm(v)
        v_per = np.array([-v_dir[1], v_dir[0]])

        noise = nc * np.linalg.norm(v) * np.random.normal(0, 1, 2)
        v_noisy[i+1] += noise @ np.array([v_dir, v_per])
    
    p_noisy = p0 + np.cumsum(
        (v_noisy[1:] + v_noisy[:-1]) / 2 * interval, 
        axis=0
    )
    p_noisy = np.insert(p_noisy, 0, p0, axis=0)

    return p_noisy, v_noisy


def interpolate_plan_segment(
    p0, v0, pn, vn,
    duration,
    interp_interval
):
    """
    Return p, v when dt elapsed from given initial state.
    This function is specialized for computing target's landing area
    at shooting moment, where it is somewhere between 50ms intervals.
    """
    # assert interp_interval < full_duration
    return otg_2d(
        p0, v0, pn, vn,
        interp_interval,
        duration,
        duration
    )

def interpolate_plan(p, v, plan_interval, interp_interval):
    interp_plan_p = [p[0:1]]
    interp_plan_v = [v[0:1]]
    for i in range(p.shape[0] - 1):
        _p, _v = interpolate_plan_segment(
            p[i], v[i], p[i+1], v[i+1], plan_interval, interp_interval
        )
        interp_plan_p.append(_p[1:])
        interp_plan_v.append(_v[1:])
    return np.concatenate(interp_plan_p), np.concatenate(interp_plan_v)


def accel_sum(v):
    if len(v) < 2: return 0
    acc = (v[1:] - v[:-1])
    return np.sum(np.sqrt(np.sum(acc**2, axis=1)))


if __name__ == "__main__":

    g = GameState()
    g.reset(session='fsw')

    g.show_monitor()

    p, v = plan_hand_movement(
        g.hpos,   # Current state
        g.hvel,   # Current state
        g.ppos,   # Current state
        g.pcam,   # Current state
        g.tmpos,  # Future state
        g.tvel_by_orbit(),   # Future state
        np.zeros(2),   # Current state
        g.sensi,  # Current state
        0.2,
        0.1
    )

    p, v = add_motor_noise(p[0], v, 0.1)

    g.move_hand(p[-1] - p[0])
    g.show_monitor()


    # p, v = otg_2d(
    #     np.zeros(2),
    #     np.array([-1, 3]),
    #     np.ones(2),
    #     -np.ones(2),
    #     2e-3,
    #     50e-3,
    #     50e-3
    # )

    # print(accel_sum(v))

    # p, v = otg_2d(
    #     np.zeros(2),
    #     np.array([-1, 3]),
    #     np.ones(2),
    #     -np.ones(2),
    #     5e-3,
    #     50e-3,
    #     50e-3
    # )

    # print(accel_sum(v))

    # import matplotlib.pyplot as plt
    # plt.scatter(*p.T, s=1, color='k')
    # #plt.scatter(*p[11:].T, s=1, color='r')
    # plt.show()