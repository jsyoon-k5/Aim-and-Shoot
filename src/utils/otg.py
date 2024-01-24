"""
Optimal Trajectory Generation

Reference:
Bye, Robin Trulssen. 
The BUMP model of response planning. Diss. Ph. D. Dissertation. 
Sydney, Australia: The University of New South Wales, 2009.

p.49 ~ 55
"""

import numpy as np
import os, pickle, sys

from tqdm import tqdm
from collections import defaultdict

sys.path.append("..")
from configs.path import PATH_MAT
from utils.utils import pickle_load, pickle_save


def prepare_otg_calculation(time_unit, time_interval):
    steps = time_interval // time_unit + 2
    mat_path = f"{PATH_MAT}/grams_{time_unit}_{steps}.pkl"

    if os.path.exists(mat_path):
        Gs, GTs, Grams, Gram_invs = pickle_load(mat_path)
    else:
        # Preparations for OTG
        tu = time_unit / 1000
        G = np.array([[1, tu], [0, 1]])
        H = np.array([[tu**2 / 2], [tu]])

        def get_Gs():
            Gs, GTs = [np.identity(2)], [np.identity(2)]
            for _ in range(1, steps):
                Gs.append(Gs[-1] @ G)
                GTs.append(GTs[-1] @ G.T)
            return Gs, GTs

        Gs, GTs = get_Gs()

        def get_gram(n, k):
            gram = np.zeros((2, 2))
            for j in range(0, k):
                gram += Gs[j] @ H @ H.T @ GTs[n - k + j]
            return gram

        def get_grammians():
            Grams, Gram_invs = defaultdict(dict), dict()
            for n in tqdm(range(steps)):
                for k in range(n + 1):
                    Grams[n][k] = get_gram(n, k)
                if n > 1:
                    Gram_invs[n] = np.linalg.inv(get_gram(n, n))
            return Grams, Gram_invs

        Grams, Gram_invs = get_grammians()

        pickle_save(mat_path, (Gs, GTs, Grams, Gram_invs))
    
    return Gs, GTs, Grams, Gram_invs


GRAMS = {
    1: prepare_otg_calculation(1, 50),
    5: prepare_otg_calculation(5, 50),
    10: prepare_otg_calculation(10, 50),
    50: prepare_otg_calculation(50, 2000),
}


# 1-dimensional OTG
def otg_1d(p0, v0, p1, v1, time_unit:int, plan_duration:int, execute_duration:int):
    '''
    Generate min squared accel. trajectory.
    Initial pos/vel -> dest pos/vel
    
    Note that if plan duration is not completely divisble by time unit,
    return value could be slightly different from analytic computation.
    '''

    Gs, _, Grams, Gram_invs = GRAMS[time_unit]
    pu = plan_duration // time_unit
    eu = execute_duration // time_unit

    x0 = np.array([[p0, v0]]).T
    x1 = np.array([[p1, v1]]).T

    g_n = Gs[pu]
    inv_gram_n = Gram_invs[pu]
    span = inv_gram_n @ (x1 - g_n @ x0)

    xs = []
    for k in range(eu+1):
        g_k = Gs[k]
        gram_k = Grams[pu][k]
        xs.append((g_k @ x0) + gram_k @ span)

    xs = np.concatenate(xs, axis=1)
    return xs[0], xs[1]     # Interpolated position and velocity


def otg_2d(p0, v0, p1, v1, time_unit, plan_duration, execute_duration):
    '''Two-dimensional otg'''
    px, vx = otg_1d(p0[0], v0[0], p1[0], v1[0], time_unit, plan_duration, execute_duration)
    py, vy = otg_1d(p0[1], v0[1], p1[1], v1[1], time_unit, plan_duration, execute_duration)

    return np.vstack((px, py)).T, np.vstack((vx, vy)).T


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    p, v = otg_2d(
        np.zeros(2),
        np.ones(2),
        np.ones(2)/2,
        np.zeros(2),
        50,
        1000,
        1000
    )
    plt.scatter(*p.T, s=1)
    
    # s = np.linalg.norm(v, axis=1)
    # plt.plot(s)
    #plt.scatter(*p[11:].T, s=1, color='r')
    plt.show()

    plt.plot(v[:,0])
    plt.show()