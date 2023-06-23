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
from configs.simulation import TIME_UNIT
from configs.path import PATH_MAT


def prepare_otg_calculation(time_unit, time_interval):
    steps = round(time_interval / time_unit) + 2
    mat_path = f"{PATH_MAT}grams_{time_unit:.4f}_{steps:d}.pkl"

    if os.path.exists(mat_path):
        with open(mat_path, "rb") as fp:
            Gs, GTs, Grams, Gram_invs = pickle.load(fp)
    else:
        # Preparations for OTG
        G = np.array([[1, time_unit], [0, 1]])
        H = np.array([[time_unit**2 / 2], [time_unit]])

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

        with open(mat_path, "wb") as fp:
            pickle.dump((Gs, GTs, Grams, Gram_invs), fp)
    
    return Gs, GTs, Grams, Gram_invs


GRAMS = {
    2: prepare_otg_calculation(TIME_UNIT[2], TIME_UNIT[50]),
    5: prepare_otg_calculation(TIME_UNIT[5], TIME_UNIT[100]),
    50: prepare_otg_calculation(TIME_UNIT[50], TIME_UNIT[2000]),
    1: prepare_otg_calculation(TIME_UNIT[1], TIME_UNIT[300])
}


# 1-dimensional OTG
def otg_1d(p0, v0, p1, v1, time_unit, plan_duration, execute_duration):
    '''
    Generate min squared accel. trajectory.
    Initial pos/vel -> dest pos/vel
    
    Note that if plan duration is not completely divisble by time unit,
    return value could be slightly different from analytic computation.
    '''

    Gs, _, Grams, Gram_invs = GRAMS[int(time_unit * 1000)]
    pu = int(np.floor(plan_duration / time_unit))
    eu = int(np.floor(execute_duration / time_unit))

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
    p, v = otg_2d(
        np.zeros(2),
        np.ones(2)*10,
        np.ones(2),
        np.zeros(2),
        1e-3,
        200e-3,
        200e-3
    )

    import matplotlib.pyplot as plt
    s = np.linalg.norm(v, axis=1)
    plt.plot(s)
    #plt.scatter(*p[11:].T, s=1, color='r')
    plt.show()