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

import sys
sys.path.append("..")

from configs.common import HEAD_POSITION, MONITOR_BOUND, UP
from configs.simulation import BUMP_INTERVAL, INTERP_INTERVAL
from utils.mymath import *
from utils.otg import *


### EMMA model
def gaze_landing_point(curr, dest, head_pos=HEAD_POSITION):
    ed = eccentricity_distance(curr, dest, head_pos=head_pos, return_in_degree=True)
    landing_error = np.random.normal(0, 0.1*ed)
    eye2dest = normalize_vector(np.array([*dest, 0]) - head_pos)
    rot = np.random.uniform(0, 360)
    base_rot_axis = np.cross(eye2dest, UP)
    base_land_vec = rotate_point_around_axis(eye2dest, landing_error, *cart2sphr(*base_rot_axis))
    rot_land_vec = rotate_point_around_axis(base_land_vec, rot, *cart2sphr(*eye2dest))

    return np.clip(
        (head_pos - head_pos[Z]/rot_land_vec[Z] * rot_land_vec)[[X, Y]],
        -MONITOR_BOUND,
        MONITOR_BOUND
    )




def peak_velocity(amp, slope, a_th=SACCADE_AMP_TH, va=SACCADE_VEL_TH):
    return max(va, va + slope * (amp-a_th))

def saccade_duration(amp, peak_vel):
    return 30 * amp / (16 * peak_vel)

def saccade_speed(timestamp, delay, amp, slope):
    dur = saccade_duration(amp, peak_velocity(amp, slope))
    return minimum_jerk(timestamp, amp, delay, dur)


def gaze_plan(
    g0, gn,
    theta_q=42.8,
    delay=0,
    exe_until=0.1, 
    interp_intv=0.005, 
    head_pos=HEAD_POSITION
):
    eye2g0 = np.array([*g0, 0]) - head_pos
    eye2gn = np.array([*gn, 0]) - head_pos
    amp = angle_between(eye2g0, eye2gn, return_in_degree=True)
    timestamp = np.linspace(0, exe_until, int(exe_until / interp_intv) + 1)
    sac_spd = saccade_speed(timestamp, delay, amp, theta_q)
    amp_cum = np.cumsum(interp_intv * (sac_spd[1:] + sac_spd[:-1])/2)
    amp_cum = np.insert(amp_cum, 0, 0)

    # return timestamp, sac_spd, amp_cum

    base_rot_axis = np.cross(eye2g0, eye2gn)
    norm_fix_vec = normalize_vector(eye2g0)
    rot_fix_vec = np.array([
        rotate_point_around_axis(norm_fix_vec, a, *cart2sphr(*base_rot_axis)) for a in amp_cum
    ])
    
    gaze_traj = (-np.multiply(head_pos[Z] / rot_fix_vec[:,Z], rot_fix_vec.T).T + head_pos)[:,X:Z]

    # if delay > 0:
    #     timestamp += delay
    #     timestamp = np.insert(timestamp, 0, 0)
    #     gaze_traj = np.insert(gaze_traj, 0, gaze_traj[0], axis=0)

    return timestamp, gaze_traj



### We're not using these...
# def saccade_preparation_time(tm=0.135, ts=0.045):
#     return gamma_distribution(tm, ts)

# def visual_encoding_time(e, K=0.006, k=0.4, fi=0.99):
#     tm = K * (-np.log(fi)) * np.exp(k * e)
#     ts = tm / 3
#     return gamma_distribution(tm, ts)

# def saccade_execution_time(e):
#     tm = 0.07 + 0.002 * e
#     ts = tm / 3
#     return gamma_distribution(tm, ts)



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # curr = CROSSHAIR
    # dest = np.array([0.2, -0.05])

    # g = []
    # for _ in range(10000):
    #     g.append(
    #         gaze_landing_point(curr, dest)
    #     )
    # g = np.array(g)

    # plt.scatter(*g.T, s=0.1, alpha=0.1)
    # plt.xlim(-MONITOR_BOUND[X], MONITOR_BOUND[X])
    # plt.ylim(-MONITOR_BOUND[Y], MONITOR_BOUND[Y])
    # plt.show()

    # t = np.linspace(0, 1, 2000)
    # # y = saccade_curve(t)
    # y = saccade_speed(t, 0.5, 17.586)

    # a = np.sum((t[1:] - t[:-1]) * (y[1:]+y[:-1])/2)
    # print(a)

    # plt.plot(t, y)
    # plt.show()

    t, gtj = gaze_plan(
        CROSSHAIR, np.array([1e-6, 0]), delay=0.07
    )
    print(t, gtj[:,0])

    plt.plot(t, gtj[:,0])
    plt.show()



## GRAVEYARD

### Main Sequence - Minimum jerk based
# def saccade_curve(x):
#     """
#     Slice original sigmoid derivative curve where x in [-5, 5]
#     and normalize it to range x in [0, 1] where amplitude is 1
#     Normalized sigmoid curve for saccade fitting
#     sig'(5) = 0.006648056670790033
#     sig(5) - sig(-5) - 10sig'(5) = 0.92013373144353
#     """
#     return np.maximum((sigmoid_derivative(10*x - 5) - 0.006648056670790033) / 0.092013373144353, 0)


# def saccade_speed(t, amp, slope, va=SACCADE_VEL_TH):
#     """
#     t := elapsed time since saccade begin
#     amp := amplitude to destination
#     theta := parameter which controls the peak velocity

#     saccade_curve(0.5) = 2.6447453779075467
#     """
#     peak_v = peak_velocity(amp, slope, va=va)
#     duration = amp * 2.6447453779075467 / peak_v
#     return saccade_curve(t / duration) / 2.6447453779075467 * peak_v