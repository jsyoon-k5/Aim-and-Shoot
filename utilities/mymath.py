import copy

import numpy as np
from numpy.lib.function_base import angle
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import scipy as sp
from sklearn.metrics import r2_score

import sys
sys.path.append("..")

from configs.common import *
from configs.simulation import MAX_SIGMA_SCALE, USER_PARAM_MEAN, USER_PARAM_STD
from configs.path import PATH_MAT
from utilities.utils import pickle_save


def normalize_vector(v):
    d = np.linalg.norm(v)
    if d == 0: return v
    else: return v / d


### Random functions
def rand_sign(): return 1 if np.random.uniform() < 0.5 else -1


### Rotations
def rot_mat_2d(theta, theta_is_degree=True):
    a = theta * TO_RADIAN if theta_is_degree else theta
    cos_a, sin_a = np.cos(a), np.sin(a)
    return np.array([
        [cos_a, -sin_a], [sin_a, cos_a]
    ])


def rot_2d(p, theta, theta_is_degree=True):
    """Rotate p by theta in 2D plane"""
    assert type(p) is np.ndarray and p.size == 2
    return rot_mat_2d(theta, theta_is_degree) @ p


def rot_mat_3d(theta, _axis, theta_is_degree=True):
    r = rot_mat_2d(theta, theta_is_degree)
    r = np.insert(r, _axis, 0, axis=0)
    r = np.insert(r, _axis, 0, axis=1)
    r[_axis][_axis] = 1
    return r


def rot_around(p, theta, az, el, angles_are_degree=True):
    """Rotate p by theta around axis represented by (az, el)"""
    assert type(p) is np.ndarray and p.size == 3
    a0 = rot_mat_3d(-az, Y, angles_are_degree)
    e0 = rot_mat_3d(-el, Z, angles_are_degree)
    t = rot_mat_3d(theta, X, angles_are_degree)
    e1 = rot_mat_3d(el, Z, angles_are_degree)
    a1 = rot_mat_3d(az, Y, angles_are_degree)
    return a1 @ e1 @ t @ e0 @ a0 @ p


### Monitor space <-> Game space converting functions
def sp2ct(az, el, angles_are_degree=True):
    """
    Azim Elev -> X, Y, Z
    Input: (n,), (n,)
    Output: (n, 3)
    """
    a = az * TO_RADIAN if angles_are_degree else az
    e = el * TO_RADIAN if angles_are_degree else el
    return np.array([np.cos(a)*np.cos(e), np.sin(e), np.sin(a)*np.cos(e)]).T
    

def ct2sp(x, y, z, output_degree=True):
    """
    X, Y, Z -> Azim, Elev
    Input: (n,), (n,), (n,)
    Output: (n, 2)
    """
    ae = np.array([np.arctan2(z, x), np.arctan2(y, np.sqrt(x**2+z**2))]).T
    if output_degree: ae *= TO_DEGREE
    return ae


def perpendicular_vector(vec, a=None):
    """Return random (or a deg rotated) perpendicular vector of given vec"""
    r = np.cross(vec, UP)
    if a is None:
        r = rot_around(r, np.random.uniform(0, 360), *ct2sp(*vec))
    else:
        r = rot_around(r, 90-a, *ct2sp(*vec))
    return ct2sp(*r)


def angle_btw(v1, v2, output_degree=True):
    cos_a = np.sum(v1 * v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    a = np.arccos(np.clip(cos_a, -1, 1))
    if output_degree: a *= TO_DEGREE
    return a


# 1D Vector
def camera_space(az, el):
    """Return front, right, up vector"""
    f = sp2ct(az, el)
    r = np.cross(f, UP)
    u = np.cross(r, f)
    return np.vstack((f, r, u)).T


def target_pos_monitor(ppos, pcam, tgpos, fov=FOV, mbound=MONITOR_BOUND):
    """Return target position on monitor"""
    cam_mat = camera_space(*pcam)
    vec_pt = tgpos - ppos
    [a, b, c], _, _, _ = np.linalg.lstsq(cam_mat, vec_pt, rcond=-1)
    return (np.array([b, c]) / (np.tan(fov/2)*a)) * mbound


def target_pos_game(ppos, pcam, tmpos, fov=FOV, mbound=MONITOR_BOUND):
    """Return target position in game"""
    [f, r, u] = camera_space(*pcam).T
    coeff = tmpos / mbound * np.tan(fov/2)
    cos_theta = 1 / np.sqrt(1 + np.sum(coeff**2))
    return cos_theta * (f + coeff[0] * r + coeff[1] * u) + ppos


# 2D Vector
def camera_multi_space(az, el):
    """Return front, right, up vector"""
    f = sp2ct(az, el)
    r = np.cross(f, UP)
    u = np.cross(r, f)
    return np.transpose(np.concatenate((f, r, u), axis=1).reshape(az.shape[0], 3, 3), axes=(0, 2, 1))


def target_multi_pos_monitor(ppos, pcam, tgpos, fov=FOV, mbound=MONITOR_BOUND):
    """
    Parameter formats:
    ppos: numpy array with shape [n, 3]. Columns X, Y, Z in order
    pcam: numpy array with shape [n, 2]. Columns azim, elev in order
    tpos: numpy array with shape [n, 3]. Columns X, Y, Z in order
    """
    tmpos = np.array(
        [target_pos_monitor(pp, pc, tg, fov=fov, mbound=mbound) for pp, pc, tg in zip(ppos, pcam, tgpos)]
    )
    return tmpos
    # pcam_space = camera_multi_space(*pcam.T)        # Camera space (shape: (frame#, 3, 3))
    # vec_p2t = tgpos - ppos                          # Vector from player to target
    # coeff = np.array([
    #     np.linalg.lstsq(ps, vec, rcond=-1)[0] for ps, vec in zip(pcam_space, vec_p2t)
    # ])
    # return ((np.array([coeff[:,1:]]) / np.tan(fov/2)) * mbound)[0]


def replicate_target_movement(
    ppos, pcam, tmpos, tvel, sensi, 
    dt=0.001,
    fov=FOV,
    mbound=MONITOR_BOUND
):
    """
    Return hand velocity that replicates target velocity.
    This function ignores distortion occurred by glu perspective
    """
    tmpos_0 = tmpos
    tmpos_n = tmpos + tvel * dt   # Slight moment after
    tgpos_0 = target_pos_game(ppos, pcam, tmpos_0, fov=fov, mbound=mbound)
    tgpos_n = target_pos_game(ppos, pcam, tmpos_n, fov=fov, mbound=mbound)
    cam_adjust = ct2sp(*tgpos_n) - ct2sp(*tgpos_0)
    hand_adjust = cam_adjust / sensi
    return hand_adjust / dt


### Deprecated
# def meter2ecc(d, h=DIST_EYE_TO_MONITOR, output_degree=True):
#     ecc = np.arctan(d / h)
#     if output_degree: ecc *= TO_DEGREE
#     return ecc


def ecc_dist(p1, p2, ep=EYE_POSITION, output_degree=False):
    return angle_btw(
        np.array([*p1, 0]) - ep,
        np.array([*p2, 0]) - ep,
        output_degree=output_degree
    )


def gamma_distribution(m, s):
    return np.random.gamma((m/s)**2, s**2/m)


def kmeans_clustering(data, N=11):
    """K-means clustering"""
    clustering = KMeans(n_clusters=N).fit(data)
    return clustering.labels_


def mean_trajectory(tjs):
    _tjs = copy.deepcopy(tjs)
    max_len = 0
    for tj in tjs:
        max_len = tj.shape[0] if tj.shape[0] > max_len else max_len
    for i in range(len(tjs)):
        _tjs[i] = np.pad(tjs[i], ((0, max_len - len(tjs[i])), (0, 0)), mode='edge')
    
    return np.array(_tjs).mean(axis=0)


def mean_unbalanced_data(data, min_valid_n=3):
    '''
    data is a list with different length np.array
    average them in axis=0
    '''
    _data = copy.deepcopy(data)
    max_len = 0
    for d in data:
        max_len = len(d) if len(d) > max_len else max_len
    for i in range(len(data)):
        _data[i] = np.pad(
            data[i], 
            (0, max_len - len(data[i])), 
            mode='constant', 
            constant_values=(np.nan, np.nan)
        )
    _data = np.array(_data)

    count_valid_num = np.sum(~np.isnan(np.array(_data)), axis=0)
    
    return np.nanmean(_data, axis=0)[count_valid_num >= min_valid_n]


def mean_distance(list_ts, list_tx, list_gx, mean_tct):
    max_ts = np.max(np.concatenate(list_ts))
    new_ts = np.linspace(0, max_ts, int(100 * max_ts))

    new_tx, new_gx = [], []
    for i in range(len(list_ts)):
        _tx = np.interp(
            new_ts[new_ts < list_ts[i][-1]],
            list_ts[i],
            list_tx[i]
        )
        _gx = np.interp(
            new_ts[new_ts < list_ts[i][-1]],
            list_ts[i],
            list_gx[i]
        )
        new_tx.append(_tx)
        new_gx.append(_gx)
    
    mean_tx = mean_unbalanced_data(new_tx, min_valid_n=10)
    mean_gx = mean_unbalanced_data(new_gx, min_valid_n=10)

    new_ts = new_ts[new_ts <= mean_tct]
    mean_tx = mean_tx[:new_ts.size]
    mean_gx = mean_gx[:new_ts.size]

    return new_ts, mean_tx, mean_gx


def apply_gaussian_filter(d, size=25, sigma=3):
    # Gaussian filter
    arr = np.arange(size // 2 * (-1), size // 2 + 1)
    gf = np.exp(-np.power(arr, 2) / (2 * sigma**2))
    gf = np.flip(gf / gf.sum())

    return np.convolve(
        np.pad(d, size // 2, mode='edge'), 
        gf, 
        mode='valid'
    )


def apply_gaussian_filter_2d(d, size=25, sigma=3):
    return np.array([
        apply_gaussian_filter(d[:,0], size=size, sigma=sigma),
        apply_gaussian_filter(d[:,1], size=size, sigma=sigma)
    ]).T


def gaussian(x, A, sigma):
    return A * np.exp(-x**2/2./sigma**2)


def derivative_central(x, y):
    xp = np.pad(x, 1, mode='edge')
    yp = np.pad(y, 1, mode='edge')
    return (yp[2:] - yp[:-2]) / (xp[2:] - xp[:-2])


def intersections(y, y0, consider_sign=False):
    signs = np.sign(y - y0)
    if consider_sign:
        sign_conv = signs[1:] - signs[:-1]
        # If positive, y is increasing. Otherwise, decreasing
        return np.where(sign_conv > 0)[0], np.where(sign_conv < 0)[0] + 1
    else:
        sign_conv = signs[1:] * signs[:-1]
        return np.where(sign_conv <= 0)[0]


def dist_point2line(p, a, d):
    '''Distance from p to line: a, vec d'''
    # Distance btw point P and line (A, vec d) = norm(vec AP ^ d) / norm(d)
    nom = np.abs(np.cross(p - a, d))
    denom = np.linalg.norm(d)
    if denom == 0: return nom
    else: return nom / denom


def int_floor(n):
    return int(np.floor(n))


def float_round(n, d):
    '''Make n/d be integer'''
    return d * int_floor(n/d)


def v_normalize(x, x_min, x_max):
    return (2*x - (x_min + x_max)) / (x_max - x_min)


def v_denormalize(z, x_min, x_max):
    return (z * (x_max - x_min) + (x_min + x_max)) / 2


# Log scale sampling
def convert_z2w(z, m, s, r=MAX_SIGMA_SCALE):
    return m * (1 + r * s/m) ** z


def convert_w2z(w, m, s, r=MAX_SIGMA_SCALE):
    return np.log(w / m) / np.log(1 + r * s/m)

# Compute mean and std using given min~max range
def logscale_mean_std(_max, _min, r=MAX_SIGMA_SCALE):
    return np.sqrt(_max*_min), (_max - np.sqrt(_max*_min))/r


def parameter_range(m, s, r=MAX_SIGMA_SCALE):
    for param in m.keys():
        if param in s.keys():
            _max = convert_z2w(1, m[param], s[param], r=r)
            _min = convert_z2w(-1, m[param], s[param], r=r)
            print(f"{param}\t- MIN: {_min:.4f}\t\tMAX: {_max:.4f}")



def np_interp_2d(x, xp, fp):
    """fp.shape == (n, 2)"""
    return np.array(
        [
            np.interp(x, xp, fp[:,0]),
            np.interp(x, xp, fp[:,1]),
        ]
    ).T


def np_interp_3d(x, xp, fp):
    """fp.shape == (n, 2)"""
    return np.array(
        [
            np.interp(x, xp, fp[:,0]),
            np.interp(x, xp, fp[:,1]),
            np.interp(x, xp, fp[:,2]),
        ]
    ).T


def peak_flag(x, i):
    if i == 0:
        if x[i] <= x[i+1]: return 1
        else: return 0
    elif i == len(x) - 1:
        if x[i] <= x[i-1]: return -1
        else: return 0
    else:
        if x[i-1] <= x[i] <= x[i+1]: return 1
        if x[i-1] > x[i] > x[i+1]: return -1
        if x[i-1] <= x[i] and x[i+1] <= x[i]: return 0
        if x[i-1] > x[i] and x[i+1] > x[i]: return -2
        return 0


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1-s)


def traj_length(points):
    """Length of given trajectory"""
    return np.sum(np.linalg.norm(
        points[1:] - points[:-1], axis=1
    ))



def geometric_entropy(points):
    try:
        h = ConvexHull(points)
        outline_points = points[np.append(h.vertices, h.vertices[0])]
        perimeter = traj_length(outline_points)
        path_length = traj_length(points)
        return np.log(2 * path_length / perimeter)
    except:
        # print("Something wrong on getting convex hull ...")
        return 0



def save_random_z(size, fn):
    z = np.random.uniform(-1, 1, size=size)
    pickle_save(f"{PATH_MAT}{fn}.pkl", z)



def img_resolution(res):
    return (res//9*16, res)


def compute_r2(x, y):
    fit = np.polyfit(x, y, 1)
    func = np.poly1d(fit)
    r2 = r2_score(y, func(x))
    return func, r2



if __name__ == "__main__":
    parameter_range(USER_PARAM_MEAN, USER_PARAM_STD)
    save_random_z((256, 4), "cog256")
    save_random_z((256, 3), "rew256")
    save_random_z((256, 7), "all256")

    # import matplotlib.pyplot as plt
    # w = []
    # z = []
    # for _ in range(10000):
    #     _z = np.random.uniform(-1, 1)
    #     z.append(_z)
    #     _w = convert_z2w(_z, 0.11, 0.031, r=3)
    #     w.append(_w)
    
    # plt.hist(w, bins=50)
    # plt.show()