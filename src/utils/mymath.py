import sys
sys.path.append("..")

from utils.utils import *
from configs.common import *
from configs.simulation import *
from configs.path import *

import copy
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import scipy as sp
from scipy import stats
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
# from sklearn.metrics import r2_score




def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize the size of vector to 1."""
    norm = np.linalg.norm(vector)
    if norm == 0: return vector
    return vector / norm


### Random functions
def random_sign():
    return np.random.choice([1, -1])


def rotation_matrix_2d(angle: float, angle_is_degree: bool=True) -> np.ndarray:
    """Note that positive angle rotation corresponds to counter-clockwise rotation"""
    angle_radians = np.radians(angle) if angle_is_degree else angle
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    return np.array([
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle]
    ])


def rotate_point_2d(p, theta, theta_is_degree=True):
    """Rotate p by theta in 2D plane"""
    # assert type(p) is np.ndarray and p.size == 2
    return rotation_matrix_2d(theta, theta_is_degree) @ p


# Old name - rot_mat_3d
def rotation_matrix_3d(angle: float, ax: int, angle_is_degree: bool = True) -> np.ndarray:
    """
    Generates a 3D rotation matrix for a right-handed coordinate system.

    Parameters:
        angle (float): The rotation angle in degrees or radians, based on the 'angle_is_degree' parameter.
        axis (str|int): The rotation axis, which can be one of 'x', 'y', 'z', 0, 1, or 2.
        angle_is_degree (bool, optional): Determines whether the given angle is in degrees (True) or radians (False).

    Returns:
        np.ndarray: A rotation matrix corresponding to the specified axis and angle.
    """
    # Generate a 2D rotation matrix
    rm = rotation_matrix_2d(angle, angle_is_degree=angle_is_degree)
    
    # Insert the 2D matrix into the 3D matrix to represent the specified axis
    rm = np.insert(rm, ax, 0, axis=0)
    rm = np.insert(rm, ax, 0, axis=1)
    
    # Set the diagonal element for the specified axis to 1
    rm[ax][ax] = 1
    
    return rm


def rotate_point_around_axis(point, angle, axis_az, axis_el, angle_is_degree=True):
    """Rotate p by theta around axis represented by (az, el)"""
    assert type(point) is np.ndarray and point.size == 3
    a0 = rotation_matrix_3d(-axis_az, Y, angle_is_degree)
    e0 = rotation_matrix_3d(-axis_el, Z, angle_is_degree)
    t = rotation_matrix_3d(angle, X, angle_is_degree)
    e1 = rotation_matrix_3d(axis_el, Z, angle_is_degree)
    a1 = rotation_matrix_3d(axis_az, Y, angle_is_degree)
    return a1 @ e1 @ t @ e0 @ a0 @ point


### Monitor space <-> Game space converting functions
def sphr2cart(az, el, angle_is_degree=True):
    """
    Azim Elev -> X, Y, Z
    Input: (n,), (n,)
    Output: (n, 3)
    """
    a = np.radians(az) if angle_is_degree else az
    e = np.radians(el) if angle_is_degree else el
    return np.array([np.cos(a)*np.cos(e), np.sin(e), np.sin(a)*np.cos(e)]).T
    

def cart2sphr(x, y, z, return_in_degree=True):
    """
    X, Y, Z -> Azim, Elev
    Input: (n,), (n,), (n,)
    Output: (n, 2)
    """
    ae = np.array([np.arctan2(z, x), np.arctan2(y, np.sqrt(x**2+z**2))]).T
    return np.degrees(ae) if return_in_degree else ae


def perpendicular_vector(vec, a=None):
    """Return random (or a deg rotated) perpendicular vector of given vec"""
    r = np.cross(vec, UP)
    if a is None:
        r = rotate_point_around_axis(r, np.random.uniform(0, 360), *cart2sphr(*vec))
    else:
        r = rotate_point_around_axis(r, 90-a, *cart2sphr(*vec))
    return cart2sphr(*r)


def angle_between(v1, v2, return_in_degree=True):
    cos_a = np.sum(v1 * v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    a = np.arccos(np.clip(cos_a, -1, 1))
    return np.degrees(a) if return_in_degree else a


# Scalar
def camera_matrix(az, el):
    """Return front, right, up vector"""
    f = sphr2cart(az, el)
    r = np.cross(f, UP)
    u = np.cross(r, f)
    return np.vstack((f, r, u)).T



def target_monitor_position(ppos, pcam, tgpos, fov=FOV, mbound=MONITOR_BOUND):
    """Return target position on monitor"""
    cam_mat = camera_matrix(*pcam)
    vec_pt = tgpos - ppos
    [a, b, c], *_ = np.linalg.lstsq(cam_mat, vec_pt, rcond=-1)
    return (np.array([b, c]) / (np.tan(fov/2)*a)) * mbound


def target_game_position(ppos, pcam, tmpos, fov=FOV, mbound=MONITOR_BOUND):
    """Return target position in game"""
    [f, r, u] = camera_matrix(*pcam).T
    coeff = tmpos / mbound * np.tan(fov/2)
    cos_theta = 1 / np.sqrt(1 + np.sum(coeff**2))
    return cos_theta * (f + coeff[0] * r + coeff[1] * u) + ppos


def target_multi_pos_monitor(ppos, pcam, tgpos, fov=FOV, mbound=MONITOR_BOUND):
    """
    Parameter formats:
    ppos: numpy array with shape [n, 3]. Columns X, Y, Z in order
    pcam: numpy array with shape [n, 2]. Columns azim, elev in order
    tpos: numpy array with shape [n, 3]. Columns X, Y, Z in order
    """
    tmpos = np.array(
        [target_monitor_position(pp, pc, tg, fov=fov, mbound=mbound) for pp, pc, tg in zip(ppos, pcam, tgpos)]
    )
    return tmpos


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
    tgpos_0 = target_game_position(ppos, pcam, tmpos_0, fov=fov, mbound=mbound)
    tgpos_n = target_game_position(ppos, pcam, tmpos_n, fov=fov, mbound=mbound)
    cam_adjust = cart2sphr(*tgpos_n) - cart2sphr(*tgpos_0)
    hand_adjust = cam_adjust / sensi
    return hand_adjust / dt




def eccentricity_distance(p1, p2, head_pos=HEAD_POSITION, return_in_degree=False):
    return angle_between(
        np.array([*p1, 0]) - head_pos,
        np.array([*p2, 0]) - head_pos,
        return_in_degree=return_in_degree
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


def mean_unbalanced_data(data, min_valid_n=3, return_err=False):
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
    
    if not return_err:
        return np.nanmean(_data, axis=0)[count_valid_num >= min_valid_n]
    else:
        return (
            np.nanmean(_data, axis=0)[count_valid_num >= min_valid_n],
            1.96 * np.nanstd(_data, axis=0)[count_valid_num >= min_valid_n] / np.sqrt(count_valid_num[count_valid_num >= min_valid_n]),
            np.nanstd(_data, axis=0)[count_valid_num >= min_valid_n]
        )


def mean_distance(list_ts, list_tx, list_gx, mean_tct, return_error=False, min_n=10):
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
    
    if not return_error:
        mean_tx = mean_unbalanced_data(new_tx, min_valid_n=min_n)
        mean_gx = mean_unbalanced_data(new_gx, min_valid_n=min_n)

        new_ts = new_ts[new_ts <= mean_tct]
        mean_tx = mean_tx[:new_ts.size]
        mean_gx = mean_gx[:new_ts.size]

        return new_ts, mean_tx, mean_gx
    else:
        mean_tx, _, std_tx = mean_unbalanced_data(new_tx, min_valid_n=min_n, return_err=True)
        mean_gx, _, std_gx = mean_unbalanced_data(new_gx, min_valid_n=min_n, return_err=True)

        new_ts = new_ts[new_ts <= mean_tct]
        mean_tx = mean_tx[:new_ts.size]
        mean_gx = mean_gx[:new_ts.size]
        std_tx = std_tx[:new_ts.size]
        std_gx = std_gx[:new_ts.size]

        return new_ts, mean_tx, mean_gx, std_tx, std_gx


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


def linear_normalize(w, v_min, v_max, dtype=np.float32):
    return np.clip((2*w - (v_min + v_max)) / (v_max - v_min), -1, 1, dtype=dtype)


def linear_denormalize(z, v_min, v_max, dtype=np.float32):
    return np.clip((z * (v_max - v_min) + (v_min + v_max)) / 2, v_min, v_max, dtype=dtype)


def log_normalize(w, v_min, v_max, scale=1, dtype=np.float32):
    new_z = (2*w - (v_max + v_min)) / (v_max - v_min)
    return np.clip(2 * (np.log(1 + (np.e*scale - 1) * ((new_z + 1)/2)) / (np.log(scale) + 1)) - 1, -1, 1, dtype=dtype) if scale != 0 else dtype(new_z)


def log_denormalize(z, v_min, v_max, scale=1, dtype=np.float32):
    new_z = ((np.e*scale) ** ((z+1)/2) - 1) / (np.e*scale - 1) * 2 - 1 if scale != 0 else dtype(z)
    return np.clip((new_z * (v_max - v_min) + v_min + v_max) / 2, v_min, v_max, dtype=dtype)


# # Log scale sampling, with max and min
# def convert_z2w(z, v_max, v_min, scale=MAX_SIGMA_SCALE):
#     new_z = ((np.e*scale) ** ((z+1)/2) - 1) / (np.e*scale - 1) * 2 - 1 if scale != 0 else z
#     return (new_z * (v_max - v_min) + v_min + v_max) / 2

# def convert_w2z(w, v_max, v_min, scale=MAX_SIGMA_SCALE):
#     new_z = (2*w - (v_max + v_min)) / (v_max - v_min)
#     return 2 * (np.log(1 + (np.e*scale - 1) * ((new_z + 1)/2)) / (np.log(scale) + 1)) - 1 if scale != 0 else new_z


def np_interp_nd(x, xp, fp):
    return np.array([np.interp(x, xp, _fp) for _fp in fp.T]).T


# def np_interp_2d(x, xp, fp):
#     """fp.shape == (n, 2)"""
#     return np.array(
#         [
#             np.interp(x, xp, fp[:,0]),
#             np.interp(x, xp, fp[:,1]),
#         ]
#     ).T


# def np_interp_3d(x, xp, fp):
#     """fp.shape == (n, 2)"""
#     return np.array(
#         [
#             np.interp(x, xp, fp[:,0]),
#             np.interp(x, xp, fp[:,1]),
#             np.interp(x, xp, fp[:,2]),
#         ]
#     ).T


def sp_interp_nd(x, xp, fp, n=2, kind='quadratic'):
    return np.array(
        [
            sp.interpolate.interp1d(
                xp, fp[:,i], 
                kind=kind, 
                bounds_error=False,
                fill_value=(fp[:,i][0], fp[:,i][-1])
            )(x) for i in range(n)
        ]
    ).T




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


# def compute_r2(x, y):
#     fit = np.polyfit(x, y, 1)
#     func = np.poly1d(fit)
#     r2 = r2_score(y, func(x))
#     return func, r2


def get_r_squared(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2
    return r_squared


def get_adjusted_r_squared(x, y, p):
    r2 = get_r_squared(x, y)
    n = len(x)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def cl_95_intv(data):
    z = 1.96
    err = np.std(data) / np.sqrt(len(data))
    return z * err


def find_divisors(N):
    for i in range(int(np.sqrt(N)), 0, -1):
        if N % i == 0:
            return i, N // i


def discrete_labeling(d, lvl=8):
    # Return label of "index of difficulty"
    # points = np.linspace(_min, _max, lvl)
    # label = np.array([np.argmin(np.abs(points - v)) for v in d])
    # return label

    # s = len(d)
    # _d = np.sort(d)
    # lvs = [_d[(lv*s)//lvl] for lv in range(1, lvl)]
    # _label = np.searchsorted(lvs, d)
    # return _label

    lvs = [np.percentile(d, q) for q in np.linspace(0, 100, lvl+1)[1:-1]]
    _label = np.searchsorted(lvs, d)
    return _label



def np_groupby(a, key_pos=0, mode=np.mean, lvl=8):
    assert len(a.shape) == 2

    _label = discrete_labeling(a[:,key_pos], lvl=lvl)
    n = np.unique(_label)
    return np.array([mode(a[_label == i], axis=0) for i in n])


def minimum_jerk(t, D, T0, dur):
    c = np.zeros(t.size)
    if dur <= 0: return c
    mask_time = (t-T0)*(t-T0-dur) <= 0
    t_ = t[mask_time]
    tau = (t_ - T0) / dur
    c[mask_time] = 30 * D / dur * (
        tau ** 4 - 2 * tau ** 3 + tau ** 2
    )
    return c


def kl_divergence(p, q):
    return stats.entropy(p / p.sum(), q / q.sum())


def dtw(traj1, traj2):
    # Dynamic Time Warping
    dist, _ = fastdtw(traj1, traj2, dist=euclidean)
    return dist



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    xx = np.linspace(0.0, 0.14, 20)
    yy = np.linspace(0.0, 0.08, 20)

    for x in xx:
        for y in yy:
            g = target_game_position(np.zeros(3), np.zeros(2), np.array([x, y]))
            m = target_monitor_position(np.zeros(3), np.zeros(2), g)
            plt.scatter(*m, s=3, color='r', zorder=0)
            plt.scatter(x, y, s=1, color='k', zorder=10)

    plt.show()

    # plt.scatter(x, y, s=0.1)
    # plt.show()

    pass


### GRAVEYARD

# def peak_flag(x, i):
#     if i == 0:
#         if x[i] <= x[i+1]: return 1
#         else: return 0
#     elif i == len(x) - 1:
#         if x[i] <= x[i-1]: return -1
#         else: return 0
#     else:
#         if x[i-1] <= x[i] <= x[i+1]: return 1
#         if x[i-1] > x[i] > x[i+1]: return -1
#         if x[i-1] <= x[i] and x[i+1] <= x[i]: return 0
#         if x[i-1] > x[i] and x[i+1] > x[i]: return -2
#         return 0