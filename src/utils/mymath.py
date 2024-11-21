import numpy as np
from scipy import stats

from ..config.constant import AXIS, VECTOR


def normalize_vector(vector, tol=1e-6):
    """Normalize the size of vector to 1."""
    norm = np.linalg.norm(vector)
    if norm <= tol: return vector
    return vector / norm


def angle_between(v1, v2, return_in_degree=True):
    cos_a = np.sum(v1 * v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    a = np.arccos(np.clip(cos_a, -1, 1))
    return np.degrees(a) if return_in_degree else a


def cos_sin_array(a):
    radians = np.radians(a)
    return np.array([np.cos(radians), np.sin(radians)])


def perpendicular_vector(vector, angle=None):
    v = normalize_vector(vector)
    r = np.cross(v, VECTOR.UP)
    if angle is None:
        angle = 90 - np.random.uniform(0, 360)
    r = Rotate.point_about_axis(r, angle, *Convert.cart2sphr(*v))
    return Convert.cart2sphr(*r)


class Rotate:
    """
    TBU: UPDATE TO QUATERNION OPERATION
    """
    def matrix_2d(angle: float, angle_is_degree: bool=True) -> np.ndarray:
        """Note that positive angle rotation corresponds to counter-clockwise rotation"""
        angle_radians = np.radians(angle) if angle_is_degree else angle
        cos_angle = np.cos(angle_radians)
        sin_angle = np.sin(angle_radians)
        return np.array([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])

    def point_2d(p, theta, theta_is_degree=True):
        """Rotate p by theta in 2D plane"""
        # assert type(p) is np.ndarray and p.size == 2
        return Rotate.matrix_2d(theta, theta_is_degree) @ p
    
    def matrix_3d(angle: float, ax: int, angle_is_degree: bool = True) -> np.ndarray:
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
        rm = Rotate.matrix_2d(angle, angle_is_degree=angle_is_degree)
        
        # Insert the 2D matrix into the 3D matrix to represent the specified axis
        rm = np.insert(rm, ax, 0, axis=0)
        rm = np.insert(rm, ax, 0, axis=1)
        
        # Set the diagonal element for the specified axis to 1
        rm[ax][ax] = 1
        
        return rm


    def point_about_axis(point, angle, axis_az, axis_el, angle_is_degree=True):
        """Rotate p by theta around axis represented by (az, el)"""
        assert type(point) is np.ndarray and point.size == 3
        a0 = Rotate.matrix_3d(-axis_az, AXIS.Y, angle_is_degree)
        e0 = Rotate.matrix_3d(-axis_el, AXIS.Z, angle_is_degree)
        t = Rotate.matrix_3d(angle, AXIS.X, angle_is_degree)
        e1 = Rotate.matrix_3d(axis_el, AXIS.Z, angle_is_degree)
        a1 = Rotate.matrix_3d(axis_az, AXIS.Y, angle_is_degree)
        return a1 @ e1 @ t @ e0 @ a0 @ point

        



class Convert:
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
    

    def _camera_matrix(az, el):
        # Return front, right, up vector
        f = Convert.sphr2cart(az, el)
        r = np.cross(f, VECTOR.UP)
        u = np.cross(r, f)
        return np.vstack((f, r, u)).T
    

    def game2monitor(ppos, pcam, gpos, fov, monitor_qt):
        # Return target position on monitor. (0, 0) is the center
        # fov unit: degree
        cam_mat = Convert._camera_matrix(*pcam)
        vec_pt = gpos - ppos
        [a, b, c], *_ = np.linalg.lstsq(cam_mat, vec_pt, rcond=-1)
        return (np.array([b, c]) / (np.tan(np.radians(fov)/2)*a)) * monitor_qt


    def monitor2game(ppos, pcam, mpos, fov, monitor_qt):
        """Return target position in game"""
        [f, r, u] = Convert._camera_matrix(*pcam).T
        coeff = mpos / monitor_qt * np.tan(np.radians(fov)/2)
        cos_theta = 1 / np.sqrt(1 + np.sum(coeff**2))
        return cos_theta * (f + coeff[0] * r + coeff[1] * u) + ppos


    def games2monitors(ppos, pcam, gpos, fov, monitor_qt):
        """
        Parameter formats:
        ppos: numpy array with shape [n, 3]. Columns X, Y, Z in order
        pcam: numpy array with shape [n, 2]. Columns azim, elev in order
        gpos: numpy array with shape [n, 3]. Columns X, Y, Z in order
        """
        tmpos = np.array(
            [Convert.game2monitor(pp, pc, tg, fov, monitor_qt) for pp, pc, tg in zip(ppos, pcam, gpos)]
        )
        return tmpos
    

    def orbit2direction(ppos, pcam, gpos, orbit, fov, monitor_qt, dangle=1, sample=5):
        # Left X-axis := 0 degree, counter-clockwise
        # Sample point orbit trajectory
        glist = [gpos]
        for _ in range(sample-1):
            glist.append(Rotate.point_about_axis(glist[-1], dangle, *orbit))
        glist = np.array(glist)
        # Compute monitor position
        mlist = Convert.games2monitors(np.tile(ppos, (sample, 1)), np.tile(pcam, (sample, 1)), glist, fov, monitor_qt)
        # Compute angle and return
        dir = np.diff(mlist, axis=0).mean(axis=0)
        return np.degrees(np.arctan2(dir[1], dir[0]))


def linear_normalize(w, v_min, v_max, dtype=np.float32, clip=True):
    return np.clip((2*w - (v_min + v_max)) / (v_max - v_min), -1, 1, dtype=dtype)


def linear_denormalize(z, v_min, v_max, dtype=np.float32, clip=True):
    return np.clip((z * (v_max - v_min) + (v_min + v_max)) / 2, v_min, v_max, dtype=dtype)


def log_normalize(w, v_min, v_max, scale=1, dtype=np.float32):
    new_z = (2*w - (v_max + v_min)) / (v_max - v_min)
    return np.clip(2 * (np.log(1 + (np.e*scale - 1) * ((new_z + 1)/2)) / (np.log(scale) + 1)) - 1, -1, 1, dtype=dtype) if scale != 0 else dtype(new_z)


def log_denormalize(z, v_min, v_max, scale=1, dtype=np.float32):
    new_z = ((np.e*scale) ** ((z+1)/2) - 1) / (np.e*scale - 1) * 2 - 1 if scale != 0 else dtype(z)
    return np.clip((new_z * (v_max - v_min) + v_min + v_max) / 2, v_min, v_max, dtype=dtype)


def get_r_squared(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2
    return r_squared


def np_interp_nd(x, xp, fp):
    return np.array([np.interp(x, xp, _fp) for _fp in fp.T]).T


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
    

def index_of_difficulty(width, distance):
    return np.log2(1 + distance / width)



def discrete_labeling(d, lvl=8):
    lvs = [np.percentile(d, q) for q in np.linspace(0, 100, lvl+1)[1:-1]]
    _label = np.searchsorted(lvs, d)
    return _label


def np_groupby(a, key_pos=0, mode=np.mean, lvl=8):
    assert len(a.shape) == 2

    _label = discrete_labeling(a[:,key_pos], lvl=lvl)
    n = np.unique(_label)
    return np.array([mode(a[_label == i], axis=0) for i in n])


def get_r_squared(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2
    return r_squared


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


def random_sampler(_min, _max, _min_sample=0, _max_sample=0, size=1):
    assert _min_sample + _max_sample <= 1

    den = 1 - (_min_sample + _max_sample)
    z = np.clip(np.random.uniform(-_min_sample / den, 1 + _max_sample / den, size=size), 0, 1)
    z = np.clip(z * (_max - _min) + _min, _min, _max)
    if z.size == 1:
        return z[0]
    return z


def random_sign():
    return 1 if np.random.uniform(0, 1) > 0.5 else -1