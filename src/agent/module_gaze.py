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

from ..utils.mymath import angle_between_vectors, minimum_jerk, normalize_vector


DEFAULT_HEAD_POSITION = np.array([0.0, 77.9, 575.0], dtype=float)
DEFAULT_MONITOR_HALF_SIZE_MM = np.array([531.3, 298.8], dtype=float) / 2.0
WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=float)
_EPS = 1e-12


def _screen_point_to_head_ray(point, head):
    return np.array([point[0], point[1], 0.0], dtype=float) - head


def _safe_perpendicular_axis(ray, reference=WORLD_UP):
    axis = np.cross(ray, reference)
    if np.linalg.norm(axis) > _EPS:
        return axis
    return np.cross(ray, np.array([1.0, 0.0, 0.0], dtype=float))


def _rotate_about_axis(vector, angle_deg, axis):
    """Rotate one 3D vector around one axis for scalar or vector angles."""
    vector = np.asarray(vector, dtype=float)
    angle_deg = np.asarray(angle_deg, dtype=float)
    axis = np.asarray(axis, dtype=float)

    axis_norm = np.linalg.norm(axis)
    if axis_norm <= _EPS:
        if angle_deg.ndim == 0:
            return vector.copy()
        return np.repeat(vector[None, :], angle_deg.size, axis=0)

    axis = axis / axis_norm
    theta = np.radians(angle_deg)
    cos_theta = np.cos(theta)[..., None]
    sin_theta = np.sin(theta)[..., None]

    cross = np.cross(axis, vector)
    dot = np.dot(vector, axis)
    return (
        vector * cos_theta
        + cross * sin_theta
        + axis * dot * (1.0 - cos_theta)
    )


def _ray_to_monitor_position(ray, head):
    ray = np.asarray(ray, dtype=float)
    head = np.asarray(head, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = head[2] / ray[..., 2]
        return head[:2] - scale[..., None] * ray[..., :2]


class Gaze:
    ### EMMA model
    def gaze_landing_point(
        curr,
        dest,
        head=DEFAULT_HEAD_POSITION,
        deviation=0.1,
        monitor_qt=DEFAULT_MONITOR_HALF_SIZE_MM,
    ):
        head = np.asarray(head, dtype=float)
        monitor_qt = np.asarray(monitor_qt, dtype=float)
        ecc = Gaze._eccentricity(curr, dest, head)
        land_error = np.random.normal(0, deviation*ecc)

        # Sample the estimated position in visual angle space
        head_to_dest = normalize_vector(_screen_point_to_head_ray(dest, head))
        head_to_dest_noisy = _rotate_about_axis(
            _rotate_about_axis(
                head_to_dest,
                land_error,
                _safe_perpendicular_axis(head_to_dest),
            ),
            np.random.uniform(0, 360),
            head_to_dest,
        )
        return np.clip(
            _ray_to_monitor_position(head_to_dest_noisy, head),
            -monitor_qt,
            monitor_qt
        )
    
    def _eccentricity(p0, p1, head):
        head = np.asarray(head, dtype=float)
        return angle_between_vectors(
            _screen_point_to_head_ray(p0, head),
            _screen_point_to_head_ray(p1, head),
            return_in_degree=True
        )


    def peak_velocity(amp, a_th=1, slope=35.6, va=78.7):
        return max(va, va + slope * (amp-a_th))

    def saccade_duration(amp, peak_vel):
        return 30 * amp / (16 * peak_vel)

    def saccade_speed(timestamp, delay, amp):
        dur = Gaze.saccade_duration(amp, Gaze.peak_velocity(amp))
        return minimum_jerk(timestamp, amp, delay, dur)


    def saccade_amplitude(timestamp, delay, amp):
        dur = Gaze.saccade_duration(amp, Gaze.peak_velocity(amp))
        if dur <= 0:
            return np.zeros_like(timestamp, dtype=float)

        tau = np.clip((timestamp - delay) / dur, 0.0, 1.0)
        return amp * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)


    def gaze_plan(
        g0, gn,
        delay:int=0,
        exe_until:int=100, 
        interp_intv:int=5, 
        head=DEFAULT_HEAD_POSITION,
    ):
        head = np.asarray(head, dtype=float)
        eye2g0 = _screen_point_to_head_ray(g0, head)
        eye2gn = _screen_point_to_head_ray(gn, head)
        amp = angle_between_vectors(eye2g0, eye2gn, return_in_degree=True)
        timestamp = np.linspace(0, exe_until, exe_until // interp_intv + 1, dtype=int)  # Millisec
        sac_spd = Gaze.saccade_speed(timestamp / 1000, delay / 1000, amp)
        amp_cum = np.cumsum(interp_intv / 1000 * (sac_spd[1:] + sac_spd[:-1]) / 2)
        amp_cum = np.insert(amp_cum, 0, 0)

        # return timestamp, sac_spd, amp_cum

        base_rot_axis = np.cross(eye2g0, eye2gn)
        norm_fix_vec = normalize_vector(eye2g0)
        rot_fix_vec = _rotate_about_axis(norm_fix_vec, amp_cum, base_rot_axis)

        gaze_traj = _ray_to_monitor_position(rot_fix_vec, head)


        return timestamp, gaze_traj
    


if __name__ == "__main__":
    t, g = Gaze.gaze_plan(np.zeros(2), np.ones(2) / 10, delay=20, interp_intv=1)
    
    import matplotlib.pyplot as plt
    plt.plot(t, g[:, 0], label="x")
    plt.plot(t, g[:, 1], label="y")
    plt.legend()
    plt.show()
