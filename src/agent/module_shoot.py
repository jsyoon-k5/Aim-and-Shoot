"""
Shoot Action Module

Intermittent Click Planning (ICP) style click timing model migrated from the
2023 Aim-and-Shoot codebase.

The central method is :meth:`Shoot.sample_shoot_timing`.  It estimates the
future time at which the target disk reaches the crosshair and then samples a
noisy click time from the ICP clock model.

Reference:
1) "An Intermittent Click Planning Model", Eunji Park and Byungjoo Lee
   https://dl.acm.org/doi/abs/10.1145/3313831.3376725
"""

from __future__ import annotations

import numpy as np

from ..configs.constants import TINTERVAL


_EPS = 1e-12
_MIN_CUE_TIME_S = 1e-6
_DEFAULT_CMU = 0.185
_DEFAULT_NU = 20.524
_DEFAULT_DELTA = 0.411


def _as_vec2(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 2:
        raise ValueError(f"{name} must be a 2D vector, got shape {np.shape(value)}")
    return arr


def _crossing_window(
    target_pos,
    crosshair_pos,
    target_vel,
    target_radius,
) -> tuple[float, float]:
    """Return future disk-entry cue time and visible time window in seconds.

    The target center is assumed to move linearly on the monitor:

        target(t) = target_pos + target_vel * t

    A click is geometrically valid while
    ``||target(t) - crosshair_pos|| <= target_radius``.  This helper solves the
    corresponding quadratic inequality directly, which keeps the moving-away
    and already-inside cases explicit.

    Returns
    -------
    cue_time_s
        First future time at which the target disk is relevant.  If the disk is
        already covering the crosshair, this is a tiny positive time.  If the
        disk never intersects in the future, this is the future closest-approach
        time and the window is zero.
    window_time_s
        Duration for which the disk covers the crosshair after cue_time_s.
    """
    target_pos = _as_vec2(target_pos, "target_pos")
    crosshair_pos = _as_vec2(crosshair_pos, "crosshair_pos")
    target_vel = _as_vec2(target_vel, "target_vel")
    radius = float(target_radius)
    if radius < 0:
        raise ValueError("target_radius must be non-negative")

    rel_pos = target_pos - crosshair_pos
    speed_sq = float(np.dot(target_vel, target_vel))
    current_margin = float(np.dot(rel_pos, rel_pos) - radius * radius)

    if speed_sq <= _EPS:
        return _MIN_CUE_TIME_S, 0.0

    b = 2.0 * float(np.dot(rel_pos, target_vel))
    discriminant = b * b - 4.0 * speed_sq * current_margin

    if discriminant < 0.0:
        closest_time = max(-b / (2.0 * speed_sq), _MIN_CUE_TIME_S)
        return closest_time, 0.0

    sqrt_disc = float(np.sqrt(max(discriminant, 0.0)))
    t_entry = (-b - sqrt_disc) / (2.0 * speed_sq)
    t_exit = (-b + sqrt_disc) / (2.0 * speed_sq)

    if t_exit < _MIN_CUE_TIME_S:
        return _MIN_CUE_TIME_S, 0.0

    cue_time = max(t_entry, _MIN_CUE_TIME_S)
    if current_margin <= 0.0:
        cue_time = _MIN_CUE_TIME_S

    window_time = max(t_exit - cue_time, 0.0)
    return float(cue_time), float(window_time)


class Shoot:
    def icp(
        Wt,
        tc,
        param_clock_noise,
        cmu: float = _DEFAULT_CMU,
        nu: float = _DEFAULT_NU,
        delta: float = _DEFAULT_DELTA,
        interp_interval_ms: int = TINTERVAL.INTERP1,
    ):
        """Sample a click time in seconds using the ICP noise equation.

        Parameters
        ----------
        Wt : float
            Target-width time window in seconds.
        tc : float
            Cue-viewing time in seconds.
        param_clock_noise : float
            Clock/timing noise parameter used by the current agent configs.
        cmu, nu, delta : float
            ICP timing constants.
        interp_interval_ms : int
            Lower bound on returned time, matching the simulation interpolation
            resolution.
        """
        Wt = max(float(Wt), 0.0)
        tc = max(float(tc), _MIN_CUE_TIME_S)
        param_clock_noise = float(param_clock_noise)
        cmu = float(cmu)
        nu = float(nu)
        delta = float(delta)
        min_time_s = float(interp_interval_ms) / 1000.0

        mean = Wt * cmu
        if param_clock_noise > 0.0:
            sigma_t = param_clock_noise * tc
            sigma_v = param_clock_noise * (
                1.0 / (np.exp(np.clip(nu * tc, 1e-5, 10.0)) - 1.0)
                + delta
            )
            sigma = np.sqrt(
                sigma_t * sigma_t * sigma_v * sigma_v
                / max(sigma_t * sigma_t + sigma_v * sigma_v, _EPS)
            )
        else:
            sigma = 0.0

        total_time = tc + np.random.normal(mean, sigma)
        return float(np.clip(total_time, min_time_s, np.inf))


    def sample_shoot_timing(
        tpos,
        cpos,
        tgvel,
        tavel,
        trad,
        param_clock_noise,
        cmu: float = _DEFAULT_CMU,
        nu: float = _DEFAULT_NU,
        delta: float = _DEFAULT_DELTA,
        interp_interval_ms: int = TINTERVAL.INTERP1,
    ):
        """Sample shoot timing from current target/crosshair geometry.

        Parameters
        ----------
        tpos, cpos : array-like, shape (2,)
            Target center and crosshair positions on the monitor in mm.
        tgvel, tavel : array-like, shape (2,)
            Target velocity terms in mm/s.  As in the 2023 implementation, the
            effective target velocity is ``tgvel + tavel``.
        trad : float
            Target radius in mm.
        param_clock_noise : float
            Clock/timing noise parameter used by the current agent configs.
        cmu, nu, delta : float
            ICP timing constants.
        interp_interval_ms : int
            Minimum returned timing resolution in milliseconds.

        Returns
        -------
        float
            Sampled click timing in seconds from the current decision moment.
        """
        target_vel = _as_vec2(tgvel, "tgvel") + _as_vec2(tavel, "tavel")
        cue_time, width_time = _crossing_window(tpos, cpos, target_vel, trad)
        return Shoot.icp(
            width_time,
            cue_time,
            param_clock_noise,
            cmu=cmu,
            nu=nu,
            delta=delta,
            interp_interval_ms=interp_interval_ms,
        )


    def internal_clock_noise(param_clock_noise):
        """Sample multiplicative clock noise with the ICP truncation rule."""
        while True:
            noise = np.random.normal(1.0, float(param_clock_noise))
            if 0.001 < noise < 2.0:
                return float(noise)


if __name__ == "__main__":
    cue, width = _crossing_window([-10.0, 0.0], [0.0, 0.0], [10.0, 0.0], 1.0)
    assert np.isclose(cue, 0.9)
    assert np.isclose(width, 0.2)
    assert np.isclose(
        Shoot.sample_shoot_timing(
            [-10.0, 0.0],
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 0.0],
            1.0,
            param_clock_noise=0.0,
            cmu=0.0,
        ),
        0.9,
    )

    cue, width = _crossing_window([0.0, 0.0], [0.0, 0.0], [10.0, 0.0], 1.0)
    assert np.isclose(cue, _MIN_CUE_TIME_S)
    assert np.isclose(width, 0.1 - _MIN_CUE_TIME_S)

    cue, width = _crossing_window([10.0, 0.0], [0.0, 0.0], [10.0, 0.0], 1.0)
    assert np.isclose(cue, _MIN_CUE_TIME_S)
    assert np.isclose(width, 0.0)

    cue, width = _crossing_window([-10.0, 1.0], [0.0, 0.0], [10.0, 0.0], 1.0)
    assert np.isclose(cue, 1.0)
    assert np.isclose(width, 0.0)

    print("Shoot.sample_shoot_timing geometry checks passed.")
