import numpy as np
import math
from scipy.interpolate import interp1d


def monitor_mm_to_view_angle_deg(offset_mm, monitor_span_mm, fov_deg):
    """Convert monitor offset/size in mm to view angle in degrees.

    Uses pinhole projection geometry:
    `offset_mm = (monitor_span_mm/2) * tan(theta) / tan(fov_deg/2)`
    -> `theta = atan((offset_mm/(monitor_span_mm/2)) * tan(fov_deg/2))`

    Notes:
    - `offset_mm` can be scalar or array.
    - sign is preserved (negative mm -> negative angle).
    """
    offset_mm = np.array(offset_mm, dtype=float)
    monitor_span_mm = float(monitor_span_mm)
    fov_deg = float(fov_deg)

    if monitor_span_mm <= 0.0:
        raise ValueError("monitor_span_mm must be positive")

    half_span = monitor_span_mm / 2.0
    scale = np.tan(np.radians(fov_deg) / 2.0)
    theta_rad = np.arctan((offset_mm / half_span) * scale)
    return np.degrees(theta_rad)


def view_angle_deg_to_monitor_mm(angle_deg, monitor_span_mm, fov_deg):
    """Convert view angle in degrees to monitor offset/size in mm.

    Inverse of :func:`monitor_mm_to_view_angle_deg`.  Uses pinhole geometry:
    `offset_mm = (monitor_span_mm/2) * tan(angle_rad) / tan(fov_rad/2)`

    Particularly useful for converting an angular target radius (deg) to its
    physical size in mm **at screen center** (i.e., when the target aligns
    with the camera direction).

    Notes:
    - `angle_deg` can be scalar or array.
    - sign is preserved (negative angle -> negative mm).
    """
    angle_deg = np.array(angle_deg, dtype=float)
    monitor_span_mm = float(monitor_span_mm)
    fov_deg = float(fov_deg)

    if monitor_span_mm <= 0.0:
        raise ValueError("monitor_span_mm must be positive")

    half_span = monitor_span_mm / 2.0
    scale = np.tan(np.radians(fov_deg) / 2.0)
    offset_mm = half_span * np.tan(np.radians(angle_deg)) / scale
    return offset_mm


class Convert:
    """Angle/coordinate conversion helpers for scalar and vector inputs."""

    @staticmethod
    def sphr2cart_scalar(az, el, angle_is_degree=True):
        """Convert scalar azimuth/elevation to Cartesian np.ndarray shape (3,)."""
        a = math.radians(az) if angle_is_degree else az
        e = math.radians(el) if angle_is_degree else el
        x = math.cos(a) * math.cos(e)
        y = math.sin(e)
        z = math.sin(a) * math.cos(e)
        return np.array([x, y, z], dtype=float)

    @staticmethod
    def cart2sphr_scalar(x, y, z, return_in_degree=True):
        """Convert scalar Cartesian (x, y, z) to np.ndarray (azimuth, elevation)."""
        az = math.atan2(z, x)
        el = math.atan2(y, math.sqrt(x * x + z * z))
        if return_in_degree:
            return np.array([math.degrees(az), math.degrees(el)], dtype=float)
        return np.array([az, el], dtype=float)

    @staticmethod
    def sphr2cart_vec(az, el, angle_is_degree=True):
        """Convert vectorized azimuth/elevation arrays to Cartesian array (n, 3)."""
        a = np.radians(az) if angle_is_degree else np.array(az)
        e = np.radians(el) if angle_is_degree else np.array(el)
        x = np.cos(a) * np.cos(e)
        y = np.sin(e)
        z = np.sin(a) * np.cos(e)
        return np.stack([x, y, z], axis=-1)

    @staticmethod
    def cart2sphr_vec(x, y, z, return_in_degree=True):
        """Convert vectorized Cartesian arrays to (azimuth, elevation) array (n, 2)."""
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        az = np.arctan2(z, x)
        el = np.arctan2(y, np.sqrt(x ** 2 + z ** 2))
        ae = np.stack([az, el], axis=-1)
        return np.degrees(ae) if return_in_degree else ae

    @staticmethod
    def camera_matrix_vec(az, el, angle_is_degree=True):
        """Return vectorized camera basis (front, right, up) for each angle pair.

        Delegates to camera_matrix_vec_fast (closed-form, no np.cross).
        Original np.cross implementation preserved as camera_matrix_vec_orig.
        """
        return Convert.camera_matrix_vec_fast(az, el, angle_is_degree=angle_is_degree)

    @staticmethod
    def camera_matrix_vec_orig(az, el, angle_is_degree=True):
        """Original implementation kept for reference / regression testing."""
        front = Convert.sphr2cart_vec(az, el, angle_is_degree=angle_is_degree)
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(front, world_up)
        up = np.cross(right, front)
        return np.stack([front, right, up], axis=1)

    @staticmethod
    def camera_matrix_vec_fast(az, el, angle_is_degree=True):
        """Optimized version of camera_matrix_vec — avoids np.cross entirely.

        For world_up = [0, 1, 0] the cross products are closed-form:
            front = [fx, fy, fz]  (sphr2cart)
            right = cross(front, world_up) = [-fz,  0,  fx]
            up    = cross(right, front)    = [-fx*fy,  fx²+fz²,  -fz*fy]

        This eliminates the ~37k np.cross calls (and their normalize_axis_tuple /
        moveaxis overhead) that dominate the simulation hot path.
        """
        a = np.radians(az) if angle_is_degree else np.asarray(az, dtype=float)
        e = np.radians(el) if angle_is_degree else np.asarray(el, dtype=float)
        cos_e = np.cos(e)
        fx = np.cos(a) * cos_e
        fy = np.sin(e)
        fz = np.sin(a) * cos_e
        if np.ndim(fx) == 0:
            # Scalar az/el — preserve original (3,3) column convention
            _fx = float(fx); _fy = float(fy); _fz = float(fz)
            front = np.array([_fx,   _fy,                 _fz     ])
            right = np.array([-_fz,  0.0,                 _fx     ])
            up    = np.array([-_fx*_fy, _fx*_fx+_fz*_fz, -_fz*_fy])
            return np.stack([front, right, up], axis=1)
        else:
            # Batch path — fill pre-allocated array, no np.stack calls
            n_out = fx.shape[0]
            out = np.empty((n_out, 3, 3), dtype=float)
            out[:, 0, 0] = fx;       out[:, 0, 1] = fy;               out[:, 0, 2] = fz
            out[:, 1, 0] = -fz;      out[:, 1, 1] = 0.0;              out[:, 1, 2] = fx
            fxfy = fx * fy;  fzfy = fz * fy
            out[:, 2, 0] = -fxfy;    out[:, 2, 1] = fx*fx + fz*fz;   out[:, 2, 2] = -fzfy
            return out

    @staticmethod
    def camera_matrix_scalar(az, el, angle_is_degree=True):
        """Return scalar camera basis (front, right, up) for one angle pair."""
        f = np.array(Convert.sphr2cart_scalar(az, el, angle_is_degree=angle_is_degree), dtype=float)
        r = np.cross(f, np.array([0.0, 1.0, 0.0]))
        u = np.cross(r, f)
        f = normalize_vector(f)
        r = normalize_vector(r)
        u = normalize_vector(u)
        return np.stack([f, r, u], axis=0)

    @staticmethod
    def game2monitor(
        camera_pos,
        camera_azel,
        target_pos,
        fov_deg,
        monitor_half_size,
        margin_mm=30.0,
        return_status=False,
    ):
        """Project world target position(s) to monitor coordinates centered at (0, 0).

        Supports both single sample and batched inputs:
        - single: `camera_pos(3), camera_azel(2), target_pos(3)` -> returns `(2,)`
        - batch : `camera_pos(n,3), camera_azel(n,2), target_pos(n,3)` -> returns `(n,2)`
        """
        # ── n=1 scalar fast path — avoids _as_rows / _broadcast_n / einsum overhead ──
        _cp = np.asarray(camera_pos,        dtype=float)
        _ae = np.asarray(camera_azel,       dtype=float)
        _tp = np.asarray(target_pos,        dtype=float)
        _mh = np.asarray(monitor_half_size, dtype=float)
        if _cp.shape == (3,) and _ae.shape == (2,) and _tp.shape == (3,) and _mh.shape == (2,):
            a_r = math.radians(_ae[0]);  e_r = math.radians(_ae[1])
            cos_e = math.cos(e_r)
            fx = math.cos(a_r)*cos_e;  fy = math.sin(e_r);  fz = math.sin(a_r)*cos_e
            relx = float(_tp[0]-_cp[0]);  rely = float(_tp[1]-_cp[1]);  relz = float(_tp[2]-_cp[2])
            a_d  = relx*fx  + rely*fy  + relz*fz
            b_d  = relx*(-fz)           + relz*fx
            c_d  = relx*(-fx*fy) + rely*(fx*fx+fz*fz) + relz*(-fz*fy)
            side_norm_sq = max(fx*fx + fz*fz, 1e-12)
            b_d /= side_norm_sq
            c_d /= side_norm_sq
            fov_r = np.asarray(fov_deg, dtype=float).ravel()
            if fov_r.size == 1:
                fov_x = fov_y = float(fov_r[0])
            else:
                fov_x = float(fov_r[0]);  fov_y = float(fov_r[1])
            sx = math.tan(math.radians(fov_x)/2.0);  sy = math.tan(math.radians(fov_y)/2.0)
            eps = 1e-6
            denom = a_d if abs(a_d) >= eps else (eps if a_d >= 0.0 else -eps)
            mhx = float(_mh[0]);  mhy = float(_mh[1])
            ox = max(-3.0*mhx, min(3.0*mhx, (b_d/(sx*denom))*mhx))
            oy = max(-3.0*mhy, min(3.0*mhy, (c_d/(sy*denom))*mhy))
            out1 = np.array([ox, oy])
            if return_status:
                st = Convert.monitor_visibility_status(out1, _mh, margin_mm=margin_mm)
                return out1, st[0]
            return out1
        # ── general batched path ──────────────────────────────────────────────────────
        def _as_rows(arr, width):
            a = np.array(arr, dtype=float)
            if a.ndim == 1:
                if a.shape[0] != width:
                    raise ValueError(f"Expected shape ({width},), got {a.shape}")
                return a[None, :]
            if a.ndim == 2 and a.shape[1] == width:
                return a
            raise ValueError(f"Expected shape ({width},) or (n,{width}), got {a.shape}")

        camera_pos        = _as_rows(_cp, 3)
        camera_azel       = _as_rows(_ae, 2)
        target_pos        = _as_rows(_tp, 3)
        monitor_half_size = _as_rows(_mh, 2)

        n = max(camera_pos.shape[0], camera_azel.shape[0], target_pos.shape[0], monitor_half_size.shape[0])

        def _broadcast_n(arr):
            if arr.shape[0] == n:
                return arr
            if arr.shape[0] == 1:
                return np.repeat(arr, n, axis=0)
            raise ValueError("Batch sizes are incompatible")

        camera_pos = _broadcast_n(camera_pos)
        camera_azel = _broadcast_n(camera_azel)
        target_pos = _broadcast_n(target_pos)
        monitor_half_size = _broadcast_n(monitor_half_size)

        basis = Convert.camera_matrix_vec_fast(camera_azel[:, 0], camera_azel[:, 1])
        front = basis[:, 0, :]
        right = basis[:, 1, :]
        up    = basis[:, 2, :]

        rel = target_pos - camera_pos
        a = np.einsum('ij,ij->i', rel, front)
        b = np.einsum('ij,ij->i', rel, right)
        c = np.einsum('ij,ij->i', rel, up)
        side_norm_sq = np.maximum(np.einsum('ij,ij->i', right, right), 1e-12)
        b = b / side_norm_sq
        c = c / side_norm_sq

        fov_arr = np.array(fov_deg, dtype=float)
        if fov_arr.ndim == 0 or fov_arr.size == 1:
            fov_x = fov_y = float(fov_arr.reshape(-1)[0])
        elif fov_arr.size == 2:
            fov_x = float(fov_arr.reshape(-1)[0])
            fov_y = float(fov_arr.reshape(-1)[1])
        else:
            raise ValueError("fov_deg must be a scalar or length-2 iterable [fov_x, fov_y]")

        scale_x = math.tan(math.radians(fov_x) / 2.0)
        scale_y = math.tan(math.radians(fov_y) / 2.0)
        # Use bounded projection instead of NaN for out-of-front targets.
        # This keeps values finite for RL observations while preserving
        # out-of-monitor status through boundary checks.
        eps = 1e-6
        denom = np.where(np.abs(a) >= eps, a, np.where(a >= 0.0, eps, -eps))
        out = np.empty((n, 2), dtype=float)
        out[:, 0] = (b / (scale_x * denom)) * monitor_half_size[:, 0]
        out[:, 1] = (c / (scale_y * denom)) * monitor_half_size[:, 1]

        max_abs = 3.0 * monitor_half_size
        out = np.clip(out, -max_abs, max_abs)

        if return_status:
            status = Convert.monitor_visibility_status(out, monitor_half_size, margin_mm=margin_mm)
            if n == 1:
                return out[0], status[0]
            return out, status

        return out[0] if n == 1 else out

    @staticmethod
    def game2monitor_fast(
        camera_pos,
        camera_azel,
        target_pos,
        fov_deg,
        monitor_half_size,
        margin_mm=30.0,
        return_status=False,
    ):
        """Optimized version of game2monitor — uses camera_matrix_vec_fast.

        Same interface and output as game2monitor; replace after validation.
        """
        # ── n=1 scalar fast path — avoids _as_rows / _broadcast_n / einsum overhead ──
        _cp = np.asarray(camera_pos,       dtype=float)
        _ae = np.asarray(camera_azel,      dtype=float)
        _tp = np.asarray(target_pos,       dtype=float)
        _mh = np.asarray(monitor_half_size, dtype=float)
        if _cp.shape == (3,) and _ae.shape == (2,) and _tp.shape == (3,) and _mh.shape == (2,):
            a_r = math.radians(_ae[0]);  e_r = math.radians(_ae[1])
            cos_e = math.cos(e_r)
            fx = math.cos(a_r) * cos_e;  fy = math.sin(e_r);  fz = math.sin(a_r) * cos_e
            # right = [-fz, 0, fx],  up = [-fx*fy, fx²+fz², -fz*fy]
            relx = float(_tp[0] - _cp[0]);  rely = float(_tp[1] - _cp[1]);  relz = float(_tp[2] - _cp[2])
            a_d  = relx*fx  + rely*fy  + relz*fz          # dot(rel, front)
            b_d  = relx*(-fz)           + relz*fx          # dot(rel, right)  [ry=0]
            c_d  = relx*(-fx*fy) + rely*(fx*fx+fz*fz) + relz*(-fz*fy)  # dot(rel, up)
            side_norm_sq = max(fx*fx + fz*fz, 1e-12)
            b_d /= side_norm_sq
            c_d /= side_norm_sq
            fov_r = np.asarray(fov_deg, dtype=float).ravel()
            if fov_r.size == 1:
                fov_x = fov_y = float(fov_r[0])
            else:
                fov_x = float(fov_r[0]);  fov_y = float(fov_r[1])
            sx = math.tan(math.radians(fov_x) / 2.0)
            sy = math.tan(math.radians(fov_y) / 2.0)
            eps = 1e-6
            denom = a_d if abs(a_d) >= eps else (eps if a_d >= 0.0 else -eps)
            mhx = float(_mh[0]);  mhy = float(_mh[1])
            ox = max(-3.0*mhx, min(3.0*mhx, (b_d / (sx * denom)) * mhx))
            oy = max(-3.0*mhy, min(3.0*mhy, (c_d / (sy * denom)) * mhy))
            out1 = np.array([ox, oy])
            if return_status:
                st = Convert.monitor_visibility_status(out1, _mh, margin_mm=margin_mm)
                return out1, st[0]
            return out1
        # ── general batched path ──────────────────────────────────────────────────────
        def _as_rows(arr, width):
            a = np.array(arr, dtype=float)
            if a.ndim == 1:
                if a.shape[0] != width:
                    raise ValueError(f"Expected shape ({width},), got {a.shape}")
                return a[None, :]
            if a.ndim == 2 and a.shape[1] == width:
                return a
            raise ValueError(f"Expected shape ({width},) or (n,{width}), got {a.shape}")

        camera_pos        = _as_rows(_cp,  3)
        camera_azel       = _as_rows(_ae,  2)
        target_pos        = _as_rows(_tp,  3)
        monitor_half_size = _as_rows(_mh,  2)

        n = max(camera_pos.shape[0], camera_azel.shape[0],
                target_pos.shape[0], monitor_half_size.shape[0])

        def _broadcast_n(arr):
            if arr.shape[0] == n:  return arr
            if arr.shape[0] == 1:  return np.repeat(arr, n, axis=0)
            raise ValueError("Batch sizes are incompatible")

        camera_pos        = _broadcast_n(camera_pos)
        camera_azel       = _broadcast_n(camera_azel)
        target_pos        = _broadcast_n(target_pos)
        monitor_half_size = _broadcast_n(monitor_half_size)

        basis = Convert.camera_matrix_vec_fast(camera_azel[:, 0], camera_azel[:, 1])
        front = basis[:, 0, :]
        right = basis[:, 1, :]
        up    = basis[:, 2, :]

        rel = target_pos - camera_pos
        a = np.einsum('ij,ij->i', rel, front)
        b = np.einsum('ij,ij->i', rel, right)
        c = np.einsum('ij,ij->i', rel, up)
        side_norm_sq = np.maximum(np.einsum('ij,ij->i', right, right), 1e-12)
        b = b / side_norm_sq
        c = c / side_norm_sq

        fov_arr = np.array(fov_deg, dtype=float)
        if fov_arr.ndim == 0 or fov_arr.size == 1:
            fov_x = fov_y = float(fov_arr.reshape(-1)[0])
        elif fov_arr.size == 2:
            fov_x = float(fov_arr.reshape(-1)[0])
            fov_y = float(fov_arr.reshape(-1)[1])
        else:
            raise ValueError("fov_deg must be a scalar or length-2 iterable [fov_x, fov_y]")

        scale_x = math.tan(math.radians(fov_x) / 2.0)
        scale_y = math.tan(math.radians(fov_y) / 2.0)
        eps = 1e-6
        denom = np.where(np.abs(a) >= eps, a, np.where(a >= 0.0, eps, -eps))
        out = np.empty((n, 2), dtype=float)
        out[:, 0] = (b / (scale_x * denom)) * monitor_half_size[:, 0]
        out[:, 1] = (c / (scale_y * denom)) * monitor_half_size[:, 1]
        out = np.clip(out, -3.0 * monitor_half_size, 3.0 * monitor_half_size)

        if return_status:
            status = Convert.monitor_visibility_status(out, monitor_half_size, margin_mm=margin_mm)
            if n == 1:
                return out[0], status[0]
            return out, status
        return out[0] if n == 1 else out

    @staticmethod
    def monitor_visibility_status(monitor_pos, monitor_half_size, margin_mm=30.0):
        """Return visibility labels against monitor bounds with margin.

        Output labels are `visible` or `out-of-monitor`.
        """
        monitor_pos = np.array(monitor_pos, dtype=float)
        monitor_half_size = np.array(monitor_half_size, dtype=float)

        if monitor_pos.ndim == 1:
            monitor_pos = monitor_pos[None, :]
        if monitor_half_size.ndim == 1:
            monitor_half_size = monitor_half_size[None, :]

        n = max(monitor_pos.shape[0], monitor_half_size.shape[0])
        if monitor_pos.shape[0] == 1 and n > 1:
            monitor_pos = np.repeat(monitor_pos, n, axis=0)
        if monitor_half_size.shape[0] == 1 and n > 1:
            monitor_half_size = np.repeat(monitor_half_size, n, axis=0)

        margin = np.array(margin_mm, dtype=float)
        if margin.ndim == 0:
            margin = np.full((n, 2), float(margin), dtype=float)
        elif margin.ndim == 1:
            if margin.shape[0] == 2:
                margin = np.repeat(margin[None, :], n, axis=0)
            elif margin.shape[0] == n:
                margin = np.repeat(margin[:, None], 2, axis=1)
            else:
                raise ValueError("margin_mm must be scalar, length-2, or length-n")
        elif margin.ndim == 2 and margin.shape == (n, 2):
            pass
        else:
            raise ValueError("margin_mm must be scalar, (2,), (n,), or (n,2)")

        bounds = monitor_half_size + margin
        finite = np.isfinite(monitor_pos).all(axis=1)
        within = (np.abs(monitor_pos[:, 0]) <= bounds[:, 0]) & (np.abs(monitor_pos[:, 1]) <= bounds[:, 1])
        visible = finite & within
        return np.where(visible, "visible", "out-of-monitor")

    @staticmethod
    def game2monitor_from_basis(
        camera_pos,
        camera_front,
        camera_right,
        camera_up,
        target_pos,
        fov_deg,
        monitor_half_size,
        margin_mm=30.0,
        return_status=False,
    ):
        """Project world target position(s) using explicit camera basis vectors.

        This is useful when the runtime camera convention does not match
        `(azimuth, elevation)` conversion helpers.
        """

        def _as_rows(arr, width):
            a = np.array(arr, dtype=float)
            if a.ndim == 1:
                if a.shape[0] != width:
                    raise ValueError(f"Expected shape ({width},), got {a.shape}")
                return a[None, :]
            if a.ndim == 2 and a.shape[1] == width:
                return a
            raise ValueError(f"Expected shape ({width},) or (n,{width}), got {a.shape}")

        camera_pos = _as_rows(camera_pos, 3)
        camera_front = _as_rows(camera_front, 3)
        camera_right = _as_rows(camera_right, 3)
        camera_up = _as_rows(camera_up, 3)
        target_pos = _as_rows(target_pos, 3)
        monitor_half_size = _as_rows(monitor_half_size, 2)

        n = max(
            camera_pos.shape[0],
            camera_front.shape[0],
            camera_right.shape[0],
            camera_up.shape[0],
            target_pos.shape[0],
            monitor_half_size.shape[0],
        )

        def _broadcast_n(arr):
            if arr.shape[0] == n:
                return arr
            if arr.shape[0] == 1:
                return np.repeat(arr, n, axis=0)
            raise ValueError("Batch sizes are incompatible")

        camera_pos = _broadcast_n(camera_pos)
        camera_front = _broadcast_n(camera_front)
        camera_right = _broadcast_n(camera_right)
        camera_up = _broadcast_n(camera_up)
        target_pos = _broadcast_n(target_pos)
        monitor_half_size = _broadcast_n(monitor_half_size)

        camera_front = camera_front / np.linalg.norm(camera_front, axis=1, keepdims=True)
        camera_right = camera_right / np.linalg.norm(camera_right, axis=1, keepdims=True)
        camera_up = camera_up / np.linalg.norm(camera_up, axis=1, keepdims=True)

        rel = target_pos - camera_pos
        a = np.sum(rel * camera_front, axis=1)
        b = np.sum(rel * camera_right, axis=1)
        c = np.sum(rel * camera_up, axis=1)

        fov_arr = np.array(fov_deg, dtype=float)
        if fov_arr.ndim == 0 or fov_arr.size == 1:
            fov_x = fov_y = float(fov_arr.reshape(-1)[0])
        elif fov_arr.size == 2:
            fov_x = float(fov_arr.reshape(-1)[0])
            fov_y = float(fov_arr.reshape(-1)[1])
        else:
            raise ValueError("fov_deg must be a scalar or length-2 iterable [fov_x, fov_y]")

        scale_x = np.tan(np.radians(fov_x) / 2.0)
        scale_y = np.tan(np.radians(fov_y) / 2.0)
        eps = 1e-6
        denom = np.where(np.abs(a) >= eps, a, np.where(a >= 0.0, eps, -eps))
        out = np.empty((n, 2), dtype=float)
        out[:, 0] = (b / (scale_x * denom)) * monitor_half_size[:, 0]
        out[:, 1] = (c / (scale_y * denom)) * monitor_half_size[:, 1]

        max_abs = 3.0 * monitor_half_size
        out = np.clip(out, -max_abs, max_abs)

        if return_status:
            status = Convert.monitor_visibility_status(out, monitor_half_size, margin_mm=margin_mm)
            if n == 1:
                return out[0], status[0]
            return out, status

        return out[0] if n == 1 else out

    @staticmethod
    def monitor2game(camera_pos, camera_azel, monitor_pos, fov_deg, monitor_half_size):
        """Map monitor coordinate(s) to world point(s) on unit-sphere surface (r=1).

        Supports both single sample and batched inputs:
        - single: `camera_pos(3), camera_azel(2), monitor_pos(2)` -> returns `(3,)`
        - batch : `camera_pos(n,3), camera_azel(n,2), monitor_pos(n,2)` -> returns `(n,3)`
        """
        # ── n=1 scalar fast path — avoids _as_rows / _broadcast_n / einsum overhead ──
        _cp = np.asarray(camera_pos,        dtype=float)
        _ae = np.asarray(camera_azel,       dtype=float)
        _mp = np.asarray(monitor_pos,       dtype=float)
        _mh = np.asarray(monitor_half_size, dtype=float)
        if _cp.shape == (3,) and _ae.shape == (2,) and _mp.shape == (2,) and _mh.shape == (2,):
            a_r = math.radians(_ae[0]);  e_r = math.radians(_ae[1])
            cos_e = math.cos(e_r)
            fx = math.cos(a_r)*cos_e;  fy = math.sin(e_r);  fz = math.sin(a_r)*cos_e
            rx = -fz;  rz = fx  # ry=0
            ux = -fx*fy;  uy = fx*fx+fz*fz;  uz = -fz*fy
            fov_r = np.asarray(fov_deg, dtype=float).ravel()
            if fov_r.size == 1:
                fov_x = fov_y = float(fov_r[0])
            else:
                fov_x = float(fov_r[0]);  fov_y = float(fov_r[1])
            sx = math.tan(math.radians(fov_x)/2.0);  sy = math.tan(math.radians(fov_y)/2.0)
            cx_ = (float(_mp[0])/float(_mh[0]))*sx;  cy_ = (float(_mp[1])/float(_mh[1]))*sy
            cpx = float(_cp[0]);  cpy = float(_cp[1]);  cpz = float(_cp[2])
            cos_theta = 1.0 / math.sqrt(1.0 + cx_*cx_ + cy_*cy_)
            return np.array([
                cpx + cos_theta * (fx + cx_*rx + cy_*ux),
                cpy + cos_theta * (fy + cy_*uy),
                cpz + cos_theta * (fz + cx_*rz + cy_*uz),
            ])
        # ── general batched path ──────────────────────────────────────────────────────
        def _as_rows(arr, width):
            a = np.array(arr, dtype=float)
            if a.ndim == 1:
                if a.shape[0] != width:
                    raise ValueError(f"Expected shape ({width},), got {a.shape}")
                return a[None, :]
            if a.ndim == 2 and a.shape[1] == width:
                return a
            raise ValueError(f"Expected shape ({width},) or (n,{width}), got {a.shape}")

        camera_pos        = _as_rows(_cp, 3)
        camera_azel       = _as_rows(_ae, 2)
        monitor_pos       = _as_rows(_mp, 2)
        monitor_half_size = _as_rows(_mh, 2)

        n = max(camera_pos.shape[0], camera_azel.shape[0], monitor_pos.shape[0], monitor_half_size.shape[0])

        def _broadcast_n(arr):
            if arr.shape[0] == n:
                return arr
            if arr.shape[0] == 1:
                return np.repeat(arr, n, axis=0)
            raise ValueError("Batch sizes are incompatible")

        camera_pos        = _broadcast_n(camera_pos)
        camera_azel       = _broadcast_n(camera_azel)
        monitor_pos       = _broadcast_n(monitor_pos)
        monitor_half_size = _broadcast_n(monitor_half_size)

        basis = Convert.camera_matrix_vec_fast(camera_azel[:, 0], camera_azel[:, 1])
        front = basis[:, 0, :]
        right = basis[:, 1, :]
        up    = basis[:, 2, :]

        fov_arr = np.array(fov_deg, dtype=float)
        if fov_arr.ndim == 0 or fov_arr.size == 1:
            fov_x = fov_y = float(fov_arr.reshape(-1)[0])
        elif fov_arr.size == 2:
            fov_x = float(fov_arr.reshape(-1)[0])
            fov_y = float(fov_arr.reshape(-1)[1])
        else:
            raise ValueError("fov_deg must be a scalar or length-2 iterable [fov_x, fov_y]")

        scale_x = math.tan(math.radians(fov_x) / 2.0)
        scale_y = math.tan(math.radians(fov_y) / 2.0)
        coeff = np.empty_like(monitor_pos, dtype=float)
        coeff[:, 0] = (monitor_pos[:, 0] / monitor_half_size[:, 0]) * scale_x
        coeff[:, 1] = (monitor_pos[:, 1] / monitor_half_size[:, 1]) * scale_y
        cos_theta = 1.0 / np.sqrt(1.0 + np.sum(coeff * coeff, axis=1))
        ray = front + coeff[:, 0:1] * right + coeff[:, 1:2] * up
        out = camera_pos + cos_theta[:, None] * ray

        return out[0] if n == 1 else out

    @staticmethod
    def monitor2game_fast(camera_pos, camera_azel, monitor_pos, fov_deg, monitor_half_size):
        """Optimized version of monitor2game — uses camera_matrix_vec_fast.

        Same interface and output as monitor2game; replace after validation.
        """
        # ── n=1 scalar fast path — avoids _as_rows / _broadcast_n / einsum overhead ──
        _cp = np.asarray(camera_pos,       dtype=float)
        _ae = np.asarray(camera_azel,      dtype=float)
        _mp = np.asarray(monitor_pos,      dtype=float)
        _mh = np.asarray(monitor_half_size, dtype=float)
        if _cp.shape == (3,) and _ae.shape == (2,) and _mp.shape == (2,) and _mh.shape == (2,):
            a_r = math.radians(_ae[0]);  e_r = math.radians(_ae[1])
            cos_e = math.cos(e_r)
            fx = math.cos(a_r)*cos_e;  fy = math.sin(e_r);  fz = math.sin(a_r)*cos_e
            # right=[-fz,0,fx], up=[-fx*fy, fx²+fz², -fz*fy]
            rx = -fz;  rz = fx  # ry = 0
            ux = -fx*fy;  uy = fx*fx + fz*fz;  uz = -fz*fy
            fov_r = np.asarray(fov_deg, dtype=float).ravel()
            if fov_r.size == 1:
                fov_x = fov_y = float(fov_r[0])
            else:
                fov_x = float(fov_r[0]);  fov_y = float(fov_r[1])
            sx = math.tan(math.radians(fov_x) / 2.0)
            sy = math.tan(math.radians(fov_y) / 2.0)
            cx_ = (float(_mp[0]) / float(_mh[0])) * sx
            cy_ = (float(_mp[1]) / float(_mh[1])) * sy
            cpx = float(_cp[0]);  cpy = float(_cp[1]);  cpz = float(_cp[2])
            cos_theta = 1.0 / math.sqrt(1.0 + cx_*cx_ + cy_*cy_)
            return np.array([
                cpx + cos_theta * (fx + cx_*rx + cy_*ux),
                cpy + cos_theta * (fy + cy_*uy),
                cpz + cos_theta * (fz + cx_*rz + cy_*uz),
            ])
        # ── general batched path ──────────────────────────────────────────────────────
        def _as_rows(arr, width):
            a = np.array(arr, dtype=float)
            if a.ndim == 1:
                if a.shape[0] != width:
                    raise ValueError(f"Expected shape ({width},), got {a.shape}")
                return a[None, :]
            if a.ndim == 2 and a.shape[1] == width:
                return a
            raise ValueError(f"Expected shape ({width},) or (n,{width}), got {a.shape}")

        camera_pos        = _as_rows(_cp,  3)
        camera_azel       = _as_rows(_ae,  2)
        monitor_pos       = _as_rows(_mp,  2)
        monitor_half_size = _as_rows(_mh,  2)

        n = max(camera_pos.shape[0], camera_azel.shape[0],
                monitor_pos.shape[0], monitor_half_size.shape[0])

        def _broadcast_n(arr):
            if arr.shape[0] == n:  return arr
            if arr.shape[0] == 1:  return np.repeat(arr, n, axis=0)
            raise ValueError("Batch sizes are incompatible")

        camera_pos        = _broadcast_n(camera_pos)
        camera_azel       = _broadcast_n(camera_azel)
        monitor_pos       = _broadcast_n(monitor_pos)
        monitor_half_size = _broadcast_n(monitor_half_size)

        basis = Convert.camera_matrix_vec_fast(camera_azel[:, 0], camera_azel[:, 1])
        front = basis[:, 0, :]
        right = basis[:, 1, :]
        up    = basis[:, 2, :]

        fov_arr = np.array(fov_deg, dtype=float)
        if fov_arr.ndim == 0 or fov_arr.size == 1:
            fov_x = fov_y = float(fov_arr.reshape(-1)[0])
        elif fov_arr.size == 2:
            fov_x = float(fov_arr.reshape(-1)[0])
            fov_y = float(fov_arr.reshape(-1)[1])
        else:
            raise ValueError("fov_deg must be a scalar or length-2 iterable [fov_x, fov_y]")

        scale_x = math.tan(math.radians(fov_x) / 2.0)
        scale_y = math.tan(math.radians(fov_y) / 2.0)
        coeff = np.empty_like(monitor_pos, dtype=float)
        coeff[:, 0] = (monitor_pos[:, 0] / monitor_half_size[:, 0]) * scale_x
        coeff[:, 1] = (monitor_pos[:, 1] / monitor_half_size[:, 1]) * scale_y
        cos_theta = 1.0 / np.sqrt(1.0 + np.sum(coeff * coeff, axis=1))
        ray = front + coeff[:, 0:1] * right + coeff[:, 1:2] * up
        out = camera_pos + cos_theta[:, None] * ray

        return out[0] if n == 1 else out





def point_about_ray(point, angle, ray_direction, ray_origin=(0.0, 0.0, 0.0), angle_is_degree: bool = True):
    """Rotate a 3D point about an arbitrary ray using Rodrigues' formula."""
    return point_about_ray_fast(point, angle, ray_direction, ray_origin=ray_origin, angle_is_degree=angle_is_degree)


def point_about_ray_orig(point, angle, ray_direction, ray_origin=(0.0, 0.0, 0.0), angle_is_degree: bool = True):
    """Original implementation kept for reference / regression testing."""
    p = np.array(point, dtype=float)
    o = np.array(ray_origin, dtype=float)
    k = np.array(ray_direction, dtype=float)
    norm = np.linalg.norm(k)
    if norm == 0:
        raise ValueError("ray_direction must be non-zero")
    k = k / norm
    t = np.radians(angle) if angle_is_degree else angle
    v = p - o
    v_rot = v * np.cos(t) + np.cross(k, v) * np.sin(t) + k * np.dot(k, v) * (1.0 - np.cos(t))
    return o + v_rot


def point_about_ray_fast(point, angle, ray_direction, ray_origin=(0.0, 0.0, 0.0), angle_is_degree: bool = True):
    """Rodrigues' rotation — closed-form, avoids np.cross / np.dot / np.linalg.norm."""
    p = np.asarray(point,      dtype=float)
    o = np.asarray(ray_origin, dtype=float)
    k = np.asarray(ray_direction, dtype=float)
    k0 = float(k[0]);  k1 = float(k[1]);  k2 = float(k[2])
    norm = math.sqrt(k0*k0 + k1*k1 + k2*k2)
    if norm == 0:
        raise ValueError("ray_direction must be non-zero")
    k0 /= norm;  k1 /= norm;  k2 /= norm
    t = math.radians(float(angle)) if angle_is_degree else float(angle)
    cos_t = math.cos(t);  sin_t = math.sin(t)
    v0 = float(p[0] - o[0]);  v1 = float(p[1] - o[1]);  v2 = float(p[2] - o[2])
    # cross(k, v) = [k1*v2-k2*v1, k2*v0-k0*v2, k0*v1-k1*v0]
    cx = k1*v2 - k2*v1
    cy = k2*v0 - k0*v2
    cz = k0*v1 - k1*v0
    kv = k0*v0 + k1*v1 + k2*v2          # dot(k, v)
    fac = kv * (1.0 - cos_t)
    return np.array([
        float(o[0]) + v0*cos_t + cx*sin_t + k0*fac,
        float(o[1]) + v1*cos_t + cy*sin_t + k1*fac,
        float(o[2]) + v2*cos_t + cz*sin_t + k2*fac,
    ])



def normalize_vector(vector, tol=1e-6):
    """Return a unit-norm copy of a vector. Near-zero vectors are returned unchanged."""
    v = np.array(vector, dtype=float)
    norm = np.linalg.norm(v)
    if norm <= tol:
        return v
    return v / norm


def angle_between_vectors(v1, v2, return_in_degree=True, tol=1e-12):
    """Return the unsigned angle between two vectors."""
    a = np.array(v1, dtype=float)
    b = np.array(v2, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= tol or nb <= tol:
        raise ValueError("Input vectors must be non-zero")
    cos_theta = np.dot(a, b) / (na * nb)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(theta) if return_in_degree else theta


def sample_perpendicular_ray(ray_direction, angle=None, angle_is_degree=True, return_spherical=False):
    """Sample a unit ray perpendicular to `ray_direction`.

    If `angle` is None, a random angle is used to choose a direction on the
    perpendicular circle around the input ray.
    """
    k = normalize_vector(ray_direction)

    # Pick a helper axis not parallel to k.
    if abs(k[1]) < 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    else:
        ref = np.array([1.0, 0.0, 0.0])

    u = normalize_vector(np.cross(k, ref))
    w = normalize_vector(np.cross(k, u))

    if angle is None:
        theta = np.random.uniform(0.0, 2.0 * np.pi)
    else:
        theta = np.radians(angle) if angle_is_degree else float(angle)

    perp = np.cos(theta) * u + np.sin(theta) * w
    perp = normalize_vector(perp)

    if return_spherical:
        return np.array(Convert.cart2sphr_scalar(perp[0], perp[1], perp[2], return_in_degree=True), dtype=float)
    return perp


def orbit_normal_from_screen_angle(camera_az, camera_el, screen_angle, angle_is_degree=True, return_spherical=False):
    """Build a unit orbit-normal whose on-screen tangent matches `screen_angle`.

    - `screen_angle = 0` means +x (right) on screen.
    - Positive angle rotates counter-clockwise on screen.
    - Uses camera forward/right/up basis from `(camera_az, camera_el)`.
    """
    basis = Convert.camera_matrix_scalar(camera_az, camera_el, angle_is_degree=angle_is_degree)
    front, right, up = basis[0], basis[1], basis[2]
    front = normalize_vector(front)
    right = normalize_vector(right)
    up = normalize_vector(up)

    theta = math.radians(screen_angle) if angle_is_degree else float(screen_angle)
    tangent = math.cos(theta) * right + math.sin(theta) * up
    tangent = normalize_vector(tangent)

    # Ensure tangent direction for positive rotation at front point:
    # n x front = tangent  ->  n = front x tangent
    normal = normalize_vector(np.cross(front, tangent))

    if return_spherical:
        return np.array(Convert.cart2sphr_scalar(normal[0], normal[1], normal[2], return_in_degree=True), dtype=float)
    return normal


def sample_uniform_with_boundaries(
    low,
    high,
    p_low_boundary=0.0,
    p_high_boundary=0.0,
    rng=None,
    random_negation=False,
):
    assert 0.0 <= p_low_boundary < 1.0, "p_low_boundary must be in [0, 1)"
    assert 0.0 <= p_high_boundary < 1.0, "p_high_boundary must be in [0, 1)"
    assert p_low_boundary + p_high_boundary < 1.0, "The sum of p_low_boundary and p_high_boundary must be less than 1"
    rng = np.random.default_rng() if rng is None else rng

    def _apply_random_negation(x):
        if not random_negation:
            return x
        arr = np.array(x)
        if arr.ndim == 0:
            sign = -1 if rng.random() < 0.5 else 1
            return arr.item() * sign
        signs = np.where(rng.random(size=arr.shape) < 0.5, -1, 1)
        return arr * signs

    u = rng.random()
    if u < p_low_boundary:
        return _apply_random_negation(low)
    if u > 1.0 - p_high_boundary:
        return _apply_random_negation(high)
    return _apply_random_negation(rng.uniform(low, high))


def sampler_func(low, high, p_low_boundary=0.0, p_high_boundary=0.0, random_negation=False):
    def sample(rng=None):
        return sample_uniform_with_boundaries(
            low,
            high,
            p_low_boundary,
            p_high_boundary,
            rng,
            random_negation=random_negation,
        )
    return sample


def linear_normalize(w, v_min, v_max, dtype=np.float32):
    """Linearly map values from [v_min, v_max] to [-1, 1] with clipping."""
    w = np.array(w, dtype=float)
    v_min = np.array(v_min, dtype=float)
    v_max = np.array(v_max, dtype=float)
    span = v_max - v_min
    if np.any(span == 0):
        raise ValueError("v_max and v_min must be different")
    z = (2.0 * w - (v_min + v_max)) / span
    return np.array(np.clip(z, -1.0, 1.0), dtype=dtype)


def linear_denormalize(z, v_min, v_max, dtype=np.float32):
    """Linearly map values from [-1, 1] back to [v_min, v_max] with clipping."""
    z = np.array(z, dtype=float)
    v_min = np.array(v_min, dtype=float)
    v_max = np.array(v_max, dtype=float)
    span = v_max - v_min
    if np.any(span == 0):
        raise ValueError("v_max and v_min must be different")
    w = (z * span + (v_min + v_max)) / 2.0
    return np.array(np.clip(w, v_min, v_max), dtype=dtype)


def log_normalize(w, v_min, v_max, scale=1, dtype=np.float32):
    """Log-shaped normalize from [v_min, v_max] to [-1, 1].

    `scale=0` falls back to linear normalization.
    """
    w = np.array(w, dtype=float)
    v_min = np.array(v_min, dtype=float)
    v_max = np.array(v_max, dtype=float)
    span = v_max - v_min
    if np.any(span == 0):
        raise ValueError("v_max and v_min must be different")
    scale = np.array(scale, dtype=float)
    if np.any(scale < 0):
        raise ValueError("scale must be >= 0")

    new_z = (2.0 * w - (v_max + v_min)) / span
    new_z = np.clip(new_z, -1.0, 1.0)

    if np.all(scale == 0):
        return np.array(new_z, dtype=dtype)

    a = np.e * scale
    a_minus_1 = np.where(scale == 0, 1.0, a - 1.0)
    denom = np.where(scale == 0, 1.0, np.log(a))
    out_log = 2.0 * (np.log(1.0 + a_minus_1 * ((new_z + 1.0) / 2.0)) / denom) - 1.0
    out = np.where(scale == 0, new_z, out_log)
    return np.array(np.clip(out, -1.0, 1.0), dtype=dtype)


def log_denormalize(z, v_min, v_max, scale=1, dtype=np.float32):
    """Inverse of `log_normalize`, mapping [-1, 1] to [v_min, v_max]."""
    z = np.clip(np.array(z, dtype=float), -1.0, 1.0)
    v_min = np.array(v_min, dtype=float)
    v_max = np.array(v_max, dtype=float)
    span = v_max - v_min
    if np.any(span == 0):
        raise ValueError("v_max and v_min must be different")
    scale = np.array(scale, dtype=float)
    if np.any(scale < 0):
        raise ValueError("scale must be >= 0")

    if np.all(scale == 0):
        new_z = z
    else:
        a = np.e * scale
        a_minus_1 = np.where(scale == 0, 1.0, a - 1.0)
        new_z_log = ((a ** ((z + 1.0) / 2.0) - 1.0) / a_minus_1) * 2.0 - 1.0
        new_z = np.where(scale == 0, z, new_z_log)

    w = (new_z * span + v_min + v_max) / 2.0
    return np.array(np.clip(w, v_min, v_max), dtype=dtype)


def np_interp_nd(x, xp, fp, kind="linear", fill_value="edge", assume_sorted=True):
    """N-D interpolation along axis 0 using scipy.interpolate.interp1d.

    Parameters
    ----------
    x : array-like or scalar
        Query points.
    xp : (n,) array-like
        Sample points.
    fp : (n, d...) array-like
        Sample values. Interpolation is performed along axis 0.
    kind : str, default="linear"
        Interpolation kind for interp1d.
    fill_value : "edge" | "extrapolate" | scalar | tuple, default="edge"
        - "edge": hold endpoint values outside range.
        - "extrapolate": linear extrapolation.
        - scalar/tuple: forwarded to interp1d.
    assume_sorted : bool, default=True
        Forwarded to interp1d.
    """
    xp = np.array(xp, dtype=float)
    fp = np.array(fp, dtype=float)
    if xp.ndim != 1:
        raise ValueError("xp must be 1D")
    if fp.ndim == 1:
        fp = fp[:, None]
    if fp.shape[0] != xp.shape[0]:
        raise ValueError("fp.shape[0] must match xp.shape[0]")

    if fill_value == "edge":
        fv = (fp[0], fp[-1])
    else:
        fv = fill_value

    f = interp1d(
        xp,
        fp,
        axis=0,
        kind=kind,
        bounds_error=False,
        fill_value=fv,
        assume_sorted=assume_sorted,
    )
    out = f(x)
    return out.squeeze(-1) if out.shape[-1] == 1 else out


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


if __name__ == "__main__":
    print(monitor_mm_to_view_angle_deg(5.5, 531.3, 103))
    print(monitor_mm_to_view_angle_deg(4.0, 531.3, 103))
    print(monitor_mm_to_view_angle_deg(15.0, 531.3, 103))
