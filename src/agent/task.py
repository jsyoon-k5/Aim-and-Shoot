"""
FPS Aim-and-Shoot Task virtual environment

This code was written by June-Seop Yoon
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from copy import deepcopy

from ..utils.mymath import (
    Convert, 
    point_about_ray,
    sample_perpendicular_ray,
    sampler_func,
    angle_between_vectors,
    monitor_mm_to_view_angle_deg,
    view_angle_deg_to_monitor_mm,
)
# from ..utils.myutils import pickle_load, pickle_save
from ..configs.constants import MAXIMUM_ELEV_ANGLE

DELTA_TIME = 1e-3
CAMERA_ELEVATION_LIMIT_DEG = (-MAXIMUM_ELEV_ANGLE, MAXIMUM_ELEV_ANGLE)
DEFAULT_TASK_CONFIG = dict(
    mouse_gain_deg_per_mm = 1.0,

    monitor_width_mm = 531.3,
    monitor_height_mm = 298.8,
    monitor_width_px = 1920,
    monitor_height_px = 1080,

    cam_fov_deg_width = 103,
    cam_fov_deg_height = 70.533,
    cam_azel_max_dev_deg = float(monitor_mm_to_view_angle_deg(4.5, 531.3, 103)),    # 4.5 mm at screen center

    target_radius_deg_range = dict(
        min=float(monitor_mm_to_view_angle_deg(4.0, 531.3, 103)),
        max=float(monitor_mm_to_view_angle_deg(13.0, 531.3, 103)),
        boundary_sample=(0.0, 0.0),
    ), # 4 mm to 13 mm at screen center
    target_aspeed_deg_s_range = dict(min=0.0, max=40.0, boundary_sample=(0.05, 0.0)),
    target_pos_azim_range = dict(min=2.0, max=37.0),
    target_pos_elev_range = dict(min=1.0, max=21.0),
    # ID-uniform sampling (optional; only used when target_pos_sampling_mode="id_uniform")
    target_pos_sampling_mode = "azim_elev_uniform",  # "azim_elev_uniform" | "id_uniform"
    target_id_range = dict(min=1.5, max=4.5),        # Fitts' ID range [bits] for id_uniform mode
)

class AimandShootSpiderShotTask:
    def __init__(self, config: dict = None):
        self.config = DEFAULT_TASK_CONFIG if config is None else config

        self.crosshair_pos_mm = np.zeros(2)

        self.monitor_half_size_mm = np.array([
            self.config["monitor_width_mm"], 
            self.config["monitor_height_mm"]
        ], dtype=float) / 2

        self.camera_pos_world = np.zeros(3)    # X, Y, Z
        self.camera_azel_deg = np.zeros(2)   # Azim, Elev
        self.camera_fov_deg = np.array([self.config["cam_fov_deg_width"], self.config["cam_fov_deg_height"]])   # deg

        self.sample_camera_init_azel_deg = sampler_func(
            low=0.0, high=self.config["cam_azel_max_dev_deg"], 
            p_low_boundary=0.0, p_high_boundary=0.0, random_negation=False
        )

        self.hand_pos_mm = np.zeros(2)    # X, Y
        self.hand_vel_mm_per_s = np.zeros(2)    # X, Y
        self.hand_sensi_deg_per_mm = self.config["mouse_gain_deg_per_mm"]

        self.target_radius_deg = 0.0
        self.target_speed_deg_s = 0.0
        self.target_orbit_axis_deg = np.zeros(2)   # Azim, Elev
        self.target_pos_world = np.zeros(3)   # X, Y, Z
        self.target_pos_monitor_mm = np.zeros(2)   # X, Y

        self.sample_target_pos_world = sampler_func(
            low=np.array([self.config["target_pos_azim_range"]["min"], self.config["target_pos_elev_range"]["min"]]),
            high=np.array([self.config["target_pos_azim_range"]["max"], self.config["target_pos_elev_range"]["max"]]),
            p_low_boundary=0.0, p_high_boundary=0.0, random_negation=True
        )
        self._sample_target_radius_in_mm = "target_radius_mm_range" in self.config
        if self._sample_target_radius_in_mm:
            radius_cfg = self.config["target_radius_mm_range"]
            self.sample_target_rad_mm = sampler_func(
                low=radius_cfg["min"],
                high=radius_cfg["max"],
                p_low_boundary=radius_cfg["boundary_sample"][0],
                p_high_boundary=radius_cfg["boundary_sample"][1],
            )
            self.sample_target_rad_deg = None
        else:
            self.sample_target_rad_mm = None
            self.sample_target_rad_deg = sampler_func(
                low=self.config["target_radius_deg_range"]["min"],
                high=self.config["target_radius_deg_range"]["max"],
                p_low_boundary=self.config["target_radius_deg_range"]["boundary_sample"][0],
                p_high_boundary=self.config["target_radius_deg_range"]["boundary_sample"][1],
            )
        self.sample_target_speed_deg_s = sampler_func(
            low=self.config["target_aspeed_deg_s_range"]["min"],
            high=self.config["target_aspeed_deg_s_range"]["max"],
            p_low_boundary=self.config["target_aspeed_deg_s_range"]["boundary_sample"][0],
            p_high_boundary=self.config["target_aspeed_deg_s_range"]["boundary_sample"][1]
        )
        self.sample_target_motion_dir_deg = sampler_func(
            low=self.config["target_motion_dir_deg_range"]["min"],
            high=self.config["target_motion_dir_deg_range"]["max"],
            p_low_boundary=self.config["target_motion_dir_deg_range"]["boundary_sample"][0],
            p_high_boundary=self.config["target_motion_dir_deg_range"]["boundary_sample"][1]
        )
        # ID-uniform position sampler (only constructed when mode requires it)
        self._pos_sampling_mode = self.config.get("target_pos_sampling_mode", "azim_elev_uniform")
        if self._pos_sampling_mode == "id_uniform":
            id_cfg = self.config.get("target_id_range", dict(min=1.5, max=4.5))
            self.sample_target_id = sampler_func(
                low=id_cfg["min"], high=id_cfg["max"],
                p_low_boundary=0.0, p_high_boundary=0.0,
            )
            # Precompute valid bounding box in mm (used in _sample_pos_by_id)
            self._pos_box_mm = self._compute_pos_box_mm()
        else:
            self.sample_target_id = None
            self._pos_box_mm = None

    def _compute_pos_box_mm(self):
        """Convert azim/elev degree ranges to monitor mm bounding box.

        Returns a dict with keys:
          ``x_min``, ``x_max``, ``y_min``, ``y_max``  – box corners in mm
          ``d_box_min``, ``d_box_max``                 – min/max reachable D
          ``r_min_mm``, ``r_max_mm``                   – radius limits in mm
        """
        cfg = self.config
        x_min = float(view_angle_deg_to_monitor_mm(
            cfg["target_pos_azim_range"]["min"], cfg["monitor_width_mm"], cfg["cam_fov_deg_width"]))
        x_max = float(view_angle_deg_to_monitor_mm(
            cfg["target_pos_azim_range"]["max"], cfg["monitor_width_mm"], cfg["cam_fov_deg_width"]))
        y_min = float(view_angle_deg_to_monitor_mm(
            cfg["target_pos_elev_range"]["min"], cfg["monitor_height_mm"], cfg["cam_fov_deg_height"]))
        y_max = float(view_angle_deg_to_monitor_mm(
            cfg["target_pos_elev_range"]["max"], cfg["monitor_height_mm"], cfg["cam_fov_deg_height"]))
        if "target_radius_mm_range" in cfg:
            r_min_mm = float(cfg["target_radius_mm_range"]["min"])
            r_max_mm = float(cfg["target_radius_mm_range"]["max"])
        else:
            r_min_mm = float(view_angle_deg_to_monitor_mm(
                cfg["target_radius_deg_range"]["min"], cfg["monitor_width_mm"], cfg["cam_fov_deg_width"]))
            r_max_mm = float(view_angle_deg_to_monitor_mm(
                cfg["target_radius_deg_range"]["max"], cfg["monitor_width_mm"], cfg["cam_fov_deg_width"]))
        return dict(
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
            # A point on the circle of radius D can lie in the box iff
            # sqrt(x_min^2+y_min^2) <= D <= sqrt(x_max^2+y_max^2).
            d_box_min=float(np.sqrt(x_min**2 + y_min**2)),
            d_box_max=float(np.sqrt(x_max**2 + y_max**2)),
            r_min_mm=r_min_mm,
            r_max_mm=r_max_mm,
        )

    def _valid_phi_arc_in_box(self, d_mm):
        """Return the valid phi arc (phi_lo, phi_hi) in the first quadrant [0, pi/2]
        such that (D cos phi, D sin phi) falls inside the bounding box.

        Derivation
        ----------
        In the first quadrant:
            cos phi in [x_min/D, x_max/D]  →  phi in [acos(x_max/D), acos(x_min/D)]
            sin phi in [y_min/D, y_max/D]  →  phi in [asin(y_min/D), asin(y_max/D)]
        The valid arc is the intersection of these two intervals.

        By symmetry across all four quadrants, sampling phi uniformly from
        this arc and then randomly flipping signs of x and y gives a
        uniform distribution over all valid monitor positions at distance D.

        Returns None if no valid phi exists (D too small or too large for box).
        """
        box = self._pos_box_mm
        x_min, x_max = box["x_min"], box["x_max"]
        y_min, y_max = box["y_min"], box["y_max"]

        if d_mm <= 0.0:
            return None

        # x constraint: phi in [acos(x_max/D), acos(x_min/D)]
        # acos is defined on [-1, 1]; clamp arguments accordingly.
        phi_lo_x = np.arccos(min(x_max / d_mm, 1.0)) if x_max / d_mm >= -1.0 else 0.0
        phi_hi_x = np.arccos(max(x_min / d_mm, 0.0))   # x_min > 0, so x_min/D <= 1

        # y constraint: phi in [asin(y_min/D), asin(y_max/D)]
        phi_lo_y = np.arcsin(min(y_min / d_mm, 1.0)) if y_min / d_mm <= 1.0 else np.pi / 2
        phi_hi_y = np.arcsin(min(y_max / d_mm, 1.0)) if y_max / d_mm <= 1.0 else np.pi / 2

        phi_lo = max(phi_lo_x, phi_lo_y)
        phi_hi = min(phi_hi_x, phi_hi_y)

        if phi_lo > phi_hi + 1e-9:
            return None  # no valid direction at this distance
        return float(phi_lo), float(phi_hi)

    def _sample_pos_by_id(self, rng):
        """Jointly sample (radius, position) so that Fitts' ID is truly uniform.

        Algorithm (O(1), rejection-free)
        ---------------------------------
        1. Sample ``ID ~ U(id_min, id_max)``.
        2. From ID compute the valid radius range:
               D = (2^ID - 1) * 2 * r   and   D must satisfy d_box_min <= D <= d_box_max
               => r ∈ [ d_box_min / (2k),  d_box_max / (2k) ]  where k = 2^ID - 1
           Intersect with [r_min_mm, r_max_mm].  If the intersection is empty,
           re-sample ID (this is rare and only occurs when id_max is set beyond
           what the geometry allows).
        3. Sample r uniformly from the valid intersection.
        4. Compute D = (2^ID - 1) * 2 * r.
        5. Compute the valid phi arc analytically (see ``_valid_phi_arc_in_box``).
        6. Sample phi ~ U(phi_lo, phi_hi); randomly flip signs of x and y.

        Returns
        -------
        (target_pos_monitor_mm, target_pos_world, r_deg)
            r_deg is the sampled radius in degrees (caller must store it).

        Design rationale
        ----------------
        Fixing r first and then rejection-sampling ID violates ID uniformity
        because large r forbids high-ID values (D would exceed the bounding box).
        By conditioning r on ID instead, every sampled ID value is equally
        likely and the phi arc is always non-empty — O(1) with no rejection.
        """
        box = self._pos_box_mm
        d_box_min = box["d_box_min"]
        d_box_max = box["d_box_max"]
        r_min_mm  = box["r_min_mm"]
        r_max_mm  = box["r_max_mm"]

        max_id_tries = 20  # only needed if id_range is set beyond geometric limits
        for _ in range(max_id_tries):
            id_val = float(self.sample_target_id(rng=rng))
            k = 2.0 ** id_val - 1.0   # D = 2*k*r
            if k <= 0.0:
                continue

            # Valid r range so that d_box_min <= D <= d_box_max
            r_lo = max(r_min_mm, d_box_min / (2.0 * k))
            r_hi = min(r_max_mm, d_box_max / (2.0 * k))
            if r_lo > r_hi:
                continue  # ID geometrically unreachable; try again

            r_mm = rng.uniform(r_lo, r_hi)
            d_mm = 2.0 * k * r_mm

            arc = self._valid_phi_arc_in_box(d_mm)
            if arc is None:  # should not happen given r_lo/r_hi derivation
                continue

            phi_lo, phi_hi = arc
            phi = rng.uniform(phi_lo, phi_hi)
            x_mm = d_mm * np.cos(phi) * rng.choice([-1.0, 1.0])
            y_mm = d_mm * np.sin(phi) * rng.choice([-1.0, 1.0])

            # Convert r_mm back to degrees (at screen centre, horizontal FOV)
            from ..utils.mymath import monitor_mm_to_view_angle_deg
            r_deg = float(monitor_mm_to_view_angle_deg(
                r_mm, self.config["monitor_width_mm"], self.config["cam_fov_deg_width"]
            ))

            target_pos_monitor_mm = np.array([x_mm, y_mm], dtype=float)
            target_pos_world = Convert.monitor2game(
                self.camera_pos_world, self.camera_azel_deg,
                target_pos_monitor_mm, self.camera_fov_deg, self.monitor_half_size_mm,
            )
            return target_pos_monitor_mm, target_pos_world, r_deg

        # Fallback: uniform azim/elev with original radius sampler
        # (only triggered if id_range is badly misconfigured)
        r_deg = self._sample_target_radius_deg(rng)
        target_azel = np.array(self.sample_target_pos_world(rng=rng), dtype=float)
        target_pos_world = Convert.sphr2cart_scalar(
            az=target_azel[0], el=target_azel[1], angle_is_degree=True,
        )
        target_pos_monitor_mm = Convert.game2monitor(
            self.camera_pos_world, self.camera_azel_deg, target_pos_world,
            self.camera_fov_deg, self.monitor_half_size_mm,
        )
        return target_pos_monitor_mm, target_pos_world, r_deg

    def _clamp_camera_azel_deg(self, camera_azel_deg):
        camera_azel_deg = np.array(camera_azel_deg, dtype=float, copy=True)
        camera_azel_deg[..., 1] = np.clip(
            camera_azel_deg[..., 1],
            CAMERA_ELEVATION_LIMIT_DEG[0],
            CAMERA_ELEVATION_LIMIT_DEG[1],
        )
        return camera_azel_deg


    def _sample_target_radius_deg(self, rng):
        if self._sample_target_radius_in_mm:
            radius_mm = float(self.sample_target_rad_mm(rng=rng))
            return float(monitor_mm_to_view_angle_deg(
                radius_mm,
                self.config["monitor_width_mm"],
                self.config["cam_fov_deg_width"],
            ))
        return float(self.sample_target_rad_deg(rng=rng))


    def _sample_target_motion_dir_with_symmetry(self, rng):
        """
        Sample monitor motion direction from base range (typically [0, 90])
        and randomly apply axis symmetries to cover the full [0, 360) domain.

        Convention: +x = 0 deg, positive is counterclockwise.
        """
        theta = float(self.sample_target_motion_dir_deg(rng=rng))

        # Random reflection across y-axis: theta -> 180 - theta
        if rng.integers(0, 2) == 1:
            theta = 180.0 - theta

        # Random reflection across x-axis: theta -> -theta
        if rng.integers(0, 2) == 1:
            theta = -theta

        return float(np.mod(theta, 360.0))

    

    def reset(
        self,
        camera_azel_deg=None,
        target_pos_monitor_mm=None,
        target_pos_world=None,
        target_orbit_axis_deg=None,
        target_motion_dir_deg=None,
        target_speed_deg_s=None,
        target_radius_deg=None,
        rng=None,
        **kwargs
    ):
        rng = np.random.default_rng() if rng is None else rng
        target_motion_dir_deg = None if target_motion_dir_deg is None else float(target_motion_dir_deg)

        # Camera reset.
        self.camera_pos_world = np.zeros(3, dtype=float)
        if camera_azel_deg is None:
            mag = float(self.sample_camera_init_azel_deg(rng=rng))
            ang = rng.uniform(-np.pi, np.pi)
            # 1) Start from +az direction on xz plane: (az, el) = (mag, 0)
            # 2) Rotate that direction around reference axis (az, el)=(0,0) == +X by `ang`
            v0 = Convert.sphr2cart_scalar(az=mag, el=0.0, angle_is_degree=True)
            ca, sa = np.cos(ang), np.sin(ang)
            rot_x = np.array([
                [1.0, 0.0, 0.0],
                [0.0, ca, -sa],
                [0.0, sa, ca],
            ], dtype=float)
            v = rot_x @ v0
            self.camera_azel_deg = self._clamp_camera_azel_deg(
                Convert.cart2sphr_scalar(v[0], v[1], v[2], return_in_degree=True)
            )
        else:
            self.camera_azel_deg = self._clamp_camera_azel_deg(camera_azel_deg)

        # Target position + radius reset.
        # In id_uniform mode, ID is sampled first and radius is derived from
        # the valid range for that ID (joint sampling).  This guarantees truly
        # uniform ID without rejection; see _sample_pos_by_id for details.
        if target_pos_monitor_mm is not None:
            self.target_pos_monitor_mm = np.array(target_pos_monitor_mm, dtype=float)
            self.target_pos_world = Convert.monitor2game(
                self.camera_pos_world, self.camera_azel_deg,
                self.target_pos_monitor_mm, self.camera_fov_deg, self.monitor_half_size_mm,
            )
            self.target_radius_deg = (
                float(target_radius_deg) if target_radius_deg is not None
                else self._sample_target_radius_deg(rng)
            )
        elif target_pos_world is not None:
            self.target_pos_world = np.array(target_pos_world, dtype=float)
            self.target_pos_monitor_mm = Convert.game2monitor(
                self.camera_pos_world, self.camera_azel_deg,
                self.target_pos_world, self.camera_fov_deg, self.monitor_half_size_mm,
            )
            self.target_radius_deg = (
                float(target_radius_deg) if target_radius_deg is not None
                else self._sample_target_radius_deg(rng)
            )
        elif self._pos_sampling_mode == "id_uniform" and target_radius_deg is None:
            # Joint sample: ID → valid r range → r; then derive D and phi.
            self.target_pos_monitor_mm, self.target_pos_world, self.target_radius_deg = \
                self._sample_pos_by_id(rng)
        else:
            # azim_elev_uniform mode (or explicit radius given).
            self.target_radius_deg = (
                float(target_radius_deg) if target_radius_deg is not None
                else self._sample_target_radius_deg(rng)
            )
            target_azel = np.array(self.sample_target_pos_world(rng=rng), dtype=float)
            self.target_pos_world = Convert.sphr2cart_scalar(
                az=target_azel[0], el=target_azel[1], angle_is_degree=True,
            )
            self.target_pos_monitor_mm = Convert.game2monitor(
                self.camera_pos_world, self.camera_azel_deg,
                self.target_pos_world, self.camera_fov_deg, self.monitor_half_size_mm,
            )
        self.target_speed_deg_s = (
            float(self.sample_target_speed_deg_s(rng=rng))
            if target_speed_deg_s is None else float(target_speed_deg_s)
        )

        # Target orbit axis reset (spherical az/el representation).
        if target_orbit_axis_deg is None:
            if target_motion_dir_deg is None:
                target_motion_dir_deg = self._sample_target_motion_dir_with_symmetry(rng=rng)
            target_motion_dir_deg = float(target_motion_dir_deg)
            # Precise screen-direction mapping:
            # 1) Build a tiny monitor-space step from current target monitor position
            #    using user convention: 0 deg -> +x, positive -> CCW.
            # 2) Convert stepped monitor point to world.
            # 3) Derive orbit axis from local world tangent so on-screen motion matches
            #    the requested direction even off-center.
            theta = np.radians(float(target_motion_dir_deg))
            monitor_step_mm = 1.0
            target_pos_monitor_mm_next = self.target_pos_monitor_mm + monitor_step_mm * np.array(
                [np.cos(theta), np.sin(theta)],
                dtype=float,
            )

            target_pos_world_next = Convert.monitor2game(
                self.camera_pos_world,
                self.camera_azel_deg,
                target_pos_monitor_mm_next,
                self.camera_fov_deg,
                self.monitor_half_size_mm,
            )

            ray = self.target_pos_world - self.camera_pos_world
            ray = ray / np.linalg.norm(ray)

            # Tangent from local monitor-driven perturbation.
            tangent = target_pos_world_next - self.target_pos_world
            tangent = tangent - np.dot(tangent, ray) * ray
            tangent_norm = np.linalg.norm(tangent)

            if tangent_norm < 1e-12:
                # Rare degenerate fallback.
                self.target_orbit_axis_deg = np.array(
                    sample_perpendicular_ray(
                        ray_direction=ray,
                        angle=90.0 - target_motion_dir_deg,
                        angle_is_degree=True,
                        return_spherical=True,
                    ),
                    dtype=float,
                )
            else:
                tangent = tangent / tangent_norm
                orbit_axis_world = np.cross(ray, tangent)
                orbit_axis_world = orbit_axis_world / np.linalg.norm(orbit_axis_world)
                self.target_orbit_axis_deg = Convert.cart2sphr_scalar(
                    orbit_axis_world[0],
                    orbit_axis_world[1],
                    orbit_axis_world[2],
                    return_in_degree=True,
                )
        else:
            self.target_orbit_axis_deg = np.array(target_orbit_axis_deg, dtype=float)
            if target_motion_dir_deg is None:
                delta_orbit_angle_deg = 1e-3
                target_pos_monitor_mm_next = self.target_monitor_position(
                    target_orbit_angle_deg=delta_orbit_angle_deg,
                )
                monitor_delta = target_pos_monitor_mm_next - self.target_pos_monitor_mm
                monitor_delta_norm = np.linalg.norm(monitor_delta)

                if np.isfinite(monitor_delta_norm) and monitor_delta_norm > 1e-12:
                    target_motion_dir_deg = float(
                        np.mod(np.degrees(np.arctan2(monitor_delta[1], monitor_delta[0])), 360.0)
                    )
                else:
                    target_motion_dir_deg = 0.0

        # Hand reset.
        self.hand_pos_mm = self.camera_azel_deg / self.hand_sensi_deg_per_mm
        self.hand_vel_mm_per_s = np.zeros(2, dtype=float)

        return dict(
            camera_azel_deg=self.camera_azel_deg.copy(),
            target_pos_monitor_mm=self.target_pos_monitor_mm.copy(),
            target_pos_world=self.target_pos_world.copy(),
            target_orbit_axis_deg=self.target_orbit_axis_deg.copy(),
            target_speed_deg_s=self.target_speed_deg_s,
            target_radius_deg=self.target_radius_deg,
            target_radius_mm=self.target_radius_mm,
            target_motion_dir_deg=target_motion_dir_deg,
        )
    

    @property
    def target_radius_mm(self) -> float:
        """Target radius in mm on the monitor.

        Computed via exact pinhole projection **at screen centre** (i.e., when
        the target aligns with the camera direction).  Uses the horizontal FOV
        and monitor width stored in ``self.config``.
        """
        return float(view_angle_deg_to_monitor_mm(
            self.target_radius_deg,
            self.config["monitor_width_mm"],
            self.config["cam_fov_deg_width"],
        ))


    def current_target_monitor_pos(self, camera_pos_world=None, camera_azel_deg=None, target_pos_world=None):
        return Convert.game2monitor(
            self.camera_pos_world if camera_pos_world is None else camera_pos_world,
            self.camera_azel_deg if camera_azel_deg is None else camera_azel_deg,
            self.target_pos_world if target_pos_world is None else target_pos_world,
            self.camera_fov_deg,
            self.monitor_half_size_mm,
        )
    

    def update_target_monitor_pos(self):
        self.target_pos_monitor_mm = self.current_target_monitor_pos()
    

    def orbit_target(self, dt, inplace=False, unit='s'):
        assert unit in ['s', 'ms'], "unit must be either 's' or 'ms'"
        target_orbit_angle_deg = self.target_speed_deg_s * dt if unit == 's' else self.target_speed_deg_s * dt / 1000.0
        orbit_axis = Convert.sphr2cart_scalar(
            az=self.target_orbit_axis_deg[0],
            el=self.target_orbit_axis_deg[1],
            angle_is_degree=True,
        )
        new_target_pos_world = point_about_ray(
            point=self.target_pos_world,
            angle=target_orbit_angle_deg,
            ray_direction=orbit_axis,
            ray_origin=np.zeros(3, dtype=float),
            angle_is_degree=True,
        )
        new_target_pos_monitor_mm = self.current_target_monitor_pos(target_pos_world=new_target_pos_world)

        if inplace:
            self.target_pos_world = np.array(new_target_pos_world, dtype=float)
            self.target_pos_monitor_mm = np.array(new_target_pos_monitor_mm, dtype=float)
        else:
            return (new_target_pos_world, new_target_pos_monitor_mm)


    def move_hand(self, hand_pos_delta_mm, hand_end_vel_mm_per_s):
        self.hand_pos_mm = self.hand_pos_mm + np.array(hand_pos_delta_mm, dtype=float)
        self.hand_vel_mm_per_s = np.array(hand_end_vel_mm_per_s, dtype=float)
        self.camera_azel_deg = self._clamp_camera_azel_deg(
            self.camera_azel_deg + np.array(hand_pos_delta_mm, dtype=float) * self.hand_sensi_deg_per_mm
        )
        self.update_target_monitor_pos()

    
    def get_current_state(self):
        return dict(
            target_pos_monitor_mm=self.target_pos_monitor_mm.copy(),
            target_pos_world=self.target_pos_world.copy(),
            target_vel_by_orbit_mm_s = self.target_monitor_velocity(),
            target_rad=self.target_radius_deg,
            target_rad_mm=self.target_radius_mm,   # exact pinhole projection at screen center
            camera_azel_deg=self.camera_azel_deg.copy(),
            hand_pos_mm=self.hand_pos_mm.copy(),
            hand_vel_mm_per_s=self.hand_vel_mm_per_s.copy(),
        )
    
    def replay_and_save(self,
        htraj_p, htraj_v, interval, unit='s', 
        keys=["target_pos_monitor_mm", "hand_pos_mm", "camera_azel_deg"]
    ):
        assert unit in ['s', 'ms'], "unit must be either 's' or 'ms'"
        hand_traj_pos_mm = np.array(htraj_p, dtype=float)
        hand_traj_vel_mm_per_s = np.array(htraj_v, dtype=float)
        assert hand_traj_pos_mm.ndim == 2 and hand_traj_pos_mm.shape[1] == 2, "htraj_p must have shape (n, 2)"
        assert hand_traj_vel_mm_per_s.ndim == 2 and hand_traj_vel_mm_per_s.shape[1] == 2, "htraj_v must have shape (n, 2)"
        assert hand_traj_pos_mm.shape[0] == hand_traj_vel_mm_per_s.shape[0], "htraj_p and htraj_v must have same length"

        n = hand_traj_pos_mm.shape[0]
        if n == 0:
            return {key: np.empty((0,), dtype=float) for key in keys}

        # Camera trajectory from hand trajectory (with elevation clamp).
        hand_delta_traj_mm = np.vstack((
            hand_traj_pos_mm[0] - self.hand_pos_mm,
            np.diff(hand_traj_pos_mm, axis=0),
        ))
        camera_azel_traj_deg = np.zeros((n, 2), dtype=float)
        cam_cur = self.camera_azel_deg.copy()
        for i in range(n):
            cam_cur = self._clamp_camera_azel_deg(cam_cur + hand_delta_traj_mm[i] * self.hand_sensi_deg_per_mm)
            camera_azel_traj_deg[i] = cam_cur

        # Vectorized target world trajectory (axis-fixed orbit around origin).
        orbit_axis = Convert.sphr2cart_scalar(
            az=self.target_orbit_axis_deg[0],
            el=self.target_orbit_axis_deg[1],
            angle_is_degree=True,
        )
        orbit_axis = orbit_axis / np.linalg.norm(orbit_axis)

        orbit_step_deg = self.target_speed_deg_s * interval if unit == 's' else self.target_speed_deg_s * interval / 1000.0
        orbit_angles_deg = orbit_step_deg * np.arange(n, dtype=float)

        v0 = np.array(self.target_pos_world, dtype=float)
        c = np.cos(np.radians(orbit_angles_deg))[:, None]
        s = np.sin(np.radians(orbit_angles_deg))[:, None]
        k = orbit_axis[None, :]
        kv = float(np.dot(orbit_axis, v0))
        k_cross_v0 = np.cross(orbit_axis, v0)[None, :]
        target_pos_world_traj = v0[None, :] * c + k_cross_v0 * s + k * kv * (1.0 - c)

        target_pos_monitor_mm_traj = Convert.game2monitor(
            self.camera_pos_world,
            camera_azel_traj_deg,
            target_pos_world_traj,
            self.camera_fov_deg,
            self.monitor_half_size_mm,
        )

        # Optional velocity key (vectorized finite-difference model used in `target_monitor_velocity`).
        target_vel_by_orbit_mm_s_traj = None
        if "target_vel_by_orbit_mm_s" in keys:
            dt = DELTA_TIME
            camera_azel_next_deg = camera_azel_traj_deg + hand_traj_vel_mm_per_s * dt * self.hand_sensi_deg_per_mm
            dtheta = np.radians(self.target_speed_deg_s * dt)
            cd, sd = np.cos(dtheta), np.sin(dtheta)
            k_cross_v = np.cross(np.repeat(k, n, axis=0), target_pos_world_traj)
            k_dot_v = np.sum(target_pos_world_traj * k, axis=1, keepdims=True)
            target_pos_world_next = target_pos_world_traj * cd + k_cross_v * sd + np.repeat(k, n, axis=0) * k_dot_v * (1.0 - cd)
            target_pos_monitor_mm_next = Convert.game2monitor(
                self.camera_pos_world,
                camera_azel_next_deg,
                target_pos_world_next,
                self.camera_fov_deg,
                self.monitor_half_size_mm,
            )
            target_vel_by_orbit_mm_s_traj = (target_pos_monitor_mm_next - target_pos_monitor_mm_traj) / dt

        # Build logs with precomputed arrays.
        trajectory_log = {}
        for key in keys:
            if key == "target_pos_monitor_mm":
                trajectory_log[key] = target_pos_monitor_mm_traj
            elif key == "target_pos_world":
                trajectory_log[key] = target_pos_world_traj
            elif key == "camera_azel_deg":
                trajectory_log[key] = camera_azel_traj_deg
            elif key == "hand_pos_mm":
                trajectory_log[key] = hand_traj_pos_mm
            elif key == "hand_vel_mm_per_s":
                trajectory_log[key] = hand_traj_vel_mm_per_s
            elif key == "target_vel_by_orbit_mm_s":
                trajectory_log[key] = target_vel_by_orbit_mm_s_traj

        return trajectory_log
    

    def target_monitor_position(self, initial_target_pos_monitor_mm=None,
                                hand_displacement_mm=np.zeros(2), target_orbit_angle_deg=0, clip_ratio=2):
        new_camera_azel_deg = self._clamp_camera_azel_deg(
            self.camera_azel_deg + np.array(hand_displacement_mm, dtype=float) * self.hand_sensi_deg_per_mm
        )
        target_pos_world_start = self.target_pos_world.copy() if initial_target_pos_monitor_mm is None \
            else self._game_position(target_pos_monitor_mm=initial_target_pos_monitor_mm)
        orbit_axis = Convert.sphr2cart_scalar(
            az=self.target_orbit_axis_deg[0],
            el=self.target_orbit_axis_deg[1],
            angle_is_degree=True,
        )
        target_pos_world_rotated = point_about_ray(
            point=target_pos_world_start,
            angle=target_orbit_angle_deg,
            ray_direction=orbit_axis,
            ray_origin=np.zeros(3, dtype=float),
            angle_is_degree=True,
        )
        return np.clip(
            self._monitor_position(camera_azel_deg=new_camera_azel_deg, target_pos_world=target_pos_world_rotated),
            a_min=-clip_ratio * self.monitor_half_size_mm,
            a_max=clip_ratio * self.monitor_half_size_mm,
        )
    

    def target_monitor_velocity(
        self, 
        initial_target_pos_monitor_mm=None, 
        hand_vel_mm=np.zeros(2), 
        target_speed_deg_s=None, 
        dt=DELTA_TIME
    ):
        new_camera_azel_deg = self._clamp_camera_azel_deg(
            self.camera_azel_deg + np.array(hand_vel_mm, dtype=float) * dt * self.hand_sensi_deg_per_mm
        )
        use_current_target_pos_world = initial_target_pos_monitor_mm is None
        initial_target_pos_monitor_mm = self.target_pos_monitor_mm.copy() if use_current_target_pos_world \
            else np.array(initial_target_pos_monitor_mm, dtype=float)
        target_pos_world_start = self.target_pos_world.copy() if use_current_target_pos_world \
            else self._game_position(target_pos_monitor_mm=initial_target_pos_monitor_mm)
        orbit_axis = Convert.sphr2cart_scalar(
            az=self.target_orbit_axis_deg[0],
            el=self.target_orbit_axis_deg[1],
            angle_is_degree=True,
        )
        target_orbit_angle_deg = self.target_speed_deg_s * dt if target_speed_deg_s is None else float(target_speed_deg_s) * dt
        target_pos_world_next = point_about_ray(
            point=target_pos_world_start,
            angle=target_orbit_angle_deg,
            ray_direction=orbit_axis,
            ray_origin=self.camera_pos_world,
            angle_is_degree=True,
        )
        target_pos_monitor_mm_next = self._monitor_position(camera_azel_deg=new_camera_azel_deg, 
                                                            target_pos_world=target_pos_world_next)
        return (target_pos_monitor_mm_next - initial_target_pos_monitor_mm) / dt


    def target_crosshair_distance_mm(self):
        return np.linalg.norm(self.target_pos_monitor_mm - self.crosshair_pos_mm)
    

    def crosshair_on_target(
        self, 
        target_pos_world=None,
        target_pos_monitor_mm=None,
        camera_azel_deg=None,
        return_dist=False
    ):
        target_pos_world = self.target_pos_world if target_pos_world is None else np.array(target_pos_world, dtype=float)
        camera_azel_deg = self.camera_azel_deg if camera_azel_deg is None else np.array(camera_azel_deg, dtype=float)
        if target_pos_monitor_mm is not None:
            target_pos_world = self._game_position(
                target_pos_monitor_mm=target_pos_monitor_mm,
                camera_azel_deg=camera_azel_deg,
            )
        dist_ang = angle_between_vectors(target_pos_world - self.camera_pos_world, 
                                     Convert.sphr2cart_scalar(*camera_azel_deg, angle_is_degree=True))
        if return_dist:
            return dist_ang <= self.target_radius_deg, dist_ang
        return dist_ang <= self.target_radius_deg
    

    def target_out_of_monitor(self):
        return np.any(self.target_pos_monitor_mm > self.monitor_half_size_mm) or \
            np.any(self.target_pos_monitor_mm < -self.monitor_half_size_mm)


    def sample_task_condition(self, n, **cond):
        return [self.reset(**cond) for _ in range(n)]
    

    def target_monitor_distance_mm(self, target_pos_monitor_mm=None):
        target_pos_monitor_mm = self.target_pos_monitor_mm if target_pos_monitor_mm is None else np.array(target_pos_monitor_mm, dtype=float)
        return np.linalg.norm(target_pos_monitor_mm - self.crosshair_pos_mm)


    def _monitor_position(
        self,
        camera_pos_world=None,
        camera_azel_deg=None,
        target_pos_world=None,
        camera_fov_deg=None,
        monitor_half_size_mm=None,
    ):
        return Convert.game2monitor(
            camera_pos_world if camera_pos_world is not None else self.camera_pos_world,
            camera_azel_deg if camera_azel_deg is not None else self.camera_azel_deg,
            target_pos_world if target_pos_world is not None else self.target_pos_world,
            camera_fov_deg if camera_fov_deg is not None else self.camera_fov_deg,
            monitor_half_size_mm if monitor_half_size_mm is not None else self.monitor_half_size_mm,
        )
    
    def _game_position(
        self,
        camera_pos_world=None,
        camera_azel_deg=None,
        target_pos_monitor_mm=None,
        camera_fov_deg=None,
        monitor_half_size_mm=None,
    ):
        return Convert.monitor2game(
            camera_pos_world if camera_pos_world is not None else self.camera_pos_world,
            camera_azel_deg if camera_azel_deg is not None else self.camera_azel_deg,
            target_pos_monitor_mm if target_pos_monitor_mm is not None else self.target_pos_monitor_mm,
            camera_fov_deg if camera_fov_deg is not None else self.camera_fov_deg,
            monitor_half_size_mm if monitor_half_size_mm is not None else self.monitor_half_size_mm,
        )


    def output_current_status(self):
        # Camera
        print(f"Camera angle: Azim {self.camera_azel_deg[0]:.3f}, Elev {self.camera_azel_deg[1]:.3f}")

        # Hand
        print(f"Hand: X {self.hand_pos_mm[0]:.3f}, Y {self.hand_pos_mm[1]:.3f}, VX {self.hand_vel_mm_per_s[0]:.3f}, VY {self.hand_vel_mm_per_s[1]:.3f}")

        # Target
        print(f"Target: GX {self.target_pos_world[0]} GY {self.target_pos_world[1]} GZ {self.target_pos_world[2]}, MX {self.target_pos_monitor_mm[0]} MY {self.target_pos_monitor_mm[1]}")


if __name__ == "__main__":
    """
    Sample distribution check for fixed_reward.yaml preset.
    Plots Fitts' ID, target radius (deg), and initial target distance (mm).
    """
    import os, sys
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # ── load preset ──────────────────────────────────────────────────────────
    from ..configs.loader import CFG_AGENT_PROFILE
    preset_cfg = CFG_AGENT_PROFILE["fixed_reward"]
    task_cfg   = preset_cfg["task"]

    task = AimandShootSpiderShotTask(config=task_cfg)
    rng  = np.random.default_rng(0)

    N = 10_000
    ids      = np.empty(N)
    radii    = np.empty(N)   # deg
    dists_mm = np.empty(N)   # initial target distance from crosshair (mm)

    for i in range(N):
        info = task.reset(rng=rng)
        r_deg = info["target_radius_deg"]
        d_mm  = float(np.linalg.norm(info["target_pos_monitor_mm"]))
        r_mm  = info["target_radius_mm"]
        id_val = np.log2(d_mm / (2.0 * r_mm) + 1.0) if r_mm > 0 else 0.0
        ids[i]      = id_val
        radii[i]    = r_deg
        dists_mm[i] = d_mm

    print(f"N = {N}")
    print(f"ID     : min={ids.min():.3f}  max={ids.max():.3f}  mean={ids.mean():.3f}  std={ids.std():.3f}")
    print(f"Radius : min={radii.min():.3f}  max={radii.max():.3f}  mean={radii.mean():.3f}  std={radii.std():.3f}  (deg)")
    print(f"Dist   : min={dists_mm.min():.1f}  max={dists_mm.max():.1f}  mean={dists_mm.mean():.1f}  std={dists_mm.std():.1f}  (mm)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(ids, bins=50, edgecolor="white", linewidth=0.3)
    axes[0].set_title("Fitts' ID distribution")
    axes[0].set_xlabel("ID (bits)")
    axes[0].set_ylabel("Count")

    axes[1].hist(radii, bins=50, edgecolor="white", linewidth=0.3)
    axes[1].set_title("Target radius distribution")
    axes[1].set_xlabel("Radius (deg)")
    axes[1].set_ylabel("Count")

    axes[2].hist(dists_mm, bins=50, edgecolor="white", linewidth=0.3)
    axes[2].set_title("Initial target distance distribution")
    axes[2].set_xlabel("Distance (mm)")
    axes[2].set_ylabel("Count")

    plt.suptitle(f"fixed_reward preset — N={N}", fontsize=12)
    plt.tight_layout()
    plt.show()
    
