"""
Visual Perception Module

Migrated from the original Aim-and-Shoot code repository
Code modified by June-Seop Yoon
"""

import numpy as np

from ..utils.mymath import (
    angle_between_vectors,
    normalize_vector,
    point_about_ray,
    sample_perpendicular_ray,
)


DEFAULT_HEAD_POSITION = np.array([0.0, 77.9, 575.0], dtype=float)
DEFAULT_MONITOR_HALF_SIZE_MM = np.array([531.3, 298.8], dtype=float) / 2.0


class Perceive:
    def position(
        opos,
        gpos,
        noise,
        head=DEFAULT_HEAD_POSITION,
        monitor_qt=DEFAULT_MONITOR_HALF_SIZE_MM,
        return_sigma=False,
    ):
        """
        opos, gpos := objective/gaze position as 2D state on the monitor
        head := 3D state in the physical world
        """
        head = np.asarray(head, dtype=float)
        monitor_qt = np.asarray(monitor_qt, dtype=float)
        opos = np.asarray(opos, dtype=float)
        gpos = np.asarray(gpos, dtype=float)

        obj_ray = np.array([opos[0], opos[1], 0.0], dtype=float) - head
        gaze_ray = np.array([gpos[0], gpos[1], 0.0], dtype=float) - head

        ecc_dist = angle_between_vectors(
            obj_ray,
            gaze_ray,
            return_in_degree=True
        )
        sigma = noise * ecc_dist
        est_error = np.random.normal(0, sigma)

        # Sample the estimated position in visual angle space
        head_to_obj = normalize_vector(obj_ray)
        head_to_obj_noisy = point_about_ray(
            head_to_obj,
            est_error,
            sample_perpendicular_ray(head_to_obj),
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            hat_pos = head[:2] - (head[2] / head_to_obj_noisy[2]) * head_to_obj_noisy[:2]
        hat_pos = np.clip(
            hat_pos,
            -monitor_qt,
            monitor_qt
        )
        if return_sigma:
            return hat_pos, sigma
        return hat_pos
    

    def velocity(vel, pos, noise, head=np.array([0., 77.9, 575.0]), s0=0.3, dt=0.01):
        """
        pos, vel := 2D state on the monitor (mm/s)
        head := 3D head position in the real world (mm); (0, 0, 0) is the center of the monitor
        """
        spd = np.linalg.norm(vel)
        if spd <= 0: return vel

        p0 = np.array([*pos, 0])
        p1 = np.array([*(pos + vel * dt), 0])
        
        # Angle between objective movement and head to objective
        aspd = angle_between_vectors(p0 - head, p1 - head) / dt
        if aspd <= 0:
            return vel
        
        aspd_hat = np.log(1 + aspd / s0)
        s_prime = np.random.lognormal(aspd_hat, noise)
        s_final = np.clip((s_prime - 1) * s0, 0, np.inf)

        return (s_final / aspd) * vel
    

    def timing(t, noise):
        while True:
            clock_noise = np.random.normal(1, noise)
            if 0.01 < clock_noise < 2: break
        return t * clock_noise


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    np.random.seed(42)

    opos = np.zeros(2, dtype=float)
    gpos_list = [
        np.array([0.20, 0.0], dtype=float),
        np.array([0.15, 0.0], dtype=float),
        np.array([0.10, 0.0], dtype=float),
        np.array([0.05, 0.0], dtype=float),
        np.array([0.00, 0.0], dtype=float),
    ]
    noise_list = [0.2, 0.1]
    colors = {0.2: "tab:blue", 0.1: "tab:orange"}
    n_sample = 8_000
    scatter_sample = 2_000

    def add_cov_ellipse(ax, samples, color, n_std=2.0):
        cov = np.cov(samples.T)
        if not np.all(np.isfinite(cov)) or np.max(np.abs(cov)) <= 1e-15:
            return
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 2.0 * n_std * np.sqrt(np.maximum(vals, 0.0))
        ellipse = Ellipse(
            xy=samples.mean(axis=0),
            width=width,
            height=height,
            angle=angle,
            fill=False,
            edgecolor=color,
            linewidth=2.0,
        )
        ax.add_patch(ellipse)

    fig, axes = plt.subplots(1, len(gpos_list), figsize=(4 * len(gpos_list), 4), sharex=True, sharey=True)
    if len(gpos_list) == 1:
        axes = [axes]

    print("=== Perceive.position validation ===")
    print(f"opos={opos.tolist()}, n_sample={n_sample}")

    for ax, gpos in zip(axes, gpos_list):
        for noise in noise_list:
            samples = np.array([
                Perceive.position(opos=opos, gpos=gpos, noise=noise)
                for _ in range(n_sample)
            ])
            std_xy = samples.std(axis=0)
            sigma_deg = Perceive.position(opos=opos, gpos=gpos, noise=noise, return_sigma=True)[1]
            print(
                f"gpos={gpos.tolist()}, noise={noise:.1f}, "
                f"sigma_deg={sigma_deg:.6f}, std_xy_mm={std_xy.round(6).tolist()}"
            )

            shown = samples[np.random.choice(n_sample, scatter_sample, replace=False)]
            ax.scatter(
                shown[:, 0],
                shown[:, 1],
                s=4,
                alpha=0.18,
                color=colors[noise],
                label=f"noise={noise:.1f}",
                rasterized=True,
            )
            add_cov_ellipse(ax, samples, colors[noise])

        ax.scatter(opos[0], opos[1], marker="+", s=80, color="black", linewidths=1.5, label="opos")
        ax.scatter(gpos[0], gpos[1], marker="x", s=60, color="gray", linewidths=1.5, label="gpos")
        ax.set_title(f"gpos=({gpos[0]:.2f}, {gpos[1]:.2f})")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.set_xlabel("estimated x (mm)")

    axes[0].set_ylabel("estimated y (mm)")
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=4)
    fig.suptitle("Perceive.position estimation noise by gaze-object eccentricity", y=1.04)
    plt.tight_layout()
    plt.show()

    if False:
        noise = 0.5
        pos = np.array([0.0, 0.0], dtype=float)
        vel = np.array([150.0, 0.0], dtype=float)

        n_sample = 10_000
        vel_hat = np.array([Perceive.velocity(vel=vel, pos=pos, noise=noise) for _ in range(n_sample)])
        spd_hat = np.linalg.norm(vel_hat, axis=1)

        print("=== Perceive.velocity validation ===")
        print(f"noise={noise}, pos={pos.tolist()}, vel={vel.tolist()}")
        print(f"sample size={n_sample}")
        print(f"speed mean={spd_hat.mean():.4f}, std={spd_hat.std():.4f}, min={spd_hat.min():.4f}, max={spd_hat.max():.4f}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(spd_hat, bins=80)
        axes[0].set_title("Perceived speed histogram")
        axes[0].set_xlabel("speed (mm/s)")
        axes[0].set_ylabel("count")

        axes[1].hist(vel_hat[:, 0], bins=80, alpha=0.8, label="vx")
        axes[1].hist(vel_hat[:, 1], bins=80, alpha=0.6, label="vy")
        axes[1].set_title("Perceived velocity components")
        axes[1].set_xlabel("velocity component (mm/s)")
        axes[1].set_ylabel("count")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

