"""
Non-hierarchical amortized inference on real (or simulated) user data.

Given a trained amortizer checkpoint and per-trial summary statistics,
this module:
  1. Normalises the raw data using the config stored in the checkpoint.
  2. Runs the trained network to produce parameter estimates in z-space.
  3. Maps the z-space estimates back to actual parameter values (w-space).
  4. Optionally re-simulates with the inferred parameters for verification.

Usage example::

    inferer = AmortizedInferer(run_name="2506011234", iteration=100)
    result = inferer.infer(stat_df)
    print(result)  # {param_motor_noise: 0.12, param_speed_noise: 0.04, ...}

Code written by June-Seop Yoon
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from ..nets.amortizer import AmortizerForTrialData, RegressionForTrialData
from ..configs.constants import FOLDERS
from ..utils.mymath import linear_normalize
from ..utils.mytorch import get_auto_device

from .dataset import AmortizedInferenceSimulator
from .local_utils import (
    DIR_PROJECT_ROOT,
    normalize_per_feature,
    _build_param_info,
    _z_to_w as params_z_to_w,
    _w_to_z as params_w_to_z,
)


# ─────────────────────────────────────────────────────────────────────────────
# Inferer
# ─────────────────────────────────────────────────────────────────────────────

class AmortizedInferer:
    """Load a trained amortizer and run inference on real data.

    Parameters
    ----------
    run_name : str
        Name of the training run (directory under ``data/amortized_inference/models/``).
    iteration : int, optional
        Specific checkpoint iteration to load.  *None* → latest.
    device : str
        ``'auto'``, ``'cuda'``, or ``'cpu'``.
    """

    def __init__(
        self,
        run_name: str,
        iteration: Optional[int] = None,
        device: str = "auto",
    ) -> None:
        model_dir = (
            DIR_PROJECT_ROOT / FOLDERS.DATA / FOLDERS.AMORTIZER
            / FOLDERS.AMT_MODELS / run_name
        )
        pts_dir = model_dir / "pts"

        if not pts_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {pts_dir}")

        # Resolve checkpoint path
        if iteration is not None:
            ckpt_path = pts_dir / f"iter_{iteration:04d}.pt"
        else:
            ckpts = sorted(pts_dir.glob("iter_*.pt"))
            if not ckpts:
                raise FileNotFoundError(f"No checkpoints in {pts_dir}")
            ckpt_path = ckpts[-1]

        if device == "auto":
            self.device = get_auto_device()
        else:
            self.device = torch.device(device)

        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)

        # ── Restore metadata ──────────────────────────────────────────────
        self.run_name = run_name
        self.point_estimation: bool = ckpt["point_estimation"]
        raw_param_names = ckpt["param_names"]
        if raw_param_names and isinstance(raw_param_names[0], dict):
            self.param_names = [p.get("output_column", p["name"]) for p in raw_param_names]
        else:
            self.param_names = list(raw_param_names)
        self.stat_feature_names: List[str] = ckpt["stat_feature_names"]
        self.stat_feature_cfgs: List[dict] = ckpt["stat_feature_cfgs"]
        self.normalize_config: dict = ckpt["normalize_config"]
        self.traj_feature_cfgs: List[dict] = ckpt.get(
            "traj_feature_cfgs",
            self.normalize_config.get("traj", {}).get("features", []),
        )
        self.traj_feature_names: List[str] = [
            f["name"] for f in self.traj_feature_cfgs
        ]
        self.n_params = len(self.param_names)

        # ── Rebuild model ─────────────────────────────────────────────────
        amt_cfg = ckpt["amortizer_config"]
        if self.point_estimation:
            self.model = RegressionForTrialData(amt_cfg)
        else:
            self.model = AmortizerForTrialData(amt_cfg)

        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()

        # ── Parameter info (for z↔w conversion) ──────────────────────────
        # Rebuild from the config stored alongside the checkpoint
        config_path = pts_dir / "config.yaml"
        if config_path.exists():
            from ..utils.myutils import load_yaml_config
            full_config = load_yaml_config(str(config_path))
            self._full_config = full_config
        else:
            self._full_config = None

        sim_cfg = self._full_config.get("simulator", {}) if self._full_config else {}
        legacy_sim_cfg = sim_cfg.get("legacy_source", {}) if isinstance(sim_cfg, dict) else {}
        self.traj_downsample_hz = float(
            sim_cfg.get("traj_downsample", legacy_sim_cfg.get("traj_downsample", 40.0))
        )

        self._param_info: Optional[list] = None  # lazily built

        print(
            f"[AmortizedInferer] loaded {ckpt_path.name}  "
            f"({'regression' if self.point_estimation else 'density'})\n"
            f"  params: {self.param_names}\n"
            f"  stat features: {len(self.stat_feature_names)}"
        )

    # ── Public API ────────────────────────────────────────────────────────

    def infer(
        self,
        stat_data: Union[pd.DataFrame, np.ndarray],
        traj_data=None,
        n_sample: int = 500,
        estimation_type: str = "mode",
        return_z: bool = False,
        return_samples: bool = False,
    ) -> Union[dict, Tuple[dict, np.ndarray]]:
        """Infer parameters from observed trial data.

        Parameters
        ----------
        stat_data : DataFrame or ndarray
            If DataFrame, column names must match ``stat_feature_names`` from
            the config.  Shape (n_trials, n_stat_features).
            If ndarray, assumed already aligned (n_trials, n_stat_features).
        traj_data : list, optional
            Per-trial trajectory data (for encoder with trajectory input).
        n_sample : int
            Number of posterior samples (density estimation only).
        estimation_type : str
            ``'mode'``, ``'mean'``, or ``'median'``  (density estimation only).
        return_z : bool
            If True, also return the z-space estimate.
        return_samples : bool
            If True, also return posterior samples (density only).

        Returns
        -------
        result : dict
            ``{param_name: value}`` in actual (w-space) units.
        z_estimate : ndarray, optional
            z-space estimate, shape (n_params,).
        """
        # Normalise input
        stat_norm = self._normalize_input(stat_data)
        traj_norm = self._normalize_traj_input(traj_data)
        stat_tensor = torch.as_tensor(
            stat_norm,
            dtype=torch.float32,
            device=self.device,
        )
        # stat_norm: (n_trials, n_stat)

        # Run inference
        with torch.no_grad():
            if self.point_estimation:
                z_est = self.model.infer(
                    stat_tensor, traj_data=traj_norm,
                )  # (n_params,) numpy
                samples = None
            else:
                result = self.model.infer(
                    stat_tensor, traj_data=traj_norm,
                    n_sample=n_sample,
                    type=estimation_type,
                    return_samples=return_samples,
                )
                if return_samples:
                    z_est, samples = result
                else:
                    z_est = result
                    samples = None

        z_est = np.asarray(z_est, dtype=np.float32).reshape(self.n_params)
        z_est = np.clip(z_est, -1.0, 1.0)

        # Convert z → w
        param_info = self._get_param_info()
        if param_info is not None:
            w_est = params_z_to_w(z_est, param_info)
        else:
            # Fallback: treat z as w (no conversion info available)
            w_est = z_est

        result_dict = {
            name: float(w_est[i])
            for i, name in enumerate(self.param_names)
        }

        if return_z:
            if return_samples and samples is not None:
                return result_dict, z_est, samples
            return result_dict, z_est
        if return_samples and samples is not None:
            return result_dict, samples
        return result_dict

    def process_data(
        self,
        stat_data: pd.DataFrame,
        traj_data: Optional[List[pd.DataFrame]] = None,
    ):
        """Legacy-compatible preprocessing helper.

        This mirrors ``src/inference/inferer.py::AnSInferer.process_data``:
        select configured features, normalize static features, and for raw
        trajectory DataFrames resample to the stored downsample rate while
        replacing ``timestamp`` with per-step ``dt``.
        """
        stat_norm = self._normalize_input(stat_data)
        if traj_data is None:
            return stat_norm
        return stat_norm, self._normalize_traj_input(traj_data)

    def infer_batch(
        self,
        stat_data_list: List[Union[pd.DataFrame, np.ndarray]],
        traj_data_list=None,
        n_sample: int = 500,
        estimation_type: str = "mode",
    ) -> List[dict]:
        """Infer parameters for multiple users (each with their own trial set).

        Parameters
        ----------
        stat_data_list : list of DataFrames or ndarrays
            One element per user.
        traj_data_list : list, optional
        n_sample, estimation_type :
            Passed to :meth:`infer`.

        Returns
        -------
        List of {param_name: value} dicts.
        """
        results = []
        for i, stat_data in enumerate(stat_data_list):
            traj = traj_data_list[i] if traj_data_list else None
            r = self.infer(
                stat_data, traj_data=traj,
                n_sample=n_sample,
                estimation_type=estimation_type,
            )
            results.append(r)
        return results

    def simulate_with_inferred(
        self,
        inferred_params: dict,
        n_episodes: int = 64,
        num_cpu: int = 1,
    ) -> pd.DataFrame:
        """Re-simulate the environment with inferred parameters for verification.

        Parameters
        ----------
        inferred_params : dict
            ``{param_name: value}`` in w-space, as returned by :meth:`infer`.
        n_episodes : int
        num_cpu : int

        Returns
        -------
        DataFrame of simulation results.
        """
        if self._full_config is None:
            raise RuntimeError(
                "Full config not available — cannot create simulator. "
                "Ensure config.yaml exists alongside the checkpoint."
            )
        sim = AmortizedInferenceSimulator(self._full_config)
        param_info = self._get_param_info()
        w_vec = np.array([inferred_params[name] for name in self.param_names], dtype=np.float32)
        z_vec = params_w_to_z(w_vec, param_info)

        summary_df, _ = sim.simulate_with_params(
            param_z=z_vec,
            n_episodes=n_episodes,
            num_cpu=num_cpu,
            save_trajectory=False,
        )
        return summary_df

    # ── Internal ──────────────────────────────────────────────────────────

    def _normalize_input(
        self,
        stat_data: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """Normalize raw stat data to [-1, 1] using stored feature configs.

        Returns ndarray of shape (n_trials, n_stat_features).
        """
        if isinstance(stat_data, pd.DataFrame):
            # Extract columns in the correct order
            arr = np.zeros((len(stat_data), len(self.stat_feature_names)), dtype=np.float32)
            for i, feat_name in enumerate(self.stat_feature_names):
                if feat_name not in stat_data.columns:
                    raise KeyError(
                        f"Feature '{feat_name}' not in input DataFrame. "
                        f"Expected: {self.stat_feature_names}"
                    )
                arr[:, i] = stat_data[feat_name].to_numpy(dtype=np.float32)
        else:
            arr = np.asarray(stat_data, dtype=np.float32)

        return normalize_per_feature(arr, self.stat_feature_cfgs)

    def _normalize_traj_input(self, traj_data):
        """Normalize raw trajectory DataFrames using the legacy convention.

        Existing callers may already pass normalized numpy trajectories; those
        are left untouched.  Raw DataFrames are converted exactly like the
        legacy inferer: absolute ``timestamp`` is used for interpolation, then
        stored as ``dt`` before feature-wise normalization.
        """
        if traj_data is None:
            return None

        if self.normalize_config.get("traj", {}).get("ignore", False):
            return None

        if not self.traj_feature_cfgs:
            return traj_data

        first = traj_data[0] if len(traj_data) else None
        if not isinstance(first, pd.DataFrame):
            return traj_data

        traj_norm = []
        traj_value_range = np.array(
            [[f["min"], f["max"]] for f in self.traj_feature_cfgs],
            dtype=np.float32,
        )
        direct_columns_available = all(name in first.columns for name in self.traj_feature_names)

        for traj_df in traj_data:
            if traj_df.empty:
                traj_norm.append(
                    np.zeros((0, len(self.traj_feature_cfgs)), dtype=np.float32)
                )
                continue

            if direct_columns_available:
                new_traj = traj_df[self.traj_feature_names].to_numpy(dtype=np.float32)
            else:
                new_traj = self._resample_legacy_traj_dataframe(traj_df)

            traj_norm.append(
                linear_normalize(
                    new_traj,
                    traj_value_range[:, 0],
                    traj_value_range[:, 1],
                    dtype=np.float32,
                )
            )

        return traj_norm

    def _resample_legacy_traj_dataframe(self, traj_df: pd.DataFrame) -> np.ndarray:
        timestamp_candidates = ["timestamp", "timestamp_s", "timestamp_ms"]
        timestamp_key = next(
            (name for name in timestamp_candidates if name in traj_df.columns),
            None,
        )
        if timestamp_key is None:
            raise KeyError(
                "Trajectory DataFrame must contain configured trajectory columns "
                f"{self.traj_feature_names} or one of {timestamp_candidates} for "
                "legacy resampling."
            )

        old_timestamp = traj_df[timestamp_key].to_numpy(dtype=np.float32)
        if timestamp_key.endswith("_ms"):
            old_timestamp = old_timestamp / 1000.0
        if old_timestamp.size == 0:
            return np.zeros((0, len(self.traj_feature_cfgs)), dtype=np.float32)

        legacy_name_map = {
            "dt_ms": "timestamp",
            "target_x_mm": "target_pos_monitor_x",
            "target_y_mm": "target_pos_monitor_y",
            "camera_az_deg": "player_cam_az",
            "camera_el_deg": "player_cam_el",
        }
        last_t = float(old_timestamp[-1])
        n_steps = max(1, int(last_t * self.traj_downsample_hz))
        new_timestamp = np.linspace(0.0, last_t, n_steps, dtype=np.float32)
        new_traj_cols = []

        for feature_name in self.traj_feature_names:
            legacy_name = legacy_name_map.get(feature_name, feature_name)
            if feature_name in ("dt_ms", "timestamp") or legacy_name == "timestamp":
                dt_ms = np.insert(np.diff(new_timestamp), 0, 0.0) * 1000.0
                new_traj_cols.append(dt_ms.astype(np.float32))
                continue
            if feature_name in traj_df.columns:
                source_name = feature_name
            elif legacy_name in traj_df.columns:
                source_name = legacy_name
            else:
                raise KeyError(
                    f"Feature '{feature_name}' not in trajectory DataFrame. "
                    f"Expected current name or legacy name '{legacy_name}'."
                )
            values = traj_df[source_name].to_numpy(dtype=np.float32)
            if source_name in ("target_pos_monitor_x", "target_pos_monitor_y"):
                values = values * 1000.0
            new_traj_cols.append(
                np.interp(new_timestamp, old_timestamp, values).astype(np.float32)
            )

        return np.array(new_traj_cols, dtype=np.float32).T

    def _get_param_info(self) -> Optional[list]:
        """Lazily build param_info from the stored config."""
        if self._param_info is not None:
            return self._param_info

        if self._full_config is None:
            return None

        self._param_info = _build_param_info(self._full_config["infer_params"])

        return self._param_info
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run amortized inference on data.")
    parser.add_argument("--run", type=str, required=True, help="Training run name")
    parser.add_argument("--iter", type=int, default=None, help="Checkpoint iteration")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    inferer = AmortizedInferer(
        run_name=args.run,
        iteration=args.iter,
        device=args.device,
    )
    print(f"Inferer ready. Param names: {inferer.param_names}")
