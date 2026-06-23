"""
Non-hierarchical dataset generation and management for amortized inference.

Generates (θ, x, y) tuples by sampling user-parameter vectors θ, running the
Aim-and-Shoot simulator with those parameters fixed, and recording per-trial
summary statistics (x = task conditions, y = performance).

**Raw storage policy**
  Summary statistics are saved to disk as *raw* (un-normalised) float32 arrays
  containing **all** simulator output columns.  Feature selection and
  normalisation are applied once in :meth:`_load_existing` (on load into
  memory), so dataset files remain valid after changing feature bounds or
  feature selection without re-running the simulator.

Storage layout under  data/amortized_inference/<config_hash>/ :
    train/
        <timestamp>_<N>p_<M>ep.h5          — raw summary data (HDF5, gzip)
        <timestamp>_<N>p_<M>ep_traj.pkl    — coarse trajectories (optional)
    valid/
        <timestamp>_<Nu>u_<Nt>t.h5
        <timestamp>_<Nu>u_<Nt>t_traj.pkl

Code written by June-Seop Yoon
"""

from __future__ import annotations

import glob
import numpy as np
# On Windows, pandas-before-torch can make PyTorch's DLL loader fail.
import torch
import pandas as pd
import psutil
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm

from ..agent.simulator import AimandShootSimulator
from ..configs.constants import FOLDERS
from ..configs.loader import CFG_AMORTIZED_INFERENCE
from ..utils.myutils import pickle_save, pickle_load, get_compact_timestamp_str
from .local_utils import (
    DIR_PROJECT_ROOT,
    normalize_per_feature,
    denormalize_per_feature,
    _sample_z,
    _z_to_w,
    _w_to_z,
    _process_traj_dicts,
    _build_param_info,
    _dataset_hash,
    _write_dataset_meta,
    h5_save,
    h5_load,
)

class AmortizedInferenceSimulator:
    """Wraps :class:`AimandShootSimulator` for dataset generation."""

    def __init__(self, config: dict):
        sim_cfg = config["simulator"]
        manifold_cfg = sim_cfg.get("manifold", {})
        self.use_manifold_flag: bool = manifold_cfg.get("enabled", False)

        # When manifold is enabled, the base SAC checkpoint is not needed —
        # only agent_config.yaml is required.  Pass ckpt=None to skip the
        # checkpoint validation that would otherwise fail on machines that
        # only have manifold weights (.pt) but no SAC checkpoint (.zip).
        sac_ckpt = None if self.use_manifold_flag else sim_cfg.get("ckpt", "latest")
        self.simulator = AimandShootSimulator(
            model_name=sim_cfg["model_name"],
            ckpt=sac_ckpt,
        )
        self.agent_config = deepcopy(self.simulator._env_config)

        if self.use_manifold_flag:
            manifold_name = manifold_cfg["manifold_name"]
            epoch = manifold_cfg.get("epoch", None)
            self.simulator.use_manifold(manifold_name, tau=None, epoch=epoch)

        self.param_info = _build_param_info(config["infer_params"])
        self.param_names: List[str] = [pi["name"] for pi in self.param_info]
        self.n_params: int = len(self.param_info)
        self.has_tau: bool = any(pi.get("is_tau", False) for pi in self.param_info)

    # ── Static parameter-space helpers (inherited by subclasses) ─────────
    # Logic lives in local_utils; kept as @staticmethod for inheritance.

    @staticmethod
    def _sample_z(n: int, dim: int) -> np.ndarray:
        return _sample_z(n, dim)

    @staticmethod
    def _z_to_w(z: np.ndarray, param_info: List[dict]) -> np.ndarray:
        return _z_to_w(z, param_info)

    @staticmethod
    def _w_to_z(w: np.ndarray, param_info: List[dict]) -> np.ndarray:
        return _w_to_z(w, param_info)

    def sample_params_z(self, n: int) -> np.ndarray:
        return self._sample_z(n, self.n_params)

    def z_to_w(self, z: np.ndarray) -> np.ndarray:
        return self._z_to_w(z, self.param_info)

    def w_to_z(self, w: np.ndarray) -> np.ndarray:
        return self._w_to_z(w, self.param_info)

    def simulate_with_params(
        self,
        param_z: np.ndarray,
        n_episodes: int,
        num_cpu: int = 1,
        resimulate_max_num: int = 3,
        save_trajectory: bool = False,
        deterministic: bool = True,
        verbose: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[List[dict]]]:
        """Run *n_episodes* with a fixed parameter vector θ.

        Returns
        -------
        summary_df : DataFrame — **all** simulator output columns, un-normalised
        trajectories : list of dicts or None
        """
        preset, tau = self._z_to_preset_and_tau(param_z)
        preset_list = [deepcopy(preset) for _ in range(n_episodes)]

        if tau is not None and self.use_manifold_flag:
            self.simulator.update_tau(tau)

        self.simulator.clear_records()
        self.simulator.simulate(
            env_preset_list=preset_list,
            num_cpu=num_cpu,
            resimulate_max_num=resimulate_max_num,
            deterministic=deterministic,
            verbose=verbose,
            save_trajectory=False,
            save_coarse_trajectory=save_trajectory,
        )

        summary_df = self.simulator.get_summarized_results()

        trajectories = None
        if save_trajectory:
            trajectories = [
                rec.get_coarse_trajectory_record()
                for rec in self.simulator.simulation_records
            ]

        return summary_df, trajectories

    def simulate_with_param_batch(
        self,
        params_z: np.ndarray,
        n_episodes: int,
        num_cpu: int = 1,
        resimulate_max_num: int = 3,
        deterministic: bool = True,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Run multiple parameter vectors in one simulator call.

        The returned rows are ordered as param 0 episodes, param 1 episodes,
        and so on, so callers can reshape by ``(n_params, n_episodes, n_cols)``.
        """
        params_z = np.asarray(params_z, dtype=np.float32)
        preset_list: list[dict] = []
        tau_list: list[float] = []
        use_tau_list = False

        for z_vec in params_z:
            preset, tau = self._z_to_preset_and_tau(z_vec)
            preset_list.extend(deepcopy(preset) for _ in range(n_episodes))
            if tau is not None and self.use_manifold_flag:
                tau_list.extend([float(tau)] * n_episodes)
                use_tau_list = True

        sim_kwargs = {}
        if use_tau_list:
            sim_kwargs = {"tau_mode": "list", "tau_list": tau_list}

        self.simulator.clear_records()
        self.simulator.simulate(
            env_preset_list=preset_list,
            num_cpu=num_cpu,
            resimulate_max_num=resimulate_max_num,
            deterministic=deterministic,
            verbose=verbose,
            save_trajectory=False,
            save_coarse_trajectory=False,
            **sim_kwargs,
        )
        return self.simulator.get_summarized_results()

    def _z_to_preset_and_tau(
        self,
        z_vec: np.ndarray,
    ) -> Tuple[dict, Optional[float]]:
        """Convert a z-vector to a simulator env-preset dict and tau."""
        w_vec = self.z_to_w(z_vec)
        preset: dict = {}
        tau: Optional[float] = None
        for i, pi in enumerate(self.param_info):
            if pi.get("is_tau", False):
                tau = float(w_vec[i])
            else:
                preset[pi["preset_key"]] = float(w_vec[i])
        return preset, tau


# ─────────────────────────────────────────────────────────────────────────────
# Training Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TrainingDataset:
    """Generates, stores, and samples training data for amortized inference.

    **Raw storage**: summary data is saved as a raw float32 array containing
    only the configured stat features (in order), un-normalised.  Normalisation
    is applied in :meth:`sample` using the feature config, so dataset files
    remain valid after changing feature bounds or feature selection.

    Parameters
    ----------
    config_preset : str
    config : dict, optional
    load_existing : bool
    verbose : bool
    """

    def __init__(
        self,
        config_preset: str = "default",
        config: dict = None,
        load_existing: bool = True,
        verbose: bool = True,
    ):
        self.config = (
            deepcopy(config) if config is not None
            else deepcopy(CFG_AMORTIZED_INFERENCE[config_preset])
        )
        self.config_preset = config_preset
        self._sim: Optional[AmortizedInferenceSimulator] = None

        self.norm_cfg = self.config["normalize"]
        self.stat_feature_cfgs: List[dict] = self.norm_cfg["stat"]["features"]
        self.stat_feature_names: List[str] = [f["name"] for f in self.stat_feature_cfgs]
        self.n_stat_features: int = len(self.stat_feature_cfgs)

        self.use_traj = not self.norm_cfg.get("traj", {}).get("ignore", True)
        self.traj_feature_cfgs: List[dict] = (
            self.norm_cfg["traj"]["features"] if self.use_traj else []
        )

        self.episodes_per_param: int = self.config["dataset"]["train"]["episodes_per_param"]

        hash_str = _dataset_hash(self.config)
        hash_dir = (
            DIR_PROJECT_ROOT / FOLDERS.DATA / FOLDERS.AMORTIZER
            / FOLDERS.AMT_DATASETS / hash_str
        )
        hash_dir.mkdir(parents=True, exist_ok=True)
        _write_dataset_meta(hash_dir, self.config, hash_str)
        self.data_dir = hash_dir / "train"
        self.data_dir.mkdir(exist_ok=True)

        self.n_param_sets: int = 0
        self.dataset: Optional[dict] = None   # {params_z, summary_raw}

        if load_existing:
            self._load_existing(verbose=verbose)

    @property
    def sim(self) -> AmortizedInferenceSimulator:
        if self._sim is None:
            self._sim = AmortizedInferenceSimulator(self.config)
        return self._sim

    # ── Load ──────────────────────────────────────────────────────────────

    def _load_existing(self, verbose: bool = True):
        """Concatenate all previously generated files for this config."""
        pattern = f"*_{self.episodes_per_param}ep.h5"
        file_list = sorted(glob.glob(str(self.data_dir / pattern)))

        if not file_list:
            if verbose:
                print(f"[TrainingDataset] No existing data in {self.data_dir}")
            return

        all_params: List[np.ndarray] = []
        all_summary: List[np.ndarray] = []
        all_columns: Optional[List[str]] = None

        for fpath in (tqdm(file_list, desc="Loading training data") if verbose else file_list):
            data = h5_load(fpath)
            all_params.append(data["params_z"])
            all_summary.append(data["summary_raw"])
            if all_columns is None:
                all_columns = data["all_columns"]

        self.n_param_sets = sum(p.shape[0] for p in all_params)
        summary_raw = np.concatenate(all_summary, axis=0)

        # Feature selection + normalisation — done once on load, not per-sample
        col_idx = [all_columns.index(f["name"]) for f in self.stat_feature_cfgs]
        selected = summary_raw[:, :, col_idx]
        self.dataset = dict(
            params_z=np.concatenate(all_params, axis=0),
            summary=normalize_per_feature(selected, self.stat_feature_cfgs),
        )

        # Trajectory data — load and process once on load if use_traj
        if self.use_traj and self.traj_feature_cfgs:
            traj_blocks: List[np.ndarray] = []
            all_have_traj = True
            for fpath in file_list:
                stem = Path(fpath).stem
                traj_fpath = self.data_dir / f"{stem}_traj.pkl"
                if traj_fpath.exists():
                    raw = pickle_load(str(traj_fpath))
                    traj_blocks.append(_process_traj_dicts(raw, self.traj_feature_cfgs))
                else:
                    all_have_traj = False
                    break
            self.dataset["traj"] = (
                np.concatenate(traj_blocks, axis=0) if all_have_traj and traj_blocks else None
            )
            if not all_have_traj:
                print(
                    "[TrainingDataset] Warning: use_traj=True but some _traj.pkl files are "
                    "missing. Set save_trajectory=True when calling generate()."
                )

        if verbose:
            traj_msg = (
                " + traj" if self.use_traj and self.dataset.get("traj") is not None else ""
            )
            print(
                f"[TrainingDataset] Loaded {self.n_param_sets} param sets "
                f"× {self.episodes_per_param} episodes{traj_msg}"
            )

    # ── Generate ──────────────────────────────────────────────────────────

    def generate(
        self,
        n_param_sets: int = 2 ** 14,
        num_cpu: int = None,
        save_trajectory: bool = False,
        repeat: int = 1,
        verbose: bool = True,
    ):
        """Generate training data.

        Each generation run saves a separate file; multiple files are
        concatenated on load, allowing incremental growth of the dataset.
        """
        if num_cpu is None:
            num_cpu = max(1, (psutil.cpu_count(logical=False) or 1))
        train_cfg = self.config.get("dataset", {}).get("train", {})
        batch_param_sets = max(1, int(train_cfg.get("batch_param_sets", 64)))

        for rep in range(repeat):
            if verbose:
                print(
                    f"[TrainingDataset] Round {rep + 1}/{repeat} — "
                    f"{n_param_sets} param sets × {self.episodes_per_param} episodes"
                )

            params_z = self.sim.sample_params_z(n_param_sets)
            summary_rows: List[np.ndarray] = []
            all_columns: Optional[List[str]] = None
            traj_all: Optional[list] = [] if save_trajectory else None

            if save_trajectory:
                for idx in (tqdm(range(n_param_sets), desc="Simulating") if verbose else range(n_param_sets)):
                    summary_df, trajs = self.sim.simulate_with_params(
                        param_z=params_z[idx],
                        n_episodes=self.episodes_per_param,
                        num_cpu=num_cpu,
                        save_trajectory=save_trajectory,
                    )
                    if all_columns is None:
                        all_columns = list(summary_df.columns)
                    summary_rows.append(summary_df.to_numpy(dtype=np.float32))

                    if trajs is not None:
                        traj_all.append(trajs)

                summary_raw = np.stack(summary_rows, axis=0)  # (n_params, n_eps, n_all_cols)
            else:
                batch_starts = range(0, n_param_sets, batch_param_sets)
                iterator = tqdm(batch_starts, desc="Simulating batches") if verbose else batch_starts
                for start in iterator:
                    end = min(start + batch_param_sets, n_param_sets)
                    n_batch = end - start
                    summary_df = self.sim.simulate_with_param_batch(
                        params_z=params_z[start:end],
                        n_episodes=self.episodes_per_param,
                        num_cpu=num_cpu,
                    )
                    if all_columns is None:
                        all_columns = list(summary_df.columns)
                    arr = summary_df[all_columns].to_numpy(dtype=np.float32)
                    expected = n_batch * self.episodes_per_param
                    if arr.shape[0] != expected:
                        raise RuntimeError(
                            f"Expected {expected} simulation rows for batch "
                            f"{start}:{end}, got {arr.shape[0]}."
                        )
                    summary_rows.append(
                        arr.reshape(n_batch, self.episodes_per_param, len(all_columns))
                    )

                summary_raw = np.concatenate(summary_rows, axis=0)
            timestamp = get_compact_timestamp_str(omit_year=True)
            filename = f"{timestamp}_{n_param_sets}p_{self.episodes_per_param}ep.h5"
            h5_save(
                self.data_dir / filename,
                arrays={"params_z": params_z, "summary_raw": summary_raw},
                attrs={
                    "all_columns": all_columns,
                    "param_names": self.sim.param_names,
                    "n_param_sets": n_param_sets,
                    "episodes_per_param": self.episodes_per_param,
                    "timestamp": timestamp,
                },
            )
            if verbose:
                print(f"  → Saved {self.data_dir / filename}")

            if save_trajectory and traj_all:
                traj_fn = f"{timestamp}_{n_param_sets}p_{self.episodes_per_param}ep_traj.pkl"
                pickle_save(str(self.data_dir / traj_fn), traj_all)
                if verbose:
                    print(f"  → Saved {self.data_dir / traj_fn}")

        self._load_existing(verbose=verbose)

    # ── Sampling ──────────────────────────────────────────────────────────

    def sample(
        self,
        batch_size: int,
        sim_per_param: int = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Sample a random batch for one training step.

        Returns
        -------
        params_z   : ndarray (batch_size, n_params)
        stat_batch : ndarray (batch_size, sim_per_param, n_stat_features) — normalised [-1,1]
        traj_batch : ndarray (batch_size, sim_per_param) of object, or None
            Each element is (n_steps, n_traj_features) float32, normalised [-1, 1].
        """
        assert self.dataset is not None, "No data loaded. Call generate() first."

        if sim_per_param is None:
            sim_per_param = self.episodes_per_param

        param_idx = np.random.choice(
            self.n_param_sets, batch_size, replace=(batch_size > self.n_param_sets)
        )
        ep_idx = np.random.choice(self.episodes_per_param, sim_per_param, replace=False)

        params_z = self.dataset["params_z"][param_idx]
        stat_batch = self.dataset["summary"][param_idx][:, ep_idx, :]

        traj_batch = None
        if self.use_traj and self.dataset.get("traj") is not None:
            traj_batch = self.dataset["traj"][param_idx][:, ep_idx]

        return params_z, stat_batch, traj_batch


# ─────────────────────────────────────────────────────────────────────────────
# Validation Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ValidationDataset:
    """Generates and manages validation data for amortized inference.

    Stores raw (un-normalised) summary statistics; normalisation applied in
    :meth:`sample`.
    """

    def __init__(
        self,
        config_preset: str = "default",
        config: dict = None,
        load_existing: bool = True,
        verbose: bool = True,
    ):
        self.config = (
            deepcopy(config) if config is not None
            else deepcopy(CFG_AMORTIZED_INFERENCE[config_preset])
        )
        self.config_preset = config_preset
        self._sim: Optional[AmortizedInferenceSimulator] = None

        self.norm_cfg = self.config["normalize"]
        self.stat_feature_cfgs: List[dict] = self.norm_cfg["stat"]["features"]
        self.stat_feature_names: List[str] = [f["name"] for f in self.stat_feature_cfgs]
        self.n_stat_features: int = len(self.stat_feature_cfgs)

        self.use_traj = not self.norm_cfg.get("traj", {}).get("ignore", True)
        self.traj_feature_cfgs: List[dict] = (
            self.norm_cfg["traj"]["features"] if self.use_traj else []
        )

        valid_cfg = self.config["dataset"]["valid"]
        self.total_users: int = valid_cfg["total_users"]
        self.trials_per_user: int = valid_cfg["trials_per_user"]

        hash_str = _dataset_hash(self.config)
        hash_dir = (
            DIR_PROJECT_ROOT / FOLDERS.DATA / FOLDERS.AMORTIZER
            / FOLDERS.AMT_DATASETS / hash_str
        )
        hash_dir.mkdir(parents=True, exist_ok=True)
        _write_dataset_meta(hash_dir, self.config, hash_str)
        self.data_dir = hash_dir / "valid"
        self.data_dir.mkdir(exist_ok=True)

        self.dataset: Optional[dict] = None

        if load_existing:
            self._load_existing(verbose=verbose)

    @property
    def sim(self) -> AmortizedInferenceSimulator:
        if self._sim is None:
            self._sim = AmortizedInferenceSimulator(self.config)
        return self._sim

    # ── Load ──────────────────────────────────────────────────────────────

    def _load_existing(self, verbose: bool = True):
        pattern = f"*_{self.total_users}u_{self.trials_per_user}t.h5"
        file_list = sorted(glob.glob(str(self.data_dir / pattern)))

        if not file_list:
            if verbose:
                print(f"[ValidationDataset] No existing data in {self.data_dir}")
            return

        fpath = file_list[-1]
        data = h5_load(fpath)

        # Feature selection + normalisation — done once on load
        col_idx = [data["all_columns"].index(f["name"]) for f in self.stat_feature_cfgs]
        selected = data["summary_raw"][:, :, col_idx]
        self.dataset = {
            **data,
            "summary": normalize_per_feature(selected, self.stat_feature_cfgs),
        }

        # Trajectory data
        if self.use_traj and self.traj_feature_cfgs:
            stem = Path(fpath).stem
            traj_fpath = self.data_dir / f"{stem}_traj.pkl"
            if traj_fpath.exists():
                raw = pickle_load(str(traj_fpath))
                self.dataset["traj"] = _process_traj_dicts(raw, self.traj_feature_cfgs)
            else:
                self.dataset["traj"] = None
                print(
                    "[ValidationDataset] Warning: use_traj=True but _traj.pkl not found. "
                    "Set save_trajectory=True when calling generate()."
                )

        if verbose:
            traj_msg = (
                " + traj" if self.use_traj and self.dataset.get("traj") is not None else ""
            )
            print(
                f"[ValidationDataset] Loaded {data['n_users']} users "
                f"× {data['trials_per_user']} trials from {Path(fpath).name}{traj_msg}"
            )

    # ── Generate ──────────────────────────────────────────────────────────

    def generate(
        self,
        num_cpu: int = None,
        save_trajectory: bool = False,
        verbose: bool = True,
    ):
        """Generate validation data (raw, un-normalised)."""
        if num_cpu is None:
            num_cpu = max(1, (psutil.cpu_count(logical=False) or 1))

        params_z = self.sim.sample_params_z(self.total_users)
        summary_rows: List[np.ndarray] = []
        all_columns: Optional[List[str]] = None
        traj_all: Optional[list] = [] if save_trajectory else None

        for u in (tqdm(range(self.total_users), desc="Valid users") if verbose else range(self.total_users)):
            summary_df, trajs = self.sim.simulate_with_params(
                param_z=params_z[u],
                n_episodes=self.trials_per_user,
                num_cpu=num_cpu,
                save_trajectory=save_trajectory,
            )
            if all_columns is None:
                all_columns = list(summary_df.columns)
            summary_rows.append(summary_df.to_numpy(dtype=np.float32))

            if save_trajectory and trajs is not None:
                traj_all.append(trajs)

        summary_raw = np.stack(summary_rows, axis=0)  # (n_users, n_trials, n_all_cols)
        timestamp = get_compact_timestamp_str(omit_year=True)
        filename = f"{timestamp}_{self.total_users}u_{self.trials_per_user}t.h5"
        h5_save(
            self.data_dir / filename,
            arrays={"params_z": params_z, "summary_raw": summary_raw},
            attrs={
                "all_columns": all_columns,
                "param_names": self.sim.param_names,
                "n_users": self.total_users,
                "trials_per_user": self.trials_per_user,
                "timestamp": timestamp,
            },
        )
        if verbose:
            print(f"  → Saved {self.data_dir / filename}")

        if save_trajectory and traj_all:
            traj_fn = f"{timestamp}_{self.total_users}u_{self.trials_per_user}t_traj.pkl"
            pickle_save(str(self.data_dir / traj_fn), traj_all)

        self._load_existing(verbose=verbose)

    # ── Sampling ──────────────────────────────────────────────────────────

    def sample(
        self,
        n_trials: int,
        n_users: int = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Sample validation data.

        Returns
        -------
        params_z   : ndarray (n_users, n_params)
        stat_batch : ndarray (n_users, n_trials, n_stat_features) — normalised
        traj_batch : ndarray (n_users, n_trials) of object, or None
            Each element is (n_steps, n_traj_features) float32, normalised [-1, 1].
        """
        assert self.dataset is not None, "No validation data loaded."
        if n_users is None:
            n_users = self.dataset["n_users"]

        user_idx = np.random.choice(
            self.dataset["n_users"], n_users, replace=(n_users > self.dataset["n_users"])
        )
        trial_idx = np.random.choice(self.trials_per_user, n_trials, replace=False)

        params_z = self.dataset["params_z"][user_idx]
        stat_batch = self.dataset["summary"][user_idx][:, trial_idx, :]

        traj_batch = None
        if self.use_traj and self.dataset.get("traj") is not None:
            traj_batch = self.dataset["traj"][user_idx][:, trial_idx]

        return params_z, stat_batch, traj_batch



# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config_preset = "ijhcs_gaze_abl"
    # ── Training Dataset ──────────────────────────────────────────────────
    ds = TrainingDataset(config_preset=config_preset, load_existing=False, verbose=True)
    ds.generate(n_param_sets=10, num_cpu=8, repeat=1, verbose=True)

    p, s, _ = ds.sample(batch_size=4)
    print(f"params_z shape: {p.shape}")
    print(f"stat_batch shape: {s.shape}")

    # ── Validation Dataset ────────────────────────────────────────────────
    vds = ValidationDataset(config_preset=config_preset, load_existing=False, verbose=True)
    vds.generate(num_cpu=8, verbose=True)

    p, s, t = vds.sample(n_trials=10)
    print(f"params_z shape:  {p.shape}")
    print(f"stat_batch shape:{s.shape}")
    print(f"traj_batch:      {t.shape if t is not None else None}")
