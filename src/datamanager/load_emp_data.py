"""
Load processed empirical experiment data.

Provides a simple API to load summary CSVs and trajectory HDF5 files
for one or more users.

Usage
-----
    from src.datamanager.load_emp_data import EmpDataLoader, load_summary

    # Standalone loaders
    df   = load_summary("yjstest_rh")
    traj = load_trajectory("yjstest_rh", kind="coarse")

    # Pre-loaded class with filtering
    loader = EmpDataLoader(users=["yjstest_rh"], traj_kind="coarse")
    summary, trajs = loader.load(session_name="session_000")

Written by June-Seop Yoon
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import pandas as pd
import yaml

from ..configs.constants import FOLDERS
from ..agent.preset_converter import ijhcs_summary_to_env_presets
from .process_emp_data import load_trajectories_h5, _TRAJ_H5_NAME

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_IJHCS_DIR = _PROJECT_ROOT / "empirical"

# Valid trajectory kinds
TrajKind = Literal["raw", "coarse"]
IJHCSTrajHz = Literal[60, 20]


class IJHCSExpDataLoader:
    """Loader for the processed IJHCS aim-and-shoot user-study dataset.

    The processed files are expected directly under ``empirical/ijhcs_exp``:
    ``summary.csv`` and ``trajectories_<hz>hz.h5``.

    By default, :meth:`load` returns analysis-ready trials: truncated trials,
    TCT/shoot-error outliers, insufficient gaze trials, non-close initial gaze
    trials, and rows with missing reaction times are filtered out. Pass
    ``filtered=False`` to get the raw processed summary rows.
    """

    def __init__(
        self,
        *,
        traj_hz: IJHCSTrajHz = 60,
        load_traj: bool = True,
        root: Path | None = None,
        filtered: bool = True,
        exclude_truncated: bool = True,
        exclude_outliers: bool = True,
        require_sufficient_gaze: bool = True,
        require_close_gaze: bool = True,
        require_reaction_times: bool = True,
    ) -> None:
        self.root = _IJHCS_DIR if root is None else Path(root)
        self.traj_hz = int(traj_hz)
        self.filtered = bool(filtered)
        self.default_filter_options = {
            "exclude_truncated": bool(exclude_truncated),
            "exclude_outliers": bool(exclude_outliers),
            "require_sufficient_gaze": bool(require_sufficient_gaze),
            "require_close_gaze": bool(require_close_gaze),
            "require_reaction_times": bool(require_reaction_times),
        }
        self.summary = load_ijhcs_summary(root=self.root)
        self.trajectory = IJHCSTrajectoryStore(self.root / f"trajectories_{self.traj_hz}hz.h5") if load_traj else None

    def load(
        self,
        *,
        return_traj: bool = True,
        filtered: bool | None = None,
        exclude_truncated: bool | None = None,
        exclude_outliers: bool | None = None,
        require_sufficient_gaze: bool | None = None,
        require_close_gaze: bool | None = None,
        require_reaction_times: bool | None = None,
        **filters,
    ) -> tuple[pd.DataFrame, list[pd.DataFrame]] | pd.DataFrame:
        if self.summary.empty:
            return (self.summary.copy(), []) if return_traj else self.summary.copy()

        options = self.default_filter_options.copy()
        overrides = {
            "exclude_truncated": exclude_truncated,
            "exclude_outliers": exclude_outliers,
            "require_sufficient_gaze": require_sufficient_gaze,
            "require_close_gaze": require_close_gaze,
            "require_reaction_times": require_reaction_times,
        }
        options.update({key: bool(val) for key, val in overrides.items() if val is not None})
        use_default_filters = self.filtered if filtered is None else bool(filtered)

        mask = _build_mask(self.summary, filters)
        if use_default_filters:
            mask &= _build_ijhcs_quality_mask(self.summary, **options)
        sub = self.summary.loc[mask].reset_index(drop=True)

        if not return_traj:
            return sub
        if self.trajectory is None:
            raise RuntimeError("IJHCSExpDataLoader was constructed with load_traj=False.")
        trajs = self.trajectory.load_many(sub["trajectory_key"].astype(str).to_list())
        return sub, trajs

    def load_presets(
        self,
        *,
        base_preset: dict | None = None,
        return_summary: bool = False,
        filtered: bool | None = None,
        exclude_truncated: bool | None = None,
        exclude_outliers: bool | None = None,
        require_sufficient_gaze: bool | None = None,
        require_close_gaze: bool | None = None,
        require_reaction_times: bool | None = None,
        **filters,
    ) -> list[dict] | tuple[pd.DataFrame, list[dict]]:
        """Return simulator presets parsed from matching IJHCS summary rows."""
        summary = self.load(
            return_traj=False,
            filtered=filtered,
            exclude_truncated=exclude_truncated,
            exclude_outliers=exclude_outliers,
            require_sufficient_gaze=require_sufficient_gaze,
            require_close_gaze=require_close_gaze,
            require_reaction_times=require_reaction_times,
            **filters,
        )
        presets = ijhcs_summary_to_env_presets(summary, base_preset=base_preset)
        if return_summary:
            return summary, presets
        return presets

    def __len__(self) -> int:
        return len(self.summary)

    def __repr__(self) -> str:
        return (
            f"IJHCSExpDataLoader(trials={len(self.summary)}, traj_hz={self.traj_hz}, "
            f"filtered={self.filtered})"
        )


class IJHCSTrajectoryStore:
    """Lazy reader for ``trajectories_<hz>hz.h5`` files."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.columns: list[str] = []
        self.keys: list[str] = []
        self.offsets = []
        self.key_to_index: dict[str, int] = {}
        self.sampling_hz: int | None = None
        self._load_index()

    def _load_index(self) -> None:
        import h5py

        if not self.path.exists():
            return
        with h5py.File(str(self.path), "r") as f:
            self.sampling_hz = int(f.attrs.get("sampling_hz", 0))
            self.columns = [_decode_h5_string(x) for x in f["columns"][...]]
            self.keys = [_decode_h5_string(x) for x in f["trajectory_key"][...]]
            self.offsets = f["offsets"][...].astype(int)
        self.key_to_index = {key: i for i, key in enumerate(self.keys)}

    def load(self, key: str) -> pd.DataFrame:
        return self.load_many([key])[0]

    def load_many(self, keys: Sequence[str]) -> list[pd.DataFrame]:
        import h5py

        if not self.path.exists():
            return [pd.DataFrame() for _ in keys]
        out: list[pd.DataFrame] = []
        with h5py.File(str(self.path), "r") as f:
            data = f["data"]
            for key in keys:
                idx = self.key_to_index.get(str(key))
                if idx is None:
                    out.append(pd.DataFrame())
                    continue
                start = int(self.offsets[idx])
                end = int(self.offsets[idx + 1])
                out.append(pd.DataFrame(data[start:end], columns=self.columns))
        return out

    def load_all(self) -> dict[str, pd.DataFrame]:
        return dict(zip(self.keys, self.load_many(self.keys)))

    def __len__(self) -> int:
        return len(self.keys)



# =====================================================================
# Public loaders
# =====================================================================


def load_ijhcs_summary(root: Path | None = None) -> pd.DataFrame:
    """Load ``empirical/summary.csv``."""
    data_dir = _IJHCS_DIR if root is None else Path(root)
    path = data_dir / "summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_ijhcs_saccade_summary(root: Path | None = None) -> pd.DataFrame:
    """Load ``empirical/saccade_summary.csv``."""
    data_dir = _IJHCS_DIR if root is None else Path(root)
    path = data_dir / "saccade_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_ijhcs_trajectory(
    hz: IJHCSTrajHz = 240,
    *,
    keys: Sequence[str] | None = None,
    root: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load IJHCS trajectories at a preprocessed sampling frequency."""
    data_dir = _IJHCS_DIR if root is None else Path(root)
    store = IJHCSTrajectoryStore(data_dir / f"trajectories_{int(hz)}hz.h5")
    keys_to_load = store.keys if keys is None else [str(k) for k in keys]
    return dict(zip(keys_to_load, store.load_many(keys_to_load)))

# =====================================================================
# Private helpers
# =====================================================================


def _decode_h5_string(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _build_mask(df: pd.DataFrame, filters: dict) -> pd.Series:
    """Combine per-column equality / isin filters into a boolean mask."""
    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        if col not in df.columns:
            raise KeyError(
                f"EmpDataLoader.load: column '{col}' not in summary "
                f"(available: {list(df.columns)})"
            )
        if isinstance(val, (list, tuple, set, pd.Series)) or (
            hasattr(val, "__iter__") and not isinstance(val, (str, bytes))
        ):
            mask &= df[col].isin(list(val))
        else:
            mask &= df[col] == val
    return mask


def _build_ijhcs_quality_mask(
    df: pd.DataFrame,
    *,
    exclude_truncated: bool = True,
    exclude_outliers: bool = True,
    require_sufficient_gaze: bool = True,
    require_close_gaze: bool = True,
    require_reaction_times: bool = True,
) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if exclude_truncated and "truncated" in df.columns:
        mask &= df["truncated"].fillna(0).astype(int) == 0
    if exclude_outliers:
        for col in ("outlier_tct_flag", "outlier_shoot_error"):
            if col in df.columns:
                mask &= df[col].fillna(0).astype(int) == 0
    if require_sufficient_gaze and "sufficient_gaze" in df.columns:
        mask &= df["sufficient_gaze"].fillna(0).astype(int) == 1
    if require_close_gaze and "close_gaze_init_pos" in df.columns:
        mask &= df["close_gaze_init_pos"].fillna(0).astype(int) == 1
    if require_reaction_times:
        for col in ("gaze_reaction_time", "hand_reaction_time"):
            if col in df.columns:
                mask &= df[col].notna()
    return mask


def _show_random_ijhcs_target_gaze_trials(n_trials: int = 6, traj_hz: IJHCSTrajHz = 20) -> None:
    import matplotlib.pyplot as plt

    from ..agent.task import DEFAULT_TASK_CONFIG

    loader = IJHCSExpDataLoader(traj_hz=traj_hz)
    summary = loader.load(return_traj=False)
    if summary.empty:
        print("No IJHCS trials available after default quality filtering.")
        return

    sampled = summary.sample(n=min(int(n_trials), len(summary))).reset_index(drop=True)
    _, trajectories = loader.load(
        trajectory_key=sampled["trajectory_key"].astype(str).to_list(),
    )

    width_mm = float(DEFAULT_TASK_CONFIG["monitor_width_mm"])
    height_mm = float(DEFAULT_TASK_CONFIG["monitor_height_mm"])
    xlim = (-width_mm / 2.0, width_mm / 2.0)
    ylim = (-height_mm / 2.0, height_mm / 2.0)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    axes = axes.ravel()
    for ax, (_, row), traj in zip(axes, sampled.iterrows(), trajectories):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.axhline(0.0, color="0.85", linewidth=0.8)
        ax.axvline(0.0, color="0.85", linewidth=0.8)

        if not traj.empty:
            ax.plot(traj["target_x_mm"], traj["target_y_mm"], color="tab:blue", linewidth=1.8, label="target")
            ax.scatter(
                traj["target_x_mm"].iloc[0],
                traj["target_y_mm"].iloc[0],
                color="tab:blue",
                s=24,
                marker="o",
                zorder=3,
            )
            ax.scatter(
                traj["target_x_mm"].iloc[-1],
                traj["target_y_mm"].iloc[-1],
                color="tab:blue",
                s=34,
                marker="x",
                zorder=3,
            )

            valid = traj["gaze_valid"].to_numpy(dtype=float) >= 0.5
            ax.plot(traj["gaze_x_mm"], traj["gaze_y_mm"], color="tab:red", alpha=0.45, linewidth=1.0, label="gaze")
            if valid.any():
                ax.scatter(
                    traj.loc[valid, "gaze_x_mm"],
                    traj.loc[valid, "gaze_y_mm"],
                    color="tab:red",
                    s=8,
                    alpha=0.75,
                )
            if (~valid).any():
                ax.scatter(
                    traj.loc[~valid, "gaze_x_mm"],
                    traj.loc[~valid, "gaze_y_mm"],
                    color="tab:orange",
                    s=8,
                    alpha=0.35,
                )

        ax.set_title(
            f"p{int(row['player_index']):02d} {row['sensitivity_mode']} "
            f"{row['target_name']} b{int(row['block_index'])} t{int(row['trial_index'])}"
        )
        ax.set_xlabel("monitor x (mm)")
        ax.set_ylabel("monitor y (mm)")

    for ax in axes[len(sampled):]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    plt.show()


if __name__ == "__main__":
    _show_random_ijhcs_target_gaze_trials()
