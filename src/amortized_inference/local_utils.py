"""
Shared low-level utilities for the amortized inference dataset pipeline.

Centralises helpers that are used by both  dataset.py  and
dataset_hierarchical.py  to avoid duplication and keep the dataset modules
focused on their class logic.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
from typing import List, Optional, Tuple, Union

from ..utils.myutils import config_hash, get_compact_timestamp_str
from ..utils.mymath import (
    linear_normalize,
    linear_denormalize,
    log_normalize,
    log_denormalize,
)

# Project root: <repo>/src/inference/local_utils.py  →  <repo>
DIR_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize_per_feature(
    arr: np.ndarray,
    feature_cfgs: List[dict],
) -> np.ndarray:
    """Map a raw feature array (..., n_feat) to [-1, 1] per-feature.

    Parameters
    ----------
    arr : ndarray of shape (..., n_feat)
    feature_cfgs : list of dicts, each with keys {name, min, max}.

    Returns
    -------
    ndarray of same shape, dtype float32.
    """
    arr = np.asarray(arr, dtype=np.float32)
    lo = np.array([float(fc["min"]) for fc in feature_cfgs], dtype=np.float32)
    hi = np.array([float(fc["max"]) for fc in feature_cfgs], dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo) * 2.0 - 1.0, -1.0, 1.0).astype(np.float32)


def denormalize_per_feature(
    arr_norm: np.ndarray,
    feature_cfgs: List[dict],
) -> np.ndarray:
    """Inverse of :func:`normalize_per_feature`."""
    arr_norm = np.asarray(arr_norm, dtype=np.float32)
    out = np.empty_like(arr_norm)
    for i, fc in enumerate(feature_cfgs):
        lo, hi = float(fc["min"]), float(fc["max"])
        out[..., i] = linear_denormalize(arr_norm[..., i], lo, hi)
    return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory helpers
# ─────────────────────────────────────────────────────────────────────────────

def _process_traj_dicts(
    raw: List[List[dict]],
    traj_feature_cfgs: List[dict],
) -> np.ndarray:
    """Convert raw trajectory dicts to a 2-D object array of normalised arrays.

    Parameters
    ----------
    raw : list (n_params) of list (n_episodes) of trajectory dicts
        Each dict is DataFrame-convertible; column names identical across all dicts.
    traj_feature_cfgs : list of {name, min, max}

    Returns
    -------
    ndarray of shape (n_params, n_episodes), dtype=object
        Each element is float32 array (n_steps, n_traj_features), normalised [-1, 1].
    """
    n_params = len(raw)
    n_eps = len(raw[0]) if n_params > 0 else 0
    out = np.empty((n_params, n_eps), dtype=object)

    col_names: Optional[List[str]] = None
    col_idx: Optional[List[int]] = None

    for i, ep_list in enumerate(raw):
        for j, traj_dict in enumerate(ep_list):
            traj_df = pd.DataFrame(traj_dict)
            if col_names is None:
                col_names = traj_df.columns.tolist()
                col_idx = [col_names.index(f["name"]) for f in traj_feature_cfgs]
            traj_arr = traj_df.to_numpy(dtype=np.float32)    # (n_steps, n_all_cols)
            out[i, j] = normalize_per_feature(traj_arr[:, col_idx], traj_feature_cfgs)

    return out


def _process_traj_dicts_hierarchical(
    raw: List,
    traj_feature_cfgs: List[dict],
) -> np.ndarray:
    """Convert hierarchical raw trajectory dicts → 3-D object array.

    Parameters
    ----------
    raw : list (n_params) of list (n_groups) of list (n_episodes) of traj dicts
    traj_feature_cfgs : list of {name, min, max}

    Returns
    -------
    ndarray of shape (n_params, n_groups, n_episodes), dtype=object
        Each element is (n_steps, n_traj_features) float32, normalised [-1, 1].
    """
    n_params = len(raw)
    n_groups = len(raw[0]) if n_params > 0 else 0
    n_eps    = len(raw[0][0]) if n_groups > 0 else 0
    out = np.empty((n_params, n_groups, n_eps), dtype=object)

    col_names: Optional[List[str]] = None
    col_idx: Optional[List[int]] = None

    for i, group_list in enumerate(raw):
        for g, ep_list in enumerate(group_list):
            for j, traj_dict in enumerate(ep_list):
                traj_df = pd.DataFrame(traj_dict)
                if col_names is None:
                    col_names = traj_df.columns.tolist()
                    col_idx = [col_names.index(f["name"]) for f in traj_feature_cfgs]
                traj_arr = traj_df.to_numpy(dtype=np.float32)
                out[i, g, j] = normalize_per_feature(traj_arr[:, col_idx], traj_feature_cfgs)

    return out


def _dense_traj_from_dicts_hierarchical(
    raw,
    pad_to=None,
):
    """Convert hierarchical raw traj dicts -> dense padded float32 arrays (ALL features, raw).

    Stores every column from the simulator, unnormalised.  Feature selection
    and normalisation are applied at *load time* by
    _select_and_normalize_traj, mirroring how summary_raw is handled
    for stat data: the on-disk file is feature-agnostic, so the config's
    traj.features list can be changed without re-generating data.

    Parameters
    ----------
    raw : list (n_params) of list (n_groups) of list (n_episodes) of traj dicts
        Each dict is DataFrame-convertible.
    pad_to : int, optional
        Fixed output T dimension.  If set, trajectories longer than pad_to
        are truncated, and the output always has shape (..., pad_to, n_feat_all).
        If None, T_max is derived from the longest trajectory in raw.

    Returns
    -------
    traj_raw     : float32  (n_params, n_groups, n_eps, T_max, n_feat_all)
        Zero-padded, unnormalised (raw simulator output).
    traj_lengths : int32    (n_params, n_groups, n_eps)
        True (unpadded) length of each trajectory, capped at T_max.
    T_max        : int  (equals pad_to when supplied)
    all_traj_columns : list of str
        Column names for the last axis of traj_raw.
    """
    import numpy as np
    import pandas as pd

    n_params = len(raw)
    n_groups = len(raw[0]) if n_params > 0 else 0
    n_eps    = len(raw[0][0]) if n_groups > 0 else 0
    all_traj_columns = None

    # Pass 1: convert to numpy arrays (all columns, raw) and find longest
    arrs = np.empty((n_params, n_groups, n_eps), dtype=object)
    T_max_data = 0
    for i, group_list in enumerate(raw):
        for g, ep_list in enumerate(group_list):
            for j, traj_dict in enumerate(ep_list):
                df = pd.DataFrame(traj_dict)
                if all_traj_columns is None:
                    all_traj_columns = df.columns.tolist()
                arr = df.to_numpy(dtype=np.float32)   # (T, F_all) -- raw, all columns
                arrs[i, g, j] = arr
                if arr.shape[0] > T_max_data:
                    T_max_data = arr.shape[0]

    # Use config-specified length when provided; otherwise data-driven
    T_max = pad_to if pad_to is not None else T_max_data
    all_traj_columns = all_traj_columns or []
    n_feat_all = len(all_traj_columns)

    # Pass 2: pack into a single dense array (truncate if arr.shape[0] > T_max)
    traj_raw     = np.zeros((n_params, n_groups, n_eps, T_max, n_feat_all), dtype=np.float32)
    traj_lengths = np.zeros((n_params, n_groups, n_eps), dtype=np.int32)
    for i in range(n_params):
        for g in range(n_groups):
            for j in range(n_eps):
                arr = arrs[i, g, j]
                T   = min(arr.shape[0], T_max)
                traj_raw[i, g, j, :T] = arr[:T]
                traj_lengths[i, g, j] = T

    return traj_raw, traj_lengths, T_max, all_traj_columns

def _obj_array_to_dense(
    traj_obj: np.ndarray,
    pad_to: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Convert an object array of variable-length traj arrays -> dense padded float32.

    Used to migrate legacy ``*_traj.pkl`` data (loaded via
    :func:`_process_traj_dicts_hierarchical`) into the dense HDF5 format
    without re-running simulations.

    Parameters
    ----------
    traj_obj : ndarray  (*shape)  dtype=object
        Each element is a float32 array of shape (T, F).
    pad_to : int, optional
        Fixed output T dimension.  Trajectories longer than *pad_to* are
        truncated.  If ``None``, T_max is derived from the longest element.

    Returns
    -------
    traj_dense   : float32  (*shape, T_max, F)
    traj_lengths : int32    (*shape)
    T_max        : int  (equals *pad_to* when supplied)
    """
    flat = [arr for arr in traj_obj.ravel() if arr is not None]
    if not flat:
        return (
            np.zeros((*traj_obj.shape, 0, 0), dtype=np.float32),
            np.zeros(traj_obj.shape, dtype=np.int32),
            0,
        )
    T_max_data = int(max(a.shape[0] for a in flat))
    T_max  = pad_to if pad_to is not None else T_max_data
    n_feat = int(flat[0].shape[1])

    traj_dense   = np.zeros((*traj_obj.shape, T_max, n_feat), dtype=np.float32)
    traj_lengths = np.zeros(traj_obj.shape, dtype=np.int32)
    for idx in np.ndindex(*traj_obj.shape):
        arr = traj_obj[idx]
        if arr is not None:
            T = min(arr.shape[0], T_max)    # cap at T_max (truncate long trajs)
            traj_dense[idx + (slice(T), slice(None))] = arr[:T]
            traj_lengths[idx] = T

    return traj_dense, traj_lengths, T_max


def _select_and_normalize_traj(
    traj_raw: np.ndarray,
    traj_lengths: np.ndarray,
    all_traj_columns: List[str],
    traj_feature_cfgs: List[dict],
) -> np.ndarray:
    """Select configured traj features, normalise, and re-zero padding positions.

    Called at *load time* so the on-disk ``traj_raw`` array remains
    feature-agnostic (analogous to ``summary_raw`` for stat data).

    Parameters
    ----------
    traj_raw     : float32  (..., T_max, F_all) — raw, unnormalised
    traj_lengths : int32    (...) — true lengths (padding starts at index L)
    all_traj_columns : column names for the last axis of *traj_raw*
    traj_feature_cfgs : list of {name, min, max} — features to select

    Returns
    -------
    float32  (..., T_max, F_cfg) — normalised to [-1, 1], padded with 0.0
    """
    col_idx    = [all_traj_columns.index(f["name"]) for f in traj_feature_cfgs]
    selected   = traj_raw[..., col_idx]                               # (..., T, F_cfg)
    normalized = normalize_per_feature(selected, traj_feature_cfgs)  # (..., T, F_cfg)
    # Raw padding zeros normalise to non-zero for offset features (e.g. dt_ms).
    # Re-zero all padded positions so the downstream model sees consistent 0.0 padding.
    T    = normalized.shape[-2]
    mask = np.arange(T) >= traj_lengths[..., None]                    # (..., T) broadcast
    normalized[mask] = 0.0
    return normalized


# ─────────────────────────────────────────────────────────────────────────────
# Parameter-space helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sample_z(n: int, dim: int) -> np.ndarray:
    """Sample *n* parameter vectors uniformly in z-space [-1, 1]^dim."""
    return np.random.uniform(-1, 1, size=(n, dim)).astype(np.float32)


def _z_to_w(z: np.ndarray, param_info: List[dict]) -> np.ndarray:
    """Denormalise a z-space vector to physical (w) space."""
    z = np.asarray(z, dtype=np.float32)
    w = np.empty_like(z)
    for i, pi in enumerate(param_info):
        if pi["type"] == "loguniform":
            w[..., i] = log_denormalize(z[..., i], pi["min"], pi["max"], scale=pi["scale"])
        else:
            w[..., i] = linear_denormalize(z[..., i], pi["min"], pi["max"])
    return w


def _w_to_z(w: np.ndarray, param_info: List[dict]) -> np.ndarray:
    """Normalise a physical (w) space vector to z-space."""
    w = np.asarray(w, dtype=np.float32)
    z = np.empty_like(w)
    for i, pi in enumerate(param_info):
        if pi["type"] == "loguniform":
            z[..., i] = log_normalize(w[..., i], pi["min"], pi["max"], scale=pi["scale"])
        else:
            z[..., i] = linear_normalize(w[..., i], pi["min"], pi["max"])
    return z


def _build_param_info(infer_params: List[dict]) -> List[dict]:
    """Build a list of per-parameter metadata dicts from explicit full-spec entries.

    Each entry in *infer_params* must have:
        name, min, max
    Optional fields (with defaults):
        preset_key    (default: name)      — key used in the simulator env-preset dict
        output_column (default: name)      — column name in the results DataFrame
        type          (default: "linear")  — "linear" or "loguniform"
        scale         (default: 1.0)       — scaling factor for loguniform mapping
        is_tau        (default: false)     — marks the manifold interpolation parameter

    Returns a list of dicts with keys:
        name, preset_key, output_column, min, max, type, scale, is_tau
    """
    info_list: List[dict] = []
    for entry in infer_params:
        name = entry["name"]
        t = entry.get("type", "linear")
        if t == "uniform":
            t = "linear"   # normalise legacy alias
        info_list.append(dict(
            name=name,
            preset_key=entry.get("preset_key", name),
            output_column=entry.get("output_column", name),
            min=float(entry["min"]),
            max=float(entry["max"]),
            type=t,
            scale=float(entry.get("scale", 1.0)),
            is_tau=bool(entry.get("is_tau", False)),
            is_tau_surface=bool(entry.get("is_tau_surface", False)),
        ))
    return info_list


# ─────────────────────────────────────────────────────────────────────────────
# Dataset identity helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dataset_hash(config: dict) -> str:
    """Deterministic hash of the data-defining config parts (excludes training hyper-params).

    The following sections contribute to the hash:
    - ``simulator``         — proxy / SAC model settings
    - ``infer_params``      — parameter names, ranges, types (incl. is_tau_surface)
    - ``episodes_per_param``
    - ``task_condition_mapper`` (top-level, non-hierarchical) — mapper class,
      thresholds, clip values.  Different mappers → different data → different hash.
    - ``hierarchical``      — global/local splits + hierarchical mapper (when enabled).
      Non-hierarchical configs never set ``hierarchical.enabled=true``, so this
      key is absent and does not collide with hierarchical hashes.
    - ``monotone_tau``      — when True, per-group taus are PAVA-projected so that
      they respect difficulty ordering.  Different from unconstrained data → separate hash.
    """
    identity = {
        "simulator": config.get("simulator", {}),
        "infer_params": config.get("infer_params", []),
        "episodes_per_param": config["dataset"]["train"]["episodes_per_param"],
    }
    # Top-level mapper (non-hierarchical path, e.g. TauSurfaceGrid234).
    # Hierarchical configs store their mapper under hierarchical.task_condition_mapper
    # and never populate this top-level key, so there is no double-counting.
    top_mapper = config.get("task_condition_mapper")
    if top_mapper:
        identity["task_condition_mapper"] = top_mapper
    hier = config.get("hierarchical", {})
    if hier.get("enabled", False):
        identity["hierarchical"] = {
            "global_params": hier.get("global_params", []),
            "local_params": hier.get("local_params", []),
            "task_condition_mapper": hier.get("task_condition_mapper", {}),
        }
    # monotone_tau changes the data distribution → must be part of the hash
    if config.get("monotone_tau", False):
        identity["monotone_tau"] = True
    return config_hash(identity)


def _write_dataset_meta(
    hash_dir: Path,
    config: dict,
    hash_str: str,
    all_columns: list = None,
) -> None:
    """Write (or patch) the human-readable meta.yaml in the dataset hash directory.

    On first call (file absent): writes full meta, including ``all_columns`` if
    provided (it is only known after the first simulation run).

    On subsequent calls (file exists): no-op *unless* ``all_columns`` is
    provided and the field is not yet recorded — in that case the field is
    patched in without touching the rest of the file.
    """
    meta_path = hash_dir / "meta.yaml"
    if meta_path.exists():
        if all_columns is not None:
            with open(meta_path, "r", encoding="utf-8") as f:
                existing = yaml.safe_load(f) or {}
            if "all_columns" not in existing:
                existing["all_columns"] = all_columns
                with open(meta_path, "w", encoding="utf-8") as f:
                    yaml.dump(existing, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        return
    meta = {
        "config_hash": hash_str,
        "created_at": get_compact_timestamp_str(),
        "simulator": config.get("simulator", {}),
        "infer_params": config.get("infer_params", []),
        "episodes_per_param": config["dataset"]["train"]["episodes_per_param"],
    }
    if all_columns is not None:
        meta["all_columns"] = all_columns
    top_mapper = config.get("task_condition_mapper")
    if top_mapper:
        meta["task_condition_mapper"] = top_mapper
    hier = config.get("hierarchical", {})
    if hier.get("enabled", False):
        meta["hierarchical"] = {
            "global_params": hier.get("global_params", []),
            "local_params": hier.get("local_params", []),
            "task_condition_mapper": hier.get("task_condition_mapper", {}),
        }
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.dump(meta, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def h5_save(
    path: Union[str, Path],
    arrays: dict,
    attrs: dict = None,
    compression: str = "gzip",
    compression_opts: int = 4,
) -> None:
    """Save numpy arrays to an HDF5 file with gzip compression.

    Parameters
    ----------
    path : file path
    arrays : mapping of name → np.ndarray, stored as compressed HDF5 datasets.
    attrs : file-level metadata; list/tuple values are JSON-encoded as strings.
    compression : HDF5 compression filter (default ``"gzip"``).
    compression_opts : compression level 1–9 (default 4).
    """
    with h5py.File(path, "w") as f:
        for key, arr in arrays.items():
            f.create_dataset(
                key, data=arr,
                compression=compression, compression_opts=compression_opts,
            )
        if attrs:
            for key, val in attrs.items():
                f.attrs[key] = json.dumps(val) if isinstance(val, (list, tuple)) else val


def h5_load(path: Union[str, Path]) -> dict:
    """Load all datasets and attributes from an HDF5 file into a flat dict.

    List-valued attributes (JSON-encoded strings) are decoded back to Python
    lists.  Integer / float attributes are converted to Python native types.
    """
    result: dict = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            result[key] = f[key][()]          # load full array into memory
        for key, val in f.attrs.items():
            if isinstance(val, bytes):
                val = val.decode("utf-8")
            if isinstance(val, str):
                try:
                    result[key] = json.loads(val)
                except (json.JSONDecodeError, ValueError):
                    result[key] = val
            elif isinstance(val, np.integer):
                result[key] = int(val)
            elif isinstance(val, np.floating):
                result[key] = float(val)
            else:
                result[key] = val
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def nll_loss(
    z: torch.Tensor,
    log_det_J: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Negative log-likelihood of a normalising flow.

    .. math::
        \\mathcal{L} = \\frac12 \\|z\\|^2 - \\log|\\det J|

    Parameters
    ----------
    z : Tensor  (batch, n_param)
    log_det_J : Tensor  (batch,)
    weights : Tensor  (n_param,), optional
        Per-parameter loss weighting (multiplied with z² component).

    Returns
    -------
    Scalar tensor.
    """
    if weights is not None:
        z_sq = 0.5 * torch.sum(z ** 2 * weights.unsqueeze(0), dim=-1)
    else:
        z_sq = 0.5 * torch.sum(z ** 2, dim=-1)
    return (z_sq - log_det_J).mean()


def weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Weighted mean-squared error.

    Parameters
    ----------
    pred, target : Tensor  (batch, n_param)
    weights : Tensor  (n_param,), optional

    Returns
    -------
    Scalar tensor.
    """
    mse = (pred - target) ** 2
    if weights is not None:
        mse = mse * weights.unsqueeze(0)
    return mse.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Network config auto-fill helpers
# ─────────────────────────────────────────────────────────────────────────────

def _auto_fill_amortizer_config(
    cfg: dict,
    n_stat_features: int,
    n_infer_params: int,
    use_traj: bool = False,
    n_traj_features: int = 0,
) -> dict:
    """Fill ``null`` placeholders in the amortizer network config section.

    Mutates *cfg* in-place and returns it.

    ``mlp.out_sz`` and ``mlp.feat_sz`` are always kept in sync: if
    ``out_sz`` is set in the YAML it overwrites ``feat_sz``; otherwise
    ``out_sz`` is derived from ``feat_sz``.
    """
    enc = cfg["encoder"]
    if enc.get("stat_sz") is None:
        enc["stat_sz"] = n_stat_features
    if enc.get("traj_sz") is None:
        enc["traj_sz"] = n_traj_features if use_traj else 0
    # mlp out_sz ↔ feat_sz sync
    _mlp = enc.get("mlp", {})
    if _mlp.get("out_sz") is None:
        _mlp["out_sz"] = _mlp.get("feat_sz", 32)
    # NOTE: Do NOT overwrite feat_sz with out_sz — they serve different
    # purposes (feat_sz = hidden layer width, out_sz = final output width).
    if _mlp.get("feat_sz") is None:
        _mlp["feat_sz"] = _mlp["out_sz"]

    # Non-hierarchical invertible / linear
    inv = cfg.get("invertible", {})
    if inv.get("param_sz") is None:
        inv["param_sz"] = n_infer_params
    inv_block = inv.get("block", {})
    if inv_block.get("cond_sz") is None:
        trial_enc_out = cfg.get("trial_encoder", {}).get("attention", {}).get("out_sz", 32)
        inv_block["cond_sz"] = trial_enc_out

    lin = cfg.get("linear", {})
    if lin.get("out_sz") is None:
        lin["out_sz"] = n_infer_params

    return cfg


def _auto_fill_hierarchical_config(
    cfg: dict,
    n_stat_features: int,
    n_global: int,
    n_local: int,
    n_groups: int,
    trial_encoder_out_sz: int,
    use_traj: bool = False,
    n_traj_features: int = 0,
    max_traj_len: int = 40,
) -> dict:
    """Fill ``null`` placeholders for hierarchical amortizer config.

    Mutates *cfg* in-place and returns it.

    ``mlp.feat_sz`` (hidden width) and ``mlp.out_sz`` (output width)
    serve different purposes and are kept independent.  If only one
    is set, the other defaults to it; if both are set, both are preserved.
    """
    enc = cfg["encoder"]
    if enc.get("stat_sz") is None:
        enc["stat_sz"] = n_stat_features
    if enc.get("traj_sz") is None:
        enc["traj_sz"] = n_traj_features if use_traj else 0
    # transformer max_step — auto-set from dataset max trajectory length
    if use_traj and max_traj_len > 0:
        trans_cfg = enc.get("transformer", {})
        if trans_cfg.get("max_step") is None:
            trans_cfg["max_step"] = max_traj_len
    # mlp out_sz ↔ feat_sz sync
    _mlp = enc.get("mlp", {})
    if _mlp.get("out_sz") is None:
        _mlp["out_sz"] = _mlp.get("feat_sz", 32)
    # NOTE: Do NOT overwrite feat_sz with out_sz — they serve different
    # purposes (feat_sz = hidden layer width, out_sz = final output width).
    # Legacy keeps them separate (e.g. feat_sz=128, out_sz=64).
    if _mlp.get("feat_sz") is None:
        _mlp["feat_sz"] = _mlp["out_sz"]

    # Optional mean-concat features: appended to the trial encoder output
    # before being fed to the global / local estimator nets.
    n_mean_concat = len(cfg.get("mean_concat_indices", []))
    cond_sz_global = trial_encoder_out_sz + n_mean_concat

    # Global invertible / linear
    inv_g = cfg.setdefault("invertible_global", {})
    if inv_g.get("param_sz") is None:
        inv_g["param_sz"] = n_global
    inv_g_block = inv_g.setdefault("block", {})
    if inv_g_block.get("cond_sz") is None:
        inv_g_block["cond_sz"] = cond_sz_global

    lin_g = cfg.setdefault("linear_global", {})
    if lin_g.get("out_sz") is None:
        lin_g["out_sz"] = n_global
    if lin_g.get("in_sz") is None:
        lin_g["in_sz"] = cond_sz_global

    # Local invertible / linear
    inv_l = cfg.setdefault("invertible_local", {})
    if inv_l.get("param_sz") is None:
        inv_l["param_sz"] = n_local
    inv_l_block = inv_l.setdefault("block", {})
    if inv_l_block.get("cond_sz") is None:
        inv_l_block["cond_sz"] = cond_sz_global * 2

    lin_l = cfg.setdefault("linear_local", {})
    if lin_l.get("out_sz") is None:
        lin_l["out_sz"] = n_local
    if lin_l.get("in_sz") is None:
        concat_fv = cfg.get("concat_feature_vector", False)
        if concat_fv:
            lin_l["in_sz"] = cond_sz_global * 2
        else:
            lin_l["in_sz"] = n_global + cond_sz_global

    return cfg


def _auto_fill_hier3_config(
    cfg: dict,
    n_stat_features: int,
    n_global: int,
    n_mid_local: int,
    n_local: int,
    trial_encoder_out_sz: int,
    use_traj: bool = False,
    n_traj_features: int = 0,
    max_traj_len: int = 40,
) -> dict:
    """Fill ``null`` placeholders for the Hier3 (3-level) amortizer config.

    Linear head input sizes:
    * global : ``enc_sz``
    * mid    : ``n_global + enc_sz``
    * local  : ``n_global + n_mid_local + enc_sz``

    where ``enc_sz = trial_encoder_out_sz + n_mean_concat``.
    """
    enc = cfg["encoder"]
    if enc.get("stat_sz") is None:
        enc["stat_sz"] = n_stat_features
    if enc.get("traj_sz") is None:
        enc["traj_sz"] = n_traj_features if use_traj else 0
    if use_traj and max_traj_len > 0:
        trans_cfg = enc.get("transformer", {})
        if trans_cfg.get("max_step") is None:
            trans_cfg["max_step"] = max_traj_len
    _mlp = enc.get("mlp", {})
    if _mlp.get("out_sz") is None:
        _mlp["out_sz"] = _mlp.get("feat_sz", 32)
    if _mlp.get("feat_sz") is None:
        _mlp["feat_sz"] = _mlp["out_sz"]

    n_mean_concat = len(cfg.get("mean_concat_indices", []))
    enc_sz = trial_encoder_out_sz + n_mean_concat

    lin_g = cfg.setdefault("linear_global", {})
    if lin_g.get("out_sz") is None:
        lin_g["out_sz"] = n_global
    if lin_g.get("in_sz") is None:
        lin_g["in_sz"] = enc_sz

    lin_m = cfg.setdefault("linear_mid", {})
    if lin_m.get("out_sz") is None:
        lin_m["out_sz"] = n_mid_local
    if lin_m.get("in_sz") is None:
        lin_m["in_sz"] = n_global + enc_sz

    lin_l = cfg.setdefault("linear_local", {})
    if lin_l.get("out_sz") is None:
        lin_l["out_sz"] = n_local
    if lin_l.get("in_sz") is None:
        lin_l["in_sz"] = n_global + n_mid_local + enc_sz

    return cfg
