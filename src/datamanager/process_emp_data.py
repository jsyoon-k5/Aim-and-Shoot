"""
Process raw SQLite experiment databases into analysis-ready CSV + metadata.

Usage
-----
    python -m src.datamanager.process_emp_data <user_name>
    python -m src.datamanager.process_emp_data --all

Written by June-Seop Yoon
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..configs.constants import (
    FEATURES,
    FOLDERS,
    DB_TO_FEATURES,
    EMP_META_COLUMNS,
)
from ..utils.mymath import Convert, view_angle_deg_to_monitor_mm

# ── Project root ──────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_RAW_DIR = _PROJECT_ROOT / FOLDERS.EMPIRICAL_RAW
_PROCESSED_DIR = _PROJECT_ROOT / FOLDERS.EMPIRICAL_PROCESSED


# =====================================================================
# Trajectory DB API
# =====================================================================

def get_trial_trajectory(
    user_name: str,
    session_name: str,
    trial_idx: int,
) -> pd.DataFrame:
    """Load per-frame player action data for a single trial.

    Reads the ``player_action`` table from the user's most recent raw DB
    and filters to the given session + trial.

    Returns
    -------
    pd.DataFrame with columns:
        timestamp_utc, mouse_dx, mouse_dy, camera_az, camera_el,
        click_event, session_name, trial_idx
    Empty DataFrame if no data is found.
    """
    db_path = _find_db(user_name)
    if db_path is None:
        return pd.DataFrame()
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query(
            "SELECT * FROM player_action "
            "WHERE session_name = ? AND trial_idx = ? "
            "ORDER BY timestamp_utc",
            conn,
            params=(session_name, trial_idx),
        )
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

def get_trial_index(user_name: str) -> list[tuple[str, int]]:
    """Return an ordered list of (session_name, trial_idx) for valid trials.

    Only includes trials that belong to completed sessions (same filtering
    logic as the main processing pipeline).  Suitable for use as a loop
    index when iterating over trajectories.

    Returns
    -------
    list of (session_name, trial_idx) tuples, sorted by session then trial.
    Empty list if the user has no processed summary CSV yet.
    """
    csv_path = _PROCESSED_DIR / user_name / "summary.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path, usecols=["session_name", "trial_idx"])
    pairs = sorted(
        set(zip(df["session_name"].astype(str), df["trial_idx"].astype(int)))
    )
    return pairs

def get_all_trajectories(
    user_name: str,
) -> dict[tuple[str, int], pd.DataFrame]:
    """Load all per-frame trajectories for a user.

    Returns
    -------
    dict mapping (session_name, trial_idx) → trajectory DataFrame.
    Missing trajectories (e.g. no player_action rows) are omitted.
    """
    index = get_trial_index(user_name)
    if not index:
        return {}

    db_path = _find_db(user_name)
    if db_path is None:
        return {}

    conn = sqlite3.connect(str(db_path))
    result: dict[tuple[str, int], pd.DataFrame] = {}
    try:
        for session_name, trial_idx in index:
            df = pd.read_sql_query(
                "SELECT * FROM player_action "
                "WHERE session_name = ? AND trial_idx = ? "
                "ORDER BY timestamp_utc",
                conn,
                params=(session_name, trial_idx),
            )
            if not df.empty:
                result[(session_name, trial_idx)] = df
    except Exception:
        pass
    finally:
        conn.close()
    return result


# =====================================================================
# Hand reaction time parsing
# =====================================================================

# HRT constants — ballistic-peak-anchored threshold-crossing onset detector.
_HRT_MIN_MS        = 50.0    # reject onsets before this (anticipation / already moving)
_HRT_MAX_MS        = 500.0   # reject onsets after this (task timed out / no reaction)
_SMOOTH_SIGMA      = 3       # Gaussian smoothing width, in frames
_MIN_PEAK_SPEED    = 0.03    # minimum smoothed ballistic peak speed, m/s
_ONSET_FRAC        = 0.10    # walk-back threshold as fraction of ballistic peak
_ONSET_ABS_MIN     = 0.02    # absolute floor for walk-back threshold, m/s

def parse_hand_reaction_time(
    traj_df: pd.DataFrame,
    trial_start_utc: float,
    mouse_dpi: float = 800.0,
) -> float:
    """Estimate hand reaction time (ms) from a trial's trajectory.

    Algorithm
    ---------
    1. Convert raw mouse counts (``dx``, ``dy``) to physical speed (m/s)
       using the user's DPI, then Gaussian-smooth with ``_SMOOTH_SIGMA``.
    2. Restrict to post-trial frames (``time_ms >= 0``).
    3. Find the **global maximum** of the smoothed speed — this is the
       ballistic movement peak toward the target. Reject the trial if
       the peak is below ``_MIN_PEAK_SPEED`` (subject barely moved).
    4. Compute an onset threshold
       ``thr = max(peak * _ONSET_FRAC, _ONSET_ABS_MIN)``.
    5. Walk **backward** from the ballistic peak index while
       ``speed >= thr``. The first index where ``speed`` drops below
       ``thr`` marks the end of the pre-reactive / baseline phase; the
       next (higher) index is the movement onset.
    6. Return the onset time if it lies in ``[_HRT_MIN_MS, _HRT_MAX_MS]``,
       otherwise NaN.

    Rationale
    ---------
    Previous persistence-based approach required a *quiet* baseline
    before the ballistic burst, so any pre-trial drift / residual motion
    from the last trial caused rejection. The threshold-crossing rule
    anchored on the largest peak is robust to small pre-trial motion:
    the walk-back naturally stops just before the ballistic phase starts
    rising, regardless of small fluctuations before.

    Parameters
    ----------
    traj_df : pd.DataFrame
        ``player_action`` rows for one trial, ordered by timestamp_utc.
    trial_start_utc : float
        UTC timestamp (seconds) when the active target appeared.
    mouse_dpi : float
        Mouse DPI (counts per inch).

    Returns
    -------
    float   Hand reaction time in milliseconds, or NaN if the trial
            cannot be parsed (no movement / onset out of physiological
            window).
    """
    from scipy.ndimage import gaussian_filter1d

    if traj_df.empty or len(traj_df) < 3:
        return float("nan")

    ts = traj_df["timestamp_utc"].values.astype(float)
    dx = traj_df["mouse_dx"].values.astype(float)
    dy = traj_df["mouse_dy"].values.astype(float)

    # ── Speed profile (full trajectory, incl. pre-trial frames) ───────
    dt = np.diff(ts)
    dt = np.where(dt > 0, dt, np.nan)
    m_per_count = 0.0254 / mouse_dpi
    disp = np.sqrt(dx[1:] ** 2 + dy[1:] ** 2) * m_per_count
    speed_raw = disp / dt                                  # m/s
    speed_raw = np.where(np.isfinite(speed_raw), speed_raw, 0.0)
    time_ms = ((ts[1:] + ts[:-1]) / 2.0 - trial_start_utc) * 1000.0

    # Smooth on the full trajectory so edge effects don't corrupt t≈0.
    speed_smooth = gaussian_filter1d(speed_raw, _SMOOTH_SIGMA)

    # Restrict to post-trial frames for HRT analysis.
    mask = time_ms >= 0.0
    time_ms = time_ms[mask]
    speed = speed_smooth[mask]

    if len(speed) == 0 or not np.isfinite(speed).any():
        return float("nan")

    # ── Ballistic peak ────────────────────────────────────────────────
    peak_idx = int(np.argmax(speed))
    peak_val = float(speed[peak_idx])
    if peak_val < _MIN_PEAK_SPEED:
        return float("nan")

    # ── Walk-back threshold crossing ──────────────────────────────────
    thresh = max(peak_val * _ONSET_FRAC, _ONSET_ABS_MIN)
    i = peak_idx
    while i > 0 and speed[i] >= thresh:
        i -= 1
    onset_idx = i + 1
    if onset_idx >= len(time_ms):
        return float("nan")

    onset_t = float(time_ms[onset_idx])
    if _HRT_MIN_MS <= onset_t <= _HRT_MAX_MS:
        return onset_t
    return float("nan")

def parse_max_hand_speed(
    traj_df: pd.DataFrame,
    mouse_dpi: float,
) -> float:
    """Compute peak physical hand speed (mm/s) from raw mouse trajectory.

    Converts raw mouse counts to mm using DPI, divides by frame Δt,
    and returns the maximum speed observed during the trial.

    Parameters
    ----------
    traj_df : pd.DataFrame
        ``player_action`` rows for one trial, ordered by timestamp_utc.
        Must contain columns: timestamp_utc, mouse_dx, mouse_dy.
    mouse_dpi : float
        Mouse DPI (counts per inch).

    Returns
    -------
    float
        Peak hand speed in mm/s. NaN if trajectory is empty or has < 2 rows.
    """
    if traj_df.empty or len(traj_df) < 2:
        return float("nan")

    from scipy.ndimage import gaussian_filter1d

    ts = traj_df["timestamp_utc"].values.astype(float)
    dx = traj_df["mouse_dx"].values.astype(float)
    dy = traj_df["mouse_dy"].values.astype(float)

    dt = np.diff(ts)
    dt = np.where(dt > 0, dt, np.nan)

    mm_per_count = 25.4 / mouse_dpi          # 1 inch = 25.4 mm
    disp_mm = np.sqrt(dx[1:] ** 2 + dy[1:] ** 2) * mm_per_count
    speed_mm_s = disp_mm / dt               # mm/s per frame
    speed_mm_s = np.where(np.isfinite(speed_mm_s), speed_mm_s, 0.0)

    # Smooth to suppress single-frame jitter before taking peak
    speed_smooth = gaussian_filter1d(speed_mm_s, _SMOOTH_SIGMA)

    return float(np.max(speed_smooth)) if speed_smooth.size > 0 else float("nan")

def parse_final_hand_speed(
    traj_df: pd.DataFrame,
    mouse_dpi: float,
    trial_start_utc: float,
) -> float:
    """Compute physical hand speed (mm/s) at the moment of the shot click.

    If no shot click is found (timeout trial), uses the last frame.
    Speed is computed from the smoothed speed profile evaluated at the
    click timestamp.

    Parameters
    ----------
    traj_df : pd.DataFrame
        ``player_action`` rows for one trial, ordered by timestamp_utc.
        Must contain columns: timestamp_utc, mouse_dx, mouse_dy, click_event.
    mouse_dpi : float
        Mouse DPI (counts per inch).
    trial_start_utc : float
        UTC timestamp (seconds) when the active target appeared.

    Returns
    -------
    float
        Hand speed in mm/s at shot moment. NaN if trajectory is empty or
        has < 2 rows.
    """
    if traj_df.empty or len(traj_df) < 2:
        return float("nan")

    from scipy.ndimage import gaussian_filter1d

    ts = traj_df["timestamp_utc"].values.astype(float)
    dx = traj_df["mouse_dx"].values.astype(float)
    dy = traj_df["mouse_dy"].values.astype(float)

    dt = np.diff(ts)
    dt = np.where(dt > 0, dt, np.nan)

    mm_per_count = 25.4 / mouse_dpi
    disp_mm = np.sqrt(dx[1:] ** 2 + dy[1:] ** 2) * mm_per_count
    speed_mm_s = disp_mm / dt
    speed_mm_s = np.where(np.isfinite(speed_mm_s), speed_mm_s, 0.0)
    speed_smooth = gaussian_filter1d(speed_mm_s, _SMOOTH_SIGMA)

    # Speed array has length len(ts)-1; index i corresponds to interval (ts[i], ts[i+1]).
    # Map to per-frame index: use midpoint convention → speed[i] ~ ts[i].
    # Find shot click frame (post trial-start only)
    click_col = traj_df.get("click_event", pd.Series([""] * len(traj_df)))
    click_col = click_col.fillna("").astype(str)
    time_ms = (ts - trial_start_utc) * 1000.0

    # Look for a shot click after trial start
    shot_idx = None
    for i, (t_ms, evt) in enumerate(zip(time_ms, click_col)):
        if t_ms >= 0.0 and evt != "":
            shot_idx = i
            break

    # speed_mm_s[i] = disp of frame (i+1) / dt[i→i+1].
    # The click is at row shot_idx; dx[shot_idx] is the movement that ENDED
    # at the click, which appears in disp_mm at index shot_idx-1, i.e.
    # speed_mm_s[shot_idx-1].  Using shot_idx directly would read the speed
    # of the frame AFTER the click (near-zero → underestimates final speed).
    if shot_idx is None:
        # Timeout: use last speed sample
        spd_idx = len(speed_smooth) - 1
    else:
        spd_idx = max(0, shot_idx - 1)

    return float(speed_smooth[spd_idx])


# =====================================================================
# Trajectory parsing  (raw per-trial & coarse simulator-matched)
# =====================================================================

COARSE_TRAJ_COLUMNS = [
    "timestamp_ms",
    "dt_ms",
    "camera_az_deg",
    "camera_el_deg",
    "target_az_deg",
    "target_el_deg",
]

RAW_TRAJ_COLUMNS = [
    "timestamp_ms",
    "dt_ms",
    "mouse_dx",
    "mouse_dy",
    "camera_az_deg",
    "camera_el_deg",
    "target_x",
    "target_y",
    "target_z",
    "target_az_deg",
    "target_el_deg",
    "click_event",       # "" / "reference" / "shot"  (strings → stored as ASCII bytes in h5)
]

def parse_trial_trajectory_raw(
    df_player: pd.DataFrame,
    df_target: pd.DataFrame,
) -> pd.DataFrame:
    """Build a per-trial raw-frequency trajectory.

    **Master time grid = ``target_trajectory`` timestamps** (already filtered
    to ``target_type='target'`` before being passed in).  The first target
    frame defines t=0; the last defines t=TCT.  This guarantees that
    ``timestamp_ms`` always starts at exactly 0 and ends at the true trial
    completion time, with no dependency on external UTC bookkeeping.

    Camera az/el and mouse dx/dy from ``player_action`` are linearly
    interpolated onto the target time grid.  A symmetric buffer around the
    target time window is used when selecting player rows so that
    interpolation at the boundaries is accurate (no clamping artefacts).
    Click events are assigned to the nearest target-grid frame.

    Parameters
    ----------
    df_player : pd.DataFrame
        Rows of ``player_action`` for one trial (may include pre-trial
        reference phase rows — they are excluded by the time-window filter).
        Must have columns: timestamp_utc, mouse_dx, mouse_dy,
        camera_az, camera_el, click_event.
    df_target : pd.DataFrame
        Rows of ``target_trajectory`` already filtered to
        ``target_type='target'`` for the same (session, trial).
        Must have columns: timestamp_utc, target_x, target_y, target_z.

    Returns
    -------
    pd.DataFrame — columns listed in :data:`RAW_TRAJ_COLUMNS`.
    ``timestamp_ms[0] == 0``, ``timestamp_ms[-1] == TCT``.
    Empty DataFrame if ``df_target`` is empty.
    """
    if df_target.empty:
        return pd.DataFrame(columns=RAW_TRAJ_COLUMNS)

    # ── Master grid from target timestamps ───────────────────────────
    dft = df_target.sort_values("timestamp_utc").reset_index(drop=True)
    ts_t = dft["timestamp_utc"].to_numpy(dtype=np.float64)
    t_start = ts_t[0]
    t_end   = ts_t[-1]

    # timestamp_ms: exactly 0 at first target frame, TCT at last
    time_ms = ((ts_t - t_start) * 1000.0).astype(np.float32)

    # target xyz (Panda3D coords: x=right, y=fwd, z=up)
    tx = dft["target_x"].to_numpy(dtype=np.float64)
    ty = dft["target_y"].to_numpy(dtype=np.float64)
    tz = dft["target_z"].to_numpy(dtype=np.float64)

    # Panda3D → agent convention (x=fwd, y=up, z=right) for cart2sphr_vec
    tgt_azel = Convert.cart2sphr_vec(ty, tz, tx, return_in_degree=True)  # (N, 2)

    # ── Interpolate player columns onto target grid ───────────────────
    # Use a ±50 ms buffer around the active target window so that
    # interpolation at the edges uses real player data, not clamped values.
    _BUFFER_S = 0.05  # 50 ms ≈ 12 frames @ 240 Hz

    n = len(ts_t)
    cam_az = np.full(n, np.nan, dtype=np.float32)
    cam_el = np.full(n, np.nan, dtype=np.float32)
    mdx    = np.zeros(n, dtype=np.float32)
    mdy    = np.zeros(n, dtype=np.float32)
    click  = np.full(n, "", dtype=object)

    if not df_player.empty:
        dfp = df_player.sort_values("timestamp_utc").reset_index(drop=True)
        mask = (
            (dfp["timestamp_utc"] >= t_start - _BUFFER_S) &
            (dfp["timestamp_utc"] <= t_end   + _BUFFER_S)
        )
        dfp_win = dfp[mask].reset_index(drop=True)
        if dfp_win.empty:
            dfp_win = dfp  # fallback: use all rows

        ts_p = dfp_win["timestamp_utc"].to_numpy(dtype=np.float64)
        cam_az = np.interp(ts_t, ts_p,
                           dfp_win["camera_az"].to_numpy(dtype=np.float64)
                           ).astype(np.float32)
        cam_el = np.interp(ts_t, ts_p,
                           dfp_win["camera_el"].to_numpy(dtype=np.float64)
                           ).astype(np.float32)
        mdx    = np.interp(ts_t, ts_p,
                           dfp_win["mouse_dx"].to_numpy(dtype=np.float64)
                           ).astype(np.float32)
        mdy    = np.interp(ts_t, ts_p,
                           dfp_win["mouse_dy"].to_numpy(dtype=np.float64)
                           ).astype(np.float32)

        # Assign click events to the nearest target-grid frame
        click_mask = dfp_win["click_event"].fillna("").astype(str) != ""
        for _, crow in dfp_win[click_mask].iterrows():
            idx = int(np.argmin(np.abs(ts_t - float(crow["timestamp_utc"]))))
            evt = str(crow["click_event"])
            click[idx] = (click[idx] + "," + evt).strip(",") if click[idx] else evt

    # ── Build output DataFrame ────────────────────────────────────────
    dt_ms = np.zeros(n, dtype=np.float32)
    if n > 1:
        dt_ms[1:] = np.diff(time_ms)

    return pd.DataFrame({
        "timestamp_ms":  time_ms,
        "dt_ms":         dt_ms,
        "mouse_dx":      mdx,
        "mouse_dy":      mdy,
        "camera_az_deg": cam_az,
        "camera_el_deg": cam_el,
        "target_x":      tx.astype(np.float32),
        "target_y":      ty.astype(np.float32),
        "target_z":      tz.astype(np.float32),
        "target_az_deg": tgt_azel[:, 0].astype(np.float32),
        "target_el_deg": tgt_azel[:, 1].astype(np.float32),
        "click_event":   click,
    })

def parse_trial_trajectory_coarse(
    raw_traj: pd.DataFrame,
    hrt_ms: float,
    tct_ms: float,
) -> pd.DataFrame:
    """Re-sample a raw per-trial trajectory onto the simulator's coarse grid.

    The coarse grid mirrors the simulator's BUMP-resolution output::

        [0.0, hrt_ms, hrt_ms + BUMP, hrt_ms + 2·BUMP, ..., tct_ms]

    ``tct_ms`` is always the final timestamp; if the last BUMP step falls
    within less than ``BUMP`` ms of ``tct_ms`` that step is omitted so no
    duplicate terminal samples are produced.  Camera and target az/el are
    linearly interpolated from the raw trajectory at these timestamps.

    Parameters
    ----------
    raw_traj : pd.DataFrame
        Output of :func:`parse_trial_trajectory_raw` (columns include
        timestamp_ms, camera_az_deg, camera_el_deg, target_az_deg,
        target_el_deg).  All samples must have ``timestamp_ms >= 0``.
    hrt_ms : float
        Parsed hand reaction time (ms).  If NaN, falls back to the first
        BUMP step time after 0 (i.e. treats the whole trial as ballistic).
    tct_ms : float
        Trial completion time (ms from target onset to shot).

    Returns
    -------
    pd.DataFrame  — columns listed in :data:`COARSE_TRAJ_COLUMNS`.  Empty
    DataFrame if the raw trajectory is empty.
    """
    from ..configs.constants import TINTERVAL

    if raw_traj.empty or not math.isfinite(tct_ms) or tct_ms <= 0.0:
        return pd.DataFrame(columns=COARSE_TRAJ_COLUMNS)

    BUMP = float(TINTERVAL.BUMP)

    # Fallback when HRT is NaN or out of range — start the BUMP grid at 0.
    if not math.isfinite(hrt_ms) or hrt_ms <= 0.0 or hrt_ms >= tct_ms:
        hrt_ms = 0.0

    # Build target timestamps: [0, HRT, HRT+BUMP, HRT+2·BUMP, ..., TCT]
    stamps: list[float] = [0.0]
    if hrt_ms > 0.0:
        stamps.append(float(hrt_ms))
    t = float(hrt_ms) + BUMP
    while t < tct_ms - 1e-6:
        stamps.append(t)
        t += BUMP
    stamps.append(float(tct_ms))
    # Deduplicate (can happen when hrt_ms == 0 or stamp equals TCT)
    stamps = sorted({round(s, 6) for s in stamps})
    times = np.array(stamps, dtype=np.float32)

    # Interpolate from the raw grid
    raw_t = raw_traj["timestamp_ms"].to_numpy(dtype=np.float64)
    def _interp(col: str) -> np.ndarray:
        y = raw_traj[col].to_numpy(dtype=np.float64)
        return np.interp(times.astype(np.float64), raw_t, y).astype(np.float32)

    dt_ms = np.zeros_like(times)
    if len(times) > 1:
        dt_ms[1:] = np.diff(times)

    return pd.DataFrame({
        "timestamp_ms":   times,
        "dt_ms":          dt_ms,
        "camera_az_deg":  _interp("camera_az_deg"),
        "camera_el_deg": _interp("camera_el_deg"),
        "target_az_deg":  _interp("target_az_deg"),
        "target_el_deg":  _interp("target_el_deg"),
    })


# =====================================================================
# Trajectory storage  (per-user HDF5 files)
# =====================================================================

_TRAJ_H5_NAME = "trajectories.h5"

def load_trajectories_h5(
    path: Path,
    kind: str = "coarse",
) -> dict[tuple[str, int], pd.DataFrame]:
    """Load per-trial trajectories from a user's trajectories.h5.

    Parameters
    ----------
    path : Path
        Path to the trajectories.h5 file.
    kind : {"coarse", "raw"}, default ``"coarse"``
        Which trajectory resolution to load.
        ``"coarse"`` — simulator-matched BUMP-grid (small, fast, used for inference).
        ``"raw"``    — full ~240 Hz player_action + interpolated target positions.

    Returns
    -------
    dict
        ``{(session_name, trial_idx): DataFrame}``
    """
    import h5py

    out: dict[tuple[str, int], pd.DataFrame] = {}
    if not path.exists():
        return out
    with h5py.File(str(path), "r") as f:
        if kind not in f:
            return out
        grp = f[kind]
        for gname in grp.keys():
            sn, ti = _parse_trial_group_name(gname)
            g = grp[gname]
            cols: dict[str, np.ndarray] = {}
            for col in g.keys():
                arr = g[col][...]
                if arr.dtype.kind == "O":
                    arr = np.array([
                        (x.decode("ascii") if isinstance(x, bytes) else str(x))
                        for x in arr
                    ])
                cols[col] = arr
            out[(sn, ti)] = pd.DataFrame(cols)
    return out


# =====================================================================
# Main processing
# =====================================================================

def process_user(user_name: str, *, verbose: bool = True) -> Path | None:
    """Process a single user's raw DB into summary CSV + metadata YAML.

    Always overwrites existing output files.
    Returns the output directory path, or None if processing failed.
    """
    db_path = _find_db(user_name)
    if db_path is None:
        if verbose:
            print(f"[process] No DB found for user '{user_name}' in {_RAW_DIR}")
        return None

    if verbose:
        print(f"[process] Using DB: {db_path.name}")

    conn = sqlite3.connect(str(db_path))

    # Read all tables
    df_trials = _read_table(conn, "session_trial_info")
    df_timestamps = _read_table(conn, "session_timestamp")
    df_events = _read_table(conn, "experiment_events")
    df_profile = _read_table(conn, "user_profile")

    conn.close()

    if df_trials.empty:
        if verbose:
            print(f"[process] No trial data found in {db_path.name}")
        return None

    # ── Validate ──────────────────────────────────────────────────────
    completed = _get_completed_sessions(df_timestamps)
    if verbose:
        all_sessions = set(df_trials["session_name"].unique())
        incomplete = all_sessions - completed
        print(f"[process] Sessions: {len(all_sessions)} total, "
              f"{len(completed)} complete, {len(incomplete)} incomplete")
        if incomplete:
            print(f"[process]   Dropping incomplete: {sorted(incomplete)}")

    df_valid = _validate_trials(df_trials, completed)
    if df_valid.empty:
        if verbose:
            print(f"[process] No valid trials after filtering.")
        return None

    if verbose:
        print(f"[process] Valid trials: {len(df_valid)}")

    # ── Rename DB columns → FEATURES names ────────────────────────────
    rename_map = {db_col: feat for db_col, feat in DB_TO_FEATURES.items()
                  if db_col in df_valid.columns}
    df_valid = df_valid.rename(columns=rename_map)

    # ── Add experiment meta columns ───────────────────────────────────
    df_valid.insert(0, "user_name", user_name)
    # session_name and trial_idx already present from DB

    # ── Compute derived features ──────────────────────────────────────
    fps_env_cfg = None
    if not df_profile.empty:
        cfg_str = df_profile.iloc[0].get("fps_env_config", None)
        if cfg_str:
            try:
                fps_env_cfg = json.loads(cfg_str)
            except (json.JSONDecodeError, TypeError):
                pass

    df_valid = _compute_derived_features(df_valid, fps_env_cfg)

    # ── Hand reaction time ────────────────────────────────────────────
    # Load all trajectories for the user and estimate HRT per trial.
    # NOTE: get_all_trajectories() depends on summary.csv which hasn't been
    # written yet.  Query the DB directly using the already-validated trial
    # index so new users are handled correctly.
    mouse_dpi = 800.0
    if not df_profile.empty:
        mouse_dpi = float(df_profile.iloc[0].get("mouse_dpi", 800.0))
    trajectories: dict[tuple[str, int], pd.DataFrame] = {}
    _conn2 = sqlite3.connect(str(db_path))
    try:
        for _, _row in df_valid.iterrows():
            _sn = str(_row["session_name"])
            _ti = int(_row["trial_idx"])
            _df = pd.read_sql_query(
                "SELECT * FROM player_action "
                "WHERE session_name = ? AND trial_idx = ? "
                "ORDER BY timestamp_utc",
                _conn2, params=(_sn, _ti),
            )
            if not _df.empty:
                trajectories[(_sn, _ti)] = _df
    except Exception:
        pass
    finally:
        _conn2.close()
    hrt_values: list[float] = []
    max_hspd_values: list[float] = []
    final_hspd_values: list[float] = []
    for _, row in df_valid.iterrows():
        key = (str(row["session_name"]), int(row["trial_idx"]))
        traj = trajectories.get(key, pd.DataFrame())
        start_utc = float(row.get("start_time_utc", 0.0))
        hrt_values.append(parse_hand_reaction_time(traj, start_utc, mouse_dpi))
        max_hspd_values.append(parse_max_hand_speed(traj, mouse_dpi))
        final_hspd_values.append(parse_final_hand_speed(traj, mouse_dpi, start_utc))

    # Fill NaN HRT / MAX_HSPD with the user's own mean of successfully parsed trials.
    # NaN rows must not appear in the output CSV — downstream simulator
    # requires every trial to have a valid scalar for both features.
    valid_hrt = [v for v in hrt_values if not math.isnan(v)]
    hrt_mean = sum(valid_hrt) / len(valid_hrt) if valid_hrt else float("nan")
    hrt_values = [v if not math.isnan(v) else hrt_mean for v in hrt_values]

    valid_hspd = [v for v in max_hspd_values if not math.isnan(v)]
    hspd_mean = sum(valid_hspd) / len(valid_hspd) if valid_hspd else float("nan")
    max_hspd_values = [v if not math.isnan(v) else hspd_mean for v in max_hspd_values]

    valid_final_hspd = [v for v in final_hspd_values if not math.isnan(v)]
    final_hspd_mean = sum(valid_final_hspd) / len(valid_final_hspd) if valid_final_hspd else float("nan")
    final_hspd_values = [v if not math.isnan(v) else final_hspd_mean for v in final_hspd_values]

    df_valid[FEATURES.HRT] = hrt_values
    df_valid[FEATURES.MAX_HSPD] = max_hspd_values
    df_valid[FEATURES.FINAL_HSPD] = final_hspd_values
    df_valid[FEATURES.MOVEMENT_TIME] = (
        df_valid[FEATURES.TCT] - df_valid[FEATURES.HRT]
    ).clip(lower=0.0)
    if verbose:
        n_filled_hrt = len(hrt_values) - len(valid_hrt)
        n_filled_hspd = len(max_hspd_values) - len(valid_hspd)
        if valid_hrt:
            print(f"[process] HRT: {len(valid_hrt)}/{len(hrt_values)} directly parsed"
                  + (f", {n_filled_hrt} mean-filled ({hrt_mean:.1f} ms)" if n_filled_hrt else "")
                  + f" — mean={hrt_mean:.1f} ms, median={sorted(valid_hrt)[len(valid_hrt)//2]:.1f} ms")
        else:
            print(f"[process] HRT: no trajectories parsed")
        if valid_hspd:
            print(f"[process] Max hand speed: {len(valid_hspd)}/{len(max_hspd_values)} parsed"
                  + (f", {n_filled_hspd} mean-filled" if n_filled_hspd else "")
                  + f" (mean={hspd_mean:.0f} mm/s)")
        else:
            print(f"[process] Max hand speed: no trajectories parsed")

    # ── Select & order output columns ─────────────────────────────────
    # Meta columns first, then FEATURES columns in a consistent order,
    # then remaining DB-only columns (shot_camera_az/el, etc.)
    feature_cols_order = [
        FEATURES.CAM_INIT_ANGLE_AZ,
        FEATURES.CAM_INIT_ANGLE_EL,
        FEATURES.TARGET_POS_WORLD_AZ,
        FEATURES.TARGET_POS_WORLD_EL,
        FEATURES.TARGET_RADIUS,
        FEATURES.TARGET_ASPEED,
        FEATURES.TARGET_MOVEMENT_MONITOR_DIRECTION,
        FEATURES.TARGET_ORBIT_AXIS_AZ,
        FEATURES.TARGET_ORBIT_AXIS_EL,
        # Derived
        FEATURES.TARGET_POS_MONITOR_X,
        FEATURES.TARGET_POS_MONITOR_Y,
        FEATURES.TARGET_POS_DISTANCE,
        FEATURES.TARGET_RADIUS_MM,
        FEATURES.FITTS_IOD,
        # Result
        FEATURES.TCT,
        FEATURES.ACC,
        FEATURES.ERR_DEG,
        FEATURES.ERR,
        # Parsed from trajectory
        FEATURES.HRT,
        FEATURES.MOVEMENT_TIME,
        FEATURES.MAX_HSPD,
        FEATURES.FINAL_HSPD,
    ]
    extra_cols = [
        FEATURES.CAM_SHOT_ANGLE_AZ,
        FEATURES.CAM_SHOT_ANGLE_EL,
        FEATURES.TARGET_INIT_X,
        FEATURES.TARGET_INIT_Y,
        FEATURES.TARGET_INIT_Z,
        FEATURES.TRIAL_START_UTC,
        FEATURES.TRIAL_END_UTC,
    ]

    ordered = EMP_META_COLUMNS.copy()
    for c in feature_cols_order:
        if c in df_valid.columns:
            ordered.append(c)
    for c in extra_cols:
        if c in df_valid.columns:
            ordered.append(c)
    # Append anything remaining that we haven't listed
    for c in df_valid.columns:
        if c not in ordered and c != "id":
            ordered.append(c)

    df_out = df_valid[[c for c in ordered if c in df_valid.columns]]

    # ── Save summary CSV ──────────────────────────────────────────────
    out_dir = _PROCESSED_DIR / user_name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    df_out.to_csv(csv_path, index=False, float_format="%.6f")
    if verbose:
        print(f"[process] Saved: {csv_path}")

    # ── Build & save per-trial trajectories (raw + coarse) ────────────
    # ``trajectories`` (player_action) is already loaded above for HRT.
    # target_trajectory is per-trial-filtered at SQL level so no
    # interpolation ever crosses a trial boundary.
    target_trajectories = _get_all_target_trajectories(user_name)
    traj_per_trial: dict[tuple[str, int], dict[str, pd.DataFrame]] = {}
    n_raw, n_coarse = 0, 0
    for _, row in df_out.iterrows():
        key = (str(row["session_name"]), int(row["trial_idx"]))
        df_p = trajectories.get(key, pd.DataFrame())
        df_t = target_trajectories.get(key, pd.DataFrame())
        raw = parse_trial_trajectory_raw(df_p, df_t)
        if raw.empty:
            continue
        hrt_ms = float(row.get(FEATURES.HRT, float("nan")))
        # tct_ms = raw's last timestamp: both start (0) and end are defined
        # by the target_trajectory table, so coarse and raw are guaranteed
        # to share identical endpoints.
        tct_ms = float(raw["timestamp_ms"].iloc[-1])
        coarse = parse_trial_trajectory_coarse(raw, hrt_ms, tct_ms)
        traj_per_trial[key] = {"raw": raw, "coarse": coarse}
        n_raw += 1
        if not coarse.empty:
            n_coarse += 1

    traj_path = out_dir / _TRAJ_H5_NAME
    _save_trajectories_h5(traj_path, traj_per_trial)
    if verbose:
        print(f"[process] Saved: {traj_path}  "
              f"(raw={n_raw}, coarse={n_coarse} / {len(df_out)} trials)")

    # ── Save metadata YAML ────────────────────────────────────────────
    meta = _build_metadata(
        user_name, db_path, df_profile, df_timestamps, df_events,
        completed, df_out,
    )
    meta_path = out_dir / "metadata.yaml"
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    if verbose:
        print(f"[process] Saved: {meta_path}")

    return out_dir


# =====================================================================
# CLI
# =====================================================================

def _list_all_users() -> list[str]:
    """Discover all user names from DB filenames in the raw directory."""
    users = set()
    if _RAW_DIR.exists():
        for p in _RAW_DIR.glob("*.db"):
            # Filename: {user}_{hash}.db — extract everything before last _
            name_parts = p.stem.rsplit("_", 1)
            if len(name_parts) == 2:
                users.add(name_parts[0])
    return sorted(users)

def main():
    parser = argparse.ArgumentParser(
        description="Process raw experiment DB into summary CSV + metadata."
    )
    parser.add_argument(
        "user", nargs="?", default=None,
        help="User name to process. Omit or use --all for all users.",
    )
    parser.add_argument("--all", action="store_true", help="Process all users.")
    args = parser.parse_args()

    if args.all or args.user is None:
        users = _list_all_users()
        if not users:
            print(f"[process] No DB files found in {_RAW_DIR}")
            return
        print(f"[process] Processing {len(users)} user(s): {users}")
        for u in users:
            print(f"\n{'='*60}")
            process_user(u)
    else:
        process_user(args.user)


# =====================================================================
# Private helpers
# =====================================================================

def _find_db(user_name: str) -> Path | None:
    """Find the most recent DB file for *user_name* under the raw dir.

    DB files are named ``{user}_{hash}.db``.  If multiple exist for the
    same user (e.g. config change → different hash), the one with the
    latest filesystem modification time is returned.
    """
    candidates = sorted(
        _RAW_DIR.glob(f"{user_name}_*.db"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None

def _read_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    """Read an entire table into a DataFrame (empty DF if table missing)."""
    try:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)
    except Exception:
        return pd.DataFrame()

def _get_completed_sessions(df_timestamps: pd.DataFrame) -> set[str]:
    """Return set of session_name strings that were fully completed.

    A session is considered complete when it appears in session_timestamp
    (engine writes a row only after the last trial's ``end_trial`` call).

    Sessions that were interrupted (user quit from pause, crash, etc.)
    do NOT get a session_timestamp row and are therefore excluded.
    """
    if df_timestamps.empty:
        return set()
    return set(df_timestamps["session_name"].unique())

def _validate_trials(
    df_trials: pd.DataFrame,
    completed_sessions: set[str],
) -> pd.DataFrame:
    """Filter trial rows to only those in completed sessions and with
    valid data (non-null required fields).

    Trials interrupted mid-action (pause during TRIAL_ACTIVE) are
    automatically excluded because the logger only writes a row on shot,
    so no session_trial_info row exists for them.
    """
    if df_trials.empty:
        return df_trials

    # 1. Keep only completed sessions
    df = df_trials[df_trials["session_name"].isin(completed_sessions)].copy()

    # 2. Drop rows with missing critical fields
    required_cols = [
        "session_name", "trial_idx", "start_time_utc", "end_time_utc",
        "hit",
        "target_radius_deg", "target_speed_deg_s",
    ]
    existing_required = [c for c in required_cols if c in df.columns]
    df = df.dropna(subset=existing_required)

    # 4. Backward compat: if completion_time_ms is missing, compute it
    if "completion_time_ms" not in df.columns:
        df["completion_time_ms"] = (
            (df["end_time_utc"] - df["start_time_utc"]) * 1000.0
        )

    return df.reset_index(drop=True)

def _compute_derived_features(
    df: pd.DataFrame,
    fps_env_cfg: dict | None,
) -> pd.DataFrame:
    """Add derived columns that the simulator normally computes.

    These include target_pos_monitor_mm, target_distance_mm,
    target_radius_mm, target_index_of_difficulty, etc.
    """
    # Get monitor/FOV geometry — prefer from the stored fps_env config,
    # fall back to the task section defaults.
    if fps_env_cfg is not None:
        task_cfg = fps_env_cfg.get("task", {})
        mon_w = float(task_cfg.get("monitor_width_mm", 531.3))
        mon_h = float(task_cfg.get("monitor_height_mm", 298.8))
        fov_w = float(task_cfg.get("cam_fov_deg_width", 103.0))
        fov_h = float(task_cfg.get("cam_fov_deg_height", 70.533))
    else:
        mon_w, mon_h = 531.3, 298.8
        fov_w, fov_h = 103.0, 70.533

    # ── target_pos_monitor_mm ──
    # We have target_pos_world (from init_x/y/z) and camera (from ref_click).
    # The simulator stores camera_azel_deg = [az, el] and target_pos_world.
    # game2monitor converts world pos → monitor mm given camera orientation.
    az_col = FEATURES.CAM_INIT_ANGLE_AZ
    el_col = FEATURES.CAM_INIT_ANGLE_EL

    if az_col in df.columns and el_col in df.columns:
        mon_x_list = []
        mon_y_list = []
        fov_wh = np.array([fov_w, fov_h])
        mon_half = np.array([mon_w / 2.0, mon_h / 2.0])
        for _, row in df.iterrows():
            camera_azel = np.array([row[az_col], row[el_col]], dtype=float)
            target_world = np.array([
                row[FEATURES.TARGET_INIT_X],
                row[FEATURES.TARGET_INIT_Y],
                row[FEATURES.TARGET_INIT_Z],
            ], dtype=float)
            mon_xy = Convert.game2monitor(
                np.zeros(3),       # camera_pos
                camera_azel,       # camera_azel
                target_world,      # target_pos
                fov_wh,            # fov_deg
                mon_half,          # monitor_half_size
            )
            mon_x_list.append(float(mon_xy[0]))
            mon_y_list.append(float(mon_xy[1]))

        df[FEATURES.TARGET_POS_MONITOR_X] = mon_x_list
        df[FEATURES.TARGET_POS_MONITOR_Y] = mon_y_list
        df[FEATURES.TARGET_POS_DISTANCE] = np.sqrt(
            np.array(mon_x_list) ** 2 + np.array(mon_y_list) ** 2
        )

    # ── target_radius_mm ──
    if FEATURES.TARGET_RADIUS in df.columns:
        df[FEATURES.TARGET_RADIUS_MM] = df[FEATURES.TARGET_RADIUS].apply(
            lambda r: float(view_angle_deg_to_monitor_mm(r, mon_w, fov_w))
        )

    # ── target_index_of_difficulty (Fitts' law ID) ──
    if FEATURES.TARGET_POS_DISTANCE in df.columns and FEATURES.TARGET_RADIUS_MM in df.columns:
        D = df[FEATURES.TARGET_POS_DISTANCE].values
        W = df[FEATURES.TARGET_RADIUS_MM].values
        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            df[FEATURES.FITTS_IOD] = np.log2(D / (2.0 * W) + 1.0)

    # ── shoot_error_mm ──
    # Convert angular shoot error to mm on the monitor (same projection as radius).
    if FEATURES.ERR_DEG in df.columns:
        df[FEATURES.ERR] = df[FEATURES.ERR_DEG].apply(
            lambda e: float(view_angle_deg_to_monitor_mm(e, mon_w, fov_w))
            if not math.isnan(float(e)) else float("nan")
        )

    return df

def _get_all_target_trajectories(
    user_name: str,
) -> dict[tuple[str, int], pd.DataFrame]:
    """Load all per-trial target position rows from the raw DB.

    Returns a dict keyed by ``(session_name, trial_idx)`` — each value is a
    DataFrame of ``target_trajectory`` rows for that trial only.  Trials with
    no logged target rows are absent from the dict.

    Per-trial filtering is done at the SQL level so later sync/interpolation
    never crosses trial boundaries.
    """
    index = get_trial_index(user_name)
    if not index:
        return {}

    db_path = _find_db(user_name)
    if db_path is None:
        return {}

    conn = sqlite3.connect(str(db_path))
    result: dict[tuple[str, int], pd.DataFrame] = {}
    try:
        df_all = pd.read_sql_query(
            "SELECT session_name, trial_idx, timestamp_utc, "
            "target_x, target_y, target_z, target_type "
            "FROM target_trajectory "
            "WHERE target_type = 'target' "
            "ORDER BY session_name, trial_idx, timestamp_utc",
            conn,
        )
    except Exception:
        df_all = pd.DataFrame()
    finally:
        conn.close()

    if df_all.empty:
        return {}

    for (sn, ti), grp in df_all.groupby(["session_name", "trial_idx"]):
        key = (str(sn), int(ti))
        result[key] = grp.reset_index(drop=True)
    return result

def _trial_group_name(session_name: str, trial_idx: int) -> str:
    """HDF5 group name encoding a (session, trial) pair — reversible."""
    return f"{session_name}__t{int(trial_idx):04d}"

def _parse_trial_group_name(name: str) -> tuple[str, int]:
    """Inverse of :func:`_trial_group_name`."""
    session_name, t_tag = name.rsplit("__t", 1)
    return session_name, int(t_tag)

def _save_trajectories_h5(
    path: Path,
    traj_per_trial: dict[tuple[str, int], dict[str, pd.DataFrame]],
) -> None:
    """Write per-trial raw + coarse trajectories to a single h5 file.

    Layout::

        trajectories.h5
            raw/<session>__t0000/  {column: 1-D dataset}
            raw/<session>__t0001/  ...
            coarse/<session>__t0000/ ...

    Parameters
    ----------
    path : Path
        Destination .h5 file (overwritten).
    traj_per_trial : dict
        ``{(session_name, trial_idx): {"raw": DataFrame, "coarse": DataFrame}}``
    """
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        raw_grp = f.create_group("raw")
        coa_grp = f.create_group("coarse")
        for (sn, ti), d in traj_per_trial.items():
            gname = _trial_group_name(sn, ti)
            for kind, sub in (("raw", raw_grp), ("coarse", coa_grp)):
                df = d.get(kind)
                if df is None or df.empty:
                    continue
                g = sub.create_group(gname)
                for col in df.columns:
                    arr = df[col].to_numpy()
                    if arr.dtype.kind in ("U", "O"):
                        # Variable-length strings → ASCII-encoded h5 vlen
                        dt = h5py.special_dtype(vlen=bytes)
                        g.create_dataset(
                            col,
                            data=np.array([str(x).encode("ascii") for x in arr]),
                            dtype=dt,
                            compression="gzip",
                            compression_opts=4,
                        )
                    else:
                        g.create_dataset(
                            col,
                            data=arr,
                            compression="gzip",
                            compression_opts=4,
                        )

def _build_metadata(
    user_name: str,
    db_path: Path,
    df_profile: pd.DataFrame,
    df_timestamps: pd.DataFrame,
    df_events: pd.DataFrame,
    completed_sessions: set[str],
    df_summary: pd.DataFrame,
) -> dict:
    """Build a metadata dict summarizing the experiment run."""
    meta: dict = {
        "user_name": user_name,
        "source_db": db_path.name,
    }

    # From user_profile table
    if not df_profile.empty:
        row = df_profile.iloc[0]
        meta["mouse_dpi"] = float(row.get("mouse_dpi", 0))
        meta["sensitivity_type"] = str(row.get("sensitivity_type", ""))
        meta["sensitivity_params"] = str(row.get("sensitivity_params", ""))
        meta["experiment_date"] = str(row.get("experiment_date", ""))
        meta["theme"] = str(row.get("theme", "default"))

    # Session timestamps
    sessions_meta = []
    if not df_timestamps.empty:
        for _, row in df_timestamps.iterrows():
            sname = str(row["session_name"])
            if sname in completed_sessions:
                sessions_meta.append({
                    "session_name": sname,
                    "start_time_utc": float(row["start_time_utc"]),
                    "end_time_utc": float(row["end_time_utc"]),
                    "duration_s": float(row["end_time_utc"] - row["start_time_utc"]),
                })
    meta["completed_sessions"] = sessions_meta
    meta["num_completed_sessions"] = len(sessions_meta)

    # Trial counts
    meta["total_valid_trials"] = len(df_summary)

    # Events summary
    if not df_events.empty:
        event_counts = df_events["event_type"].value_counts().to_dict()
        meta["event_counts"] = {str(k): int(v) for k, v in event_counts.items()}

    # FPS env config (stored as JSON string in user_profile)
    if not df_profile.empty:
        cfg_str = df_profile.iloc[0].get("fps_env_config", None)
        if cfg_str:
            try:
                meta["fps_env_config"] = json.loads(cfg_str)
            except (json.JSONDecodeError, TypeError):
                meta["fps_env_config"] = str(cfg_str)

    return meta

if __name__ == "__main__":
    main()