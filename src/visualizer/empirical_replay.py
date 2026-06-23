"""
Empirical trajectory replay video renderer.

This module renders processed IJHCS empirical trajectories directly from the
60 Hz trajectory store.  It reuses EpisodeVideoRenderer for Panda3D scene
setup and video writing, but bypasses the simulator trajectory reconstruction.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..agent.task import AimandShootSpiderShotTask as AnSTask
from ..configs.constants import FEATURES
from ..configs.loader import load_yaml_config_file
from ..datamanager.load_emp_data import IJHCSExpDataLoader
from ..utils.mymath import Convert
from ..utils.myutils import get_compact_timestamp_str
from .renderer import EpisodeVideoRenderer


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "empirical" / "video"
DEFAULT_MAX_TRIALS = 100
REQUIRED_TRAJ_COLUMNS = (
    "timestamp_ms",
    "camera_az_deg",
    "camera_el_deg",
    "target_az_deg",
    "target_el_deg",
    "target_x_mm",
    "target_y_mm",
)


@dataclass
class EmpiricalReplayRecord:
    """Minimal episode-like record consumed by EpisodeVideoRenderer."""

    trajectory_key: str
    summary: dict[str, Any]
    trajectory: pd.DataFrame
    player_state: dict[str, Any]
    init_game_env: dict[str, Any]
    result_report: dict[str, Any]


class EmpiricalReplayRenderer(EpisodeVideoRenderer):
    """Renderer that samples empirical 60 Hz trajectory rows into video frames."""

    def _reconstruct_trajectory(self, ep_rec: EmpiricalReplayRecord):
        traj = ep_rec.trajectory
        _assert_required_columns(traj, REQUIRED_TRAJ_COLUMNS, ep_rec.trajectory_key)
        if traj.empty:
            raise ValueError(f"Empty empirical trajectory: {ep_rec.trajectory_key}")

        t_raw = _series_float(traj, "timestamp_ms")
        if t_raw.size == 0:
            raise ValueError(f"Missing timestamp data: {ep_rec.trajectory_key}")
        t_raw = t_raw - float(t_raw[0])

        order = np.argsort(t_raw)
        t_raw = t_raw[order]
        keep = np.concatenate(([True], np.diff(t_raw) > 1e-9))
        t_src = t_raw[keep]
        if t_src.size == 0:
            raise ValueError(f"No usable timestamp data: {ep_rec.trajectory_key}")

        tct_ms = _finite_float(
            ep_rec.result_report.get(FEATURES.TCT),
            default=float(t_src[-1]),
        )
        if tct_ms <= 0.0:
            tct_ms = float(t_src[-1])

        ms_per_frame = 1000.0 / float(self._fps)
        frame_times = np.arange(0.0, tct_ms + 1e-6, ms_per_frame)
        if frame_times.size == 0 or frame_times[-1] < tct_ms - 1e-6:
            frame_times = np.append(frame_times, tct_ms)

        camera_az = _interp_series(traj, "camera_az_deg", order, keep, t_src, frame_times)
        camera_el = _interp_series(traj, "camera_el_deg", order, keep, t_src, frame_times)
        target_az = _interp_series(traj, "target_az_deg", order, keep, t_src, frame_times)
        target_el = _interp_series(traj, "target_el_deg", order, keep, t_src, frame_times)
        target_x = _interp_series(traj, "target_x_mm", order, keep, t_src, frame_times)
        target_y = _interp_series(traj, "target_y_mm", order, keep, t_src, frame_times)

        target_world = Convert.sphr2cart_vec(target_az, target_el, angle_is_degree=True)
        camera_azel = np.column_stack((camera_az, camera_el))
        target_monitor = np.column_stack((target_x, target_y))
        gaze_monitor = _sample_gaze_mm(traj, order, keep, t_src, frame_times)

        frames = [
            (
                camera_azel[i],
                target_world[i],
                target_monitor[i],
                gaze_monitor[i],
                float(frame_times[i]),
            )
            for i in range(len(frame_times))
        ]

        visual_radius = float(np.radians(ep_rec.init_game_env.get("target_radius_deg", 0.05)))
        return frames, float(tct_ms), visual_radius


def render_empirical_replay(
    filters: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    *,
    output_root: str | Path | None = None,
    output_name: str | None = None,
    combine: bool = True,
    max_trials: int = DEFAULT_MAX_TRIALS,
    confirm_large: bool = False,
    limit: int | None = None,
    loader_kwargs: Mapping[str, Any] | None = None,
    render_config: Mapping[str, Any] | None = None,
    task_config: Mapping[str, Any] | None = None,
) -> str | list[str]:
    """Render empirical IJHCS trials to replay video.

    Parameters
    ----------
    filters
        One loader filter dict or a list of filter dicts.  Filter keys are the
        summary columns accepted by IJHCSExpDataLoader.load, for example
        ``{"player_index": 1, "trial_index": [0, 1, 2]}``.
    output_root
        Root directory for generated videos.  Defaults to ``empirical/video``.
    output_name
        File name for combined output.  ``.mp4`` is appended when omitted.
    combine
        If True, concatenate trials into one MP4.  If False, write one MP4 per
        trial under ``output_root``.
    max_trials
        Hard guard against accidental large renders.  More trials raises by
        default, or prompts when ``confirm_large=True``.
    confirm_large
        Ask for interactive confirmation instead of raising when trial count is
        above ``max_trials``.
    limit
        Optional cap after filtering.  Useful for quick test runs.
    loader_kwargs
        Extra keyword arguments for IJHCSExpDataLoader.  ``traj_hz`` defaults to
        60 and ``load_traj`` defaults to True.
    render_config
        Extra EpisodeVideoRenderer config.  ``fps`` defaults to 60.
    task_config
        Optional task config for renderer monitor/FOV geometry.  Defaults to
        ``configs/ans_agent/default.yaml``.
    """

    output_dir = DEFAULT_OUTPUT_ROOT if output_root is None else Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_specs = _normalize_filters(filters)
    loader_options = {"traj_hz": 60, "load_traj": True}
    loader_options.update(dict(loader_kwargs or {}))
    loader = IJHCSExpDataLoader(**loader_options)

    summary, trajectories = _load_filtered_trajectories(loader, load_specs)
    if limit is not None:
        limit = max(0, int(limit))
        summary = summary.iloc[:limit].reset_index(drop=True)
        trajectories = trajectories[:limit]

    if summary.empty:
        raise ValueError(f"No empirical trials matched filters: {load_specs}")

    valid_rows = []
    valid_trajs = []
    for (_, row), traj in zip(summary.iterrows(), trajectories):
        if traj is None or traj.empty:
            print(f"[EmpiricalReplay] Skipping empty trajectory: {row.get('trajectory_key', '<missing>')}")
            continue
        valid_rows.append(row)
        valid_trajs.append(traj)

    if not valid_rows:
        raise ValueError("Matched trials have no readable trajectory data.")

    n_trials = len(valid_rows)
    _guard_trial_count(n_trials, int(max_trials), bool(confirm_large))

    records = [
        _make_replay_record(row, traj)
        for row, traj in zip(valid_rows, valid_trajs)
    ]

    cfg = {"fps": 60, "freeze_ms": 0, "output_dir": str(output_dir)}
    cfg.update(dict(render_config or {}))
    renderer = EmpiricalReplayRenderer(task=_make_task(task_config), config=cfg)

    if combine:
        name = output_name or f"empirical_replay_{get_compact_timestamp_str()}_n{n_trials}.mp4"
        if not name.lower().endswith(".mp4"):
            name += ".mp4"
        return renderer.render_all(records, str(output_dir / _safe_filename(name)))

    paths = [
        str(output_dir / f"{i:04d}_{_safe_filename(record.trajectory_key)}.mp4")
        for i, record in enumerate(records)
    ]
    return renderer.render_episodes(records, output_paths=paths)


def _normalize_filters(
    filters: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    if filters is None:
        return [{}]
    if isinstance(filters, Mapping):
        return [dict(filters)]
    specs = [dict(spec) for spec in filters]
    if not specs:
        raise ValueError("filters list is empty.")
    return specs


def _make_task(task_config: Mapping[str, Any] | None = None) -> AnSTask:
    if task_config is None:
        task_config = load_yaml_config_file("ans_agent/default.yaml").get("task")
    if task_config:
        return AnSTask(config=dict(task_config))
    return AnSTask()


def _load_filtered_trajectories(
    loader: IJHCSExpDataLoader,
    specs: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    summary_parts = []
    trajectories: list[pd.DataFrame] = []
    for spec in specs:
        sub, trajs = loader.load(return_traj=True, **dict(spec))
        summary_parts.append(sub)
        trajectories.extend(trajs)

    if not summary_parts:
        return pd.DataFrame(), []
    summary = pd.concat(summary_parts, ignore_index=True) if len(summary_parts) > 1 else summary_parts[0]
    return summary.reset_index(drop=True), trajectories


def _make_replay_record(row: pd.Series, trajectory: pd.DataFrame) -> EmpiricalReplayRecord:
    summary = row.to_dict()
    trajectory_key = str(summary.get("trajectory_key", f"trial_{row.name}"))

    first = trajectory.iloc[0] if not trajectory.empty else pd.Series(dtype=float)
    player_state = _player_state_from_summary(summary)
    init_game_env = _init_env_from_summary(summary, first)
    result_report = _result_report_from_summary(summary, trajectory)

    return EmpiricalReplayRecord(
        trajectory_key=trajectory_key,
        summary=summary,
        trajectory=trajectory.reset_index(drop=True),
        player_state=player_state,
        init_game_env=init_game_env,
        result_report=result_report,
    )


def _player_state_from_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for key in (
        "expertise_group",
        "player_index",
        "sensitivity_mode",
        "target_name",
        "block_index",
        "trial_index",
    ):
        if key in summary and pd.notna(summary[key]):
            state[key] = _compact_scalar(summary[key])

    for feature in (FEATURES.GRT, FEATURES.HRT):
        if feature in summary and pd.notna(summary[feature]):
            state[feature] = float(summary[feature])

    head_keys = (FEATURES.HEAD_POS_X, FEATURES.HEAD_POS_Y, FEATURES.HEAD_POS_Z)
    if all(key in summary and pd.notna(summary[key]) for key in head_keys):
        state["head_position"] = np.array([float(summary[key]) for key in head_keys], dtype=float)
    return state


def _init_env_from_summary(summary: Mapping[str, Any], first: pd.Series) -> dict[str, Any]:
    cam_az = _coalesce_float(summary.get(FEATURES.CAM_INIT_ANGLE_AZ), first.get("camera_az_deg"), 0.0)
    cam_el = _coalesce_float(summary.get(FEATURES.CAM_INIT_ANGLE_EL), first.get("camera_el_deg"), 0.0)
    target_x = _coalesce_float(summary.get(FEATURES.TARGET_POS_MONITOR_X), first.get("target_x_mm"), 0.0)
    target_y = _coalesce_float(summary.get(FEATURES.TARGET_POS_MONITOR_Y), first.get("target_y_mm"), 0.0)

    target_az = _coalesce_float(summary.get(FEATURES.TARGET_POS_WORLD_AZ), first.get("target_az_deg"), 0.0)
    target_el = _coalesce_float(summary.get(FEATURES.TARGET_POS_WORLD_EL), first.get("target_el_deg"), 0.0)

    return {
        "camera_azel_deg": np.array([cam_az, cam_el], dtype=float),
        "target_pos_world": Convert.sphr2cart_scalar(target_az, target_el, angle_is_degree=True),
        "target_pos_monitor_mm": np.array([target_x, target_y], dtype=float),
        "target_radius_deg": _coalesce_float(summary.get(FEATURES.TARGET_RADIUS), None, 0.05),
        "target_speed_deg_s": _coalesce_float(summary.get(FEATURES.TARGET_ASPEED), None, 0.0),
        "target_orbit_axis_deg": np.array(
            [
                _coalesce_float(summary.get(FEATURES.TARGET_ORBIT_AXIS_AZ), None, 0.0),
                _coalesce_float(summary.get(FEATURES.TARGET_ORBIT_AXIS_EL), None, 0.0),
            ],
            dtype=float,
        ),
        "target_motion_dir_deg": _coalesce_float(
            summary.get(FEATURES.TARGET_MOVEMENT_MONITOR_DIRECTION),
            None,
            0.0,
        ),
    }


def _result_report_from_summary(
    summary: Mapping[str, Any],
    trajectory: pd.DataFrame,
) -> dict[str, Any]:
    t_default = 0.0
    if "timestamp_ms" in trajectory and not trajectory.empty:
        ts = _series_float(trajectory, "timestamp_ms")
        if ts.size:
            t_default = float(ts[-1] - ts[0])

    return {
        FEATURES.TCT: _coalesce_float(summary.get(FEATURES.TCT), None, t_default),
        FEATURES.RES: bool(_coalesce_float(summary.get(FEATURES.RES), summary.get(FEATURES.ACC), 0.0)),
        FEATURES.ACC: _coalesce_float(summary.get(FEATURES.ACC), summary.get(FEATURES.RES), 0.0),
        FEATURES.ERR: _coalesce_float(summary.get(FEATURES.ERR), None, 0.0),
        FEATURES.TRUNC: bool(_coalesce_float(summary.get(FEATURES.TRUNC), None, 0.0)),
    }


def _sample_gaze_mm(
    traj: pd.DataFrame,
    order: np.ndarray,
    keep: np.ndarray,
    t_src: np.ndarray,
    frame_times: np.ndarray,
) -> list[np.ndarray | None]:
    required = ("gaze_x_mm", "gaze_y_mm", "gaze_valid")
    if any(col not in traj.columns for col in required):
        return [None] * len(frame_times)

    gaze_x = _interp_series(traj, "gaze_x_mm", order, keep, t_src, frame_times)
    gaze_y = _interp_series(traj, "gaze_y_mm", order, keep, t_src, frame_times)
    valid_src = _series_float(traj, "gaze_valid")[order][keep]
    valid = np.interp(frame_times, t_src, np.nan_to_num(valid_src, nan=0.0), left=0.0, right=0.0) >= 0.5

    gaze = []
    for ok, gx, gy in zip(valid, gaze_x, gaze_y):
        if ok and np.isfinite(gx) and np.isfinite(gy):
            gaze.append(np.array([gx, gy], dtype=float))
        else:
            gaze.append(None)
    return gaze


def _interp_series(
    df: pd.DataFrame,
    column: str,
    order: np.ndarray,
    keep: np.ndarray,
    t_src: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    values = _series_float(df, column)[order][keep]
    finite = np.isfinite(values) & np.isfinite(t_src)
    if not finite.any():
        return np.zeros_like(query, dtype=float)
    return np.interp(query, t_src[finite], values[finite], left=values[finite][0], right=values[finite][-1])


def _series_float(df: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)


def _assert_required_columns(df: pd.DataFrame, columns: Sequence[str], trajectory_key: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Trajectory {trajectory_key} missing required columns: {missing}")


def _guard_trial_count(n_trials: int, max_trials: int, confirm_large: bool) -> None:
    if max_trials <= 0 or n_trials <= max_trials:
        return
    message = (
        f"Requested {n_trials} empirical replay videos/episodes, which exceeds "
        f"max_trials={max_trials}."
    )
    if not confirm_large:
        raise AssertionError(message + " Narrow filters, set limit, or pass confirm_large=True.")

    answer = input(message + " Type 'yes' to continue: ").strip().lower()
    if answer != "yes":
        raise RuntimeError("Empirical replay rendering cancelled by user.")


def _coalesce_float(primary: Any, secondary: Any, default: float) -> float:
    for value in (primary, secondary):
        parsed = _maybe_float(value)
        if parsed is not None:
            return parsed
    return float(default)


def _finite_float(value: Any, default: float) -> float:
    parsed = _maybe_float(value)
    return float(default) if parsed is None else parsed


def _maybe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _compact_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _safe_filename(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return name or "empirical_replay"


def _parse_key_value(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Filters must use key=value syntax.")
    key, value = raw.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        raise argparse.ArgumentTypeError("Filter key cannot be empty.")
    if "," in value:
        return key, [_parse_scalar(part.strip()) for part in value.split(",")]
    return key, _parse_scalar(value)


def _parse_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _main() -> None:
    parser = argparse.ArgumentParser(description="Render empirical IJHCS 60 Hz replay video.")
    parser.add_argument("--filter", action="append", type=_parse_key_value, default=[], help="Summary filter as key=value. Repeatable; comma values use isin.")
    parser.add_argument("--limit", type=int, default=3, help="Trial cap for quick test runs. Default: 3.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--separate", action="store_true", help="Write one MP4 per trial instead of one combined MP4.")
    parser.add_argument("--max-trials", type=int, default=DEFAULT_MAX_TRIALS)
    parser.add_argument("--confirm-large", action="store_true")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--freeze-ms", type=float, default=0.0)
    args = parser.parse_args()

    filter_dict = dict(args.filter)
    render_config: dict[str, Any] = {"fps": args.fps}
    if args.width is not None:
        render_config["width"] = args.width
    if args.height is not None:
        render_config["height"] = args.height
    render_config["freeze_ms"] = args.freeze_ms

    result = render_empirical_replay(
        filter_dict,
        output_root=args.output_root,
        output_name=args.output_name,
        combine=not args.separate,
        max_trials=args.max_trials,
        confirm_large=args.confirm_large,
        limit=args.limit,
        render_config=render_config,
    )
    print(f"[EmpiricalReplay] Saved: {result}")


if __name__ == "__main__":
    _main()
