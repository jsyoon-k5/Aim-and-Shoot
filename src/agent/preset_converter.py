"""Convert empirical summary rows into simulator episode presets.

The simulator accepts one flat ``env_preset`` dict per episode.  Keys are split
inside ``vplayer.update_env_presets`` into task presets and player-state
presets, so this module intentionally returns a single merged dict.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping

import numpy as np

from ..configs.constants import FEATURES
from ..utils.mymath import Convert, monitor_mm_to_view_angle_deg

_DEFAULT_TASK_GEOMETRY = {
    "monitor_width_mm": 531.3,
    "cam_fov_deg_width": 103.0,
}


def summary_row_to_env_preset(
    row,
    *,
    base_preset: Mapping | None = None,
    include_task: bool = True,
    include_player: bool = True,
    task_config: Mapping | None = None,
) -> dict:
    """Convert one empirical/simulator summary row to an episode preset.

    The conversion is conservative: fields are included only when the source
    row has finite values.  This keeps older empirical tables compatible while
    allowing richer IJHCS rows to replay gaze/head/reaction conditions exactly.
    """
    preset = deepcopy(dict(base_preset or {}))

    if include_task:
        preset.update(_task_preset_from_summary_row(row, task_config=task_config))

    if include_player:
        preset.update(_player_preset_from_summary_row(row))

    return preset


def summary_rows_to_env_presets(
    rows,
    *,
    base_preset: Mapping | None = None,
    include_task: bool = True,
    include_player: bool = True,
    task_config: Mapping | None = None,
) -> list[dict]:
    """Vectorized convenience wrapper for a DataFrame or iterable of rows."""
    if hasattr(rows, "iterrows"):
        iterator = (row for _, row in rows.iterrows())
    else:
        iterator = iter(rows)

    return [
        summary_row_to_env_preset(
            row,
            base_preset=base_preset,
            include_task=include_task,
            include_player=include_player,
            task_config=task_config,
        )
        for row in iterator
    ]


def ijhcs_summary_row_to_env_preset(row, *, base_preset: Mapping | None = None) -> dict:
    """Explicit IJHCS alias for call sites that want domain-specific naming."""
    return summary_row_to_env_preset(row, base_preset=base_preset)


def ijhcs_summary_to_env_presets(rows, *, base_preset: Mapping | None = None) -> list[dict]:
    """Explicit IJHCS alias for converting many rows."""
    return summary_rows_to_env_presets(rows, base_preset=base_preset)


def _task_preset_from_summary_row(row, *, task_config: Mapping | None = None) -> dict:
    preset: dict = {}

    cam_az = _get_float(row, FEATURES.CAM_INIT_ANGLE_AZ)
    cam_el = _get_float(row, FEATURES.CAM_INIT_ANGLE_EL)
    if cam_az is not None and cam_el is not None:
        preset["camera_azel_deg"] = np.array([cam_az, cam_el], dtype=np.float64)

    target_world = _target_world_from_summary_row(row)
    if target_world is not None:
        preset["target_pos_world"] = target_world
    else:
        target_monitor = _target_monitor_from_summary_row(row)
        if target_monitor is not None:
            preset["target_pos_monitor_mm"] = target_monitor

    orbit_az = _get_float(row, FEATURES.TARGET_ORBIT_AXIS_AZ)
    orbit_el = _get_float(row, FEATURES.TARGET_ORBIT_AXIS_EL)
    if orbit_az is not None and orbit_el is not None:
        preset["target_orbit_axis_deg"] = np.array([orbit_az, orbit_el], dtype=np.float64)

    speed = _get_float(row, FEATURES.TARGET_ASPEED)
    if speed is not None:
        preset["target_speed_deg_s"] = speed

    radius_deg = _get_float(row, FEATURES.TARGET_RADIUS)
    if radius_deg is None:
        radius_mm = _get_float(row, FEATURES.TARGET_RADIUS_MM)
        if radius_mm is not None:
            cfg = task_config or _DEFAULT_TASK_GEOMETRY
            radius_deg = float(
                monitor_mm_to_view_angle_deg(
                    radius_mm,
                    float(cfg["monitor_width_mm"]),
                    float(cfg["cam_fov_deg_width"]),
                )
            )
    if radius_deg is not None:
        preset["target_radius_deg"] = radius_deg

    motion_dir = _get_float(row, FEATURES.TARGET_MOVEMENT_MONITOR_DIRECTION)
    if motion_dir is not None:
        preset["target_motion_dir_deg"] = motion_dir

    return preset


def _player_preset_from_summary_row(row) -> dict:
    preset: dict = {}

    hrt = _first_float(row, FEATURES.HRT, "mouse_reaction_time_ms")
    if hrt is None:
        hrt_s = _get_float(row, "mouse_reaction_time_s")
        if hrt_s is not None:
            hrt = hrt_s * 1000.0
    if hrt is not None:
        preset["hand_reaction_time"] = hrt

    grt = _get_float(row, FEATURES.GRT)
    if grt is None:
        grt_s = _get_float(row, "gaze_reaction_time_s")
        if grt_s is not None:
            grt = grt_s * 1000.0
    if grt is not None:
        preset["gaze_reaction_time"] = grt

    head = _vector_from_columns(
        row,
        FEATURES.HEAD_POS_X,
        FEATURES.HEAD_POS_Y,
        FEATURES.HEAD_POS_Z,
    )
    if head is not None:
        preset["head_position"] = head

    gaze = _vector_from_columns(row, FEATURES.GAZE_INIT_X, FEATURES.GAZE_INIT_Y)
    if gaze is not None:
        preset["gaze_position"] = gaze

    return preset


def _target_world_from_summary_row(row) -> np.ndarray | None:
    world = _vector_from_columns(
        row,
        FEATURES.TARGET_INIT_X,
        FEATURES.TARGET_INIT_Y,
        FEATURES.TARGET_INIT_Z,
    )
    if world is not None:
        return world

    az = _get_float(row, FEATURES.TARGET_POS_WORLD_AZ)
    el = _get_float(row, FEATURES.TARGET_POS_WORLD_EL)
    if az is None or el is None:
        return None
    return np.array(
        Convert.sphr2cart_scalar(az=az, el=el, angle_is_degree=True),
        dtype=np.float64,
    )


def _target_monitor_from_summary_row(row) -> np.ndarray | None:
    return _vector_from_columns(
        row,
        FEATURES.TARGET_POS_MONITOR_X,
        FEATURES.TARGET_POS_MONITOR_Y,
    )


def _vector_from_columns(row, *columns: str) -> np.ndarray | None:
    values = [_get_float(row, col) for col in columns]
    if any(v is None for v in values):
        return None
    return np.array(values, dtype=np.float64)


def _first_float(row, *columns: str) -> float | None:
    for col in columns:
        val = _get_float(row, col)
        if val is not None:
            return val
    return None


def _get_float(row, key: str) -> float | None:
    val = _get_value(row, key)
    if val is None:
        return None
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _get_value(row, key: str):
    if isinstance(row, Mapping):
        return row.get(key, None)
    if hasattr(row, "index") and key not in row.index:
        return None
    try:
        return row[key]
    except (KeyError, TypeError, IndexError):
        return None
