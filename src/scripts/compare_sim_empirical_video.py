"""
Run IJHCS-matched simulations and render simulation-vs-empirical videos.

Default usage:
    python -m src.scripts.compare_sim_empirical_video

The script:
  1. Loads empirical IJHCS trials and corresponding inferred parameters.
  2. Runs/caches simulations under each empirical trial's task/player preset.
  3. Selects the top-N trials with matching accuracy and lowest TCT error.
  4. Renders one FHD 60 Hz side-by-side video:
       left  = simulation
       right = empirical human replay

By default, no player/expertise filter is applied.  All IJHCS trials that pass
IJHCSExpDataLoader's quality filters are simulated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..agent.preset_converter import ijhcs_summary_to_env_presets
from ..agent.simulator import AimandShootSimulator
from ..agent.task import AimandShootSpiderShotTask as AnSTask
from ..configs.constants import FEATURES
from ..datamanager.load_emp_data import IJHCSExpDataLoader
from ..utils.myutils import get_compact_timestamp_str
from ..visualizer.empirical_replay import EmpiricalReplayRenderer, _make_replay_record, _make_task
from ..visualizer.renderer import EpisodeVideoRenderer


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INFERENCE_PATH = PROJECT_ROOT / "empirical" / "ijhcs_inference_results.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "empirical" / "video" / "sim_empirical_compare"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "empirical" / "video" / "cache" / "sim_empirical_compare"
SCRIPT_CACHE_VERSION = 1

PARAM_KEYS = (
    "param_motor_noise",
    "param_position_noise",
    "param_speed_noise",
    "param_clock_noise",
    "param_succ_reward",
    "param_fail_penalty",
    "param_reward_decay",
    "param_penalty_decay",
)


@dataclass
class TrialPair:
    trajectory_key: str
    sim_record: Any
    emp_record: Any
    score_row: pd.Series


@dataclass(frozen=True)
class ComparisonLayout:
    final_width: int
    final_height: int
    panel_width: int
    panel_height: int
    left_xy: tuple[int, int]
    right_xy: tuple[int, int]
    border_px: int
    label_y: int


class SideBySideComparisonRenderer(EmpiricalReplayRenderer):
    """Render simulation and empirical replay panels into one FHD video."""

    def __init__(self, task: AnSTask, config: dict | None = None):
        super().__init__(task=task, config=config)
        hud_layer = self.aspect2d.find("**/hud_layer")
        if not hud_layer.isEmpty():
            hud_layer.hide()

    def render_pairs(
        self,
        pairs: Sequence[TrialPair],
        output_path: str | Path,
        *,
        final_width: int = 1920,
        final_height: int = 1080,
        left_label: str = "Simulation",
        right_label: str = "Empirical",
        panel_gap: int = 24,
        outer_margin: int = 24,
        label_height: int = 56,
        border_px: int = 2,
    ) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        panel_width = int(self._width)
        panel_height = int(self._height)
        final_width = int(final_width)
        final_height = int(final_height)
        layout = _comparison_layout(
            final_width=final_width,
            final_height=final_height,
            panel_width=panel_width,
            panel_height=panel_height,
            panel_gap=int(panel_gap),
            outer_margin=int(outer_margin),
            label_height=int(label_height),
            border_px=int(border_px),
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, self._fps, (final_width, final_height))
        if not writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter could not open: {output_path}")

        freeze_n = int(round(self._fps * float(self.cfg["freeze_ms"]) / 1000.0))
        total_frames = 0

        for pair_idx, pair in enumerate(tqdm(pairs, desc="Rendering comparisons")):
            sim_frames, sim_tct, sim_radius = EpisodeVideoRenderer._reconstruct_trajectory(self, pair.sim_record)
            emp_frames, emp_tct, emp_radius = EmpiricalReplayRenderer._reconstruct_trajectory(self, pair.emp_record)
            n_body = max(len(sim_frames), len(emp_frames))
            n_total = n_body + freeze_n

            sim_gaze_history: list[tuple[float, np.ndarray]] = []
            emp_gaze_history: list[tuple[float, np.ndarray]] = []

            for frame_idx in range(n_total):
                sim_panel = self._render_panel_frame(
                    pair.sim_record,
                    sim_frames,
                    sim_tct,
                    sim_radius,
                    frame_idx,
                    sim_gaze_history,
                )
                emp_panel = self._render_panel_frame(
                    pair.emp_record,
                    emp_frames,
                    emp_tct,
                    emp_radius,
                    frame_idx,
                    emp_gaze_history,
                )
                frame = _compose_comparison_frame(
                    sim_panel,
                    emp_panel,
                    layout=layout,
                    labels=(left_label, right_label),
                )
                writer.write(frame)
            total_frames += n_total

        writer.release()
        print(f"[CompareRenderer] {len(pairs)} pairs, {total_frames} frames -> {output_path}")
        return str(output_path)

    def _render_panel_frame(
        self,
        record: Any,
        frames_data: Sequence[tuple],
        tct_ms: float,
        visual_radius: float,
        frame_idx: int,
        gaze_history: list[tuple[float, np.ndarray]],
    ) -> np.ndarray:
        if not frames_data:
            raise ValueError("Cannot render an empty frame sequence.")

        idx = min(int(frame_idx), len(frames_data) - 1)
        cam_azel, tgt_world, tgt_monitor_mm, gaze_mm, t_ms = frames_data[idx]
        if frame_idx >= len(frames_data):
            t_ms = tct_ms

        self._apply_camera(cam_azel)
        self._place_target(tgt_world, visual_radius)

        if gaze_mm is not None:
            gaze_history.append((float(t_ms), np.asarray(gaze_mm, dtype=float)))
        self._update_gaze_overlay(gaze_mm, self._gaze_trail_for_time(gaze_history, float(t_ms)))
        return self._grab_frame_bgr()


def _panel_size_from_canvas(
    *,
    final_width: int,
    final_height: int | None,
    panel_gap: int,
    outer_margin: int,
    label_height: int,
    border_px: int,
) -> tuple[int, int]:
    available_width = final_width - 2 * outer_margin - panel_gap - 4 * border_px
    panel_width_by_w = available_width // 2
    if panel_width_by_w <= 0:
        raise ValueError("Canvas width is too small for the requested margins, gap, and borders.")

    panel_width = panel_width_by_w
    if final_height is not None:
        available_height = final_height - label_height - 2 * outer_margin - 2 * border_px
        if available_height <= 0:
            raise ValueError("Canvas height is too small for the requested margins, labels, and borders.")
        panel_width_by_h = int(np.floor(available_height * 16.0 / 9.0))
        panel_width = min(panel_width, panel_width_by_h)

    panel_width = max(16, panel_width)
    panel_height = int(round(panel_width * 9.0 / 16.0))
    panel_width = int(round(panel_height * 16.0 / 9.0))
    return panel_width, panel_height


def _auto_canvas_height(
    *,
    panel_height: int,
    outer_margin: int,
    label_height: int,
    border_px: int,
) -> int:
    height = 2 * outer_margin + label_height + panel_height + 2 * border_px
    return int(height + height % 2)


def _comparison_layout(
    *,
    final_width: int,
    final_height: int,
    panel_width: int,
    panel_height: int,
    panel_gap: int,
    outer_margin: int,
    label_height: int,
    border_px: int,
) -> ComparisonLayout:
    expected_height = int(round(panel_width * 9.0 / 16.0))
    if abs(panel_height - expected_height) > 1:
        raise ValueError(f"Panel must be 16:9. Got {panel_width}x{panel_height}.")

    block_width = 2 * (panel_width + 2 * border_px) + panel_gap
    block_height = panel_height + 2 * border_px
    if block_width > final_width or block_height + label_height + 2 * outer_margin > final_height:
        raise ValueError("Panel layout does not fit inside the final canvas.")

    border_x0 = (final_width - block_width) // 2
    border_y = outer_margin + label_height

    left_xy = (border_x0 + border_px, border_y + border_px)
    right_border_x = border_x0 + panel_width + 2 * border_px + panel_gap
    right_xy = (right_border_x + border_px, border_y + border_px)
    label_y = outer_margin + max(14, label_height // 2)

    return ComparisonLayout(
        final_width=final_width,
        final_height=final_height,
        panel_width=panel_width,
        panel_height=panel_height,
        left_xy=left_xy,
        right_xy=right_xy,
        border_px=border_px,
        label_y=label_y,
    )


def _compose_comparison_frame(
    sim_panel: np.ndarray,
    emp_panel: np.ndarray,
    *,
    layout: ComparisonLayout,
    labels: tuple[str, str],
) -> np.ndarray:
    frame = np.zeros((layout.final_height, layout.final_width, 3), dtype=np.uint8)
    border_color = (215, 215, 215)

    for panel, (x, y) in ((sim_panel, layout.left_xy), (emp_panel, layout.right_xy)):
        if panel.shape[:2] != (layout.panel_height, layout.panel_width):
            raise ValueError(
                f"Panel frame has shape {panel.shape[:2]}, expected "
                f"{(layout.panel_height, layout.panel_width)}."
            )
        b = layout.border_px
        cv2.rectangle(
            frame,
            (x - b, y - b),
            (x + layout.panel_width + b - 1, y + layout.panel_height + b - 1),
            border_color,
            thickness=-1,
        )
        frame[y:y + layout.panel_height, x:x + layout.panel_width] = panel

    for label, (x, _) in zip(labels, (layout.left_xy, layout.right_xy)):
        _draw_centered_text(
            frame,
            label,
            center_x=x + layout.panel_width // 2,
            baseline_y=layout.label_y,
        )
    return frame


def _draw_centered_text(frame: np.ndarray, text: str, *, center_x: int, baseline_y: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.82
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = int(center_x - text_w / 2)
    y = int(baseline_y + text_h / 2)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    output_root = Path(args.output_root)
    cache_root = Path(args.cache_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    trials = _load_empirical_summary(args)
    inference = pd.read_csv(args.inference_path)
    env_preset_list = _build_env_presets(trials, inference)

    cache_path = _cache_path(args, trials, cache_root)
    cache = None if args.refresh_cache else _load_cache(cache_path)
    if cache is None:
        simulator = AimandShootSimulator(model_name=args.model_name, ckpt=_parse_ckpt(args.ckpt))
        simulator.simulate(
            env_preset_list=env_preset_list,
            num_cpu=args.num_cpu,
            resimulate_max_num=args.resimulate_max_num,
            deterministic=not args.stochastic,
            save_trajectory=True,
            save_coarse_trajectory=False,
            verbose=True,
        )
        cache = {
            "metadata": _cache_metadata(args, trials),
            "summary": trials.reset_index(drop=True),
            "sim_records": simulator.simulation_records,
            "sim_summary": simulator.get_summarized_results(),
            "task_config": getattr(simulator.agent.game_env, "config", None),
        }
        _save_cache(cache_path, cache)
        print(f"[Cache] Saved simulations: {cache_path}")
    else:
        print(f"[Cache] Loaded simulations: {cache_path}")

    score = _score_trials(cache["summary"], cache["sim_records"])
    top_score = _select_top_trials(score, top_n=args.top_n)
    print(_format_top_table(top_score))

    pairs = _build_trial_pairs(top_score, cache["summary"], cache["sim_records"])
    panel_width, panel_height = _panel_size_from_canvas(
        final_width=int(args.width),
        final_height=None if args.height is None else int(args.height),
        panel_gap=int(args.panel_gap),
        outer_margin=int(args.outer_margin),
        label_height=int(args.label_height),
        border_px=int(args.border_px),
    )
    final_height = (
        _auto_canvas_height(
            panel_height=panel_height,
            outer_margin=int(args.outer_margin),
            label_height=int(args.label_height),
            border_px=int(args.border_px),
        )
        if args.height is None
        else int(args.height)
    )
    print(
        "[Render] "
        f"canvas={int(args.width)}x{final_height}, "
        f"panel={panel_width}x{panel_height}, "
        f"gap={int(args.panel_gap)}, border={int(args.border_px)}"
    )
    renderer = SideBySideComparisonRenderer(
        task=_task_from_cache(cache),
        config={
            "width": panel_width,
            "height": panel_height,
            "fps": int(args.fps),
            "freeze_ms": float(args.freeze_ms),
            "output_dir": str(output_root),
        },
    )

    output_name = args.output_name or f"sim_empirical_compare_{get_compact_timestamp_str()}_top{len(pairs)}.mp4"
    if not output_name.lower().endswith(".mp4"):
        output_name += ".mp4"
    output_path = output_root / output_name
    renderer.render_pairs(
        pairs,
        output_path,
        final_width=int(args.width),
        final_height=final_height,
        panel_gap=int(args.panel_gap),
        outer_margin=int(args.outer_margin),
        label_height=int(args.label_height),
        border_px=int(args.border_px),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare simulated and empirical IJHCS replay videos.")
    parser.add_argument("--model-name", default="ijhcs_default")
    parser.add_argument("--ckpt", default="latest", help="'latest', 'best', or an integer checkpoint step.")
    parser.add_argument("--inference-path", type=Path, default=DEFAULT_INFERENCE_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-name", default=None)

    parser.add_argument("--expertise-group", default=None)
    parser.add_argument("--player-index", type=int, default=None)
    parser.add_argument("--sensitivity-mode", default=None)
    parser.add_argument("--target-color-level", default=None)
    parser.add_argument("--filter", action="append", type=_parse_key_value, default=[], help="Additional summary filter as key=value. Repeatable; comma values use isin.")
    parser.add_argument("--raw", action="store_true", help="Disable IJHCS quality filtering.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on trials before simulation.")

    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--num-cpu", type=int, default=10)
    parser.add_argument("--resimulate-max-num", type=int, default=5)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic SAC actions instead of deterministic actions.")
    parser.add_argument("--refresh-cache", action="store_true", help="Ignore existing simulation cache and rerun simulations.")

    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=None, help="Final canvas height. Omit for tight auto-crop.")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--freeze-ms", type=float, default=250.0)
    parser.add_argument("--panel-gap", type=int, default=24)
    parser.add_argument("--outer-margin", type=int, default=24)
    parser.add_argument("--label-height", type=int, default=56)
    parser.add_argument("--border-px", type=int, default=2)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.width) <= 0:
        raise ValueError("--width must be positive.")
    if args.height is not None and int(args.height) <= 0:
        raise ValueError("--height must be positive when provided.")
    if int(args.fps) <= 0:
        raise ValueError("--fps must be positive.")
    if int(args.num_cpu) <= 0:
        raise ValueError("--num-cpu must be positive.")
    if int(args.top_n) <= 0:
        raise ValueError("--top-n must be positive.")
    if int(args.panel_gap) < 0 or int(args.outer_margin) < 0:
        raise ValueError("--panel-gap and --outer-margin must be non-negative.")
    if int(args.label_height) < 0:
        raise ValueError("--label-height must be non-negative.")
    if int(args.border_px) < 1:
        raise ValueError("--border-px must be at least 1.")


def _load_empirical_summary(args: argparse.Namespace) -> pd.DataFrame:
    filters: dict[str, Any] = dict(args.filter)
    if args.expertise_group is not None:
        filters["expertise_group"] = args.expertise_group
    if args.player_index is not None:
        filters["player_index"] = int(args.player_index)
    if args.sensitivity_mode is not None:
        filters["sensitivity_mode"] = args.sensitivity_mode
    if args.target_color_level is not None:
        filters["target_color_level"] = args.target_color_level

    loader = IJHCSExpDataLoader(traj_hz=60, load_traj=False, filtered=not args.raw)
    summary = loader.load(return_traj=False, **filters)
    if args.limit is not None:
        summary = summary.iloc[: max(0, int(args.limit))].reset_index(drop=True)
    if summary.empty:
        raise ValueError(f"No empirical IJHCS trials matched filters: {filters}")
    print(f"[Data] Loaded {len(summary)} empirical trials with filters: {filters}")
    return summary.reset_index(drop=True)


def _build_env_presets(trials: pd.DataFrame, inference: pd.DataFrame) -> list[dict]:
    task_presets = ijhcs_summary_to_env_presets(trials)
    return [
        {**task_preset, **_parameter_preset_for_trial(row, inference)}
        for task_preset, (_, row) in zip(task_presets, trials.iterrows())
    ]


def _parameter_preset_for_trial(row: pd.Series, inference: pd.DataFrame) -> dict[str, float]:
    required = ("expertise_group", "player_index", "sensitivity_mode", "target_color_level")
    missing = [key for key in required if key not in row.index]
    if missing:
        raise KeyError(f"Empirical summary row is missing inference keys: {missing}")

    matches = inference.copy()
    for key in required:
        if key == "player_index":
            matches = matches[matches[key].astype(int) == int(row[key])]
        else:
            matches = matches[matches[key].astype(str) == str(row[key])]

    if matches.empty:
        raise ValueError(
            "No inferred parameter row found for "
            f"expertise_group={row['expertise_group']}, "
            f"player_index={row['player_index']}, "
            f"sensitivity_mode={row['sensitivity_mode']}, "
            f"target_color_level={row['target_color_level']}."
        )

    param_row = matches.iloc[0]
    return {key: float(param_row[key]) for key in PARAM_KEYS}


def _score_trials(emp_summary: pd.DataFrame, sim_records: Sequence[Any]) -> pd.DataFrame:
    if len(emp_summary) != len(sim_records):
        raise ValueError("Empirical summary and simulation records must have the same length.")

    rows = []
    for idx, (record, (_, emp_row)) in enumerate(zip(sim_records, emp_summary.iterrows())):
        sim_tct = float(record.result_report[FEATURES.TCT])
        emp_tct = float(emp_row[FEATURES.TCT])
        sim_acc = int(bool(record.result_report[FEATURES.RES]))
        emp_acc = int(bool(emp_row.get(FEATURES.RES, emp_row.get(FEATURES.ACC, 0))))
        rows.append(
            {
                "trial_cache_index": idx,
                "trajectory_key": str(emp_row["trajectory_key"]),
                "expertise_group": emp_row.get("expertise_group", None),
                "player_index": emp_row.get("player_index", None),
                "sensitivity_mode": emp_row.get("sensitivity_mode", None),
                "target_color_level": emp_row.get("target_color_level", None),
                "emp_acc": emp_acc,
                "sim_acc": sim_acc,
                "acc_match": sim_acc == emp_acc,
                "emp_tct_ms": emp_tct,
                "sim_tct_ms": sim_tct,
                "tct_abs_error_ms": abs(sim_tct - emp_tct),
            }
        )
    return pd.DataFrame(rows)


def _select_top_trials(score: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    top_n = max(1, int(top_n))
    matched = score[score["acc_match"]].sort_values("tct_abs_error_ms", ascending=True)
    if len(matched) >= top_n:
        return matched.head(top_n).reset_index(drop=True)

    fallback = score[~score["acc_match"]].sort_values("tct_abs_error_ms", ascending=True)
    if not fallback.empty:
        print(
            f"[Select] Only {len(matched)} ACC-matched trials available; "
            f"filling remaining slots with {min(top_n - len(matched), len(fallback))} ACC-mismatched trials."
        )
    return pd.concat([matched, fallback], ignore_index=True).head(top_n).reset_index(drop=True)


def _build_trial_pairs(
    top_score: pd.DataFrame,
    emp_summary: pd.DataFrame,
    sim_records: Sequence[Any],
) -> list[TrialPair]:
    loader = IJHCSExpDataLoader(traj_hz=60, load_traj=True, filtered=False)
    pairs: list[TrialPair] = []
    for _, score_row in top_score.iterrows():
        idx = int(score_row["trial_cache_index"])
        emp_row = emp_summary.iloc[idx]
        trajectory_key = str(emp_row["trajectory_key"])
        trajectory = loader.trajectory.load(trajectory_key)
        emp_record = _make_replay_record(emp_row, trajectory)
        pairs.append(
            TrialPair(
                trajectory_key=trajectory_key,
                sim_record=sim_records[idx],
                emp_record=emp_record,
                score_row=score_row,
            )
        )
    return pairs


def _task_from_cache(cache: Mapping[str, Any]) -> AnSTask:
    task_config = cache.get("task_config")
    if task_config:
        return AnSTask(config=dict(task_config))
    return _make_task(None)


def _cache_path(args: argparse.Namespace, trials: pd.DataFrame, cache_root: Path) -> Path:
    digest = _cache_digest(args, trials)
    return cache_root / f"sim_records_{digest}.pkl"


def _cache_digest(args: argparse.Namespace, trials: pd.DataFrame) -> str:
    inference_path = Path(args.inference_path)
    stat = inference_path.stat()
    payload = {
        "version": SCRIPT_CACHE_VERSION,
        "model_name": args.model_name,
        "ckpt": str(args.ckpt),
        "stochastic": bool(args.stochastic),
        "resimulate_max_num": int(args.resimulate_max_num),
        "inference_path": str(inference_path.resolve()),
        "inference_mtime_ns": stat.st_mtime_ns,
        "inference_size": stat.st_size,
        "trajectory_keys": trials["trajectory_key"].astype(str).tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def _cache_metadata(args: argparse.Namespace, trials: pd.DataFrame) -> dict[str, Any]:
    return {
        "version": SCRIPT_CACHE_VERSION,
        "model_name": args.model_name,
        "ckpt": str(args.ckpt),
        "stochastic": bool(args.stochastic),
        "num_trials": int(len(trials)),
        "trajectory_keys": trials["trajectory_key"].astype(str).tolist(),
    }


def _load_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def _save_cache(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(dict(payload), f, protocol=pickle.HIGHEST_PROTOCOL)


def _format_top_table(top_score: pd.DataFrame) -> str:
    cols = [
        "trajectory_key",
        "acc_match",
        "emp_acc",
        "sim_acc",
        "emp_tct_ms",
        "sim_tct_ms",
        "tct_abs_error_ms",
    ]
    return "[Select] Top trials:\n" + top_score[cols].to_string(index=False)


def _parse_ckpt(value: str) -> int | str:
    text = str(value)
    return int(text) if text.isdigit() else text


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


if __name__ == "__main__":
    main()
