"""
Interactive GUI simulator for Aim-and-Shoot.

Run:
    python -m src.scripts.gui_simulator

The GUI exposes:
  - 8 model-parameter sliders.
  - Task-condition sliders with fixed or random-range mode.
  - Live monitor-space task preview.
  - A Run button that simulates N episodes.
  - In-memory video playback of simulated episodes without writing MP4 files.
"""

from __future__ import annotations

import argparse
import queue
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk
from typing import Any

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ..agent.simulator import AimandShootSimulator
from ..configs.constants import FEATURES
from ..configs.loader import load_yaml_config_file
from ..utils.mymath import monitor_mm_to_view_angle_deg
from ..visualizer.renderer import EpisodeVideoRenderer


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


@dataclass(frozen=True)
class NumericSpec:
    key: str
    label: str
    minimum: float
    maximum: float
    default: float
    resolution: float


class InMemoryEpisodeRenderer(EpisodeVideoRenderer):
    """Render EpisodeVideoRenderer frames into RGB numpy arrays instead of MP4."""

    def __init__(self, task, config: dict | None = None, *, show_hud: bool = False):
        super().__init__(task=task, config=config)
        self.show_hud = bool(show_hud)
        if not self.show_hud:
            hud_layer = self.aspect2d.find("**/hud_layer")
            if not hud_layer.isEmpty():
                hud_layer.hide()

    def render_record_to_rgb_frames(self, record) -> list[np.ndarray]:
        frames_data, tct_ms, visual_radius = self._reconstruct_trajectory(record)
        if self.show_hud:
            self._hud_set_static(record.player_state, record.init_game_env, record.result_report)

        frames: list[np.ndarray] = []
        gaze_history: list[tuple[float, np.ndarray]] = []
        for cam_azel, tgt_world, tgt_monitor_mm, gaze_mm, t_ms in frames_data:
            self._apply_camera(cam_azel)
            self._place_target(tgt_world, visual_radius)
            if self.show_hud:
                self._hud_set_dynamic(float(t_ms), float(tct_ms), tgt_monitor_mm)
            if gaze_mm is not None:
                gaze_history.append((float(t_ms), np.asarray(gaze_mm, dtype=float)))
            self._update_gaze_overlay(gaze_mm, self._gaze_trail_for_time(gaze_history, float(t_ms)))
            frames.append(self._grab_frame_bgr()[..., ::-1])

        freeze_n = int(round(self._fps * float(self.cfg["freeze_ms"]) / 1000.0))
        if freeze_n > 0 and frames_data:
            last_cam, last_tgt, _last_mon, last_gaze, _ = frames_data[-1]
            self._apply_camera(last_cam)
            self._place_target(last_tgt, visual_radius)
            self._update_gaze_overlay(last_gaze, self._gaze_trail_for_time(gaze_history, float(tct_ms)))
            still = self._grab_frame_bgr()[..., ::-1]
            frames.extend([still.copy() for _ in range(freeze_n)])
        return frames


class ParameterSlider:
    def __init__(self, parent, spec: NumericSpec, on_change):
        self.spec = spec
        self.fixed_var = tk.DoubleVar(value=float(spec.default))
        self.min_var = tk.DoubleVar(value=float(spec.minimum))
        self.max_var = tk.DoubleVar(value=float(spec.maximum))
        self.random_var = tk.BooleanVar(value=False)
        self.frame = ttk.LabelFrame(parent, text=spec.label)
        self.label_var = tk.StringVar()

        self._make_row(row=0, label="value", var=self.fixed_var, on_change=on_change)
        self._make_row(row=1, label="min", var=self.min_var, on_change=on_change)
        self._make_row(row=2, label="max", var=self.max_var, on_change=on_change)
        ttk.Checkbutton(
            self.frame,
            text="sample range",
            variable=self.random_var,
            command=lambda: self._changed(on_change),
        ).grid(row=3, column=0, sticky="w", padx=4, pady=(2, 4))
        ttk.Label(self.frame, textvariable=self.label_var).grid(row=3, column=1, columnspan=2, sticky="e", padx=4)
        self.frame.columnconfigure(1, weight=1)
        self._refresh_label()

    def _make_row(self, row: int, label: str, var: tk.DoubleVar, on_change):
        ttk.Label(self.frame, text=label, width=7).grid(row=row, column=0, sticky="w", padx=4, pady=1)
        scale = ttk.Scale(
            self.frame,
            from_=float(self.spec.minimum),
            to=float(self.spec.maximum),
            orient="horizontal",
            variable=var,
            command=lambda _value: self._changed(on_change),
        )
        scale.grid(row=row, column=1, sticky="ew", padx=(4, 6), pady=1)
        spin = ttk.Spinbox(
            self.frame,
            from_=float(self.spec.minimum),
            to=float(self.spec.maximum),
            increment=float(self.spec.resolution),
            textvariable=var,
            width=13,
            command=lambda: self._changed(on_change),
        )
        spin.grid(row=row, column=2, sticky="e", padx=(0, 4), pady=1)
        spin.bind("<Return>", lambda _event: self._changed(on_change))
        spin.bind("<FocusOut>", lambda _event: self._changed(on_change))
        return scale

    def _changed(self, on_change) -> None:
        lo = min(float(self.min_var.get()), float(self.max_var.get()))
        hi = max(float(self.min_var.get()), float(self.max_var.get()))
        if lo != float(self.min_var.get()):
            self.min_var.set(lo)
            self.max_var.set(hi)
        self._refresh_label()
        on_change()

    def _refresh_label(self) -> None:
        if self.random_var.get():
            self.label_var.set(f"U[{self.low():.6g}, {self.high():.6g}]")
        else:
            self.label_var.set(f"{self.value():.6g}")

    def value(self) -> float:
        return float(self.fixed_var.get())

    def low(self) -> float:
        return min(float(self.min_var.get()), float(self.max_var.get()))

    def high(self) -> float:
        return max(float(self.min_var.get()), float(self.max_var.get()))

    def sample(self, rng: np.random.Generator) -> float:
        if self.random_var.get():
            return float(rng.uniform(self.low(), self.high()))
        return self.value()


class TaskConditionControl:
    def __init__(self, parent, spec: NumericSpec, on_change):
        self.spec = spec
        self.fixed_var = tk.DoubleVar(value=float(spec.default))
        self.min_var = tk.DoubleVar(value=float(spec.minimum))
        self.max_var = tk.DoubleVar(value=float(spec.maximum))
        self.random_var = tk.BooleanVar(value=False)
        self.value_label_var = tk.StringVar()

        self.frame = ttk.LabelFrame(parent, text=spec.label)
        self.fixed_scale = self._make_scale(self.fixed_var, 0, "value", on_change)
        self.min_scale = self._make_scale(self.min_var, 1, "min", on_change)
        self.max_scale = self._make_scale(self.max_var, 2, "max", on_change)
        self.random_check = ttk.Checkbutton(
            self.frame,
            text="sample range",
            variable=self.random_var,
            command=lambda: self._changed(on_change),
        )
        self.random_check.grid(row=3, column=0, sticky="w", padx=4, pady=(2, 4))
        ttk.Label(self.frame, textvariable=self.value_label_var).grid(row=3, column=1, columnspan=2, sticky="e")
        self.frame.columnconfigure(1, weight=1)
        self._refresh_label()

    def _make_scale(self, var: tk.DoubleVar, row: int, label: str, on_change):
        ttk.Label(self.frame, text=label, width=7).grid(row=row, column=0, sticky="w", padx=4)
        scale = ttk.Scale(
            self.frame,
            from_=float(self.spec.minimum),
            to=float(self.spec.maximum),
            orient="horizontal",
            variable=var,
            command=lambda _value: self._changed(on_change),
        )
        scale.grid(row=row, column=1, sticky="ew", padx=(4, 6))
        spin = ttk.Spinbox(
            self.frame,
            from_=float(self.spec.minimum),
            to=float(self.spec.maximum),
            increment=float(self.spec.resolution),
            textvariable=var,
            width=13,
            command=lambda: self._changed(on_change),
        )
        spin.grid(row=row, column=2, sticky="e", padx=(0, 4), pady=1)
        spin.bind("<Return>", lambda _event: self._changed(on_change))
        spin.bind("<FocusOut>", lambda _event: self._changed(on_change))
        return scale

    def _changed(self, on_change) -> None:
        lo = min(float(self.min_var.get()), float(self.max_var.get()))
        hi = max(float(self.min_var.get()), float(self.max_var.get()))
        if lo != float(self.min_var.get()):
            self.min_var.set(lo)
            self.max_var.set(hi)
        self._refresh_label()
        on_change()

    def _refresh_label(self) -> None:
        if self.random_var.get():
            self.value_label_var.set(f"U[{self.low():.6g}, {self.high():.6g}]")
        else:
            self.value_label_var.set(f"{self.fixed_value():.6g}")

    def fixed_value(self) -> float:
        return float(self.fixed_var.get())

    def low(self) -> float:
        return min(float(self.min_var.get()), float(self.max_var.get()))

    def high(self) -> float:
        return max(float(self.min_var.get()), float(self.max_var.get()))

    def preview_value(self) -> float:
        return self.fixed_value()

    def sample(self, rng: np.random.Generator) -> float:
        if self.random_var.get():
            return float(rng.uniform(self.low(), self.high()))
        return self.fixed_value()


class ScrollFrame(ttk.Frame):
    def __init__(self, parent, *, width: int = 640):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, width=int(width))
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas)
        self.content.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.canvas.bind("<Configure>", self._sync_content_width)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    def _sync_content_width(self, event) -> None:
        self.canvas.itemconfigure(self.window_id, width=max(1, int(event.width)))


class InteractiveSimulatorGUI:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = tk.Tk()
        self.root.title("Aim-and-Shoot Interactive Simulator")
        self.root.geometry(f"{int(self.args.window_width)}x{int(self.args.window_height)}")
        self.root.minsize(1500, 900)
        self.agent_config = load_yaml_config_file("ans_agent/default.yaml")
        self.task_config = self.agent_config["task"]
        self.player_config = self.agent_config["player"]

        self.simulator: AimandShootSimulator | None = None
        self.renderer: InMemoryEpisodeRenderer | None = None
        self.records: list[Any] = []
        self.current_frames: list[np.ndarray] = []
        self.current_episode_idx = 0
        self.current_frame_idx = 0
        self.playing = False
        self._updating_frame_slider = False
        self.worker_queue: queue.Queue = queue.Queue()
        self.worker_thread: threading.Thread | None = None

        self.param_controls: dict[str, ParameterSlider] = {}
        self.task_controls: dict[str, TaskConditionControl] = {}

        self._build_ui()
        self._update_task_preview()
        self._poll_worker_queue()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def run(self) -> None:
        self.root.mainloop()

    def _build_ui(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        controls = ScrollFrame(self.root, width=int(self.args.control_width))
        controls.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        viewer = ttk.Frame(self.root)
        viewer.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=8)
        viewer.columnconfigure(0, weight=1)
        viewer.rowconfigure(0, weight=1)

        self._build_controls(controls.content)
        self._build_viewer(viewer)

    def _build_controls(self, parent) -> None:
        runtime = ttk.LabelFrame(parent, text="Runtime")
        runtime.pack(fill="x", pady=(0, 8))

        self.model_name_var = tk.StringVar(value=self.args.model_name)
        self.ckpt_var = tk.StringVar(value=str(self.args.ckpt))
        self.episodes_var = tk.IntVar(value=int(self.args.episodes))
        self.cpu_var = tk.IntVar(value=int(self.args.num_cpu))
        self.seed_var = tk.StringVar(value="")
        self.deterministic_var = tk.BooleanVar(value=True)

        self._entry_row(runtime, 0, "model", self.model_name_var)
        self._entry_row(runtime, 1, "ckpt", self.ckpt_var)
        self._spin_row(runtime, 2, "episodes", self.episodes_var, 1, 1000)
        self._spin_row(runtime, 3, "cpus", self.cpu_var, 1, 32)
        self._entry_row(runtime, 4, "seed", self.seed_var)
        ttk.Checkbutton(runtime, text="deterministic SAC action", variable=self.deterministic_var).grid(
            row=5, column=0, columnspan=2, sticky="w", padx=4, pady=2
        )

        buttons = ttk.Frame(runtime)
        buttons.grid(row=6, column=0, columnspan=2, sticky="ew", padx=4, pady=(6, 4))
        ttk.Button(buttons, text="Draw Sample", command=self._sample_preview).pack(side="left", padx=(0, 4))
        ttk.Button(buttons, text="Run", command=self._run_simulation_clicked).pack(side="left", padx=(0, 4))

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(runtime, textvariable=self.status_var, wraplength=460).grid(
            row=7, column=0, columnspan=2, sticky="ew", padx=4, pady=(4, 4)
        )

        params = ttk.LabelFrame(parent, text="Model Parameters")
        params.pack(fill="x", pady=(0, 8))
        for spec in self._parameter_specs():
            control = ParameterSlider(params, spec, on_change=self._update_status_from_controls)
            control.frame.pack(fill="x", padx=4, pady=2)
            self.param_controls[spec.key] = control

        task = ttk.LabelFrame(parent, text="Task Conditions")
        task.pack(fill="x", pady=(0, 8))
        for spec in self._task_specs():
            control = TaskConditionControl(task, spec, on_change=self._update_task_preview)
            control.frame.pack(fill="x", padx=4, pady=4)
            self.task_controls[spec.key] = control

        summary = ttk.LabelFrame(parent, text="Episode Summary")
        summary.pack(fill="x", pady=(0, 8))
        self.summary_var = tk.StringVar(value="No simulation yet.")
        ttk.Label(summary, textvariable=self.summary_var, wraplength=460, justify="left").pack(fill="x", padx=4, pady=4)

    def _build_viewer(self, parent) -> None:
        parent.rowconfigure(0, weight=1)
        parent.rowconfigure(1, weight=0)

        fig = Figure(figsize=(11.5, 8.6), dpi=100)
        gs = fig.add_gridspec(2, 1, height_ratios=[4.0, 1.25])
        self.ax_video = fig.add_subplot(gs[0])
        self.ax_preview = fig.add_subplot(gs[1])
        fig.tight_layout(pad=1.4)

        self.ax_video.set_title("Simulation video")
        self.ax_video.axis("off")
        blank = np.zeros((int(self.args.render_height), int(self.args.render_width), 3), dtype=np.uint8)
        self.video_artist = self.ax_video.imshow(blank)

        self.canvas = FigureCanvasTkAgg(fig, master=parent)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        playback = ttk.LabelFrame(parent, text="Video playback")
        playback.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        playback.columnconfigure(4, weight=1)

        ttk.Button(playback, text="Prev episode", command=lambda: self._load_episode(self.current_episode_idx - 1)).grid(row=0, column=0, padx=4, pady=4)
        ttk.Button(playback, text="Next episode", command=lambda: self._load_episode(self.current_episode_idx + 1)).grid(row=0, column=1, padx=4, pady=4)
        self.play_button_text = tk.StringVar(value="Play")
        ttk.Button(playback, textvariable=self.play_button_text, command=self._toggle_play).grid(row=0, column=2, padx=4, pady=4)
        ttk.Button(playback, text="Restart", command=lambda: self._show_frame(0)).grid(row=0, column=3, padx=4, pady=4)

        self.frame_var = tk.DoubleVar(value=0.0)
        self.frame_scale = ttk.Scale(
            playback,
            from_=0,
            to=1,
            orient="horizontal",
            variable=self.frame_var,
            command=self._seek_frame_from_slider,
        )
        self.frame_scale.grid(row=0, column=4, sticky="ew", padx=(8, 4), pady=4)
        self.frame_label_var = tk.StringVar(value="frame 0 / 0")
        ttk.Label(playback, textvariable=self.frame_label_var, width=18).grid(row=0, column=5, padx=4, pady=4)

    def _entry_row(self, parent, row: int, label: str, var: tk.StringVar) -> None:
        ttk.Label(parent, text=label, width=12).grid(row=row, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        parent.columnconfigure(1, weight=1)

    def _spin_row(self, parent, row: int, label: str, var: tk.IntVar, lo: int, hi: int) -> None:
        ttk.Label(parent, text=label, width=12).grid(row=row, column=0, sticky="w", padx=4, pady=2)
        ttk.Spinbox(parent, from_=lo, to=hi, textvariable=var, width=10).grid(
            row=row, column=1, sticky="w", padx=4, pady=2
        )

    def _parameter_specs(self) -> list[NumericSpec]:
        specs = []
        for key in PARAM_KEYS:
            cfg = self.player_config[key]
            lo = float(cfg.get("min", 0.0))
            hi = float(cfg.get("max", 1.0))
            default = float(cfg.get("mean", 0.5 * (lo + hi)))
            resolution = 0.001 if hi <= 1.0 else 0.1
            specs.append(NumericSpec(key, key, lo, hi, default, resolution))
        return specs

    def _task_specs(self) -> list[NumericSpec]:
        task = self.task_config
        half_w = float(task["monitor_width_mm"]) / 2.0
        half_h = float(task["monitor_height_mm"]) / 2.0
        radius_cfg = task.get("target_radius_mm_range")
        if radius_cfg is None:
            radius_cfg = {"min": 4.0, "max": 13.0}
        speed_cfg = task["target_aspeed_deg_s_range"]
        return [
            NumericSpec("target_radius_mm", "target radius (mm)", float(radius_cfg["min"]), float(radius_cfg["max"]), 0.5 * (float(radius_cfg["min"]) + float(radius_cfg["max"])), 0.1),
            NumericSpec("target_speed_deg_s", "target speed (deg/s)", float(speed_cfg["min"]), float(speed_cfg["max"]), 0.5 * (float(speed_cfg["min"]) + float(speed_cfg["max"])), 0.1),
            NumericSpec("target_motion_dir_deg", "motion direction (deg)", 0.0, 360.0, 0.0, 1.0),
            NumericSpec("target_x_mm", "target x (mm)", -0.85 * half_w, 0.85 * half_w, 0.25 * half_w, 1.0),
            NumericSpec("target_y_mm", "target y (mm)", -0.85 * half_h, 0.85 * half_h, 0.25 * half_h, 1.0),
        ]

    def _sample_preview(self) -> None:
        try:
            rng = np.random.default_rng(self._seed_or_none())
        except ValueError as exc:
            self.status_var.set(str(exc))
            return
        for control in self.param_controls.values():
            if control.random_var.get():
                control.fixed_var.set(control.sample(rng))
                control._refresh_label()
        for control in self.task_controls.values():
            if control.random_var.get():
                control.fixed_var.set(control.sample(rng))
                control._refresh_label()
        self._update_task_preview()

    def _task_preview_values(self) -> dict[str, float]:
        return {key: control.preview_value() for key, control in self.task_controls.items()}

    def _sample_task_values(self, rng: np.random.Generator) -> dict[str, float]:
        return {key: control.sample(rng) for key, control in self.task_controls.items()}

    def _build_env_preset(self, rng: np.random.Generator) -> dict[str, Any]:
        task_values = self._sample_task_values(rng)
        radius_deg = float(
            monitor_mm_to_view_angle_deg(
                task_values["target_radius_mm"],
                float(self.task_config["monitor_width_mm"]),
                float(self.task_config["cam_fov_deg_width"]),
            )
        )
        preset: dict[str, Any] = {
            "target_pos_monitor_mm": np.array(
                [task_values["target_x_mm"], task_values["target_y_mm"]],
                dtype=float,
            ),
            "target_speed_deg_s": float(task_values["target_speed_deg_s"]),
            "target_motion_dir_deg": float(task_values["target_motion_dir_deg"]),
            "target_radius_deg": radius_deg,
        }
        preset.update({key: control.sample(rng) for key, control in self.param_controls.items()})
        return preset

    def _update_task_preview(self) -> None:
        values = self._task_preview_values() if self.task_controls else {}
        if not values:
            return

        half_w = float(self.task_config["monitor_width_mm"]) / 2.0
        half_h = float(self.task_config["monitor_height_mm"]) / 2.0
        x = values["target_x_mm"]
        y = values["target_y_mm"]
        r = values["target_radius_mm"]
        direction = np.radians(values["target_motion_dir_deg"])
        speed = values["target_speed_deg_s"]

        self.ax_preview.clear()
        self.ax_preview.set_title("Task preview")
        self.ax_preview.set_xlim(-half_w, half_w)
        self.ax_preview.set_ylim(-half_h, half_h)
        self.ax_preview.set_aspect("equal", adjustable="box")
        self.ax_preview.axhline(0.0, color="0.7", linewidth=0.8)
        self.ax_preview.axvline(0.0, color="0.7", linewidth=0.8)
        self.ax_preview.add_patch(
            plt_rectangle((-half_w, -half_h), 2 * half_w, 2 * half_h, fill=False, edgecolor="0.4")
        )

        x_control = self.task_controls["target_x_mm"]
        y_control = self.task_controls["target_y_mm"]
        if x_control.random_var.get() or y_control.random_var.get():
            x0, x1 = x_control.low(), x_control.high()
            y0, y1 = y_control.low(), y_control.high()
            self.ax_preview.add_patch(
                plt_rectangle((x0, y0), x1 - x0, y1 - y0, fill=True, color="tab:blue", alpha=0.12)
            )

        circle = plt_circle((x, y), r, color="tab:blue", fill=False, linewidth=2)
        self.ax_preview.add_patch(circle)
        radius_control = self.task_controls["target_radius_mm"]
        if radius_control.random_var.get():
            self.ax_preview.add_patch(
                plt_circle((x, y), radius_control.low(), color="tab:blue", fill=False, linewidth=1, linestyle=":")
            )
            self.ax_preview.add_patch(
                plt_circle((x, y), radius_control.high(), color="tab:blue", fill=False, linewidth=1, linestyle=":")
            )
        arrow_len = max(20.0, min(90.0, 2.0 * speed + 20.0))
        self.ax_preview.arrow(
            x,
            y,
            arrow_len * np.cos(direction),
            arrow_len * np.sin(direction),
            width=1.4,
            head_width=7.0,
            head_length=10.0,
            color="tab:orange",
            length_includes_head=True,
        )
        direction_control = self.task_controls["target_motion_dir_deg"]
        if direction_control.random_var.get():
            for bound in (direction_control.low(), direction_control.high()):
                theta = np.radians(bound)
                self.ax_preview.arrow(
                    x,
                    y,
                    0.65 * arrow_len * np.cos(theta),
                    0.65 * arrow_len * np.sin(theta),
                    width=0.7,
                    head_width=5.0,
                    head_length=7.0,
                    color="tab:orange",
                    alpha=0.35,
                    length_includes_head=True,
                )
        self.ax_preview.scatter([0.0], [0.0], marker="+", color="limegreen", s=90)
        self.ax_preview.set_xlabel("monitor x (mm)")
        self.ax_preview.set_ylabel("monitor y (mm)")
        self.ax_preview.grid(True, color="0.88", linewidth=0.6)
        self.canvas.draw_idle()
        self._update_status_from_controls()

    def _update_status_from_controls(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            return
        self.status_var.set("Ready. Adjust sliders or run simulation.")

    def _run_simulation_clicked(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            self.status_var.set("Simulation is already running.")
            return
        try:
            presets = self._build_presets_for_run()
        except ValueError as exc:
            self.status_var.set(str(exc))
            return
        self.status_var.set(f"Running {len(presets)} episodes...")
        self.summary_var.set("Simulation running...")
        self.worker_thread = threading.Thread(target=self._simulation_worker, args=(presets,), daemon=True)
        self.worker_thread.start()

    def _build_presets_for_run(self) -> list[dict[str, Any]]:
        n_episodes = max(1, int(self.episodes_var.get()))
        rng = np.random.default_rng(self._seed_or_none())
        return [self._build_env_preset(rng) for _ in range(n_episodes)]

    def _simulation_worker(self, presets: list[dict[str, Any]]) -> None:
        try:
            simulator = AimandShootSimulator(model_name=self.model_name_var.get(), ckpt=self._parse_ckpt())
            simulator.simulate(
                env_preset_list=presets,
                num_cpu=max(1, int(self.cpu_var.get())),
                resimulate_max_num=3,
                deterministic=bool(self.deterministic_var.get()),
                save_trajectory=True,
                save_coarse_trajectory=False,
                verbose=True,
            )
            self.worker_queue.put(("done", simulator))
        except Exception as exc:
            self.worker_queue.put(("error", exc))

    def _poll_worker_queue(self) -> None:
        try:
            kind, payload = self.worker_queue.get_nowait()
        except queue.Empty:
            self.root.after(200, self._poll_worker_queue)
            return

        if kind == "done":
            self.simulator = payload
            if self.renderer is not None:
                self.renderer.destroy()
                self.renderer = None
            self.records = list(payload.simulation_records)
            self.current_episode_idx = 0
            self.status_var.set(f"Simulation complete: {len(self.records)} episodes.")
            self._load_episode(0)
        else:
            self.status_var.set(f"Simulation failed: {payload}")
            self.summary_var.set(str(payload))
        self.root.after(200, self._poll_worker_queue)

    def _load_episode(self, index: int) -> None:
        if not self.records:
            self.status_var.set("No simulated episodes to load.")
            return
        self.playing = False
        if hasattr(self, "play_button_text"):
            self.play_button_text.set("Play")
        index = int(np.clip(index, 0, len(self.records) - 1))
        self.current_episode_idx = index
        record = self.records[index]
        self.status_var.set(f"Rendering episode {index + 1}/{len(self.records)} in memory...")
        self.root.update_idletasks()
        self.current_frames = self._renderer().render_record_to_rgb_frames(record)
        self.current_frame_idx = 0
        self._show_frame(0)
        self._update_summary(record, index)
        self.status_var.set(f"Loaded episode {index + 1}/{len(self.records)}.")

    def _renderer(self) -> InMemoryEpisodeRenderer:
        if self.renderer is None:
            if self.simulator is None:
                raise RuntimeError("Simulator must be available before renderer construction.")
            self.renderer = InMemoryEpisodeRenderer(
                task=self.simulator.agent.game_env,
                config={
                    "width": int(self.args.render_width),
                    "height": int(self.args.render_height),
                    "fps": int(self.args.fps),
                    "freeze_ms": int(self.args.freeze_ms),
                },
                show_hud=False,
            )
        return self.renderer

    def _show_frame(self, frame_idx: int) -> None:
        if not self.current_frames:
            return
        frame_idx = int(np.clip(frame_idx, 0, len(self.current_frames) - 1))
        self.current_frame_idx = frame_idx
        if hasattr(self, "frame_scale"):
            self.frame_scale.configure(to=max(1, len(self.current_frames) - 1))
            self._updating_frame_slider = True
            self.frame_var.set(float(frame_idx))
            self._updating_frame_slider = False
            self.frame_label_var.set(f"frame {frame_idx + 1} / {len(self.current_frames)}")
        self.video_artist.set_data(self.current_frames[frame_idx])
        self.ax_video.set_title(
            f"Episode {self.current_episode_idx + 1}/{len(self.records)} - "
            f"frame {frame_idx + 1}/{len(self.current_frames)}"
        )
        self.canvas.draw_idle()

    def _seek_frame_from_slider(self, value) -> None:
        if self._updating_frame_slider or not self.current_frames:
            return
        self.playing = False
        if hasattr(self, "play_button_text"):
            self.play_button_text.set("Play")
        self._show_frame(int(round(float(value))))

    def _toggle_play(self) -> None:
        if not self.current_frames:
            self.status_var.set("Run a simulation first.")
            return
        self.playing = not self.playing
        if hasattr(self, "play_button_text"):
            self.play_button_text.set("Pause" if self.playing else "Play")
        if self.playing:
            self._play_tick()

    def _play_tick(self) -> None:
        if not self.playing or not self.current_frames:
            return
        next_idx = self.current_frame_idx + 1
        if next_idx >= len(self.current_frames):
            self.playing = False
            if hasattr(self, "play_button_text"):
                self.play_button_text.set("Play")
            return
        self._show_frame(next_idx)
        self.root.after(max(1, int(1000 / int(self.args.fps))), self._play_tick)

    def _update_summary(self, record, index: int) -> None:
        result = record.result_report
        summary = (
            f"Episode {index + 1}/{len(self.records)}\n"
            f"result: {'HIT' if result.get(FEATURES.RES, False) else 'MISS'}\n"
            f"TCT: {float(result.get(FEATURES.TCT, np.nan)):.1f} ms\n"
            f"error: {float(result.get(FEATURES.ERR, np.nan)):.2f} mm\n"
            f"max hand speed: {float(result.get(FEATURES.MAX_HSPD, np.nan)):.1f} mm/s"
        )
        self.summary_var.set(summary)

    def _seed_or_none(self) -> int | None:
        text = self.seed_var.get().strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError("Seed must be blank or an integer.") from exc

    def _parse_ckpt(self) -> str | int:
        text = self.ckpt_var.get().strip()
        return int(text) if text.isdigit() else text

    def _on_close(self) -> None:
        self.playing = False
        try:
            if self.renderer is not None:
                self.renderer.destroy()
        finally:
            self.root.destroy()


def plt_rectangle(*args, **kwargs):
    from matplotlib.patches import Rectangle
    return Rectangle(*args, **kwargs)


def plt_circle(*args, **kwargs):
    from matplotlib.patches import Circle
    return Circle(*args, **kwargs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI Aim-and-Shoot simulator.")
    parser.add_argument("--model-name", default="ijhcs_default")
    parser.add_argument("--ckpt", default="latest")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--num-cpu", type=int, default=1)
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--window-width", type=int, default=1800)
    parser.add_argument("--window-height", type=int, default=1060)
    parser.add_argument("--control-width", type=int, default=680)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--freeze-ms", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    gui = InteractiveSimulatorGUI(_parse_args())
    gui.run()


if __name__ == "__main__":
    main()
