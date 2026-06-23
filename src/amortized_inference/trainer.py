"""
Non-hierarchical amortizer training loop.

Trains either a density-estimation amortizer (:class:`AmortizerForTrialData`,
loss = negative log-likelihood of a normalizing flow) or a point-estimation
regressor (:class:`RegressionForTrialData`, loss = weighted MSE).

Directory layout   data/amortized_inference/models/<run_name>/ :
    config.yaml         — full inference config snapshot
    pts/
        iter_0001.pt    — checkpoint (model + optimizer + meta)
        …
    board/
        events.out.tfevents.*   — TensorBoard scalars

Code written by June-Seop Yoon
"""

from __future__ import annotations

from copy import deepcopy
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

from ..nets.amortizer import (
    AmortizerForTrialData,
    RegressionForTrialData,
)
from ..configs.constants import FOLDERS
from ..configs.loader import CFG_AMORTIZED_INFERENCE
from scipy import stats as sp_stats
from ..utils.myutils import get_compact_timestamp_str, save_dict_to_yaml
from ..utils.mytorch import get_auto_device
from ..utils.loggers import Logger
from ..utils.myplot import figure_grid, figure_save
from ..utils.scheduler import CosAnnealWR

from .dataset import TrainingDataset, ValidationDataset
from .local_utils import (
    DIR_PROJECT_ROOT,
    nll_loss,
    weighted_mse_loss,
    _auto_fill_amortizer_config,
    _build_param_info,
    _z_to_w,
)


_PARAM_SYMBOL = {
    "param_motor_noise": r"$\theta_m$",
    "param_position_noise": r"$\theta_p$",
    "param_speed_noise": r"$\theta_s$",
    "param_clock_noise": r"$\theta_c$",
    "param_succ_reward": r"$r_{succ}$",
    "param_fail_penalty": r"$r_{fail}$",
    "param_reward_decay": r"$\lambda_{succ}$",
    "param_penalty_decay": r"$\lambda_{fail}$",
}


def _param_name(param) -> str:
    return str(param.get("name", param)) if isinstance(param, dict) else str(param)


def _param_label(param_name: str) -> str:
    return _PARAM_SYMBOL.get(param_name, param_name.replace("param_", ""))


def _legacy_r2(xdata, ydata) -> float:
    x = np.asarray(xdata, dtype=float).reshape(-1)
    y = np.asarray(ydata, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    _, _, r_value, _, _ = sp_stats.linregress(x, y)
    return float(r_value ** 2)


def _draw_legacy_r2_plot(ax, xdata, ydata, xlabel: str, ylabel: str) -> float:
    x = np.asarray(xdata, dtype=float).reshape(-1)
    y = np.asarray(ydata, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(linestyle="--", linewidth=0.5, color="gray", alpha=0.5)
        ax.legend(
            [Line2D([0], [0], color="black", lw=2.5)],
            [r"$R^2=nan$"],
            fontsize=8,
            loc="lower right",
        )
        return float("nan")

    r2 = _legacy_r2(x, y)
    min_val = float(min(np.min(x), np.min(y)))
    max_val = float(max(np.max(x), np.max(y)))
    edge = (max_val - min_val) * 0.1
    if edge <= 0:
        edge = max(abs(max_val), 1.0) * 0.1
    lo, hi = min_val - edge, max_val + edge

    ax.scatter(x, y, color="black", alpha=0.3, s=5, zorder=3)
    if x.size >= 2 and np.std(x) >= 1e-12:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(lo, hi, 100)
        ax.plot(xs, slope * xs + intercept, color="red", lw=2.5, zorder=2)
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", lw=1.0, zorder=1)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(linestyle="--", linewidth=0.5, color="gray", alpha=0.5)
    ax.legend(
        [Line2D([0], [0], color="black", lw=2.5)],
        [fr"$R^2={r2:.2f}$" if np.isfinite(r2) else r"$R^2=nan$"],
        fontsize=8,
        loc="lower right",
    )
    ax.set_aspect("equal", adjustable="box")
    return r2


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class AmortizedInferenceTrainer:
    """Trains a non-hierarchical amortizer (density or point estimation).

    Parameters
    ----------
    config_preset : str
        Key into ``CFG_AMORTIZED_INFERENCE``.
    name : str, optional
        Human-readable run identifier; auto-timestamp when *None*.
    device : str
        ``'auto'``, ``'cuda'``, or ``'cpu'``.
    """

    def __init__(
        self,
        config_preset: str = "default",
        name: Optional[str] = None,
        device: str = "auto",
    ) -> None:
        cfg_all = CFG_AMORTIZED_INFERENCE.get(config_preset)
        if cfg_all is None:
            raise KeyError(
                f"Config preset '{config_preset}' not found in ans_inference configs."
            )

        self.config = deepcopy(cfg_all)
        self.config_preset = config_preset
        train_cfg = self.config["training"]
        amt_cfg = deepcopy(train_cfg["amortizer"])

        # ── Datasets ──────────────────────────────────────────────────────
        self.train_ds = TrainingDataset(
            config_preset=config_preset,
            load_existing=True,
            verbose=True,
        )
        self.valid_ds = ValidationDataset(
            config_preset=config_preset,
            load_existing=True,
            verbose=True,
        )

        n_stat = self.train_ds.n_stat_features
        n_params = len(self.config["infer_params"])
        use_traj = self.train_ds.use_traj
        n_traj = len(self.train_ds.traj_feature_cfgs) if use_traj else 0

        # ── Model ─────────────────────────────────────────────────────────
        self.point_estimation: bool = train_cfg.get("point_estimation", True)
        amt_cfg = _auto_fill_amortizer_config(
            amt_cfg, n_stat, n_params, use_traj, n_traj,
        )
        self._amt_cfg = amt_cfg

        if self.point_estimation:
            self.model = RegressionForTrialData(amt_cfg)
        else:
            self.model = AmortizerForTrialData(amt_cfg)

        if device == "auto":
            self.device = get_auto_device()
        else:
            self.device = torch.device(device)
        self.model.to(self.device)

        # ── Training hyper-parameters ─────────────────────────────────────
        self.lr: float = float(train_cfg["learning_rate"])
        self.lr_gamma: float = float(train_cfg.get("lr_gamma", 1.0))
        self.grad_clip: float = float(train_cfg.get("clipping", float("inf")))
        self.max_iter: int = int(train_cfg["max_iterations"])
        self.val_interval: int = int(train_cfg.get("valid_interval", 10))
        self.batch_size: int = int(train_cfg["batch_size"])
        self.step_per_iter: int = int(train_cfg.get("step_per_iter", 50))
        self.sim_per_param: int = int(train_cfg.get("sim_per_param", 32))

        # Per-parameter loss weights
        loss_w = amt_cfg.get("loss_weight")
        if loss_w is not None:
            self.loss_weights = torch.tensor(loss_w, dtype=torch.float32, device=self.device)
        else:
            self.loss_weights = None

        # ── Optimizer / Scheduler ─────────────────────────────────────────
        # Optimizer starts at near-zero lr; CosAnnealWR ramps up to eta_max=self.lr.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-9)
        self.scheduler = CosAnnealWR(
            self.optimizer,
            T_0=10,
            T_mult=1,
            eta_max=self.lr,
            T_up=1,
            gamma=self.lr_gamma,
        )
        self.iteration = 0

        # ── Parameter info (for physical-space denormalization in plots) ───
        self.param_info = _build_param_info(self.config["infer_params"])

        # ── Paths ─────────────────────────────────────────────────────────
        self.run_name = name or get_compact_timestamp_str()
        self.run_dir = (
            DIR_PROJECT_ROOT / FOLDERS.DATA / FOLDERS.AMORTIZER
            / FOLDERS.AMT_MODELS / self.run_name
        )
        self.pts_dir = self.run_dir / "pts"
        self.board_dir = self.run_dir / "board"
        self.results_dir = self.run_dir / "results"
        for d in (self.pts_dir, self.board_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Persist config
        save_dict_to_yaml(self.config, str(self.pts_dir / "config.yaml"))

        # Logger is lazily initialised in train() so that last_step
        # can be set correctly after resuming from a checkpoint.
        self.logger: Optional[Logger] = None

        n_model_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"\n[AmortizedInferenceTrainer] run={self.run_name}  device={self.device}\n"
            f"  model type    : {'Regression' if self.point_estimation else 'Amortizer'}\n"
            f"  model params  : {n_model_params:,}\n"
            f"  infer params  : {n_params}  ({self.config['infer_params']})\n"
            f"  stat features : {n_stat}\n"
            f"  train data    : {self.train_ds.n_param_sets} param sets"
        )

    # ── Public API ────────────────────────────────────────────────────────

    def train(
        self,
        max_iterations: Optional[int] = None,
        batch_size: Optional[int] = None,
        step_per_iter: Optional[int] = None,
        resume: bool = True,
    ) -> None:
        """Run the full training loop.

        Parameters
        ----------
        max_iterations : int, optional
            Override config ``max_iterations``.
        batch_size : int, optional
            Override config ``batch_size``.
        step_per_iter : int, optional
            Inner gradient steps per iteration.  Override config ``step_per_iter``.
        resume : bool
            When True, scan ``pts/`` for the latest checkpoint and continue.
        """
        if resume:
            self._resume_latest()

        max_iterations = max_iterations or self.max_iter
        batch_size = batch_size or self.batch_size
        step_per_iter = step_per_iter or self.step_per_iter
        start = self.iteration + 1

        # Initialise Logger with gradient-step counter so that the x-axis
        # is continuous across resume boundaries.
        last_step = self.iteration * step_per_iter
        self.logger = Logger(
            "board",
            last_step=last_step,
            board=True,
            board_path=str(self.board_dir.parent),
        )

        print(f"\n  iterations {start}..{max_iterations}  "
              f"(batch_size={batch_size}, step_per_iter={step_per_iter})")

        pbar_total = tqdm(
            range(start, max_iterations + 1),
            desc="Training",
            unit="iter",
            position=0,
        )
        for it in pbar_total:
            self.iteration = it
            step_losses = []

            # ── Inner training loop ───────────────────────────────────────
            pbar_step = tqdm(
                range(step_per_iter),
                desc=f"  Iter {it:4d}/{max_iterations}",
                leave=False,
                unit="step",
                position=1,
            )
            for step in pbar_step:
                loss = self._train_step(batch_size=batch_size)
                step_losses.append(loss)
                if step % 10 == 0:
                    self.logger.write_scalar(
                        train_loss=loss,
                        lr=self.optimizer.param_groups[0]["lr"],
                    )
                pbar_step.set_postfix(loss=f"{loss:.6f}")
                self.logger.step()
                # Step scheduler once per gradient step with fractional epoch
                self.scheduler.step((it - 1) + step / step_per_iter)
                if np.isnan(loss):
                    raise RuntimeError("NaN loss computed.")

            avg_train_loss = float(np.mean(step_losses))
            postfix = dict(
                train=f"{avg_train_loss:.5f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
            )

            if it % self.val_interval == 0:
                self._save_checkpoint(it)
                val_loss = self._validate_loss()
                recovery = self._parameter_recovery()
                self.logger.write_scalar(valid_loss=val_loss, **recovery)

                postfix["val"] = f"{val_loss:.5f}"
                tqdm.write(
                    f"  [iter {it:4d}]  train={avg_train_loss:.6f}  "
                    f"val={val_loss:.6f}  lr={self.optimizer.param_groups[0]['lr']:.2e}"
                )

            pbar_total.set_postfix(**postfix)

        self._save_checkpoint(self.iteration)
        print(f"\n[AmortizedInferenceTrainer] Training done - {self.run_name}")

    def save(self, iteration: int) -> None:
        """Alias for :meth:`_save_checkpoint`."""
        self._save_checkpoint(iteration)

    # ── Internal ──────────────────────────────────────────────────────────

    def _train_step(self, batch_size: Optional[int] = None) -> float:
        """One training step (one batch)."""
        self.model.train()
        batch_size = batch_size or self.batch_size

        params_z, stat_batch, traj_batch = self.train_ds.sample(
            batch_size=batch_size,
            sim_per_param=self.sim_per_param,
        )

        stat_t = torch.tensor(stat_batch, dtype=torch.float32, device=self.device)

        if self.point_estimation:
            # RegressionForTrialData.forward(batch_stat, batch_traj)
            pred = self.model.forward(stat_t, None)
            target = torch.tensor(params_z, dtype=torch.float32, device=self.device)
            loss = weighted_mse_loss(pred, target, self.loss_weights)
        else:
            # AmortizerForTrialData.forward(batch_param, batch_stat, batch_traj)
            params_t = torch.tensor(params_z, dtype=torch.float32, device=self.device)
            out = self.model.forward(params_t, stat_t, None)
            z, log_det_J = out
            loss = nll_loss(z, log_det_J, self.loss_weights)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip < float("inf"):
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        return loss.item()

    def _validate_loss(self) -> float:
        """Compute loss on the validation set."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        n_users = self.valid_ds.dataset.get("n_users", 0) if self.valid_ds.dataset else 0
        if n_users == 0:
            return float("nan")

        # Evaluate on all validation users
        batch_size = min(self.batch_size, n_users)
        n_batches_total = max(1, n_users // batch_size)

        with torch.no_grad():
            for _ in range(n_batches_total):
                params_z, stat_batch, traj_batch = self.valid_ds.sample(
                    n_trials=self.sim_per_param,
                    n_users=batch_size,
                )

                stat_t = torch.tensor(stat_batch, dtype=torch.float32, device=self.device)

                if self.point_estimation:
                    pred = self.model.forward(stat_t, None)
                    target = torch.tensor(params_z, dtype=torch.float32, device=self.device)
                    loss = weighted_mse_loss(pred, target, self.loss_weights)
                else:
                    params_t = torch.tensor(params_z, dtype=torch.float32, device=self.device)
                    out = self.model.forward(params_t, stat_t, None)
                    z, log_det_J = out
                    loss = nll_loss(z, log_det_J, self.loss_weights)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def _parameter_recovery(
        self,
        n_trial_list: List[int] = None,
    ) -> dict:
        """Run parameter recovery on the validation set.

        For each trial count in *n_trial_list*, infer parameters for every
        validation user, compare against ground truth, and compute R² per
        parameter.  Generates scatter-plot figures logged to TensorBoard.

        Returns a dict of  ``r2_<param>_n<n_trial>: value``  scalars.
        """
        if n_trial_list is None:
            n_trial_list = [16, 32, 64, 128]

        if self.valid_ds.dataset is None:
            return {}

        n_users = self.valid_ds.dataset.get("n_users", 0)
        if n_users == 0:
            return {}

        param_names = [_param_name(p) for p in self.config["infer_params"]]
        n_params = len(param_names)
        all_true_z = self.valid_ds.dataset["params_z"][:n_users]  # (n_users, n_params)
        all_summary = self.valid_ds.dataset["summary"][:n_users]  # (n_users, trials, n_stat)
        trials_per_user = all_summary.shape[1]

        # Per-iteration results directory (mirrors hierarchical trainer pattern)
        iter_dir = self.results_dir / f"iter_{self.iteration:04d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        result: dict = {}

        for n_trial in n_trial_list:
            if n_trial > trials_per_user:
                continue

            pred_z_list: List[np.ndarray] = []

            with torch.no_grad():
                for u in range(n_users):
                    # Sub-sample trials
                    trial_idx = np.random.choice(trials_per_user, n_trial, replace=False)
                    stat_sub = all_summary[u, trial_idx, :]  # (n_trial, n_stat)
                    stat_t = torch.tensor(
                        stat_sub, dtype=torch.float32, device=self.device
                    )  # (n_trial, n_stat) — 2D, no batch dim

                    if self.point_estimation:
                        pred = self.model.infer(stat_t, None)  # (n_params,)
                    else:
                        pred = self.model.infer(
                            stat_t, None, n_sample=200, type="mode",
                        )
                    pred_z_list.append(np.asarray(pred, dtype=np.float32).reshape(n_params))

            pred_arr = np.array(pred_z_list)  # (n_users, n_params)
            true_arr = all_true_z             # (n_users, n_params)
            pred_w_arr = _z_to_w(pred_arr, self.param_info)
            true_w_arr = _z_to_w(true_arr, self.param_info)

            # Compute R² per parameter and log
            rsq_list = []
            for p in range(n_params):
                r2 = _legacy_r2(true_w_arr[:, p], pred_w_arr[:, p])
                rsq_list.append(r2)
                result[f"Parameter_Recovery/r2_{n_trial:03d}_{param_names[p]}_sim"] = r2

            # Mean R²
            result[f"Parameter_Recovery/r2_{n_trial:03d}_mean_sim"] = float(np.mean(rsq_list))

            # Generate scatter plot figure — save to disk and tensorboard
            fig = self._plot_parameter_recovery(
                true_w_arr, pred_w_arr, param_names, rsq_list, n_trial,
            )
            figure_save(fig, str(iter_dir / f"recovery_n{n_trial}.png"))
            if self.logger is not None and self.logger.board:
                self.logger.writer.add_figure(
                    f"Parameter_Recovery/fig_n{n_trial}",
                    fig, self.logger.train_step,
                )
            plt.close(fig)


        return result

    def _plot_parameter_recovery(
        self,
        true_w: np.ndarray,
        pred_w: np.ndarray,
        param_names: List[str],
        rsq_list: List[float],
        n_trial: int,
    ):
        """Create legacy-style true-vs-inferred parameter recovery plots."""
        n_params = len(param_names)
        fig, axes = figure_grid(1, n_params, size_ax=3)
        axes = np.asarray(axes).reshape(-1)

        for p in range(n_params):
            label = _param_label(param_names[p])
            r2 = _draw_legacy_r2_plot(
                ax=axes[p],
                xdata=true_w[:, p],
                ydata=pred_w[:, p],
                xlabel=f"True {label}",
                ylabel=f"Inferred {label}",
            )
            rsq_list[p] = r2
        fig.suptitle(f"Parameter Recovery (n_trial={n_trial})", fontsize=12)
        return fig


    def _save_checkpoint(self, iteration: int) -> None:
        """Persist model, optimizer, and metadata."""
        ckpt = dict(
            iteration=iteration,
            model_state=self.model.state_dict(),
            optim_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict(),
            # Meta
            config_preset=self.config_preset,
            amortizer_config=self._amt_cfg,
            point_estimation=self.point_estimation,
            param_names=self.config["infer_params"],
            stat_feature_names=self.train_ds.stat_feature_names,
            stat_feature_cfgs=self.train_ds.stat_feature_cfgs,
            normalize_config=self.config["normalize"],
        )
        path = self.pts_dir / f"iter_{iteration:04d}.pt"
        torch.save(ckpt, str(path))

    def _resume_latest(self) -> None:
        """Load the latest checkpoint from ``pts/``."""
        ckpts = sorted(self.pts_dir.glob("iter_*.pt"))
        if not ckpts:
            return

        path = ckpts[-1]
        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        if ckpt.get("scheduler_state"):
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state"])
            except (ValueError, KeyError):
                # Scheduler type changed (e.g. ExponentialLR → CosAnnealWR);
                # ignore stale state and start fresh.
                pass
        self.iteration = int(ckpt["iteration"])
        print(f"  Resumed from {path.name}  (iteration {self.iteration})")


# ─────────────────────────────────────────────────────────────────────────────
# __main__
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train an amortized inference network.")
    parser.add_argument("--preset", type=str, default="ijhcs_gaze_abl",
                        help="Config preset name (stem of configs/ans_inference/<preset>.yaml)")
    parser.add_argument("--name", type=str, default=None,
                        help="Run name to resume.  Omit to start a fresh run.")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size from config.")
    parser.add_argument("--step-per-iter", type=int, default=None,
                        help="Inner gradient steps per iteration.")
    parser.add_argument("--max-iter", type=int, default=None,
                        help="Override max_iterations from config.")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    trainer = AmortizedInferenceTrainer(
        config_preset=args.preset,
        name=args.name,
        device=args.device,
    )
    trainer.train(
        max_iterations=args.max_iter,
        batch_size=args.batch_size,
        step_per_iter=args.step_per_iter,
        resume=args.resume,
    )
