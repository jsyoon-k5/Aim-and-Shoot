import os, datetime
import numpy as np
import torch
from scipy import stats as scipy_stats
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

class Logger(object):
    def __init__(self, name, last_step=0, board=True, board_path="./data/board"):
        self.name = name
        self.train_step = last_step
        self.board = board
        if self.board:
            self._set_tb_writer(board_path)

    def _set_tb_writer(self, board_path):
        os.makedirs(f"{board_path}/{self.name}", exist_ok=True)
        # datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        self.writer = SummaryWriter(f"{board_path}/{self.name}")

    def step(self):
        self.train_step += 1

    def write_scalar(self, verbose=False, **kwargs):
        if self.board:
            for key, value in kwargs.items():
                self.writer.add_scalar(key, value, self.train_step)


class TensorboardStdCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardStdCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        actor = self.model.policy.actor

        # Ensure that current_log_std is a tensor
        if actor.current_log_std is not None:
            std = torch.exp(actor.current_log_std).mean().item()
            self.logger.record('rollout/action_std', std)
        else:
            self.logger.record('rollout/action_std', float('nan'))

        if "infos" in self.locals and self.locals["infos"]:
            info = self.locals["infos"][-1]
        return True


class TruncationLoggingEvalCallback(EvalCallback):
    """EvalCallback that additionally logs the episode truncation rate to TensorBoard.

    Adds ``eval/truncation_rate``: fraction of evaluation episodes that ended
    in truncation (time-limit / target escaped) rather than a shot outcome.

    Adds ``eval/tct_lognorm_shapiro_w`` and ``eval/tct_lognorm_ks_stat``:
    goodness-of-fit of the successful-episode TCT distribution to lognormal.
      - Shapiro-Wilk W on log(TCT): W → 1.0 means strong lognormal fit.
      - KS statistic against the MLE-fitted lognormal CDF: smaller is better.
    Only computed from non-truncated episodes (both hit and miss).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._truncation_buffer: list = []
        self._tct_buffer: list = []         # TCT (ms) for successful episodes

    # Called by evaluate_policy for every completed episode.
    # super() handles is_success; we additionally record is_truncated.
    def _log_success_callback(self, locals_: dict, globals_: dict) -> None:
        super()._log_success_callback(locals_, globals_)
        if locals_["done"]:
            info = locals_["info"]
            is_trunc = bool(info.get("is_truncated", False))
            self._truncation_buffer.append(float(is_trunc))
            if not is_trunc:
                tct = info.get("time", None)
                if tct is not None and float(tct) > 0:
                    self._tct_buffer.append(float(tct))

    def _on_step(self) -> bool:
        # Clear the buffer just before the eval run so old data never leaks in.
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._truncation_buffer = []
            self._tct_buffer = []

        result = super()._on_step()  # runs eval, populates buffers

        # Log truncation rate after the eval (super already called logger.dump).
        if len(self._truncation_buffer) > 0:
            trunc_rate = float(np.mean(self._truncation_buffer))
            self.logger.record("eval/truncation_rate", trunc_rate)

        # Log lognormal goodness-of-fit for TCT.
        tct = np.array(self._tct_buffer, dtype=float)
        if len(tct) >= 8:   # need enough samples for meaningful statistics
            log_tct = np.log(tct)
            # Shapiro-Wilk on log(TCT): W → 1 means lognormal fit is good.
            sw_w, _ = scipy_stats.shapiro(log_tct)
            # KS test against MLE-fitted lognormal.
            mu, sigma = float(log_tct.mean()), float(log_tct.std(ddof=1))
            ks_stat, _ = scipy_stats.kstest(
                log_tct, "norm", args=(mu, sigma)
            )
            self.logger.record("eval/tct_lognorm_shapiro_w", float(sw_w))
            self.logger.record("eval/tct_lognorm_ks_stat", float(ks_stat))
            self.logger.record("eval/tct_n_success", len(tct))

        if len(self._truncation_buffer) > 0 or len(tct) >= 8:
            self.logger.dump(self.num_timesteps)

        self._truncation_buffer = []
        self._tct_buffer = []

        return result