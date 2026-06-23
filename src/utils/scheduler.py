"""Learning-rate scheduler factory for policy manifold training.

Scheduler type is specified in ``train.scheduler.type`` of the distillation
config YAML.  Supported types:

    none          – constant LR (no scheduler)
    cosine        – CosineAnnealingLR: smoothly decays from ``lr`` to ``eta_min``
    step          – StepLR: multiplicative decay every ``step_size`` epochs
    exponential   – ExponentialLR: multiplicative decay every epoch

Example YAML
------------
::

    train:
      scheduler:
        type: cosine
        eta_min: 1.0e-5

Notes
-----
``last_epoch`` semantics (PyTorch convention):
  - In the scheduler constructor, ``__init__`` always calls ``step()`` once
    internally, which increments ``last_epoch`` by 1 and applies ``get_lr()``.
  - For a **fresh** run, pass ``last_epoch=-1`` (default):
    ``__init__`` → ``step()`` → ``last_epoch=0`` → LR = initial (cosine peak).
  - When **resuming** from epoch ``N`` (i.e. ``N`` explicit ``step()`` calls
    have already happened), pass ``last_epoch = N - 1``:
    ``__init__`` → ``step()`` → ``last_epoch=N`` → LR = cosine(N/T_max),
    which exactly matches the LR stored in the optimizer checkpoint.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import optim
import math
from torch.optim.lr_scheduler import _LRScheduler


def build_scheduler(
	optimizer: optim.Optimizer,
	config: Optional[dict],
	total_epochs: int,
	last_epoch: int = -1,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
	"""Instantiate a PyTorch LR scheduler from a config dict.

	Parameters
	----------
	optimizer : torch.optim.Optimizer
	config : dict | None
	    Contents of the ``train.scheduler`` YAML key.
	    Required key:
	      ``type`` : str — one of ``"none"`` | ``"cosine"`` | ``"step"`` |
	      ``"exponential"``
	    Type-specific optional keys:

	    ``cosine``
	      ``eta_min``   – minimum LR at end of cosine curve (default ``1e-6``)
	    ``cosine_wr``   – :class:`CosAnnealWR`: cosine annealing with warm restarts,
	      per-cycle linear warmup, and decaying peak LR
	      ``t_0``       – initial restart period in epochs (default ``20``)
	      ``t_mult``    – restart period multiplier per cycle, int ≥ 1 (default ``1``)
	      ``t_up``      – linear warmup epochs per cycle (default ``5``)
	      ``gamma``     – peak-LR multiplicative factor per cycle, 1.0 = no decay (default ``1.0``)
	    ``step``
	      ``step_size`` – decay period in epochs (default ``10``)
	      ``gamma``     – multiplicative factor (default ``0.5``)
	    ``exponential``
	      ``gamma``     – multiplicative factor per epoch (default ``0.95``)
	total_epochs : int
	    Full number of training epochs in this run.  Used as ``T_max`` for
	    :class:`~torch.optim.lr_scheduler.CosineAnnealingLR`.
	last_epoch : int
	    Passed straight to the scheduler constructor.  For a **fresh** run use
	    ``-1`` (default).  When **resuming** from epoch ``N``, pass ``N - 1``
	    so that the scheduler's first internal ``step()`` in ``__init__``
	    restores the LR to exactly the value it had at epoch ``N``, and the
	    next explicit ``step()`` (at end of epoch ``N+1``) advances it
	    correctly.

	Returns
	-------
	torch.optim.lr_scheduler.LRScheduler or None
	"""
	if config is None:
		return None

	stype = str(config.get("type", "none")).lower()
	if stype == "none":
		return None

	if stype == "cosine":
		eta_min = float(config.get("eta_min", 1e-6))
		return torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer,
			T_max=total_epochs,
			eta_min=eta_min,
			last_epoch=last_epoch,
		)

	if stype == "cosine_wr":
		# eta_max defaults to current (initial) LR so the YAML lr key stays
		# the single source of truth for the peak learning rate.
		eta_max = float(config.get("eta_max", optimizer.param_groups[0]["lr"]))
		t_0    = int(config.get("t_0",    20))
		t_mult = int(config.get("t_mult",  1))
		t_up   = int(config.get("t_up",    5))
		gamma  = float(config.get("gamma",  1.0))
		return CosAnnealWR(
			optimizer,
			T_0=t_0,
			T_mult=t_mult,
			eta_max=eta_max,
			T_up=t_up,
			gamma=gamma,
			last_epoch=last_epoch,
		)

	if stype == "step":
		step_size = int(config.get("step_size", 10))
		gamma = float(config.get("gamma", 0.5))
		return torch.optim.lr_scheduler.StepLR(
			optimizer,
			step_size=step_size,
			gamma=gamma,
			last_epoch=last_epoch,
		)

	if stype == "exponential":
		gamma = float(config.get("gamma", 0.95))
		return torch.optim.lr_scheduler.ExponentialLR(
			optimizer,
			gamma=gamma,
			last_epoch=last_epoch,
		)

	if stype == "warmup_cosine":
		# Linear warmup for n_warmup epochs, then cosine decay to eta_min.
		# LR at epoch e:
		#   e < n_warmup : lr * (start_factor + (1-start_factor) * e / n_warmup)
		#   e >= n_warmup: lr * (eta_min_frac + 0.5*(1-eta_min_frac)*(1+cos(π*(e-n_warmup)/T_decay)))
		# where lr = optimizer initial LR (= peak LR).
		n_warmup     = int(config.get("warmup_epochs", 5))
		eta_min      = float(config.get("eta_min", 1e-6))
		start_factor = float(config.get("start_factor", 0.1))
		base_lr      = optimizer.param_groups[0]["lr"]
		eta_min_frac = eta_min / base_lr if base_lr > 0 else 0.0
		T_decay      = max(1, total_epochs - n_warmup)

		def _lr_lambda(epoch: int) -> float:
			if epoch < n_warmup:
				# linear ramp: start_factor → 1.0
				return start_factor + (1.0 - start_factor) * epoch / max(1, n_warmup)
			# cosine decay: 1.0 → eta_min_frac
			progress = (epoch - n_warmup) / T_decay
			return eta_min_frac + 0.5 * (1.0 - eta_min_frac) * (
				1.0 + math.cos(math.pi * min(progress, 1.0))
			)

		return torch.optim.lr_scheduler.LambdaLR(
			optimizer, _lr_lambda, last_epoch=last_epoch
		)

	raise ValueError(
		f"Unknown scheduler type: '{stype}'. "
		"Supported: 'none', 'cosine', 'cosine_wr', 'step', 'exponential', 'warmup_cosine'."
	)


class Noam(_LRScheduler):
    """
    https://github.com/tugstugi/pytorch-saltnet/blob/master/utils/lr_scheduler.py

    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    Parameters
    ----------
    warmup_steps: ``int``, required.
        The number of steps to linearly increase the learning rate.
    """
    def __init__(self, optimizer, warmup_steps):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        # Compared to the original Noam scheduler,
        # we simplified it to follow [ model_size ** (-0.5) = warmup_steps ** (0.5) ]
        scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]


class CosAnnealWR(_LRScheduler):
    """
    https://github.com/gaussian37/pytorch_deep_learning_models/blob/master/cosine_annealing_with_warmup
    """
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = -1
        super().__init__(optimizer, last_epoch)
        self.T_cur = last_epoch
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)

        for group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            group["lr"] = lr
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]