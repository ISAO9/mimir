"""
mimir.training.trainer
======================

Training loop for the Neural Velocity Field on a fixed observation set.

Design
------
* Every fixed number of iterations (`val_every`), evaluate the field on the
  same regular grid where the ground truth lives and compute RMSE / SSIM /
  Pearson correlation against the truth. The trainer keeps the *best* RMSE
  checkpoint on disk (`models/<run>/best.pt`) and the *last* checkpoint
  (`models/<run>/last.pt`).
* The training data are flat arrays (sources, receivers, observed travel
  times). The trainer runs full-batch by default — for our problem sizes
  (tens of thousands of rays, max), that fits comfortably on a 16 GB GPU.
* LR schedule is cosine with linear warm-up.
* Gradient clipping is on by default (`grad_clip=1.0`) — without it the
  early phase of training is occasionally unstable when the field starts
  far from the truth.

What this trainer is *not* yet
------------------------------
* Mini-batch / DataLoader-based — full-batch is fine here.
* Multi-GPU — single-device only (we will revisit when scaling to OpenFWI).
* Curriculum / coarse-to-fine — the Fourier scale can be ramped up
  inside the loop in a future iteration; for now we expose a fixed scale.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from tqdm import tqdm

from mimir.fields.neural_velocity_field import NeuralVelocityField, NVFConfig
from mimir.losses.physics_losses import travel_time_data_loss, total_variation_2d
from mimir.physics.ray_tracing import batched_straight_ray_travel_time


# ---------------------------------------------------------------------------
# config and state objects
# ---------------------------------------------------------------------------


@dataclass
class NVFTrainingConfig:
    iterations: int = 8000
    learning_rate: float = 2.0e-3
    lr_schedule: str = "cosine"          # constant | cosine
    warmup_iters: int = 200
    grad_clip: float = 1.0
    data_loss: str = "huber"             # mse | huber | l1
    huber_delta: float = 0.05
    smoothness_weight: float = 1.0e-3
    smoothness_grid: int = 64            # resolution at which TV is computed
    ray_samples: int = 64
    val_every: int = 100
    log_every: int = 50
    save_best: bool = True
    save_last: bool = True               # set False in ablations to avoid disk-fill
    out_dir: str = "./logs/nvf_default"


@dataclass
class NVFTrainingState:
    iter: int = 0
    best_val_rmse: float = math.inf
    history: dict = field(default_factory=lambda: {
        "iter": [], "loss": [], "data_loss": [], "tv": [],
        "lr": [], "val_iter": [], "val_rmse": [], "val_ssim": [], "val_pearson": [],
    })


# ---------------------------------------------------------------------------
# evaluation metrics
# ---------------------------------------------------------------------------


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM via scikit-image; auto data range from joint span."""
    from skimage.metrics import structural_similarity as ski_ssim

    data_range = float(max(a.max(), b.max()) - min(a.min(), b.min()))
    if data_range <= 0:
        return float("nan")
    return float(ski_ssim(a, b, data_range=data_range))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    av = a.flatten() - a.mean()
    bv = b.flatten() - b.mean()
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 0:
        return float("nan")
    return float((av * bv).sum() / denom)


# ---------------------------------------------------------------------------
# learning-rate schedules
# ---------------------------------------------------------------------------


def _lr_schedule_value(it: int, base_lr: float, total: int, warmup: int, kind: str) -> float:
    if kind == "constant":
        return base_lr
    if kind == "cosine":
        if it < warmup:
            return base_lr * (it + 1) / max(1, warmup)
        progress = (it - warmup) / max(1, total - warmup)
        progress = min(max(progress, 0.0), 1.0)
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"Unknown lr_schedule: {kind}")


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------


class NVFTrainer:
    """
    Train a NeuralVelocityField against a fixed set of observed travel times.

    Parameters
    ----------
    field : NeuralVelocityField
        The trainable model.
    sources, receivers : torch.Tensor
        (B, 2) coordinates.
    observed_tt : torch.Tensor
        (B,) observed travel times in seconds.
    truth_grid : np.ndarray | None
        Optional ground-truth velocity grid for validation metrics.
    cfg : NVFTrainingConfig
    """

    def __init__(
        self,
        field: NeuralVelocityField,
        sources: torch.Tensor,
        receivers: torch.Tensor,
        observed_tt: torch.Tensor,
        truth_grid: Optional[np.ndarray] = None,
        cfg: NVFTrainingConfig = NVFTrainingConfig(),
    ) -> None:
        self.field = field
        self.sources = sources
        self.receivers = receivers
        self.observed_tt = observed_tt
        self.truth_grid = truth_grid
        self.cfg = cfg
        self.state = NVFTrainingState()

        self.optimizer: Optimizer = Adam(
            self.field.parameters(), lr=cfg.learning_rate
        )

        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # one-step
    # ------------------------------------------------------------------

    def _train_step(self) -> dict:
        self.field.train()

        # 1. data-fit term
        pred_tt = batched_straight_ray_travel_time(
            self.field, self.sources, self.receivers, n_samples=self.cfg.ray_samples
        )
        data_loss = travel_time_data_loss(
            pred_tt, self.observed_tt, kind=self.cfg.data_loss, huber_delta=self.cfg.huber_delta
        )

        # 2. TV regularizer on the velocity grid
        if self.cfg.smoothness_weight > 0:
            v_grid = self.field.velocity_grid(self.cfg.smoothness_grid, self.cfg.smoothness_grid)
            tv = total_variation_2d(v_grid)
        else:
            tv = torch.tensor(0.0, device=data_loss.device)

        loss = data_loss + self.cfg.smoothness_weight * tv

        # 3. step
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.field.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        return {
            "loss": float(loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "tv": float(tv.detach().cpu()),
        }

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self) -> dict:
        if self.truth_grid is None:
            return {}
        self.field.eval()
        nz, nx = self.truth_grid.shape
        v_pred = self.field.velocity_grid(nx, nz).detach().cpu().numpy()
        v_true = self.truth_grid

        rmse = float(np.sqrt(np.mean((v_pred - v_true) ** 2)))
        ssim = _ssim(v_true, v_pred)
        pear = _pearson(v_true, v_pred)
        return {"rmse": rmse, "ssim": ssim, "pearson": pear}

    # ------------------------------------------------------------------
    # checkpoint
    # ------------------------------------------------------------------

    def _save(self, name: str) -> Path:
        path = self.out_dir / name
        torch.save(
            {
                "iter": self.state.iter,
                "model_state_dict": self.field.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "field_config": asdict(self.field.cfg) if hasattr(self.field, "cfg") else None,
                "training_config": asdict(self.cfg),
                "history": self.state.history,
                "best_val_rmse": self.state.best_val_rmse,
            },
            path,
        )
        return path

    # ------------------------------------------------------------------
    # the loop
    # ------------------------------------------------------------------

    def fit(self, on_iter: Optional[Callable[[NVFTrainingState], None]] = None) -> NVFTrainingState:
        """
        Run the full training loop. ``on_iter`` is an optional callback
        invoked after each completed iteration with the trainer state
        (useful for live plotting).
        """
        pbar = tqdm(range(self.cfg.iterations), desc="MIMIR-NVF", dynamic_ncols=True)
        t0 = time.time()

        for it in pbar:
            # LR schedule
            lr = _lr_schedule_value(
                it, self.cfg.learning_rate, self.cfg.iterations,
                self.cfg.warmup_iters, self.cfg.lr_schedule,
            )
            for g in self.optimizer.param_groups:
                g["lr"] = lr

            stats = self._train_step()
            self.state.iter = it + 1

            # Log
            if (it % self.cfg.log_every) == 0 or it == self.cfg.iterations - 1:
                self.state.history["iter"].append(it)
                self.state.history["loss"].append(stats["loss"])
                self.state.history["data_loss"].append(stats["data_loss"])
                self.state.history["tv"].append(stats["tv"])
                self.state.history["lr"].append(lr)

            # Validate
            if (it % self.cfg.val_every) == 0 or it == self.cfg.iterations - 1:
                val = self._validate()
                if val:
                    self.state.history["val_iter"].append(it)
                    self.state.history["val_rmse"].append(val["rmse"])
                    self.state.history["val_ssim"].append(val["ssim"])
                    self.state.history["val_pearson"].append(val["pearson"])

                    if self.cfg.save_best and val["rmse"] < self.state.best_val_rmse:
                        self.state.best_val_rmse = val["rmse"]
                        self._save("best.pt")

                    pbar.set_postfix({
                        "loss": f"{stats['loss']:.4f}",
                        "rmse": f"{val['rmse']:.4f}",
                        "ssim": f"{val['ssim']:.3f}",
                    })
                else:
                    pbar.set_postfix({"loss": f"{stats['loss']:.4f}"})

            if on_iter is not None:
                on_iter(self.state)

        # Save the last (unless explicitly disabled, e.g. in ablation runs)
        if self.cfg.save_last:
            self._save("last.pt")
        elapsed = time.time() - t0
        print(f"[trainer] finished {self.cfg.iterations} iters in {elapsed:.1f}s; best RMSE = {self.state.best_val_rmse:.4f}")
        return self.state
