"""
mimir.training.tgv_trainer
==========================

Joint trainer for the velocity field (v) and the auxiliary vector field
(w) used in TGV² regularization (Bredies-Kunisch-Pock 2010).

Both fields are coordinate-based MLPs with Fourier feature embedding;
they share an Adam optimizer and a single training loop. The TGV²
regularizer effectively supplies a coupling between v and w via the
loss term

    α₁ * |∇v − w|  +  α₀ * |ε(w)|

so the gradients flow naturally into both networks at every step.

This subclass overrides only `_train_step` of the base NVFTrainer to:

  1. Compute the velocity grid v on the smoothness grid (same as TV).
  2. Compute the auxiliary vector field grid w on the same grid.
  3. Use TGV² instead of TV in the loss.

Everything else — checkpointing, validation, LR schedule, logging — is
inherited from the base trainer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mimir.fields.neural_velocity_field import NeuralVelocityField
from mimir.losses.physics_losses import travel_time_data_loss
from mimir.losses.tgv import (
    AuxiliaryVectorField,
    total_generalized_variation_2d,
)
from mimir.physics.ray_tracing import batched_straight_ray_travel_time
from mimir.training.trainer import NVFTrainer, NVFTrainingConfig


@dataclass
class TGVTrainingConfig(NVFTrainingConfig):
    """
    Same as NVFTrainingConfig but with TGV-specific weights replacing
    `smoothness_weight`.

    The base class's `smoothness_weight` is ignored when this config is used.
    """

    tgv_alpha_0: float = 1.0   # weight on |ε(w)|  (Hessian-like term)
    tgv_alpha_1: float = 2.0   # weight on |∇v − w|  (TV-like term)


class TGVTrainer(NVFTrainer):
    """
    Joint trainer for v (NeuralVelocityField) and w (AuxiliaryVectorField)
    using TGV² regularization.

    Parameters
    ----------
    field : NeuralVelocityField
    aux_field : AuxiliaryVectorField
    sources, receivers, observed_tt : torch.Tensor
    truth_grid : np.ndarray | None
    cfg : TGVTrainingConfig
    """

    def __init__(
        self,
        field: NeuralVelocityField,
        aux_field: AuxiliaryVectorField,
        sources: torch.Tensor,
        receivers: torch.Tensor,
        observed_tt: torch.Tensor,
        truth_grid=None,
        cfg: TGVTrainingConfig = TGVTrainingConfig(),
    ) -> None:
        # Initialise the base trainer first (this builds the optimizer over
        # `field` parameters only).
        super().__init__(
            field=field, sources=sources, receivers=receivers,
            observed_tt=observed_tt, truth_grid=truth_grid, cfg=cfg,
        )
        self.aux_field = aux_field

        # Re-build the optimizer to include the aux field's parameters.
        self.optimizer = torch.optim.Adam(
            list(self.field.parameters()) + list(self.aux_field.parameters()),
            lr=cfg.learning_rate,
        )

    def _train_step(self) -> dict:
        self.field.train()
        self.aux_field.train()

        # 1. Data-fit term (velocity field only)
        pred_tt = batched_straight_ray_travel_time(
            self.field, self.sources, self.receivers, n_samples=self.cfg.ray_samples
        )
        data_loss = travel_time_data_loss(
            pred_tt, self.observed_tt,
            kind=self.cfg.data_loss, huber_delta=self.cfg.huber_delta,
        )

        # 2. TGV regularizer — couples v and w
        v_grid = self.field.velocity_grid(
            self.cfg.smoothness_grid, self.cfg.smoothness_grid
        )
        w_grid = self.aux_field.vector_field_grid(
            self.cfg.smoothness_grid, self.cfg.smoothness_grid
        )
        tgv = total_generalized_variation_2d(
            v_grid, w_grid,
            alpha_0=self.cfg.tgv_alpha_0,
            alpha_1=self.cfg.tgv_alpha_1,
        )

        # The smoothness_weight from the base config still scales the *whole*
        # TGV term, allowing curriculum-style annealing if desired.
        loss = data_loss + self.cfg.smoothness_weight * tgv

        # 3. Step
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.field.parameters()) + list(self.aux_field.parameters()),
                self.cfg.grad_clip,
            )
        self.optimizer.step()

        return {
            "loss": float(loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "tv": float(tgv.detach().cpu()),    # logged under "tv" for compatibility
        }
