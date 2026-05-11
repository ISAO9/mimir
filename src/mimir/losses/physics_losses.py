"""
mimir.losses.physics_losses
===========================

Loss functions used by the MIMIR training loop:

* ``travel_time_data_loss``
    Discrepancy between predicted and observed travel times. Defaults to
    Huber (smooth-L1) which is robust to outliers from velocity-field
    initialization while still being smooth-quadratic near the optimum.

* ``total_variation_2d``
    Anisotropic TV regularizer on a regular grid sample of the velocity
    field. Encourages piecewise-smooth structures with sharp transitions —
    which is geophysically reasonable (lithology contrasts) and is the
    standard regularizer in the SBAS / FWI literature.

Why Huber over MSE?
-------------------
Travel-time residuals near initialization can be large (tens of percent) and
MSE squares those, which dominates gradients and creates "explode-or-collapse"
training. Huber clips the influence of large residuals to L1 behaviour beyond
a threshold ``delta``, which empirically gives much more reliable convergence
on the OpenFWI-style benchmarks we target.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


DataLossKind = Literal["mse", "huber", "l1"]


def travel_time_data_loss(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    kind: DataLossKind = "huber",
    huber_delta: float = 0.05,
) -> torch.Tensor:
    """
    Scalar discrepancy between predicted and observed travel times.

    Parameters
    ----------
    predicted, observed : torch.Tensor
        Same shape (e.g. (B,) for a batch of rays).
    kind : {"mse", "huber", "l1"}
        Loss type.
    huber_delta : float
        Smoothness threshold for Huber loss; ignored for "mse" or "l1".
    """
    if kind == "mse":
        return F.mse_loss(predicted, observed)
    if kind == "huber":
        # PyTorch's smooth_l1_loss is Huber with beta = huber_delta
        return F.smooth_l1_loss(predicted, observed, beta=huber_delta)
    if kind == "l1":
        return F.l1_loss(predicted, observed)
    raise ValueError(f"Unknown data-loss kind: {kind}")


def total_variation_2d(field_grid: torch.Tensor) -> torch.Tensor:
    """
    Anisotropic 2D total variation of a 2D tensor.

    Parameters
    ----------
    field_grid : torch.Tensor
        Shape (H, W). Typically obtained by sampling the Neural Velocity
        Field on a coarse regular grid.

    Returns
    -------
    torch.Tensor
        Scalar — mean of |dz| + |dx| over the grid (excluding the boundary).

    Notes
    -----
    Using the *mean* (not sum) keeps the magnitude scale-invariant in the
    grid resolution, which makes the regularization weight transferable
    between experiments.
    """
    if field_grid.dim() != 2:
        raise ValueError(f"Expected 2D tensor (H, W); got shape {tuple(field_grid.shape)}.")

    # Forward differences along each axis
    dz = field_grid[1:, :] - field_grid[:-1, :]   # (H-1, W)
    dx = field_grid[:, 1:] - field_grid[:, :-1]   # (H, W-1)
    return dz.abs().mean() + dx.abs().mean()
