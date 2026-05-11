"""
mimir.physics.ray_tracing
=========================

Differentiable approximations of the ray-theoretical travel time

        T(s -> r) = ∫_path  1 / v(x(s)) ds

where the integration path goes from a source `s` to a receiver `r` through
a velocity field `v`. The whole integral is autograd-friendly so gradients
flow back to whatever parametrizes `v` (in our case a NeuralVelocityField).

Two solvers are provided:

* ``straight_ray_travel_time`` / ``batched_straight_ray_travel_time``
    Approximation valid in nearly homogeneous media. The path is the
    straight line from source to receiver, sampled at `n_samples` points
    by trapezoidal-rule weights. This is what the prototype scripts
    (76–82) used and is the right starting point. It is also exact in
    a constant-velocity medium.

* (deferred) ``curved_ray_travel_time``
    Eikonal-aware iterative path refinement, in which the path is
    re-bent each outer iteration to follow Snell's law in the current
    velocity estimate. Implemented later (script 23) once the straight-ray
    pipeline is validated against the FMM baseline.

We use trapezoidal rule rather than midpoint or Simpson because:
  - it is the simplest scheme that integrates affine functions exactly
  - boundary endpoints (source, receiver) are explicitly included, which is
    cleaner for joint source-position inversion in the future.
"""

from __future__ import annotations

import torch

from mimir.fields.neural_velocity_field import NeuralVelocityField


def _trapezoidal_path_integral_of_slowness(
    velocity_field: NeuralVelocityField,
    src: torch.Tensor,           # shape (2,)
    rec: torch.Tensor,           # shape (2,)
    n_samples: int,
) -> torch.Tensor:
    """
    Single straight ray from `src` to `rec` integrated by the trapezoidal rule.

    T = sum_{i=0}^{n-1}  ds * 0.5 * (s_i + s_{i+1})
      = ds * (0.5 * s_0 + s_1 + ... + s_{n-1} + 0.5 * s_n)

    where ``s_i = 1 / v(x_i)`` and ``x_i`` is the i-th sample point on the
    straight line from src to rec, ``ds = ||rec - src|| / n_samples``.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2 for trapezoidal rule.")

    n_pts = n_samples + 1  # endpoints inclusive
    t_steps = torch.linspace(0.0, 1.0, n_pts, device=src.device, dtype=src.dtype)
    path_x = src[0] + (rec[0] - src[0]) * t_steps
    path_z = src[1] + (rec[1] - src[1]) * t_steps

    velocities = velocity_field(path_x, path_z)
    slowness = 1.0 / velocities

    distance = torch.linalg.norm(rec - src)
    ds = distance / n_samples
    weights = torch.ones_like(slowness)
    weights[0] = 0.5
    weights[-1] = 0.5
    return ds * (weights * slowness).sum()


def straight_ray_travel_time(
    velocity_field: NeuralVelocityField,
    src: torch.Tensor,
    rec: torch.Tensor,
    n_samples: int = 64,
) -> torch.Tensor:
    """
    Compute the travel time along the straight line from `src` to `rec`
    through `velocity_field`.

    Parameters
    ----------
    velocity_field : NeuralVelocityField
        The continuous velocity model (autograd-tracked).
    src, rec : torch.Tensor
        Shape (2,) — (x, z) coordinates in physical units (km).
    n_samples : int, default 64
        Number of integration *intervals*; the actual number of evaluation
        points is `n_samples + 1`.

    Returns
    -------
    torch.Tensor
        Scalar (0-dim) travel time in seconds (since velocity is km/s and
        distance is km).
    """
    return _trapezoidal_path_integral_of_slowness(velocity_field, src, rec, n_samples)


def batched_straight_ray_travel_time(
    velocity_field: NeuralVelocityField,
    sources: torch.Tensor,        # shape (B, 2)
    receivers: torch.Tensor,      # shape (B, 2)
    n_samples: int = 64,
) -> torch.Tensor:
    """
    Vectorized version: B independent rays in one call. The whole computation
    is one tensor op, so this is dramatically faster than a Python loop and
    fully GPU-parallel.

    Parameters
    ----------
    velocity_field : NeuralVelocityField
    sources : torch.Tensor of shape (B, 2)
    receivers : torch.Tensor of shape (B, 2)
    n_samples : int

    Returns
    -------
    torch.Tensor of shape (B,) — travel times in seconds.
    """
    if sources.shape != receivers.shape:
        raise ValueError("sources and receivers must have the same shape.")
    if sources.dim() != 2 or sources.shape[-1] != 2:
        raise ValueError(f"Expected shape (B, 2); got {tuple(sources.shape)}.")

    B = sources.shape[0]
    n_pts = n_samples + 1
    device = sources.device
    dtype = sources.dtype

    # t in [0, 1] for the n_pts samples per ray; broadcast to (B, n_pts)
    t_line = torch.linspace(0.0, 1.0, n_pts, device=device, dtype=dtype)  # (n_pts,)
    t_line = t_line.unsqueeze(0).expand(B, n_pts)                          # (B, n_pts)

    # Path coordinates per ray: (B, n_pts)
    src_x = sources[:, 0:1]    # (B, 1)
    src_z = sources[:, 1:2]
    rec_x = receivers[:, 0:1]
    rec_z = receivers[:, 1:2]
    path_x = src_x + (rec_x - src_x) * t_line
    path_z = src_z + (rec_z - src_z) * t_line

    # Flatten so the field gets a 1D batch
    flat_x = path_x.reshape(-1)
    flat_z = path_z.reshape(-1)
    flat_v = velocity_field(flat_x, flat_z)
    velocities = flat_v.view(B, n_pts)

    slowness = 1.0 / velocities  # (B, n_pts)

    # Trapezoidal weights: 0.5 at endpoints, 1.0 elsewhere
    w = torch.ones(n_pts, device=device, dtype=dtype)
    w[0] = 0.5
    w[-1] = 0.5
    integrand = (slowness * w.unsqueeze(0)).sum(dim=-1)  # (B,)

    distances = torch.linalg.norm(receivers - sources, dim=-1)  # (B,)
    ds = distances / n_samples
    return ds * integrand
