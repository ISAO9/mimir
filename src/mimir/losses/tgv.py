"""
mimir.losses.tgv
================

Total Generalized Variation (TGV²) regularizer for 2D fields, after
Bredies, Kunisch & Pock (2010), "Total Generalized Variation", SIAM J.
Imaging Sciences, 3(3), 492–526.

Why TGV instead of TV
---------------------
Plain TV penalizes the L1 norm of the gradient ‖∇v‖₁. Its minimizers
are *piecewise-constant* fields (the staircase artifact). For seismic
velocities we often want *piecewise-affine* (smoothly varying within
each rock unit, sharp jumps at lithological contacts) — exactly what
TGV² is designed to recover.

The mathematical definition is

    TGV²_{α₀, α₁}(v) = min_w  α₁ ‖∇v − w‖₁  +  α₀ ‖ε(w)‖₁

where w : Ω → ℝ² is an auxiliary vector field that absorbs the smooth
part of the gradient, and ε(w) = ½(∇w + (∇w)ᵀ) is the symmetric gradient
(2x2 tensor). When the inner minimum is attained:

  * On flat regions of v, w ≈ 0 and the TV-like first term penalizes nothing.
  * On smoothly varying regions, w ≈ ∇v, so the first term vanishes and the
    second term penalizes the *Hessian* (a smooth-affine prior).
  * On true edges where ∇v is large and singular, w cannot follow the
    discontinuity, so the first term penalizes the jump (preserving edges).

Implementation note
-------------------
The classical TGV inner minimization over w is solved by Chambolle-Pock
primal-dual iterations. We instead exploit our differentiable framework:
we represent w with the **same architecture as the velocity field** (a
small Fourier-feature MLP) and let Adam optimize w *jointly* with the
NVF parameters during training. The TGV term is then

    TGV²(v, w) = α₁ * mean(|∇v − w|) + α₀ * mean(|ε(w)|)

where mean(|·|) is the per-component absolute value averaged over the
evaluation grid. The min over w is approximated by the Adam update flowing
into w's parameters at every step. This is *not* the exact TGV, but is
provably an upper bound and converges to it as training proceeds (the
Bredies-Holler 2014 connection between TGV and bilevel optimization).

We use isotropic L1 norms (joint magnitude across components) rather than
anisotropic (per-component sum) — these give visually more pleasing,
rotation-invariant results on geophysical fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from mimir.fields.encoding import FourierFeatureEncoding


# ---------------------------------------------------------------------------
# auxiliary vector field — same family as NeuralVelocityField but unbounded
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuxFieldConfig:
    """
    Configuration for the auxiliary vector field w(x, z) ∈ ℝ² used by TGV².

    We use a smaller / shallower network than the velocity field because
    the auxiliary field is intrinsically smoother: it should track ∇v in
    smoothly varying regions, where ∇v itself is bandlimited.
    """

    domain_x: tuple[float, float] = (0.0, 10.0)
    domain_z: tuple[float, float] = (0.0, 10.0)
    fourier_dim: int = 32
    fourier_scale: float = 2.0     # half of the velocity-field scale
    hidden: int = 64
    depth: int = 3
    activation: str = "tanh"


class AuxiliaryVectorField(nn.Module):
    """
    A coordinate-based MLP w(x, z) → ℝ² used as the auxiliary vector field
    in TGV². Has no output bound (unlike the velocity field) because w
    naturally takes signed real values.
    """

    def __init__(self, cfg: AuxFieldConfig) -> None:
        super().__init__()
        self.cfg = cfg

        x_lo, x_hi = cfg.domain_x
        z_lo, z_hi = cfg.domain_z
        self.register_buffer(
            "_xc", torch.tensor([(x_lo + x_hi) / 2.0, (z_lo + z_hi) / 2.0])
        )
        self.register_buffer(
            "_xr", torch.tensor([(x_hi - x_lo) / 2.0, (z_hi - z_lo) / 2.0])
        )

        self.encoding = FourierFeatureEncoding(
            in_dim=2, num_features=cfg.fourier_dim, scale=cfg.fourier_scale
        )

        layers: list[nn.Module] = []
        in_d = self.encoding.out_dim
        for _ in range(cfg.depth):
            layers.append(nn.Linear(in_d, cfg.hidden))
            if cfg.activation == "tanh":
                layers.append(nn.Tanh())
            elif cfg.activation == "gelu":
                layers.append(nn.GELU())
            elif cfg.activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.1))
            else:
                raise ValueError(f"Unknown activation: {cfg.activation}")
            in_d = cfg.hidden
        layers.append(nn.Linear(in_d, 2))   # output ∈ ℝ²
        self.mlp = nn.Sequential(*layers)

        # Initialize the final layer to ≈ 0 so w starts as the zero field
        # (TV-like behaviour at iteration 0).
        last = self.mlp[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def normalize_coords(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        coords = torch.stack([x, z], dim=-1)
        return (coords - self._xc) / self._xr

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if x.dim() == 0:
            x = x.unsqueeze(0)
            z = z.unsqueeze(0)
        coords = self.normalize_coords(x, z)
        feats = self.encoding(coords)
        return self.mlp(feats)   # shape (N, 2): w_x, w_z

    def vector_field_grid(self, nx: int, nz: int, device: torch.device | None = None) -> torch.Tensor:
        """
        Evaluate w on a regular nx-by-nz grid. Returns shape (2, nz, nx)
        — channel-first to match PyTorch conv conventions.
        """
        if device is None:
            device = next(self.parameters()).device
        xs = torch.linspace(self.cfg.domain_x[0], self.cfg.domain_x[1], nx, device=device)
        zs = torch.linspace(self.cfg.domain_z[0], self.cfg.domain_z[1], nz, device=device)
        zz, xx = torch.meshgrid(zs, xs, indexing="ij")
        w = self.forward(xx.flatten(), zz.flatten())   # (N, 2)
        return w.view(nz, nx, 2).permute(2, 0, 1)       # (2, nz, nx)


# ---------------------------------------------------------------------------
# differential operators on grids
# ---------------------------------------------------------------------------


def _grad_2d(field_grid: torch.Tensor) -> torch.Tensor:
    """
    Forward-difference gradient of a (H, W) tensor.

    Returns a (2, H, W) tensor: channel 0 = dz, channel 1 = dx.
    Boundary cells use a one-sided (zero-padded) difference, matching the
    behaviour of the TV regularizer.
    """
    if field_grid.dim() != 2:
        raise ValueError(f"Expected (H, W); got {tuple(field_grid.shape)}.")
    dz = torch.zeros_like(field_grid)
    dx = torch.zeros_like(field_grid)
    dz[:-1, :] = field_grid[1:, :] - field_grid[:-1, :]
    dx[:, :-1] = field_grid[:, 1:] - field_grid[:, :-1]
    return torch.stack([dz, dx], dim=0)


def _symmetric_gradient_2d(w: torch.Tensor) -> torch.Tensor:
    """
    Symmetric gradient ε(w) = ½(∇w + ∇wᵀ) of a vector field w of shape
    (2, H, W). Returns the three independent components of the resulting
    2x2 symmetric tensor: [ε_zz, ε_xx, ε_zx], each of shape (H, W).

    For a 2D vector field w = (w_z, w_x):
        ε_zz = ∂w_z/∂z
        ε_xx = ∂w_x/∂x
        ε_zx = ½(∂w_z/∂x + ∂w_x/∂z)
    """
    if w.dim() != 3 or w.shape[0] != 2:
        raise ValueError(f"Expected (2, H, W); got {tuple(w.shape)}.")
    w_z = w[0]
    w_x = w[1]

    # Forward differences with zero-padding at the trailing boundary
    dwz_dz = torch.zeros_like(w_z)
    dwz_dx = torch.zeros_like(w_z)
    dwx_dz = torch.zeros_like(w_x)
    dwx_dx = torch.zeros_like(w_x)
    dwz_dz[:-1, :] = w_z[1:, :] - w_z[:-1, :]
    dwz_dx[:, :-1] = w_z[:, 1:] - w_z[:, :-1]
    dwx_dz[:-1, :] = w_x[1:, :] - w_x[:-1, :]
    dwx_dx[:, :-1] = w_x[:, 1:] - w_x[:, :-1]

    eps_zz = dwz_dz
    eps_xx = dwx_dx
    eps_zx = 0.5 * (dwz_dx + dwx_dz)
    return torch.stack([eps_zz, eps_xx, eps_zx], dim=0)


# ---------------------------------------------------------------------------
# TGV² loss
# ---------------------------------------------------------------------------


def total_generalized_variation_2d(
    field_grid: torch.Tensor,
    w_grid: torch.Tensor,
    alpha_0: float = 1.0,
    alpha_1: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Differentiable TGV² regularizer:

        TGV²(v, w) = α₁ * E_grid[ ‖∇v − w‖_iso ] + α₀ * E_grid[ ‖ε(w)‖_iso ]

    where ‖·‖_iso is the isotropic (joint) L1 norm of the components at
    each pixel:

        ‖(a, b)‖_iso = √(a² + b² + ε)         (TV-like term)
        ‖(a, b, c)‖_iso = √(a² + b² + 2c² + ε)  (symmetric tensor term)

    Units convention
    ----------------
    Both ∇v and w are expressed in **pixel-step units** — i.e. ∇v[i, j]
    represents the difference between adjacent grid cells, not the physical
    derivative ∂v/∂x. This keeps the regularizer scale-free and lets
    `alpha_0`, `alpha_1` carry purely dimensionless meaning. Critically,
    the auxiliary network must learn to output values matching this same
    pixel-step convention; this happens naturally during joint training
    because TGV² couples them through the `∇v − w` term.

    The 2c² in the second norm is the standard convention for the Frobenius
    norm of a symmetric 2x2 tensor written via its three unique entries.

    Parameters
    ----------
    field_grid : torch.Tensor of shape (H, W)
    w_grid : torch.Tensor of shape (2, H, W)
    alpha_0, alpha_1 : float
        TGV² weights. Bredies-Kunisch-Pock 2010 recommend α₀/α₁ ≈ 2.0
        for natural images; we expose both for ablation.
    eps : small float
        Regularizes √(·) at zero to avoid NaN gradients.

    Returns
    -------
    Scalar torch.Tensor.
    """
    if field_grid.dim() != 2:
        raise ValueError(f"field_grid must be (H, W); got {tuple(field_grid.shape)}.")
    if w_grid.dim() != 3 or w_grid.shape[0] != 2:
        raise ValueError(f"w_grid must be (2, H, W); got {tuple(w_grid.shape)}.")
    if w_grid.shape[1:] != field_grid.shape:
        raise ValueError(
            f"w_grid spatial dims {tuple(w_grid.shape[1:])} != "
            f"field_grid {tuple(field_grid.shape)}."
        )

    grad_v = _grad_2d(field_grid)              # (2, H, W) = [dz, dx]
    diff = grad_v - w_grid                      # (2, H, W)
    tv_like = torch.sqrt(diff[0] ** 2 + diff[1] ** 2 + eps).mean()

    eps_w = _symmetric_gradient_2d(w_grid)      # (3, H, W) = [εzz, εxx, εzx]
    sym_norm = torch.sqrt(
        eps_w[0] ** 2 + eps_w[1] ** 2 + 2.0 * eps_w[2] ** 2 + eps
    ).mean()

    return alpha_1 * tv_like + alpha_0 * sym_norm


# ---------------------------------------------------------------------------
# Huber-TV (cheap alternative for ablation)
# ---------------------------------------------------------------------------


def huber_total_variation_2d(
    field_grid: torch.Tensor,
    delta: float = 0.05,
) -> torch.Tensor:
    """
    Huber-smoothed total variation. Equivalent to TV for large gradients
    (preserving edges) but quadratic for small gradients (avoiding the
    staircase artifact in smooth regions). Concretely:

        ρ_δ(t) = ½ t² / δ      if |t| ≤ δ
                = |t| − δ/2     if |t| > δ

    applied to the joint magnitude |∇v|. Cheaper than TGV (no auxiliary
    field) but theoretically less powerful — included for reference and
    as a sanity baseline against TGV.
    """
    if field_grid.dim() != 2:
        raise ValueError(f"Expected (H, W); got {tuple(field_grid.shape)}.")

    grad = _grad_2d(field_grid)                # (2, H, W)
    mag = torch.sqrt(grad[0] ** 2 + grad[1] ** 2 + 1e-12)
    quadratic = 0.5 * (mag ** 2) / delta
    linear = mag - 0.5 * delta
    return torch.where(mag <= delta, quadratic, linear).mean()
