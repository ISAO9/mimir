"""
mimir.fields.neural_velocity_field
==================================

Neural Velocity Field (NVF) — a coordinate-based MLP

        v(x, z) = base_velocity + (v_max - v_min) * sigmoid( MLP( gamma(x, z) ) )

that represents a 2D P-wave velocity field. The whole field is a closed-form,
infinitely-differentiable function of the coordinates, which gives us:

  • analytical gradients w.r.t. any model parameter for the inverse problem
  • mesh-free evaluation at arbitrary query points (essential for ray
    integration which samples non-grid-aligned points along each ray)
  • a well-defined output range (no postprocessing clamps required)

Empirical recipe (validated across scripts 76–82 of the prototype lineage):
  1. Normalize the input coordinate domain to [-1, 1].
  2. Apply a Fourier feature encoding (Tancik 2020) with scale ≈ 4 for a
     10 km x 10 km domain.
  3. Use Tanh activations (best for smooth physical fields) or LeakyReLU
     (slightly sharper but noisier).
  4. Output a *bounded* velocity via base + range * sigmoid(...). This
     prevents pathological negative or absurdly high velocities during the
     early, high-LR phase of training.
  5. Initialize so the field starts close to a uniform `base_velocity`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from mimir.fields.encoding import FourierFeatureEncoding


ActivationName = Literal["tanh", "gelu", "leaky_relu"]


def _make_activation(name: ActivationName) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    raise ValueError(f"Unknown activation: {name}")


@dataclass(frozen=True)
class NVFConfig:
    """Configuration object for NeuralVelocityField — used by the trainer."""

    domain_x: tuple[float, float] = (0.0, 10.0)   # km
    domain_z: tuple[float, float] = (0.0, 10.0)   # km
    base_velocity: float = 3.0                    # km/s, mid of expected range
    velocity_range: tuple[float, float] = (2.0, 5.5)
    fourier_dim: int = 64
    fourier_scale: float = 4.0
    hidden: int = 128
    depth: int = 4
    activation: ActivationName = "tanh"


class NeuralVelocityField(nn.Module):
    """
    A continuous 2D P-wave velocity field parametrized by an MLP over
    Fourier-embedded normalized coordinates.

    The output is constrained to ``[v_min, v_max]`` via a sigmoid:

        v_hat(x, z) = v_min + (v_max - v_min) * sigmoid(g(x, z))

    where ``g`` is the raw MLP output. Initialization sets g ≈ 0 so that
    v_hat starts at (v_min + v_max) / 2 — adjust ``base_velocity`` to taste
    if your prior expectation differs.
    """

    def __init__(self, cfg: NVFConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Coordinate normalization parameters (registered so they save)
        x_lo, x_hi = cfg.domain_x
        z_lo, z_hi = cfg.domain_z
        self.register_buffer(
            "_xc", torch.tensor([(x_lo + x_hi) / 2.0, (z_lo + z_hi) / 2.0])
        )
        self.register_buffer(
            "_xr", torch.tensor([(x_hi - x_lo) / 2.0, (z_hi - z_lo) / 2.0])
        )

        # Output range (in km/s); sigmoid activates onto this interval
        v_lo, v_hi = cfg.velocity_range
        self.register_buffer("_v_lo", torch.tensor(float(v_lo)))
        self.register_buffer("_v_hi", torch.tensor(float(v_hi)))

        # Fourier embedding
        self.encoding = FourierFeatureEncoding(
            in_dim=2, num_features=cfg.fourier_dim, scale=cfg.fourier_scale
        )

        # MLP backbone
        layers: list[nn.Module] = []
        in_d = self.encoding.out_dim
        for _ in range(cfg.depth):
            layers.append(nn.Linear(in_d, cfg.hidden))
            layers.append(_make_activation(cfg.activation))
            in_d = cfg.hidden
        layers.append(nn.Linear(in_d, 1))
        self.mlp = nn.Sequential(*layers)

        # Initialise the final layer so g(x,z) ≈ 0 → output ≈ midpoint
        # of velocity_range, then offset by the prior `base_velocity` via
        # an analytical bias trick computed below.
        self._init_for_base_velocity()

    # ------------------------------------------------------------------
    # core forward
    # ------------------------------------------------------------------

    def normalize_coords(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Stack and normalize coordinates to [-1, 1].

        Parameters
        ----------
        x, z : 1D tensors of shape (N,) in physical units (km).

        Returns
        -------
        torch.Tensor of shape (N, 2), each component in [-1, 1].
        """
        coords = torch.stack([x, z], dim=-1)
        return (coords - self._xc) / self._xr

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the velocity field at a batch of (x, z) coordinates.

        Parameters
        ----------
        x, z : torch.Tensor
            Shape (N,). Coordinates in *physical* units (km).

        Returns
        -------
        torch.Tensor
            Shape (N,). Velocities in km/s, guaranteed within
            ``self.cfg.velocity_range``.
        """
        if x.dim() == 0:
            x = x.unsqueeze(0)
            z = z.unsqueeze(0)
        coords = self.normalize_coords(x, z)
        feats = self.encoding(coords)
        g = self.mlp(feats).squeeze(-1)  # shape (N,)
        return self._v_lo + (self._v_hi - self._v_lo) * torch.sigmoid(g)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def velocity_grid(self, nx: int, nz: int, device: torch.device | None = None) -> torch.Tensor:
        """
        Evaluate the field on a regular nx-by-nz grid spanning the domain.
        Used for visualization and metric computation.

        Returns
        -------
        torch.Tensor of shape (nz, nx) — convention is image-style (row=z).
        """
        if device is None:
            device = next(self.parameters()).device
        x_lo, x_hi = self.cfg.domain_x
        z_lo, z_hi = self.cfg.domain_z
        xs = torch.linspace(x_lo, x_hi, nx, device=device)
        zs = torch.linspace(z_lo, z_hi, nz, device=device)
        zz, xx = torch.meshgrid(zs, xs, indexing="ij")
        v = self.forward(xx.flatten(), zz.flatten())
        return v.view(nz, nx)

    # ------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------

    def _init_for_base_velocity(self) -> None:
        """
        Set the MLP biases so that, at initialisation, the predicted velocity
        is approximately ``self.cfg.base_velocity`` everywhere.

        We solve for the sigmoid logit:
            base = v_lo + (v_hi - v_lo) * sigmoid(b)
            =>  b = logit((base - v_lo) / (v_hi - v_lo))
        Then we set the final-layer bias to b and zero its weights, while
        keeping prior layers at PyTorch's default initialisation. With small
        random projections via Fourier features the MLP output near init is
        approximately the final bias.
        """
        v_lo = self.cfg.velocity_range[0]
        v_hi = self.cfg.velocity_range[1]
        base = float(self.cfg.base_velocity)
        if not v_lo < base < v_hi:
            raise ValueError(
                f"base_velocity {base} must lie strictly inside velocity_range {(v_lo, v_hi)}."
            )
        p = (base - v_lo) / (v_hi - v_lo)
        b = torch.log(torch.tensor(p / (1.0 - p)))

        last_linear = self.mlp[-1]
        assert isinstance(last_linear, nn.Linear)
        nn.init.zeros_(last_linear.weight)
        nn.init.constant_(last_linear.bias, b.item())
