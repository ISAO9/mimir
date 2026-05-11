"""
mimir.fields.encoding
=====================

Random Fourier feature embedding for coordinate-based neural networks
(Tancik et al. 2020; "Fourier Features Let Networks Learn High Frequency
Functions in Low Dimensional Domains", NeurIPS 2020).

Why this exists
---------------
Plain MLPs that ingest raw (x, z) coordinates exhibit a strong spectral bias
toward low-frequency outputs — they fail to recover sharp boundaries, which
is fatal for tomography where the target structures are precisely the sharp
heterogeneities. Mapping the input through

        gamma(v) = [v, sin(2 pi B v), cos(2 pi B v)]

with a fixed random projection matrix `B` removes that bias and lets the
downstream MLP fit high-frequency content without the depth blow-up that
would otherwise be required.

The choice of `B`'s standard deviation (`scale`) is the *single most
sensitive hyperparameter* in our pipeline. Too low → blurry reconstructions;
too high → noisy / chequer-board artefacts. We therefore expose it
explicitly and ablate it in `scripts/41_ablation_fourier.py`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierFeatureEncoding(nn.Module):
    """
    Map an input of shape (..., in_dim) to features of shape (..., out_dim)
    where ``out_dim = in_dim + 2 * num_features``.

    The projection matrix B is sampled once at construction from N(0, scale^2)
    and registered as a non-trainable buffer (so it survives a checkpoint
    round-trip and is shared across processes via DDP).

    Parameters
    ----------
    in_dim : int
        Input coordinate dimension. For 2D tomography this is 2.
    num_features : int
        Number of frequencies in the embedding. Total output dimension is
        ``in_dim + 2 * num_features``.
    scale : float
        Standard deviation of the random frequencies. Larger scale → sharper
        learnable structures, but more noise sensitivity. For coordinates
        normalized to [-1, 1] (which is what we expect upstream), `scale ~ 4`
        is a good starting point.
    learnable : bool, default False
        If True, allow gradient flow into B. Empirically does not help for
        sparsely sampled tomography problems and risks overfitting; we keep
        the default to match the classical NeRF formulation.
    """

    def __init__(
        self,
        in_dim: int = 2,
        num_features: int = 64,
        scale: float = 4.0,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive.")
        if scale <= 0:
            raise ValueError("scale must be positive.")

        self.in_dim = in_dim
        self.num_features = num_features
        self.scale = scale
        self.learnable = learnable

        B = torch.randn(in_dim, num_features) * scale
        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

    @property
    def out_dim(self) -> int:
        return self.in_dim + 2 * self.num_features

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        coords : torch.Tensor
            Shape (..., in_dim). Should be normalized (e.g. to [-1, 1])
            *before* being passed in. Normalization lives in the field
            module so we keep this layer pure-mathematical.

        Returns
        -------
        torch.Tensor
            Shape (..., out_dim) = (..., in_dim + 2 * num_features).
        """
        if coords.shape[-1] != self.in_dim:
            raise ValueError(
                f"Expected last dim {self.in_dim}, got {coords.shape[-1]}."
            )
        proj = 2.0 * math.pi * coords @ self.B  # shape (..., num_features)
        return torch.cat([coords, torch.sin(proj), torch.cos(proj)], dim=-1)
