"""
mimir.data.benchmarks
=====================

Ground-truth synthetic velocity models used to validate the inversion
pipeline end-to-end. These deliberately span the spectrum of difficulty:

* ``make_gaussian_anomaly`` — a smooth, isolated high-velocity blob in a
  homogeneous background. Easy regime; the appropriate sanity check.
* ``make_layered`` — a horizontally layered Earth with velocities increasing
  with depth, plus a localized anomaly. Standard in seismology textbooks.
* ``make_curvefault_lookalike`` — qualitatively reproduces the OpenFWI
  CurveFault family: sharp velocity contrasts across a curved fault. Hard
  regime; tests Fourier-feature ability to resolve discontinuities.

We deliberately do not bundle the actual OpenFWI dataset (it is several GB);
script ``02_download_openfwi.py`` is responsible for that, and benchmarks
created here are roughly distribution-matched.

All ground truths return a (nz, nx) NumPy array (image-style, row=z, col=x)
with velocities in km/s.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BenchmarkSpec:
    """Metadata describing a synthetic ground-truth velocity model."""

    name: str
    nx: int
    nz: int
    domain_x: tuple[float, float]
    domain_z: tuple[float, float]
    velocity_min: float
    velocity_max: float

    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, Z) meshgrids in physical units."""
        xs = np.linspace(self.domain_x[0], self.domain_x[1], self.nx)
        zs = np.linspace(self.domain_z[0], self.domain_z[1], self.nz)
        Z, X = np.meshgrid(zs, xs, indexing="ij")
        return X, Z


def make_gaussian_anomaly(
    nx: int = 128,
    nz: int = 128,
    domain: tuple[float, float] = (0.0, 10.0),
    base_velocity: float = 3.0,
    anomaly_velocity: float = 4.5,
    anomaly_center: tuple[float, float] = (5.0, 5.0),
    anomaly_sigma: float = 1.5,
) -> tuple[np.ndarray, BenchmarkSpec]:
    """
    Smooth high-velocity Gaussian blob over a homogeneous background.

    Returns
    -------
    velocity : ndarray (nz, nx)
    spec : BenchmarkSpec
    """
    spec = BenchmarkSpec(
        name="gaussian_anomaly",
        nx=nx, nz=nz,
        domain_x=domain, domain_z=domain,
        velocity_min=float(base_velocity),
        velocity_max=float(anomaly_velocity),
    )
    X, Z = spec.grid()
    cx, cz = anomaly_center
    delta = anomaly_velocity - base_velocity
    v = base_velocity + delta * np.exp(-((X - cx) ** 2 + (Z - cz) ** 2) / (2.0 * anomaly_sigma ** 2))
    return v.astype(np.float32), spec


def make_layered(
    nx: int = 128,
    nz: int = 128,
    domain: tuple[float, float] = (0.0, 10.0),
    surface_velocity: float = 2.5,
    deep_velocity: float = 4.5,
    n_layers: int = 4,
    anomaly_velocity: float = 5.5,
    anomaly_center: tuple[float, float] = (7.0, 6.0),
    anomaly_sigma: float = 0.8,
) -> tuple[np.ndarray, BenchmarkSpec]:
    """
    Horizontally layered model with velocity increasing in `n_layers` step
    increments from `surface_velocity` to `deep_velocity`, plus an embedded
    Gaussian anomaly that breaks horizontal symmetry.
    """
    spec = BenchmarkSpec(
        name="layered",
        nx=nx, nz=nz,
        domain_x=domain, domain_z=domain,
        velocity_min=float(surface_velocity),
        velocity_max=float(max(deep_velocity, anomaly_velocity)),
    )
    X, Z = spec.grid()

    # Layer index (0..n_layers-1) from depth fraction
    z_frac = (Z - domain[0]) / (domain[1] - domain[0])
    layer = np.clip((z_frac * n_layers).astype(int), 0, n_layers - 1)
    # Velocity per layer
    layer_velocities = np.linspace(surface_velocity, deep_velocity, n_layers, dtype=np.float32)
    v = layer_velocities[layer]

    # Anomaly
    cx, cz = anomaly_center
    delta = anomaly_velocity - deep_velocity
    v = v + delta * np.exp(-((X - cx) ** 2 + (Z - cz) ** 2) / (2.0 * anomaly_sigma ** 2))
    return v.astype(np.float32), spec


def make_curvefault_lookalike(
    nx: int = 128,
    nz: int = 128,
    domain: tuple[float, float] = (0.0, 10.0),
    background_velocity: float = 3.0,
    fault_throw: float = 4.5,
    seed: int = 0,
) -> tuple[np.ndarray, BenchmarkSpec]:
    """
    A toy stand-in for the OpenFWI CurveFault family: two layers separated
    by a sinusoidally curved interface, with a sharp velocity contrast
    across the fault. This exercises the Fourier-feature embedding's
    ability to resolve discontinuities.

    Note: this is *not* OpenFWI; it is a distribution-similar synthetic.
    Real OpenFWI is downloaded by `scripts/02_download_openfwi.py`.
    """
    rng = np.random.default_rng(seed)
    spec = BenchmarkSpec(
        name="curvefault_lookalike",
        nx=nx, nz=nz,
        domain_x=domain, domain_z=domain,
        velocity_min=float(background_velocity),
        velocity_max=float(fault_throw),
    )
    X, Z = spec.grid()

    # Curved interface depth as a function of x
    amp = 1.0 + 0.5 * rng.random()
    period = (domain[1] - domain[0]) / (1.0 + rng.random())  # 1 to 2 wavelengths
    z_interface = 0.5 * (domain[0] + domain[1]) + amp * np.sin(2 * np.pi * X / period)

    v = np.where(Z < z_interface, background_velocity, fault_throw)
    # Tiny smoothing so the interface is not a single-pixel step (helps the
    # ray integral). One-pixel Gaussian-like smoothing.
    from scipy.ndimage import gaussian_filter
    v = gaussian_filter(v.astype(np.float32), sigma=0.7)
    return v.astype(np.float32), spec
