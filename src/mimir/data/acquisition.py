"""
mimir.data.acquisition
======================

Source / receiver geometries and the (clean + noisy) observation generator
used to produce the inverse-problem inputs from a known ground truth.

Geometries
----------
* ``cross_well_layout``
    Sources on the left wall, receivers on the right wall, and *also* sources
    on the top with receivers on the bottom. This is the "CT-scan"
    cross-cutting pattern that gives uniform coverage of the central region.
    Empirically (per the prototype 76–82 lineage) this is the geometry that
    yields recoverable central anomalies; pure surface acquisition leaves
    the centre under-illuminated.

* ``surface_layout``
    Both sources and receivers along the surface (z=z_min). Mirrors most
    real seismological networks but is adversarial — central anomaly
    recovery needs many more rays.

The observation generator computes ground-truth travel times by FMM
(scikit-fmm) for accuracy, then adds Gaussian noise calibrated as a
percentage of each clean travel time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import skfmm

from mimir.data.benchmarks import BenchmarkSpec


@dataclass(frozen=True)
class AcquisitionGeometry:
    """A list of source positions and a list of receiver positions per source."""

    sources: np.ndarray         # shape (S, 2), columns = (x, z)
    receivers_per_source: list[np.ndarray]  # length S, each (R_s, 2)

    def n_sources(self) -> int:
        return self.sources.shape[0]

    def total_rays(self) -> int:
        return sum(r.shape[0] for r in self.receivers_per_source)


# ---------------------------------------------------------------------------
# layouts
# ---------------------------------------------------------------------------


def cross_well_layout(
    spec: BenchmarkSpec,
    n_sources_per_side: int = 12,
    n_receivers_per_side: int = 12,
    margin: float = 0.5,
) -> AcquisitionGeometry:
    """
    Sources/receivers along all four walls, paired left-right and top-bottom.

    Parameters
    ----------
    spec : BenchmarkSpec
        Defines the domain.
    n_sources_per_side : int
        Number of sources distributed evenly along each "source wall".
    n_receivers_per_side : int
        Number of receivers along each "receiver wall".
    margin : float
        Distance from the actual domain edge to where we place the points
        (so they sit *inside* the modelled region — important because the
        FMM solver needs the source point to be a grid cell, not on the
        Dirichlet boundary).
    """
    x_lo, x_hi = spec.domain_x[0] + margin, spec.domain_x[1] - margin
    z_lo, z_hi = spec.domain_z[0] + margin, spec.domain_z[1] - margin

    sources_list: list[np.ndarray] = []
    recs_per_src: list[np.ndarray] = []

    # 1. Sources on left wall, receivers along right wall
    src_z = np.linspace(z_lo, z_hi, n_sources_per_side)
    rec_z = np.linspace(z_lo, z_hi, n_receivers_per_side)
    for sz in src_z:
        sources_list.append(np.array([x_lo, sz], dtype=np.float32))
        recs_per_src.append(np.column_stack(
            [np.full(n_receivers_per_side, x_hi, dtype=np.float32), rec_z.astype(np.float32)]
        ))

    # 2. Sources on top wall, receivers along bottom wall
    src_x = np.linspace(x_lo, x_hi, n_sources_per_side)
    rec_x = np.linspace(x_lo, x_hi, n_receivers_per_side)
    for sx in src_x:
        sources_list.append(np.array([sx, z_lo], dtype=np.float32))
        recs_per_src.append(np.column_stack(
            [rec_x.astype(np.float32), np.full(n_receivers_per_side, z_hi, dtype=np.float32)]
        ))

    sources = np.stack(sources_list, axis=0)
    return AcquisitionGeometry(sources=sources, receivers_per_source=recs_per_src)


def surface_layout(
    spec: BenchmarkSpec,
    n_sources: int = 16,
    n_receivers: int = 32,
    margin: float = 0.5,
) -> AcquisitionGeometry:
    """
    Sources and receivers along the top edge (surface). Adversarial geometry.
    """
    x_lo, x_hi = spec.domain_x[0] + margin, spec.domain_x[1] - margin
    z_top = spec.domain_z[0] + margin

    src_x = np.linspace(x_lo, x_hi, n_sources)
    rec_x = np.linspace(x_lo, x_hi, n_receivers)

    sources = np.column_stack([src_x.astype(np.float32), np.full(n_sources, z_top, dtype=np.float32)])
    rec_array = np.column_stack([rec_x.astype(np.float32), np.full(n_receivers, z_top, dtype=np.float32)])

    receivers_per_source: list[np.ndarray] = []
    for sx in src_x:
        # Exclude the receiver coincident with the source (zero-distance ray)
        keep = np.abs(rec_array[:, 0] - sx) > 1e-6
        receivers_per_source.append(rec_array[keep])

    return AcquisitionGeometry(sources=sources, receivers_per_source=receivers_per_source)


# ---------------------------------------------------------------------------
# observation generator (uses ground-truth velocity grid + FMM for accuracy)
# ---------------------------------------------------------------------------


def _world_to_grid(coord: np.ndarray, spec: BenchmarkSpec) -> tuple[int, int]:
    """Convert a world-frame (x, z) point to grid (col, row) indices."""
    x, z = coord
    col = int(round((x - spec.domain_x[0]) / (spec.domain_x[1] - spec.domain_x[0]) * (spec.nx - 1)))
    row = int(round((z - spec.domain_z[0]) / (spec.domain_z[1] - spec.domain_z[0]) * (spec.nz - 1)))
    col = int(np.clip(col, 0, spec.nx - 1))
    row = int(np.clip(row, 0, spec.nz - 1))
    return col, row


def _fmm_travel_time_field(
    velocity_grid: np.ndarray, source_xy: np.ndarray, spec: BenchmarkSpec,
) -> np.ndarray:
    """
    Solve the eikonal equation |grad T| = 1/v from a single source by
    fast marching, returning the full travel-time field on the grid.
    """
    # phi: signed distance from the source, negative inside a tiny disk so
    # skfmm interprets the source as a region. We use a single-pixel hot
    # spot which is sufficient for our cell sizes.
    phi = np.ones_like(velocity_grid, dtype=np.float64)
    col, row = _world_to_grid(source_xy, spec)
    phi[row, col] = -1.0

    dx = (spec.domain_x[1] - spec.domain_x[0]) / (spec.nx - 1)
    dz = (spec.domain_z[1] - spec.domain_z[0]) / (spec.nz - 1)
    if not np.isclose(dx, dz):
        # FMM here assumes square cells; warn the user to keep nx == nz with
        # equal domains, which is also our convention. We keep this guard.
        raise ValueError(f"FMM requires square grid cells; got dx={dx}, dz={dz}.")

    return skfmm.travel_time(phi, velocity_grid.astype(np.float64), dx=dx)


def sample_observed_travel_times(
    velocity_grid: np.ndarray,
    spec: BenchmarkSpec,
    geometry: AcquisitionGeometry,
    noise_pct: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run FMM from each source and sample travel times at the corresponding
    receivers, returning a flat (sources × receivers_for_that_source) record.

    Parameters
    ----------
    velocity_grid : ndarray (nz, nx)
        Ground-truth velocity (km/s).
    spec : BenchmarkSpec
        Domain metadata.
    geometry : AcquisitionGeometry
    noise_pct : float
        Percentage of each clean travel time used as the standard deviation
        of additive Gaussian noise. Set to 0 for noise-free experiments.
    rng : np.random.Generator | None
        For reproducible noise.

    Returns
    -------
    sources_flat : ndarray (R_total, 2)
    receivers_flat : ndarray (R_total, 2)
    travel_times : ndarray (R_total,)
    """
    if rng is None:
        rng = np.random.default_rng()

    sources_flat: list[np.ndarray] = []
    receivers_flat: list[np.ndarray] = []
    times_flat: list[float] = []

    for s_idx in range(geometry.n_sources()):
        src = geometry.sources[s_idx]
        recs = geometry.receivers_per_source[s_idx]

        tt_field = _fmm_travel_time_field(velocity_grid, src, spec)

        for rec in recs:
            col, row = _world_to_grid(rec, spec)
            t_clean = float(tt_field[row, col])
            if not np.isfinite(t_clean):
                continue
            if noise_pct > 0:
                noise = rng.normal(loc=0.0, scale=(noise_pct / 100.0) * t_clean)
                t = t_clean + float(noise)
            else:
                t = t_clean
            sources_flat.append(src.copy())
            receivers_flat.append(rec.copy())
            times_flat.append(t)

    return (
        np.stack(sources_flat, axis=0).astype(np.float32),
        np.stack(receivers_flat, axis=0).astype(np.float32),
        np.asarray(times_flat, dtype=np.float32),
    )
