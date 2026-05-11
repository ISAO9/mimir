"""
01_generate_synthetic_benchmarks.py
===================================

Generate the three synthetic ground-truth velocity benchmarks used to
validate the MIMIR pipeline end-to-end, and render visual previews to PDF.

Outputs
-------
* data/synthetic/<name>.npz       — ground-truth velocity grid + metadata
* PDF/01_<name>_truth.pdf         — preview of the velocity field
* PDF/01_<name>_geometry.pdf      — acquisition geometry overlay

Benchmarks
----------
* `gaussian_anomaly` — easy regime, one Gaussian blob
* `layered`          — moderate, layered with one anomaly
* `curvefault_lookalike` — hard, sharp curved interface

For each benchmark we save both the truth and a default cross-well
acquisition geometry (12 sources × 12 receivers per side = 576 rays).

Usage
-----
    python scripts/01_generate_synthetic_benchmarks.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mimir.data.benchmarks import (
    BenchmarkSpec,
    make_curvefault_lookalike,
    make_gaussian_anomaly,
    make_layered,
)
from mimir.data.acquisition import cross_well_layout
from mimir.utils.seed import set_global_seed
from mimir.viz.figures import (
    apply_paper_style,
    plot_acquisition_geometry,
    save_pdf,
)
import matplotlib.pyplot as plt


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic"
PDF_DIR = Path(__file__).resolve().parents[1] / "PDF"


def _save_truth_npz(name: str, velocity: np.ndarray, spec: BenchmarkSpec) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{name}.npz"
    np.savez(
        path,
        velocity=velocity,
        nx=spec.nx, nz=spec.nz,
        domain_x=np.asarray(spec.domain_x, dtype=np.float32),
        domain_z=np.asarray(spec.domain_z, dtype=np.float32),
        velocity_min=spec.velocity_min, velocity_max=spec.velocity_max,
        name=spec.name,
    )
    return path


def _plot_truth(velocity: np.ndarray, spec: BenchmarkSpec, name: str) -> Path:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(4.8, 4.0), constrained_layout=True)
    extent = [spec.domain_x[0], spec.domain_x[1], spec.domain_z[1], spec.domain_z[0]]
    im = ax.imshow(velocity, extent=extent, cmap="viridis", aspect="equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("z (km)")
    ax.set_title(f"Ground truth — {spec.name}")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.04)
    cbar.set_label("Velocity (km/s)")
    return save_pdf(fig, f"01_{name}_truth", out_dir=PDF_DIR)


def main() -> None:
    set_global_seed(20260507)

    builders = [
        ("gaussian_anomaly",      lambda: make_gaussian_anomaly()),
        ("layered",               lambda: make_layered()),
        ("curvefault_lookalike",  lambda: make_curvefault_lookalike(seed=42)),
    ]

    for name, builder in builders:
        velocity, spec = builder()
        npz_path = _save_truth_npz(name, velocity, spec)
        truth_pdf = _plot_truth(velocity, spec, name)

        # Default acquisition geometry — saved to PDF for record
        geometry = cross_well_layout(spec, n_sources_per_side=12, n_receivers_per_side=12)
        fig = plot_acquisition_geometry(spec, geometry, title=f"Acquisition — {spec.name}")
        geo_pdf = save_pdf(fig, f"01_{name}_geometry", out_dir=PDF_DIR)

        print(f"[ok] {name}: truth -> {npz_path}")
        print(f"          truth pdf    -> {truth_pdf}")
        print(f"          geometry pdf -> {geo_pdf}")
        print(f"          rays: {geometry.total_rays()}")


if __name__ == "__main__":
    main()
