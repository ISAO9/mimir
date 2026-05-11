"""
41_combine_ablation_heatmaps.py
================================

Combine the three benchmark ablation heatmaps (RMSE only) produced by
script 21 into a single 1x3 figure suitable as Figure 1 of the GJI
manuscript.

Reads:
    logs/ablation_fourier_smoothness/results.csv          (long-format,
        produced by 21_ablation_fourier_smoothness.py)

Writes:
    PDF/21_ablation_heatmap_rmse_combined.pdf             (paper Figure 1)

Layout
------
    +---------------+---------------+---------------+----------+
    |  Gaussian     |   Layered     | Curve-fault   | colorbar |
    |  RMSE heatmap |  RMSE heatmap | RMSE heatmap  |          |
    +---------------+---------------+---------------+----------+

A single shared colour bar on the right keeps the panels visually compact.
The colour scale is shared across all three panels for direct visual
comparison; this means the layered/curvefault panels (which have higher
RMSE values) drive the upper end of the colour scale, and the Gaussian
panel (with lower values) appears in the lower part of the scale - this
is the desired behaviour for "is one benchmark harder than another?"
visual reading.

Usage
-----
    python scripts/41_combine_ablation_heatmaps.py

    # Override the colour scale upper bound (e.g. to clip extreme outliers)
    python scripts/41_combine_ablation_heatmaps.py --vmax 0.7
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mimir.viz.figures import apply_paper_style, save_pdf


REPO_ROOT = Path(__file__).resolve().parents[1]
ABL_CSV = REPO_ROOT / "logs" / "ablation_fourier_smoothness" / "results.csv"
PDF_DIR = REPO_ROOT / "PDF"

BENCHMARKS = ["gaussian_anomaly", "layered", "curvefault_lookalike"]
NICE_NAMES = {
    "gaussian_anomaly": "(a) Gaussian anomaly",
    "layered": "(b) Layered",
    "curvefault_lookalike": "(c) Curve-fault",
}


def _load_results(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/21_ablation_fourier_smoothness.py first."
        )
    rows = []
    with path.open("r", newline="") as f:
        for r in csv.DictReader(f):
            for k in ("fourier_scale", "smoothness_weight", "best_rmse",
                     "best_ssim", "best_pearson"):
                if k in r and r[k] != "":
                    r[k] = float(r[k])
            for k in ("seed", "best_iter"):
                if k in r and r[k] != "":
                    r[k] = int(r[k])
            rows.append(r)
    return rows


def _build_grid(rows: list[dict], benchmark: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate to mean(best_rmse) over seeds for one benchmark, on the
    (smoothness_weight, fourier_scale) grid.

    Returns
    -------
    rmse_grid : (n_smoothness, n_scale) array, ordered with smoothness
                ascending top-to-bottom (so 1.0 appears at the BOTTOM
                in axis y; we'll invert for display).
    scales    : sorted unique fourier_scale values
    smooths   : sorted unique smoothness_weight values
    """
    cell = [r for r in rows if r["benchmark"] == benchmark]
    if not cell:
        raise RuntimeError(f"No rows for benchmark {benchmark}.")

    scales = sorted({r["fourier_scale"] for r in cell})
    smooths = sorted({r["smoothness_weight"] for r in cell})

    grid = np.full((len(smooths), len(scales)), np.nan)
    for i, sm in enumerate(smooths):
        for j, sc in enumerate(scales):
            entries = [r["best_rmse"] for r in cell
                       if abs(r["smoothness_weight"] - sm) < 1e-12
                       and abs(r["fourier_scale"] - sc) < 1e-12]
            if entries:
                grid[i, j] = float(np.mean(entries))
    return grid, np.asarray(scales), np.asarray(smooths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Combine ablation heatmaps.")
    parser.add_argument("--vmin", type=float, default=None,
                        help="Override colour scale lower bound.")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Override colour scale upper bound. Default: clipped at 0.7 km/s.")
    parser.add_argument("--cmap", default="viridis_r",
                        help="Matplotlib colour map name (default: viridis_r so low-RMSE = bright).")
    args = parser.parse_args(argv)

    apply_paper_style()
    rows = _load_results(ABL_CSV)

    # Build per-benchmark grids
    grids = {}
    scales_ref = None
    smooths_ref = None
    for b in BENCHMARKS:
        g, scales, smooths = _build_grid(rows, b)
        grids[b] = g
        if scales_ref is None:
            scales_ref = scales
            smooths_ref = smooths
        else:
            assert np.allclose(scales, scales_ref)
            assert np.allclose(smooths, smooths_ref)

    # Auto colour-scale bounds
    all_vals = np.concatenate([g.flatten() for g in grids.values()])
    all_vals = all_vals[~np.isnan(all_vals)]
    vmin = args.vmin if args.vmin is not None else float(all_vals.min())
    if args.vmax is not None:
        vmax = args.vmax
    else:
        # Clip at 0.7 km/s — values above this are saturated/diverged anyway
        vmax = min(0.7, float(np.percentile(all_vals, 95)))

    # Figure
    fig, axes = plt.subplots(
        1, 3, figsize=(8.4, 2.9),
        gridspec_kw={"wspace": 0.18, "right": 0.88},
    )
    cbar_ax = fig.add_axes([0.90, 0.18, 0.018, 0.7])

    extent = (-0.5, len(scales_ref) - 0.5, len(smooths_ref) - 0.5, -0.5)

    im = None
    for col, b in enumerate(BENCHMARKS):
        ax = axes[col]
        # Display: smoothness ascending top-to-bottom in the data, but we
        # want HIGH smoothness (=1.0, the optimum) at the TOP of the
        # axis to match the ablation paper's visual convention. So we
        # flip the array vertically.
        display_grid = grids[b][::-1, :]

        im = ax.imshow(
            display_grid,
            cmap=args.cmap, vmin=vmin, vmax=vmax,
            aspect="auto", interpolation="nearest",
        )

        # Tick labels
        ax.set_xticks(np.arange(len(scales_ref)))
        ax.set_xticklabels([f"{s:g}" for s in scales_ref])
        ax.set_yticks(np.arange(len(smooths_ref)))
        # Reversed because we flipped the data
        ax.set_yticklabels([f"$10^{{{int(np.log10(s))}}}$"
                             for s in smooths_ref[::-1]])

        ax.set_xlabel("Fourier scale $\\sigma_{f}$")
        if col == 0:
            ax.set_ylabel("Smoothness weight $\\lambda$")

        ax.set_title(NICE_NAMES[b], fontsize=10, pad=4)

        # Annotate each cell with the RMSE value
        for i in range(display_grid.shape[0]):
            for j in range(display_grid.shape[1]):
                val = display_grid[i, j]
                if np.isnan(val):
                    continue
                color = "white" if val > (vmin + 0.45 * (vmax - vmin)) else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=7)

        # Mark the optimum (sm=1.0, sf=4) — the recommended config
        try:
            best_smooth_disp = list(smooths_ref[::-1]).index(1.0)
            best_scale_disp = list(scales_ref).index(4.0)
            ax.scatter([best_scale_disp], [best_smooth_disp],
                       marker="o", s=80, facecolors="none",
                       edgecolors="red", linewidths=1.3, zorder=3)
        except ValueError:
            pass

    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("Validation RMSE (km s$^{-1}$)")

    out_path = save_pdf(fig, "21_ablation_heatmap_rmse_combined", out_dir=PDF_DIR)
    plt.close(fig)
    print(f"[ablation-combined] wrote {out_path}")
    print(f"[ablation-combined] colour scale: [{vmin:.3f}, {vmax:.3f}] km s^-1")
    print(f"[ablation-combined] red ring marks the recommended configuration "
          f"(sigma_f=4, lambda=1) at RMSE values:")
    for b in BENCHMARKS:
        opt_smooth_idx = list(smooths_ref).index(1.0)
        opt_scale_idx = list(scales_ref).index(4.0)
        rmse_at_opt = grids[b][opt_smooth_idx, opt_scale_idx]
        print(f"  {b:<24} {rmse_at_opt:.4f} km/s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
