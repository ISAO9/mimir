"""
mimir.viz.figures
=================

Paper-grade figure helpers. Every plot in MIMIR goes through this module so
the figure style is consistent and the IVXA conventions are enforced:

  * Background is **white** everywhere (figure + axes).
  * All text is **English-only** (no Japanese / no kanji in deliverables).
  * Legends, colour bars, and titles sit in the **margins**, never on top
    of data.
  * Files are saved as **PDF** under a `PDF/` directory by default.
  * Sizes are chosen to fit two-column journal layouts (~6.7 in wide).

Anything inside MIMIR that draws a figure should call `apply_paper_style()`
before plotting and `save_pdf(...)` afterwards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from mimir.data.benchmarks import BenchmarkSpec
from mimir.data.acquisition import AcquisitionGeometry


# ---------------------------------------------------------------------------
# style
# ---------------------------------------------------------------------------


def apply_paper_style() -> None:
    """
    Apply the MIMIR paper style to the current Matplotlib runtime.

    Idempotent — safe to call repeatedly. Sets fonts, sizes, white
    backgrounds, no top/right spines, tight ticks.
    """
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            # Fonts
            "font.family": "DejaVu Sans",   # available everywhere; English-safe
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            # Axes look
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            # Grid for line plots
            "grid.linewidth": 0.5,
            "grid.alpha": 0.4,
            # Vector PDFs
            "pdf.fonttype": 42,           # embed TrueType (editable in Illustrator)
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


# ---------------------------------------------------------------------------
# saving
# ---------------------------------------------------------------------------


def save_pdf(fig: plt.Figure, name: str, out_dir: Path | str = "PDF", dpi: int = 200) -> Path:
    """
    Save `fig` to ``out_dir/name.pdf`` (always PDF), creating the directory
    if needed. Returns the resolved path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not name.lower().endswith(".pdf"):
        name = name + ".pdf"
    path = out / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return path


# ---------------------------------------------------------------------------
# plot helpers
# ---------------------------------------------------------------------------


def plot_velocity_pair(
    truth: np.ndarray,
    estimate: np.ndarray,
    spec: BenchmarkSpec,
    title_truth: str = "Ground truth",
    title_estimate: str = "MIMIR estimate",
    vmin: float | None = None,
    vmax: float | None = None,
    rays: tuple[np.ndarray, np.ndarray] | None = None,
    max_rays_drawn: int = 30,
    ray_alpha: float = 0.20,
    ray_seed: int = 0,
) -> plt.Figure:
    """
    Side-by-side image of ground-truth and estimated velocity fields with
    a single shared colourbar in the right margin (so it does not overlap
    either image).

    Parameters
    ----------
    truth, estimate : ndarray (nz, nx)
        Velocity fields in km/s.
    spec : BenchmarkSpec
        Domain bounds.
    rays : (sources, receivers) optional
        If given, plot light grey lines for a *random subset* of rays on
        top of the estimate panel — useful for a "coverage" hint without
        obscuring the underlying field.
    max_rays_drawn : int, default 30
        Cap on how many rays are visualised, regardless of how many were
        used in the inversion. With ~300 rays drawn at any reasonable
        alpha, the overlay accumulates to opaque and hides the science.
        Always subsample for paper figures; show the full geometry in a
        dedicated acquisition figure (`plot_acquisition_geometry`).
    ray_alpha : float, default 0.20
        Per-line alpha for the ray overlay. Even with subsampling, keep
        this small so the underlying velocity field stays readable.
    ray_seed : int, default 0
        Reproducibility for the random subsample.
    """
    apply_paper_style()
    if vmin is None:
        vmin = float(min(truth.min(), estimate.min()))
    if vmax is None:
        vmax = float(max(truth.max(), estimate.max()))

    extent = [spec.domain_x[0], spec.domain_x[1], spec.domain_z[1], spec.domain_z[0]]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), constrained_layout=True)

    im0 = axes[0].imshow(truth, extent=extent, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
    axes[0].set_title(title_truth)
    axes[0].set_xlabel("x (km)")
    axes[0].set_ylabel("z (km)")

    axes[1].imshow(estimate, extent=extent, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
    axes[1].set_title(title_estimate)
    axes[1].set_xlabel("x (km)")
    axes[1].tick_params(labelleft=False)

    if rays is not None:
        sources, receivers = rays
        n_rays = sources.shape[0]
        if n_rays > max_rays_drawn:
            rng = np.random.default_rng(ray_seed)
            idx = rng.choice(n_rays, size=max_rays_drawn, replace=False)
            sources = sources[idx]
            receivers = receivers[idx]
        for s, r in zip(sources, receivers):
            axes[1].plot([s[0], r[0]], [s[1], r[1]],
                         color="white", linewidth=0.4, alpha=ray_alpha)

    cbar = fig.colorbar(im0, ax=axes, location="right", shrink=0.85, pad=0.02)
    cbar.set_label("Velocity (km/s)")
    return fig


def plot_acquisition_geometry(
    spec: BenchmarkSpec,
    geometry: AcquisitionGeometry,
    title: str = "Acquisition geometry",
) -> plt.Figure:
    """
    Plot domain box with sources (filled triangles) and receivers (open
    circles); legend placed outside the axes so it never sits on the data.
    """
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.5, 5.0), constrained_layout=True)

    # Domain box
    x_lo, x_hi = spec.domain_x
    z_lo, z_hi = spec.domain_z
    ax.add_patch(
        plt.Rectangle((x_lo, z_lo), x_hi - x_lo, z_hi - z_lo, fill=False, edgecolor="0.4", linewidth=0.8)
    )

    # Receivers (open circles) — collect all unique
    all_recs = np.concatenate(geometry.receivers_per_source, axis=0)
    ax.scatter(
        all_recs[:, 0], all_recs[:, 1],
        marker="o", facecolors="none", edgecolors="tab:blue",
        s=18, linewidths=0.8, label="Receivers",
    )

    # Sources (filled triangles)
    ax.scatter(
        geometry.sources[:, 0], geometry.sources[:, 1],
        marker="v", color="tab:red", s=24, label="Sources",
    )

    ax.set_xlim(x_lo - 0.2, x_hi + 0.2)
    ax.set_ylim(z_hi + 0.2, z_lo - 0.2)        # invert: depth grows downward
    ax.set_xlabel("x (km)")
    ax.set_ylabel("z (km)")
    ax.set_aspect("equal")
    ax.set_title(title)

    # Legend in upper-right *outside* the axes
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    return fig


def plot_loss_history(
    history: Sequence[float],
    iterations: Sequence[int] | None = None,
    title: str = "Training loss",
    smoothing: int = 1,
    ylabel: str = "Loss",
) -> plt.Figure:
    """
    Semilog-y plot of a loss curve with optional moving-average smoothing.

    Parameters
    ----------
    history : Sequence[float]
        Loss values, one per logged iteration.
    iterations : Sequence[int] | None
        Actual iteration indices corresponding to ``history``. If ``None``,
        falls back to ``range(len(history))`` for backward compatibility,
        but you almost always want to pass the real iteration list (e.g.
        ``state.history["iter"]``) so the x-axis shows true training
        iterations rather than log-entry indices.
    """
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.0), constrained_layout=True)

    history_arr = np.asarray(history, dtype=float)
    if iterations is None:
        iters = np.arange(len(history_arr))
    else:
        iters = np.asarray(iterations, dtype=float)
        if len(iters) != len(history_arr):
            raise ValueError(
                f"iterations length {len(iters)} != history length {len(history_arr)}"
            )

    ax.plot(iters, history_arr, color="0.7", linewidth=0.6, label="raw")
    if smoothing > 1 and len(history_arr) >= smoothing:
        kernel = np.ones(smoothing) / smoothing
        smoothed = np.convolve(history_arr, kernel, mode="valid")
        # keep the smoothed curve aligned with the centre of the moving window
        center_offset = smoothing // 2
        ax.plot(iters[center_offset : center_offset + len(smoothed)], smoothed,
                color="tab:blue", linewidth=1.4, label=f"moving avg ({smoothing})")

    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both")
    ax.legend(loc="upper right")
    return fig
