"""
11_compare_mimir_vs_baseline.py
================================

Build the manuscript's composite comparison figure: for each of the three
synthetic benchmarks, render ground truth, MIMIR-TGV² best-of-five-seeds,
classical FMM-LSMR best-of-five-seeds, and the per-pixel difference field
in a single 3x4 panel.

Reads
-----
    logs/tgv/results_per_seed.csv               (from script 40, MIMIR-TGV²)
    logs/baseline_eikonal_fmm/results_per_seed.csv  (from script 10, classical)
    models/tgv/<benchmark>/seed_<N>/best.pt     (per-seed MIMIR checkpoints)
    data/synthetic/<benchmark>.npz              (from script 01)

Writes
------
    PDF/11_comparison_composite.pdf             (paper Figure 2)

For each benchmark, selects the seed achieving the lowest validation RMSE
on each method side and reconstructs the velocity field from the saved
checkpoint (MIMIR) or the saved baseline grid (classical), then computes
per-pixel difference and overlays SSIM/RMSE annotations.

Usage
-----
    cd /path/to/mimir
    uv run python scripts/11_compare_mimir_vs_baseline.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import ticker
from scipy import stats

from mimir.data.benchmarks import BenchmarkSpec
from mimir.fields import NeuralVelocityField
from mimir.fields.neural_velocity_field import NVFConfig
from mimir.utils import resolve_device
from mimir.utils.device import banner
from mimir.viz.figures import apply_paper_style, save_pdf


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "synthetic"
PDF_DIR = REPO_ROOT / "PDF"
COMP_LOG = REPO_ROOT / "logs" / "comparison"
# >>> THE CRITICAL FIX: point at TGV² results, not TV results <<<
MIMIR_CSV = REPO_ROOT / "logs" / "tgv" / "results_per_seed.csv"
BASELINE_CSV = REPO_ROOT / "logs" / "baseline_eikonal_fmm" / "results_per_seed.csv"
MIMIR_MODELS = REPO_ROOT / "models" / "tgv"


# ============================================================================
# Loaders
# ============================================================================


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the corresponding script first."
        )
    rows = []
    with path.open("r", newline="") as f:
        for r in csv.DictReader(f):
            for k in ("best_rmse", "best_ssim", "best_pearson",
                      "final_rmse", "elapsed_s"):
                if k in r and r[k] != "":
                    r[k] = float(r[k])
            for k in ("seed", "best_iter"):
                if k in r and r[k] != "":
                    r[k] = int(r[k])
            rows.append(r)
    return rows


def _load_truth(name: str) -> tuple[np.ndarray, BenchmarkSpec]:
    npz = DATA_DIR / f"{name}.npz"
    arch = np.load(npz, allow_pickle=False)
    velocity = arch["velocity"].astype(np.float32)
    spec = BenchmarkSpec(
        name=str(arch["name"]),
        nx=int(arch["nx"]), nz=int(arch["nz"]),
        domain_x=tuple(arch["domain_x"].tolist()),
        domain_z=tuple(arch["domain_z"].tolist()),
        velocity_min=float(arch["velocity_min"]),
        velocity_max=float(arch["velocity_max"]),
    )
    return velocity, spec


def _reload_mimir_velocity(name: str, spec: BenchmarkSpec, device: torch.device) -> np.ndarray:
    """
    Reload the MIMIR-TGV² best_overall.pt checkpoint (produced by
    scripts/40_train_tgv.py) and evaluate on the spec grid.
    """
    ckpt_path = MIMIR_MODELS / name / "best_overall.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} not found. "
            f"Run scripts/40_train_tgv.py first to produce TGV² checkpoints."
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    field_cfg_dict = ckpt.get("field_config", {}) or {}
    nvf_cfg = NVFConfig(
        domain_x=tuple(field_cfg_dict.get("domain_x", spec.domain_x)),
        domain_z=tuple(field_cfg_dict.get("domain_z", spec.domain_z)),
        base_velocity=float(field_cfg_dict.get("base_velocity", 3.0)),
        velocity_range=tuple(field_cfg_dict.get("velocity_range", (2.0, 5.5))),
        fourier_dim=int(field_cfg_dict.get("fourier_dim", 64)),
        fourier_scale=float(field_cfg_dict.get("fourier_scale", 4.0)),
        hidden=int(field_cfg_dict.get("hidden", 128)),
        depth=int(field_cfg_dict.get("depth", 4)),
        activation=field_cfg_dict.get("activation", "tanh"),
    )
    field = NeuralVelocityField(nvf_cfg).to(device)
    field.load_state_dict(ckpt["model_state_dict"])
    field.eval()
    with torch.no_grad():
        v = field.velocity_grid(spec.nx, spec.nz).cpu().numpy()
    return v


def _reload_baseline_velocity(name: str, spec: BenchmarkSpec):
    """Re-runs the classical FMM-LSMR at its tuned config on the
    seed with the lowest RMSE. Returns (v_best, geometry, best_seed)."""
    cfg_path = REPO_ROOT / "logs" / "baseline_eikonal_fmm" / name / "best_config.txt"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path} not found.")
    kv = {}
    for line in cfg_path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    damping = float(kv["damping"])
    smoothing = float(kv["smoothing"])
    max_iter = int(kv.get("max_iter", "20"))

    rows = _load_csv(BASELINE_CSV)
    bench_rows = [r for r in rows if r["benchmark"] == name]
    if not bench_rows:
        raise RuntimeError(f"No rows for benchmark {name} in {BASELINE_CSV}")
    best_row = min(bench_rows, key=lambda r: r["best_rmse"])
    best_seed = int(best_row["seed"])

    from mimir.baselines import ClassicalEikonalTomography, EikonalTomographyConfig
    from mimir.data.acquisition import cross_well_layout, sample_observed_travel_times
    from mimir.utils.seed import set_global_seed

    truth, _ = _load_truth(name)
    set_global_seed(best_seed)
    geometry = cross_well_layout(spec, n_sources_per_side=12, n_receivers_per_side=12)
    rng = np.random.default_rng(best_seed + 1)
    sources, receivers, tt = sample_observed_travel_times(
        truth, spec, geometry, noise_pct=5.0, rng=rng,
    )
    cfg = EikonalTomographyConfig(
        damping=damping, smoothing=smoothing, max_iter=max_iter,
        log_every=max_iter,
    )
    solver = ClassicalEikonalTomography(spec, cfg)
    v_best, _ = solver.fit(sources, receivers, tt, truth_grid=truth, verbose=False)
    return v_best, geometry, best_seed


# ============================================================================
# Stats helpers
# ============================================================================


def _per_benchmark_stats(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for b in sorted({r["benchmark"] for r in rows}):
        cell = [r for r in rows if r["benchmark"] == b]
        rmses = np.asarray([c["best_rmse"] for c in cell])
        ssims = np.asarray([c["best_ssim"] for c in cell])
        pears = np.asarray([c["best_pearson"] for c in cell])
        out[b] = {
            "n": len(cell),
            "mean_rmse": float(rmses.mean()), "std_rmse": float(rmses.std()),
            "mean_ssim": float(ssims.mean()), "std_ssim": float(ssims.std()),
            "mean_pearson": float(pears.mean()), "std_pearson": float(pears.std()),
            "rmse_array": rmses, "ssim_array": ssims,
        }
    return out


def _best_seed_metrics(rows: list[dict], bench: str) -> tuple[float, float]:
    cell = [r for r in rows if r["benchmark"] == bench]
    if not cell:
        return float("nan"), float("nan")
    best = min(cell, key=lambda r: r["best_rmse"])
    return float(best["best_rmse"]), float(best["best_ssim"])


# ============================================================================
# Composite figure
# ============================================================================


def _add_acquisition_markers(ax, geometry, extent):
    try:
        srcs = np.asarray(geometry.sources)
        rcvs = np.asarray(geometry.receivers)
    except Exception:
        srcs = np.asarray(getattr(geometry, "source_positions", []))
        rcvs = np.asarray(getattr(geometry, "receiver_positions", []))

    if srcs.size:
        ax.scatter(srcs[:, 0], srcs[:, 1], marker="*", s=22,
                   c="white", edgecolors="black", linewidths=0.6,
                   zorder=10)
    if rcvs.size:
        ax.scatter(rcvs[:, 0], rcvs[:, 1], marker="v", s=12,
                   c="black", edgecolors="white", linewidths=0.4,
                   zorder=10)


def _annotate_metric_box(ax, rmse: float, ssim: float):
    if np.isnan(rmse):
        return
    txt = f"RMSE = {rmse:.3f} km/s\nSSIM = {ssim:.3f}"
    ax.text(
        0.04, 0.96, txt,
        transform=ax.transAxes,
        fontsize=8, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.28",
                  facecolor="white", edgecolor="gray", alpha=0.88),
        zorder=12,
    )


def plot_composite(
    benches, truths, specs, mimir_v, baseline_v,
    geometries, mimir_metrics, fmm_metrics,
) -> plt.Figure:
    apply_paper_style()
    n = len(benches)
    fig = plt.figure(figsize=(11.5, 2.9 * n + 0.6), facecolor="white")
    gs = fig.add_gridspec(
        nrows=n, ncols=6,
        width_ratios=[1.0, 1.0, 1.0, 0.04, 1.0, 0.04],
        wspace=0.20, hspace=0.30,
        left=0.06, right=0.97, top=0.93, bottom=0.08,
    )
    col_titles = ["Ground truth", "MIMIR-TGV²", "Classical FMM-LSMR", "MIMIR − FMM"]

    for row, name in enumerate(benches):
        truth = truths[name]
        spec = specs[name]
        v_mim = mimir_v[name]
        v_fmm = baseline_v[name]
        geom = geometries[name]
        diff = v_mim - v_fmm

        extent = [spec.domain_x[0], spec.domain_x[1],
                  spec.domain_z[1], spec.domain_z[0]]
        v_lo = float(min(truth.min(), v_mim.min(), v_fmm.min()))
        v_hi = float(max(truth.max(), v_mim.max(), v_fmm.max()))
        d_abs = float(np.max(np.abs(diff))) if np.any(diff) else 1.0

        # Truth
        ax_t = fig.add_subplot(gs[row, 0])
        im_v = ax_t.imshow(truth, extent=extent, cmap="viridis",
                           vmin=v_lo, vmax=v_hi, aspect="equal",
                           interpolation="nearest")
        _add_acquisition_markers(ax_t, geom, extent)
        if row == 0:
            ax_t.set_title(col_titles[0], fontsize=10)
        ax_t.set_ylabel(f"{name.replace('_', ' ')}\nz (km)", fontsize=9)
        if row == n - 1:
            ax_t.set_xlabel("x (km)", fontsize=9)
        else:
            ax_t.tick_params(labelbottom=False)
        ax_t.tick_params(labelsize=8)

        # MIMIR
        ax_m = fig.add_subplot(gs[row, 1])
        ax_m.imshow(v_mim, extent=extent, cmap="viridis",
                    vmin=v_lo, vmax=v_hi, aspect="equal",
                    interpolation="nearest")
        _add_acquisition_markers(ax_m, geom, extent)
        rmse_m, ssim_m = mimir_metrics.get(name, (float("nan"), float("nan")))
        _annotate_metric_box(ax_m, rmse_m, ssim_m)
        if row == 0:
            ax_m.set_title(col_titles[1], fontsize=10)
        if row == n - 1:
            ax_m.set_xlabel("x (km)", fontsize=9)
        else:
            ax_m.tick_params(labelbottom=False)
        ax_m.tick_params(labelleft=False, labelsize=8)

        # FMM
        ax_f = fig.add_subplot(gs[row, 2])
        ax_f.imshow(v_fmm, extent=extent, cmap="viridis",
                    vmin=v_lo, vmax=v_hi, aspect="equal",
                    interpolation="nearest")
        _add_acquisition_markers(ax_f, geom, extent)
        rmse_f, ssim_f = fmm_metrics.get(name, (float("nan"), float("nan")))
        _annotate_metric_box(ax_f, rmse_f, ssim_f)
        if row == 0:
            ax_f.set_title(col_titles[2], fontsize=10)
        if row == n - 1:
            ax_f.set_xlabel("x (km)", fontsize=9)
        else:
            ax_f.tick_params(labelbottom=False)
        ax_f.tick_params(labelleft=False, labelsize=8)

        cax_v = fig.add_subplot(gs[row, 3])
        cb_v = fig.colorbar(im_v, cax=cax_v)
        cb_v.set_label("v (km/s)", fontsize=8)
        cb_v.ax.tick_params(labelsize=7)

        # Difference
        ax_d = fig.add_subplot(gs[row, 4])
        im_d = ax_d.imshow(diff, extent=extent, cmap="seismic",
                           vmin=-d_abs, vmax=+d_abs, aspect="equal",
                           interpolation="nearest")
        _add_acquisition_markers(ax_d, geom, extent)
        if row == 0:
            ax_d.set_title(col_titles[3], fontsize=10)
        if row == n - 1:
            ax_d.set_xlabel("x (km)", fontsize=9)
        else:
            ax_d.tick_params(labelbottom=False)
        ax_d.tick_params(labelleft=False, labelsize=8)

        cax_d = fig.add_subplot(gs[row, 5])
        cb_d = fig.colorbar(im_d, cax=cax_d)
        cb_d.set_label("Δv (km/s)", fontsize=8)
        cb_d.ax.tick_params(labelsize=7)

    return fig


# ============================================================================
# Main
# ============================================================================


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Paper-grade composite — MIMIR-TGV² vs FMM-LSMR (corrected paths)."
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    print("[compare] loading per-seed CSVs...")
    print(f"             MIMIR-TGV² source: {MIMIR_CSV}")
    print(f"             FMM source:        {BASELINE_CSV}")
    mimir_rows = _load_csv(MIMIR_CSV)
    baseline_rows = _load_csv(BASELINE_CSV)

    mimir_stats = _per_benchmark_stats(mimir_rows)
    baseline_stats = _per_benchmark_stats(baseline_rows)
    benches = sorted(set(mimir_stats) & set(baseline_stats))

    # Quick sanity check vs paper Table 1
    print("\n[compare] Sanity check against paper Table 1:")
    print(f"   {'benchmark':<25}{'mean RMSE':>15}  expected Table 1")
    expected = {
        "gaussian_anomaly":     "0.118 ± 0.015",
        "layered":              "0.226 ± 0.007",
        "curvefault_lookalike": "0.286 ± 0.010",
    }
    for b in benches:
        actual = f"{mimir_stats[b]['mean_rmse']:.3f} ± {mimir_stats[b]['std_rmse']:.3f}"
        match = "  ✓" if abs(mimir_stats[b]['mean_rmse']
                              - float(expected.get(b, "0.0").split()[0])) < 0.01 else "  ⚠ MISMATCH"
        print(f"   {b:<25}{actual:>15}  {expected.get(b, '?')}{match}")

    print("\n[compare] re-evaluating MIMIR-TGV² + FMM velocity grids...")
    device = resolve_device(args.device)
    print(banner(device))

    truths, specs = {}, {}
    mimir_v, baseline_v, geometries = {}, {}, {}
    mimir_metrics, fmm_metrics = {}, {}

    for b in benches:
        truth, spec = _load_truth(b)
        truths[b] = truth
        specs[b] = spec

        print(f"[compare]   reloading MIMIR-TGV²  for {b} ...")
        mimir_v[b] = _reload_mimir_velocity(b, spec, device)

        print(f"[compare]   re-running FMM         for {b} ...")
        v_b, geom, _seed = _reload_baseline_velocity(b, spec)
        baseline_v[b] = v_b
        geometries[b] = geom

        mimir_metrics[b] = _best_seed_metrics(mimir_rows, b)
        fmm_metrics[b]   = _best_seed_metrics(baseline_rows, b)

    fig = plot_composite(
        benches, truths, specs, mimir_v, baseline_v,
        geometries, mimir_metrics, fmm_metrics,
    )
    comp_pdf = save_pdf(fig, "11_comparison_composite", out_dir=PDF_DIR)
    plt.close(fig)
    print(f"[compare]   -> {comp_pdf}")
    print("\n[compare] Done. Composite PDF regenerated with TRUE MIMIR-TGV² data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
