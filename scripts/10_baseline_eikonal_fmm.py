"""
10_baseline_eikonal_fmm.py
==========================

Run the classical iterative travel-time tomography baseline (FMM + LSMR +
Tikhonov + 2D Laplacian smoothing) on the same three synthetic benchmarks
and the same multi-seed observation set used by `22_train_paper_grade.py`,
producing strictly comparable metrics.

Algorithm
---------
See `mimir.baselines.eikonal_tomography` for the full algorithm description.
Briefly: each outer iteration solves the eikonal forward problem with FMM,
back-traces curved rays from receivers along -∇T, builds a sparse Jacobian
of path lengths through cells, and applies a regularized LSMR update to the
slowness field. We track the iteration with the lowest validation RMSE
("best") and report it.

Hyperparameter selection
------------------------
For a fair comparison we do not hand-pick (damping, smoothing) — instead we
auto-grid-search over a modest range *per benchmark*, using seed 0 to pick
the optimum, and only then run the remaining seeds at that fixed setting.
This mirrors how a careful seismologist would tune the inversion using a
small validation slice. See `--no-grid-search` to disable.

Outputs
-------
    logs/baseline_eikonal_fmm/<benchmark>/best_config.txt
    logs/baseline_eikonal_fmm/results_per_seed.csv
    logs/baseline_eikonal_fmm/results_summary.csv
    PDF/10_baseline_velocity_<benchmark>.pdf       — truth vs FMM (best seed)
    PDF/10_baseline_composite.pdf                  — 3 benchmarks composite

Usage
-----
    # Full run, all benchmarks, 5 seeds, with hyperparameter grid search
    python scripts/10_baseline_eikonal_fmm.py

    # Quick smoke (gaussian only, 2 seeds, 8 outer iter)
    python scripts/10_baseline_eikonal_fmm.py --quick

    # Skip grid search (use specified damping / smoothing)
    python scripts/10_baseline_eikonal_fmm.py --no-grid-search --damping 1e-3 --smoothing 50
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mimir.baselines import (
    ClassicalEikonalTomography,
    EikonalTomographyConfig,
    EikonalTomographyState,
)
from mimir.data.acquisition import cross_well_layout, sample_observed_travel_times
from mimir.data.benchmarks import BenchmarkSpec
from mimir.utils.seed import set_global_seed
from mimir.viz.figures import (
    apply_paper_style,
    plot_velocity_pair,
    save_pdf,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "synthetic"
PDF_DIR = REPO_ROOT / "PDF"
LOG_ROOT = REPO_ROOT / "logs" / "baseline_eikonal_fmm"


# Hyperparameter grid for the per-benchmark auto-tune
GRID_DAMPING = [1e-3, 1e-2]
GRID_SMOOTHING = [5.0, 20.0, 50.0]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_truth(name: str) -> tuple[np.ndarray, BenchmarkSpec]:
    npz_path = DATA_DIR / f"{name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found. Run scripts/01_generate_synthetic_benchmarks.py first."
        )
    arch = np.load(npz_path, allow_pickle=False)
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


def _make_observations(truth: np.ndarray, spec: BenchmarkSpec, seed: int):
    """Reproduce exactly the same observations as MIMIR's script 22."""
    geometry = cross_well_layout(spec, n_sources_per_side=12, n_receivers_per_side=12)
    rng = np.random.default_rng(seed + 1)
    return sample_observed_travel_times(truth, spec, geometry, noise_pct=5.0, rng=rng)


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity as ski_ssim
    rng = float(max(a.max(), b.max()) - min(a.min(), b.min()))
    if rng <= 0:
        return float("nan")
    return float(ski_ssim(a.astype(np.float64), b.astype(np.float64), data_range=rng))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    av = a.flatten() - a.mean()
    bv = b.flatten() - b.mean()
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 0:
        return float("nan")
    return float((av * bv).sum() / denom)


# ---------------------------------------------------------------------------
# per-seed run
# ---------------------------------------------------------------------------


def run_one_seed(
    truth: np.ndarray,
    spec: BenchmarkSpec,
    seed: int,
    damping: float,
    smoothing: float,
    max_iter: int,
    verbose: bool = False,
) -> dict:
    """Run classical inversion for one seed, return a dict of metrics."""
    set_global_seed(seed)
    sources, receivers, tt = _make_observations(truth, spec, seed)

    cfg = EikonalTomographyConfig(
        base_velocity=3.0,
        velocity_min=2.0,
        velocity_max=5.5,
        max_iter=max_iter,
        damping=damping,
        smoothing=smoothing,
        step_size=0.5,
        log_every=max_iter,        # silence per-iter logging
    )
    solver = ClassicalEikonalTomography(spec, cfg)

    t0 = time.time()
    v_best, state = solver.fit(sources, receivers, tt, truth_grid=truth, verbose=verbose)
    elapsed = time.time() - t0

    # Compute per-metric results at best iter
    val_iter = np.asarray(state.history["iter"])
    val_rmse = np.asarray(state.history["val_rmse"])
    val_ssim = np.asarray(state.history["val_ssim"])
    val_pear = np.asarray(state.history["val_pearson"])
    best_idx = int(np.argmin(val_rmse))

    return {
        "seed": int(seed),
        "best_rmse": float(state.best_val_rmse),
        "best_iter": int(val_iter[best_idx]),
        "best_ssim": float(val_ssim[best_idx]),
        "best_pearson": float(val_pear[best_idx]),
        "final_rmse": float(val_rmse[-1]),
        "elapsed_s": float(elapsed),
        "v_best": v_best,
        "src_np": sources,
        "rec_np": receivers,
    }


# ---------------------------------------------------------------------------
# composite figure (matches 22_paper_composite layout)
# ---------------------------------------------------------------------------


def make_composite(
    bench_results: dict[str, dict],
    truths: dict[str, np.ndarray],
    specs: dict[str, BenchmarkSpec],
) -> plt.Figure:
    """3 x 3 paper figure: rows = benchmarks, columns = (truth, FMM, error)."""
    apply_paper_style()
    bnames = list(bench_results.keys())
    n = len(bnames)
    fig, axes = plt.subplots(n, 3, figsize=(8.4, 2.6 * n + 0.4),
                             constrained_layout=True)
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, name in enumerate(bnames):
        truth = truths[name]
        spec = specs[name]
        v_pred = bench_results[name]["v_best"]
        err = v_pred - truth

        extent = [spec.domain_x[0], spec.domain_x[1],
                  spec.domain_z[1], spec.domain_z[0]]

        vmin = float(min(truth.min(), v_pred.min()))
        vmax = float(max(truth.max(), v_pred.max()))
        im0 = axes[row, 0].imshow(truth, extent=extent, cmap="viridis",
                                  vmin=vmin, vmax=vmax, aspect="equal")
        axes[row, 1].imshow(v_pred, extent=extent, cmap="viridis",
                            vmin=vmin, vmax=vmax, aspect="equal")
        emax = float(np.abs(err).max()) if np.any(err) else 1.0
        im2 = axes[row, 2].imshow(err, extent=extent, cmap="RdBu_r",
                                  vmin=-emax, vmax=+emax, aspect="equal")

        nice = name.replace("_", " ")
        axes[row, 0].set_ylabel(f"{nice}\nz (km)")
        if row == n - 1:
            for c in range(3):
                axes[row, c].set_xlabel("x (km)")
        else:
            for c in range(3):
                axes[row, c].tick_params(labelbottom=False)
        for c in (1, 2):
            axes[row, c].tick_params(labelleft=False)

        if row == 0:
            axes[row, 0].set_title("Ground truth")
            axes[row, 1].set_title("Classical FMM-LSMR")
            axes[row, 2].set_title("Error (FMM − truth)")

        cb0 = fig.colorbar(im0, ax=axes[row, :2], location="right",
                           shrink=0.85, pad=0.02)
        cb0.set_label("v (km/s)")
        cb2 = fig.colorbar(im2, ax=axes[row, 2], location="right",
                           shrink=0.85, pad=0.02)
        cb2.set_label("Δv (km/s)")

    return fig


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MIMIR classical FMM-LSMR baseline.")
    parser.add_argument("--benchmark", default=None,
                        choices=["gaussian_anomaly", "layered", "curvefault_lookalike"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--smoothing", type=float, default=20.0)
    parser.add_argument("--no-grid-search", action="store_true",
                        help="Skip per-benchmark hyperparameter grid search.")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: gaussian only, 2 seeds, max_iter=8.")
    args = parser.parse_args(argv)

    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    n_seeds = args.seeds
    max_iter = args.max_iter
    if args.quick:
        n_seeds = 2
        max_iter = 8
        args.benchmark = args.benchmark or "gaussian_anomaly"

    benchmarks = (
        [args.benchmark] if args.benchmark
        else ["gaussian_anomaly", "layered", "curvefault_lookalike"]
    )

    seeds = [20260507 + 17 * i for i in range(n_seeds)]
    print(f"[baseline] benchmarks: {benchmarks}, seeds: {seeds}, max_iter: {max_iter}")
    print(f"[baseline] grid search: {not args.no_grid_search}")

    all_rows: list[dict] = []
    bench_results: dict[str, dict] = {}
    truths: dict[str, np.ndarray] = {}
    specs: dict[str, BenchmarkSpec] = {}
    best_configs: dict[str, tuple[float, float]] = {}

    t_start = time.time()

    for bname in benchmarks:
        print(f"\n[baseline] === benchmark: {bname} ===")
        truth, spec = _load_truth(bname)
        truths[bname] = truth
        specs[bname] = spec

        # ---- 1. Per-benchmark hyperparameter grid search on seeds[0] ----
        if args.no_grid_search:
            best_damp, best_sm = args.damping, args.smoothing
            print(f"[baseline]   using fixed damping={best_damp}, smoothing={best_sm}")
        else:
            print(f"[baseline]   grid search on seed={seeds[0]}: "
                  f"damping={GRID_DAMPING}, smoothing={GRID_SMOOTHING}")
            best_rmse_so_far = np.inf
            best_damp, best_sm = GRID_DAMPING[0], GRID_SMOOTHING[0]
            for damp, sm in product(GRID_DAMPING, GRID_SMOOTHING):
                t0 = time.time()
                row = run_one_seed(truth, spec, seeds[0], damp, sm,
                                   max_iter=max_iter, verbose=False)
                if row["best_rmse"] < best_rmse_so_far:
                    best_rmse_so_far = row["best_rmse"]
                    best_damp, best_sm = damp, sm
                print(f"[baseline]     damping={damp:.0e} smoothing={sm:>5.1f} "
                      f"-> rmse={row['best_rmse']:.4f}  ({time.time()-t0:.0f}s)")
            print(f"[baseline]   winner: damping={best_damp:.0e} smoothing={best_sm}")

        best_configs[bname] = (best_damp, best_sm)
        # Persist best config for reproducibility
        cfg_path = LOG_ROOT / bname / "best_config.txt"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(
            f"damping={best_damp}\nsmoothing={best_sm}\nmax_iter={max_iter}\n"
        )

        # ---- 2. Run all seeds at the chosen config ----
        seed_rows: list[dict] = []
        for s_idx, seed in enumerate(seeds):
            print(f"[baseline]   seed {s_idx + 1}/{n_seeds} = {seed}")
            t0 = time.time()
            row = run_one_seed(truth, spec, seed, best_damp, best_sm,
                               max_iter=max_iter, verbose=False)
            print(f"[baseline]     rmse={row['best_rmse']:.4f}  "
                  f"ssim={row['best_ssim']:.3f}  ({time.time()-t0:.0f}s)")
            seed_rows.append(row)
            all_rows.append({
                "benchmark": bname,
                "seed": row["seed"],
                "best_rmse": row["best_rmse"],
                "best_ssim": row["best_ssim"],
                "best_pearson": row["best_pearson"],
                "best_iter": row["best_iter"],
                "final_rmse": row["final_rmse"],
                "elapsed_s": row["elapsed_s"],
                "damping": best_damp,
                "smoothing": best_sm,
            })

        # ---- 3. Pick winner seed and produce per-benchmark figure ----
        best_seed_idx = int(np.argmin([r["best_rmse"] for r in seed_rows]))
        winner = seed_rows[best_seed_idx]
        bench_results[bname] = winner

        fig = plot_velocity_pair(
            truth, winner["v_best"], spec,
            title_truth="Ground truth",
            title_estimate=f"Classical FMM-LSMR (best of {n_seeds} seeds)",
            rays=None,
        )
        pair_pdf = save_pdf(fig, f"10_baseline_velocity_{bname}", out_dir=PDF_DIR)
        plt.close(fig)
        print(f"[baseline]   figure -> {pair_pdf}")

    # ---- per-seed CSV ----
    per_seed_csv = LOG_ROOT / "results_per_seed.csv"
    with per_seed_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n[baseline] per-seed CSV  -> {per_seed_csv}")

    # ---- summary CSV ----
    summary: list[dict] = []
    for bname in benchmarks:
        cell = [r for r in all_rows if r["benchmark"] == bname]
        rmses = np.asarray([c["best_rmse"] for c in cell])
        ssims = np.asarray([c["best_ssim"] for c in cell])
        pears = np.asarray([c["best_pearson"] for c in cell])
        summary.append({
            "benchmark": bname,
            "n_seeds": len(cell),
            "damping": best_configs[bname][0],
            "smoothing": best_configs[bname][1],
            "mean_rmse": float(rmses.mean()),
            "std_rmse": float(rmses.std()),
            "mean_ssim": float(ssims.mean()),
            "std_ssim": float(ssims.std()),
            "mean_pearson": float(pears.mean()),
            "std_pearson": float(pears.std()),
        })
    summary_csv = LOG_ROOT / "results_summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"[baseline] summary CSV   -> {summary_csv}")

    # ---- composite ----
    if len(benchmarks) >= 1:
        fig = make_composite(bench_results, truths, specs)
        comp_pdf = save_pdf(fig, "10_baseline_composite", out_dir=PDF_DIR)
        plt.close(fig)
        print(f"[baseline] composite     -> {comp_pdf}")

    # ---- final report ----
    elapsed = time.time() - t_start
    print("\n" + "=" * 72)
    print(f"[baseline] total wall-clock: {elapsed / 60:.1f} min")
    print("[baseline] BASELINE TABLE — Classical FMM-LSMR Tomography")
    print("-" * 72)
    print(f"{'benchmark':<24}{'RMSE (km/s)':>16}{'SSIM':>14}{'Pearson':>14}")
    print("-" * 72)
    for r in summary:
        print(f"{r['benchmark']:<24}"
              f"{r['mean_rmse']:>10.4f} ± {r['std_rmse']:.4f}"
              f"{r['mean_ssim']:>9.3f} ± {r['std_ssim']:.3f}"
              f"{r['mean_pearson']:>9.3f} ± {r['std_pearson']:.3f}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
