"""
21_ablation_fourier_smoothness.py
=================================

Systematic 2D ablation study of MIMIR's two most critical hyperparameters:

   * ``fourier_scale``        — controls the spectral bandwidth of the field
   * ``smoothness_weight``    — controls the TV regularizer strength

Motivation
----------
Empirical reconnaissance (script 20 single-shot runs) revealed that the
Neural Velocity Field exhibits *three distinct failure modes* depending on
where in the (scale, smoothness) plane the experiment lies:

   (i)   low smoothness              -> RMSE diverges monotonically
   (ii)  moderate smoothness         -> RMSE stabilises but worse than init
   (iii) high smoothness + low scale -> RMSE briefly improves, then overshoots

This script maps the entire 2D landscape with controlled multi-seed
statistics, so the paper has a defensible recommendation: "use values in
this region".

Sweep
-----
   fourier_scale       ∈ {0.5, 1.0, 2.0, 4.0, 8.0}             (5 values)
   smoothness_weight   ∈ {1e-3, 1e-2, 1e-1, 1e+0}              (4 values)
   seeds               = 3 (paper-grade min; the plain runner uses 5)
   iterations          = 2000                                  (enough to see all 3 modes)

Total = 5 x 4 x 3 = 60 runs. On a laptop GPU (e.g. Apple M-series MPS) (≈67 it/s) this is roughly
60 * 30 s = 30 minutes. On CPU expect ~6–7 hours.

Outputs
-------
   logs/ablation_fourier_smoothness/
      results.csv                  — every run, every metric
      summary.csv                  — mean ± std per (scale, smoothness)
   PDF/
      21_ablation_heatmap_rmse.pdf
      21_ablation_heatmap_ssim.pdf
      21_ablation_curves.pdf       — RMSE vs iteration, faceted

Usage
-----
   # Default (gaussian_anomaly, overnight friendly):
   python scripts/21_ablation_fourier_smoothness.py

   # Quick smoke (12 runs, ~3 min on a modern laptop):
   python scripts/21_ablation_fourier_smoothness.py --quick

   # Different benchmark:
   python scripts/21_ablation_fourier_smoothness.py --benchmark layered
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from mimir.data.acquisition import cross_well_layout, sample_observed_travel_times
from mimir.data.benchmarks import BenchmarkSpec
from mimir.fields import NeuralVelocityField
from mimir.fields.neural_velocity_field import NVFConfig
from mimir.training import NVFTrainer, NVFTrainingConfig
from mimir.utils import resolve_device, set_global_seed
from mimir.utils.device import banner
from mimir.viz.figures import apply_paper_style, save_pdf


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "synthetic"
PDF_DIR = REPO_ROOT / "PDF"
LOG_DIR = REPO_ROOT / "logs" / "ablation_fourier_smoothness"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def _load_truth(name: str) -> tuple[np.ndarray, BenchmarkSpec]:
    npz_path = DATA_DIR / f"{name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Truth file {npz_path} not found. "
            "Run scripts/01_generate_synthetic_benchmarks.py first."
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


# ---------------------------------------------------------------------------
# single run
# ---------------------------------------------------------------------------


def run_one(
    truth: np.ndarray,
    spec: BenchmarkSpec,
    fourier_scale: float,
    smoothness_weight: float,
    seed: int,
    iterations: int,
    device: torch.device,
    out_root: Path,
) -> dict:
    """Run a single training and return a flat dict of metrics."""
    set_global_seed(seed)

    geometry = cross_well_layout(spec, n_sources_per_side=12, n_receivers_per_side=12)
    rng = np.random.default_rng(seed + 1)
    src_np, rec_np, tt_np = sample_observed_travel_times(
        truth, spec, geometry, noise_pct=5.0, rng=rng,
    )

    nvf_cfg = NVFConfig(
        domain_x=spec.domain_x,
        domain_z=spec.domain_z,
        base_velocity=3.0,
        velocity_range=(2.0, 5.5),
        fourier_dim=64,
        fourier_scale=fourier_scale,
        hidden=128,
        depth=4,
        activation="tanh",
    )
    field = NeuralVelocityField(nvf_cfg).to(device)

    src = torch.from_numpy(src_np).to(device)
    rec = torch.from_numpy(rec_np).to(device)
    obs = torch.from_numpy(tt_np).to(device)

    tag = f"fs{fourier_scale}_sm{smoothness_weight:.0e}_seed{seed}"
    run_dir = out_root / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    trn_cfg = NVFTrainingConfig(
        iterations=iterations,
        learning_rate=5e-3,
        lr_schedule="cosine",
        warmup_iters=100,
        grad_clip=1.0,
        data_loss="huber",
        huber_delta=0.05,
        smoothness_weight=smoothness_weight,
        smoothness_grid=64,
        ray_samples=64,
        val_every=50,
        log_every=200,
        save_best=True,
        out_dir=str(run_dir),
    )
    trainer = NVFTrainer(field, src, rec, obs, truth_grid=truth, cfg=trn_cfg)

    t0 = time.time()
    state = trainer.fit()
    elapsed = time.time() - t0

    # Identify iter at which best RMSE was achieved
    val_iter = np.asarray(state.history["val_iter"])
    val_rmse = np.asarray(state.history["val_rmse"])
    val_ssim = np.asarray(state.history["val_ssim"])
    best_idx = int(np.argmin(val_rmse))

    return {
        "fourier_scale": float(fourier_scale),
        "smoothness_weight": float(smoothness_weight),
        "seed": int(seed),
        "best_rmse": float(state.best_val_rmse),
        "best_iter": int(val_iter[best_idx]),
        "best_ssim_at_best_rmse": float(val_ssim[best_idx]),
        "final_rmse": float(val_rmse[-1]),
        "final_ssim": float(val_ssim[-1]),
        "elapsed_s": float(elapsed),
    }


# ---------------------------------------------------------------------------
# heat-map plotting
# ---------------------------------------------------------------------------


def plot_heatmap(
    summary_rows: list[dict],
    metric: str,
    title: str,
    cbar_label: str,
    invert_cmap: bool = False,
) -> plt.Figure:
    """Plot a 2D heatmap of mean(metric) over (scale, smoothness)."""
    apply_paper_style()
    scales = sorted(set(r["fourier_scale"] for r in summary_rows))
    sms = sorted(set(r["smoothness_weight"] for r in summary_rows))

    grid = np.full((len(sms), len(scales)), np.nan)
    for r in summary_rows:
        i = sms.index(r["smoothness_weight"])
        j = scales.index(r["fourier_scale"])
        grid[i, j] = r[metric]

    cmap = "viridis_r" if invert_cmap else "viridis"
    fig, ax = plt.subplots(figsize=(5.4, 4.0), constrained_layout=True)
    im = ax.imshow(grid, cmap=cmap, aspect="auto", origin="lower")
    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels([f"{s:g}" for s in scales])
    ax.set_yticks(range(len(sms)))
    ax.set_yticklabels([f"{s:.0e}" for s in sms])
    ax.set_xlabel("Fourier scale")
    ax.set_ylabel("Smoothness weight")
    ax.set_title(title)

    # annotate cells with the value
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=7, color="white",
                        path_effects=None)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.04)
    cbar.set_label(cbar_label)
    return fig


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MIMIR ablation: Fourier scale × smoothness.")
    parser.add_argument("--benchmark", default="gaussian_anomaly",
                        choices=["gaussian_anomaly", "layered", "curvefault_lookalike"])
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--quick", action="store_true",
                        help="Smaller sweep for a fast smoke test.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(banner(device))

    truth, spec = _load_truth(args.benchmark)
    print(f"[ablation] benchmark={args.benchmark}, iter/run={args.iterations}, seeds/cell={args.seeds}")

    if args.quick:
        scales = [1.0, 2.0, 4.0]
        sms = [1e-2, 1e-1]
    else:
        scales = [0.5, 1.0, 2.0, 4.0, 8.0]
        sms = [1e-3, 1e-2, 1e-1, 1e0]

    seeds = [20260507 + 17 * i for i in range(args.seeds)]
    total = len(scales) * len(sms) * len(seeds)
    print(f"[ablation] total runs: {total} "
          f"(scales={scales}, sms={sms}, seeds={seeds})")

    results: list[dict] = []
    run_idx = 0
    t_start = time.time()
    for sc in scales:
        for sm in sms:
            for seed in seeds:
                run_idx += 1
                print(f"\n[ablation] === run {run_idx}/{total} === "
                      f"scale={sc}  sm={sm:.0e}  seed={seed}")
                row = run_one(truth, spec, sc, sm, seed, args.iterations,
                              device, LOG_DIR)
                results.append(row)

    elapsed_total = time.time() - t_start
    print(f"\n[ablation] total wall-clock: {elapsed_total/60:.1f} min")

    # ---- write per-run CSV ----
    results_csv = LOG_DIR / "results.csv"
    with results_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"[ablation] per-run results -> {results_csv}")

    # ---- summarise: mean ± std per (scale, smoothness) ----
    summary: list[dict] = []
    for sc in scales:
        for sm in sms:
            cell = [r for r in results if r["fourier_scale"] == sc and r["smoothness_weight"] == sm]
            best_rmses = np.asarray([c["best_rmse"] for c in cell])
            best_ssims = np.asarray([c["best_ssim_at_best_rmse"] for c in cell])
            summary.append({
                "fourier_scale": sc,
                "smoothness_weight": sm,
                "n_seeds": len(cell),
                "mean_best_rmse": float(best_rmses.mean()),
                "std_best_rmse": float(best_rmses.std()),
                "mean_best_ssim": float(best_ssims.mean()),
                "std_best_ssim": float(best_ssims.std()),
            })

    summary_csv = LOG_DIR / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"[ablation] summary       -> {summary_csv}")

    # ---- heatmap PDFs ----
    fig_rmse = plot_heatmap(
        [{"fourier_scale": s["fourier_scale"],
          "smoothness_weight": s["smoothness_weight"],
          "mean_best_rmse": s["mean_best_rmse"]} for s in summary],
        metric="mean_best_rmse",
        title=f"Best validation RMSE — {args.benchmark} (mean of {args.seeds} seeds)",
        cbar_label="RMSE (km/s)",
        invert_cmap=True,    # lower is better, so invert so green=good
    )
    rmse_pdf = save_pdf(fig_rmse, f"21_ablation_heatmap_rmse_{args.benchmark}", out_dir=PDF_DIR)
    print(f"[ablation] RMSE heatmap  -> {rmse_pdf}")

    fig_ssim = plot_heatmap(
        [{"fourier_scale": s["fourier_scale"],
          "smoothness_weight": s["smoothness_weight"],
          "mean_best_ssim": s["mean_best_ssim"]} for s in summary],
        metric="mean_best_ssim",
        title=f"SSIM at best RMSE — {args.benchmark} (mean of {args.seeds} seeds)",
        cbar_label="SSIM",
        invert_cmap=False,   # higher SSIM is better
    )
    ssim_pdf = save_pdf(fig_ssim, f"21_ablation_heatmap_ssim_{args.benchmark}", out_dir=PDF_DIR)
    print(f"[ablation] SSIM heatmap  -> {ssim_pdf}")

    # ---- best config recommendation ----
    best_idx = int(np.argmin([s["mean_best_rmse"] for s in summary]))
    rec = summary[best_idx]
    print("\n" + "=" * 72)
    print("[ablation] RECOMMENDED CONFIG")
    print(f"  fourier_scale       = {rec['fourier_scale']}")
    print(f"  smoothness_weight   = {rec['smoothness_weight']:.0e}")
    print(f"  expected best RMSE  = {rec['mean_best_rmse']:.4f} ± {rec['std_best_rmse']:.4f} km/s")
    print(f"  expected best SSIM  = {rec['mean_best_ssim']:.4f} ± {rec['std_best_ssim']:.4f}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
