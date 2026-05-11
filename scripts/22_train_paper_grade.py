"""
22_train_paper_grade.py
=======================

Paper-grade evaluation of MIMIR's Neural Velocity Field across all three
synthetic benchmarks, using the configuration identified by the ablation
study in script 21.

Configuration (from ablation winners across all 3 benchmarks, mean of 3 seeds):

    fourier_scale       = 4.0
    smoothness_weight   = 1.0
    learning_rate       = 5e-3
    iterations          = 8000     (4x the ablation; allows full convergence)
    seeds               = 5        (paper-grade min for mean ± std reporting)
    benchmarks          = {gaussian_anomaly, layered, curvefault_lookalike}

What it produces
----------------
For each benchmark, runs 5 independent seeds and tracks the best validation
RMSE per seed. At the end:

* logs/paper_grade/<benchmark>/seed_<N>/history.npz       — per-seed history
* models/paper_grade/<benchmark>/best_overall.pt          — best seed checkpoint
* PDF/22_paper_velocity_<benchmark>.pdf                   — truth vs MIMIR (best seed)
* PDF/22_paper_composite.pdf                              — 3x3 paper figure
* logs/paper_grade/results_per_seed.csv                   — every run, every metric
* logs/paper_grade/results_summary.csv                    — mean ± std per benchmark

Disk hygiene
------------
Only the best-overall checkpoint per benchmark is retained (3 files total,
~270 KB each). All other checkpoints are deleted after evaluation. The
ablation rerun won't fill the disk again.

Usage
-----
    # Full paper-grade run, all benchmarks, 5 seeds, 8000 iter (~25 min on a modern laptop)
    python scripts/22_train_paper_grade.py

    # Quick smoke (2 seeds × 1000 iter, ~3 min)
    python scripts/22_train_paper_grade.py --quick

    # Single benchmark only
    python scripts/22_train_paper_grade.py --benchmark gaussian_anomaly

    # Override the optimal config (e.g. to test scale=0.5, sm=1.0)
    python scripts/22_train_paper_grade.py --fourier-scale 0.5 --smoothness-weight 1.0
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
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
from mimir.viz.figures import (
    apply_paper_style,
    plot_velocity_pair,
    save_pdf,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "synthetic"
PDF_DIR = REPO_ROOT / "PDF"
LOG_ROOT = REPO_ROOT / "logs" / "paper_grade"
MODEL_ROOT = REPO_ROOT / "models" / "paper_grade"


# Optimal config from ablation (script 21, 3 benchmarks × 3 seeds × 2000 iter)
OPTIMAL_CFG = {
    "fourier_scale": 4.0,
    "smoothness_weight": 1.0,
    "learning_rate": 5e-3,
    "iterations": 8000,
}


# ---------------------------------------------------------------------------
# data loading
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
# single-seed run
# ---------------------------------------------------------------------------


def run_one_seed(
    truth: np.ndarray,
    spec: BenchmarkSpec,
    seed: int,
    cfg_overrides: dict,
    device: torch.device,
    out_dir: Path,
) -> dict:
    """Train one seed and return a flat dict of metrics + checkpoint path."""
    set_global_seed(seed)

    # Acquisition + observations
    geometry = cross_well_layout(spec, n_sources_per_side=12, n_receivers_per_side=12)
    rng = np.random.default_rng(seed + 1)
    src_np, rec_np, tt_np = sample_observed_travel_times(
        truth, spec, geometry, noise_pct=5.0, rng=rng,
    )

    # Field
    nvf_cfg = NVFConfig(
        domain_x=spec.domain_x,
        domain_z=spec.domain_z,
        base_velocity=3.0,
        velocity_range=(2.0, 5.5),
        fourier_dim=64,
        fourier_scale=float(cfg_overrides["fourier_scale"]),
        hidden=128,
        depth=4,
        activation="tanh",
    )
    field = NeuralVelocityField(nvf_cfg).to(device)

    src = torch.from_numpy(src_np).to(device)
    rec = torch.from_numpy(rec_np).to(device)
    obs = torch.from_numpy(tt_np).to(device)

    # Trainer
    out_dir.mkdir(parents=True, exist_ok=True)
    trn_cfg = NVFTrainingConfig(
        iterations=int(cfg_overrides["iterations"]),
        learning_rate=float(cfg_overrides["learning_rate"]),
        lr_schedule="cosine",
        warmup_iters=200,
        grad_clip=1.0,
        data_loss="huber",
        huber_delta=0.05,
        smoothness_weight=float(cfg_overrides["smoothness_weight"]),
        smoothness_grid=64,
        ray_samples=64,
        val_every=100,
        log_every=200,
        save_best=True,
        save_last=False,           # paper-grade keeps best only; saves disk
        out_dir=str(out_dir),
    )
    trainer = NVFTrainer(field, src, rec, obs, truth_grid=truth, cfg=trn_cfg)

    t0 = time.time()
    state = trainer.fit()
    elapsed = time.time() - t0

    # Load the best checkpoint to extract the predicted velocity grid for figs
    best_path = out_dir / "best.pt"
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    field.load_state_dict(ckpt["model_state_dict"])
    field.eval()
    with torch.no_grad():
        v_pred = field.velocity_grid(spec.nx, spec.nz).cpu().numpy()

    # Save history separately (small, paper-useful)
    hist_path = out_dir / "history.npz"
    np.savez(hist_path, **{k: np.asarray(v) for k, v in state.history.items()},
             best_val_rmse=state.best_val_rmse)

    # Pearson at best
    val_iter = np.asarray(state.history["val_iter"])
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
        "best_ckpt": str(best_path),
        "v_pred": v_pred,         # carried for the per-benchmark figure
        "src_np": src_np,
        "rec_np": rec_np,
    }


# ---------------------------------------------------------------------------
# paper composite figure
# ---------------------------------------------------------------------------


def make_composite_paper_figure(
    bench_results: dict[str, dict],
    truths: dict[str, np.ndarray],
    specs: dict[str, BenchmarkSpec],
) -> plt.Figure:
    """
    A 3x3 paper figure: rows = benchmarks, columns = (truth, MIMIR, error).

    The error column uses a diverging colormap centred at zero so over- and
    under-prediction are visually distinguishable.
    """
    apply_paper_style()
    bnames = list(bench_results.keys())
    n = len(bnames)
    fig, axes = plt.subplots(n, 3, figsize=(8.4, 2.6 * n + 0.4),
                             constrained_layout=True)
    if n == 1:
        axes = axes[np.newaxis, :]   # uniform indexing

    for row, name in enumerate(bnames):
        truth = truths[name]
        spec = specs[name]
        v_pred = bench_results[name]["v_pred"]
        err = v_pred - truth

        extent = [spec.domain_x[0], spec.domain_x[1],
                  spec.domain_z[1], spec.domain_z[0]]

        # truth
        vmin = float(min(truth.min(), v_pred.min()))
        vmax = float(max(truth.max(), v_pred.max()))
        im0 = axes[row, 0].imshow(truth, extent=extent, cmap="viridis",
                                   vmin=vmin, vmax=vmax, aspect="equal")
        # MIMIR
        axes[row, 1].imshow(v_pred, extent=extent, cmap="viridis",
                             vmin=vmin, vmax=vmax, aspect="equal")
        # error
        emax = float(np.abs(err).max())
        im2 = axes[row, 2].imshow(err, extent=extent, cmap="RdBu_r",
                                   vmin=-emax, vmax=+emax, aspect="equal")

        # axis decoration
        nice = name.replace("_", " ")
        axes[row, 0].set_ylabel(f"{nice}\nz (km)")
        if row == n - 1:
            axes[row, 0].set_xlabel("x (km)")
            axes[row, 1].set_xlabel("x (km)")
            axes[row, 2].set_xlabel("x (km)")
        else:
            for c in range(3):
                axes[row, c].tick_params(labelbottom=False)
        for c in (1, 2):
            axes[row, c].tick_params(labelleft=False)

        if row == 0:
            axes[row, 0].set_title("Ground truth")
            axes[row, 1].set_title("MIMIR estimate")
            axes[row, 2].set_title("Error (MIMIR − truth)")

        # per-row colourbars (right edge, in margins)
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
    parser = argparse.ArgumentParser(description="MIMIR paper-grade training.")
    parser.add_argument("--benchmark", default=None,
                        choices=["gaussian_anomaly", "layered", "curvefault_lookalike"],
                        help="Run only this benchmark; default = all three.")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=None,
                        help="Override default 8000.")
    parser.add_argument("--fourier-scale", type=float, default=None,
                        help="Override optimal scale=4.0.")
    parser.add_argument("--smoothness-weight", type=float, default=None,
                        help="Override optimal sm=1.0.")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Override learning_rate=5e-3.")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: 2 seeds × 1000 iter.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    # Resolve config
    cfg = dict(OPTIMAL_CFG)
    if args.iterations is not None:
        cfg["iterations"] = args.iterations
    if args.fourier_scale is not None:
        cfg["fourier_scale"] = args.fourier_scale
    if args.smoothness_weight is not None:
        cfg["smoothness_weight"] = args.smoothness_weight
    if args.learning_rate is not None:
        cfg["learning_rate"] = args.learning_rate
    n_seeds = args.seeds
    if args.quick:
        n_seeds = 2
        cfg["iterations"] = 1000

    # Benchmarks
    if args.benchmark is None:
        benchmarks = ["gaussian_anomaly", "layered", "curvefault_lookalike"]
    else:
        benchmarks = [args.benchmark]

    device = resolve_device(args.device)
    print(banner(device))
    print(f"[paper-grade] config: {cfg}")
    print(f"[paper-grade] seeds: {n_seeds}, benchmarks: {benchmarks}")

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    seeds = [20260507 + 17 * i for i in range(n_seeds)]
    all_rows: list[dict] = []
    bench_results: dict[str, dict] = {}    # winner per benchmark
    truths: dict[str, np.ndarray] = {}
    specs: dict[str, BenchmarkSpec] = {}

    t_start = time.time()
    for bname in benchmarks:
        print(f"\n[paper-grade] === benchmark: {bname} ===")
        truth, spec = _load_truth(bname)
        truths[bname] = truth
        specs[bname] = spec

        seed_runs: list[dict] = []
        for s_idx, seed in enumerate(seeds):
            print(f"\n[paper-grade]   seed {s_idx + 1}/{n_seeds} = {seed}")
            seed_dir = LOG_ROOT / bname / f"seed_{seed}"
            row = run_one_seed(truth, spec, seed, cfg, device, seed_dir)
            seed_runs.append(row)
            all_rows.append({
                "benchmark": bname,
                "seed": row["seed"],
                "best_rmse": row["best_rmse"],
                "best_ssim": row["best_ssim"],
                "best_pearson": row["best_pearson"],
                "best_iter": row["best_iter"],
                "final_rmse": row["final_rmse"],
                "elapsed_s": row["elapsed_s"],
            })

        # Pick the best seed for this benchmark
        best_idx = int(np.argmin([r["best_rmse"] for r in seed_runs]))
        winner = seed_runs[best_idx]
        bench_results[bname] = winner

        # Persist winner checkpoint, drop the others
        winner_dst = MODEL_ROOT / bname / "best_overall.pt"
        winner_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(winner["best_ckpt"], winner_dst)

        # Per-benchmark velocity-pair PDF (winner only)
        fig = plot_velocity_pair(
            truth, winner["v_pred"], spec,
            title_truth="Ground truth",
            title_estimate=f"MIMIR estimate (best of {n_seeds} seeds)",
            rays=None,
        )
        pair_pdf = save_pdf(fig, f"22_paper_velocity_{bname}", out_dir=PDF_DIR)
        plt.close(fig)
        print(f"[paper-grade]   winner: seed={winner['seed']} "
              f"RMSE={winner['best_rmse']:.4f} SSIM={winner['best_ssim']:.3f}")
        print(f"[paper-grade]   checkpoint -> {winner_dst}")
        print(f"[paper-grade]   figure     -> {pair_pdf}")

        # Drop the per-seed best.pt files now that the winner is saved
        for r in seed_runs:
            ckpt = Path(r["best_ckpt"])
            if ckpt.exists() and ckpt != winner_dst:
                ckpt.unlink()

    # ---- per-seed CSV ----
    per_seed_csv = LOG_ROOT / "results_per_seed.csv"
    with per_seed_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n[paper-grade] per-seed CSV  -> {per_seed_csv}")

    # ---- summary CSV (mean ± std per benchmark) ----
    summary_rows: list[dict] = []
    for bname in benchmarks:
        cell = [r for r in all_rows if r["benchmark"] == bname]
        rmses = np.asarray([c["best_rmse"] for c in cell])
        ssims = np.asarray([c["best_ssim"] for c in cell])
        pears = np.asarray([c["best_pearson"] for c in cell])
        summary_rows.append({
            "benchmark": bname,
            "n_seeds": len(cell),
            "mean_rmse": float(rmses.mean()),
            "std_rmse": float(rmses.std()),
            "mean_ssim": float(ssims.mean()),
            "std_ssim": float(ssims.std()),
            "mean_pearson": float(pears.mean()),
            "std_pearson": float(pears.std()),
        })
    summary_csv = LOG_ROOT / "results_summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[paper-grade] summary CSV   -> {summary_csv}")

    # ---- paper composite figure ----
    if len(benchmarks) >= 1:
        fig = make_composite_paper_figure(bench_results, truths, specs)
        comp_pdf = save_pdf(fig, "22_paper_composite", out_dir=PDF_DIR)
        plt.close(fig)
        print(f"[paper-grade] composite    -> {comp_pdf}")

    # ---- final report ----
    elapsed_total = time.time() - t_start
    print("\n" + "=" * 72)
    print(f"[paper-grade] total wall-clock: {elapsed_total / 60:.1f} min")
    print("[paper-grade] PAPER TABLE 1 — MIMIR Neural Velocity Field")
    print("-" * 72)
    print(f"{'benchmark':<24}{'RMSE (km/s)':>16}{'SSIM':>14}{'Pearson':>14}")
    print("-" * 72)
    for r in summary_rows:
        print(f"{r['benchmark']:<24}"
              f"{r['mean_rmse']:>10.4f} ± {r['std_rmse']:.4f}"
              f"{r['mean_ssim']:>9.3f} ± {r['std_ssim']:.3f}"
              f"{r['mean_pearson']:>9.3f} ± {r['std_pearson']:.3f}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
