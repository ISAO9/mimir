"""
42_ablation_tgv_alphas.py
==========================

Ablation over the TGV^2 weight pair (alpha_0, alpha_1) for Supplementary
material. This addresses the natural reviewer concern: "your TGV^2 result
depends on the choice (alpha_0, alpha_1) = (1.0, 2.0). Did you tune?"

Hypothesis (from Bredies-Kunisch-Pock 2010 imaging recommendations):
    The optimum is at alpha_0/alpha_1 ≈ 0.5, i.e. alpha_1 should be
    roughly twice alpha_0. Robustness within ~30% of this ratio is
    expected.

Configuration
-------------
* Single benchmark: gaussian_anomaly (the most TGV-sensitive)
* 4 x 4 grid:
    alpha_0 in {0.5, 1.0, 2.0, 4.0}
    alpha_1 in {0.5, 1.0, 2.0, 4.0}
* 3 seeds per cell (16 cells x 3 seeds = 48 runs)
* 4000 iter (half of paper-grade — saves time, RMSE has plateaued by 4k)
* Otherwise identical to script 40

Estimated wall-clock on a modern laptop: ~10-12 min.

Outputs
-------
    logs/ablation_tgv/results.csv           — per-run metrics
    logs/ablation_tgv/summary.csv           — mean ± std per cell
    PDF/42_ablation_tgv_heatmap.pdf         — supplementary figure

Usage
-----
    python scripts/42_ablation_tgv_alphas.py

    # Single seed quick test (~3 min)
    python scripts/42_ablation_tgv_alphas.py --seeds 1 --iterations 1000
"""

from __future__ import annotations

import argparse
import csv
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
from mimir.losses import AuxFieldConfig, AuxiliaryVectorField
from mimir.training import TGVTrainer, TGVTrainingConfig
from mimir.utils import resolve_device, set_global_seed
from mimir.utils.device import banner
from mimir.viz.figures import apply_paper_style, save_pdf


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "synthetic"
PDF_DIR = REPO_ROOT / "PDF"
LOG_ROOT = REPO_ROOT / "logs" / "ablation_tgv"


GRID_ALPHA_0 = [0.5, 1.0, 2.0, 4.0]
GRID_ALPHA_1 = [0.5, 1.0, 2.0, 4.0]


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


def run_one(
    truth: np.ndarray,
    spec: BenchmarkSpec,
    seed: int,
    alpha_0: float,
    alpha_1: float,
    iterations: int,
    device: torch.device,
    out_dir: Path,
) -> dict:
    """Single run; returns per-run metrics (no checkpoint kept)."""
    set_global_seed(seed)
    geometry = cross_well_layout(spec, n_sources_per_side=12, n_receivers_per_side=12)
    rng = np.random.default_rng(seed + 1)
    src_np, rec_np, tt_np = sample_observed_travel_times(
        truth, spec, geometry, noise_pct=5.0, rng=rng,
    )

    nvf_cfg = NVFConfig(
        domain_x=spec.domain_x, domain_z=spec.domain_z,
        base_velocity=3.0, velocity_range=(2.0, 5.5),
        fourier_dim=64, fourier_scale=4.0,
        hidden=128, depth=4, activation="tanh",
    )
    field = NeuralVelocityField(nvf_cfg).to(device)

    aux_cfg = AuxFieldConfig(
        domain_x=spec.domain_x, domain_z=spec.domain_z,
        fourier_dim=32, fourier_scale=2.0,
        hidden=64, depth=3, activation="tanh",
    )
    aux_field = AuxiliaryVectorField(aux_cfg).to(device)

    src = torch.from_numpy(src_np).to(device)
    rec = torch.from_numpy(rec_np).to(device)
    obs = torch.from_numpy(tt_np).to(device)

    out_dir.mkdir(parents=True, exist_ok=True)
    trn_cfg = TGVTrainingConfig(
        iterations=iterations, learning_rate=5e-3,
        lr_schedule="cosine", warmup_iters=200,
        grad_clip=1.0,
        data_loss="huber", huber_delta=0.05,
        smoothness_weight=1.0,
        smoothness_grid=64, ray_samples=64,
        val_every=100, log_every=iterations,    # silence
        save_best=True,                          # MUST be True for trainer to track best (bug in trainer)
        save_last=False,
        out_dir=str(out_dir),
        tgv_alpha_0=alpha_0, tgv_alpha_1=alpha_1,
    )
    trainer = TGVTrainer(
        field=field, aux_field=aux_field,
        sources=src, receivers=rec, observed_tt=obs,
        truth_grid=truth, cfg=trn_cfg,
    )
    state = trainer.fit()

    # Delete the checkpoint we don't need (ablation only cares about metrics)
    ckpt_path = out_dir / "best.pt"
    if ckpt_path.exists():
        ckpt_path.unlink()

    val_rmse = np.asarray(state.history["val_rmse"])
    val_ssim = np.asarray(state.history["val_ssim"])
    best_idx = int(np.argmin(val_rmse))
    return {
        "alpha_0": alpha_0,
        "alpha_1": alpha_1,
        "seed": int(seed),
        "best_rmse": float(state.best_val_rmse),
        "best_ssim": float(val_ssim[best_idx]),
    }


def plot_heatmap(summary: list[dict], out_path: Path) -> Path:
    """4x4 heatmap of mean RMSE over (alpha_0, alpha_1)."""
    apply_paper_style()
    n0 = len(GRID_ALPHA_0)
    n1 = len(GRID_ALPHA_1)
    grid = np.full((n0, n1), np.nan)
    for r in summary:
        i = GRID_ALPHA_0.index(r["alpha_0"])
        j = GRID_ALPHA_1.index(r["alpha_1"])
        grid[i, j] = r["mean_rmse"]

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    im = ax.imshow(grid, cmap="viridis_r", aspect="auto")

    ax.set_xticks(np.arange(n1))
    ax.set_xticklabels([f"{a:g}" for a in GRID_ALPHA_1])
    ax.set_yticks(np.arange(n0))
    ax.set_yticklabels([f"{a:g}" for a in GRID_ALPHA_0])
    ax.set_xlabel(r"$\alpha_1$  (TV-like term weight)")
    ax.set_ylabel(r"$\alpha_0$  (Hessian-like term weight)")
    ax.set_title(r"TGV$^2$ ablation on gaussian\_anomaly: best RMSE")

    for i in range(n0):
        for j in range(n1):
            val = grid[i, j]
            if np.isnan(val):
                continue
            v_norm = (val - grid[~np.isnan(grid)].min()) / (
                grid[~np.isnan(grid)].max() - grid[~np.isnan(grid)].min() + 1e-9
            )
            color = "white" if v_norm > 0.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    color=color, fontsize=8)

    # Highlight the recommended (1.0, 2.0)
    try:
        oi = GRID_ALPHA_0.index(1.0)
        oj = GRID_ALPHA_1.index(2.0)
        ax.scatter([oj], [oi], marker="o", s=120, facecolors="none",
                   edgecolors="red", linewidths=1.5, zorder=3)
        ax.annotate("BKP 2010\nrecommended", xy=(oj, oi), xytext=(oj + 0.7, oi - 0.7),
                    fontsize=7, color="red",
                    arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
    except ValueError:
        pass

    cb = fig.colorbar(im, ax=ax)
    cb.set_label("RMSE (km s$^{-1}$)")

    save_pdf(fig, out_path.stem, out_dir=out_path.parent)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TGV alpha-pair ablation.")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    print(banner(device))
    print(f"[tgv-ablation] grid: alpha_0={GRID_ALPHA_0}, alpha_1={GRID_ALPHA_1}")
    print(f"[tgv-ablation] {args.seeds} seeds, {args.iterations} iter, "
          f"{len(GRID_ALPHA_0)*len(GRID_ALPHA_1)*args.seeds} runs total")

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    truth, spec = _load_truth("gaussian_anomaly")
    seeds = [20260507 + 17 * i for i in range(args.seeds)]

    rows: list[dict] = []
    t_start = time.time()
    n_runs = len(GRID_ALPHA_0) * len(GRID_ALPHA_1) * args.seeds
    run_idx = 0
    for a0 in GRID_ALPHA_0:
        for a1 in GRID_ALPHA_1:
            for seed in seeds:
                run_idx += 1
                t0 = time.time()
                row = run_one(
                    truth, spec, seed, a0, a1, args.iterations, device,
                    LOG_ROOT / f"a0_{a0}_a1_{a1}" / f"seed_{seed}",
                )
                rows.append(row)
                elapsed = time.time() - t0
                eta = (n_runs - run_idx) * elapsed
                print(f"  [{run_idx:2d}/{n_runs}] a0={a0} a1={a1} seed={seed}  "
                      f"RMSE={row['best_rmse']:.4f}  SSIM={row['best_ssim']:.3f}  "
                      f"({elapsed:.0f}s, ETA {eta/60:.1f}min)")

    # Per-cell summary
    summary: list[dict] = []
    for a0 in GRID_ALPHA_0:
        for a1 in GRID_ALPHA_1:
            cell = [r for r in rows
                    if abs(r["alpha_0"] - a0) < 1e-12
                    and abs(r["alpha_1"] - a1) < 1e-12]
            rmses = np.asarray([c["best_rmse"] for c in cell])
            ssims = np.asarray([c["best_ssim"] for c in cell])
            summary.append({
                "alpha_0": a0,
                "alpha_1": a1,
                "n_seeds": len(cell),
                "mean_rmse": float(rmses.mean()),
                "std_rmse": float(rmses.std()),
                "mean_ssim": float(ssims.mean()),
                "std_ssim": float(ssims.std()),
            })

    # Persist CSVs
    per_run_csv = LOG_ROOT / "results.csv"
    with per_run_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary_csv = LOG_ROOT / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # Heatmap
    heatmap_path = plot_heatmap(summary, PDF_DIR / "42_ablation_tgv_heatmap.pdf")

    # Final report
    elapsed = time.time() - t_start
    print(f"\n[tgv-ablation] total wall-clock: {elapsed/60:.1f} min")
    print(f"[tgv-ablation] per-run CSV  -> {per_run_csv}")
    print(f"[tgv-ablation] summary CSV  -> {summary_csv}")
    print(f"[tgv-ablation] heatmap      -> {heatmap_path}")

    # Find the optimum
    best = min(summary, key=lambda r: r["mean_rmse"])
    bkp = next(r for r in summary if abs(r["alpha_0"] - 1.0) < 1e-12
                                    and abs(r["alpha_1"] - 2.0) < 1e-12)
    print()
    print(f"[tgv-ablation] grid-best: a0={best['alpha_0']}, a1={best['alpha_1']}, "
          f"RMSE={best['mean_rmse']:.4f} ± {best['std_rmse']:.4f}")
    print(f"[tgv-ablation] BKP 2010:  a0={bkp['alpha_0']}, a1={bkp['alpha_1']}, "
          f"RMSE={bkp['mean_rmse']:.4f} ± {bkp['std_rmse']:.4f}")
    delta = best['mean_rmse'] - bkp['mean_rmse']
    print(f"[tgv-ablation] grid-best is {delta:+.4f} km/s vs BKP recommendation "
          f"({100*delta/bkp['mean_rmse']:+.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
