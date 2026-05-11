"""
40_train_tgv.py
===============

Paper-grade evaluation of MIMIR with **Total Generalized Variation (TGV²)**
regularization in place of Total Variation (TV).

Why this script exists
----------------------
Script 22 used TV regularization (lambda=1.0) and lost the Gaussian
benchmark to the classical FMM-LSMR baseline by 52% RMSE because TV is
biased toward piecewise-constant solutions (the staircase artifact),
which clashes with the smooth Gaussian-like targets that L2 smoothing
naturally favours.

Curriculum annealing of the TV weight (script 30) produced only a small
(5.4%, p=0.019) improvement on Gaussian — confirming that TV's bias is
intrinsic and not a scheduling artifact.

This script tests the principled remedy: replace TV with TGV² (Bredies,
Kunisch & Pock 2010), which prefers piecewise-*affine* (smoothly varying
within layers, sharp jumps at boundaries) solutions. The hypothesis is
that TGV² will close the Gaussian gap *without* sacrificing the curvefault
and layered wins.

Configuration
-------------
We follow the same protocol as script 22:
  fourier_scale       = 4.0
  learning_rate       = 5e-3
  iterations          = 8000
  seeds               = 5
  benchmarks          = {gaussian_anomaly, layered, curvefault_lookalike}

TGV-specific hyperparameters (default from BKP 2010 imaging recommendations):
  tgv_alpha_0         = 1.0   (weight on |ε(w)|, Hessian-like)
  tgv_alpha_1         = 2.0   (weight on |∇v − w|, TV-like)
  smoothness_weight   = 1.0   (overall TGV² scale, matched to script 22's TV weight)

Outputs match script 22's layout:

    logs/tgv/<benchmark>/seed_<N>/history.npz
    models/tgv/<benchmark>/best_overall.pt
    PDF/40_tgv_velocity_<benchmark>.pdf
    PDF/40_tgv_composite.pdf
    logs/tgv/results_per_seed.csv
    logs/tgv/results_summary.csv
    logs/tgv/comparison_with_paper_grade.csv      — automatic if script 22 was run

Usage
-----
    # Full TGV paper-grade run (~12-15 min on a modern laptop)
    python scripts/40_train_tgv.py

    # Quick smoke (gaussian only, 2 seeds × 1000 iter)
    python scripts/40_train_tgv.py --quick

    # Override TGV weights
    python scripts/40_train_tgv.py --tgv-alpha-0 0.5 --tgv-alpha-1 1.0
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
from mimir.losses import AuxFieldConfig, AuxiliaryVectorField
from mimir.training import TGVTrainer, TGVTrainingConfig
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
LOG_ROOT = REPO_ROOT / "logs" / "tgv"
MODEL_ROOT = REPO_ROOT / "models" / "tgv"
PAPER_GRADE_CSV = REPO_ROOT / "logs" / "paper_grade" / "results_per_seed.csv"


BASE_CFG = {
    "fourier_scale": 4.0,
    "smoothness_weight": 1.0,    # overall TGV² scale
    "learning_rate": 5e-3,
    "iterations": 8000,
    "tgv_alpha_0": 1.0,
    "tgv_alpha_1": 2.0,
}


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


def run_one_seed(
    truth: np.ndarray,
    spec: BenchmarkSpec,
    seed: int,
    cfg: dict,
    device: torch.device,
    out_dir: Path,
) -> dict:
    """Run one seed of TGV training; return metrics + reloaded best velocity."""
    set_global_seed(seed)

    # Same observation protocol as script 22 — direct comparability per seed
    geometry = cross_well_layout(spec, n_sources_per_side=12, n_receivers_per_side=12)
    rng = np.random.default_rng(seed + 1)
    src_np, rec_np, tt_np = sample_observed_travel_times(
        truth, spec, geometry, noise_pct=5.0, rng=rng,
    )

    # Velocity field (identical config to script 22)
    nvf_cfg = NVFConfig(
        domain_x=spec.domain_x,
        domain_z=spec.domain_z,
        base_velocity=3.0,
        velocity_range=(2.0, 5.5),
        fourier_dim=64,
        fourier_scale=float(cfg["fourier_scale"]),
        hidden=128,
        depth=4,
        activation="tanh",
    )
    field = NeuralVelocityField(nvf_cfg).to(device)

    # Auxiliary vector field for TGV
    aux_cfg = AuxFieldConfig(
        domain_x=spec.domain_x,
        domain_z=spec.domain_z,
        fourier_dim=32,
        fourier_scale=2.0,
        hidden=64,
        depth=3,
        activation="tanh",
    )
    aux_field = AuxiliaryVectorField(aux_cfg).to(device)

    src = torch.from_numpy(src_np).to(device)
    rec = torch.from_numpy(rec_np).to(device)
    obs = torch.from_numpy(tt_np).to(device)

    out_dir.mkdir(parents=True, exist_ok=True)
    trn_cfg = TGVTrainingConfig(
        iterations=int(cfg["iterations"]),
        learning_rate=float(cfg["learning_rate"]),
        lr_schedule="cosine",
        warmup_iters=200,
        grad_clip=1.0,
        data_loss="huber",
        huber_delta=0.05,
        smoothness_weight=float(cfg["smoothness_weight"]),
        smoothness_grid=64,
        ray_samples=64,
        val_every=100,
        log_every=200,
        save_best=True,
        save_last=False,
        out_dir=str(out_dir),
        tgv_alpha_0=float(cfg["tgv_alpha_0"]),
        tgv_alpha_1=float(cfg["tgv_alpha_1"]),
    )
    trainer = TGVTrainer(
        field=field, aux_field=aux_field,
        sources=src, receivers=rec, observed_tt=obs,
        truth_grid=truth, cfg=trn_cfg,
    )

    t0 = time.time()
    state = trainer.fit()
    elapsed = time.time() - t0

    # Reload best for figure rendering
    best_path = out_dir / "best.pt"
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    field.load_state_dict(ckpt["model_state_dict"])
    field.eval()
    with torch.no_grad():
        v_pred = field.velocity_grid(spec.nx, spec.nz).cpu().numpy()

    hist_path = out_dir / "history.npz"
    np.savez(
        hist_path,
        **{k: np.asarray(v) for k, v in state.history.items()},
        best_val_rmse=state.best_val_rmse,
    )

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
        "v_pred": v_pred,
    }


def make_composite(
    bench_results: dict[str, dict],
    truths: dict[str, np.ndarray],
    specs: dict[str, BenchmarkSpec],
) -> plt.Figure:
    """3 x 3: rows = benchmarks, cols = (truth, TGV, error)."""
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
        v_pred = bench_results[name]["v_pred"]
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
            axes[row, 1].set_title("MIMIR + TGV²")
            axes[row, 2].set_title("Error")

        cb0 = fig.colorbar(im0, ax=axes[row, :2], location="right",
                           shrink=0.85, pad=0.02)
        cb0.set_label("v (km/s)")
        cb2 = fig.colorbar(im2, ax=axes[row, 2], location="right",
                           shrink=0.85, pad=0.02)
        cb2.set_label("Δv (km/s)")
    return fig


def _read_paper_grade_csv() -> list[dict] | None:
    if not PAPER_GRADE_CSV.exists():
        return None
    rows = []
    with PAPER_GRADE_CSV.open("r", newline="") as f:
        for r in csv.DictReader(f):
            for k in ("best_rmse", "best_ssim", "best_pearson", "final_rmse", "elapsed_s"):
                if k in r and r[k] != "":
                    r[k] = float(r[k])
            for k in ("seed", "best_iter"):
                if k in r and r[k] != "":
                    r[k] = int(r[k])
            rows.append(r)
    return rows


def _per_benchmark_summary(rows: list[dict]) -> list[dict]:
    benches = sorted({r["benchmark"] for r in rows})
    out = []
    for b in benches:
        cell = [r for r in rows if r["benchmark"] == b]
        rmses = np.asarray([c["best_rmse"] for c in cell])
        ssims = np.asarray([c["best_ssim"] for c in cell])
        pears = np.asarray([c["best_pearson"] for c in cell])
        out.append({
            "benchmark": b,
            "n_seeds": len(cell),
            "mean_rmse": float(rmses.mean()),
            "std_rmse": float(rmses.std()),
            "mean_ssim": float(ssims.mean()),
            "std_ssim": float(ssims.std()),
            "mean_pearson": float(pears.mean()),
            "std_pearson": float(pears.std()),
        })
    return out


def _write_comparison_csv(
    tgv_summary: list[dict], paper_summary: list[dict], out_path: Path,
) -> list[dict]:
    pap_by = {r["benchmark"]: r for r in paper_summary}
    rows: list[dict] = []
    for r in tgv_summary:
        b = r["benchmark"]
        if b not in pap_by:
            continue
        p = pap_by[b]
        rmse_pct = (p["mean_rmse"] - r["mean_rmse"]) / max(p["mean_rmse"], 1e-9) * 100.0
        ssim_pct = (r["mean_ssim"] - p["mean_ssim"]) / max(p["mean_ssim"], 1e-9) * 100.0
        rows.append({
            "benchmark": b,
            "tv_rmse_mean": round(p["mean_rmse"], 4),
            "tv_rmse_std": round(p["std_rmse"], 4),
            "tgv_rmse_mean": round(r["mean_rmse"], 4),
            "tgv_rmse_std": round(r["std_rmse"], 4),
            "rmse_improvement_pct": round(rmse_pct, 2),
            "tv_ssim_mean": round(p["mean_ssim"], 3),
            "tgv_ssim_mean": round(r["mean_ssim"], 3),
            "ssim_improvement_pct": round(ssim_pct, 2),
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MIMIR with TGV² regularization.")
    parser.add_argument("--benchmark", default=None,
                        choices=["gaussian_anomaly", "layered", "curvefault_lookalike"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--smoothness-weight", type=float, default=None)
    parser.add_argument("--tgv-alpha-0", type=float, default=None,
                        help="Weight on |ε(w)| (Hessian-like). Default 1.0.")
    parser.add_argument("--tgv-alpha-1", type=float, default=None,
                        help="Weight on |∇v − w| (TV-like). Default 2.0.")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: 2 seeds × 1000 iter, gaussian only.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    cfg = dict(BASE_CFG)
    if args.iterations is not None:
        cfg["iterations"] = args.iterations
    if args.smoothness_weight is not None:
        cfg["smoothness_weight"] = args.smoothness_weight
    if args.tgv_alpha_0 is not None:
        cfg["tgv_alpha_0"] = args.tgv_alpha_0
    if args.tgv_alpha_1 is not None:
        cfg["tgv_alpha_1"] = args.tgv_alpha_1
    n_seeds = args.seeds
    if args.quick:
        n_seeds = 2
        cfg["iterations"] = 1000
        if args.benchmark is None:
            args.benchmark = "gaussian_anomaly"

    benchmarks = (
        [args.benchmark] if args.benchmark
        else ["gaussian_anomaly", "layered", "curvefault_lookalike"]
    )

    device = resolve_device(args.device)
    print(banner(device))
    print(f"[tgv] config: {cfg}")
    print(f"[tgv] seeds: {n_seeds}, benchmarks: {benchmarks}")

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    seeds = [20260507 + 17 * i for i in range(n_seeds)]
    all_rows: list[dict] = []
    bench_results: dict[str, dict] = {}
    truths: dict[str, np.ndarray] = {}
    specs: dict[str, BenchmarkSpec] = {}

    t_start = time.time()

    for bname in benchmarks:
        print(f"\n[tgv] === benchmark: {bname} ===")
        truth, spec = _load_truth(bname)
        truths[bname] = truth
        specs[bname] = spec

        seed_runs: list[dict] = []
        for s_idx, seed in enumerate(seeds):
            print(f"\n[tgv]   seed {s_idx + 1}/{n_seeds} = {seed}")
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

        # Pick winner
        best_idx = int(np.argmin([r["best_rmse"] for r in seed_runs]))
        winner = seed_runs[best_idx]
        bench_results[bname] = winner

        winner_dst = MODEL_ROOT / bname / "best_overall.pt"
        winner_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(winner["best_ckpt"], winner_dst)
        for r in seed_runs:
            ckpt = Path(r["best_ckpt"])
            if ckpt.exists() and ckpt != winner_dst:
                ckpt.unlink()

        fig = plot_velocity_pair(
            truth, winner["v_pred"], spec,
            title_truth="Ground truth",
            title_estimate=f"MIMIR + TGV² (best of {n_seeds} seeds)",
            rays=None,
        )
        pair_pdf = save_pdf(fig, f"40_tgv_velocity_{bname}", out_dir=PDF_DIR)
        plt.close(fig)
        print(f"[tgv]   winner: seed={winner['seed']} "
              f"RMSE={winner['best_rmse']:.4f} SSIM={winner['best_ssim']:.3f}")
        print(f"[tgv]   figure -> {pair_pdf}")

    # CSVs
    per_seed_csv = LOG_ROOT / "results_per_seed.csv"
    with per_seed_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n[tgv] per-seed CSV  -> {per_seed_csv}")

    tgv_summary = _per_benchmark_summary(all_rows)
    summary_csv = LOG_ROOT / "results_summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tgv_summary[0].keys()))
        w.writeheader()
        w.writerows(tgv_summary)
    print(f"[tgv] summary CSV   -> {summary_csv}")

    # Composite
    fig = make_composite(bench_results, truths, specs)
    comp_pdf = save_pdf(fig, "40_tgv_composite", out_dir=PDF_DIR)
    plt.close(fig)
    print(f"[tgv] composite     -> {comp_pdf}")

    # Comparison vs paper-grade (TV)
    paper_rows = _read_paper_grade_csv()
    cmp_rows: list[dict] = []
    if paper_rows is not None:
        paper_summary = _per_benchmark_summary(paper_rows)
        comp_csv = LOG_ROOT / "comparison_with_paper_grade.csv"
        cmp_rows = _write_comparison_csv(tgv_summary, paper_summary, comp_csv)
        print(f"[tgv] vs-paper CSV  -> {comp_csv}")

    elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"[tgv] total wall-clock: {elapsed / 60:.1f} min")
    print("[tgv] TGV² TABLE")
    print("-" * 80)
    print(f"{'benchmark':<24}{'RMSE (km/s)':>16}{'SSIM':>14}{'Pearson':>14}")
    print("-" * 80)
    for r in tgv_summary:
        print(f"{r['benchmark']:<24}"
              f"{r['mean_rmse']:>10.4f} ± {r['std_rmse']:.4f}"
              f"{r['mean_ssim']:>9.3f} ± {r['std_ssim']:.3f}"
              f"{r['mean_pearson']:>9.3f} ± {r['std_pearson']:.3f}")
    print("=" * 80)

    if cmp_rows:
        print("\n[tgv] TGV² vs FIXED-TV (script 22)")
        print("-" * 80)
        print(f"{'benchmark':<24}{'TV RMSE':>16}{'TGV RMSE':>16}{'Δ RMSE %':>14}")
        print("-" * 80)
        for r in cmp_rows:
            sign = "+" if r["rmse_improvement_pct"] >= 0 else ""
            print(f"{r['benchmark']:<24}"
                  f"{r['tv_rmse_mean']:>10.4f} ± {r['tv_rmse_std']:.4f}"
                  f"{r['tgv_rmse_mean']:>10.4f} ± {r['tgv_rmse_std']:.4f}"
                  f"{sign}{r['rmse_improvement_pct']:>10.1f}%")
        print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
