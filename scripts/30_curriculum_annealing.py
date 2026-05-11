"""
30_curriculum_annealing.py
==========================

Train MIMIR's Neural Velocity Field with a *time-varying* TV regularization
weight (curriculum annealing), in an attempt to combine TV's strength on
sharp interfaces (curvefault, layered) with L2-style accuracy on smooth
fields (gaussian).

Motivation
----------
Script 22 used a fixed lambda_TV = 1.0 throughout training, which:

  * wins curvefault by +29% RMSE (p=0.0005)
  * wins layered    by +33% RMSE (p=0.0002)
  * loses gaussian  by -52% RMSE (p=0.0096)

We hypothesise that the gaussian loss is caused by TV's well-known staircase
bias on smooth fields (Rudin-Osher-Fatemi 1992). If we let lambda_TV decrease
during training — strong early to capture interfaces, weak late to refine
smooth regions — we may close the gaussian gap without sacrificing the wins.

Schedule options
----------------
  * `log_linear`        lambda = exp( log(start) + (log(end) - log(start)) * t/T )
                        Smooth, monotone, geometric decay. Default.
  * `step`              lambda = start  (t < T/2)
                        lambda = end    (t >= T/2)
                        Abrupt switch — may destabilise the optimum.
  * `step_then_log`     lambda = start                                         (t < 0.3 T)
                        lambda = log_linear(start, end) on remaining 70%       (t >= 0.3 T)
                        Hybrid — locks in interfaces, then refines.

What this script produces
-------------------------
For each benchmark, runs `--seeds` independent seeds with the chosen
schedule, then:

  logs/curriculum/<benchmark>/seed_<N>/history.npz  — incl. per-iter lambda
  models/curriculum/<benchmark>/best_overall.pt     — best seed only
  PDF/30_curriculum_velocity_<benchmark>.pdf        — truth vs curriculum
  PDF/30_curriculum_composite.pdf                   — 3 x 3 paper figure
  PDF/30_curriculum_lambda_schedule.pdf             — visualises the schedule
  PDF/30_curriculum_vs_paper_grade.pdf              — bar comparison
  logs/curriculum/results_per_seed.csv
  logs/curriculum/results_summary.csv
  logs/curriculum/comparison_with_paper_grade.csv   — if script 22 was run

Disk hygiene
------------
Same as script 22 — only the per-benchmark winner checkpoint is retained.

Usage
-----
    # Full default — log_linear schedule, 5 seeds, 8000 iter, all benchmarks
    python scripts/30_curriculum_annealing.py

    # Test the step schedule
    python scripts/30_curriculum_annealing.py --schedule step

    # Single benchmark + custom range
    python scripts/30_curriculum_annealing.py \
        --benchmark gaussian_anomaly \
        --schedule step_then_log \
        --lambda-start 1.0 --lambda-end 1e-3

    # Smoke test (2 seeds × 1000 iter, ~1 min on a modern laptop)
    python scripts/30_curriculum_annealing.py --quick
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Callable

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
LOG_ROOT = REPO_ROOT / "logs" / "curriculum"
MODEL_ROOT = REPO_ROOT / "models" / "curriculum"
PAPER_GRADE_CSV = REPO_ROOT / "logs" / "paper_grade" / "results_per_seed.csv"


# Default fixed-config (everything except smoothness_weight, which is annealed)
BASE_CFG = {
    "fourier_scale": 4.0,
    "learning_rate": 5e-3,
    "iterations": 8000,
}


# ---------------------------------------------------------------------------
# schedule factory
# ---------------------------------------------------------------------------


ScheduleFn = Callable[[int, int], float]


def make_schedule(kind: str, start: float, end: float, warmup_frac: float = 0.3) -> ScheduleFn:
    """
    Build a callable schedule(iter, total) -> lambda for the chosen kind.

    Parameters
    ----------
    kind : {"log_linear", "step", "step_then_log"}
    start, end : positive floats — lambda range
    warmup_frac : only used by `step_then_log`; fraction of total iters held
                  at `start` before annealing kicks in.
    """
    if start <= 0 or end <= 0:
        raise ValueError("lambda start and end must be positive (log-space).")

    log_start = math.log(start)
    log_end = math.log(end)

    if kind == "log_linear":
        def schedule(it: int, total: int) -> float:
            if total <= 1:
                return start
            t = it / (total - 1)
            return math.exp(log_start + (log_end - log_start) * t)
        return schedule

    if kind == "step":
        def schedule(it: int, total: int) -> float:
            return start if it < total // 2 else end
        return schedule

    if kind == "step_then_log":
        def schedule(it: int, total: int) -> float:
            warmup = int(total * warmup_frac)
            if it < warmup:
                return start
            denom = max(1, total - warmup - 1)
            t = (it - warmup) / denom
            t = min(max(t, 0.0), 1.0)
            return math.exp(log_start + (log_end - log_start) * t)
        return schedule

    raise ValueError(f"Unknown schedule kind: {kind}")


# ---------------------------------------------------------------------------
# curriculum trainer (subclass; does not touch trainer.py)
# ---------------------------------------------------------------------------


class CurriculumNVFTrainer(NVFTrainer):
    """
    NVFTrainer that mutates ``self.cfg.smoothness_weight`` immediately before
    each training step, according to a schedule callable. The lambda values
    actually used per step are captured in ``self.lambda_history`` for
    later visualisation.

    Implementation note: We override `_train_step` (called once per outer
    iteration) and update lambda there. The base trainer's `_train_step`
    reads ``self.cfg.smoothness_weight`` when it computes the loss, so by
    setting it here we control the regularizer strength for every iteration.
    """

    def __init__(
        self,
        *args,
        smoothness_schedule: ScheduleFn,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._smoothness_schedule = smoothness_schedule
        self._curriculum_iter = 0
        self.lambda_history: list[float] = []

    def _train_step(self) -> dict:
        cur_lambda = self._smoothness_schedule(
            self._curriculum_iter, self.cfg.iterations
        )
        self.cfg.smoothness_weight = float(cur_lambda)
        self.lambda_history.append(float(cur_lambda))
        out = super()._train_step()
        self._curriculum_iter += 1
        return out


# ---------------------------------------------------------------------------
# data loading
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


# ---------------------------------------------------------------------------
# per-seed training run
# ---------------------------------------------------------------------------


def run_one_seed(
    truth: np.ndarray,
    spec: BenchmarkSpec,
    seed: int,
    schedule_fn: ScheduleFn,
    schedule_label: str,
    base_cfg: dict,
    device: torch.device,
    out_dir: Path,
) -> dict:
    """Run one seed with the curriculum schedule. Returns a flat dict + tensors."""
    set_global_seed(seed)

    # Acquisition + observations — identical to script 22, so observations
    # match seed-by-seed and metrics are directly comparable.
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
        fourier_scale=float(base_cfg["fourier_scale"]),
        hidden=128,
        depth=4,
        activation="tanh",
    )
    field = NeuralVelocityField(nvf_cfg).to(device)

    src = torch.from_numpy(src_np).to(device)
    rec = torch.from_numpy(rec_np).to(device)
    obs = torch.from_numpy(tt_np).to(device)

    # Trainer — note `smoothness_weight` is a placeholder; the schedule overrides it.
    out_dir.mkdir(parents=True, exist_ok=True)
    trn_cfg = NVFTrainingConfig(
        iterations=int(base_cfg["iterations"]),
        learning_rate=float(base_cfg["learning_rate"]),
        lr_schedule="cosine",
        warmup_iters=200,
        grad_clip=1.0,
        data_loss="huber",
        huber_delta=0.05,
        smoothness_weight=1.0,         # overridden every step by schedule
        smoothness_grid=64,
        ray_samples=64,
        val_every=100,
        log_every=200,
        save_best=True,
        save_last=False,
        out_dir=str(out_dir),
    )
    trainer = CurriculumNVFTrainer(
        field=field,
        sources=src,
        receivers=rec,
        observed_tt=obs,
        truth_grid=truth,
        cfg=trn_cfg,
        smoothness_schedule=schedule_fn,
    )

    t0 = time.time()
    state = trainer.fit()
    elapsed = time.time() - t0

    # Reload best to compute predicted velocity grid for figures
    best_path = out_dir / "best.pt"
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    field.load_state_dict(ckpt["model_state_dict"])
    field.eval()
    with torch.no_grad():
        v_pred = field.velocity_grid(spec.nx, spec.nz).cpu().numpy()

    # Persist history (incl. lambda schedule)
    hist_path = out_dir / "history.npz"
    np.savez(
        hist_path,
        **{k: np.asarray(v) for k, v in state.history.items()},
        best_val_rmse=state.best_val_rmse,
        lambda_history=np.asarray(trainer.lambda_history),
        schedule_label=schedule_label,
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
        "lambda_at_best": float(np.asarray(trainer.lambda_history)[val_iter[best_idx]]),
        "best_ckpt": str(best_path),
        "v_pred": v_pred,
        "src_np": src_np,
        "rec_np": rec_np,
        "lambda_history": np.asarray(trainer.lambda_history),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def make_composite(
    bench_results: dict[str, dict],
    truths: dict[str, np.ndarray],
    specs: dict[str, BenchmarkSpec],
    schedule_label: str,
) -> plt.Figure:
    """3 x 3: rows = benchmarks, cols = (truth, curriculum-MIMIR, error)."""
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
            axes[row, 1].set_title(f"MIMIR — curriculum ({schedule_label})")
            axes[row, 2].set_title("Error")

        cb0 = fig.colorbar(im0, ax=axes[row, :2], location="right",
                           shrink=0.85, pad=0.02)
        cb0.set_label("v (km/s)")
        cb2 = fig.colorbar(im2, ax=axes[row, 2], location="right",
                           shrink=0.85, pad=0.02)
        cb2.set_label("Δv (km/s)")

    return fig


def plot_lambda_schedule(
    schedule_fn: ScheduleFn,
    total_iter: int,
    label: str,
    realised_history: np.ndarray | None = None,
) -> plt.Figure:
    """Visualise the schedule lambda(iter)."""
    apply_paper_style()
    iters = np.arange(total_iter)
    lambdas = np.array([schedule_fn(int(i), total_iter) for i in iters])

    fig, ax = plt.subplots(figsize=(5.5, 3.0), constrained_layout=True)
    ax.plot(iters, lambdas, color="tab:blue", linewidth=1.5,
            label=f"schedule: {label}")
    if realised_history is not None and len(realised_history) > 0:
        ax.plot(np.arange(len(realised_history)), realised_history,
                color="tab:orange", linewidth=0.8, alpha=0.6,
                linestyle="--", label="realised (one seed)")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\lambda_{\mathrm{TV}}$")
    ax.set_title(f"TV regularization curriculum schedule — {label}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", frameon=False)
    return fig


def plot_curriculum_vs_paper_grade(
    cur_summary: list[dict],
    paper_summary: list[dict],
    schedule_label: str,
) -> plt.Figure:
    """
    Two-panel grouped bar (RMSE + SSIM) comparing curriculum vs paper-grade
    (fixed lambda) MIMIR runs, side-by-side per benchmark.
    """
    apply_paper_style()
    benches = sorted(set(r["benchmark"] for r in cur_summary)
                     & set(r["benchmark"] for r in paper_summary))
    n = len(benches)
    if n == 0:
        return plt.figure()

    x = np.arange(n)
    width = 0.36

    # Index by benchmark
    cur_by = {r["benchmark"]: r for r in cur_summary}
    pap_by = {r["benchmark"]: r for r in paper_summary}

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), constrained_layout=True)

    # RMSE
    ax = axes[0]
    pap_means = [pap_by[b]["mean_rmse"] for b in benches]
    pap_stds = [pap_by[b]["std_rmse"] for b in benches]
    cur_means = [cur_by[b]["mean_rmse"] for b in benches]
    cur_stds = [cur_by[b]["std_rmse"] for b in benches]
    ax.bar(x - width / 2, pap_means, yerr=pap_stds, width=width, color="tab:blue",
           label=r"Fixed $\lambda$ (script 22)", capsize=3)
    ax.bar(x + width / 2, cur_means, yerr=cur_stds, width=width, color="tab:green",
           label=f"Curriculum ({schedule_label})", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("_", "\n") for b in benches])
    ax.set_ylabel("Validation RMSE (km/s)")
    ax.set_title("RMSE (lower is better)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, axis="y", alpha=0.3)

    # SSIM
    ax = axes[1]
    pap_means = [pap_by[b]["mean_ssim"] for b in benches]
    pap_stds = [pap_by[b]["std_ssim"] for b in benches]
    cur_means = [cur_by[b]["mean_ssim"] for b in benches]
    cur_stds = [cur_by[b]["std_ssim"] for b in benches]
    ax.bar(x - width / 2, pap_means, yerr=pap_stds, width=width, color="tab:blue",
           label=r"Fixed $\lambda$ (script 22)", capsize=3)
    ax.bar(x + width / 2, cur_means, yerr=cur_stds, width=width, color="tab:green",
           label=f"Curriculum ({schedule_label})", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("_", "\n") for b in benches])
    ax.set_ylabel("Validation SSIM")
    ax.set_title("SSIM (higher is better)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, axis="y", alpha=0.3)

    return fig


# ---------------------------------------------------------------------------
# helpers for paper-grade comparison
# ---------------------------------------------------------------------------


def _read_paper_grade_csv() -> list[dict] | None:
    """Read script 22's per-seed CSV if present, else return None."""
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


def _write_paired_comparison_csv(
    cur_summary: list[dict],
    paper_summary: list[dict],
    schedule_label: str,
    out_path: Path,
) -> list[dict]:
    """Per-benchmark improvement table: curriculum vs paper-grade."""
    pap_by = {r["benchmark"]: r for r in paper_summary}
    rows: list[dict] = []
    for r in cur_summary:
        b = r["benchmark"]
        if b not in pap_by:
            continue
        p = pap_by[b]
        rmse_pct = (p["mean_rmse"] - r["mean_rmse"]) / max(p["mean_rmse"], 1e-9) * 100.0
        ssim_pct = (r["mean_ssim"] - p["mean_ssim"]) / max(p["mean_ssim"], 1e-9) * 100.0
        rows.append({
            "benchmark": b,
            "schedule": schedule_label,
            "fixed_rmse_mean": round(p["mean_rmse"], 4),
            "fixed_rmse_std": round(p["std_rmse"], 4),
            "curriculum_rmse_mean": round(r["mean_rmse"], 4),
            "curriculum_rmse_std": round(r["std_rmse"], 4),
            "rmse_improvement_pct": round(rmse_pct, 2),
            "fixed_ssim_mean": round(p["mean_ssim"], 3),
            "fixed_ssim_std": round(p["std_ssim"], 3),
            "curriculum_ssim_mean": round(r["mean_ssim"], 3),
            "curriculum_ssim_std": round(r["std_ssim"], 3),
            "ssim_improvement_pct": round(ssim_pct, 2),
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MIMIR curriculum annealing of TV weight.")
    parser.add_argument(
        "--schedule", default="log_linear",
        choices=["log_linear", "step", "step_then_log"],
        help="Form of the lambda(iter) curriculum.",
    )
    parser.add_argument("--lambda-start", type=float, default=1.0,
                        help="Initial TV weight.")
    parser.add_argument("--lambda-end", type=float, default=1e-2,
                        help="Final TV weight.")
    parser.add_argument("--warmup-frac", type=float, default=0.3,
                        help="(step_then_log only) fraction of iters held at lambda_start.")
    parser.add_argument("--benchmark", default=None,
                        choices=["gaussian_anomaly", "layered", "curvefault_lookalike"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--fourier-scale", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: 2 seeds × 1000 iter.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    # Resolve config
    base_cfg = dict(BASE_CFG)
    if args.iterations is not None:
        base_cfg["iterations"] = args.iterations
    if args.fourier_scale is not None:
        base_cfg["fourier_scale"] = args.fourier_scale
    if args.learning_rate is not None:
        base_cfg["learning_rate"] = args.learning_rate
    n_seeds = args.seeds
    if args.quick:
        n_seeds = 2
        base_cfg["iterations"] = 1000

    # Schedule
    schedule_fn = make_schedule(
        args.schedule, args.lambda_start, args.lambda_end, args.warmup_frac
    )
    schedule_label = (
        f"{args.schedule}: {args.lambda_start:g}→{args.lambda_end:g}"
        + (f" warmup={args.warmup_frac:.2f}" if args.schedule == "step_then_log" else "")
    )

    # Benchmarks
    if args.benchmark is None:
        benchmarks = ["gaussian_anomaly", "layered", "curvefault_lookalike"]
    else:
        benchmarks = [args.benchmark]

    device = resolve_device(args.device)
    print(banner(device))
    print(f"[curriculum] schedule: {schedule_label}")
    print(f"[curriculum] base config: {base_cfg}")
    print(f"[curriculum] seeds: {n_seeds}, benchmarks: {benchmarks}")

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    seeds = [20260507 + 17 * i for i in range(n_seeds)]
    all_rows: list[dict] = []
    bench_results: dict[str, dict] = {}
    truths: dict[str, np.ndarray] = {}
    specs: dict[str, BenchmarkSpec] = {}

    t_start = time.time()
    realised_lambda_for_plot: np.ndarray | None = None

    for bname in benchmarks:
        print(f"\n[curriculum] === benchmark: {bname} ===")
        truth, spec = _load_truth(bname)
        truths[bname] = truth
        specs[bname] = spec

        seed_runs: list[dict] = []
        for s_idx, seed in enumerate(seeds):
            print(f"\n[curriculum]   seed {s_idx + 1}/{n_seeds} = {seed}")
            seed_dir = LOG_ROOT / bname / f"seed_{seed}"
            row = run_one_seed(
                truth, spec, seed, schedule_fn, schedule_label,
                base_cfg, device, seed_dir,
            )
            seed_runs.append(row)
            all_rows.append({
                "benchmark": bname,
                "seed": row["seed"],
                "best_rmse": row["best_rmse"],
                "best_ssim": row["best_ssim"],
                "best_pearson": row["best_pearson"],
                "best_iter": row["best_iter"],
                "lambda_at_best": row["lambda_at_best"],
                "final_rmse": row["final_rmse"],
                "elapsed_s": row["elapsed_s"],
                "schedule": schedule_label,
            })
            # Capture one realised lambda history for the schedule plot
            if realised_lambda_for_plot is None:
                realised_lambda_for_plot = row["lambda_history"]

        # Pick winner seed
        best_idx = int(np.argmin([r["best_rmse"] for r in seed_runs]))
        winner = seed_runs[best_idx]
        bench_results[bname] = winner

        # Persist winner checkpoint, drop the others
        winner_dst = MODEL_ROOT / bname / "best_overall.pt"
        winner_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(winner["best_ckpt"], winner_dst)
        for r in seed_runs:
            ckpt = Path(r["best_ckpt"])
            if ckpt.exists() and ckpt != winner_dst:
                ckpt.unlink()

        # Per-benchmark velocity-pair PDF
        fig = plot_velocity_pair(
            truth, winner["v_pred"], spec,
            title_truth="Ground truth",
            title_estimate=f"MIMIR curriculum ({args.schedule}, best of {n_seeds})",
            rays=None,
        )
        pair_pdf = save_pdf(fig, f"30_curriculum_velocity_{bname}", out_dir=PDF_DIR)
        plt.close(fig)
        print(f"[curriculum]   winner: seed={winner['seed']} "
              f"RMSE={winner['best_rmse']:.4f} SSIM={winner['best_ssim']:.3f} "
              f"(at λ={winner['lambda_at_best']:.4f})")
        print(f"[curriculum]   figure -> {pair_pdf}")

    # ---- per-seed CSV ----
    per_seed_csv = LOG_ROOT / "results_per_seed.csv"
    with per_seed_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n[curriculum] per-seed CSV  -> {per_seed_csv}")

    # ---- summary CSV ----
    cur_summary = _per_benchmark_summary(all_rows)
    summary_csv = LOG_ROOT / "results_summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cur_summary[0].keys()))
        w.writeheader()
        w.writerows(cur_summary)
    print(f"[curriculum] summary CSV   -> {summary_csv}")

    # ---- schedule visualization ----
    fig = plot_lambda_schedule(
        schedule_fn, base_cfg["iterations"], schedule_label,
        realised_history=realised_lambda_for_plot,
    )
    sched_pdf = save_pdf(fig, "30_curriculum_lambda_schedule", out_dir=PDF_DIR)
    plt.close(fig)
    print(f"[curriculum] schedule plot -> {sched_pdf}")

    # ---- composite ----
    fig = make_composite(bench_results, truths, specs, args.schedule)
    comp_pdf = save_pdf(fig, "30_curriculum_composite", out_dir=PDF_DIR)
    plt.close(fig)
    print(f"[curriculum] composite     -> {comp_pdf}")

    # ---- comparison vs paper-grade (if available) ----
    paper_rows = _read_paper_grade_csv()
    if paper_rows is not None:
        paper_summary = _per_benchmark_summary(paper_rows)
        comp_csv = LOG_ROOT / "comparison_with_paper_grade.csv"
        cmp_rows = _write_paired_comparison_csv(
            cur_summary, paper_summary, schedule_label, comp_csv,
        )
        print(f"[curriculum] vs-paper CSV  -> {comp_csv}")

        fig = plot_curriculum_vs_paper_grade(
            cur_summary, paper_summary, args.schedule,
        )
        cmp_pdf = save_pdf(fig, "30_curriculum_vs_paper_grade", out_dir=PDF_DIR)
        plt.close(fig)
        print(f"[curriculum] vs-paper PDF  -> {cmp_pdf}")
    else:
        print("[curriculum] (paper-grade results not found at "
              f"{PAPER_GRADE_CSV}; skipping comparison)")
        cmp_rows = []

    # ---- final report ----
    elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"[curriculum] total wall-clock: {elapsed / 60:.1f} min")
    print("[curriculum] CURRICULUM TABLE")
    print("-" * 80)
    print(f"{'benchmark':<24}{'RMSE (km/s)':>16}{'SSIM':>14}{'Pearson':>14}")
    print("-" * 80)
    for r in cur_summary:
        print(f"{r['benchmark']:<24}"
              f"{r['mean_rmse']:>10.4f} ± {r['std_rmse']:.4f}"
              f"{r['mean_ssim']:>9.3f} ± {r['std_ssim']:.3f}"
              f"{r['mean_pearson']:>9.3f} ± {r['std_pearson']:.3f}")
    print("=" * 80)

    if cmp_rows:
        print("\n[curriculum] CURRICULUM vs FIXED-λ (script 22)")
        print("-" * 80)
        print(f"{'benchmark':<24}{'fixed RMSE':>16}{'curr. RMSE':>16}{'Δ RMSE %':>14}")
        print("-" * 80)
        for r in cmp_rows:
            sign = "+" if r["rmse_improvement_pct"] >= 0 else ""
            print(f"{r['benchmark']:<24}"
                  f"{r['fixed_rmse_mean']:>10.4f} ± {r['fixed_rmse_std']:.4f}"
                  f"{r['curriculum_rmse_mean']:>10.4f} ± {r['curriculum_rmse_std']:.4f}"
                  f"{sign}{r['rmse_improvement_pct']:>10.1f}%")
        print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
