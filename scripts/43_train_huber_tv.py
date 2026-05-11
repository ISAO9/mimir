"""
43_train_huber_tv.py
=====================

Train MIMIR with Huber-TV regularization, the natural smoothed
intermediate between plain TV and TGV^2. This addresses the reviewer
question: "Is the TGV^2 advantage really about piecewise-affine priors,
or does any smoothed L1 give a similar improvement?"

Huber-TV definition
-------------------
Plain TV penalizes the L1 norm of the gradient:
    TV(v) = E[ |∇v|_iso ]

Huber-TV smooths the L1 at zero:
    HTV(v) = E[ ρ_δ(|∇v|_iso) ]
where
    ρ_δ(t) = (1/2) t^2 / δ      if |t| ≤ δ
           = |t| − δ/2          if |t| > δ

The threshold δ ≈ 0.05 makes the regularizer quadratic for small gradients
(removing the staircase artifact in smooth regions, like TGV^2) but
linear for large gradients (preserving edges, like TV).

Hypothesis (per BKP 2010 theory): Huber-TV will improve over plain TV on
the Gaussian benchmark (same intuition as TGV^2 — smooth gradients no
longer staircased) but will NOT match TGV^2 because Huber-TV cannot
distinguish "smooth gradient" from "constant region", whereas TGV^2's
auxiliary field can.

Configuration
-------------
Identical protocol to scripts 22 and 40 (5 seeds × 8000 iter × 3
benchmarks), Huber-TV replacing TV/TGV in the loss.

Outputs
-------
    logs/huber_tv/<benchmark>/seed_<N>/history.npz
    models/huber_tv/<benchmark>/best_overall.pt
    PDF/43_huber_tv_velocity_<benchmark>.pdf
    PDF/43_huber_tv_composite.pdf
    logs/huber_tv/results_per_seed.csv
    logs/huber_tv/results_summary.csv
    logs/huber_tv/comparison_with_tgv.csv
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
from mimir.losses import huber_total_variation_2d
from mimir.losses.physics_losses import travel_time_data_loss
from mimir.physics.ray_tracing import batched_straight_ray_travel_time
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
LOG_ROOT = REPO_ROOT / "logs" / "huber_tv"
MODEL_ROOT = REPO_ROOT / "models" / "huber_tv"
TGV_CSV = REPO_ROOT / "logs" / "tgv" / "results_per_seed.csv"


BASE_CFG = {
    "fourier_scale": 4.0,
    "smoothness_weight": 1.0,
    "learning_rate": 5e-3,
    "iterations": 8000,
    "huber_delta": 0.05,
}


# ---------------------------------------------------------------------------
# Trainer subclass that swaps the regularizer
# ---------------------------------------------------------------------------


class HuberTVTrainer(NVFTrainer):
    """
    Override the regularization step in NVFTrainer to use Huber-TV in
    place of plain TV. The huber-TV threshold delta is taken from
    self.cfg.huber_delta (already configured for the data loss); we
    deliberately reuse it so a single delta governs both the data loss
    Huber and the regularizer Huber, keeping hyperparameter complexity
    minimal.
    """

    def _train_step(self) -> dict:
        self.field.train()
        pred_tt = batched_straight_ray_travel_time(
            self.field, self.sources, self.receivers,
            n_samples=self.cfg.ray_samples,
        )
        data_loss = travel_time_data_loss(
            pred_tt, self.observed_tt,
            kind=self.cfg.data_loss, huber_delta=self.cfg.huber_delta,
        )

        v_grid = self.field.velocity_grid(
            self.cfg.smoothness_grid, self.cfg.smoothness_grid,
        )
        htv = huber_total_variation_2d(v_grid, delta=self.cfg.huber_delta)

        loss = data_loss + self.cfg.smoothness_weight * htv

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.field.parameters(), self.cfg.grad_clip,
            )
        self.optimizer.step()

        return {
            "loss": float(loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "tv": float(htv.detach().cpu()),     # logged under "tv" key for compat
        }


# ---------------------------------------------------------------------------
# helpers (mirror script 40)
# ---------------------------------------------------------------------------


def _load_truth(name: str) -> tuple[np.ndarray, BenchmarkSpec]:
    npz_path = DATA_DIR / f"{name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} not found.")
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
    set_global_seed(seed)
    geometry = cross_well_layout(spec, n_sources_per_side=12, n_receivers_per_side=12)
    rng = np.random.default_rng(seed + 1)
    src_np, rec_np, tt_np = sample_observed_travel_times(
        truth, spec, geometry, noise_pct=5.0, rng=rng,
    )

    nvf_cfg = NVFConfig(
        domain_x=spec.domain_x, domain_z=spec.domain_z,
        base_velocity=3.0, velocity_range=(2.0, 5.5),
        fourier_dim=64, fourier_scale=float(cfg["fourier_scale"]),
        hidden=128, depth=4, activation="tanh",
    )
    field = NeuralVelocityField(nvf_cfg).to(device)

    src = torch.from_numpy(src_np).to(device)
    rec = torch.from_numpy(rec_np).to(device)
    obs = torch.from_numpy(tt_np).to(device)

    out_dir.mkdir(parents=True, exist_ok=True)
    trn_cfg = NVFTrainingConfig(
        iterations=int(cfg["iterations"]),
        learning_rate=float(cfg["learning_rate"]),
        lr_schedule="cosine", warmup_iters=200,
        grad_clip=1.0,
        data_loss="huber", huber_delta=float(cfg["huber_delta"]),
        smoothness_weight=float(cfg["smoothness_weight"]),
        smoothness_grid=64, ray_samples=64,
        val_every=100, log_every=200,
        save_best=True, save_last=False,
        out_dir=str(out_dir),
    )
    trainer = HuberTVTrainer(field, src, rec, obs, truth_grid=truth, cfg=trn_cfg)

    t0 = time.time()
    state = trainer.fit()
    elapsed = time.time() - t0

    best_path = out_dir / "best.pt"
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    field.load_state_dict(ckpt["model_state_dict"])
    field.eval()
    with torch.no_grad():
        v_pred = field.velocity_grid(spec.nx, spec.nz).cpu().numpy()

    np.savez(out_dir / "history.npz",
             **{k: np.asarray(v) for k, v in state.history.items()},
             best_val_rmse=state.best_val_rmse)

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
    """3 x 3: rows = benchmarks, cols = (truth, Huber-TV, error)."""
    apply_paper_style()
    bnames = list(bench_results.keys())
    n = len(bnames)
    fig, axes = plt.subplots(n, 3, figsize=(8.4, 2.6 * n + 0.4),
                             constrained_layout=True)
    if n == 1:
        axes = axes[np.newaxis, :]
    for row, name in enumerate(bnames):
        truth = truths[name]; spec = specs[name]
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
            axes[row, 1].set_title("MIMIR + Huber-TV")
            axes[row, 2].set_title("Error")
        cb0 = fig.colorbar(im0, ax=axes[row, :2], location="right",
                           shrink=0.85, pad=0.02)
        cb0.set_label("v (km/s)")
        cb2 = fig.colorbar(im2, ax=axes[row, 2], location="right",
                           shrink=0.85, pad=0.02)
        cb2.set_label("Δv (km/s)")
    return fig


def _read_csv(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    rows = []
    with path.open("r", newline="") as f:
        for r in csv.DictReader(f):
            for k in ("best_rmse", "best_ssim", "best_pearson", "final_rmse", "elapsed_s"):
                if k in r and r[k] != "":
                    r[k] = float(r[k])
            for k in ("seed", "best_iter"):
                if k in r and r[k] != "":
                    r[k] = int(r[k])
            rows.append(r)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MIMIR with Huber-TV regularization.")
    parser.add_argument("--benchmark", default=None,
                        choices=["gaussian_anomaly", "layered", "curvefault_lookalike"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--huber-delta", type=float, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    cfg = dict(BASE_CFG)
    if args.iterations is not None:
        cfg["iterations"] = args.iterations
    if args.huber_delta is not None:
        cfg["huber_delta"] = args.huber_delta
    n_seeds = args.seeds
    if args.quick:
        n_seeds = 2
        cfg["iterations"] = 1000
        if args.benchmark is None:
            args.benchmark = "gaussian_anomaly"

    benchmarks = ([args.benchmark] if args.benchmark
                  else ["gaussian_anomaly", "layered", "curvefault_lookalike"])

    device = resolve_device(args.device)
    print(banner(device))
    print(f"[huber-tv] config: {cfg}")
    print(f"[huber-tv] seeds: {n_seeds}, benchmarks: {benchmarks}")

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    seeds = [20260507 + 17 * i for i in range(n_seeds)]

    all_rows: list[dict] = []
    bench_results: dict[str, dict] = {}
    truths: dict[str, np.ndarray] = {}
    specs: dict[str, BenchmarkSpec] = {}
    t_start = time.time()

    for bname in benchmarks:
        print(f"\n[huber-tv] === benchmark: {bname} ===")
        truth, spec = _load_truth(bname)
        truths[bname] = truth; specs[bname] = spec

        seed_runs: list[dict] = []
        for s_idx, seed in enumerate(seeds):
            print(f"\n[huber-tv]   seed {s_idx + 1}/{n_seeds} = {seed}")
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
            title_estimate=f"MIMIR + Huber-TV (best of {n_seeds} seeds)",
            rays=None,
        )
        save_pdf(fig, f"43_huber_tv_velocity_{bname}", out_dir=PDF_DIR)
        plt.close(fig)
        print(f"[huber-tv]   winner: seed={winner['seed']} "
              f"RMSE={winner['best_rmse']:.4f} SSIM={winner['best_ssim']:.3f}")

    # CSVs
    per_seed_csv = LOG_ROOT / "results_per_seed.csv"
    with per_seed_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n[huber-tv] per-seed CSV  -> {per_seed_csv}")

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
    print(f"[huber-tv] summary CSV   -> {summary_csv}")

    # Composite
    fig = make_composite(bench_results, truths, specs)
    save_pdf(fig, "43_huber_tv_composite", out_dir=PDF_DIR)
    plt.close(fig)
    print(f"[huber-tv] composite     -> {PDF_DIR / '43_huber_tv_composite.pdf'}")

    # Compare vs TGV
    tgv_rows = _read_csv(TGV_CSV)
    if tgv_rows is not None:
        cmp_rows: list[dict] = []
        tgv_by = {(r["benchmark"], r["seed"]): r for r in tgv_rows}
        htv_by = {(r["benchmark"], r["seed"]): r for r in all_rows}
        for bname in benchmarks:
            seeds_both = sorted({r["seed"] for r in all_rows if r["benchmark"] == bname})
            tgv_arr = np.asarray([tgv_by[(bname, s)]["best_rmse"] for s in seeds_both
                                   if (bname, s) in tgv_by])
            htv_arr = np.asarray([htv_by[(bname, s)]["best_rmse"] for s in seeds_both])
            from scipy import stats as _sp
            try:
                _, p = _sp.ttest_rel(htv_arr, tgv_arr)
            except Exception:
                p = float("nan")
            cmp_rows.append({
                "benchmark": bname,
                "tgv_rmse_mean": float(tgv_arr.mean()),
                "tgv_rmse_std": float(tgv_arr.std()),
                "huber_tv_rmse_mean": float(htv_arr.mean()),
                "huber_tv_rmse_std": float(htv_arr.std()),
                "delta_rmse": float(htv_arr.mean() - tgv_arr.mean()),
                "p_paired_t": float(p),
            })
        cmp_csv = LOG_ROOT / "comparison_with_tgv.csv"
        with cmp_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cmp_rows[0].keys()))
            w.writeheader()
            w.writerows(cmp_rows)
        print(f"[huber-tv] vs-TGV CSV    -> {cmp_csv}")

    elapsed = time.time() - t_start
    print(f"\n{'='*72}")
    print(f"[huber-tv] total wall-clock: {elapsed/60:.1f} min")
    print("[huber-tv] HUBER-TV TABLE")
    print("-"*72)
    print(f"{'benchmark':<24}{'RMSE (km/s)':>16}{'SSIM':>14}{'Pearson':>14}")
    print("-"*72)
    for r in summary_rows:
        print(f"{r['benchmark']:<24}"
              f"{r['mean_rmse']:>10.4f} ± {r['std_rmse']:.4f}"
              f"{r['mean_ssim']:>9.3f} ± {r['std_ssim']:.3f}"
              f"{r['mean_pearson']:>9.3f} ± {r['std_pearson']:.3f}")
    print("="*72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
