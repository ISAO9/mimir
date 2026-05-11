"""
20_train_neural_velocity_field.py
=================================

End-to-end training of a Neural Velocity Field on a chosen synthetic
benchmark.

Pipeline
--------
1. Load YAML config (defaults to ``configs/nvf_default.yaml``).
2. Load the saved ground-truth velocity from ``data/synthetic/<name>.npz``
   produced by ``01_generate_synthetic_benchmarks.py``.
3. Build the cross-well acquisition geometry and compute *clean* observed
   travel times via FMM (skfmm), then add Gaussian noise as a percentage
   of each clean travel time.
4. Build a NeuralVelocityField with Fourier features and the recipe
   validated across the prototype 76–82 lineage.
5. Train with the ``NVFTrainer``, which keeps the best-RMSE checkpoint.
6. Render paper-grade figures to ``PDF/``:
     * ground-truth vs. estimate side-by-side
     * loss curve
     * validation-RMSE / SSIM curves

Outputs
-------
* models/<run_name>/best.pt       — best-validation checkpoint
* models/<run_name>/last.pt       — final-iteration checkpoint
* logs/<run_name>/history.npz     — full per-iteration log
* PDF/20_<run_name>_*.pdf         — three paper figures

Usage
-----
    python scripts/20_train_neural_velocity_field.py
    python scripts/20_train_neural_velocity_field.py \\
        --config configs/nvf_default.yaml \\
        --benchmark layered \\
        --seed 20260101
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from mimir.data.acquisition import (
    cross_well_layout,
    sample_observed_travel_times,
    surface_layout,
)
from mimir.data.benchmarks import BenchmarkSpec
from mimir.fields import NeuralVelocityField
from mimir.fields.neural_velocity_field import NVFConfig
from mimir.training import NVFTrainer, NVFTrainingConfig
from mimir.utils import resolve_device, set_global_seed
from mimir.utils.device import banner
from mimir.viz.figures import (
    apply_paper_style,
    plot_loss_history,
    plot_velocity_pair,
    save_pdf,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "synthetic"
PDF_DIR = REPO_ROOT / "PDF"
MODELS_DIR = REPO_ROOT / "models"
LOGS_DIR = REPO_ROOT / "logs"


# ---------------------------------------------------------------------------
# config helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _override(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    sub = cfg
    for k in keys[:-1]:
        sub = sub.setdefault(k, {})
    sub[keys[-1]] = value


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
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MIMIR NVF training.")
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs" / "nvf_default.yaml"),
        help="YAML config file.",
    )
    parser.add_argument("--benchmark", default=None, help="Override data.benchmark.")
    parser.add_argument("--seed", type=int, default=None, help="Override experiment.seed.")
    parser.add_argument(
        "--source-layout", default=None, choices=["cross", "surface"],
        help="Override data.source_layout.",
    )
    parser.add_argument("--noise-pct", type=float, default=None, help="Override data.noise_pct.")
    parser.add_argument("--iterations", type=int, default=None, help="Override training.iterations.")
    parser.add_argument(
        "--fourier-scale", type=float, default=None,
        help="Override field.fourier_scale. Lower (~1-2) for smooth fields, "
        "higher (~4-8) for sharp interfaces. CRITICAL hyperparameter.",
    )
    parser.add_argument(
        "--fourier-dim", type=int, default=None,
        help="Override field.fourier_dim (number of random frequencies).",
    )
    parser.add_argument(
        "--smoothness-weight", type=float, default=None,
        help="Override loss.smoothness_weight (TV regularizer strength).",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="Override training.learning_rate.",
    )
    parser.add_argument(
        "--no-rays", action="store_true",
        help="Do not draw ray overlay on the velocity-pair figure (cleaner).",
    )
    parser.add_argument(
        "--tag", default=None,
        help="Optional run-name tag, appended to the auto-generated run name.",
    )
    args = parser.parse_args(argv)

    cfg = _load_yaml(Path(args.config))

    # Apply CLI overrides
    if args.benchmark is not None:
        _override(cfg, "data.benchmark", args.benchmark)
    if args.seed is not None:
        _override(cfg, "experiment.seed", args.seed)
    if args.source_layout is not None:
        _override(cfg, "data.source_layout", args.source_layout)
    if args.noise_pct is not None:
        _override(cfg, "data.noise_pct", args.noise_pct)
    if args.iterations is not None:
        _override(cfg, "training.iterations", args.iterations)
    if args.fourier_scale is not None:
        _override(cfg, "field.fourier_scale", args.fourier_scale)
    if args.fourier_dim is not None:
        _override(cfg, "field.fourier_dim", args.fourier_dim)
    if args.smoothness_weight is not None:
        _override(cfg, "loss.smoothness_weight", args.smoothness_weight)
    if args.learning_rate is not None:
        _override(cfg, "training.learning_rate", args.learning_rate)

    # Names / dirs
    benchmark = cfg["data"]["benchmark"]
    run_name = f"{cfg['experiment']['name']}_{benchmark}_seed{cfg['experiment']['seed']}"
    if args.tag:
        run_name = f"{run_name}_{args.tag}"
    out_dir = LOGS_DIR / run_name
    model_dir = MODELS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Reproducibility & device
    seed = int(cfg["experiment"]["seed"])
    set_global_seed(seed)
    device = resolve_device(cfg["experiment"]["device"])
    print(banner(device))

    # 2. Load truth
    truth, spec = _load_truth(benchmark)
    print(f"[data] benchmark={spec.name} grid={spec.nz}x{spec.nx} "
          f"velocity range=[{truth.min():.2f}, {truth.max():.2f}] km/s")

    # 3. Acquisition + observations
    if cfg["data"]["source_layout"] == "cross":
        geometry = cross_well_layout(
            spec,
            n_sources_per_side=cfg["data"]["n_sources_per_side"],
            n_receivers_per_side=cfg["data"]["n_receivers_per_side"],
        )
    else:
        geometry = surface_layout(spec)
    rng = np.random.default_rng(seed + 1)
    sources_np, receivers_np, observed_tt_np = sample_observed_travel_times(
        truth, spec, geometry, noise_pct=float(cfg["data"]["noise_pct"]), rng=rng,
    )
    print(f"[data] {sources_np.shape[0]} rays, "
          f"travel times in [{observed_tt_np.min():.3f}, {observed_tt_np.max():.3f}] s, "
          f"noise={cfg['data']['noise_pct']:.1f}%")

    # 4. Build NVF
    nvf_cfg = NVFConfig(
        domain_x=tuple(cfg["data"]["domain_x"]),
        domain_z=tuple(cfg["data"]["domain_z"]),
        base_velocity=float(cfg["field"]["base_velocity"]),
        velocity_range=tuple(cfg["field"]["velocity_range"]),
        fourier_dim=int(cfg["field"]["fourier_dim"]),
        fourier_scale=float(cfg["field"]["fourier_scale"]),
        hidden=int(cfg["field"]["hidden"]),
        depth=int(cfg["field"]["depth"]),
        activation=cfg["field"]["activation"],
    )
    field = NeuralVelocityField(nvf_cfg).to(device)
    n_params = sum(p.numel() for p in field.parameters() if p.requires_grad)
    print(f"[field] NVF parameters: {n_params:,}")

    # 5. Move data to device
    sources = torch.from_numpy(sources_np).to(device)
    receivers = torch.from_numpy(receivers_np).to(device)
    observed_tt = torch.from_numpy(observed_tt_np).to(device)

    # 6. Trainer
    trn_cfg = NVFTrainingConfig(
        iterations=int(cfg["training"]["iterations"]),
        learning_rate=float(cfg["training"]["learning_rate"]),
        lr_schedule=cfg["training"]["lr_schedule"],
        warmup_iters=int(cfg["training"]["warmup_iters"]),
        grad_clip=float(cfg["training"]["grad_clip"]),
        data_loss=cfg["loss"]["data_loss"],
        huber_delta=float(cfg["loss"]["huber_delta"]),
        smoothness_weight=float(cfg["loss"]["smoothness_weight"]),
        smoothness_grid=64,
        ray_samples=int(cfg["physics"]["ray_samples"]),
        val_every=int(cfg["training"]["val_every"]),
        log_every=int(cfg["training"]["log_every"]),
        save_best=bool(cfg["training"]["save_best"]),
        out_dir=str(model_dir),
    )
    trainer = NVFTrainer(
        field=field,
        sources=sources,
        receivers=receivers,
        observed_tt=observed_tt,
        truth_grid=truth,
        cfg=trn_cfg,
    )

    # 7. Fit
    state = trainer.fit()

    # 8. Save full history (separately from checkpoints)
    hist_path = out_dir / "history.npz"
    np.savez(
        hist_path,
        **{k: np.asarray(v) for k, v in state.history.items()},
        best_val_rmse=state.best_val_rmse,
    )

    # 9. Reload best checkpoint for figure generation
    best_ckpt = torch.load(model_dir / "best.pt", map_location=device, weights_only=False)
    field.load_state_dict(best_ckpt["model_state_dict"])
    field.eval()
    with torch.no_grad():
        v_pred = field.velocity_grid(spec.nx, spec.nz).cpu().numpy()

    # 10. Figures (white bg, English, PDF, legends in margins — handled by viz)
    apply_paper_style()

    fig_pair = plot_velocity_pair(
        truth, v_pred, spec,
        title_truth="Ground truth", title_estimate="MIMIR NVF estimate",
        rays=None if args.no_rays else (sources_np, receivers_np),
    )
    pair_pdf = save_pdf(fig_pair, f"20_{run_name}_velocity_pair", out_dir=PDF_DIR)

    fig_loss = plot_loss_history(
        state.history["loss"],
        iterations=state.history["iter"],
        title="Total loss", smoothing=10,
    )
    loss_pdf = save_pdf(fig_loss, f"20_{run_name}_loss", out_dir=PDF_DIR)

    # Validation curves (RMSE & SSIM)
    if state.history["val_iter"]:
        fig, ax1 = plt.subplots(figsize=(5.5, 3.0), constrained_layout=True)
        ax1.plot(state.history["val_iter"], state.history["val_rmse"],
                 marker="o", markersize=3, color="tab:blue", label="RMSE")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Validation RMSE (km/s)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(state.history["val_iter"], state.history["val_ssim"],
                 marker="s", markersize=3, color="tab:orange", label="SSIM")
        ax2.set_ylabel("Validation SSIM", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        # Spines fix for twin axes (Matplotlib defaults reintroduce the right spine)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(True)

        ax1.set_title("Validation RMSE & SSIM during training")
        val_pdf = save_pdf(fig, f"20_{run_name}_validation", out_dir=PDF_DIR)
    else:
        val_pdf = None

    # 11. Summary
    print("=" * 72)
    print(f"[done] best validation RMSE = {state.best_val_rmse:.4f} km/s")
    print(f"       checkpoints: {model_dir}")
    print(f"       history    : {hist_path}")
    print(f"       figures    : {pair_pdf}")
    print(f"                    {loss_pdf}")
    if val_pdf is not None:
        print(f"                    {val_pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
