"""
44_verify_final_results.py
===========================

Load every experimental result produced by the MIMIR pipeline and verify
the headline numerical claims of the paper. Outputs a single CSV/console
report containing all paired-t-test p-values and confidence intervals for
review before submission.

Methods compared (each is a per-seed CSV):
    classical   FMM-LSMR baseline (script 10)
    tv          fixed-lambda TV (script 22)
    curriculum  log-linear annealed TV (script 30)
    huber       Huber-TV (script 43, optional)
    tgv         TGV^2 (script 40)

For every benchmark, every method-pair, computes:
    Δ RMSE  (M2 - M1, mean ± std of paired differences)
    paired Student's t-test p-value
    95% CI of Δ RMSE
    Δ SSIM and its p-value
    Δ Pearson and its p-value

Outputs
-------
    logs/final_verification/all_methods_summary.csv
    logs/final_verification/all_pairs_paired_t.csv
    logs/final_verification/headline_numbers.txt
    PDF/44_methods_overview_bars.pdf

Usage
-----
    python scripts/44_verify_final_results.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from mimir.viz.figures import apply_paper_style, save_pdf


REPO_ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = REPO_ROOT / "PDF"
OUT_DIR = REPO_ROOT / "logs" / "final_verification"

CSV_PATHS = {
    "classical":   REPO_ROOT / "logs" / "baseline_eikonal_fmm" / "results_per_seed.csv",
    "tv":          REPO_ROOT / "logs" / "paper_grade" / "results_per_seed.csv",
    "curriculum":  REPO_ROOT / "logs" / "curriculum" / "results_per_seed.csv",
    "huber":       REPO_ROOT / "logs" / "huber_tv" / "results_per_seed.csv",
    "tgv":         REPO_ROOT / "logs" / "tgv" / "results_per_seed.csv",
}

NICE_LABEL = {
    "classical":  "FMM-LSMR (classical)",
    "tv":         "MIMIR-TV",
    "curriculum": "MIMIR-TV (curriculum)",
    "huber":      "MIMIR-Huber-TV",
    "tgv":        "MIMIR-TGV²",
}


def _load_csv(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
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


def _per_method_per_benchmark(rows: list[dict]) -> dict[str, dict[str, np.ndarray]]:
    """Return dict[benchmark][metric] -> array(n_seeds,) sorted by seed."""
    benches = sorted({r["benchmark"] for r in rows})
    out: dict[str, dict[str, np.ndarray]] = {}
    for b in benches:
        cell = sorted([r for r in rows if r["benchmark"] == b],
                      key=lambda r: r["seed"])
        out[b] = {
            "rmse": np.asarray([r["best_rmse"] for r in cell]),
            "ssim": np.asarray([r["best_ssim"] for r in cell]),
            "pearson": np.asarray([r["best_pearson"] for r in cell]),
            "seeds": np.asarray([r["seed"] for r in cell]),
        }
    return out


def _paired_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    """Returns (mean diff, std diff, p-value, ci95_half_width)."""
    if len(a) != len(b) or len(a) < 2:
        return (float("nan"),) * 4
    diff = a - b
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1))
    n = len(diff)
    se = sd / np.sqrt(n)
    try:
        _, p = stats.ttest_rel(a, b)
    except Exception:
        p = float("nan")
    # 95% CI half-width
    if n >= 2:
        t_crit = stats.t.ppf(0.975, df=n - 1)
        ci = float(t_crit * se)
    else:
        ci = float("nan")
    return mean, sd, float(p), ci


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load every available results CSV ----
    method_rows: dict[str, list[dict]] = {}
    for m, path in CSV_PATHS.items():
        rows = _load_csv(path)
        if rows is None:
            print(f"[verify]   ✗ {m:<12} NOT FOUND  ({path})")
            continue
        method_rows[m] = rows
        n_b = len({r["benchmark"] for r in rows})
        n_s = len({(r["benchmark"], r["seed"]) for r in rows})
        print(f"[verify]   ✓ {m:<12} loaded ({n_b} benchmarks × {n_s // n_b if n_b else 0} seeds)")

    if not method_rows:
        print("[verify] FATAL: no results found. Run earlier scripts first.")
        return 1

    # ---- 2. Per-method per-benchmark stats ----
    by_method = {m: _per_method_per_benchmark(rows)
                 for m, rows in method_rows.items()}
    benches = sorted(set().union(*[set(by_method[m].keys()) for m in by_method]))

    # Summary table (mean ± std per cell)
    summary_csv = OUT_DIR / "all_methods_summary.csv"
    summary_rows: list[dict] = []
    for b in benches:
        for m in method_rows:
            if b not in by_method[m]:
                continue
            d = by_method[m][b]
            summary_rows.append({
                "benchmark": b,
                "method": m,
                "method_label": NICE_LABEL[m],
                "n_seeds": len(d["rmse"]),
                "mean_rmse": float(d["rmse"].mean()),
                "std_rmse": float(d["rmse"].std()),
                "mean_ssim": float(d["ssim"].mean()),
                "std_ssim": float(d["ssim"].std()),
                "mean_pearson": float(d["pearson"].mean()),
                "std_pearson": float(d["pearson"].std()),
            })
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\n[verify] all_methods_summary    -> {summary_csv}")

    # ---- 3. All paired t-tests ----
    pairs_csv = OUT_DIR / "all_pairs_paired_t.csv"
    pair_rows: list[dict] = []
    for b in benches:
        for m1 in method_rows:
            for m2 in method_rows:
                if m1 >= m2:                    # alphabetical, no duplicates
                    continue
                if b not in by_method[m1] or b not in by_method[m2]:
                    continue
                d1 = by_method[m1][b]; d2 = by_method[m2][b]
                # Align by seed
                common = sorted(set(d1["seeds"].tolist()) & set(d2["seeds"].tolist()))
                if len(common) < 2:
                    continue
                idx1 = [list(d1["seeds"]).index(s) for s in common]
                idx2 = [list(d2["seeds"]).index(s) for s in common]
                a_r = d1["rmse"][idx1]; b_r = d2["rmse"][idx2]
                a_s = d1["ssim"][idx1]; b_s = d2["ssim"][idx2]
                a_p = d1["pearson"][idx1]; b_p = d2["pearson"][idx2]
                m_r, sd_r, p_r, ci_r = _paired_test(b_r, a_r)
                m_s, sd_s, p_s, ci_s = _paired_test(b_s, a_s)
                m_p, sd_p, p_p, ci_p = _paired_test(b_p, a_p)
                pair_rows.append({
                    "benchmark": b,
                    "method_a": m1,
                    "method_b": m2,
                    "n_paired_seeds": len(common),
                    "delta_rmse_mean": round(m_r, 4),
                    "delta_rmse_std": round(sd_r, 4),
                    "delta_rmse_ci95": round(ci_r, 4),
                    "p_rmse": round(p_r, 4),
                    "delta_ssim_mean": round(m_s, 4),
                    "delta_ssim_std": round(sd_s, 4),
                    "p_ssim": round(p_s, 4),
                    "delta_pearson_mean": round(m_p, 4),
                    "p_pearson": round(p_p, 4),
                })
    with pairs_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        w.writerows(pair_rows)
    print(f"[verify] all_pairs_paired_t     -> {pairs_csv}")

    # ---- 4. Headline numbers (TGV vs FMM, TGV vs TV) ----
    headline_lines: list[str] = []

    def fmt_method(m: str) -> str:
        return NICE_LABEL.get(m, m)

    headline_lines.append("=" * 90)
    headline_lines.append("HEADLINE NUMBERS — copy-paste into manuscript")
    headline_lines.append("=" * 90)

    # Per-method table per benchmark
    headline_lines.append("\n=== Table 1: per-method per-benchmark mean ± std (5 seeds, 8000 iter) ===")
    headline_lines.append(f"{'benchmark':<24}{'method':<28}{'RMSE (km/s)':>20}{'SSIM':>16}")
    headline_lines.append("-" * 90)
    for b in benches:
        for m in ("classical", "tv", "curriculum", "huber", "tgv"):
            if m in by_method and b in by_method[m]:
                d = by_method[m][b]
                headline_lines.append(
                    f"{b:<24}{fmt_method(m):<28}"
                    f"{d['rmse'].mean():>10.4f} ± {d['rmse'].std():.4f}"
                    f"{d['ssim'].mean():>10.3f} ± {d['ssim'].std():.3f}"
                )
        headline_lines.append("")

    # Critical pairs
    headline_lines.append("=== Table 2: critical paired comparisons (paired Student's t-test, 5 seeds) ===")
    critical_pairs = [
        ("classical", "tgv",     "MIMIR-TGV² vs FMM-LSMR  (does the headline claim hold?)"),
        ("tv",        "tgv",     "MIMIR-TGV² vs MIMIR-TV  (does TGV beat TV?)"),
        ("classical", "tv",      "MIMIR-TV vs FMM-LSMR    (TV-only baseline check)"),
        ("tv",        "curriculum", "Curriculum-TV vs Fixed-TV (does annealing help?)"),
        ("huber",     "tgv",     "MIMIR-TGV² vs Huber-TV  (Supplementary)"),
    ]
    headline_lines.append(f"\n{'comparison':<55}{'benchmark':<24}{'Δ RMSE (km/s)':>20}{'p':>8}")
    headline_lines.append("-" * 110)
    for m_a, m_b, label in critical_pairs:
        if m_a not in by_method or m_b not in by_method:
            continue
        for b in benches:
            row = next((r for r in pair_rows
                        if r["benchmark"] == b
                        and ((r["method_a"] == m_a and r["method_b"] == m_b)
                             or (r["method_a"] == m_b and r["method_b"] == m_a))),
                       None)
            if row is None:
                continue
            sign = +1 if row["method_a"] == m_a else -1
            d = sign * row["delta_rmse_mean"]
            sd = row["delta_rmse_std"]
            p = row["p_rmse"]
            star = " *" if p < 0.05 else "  "
            headline_lines.append(
                f"{label[:54]:<55}{b:<24}{d:>+12.4f} ± {sd:.4f}{p:>7.4f}{star}"
            )
        headline_lines.append("")

    txt = "\n".join(headline_lines)
    headline_path = OUT_DIR / "headline_numbers.txt"
    headline_path.write_text(txt + "\n")
    print(f"[verify] headline_numbers       -> {headline_path}")
    print()
    print(txt)

    # ---- 5. Methods overview bar chart ----
    apply_paper_style()
    methods_for_plot = [m for m in ("classical", "tv", "curriculum", "huber", "tgv")
                        if m in by_method]
    n_methods = len(methods_for_plot)
    n_benches = len(benches)
    fig, ax = plt.subplots(figsize=(8.4, 3.6), constrained_layout=True)
    width = 0.8 / n_methods
    x = np.arange(n_benches)
    palette = ["tab:orange", "tab:blue", "tab:green", "tab:purple", "tab:red"]
    for i, m in enumerate(methods_for_plot):
        means = [by_method[m][b]["rmse"].mean() if b in by_method[m] else np.nan
                 for b in benches]
        stds = [by_method[m][b]["rmse"].std() if b in by_method[m] else 0.0
                for b in benches]
        ax.bar(x + (i - (n_methods - 1) / 2) * width,
               means, yerr=stds, width=width,
               label=NICE_LABEL[m], color=palette[i], capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("_", "\n") for b in benches])
    ax.set_ylabel("Validation RMSE (km s$^{-1}$)")
    ax.set_title("All methods — mean ± std across 5 seeds")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    save_pdf(fig, "44_methods_overview_bars", out_dir=PDF_DIR)
    plt.close(fig)
    print(f"\n[verify] methods overview PDF   -> {PDF_DIR / '44_methods_overview_bars.pdf'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
