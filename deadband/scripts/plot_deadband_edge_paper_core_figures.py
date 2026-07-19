#!/usr/bin/env python3
"""
Generate core paper figures for the deadband-edge accumulation study.

Outputs:
- baseline vs best-candidate distribution comparison
- baseline vs best-candidate 96-dispatch envelope comparison
- representative dispatch-pair mechanism comparison
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "openandes-mpl"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-results-dir", type=Path, required=True)
    parser.add_argument("--best-results-dir", type=Path, required=True)
    parser.add_argument("--baseline-trace-csv", type=Path, required=True)
    parser.add_argument("--best-trace-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--summary-name", type=str, default="daily_hotstart_summary.csv")
    parser.add_argument("--dispatch-interval", type=int, default=900)
    parser.add_argument("--zoom-seconds", type=float, default=360.0)
    parser.add_argument("--db-edge-hz", type=float, default=0.036)
    return parser.parse_args()


def load_summary(results_dir: Path, summary_name: str) -> pd.DataFrame:
    summary = pd.read_csv(results_dir / summary_name).copy()
    summary["freq_csv"] = summary["freq_csv"].astype(str)
    summary = summary[summary["freq_csv"].str.len() > 0].copy()
    summary = summary[summary["freq_csv"].map(lambda p: Path(p).exists())].copy()
    if summary.empty:
        raise RuntimeError(f"No usable rows found in {results_dir / summary_name}")
    return summary.sort_values(["hour", "dispatch"]).reset_index(drop=True)


def load_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["time_s", "freq_dev_hz"]].copy()


def collect_samples(summary: pd.DataFrame) -> np.ndarray:
    blocks = []
    for _, row in summary.iterrows():
        df = load_series(Path(row["freq_csv"]))
        y = df["freq_dev_hz"].to_numpy(dtype=float)
        if y.size:
            blocks.append(y)
    if not blocks:
        raise RuntimeError("No frequency samples found.")
    return np.concatenate(blocks)


def load_grid(summary: pd.DataFrame, duration_seconds: int) -> tuple[np.ndarray, np.ndarray]:
    t_grid = np.arange(duration_seconds, dtype=float)
    rows = []
    for _, row in summary.iterrows():
        df = load_series(Path(row["freq_csv"]))
        t = df["time_s"].to_numpy(dtype=float)
        y = df["freq_dev_hz"].to_numpy(dtype=float)
        if t.size == 0:
            rows.append(np.full_like(t_grid, np.nan))
            continue
        left = y[0]
        right = y[-1] if t_grid[-1] <= t[-1] else np.nan
        rows.append(np.interp(t_grid, t, y, left=left, right=right))
    return t_grid, np.vstack(rows)


def freq_stats(samples: np.ndarray) -> dict[str, float]:
    abs_samples = np.abs(samples)
    return {
        "mean_abs_hz": float(np.mean(abs_samples)),
        "share_abs_gt_0p036": float(np.mean(abs_samples > 0.036)),
        "share_abs_gt_0p05": float(np.mean(abs_samples > 0.05)),
        "max_abs_hz": float(np.max(abs_samples)),
    }


def write_distribution_compare(
    *,
    baseline_samples: np.ndarray,
    best_samples: np.ndarray,
    out_path: Path,
    db_edge_hz: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.0))
    colors = {"baseline": "#8c564b", "best": "#0f5c78"}

    low = float(min(np.quantile(baseline_samples, 0.001), np.quantile(best_samples, 0.001)))
    high = float(max(np.quantile(baseline_samples, 0.999), np.quantile(best_samples, 0.999)))
    bins = np.linspace(low, high, 160)
    abs_bins = np.linspace(0.0, max(0.08, float(max(np.quantile(np.abs(baseline_samples), 0.999), np.quantile(np.abs(best_samples), 0.999))) * 1.02), 120)

    axes[0].hist(
        baseline_samples,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.0,
        color=colors["baseline"],
        label="Baseline",
    )
    axes[0].hist(
        best_samples,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.0,
        color=colors["best"],
        label="Best candidate",
    )
    for sign in (-1.0, 1.0):
        axes[0].axvline(sign * db_edge_hz, color="#666666", linestyle=":", linewidth=1.0)
    axes[0].axvline(0.0, color="#999999", linestyle="--", linewidth=0.9)
    axes[0].set_title("Signed Frequency-Deviation Distribution")
    axes[0].set_xlabel("Frequency deviation [Hz]")
    axes[0].set_ylabel("Probability density")
    axes[0].grid(True, alpha=0.22)
    axes[0].legend(frameon=False, loc="upper left")

    axes[1].hist(
        np.abs(baseline_samples),
        bins=abs_bins,
        density=True,
        histtype="step",
        linewidth=2.0,
        color=colors["baseline"],
        label="Baseline",
    )
    axes[1].hist(
        np.abs(best_samples),
        bins=abs_bins,
        density=True,
        histtype="step",
        linewidth=2.0,
        color=colors["best"],
        label="Best candidate",
    )
    axes[1].axvline(db_edge_hz, color="#666666", linestyle=":", linewidth=1.0)
    axes[1].set_xlim(0.0, max(0.07, db_edge_hz * 1.9))
    axes[1].set_title("Absolute-Deviation Distribution (Shoulder View)")
    axes[1].set_xlabel("|Frequency deviation| [Hz]")
    axes[1].set_ylabel("Probability density")
    axes[1].grid(True, alpha=0.22)

    base_stats = freq_stats(baseline_samples)
    best_stats = freq_stats(best_samples)
    text = "\n".join([
        "Baseline",
        f"mean|f| = {base_stats['mean_abs_hz']:.4f} Hz",
        f"share>|0.036| = {base_stats['share_abs_gt_0p036']:.2%}",
        f"share>|0.05| = {base_stats['share_abs_gt_0p05']:.2%}",
        "",
        "Best candidate",
        f"mean|f| = {best_stats['mean_abs_hz']:.4f} Hz",
        f"share>|0.036| = {best_stats['share_abs_gt_0p036']:.2%}",
        f"share>|0.05| = {best_stats['share_abs_gt_0p05']:.2%}",
    ])
    axes[1].text(
        0.98,
        0.97,
        text,
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )

    fig.suptitle("Baseline vs Best Candidate: Frequency Distribution and 36 mHz Shoulder", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _envelope_stats(summary: pd.DataFrame, duration_seconds: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    t_grid, grid = load_grid(summary, duration_seconds)
    stats = {
        "mean": np.nanmean(grid, axis=0),
        "median": np.nanmedian(grid, axis=0),
        "p10": np.nanpercentile(grid, 10, axis=0),
        "p90": np.nanpercentile(grid, 90, axis=0),
        "min": np.nanmin(grid, axis=0),
        "max": np.nanmax(grid, axis=0),
    }
    return t_grid, stats


def write_curve_compare(
    *,
    baseline_summary: pd.DataFrame,
    best_summary: pd.DataFrame,
    out_path: Path,
    duration_seconds: int,
    db_edge_hz: float,
) -> None:
    t_b, s_b = _envelope_stats(baseline_summary, duration_seconds)
    t_c, s_c = _envelope_stats(best_summary, duration_seconds)
    y_lim = max(
        float(np.nanquantile(np.abs(np.concatenate([s_b["min"], s_b["max"], s_c["min"], s_c["max"]])), 0.995)),
        db_edge_hz * 1.6,
    )

    fig, axes = plt.subplots(1, 2, figsize=(15.4, 5.8), sharey=True)
    panels = [
        ("Baseline", t_b, s_b, axes[0], "#8c564b"),
        ("Best candidate", t_c, s_c, axes[1], "#0f5c78"),
    ]
    for title, t_grid, stats, ax, color in panels:
        ax.fill_between(t_grid, stats["min"], stats["max"], color="#d9e5ec", alpha=0.85, label="Min-Max")
        ax.fill_between(t_grid, stats["p10"], stats["p90"], color="#8fb6c9", alpha=0.9, label="10-90 percentile")
        ax.plot(t_grid, stats["median"], color=color, linewidth=1.8, label="Median")
        ax.plot(t_grid, stats["mean"], color="#111111", linewidth=1.4, linestyle="--", label="Mean")
        for sign in (-1.0, 1.0):
            ax.axhline(sign * db_edge_hz, color="#666666", linestyle=":", linewidth=0.9)
        ax.axhline(0.0, color="#999999", linestyle="-", linewidth=0.8)
        ax.set_xlim(0.0, duration_seconds - 1)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_title(title)
        ax.set_xlabel("Time within dispatch [s]")
        ax.grid(True, alpha=0.22)
    axes[0].set_ylabel("Frequency deviation [Hz]")
    axes[1].legend(frameon=False, loc="upper right")

    fig.suptitle("Baseline vs Best Candidate: 96-Dispatch Frequency Envelope", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_mechanism_compare(
    *,
    baseline_trace: pd.DataFrame,
    best_trace: pd.DataFrame,
    out_path: Path,
    zoom_seconds: float,
    db_edge_hz: float,
) -> None:
    base = baseline_trace.copy()
    best = best_trace.copy()
    base["pvd_pe_pref_gap"] = base["pvd_pe_sum"] - base["pvd_pref_sum"]
    base["esd_pe_pref_gap"] = base["esd_pe_sum"] - base["esd_pref_sum"]
    best["pvd_pe_pref_gap"] = best["pvd_pe_sum"] - best["pvd_pref_sum"]
    best["esd_pe_pref_gap"] = best["esd_pe_sum"] - best["esd_pref_sum"]

    row_specs = [
        ("Frequency deviation [Hz]", [("freq_dev_hz", "Freq", "#111111")]),
        ("Governor droop [pu]", [("gov_droop_sum", "Gov droop", "#1f77b4")]),
        ("DER droop [pu]", [("pvd_droop_sum", "PVD droop", "#ff7f0e"), ("esd_droop_sum", "ESD droop", "#2ca02c")]),
        ("Pe - Pref [pu]", [("pvd_pe_pref_gap", "PVD Pe-Pref", "#ff7f0e"), ("esd_pe_pref_gap", "ESD Pe-Pref", "#2ca02c")]),
    ]

    fig, axes = plt.subplots(len(row_specs), 2, figsize=(15.6, 10.6), sharex="col", constrained_layout=True)
    datasets = [("Baseline", base), ("Best candidate", best)]

    for row_idx, (ylabel, series_specs) in enumerate(row_specs):
        y_min = np.inf
        y_max = -np.inf
        for _, df in datasets:
            mask = df["time_s"] <= zoom_seconds
            for col, _, _ in series_specs:
                vals = df.loc[mask, col].to_numpy(dtype=float)
                if vals.size:
                    y_min = min(y_min, float(np.min(vals)))
                    y_max = max(y_max, float(np.max(vals)))
        pad = 0.08 * max(abs(y_min), abs(y_max), 1e-6)
        for col_idx, (title, df) in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            mask = df["time_s"] <= zoom_seconds
            for col, label, color in series_specs:
                ax.plot(df.loc[mask, "time_s"], df.loc[mask, col], color=color, linewidth=1.7, label=label)
            if row_idx == 0:
                ax.axhline(db_edge_hz, color="#666666", linestyle=":", linewidth=0.9)
                ax.axhline(-db_edge_hz, color="#666666", linestyle=":", linewidth=0.9)
            ax.axhline(0.0, color="#999999", linestyle="-", linewidth=0.8)
            ax.set_xlim(0.0, zoom_seconds)
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.grid(True, alpha=0.22)
            if row_idx == 0:
                ax.set_title(title)
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            if row_idx == len(row_specs) - 1:
                ax.set_xlabel("Time in selected dispatch [s]")
            if row_idx == 0 or row_idx >= 2:
                ax.legend(frameon=False, loc="upper right")

    fig.suptitle("Representative Dispatch Pair: Baseline vs Best Candidate Mechanism Comparison", fontsize=14)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary = load_summary(args.baseline_results_dir, args.summary_name)
    best_summary = load_summary(args.best_results_dir, args.summary_name)
    baseline_samples = collect_samples(baseline_summary)
    best_samples = collect_samples(best_summary)

    write_distribution_compare(
        baseline_samples=baseline_samples,
        best_samples=best_samples,
        out_path=args.out_dir / "fig01_distribution_compare.png",
        db_edge_hz=float(args.db_edge_hz),
    )
    write_curve_compare(
        baseline_summary=baseline_summary,
        best_summary=best_summary,
        out_path=args.out_dir / "fig02_curves_compare.png",
        duration_seconds=int(args.dispatch_interval),
        db_edge_hz=float(args.db_edge_hz),
    )

    baseline_trace = pd.read_csv(args.baseline_trace_csv)
    best_trace = pd.read_csv(args.best_trace_csv)
    write_mechanism_compare(
        baseline_trace=baseline_trace,
        best_trace=best_trace,
        out_path=args.out_dir / "fig03_mechanism_compare.png",
        zoom_seconds=float(args.zoom_seconds),
        db_edge_hz=float(args.db_edge_hz),
    )

    print(f"distribution_compare_png={args.out_dir / 'fig01_distribution_compare.png'}")
    print(f"curves_compare_png={args.out_dir / 'fig02_curves_compare.png'}")
    print(f"mechanism_compare_png={args.out_dir / 'fig03_mechanism_compare.png'}")


if __name__ == "__main__":
    main()
