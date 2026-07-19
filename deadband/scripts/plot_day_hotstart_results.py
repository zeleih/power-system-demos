#!/usr/bin/env python3
"""
Generate aggregate plots for a hot-start 96-dispatch run.

Inputs:
- daily hot-start summary CSV produced by ``run_day_dispatch_hotstart.py``
- per-dispatch frequency CSVs referenced by that summary

Outputs:
- frequency distribution PNG + stats CSV
- one all-96 overview PNG
- four 6-hour grouped PNGs covering all 96 dispatch intervals
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
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--summary-name", type=str, default="daily_hotstart_summary.csv")
    parser.add_argument("--duration-seconds", type=int, default=900)
    parser.add_argument("--bins", type=int, default=160)
    return parser.parse_args()


def load_summary(results_dir: Path, summary_name: str) -> pd.DataFrame:
    summary_path = results_dir / summary_name
    summary = pd.read_csv(summary_path)
    summary = summary.copy()
    summary["freq_csv"] = summary["freq_csv"].astype(str)
    summary = summary[summary["freq_csv"].str.len() > 0].copy()
    summary = summary[summary["freq_csv"].map(lambda p: Path(p).exists())].copy()
    if summary.empty:
        raise RuntimeError(f"No usable rows found in {summary_path}")
    return summary.sort_values(["hour", "dispatch"]).reset_index(drop=True)


def load_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["time_s", "freq_dev_hz"]].copy()


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
        grid = np.interp(t_grid, t, y, left=left, right=right)
        rows.append(grid)
    return t_grid, np.vstack(rows)


def collect_samples(summary: pd.DataFrame) -> np.ndarray:
    blocks = []
    for _, row in summary.iterrows():
        df = load_series(Path(row["freq_csv"]))
        y = df["freq_dev_hz"].to_numpy(dtype=float)
        if y.size:
            blocks.append(y)
    if not blocks:
        raise RuntimeError("No frequency samples found in summary.")
    return np.concatenate(blocks)


def write_threshold_stats(results_dir: Path, samples: np.ndarray) -> None:
    abs_samples = np.abs(samples)
    thresholds = [0.01, 0.02, 0.036, 0.05, 0.06, 0.08, 0.10]
    rows = []
    for threshold in thresholds:
        rows.append({
            "threshold_hz": float(threshold),
            "share_abs_gt": float(np.mean(abs_samples > threshold)),
            "share_abs_ge": float(np.mean(abs_samples >= threshold)),
        })
    pd.DataFrame(rows).to_csv(results_dir / "frequency_threshold_stats.csv", index=False)


def write_distribution(results_dir: Path, samples: np.ndarray, bins: int) -> None:
    abs_samples = np.abs(samples)
    stats = pd.DataFrame([{
        "count": int(samples.size),
        "mean_hz": float(np.mean(samples)),
        "std_hz": float(np.std(samples)),
        "min_hz": float(np.min(samples)),
        "p01_hz": float(np.quantile(samples, 0.01)),
        "p05_hz": float(np.quantile(samples, 0.05)),
        "median_hz": float(np.quantile(samples, 0.50)),
        "p95_hz": float(np.quantile(samples, 0.95)),
        "p99_hz": float(np.quantile(samples, 0.99)),
        "max_hz": float(np.max(samples)),
        "mean_abs_hz": float(np.mean(abs_samples)),
        "p95_abs_hz": float(np.quantile(abs_samples, 0.95)),
        "p99_abs_hz": float(np.quantile(abs_samples, 0.99)),
        "max_abs_hz": float(np.max(abs_samples)),
        "share_abs_gt_0p05": float(np.mean(abs_samples > 0.05)),
        "share_abs_gt_0p036": float(np.mean(abs_samples > 0.036)),
    }])
    stats.to_csv(results_dir / "frequency_distribution_stats.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    hist_range = np.quantile(samples, [0.001, 0.999])
    if hist_range[0] == hist_range[1]:
        hist_range = (hist_range[0] - 1e-6, hist_range[1] + 1e-6)

    axes[0].hist(
        samples,
        bins=bins,
        range=(float(hist_range[0]), float(hist_range[1])),
        density=True,
        color="#0f5c78",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    axes[0].axvline(0.0, color="#777777", linewidth=0.9, linestyle="--")
    axes[0].set_title("Aggregate Frequency-Deviation PDF")
    axes[0].set_xlabel("Frequency deviation [Hz]")
    axes[0].set_ylabel("Probability density")
    axes[0].grid(True, alpha=0.22)

    xs = np.sort(samples)
    ys = np.arange(1, xs.size + 1, dtype=float) / xs.size
    axes[1].plot(xs, ys, color="#b24c2a", linewidth=1.5)
    axes[1].axvline(0.0, color="#777777", linewidth=0.9, linestyle="--")
    axes[1].set_title("Aggregate Frequency-Deviation CDF")
    axes[1].set_xlabel("Frequency deviation [Hz]")
    axes[1].set_ylabel("Cumulative probability")
    axes[1].grid(True, alpha=0.22)

    text = "\n".join([
        f"N = {samples.size}",
        f"mean = {np.mean(samples):.4f} Hz",
        f"std = {np.std(samples):.4f} Hz",
        f"P95(|f|) = {np.quantile(abs_samples, 0.95):.4f} Hz",
        f"P99(|f|) = {np.quantile(abs_samples, 0.99):.4f} Hz",
        f"share(|f|>0.05) = {np.mean(abs_samples > 0.05):.4%}",
    ])
    axes[1].text(
        0.98,
        0.04,
        text,
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="#cccccc"),
    )

    fig.suptitle("Daily Frequency-Deviation Distribution Across 96 Hot-Start Dispatches", fontsize=14)
    fig.tight_layout()
    fig.savefig(results_dir / "frequency_distribution.png", dpi=220)
    plt.close(fig)


def write_all_curves(results_dir: Path, summary: pd.DataFrame, duration_seconds: int) -> None:
    t_grid, grid = load_grid(summary, duration_seconds)

    mean_curve = np.nanmean(grid, axis=0)
    median_curve = np.nanmedian(grid, axis=0)
    p10 = np.nanpercentile(grid, 10, axis=0)
    p90 = np.nanpercentile(grid, 90, axis=0)
    ymin = np.nanmin(grid, axis=0)
    ymax = np.nanmax(grid, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    cmap = plt.get_cmap("turbo")
    norm = plt.Normalize(summary["hour"].min(), summary["hour"].max())

    for _, row in summary.iterrows():
        color = cmap(norm(row["hour"]))
        df = load_series(Path(row["freq_csv"]))
        axes[0].plot(df["time_s"], df["freq_dev_hz"], color=color, alpha=0.32, linewidth=0.8)

    axes[0].plot(t_grid, mean_curve, color="#111111", linewidth=2.0, label="Mean across dispatches")
    axes[0].set_title("All 96 Dispatch Frequency Curves")
    axes[0].set_ylabel("Frequency deviation [Hz]")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, loc="upper right")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[0], pad=0.01)
    cbar.set_label("Hour of day")

    axes[1].fill_between(t_grid, ymin, ymax, color="#d9e5ec", alpha=0.9, label="Min-Max envelope")
    axes[1].fill_between(t_grid, p10, p90, color="#8fb6c9", alpha=0.95, label="10-90 percentile")
    axes[1].plot(t_grid, median_curve, color="#0f5c78", linewidth=1.8, label="Median")
    axes[1].plot(t_grid, mean_curve, color="#b85c38", linewidth=1.4, label="Mean")
    axes[1].axhline(0.0, color="#666666", linewidth=0.8, linestyle="--")
    axes[1].set_title("Daily Frequency Response Envelope")
    axes[1].set_xlabel("Time within dispatch [s]")
    axes[1].set_ylabel("Frequency deviation [Hz]")
    axes[1].set_xlim(0, duration_seconds - 1)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, ncol=4, loc="upper right")

    fig.tight_layout()
    fig.savefig(results_dir / "frequency_curves_all_96.png", dpi=220)
    plt.close(fig)


def write_curve_heatmap(results_dir: Path, summary: pd.DataFrame, duration_seconds: int) -> None:
    t_grid, grid = load_grid(summary, duration_seconds)
    valid = np.isfinite(grid)
    x = np.broadcast_to(t_grid[None, :], grid.shape)[valid]
    y = grid[valid]

    y_abs = np.abs(y)
    q = float(np.quantile(y_abs, 0.999))
    y_lim = max(0.06, q * 1.05)
    y_bins = np.linspace(-y_lim, y_lim, 240)
    x_bins = np.linspace(float(t_grid[0]), float(t_grid[-1]), 181)

    heat, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])

    fig, ax = plt.subplots(figsize=(15.5, 6.2))
    mesh = ax.pcolormesh(
        xedges,
        yedges,
        heat.T,
        cmap="magma",
        shading="auto",
    )
    ax.axhline(0.0, color="white", linewidth=0.9, linestyle="--", alpha=0.7)
    ax.axhline(0.036, color="#8fd3ff", linewidth=1.0, linestyle="--", alpha=0.9)
    ax.axhline(-0.036, color="#8fd3ff", linewidth=1.0, linestyle="--", alpha=0.9)
    ax.axhline(0.05, color="#9ef0a0", linewidth=1.0, linestyle=":", alpha=0.9)
    ax.axhline(-0.05, color="#9ef0a0", linewidth=1.0, linestyle=":", alpha=0.9)
    ax.set_title("96-Dispatch Frequency-Curve Density Heatmap")
    ax.set_xlabel("Time within dispatch [s]")
    ax.set_ylabel("Frequency deviation [Hz]")
    ax.set_xlim(0, duration_seconds - 1)
    ax.set_ylim(-y_lim, y_lim)
    ax.grid(True, alpha=0.08)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
    cbar.set_label("Sample count in bin")

    text = "\n".join([
        f"dispatches = {len(summary)}",
        f"samples = {int(valid.sum())}",
        f"share(|f|>0.036) = {np.mean(y_abs > 0.036):.2%}",
        f"share(|f|>0.05) = {np.mean(y_abs > 0.05):.2%}",
    ])
    ax.text(
        0.985,
        0.03,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#111111",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="#cccccc"),
    )

    fig.tight_layout()
    fig.savefig(results_dir / "frequency_curve_heatmap.png", dpi=220)
    plt.close(fig)


def write_hour_groups(results_dir: Path, summary: pd.DataFrame, duration_seconds: int) -> None:
    colors = ["#0f5c78", "#2a7f3f", "#b85c38", "#7a4ba0"]
    global_min = float(summary["min_hz"].min())
    global_max = float(summary["max_hz"].max())
    pad = 0.08 * max(abs(global_min), abs(global_max), 1e-4)

    for hour0 in range(0, 24, 6):
        hour1 = hour0 + 5
        fig, axes = plt.subplots(3, 2, figsize=(16, 13), sharex=True, sharey=True)
        for offset, hour in enumerate(range(hour0, hour1 + 1)):
            ax = axes.flat[offset]
            hour_rows = summary[summary["hour"] == hour].sort_values("dispatch")
            for _, row in hour_rows.iterrows():
                dispatch_id = int(row["dispatch"])
                df = load_series(Path(row["freq_csv"]))
                ax.plot(
                    df["time_s"],
                    df["freq_dev_hz"],
                    color=colors[dispatch_id % len(colors)],
                    linewidth=1.2,
                    label=f"d{dispatch_id}",
                )
            ax.axhline(0.0, color="#666666", linewidth=0.7, linestyle="--")
            ax.set_title(f"Hour {hour:02d}")
            ax.grid(True, alpha=0.22)
            ax.set_xlim(0, duration_seconds - 1)
            ax.set_ylim(global_min - pad, global_max + pad)
            if offset == 0 and len(hour_rows) > 0:
                ax.legend(frameon=False, ncol=4, fontsize=9, loc="upper right")

        for ax in axes[-1, :]:
            ax.set_xlabel("Time [s]")
        for ax in axes[:, 0]:
            ax.set_ylabel("Freq dev [Hz]")

        fig.suptitle(f"Frequency Curves for Hours {hour0:02d}-{hour1:02d}", fontsize=15)
        fig.tight_layout()
        fig.savefig(results_dir / f"frequency_curves_h{hour0:02d}_to_h{hour1:02d}.png", dpi=220)
        plt.close(fig)


def write_96_panel_grid(results_dir: Path, summary: pd.DataFrame, duration_seconds: int) -> None:
    n_hours = 24
    n_dispatch = 4

    global_min = float(summary["min_hz"].min())
    global_max = float(summary["max_hz"].max())
    pad = 0.08 * max(abs(global_min), abs(global_max), 1e-4)

    fig, axes = plt.subplots(
        n_hours,
        n_dispatch,
        figsize=(18, 40),
        sharex=True,
        sharey=True,
    )

    for hour in range(n_hours):
        for dispatch in range(n_dispatch):
            ax = axes[hour, dispatch]
            row = summary[(summary["hour"] == hour) & (summary["dispatch"] == dispatch)]
            if row.empty:
                ax.axis("off")
                continue

            item = row.iloc[0]
            df = load_series(Path(item["freq_csv"]))
            ax.plot(df["time_s"], df["freq_dev_hz"], color="#0f5c78", linewidth=1.0)
            ax.axhline(0.0, color="#666666", linewidth=0.6, linestyle="--")
            ax.axhline(0.036, color="#9aa0a6", linewidth=0.5, linestyle=":")
            ax.axhline(-0.036, color="#9aa0a6", linewidth=0.5, linestyle=":")
            ax.set_xlim(0, duration_seconds - 1)
            ax.set_ylim(global_min - pad, global_max + pad)
            ax.grid(True, alpha=0.18)
            ax.set_title(f"h{hour}d{dispatch}", fontsize=8, pad=2.5)

    for ax in axes[-1, :]:
        ax.set_xlabel("t [s]")
    for ax in axes[:, 0]:
        ax.set_ylabel("f [Hz]")

    fig.suptitle("Frequency Curves for All 96 Dispatch Intervals (24x4 grid)", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.992))
    fig.savefig(results_dir / "frequency_curves_96_panels.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary = load_summary(args.results_dir, args.summary_name)
    samples = collect_samples(summary)

    write_threshold_stats(args.results_dir, samples)
    write_distribution(args.results_dir, samples, args.bins)
    write_all_curves(args.results_dir, summary, args.duration_seconds)
    write_curve_heatmap(args.results_dir, summary, args.duration_seconds)
    write_hour_groups(args.results_dir, summary, args.duration_seconds)
    write_96_panel_grid(args.results_dir, summary, args.duration_seconds)

    print(f"summary_rows={len(summary)}")
    print(f"frequency_distribution_png={args.results_dir / 'frequency_distribution.png'}")
    print(f"frequency_curves_all_96_png={args.results_dir / 'frequency_curves_all_96.png'}")


if __name__ == "__main__":
    main()
