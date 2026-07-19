#!/usr/bin/env python3
"""
Plot the aggregate frequency-deviation distribution for one day of dispatch TDS runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_samples(results_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    summary = pd.read_csv(results_dir / "daily_summary.csv")
    samples: list[np.ndarray] = []

    for _, row in summary.iterrows():
        if int(row.get("success", 0)) != 1:
            continue
        freq_csv = Path(row["freq_csv"])
        df = pd.read_csv(freq_csv)
        y = df["freq_dev_hz"].to_numpy(dtype=float)
        if y.size:
            samples.append(y)

    if not samples:
        raise RuntimeError(f"No successful frequency samples found in {results_dir}")

    return summary, np.concatenate(samples)


def compute_stats(samples: np.ndarray) -> pd.DataFrame:
    abs_samples = np.abs(samples)
    stats = {
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
    }
    return pd.DataFrame([stats])


def make_plot(fig_path: Path, samples: np.ndarray, bins: int) -> None:
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

    stats_text = "\n".join([
        f"N = {samples.size}",
        f"mean = {np.mean(samples):.4f} Hz",
        f"std = {np.std(samples):.4f} Hz",
        f"P95(|f|) = {np.quantile(np.abs(samples), 0.95):.4f} Hz",
        f"P99(|f|) = {np.quantile(np.abs(samples), 0.99):.4f} Hz",
        f"max(|f|) = {np.max(np.abs(samples)):.4f} Hz",
    ])
    axes[1].text(
        0.98,
        0.04,
        stats_text,
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="#cccccc"),
    )

    fig.suptitle("Daily Frequency-Deviation Distribution Across 96 Dispatches", fontsize=14)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    _, samples = load_samples(args.results_dir)
    stats = compute_stats(samples)

    stats_csv = args.results_dir / "frequency_distribution_stats.csv"
    fig_path = args.results_dir / "frequency_distribution.png"

    stats.to_csv(stats_csv, index=False)
    make_plot(fig_path, samples, args.bins)

    print(f"stats_csv={stats_csv}")
    print(f"fig_png={fig_path}")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
