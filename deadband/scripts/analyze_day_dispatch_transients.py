#!/usr/bin/env python3
"""
Analyze daily dispatch TDS runs for large initial deviations and slow recovery.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def resample_series(freq_csv: Path, duration_seconds: int) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(freq_csv)
    df = df.drop_duplicates(subset="time_s", keep="last").sort_values("time_s")
    t = df["time_s"].to_numpy(dtype=float)
    y = df["freq_dev_hz"].to_numpy(dtype=float)
    grid = np.arange(duration_seconds, dtype=float)
    out = np.interp(grid, t, y)
    return grid, out


def find_settle_time(y: np.ndarray, threshold_hz: float, hold_seconds: int) -> int:
    mask = np.abs(y) <= threshold_hz
    limit = len(y) - hold_seconds + 1
    for i in range(max(0, limit)):
        if mask[i:i + hold_seconds].all():
            return i
    return len(y)


def compute_metrics(
    results_dir: Path,
    duration_seconds: int,
    early_window_seconds: int,
    settle_threshold_hz: float,
    settle_hold_seconds: int,
) -> pd.DataFrame:
    summary = pd.read_csv(results_dir / "daily_summary.csv")
    rows: list[dict] = []

    for _, row in summary.iterrows():
        label = str(row["label"])
        t_grid, y = resample_series(Path(row["freq_csv"]), duration_seconds)
        early_end = min(len(y), early_window_seconds + 1)
        early_slice = y[:early_end]
        early_idx = int(np.argmax(np.abs(early_slice)))
        settle_time = find_settle_time(y, settle_threshold_hz, settle_hold_seconds)

        rows.append({
            "label": label,
            "hour": int(row["hour"]),
            "dispatch": int(row["dispatch"]),
            "early_peak_hz": float(np.max(np.abs(early_slice))),
            "early_peak_t_s": int(t_grid[early_idx]),
            "settle_time_s": int(settle_time),
            "final_abs_hz": float(abs(y[-1])),
            "abs_mean_hz": float(np.mean(np.abs(y))),
            "max_abs_hz": float(np.max(np.abs(y))),
            "freq_csv": row["freq_csv"],
        })

    return pd.DataFrame(rows).sort_values(["hour", "dispatch"]).reset_index(drop=True)


def make_scatter(fig_path: Path, metrics: pd.DataFrame, settle_threshold_hz: float) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    size = 40 + 3500 * metrics["abs_mean_hz"]
    sc = ax.scatter(
        metrics["early_peak_hz"],
        metrics["settle_time_s"],
        c=metrics["hour"],
        s=size,
        cmap="turbo",
        alpha=0.82,
        edgecolors="white",
        linewidths=0.5,
    )

    top_labels = pd.concat([
        metrics.nlargest(8, "early_peak_hz"),
        metrics.nlargest(8, "settle_time_s"),
    ]).drop_duplicates("label")
    for _, row in top_labels.iterrows():
        ax.annotate(
            row["label"],
            (row["early_peak_hz"], row["settle_time_s"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Hour of day")

    ax.set_title(
        f"Daily Dispatch Initial Peak vs Recovery Time (settle within +/-{settle_threshold_hz:.3f} Hz)"
    )
    ax.set_xlabel("Peak |frequency deviation| in first 60 s [Hz]")
    ax.set_ylabel("First settle time [s] with 30 s hold")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def make_grid_plot(
    fig_path: Path,
    metrics: pd.DataFrame,
    duration_seconds: int,
    title: str,
    settle_threshold_hz: float,
) -> None:
    n = len(metrics)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.6 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    global_max = 0.0
    series_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for _, row in metrics.iterrows():
        t_grid, y = resample_series(Path(row["freq_csv"]), duration_seconds)
        series_cache[row["label"]] = (t_grid, y)
        global_max = max(global_max, float(np.max(np.abs(y))))

    ylim = max(0.05, min(0.38, global_max * 1.08))

    for ax, (_, row) in zip(axes.flat, metrics.iterrows()):
        t_grid, y = series_cache[row["label"]]
        ax.plot(t_grid, y, color="#0f5c78", linewidth=1.3)
        ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle="--")
        ax.axhline(settle_threshold_hz, color="#b85c38", linewidth=0.7, linestyle=":")
        ax.axhline(-settle_threshold_hz, color="#b85c38", linewidth=0.7, linestyle=":")
        ax.set_title(
            f"{row['label']} | peak60={row['early_peak_hz']:.3f} Hz | settle={int(row['settle_time_s'])} s",
            fontsize=9,
        )
        ax.grid(True, alpha=0.22)
        ax.set_xlim(0, duration_seconds - 1)
        ax.set_ylim(-ylim, ylim)

    for ax in axes.flat[n:]:
        ax.axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [s]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Freq dev [Hz]")

    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/Applications/openandes/demo/demo/deadband/results/day96_agc4_kp0p05_ki0p0625"),
    )
    parser.add_argument("--duration-seconds", type=int, default=900)
    parser.add_argument("--early-window-seconds", type=int, default=60)
    parser.add_argument("--settle-threshold-hz", type=float, default=0.02)
    parser.add_argument("--settle-hold-seconds", type=int, default=30)
    parser.add_argument("--top-n", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metrics = compute_metrics(
        results_dir=args.results_dir,
        duration_seconds=args.duration_seconds,
        early_window_seconds=args.early_window_seconds,
        settle_threshold_hz=args.settle_threshold_hz,
        settle_hold_seconds=args.settle_hold_seconds,
    )

    metrics_csv = args.results_dir / "transient_metrics.csv"
    metrics.to_csv(metrics_csv, index=False)

    top_early = metrics.nlargest(args.top_n, "early_peak_hz").copy()
    top_slow = metrics.nlargest(args.top_n, "settle_time_s").copy()
    top_early_csv = args.results_dir / "top_initial_peaks.csv"
    top_slow_csv = args.results_dir / "top_slowest_recovery.csv"
    top_early.to_csv(top_early_csv, index=False)
    top_slow.to_csv(top_slow_csv, index=False)

    scatter_png = args.results_dir / "transient_peak_vs_settle_scatter.png"
    top_early_png = args.results_dir / "top_initial_peaks_grid.png"
    top_slow_png = args.results_dir / "top_slowest_recovery_grid.png"

    make_scatter(scatter_png, metrics, args.settle_threshold_hz)
    make_grid_plot(
        top_early_png,
        top_early,
        args.duration_seconds,
        title="Dispatches with the Largest Initial Frequency Excursions",
        settle_threshold_hz=args.settle_threshold_hz,
    )
    make_grid_plot(
        top_slow_png,
        top_slow,
        args.duration_seconds,
        title="Dispatches with the Slowest Recovery into +/-0.02 Hz",
        settle_threshold_hz=args.settle_threshold_hz,
    )

    print(f"metrics_csv={metrics_csv}")
    print(f"top_early_csv={top_early_csv}")
    print(f"top_slow_csv={top_slow_csv}")
    print(f"scatter_png={scatter_png}")
    print(f"top_early_png={top_early_png}")
    print(f"top_slow_png={top_slow_png}")
    print("top_early_labels=" + ",".join(top_early["label"].tolist()))
    print("top_slow_labels=" + ",".join(top_slow["label"].tolist()))


if __name__ == "__main__":
    main()
