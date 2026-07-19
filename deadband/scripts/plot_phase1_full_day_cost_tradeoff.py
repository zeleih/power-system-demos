#!/usr/bin/env python3
"""
Plot safety-cost tradeoff for phase-1 full-day candidates.

The current baseline may not yet have day-level cost columns, so the script
uses baseline frequency metrics as reference lines and plots the candidate
points in cost-frequency space.
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
    parser.add_argument("--ranked-csv", type=Path, required=True)
    parser.add_argument("--baseline-stats-csv", type=Path, required=True)
    parser.add_argument("--out-png", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    return parser.parse_args()


def pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    mask = np.ones(x.size, dtype=bool)
    for i in range(x.size):
        dominated = (x <= x[i]) & (y <= y[i]) & ((x < x[i]) | (y < y[i]))
        dominated[i] = False
        if np.any(dominated):
            mask[i] = False
    return mask


def annotate_points(ax: plt.Axes, df: pd.DataFrame, xcol: str, ycol: str, *, xscale: float = 1.0, yscale: float = 1.0) -> None:
    for _, row in df.iterrows():
        ax.annotate(
            f"#{int(row['rank'])}",
            (float(row[xcol]) * xscale, float(row[ycol]) * yscale),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
            color="#1d1d1d",
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="#d4d4d4", alpha=0.85),
        )


def main() -> None:
    args = parse_args()
    ranked = pd.read_csv(args.ranked_csv)
    ranked = ranked.sort_values("rank").reset_index(drop=True)
    baseline = pd.read_csv(args.baseline_stats_csv).iloc[0].to_dict()

    pareto = ranked[pareto_mask(
        ranked["esd_throughput"].to_numpy(dtype=float),
        ranked["share_abs_gt_0p05"].to_numpy(dtype=float),
    )].copy()
    pareto = pareto.sort_values(["esd_throughput", "share_abs_gt_0p05"]).reset_index(drop=True)

    export = ranked[[
        "rank",
        "combo_id",
        "wind_deadband_hz",
        "solar_deadband_hz",
        "esd_deadband_hz",
        "mean_abs_hz",
        "share_abs_gt_0p05",
        "max_abs_hz",
        "esd_throughput",
        "wind_effort",
        "pv_effort",
        "gov_droop_effort",
    ]].copy()
    export["pareto_esd_tail"] = export["combo_id"].isin(set(pareto["combo_id"])).astype(int)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(args.out_csv, index=False)

    color = ranked["max_abs_hz"].to_numpy(dtype=float)
    scatter_style = dict(
        c=color,
        cmap="viridis",
        s=88,
        alpha=0.94,
        edgecolors="white",
        linewidths=0.8,
    )

    fig, axes = plt.subplots(1, 2, figsize=(15.6, 6.6))

    x_esd = ranked["esd_throughput"].to_numpy(dtype=float)
    y_tail = 100.0 * ranked["share_abs_gt_0p05"].to_numpy(dtype=float)
    y_mean = 1000.0 * ranked["mean_abs_hz"].to_numpy(dtype=float)

    sc0 = axes[0].scatter(x_esd, y_tail, **scatter_style)
    axes[0].plot(
        pareto["esd_throughput"].to_numpy(dtype=float),
        100.0 * pareto["share_abs_gt_0p05"].to_numpy(dtype=float),
        color="#c44e52",
        linewidth=1.8,
        marker="o",
        markersize=4,
        label="Pareto front",
        zorder=4,
    )
    annotate_points(axes[0], ranked, "esd_throughput", "share_abs_gt_0p05", yscale=100.0)
    axes[0].axhline(
        100.0 * float(baseline["share_abs_gt_0p05"]),
        color="#8c564b",
        linestyle="--",
        linewidth=1.4,
        label="Baseline tail-risk level",
    )
    best = ranked.iloc[0]
    axes[0].scatter(
        [float(best["esd_throughput"])],
        [100.0 * float(best["share_abs_gt_0p05"])],
        marker="*",
        s=240,
        color="#ffcc33",
        edgecolors="#333333",
        linewidths=1.0,
        zorder=5,
        label="Best candidate",
    )
    axes[0].set_title("Storage Motion vs Tail Risk")
    axes[0].set_xlabel("ESD throughput [model-unit * s]")
    axes[0].set_ylabel("share(|f| > 0.05 Hz) [%]")
    axes[0].grid(True, alpha=0.22)
    axes[0].legend(frameon=False, loc="upper right")

    sc1 = axes[1].scatter(x_esd, y_mean, **scatter_style)
    annotate_points(axes[1], ranked, "esd_throughput", "mean_abs_hz", yscale=1000.0)
    axes[1].axhline(
        1000.0 * float(baseline["mean_abs_hz"]),
        color="#8c564b",
        linestyle="--",
        linewidth=1.4,
        label="Baseline mean |f| level",
    )
    axes[1].scatter(
        [float(best["esd_throughput"])],
        [1000.0 * float(best["mean_abs_hz"])],
        marker="*",
        s=240,
        color="#ffcc33",
        edgecolors="#333333",
        linewidths=1.0,
        zorder=5,
        label="Best candidate",
    )
    axes[1].set_title("Storage Motion vs Frequency Quality")
    axes[1].set_xlabel("ESD throughput [model-unit * s]")
    axes[1].set_ylabel("mean |f| [mHz]")
    axes[1].grid(True, alpha=0.22)
    axes[1].legend(frameon=False, loc="upper right")

    cbar = fig.colorbar(sc1, ax=axes, pad=0.015, shrink=0.96)
    cbar.set_label("max_abs_hz [Hz]")

    fig.suptitle(
        "Phase-1 Full-Day Cost Tradeoff (6 accepted candidates, baseline as frequency reference lines)",
        fontsize=14,
    )
    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
