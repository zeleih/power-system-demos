#!/usr/bin/env python3
"""
Plot the eligible phase-1 deadband coarse-sweep candidates on scatter/Pareto charts.
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
    parser.add_argument("--out-png", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=6)
    return parser.parse_args()


def pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    mask = np.ones(x.size, dtype=bool)
    for i in range(x.size):
        dominated = (x <= x[i]) & (y <= y[i]) & ((x < x[i]) | (y < y[i]))
        dominated[i] = False
        if np.any(dominated):
            mask[i] = False
    return mask


def annotate_topk(ax: plt.Axes, df: pd.DataFrame, xcol: str, ycol: str) -> None:
    for _, row in df.iterrows():
        ax.annotate(
            f"#{int(row['rank'])}",
            (100.0 * float(row[xcol]), 100.0 * float(row[ycol])),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
            color="#1d1d1d",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#d4d4d4", alpha=0.85),
        )


def main() -> None:
    args = parse_args()
    ranked = pd.read_csv(args.ranked_csv)
    eligible = ranked[ranked["eligible"] == 1].copy().reset_index(drop=True)
    if eligible.empty:
        raise RuntimeError(f"No eligible combos found in {args.ranked_csv}")

    pareto = eligible[pareto_mask(
        eligible["edge_mass_36"].to_numpy(dtype=float),
        eligible["share_abs_gt_0p05"].to_numpy(dtype=float),
    )].copy()
    pareto = pareto.sort_values(["edge_mass_36", "share_abs_gt_0p05"]).reset_index(drop=True)
    pareto.to_csv(args.out_csv, index=False)

    topk = eligible.sort_values("rank").head(int(args.top_k)).copy()

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.4))
    color = eligible["max_abs_hz"].to_numpy(dtype=float)
    scatter_style = dict(
        c=color,
        cmap="viridis",
        s=62,
        alpha=0.92,
        edgecolors="white",
        linewidths=0.6,
    )

    x1 = 100.0 * eligible["edge_mass_36"].to_numpy(dtype=float)
    y = 100.0 * eligible["share_abs_gt_0p05"].to_numpy(dtype=float)
    x2 = 100.0 * eligible["edge_asymmetry_36"].to_numpy(dtype=float)

    sc0 = axes[0].scatter(x1, y, **scatter_style)
    axes[0].plot(
        100.0 * pareto["edge_mass_36"].to_numpy(dtype=float),
        100.0 * pareto["share_abs_gt_0p05"].to_numpy(dtype=float),
        color="#c44e52",
        linewidth=1.8,
        marker="o",
        markersize=4,
        label="Pareto front",
        zorder=4,
    )
    axes[0].scatter(
        100.0 * topk["edge_mass_36"].to_numpy(dtype=float),
        100.0 * topk["share_abs_gt_0p05"].to_numpy(dtype=float),
        marker="*",
        s=185,
        color="#ffcc33",
        edgecolors="#333333",
        linewidths=0.9,
        label="Top-6 by coarse ranking",
        zorder=5,
    )
    annotate_topk(axes[0], topk, "edge_mass_36", "share_abs_gt_0p05")
    axes[0].set_title("Pareto Plane: 36 mHz Shoulder vs Tail Risk")
    axes[0].set_xlabel(r"$\mathrm{EM}_{36}$ [% of samples]")
    axes[0].set_ylabel(r"share($|\Delta f| > 0.05$ Hz) [%]")
    axes[0].grid(True, alpha=0.22)
    axes[0].legend(frameon=False, loc="upper right")

    sc1 = axes[1].scatter(x2, y, **scatter_style)
    axes[1].scatter(
        100.0 * topk["edge_asymmetry_36"].to_numpy(dtype=float),
        100.0 * topk["share_abs_gt_0p05"].to_numpy(dtype=float),
        marker="*",
        s=185,
        color="#ffcc33",
        edgecolors="#333333",
        linewidths=0.9,
        zorder=5,
    )
    annotate_topk(axes[1], topk, "edge_asymmetry_36", "share_abs_gt_0p05")
    axes[1].set_title("Scatter: 36 mHz Asymmetry vs Tail Risk")
    axes[1].set_xlabel(r"$\mathrm{EA}_{36}$ [% of samples]")
    axes[1].set_ylabel(r"share($|\Delta f| > 0.05$ Hz) [%]")
    axes[1].grid(True, alpha=0.22)

    cbar = fig.colorbar(sc1, ax=axes, pad=0.015, shrink=0.96)
    cbar.set_label("max_abs_hz [Hz]")

    fig.suptitle(
        f"Phase-1 Eligible Deadband Combos ({len(eligible)} candidates, top-{int(args.top_k)} highlighted)",
        fontsize=14,
    )
    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=220)
    pdf_path = args.out_png.with_suffix(".pdf")
    fig.savefig(pdf_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
