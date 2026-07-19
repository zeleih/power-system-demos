"""Fig: top-6 candidates in two coupled planes (single-column, two panels).

(a) shoulder intensity EM36 vs tail risk share(|Df|>0.05 Hz), with baseline;
(b) storage throughput E_ESD vs the same tail risk, colored by wind deadband.

Same candidates, same y-axis in both panels; labels give final full-day rank.
Data: phase1_full_day_ranked.csv. Output: fig5_pareto.pdf/.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fig_style import COLORS, apply_style, polish_axes, save_figure

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent

WIND_SHADES = {0.036: "#9DB9D6", 0.045: "#5B87B0", 0.055: "#2E5F8A", 0.065: "#173A5C"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-csv", type=Path,
        default=ROOT / "results"
                / "phase1_full_day_tip100_alpha098_disable_pvd_agc_disable_esd_agc_kp0p1_ki0p002"
                / "phase1_full_day_ranked.csv",
    )
    parser.add_argument("--out", type=Path, default=HERE / "fig5_pareto.pdf")
    parser.add_argument("--baseline-edge-mass", type=float, default=0.1690)
    parser.add_argument("--baseline-share-gt-0p05", type=float, default=0.0287)
    args = parser.parse_args()

    df = pd.read_csv(args.summary_csv).sort_values("rank").reset_index(drop=True)
    y = 100 * df["share_abs_gt_0p05"].values
    ranks = df["rank"].astype(int).values
    best = ranks == 1

    apply_style()
    fig, (axa, axb) = plt.subplots(
        2, 1, figsize=(3.5, 4.0), sharey=True,
        gridspec_kw={"hspace": 0.28})

    # ----- (a) EM36 vs tail risk -------------------------------------------
    xa = 100 * df["edge_mass_36"].values
    xb_base = 100 * args.baseline_edge_mass
    yb_base = 100 * args.baseline_share_gt_0p05
    axa.scatter(xb_base, yb_base, marker="X", s=80, color=COLORS["red"],
                edgecolors="white", linewidths=0.6, zorder=6, label="baseline")
    axa.annotate("baseline", (xb_base, yb_base), xytext=(-7, -2),
                 textcoords="offset points", ha="right", va="center",
                 fontsize=6.8, color=COLORS["red"])
    axa.scatter(xa[~best], y[~best], s=46, color=COLORS["blue"],
                edgecolors="white", linewidths=0.55, zorder=5,
                label="top candidates")
    axa.scatter(xa[best], y[best], s=110, marker="*", color=COLORS["amber"],
                edgecolors=COLORS["ink"], linewidths=0.45, zorder=7,
                label="selected best (#1)")
    offs_a = {1: (9, -6), 2: (9, 1), 3: (9, 1), 4: (-9, 0), 5: (9, -2), 6: (-9, 3)}
    for xi, yi, r in zip(xa, y, ranks):
        dx, dy = offs_a.get(int(r), (8, 4))
        axa.annotate(f"#{r}", xy=(xi, yi), xytext=(dx, dy),
                     textcoords="offset points", fontsize=6.6,
                     color=COLORS["ink"], ha="left" if dx >= 0 else "right",
                     va="center",
                     arrowprops=dict(arrowstyle="-", color=COLORS["muted"],
                                     linewidth=0.45, shrinkA=1.5, shrinkB=2.5,
                                     alpha=0.80))
    axa.set_xlabel(r"$\mathrm{EM}_{36}$ (% of samples)")
    axa.set_ylabel(r"$\mathrm{share}(|\Delta f|\!>\!0.05\,\mathrm{Hz})$ (%)")
    axa.set_xlim(min(xa.min(), xb_base) - 0.6, max(xa.max(), xb_base) + 0.6)
    axa.set_ylim(min(y.min(), yb_base) - 0.18, max(y.max(), yb_base) + 0.33)
    axa.legend(loc="upper left", frameon=False, handletextpad=0.4,
               labelspacing=0.25, fontsize=6.6)
    axa.text(0.012, 0.03, "(a)", transform=axa.transAxes, fontsize=8,
             fontweight="bold", ha="left", va="bottom", color=COLORS["ink"])

    # ----- (b) storage throughput vs tail risk ------------------------------
    xe = df["esd_throughput"].values
    for wdb in sorted(WIND_SHADES):
        m = np.isclose(df["wind_deadband_hz"].values, wdb)
        if not m.any():
            continue
        axb.scatter(xe[m & ~best], y[m & ~best], s=46,
                    color=WIND_SHADES[wdb], edgecolors="white",
                    linewidths=0.55, zorder=5,
                    label=rf"$d_{{\mathrm{{b,Wind}}}}$ = {wdb*1000:.0f} mHz")
        if (m & best).any():
            axb.scatter(xe[m & best], y[m & best], s=110, marker="*",
                        color=COLORS["amber"], edgecolors=COLORS["ink"],
                        linewidths=0.45, zorder=7)
    axb.axhline(yb_base, color=COLORS["red"], linestyle="--", linewidth=0.8,
                alpha=0.8)
    axb.text(0.985, yb_base - 0.05, "baseline tail share",
             transform=axb.get_yaxis_transform(), ha="right", va="top",
             fontsize=6.4, color=COLORS["red"])
    offs_b = {1: (-9, -4), 2: (9, 1), 3: (9, 1), 4: (-9, 2), 5: (9, -3), 6: (-9, 4)}
    for xi, yi, r in zip(xe, y, ranks):
        dx, dy = offs_b.get(int(r), (8, 4))
        axb.annotate(f"#{r}", xy=(xi, yi), xytext=(dx, dy),
                     textcoords="offset points", fontsize=6.6,
                     color=COLORS["ink"], ha="left" if dx >= 0 else "right",
                     va="center",
                     arrowprops=dict(arrowstyle="-", color=COLORS["muted"],
                                     linewidth=0.45, shrinkA=1.5, shrinkB=2.5,
                                     alpha=0.80))
    axb.set_xlabel(r"Storage throughput $E_{\mathrm{ESD}}$ (p.u.$\cdot$s)")
    axb.set_ylabel(r"$\mathrm{share}(|\Delta f|\!>\!0.05\,\mathrm{Hz})$ (%)")
    axb.set_xlim(xe.min() - 1.2, xe.max() + 1.2)
    axb.legend(loc="upper left", bbox_to_anchor=(0.0, 0.84), frameon=False,
               handletextpad=0.4, labelspacing=0.25, fontsize=6.4)
    axb.text(0.012, 0.03, "(b)", transform=axb.transAxes, fontsize=8,
             fontweight="bold", ha="left", va="bottom", color=COLORS["ink"])

    for ax in (axa, axb):
        polish_axes(ax)
    fig.align_ylabels((axa, axb))
    save_figure(fig, args.out)


if __name__ == "__main__":
    main()
