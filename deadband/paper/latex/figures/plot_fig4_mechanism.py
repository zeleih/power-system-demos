"""Fig. 4: Mechanism comparison for one representative window (h11d2-h11d3).

Two columns, same window, same inputs, nominal inertia:
  left  = uniform deadband 36/36/36 mHz (baseline)  -> engage/disengage cycle
  right = heterogeneous 36/25/15 mHz (selected)     -> layered release

Three rows:
  (a)/(b) frequency deviation with the class deadband thresholds
  (c)/(d) aggregate droop breakdown (governor / wind+PV / storage)
  (e)/(f) engaged-unit fraction per converter class (raw 1-s + 25-s mean)

Data: results/inertia_multiscenario_screen_20260503/H100_{uniform,best}_
h11d2_h11d3_trace.csv.  The H100 (nominal-inertia) runs use the same
configuration as the phase-1 day-long study; the best-case trace is
bit-identical to the dedicated run in results/fig4_mechanism_best_h11d2_h11d3.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fig_style import COLORS, apply_style, polish_axes, save_figure

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
N_PVD = 22
N_ESD = 2


def smooth(y: np.ndarray, window: int = 25) -> np.ndarray:
    return pd.Series(y).rolling(window, center=True, min_periods=1).mean().to_numpy()


def panel_label(ax, panel: str, title: str, color=None):
    ax.text(0.008, 0.97, rf"$\bf{{{panel}}}$  {title}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7.0, color=color or COLORS["ink"])


def threshold_lines(ax, thresholds):
    """Amber threshold pairs; linestyle encodes the class."""
    for thr, ls in thresholds:
        ax.axhline(thr, color=COLORS["amber"], linestyle=ls,
                   linewidth=0.6, alpha=0.85, zorder=1)
        ax.axhline(-thr, color=COLORS["amber"], linestyle=ls,
                   linewidth=0.6, alpha=0.85, zorder=1)
    ax.axhline(0, color=COLORS["axis"], linewidth=0.5, alpha=0.6, zorder=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    base = ROOT / "results" / "inertia_multiscenario_screen_20260503"
    parser.add_argument("--uniform-csv", type=Path,
                        default=base / "H100_uniform_h11d2_h11d3_trace.csv")
    parser.add_argument("--best-csv", type=Path,
                        default=base / "H100_best_h11d2_h11d3_trace.csv")
    parser.add_argument("--out", type=Path, default=HERE / "fig4_mechanism.pdf")
    args = parser.parse_args()

    apply_style()
    du = pd.read_csv(args.uniform_csv)
    db = pd.read_csv(args.best_csv)

    fig, axes = plt.subplots(
        3, 2, figsize=(7.16, 3.4), sharex=True, sharey="row",
        gridspec_kw={"hspace": 0.13, "wspace": 0.06,
                     "height_ratios": [1.15, 1.0, 1.0]})

    col_meta = [
        (du, "uniform 36/36/36 mHz", [(0.036, "-")]),
        (db, "heterogeneous 36/25/15 mHz",
         [(0.036, "-"), (0.025, "--"), (0.015, ":")]),
    ]
    panels = [["(a)", "(b)"], ["(c)", "(d)"], ["(e)", "(f)"]]

    for j, (df, title, thrs) in enumerate(col_meta):
        t = df["time_s"].to_numpy()

        # ---- row 1: frequency deviation -------------------------------
        ax = axes[0, j]
        f = df["freq_dev_hz"].to_numpy()
        ax.axhspan(-0.036, 0.036, color=COLORS["band"], alpha=0.55,
                   zorder=0, linewidth=0)
        threshold_lines(ax, thrs)
        ax.plot(t, f, color=COLORS["blue"], alpha=0.18, linewidth=0.35,
                rasterized=True)
        ax.plot(t, smooth(f), color=COLORS["blue"], linewidth=1.15, zorder=4)
        panel_label(ax, panels[0][j], title, COLORS["blue"])
        ax.set_ylim(-0.068, 0.068)

        # ---- row 2: droop breakdown ------------------------------------
        ax = axes[1, j]
        ax.axhline(0, color=COLORS["axis"], linewidth=0.5, alpha=0.6)
        for colname, c, lab in (
            ("gov_droop_sum", COLORS["green"], "governor"),
            ("pvd_droop_sum", COLORS["blue"], "wind+PV"),
            ("esd_droop_sum", COLORS["amber"], "storage"),
        ):
            y = df[colname].to_numpy()
            ax.plot(t, smooth(y), color=c, linewidth=1.0,
                    label=lab if j == 0 else None)
        panel_label(ax, panels[1][j], "droop breakdown (25-s mean)")
        if j == 0:
            ax.legend(loc="lower center", ncol=3, borderaxespad=0.2,
                      handlelength=1.2, fontsize=6.4)

        # ---- row 3: engaged-unit fraction ------------------------------
        ax = axes[2, j]
        # storage is drawn dashed on top: in the uniform case the two
        # classes switch in the same seconds and the traces coincide.
        for colname, n, c, ls, lab in (
            ("pvd_active_count", N_PVD, COLORS["blue"], "-", "wind+PV"),
            ("esd_active_count", N_ESD, COLORS["amber"], (0, (4, 2)), "storage"),
        ):
            y = 100.0 * df[colname].to_numpy() / n
            ax.plot(t, y, color=c, alpha=0.20, linewidth=0.35,
                    rasterized=True, drawstyle="steps-post")
            ax.plot(t, smooth(y), color=c, linewidth=1.05, linestyle=ls,
                    label=lab if j == 0 else None)
        no_der = 100.0 * ((df["pvd_active_count"] == 0)
                          & (df["esd_active_count"] == 0)).mean()
        ax.text(0.985, 0.10, f"no DER support: {no_der:.1f}% of time",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.4, color=COLORS["ink"])
        panel_label(ax, panels[2][j], "engaged units (raw + 25-s mean)")
        ax.set_ylim(-4, 119)
        ax.set_yticks([0, 25, 50, 75, 100])
        if j == 0:
            ax.legend(loc="upper right", ncol=2, borderaxespad=0.2,
                      handlelength=1.2, fontsize=6.4)
        ax.set_xlabel("Time within interval (s)")

    axes[0, 0].set_ylabel(r"$\Delta f$ (Hz)")
    axes[1, 0].set_ylabel(r"$\Delta P$ (p.u.)")
    axes[2, 0].set_ylabel("engaged (%)")
    axes[0, 0].set_xlim(0, 900)

    for ax in axes.flat:
        polish_axes(ax)
    fig.align_ylabels(axes[:, 0])
    save_figure(fig, args.out)


if __name__ == "__main__":
    main()
