"""Fig. 6: Baseline vs. best-candidate day-long frequency distribution.

Main result figure. Overlays the baseline (uniform +/-0.036 Hz) and the
best candidate (0.036/0.025/0.015 Hz) day-long frequency densities, with
+/-0.036 Hz and +/-0.05 Hz reference lines and an inline tail-reduction
callout. Output: fig6_dist_compare.pdf/.png
"""
import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fig_style import (COLORS, add_deadband_band, add_threshold_lines,
                       apply_style, polish_axes, save_figure,
                       smooth_histogram)

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent


def load_full_day(results_dir: Path) -> np.ndarray:
    files = sorted(glob.glob(str(results_dir / "h*d*_frequency.csv")))
    if not files:
        raise FileNotFoundError(f"No h*d*_frequency.csv files found under {results_dir}")
    chunks = [pd.read_csv(f)["freq_dev_hz"].values for f in files]
    return np.concatenate(chunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-dir", type=Path,
        default=ROOT / "results"
                / "phase1_baseline_full_day_tip100_alpha098_disable_pvd_agc_disable_esd_agc_kp0p1_ki0p002"
                / "wind036_pv036_esd036",
    )
    parser.add_argument(
        "--best-dir", type=Path,
        default=ROOT / "results"
                / "phase1_full_day_tip100_alpha098_disable_pvd_agc_disable_esd_agc_kp0p1_ki0p002"
                / "wind036_pv025_esd015",
    )
    parser.add_argument("--out", type=Path, default=HERE / "fig6_dist_compare.pdf")
    args = parser.parse_args()

    base = load_full_day(args.baseline_dir)
    best = load_full_day(args.best_dir)

    apply_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    bins = np.linspace(-0.095, 0.095, 191)
    cb, _, sb = smooth_histogram(base, bins, density=True, window=13)
    cw, _, sw = smooth_histogram(best, bins, density=True, window=13)
    ymax = max(sb.max(), sw.max())

    add_deadband_band(ax, half_width=0.036, alpha=0.55)
    ax.fill_between(cb, 0, sb, color=COLORS["red"], alpha=0.14, linewidth=0)
    ax.fill_between(cw, 0, sw, color=COLORS["blue"], alpha=0.16, linewidth=0)
    ax.plot(cb, sb, color=COLORS["red"], linewidth=1.35,
            label=r"baseline (uniform $\pm36$ mHz)")
    ax.plot(cw, sw, color=COLORS["blue"], linewidth=1.45,
            label="best: 36/25/15 mHz")
    add_threshold_lines(ax)

    # Tail-reduction callout: a single short label, no decorative arrow.
    s05_base = np.mean(np.abs(base) > 0.05) * 100
    s05_best = np.mean(np.abs(best) > 0.05) * 100
    pct_red = (s05_base - s05_best) / s05_base * 100
    ax.annotate(r"$|\Delta f|>50$ mHz" + f"\n$-{pct_red:.1f}$%",
                xy=(0.058, np.interp(0.058, cw, sw)),
                xytext=(0.072, 0.32 * ymax),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["blue"],
                                linewidth=0.6, shrinkA=0.5, shrinkB=2,
                                mutation_scale=6),
                fontsize=6.6, color=COLORS["blue"], ha="left", va="center")

    ax.set_xlabel(r"$\Delta f$ (Hz)")
    ax.set_ylabel("density")
    ax.set_xlim(-0.085, 0.085)
    ax.set_ylim(0, ymax * 1.30)
    ax.set_xticks(np.arange(-0.08, 0.0801, 0.02))
    polish_axes(ax)
    leg = ax.legend(loc="upper center", frameon=False, ncol=1,
                    bbox_to_anchor=(0.5, 1.02), handletextpad=0.4,
                    handlelength=1.5, labelspacing=0.18)
    for txt in leg.get_texts():
        txt.set_fontsize(6.6)
    save_figure(fig, args.out)


if __name__ == "__main__":
    main()
