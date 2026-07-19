"""Fig. 3: Baseline day-long frequency distribution (the 0.036 Hz shoulder).

Reads all 96 ``h*d*_frequency.csv`` files from the baseline full-day result
directory, concatenates them, and renders a single-column density plot with
the deadband boundary (+/-0.036 Hz, amber) and tail threshold (+/-0.05 Hz, red).
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
ROOT = HERE.parent.parent.parent  # .../deadband/


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
        default=ROOT / "results" /
                "phase1_baseline_full_day_tip100_alpha098_disable_pvd_agc_disable_esd_agc_kp0p1_ki0p002" /
                "wind036_pv036_esd036",
    )
    parser.add_argument("--out", type=Path, default=HERE / "fig3_baseline_dist.pdf")
    args = parser.parse_args()

    freq = load_full_day(args.baseline_dir)
    apply_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.35))

    bins = np.linspace(-0.095, 0.095, 161)
    centers, _, smooth = smooth_histogram(freq, bins, density=True, window=11)

    add_deadband_band(ax, half_width=0.036, alpha=0.55)
    ax.fill_between(centers, 0, smooth, color=COLORS["blue"], alpha=0.16,
                    linewidth=0, zorder=2)
    ax.plot(centers, smooth, color=COLORS["blue"], linewidth=1.4, zorder=4,
            label=r"baseline: uniform $\pm36$ mHz")
    add_threshold_lines(ax)

    # Shoulder annotation: anchor to the positive shoulder, label inside the plot.
    ymax = float(smooth.max())
    sx = 0.0395
    sy = float(np.interp(sx, centers, smooth))
    ax.annotate(
        "deadband-edge\nshoulder",
        xy=(sx, sy),
        xytext=(0.066, 0.50 * ymax),
        arrowprops=dict(arrowstyle="-|>", color=COLORS["amber"], linewidth=0.7,
                        shrinkA=0.5, shrinkB=2, mutation_scale=7),
        ha="center", va="center", fontsize=6.6, color=COLORS["amber"],
    )

    ax.set_xlabel(r"$\Delta f$ (Hz)")
    ax.set_ylabel("density")
    ax.set_xlim(-0.085, 0.085)
    ax.set_ylim(0, ymax * 1.20)
    ax.set_xticks(np.arange(-0.08, 0.0801, 0.02))
    polish_axes(ax)
    ax.legend(loc="upper left", frameon=False, handlelength=1.4)
    save_figure(fig, args.out)


if __name__ == "__main__":
    main()
