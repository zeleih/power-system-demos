"""Fig. 7: 96-interval frequency panorama (baseline vs. best).

Concatenates all 96 h*d*_frequency.csv files from each run into a single
24-hour timeline and plots them as two stacked panels. Horizontal reference
lines mark +/-0.036 Hz and +/-0.05 Hz; vertical light ticks mark every 4 hours.
Output: fig7_panorama.pdf/.png
"""
import argparse
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fig_style import (COLORS, add_deadband_band, add_threshold_lines,
                       apply_style, pct, polish_axes, save_figure)

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent

_FNAME = re.compile(r"h(\d+)d(\d+)_frequency\.csv$")


def index_from_name(p: Path) -> int:
    m = _FNAME.search(p.name)
    if not m:
        return 1 << 30
    h, d = int(m.group(1)), int(m.group(2))
    return h * 4 + d


def load_day_timeline(results_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(
        [Path(p) for p in glob.glob(str(results_dir / "h*d*_frequency.csv"))],
        key=index_from_name,
    )
    if not files:
        raise FileNotFoundError(f"No h*d*_frequency.csv under {results_dir}")
    t_all, f_all = [], []
    for pos, fp in enumerate(files):
        df = pd.read_csv(fp)
        offset = 900.0 * pos
        t_all.append(df["time_s"].values + offset)
        f_all.append(df["freq_dev_hz"].values)
    return np.concatenate(t_all), np.concatenate(f_all)


def rolling(y: np.ndarray, window: int = 151) -> np.ndarray:
    return pd.Series(y).rolling(window, center=True, min_periods=1).mean().to_numpy()


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
    parser.add_argument("--out", type=Path, default=HERE / "fig7_panorama.pdf")
    args = parser.parse_args()

    t_b, f_b = load_day_timeline(args.baseline_dir)
    t_w, f_w = load_day_timeline(args.best_dir)
    h_b = t_b / 3600.0
    h_w = t_w / 3600.0

    apply_style()
    # Two-column figure justified by the 24-hour timeline, two traces, and an
    # hourly edge-mass summary that shows where the shoulder is concentrated.
    fig, axes = plt.subplots(3, 1, figsize=(7.16, 3.35), sharex=True,
                             gridspec_kw={"hspace": 0.09,
                                          "height_ratios": [1.08, 1.08, 0.55]})

    cases = [
        (h_b, f_b, r"Baseline: uniform $\pm36$ mHz", COLORS["red"]),
        (h_w, f_w, "Best: 36/25/15 mHz",   COLORS["blue"]),
    ]
    for ax, (h, f, title, color) in zip(axes[:2], cases):
        add_deadband_band(ax, half_width=0.036, vertical=False, alpha=0.45)
        add_threshold_lines(ax, vertical=False)
        # raw trace at 1 Hz, very thin and translucent for context
        ax.plot(h, f, linewidth=0.16, color=color, alpha=0.20, rasterized=True)
        ax.plot(h, rolling(f), linewidth=1.02, color=color, alpha=0.98)
        # mark the few samples that breach the tail threshold
        exceed = np.abs(f) > 0.05
        if exceed.any():
            step = max(1, exceed.sum() // 600)
            ax.scatter(h[exceed][::step], f[exceed][::step], s=2.2,
                       color=COLORS["red"], alpha=0.72, linewidths=0,
                       rasterized=True, zorder=4)
        # 4-hour gridlines for orientation
        for hour in range(0, 25, 4):
            ax.axvline(hour, color=COLORS["grid"], linewidth=0.4,
                       alpha=0.7, zorder=0)
        ax.set_ylabel(r"$\Delta f$ (Hz)")
        ax.text(0.006, 0.94, title,
                transform=ax.transAxes, ha="left", va="top",
                fontsize=7.2, color=color,
                bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.03",
                          facecolor="white", alpha=0.92,
                          edgecolor=COLORS["grid"], linewidth=0.45))
        share036 = np.mean(np.abs(f) > 0.036) * 100
        share05 = np.mean(np.abs(f) > 0.05) * 100
        ax.text(0.998, 0.04,
                rf"$|\Delta f|>36$ mHz: {pct(share036,1)}    "
                rf"$|\Delta f|>50$ mHz: {pct(share05,2)}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.6,
                color=COLORS["ink"],
                bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.03",
                          facecolor="white", alpha=0.95,
                          edgecolor=COLORS["grid"], linewidth=0.55))
        polish_axes(ax, grid=False)

    for ax in axes[:2]:
        ax.set_ylim(-0.105, 0.085)

    def hourly_edge_mass(t_hours: np.ndarray, freq: np.ndarray) -> np.ndarray:
        out = []
        for hour in range(24):
            mask = (t_hours >= hour) & (t_hours < hour + 1)
            abs_f = np.abs(freq[mask])
            edge = (abs_f >= 0.032) & (abs_f <= 0.040)
            out.append(float(edge.mean() * 100.0) if abs_f.size else np.nan)
        return np.asarray(out)

    hours = np.arange(24)
    ax_bar = axes[2]
    em_b = hourly_edge_mass(h_b, f_b)
    em_w = hourly_edge_mass(h_w, f_w)
    ax_bar.bar(hours - 0.18, em_b, width=0.36, color=COLORS["red"],
               alpha=0.50, linewidth=0, label="baseline")
    ax_bar.bar(hours + 0.18, em_w, width=0.36, color=COLORS["blue"],
               alpha=0.58, linewidth=0, label="best")
    ax_bar.set_ylabel(r"hourly $\mathrm{EM}_{36}$ (%)")
    ax_bar.set_xlabel("Time (hours)")
    ax_bar.set_ylim(0, max(np.nanmax(em_b), np.nanmax(em_w)) * 1.18)
    ax_bar.legend(frameon=False, loc="upper right", ncol=2,
                  handlelength=1.2, columnspacing=0.8)
    for hour in range(0, 25, 4):
        ax_bar.axvline(hour, color=COLORS["grid"], linewidth=0.4,
                       alpha=0.7, zorder=0)
    polish_axes(ax_bar, grid=False)

    axes[-1].set_xlim(0, 24)
    axes[-1].set_xticks(range(0, 25, 4))
    fig.align_ylabels(axes)
    save_figure(fig, args.out, dpi=240)


if __name__ == "__main__":
    main()
