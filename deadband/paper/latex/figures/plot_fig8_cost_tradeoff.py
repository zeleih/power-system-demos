"""Fig. 8: Day-long safety-cost trade-off (share>0.05 Hz vs E_ESD).

Regenerates the cost trade-off plot from phase1_full_day_ranked.csv.
Baseline tail-share is drawn as a red dashed reference because cost proxies
were not recomputed for the baseline run under the cost-split upgrade.
Output: fig8_cost_tradeoff.pdf/.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from fig_style import COLORS, apply_style, polish_axes, save_figure

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
SUMMARY_CSV = (
    ROOT / "results"
    / "phase1_full_day_tip100_alpha098_disable_pvd_agc_disable_esd_agc_kp0p1_ki0p002"
    / "phase1_full_day_ranked.csv"
)
OUT_PDF = HERE / "fig8_cost_tradeoff.pdf"

BASELINE_SHARE_05 = 0.0287
BASELINE_LABEL = "baseline tail share (uniform +/-0.036 Hz)"

df = pd.read_csv(SUMMARY_CSV).sort_values("esd_throughput").reset_index(drop=True)

apply_style()
fig, ax = plt.subplots(figsize=(3.5, 2.6))

# Wind-deadband colour key (ordinal blue ramp, light to dark).
unique_wind = sorted(df["wind_deadband_hz"].unique())
palette = ["#9BB9D5", "#5D8FBA", "#1F4E79", "#12314F"]
color_for = {w: palette[i % len(palette)] for i, w in enumerate(unique_wind)}

# Reference line for the baseline tail-share (red dashed).
ax.axhline(BASELINE_SHARE_05 * 100, color=COLORS["red"], linestyle="--",
           linewidth=0.9, zorder=1, label=BASELINE_LABEL)

# Candidate markers — circles, with a star for #1.
y_min = float(df["share_abs_gt_0p05"].min()) * 100
y_max = BASELINE_SHARE_05 * 100
for _, r in df.iterrows():
    c = color_for[r["wind_deadband_hz"]]
    rank = int(r["rank"])
    if rank == 1:
        ax.scatter(r["esd_throughput"], r["share_abs_gt_0p05"] * 100,
                   s=110, marker="*", color=c, edgecolor=COLORS["ink"],
                   linewidth=0.5, zorder=5)
    else:
        ax.scatter(r["esd_throughput"], r["share_abs_gt_0p05"] * 100,
                   s=44, marker="o", color=c, edgecolor="white",
                   linewidth=0.55, zorder=4)

# Compact rank annotations with non-colliding offsets.
offsets = {
    1: (-18, -2),
    2: (10, -4),
    3: (10, 3),
    4: (-18, 16),
    5: (13, -8),
    6: (13, 9),
}
for _, r in df.iterrows():
    rk = int(r["rank"])
    dx, dy = offsets.get(rk, (8, 4))
    ax.annotate(f"#{rk}",
                xy=(r["esd_throughput"], r["share_abs_gt_0p05"] * 100),
                xytext=(dx, dy), textcoords="offset points",
                fontsize=6.6, color=COLORS["ink"],
                ha="left" if dx >= 0 else "right",
                va="center",
                arrowprops=dict(arrowstyle="-", color=COLORS["muted"],
                                linewidth=0.45, shrinkA=1.5, shrinkB=2.5,
                                alpha=0.78))

# Compact legend for wind-deadband colour key + baseline reference.
legend_handles = [Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=color_for[w], markeredgecolor='white',
                         markersize=5.5,
                         label=rf"$d_{{b,\mathrm{{W}}}}={int(round(w*1000))}$ mHz")
                  for w in unique_wind]
legend_handles.append(Line2D([0], [0], color=COLORS["red"], linestyle='--',
                             label="baseline tail share"))
leg = ax.legend(handles=legend_handles, loc="upper left",
                bbox_to_anchor=(0.02, 0.98),
                frameon=True, framealpha=0.94, edgecolor=COLORS["grid"],
                ncol=1, handletextpad=0.4,
                handlelength=1.4, labelspacing=0.22)
leg.get_frame().set_linewidth(0.5)
for txt in leg.get_texts():
    txt.set_fontsize(6.5)

ax.set_xlabel(r"Storage throughput $E_{\mathrm{ESD}}$ (p.u.$\cdot$s)")
ax.set_ylabel(r"$\mathrm{share}(|\Delta f|\!>\!0.05\,\mathrm{Hz})$ (%)")
ax.set_xlim(df["esd_throughput"].min() - 0.7, df["esd_throughput"].max() + 1.0)
ax.set_ylim(y_min - 0.18, y_max + 0.30)
polish_axes(ax)
save_figure(fig, OUT_PDF)
