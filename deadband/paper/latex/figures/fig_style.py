"""Shared plotting helpers for the paper figures.

Restrained IEEE-friendly palette:
- 2 neutrals (ink, axis, grid) carry the chrome.
- 3 data hues (blue, red, amber) carry meaning consistently across figures:
    blue   -> "best candidate" / DER / primary series
    red    -> "baseline" / tail-risk threshold
    amber  -> deadband boundary / design-variable accent
- Optional supports (green, slate, teal) only when an extra channel is
  unavoidable (e.g., governor channel in Fig.4, wind-deadband colouring in
  Fig.8).
"""

from __future__ import annotations

import numpy as np

COLORS = {
    # --- neutrals (chrome) ---
    "ink":   "#1F2937",   # axis, primary text
    "axis":  "#4B5563",   # secondary text, faint arrows
    "muted": "#6B7280",   # tertiary text
    "grid":  "#E5E7EB",   # subtle grid
    "band":  "#EEF2F7",   # inside-deadband shading
    # --- semantic data hues (limited palette) ---
    "blue":  "#1F4E79",   # best / primary series
    "red":   "#A52828",   # baseline / tail-risk
    "amber": "#B45309",   # deadband boundary, design-variable accent
    "green": "#2F6F4E",   # governor channel
    "teal":  "#1E6B6B",   # auxiliary categorical
    # --- tints for fills ---
    "blue_tint":  "#DCE6F1",
    "red_tint":   "#F1DCDC",
    "amber_tint": "#F4E4CB",
    "green_tint": "#D7E8DD",
    # --- aliases retained for symmetric API ---
    "best":      "#1F4E79",
    "baseline":  "#A52828",
    "slate":     "#475569",
    "text":      "#1F2937",
    "orange":    "#B45309",
    "red_light":  "#F1DCDC",
    "blue_light": "#DCE6F1",
}


def apply_style() -> None:
    """Apply consistent matplotlib rcParams across all figures."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlesize": 8.5,
            "axes.titleweight": "regular",
            "axes.labelsize": 8,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.0,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": False,
        }
    )


def polish_axes(ax, grid: bool = True) -> None:
    """Strip top/right spines, soften the remaining ones, optional grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["ink"])
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(colors=COLORS["ink"], width=0.6, length=2.6)
    if grid:
        ax.grid(True, color=COLORS["grid"], linewidth=0.5, alpha=0.85)
        ax.set_axisbelow(True)


def save_figure(fig, out_pdf, dpi: int = 300) -> None:
    """Write PDF + a same-name PNG at the given DPI."""
    out_pdf = str(out_pdf)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    if out_pdf.lower().endswith(".pdf"):
        fig.savefig(
            out_pdf[:-4] + ".png", dpi=dpi, bbox_inches="tight", pad_inches=0.03
        )


def smooth_histogram(values, bins, density: bool = True, window: int = 9):
    """Histogram with a Hanning-smoothed companion curve.

    Returns (centers, raw_hist, smoothed_hist).
    """
    hist, edges = np.histogram(values, bins=bins, density=density)
    centers = 0.5 * (edges[:-1] + edges[1:])
    window = max(3, int(window) | 1)
    kernel = np.hanning(window)
    kernel = kernel / kernel.sum()
    smooth = np.convolve(hist, kernel, mode="same")
    return centers, hist, smooth


def add_deadband_band(ax, half_width: float = 0.036, *, vertical: bool = True,
                      color: str | None = None, alpha: float = 0.55) -> None:
    """Shade the inside-deadband region (default ±36 mHz)."""
    color = color or COLORS["band"]
    if vertical:
        ax.axvspan(-half_width, half_width, color=color, alpha=alpha, zorder=0)
    else:
        ax.axhspan(-half_width, half_width, color=color, alpha=alpha, zorder=0)


def add_threshold_lines(ax, *, vertical: bool = True,
                        primary: float = 0.036, secondary: float = 0.05) -> None:
    """Two reference thresholds: amber for the deadband boundary, red for tail."""
    pairs = [
        (primary,   COLORS["amber"], "-",  0.85),
        (secondary, COLORS["red"],   "--", 0.75),
    ]
    for thr, color, ls, alpha in pairs:
        if vertical:
            ax.axvline(thr,  color=color, linestyle=ls, linewidth=0.7, alpha=alpha, zorder=1)
            ax.axvline(-thr, color=color, linestyle=ls, linewidth=0.7, alpha=alpha, zorder=1)
        else:
            ax.axhline(thr,  color=color, linestyle=ls, linewidth=0.7, alpha=alpha, zorder=1)
            ax.axhline(-thr, color=color, linestyle=ls, linewidth=0.7, alpha=alpha, zorder=1)


# Backwards-compatible alias used by fig4/fig6/fig7.
add_frequency_thresholds = add_threshold_lines


def stat_box(ax, text: str, xy=(0.98, 0.96), ha: str = "right",
             facecolor: str = "white", edgecolor: str | None = None,
             fontsize: float = 6.6, **kwargs) -> None:
    """Compact white stat callout, used sparingly."""
    edge = edgecolor or COLORS["grid"]
    box = {
        "boxstyle": "round,pad=0.3,rounding_size=0.04",
        "facecolor": facecolor,
        "edgecolor": edge,
        "linewidth": 0.6,
        "alpha": 0.94,
    }
    box.update(kwargs.pop("bbox", {}))
    ax.text(
        xy[0], xy[1], text,
        ha=ha, va="top",
        fontsize=fontsize,
        color=COLORS["ink"],
        transform=ax.transAxes,
        bbox=box,
        **kwargs,
    )


def pct(value: float, decimals: int = 2) -> str:
    """Format a 0..100 number as a literal percent string ('12.34%').

    matplotlib without usetex renders '\\%' literally, so always feed it '%'.
    """
    return f"{value:.{decimals}f}%"
