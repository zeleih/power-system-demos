#!/usr/bin/env python3
"""
Minimal-model figure for §4.4: piecewise-linear stochastic model.

Two panels:
  (a) Theoretical sensitivity to converter-droop-to-damping ratio beta/alpha
      under a fixed deadband d_b = 36 mHz. Shows that the shoulder is created
      by the kink in drift slope and intensifies as beta/alpha grows.
  (b) Empirical day-long |delta f| histogram from the baseline 96-dispatch
      run versus the analytical stationary density fitted from
      minimum L2 distance to the empirical histogram.

Output: paper/latex/figures/fig9_minimal_model.pdf
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


REPO = Path(__file__).resolve().parents[1]
BASELINE_DIR = (
    REPO
    / "results"
    / "phase1_baseline_full_day_tip100_alpha098_disable_pvd_agc_disable_esd_agc_kp0p1_ki0p002"
    / "wind036_pv036_esd036"
)
OUT_PDF = REPO / "paper" / "latex" / "figures" / "fig9_minimal_model.pdf"
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)


D_B = 0.036  # deadband boundary (Hz)


# ---------------------------------------------------------------------------
# Model density
# ---------------------------------------------------------------------------
def stationary_density(x: np.ndarray, alpha: float, beta: float, D: float, d_b: float = D_B) -> np.ndarray:
    """
    Unnormalized stationary density of the piecewise-linear Langevin model.

    p(x) ∝ exp(-2 V(x) / sigma^2) with sigma^2 = 2 D.

    Inside |x| <= d_b: V = alpha x^2 / 2.
    Outside |x| > d_b: V = (alpha + beta) x^2 / 2 - beta d_b |x| + beta d_b^2 / 2.
    """

    ax = np.abs(x)
    inside = ax <= d_b
    outside = ~inside

    p = np.empty_like(x, dtype=float)

    # inside the deadband: pure OU
    p[inside] = np.exp(-(alpha * x[inside] ** 2) / (2.0 * D))

    if np.any(outside):
        x_out = x[outside]
        ax_out = np.abs(x_out)
        # outer-branch potential, glued continuously at |x| = d_b
        V_out = (alpha + beta) * x_out ** 2 / 2.0 - beta * d_b * ax_out + beta * d_b ** 2 / 2.0
        p[outside] = np.exp(-V_out / D)
    return p


def normalize(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    area = np.trapezoid(p, x)
    if area <= 0:
        return p
    return p / area


# ---------------------------------------------------------------------------
# Empirical baseline aggregation
# ---------------------------------------------------------------------------
def load_baseline_freq() -> np.ndarray:
    csvs = sorted(BASELINE_DIR.glob("h*d*_frequency.csv"))
    if len(csvs) != 96:
        raise SystemExit(
            f"Expected 96 frequency CSVs, found {len(csvs)} in {BASELINE_DIR}"
        )
    chunks = [pd.read_csv(p)["freq_dev_hz"].to_numpy() for p in csvs]
    return np.concatenate(chunks)


def fit_alpha_beta_D_from_data(samples: np.ndarray) -> tuple[float, float, float]:
    """
    Joint 2-D calibration of (beta/alpha, D/alpha) with alpha fixed at 1.

    Only the dimensionless ratios beta/alpha and D/alpha affect the density
    shape, so fixing alpha is a normalization choice. We pick the pair that
    minimizes the L2 error between the model density and the empirical
    density on a fine x-grid.
    """

    samples = samples[np.abs(samples) <= 0.10]  # restrict to the modeled support

    bins = np.linspace(-0.10, 0.10, 401)
    hist, edges = np.histogram(samples, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    alpha = 1.0
    var_full = float(np.var(samples, ddof=0))

    # search D/alpha around var_full (OU would give D/alpha = var; truncation
    # and the outer branch break this exact relation, so we widen the window).
    D_grid = np.geomspace(0.3 * var_full, 5.0 * var_full, 30)
    ratio_grid = np.geomspace(0.05, 100.0, 80)

    best = (None, None, np.inf)
    for D_try in D_grid:
        for ratio in ratio_grid:
            beta = ratio * alpha
            p = stationary_density(centers, alpha, beta, D_try)
            p = normalize(p, centers)
            err = float(np.mean((p - hist) ** 2))
            if err < best[2]:
                best = (ratio, D_try, err)

    ratio_fit = best[0] if best[0] is not None else 1.0
    D_fit = best[1] if best[1] is not None else var_full
    return alpha, ratio_fit * alpha, D_fit


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
PALETTE = {
    "ink": "#1F2937",
    "slate": "#475569",
    "blue": "#1F4E79",
    "amber": "#B45309",
    "red": "#A52828",
    "red_tint": "#F1DCDC",
    "soft": "#94A3B8",
}


def _setup_axes(ax, x_max=0.10):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(PALETTE["ink"])
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", length=3.0, width=0.7, colors=PALETTE["ink"])
    ax.set_xlim(-x_max, x_max)


def main() -> None:
    samples = load_baseline_freq()
    alpha, beta, D = fit_alpha_beta_D_from_data(samples)
    ratio_fit = beta / alpha

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.9), constrained_layout=True)
    ax_a, ax_b = axes

    # -----------------------------------------------------------------------
    # Panel (a): theoretical sweep over beta/alpha
    # -----------------------------------------------------------------------
    x = np.linspace(-0.08, 0.08, 4001)
    sweep_ratios = [0.0, 5.0, 20.0, 80.0]
    sweep_colors = ["#BFD0E0", "#7EA6C8", "#3E729D", PALETTE["blue"]]
    for r, c in zip(sweep_ratios, sweep_colors):
        beta_r = r * alpha
        p = stationary_density(x, alpha, beta_r, D)
        p = normalize(p, x)
        label = (
            r"$\beta = 0$ (no deadband)" if r == 0 else fr"$\beta/\alpha = {int(r)}$"
        )
        ax_a.plot(x, p, color=c, linewidth=1.3, label=label)
    ax_a.axvline(D_B, color=PALETTE["amber"], linewidth=0.65, linestyle=(0, (3, 2)))
    ax_a.axvline(-D_B, color=PALETTE["amber"], linewidth=0.65, linestyle=(0, (3, 2)))
    ax_a.set_xlabel(r"$\Delta f$ (Hz)")
    ax_a.set_ylabel("stationary density (1/Hz)")
    _setup_axes(ax_a, x_max=0.08)
    ax_a.xaxis.set_major_locator(MultipleLocator(0.02))
    ax_a.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.50, 0.03), ncol=2)
    ax_a.text(
        0.02,
        0.98,
        "(a)",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )

    # annotate boundary
    ymax = ax_a.get_ylim()[1]
    ax_a.text(D_B + 0.001, ymax * 0.93, r"$+d_b$", color=PALETTE["amber"], fontsize=7)
    ax_a.text(-D_B - 0.012, ymax * 0.93, r"$-d_b$", color=PALETTE["amber"], fontsize=7)

    # -----------------------------------------------------------------------
    # Panel (b): baseline empirical histogram vs fitted analytical density
    # -----------------------------------------------------------------------
    bins = np.linspace(-0.10, 0.10, 161)
    ax_b.hist(
        samples,
        bins=bins,
        density=True,
        color=PALETTE["red_tint"],
        alpha=0.82,
        edgecolor="none",
        label="baseline empirical (96 dispatch)",
    )
    x_b = np.linspace(-0.10, 0.10, 4001)
    p_fit = stationary_density(x_b, alpha, beta, D)
    p_fit = normalize(p_fit, x_b)
    ax_b.plot(
        x_b,
        p_fit,
        color=PALETTE["blue"],
        linewidth=1.6,
        label=fr"minimal model ($\beta/\alpha={ratio_fit:.1f}$)",
    )
    ax_b.axvline(D_B, color=PALETTE["amber"], linewidth=0.65, linestyle=(0, (3, 2)))
    ax_b.axvline(-D_B, color=PALETTE["amber"], linewidth=0.65, linestyle=(0, (3, 2)))
    ax_b.set_xlabel(r"$\Delta f$ (Hz)")
    ax_b.set_ylabel("density (1/Hz)")
    _setup_axes(ax_b, x_max=0.08)
    ax_b.xaxis.set_major_locator(MultipleLocator(0.02))
    ax_b.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.52, 0.03))
    ax_b.text(
        0.02,
        0.98,
        "(b)",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )

    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Saved {OUT_PDF}")
    print(f"Calibration: alpha=1 (unit), beta/alpha={ratio_fit:.2f}, D={D:.3e}")
    inside = samples[np.abs(samples) <= D_B]
    outside = samples[np.abs(samples) > D_B]
    print(
        f"Empirical: {samples.size} samples, |inside|={inside.size}, "
        f"|outside|={outside.size}, share(>0.036Hz)={outside.size/samples.size:.3%}"
    )


if __name__ == "__main__":
    main()
