"""Fig. 11: bounded-inertia and multi-window robustness screen.

Reads inertia_window_metrics.csv from run_inertia_sensitivity_windows.py and
plots pairwise relative reductions of the selected heterogeneous candidate
against the uniform 36 mHz baseline for each inertia/window pair.
Output: fig11_inertia_screen.pdf/.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from fig_style import COLORS, apply_style, polish_axes, save_figure

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
RESULTS_DIR = ROOT / "results" / "inertia_multiscenario_screen_20260503"
WINDOW_CSV = RESULTS_DIR / "inertia_window_metrics.csv"
OUT_PDF = HERE / "fig11_inertia_screen.pdf"
OUT_PAIRWISE = RESULTS_DIR / "inertia_pairwise_reductions.csv"

WINDOW_LABELS = {
    "h20d3_h21d0": "h20 low ramp\nh20d3→h21d0",
    "h11d2_h11d3": "h11 high PV\nh11d2→h11d3",
    "h7d3_h8d2": "h07 max +ramp\nh7d3→h8d2",
}
WINDOW_ORDER = ["h20d3_h21d0", "h11d2_h11d3", "h7d3_h8d2"]
INERTIA_ORDER = [0.5, 1.0, 1.25]


def _paired_reductions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "mean_abs_hz",
        "share_abs_gt_0p036",
        "share_abs_gt_0p05",
        "max_abs_hz",
        "edge_mass_36",
    ]
    ok = df[df.get("failed", 0).fillna(0).astype(int) == 0].copy()
    for (mult, window), sub in ok.groupby(["inertia_multiplier", "window"]):
        cases = {str(row["case"]): row for _, row in sub.iterrows()}
        if "uniform" not in cases or "best" not in cases:
            continue
        row = {"inertia_multiplier": float(mult), "window": window}
        for metric in metrics:
            uniform = float(cases["uniform"][metric])
            best = float(cases["best"][metric])
            row[f"uniform_{metric}"] = uniform
            row[f"best_{metric}"] = best
            row[f"delta_{metric}"] = best - uniform
            row[f"reduction_{metric}"] = (
                (uniform - best) / uniform * 100.0 if abs(uniform) > 1e-12 else np.nan
            )
        rows.append(row)
    pairwise = pd.DataFrame(rows)
    if pairwise.empty:
        raise RuntimeError(f"No paired uniform/best rows found in {WINDOW_CSV}")
    return pairwise.sort_values(["window", "inertia_multiplier"])


def _matrix(pairwise: pd.DataFrame, metric: str) -> np.ndarray:
    out = np.full((len(WINDOW_ORDER), len(INERTIA_ORDER)), np.nan)
    for i, window in enumerate(WINDOW_ORDER):
        for j, mult in enumerate(INERTIA_ORDER):
            hit = pairwise[
                (pairwise["window"] == window)
                & np.isclose(pairwise["inertia_multiplier"], mult)
            ]
            if not hit.empty:
                out[i, j] = float(hit.iloc[0][f"reduction_{metric}"])
    return out


def _draw_heatmap(ax, data: np.ndarray, title: str, *, show_ylabels: bool,
                  span_override: float | None = None) -> None:
    span = max(5.0, float(np.nanmax(np.abs(data))))
    span = float(np.ceil(span / 5.0) * 5.0)
    if span_override is not None:
        span = float(span_override)
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#F3F4F6")
    norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_title(title, pad=2.5, fontsize=7.6)
    ax.set_xticks(range(len(INERTIA_ORDER)))
    ax.set_xticklabels([f"{x:g}x" for x in INERTIA_ORDER], fontsize=6.6)
    ax.set_yticks(range(len(WINDOW_ORDER)))
    if show_ylabels:
        ax.set_yticklabels([WINDOW_LABELS[w] for w in WINDOW_ORDER],
                           fontsize=6.5)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", left=False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if np.isfinite(value):
                color = "white" if abs(value) > 0.55 * span else COLORS["ink"]
                ax.text(j, i, f"{value:+.1f}%", ha="center", va="center",
                        fontsize=6.4, color=color)
            else:
                ax.text(j, i, "n/a", ha="center", va="center",
                        fontsize=6.2, color=COLORS["muted"])

    ax.set_xticks(np.arange(-0.5, len(INERTIA_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(WINDOW_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    polish_axes(ax, grid=False)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.052, pad=0.03)
    cbar.set_ticks([-span, 0, span])
    cbar.ax.tick_params(labelsize=6.2, width=0.5, length=2.0)
    cbar.outline.set_linewidth(0.5)


def main() -> None:
    df = pd.read_csv(WINDOW_CSV)
    pairwise = _paired_reductions(df)
    OUT_PAIRWISE.parent.mkdir(parents=True, exist_ok=True)
    pairwise.to_csv(OUT_PAIRWISE, index=False)

    tail = _matrix(pairwise, "share_abs_gt_0p05")
    edge = _matrix(pairwise, "edge_mass_36")
    apply_style()
    fig, (ax_tail, ax_edge) = plt.subplots(
        1, 2, figsize=(3.5, 1.9), gridspec_kw={"wspace": 0.5}
    )
    _draw_heatmap(
        ax_tail,
        tail,
        r"(a) share($|\Delta f|>50$ mHz)",
        show_ylabels=True,
        span_override=100.0,
    )
    _draw_heatmap(
        ax_edge,
        edge,
        r"(b) $\mathrm{EM}_{36}$",
        show_ylabels=False,
        span_override=30.0,
    )
    fig.supxlabel(r"synchronous inertia scaling $M_{\mathrm{scale}}$",
                  fontsize=7.2, y=0.015)
    fig.subplots_adjust(left=0.165, right=0.965, bottom=0.20, top=0.88)
    save_figure(fig, OUT_PDF)


if __name__ == "__main__":
    main()
