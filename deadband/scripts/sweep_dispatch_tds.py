#!/usr/bin/env python3
"""
Sweep AGC PI gains for the deadband dispatch-to-TDS workflow.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_dispatch_tds as rdt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_DISPATCH_JSON = RESULTS / "h13d2_dispatch.json"
DEFAULT_SWEEP_DIR = RESULTS / "agc4_sweep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatch-json", type=Path, default=DEFAULT_DISPATCH_JSON)
    parser.add_argument("--curve-file", type=Path, default=rdt.DEFAULT_CURVE_FILE)
    parser.add_argument("--dyn-case", type=Path, default=rdt.DEFAULT_DYN_CASE)
    parser.add_argument("--stable-dyn-case", type=Path,
                        default=rdt.CASES / "IL200_dyn_db2_stable_agc4_sweep.xlsx")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--agc-interval", type=int, default=4)
    parser.add_argument("--duration-seconds", type=int, default=900)
    parser.add_argument("--init-mode", choices=("dispatch", "first"),
                        default="first")
    parser.add_argument("--kp-list", type=float, nargs="+",
                        default=[0.05, 0.10, 0.15, 0.20, 0.25])
    parser.add_argument("--ki-list", type=float, nargs="+",
                        default=[0.0125, 0.0250, 0.0375, 0.0500, 0.0625])
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--save-series", action="store_true",
                        help="Save per-case CSV/PNG outputs for successful runs.")
    return parser.parse_args()


def safe_float(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and math.isnan(value):
        return None
    return float(value)


def summarize_case(
    kp: float,
    ki: float,
    init_mode: str,
    success: bool,
    t: np.ndarray | None = None,
    f_dev_hz: np.ndarray | None = None,
    error: str | None = None,
) -> dict:
    row = {
        "kp": kp,
        "ki": ki,
        "init_mode": init_mode,
        "success": int(success),
        "samples": None,
        "t_end_s": None,
        "min_hz": None,
        "t_min_s": None,
        "max_hz": None,
        "t_max_s": None,
        "final_hz": None,
        "abs_mean_hz": None,
        "rms_hz": None,
        "error": error or "",
    }

    if not success or t is None or f_dev_hz is None or len(t) == 0:
        return row

    imin = int(np.argmin(f_dev_hz))
    imax = int(np.argmax(f_dev_hz))
    row.update({
        "samples": int(len(t)),
        "t_end_s": float(t[-1]),
        "min_hz": float(f_dev_hz[imin]),
        "t_min_s": float(t[imin]),
        "max_hz": float(f_dev_hz[imax]),
        "t_max_s": float(t[imax]),
        "final_hz": float(f_dev_hz[-1]),
        "abs_mean_hz": float(np.mean(np.abs(f_dev_hz))),
        "rms_hz": float(np.sqrt(np.mean(np.square(f_dev_hz)))),
    })
    return row


def make_heatmap(fig_path: Path, summary: pd.DataFrame, init_mode: str) -> None:
    metrics = [
        ("success", "Success (1=yes)"),
        ("abs_mean_hz", "Mean |freq dev| [Hz]"),
        ("max_hz", "Max freq dev [Hz]"),
        ("final_hz", "Final freq dev [Hz]"),
    ]

    kp_vals = sorted(summary["kp"].unique())
    ki_vals = sorted(summary["ki"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (metric, title) in zip(axes.flat, metrics):
        pivot = summary.pivot(index="ki", columns="kp", values=metric).reindex(index=ki_vals, columns=kp_vals)
        data = pivot.to_numpy(dtype=float)
        cmap = "viridis" if metric == "success" else "coolwarm"
        if metric == "success":
            vmin, vmax = 0, 1
        else:
            finite = data[np.isfinite(data)]
            if finite.size == 0:
                vmin, vmax = 0.0, 1.0
            else:
                vmin, vmax = float(np.min(finite)), float(np.max(finite))
                if vmin == vmax:
                    vmin -= 1e-6
                    vmax += 1e-6
        im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(range(len(kp_vals)), [f"{v:.3f}" for v in kp_vals], rotation=45, ha="right")
        ax.set_yticks(range(len(ki_vals)), [f"{v:.4f}" for v in ki_vals])
        ax.set_xlabel("KP")
        ax.set_ylabel("KI")

        for i, ki in enumerate(ki_vals):
            for j, kp in enumerate(kp_vals):
                value = data[i, j]
                if np.isnan(value):
                    text = "fail"
                elif metric == "success":
                    text = str(int(value))
                else:
                    text = f"{value:.4f}"
                ax.text(j, i, text, ha="center", va="center", color="white", fontsize=8)

        fig.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(f"AGC PI Sweep at 4-second AGC Interval ({init_mode} init)", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)


def make_top_plot(
    fig_path: Path,
    series: list[tuple[str, np.ndarray, np.ndarray]],
    init_mode: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    colors = ["#0f5c78", "#2a7f3f", "#b24c2a", "#c48f00", "#7a4ba0", "#444444"]

    for i, (label, t, f_dev_hz) in enumerate(series):
        color = colors[i % len(colors)]
        axes[0].plot(t, f_dev_hz, label=label, color=color, linewidth=1.4)
        axes[1].plot(t, f_dev_hz, label=label, color=color, linewidth=1.4)

    for ax in axes:
        ax.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")
        ax.grid(True, alpha=0.25)
        ax.legend()

    axes[0].set_title(f"Best Stable AGC PI Candidates at 4-second AGC Interval ({init_mode} init)")
    axes[0].set_ylabel("Frequency deviation [Hz]")
    axes[0].set_ylim(-0.07, 0.05)
    axes[1].set_title("Tail Response")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Frequency deviation [Hz]")
    axes[1].set_xlim(500, 900)
    axes[1].set_ylim(-0.02, 0.04)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    rdt.andes.config_logger(stream_level=30)

    curve = rdt.load_curve(args.curve_file)
    dyn_case = rdt.adapt_dyn_case(args.dyn_case, args.stable_dyn_case)
    dispatch_record = rdt.DispatchRecord.from_json(args.dispatch_json)

    rows: list[dict] = []
    successful_series: list[tuple[dict, np.ndarray, np.ndarray]] = []
    total = len(args.kp_list) * len(args.ki_list)
    run_no = 0

    for kp in args.kp_list:
        for ki in args.ki_list:
            run_no += 1
            print(f"[{run_no}/{total}] kp={kp:.4f}, ki={ki:.4f}")
            try:
                t, f_dev_hz = rdt.run_tds(
                    dispatch_record=dispatch_record,
                    curve=curve,
                    dyn_case=dyn_case,
                    duration_seconds=args.duration_seconds,
                    agc_interval=args.agc_interval,
                    kp=kp,
                    ki=ki,
                    wind_prefixes=rdt.DEFAULT_WIND_PREFIXES,
                    solar_prefixes=rdt.DEFAULT_SOLAR_PREFIXES,
                    init_mode=args.init_mode,
                )
            except Exception as exc:
                row = summarize_case(
                    kp=kp,
                    ki=ki,
                    init_mode=args.init_mode,
                    success=False,
                    error=str(exc),
                )
                rows.append(row)
                print(f"  fail: {exc}")
                continue

            row = summarize_case(
                kp=kp,
                ki=ki,
                init_mode=args.init_mode,
                success=True,
                t=t,
                f_dev_hz=f_dev_hz,
            )
            rows.append(row)
            successful_series.append((row, t, f_dev_hz))
            print(
                "  ok:"
                f" min={row['min_hz']:.4f},"
                f" max={row['max_hz']:.4f},"
                f" final={row['final_hz']:.4f},"
                f" abs_mean={row['abs_mean_hz']:.4f}"
            )

            if args.save_series:
                label = f"kp{kp:.3f}_ki{ki:.4f}".replace(".", "p")
                rdt.write_dispatch_json(dispatch_record, args.results_dir, label=label)
                rdt.save_outputs(t, f_dev_hz, dispatch_record, args.results_dir, label=label)

    summary = pd.DataFrame(rows).sort_values(["kp", "ki"]).reset_index(drop=True)
    summary_csv = args.results_dir / "sweep_summary.csv"
    summary.to_csv(summary_csv, index=False)

    heatmap_png = args.results_dir / "sweep_heatmaps.png"
    make_heatmap(heatmap_png, summary, args.init_mode)

    stable = summary[summary["success"] == 1].copy()
    stable["score"] = (
        stable["abs_mean_hz"]
        + 0.5 * stable["rms_hz"]
        + 0.5 * stable["final_hz"].abs()
    )
    stable = stable.sort_values(["score", "abs_mean_hz", "final_hz"])
    top_csv = args.results_dir / "top_candidates.csv"
    stable.head(args.top_n).to_csv(top_csv, index=False)

    series_lookup = {(row["kp"], row["ki"]): (t, f_dev_hz) for row, t, f_dev_hz in successful_series}
    top_series = []
    for _, row in stable.head(args.top_n).iterrows():
        t, f_dev_hz = series_lookup[(row["kp"], row["ki"])]
        label = (
            f"KP={row['kp']:.3f}, KI={row['ki']:.4f} | "
            f"mean={row['abs_mean_hz']:.4f}, final={row['final_hz']:.4f}"
        )
        top_series.append((label, t, f_dev_hz))
    top_png = args.results_dir / "top_candidates.png"
    if top_series:
        make_top_plot(top_png, top_series, args.init_mode)

    print(f"summary_csv={summary_csv}")
    print(f"heatmap_png={heatmap_png}")
    print(f"top_csv={top_csv}")
    if top_series:
        print(f"top_png={top_png}")


if __name__ == "__main__":
    main()
