#!/usr/bin/env python3
"""
Run two consecutive dispatch intervals as one continuous TDS simulation.

This script starts from the first dispatch OPF state and follows the curve for
``2 * dispatch_interval`` seconds without reinitializing the dynamic model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_dispatch_tds as rdt
from compare_dispatch_pair_hotstart import compute_bf, dispatch_offset, prepare_system, run_segment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-dispatch-json", type=Path, required=True)
    parser.add_argument("--second-dispatch-json", type=Path, required=True)
    parser.add_argument("--first-cold-csv", type=Path, default=None)
    parser.add_argument("--second-cold-csv", type=Path, default=None)
    parser.add_argument("--kp", type=float, default=0.03)
    parser.add_argument("--ki", type=float, default=0.01)
    parser.add_argument("--agc-interval", type=int, default=4)
    parser.add_argument("--dispatch-interval", type=int, default=900)
    parser.add_argument("--init-mode", choices=("dispatch", "first"), default="first")
    parser.add_argument("--dyn-case", type=Path, default=rdt.DEFAULT_DYN_CASE)
    parser.add_argument("--stable-dyn-case", type=Path, default=rdt.DEFAULT_STABLE_DYN_CASE)
    parser.add_argument("--curve-file", type=Path, default=rdt.DEFAULT_CURVE_FILE)
    parser.add_argument("--results-dir", type=Path, default=rdt.RESULTS / "continuous_pair")
    parser.add_argument("--label", type=str, default=None)
    return parser.parse_args()


def boundary_value(t: np.ndarray, f: np.ndarray, target_s: float) -> float:
    hits = np.where(np.isclose(t, target_s))[0]
    if hits.size:
        return float(f[hits[0]])

    idx = int(np.argmin(np.abs(t - target_s)))
    return float(f[idx])


def main() -> None:
    args = parse_args()
    rdt.andes.config_logger(stream_level=30)

    first = rdt.DispatchRecord.from_json(args.first_dispatch_json)
    second = rdt.DispatchRecord.from_json(args.second_dispatch_json)
    label = args.label or f"{first.label}_{second.label}_continuous"
    args.results_dir.mkdir(parents=True, exist_ok=True)

    curve = rdt.load_curve(args.curve_file)
    total_seconds = 2 * args.dispatch_interval
    start = dispatch_offset(first, args.dispatch_interval)
    end = start + total_seconds
    if end > len(curve):
        raise RuntimeError(f"Curve window [{start}, {end}) exceeds curve length {len(curve)}")

    dyn_case = rdt.adapt_dyn_case(args.dyn_case, args.stable_dyn_case)
    sa, ctx = prepare_system(
        dispatch_record=first,
        curve=curve,
        dyn_case=dyn_case,
        dispatch_interval=args.dispatch_interval,
        init_mode=args.init_mode,
        wind_prefixes=rdt.DEFAULT_WIND_PREFIXES,
        solar_prefixes=rdt.DEFAULT_SOLAR_PREFIXES,
    )
    bf = compute_bf(sa, first)
    t, f, ace_integral, ace_raw = run_segment(
        sa=sa,
        ctx=ctx,
        start_offset=start,
        duration_seconds=total_seconds,
        agc_interval=args.agc_interval,
        kp=args.kp,
        ki=args.ki,
        bf=bf,
        ace_integral=0.0,
        ace_raw=0.0,
        local_start=0.0,
        include_initial=True,
    )

    out_csv = args.results_dir / f"{label}_frequency.csv"
    pd.DataFrame({"time_s": t, "freq_dev_hz": f}).to_csv(out_csv, index=False)

    summary = pd.DataFrame([{
        "label": label,
        "first_label": first.label,
        "second_label": second.label,
        "kp": args.kp,
        "ki": args.ki,
        "agc_interval": args.agc_interval,
        "dispatch_interval": args.dispatch_interval,
        "samples": int(len(t)),
        "t_end_s": float(t[-1]),
        "min_hz": float(np.min(f)),
        "max_hz": float(np.max(f)),
        "final_hz": float(f[-1]),
        "abs_mean_hz": float(np.mean(np.abs(f))),
        "ace_integral_end": float(ace_integral),
        "ace_raw_end": float(ace_raw),
    }])
    summary_csv = args.results_dir / f"{label}_summary.csv"
    summary.to_csv(summary_csv, index=False)

    compare_summary_csv: Path | None = None
    compare_plot: Path | None = None
    if args.first_cold_csv is not None and args.second_cold_csv is not None:
        cold1 = pd.read_csv(args.first_cold_csv)
        cold2 = pd.read_csv(args.second_cold_csv)
        cold_x = np.concatenate([
            cold1["time_s"].to_numpy(dtype=float),
            cold2["time_s"].to_numpy(dtype=float) + args.dispatch_interval,
        ])
        cold_y = np.concatenate([
            cold1["freq_dev_hz"].to_numpy(dtype=float),
            cold2["freq_dev_hz"].to_numpy(dtype=float),
        ])

        continuous_f_899 = boundary_value(t, f, args.dispatch_interval - 1)
        continuous_f_900 = boundary_value(t, f, args.dispatch_interval)
        stitched_jump = float(cold2["freq_dev_hz"].iloc[0] - cold1["freq_dev_hz"].iloc[-1])
        continuous_step = float(continuous_f_900 - continuous_f_899)

        compare_summary = pd.DataFrame([{
            "stitched_end_first_hz": float(cold1["freq_dev_hz"].iloc[-1]),
            "stitched_start_second_hz": float(cold2["freq_dev_hz"].iloc[0]),
            "stitched_jump_hz": stitched_jump,
            "continuous_f_899_hz": continuous_f_899,
            "continuous_f_900_hz": continuous_f_900,
            "continuous_step_899_to_900_hz": continuous_step,
            "continuous_min_hz": float(np.min(f)),
            "continuous_max_hz": float(np.max(f)),
        }])
        compare_summary_csv = args.results_dir / f"{label}_vs_stitched_summary.csv"
        compare_summary.to_csv(compare_summary_csv, index=False)

        fig, axes = plt.subplots(2, 1, figsize=(15.5, 10.2), sharex=False)
        axes[0].plot(cold_x, cold_y, color="#b24c2a", linewidth=1.25, label="cold stitched")
        axes[0].plot(t, f, color="#0f5c78", linewidth=1.4, label="continuous 1800 s")
        axes[0].axvline(args.dispatch_interval, color="#666666", linestyle="--", linewidth=0.9)
        axes[0].axhline(0.0, color="#999999", linestyle=":", linewidth=0.8)
        axes[0].set_title(f"{first.label} -> {second.label}: cold stitched vs continuous run")
        axes[0].set_ylabel("Frequency deviation [Hz]")
        axes[0].grid(True, alpha=0.22)
        axes[0].legend(loc="upper right")

        axes[1].plot(cold_x, cold_y, color="#b24c2a", linewidth=1.35, label="cold stitched")
        axes[1].plot(t, f, color="#0f5c78", linewidth=1.45, label="continuous 1800 s")
        axes[1].axvline(args.dispatch_interval, color="#666666", linestyle="--", linewidth=0.9)
        axes[1].axhline(0.0, color="#999999", linestyle=":", linewidth=0.8)
        axes[1].set_xlim(args.dispatch_interval - 60, args.dispatch_interval + 120)
        axes[1].set_title("Zoom around the dispatch boundary")
        axes[1].set_xlabel("Combined time [s]")
        axes[1].set_ylabel("Frequency deviation [Hz]")
        axes[1].grid(True, alpha=0.22)
        axes[1].legend(loc="upper right")
        axes[1].text(
            0.985,
            0.05,
            f"stitched jump = {stitched_jump:+.4f} Hz\n"
            f"continuous step = {continuous_step:+.4f} Hz",
            transform=axes[1].transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.92),
        )
        fig.tight_layout()
        compare_plot = args.results_dir / f"{label}_vs_stitched.png"
        fig.savefig(compare_plot, dpi=220)
        plt.close(fig)

    manifest = {
        "first_dispatch_json": str(args.first_dispatch_json),
        "second_dispatch_json": str(args.second_dispatch_json),
        "first_cold_csv": str(args.first_cold_csv) if args.first_cold_csv is not None else "",
        "second_cold_csv": str(args.second_cold_csv) if args.second_cold_csv is not None else "",
        "kp": args.kp,
        "ki": args.ki,
        "agc_interval": args.agc_interval,
        "dispatch_interval": args.dispatch_interval,
        "init_mode": args.init_mode,
        "curve_file": str(args.curve_file),
        "dyn_case": str(args.dyn_case),
        "stable_dyn_case": str(dyn_case),
        "results_dir": str(args.results_dir),
        "frequency_csv": str(out_csv),
        "summary_csv": str(summary_csv),
        "compare_summary_csv": str(compare_summary_csv) if compare_summary_csv is not None else "",
        "compare_plot": str(compare_plot) if compare_plot is not None else "",
    }
    (args.results_dir / f"{label}_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"frequency_csv={out_csv}")
    print(f"summary_csv={summary_csv}")
    if compare_summary_csv is not None:
        print(f"compare_summary_csv={compare_summary_csv}")
    if compare_plot is not None:
        print(f"compare_plot={compare_plot}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
