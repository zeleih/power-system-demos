#!/usr/bin/env python3
"""
Compare a midpoint-trajectory hot-start pair against a continuous two-segment run.

This script is designed for the deadband demo workflow where:

- dispatch ``k`` starts from the terminal checkpoint of dispatch ``k-1``
- conventional generator governor targets follow ``midpoint_trajectory``
- the user wants to verify that
  1) running ``k`` and ``k+1`` as separate hot-start segments and stitching them
     together, and
  2) running the same two segments continuously from the same checkpoint
     produce nearly identical frequency trajectories
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hotstart_checkpoint as hcp
import run_dispatch_tds as rdt
from compare_dispatch_pair_hotstart import (
    activate_dispatch_target_transition,
    apply_second_dispatch_targets,
    compute_bf,
    dispatch_offset,
    run_segment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-in", type=Path, required=True)
    parser.add_argument("--first-dispatch-json", type=Path, required=True)
    parser.add_argument("--second-dispatch-json", type=Path, required=True)
    parser.add_argument("--third-dispatch-json", type=Path, required=True)
    parser.add_argument("--first-hotstart-csv", type=Path, required=True)
    parser.add_argument("--second-hotstart-csv", type=Path, required=True)
    parser.add_argument("--curve-file", type=Path, default=rdt.DEFAULT_CURVE_FILE)
    parser.add_argument("--results-dir", type=Path, default=rdt.RESULTS / "trajectory_chain" / "pair_compare")
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--kp", type=float, default=0.03)
    parser.add_argument("--ki", type=float, default=0.01)
    parser.add_argument("--agc-interval", type=int, default=4)
    parser.add_argument("--dispatch-interval", type=int, default=900)
    return parser.parse_args()


def summarize_series(t: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    imin = int(np.argmin(y))
    imax = int(np.argmax(y))
    return {
        "samples": int(len(t)),
        "t_end_s": float(t[-1]),
        "min_hz": float(y[imin]),
        "t_min_s": float(t[imin]),
        "max_hz": float(y[imax]),
        "t_max_s": float(t[imax]),
        "final_hz": float(y[-1]),
        "abs_mean_hz": float(np.mean(np.abs(y))),
        "rms_hz": float(np.sqrt(np.mean(np.square(y)))),
    }


def load_series_pair(first_csv: Path, second_csv: Path, dispatch_interval: int) -> pd.DataFrame:
    first = pd.read_csv(first_csv)
    second = pd.read_csv(second_csv)
    out = pd.concat(
        [
            pd.DataFrame(
                {
                    "time_s": first["time_s"].to_numpy(dtype=float),
                    "freq_dev_hz": first["freq_dev_hz"].to_numpy(dtype=float),
                }
            ),
            pd.DataFrame(
                {
                    "time_s": second["time_s"].to_numpy(dtype=float) + float(dispatch_interval),
                    "freq_dev_hz": second["freq_dev_hz"].to_numpy(dtype=float),
                }
            ),
        ],
        ignore_index=True,
    )
    out["time_key"] = np.rint(out["time_s"]).astype(int)
    return out


def run_dispatch_window(
    *,
    sa,
    ctx: dict[str, object],
    dispatch_record: rdt.DispatchRecord,
    next_dispatch_record: rdt.DispatchRecord,
    ace_integral: float,
    ace_raw: float,
    kp: float,
    ki: float,
    agc_interval: int,
    dispatch_interval: int,
    local_start: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    transition = apply_second_dispatch_targets(
        sa,
        ctx["link"],  # type: ignore[arg-type]
        dispatch_record,
        apply_governor_targets=True,
        apply_dg_targets=False,
        duration_seconds=dispatch_interval,
        schedule_mode="midpoint_trajectory",
        next_dispatch_record=next_dispatch_record,
    )
    activate_dispatch_target_transition(sa, transition, step=0)
    bf = compute_bf(sa, dispatch_record)
    return run_segment(
        sa=sa,
        ctx=ctx,
        start_offset=dispatch_offset(dispatch_record, dispatch_interval),
        duration_seconds=dispatch_interval,
        agc_interval=agc_interval,
        kp=kp,
        ki=ki,
        bf=bf,
        ace_integral=ace_integral,
        ace_raw=ace_raw,
        local_start=local_start,
        include_initial=True,
        dispatch_target_transition=transition,
    )


def main() -> None:
    args = parse_args()
    rdt.andes.config_logger(stream_level=30)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    curve = rdt.load_curve(args.curve_file)
    first = rdt.DispatchRecord.from_json(args.first_dispatch_json)
    second = rdt.DispatchRecord.from_json(args.second_dispatch_json)
    third = rdt.DispatchRecord.from_json(args.third_dispatch_json)
    label = args.label or f"{first.label}_{second.label}_midpoint_compare"

    sa, stored_ctx, agc_state, _manifest = hcp.load_checkpoint(args.checkpoint_in)
    ctx = hcp.build_runtime_context(sa=sa, curve=curve, stored_ctx=stored_ctx)
    ace_integral = float(agc_state["ace_integral"])
    ace_raw = float(agc_state["ace_raw"])

    t1, f1, ace_integral, ace_raw = run_dispatch_window(
        sa=sa,
        ctx=ctx,
        dispatch_record=first,
        next_dispatch_record=second,
        ace_integral=ace_integral,
        ace_raw=ace_raw,
        kp=args.kp,
        ki=args.ki,
        agc_interval=args.agc_interval,
        dispatch_interval=args.dispatch_interval,
        local_start=0.0,
    )
    t2, f2, ace_integral, ace_raw = run_dispatch_window(
        sa=sa,
        ctx=ctx,
        dispatch_record=second,
        next_dispatch_record=third,
        ace_integral=ace_integral,
        ace_raw=ace_raw,
        kp=args.kp,
        ki=args.ki,
        agc_interval=args.agc_interval,
        dispatch_interval=args.dispatch_interval,
        local_start=float(args.dispatch_interval),
    )

    t_cont = np.concatenate([t1, t2])
    f_cont = np.concatenate([f1, f2])
    continuous_df = pd.DataFrame({"time_s": t_cont, "freq_dev_hz": f_cont})
    continuous_df["time_key"] = np.rint(continuous_df["time_s"]).astype(int)

    stitched_df = load_series_pair(
        args.first_hotstart_csv,
        args.second_hotstart_csv,
        args.dispatch_interval,
    )

    merged = continuous_df.merge(
        stitched_df[["time_key", "freq_dev_hz"]].rename(columns={"freq_dev_hz": "hotstart_freq_dev_hz"}),
        on="time_key",
        how="inner",
    )
    merged = merged.rename(columns={"freq_dev_hz": "continuous_freq_dev_hz"})
    merged["diff_hz"] = merged["hotstart_freq_dev_hz"] - merged["continuous_freq_dev_hz"]

    continuous_csv = args.results_dir / f"{label}_continuous_frequency.csv"
    stitched_csv = args.results_dir / f"{label}_hotstart_stitched_frequency.csv"
    diff_csv = args.results_dir / f"{label}_diff.csv"
    continuous_df[["time_s", "freq_dev_hz"]].to_csv(continuous_csv, index=False)
    stitched_df[["time_s", "freq_dev_hz"]].to_csv(stitched_csv, index=False)
    merged.to_csv(diff_csv, index=False)

    summary = {
        "label": label,
        "checkpoint_in": str(args.checkpoint_in),
        "first_dispatch_json": str(args.first_dispatch_json),
        "second_dispatch_json": str(args.second_dispatch_json),
        "third_dispatch_json": str(args.third_dispatch_json),
        "first_hotstart_csv": str(args.first_hotstart_csv),
        "second_hotstart_csv": str(args.second_hotstart_csv),
        "kp": float(args.kp),
        "ki": float(args.ki),
        "agc_interval": int(args.agc_interval),
        "dispatch_interval": int(args.dispatch_interval),
    }
    summary.update({f"continuous_{k}": v for k, v in summarize_series(t_cont, f_cont).items()})
    summary.update({f"hotstart_{k}": v for k, v in summarize_series(stitched_df["time_s"].to_numpy(), stitched_df["freq_dev_hz"].to_numpy()).items()})
    summary.update(
        {
            "diff_max_abs_hz": float(np.max(np.abs(merged["diff_hz"].to_numpy()))),
            "diff_rms_hz": float(np.sqrt(np.mean(np.square(merged["diff_hz"].to_numpy())))),
            "diff_final_hz": float(merged["diff_hz"].iloc[-1]),
        }
    )

    summary_csv = args.results_dir / f"{label}_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    fig, axes = plt.subplots(3, 1, figsize=(17.5, 13.5), sharex=False)

    axes[0].plot(
        stitched_df["time_s"],
        stitched_df["freq_dev_hz"],
        color="#b24c2a",
        linewidth=1.5,
        label="hot-start stitched",
    )
    axes[0].plot(
        continuous_df["time_s"],
        continuous_df["freq_dev_hz"],
        color="#0f5c78",
        linewidth=1.4,
        label="continuous replay",
    )
    axes[0].axvline(args.dispatch_interval, color="#666666", linestyle="--", linewidth=0.9)
    axes[0].axhline(0.0, color="#999999", linestyle=":", linewidth=0.8)
    axes[0].set_title(f"{first.label} + {second.label}: midpoint hot-start stitched vs continuous replay")
    axes[0].set_ylabel("Frequency deviation [Hz]")
    axes[0].grid(True, alpha=0.22)
    axes[0].legend(loc="upper right")

    axes[1].plot(
        stitched_df["time_s"],
        stitched_df["freq_dev_hz"],
        color="#b24c2a",
        linewidth=1.55,
        label="hot-start stitched",
    )
    axes[1].plot(
        continuous_df["time_s"],
        continuous_df["freq_dev_hz"],
        color="#0f5c78",
        linewidth=1.45,
        label="continuous replay",
    )
    axes[1].axvline(args.dispatch_interval, color="#666666", linestyle="--", linewidth=0.9)
    axes[1].axhline(0.0, color="#999999", linestyle=":", linewidth=0.8)
    axes[1].set_xlim(args.dispatch_interval - 120, args.dispatch_interval + 180)
    axes[1].set_title("Zoom around the dispatch boundary")
    axes[1].set_ylabel("Frequency deviation [Hz]")
    axes[1].grid(True, alpha=0.22)
    axes[1].legend(loc="upper right")

    axes[2].plot(
        merged["time_key"],
        1000.0 * merged["diff_hz"],
        color="#385f3b",
        linewidth=1.2,
    )
    axes[2].axvline(args.dispatch_interval, color="#666666", linestyle="--", linewidth=0.9)
    axes[2].axhline(0.0, color="#999999", linestyle=":", linewidth=0.8)
    axes[2].set_title("Hot-start minus continuous difference")
    axes[2].set_xlabel("Combined time [s]")
    axes[2].set_ylabel("Difference [mHz]")
    axes[2].grid(True, alpha=0.22)
    axes[2].text(
        0.985,
        0.05,
        f"max |diff| = {1000.0 * summary['diff_max_abs_hz']:.3f} mHz\n"
        f"RMS diff = {1000.0 * summary['diff_rms_hz']:.3f} mHz",
        transform=axes[2].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )

    fig.tight_layout()
    plot_path = args.results_dir / f"{label}_hotstart_vs_continuous.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    manifest = {
        "label": label,
        "results_dir": str(args.results_dir),
        "continuous_csv": str(continuous_csv),
        "stitched_csv": str(stitched_csv),
        "diff_csv": str(diff_csv),
        "summary_csv": str(summary_csv),
        "plot_path": str(plot_path),
        "summary": summary,
    }
    (args.results_dir / f"{label}_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"continuous_csv={continuous_csv}")
    print(f"stitched_csv={stitched_csv}")
    print(f"diff_csv={diff_csv}")
    print(f"summary_csv={summary_csv}")
    print(f"plot_path={plot_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
