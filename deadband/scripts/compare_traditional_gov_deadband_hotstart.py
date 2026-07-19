#!/usr/bin/env python3
"""
Compare one hot-start dispatch with and without conventional-governor deadband.

The intent is to isolate the primary-frequency-response deadband of traditional
governors while keeping:

- the same hot-start terminal state,
- the same AGC gains and dispatch-target trajectory,
- DER frequency deadband explicitly disabled.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
if "MPLCONFIGDIR" not in os.environ:
    _mpl_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "openandes-mpl"
    _mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

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
    parser.add_argument("--dispatch-json", type=Path, required=True)
    parser.add_argument("--next-dispatch-json", type=Path, default=None)
    parser.add_argument("--curve-file", type=Path, default=rdt.DEFAULT_CURVE_FILE)
    parser.add_argument("--results-dir", type=Path, default=rdt.RESULTS / "traditional_gov_deadband")
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--duration-seconds", type=int, default=900)
    parser.add_argument("--agc-interval", type=int, default=4)
    parser.add_argument("--kp", type=float, default=0.03)
    parser.add_argument("--ki", type=float, default=0.003)
    parser.add_argument("--governor-deadband-hz", type=float, default=0.036)
    parser.add_argument(
        "--governor-target-schedule",
        choices=("step", "boundary_ramp", "midpoint_trajectory"),
        default="midpoint_trajectory",
    )
    parser.add_argument("--save-plot", dest="save_plot", action="store_true")
    parser.add_argument("--no-save-plot", dest="save_plot", action="store_false")
    parser.set_defaults(save_plot=True)
    return parser.parse_args()


def disable_der_frequency_deadband(sa) -> dict[str, object]:
    touched: list[dict[str, object]] = []
    for model_name in ("PVD1", "ESD1"):
        if not hasattr(sa, model_name):
            continue
        mdl = getattr(sa, model_name)
        if mdl.n == 0:
            continue
        idx = mdl.idx.v
        zeros = np.zeros(mdl.n, dtype=float)
        for field in ("fdbd", "fdbdu", "ddn"):
            if hasattr(mdl, field):
                mdl.set(src=field, idx=idx, attr="v", value=zeros)
        touched.append({"model": model_name, "count": int(mdl.n)})
    return {"der_deadband_disabled": touched}


def apply_traditional_governor_deadband(sa, deadband_hz: float) -> dict[str, object]:
    db_pu = float(deadband_hz) / float(sa.config.freq)
    touched: list[dict[str, object]] = []

    for model_name in ("TGOV1NDB", "TGOV1DB", "HYGOVDB"):
        if not hasattr(sa, model_name):
            continue
        mdl = getattr(sa, model_name)
        if mdl.n == 0:
            continue
        idx = mdl.idx.v
        mdl.set(src="dbL", idx=idx, attr="v", value=np.full(mdl.n, -db_pu, dtype=float))
        mdl.set(src="dbU", idx=idx, attr="v", value=np.full(mdl.n, db_pu, dtype=float))
        r_values = mdl.get(src="R", attr="v", idx=idx)
        touched.append({
            "model": model_name,
            "count": int(mdl.n),
            "deadband_hz": float(deadband_hz),
            "deadband_pu": float(db_pu),
            "R_runtime_min": float(np.min(r_values)),
            "R_runtime_max": float(np.max(r_values)),
            "R_runtime_mean": float(np.mean(r_values)),
        })

    if not touched:
        raise RuntimeError("No conventional governor models with configurable deadband were found.")

    return {"traditional_governor_deadband": touched}


def summarize_variant(
    *,
    variant: str,
    t: np.ndarray,
    f_dev_hz: np.ndarray,
    deadband_hz: float,
) -> dict[str, object]:
    abs_f = np.abs(f_dev_hz)
    imin = int(np.argmin(f_dev_hz))
    imax = int(np.argmax(f_dev_hz))
    return {
        "variant": variant,
        "samples": int(len(t)),
        "min_hz": float(f_dev_hz[imin]),
        "t_min_s": float(t[imin]),
        "max_hz": float(f_dev_hz[imax]),
        "t_max_s": float(t[imax]),
        "final_hz": float(f_dev_hz[-1]),
        "abs_mean_hz": float(np.mean(abs_f)),
        "rms_hz": float(np.sqrt(np.mean(np.square(f_dev_hz)))),
        "share_pos_gt_deadband": float(np.mean(f_dev_hz > deadband_hz)),
        "share_neg_lt_deadband": float(np.mean(f_dev_hz < -deadband_hz)),
        "share_abs_gt_deadband": float(np.mean(abs_f > deadband_hz)),
    }


def run_one_variant(
    *,
    checkpoint_in: Path,
    curve: pd.DataFrame,
    dispatch_record: rdt.DispatchRecord,
    next_dispatch_record: rdt.DispatchRecord | None,
    duration_seconds: int,
    agc_interval: int,
    kp: float,
    ki: float,
    governor_target_schedule: str,
    governor_deadband_hz: float,
    enable_traditional_deadband: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    sa, stored_ctx, agc_state, _ = hcp.load_checkpoint(checkpoint_in)
    ctx = hcp.build_runtime_context(sa=sa, curve=curve, stored_ctx=stored_ctx)

    meta: dict[str, object] = {}
    meta.update(disable_der_frequency_deadband(sa))
    if enable_traditional_deadband:
        meta.update(apply_traditional_governor_deadband(sa, governor_deadband_hz))
    else:
        meta["traditional_governor_deadband"] = []

    transition = apply_second_dispatch_targets(
        sa,
        ctx["link"],  # type: ignore[arg-type]
        dispatch_record,
        apply_governor_targets=True,
        apply_dg_targets=False,
        duration_seconds=duration_seconds if governor_target_schedule == "midpoint_trajectory" else None,
        schedule_mode=governor_target_schedule,
        next_dispatch_record=(
            next_dispatch_record if governor_target_schedule == "midpoint_trajectory" else None
        ),
    )
    transition["ramp_seconds"] = 0
    activate_dispatch_target_transition(sa, transition, step=0)

    bf = compute_bf(sa, dispatch_record)
    t, f_dev_hz, ace_integral_end, ace_raw_end = run_segment(
        sa=sa,
        ctx=ctx,
        start_offset=dispatch_offset(dispatch_record, duration_seconds),
        duration_seconds=duration_seconds,
        agc_interval=agc_interval,
        kp=kp,
        ki=ki,
        bf=bf,
        ace_integral=float(agc_state["ace_integral"]),
        ace_raw=float(agc_state["ace_raw"]),
        local_start=0.0,
        include_initial=True,
        dispatch_target_transition=transition,
    )

    meta["ace_integral_end"] = float(ace_integral_end)
    meta["ace_raw_end"] = float(ace_raw_end)
    return t, f_dev_hz, meta


def main() -> None:
    args = parse_args()
    rdt.andes.config_logger(stream_level=30)

    curve = rdt.load_curve(args.curve_file)
    dispatch_record = rdt.DispatchRecord.from_json(args.dispatch_json)
    next_dispatch_record = (
        rdt.DispatchRecord.from_json(args.next_dispatch_json)
        if args.next_dispatch_json is not None else None
    )
    label = args.label or f"{dispatch_record.label}_traditional_gov_deadband"
    args.results_dir.mkdir(parents=True, exist_ok=True)

    baseline_t, baseline_f, baseline_meta = run_one_variant(
        checkpoint_in=args.checkpoint_in,
        curve=curve,
        dispatch_record=dispatch_record,
        next_dispatch_record=next_dispatch_record,
        duration_seconds=args.duration_seconds,
        agc_interval=args.agc_interval,
        kp=args.kp,
        ki=args.ki,
        governor_target_schedule=args.governor_target_schedule,
        governor_deadband_hz=args.governor_deadband_hz,
        enable_traditional_deadband=False,
    )

    deadband_t, deadband_f, deadband_meta = run_one_variant(
        checkpoint_in=args.checkpoint_in,
        curve=curve,
        dispatch_record=dispatch_record,
        next_dispatch_record=next_dispatch_record,
        duration_seconds=args.duration_seconds,
        agc_interval=args.agc_interval,
        kp=args.kp,
        ki=args.ki,
        governor_target_schedule=args.governor_target_schedule,
        governor_deadband_hz=args.governor_deadband_hz,
        enable_traditional_deadband=True,
    )

    baseline_csv = args.results_dir / f"{label}_baseline_frequency.csv"
    deadband_csv = args.results_dir / f"{label}_traditional_deadband_frequency.csv"
    pd.DataFrame({"time_s": baseline_t, "freq_dev_hz": baseline_f}).to_csv(baseline_csv, index=False)
    pd.DataFrame({"time_s": deadband_t, "freq_dev_hz": deadband_f}).to_csv(deadband_csv, index=False)

    summary = pd.DataFrame([
        summarize_variant(
            variant="baseline_no_traditional_deadband",
            t=baseline_t,
            f_dev_hz=baseline_f,
            deadband_hz=args.governor_deadband_hz,
        ),
        summarize_variant(
            variant="traditional_deadband_on",
            t=deadband_t,
            f_dev_hz=deadband_f,
            deadband_hz=args.governor_deadband_hz,
        ),
    ])
    summary["delta_vs_baseline_abs_mean_hz"] = (
        summary["abs_mean_hz"] - float(summary.loc[summary["variant"] == "baseline_no_traditional_deadband", "abs_mean_hz"].iloc[0])
    )
    summary["delta_vs_baseline_rms_hz"] = (
        summary["rms_hz"] - float(summary.loc[summary["variant"] == "baseline_no_traditional_deadband", "rms_hz"].iloc[0])
    )
    summary["delta_vs_baseline_share_abs_gt_deadband"] = (
        summary["share_abs_gt_deadband"] - float(
            summary.loc[summary["variant"] == "baseline_no_traditional_deadband", "share_abs_gt_deadband"].iloc[0]
        )
    )
    summary_csv = args.results_dir / f"{label}_summary.csv"
    summary.to_csv(summary_csv, index=False)

    diff = deadband_f - baseline_f
    diff_csv = args.results_dir / f"{label}_difference.csv"
    pd.DataFrame({
        "time_s": baseline_t,
        "freq_dev_hz_baseline": baseline_f,
        "freq_dev_hz_traditional_deadband": deadband_f,
        "delta_hz": diff,
    }).to_csv(diff_csv, index=False)

    plot_path = args.results_dir / f"{label}_comparison.png"
    if args.save_plot:
        fig, axes = plt.subplots(2, 1, figsize=(14.5, 9.0), sharex=True)

        axes[0].plot(baseline_t, baseline_f, color="#0f5c78", linewidth=1.5, label="baseline")
        axes[0].plot(deadband_t, deadband_f, color="#b24c2a", linewidth=1.5, label="traditional deadband on")
        axes[0].axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")
        axes[0].axhline(args.governor_deadband_hz, color="#888888", linewidth=0.9, linestyle=":")
        axes[0].axhline(-args.governor_deadband_hz, color="#888888", linewidth=0.9, linestyle=":")
        axes[0].fill_between(
            baseline_t,
            -args.governor_deadband_hz,
            args.governor_deadband_hz,
            color="#ebe3d4",
            alpha=0.35,
        )
        axes[0].set_title(
            f"{dispatch_record.label}: traditional-governor deadband only "
            f"(db = +/-{args.governor_deadband_hz:.3f} Hz)"
        )
        axes[0].set_ylabel("Frequency deviation [Hz]")
        axes[0].grid(True, alpha=0.22)
        axes[0].legend(frameon=False, loc="upper right")

        axes[1].plot(baseline_t, diff, color="#5b3f8c", linewidth=1.4)
        axes[1].axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")
        axes[1].set_title("Deadband-on minus baseline")
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("Delta f [Hz]")
        axes[1].grid(True, alpha=0.22)
        axes[1].text(
            0.985,
            0.04,
            "\n".join([
                f"max |delta| = {np.max(np.abs(diff)):.4f} Hz",
                f"baseline share(|f|>{args.governor_deadband_hz:.3f}) = {summary.loc[0, 'share_abs_gt_deadband']:.2%}",
                f"deadband share(|f|>{args.governor_deadband_hz:.3f}) = {summary.loc[1, 'share_abs_gt_deadband']:.2%}",
            ]),
            transform=axes[1].transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92, edgecolor="#cccccc"),
        )

        fig.tight_layout()
        fig.savefig(plot_path, dpi=220)
        plt.close(fig)

    config = {
        "checkpoint_in": str(args.checkpoint_in),
        "dispatch_json": str(args.dispatch_json),
        "next_dispatch_json": str(args.next_dispatch_json) if args.next_dispatch_json else "",
        "curve_file": str(args.curve_file),
        "kp": float(args.kp),
        "ki": float(args.ki),
        "agc_interval": int(args.agc_interval),
        "duration_seconds": int(args.duration_seconds),
        "governor_deadband_hz": float(args.governor_deadband_hz),
        "governor_target_schedule": str(args.governor_target_schedule),
        "baseline_meta": baseline_meta,
        "traditional_deadband_meta": deadband_meta,
        "baseline_csv": str(baseline_csv),
        "traditional_deadband_csv": str(deadband_csv),
        "difference_csv": str(diff_csv),
        "summary_csv": str(summary_csv),
        "plot_path": str(plot_path),
    }
    config_json = args.results_dir / f"{label}_config.json"
    config_json.write_text(json.dumps(config, indent=2))

    print(f"baseline_csv={baseline_csv}")
    print(f"traditional_deadband_csv={deadband_csv}")
    print(f"difference_csv={diff_csv}")
    print(f"summary_csv={summary_csv}")
    if args.save_plot:
        print(f"plot_png={plot_path}")
    print(f"config_json={config_json}")


if __name__ == "__main__":
    main()
