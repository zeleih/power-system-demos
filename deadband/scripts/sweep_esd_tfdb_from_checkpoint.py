#!/usr/bin/env python3
"""
Sweep ESD droop lag (Tfdb) from checkpoint-replayed segments.

This helper replays selected hot-start segments from existing checkpoints so we
can compare frequency and ESD output behavior without rerunning the full
preceding warmup segment each time.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import hotstart_checkpoint as hcp
import run_dispatch_tds as rdt
from compare_dispatch_pair_hotstart import (
    AGC_ANTI_WINDUP_FREEZE,
    activate_dispatch_target_transition,
    apply_second_dispatch_targets,
    compute_bf,
    dispatch_offset,
    prepare_system,
    run_segment,
)
from plot_hotstart_droop_breakdown import trace_second_segment


SEGMENT_SPECS = (
    ("h15d1", "end_h15d0", "h15d0_dispatch.json", "h15d1_dispatch.json"),
    ("h15d2", "end_h15d1", "h15d1_dispatch.json", "h15d2_dispatch.json"),
    ("h21d0", "end_h20d3", "h20d3_dispatch.json", "h21d0_dispatch.json"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dispatch-dir", type=Path, required=True)
    parser.add_argument("--curve-file", type=Path, required=True)
    parser.add_argument("--dyn-case", type=Path, required=True)
    parser.add_argument("--stable-dyn-case", type=Path, default=rdt.DEFAULT_STABLE_DYN_CASE)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--segments", nargs="*", default=None)
    parser.add_argument("--dispatch-interval", type=int, default=900)
    parser.add_argument("--agc-interval", type=int, default=4)
    parser.add_argument("--kp", type=float, default=0.1)
    parser.add_argument("--ki", type=float, default=0.004)
    parser.add_argument("--pvd-ddn", type=float, default=0.0)
    parser.add_argument("--esd-ddn", type=float, default=1.6666666667)
    parser.add_argument("--tfdb-start", type=float, default=1.0)
    parser.add_argument("--tfdb-stop", type=float, default=5.0)
    parser.add_argument("--tfdb-step", type=float, default=0.5)
    parser.add_argument("--include-zero-baseline", action="store_true")
    parser.add_argument(
        "--governor-target-schedule",
        choices=("step", "boundary_ramp", "midpoint_trajectory", "ramp_limited_basepoint"),
        default="ramp_limited_basepoint",
    )
    parser.add_argument(
        "--governor-basepoint-ramp-floor-frac-pmax-per-min",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--governor-basepoint-ramp-gap-factor",
        type=float,
        default=1.25,
    )
    return parser.parse_args()


def tfdb_grid(start: float, stop: float, step: float, include_zero_baseline: bool) -> list[float]:
    values: list[float] = []
    if include_zero_baseline:
        values.append(0.0)
    n = int(round((stop - start) / step))
    for idx in range(n + 1):
        values.append(round(start + idx * step, 10))
    return values


def tfdb_label(value: float) -> str:
    if math.isclose(value, 0.0):
        return "0p0"
    text = f"{value:.1f}".replace(".", "p")
    return text


def replay_segment(
    *,
    checkpoint_dir: Path,
    first_dispatch_json: Path,
    second_dispatch_json: Path,
    curve: pd.DataFrame,
    dyn_case: Path,
    dispatch_interval: int,
    agc_interval: int,
    kp: float,
    ki: float,
    pvd_ddn: float,
    esd_ddn: float,
    esd_tfdb: float,
    governor_target_schedule: str,
    governor_basepoint_ramp_floor_frac_pmax_per_min: float,
    governor_basepoint_ramp_gap_factor: float,
) -> pd.DataFrame:
    del checkpoint_dir
    first = rdt.DispatchRecord.from_json(first_dispatch_json)
    second = rdt.DispatchRecord.from_json(second_dispatch_json)
    rdt.validate_curve_window(curve, first, dispatch_interval)
    rdt.validate_curve_window(curve, second, dispatch_interval)

    sa, ctx = prepare_system(
        dispatch_record=first,
        curve=curve,
        dyn_case=dyn_case,
        dispatch_interval=dispatch_interval,
        init_mode="first",
        wind_prefixes=rdt.DEFAULT_WIND_PREFIXES,
        solar_prefixes=rdt.DEFAULT_SOLAR_PREFIXES,
    )
    ctx["link"] = rdt.configure_der_agc_participation(
        sa,
        ctx["link"],  # type: ignore[arg-type]
        enable_der_agc=False,
    )

    if hasattr(sa, "PVD1") and sa.PVD1.n:
        sa.PVD1.set(src="ddn", idx=sa.PVD1.idx.v, attr="v", value=np.full(sa.PVD1.n, float(pvd_ddn)))
        sa.PVD1.set(src="Tfdb", idx=sa.PVD1.idx.v, attr="v", value=np.zeros(sa.PVD1.n))
    if hasattr(sa, "ESD1") and sa.ESD1.n:
        sa.ESD1.set(src="ddn", idx=sa.ESD1.idx.v, attr="v", value=np.full(sa.ESD1.n, float(esd_ddn)))
        sa.ESD1.set(src="Tfdb", idx=sa.ESD1.idx.v, attr="v", value=np.full(sa.ESD1.n, float(esd_tfdb)))

    first_transition = apply_second_dispatch_targets(
        sa,
        ctx["link"],  # type: ignore[arg-type]
        first,
        apply_governor_targets=True,
        apply_dg_targets=False,
        duration_seconds=dispatch_interval,
        schedule_mode=governor_target_schedule,
        next_dispatch_record=None,
        basepoint_ramp_floor_frac_pmax_per_min=governor_basepoint_ramp_floor_frac_pmax_per_min,
        basepoint_ramp_gap_factor=governor_basepoint_ramp_gap_factor,
    )
    first_transition["ramp_seconds"] = 0
    activate_dispatch_target_transition(sa, first_transition, step=0)

    bf_first = compute_bf(sa, first)
    _, _, ace_integral_end, ace_raw_end = run_segment(
        sa=sa,
        ctx=ctx,
        start_offset=dispatch_offset(first, dispatch_interval),
        duration_seconds=dispatch_interval,
        agc_interval=agc_interval,
        kp=kp,
        ki=ki,
        bf=bf_first,
        ace_integral=0.0,
        ace_raw=0.0,
        local_start=0.0,
        include_initial=True,
        dispatch_target_transition=first_transition,
        gov_output_ramp_frac_pmax_per_min=0.0,
        dg_output_ramp_frac_pmax_per_min=0.0,
        agc_anti_windup_mode=AGC_ANTI_WINDUP_FREEZE,
    )

    ctx2 = ctx.copy()
    ctx2["link"] = rdt.configure_der_agc_participation(
        sa,
        rdt.build_andes_link(sa),
        enable_der_agc=False,
    )
    second_transition = apply_second_dispatch_targets(
        sa,
        ctx2["link"],  # type: ignore[arg-type]
        second,
        apply_governor_targets=True,
        apply_dg_targets=False,
        duration_seconds=dispatch_interval,
        schedule_mode=governor_target_schedule,
        next_dispatch_record=None,
        basepoint_ramp_floor_frac_pmax_per_min=governor_basepoint_ramp_floor_frac_pmax_per_min,
        basepoint_ramp_gap_factor=governor_basepoint_ramp_gap_factor,
    )
    second_transition["ramp_seconds"] = 0
    activate_dispatch_target_transition(sa, second_transition, step=0)

    bf = compute_bf(sa, second)
    trace = trace_second_segment(
        sa=sa,
        ctx=ctx2,
        dispatch_record=second,
        duration_seconds=dispatch_interval,
        agc_interval=agc_interval,
        kp=kp,
        ki=ki,
        bf=bf,
        ace_integral=ace_integral_end,
        ace_raw=ace_raw_end,
        dispatch_target_transition=second_transition,
        gov_output_ramp_frac_pmax_per_min=0.0,
        dg_output_ramp_frac_pmax_per_min=0.0,
        agc_anti_windup_mode=AGC_ANTI_WINDUP_FREEZE,
    )
    return trace


def summarize_trace(segment: str, tfdb: float, trace: pd.DataFrame) -> dict[str, float | str]:
    freq = trace["freq_dev_hz"].to_numpy(dtype=float)
    esd_dby = trace["esd_droop_sum"].to_numpy(dtype=float)
    esd_pe = trace["esd_pe_sum"].to_numpy(dtype=float)
    return {
        "segment": segment,
        "esd_tfdb_s": float(tfdb),
        "freq_mean_hz": float(freq.mean()),
        "freq_abs_mean_hz": float(np.abs(freq).mean()),
        "freq_p95_abs_hz": float(np.quantile(np.abs(freq), 0.95)),
        "freq_p99_abs_hz": float(np.quantile(np.abs(freq), 0.99)),
        "freq_max_abs_hz": float(np.abs(freq).max()),
        "share_abs_gt_0p036": float((np.abs(freq) > 0.036).mean()),
        "share_abs_gt_0p05": float((np.abs(freq) > 0.05).mean()),
        "esd_dby_abs_max": float(np.abs(esd_dby).max()),
        "esd_dby_step_abs_max": float(np.abs(np.diff(esd_dby)).max()) if len(esd_dby) > 1 else 0.0,
        "esd_pe_abs_max": float(np.abs(esd_pe).max()),
        "esd_pe_step_abs_max": float(np.abs(np.diff(esd_pe)).max()) if len(esd_pe) > 1 else 0.0,
    }


def make_segment_plot(segment: str, traces: dict[float, pd.DataFrame], out_path: Path) -> None:
    ordered = sorted(traces.items(), key=lambda item: item[0])
    cmap = plt.get_cmap("viridis")
    nonzero = [item for item in ordered if not math.isclose(item[0], 0.0)]
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), constrained_layout=True, sharex=True)

    for idx, (tfdb, trace) in enumerate(nonzero):
        color = cmap(idx / max(len(nonzero) - 1, 1))
        label = f"Tfdb={tfdb:.1f}s"
        axes[0].plot(trace["time_s"], trace["freq_dev_hz"], color=color, linewidth=1.5, label=label)
        axes[1].plot(trace["time_s"], trace["esd_droop_sum"], color=color, linewidth=1.5)
        axes[2].plot(trace["time_s"], trace["esd_pe_sum"], color=color, linewidth=1.5)

    if ordered and math.isclose(ordered[0][0], 0.0):
        base = ordered[0][1]
        axes[0].plot(base["time_s"], base["freq_dev_hz"], color="black", linewidth=2.0, linestyle="--", label="Tfdb=0.0s")
        axes[1].plot(base["time_s"], base["esd_droop_sum"], color="black", linewidth=2.0, linestyle="--")
        axes[2].plot(base["time_s"], base["esd_pe_sum"], color="black", linewidth=2.0, linestyle="--")

    axes[0].axhline(0.036, color="#666666", linestyle=":", linewidth=1.0)
    axes[0].axhline(-0.036, color="#666666", linestyle=":", linewidth=1.0)
    axes[0].axhline(0.0, color="#999999", linestyle="-", linewidth=0.8)
    axes[1].axhline(0.0, color="#999999", linestyle="-", linewidth=0.8)

    axes[0].set_title(f"{segment} | PVD off, ESD on | frequency vs ESD lag")
    axes[0].set_ylabel("Freq [Hz]")
    axes[1].set_ylabel("ESD DB_y [pu]")
    axes[2].set_ylabel("ESD Pe [pu]")
    axes[2].set_xlabel("Time [s]")
    axes[0].legend(loc="upper right", ncol=2, frameon=True)

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0.0, 899.0)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_metrics_plot(summary: pd.DataFrame, out_path: Path) -> None:
    segments = list(summary["segment"].drop_duplicates())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True, sharex=True)
    metrics = [
        ("freq_abs_mean_hz", "Mean |f| [Hz]"),
        ("freq_p95_abs_hz", "P95 |f| [Hz]"),
        ("share_abs_gt_0p036", "Share |f| > 0.036"),
        ("esd_dby_abs_max", "Max |ESD DB_y| [pu]"),
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        for color, segment in zip(colors, segments):
            part = summary[summary["segment"] == segment].sort_values("esd_tfdb_s")
            ax.plot(part["esd_tfdb_s"], part[metric], marker="o", linewidth=1.6, color=color, label=segment)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(loc="upper right", frameon=True)
    axes[1, 0].set_xlabel("ESD Tfdb [s]")
    axes[1, 1].set_xlabel("ESD Tfdb [s]")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    curve = rdt.load_curve(args.curve_file)
    tfdb_values = tfdb_grid(args.tfdb_start, args.tfdb_stop, args.tfdb_step, args.include_zero_baseline)

    all_rows: list[dict[str, float | str]] = []
    traces_by_segment: dict[str, dict[float, pd.DataFrame]] = {}

    dyn_case = rdt.adapt_dyn_case(args.dyn_case, args.stable_dyn_case)

    allowed_segments = None if not args.segments else set(args.segments)

    for segment, checkpoint_name, first_dispatch_name, second_dispatch_name in SEGMENT_SPECS:
        if allowed_segments is not None and segment not in allowed_segments:
            continue
        checkpoint_dir = args.checkpoint_root / checkpoint_name
        first_dispatch_json = args.dispatch_dir / first_dispatch_name
        second_dispatch_json = args.dispatch_dir / second_dispatch_name
        traces_by_segment[segment] = {}
        segment_dir = args.results_dir / segment
        segment_dir.mkdir(parents=True, exist_ok=True)

        for tfdb in tfdb_values:
            trace = replay_segment(
                checkpoint_dir=checkpoint_dir,
                first_dispatch_json=first_dispatch_json,
                second_dispatch_json=second_dispatch_json,
                curve=curve,
                dyn_case=dyn_case,
                dispatch_interval=args.dispatch_interval,
                agc_interval=args.agc_interval,
                kp=args.kp,
                ki=args.ki,
                pvd_ddn=args.pvd_ddn,
                esd_ddn=args.esd_ddn,
                esd_tfdb=tfdb,
                governor_target_schedule=args.governor_target_schedule,
                governor_basepoint_ramp_floor_frac_pmax_per_min=args.governor_basepoint_ramp_floor_frac_pmax_per_min,
                governor_basepoint_ramp_gap_factor=args.governor_basepoint_ramp_gap_factor,
            )
            traces_by_segment[segment][tfdb] = trace
            trace.to_csv(segment_dir / f"trace_tfdb_{tfdb_label(tfdb)}.csv", index=False)
            all_rows.append(summarize_trace(segment, tfdb, trace))

        make_segment_plot(
            segment,
            traces_by_segment[segment],
            args.results_dir / f"{segment}_freq_esd_curves.png",
        )

    summary = pd.DataFrame(all_rows).sort_values(["segment", "esd_tfdb_s"]).reset_index(drop=True)
    summary.to_csv(args.results_dir / "esd_tfdb_sweep_summary.csv", index=False)
    make_metrics_plot(summary, args.results_dir / "esd_tfdb_sweep_metrics.png")


if __name__ == "__main__":
    main()
