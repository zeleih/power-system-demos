#!/usr/bin/env python3
"""
Study how dispatch-boundary actions affect frequency for one dispatch pair.

The experiment decomposes the second-segment boundary behavior into:

1. no boundary dispatch action: keep the first-segment AGC participation factors
   and do not apply new governor targets;
2. AGC redistribution only: switch to the second-segment participation factors
   but still do not apply new governor targets;
3. governor dispatch target step: switch participation factors and apply the new
   conventional-generator targets immediately at the boundary;
4. governor dispatch target ramp: same as (3) but ramp the targets over a
   configurable number of seconds.

DG/PVD1/ESD1 dispatch targets are intentionally excluded.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from andes.utils.snapshot import load_ss, save_ss

import hotstart_checkpoint as hcp
import run_dispatch_tds as rdt
from compare_dispatch_pair_hotstart import (
    activate_dispatch_target_transition,
    apply_second_dispatch_targets,
    compute_bf,
    dispatch_offset,
    prepare_system,
    run_segment,
)


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    second_bf_source: str
    apply_governor_targets: bool
    ramp_seconds: int
    color: str


VARIANTS = (
    Variant(
        name="carryover",
        label="carry-over (bf=first, no target)",
        second_bf_source="first",
        apply_governor_targets=False,
        ramp_seconds=0,
        color="#5b3f8c",
    ),
    Variant(
        name="bf_only",
        label="AGC redistribution only (bf=second)",
        second_bf_source="second",
        apply_governor_targets=False,
        ramp_seconds=0,
        color="#1f7a8c",
    ),
    Variant(
        name="gov_step",
        label="gov target step",
        second_bf_source="second",
        apply_governor_targets=True,
        ramp_seconds=0,
        color="#c05621",
    ),
    Variant(
        name="gov_ramp",
        label="gov target ramp",
        second_bf_source="second",
        apply_governor_targets=True,
        ramp_seconds=300,
        color="#2f855a",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-dispatch-json", type=Path, required=True)
    parser.add_argument("--second-dispatch-json", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=rdt.RESULTS / "dispatch_target_effect")
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--dispatch-interval", type=int, default=900)
    parser.add_argument("--agc-interval", type=int, default=4)
    parser.add_argument("--kp", type=float, default=0.03)
    parser.add_argument("--ki", type=float, default=0.01)
    parser.add_argument("--init-mode", choices=("dispatch", "first"), default="first")
    parser.add_argument("--resume-mode", choices=("memory", "snapshot"), default="snapshot")
    parser.add_argument("--ramp-seconds", type=int, default=300)
    parser.add_argument("--dyn-case", type=Path, default=rdt.DEFAULT_DYN_CASE)
    parser.add_argument("--stable-dyn-case", type=Path, default=rdt.DEFAULT_STABLE_DYN_CASE)
    parser.add_argument("--curve-file", type=Path, default=rdt.DEFAULT_CURVE_FILE)
    parser.add_argument("--zoom-before", type=int, default=60)
    parser.add_argument("--zoom-after", type=int, default=180)
    return parser.parse_args()


def boundary_jump(series: np.ndarray) -> float:
    if len(series) < 2:
        return float("nan")
    return float(series[1] - series[0])


def window_stat(series: np.ndarray, end_idx: int, fn) -> float:
    if len(series) <= 1:
        return float("nan")
    stop = min(len(series), max(2, end_idx))
    return float(fn(series[1:stop]))


def fraction_outside(series: np.ndarray, limit_hz: float) -> float:
    if len(series) <= 1:
        return float("nan")
    return float(np.mean(np.abs(series[1:]) > limit_hz))


def seconds_to_reenter(series: np.ndarray, limit_hz: float) -> float:
    if len(series) <= 1:
        return 0.0
    mask = np.abs(series[1:]) > limit_hz
    if not np.any(mask):
        return 0.0
    last_out = int(np.max(np.where(mask)[0]))
    return float(last_out + 1)


def choose_bf(sa: Any, first: rdt.DispatchRecord, second: rdt.DispatchRecord, source: str) -> np.ndarray:
    if source == "first":
        return compute_bf(sa, first)
    if source == "second":
        return compute_bf(sa, second)
    raise ValueError(f"Unsupported bf source: {source}")


def make_transition_summary(transition: dict[str, object]) -> dict[str, float | int]:
    gov_start = transition.get("gov_pref_start")
    gov_target = transition.get("gov_pref_target")
    if gov_start is None or gov_target is None:
        return {
            "gov_target_count": 0,
            "gov_target_delta_sum": 0.0,
            "gov_target_delta_abs_sum": 0.0,
            "gov_target_delta_max_abs": 0.0,
        }

    delta = np.asarray(gov_target, dtype=float) - np.asarray(gov_start, dtype=float)
    return {
        "gov_target_count": int(delta.size),
        "gov_target_delta_sum": float(delta.sum()),
        "gov_target_delta_abs_sum": float(np.abs(delta).sum()),
        "gov_target_delta_max_abs": float(np.abs(delta).max()) if delta.size else 0.0,
    }


def main() -> None:
    args = parse_args()
    rdt.andes.config_logger(stream_level=30)

    first = rdt.DispatchRecord.from_json(args.first_dispatch_json)
    second = rdt.DispatchRecord.from_json(args.second_dispatch_json)
    label = args.label or f"{first.label}_{second.label}"
    results_dir = args.results_dir / label
    results_dir.mkdir(parents=True, exist_ok=True)

    curve = rdt.load_curve(args.curve_file)
    for record in (first, second):
        rdt.validate_curve_window(curve, record, args.dispatch_interval)

    dyn_case = rdt.adapt_dyn_case(args.dyn_case, args.stable_dyn_case)
    sa1, ctx1 = prepare_system(
        dispatch_record=first,
        curve=curve,
        dyn_case=dyn_case,
        dispatch_interval=args.dispatch_interval,
        init_mode=args.init_mode,
        wind_prefixes=rdt.DEFAULT_WIND_PREFIXES,
        solar_prefixes=rdt.DEFAULT_SOLAR_PREFIXES,
    )
    bf1 = compute_bf(sa1, first)
    t1, f1, ace_integral_end, ace_raw_end = run_segment(
        sa=sa1,
        ctx=ctx1,
        start_offset=dispatch_offset(first, args.dispatch_interval),
        duration_seconds=args.dispatch_interval,
        agc_interval=args.agc_interval,
        kp=args.kp,
        ki=args.ki,
        bf=bf1,
        ace_integral=0.0,
        ace_raw=0.0,
        local_start=0.0,
        include_initial=True,
    )

    snapshot_path = results_dir / f"{label}_first_segment_snapshot.pkl"
    sa1._deadband_hotstart_meta = {  # type: ignore[attr-defined]
        "ace_integral": ace_integral_end,
        "ace_raw": ace_raw_end,
    }
    save_ss(snapshot_path, sa1)

    base_probe = load_ss(snapshot_path)
    hcp.rehydrate_loaded_snapshot(base_probe)
    probe_transition = apply_second_dispatch_targets(
        base_probe,
        rdt.build_andes_link(base_probe),
        second,
        apply_governor_targets=True,
        apply_dg_targets=False,
    )
    transition_summary = make_transition_summary(probe_transition)

    rows: list[dict[str, object]] = []
    series: dict[str, pd.DataFrame] = {}
    ref_name = "bf_only"
    ref_second = None

    variants = list(VARIANTS)
    variants[-1] = Variant(
        name=variants[-1].name,
        label=f"gov target ramp ({args.ramp_seconds}s)",
        second_bf_source=variants[-1].second_bf_source,
        apply_governor_targets=variants[-1].apply_governor_targets,
        ramp_seconds=args.ramp_seconds,
        color=variants[-1].color,
    )

    for variant in variants:
        sa2 = load_ss(snapshot_path) if args.resume_mode == "snapshot" else sa1
        if args.resume_mode == "snapshot":
            hcp.rehydrate_loaded_snapshot(sa2)
            hot_meta = getattr(sa2, "_deadband_hotstart_meta", {})
            ace_integral_hot = float(hot_meta.get("ace_integral", 0.0))
            ace_raw_hot = float(hot_meta.get("ace_raw", 0.0))
        else:
            ace_integral_hot = ace_integral_end
            ace_raw_hot = ace_raw_end

        ctx2 = ctx1.copy()
        ctx2["link"] = rdt.build_andes_link(sa2)
        bf2 = choose_bf(sa2, first, second, variant.second_bf_source)
        transition = apply_second_dispatch_targets(
            sa2,
            ctx2["link"],  # type: ignore[arg-type]
            second,
            apply_governor_targets=variant.apply_governor_targets,
            apply_dg_targets=False,
        )
        transition["ramp_seconds"] = int(variant.ramp_seconds)
        if variant.apply_governor_targets and variant.ramp_seconds <= 0:
            activate_dispatch_target_transition(sa2, transition, step=0)

        t2, f2, _, _ = run_segment(
            sa=sa2,
            ctx=ctx2,
            start_offset=dispatch_offset(second, args.dispatch_interval),
            duration_seconds=args.dispatch_interval,
            agc_interval=args.agc_interval,
            kp=args.kp,
            ki=args.ki,
            bf=bf2,
            ace_integral=ace_integral_hot,
            ace_raw=ace_raw_hot,
            local_start=float(args.dispatch_interval),
            include_initial=True,
            dispatch_target_transition=transition,
        )

        combined = pd.DataFrame(
            {
                "time_s": np.concatenate([t1, t2]),
                "freq_dev_hz": np.concatenate([f1, f2]),
            }
        )
        combined["variant"] = variant.name
        combined["variant_label"] = variant.label
        variant_csv = results_dir / f"{label}_{variant.name}_frequency.csv"
        combined.to_csv(variant_csv, index=False)
        series[variant.name] = combined

        if variant.name == ref_name:
            ref_second = f2.copy()

        row = {
            "variant": variant.name,
            "variant_label": variant.label,
            "second_bf_source": variant.second_bf_source,
            "apply_governor_targets": int(variant.apply_governor_targets),
            "ramp_seconds": int(variant.ramp_seconds),
            "resume_mode": args.resume_mode,
            "boundary_start_hz": float(f2[0]),
            "boundary_step_0_to_1_hz": boundary_jump(f2),
            "second_min_hz": float(np.min(f2)),
            "second_max_hz": float(np.max(f2)),
            "second_abs_mean_hz": float(np.mean(np.abs(f2))),
            "second_rms_hz": float(np.sqrt(np.mean(np.square(f2)))),
            "first_60s_min_hz": window_stat(f2, 61, np.min),
            "first_60s_max_hz": window_stat(f2, 61, np.max),
            "first_300s_min_hz": window_stat(f2, 301, np.min),
            "first_300s_max_hz": window_stat(f2, 301, np.max),
            "frac_abs_gt_0p036": fraction_outside(f2, 0.036),
            "seconds_until_last_abs_gt_0p036": seconds_to_reenter(f2, 0.036),
            "freq_end_hz": float(f2[-1]),
        }
        row.update(transition_summary)
        rows.append(row)

    if ref_second is None:
        raise RuntimeError("Reference variant not produced.")

    summary = pd.DataFrame(rows)
    for variant in variants:
        name = variant.name
        second_only = series[name]["freq_dev_hz"].to_numpy(dtype=float)[len(t1):]
        delta = second_only - ref_second
        summary.loc[summary["variant"] == name, "delta_vs_bf_only_abs_mean_hz"] = float(np.mean(np.abs(delta)))
        summary.loc[summary["variant"] == name, "delta_vs_bf_only_max_abs_hz"] = float(np.max(np.abs(delta)))

    summary_csv = results_dir / f"{label}_dispatch_target_effect_summary.csv"
    summary.to_csv(summary_csv, index=False)

    merged = pd.concat(series.values(), ignore_index=True)
    merged_csv = results_dir / f"{label}_dispatch_target_effect_all_series.csv"
    merged.to_csv(merged_csv, index=False)

    fig, axes = plt.subplots(3, 1, figsize=(16.5, 13.0), sharex=False)
    for variant in variants:
        df = series[variant.name]
        axes[0].plot(df["time_s"], df["freq_dev_hz"], label=variant.label, color=variant.color, linewidth=1.5)
    axes[0].axvline(args.dispatch_interval, color="#666666", linestyle="--", linewidth=0.9)
    axes[0].axhline(0.0, color="#999999", linestyle=":", linewidth=0.8)
    axes[0].set_title(f"{first.label} -> {second.label}: dispatch-target effect decomposition")
    axes[0].set_ylabel("Frequency deviation [Hz]")
    axes[0].grid(True, alpha=0.22)
    axes[0].legend(loc="upper right", frameon=False)

    xmin = args.dispatch_interval - args.zoom_before
    xmax = args.dispatch_interval + args.zoom_after
    for variant in variants:
        df = series[variant.name]
        axes[1].plot(df["time_s"], df["freq_dev_hz"], label=variant.label, color=variant.color, linewidth=1.6)
    axes[1].axvline(args.dispatch_interval, color="#666666", linestyle="--", linewidth=0.9)
    axes[1].axhline(0.0, color="#999999", linestyle=":", linewidth=0.8)
    axes[1].set_xlim(xmin, xmax)
    axes[1].set_title("Zoom around the dispatch boundary")
    axes[1].set_ylabel("Frequency deviation [Hz]")
    axes[1].grid(True, alpha=0.22)
    axes[1].legend(loc="upper right", frameon=False)

    ref_df = series[ref_name]
    ref_y = ref_df["freq_dev_hz"].to_numpy(dtype=float)[len(t1):]
    ref_x = ref_df["time_s"].to_numpy(dtype=float)[len(t1):]
    for variant in variants:
        df = series[variant.name]
        y = df["freq_dev_hz"].to_numpy(dtype=float)[len(t1):]
        axes[2].plot(ref_x, y - ref_y, label=variant.label, color=variant.color, linewidth=1.55)
    axes[2].axhline(0.0, color="#999999", linestyle=":", linewidth=0.8)
    axes[2].set_xlim(args.dispatch_interval, xmax)
    axes[2].set_title("Incremental effect relative to AGC redistribution only")
    axes[2].set_xlabel("Combined time [s]")
    axes[2].set_ylabel("Delta frequency [Hz]")
    axes[2].grid(True, alpha=0.22)
    axes[2].legend(loc="upper right", frameon=False)

    fig.tight_layout()
    plot_path = results_dir / f"{label}_dispatch_target_effect.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    manifest = {
        "first_dispatch_json": str(args.first_dispatch_json),
        "second_dispatch_json": str(args.second_dispatch_json),
        "results_dir": str(results_dir),
        "summary_csv": str(summary_csv),
        "merged_csv": str(merged_csv),
        "plot_path": str(plot_path),
        "snapshot_path": str(snapshot_path),
        "kp": args.kp,
        "ki": args.ki,
        "agc_interval": args.agc_interval,
        "dispatch_interval": args.dispatch_interval,
        "init_mode": args.init_mode,
        "resume_mode": args.resume_mode,
        "ramp_seconds": args.ramp_seconds,
        "transition_summary": transition_summary,
        "variants": [variant.__dict__ for variant in variants],
    }
    (results_dir / f"{label}_dispatch_target_effect_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"summary_csv={summary_csv}")
    print(f"merged_csv={merged_csv}")
    print(f"plot={plot_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
