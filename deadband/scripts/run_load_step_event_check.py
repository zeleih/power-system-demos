#!/usr/bin/env python3
"""Load-step event check for baseline vs. heterogeneous deadbands.

The script reuses the Stage-1 hot-start window runner and injects a persistent
load multiplier during the evaluated dispatch interval by perturbing the daily
curve. It is intended as a compact transient-safety screen, not as a full N-1
security study.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import run_dispatch_tds as rdt
from compare_dispatch_pair_hotstart import AGC_ALLOCATION_HEADROOM, AGC_ANTI_WINDUP_FREEZE
from sweep_deadband_phase1_windows import (
    DeadbandCombo,
    WindowSpec,
    compute_trace_metrics,
    dispatch_json,
    ordered_dispatch_labels,
    run_window_trace,
)

COMPLEX_WARNING = getattr(getattr(np, "exceptions", object()), "ComplexWarning", None)
if COMPLEX_WARNING is not None:
    warnings.filterwarnings("ignore", category=COMPLEX_WARNING)
warnings.filterwarnings("ignore", message="Casting complex values to real discards")


def parse_window(text: str) -> WindowSpec:
    first, second = [part.strip() for part in text.split(",", 1)]
    return WindowSpec(first, second)


def perturb_load_curve(
    curve: pd.DataFrame,
    *,
    eval_record: rdt.DispatchRecord,
    dispatch_interval: int,
    event_time_s: int,
    load_step_frac: float,
) -> pd.DataFrame:
    out = curve.copy()
    start = eval_record.hour * 3600 + eval_record.dispatch * dispatch_interval + int(event_time_s)
    stop = eval_record.hour * 3600 + (eval_record.dispatch + 1) * dispatch_interval
    if start < 0 or stop > len(out) or start >= stop:
        raise ValueError(f"Invalid load-step slice [{start}, {stop}) for curve length {len(out)}")
    load_col = out.columns.get_loc("Load")
    out.iloc[start:stop, load_col] = out.iloc[start:stop, load_col].to_numpy(dtype=float) * (1.0 + load_step_frac)
    return out


def first_settling_time(
    time_s: np.ndarray,
    freq_hz: np.ndarray,
    *,
    event_time_s: float,
    band_hz: float = 0.036,
    hold_s: float = 30.0,
) -> float:
    post_idx = np.flatnonzero(time_s >= event_time_s)
    if post_idx.size == 0:
        return float("nan")
    dt = float(np.median(np.diff(time_s))) if time_s.size > 1 else 1.0
    hold_n = max(1, int(round(hold_s / max(dt, 1e-9))))
    inside = np.abs(freq_hz) <= band_hz
    for i in post_idx:
        j = min(i + hold_n, inside.size)
        if j - i >= hold_n and bool(np.all(inside[i:j])):
            return float(time_s[i] - event_time_s)
    return float("nan")


def event_metrics(trace: pd.DataFrame, *, event_time_s: int) -> dict[str, float]:
    time_s = trace["time_s"].to_numpy(dtype=float)
    freq = trace["freq_dev_hz"].to_numpy(dtype=float)
    post = time_s >= float(event_time_s)
    if not np.any(post):
        raise RuntimeError("Trace has no post-event samples")
    post_freq = freq[post]
    rocof = np.diff(freq) / np.maximum(np.diff(time_s), 1e-9)
    rocof_time = time_s[1:]
    event_rocof = rocof[(rocof_time >= event_time_s) & (rocof_time <= event_time_s + 10)]
    summary = compute_trace_metrics(trace)
    summary.update(
        {
            "post_min_hz": float(np.min(post_freq)),
            "post_max_hz": float(np.max(post_freq)),
            "post_max_abs_hz": float(np.max(np.abs(post_freq))),
            "event_rocof_abs_max_hz_s": float(np.max(np.abs(event_rocof))) if event_rocof.size else float("nan"),
            "settle_36mHz_30s_s": first_settling_time(time_s, freq, event_time_s=float(event_time_s)),
        }
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatch-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--curve-file", type=Path, default=rdt.DEFAULT_CURVE_FILE)
    parser.add_argument("--dyn-case", type=Path, required=True)
    parser.add_argument("--stable-dyn-case", type=Path, default=rdt.DEFAULT_STABLE_DYN_CASE)
    parser.add_argument("--window", type=parse_window, default=parse_window("h11d2,h11d3"))
    parser.add_argument("--event-time-s", type=int, default=120)
    parser.add_argument("--load-step-frac", type=float, action="append", default=None)
    parser.add_argument("--dispatch-interval", type=int, default=900)
    parser.add_argument("--agc-interval", type=int, default=4)
    parser.add_argument("--kp", type=float, default=0.1)
    parser.add_argument("--ki", type=float, default=0.002)
    parser.add_argument("--wind-pref-alpha", type=float, default=0.98)
    parser.add_argument("--solar-pref-alpha", type=float, default=0.98)
    parser.add_argument("--disable-der-agc", action="store_true")
    parser.add_argument("--disable-pvd-agc", action="store_true")
    parser.add_argument("--disable-esd-agc", action="store_true")
    parser.add_argument("--agc-allocation-mode", default=AGC_ALLOCATION_HEADROOM)
    parser.add_argument("--agc-anti-windup-mode", default=AGC_ANTI_WINDUP_FREEZE)
    parser.add_argument("--agc-gov-output-ramp-frac-pmax-per-min", type=float, default=0.0)
    parser.add_argument("--agc-dg-output-ramp-frac-pmax-per-min", type=float, default=0.0)
    parser.add_argument("--init-mode", choices=("dispatch", "first"), default="first")
    parser.add_argument("--governor-target-schedule", default="ramp_limited_basepoint")
    parser.add_argument("--governor-basepoint-ramp-floor-frac-pmax-per-min", type=float, default=0.005)
    parser.add_argument("--governor-basepoint-ramp-gap-factor", type=float, default=1.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rdt.andes.config_logger(stream_level=30)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    load_steps = args.load_step_frac or [0.05, -0.05]
    curve = rdt.load_curve(args.curve_file)
    dyn_case = rdt.adapt_dyn_case(args.dyn_case, args.stable_dyn_case)
    available_labels = ordered_dispatch_labels(args.dispatch_dir)
    label_span = args.window.labels(available_labels)
    eval_record = rdt.DispatchRecord.from_json(dispatch_json(args.dispatch_dir, label_span[-1]))

    runner_args = SimpleNamespace(
        dispatch_dir=args.dispatch_dir,
        dispatch_interval=args.dispatch_interval,
        agc_interval=args.agc_interval,
        kp=args.kp,
        ki=args.ki,
        wind_pref_alpha=args.wind_pref_alpha,
        solar_pref_alpha=args.solar_pref_alpha,
        disable_der_agc=args.disable_der_agc,
        disable_pvd_agc=args.disable_pvd_agc,
        disable_esd_agc=args.disable_esd_agc,
        agc_allocation_mode=args.agc_allocation_mode,
        agc_anti_windup_mode=args.agc_anti_windup_mode,
        agc_gov_output_ramp_frac_pmax_per_min=args.agc_gov_output_ramp_frac_pmax_per_min,
        agc_dg_output_ramp_frac_pmax_per_min=args.agc_dg_output_ramp_frac_pmax_per_min,
        init_mode=args.init_mode,
        governor_target_schedule=args.governor_target_schedule,
        governor_basepoint_ramp_floor_frac_pmax_per_min=args.governor_basepoint_ramp_floor_frac_pmax_per_min,
        governor_basepoint_ramp_gap_factor=args.governor_basepoint_ramp_gap_factor,
    )

    combos = {
        "baseline_36_36_36": DeadbandCombo(0.036, 0.036, 0.036),
        "best_36_25_15": DeadbandCombo(0.036, 0.025, 0.015),
    }
    rows: list[dict[str, object]] = []
    for step_frac in load_steps:
        event_curve = perturb_load_curve(
            curve,
            eval_record=eval_record,
            dispatch_interval=args.dispatch_interval,
            event_time_s=args.event_time_s,
            load_step_frac=float(step_frac),
        )
        step_tag = f"{int(round(step_frac * 100)):+d}pct".replace("+", "plus").replace("-", "minus")
        for label, combo in combos.items():
            trace = run_window_trace(
                curve=event_curve,
                dyn_case=dyn_case,
                window=args.window,
                combo=combo,
                args=runner_args,
                available_labels=available_labels,
            )
            trace_path = args.results_dir / f"{args.window.name}_{step_tag}_{label}_trace.csv"
            trace.to_csv(trace_path, index=False)
            metrics = event_metrics(trace, event_time_s=args.event_time_s)
            metrics.update(
                {
                    "case": label,
                    "window": args.window.name,
                    "eval_dispatch": eval_record.label,
                    "load_step_frac": float(step_frac),
                    "event_time_s": int(args.event_time_s),
                    "trace_csv": str(trace_path),
                }
            )
            rows.append(metrics)
            print(
                f"{step_tag} {label}: post_max_abs={metrics['post_max_abs_hz']:.5f} "
                f"rocof={metrics['event_rocof_abs_max_hz_s']:.5f}"
            )

    summary = pd.DataFrame(rows)
    summary_path = args.results_dir / "load_step_event_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"summary_csv={summary_path}")


if __name__ == "__main__":
    main()
