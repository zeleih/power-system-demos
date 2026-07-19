#!/usr/bin/env python3
"""
Run the phase-1 deadband coarse sweep on representative hot-start windows.

This script keeps the control baseline fixed and only scans independent
wind/PV/ESD deadband combinations. Each candidate is evaluated on a fixed set
of hot-start windows, aggregated, filtered by safety constraints, and ranked
with a safety-first rule set.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import run_dispatch_tds as rdt
from compare_dispatch_pair_hotstart import (
    AGC_ALLOCATION_HEADROOM,
    AGC_ANTI_WINDUP_FREEZE,
    activate_dispatch_target_transition,
    apply_second_dispatch_targets,
    compute_bf,
    dispatch_offset,
    prepare_system,
    run_segment,
)
from plot_hotstart_droop_breakdown import trace_second_segment


DEFAULT_WINDOWS = (
    "h2d3,h3d0",
    "h7d3,h8d2",
    "h11d2,h11d3",
    "h15d1,h15d2",
    "h20d3,h21d0",
)
DEFAULT_ESD_DEADBAND_LIST = (0.015, 0.020, 0.025, 0.030)
DEFAULT_SOLAR_DEADBAND_LIST = (0.025, 0.030, 0.036, 0.042)
DEFAULT_WIND_DEADBAND_LIST = (0.036, 0.045, 0.055, 0.065)
COMPLEX_WARNING = getattr(getattr(np, "exceptions", object()), "ComplexWarning", None)


@dataclass(frozen=True)
class WindowSpec:
    warmup_label: str
    eval_label: str

    @property
    def name(self) -> str:
        return f"{self.warmup_label}_{self.eval_label}"

    def labels(self, available_labels: list[str]) -> list[str]:
        try:
            start = available_labels.index(self.warmup_label)
        except ValueError as exc:
            raise FileNotFoundError(f"Missing warmup dispatch label: {self.warmup_label}") from exc
        try:
            end = available_labels.index(self.eval_label)
        except ValueError as exc:
            raise FileNotFoundError(f"Missing eval dispatch label: {self.eval_label}") from exc
        if end <= start:
            raise ValueError(
                f"Window {self.name} must advance forward in dispatch order; "
                f"got start={self.warmup_label}, end={self.eval_label}"
            )
        return available_labels[start : end + 1]


@dataclass(frozen=True)
class DeadbandCombo:
    wind_deadband_hz: float
    solar_deadband_hz: float
    esd_deadband_hz: float

    @property
    def combo_id(self) -> str:
        return (
            f"wind{int(round(self.wind_deadband_hz * 1000)):03d}_"
            f"pv{int(round(self.solar_deadband_hz * 1000)):03d}_"
            f"esd{int(round(self.esd_deadband_hz * 1000)):03d}"
        )


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Compat wrapper for NumPy 1.x/2.x trapezoidal integration."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatch-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--curve-file", type=Path, default=rdt.DEFAULT_CURVE_FILE)
    parser.add_argument("--dyn-case", type=Path, required=True)
    parser.add_argument("--stable-dyn-case", type=Path, default=rdt.DEFAULT_STABLE_DYN_CASE)
    parser.add_argument("--dispatch-interval", type=int, default=900)
    parser.add_argument("--agc-interval", type=int, default=4)
    parser.add_argument("--kp", type=float, default=0.1)
    parser.add_argument("--ki", type=float, default=0.002)
    parser.add_argument("--wind-pref-alpha", type=float, default=0.98)
    parser.add_argument("--solar-pref-alpha", type=float, default=0.98)
    parser.add_argument(
        "--agc-allocation-mode",
        choices=rdt.AGC_ALLOCATION_MODES,
        default=AGC_ALLOCATION_HEADROOM,
    )
    parser.add_argument(
        "--agc-anti-windup-mode",
        choices=("off", "freeze_on_saturation"),
        default=AGC_ANTI_WINDUP_FREEZE,
    )
    parser.add_argument("--agc-gov-output-ramp-frac-pmax-per-min", type=float, default=0.0)
    parser.add_argument("--agc-dg-output-ramp-frac-pmax-per-min", type=float, default=0.0)
    parser.add_argument("--disable-der-agc", action="store_true")
    parser.add_argument("--disable-pvd-agc", action="store_true")
    parser.add_argument("--disable-esd-agc", action="store_true")
    parser.add_argument("--init-mode", choices=("dispatch", "first"), default="first")
    parser.add_argument(
        "--governor-target-schedule",
        choices=("step", "boundary_ramp", "midpoint_trajectory", "ramp_limited_basepoint"),
        default="ramp_limited_basepoint",
    )
    parser.add_argument("--governor-basepoint-ramp-floor-frac-pmax-per-min", type=float, default=0.005)
    parser.add_argument("--governor-basepoint-ramp-gap-factor", type=float, default=1.25)
    parser.add_argument("--window", action="append", default=None,
                        help="Representative window in warmup_label,eval_label form. Repeatable.")
    parser.add_argument("--wind-deadband-list", type=float, nargs="+", default=list(DEFAULT_WIND_DEADBAND_LIST))
    parser.add_argument("--solar-deadband-list", type=float, nargs="+", default=list(DEFAULT_SOLAR_DEADBAND_LIST))
    parser.add_argument("--esd-deadband-list", type=float, nargs="+", default=list(DEFAULT_ESD_DEADBAND_LIST))
    parser.add_argument("--max-abs-hz-threshold", type=float, default=0.10)
    parser.add_argument("--share-abs-gt-0p05-threshold", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--save-window-traces", action="store_true")
    parser.add_argument("--force-restart", action="store_true",
                        help="Ignore previous phase1 CSVs and recompute every combo.")
    return parser.parse_args()


def parse_window_specs(values: list[str] | None) -> list[WindowSpec]:
    raw = values if values else list(DEFAULT_WINDOWS)
    specs: list[WindowSpec] = []
    for item in raw:
        first, second = [part.strip() for part in item.split(",", 1)]
        if not first or not second:
            raise ValueError(f"Invalid window spec: {item!r}")
        specs.append(WindowSpec(first, second))
    return specs


def build_combos(args: argparse.Namespace) -> list[DeadbandCombo]:
    wind_vals = sorted(float(v) for v in args.wind_deadband_list)
    solar_vals = sorted(float(v) for v in args.solar_deadband_list)
    esd_vals = sorted(float(v) for v in args.esd_deadband_list)

    combos: list[DeadbandCombo] = []
    for esd in esd_vals:
        for solar in solar_vals:
            if esd > solar:
                continue
            for wind in wind_vals:
                if solar > wind:
                    continue
                combos.append(
                    DeadbandCombo(
                        wind_deadband_hz=wind,
                        solar_deadband_hz=solar,
                        esd_deadband_hz=esd,
                    )
                )
    return combos


def dispatch_json(dispatch_dir: Path, label: str) -> Path:
    path = dispatch_dir / f"{label}_dispatch.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing dispatch JSON: {path}")
    return path


def ordered_dispatch_labels(dispatch_dir: Path) -> list[str]:
    pattern = re.compile(r"^h(\d+)d(\d+)_dispatch\.json$")
    labels: list[tuple[tuple[int, int], str]] = []
    for path in dispatch_dir.glob("h*d*_dispatch.json"):
        match = pattern.match(path.name)
        if match is None:
            continue
        hour = int(match.group(1))
        dispatch = int(match.group(2))
        label = path.stem.removesuffix("_dispatch")
        labels.append(((hour, dispatch), label))
    labels.sort(key=lambda item: item[0])
    return [label for _, label in labels]


def compute_trace_metrics(trace: pd.DataFrame) -> dict[str, float]:
    freq = trace["freq_dev_hz"].to_numpy(dtype=float)
    abs_f = np.abs(freq)
    time_s = trace["time_s"].to_numpy(dtype=float)
    edge_mask = (abs_f >= 0.032) & (abs_f <= 0.040)
    pos_edge = (freq >= 0.032) & (freq <= 0.040)
    neg_edge = (freq <= -0.032) & (freq >= -0.040)

    pvd_throughput = _trapz(
        np.abs((trace["pvd_pe_sum"] - trace["pvd_pref_sum"]).to_numpy(dtype=float)),
        time_s,
    )
    esd_throughput = _trapz(
        np.abs((trace["esd_pe_sum"] - trace["esd_pref_sum"]).to_numpy(dtype=float)),
        time_s,
    )
    gov_droop_effort = _trapz(np.abs(trace["gov_droop_sum"].to_numpy(dtype=float)), time_s)

    return {
        "samples": float(freq.size),
        "mean_abs_hz": float(np.mean(abs_f)),
        "p95_abs_hz": float(np.quantile(abs_f, 0.95)),
        "p99_abs_hz": float(np.quantile(abs_f, 0.99)),
        "max_abs_hz": float(np.max(abs_f)),
        "share_abs_gt_0p036": float(np.mean(abs_f > 0.036)),
        "share_abs_gt_0p05": float(np.mean(abs_f > 0.05)),
        "edge_mass_36": float(np.mean(edge_mask)),
        "edge_pos_36": float(np.mean(pos_edge)),
        "edge_neg_36": float(np.mean(neg_edge)),
        "edge_asymmetry_36": float(abs(np.mean(pos_edge) - np.mean(neg_edge))),
        "esd_throughput": float(esd_throughput),
        "pvd_effort": float(pvd_throughput),
        "gov_droop_effort": float(gov_droop_effort),
    }


def run_window_trace(
    *,
    curve: pd.DataFrame,
    dyn_case: Path,
    window: WindowSpec,
    combo: DeadbandCombo,
    args: argparse.Namespace,
    available_labels: list[str],
) -> pd.DataFrame:
    label_span = window.labels(available_labels)
    records = [rdt.DispatchRecord.from_json(dispatch_json(args.dispatch_dir, label)) for label in label_span]
    for record in records:
        rdt.validate_curve_window(curve, record, args.dispatch_interval)

    wind_prefixes = rdt.DEFAULT_WIND_PREFIXES
    solar_prefixes = rdt.DEFAULT_SOLAR_PREFIXES
    sa, ctx = prepare_system(
        dispatch_record=records[0],
        curve=curve,
        dyn_case=dyn_case,
        dispatch_interval=args.dispatch_interval,
        init_mode=args.init_mode,
        wind_prefixes=wind_prefixes,
        solar_prefixes=solar_prefixes,
        wind_pref_alpha=args.wind_pref_alpha,
        solar_pref_alpha=args.solar_pref_alpha,
    )
    ctx["link"] = rdt.configure_der_agc_participation(
        sa,
        ctx["link"],  # type: ignore[arg-type]
        enable_der_agc=not args.disable_der_agc,
        enable_pvd_agc=not args.disable_pvd_agc,
        enable_esd_agc=not args.disable_esd_agc,
    )
    rdt.apply_resource_deadband_overrides(
        sa,
        wind_prefixes=wind_prefixes,
        solar_prefixes=solar_prefixes,
        wind_deadband_hz=combo.wind_deadband_hz,
        solar_deadband_hz=combo.solar_deadband_hz,
        esd_deadband_hz=combo.esd_deadband_hz,
    )

    ace_integral = 0.0
    ace_raw = 0.0
    for idx, record in enumerate(records):
        current_ctx = ctx.copy()
        current_ctx["link"] = rdt.configure_der_agc_participation(
            sa,
            rdt.build_andes_link(sa),
            enable_der_agc=not args.disable_der_agc,
            enable_pvd_agc=not args.disable_pvd_agc,
            enable_esd_agc=not args.disable_esd_agc,
        )
        next_record = records[idx + 1] if idx + 1 < len(records) else None
        transition = apply_second_dispatch_targets(
            sa,
            current_ctx["link"],  # type: ignore[arg-type]
            record,
            apply_governor_targets=True,
            apply_dg_targets=False,
            duration_seconds=(
                args.dispatch_interval
                if args.governor_target_schedule in ("midpoint_trajectory", "ramp_limited_basepoint")
                else None
            ),
            schedule_mode=args.governor_target_schedule,
            next_dispatch_record=next_record if args.governor_target_schedule == "midpoint_trajectory" else None,
            basepoint_ramp_floor_frac_pmax_per_min=args.governor_basepoint_ramp_floor_frac_pmax_per_min,
            basepoint_ramp_gap_factor=args.governor_basepoint_ramp_gap_factor,
        )
        transition["ramp_seconds"] = 0
        activate_dispatch_target_transition(sa, transition, step=0)
        bf = compute_bf(sa, record)
        if idx < len(records) - 1:
            _, _, ace_integral, ace_raw = run_segment(
                sa=sa,
                ctx=current_ctx,
                start_offset=dispatch_offset(record, args.dispatch_interval),
                duration_seconds=args.dispatch_interval,
                agc_interval=args.agc_interval,
                kp=args.kp,
                ki=args.ki,
                bf=bf,
                agc_allocation_mode=args.agc_allocation_mode,
                ace_integral=ace_integral,
                ace_raw=ace_raw,
                local_start=0.0,
                include_initial=True,
                dispatch_target_transition=transition,
                gov_output_ramp_frac_pmax_per_min=args.agc_gov_output_ramp_frac_pmax_per_min,
                dg_output_ramp_frac_pmax_per_min=args.agc_dg_output_ramp_frac_pmax_per_min,
                agc_anti_windup_mode=args.agc_anti_windup_mode,
                wind_pref_alpha=args.wind_pref_alpha,
                solar_pref_alpha=args.solar_pref_alpha,
            )
        else:
            return trace_second_segment(
                sa=sa,
                ctx=current_ctx,
                dispatch_record=record,
                duration_seconds=args.dispatch_interval,
                agc_interval=args.agc_interval,
                kp=args.kp,
                ki=args.ki,
                bf=bf,
                ace_integral=ace_integral,
                ace_raw=ace_raw,
                dispatch_target_transition=transition,
                gov_output_ramp_frac_pmax_per_min=args.agc_gov_output_ramp_frac_pmax_per_min,
                dg_output_ramp_frac_pmax_per_min=args.agc_dg_output_ramp_frac_pmax_per_min,
                agc_anti_windup_mode=args.agc_anti_windup_mode,
            )
    raise RuntimeError(f"Empty record span for window {window.name}")


def summarize_combo(
    combo: DeadbandCombo,
    window_rows: list[dict[str, object]],
    traces: list[pd.DataFrame],
    args: argparse.Namespace,
) -> dict[str, object]:
    failed_windows = sum(int(bool(row["failed"])) for row in window_rows)
    samples = np.concatenate([trace["freq_dev_hz"].to_numpy(dtype=float) for trace in traces]) if traces else np.asarray([], dtype=float)
    abs_samples = np.abs(samples)
    edge_mask = (abs_samples >= 0.032) & (abs_samples <= 0.040)
    pos_edge = (samples >= 0.032) & (samples <= 0.040)
    neg_edge = (samples <= -0.032) & (samples >= -0.040)

    summary: dict[str, object] = {
        "combo_id": combo.combo_id,
        "wind_deadband_hz": combo.wind_deadband_hz,
        "solar_deadband_hz": combo.solar_deadband_hz,
        "esd_deadband_hz": combo.esd_deadband_hz,
        "window_count": len(window_rows),
        "failed_windows": failed_windows,
        "successful_windows": len(window_rows) - failed_windows,
    }
    if traces:
        max_abs_window = max(float(row["max_abs_hz"]) for row in window_rows if not row["failed"])
        share_gt_0p05 = float(np.mean(abs_samples > 0.05))
        summary.update({
            "mean_abs_hz": float(np.mean(abs_samples)),
            "p95_abs_hz": float(np.quantile(abs_samples, 0.95)),
            "p99_abs_hz": float(np.quantile(abs_samples, 0.99)),
            "max_abs_hz": float(max_abs_window),
            "share_abs_gt_0p036": float(np.mean(abs_samples > 0.036)),
            "share_abs_gt_0p05": share_gt_0p05,
            "edge_mass_36": float(np.mean(edge_mask)),
            "edge_asymmetry_36": float(abs(np.mean(pos_edge) - np.mean(neg_edge))),
            "esd_throughput": float(np.mean([row["esd_throughput"] for row in window_rows if not row["failed"]])),
            "pvd_effort": float(np.mean([row["pvd_effort"] for row in window_rows if not row["failed"]])),
            "gov_droop_effort": float(np.mean([row["gov_droop_effort"] for row in window_rows if not row["failed"]])),
        })
        summary["filter_fail_max_abs"] = int(max_abs_window > float(args.max_abs_hz_threshold))
        summary["filter_fail_share_gt_0p05"] = int(share_gt_0p05 > float(args.share_abs_gt_0p05_threshold))
    else:
        summary.update({
            "mean_abs_hz": np.nan,
            "p95_abs_hz": np.nan,
            "p99_abs_hz": np.nan,
            "max_abs_hz": np.nan,
            "share_abs_gt_0p036": np.nan,
            "share_abs_gt_0p05": np.nan,
            "edge_mass_36": np.nan,
            "edge_asymmetry_36": np.nan,
            "esd_throughput": np.nan,
            "pvd_effort": np.nan,
            "gov_droop_effort": np.nan,
            "filter_fail_max_abs": 1,
            "filter_fail_share_gt_0p05": 1,
        })
    summary["eligible"] = int(
        failed_windows == 0
        and not bool(summary["filter_fail_max_abs"])
        and not bool(summary["filter_fail_share_gt_0p05"])
    )
    return summary


def write_markdown_summary(out_path: Path, ranked: pd.DataFrame, top_k: int) -> None:
    lines = [
        "# Phase-1 Deadband Window Sweep",
        "",
        f"- total combos: {len(ranked)}",
        f"- eligible combos: {int(ranked['eligible'].sum())}",
        f"- top_k: {top_k}",
        "",
        "## Top candidates",
        "",
        "| rank | combo_id | wind | pv | esd | share>|0.05 | max_abs | edge_mass_36 | edge_asym_36 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in ranked.head(top_k).iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['combo_id']} | "
            f"{row['wind_deadband_hz']:.3f} | {row['solar_deadband_hz']:.3f} | {row['esd_deadband_hz']:.3f} | "
            f"{row['share_abs_gt_0p05']:.2%} | {row['max_abs_hz']:.4f} | "
            f"{row['edge_mass_36']:.2%} | {row['edge_asymmetry_36']:.2%} |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def write_outputs(
    *,
    results_dir: Path,
    window_rows: list[dict[str, object]],
    combo_rows: list[dict[str, object]],
    top_k: int,
    manifest: dict[str, object],
    total_combos: int,
    total_windows: int,
) -> None:
    window_df = pd.DataFrame(window_rows)
    window_df.to_csv(results_dir / "phase1_window_metrics.csv", index=False)

    combo_df = pd.DataFrame(combo_rows)
    combo_df.to_csv(results_dir / "phase1_combo_summary.csv", index=False)

    ranked = combo_df.sort_values(
        by=[
            "eligible",
            "share_abs_gt_0p05",
            "max_abs_hz",
            "edge_mass_36",
            "edge_asymmetry_36",
            "esd_throughput",
            "pvd_effort",
            "gov_droop_effort",
        ],
        ascending=[False, True, True, True, True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked.to_csv(results_dir / "phase1_combo_ranked.csv", index=False)

    top_candidates = ranked[ranked["eligible"] == 1].head(int(top_k)).copy()
    top_candidates.to_csv(results_dir / "phase1_top_candidates.csv", index=False)
    write_markdown_summary(results_dir / "phase1_top_candidates.md", top_candidates, int(top_k))

    progress = {
        "combo_count": total_combos,
        "window_count": total_windows,
        "completed_combo_count": len(combo_rows),
        "completed_window_count": len(window_rows),
        "last_completed_combo_id": combo_rows[-1]["combo_id"] if combo_rows else "",
        "eligible_combo_count": int(ranked["eligible"].sum()) if not ranked.empty else 0,
        "top_candidate_ids": top_candidates["combo_id"].tolist() if not top_candidates.empty else [],
        "manifest": manifest,
    }
    (results_dir / "phase1_progress.json").write_text(json.dumps(progress, indent=2))


def load_resume_state(results_dir: Path, *, force_restart: bool) -> tuple[list[dict[str, object]], list[dict[str, object]], set[str]]:
    if force_restart:
        return [], [], set()

    combo_path = results_dir / "phase1_combo_summary.csv"
    window_path = results_dir / "phase1_window_metrics.csv"
    if not combo_path.exists():
        return [], [], set()

    combo_df = pd.read_csv(combo_path)
    if combo_df.empty:
        return [], [], set()

    completed_combo_ids = set(combo_df["combo_id"].astype(str))
    if window_path.exists():
        window_df = pd.read_csv(window_path)
        if not window_df.empty:
            window_df = window_df[window_df["combo_id"].astype(str).isin(completed_combo_ids)].copy()
            window_rows = window_df.to_dict("records")
        else:
            window_rows = []
    else:
        window_rows = []

    combo_rows = combo_df.to_dict("records")
    return window_rows, combo_rows, completed_combo_ids


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if COMPLEX_WARNING is not None:
        warnings.filterwarnings("ignore", category=COMPLEX_WARNING)

    curve = rdt.load_curve(args.curve_file)
    dyn_case = rdt.adapt_dyn_case(args.dyn_case, args.stable_dyn_case)
    windows = parse_window_specs(args.window)
    combos = build_combos(args)
    available_labels = ordered_dispatch_labels(args.dispatch_dir)

    manifest = {
        "dyn_case": str(args.dyn_case.resolve()),
        "stable_dyn_case": str(dyn_case.resolve()),
        "dispatch_dir": str(args.dispatch_dir.resolve()),
        "curve_file": str(args.curve_file.resolve()),
        "dispatch_interval": int(args.dispatch_interval),
        "agc_interval": int(args.agc_interval),
        "kp": float(args.kp),
        "ki": float(args.ki),
        "wind_pref_alpha": float(args.wind_pref_alpha),
        "solar_pref_alpha": float(args.solar_pref_alpha),
        "agc_allocation_mode": str(args.agc_allocation_mode),
        "agc_anti_windup_mode": str(args.agc_anti_windup_mode),
        "disable_der_agc": bool(args.disable_der_agc),
        "disable_pvd_agc": bool(args.disable_pvd_agc),
        "disable_esd_agc": bool(args.disable_esd_agc),
        "init_mode": str(args.init_mode),
        "governor_target_schedule": str(args.governor_target_schedule),
        "governor_basepoint_ramp_floor_frac_pmax_per_min": float(args.governor_basepoint_ramp_floor_frac_pmax_per_min),
        "governor_basepoint_ramp_gap_factor": float(args.governor_basepoint_ramp_gap_factor),
        "windows": [spec.name for spec in windows],
        "wind_deadband_list": list(map(float, args.wind_deadband_list)),
        "solar_deadband_list": list(map(float, args.solar_deadband_list)),
        "esd_deadband_list": list(map(float, args.esd_deadband_list)),
        "max_abs_hz_threshold": float(args.max_abs_hz_threshold),
        "share_abs_gt_0p05_threshold": float(args.share_abs_gt_0p05_threshold),
        "combo_count": len(combos),
    }
    (args.results_dir / "phase1_sweep_manifest.json").write_text(json.dumps(manifest, indent=2))

    window_rows, combo_rows, completed_combo_ids = load_resume_state(
        args.results_dir,
        force_restart=bool(args.force_restart),
    )
    if completed_combo_ids:
        print(
            f"[resume] loaded {len(combo_rows)} completed combos and {len(window_rows)} window rows",
            flush=True,
        )

    total_jobs = len(combos) * len(windows)
    done = len(window_rows)
    for combo_no, combo in enumerate(combos, start=1):
        if combo.combo_id in completed_combo_ids:
            print(f"[combo {combo_no}/{len(combos)}] {combo.combo_id} already complete, skipping", flush=True)
            continue
        traces: list[pd.DataFrame] = []
        combo_window_rows: list[dict[str, object]] = []
        print(f"[combo {combo_no}/{len(combos)}] {combo.combo_id}", flush=True)
        for window in windows:
            done += 1
            try:
                trace = run_window_trace(
                    curve=curve,
                    dyn_case=dyn_case,
                    window=window,
                    combo=combo,
                    args=args,
                    available_labels=available_labels,
                )
                metrics = compute_trace_metrics(trace)
                row: dict[str, object] = {
                    "combo_id": combo.combo_id,
                    "window": window.name,
                    "warmup_label": window.warmup_label,
                    "eval_label": window.eval_label,
                    "wind_deadband_hz": combo.wind_deadband_hz,
                    "solar_deadband_hz": combo.solar_deadband_hz,
                    "esd_deadband_hz": combo.esd_deadband_hz,
                    "failed": 0,
                    "error": "",
                }
                row.update(metrics)
                traces.append(trace)
                if args.save_window_traces:
                    trace_dir = args.results_dir / "window_traces"
                    trace_dir.mkdir(parents=True, exist_ok=True)
                    trace.to_csv(trace_dir / f"{combo.combo_id}_{window.name}.csv", index=False)
                print(
                    f"  [{done}/{total_jobs}] {window.name} ok "
                    f"mean|f|={row['mean_abs_hz']:.4f} share>|0.05={row['share_abs_gt_0p05']:.2%}",
                    flush=True,
                )
            except Exception as exc:
                row = {
                    "combo_id": combo.combo_id,
                    "window": window.name,
                    "warmup_label": window.warmup_label,
                    "eval_label": window.eval_label,
                    "wind_deadband_hz": combo.wind_deadband_hz,
                    "solar_deadband_hz": combo.solar_deadband_hz,
                    "esd_deadband_hz": combo.esd_deadband_hz,
                    "failed": 1,
                    "error": str(exc),
                    "samples": 0.0,
                    "mean_abs_hz": np.nan,
                    "p95_abs_hz": np.nan,
                    "p99_abs_hz": np.nan,
                    "max_abs_hz": np.nan,
                    "share_abs_gt_0p036": np.nan,
                    "share_abs_gt_0p05": np.nan,
                    "edge_mass_36": np.nan,
                    "edge_pos_36": np.nan,
                    "edge_neg_36": np.nan,
                    "edge_asymmetry_36": np.nan,
                    "esd_throughput": np.nan,
                    "pvd_effort": np.nan,
                    "gov_droop_effort": np.nan,
                }
                print(f"  [{done}/{total_jobs}] {window.name} failed: {exc}", flush=True)
            combo_window_rows.append(row)
            window_rows.append(row)

        combo_rows.append(summarize_combo(combo, combo_window_rows, traces, args))
        completed_combo_ids.add(combo.combo_id)
        write_outputs(
            results_dir=args.results_dir,
            window_rows=window_rows,
            combo_rows=combo_rows,
            top_k=int(args.top_k),
            manifest=manifest,
            total_combos=len(combos),
            total_windows=total_jobs,
        )

    write_outputs(
        results_dir=args.results_dir,
        window_rows=window_rows,
        combo_rows=combo_rows,
        top_k=int(args.top_k),
        manifest=manifest,
        total_combos=len(combos),
        total_windows=total_jobs,
    )

    print(f"window_metrics_csv={args.results_dir / 'phase1_window_metrics.csv'}")
    print(f"combo_summary_csv={args.results_dir / 'phase1_combo_summary.csv'}")
    print(f"combo_ranked_csv={args.results_dir / 'phase1_combo_ranked.csv'}")
    print(f"top_candidates_csv={args.results_dir / 'phase1_top_candidates.csv'}")


if __name__ == "__main__":
    main()
