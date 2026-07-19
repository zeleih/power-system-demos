#!/usr/bin/env python3
"""
Trace cold-start AGC signals for one dispatch across a list of KP/KI cases.

Outputs per-case CSV traces plus combined figures for:
- frequency deviation
- ACE
- held AGC total command (ace_raw)
- summed governor paux0
- summed DG Pext0
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_dispatch_tds as rdt
from compare_dispatch_pair_hotstart import (
    AGC_ANTI_WINDUP_FREEZE,
    AGC_ANTI_WINDUP_OFF,
    activate_dispatch_target_transition,
    apply_agc_dispatch_update,
    apply_second_dispatch_targets,
    backcalculate_ace_integral,
    compute_bf,
    dispatch_offset,
    initial_agc_aw_state,
    prepare_system,
    total_der_output_pe,
    total_governor_output_pe,
)
from run_dispatch_hotstart import (
    configure_all_der_deadband,
    scale_storage_capacity,
    solve_scale_factor,
    storage_share,
)


@dataclass(frozen=True)
class GainCase:
    label: str
    kp: float
    ki: float


DEFAULT_CASES = (
    "kp0_ki0,0,0",
    "kp0p03_ki0p003,0.03,0.003",
    "kp0p05_ki0p005,0.05,0.005",
    "kp0p08_ki0p008,0.08,0.008",
    "kp0p10_ki0p010,0.10,0.010",
)


def parse_case(text: str) -> GainCase:
    parts = [item.strip() for item in text.split(",") if item.strip()]
    if len(parts) == 3:
        label, kp_text, ki_text = parts
    elif len(parts) == 2:
        kp_text, ki_text = parts
        label = f"kp{kp_text.replace('.', 'p')}_ki{ki_text.replace('.', 'p')}"
    else:
        raise argparse.ArgumentTypeError(
            f"invalid --case '{text}', expected 'label,kp,ki' or 'kp,ki'"
        )
    return GainCase(label=label, kp=float(kp_text), ki=float(ki_text))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatch-json", type=Path, required=True)
    parser.add_argument("--next-dispatch-json", type=Path, required=True)
    parser.add_argument("--curve-file", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--dyn-case", type=Path, default=rdt.DEFAULT_DYN_CASE)
    parser.add_argument("--stable-dyn-case", type=Path, default=rdt.DEFAULT_STABLE_DYN_CASE)
    parser.add_argument("--duration-seconds", type=int, default=900)
    parser.add_argument("--agc-interval", type=int, default=4)
    parser.add_argument(
        "--ace-filter-tau",
        type=float,
        default=0.0,
        help="First-order low-pass time constant for ACE in seconds. 0 disables filtering.",
    )
    parser.add_argument("--agc-gov-output-ramp-frac-pmax-per-min", type=float, default=0.0)
    parser.add_argument("--agc-dg-output-ramp-frac-pmax-per-min", type=float, default=0.0)
    parser.add_argument(
        "--agc-allocation-mode",
        choices=rdt.AGC_ALLOCATION_MODES,
        default=rdt.AGC_ALLOCATION_HEADROOM,
    )
    parser.add_argument(
        "--agc-anti-windup-mode",
        choices=(AGC_ANTI_WINDUP_OFF, AGC_ANTI_WINDUP_FREEZE),
        default=AGC_ANTI_WINDUP_OFF,
    )
    parser.add_argument("--init-mode", choices=("dispatch", "first"), default="first")
    parser.add_argument(
        "--governor-target-schedule",
        choices=("step", "boundary_ramp", "midpoint_trajectory", "ramp_limited_basepoint"),
        default="midpoint_trajectory",
    )
    parser.add_argument(
        "--governor-basepoint-ramp-floor-frac-pmax-per-min",
        type=float,
        default=0.005,
        help="Per-unit-of-pmax per minute floor used to build ramp-limited governor basepoints.",
    )
    parser.add_argument(
        "--governor-basepoint-ramp-gap-factor",
        type=float,
        default=1.25,
        help="Multiplier on target-gap/duration used to build ramp-limited governor basepoints.",
    )
    parser.add_argument("--traditional-governor-deadband-hz", type=float, default=None)
    parser.add_argument("--traditional-governor-deadband-csv", type=Path, default=None)
    parser.add_argument("--der-deadband-hz", type=float, default=None)
    parser.add_argument("--der-base-ddn", type=float, default=None)
    parser.add_argument("--pvd1-base-ddn", type=float, default=None)
    parser.add_argument("--esd1-base-ddn", type=float, default=None)
    parser.add_argument("--target-storage-share", type=float, default=None)
    parser.add_argument("--scale-esd1-ddn-with-storage", action="store_true")
    parser.add_argument("--disable-der-agc", action="store_true")
    parser.add_argument("--wind-prefix", action="append", default=None)
    parser.add_argument("--solar-prefix", action="append", default=None)
    parser.add_argument("--case", dest="cases", action="append", type=parse_case, default=None)
    return parser.parse_args()


def summed(values: np.ndarray | list[float] | None) -> float:
    if values is None:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float(arr.sum()) if arr.size else 0.0


def summed_abs(values: np.ndarray | list[float] | None) -> float:
    if values is None:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float(np.abs(arr).sum()) if arr.size else 0.0


def filter_alpha(tau_s: float, dt_s: float = 1.0) -> float:
    if tau_s <= 0.0:
        return 0.0
    return float(np.exp(-dt_s / tau_s))


def trace_segment(
    *,
    sa,
    ctx: dict[str, object],
    start_offset: int,
    duration_seconds: int,
    agc_interval: int,
    kp: float,
    ki: float,
    bf: np.ndarray,
    agc_allocation_mode: str = rdt.AGC_ALLOCATION_HEADROOM,
    dispatch_target_transition: dict[str, object] | None = None,
    gov_output_ramp_frac_pmax_per_min: float = 0.0,
    dg_output_ramp_frac_pmax_per_min: float = 0.0,
    agc_anti_windup_mode: str = AGC_ANTI_WINDUP_OFF,
) -> pd.DataFrame:
    curve: pd.DataFrame = ctx["curve"]  # type: ignore[assignment]
    link: pd.DataFrame = ctx["link"]  # type: ignore[assignment]
    pq_idx = ctx["pq_idx"]
    sap0 = ctx["sap0"]
    saq0 = ctx["saq0"]
    p0_w2t = ctx["p0_w2t"]
    p0_pv = ctx["p0_pv"]
    pvd1_w2t = ctx["pvd1_w2t"]
    pvd1_pv = ctx["pvd1_pv"]
    pext_max = ctx["pext_max"]

    ace_integral = 0.0
    ace_raw = 0.0
    ace_filtered = 0.0
    last_agov_req_sum = 0.0
    last_agov_req_abs_sum = 0.0
    last_adg_req_sum = 0.0
    last_adg_req_abs_sum = 0.0
    cycle_saturation_active = 0
    cycle_integrator_freeze_active = 0
    cycle_agc_request_sum_total = 0.0
    cycle_agc_applied_sum_total = 0.0
    cycle_agc_clip_deficit_sum_total = 0.0
    cycle_agc_saturation_ratio = 0.0
    cycle_agc_freeze_streak = 0
    agc_aw_state = initial_agc_aw_state()

    rows: list[dict[str, float | int]] = []

    gov_all_idx = [idx for idx in link["gov_idx"].tolist() if pd.notna(idx)]
    dg_all_idx = [idx for idx in link["dg_idx"].tolist() if pd.notna(idx)]

    def snapshot(ts: float, agc_update: int) -> None:
        ace_sum = float(np.asarray(sa.ACEc.ace.v, dtype=float).sum())
        gov_paux0 = (
            np.asarray(sa.TurbineGov.get(src="paux0", attr="v", idx=gov_all_idx), dtype=float)
            if gov_all_idx else np.asarray([])
        )
        dg_pext0 = (
            np.asarray(sa.DG.get(src="Pext0", attr="v", idx=dg_all_idx), dtype=float)
            if dg_all_idx else np.asarray([])
        )
        gov_pref0 = (
            np.asarray(sa.TurbineGov.get(src="pref0", attr="v", idx=gov_all_idx), dtype=float)
            if gov_all_idx else np.asarray([])
        )
        rows.append({
            "time_s": float(ts),
            "freq_dev_hz": float((sa.ACEc.f.v[0] - 1.0) * sa.config.freq),
            "ace_sum": ace_sum,
            "ace_filtered_hold": float(ace_filtered),
            "ace_integral_hold": float(ace_integral),
            "ace_raw_hold": float(ace_raw),
            "agov_request_sum": float(last_agov_req_sum),
            "agov_request_abs_sum": float(last_agov_req_abs_sum),
            "adg_request_sum": float(last_adg_req_sum),
            "adg_request_abs_sum": float(last_adg_req_abs_sum),
            "gov_paux0_sum": summed(gov_paux0),
            "gov_paux0_abs_sum": summed_abs(gov_paux0),
            "dg_pext0_sum": summed(dg_pext0),
            "dg_pext0_abs_sum": summed_abs(dg_pext0),
            "gov_pref0_sum": summed(gov_pref0),
            "gov_pe_sum": float(total_governor_output_pe(sa, gov_all_idx)),
            "dg_pe_sum": float(total_der_output_pe(sa)),
            "saturation_active": int(cycle_saturation_active),
            "integrator_freeze_active": int(cycle_integrator_freeze_active),
            "agc_request_sum_total": float(cycle_agc_request_sum_total),
            "agc_applied_sum_total": float(cycle_agc_applied_sum_total),
            "agc_clip_deficit_sum_total": float(cycle_agc_clip_deficit_sum_total),
            "agc_saturation_ratio": float(cycle_agc_saturation_ratio),
            "agc_freeze_streak": int(cycle_agc_freeze_streak),
            "agc_update": int(agc_update),
        })

    snapshot(0.0, agc_update=0)

    alpha = filter_alpha(float(getattr(sa, "_ace_filter_tau_s", 0.0)))
    current_tf = float(sa.dae.t)
    for step in range(1, duration_seconds):
        activate_dispatch_target_transition(sa, dispatch_target_transition, step)

        agc_update = 0
        if step % agc_interval == 0:
            agc_update = 1
            agc_meta = apply_agc_dispatch_update(
                sa=sa,
                link=link,
                bf=bf,
                ace_raw=ace_raw,
                pext_max=pext_max,
                agc_allocation_mode=agc_allocation_mode,
                gov_output_ramp_frac_pmax_per_min=gov_output_ramp_frac_pmax_per_min,
                dg_output_ramp_frac_pmax_per_min=dg_output_ramp_frac_pmax_per_min,
                agc_anti_windup_mode=agc_anti_windup_mode,
                prev_freeze_active=agc_aw_state["freeze_active"],
                prev_freeze_on_streak=agc_aw_state["freeze_on_streak"],
                prev_freeze_off_streak=agc_aw_state["freeze_off_streak"],
                prev_freeze_dir=agc_aw_state["freeze_dir"],
            )
            last_agov_req_sum = float(agc_meta["agov_request_sum"])
            last_agov_req_abs_sum = float(agc_meta["agov_request_abs_sum"])
            last_adg_req_sum = float(agc_meta["adg_request_sum"])
            last_adg_req_abs_sum = float(agc_meta["adg_request_abs_sum"])
            cycle_saturation_active = int(agc_meta["saturation_active"])
            cycle_integrator_freeze_active = int(agc_meta["integrator_freeze_active"])
            cycle_agc_request_sum_total = float(agc_meta["agc_request_sum_total"])
            cycle_agc_applied_sum_total = float(agc_meta["agc_applied_sum_total"])
            cycle_agc_clip_deficit_sum_total = float(agc_meta["agc_clip_deficit_sum_total"])
            cycle_agc_saturation_ratio = float(agc_meta["agc_saturation_ratio"])
            cycle_agc_freeze_streak = int(agc_meta["agc_freeze_streak"])
            agc_aw_state["freeze_active"] = int(agc_meta["agc_freeze_active"])
            agc_aw_state["freeze_on_streak"] = int(agc_meta["agc_freeze_streak"])
            agc_aw_state["freeze_off_streak"] = int(agc_meta["agc_unfreeze_streak"])
            agc_aw_state["freeze_dir"] = int(agc_meta["agc_freeze_dir"])

        kload = curve["Load"].iloc[start_offset + step]
        sa.PQ.set(src="Ppf", idx=pq_idx, attr="v", value=kload * sap0)
        sa.PQ.set(src="Qpf", idx=pq_idx, attr="v", value=kload * saq0)

        wind = curve["Wind"].iloc[start_offset + step]
        sa.PVD1.set(src="pref0", idx=pvd1_w2t, attr="v", value=wind * p0_w2t)

        solar = curve["PV"].iloc[start_offset + step]
        sa.PVD1.set(src="pref0", idx=pvd1_pv, attr="v", value=solar * p0_pv)

        current_tf += 1.0
        sa.TDS.config.tf = current_tf
        sa.TDS.run()
        if sa.exit_code != 0:
            raise RuntimeError(f"TDS failed at local step={step} with exit_code={sa.exit_code}")

        snapshot(float(step), agc_update=agc_update)

        ace_sum = float(np.asarray(sa.ACEc.ace.v, dtype=float).sum())
        if alpha > 0.0:
            ace_filtered = alpha * ace_filtered + (1.0 - alpha) * ace_sum
        else:
            ace_filtered = ace_sum
        ace_raw = -(kp * ace_filtered + ki * ace_integral)
        if cycle_integrator_freeze_active:
            ace_integral, ace_raw = backcalculate_ace_integral(
                kp=kp,
                ki=ki,
                ace_sum=float(ace_filtered),
                ace_raw=float(ace_raw),
                agc_request_sum_total=cycle_agc_request_sum_total,
                agc_applied_sum_total=cycle_agc_applied_sum_total,
            )
        else:
            ace_integral = ace_integral + ace_filtered

    trace = pd.DataFrame(rows)
    freq = trace["freq_dev_hz"].to_numpy(dtype=float)
    freq_d1 = np.abs(np.diff(freq))
    trace["gov_paux0_tv"] = float(np.sum(np.abs(np.diff(trace["gov_paux0_sum"].to_numpy(dtype=float)))))
    trace["dg_pext0_tv"] = float(np.sum(np.abs(np.diff(trace["dg_pext0_sum"].to_numpy(dtype=float)))))
    trace["freq_d1_abs_mean"] = float(freq_d1.mean()) if freq_d1.size else 0.0
    trace["freq_d1_abs_p95"] = float(np.quantile(freq_d1, 0.95)) if freq_d1.size else 0.0
    trace["saturation_fraction"] = float(trace["saturation_active"].mean()) if not trace.empty else 0.0
    trace["integrator_freeze_fraction"] = (
        float(trace["integrator_freeze_active"].mean()) if not trace.empty else 0.0
    )
    trace["agc_saturation_ratio_mean"] = float(trace["agc_saturation_ratio"].mean()) if not trace.empty else 0.0
    trace["agc_saturation_ratio_max"] = float(trace["agc_saturation_ratio"].max()) if not trace.empty else 0.0
    trace["agc_freeze_streak_max"] = int(trace["agc_freeze_streak"].max()) if not trace.empty else 0
    return trace


def configure_case(
    *,
    sa,
    args: argparse.Namespace,
) -> None:
    share_before = storage_share(sa)
    if args.target_storage_share is not None:
        current_share = float(share_before["pmax_share"])
        if abs(float(args.target_storage_share) - current_share) >= 1e-12:
            factor = solve_scale_factor(current_share, float(args.target_storage_share))
            if abs(factor - 1.0) > 1e-12:
                scale_storage_capacity(sa, factor)

    if args.der_deadband_hz is not None and args.der_deadband_hz > 0.0:
        total_storage_factor = 1.0
        if args.target_storage_share is not None:
            current_share = float(share_before["pmax_share"])
            if abs(float(args.target_storage_share) - current_share) < 1e-12:
                total_storage_factor = 1.0
            else:
                total_storage_factor = solve_scale_factor(current_share, float(args.target_storage_share))
        configure_all_der_deadband(
            sa,
            traditional_governor_deadband_hz=(
                None if args.traditional_governor_deadband_csv is not None
                else args.traditional_governor_deadband_hz
            ),
            der_deadband_hz=args.der_deadband_hz,
            der_base_ddn=args.der_base_ddn,
            pvd1_base_ddn=args.pvd1_base_ddn,
            esd1_base_ddn=args.esd1_base_ddn,
            esd1_ddn_scale=(
                float(total_storage_factor) if args.scale_esd1_ddn_with_storage else 1.0
            ),
        )
    if args.traditional_governor_deadband_csv is not None:
        rdt.apply_traditional_governor_deadband_csv(
            sa,
            args.traditional_governor_deadband_csv,
            model_name="TGOV1NDB",
        )
    elif args.traditional_governor_deadband_hz is not None and args.traditional_governor_deadband_hz > 0.0:
        rdt.apply_traditional_governor_deadband(sa, float(args.traditional_governor_deadband_hz))


def run_case(
    *,
    gain_case: GainCase,
    args: argparse.Namespace,
    dispatch_record: rdt.DispatchRecord,
    next_dispatch_record: rdt.DispatchRecord,
    curve: pd.DataFrame,
    dyn_case: Path,
    wind_prefixes: tuple[str, ...],
    solar_prefixes: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    sa, ctx = prepare_system(
        dispatch_record=dispatch_record,
        curve=curve,
        dyn_case=dyn_case,
        dispatch_interval=args.duration_seconds,
        init_mode=args.init_mode,
        wind_prefixes=wind_prefixes,
        solar_prefixes=solar_prefixes,
    )
    configure_case(sa=sa, args=args)
    ctx["link"] = rdt.configure_der_agc_participation(
        sa,
        ctx["link"],  # type: ignore[arg-type]
        enable_der_agc=not args.disable_der_agc,
    )
    sa._ace_filter_tau_s = float(args.ace_filter_tau)

    dispatch_target_transition = apply_second_dispatch_targets(
        sa,
        ctx["link"],  # type: ignore[arg-type]
        dispatch_record,
        apply_governor_targets=True,
        apply_dg_targets=False,
        duration_seconds=(
            args.duration_seconds
            if args.governor_target_schedule in ("midpoint_trajectory", "ramp_limited_basepoint")
            else None
        ),
        schedule_mode=args.governor_target_schedule,
        next_dispatch_record=next_dispatch_record if args.governor_target_schedule == "midpoint_trajectory" else None,
        basepoint_ramp_floor_frac_pmax_per_min=args.governor_basepoint_ramp_floor_frac_pmax_per_min,
        basepoint_ramp_gap_factor=args.governor_basepoint_ramp_gap_factor,
    )
    dispatch_target_transition["ramp_seconds"] = 0
    activate_dispatch_target_transition(sa, dispatch_target_transition, step=0)

    bf = compute_bf(sa, dispatch_record)
    trace = trace_segment(
        sa=sa,
        ctx=ctx,
        start_offset=dispatch_offset(dispatch_record, args.duration_seconds),
        duration_seconds=args.duration_seconds,
        agc_interval=args.agc_interval,
        kp=gain_case.kp,
        ki=gain_case.ki,
        bf=bf,
        agc_allocation_mode=args.agc_allocation_mode,
        dispatch_target_transition=dispatch_target_transition,
        gov_output_ramp_frac_pmax_per_min=args.agc_gov_output_ramp_frac_pmax_per_min,
        dg_output_ramp_frac_pmax_per_min=args.agc_dg_output_ramp_frac_pmax_per_min,
        agc_anti_windup_mode=args.agc_anti_windup_mode,
    )

    freq = trace["freq_dev_hz"].to_numpy(dtype=float)
    summary = {
        "label": gain_case.label,
        "kp": float(gain_case.kp),
        "ki": float(gain_case.ki),
        "max_hz": float(freq.max()),
        "min_hz": float(freq.min()),
        "final_hz": float(freq[-1]),
        "mean_abs_hz": float(np.mean(np.abs(freq))),
        "share_abs_gt_0p036": float(np.mean(np.abs(freq) > 0.036)),
        "zero_crossings": int(np.sum(np.signbit(freq[1:]) != np.signbit(freq[:-1]))),
        "ace_filter_tau_s": float(args.ace_filter_tau),
        "ace_raw_abs_max": float(trace["ace_raw_hold"].abs().max()),
        "ace_filtered_abs_max": float(trace["ace_filtered_hold"].abs().max()),
        "gov_paux0_abs_max": float(trace["gov_paux0_abs_sum"].max()),
        "dg_pext0_abs_max": float(trace["dg_pext0_abs_sum"].max()),
        "gov_paux0_tv": float(trace["gov_paux0_tv"].iloc[-1]),
        "dg_pext0_tv": float(trace["dg_pext0_tv"].iloc[-1]),
        "freq_d1_abs_mean": float(trace["freq_d1_abs_mean"].iloc[-1]),
        "freq_d1_abs_p95": float(trace["freq_d1_abs_p95"].iloc[-1]),
        "saturation_fraction": float(trace["saturation_fraction"].iloc[-1]),
        "integrator_freeze_fraction": float(trace["integrator_freeze_fraction"].iloc[-1]),
        "agc_request_sum_total_abs_max": float(trace["agc_request_sum_total"].abs().max()),
        "agc_applied_sum_total_abs_max": float(trace["agc_applied_sum_total"].abs().max()),
        "agc_clip_deficit_sum_total_abs_max": float(trace["agc_clip_deficit_sum_total"].abs().max()),
        "agc_saturation_ratio_mean": float(trace["agc_saturation_ratio_mean"].iloc[-1]),
        "agc_saturation_ratio_max": float(trace["agc_saturation_ratio_max"].iloc[-1]),
        "agc_freeze_streak_max": int(trace["agc_freeze_streak_max"].iloc[-1]),
    }
    return trace, summary


def make_plot(
    traces: list[tuple[GainCase, pd.DataFrame]],
    out_path: Path,
    *,
    xmax: float,
    title_suffix: str,
    dispatch_label: str,
) -> None:
    colors = ["#444444", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    fig, axes = plt.subplots(6, 1, figsize=(14, 17), constrained_layout=True, sharex=True)
    for i, (gain_case, trace) in enumerate(traces):
        color = colors[i % len(colors)]
        label = f"KP={gain_case.kp:g}, KI={gain_case.ki:g}"
        axes[0].plot(trace["time_s"], trace["freq_dev_hz"], color=color, linewidth=2.0, label=label)
        axes[1].plot(trace["time_s"], trace["ace_raw_hold"], color=color, linewidth=1.8, label=label)
        axes[2].step(trace["time_s"], trace["gov_paux0_sum"], where="post", color=color, linewidth=1.8, label=label)
        axes[3].step(trace["time_s"], trace["dg_pext0_sum"], where="post", color=color, linewidth=1.8, label=label)
        axes[4].plot(trace["time_s"], trace["gov_pe_sum"], color=color, linewidth=1.6, label=label)
        axes[5].plot(trace["time_s"], trace["dg_pe_sum"], color=color, linewidth=1.6, label=label)

    axes[0].axhline(0.036, color="black", linestyle=":", linewidth=1.0)
    axes[0].axhline(-0.036, color="black", linestyle=":", linewidth=1.0)
    axes[0].axhline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    axes[1].axhline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    axes[2].axhline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    axes[3].axhline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    axes[4].axhline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    axes[5].axhline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)

    axes[0].set_ylabel("Freq dev [Hz]")
    axes[1].set_ylabel("AGC total\nace_raw")
    axes[2].set_ylabel("Gov AGC\nsum paux0")
    axes[3].set_ylabel("DER AGC\nsum Pext0")
    axes[4].set_ylabel("Gov actual\nsum Pe")
    axes[5].set_ylabel("DER actual\nsum Pe")
    axes[5].set_xlabel("Time [s]")

    axes[0].set_title(f"{dispatch_label} cold-start AGC trace comparison ({title_suffix})")
    for ax in axes:
        ax.set_xlim(0, xmax)
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", ncol=2, frameon=True)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    curve = rdt.load_curve(args.curve_file)
    dispatch_record = rdt.DispatchRecord.from_json(args.dispatch_json)
    next_dispatch_record = rdt.DispatchRecord.from_json(args.next_dispatch_json)
    dyn_case = rdt.adapt_dyn_case(args.dyn_case, args.stable_dyn_case)
    wind_prefixes = rdt.normalize_prefixes(args.wind_prefix, rdt.DEFAULT_WIND_PREFIXES)
    solar_prefixes = rdt.normalize_prefixes(args.solar_prefix, rdt.DEFAULT_SOLAR_PREFIXES)
    cases = args.cases or [parse_case(text) for text in DEFAULT_CASES]

    traces: list[tuple[GainCase, pd.DataFrame]] = []
    summaries: list[dict[str, float | int | str]] = []

    for gain_case in cases:
        print(f"running {gain_case.label}: kp={gain_case.kp}, ki={gain_case.ki}", flush=True)
        trace, summary = run_case(
            gain_case=gain_case,
            args=args,
            dispatch_record=dispatch_record,
            next_dispatch_record=next_dispatch_record,
            curve=curve,
            dyn_case=dyn_case,
            wind_prefixes=wind_prefixes,
            solar_prefixes=solar_prefixes,
        )
        trace_csv = args.results_dir / f"{gain_case.label}_trace.csv"
        trace.to_csv(trace_csv, index=False)
        summary["trace_csv"] = str(trace_csv)
        traces.append((gain_case, trace))
        summaries.append(summary)
        print(f"  trace_csv={trace_csv}", flush=True)

    summary_df = pd.DataFrame(summaries)
    summary_csv = args.results_dir / "trace_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    full_png = args.results_dir / "h5d2_agc_trace_compare_full.png"
    zoom_png = args.results_dir / "h5d2_agc_trace_compare_first180s.png"
    dispatch_label = dispatch_record.label
    make_plot(
        traces,
        full_png,
        xmax=float(args.duration_seconds - 1),
        title_suffix="full window",
        dispatch_label=dispatch_label,
    )
    make_plot(
        traces,
        zoom_png,
        xmax=180.0,
        title_suffix="first 180 s",
        dispatch_label=dispatch_label,
    )

    print(f"summary_csv={summary_csv}")
    print(f"full_png={full_png}")
    print(f"zoom_png={zoom_png}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
