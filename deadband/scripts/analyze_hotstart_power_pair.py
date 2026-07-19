#!/usr/bin/env python3
"""
Trace one hot-start dispatch pair and export governor power-path signals.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_dispatch_tds as rdt
from compare_dispatch_pair_hotstart import (
    AGC_ALLOCATION_HEADROOM,
    AGC_ALLOCATION_MODES,
    AGC_ANTI_WINDUP_FREEZE,
    AGC_ANTI_WINDUP_OFF,
    activate_dispatch_target_transition,
    apply_agc_dispatch_update,
    apply_second_dispatch_targets,
    backcalculate_ace_integral_partial,
    compute_bf,
    dispatch_offset,
    initial_agc_aw_state,
    prepare_system,
    run_segment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-dispatch-json", type=Path, required=True)
    parser.add_argument("--second-dispatch-json", type=Path, required=True)
    parser.add_argument("--curve-file", type=Path, required=True)
    parser.add_argument("--dyn-case", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--dispatch-interval", type=int, default=900)
    parser.add_argument("--agc-interval", type=int, default=4)
    parser.add_argument("--kp", type=float, default=0.1)
    parser.add_argument("--ki", type=float, default=0.01)
    parser.add_argument(
        "--agc-anti-windup-mode",
        choices=(AGC_ANTI_WINDUP_OFF, AGC_ANTI_WINDUP_FREEZE),
        default=AGC_ANTI_WINDUP_FREEZE,
    )
    parser.add_argument(
        "--agc-allocation-mode",
        choices=AGC_ALLOCATION_MODES,
        default=AGC_ALLOCATION_HEADROOM,
    )
    parser.add_argument("--disable-der-agc", action="store_true")
    parser.add_argument("--init-mode", choices=("dispatch", "first"), default="first")
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


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    first = rdt.DispatchRecord.from_json(args.first_dispatch_json)
    second = rdt.DispatchRecord.from_json(args.second_dispatch_json)
    curve = rdt.load_curve(args.curve_file)

    sa, ctx = prepare_system(
        dispatch_record=first,
        curve=curve,
        dyn_case=args.dyn_case,
        dispatch_interval=args.dispatch_interval,
        init_mode=args.init_mode,
        wind_prefixes=rdt.DEFAULT_WIND_PREFIXES,
        solar_prefixes=rdt.DEFAULT_SOLAR_PREFIXES,
    )
    ctx["link"] = rdt.configure_der_agc_participation(
        sa,
        ctx["link"],  # type: ignore[arg-type]
        enable_der_agc=not args.disable_der_agc,
    )

    first_transition = apply_second_dispatch_targets(
        sa,
        ctx["link"],  # type: ignore[arg-type]
        first,
        apply_governor_targets=True,
        apply_dg_targets=False,
        duration_seconds=args.dispatch_interval,
        schedule_mode=args.governor_target_schedule,
        next_dispatch_record=None,
        basepoint_ramp_floor_frac_pmax_per_min=args.governor_basepoint_ramp_floor_frac_pmax_per_min,
        basepoint_ramp_gap_factor=args.governor_basepoint_ramp_gap_factor,
    )
    first_transition["ramp_seconds"] = 0
    activate_dispatch_target_transition(sa, first_transition, step=0)

    bf_first = compute_bf(sa, first)
    _, _, ace_integral, ace_raw = run_segment(
        sa=sa,
        ctx=ctx,
        start_offset=dispatch_offset(first, args.dispatch_interval),
        duration_seconds=args.dispatch_interval,
        agc_interval=args.agc_interval,
        kp=args.kp,
        ki=args.ki,
        bf=bf_first,
        ace_integral=0.0,
        ace_raw=0.0,
        local_start=0.0,
        include_initial=True,
        dispatch_target_transition=first_transition,
        gov_output_ramp_frac_pmax_per_min=0.0,
        dg_output_ramp_frac_pmax_per_min=0.0,
        agc_anti_windup_mode=args.agc_anti_windup_mode,
        agc_allocation_mode=args.agc_allocation_mode,
    )

    ctx2 = ctx.copy()
    ctx2["link"] = rdt.configure_der_agc_participation(
        sa,
        rdt.build_andes_link(sa),
        enable_der_agc=not args.disable_der_agc,
    )
    second_transition = apply_second_dispatch_targets(
        sa,
        ctx2["link"],  # type: ignore[arg-type]
        second,
        apply_governor_targets=True,
        apply_dg_targets=False,
        duration_seconds=args.dispatch_interval,
        schedule_mode=args.governor_target_schedule,
        next_dispatch_record=None,
        basepoint_ramp_floor_frac_pmax_per_min=args.governor_basepoint_ramp_floor_frac_pmax_per_min,
        basepoint_ramp_gap_factor=args.governor_basepoint_ramp_gap_factor,
    )
    second_transition["ramp_seconds"] = 0
    activate_dispatch_target_transition(sa, second_transition, step=0)

    link = ctx2["link"]  # type: ignore[assignment]
    gov_all_idx = [idx for idx in link["gov_idx"].tolist() if pd.notna(idx)]
    gov47_idx = link.loc[link["stg_idx"] == 47, "gov_idx"].iloc[0]
    syn47_idx = sa.TurbineGov.get(src="syn", attr="v", idx=[gov47_idx])[0]
    bf_second = compute_bf(sa, second)
    current_tf = float(sa.dae.t)
    cycle_req = 0.0
    cycle_applied = 0.0
    aw_state = initial_agc_aw_state()
    rows: list[dict[str, float | str]] = []

    for step in range(args.dispatch_interval):
        if step > 0:
            activate_dispatch_target_transition(sa, second_transition, step)
            if step % args.agc_interval == 0:
                meta = apply_agc_dispatch_update(
                    sa=sa,
                    link=link,
                    bf=bf_second,
                    ace_raw=ace_raw,
                    pext_max=ctx2["pext_max"],  # type: ignore[index]
                    agc_allocation_mode=args.agc_allocation_mode,
                    gov_output_ramp_frac_pmax_per_min=0.0,
                    dg_output_ramp_frac_pmax_per_min=0.0,
                    agc_anti_windup_mode=args.agc_anti_windup_mode,
                    prev_freeze_active=aw_state["freeze_active"],
                    prev_freeze_on_streak=aw_state["freeze_on_streak"],
                    prev_freeze_off_streak=aw_state["freeze_off_streak"],
                    prev_freeze_dir=aw_state["freeze_dir"],
                )
                cycle_req = float(meta["agc_request_sum_total"])
                cycle_applied = float(meta["agc_applied_sum_total"])
                aw_state["freeze_active"] = int(meta["agc_freeze_active"])
                aw_state["freeze_on_streak"] = int(meta["agc_freeze_streak"])
                aw_state["freeze_off_streak"] = int(meta["agc_unfreeze_streak"])
                aw_state["freeze_dir"] = int(meta["agc_freeze_dir"])

            offset = dispatch_offset(second, args.dispatch_interval) + step
            kload = curve["Load"].iloc[offset]
            sa.PQ.set(src="Ppf", idx=ctx2["pq_idx"], attr="v", value=kload * ctx2["sap0"])  # type: ignore[index]
            sa.PQ.set(src="Qpf", idx=ctx2["pq_idx"], attr="v", value=kload * ctx2["saq0"])  # type: ignore[index]

            wind = curve["Wind"].iloc[offset]
            sa.PVD1.set(src="pref0", idx=ctx2["pvd1_w2t"], attr="v", value=wind * ctx2["p0_w2t"])  # type: ignore[index]
            solar = curve["PV"].iloc[offset]
            sa.PVD1.set(src="pref0", idx=ctx2["pvd1_pv"], attr="v", value=solar * ctx2["p0_pv"])  # type: ignore[index]

            current_tf += 1.0
            sa.TDS.config.tf = current_tf
            sa.TDS.run()
            if sa.exit_code != 0:
                raise RuntimeError(f"TDS failed at step={step} for {second.label}")

            ace_sum = float(np.asarray(sa.ACEc.ace.v, dtype=float).sum())
            ace_raw = -(args.kp * ace_sum + args.ki * ace_integral)
            if aw_state["freeze_active"]:
                ace_integral, ace_raw = backcalculate_ace_integral_partial(
                    ace_integral=ace_integral,
                    kp=args.kp,
                    ki=args.ki,
                    ace_sum=ace_sum,
                    ace_raw=float(ace_raw),
                    agc_request_sum_total=cycle_req,
                    agc_applied_sum_total=cycle_applied,
                )
            else:
                ace_integral = ace_integral + ace_sum

        gov_paux = float(np.sum(np.asarray(sa.TurbineGov.get(src="paux0", attr="v", idx=gov_all_idx), dtype=float)))
        gov_pref = float(np.sum(np.asarray(sa.TurbineGov.get(src="pref0", attr="v", idx=gov_all_idx), dtype=float)))
        gov_db = float(
            np.sum(
                -np.asarray(sa.TurbineGov.get(src="DB_y", attr="v", idx=gov_all_idx), dtype=float)
                * np.asarray(sa.TurbineGov.get(src="gain", attr="v", idx=gov_all_idx), dtype=float)
            )
        )
        gov_syn = sa.TurbineGov.get(src="syn", attr="v", idx=gov_all_idx)
        gov_pe = float(np.sum(np.asarray(sa.SynGen.get(src="Pe", attr="v", idx=gov_syn), dtype=float)))
        gen47_pe = float(sa.SynGen.get(src="Pe", attr="v", idx=[syn47_idx])[0])
        gen47_pref = float(sa.TurbineGov.get(src="pref0", attr="v", idx=[gov47_idx])[0])
        gen47_paux = float(sa.TurbineGov.get(src="paux0", attr="v", idx=[gov47_idx])[0])
        gen47_pd = float(sa.TurbineGov.get(src="pd", attr="v", idx=[gov47_idx])[0])
        gen47_lag = float(sa.TurbineGov.get(src="LAG_y", attr="v", idx=[gov47_idx])[0])
        gen47_ll = float(sa.TurbineGov.get(src="LL_y", attr="v", idx=[gov47_idx])[0])
        gen47_pout = float(sa.TurbineGov.get(src="pout", attr="v", idx=[gov47_idx])[0])
        freq = float((sa.ACEc.f.v[0] - 1.0) * sa.config.freq)
        rows.append({
            "segment": second.label,
            "time_s": float(step),
            "freq_hz": freq,
            "gov_paux_sum": gov_paux,
            "gov_pref_sum": gov_pref,
            "gov_droop_sum": gov_db,
            "gov_pe_sum": gov_pe,
            "gen47_pe": gen47_pe,
            "gen47_pref": gen47_pref,
            "gen47_paux": gen47_paux,
            "gen47_pd": gen47_pd,
            "gen47_lag_y": gen47_lag,
            "gen47_ll_y": gen47_ll,
            "gen47_pout": gen47_pout,
            "agc_req_sum": cycle_req,
            "agc_applied_sum": cycle_applied,
        })

    trace = pd.DataFrame(rows)
    out_csv = args.results_dir / f"{first.label}_{second.label}_power_trace.csv"
    trace.to_csv(out_csv, index=False)
    print(out_csv)


if __name__ == "__main__":
    main()
