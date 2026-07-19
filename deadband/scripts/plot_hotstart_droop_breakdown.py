#!/usr/bin/env python3
"""
Plot hot-start response breakdown for one dispatch pair.

The script runs the first dispatch to build the hot-start state, then traces the
second dispatch and records:
- frequency deviation
- conventional governor droop contribution sum
- PVD1 droop contribution sum
- ESD1 droop contribution sum
- actual output deltas for governor / PVD1 / ESD1
- active deadband device counts
- AGC command sums for context
"""

from __future__ import annotations

import argparse
import sys
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
    backcalculate_ace_integral_partial,
    compute_bf,
    dispatch_offset,
    initial_agc_aw_state,
    normalize_agc_aw_state,
    prepare_system,
    run_segment,
)
from run_dispatch_hotstart import configure_all_der_deadband


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-dispatch-json", type=Path, required=True)
    parser.add_argument("--second-dispatch-json", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--curve-file", type=Path, required=True)
    parser.add_argument("--dyn-case", type=Path, required=True)
    parser.add_argument("--stable-dyn-case", type=Path, default=rdt.DEFAULT_STABLE_DYN_CASE)
    parser.add_argument("--dispatch-interval", type=int, default=900)
    parser.add_argument("--agc-interval", type=int, default=4)
    parser.add_argument("--kp", type=float, default=0.03)
    parser.add_argument("--ki", type=float, default=0.003)
    parser.add_argument(
        "--agc-anti-windup-mode",
        choices=(AGC_ANTI_WINDUP_OFF, AGC_ANTI_WINDUP_FREEZE),
        default=AGC_ANTI_WINDUP_OFF,
    )
    parser.add_argument("--agc-gov-output-ramp-frac-pmax-per-min", type=float, default=0.0)
    parser.add_argument("--agc-dg-output-ramp-frac-pmax-per-min", type=float, default=0.0)
    parser.add_argument("--disable-der-agc", action="store_true")
    parser.add_argument("--disable-pvd-agc", action="store_true")
    parser.add_argument("--disable-esd-agc", action="store_true")
    parser.add_argument("--der-deadband-hz", type=float, default=None)
    parser.add_argument("--pvd1-base-ddn", type=float, default=None)
    parser.add_argument("--esd1-base-ddn", type=float, default=None)
    parser.add_argument("--pvd1-tfdb", type=float, default=None)
    parser.add_argument("--esd1-tfdb", type=float, default=None)
    parser.add_argument("--wind-pref-alpha", type=float, default=1.0)
    parser.add_argument("--solar-pref-alpha", type=float, default=1.0)
    parser.add_argument("--wind-deadband-hz", type=float, default=None)
    parser.add_argument("--solar-deadband-hz", type=float, default=None)
    parser.add_argument("--esd-deadband-hz", type=float, default=None)
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
    parser.add_argument("--zoom-seconds", type=int, default=180)
    return parser.parse_args()


def _sum_series(values: np.ndarray) -> float:
    return float(values.sum()) if values.size else 0.0


def _count_active(values: np.ndarray, tol: float = 1e-9) -> int:
    return int(np.sum(np.abs(values) > tol)) if values.size else 0


def _sum_pvd_output(sa) -> float:
    if not hasattr(sa, "PVD1") or sa.PVD1.n == 0:
        return 0.0
    return float(np.sum(np.asarray(sa.PVD1.Ipout_y.v, dtype=float) * np.asarray(sa.PVD1.v.v, dtype=float)))


def _sum_esd_output(sa) -> float:
    if not hasattr(sa, "ESD1") or sa.ESD1.n == 0:
        return 0.0
    return float(np.sum(np.asarray(sa.ESD1.Ipout_y.v, dtype=float) * np.asarray(sa.ESD1.v.v, dtype=float)))


def _sum_pref(model) -> float:
    if not hasattr(model, "pref0") or model.n == 0:
        return 0.0
    return _sum_series(np.asarray(model.pref0.v, dtype=float))


def _sum_pavail(model) -> float:
    if not hasattr(model, "pavail0") or model.n == 0:
        return 0.0
    return _sum_series(np.asarray(model.pavail0.v, dtype=float))


def _sum_pvd_droop(sa) -> float:
    if not hasattr(sa, "PVD1") or sa.PVD1.n == 0:
        return 0.0
    if hasattr(sa.PVD1, "DBRamp_y") and hasattr(sa.PVD1.DBRamp_y, "v"):
        return _sum_series(np.asarray(sa.PVD1.DBRamp_y.v, dtype=float))
    if hasattr(sa.PVD1, "DBLag_y") and hasattr(sa.PVD1.DBLag_y, "v"):
        return _sum_series(np.asarray(sa.PVD1.DBLag_y.v, dtype=float))
    return _sum_series(np.asarray(sa.PVD1.DB_y.v, dtype=float))


def _sum_esd_droop(sa) -> float:
    if not hasattr(sa, "ESD1") or sa.ESD1.n == 0:
        return 0.0
    if hasattr(sa.ESD1, "DBRamp_y") and hasattr(sa.ESD1.DBRamp_y, "v"):
        return _sum_series(np.asarray(sa.ESD1.DBRamp_y.v, dtype=float))
    if hasattr(sa.ESD1, "DBLag_y") and hasattr(sa.ESD1.DBLag_y, "v"):
        return _sum_series(np.asarray(sa.ESD1.DBLag_y.v, dtype=float))
    return _sum_series(np.asarray(sa.ESD1.DB_y.v, dtype=float))


def _sum_pvd_pext(sa) -> float:
    if not hasattr(sa, "PVD1") or sa.PVD1.n == 0:
        return 0.0
    return _sum_series(np.asarray(sa.PVD1.Pext0.v, dtype=float))


def _sum_esd_pext(sa) -> float:
    if not hasattr(sa, "ESD1") or sa.ESD1.n == 0:
        return 0.0
    return _sum_series(np.asarray(sa.ESD1.Pext0.v, dtype=float))


def _esd_soc_values(sa) -> np.ndarray:
    if not hasattr(sa, "ESD1") or sa.ESD1.n == 0:
        return np.asarray([], dtype=float)
    if hasattr(sa.ESD1, "SOC") and hasattr(sa.ESD1.SOC, "v"):
        return np.asarray(sa.ESD1.SOC.v, dtype=float)
    if hasattr(sa.ESD1, "pIG_y") and hasattr(sa.ESD1.pIG_y, "v"):
        return np.asarray(sa.ESD1.pIG_y.v, dtype=float)
    return np.asarray([], dtype=float)


def _count_pvd_active(sa) -> int:
    if not hasattr(sa, "PVD1") or sa.PVD1.n == 0:
        return 0
    if hasattr(sa.PVD1, "DBRamp_y") and hasattr(sa.PVD1.DBRamp_y, "v"):
        return _count_active(np.asarray(sa.PVD1.DBRamp_y.v, dtype=float))
    if hasattr(sa.PVD1, "DBLag_y") and hasattr(sa.PVD1.DBLag_y, "v"):
        return _count_active(np.asarray(sa.PVD1.DBLag_y.v, dtype=float))
    return _count_active(np.asarray(sa.PVD1.DB_y.v, dtype=float))


def _count_esd_active(sa) -> int:
    if not hasattr(sa, "ESD1") or sa.ESD1.n == 0:
        return 0
    if hasattr(sa.ESD1, "DBRamp_y") and hasattr(sa.ESD1.DBRamp_y, "v"):
        return _count_active(np.asarray(sa.ESD1.DBRamp_y.v, dtype=float))
    if hasattr(sa.ESD1, "DBLag_y") and hasattr(sa.ESD1.DBLag_y, "v"):
        return _count_active(np.asarray(sa.ESD1.DBLag_y.v, dtype=float))
    return _count_active(np.asarray(sa.ESD1.DB_y.v, dtype=float))


def _sum_governor_droop(sa, gov_idx: list[int]) -> float:
    if not gov_idx:
        return 0.0
    db_y = np.asarray(sa.TurbineGov.get(src="DB_y", attr="v", idx=gov_idx), dtype=float)
    gain = np.asarray(sa.TurbineGov.get(src="gain", attr="v", idx=gov_idx), dtype=float)
    return _sum_series(-db_y * gain)


def _count_governor_active(sa, gov_idx: list[int]) -> int:
    if not gov_idx:
        return 0
    db_y = np.asarray(sa.TurbineGov.get(src="DB_y", attr="v", idx=gov_idx), dtype=float)
    return _count_active(db_y)


def _sum_governor_output(sa, gov_idx: list[int]) -> float:
    if not gov_idx:
        return 0.0
    gov_syn = sa.TurbineGov.get(src="syn", attr="v", idx=gov_idx)
    pe = np.asarray(sa.SynGen.get(src="Pe", attr="v", idx=gov_syn), dtype=float)
    return _sum_series(pe)


def trace_second_segment(
    *,
    sa,
    ctx: dict[str, object],
    dispatch_record: rdt.DispatchRecord,
    duration_seconds: int,
    agc_interval: int,
    kp: float,
    ki: float,
    bf: np.ndarray,
    ace_integral: float,
    ace_raw: float,
    dispatch_target_transition: dict[str, object],
    gov_output_ramp_frac_pmax_per_min: float,
    dg_output_ramp_frac_pmax_per_min: float,
    agc_anti_windup_mode: str,
    agc_aw_state_init: dict[str, int] | None = None,
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
    wind_pref_alpha = float(ctx.get("wind_pref_alpha", 1.0))
    solar_pref_alpha = float(ctx.get("solar_pref_alpha", 1.0))

    gov_all_idx = [idx for idx in link["gov_idx"].tolist() if pd.notna(idx)]
    dg_all_idx = [idx for idx in link["dg_idx"].tolist() if pd.notna(idx)]
    start_offset = dispatch_offset(dispatch_record, duration_seconds)

    last_gov_req_sum = 0.0
    last_dg_req_sum = 0.0
    cycle_agc_request_sum_total = 0.0
    cycle_agc_applied_sum_total = 0.0
    agc_aw_state = normalize_agc_aw_state(agc_aw_state_init)

    rows: list[dict[str, float | int]] = []

    def snapshot(ts: float, agc_update: int) -> None:
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
            "gov_droop_sum": _sum_governor_droop(sa, gov_all_idx),
            "gov_active_count": _count_governor_active(sa, gov_all_idx),
            "gov_pe_sum": _sum_governor_output(sa, gov_all_idx),
            "pvd_droop_sum": _sum_pvd_droop(sa),
            "pvd_active_count": _count_pvd_active(sa),
            "pvd_pe_sum": _sum_pvd_output(sa),
            "pvd_pref_sum": _sum_pref(sa.PVD1),
            "pvd_pavail_sum": _sum_pavail(sa.PVD1),
            "pvd_pext0_sum": _sum_pvd_pext(sa),
            "esd_droop_sum": _sum_esd_droop(sa),
            "esd_active_count": _count_esd_active(sa),
            "esd_pe_sum": _sum_esd_output(sa),
            "esd_pref_sum": _sum_pref(sa.ESD1),
            "esd_pext0_sum": _sum_esd_pext(sa),
            "esd_soc_mean": _sum_series(_esd_soc_values(sa)) / max(len(_esd_soc_values(sa)), 1),
            "esd_soc_min": float(np.min(_esd_soc_values(sa))) if _esd_soc_values(sa).size else np.nan,
            "esd_soc_max": float(np.max(_esd_soc_values(sa))) if _esd_soc_values(sa).size else np.nan,
            "esd_soc_1": float(_esd_soc_values(sa)[0]) if _esd_soc_values(sa).size >= 1 else np.nan,
            "esd_soc_2": float(_esd_soc_values(sa)[1]) if _esd_soc_values(sa).size >= 2 else np.nan,
            "gov_paux0_sum": _sum_series(gov_paux0),
            "dg_pext0_sum": _sum_series(dg_pext0),
            "gov_pref0_sum": _sum_series(gov_pref0),
            "agc_request_sum_total": float(last_gov_req_sum + last_dg_req_sum),
            "agc_update": int(agc_update),
        })

    snapshot(0.0, agc_update=0)

    current_tf = float(sa.dae.t)
    for step in range(1, duration_seconds):
        activate_dispatch_target_transition(sa, dispatch_target_transition, step)

        agc_update = 0
        if step % agc_interval == 0:
            agc_update = 1
            meta = apply_agc_dispatch_update(
                sa=sa,
                link=link,
                bf=bf,
                ace_raw=ace_raw,
                pext_max=pext_max,
                gov_output_ramp_frac_pmax_per_min=gov_output_ramp_frac_pmax_per_min,
                dg_output_ramp_frac_pmax_per_min=dg_output_ramp_frac_pmax_per_min,
                agc_anti_windup_mode=agc_anti_windup_mode,
                prev_freeze_active=agc_aw_state["freeze_active"],
                prev_freeze_on_streak=agc_aw_state["freeze_on_streak"],
                prev_freeze_off_streak=agc_aw_state["freeze_off_streak"],
                prev_freeze_dir=agc_aw_state["freeze_dir"],
            )
            last_gov_req_sum = float(meta["agov_request_sum"])
            last_dg_req_sum = float(meta["adg_request_sum"])
            cycle_agc_request_sum_total = float(meta["agc_request_sum_total"])
            cycle_agc_applied_sum_total = float(meta["agc_applied_sum_total"])
            agc_aw_state["freeze_active"] = int(meta["agc_freeze_active"])
            agc_aw_state["freeze_on_streak"] = int(meta["agc_freeze_streak"])
            agc_aw_state["freeze_off_streak"] = int(meta["agc_unfreeze_streak"])
            agc_aw_state["freeze_dir"] = int(meta["agc_freeze_dir"])

        kload = curve["Load"].iloc[start_offset + step]
        sa.PQ.set(src="Ppf", idx=pq_idx, attr="v", value=kload * sap0)
        sa.PQ.set(src="Qpf", idx=pq_idx, attr="v", value=kload * saq0)

        wind = curve["Wind"].iloc[start_offset + step]
        wind_pavail = rdt.der_available_from_curve(wind, p0_w2t)
        wind_pref = rdt.der_pref_from_available(wind_pavail, alpha=wind_pref_alpha)
        sa.PVD1.set(src="pref0", idx=pvd1_w2t, attr="v", value=wind_pref)
        sa.PVD1.set(src="pavail0", idx=pvd1_w2t, attr="v", value=wind_pavail)

        solar = curve["PV"].iloc[start_offset + step]
        solar_pavail = rdt.der_available_from_curve(solar, p0_pv)
        solar_pref = rdt.der_pref_from_available(solar_pavail, alpha=solar_pref_alpha)
        sa.PVD1.set(src="pref0", idx=pvd1_pv, attr="v", value=solar_pref)
        sa.PVD1.set(src="pavail0", idx=pvd1_pv, attr="v", value=solar_pavail)

        current_tf += 1.0
        sa.TDS.config.tf = current_tf
        sa.TDS.run()
        if sa.exit_code != 0:
            raise RuntimeError(f"TDS failed at local step={step} with exit_code={sa.exit_code}")

        snapshot(float(step), agc_update=agc_update)

        ace_sum = float(np.asarray(sa.ACEc.ace.v, dtype=float).sum())
        ace_raw = -(kp * ace_sum + ki * ace_integral)
        if agc_aw_state["freeze_active"]:
            ace_integral, ace_raw = backcalculate_ace_integral_partial(
                ace_integral=ace_integral,
                kp=kp,
                ki=ki,
                ace_sum=ace_sum,
                ace_raw=float(ace_raw),
                agc_request_sum_total=cycle_agc_request_sum_total,
                agc_applied_sum_total=cycle_agc_applied_sum_total,
            )
        else:
            ace_integral = ace_integral + ace_sum

    trace = pd.DataFrame(rows)
    for col in ("gov_pe_sum", "pvd_pe_sum", "esd_pe_sum", "gov_pref0_sum", "pvd_pref_sum", "esd_pref_sum"):
        trace[f"{col}_delta"] = trace[col] - float(trace[col].iloc[0])
    return trace


def make_figure(
    trace: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    xmax: float,
) -> None:
    fig, axes = plt.subplots(6, 1, figsize=(15, 18), constrained_layout=True, sharex=True)

    axes[0].plot(trace["time_s"], trace["freq_dev_hz"], color="#111111", linewidth=1.8)
    axes[0].axhline(0.036, color="#666666", linestyle=":", linewidth=1.0)
    axes[0].axhline(-0.036, color="#666666", linestyle=":", linewidth=1.0)
    axes[0].axhline(0.0, color="#999999", linestyle="-", linewidth=0.8)
    axes[0].set_ylabel("Freq [Hz]")
    axes[0].set_title(title)

    axes[1].plot(trace["time_s"], trace["gov_droop_sum"], color="#1f77b4", linewidth=1.6, label="TGOV1NDB droop term")
    axes[1].plot(trace["time_s"], trace["pvd_droop_sum"], color="#ff7f0e", linewidth=1.6, label="PVD1 droop term")
    axes[1].plot(trace["time_s"], trace["esd_droop_sum"], color="#2ca02c", linewidth=1.6, label="ESD1 droop term")
    axes[1].axhline(0.0, color="#999999", linestyle="-", linewidth=0.8)
    axes[1].set_ylabel("Droop term [pu]")
    axes[1].legend(loc="upper right", frameon=True)

    axes[2].plot(trace["time_s"], trace["gov_active_count"], color="#1f77b4", linewidth=1.5, label="TGOV1NDB active")
    axes[2].plot(trace["time_s"], trace["pvd_active_count"], color="#ff7f0e", linewidth=1.5, label="PVD1 active")
    axes[2].plot(trace["time_s"], trace["esd_active_count"], color="#2ca02c", linewidth=1.5, label="ESD1 active")
    axes[2].set_ylabel("Active count")
    axes[2].legend(loc="upper right", frameon=True)

    axes[3].plot(trace["time_s"], trace["gov_pe_sum_delta"], color="#1f77b4", linewidth=1.6, label="Gov Pe delta")
    axes[3].plot(trace["time_s"], trace["pvd_pe_sum_delta"], color="#ff7f0e", linewidth=1.6, label="PVD Pe delta")
    axes[3].plot(trace["time_s"], trace["esd_pe_sum_delta"], color="#2ca02c", linewidth=1.6, label="ESD Pe delta")
    axes[3].axhline(0.0, color="#999999", linestyle="-", linewidth=0.8)
    axes[3].set_ylabel("Output delta [pu]")
    axes[3].legend(loc="upper right", frameon=True)

    axes[4].step(trace["time_s"], trace["gov_paux0_sum"], where="post", color="#1f77b4", linewidth=1.5, label="Gov paux0")
    axes[4].step(trace["time_s"], trace["dg_pext0_sum"], where="post", color="#2ca02c", linewidth=1.5, label="DER Pext0")
    axes[4].plot(trace["time_s"], trace["gov_pref0_sum_delta"], color="#aa3377", linewidth=1.4, label="Gov pref0 delta")
    axes[4].axhline(0.0, color="#999999", linestyle="-", linewidth=0.8)
    axes[4].set_ylabel("AGC / pref [pu]")
    axes[4].legend(loc="upper right", frameon=True)

    axes[5].plot(trace["time_s"], trace["esd_soc_1"], color="#2ca02c", linewidth=1.5, label="ESS_1 SOC")
    axes[5].plot(trace["time_s"], trace["esd_soc_2"], color="#1b9e77", linewidth=1.5, label="ESS_2 SOC")
    axes[5].plot(trace["time_s"], trace["esd_soc_mean"], color="#111111", linewidth=1.2, linestyle="--", label="SOC mean")
    axes[5].set_ylabel("ESD SOC [-]")
    axes[5].set_xlabel("Time [s]")
    axes[5].legend(loc="upper right", frameon=True)

    for ax in axes:
        ax.set_xlim(0.0, xmax)
        ax.grid(True, alpha=0.25)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    first = rdt.DispatchRecord.from_json(args.first_dispatch_json)
    second = rdt.DispatchRecord.from_json(args.second_dispatch_json)
    curve = rdt.load_curve(args.curve_file)
    for record in (first, second):
        rdt.validate_curve_window(curve, record, args.dispatch_interval)

    dyn_case = rdt.adapt_dyn_case(args.dyn_case, args.stable_dyn_case)
    wind_prefixes = rdt.DEFAULT_WIND_PREFIXES
    solar_prefixes = rdt.DEFAULT_SOLAR_PREFIXES

    sa, ctx = prepare_system(
        dispatch_record=first,
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
    if args.der_deadband_hz is not None and args.der_deadband_hz > 0.0:
        configure_all_der_deadband(
            sa,
            traditional_governor_deadband_hz=None,
            der_deadband_hz=float(args.der_deadband_hz),
            der_base_ddn=None,
            pvd1_base_ddn=args.pvd1_base_ddn,
            esd1_base_ddn=args.esd1_base_ddn,
            esd1_ddn_scale=1.0,
        )
    rdt.apply_resource_deadband_overrides(
        sa,
        wind_prefixes=wind_prefixes,
        solar_prefixes=solar_prefixes,
        wind_deadband_hz=args.wind_deadband_hz,
        solar_deadband_hz=args.solar_deadband_hz,
        esd_deadband_hz=args.esd_deadband_hz,
    )
    if args.pvd1_tfdb is not None and hasattr(sa, "PVD1") and sa.PVD1.n:
        sa.PVD1.set(src="Tfdb", idx=sa.PVD1.idx.v, attr="v", value=np.full(sa.PVD1.n, float(args.pvd1_tfdb)))
    if args.esd1_tfdb is not None and hasattr(sa, "ESD1") and sa.ESD1.n:
        sa.ESD1.set(src="Tfdb", idx=sa.ESD1.idx.v, attr="v", value=np.full(sa.ESD1.n, float(args.esd1_tfdb)))

    first_transition = apply_second_dispatch_targets(
        sa,
        ctx["link"],  # type: ignore[arg-type]
        first,
        apply_governor_targets=True,
        apply_dg_targets=False,
        duration_seconds=(
            args.dispatch_interval
            if args.governor_target_schedule in ("midpoint_trajectory", "ramp_limited_basepoint")
            else None
        ),
        schedule_mode=args.governor_target_schedule,
        next_dispatch_record=second if args.governor_target_schedule == "midpoint_trajectory" else None,
        basepoint_ramp_floor_frac_pmax_per_min=args.governor_basepoint_ramp_floor_frac_pmax_per_min,
        basepoint_ramp_gap_factor=args.governor_basepoint_ramp_gap_factor,
    )
    first_transition["ramp_seconds"] = 0
    activate_dispatch_target_transition(sa, first_transition, step=0)

    bf_first = compute_bf(sa, first)
    _, _, ace_integral_end, ace_raw_end = run_segment(
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
        gov_output_ramp_frac_pmax_per_min=args.agc_gov_output_ramp_frac_pmax_per_min,
        dg_output_ramp_frac_pmax_per_min=args.agc_dg_output_ramp_frac_pmax_per_min,
        agc_anti_windup_mode=args.agc_anti_windup_mode,
        wind_pref_alpha=args.wind_pref_alpha,
        solar_pref_alpha=args.solar_pref_alpha,
    )

    ctx2 = ctx.copy()
    ctx2["link"] = rdt.configure_der_agc_participation(
        sa,
        rdt.build_andes_link(sa),
        enable_der_agc=not args.disable_der_agc,
        enable_pvd_agc=not args.disable_pvd_agc,
        enable_esd_agc=not args.disable_esd_agc,
    )
    second_transition = apply_second_dispatch_targets(
        sa,
        ctx2["link"],  # type: ignore[arg-type]
        second,
        apply_governor_targets=True,
        apply_dg_targets=False,
        duration_seconds=(
            args.dispatch_interval
            if args.governor_target_schedule in ("midpoint_trajectory", "ramp_limited_basepoint")
            else None
        ),
        schedule_mode=args.governor_target_schedule,
        next_dispatch_record=None,
        basepoint_ramp_floor_frac_pmax_per_min=args.governor_basepoint_ramp_floor_frac_pmax_per_min,
        basepoint_ramp_gap_factor=args.governor_basepoint_ramp_gap_factor,
    )
    second_transition["ramp_seconds"] = 0
    activate_dispatch_target_transition(sa, second_transition, step=0)

    bf_second = compute_bf(sa, second)
    trace = trace_second_segment(
        sa=sa,
        ctx=ctx2,
        dispatch_record=second,
        duration_seconds=args.dispatch_interval,
        agc_interval=args.agc_interval,
        kp=args.kp,
        ki=args.ki,
        bf=bf_second,
        ace_integral=ace_integral_end,
        ace_raw=ace_raw_end,
        dispatch_target_transition=second_transition,
        gov_output_ramp_frac_pmax_per_min=args.agc_gov_output_ramp_frac_pmax_per_min,
        dg_output_ramp_frac_pmax_per_min=args.agc_dg_output_ramp_frac_pmax_per_min,
        agc_anti_windup_mode=args.agc_anti_windup_mode,
    )

    label = f"{first.label}_{second.label}_hotstart_breakdown"
    csv_path = args.results_dir / f"{label}.csv"
    trace.to_csv(csv_path, index=False)

    full_path = args.results_dir / f"{label}_full.png"
    zoom_path = args.results_dir / f"{label}_zoom.png"
    title = (
        f"{first.label} -> {second.label} hot-start breakdown | "
        f"KP={args.kp:g}, KI={args.ki:g}, AGC={args.agc_interval}s"
    )
    make_figure(trace, full_path, title=title, xmax=float(args.dispatch_interval))
    make_figure(trace, zoom_path, title=title + f" | first {args.zoom_seconds}s", xmax=float(args.zoom_seconds))

    summary = pd.DataFrame([{
        "first_dispatch": first.label,
        "second_dispatch": second.label,
        "kp": float(args.kp),
        "ki": float(args.ki),
        "agc_interval": int(args.agc_interval),
        "freq_max_abs_hz": float(np.max(np.abs(trace["freq_dev_hz"].to_numpy(dtype=float)))),
        "freq_mean_abs_hz": float(np.mean(np.abs(trace["freq_dev_hz"].to_numpy(dtype=float)))),
        "gov_droop_abs_max": float(np.max(np.abs(trace["gov_droop_sum"].to_numpy(dtype=float)))),
        "pvd_droop_abs_max": float(np.max(np.abs(trace["pvd_droop_sum"].to_numpy(dtype=float)))),
        "esd_droop_abs_max": float(np.max(np.abs(trace["esd_droop_sum"].to_numpy(dtype=float)))),
        "pvd_pext0_abs_max": float(np.max(np.abs(trace["pvd_pext0_sum"].to_numpy(dtype=float)))),
        "esd_pext0_abs_max": float(np.max(np.abs(trace["esd_pext0_sum"].to_numpy(dtype=float)))),
        "gov_active_mean": float(np.mean(trace["gov_active_count"].to_numpy(dtype=float))),
        "pvd_active_mean": float(np.mean(trace["pvd_active_count"].to_numpy(dtype=float))),
        "esd_active_mean": float(np.mean(trace["esd_active_count"].to_numpy(dtype=float))),
        "gov_pe_delta_abs_max": float(np.max(np.abs(trace["gov_pe_sum_delta"].to_numpy(dtype=float)))),
        "pvd_pe_delta_abs_max": float(np.max(np.abs(trace["pvd_pe_sum_delta"].to_numpy(dtype=float)))),
        "esd_pe_delta_abs_max": float(np.max(np.abs(trace["esd_pe_sum_delta"].to_numpy(dtype=float)))),
        "pvd_throughput": float(np.trapezoid(np.abs((trace["pvd_pe_sum"] - trace["pvd_pref_sum"]).to_numpy(dtype=float)), trace["time_s"].to_numpy(dtype=float))),
        "esd_throughput": float(np.trapezoid(np.abs((trace["esd_pe_sum"] - trace["esd_pref_sum"]).to_numpy(dtype=float)), trace["time_s"].to_numpy(dtype=float))),
        "gov_droop_effort": float(np.trapezoid(np.abs(trace["gov_droop_sum"].to_numpy(dtype=float)), trace["time_s"].to_numpy(dtype=float))),
    }])
    summary_path = args.results_dir / f"{label}_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"trace_csv={csv_path}")
    print(f"full_plot={full_path}")
    print(f"zoom_plot={zoom_path}")
    print(f"summary_csv={summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
