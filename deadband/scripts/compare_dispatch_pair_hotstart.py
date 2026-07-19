#!/usr/bin/env python3
"""
Compare cold-start stitched dispatches against a hot-start second dispatch.

The hot-start workflow is:

1. Run the first dispatch interval with the regular deadband-demo TDS setup.
2. Save the terminal ANDES system snapshot and the external AGC integrator state.
3. Reload that snapshot and continue the second dispatch from the saved terminal
   dynamic state, while switching the AGC participation and governor basepoints
   to the second dispatch schedule.

This provides a practical "warm start" for the second dispatch without
re-initializing the dynamic model from a fresh power flow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_dispatch_tds as rdt
import andes
from andes.utils.snapshot import load_ss, save_ss
import hotstart_checkpoint as hcp

AGC_ANTI_WINDUP_OFF = "off"
AGC_ANTI_WINDUP_FREEZE = "freeze_on_saturation"
AGC_ALLOCATION_FIXED = rdt.AGC_ALLOCATION_FIXED
AGC_ALLOCATION_HEADROOM = rdt.AGC_ALLOCATION_HEADROOM
AGC_ALLOCATION_MODES = rdt.AGC_ALLOCATION_MODES
AGC_FREEZE_RATIO_THRESHOLD = 0.15
AGC_UNFREEZE_RATIO_THRESHOLD = 0.05
AGC_FREEZE_REQUIRED_STREAK = 2
AGC_UNFREEZE_REQUIRED_STREAK = 2
AGC_RATIO_EPS = 1e-9
AGC_BACKCALC_BLEND = 0.8


def dispatch_offset(dispatch_record: rdt.DispatchRecord, dispatch_interval: int) -> int:
    return dispatch_record.hour * 3600 + dispatch_record.dispatch * dispatch_interval


def compute_bf(sa: andes.system.System, dispatch_record: rdt.DispatchRecord) -> np.ndarray:
    stg = sa.StaticGen.get_all_idxes()
    stg_on = rdt.dispatch_online_mask(stg, dispatch_record)
    sn = sa.StaticGen.get(src="Sn", attr="v", idx=stg)
    denom = float((stg_on * sn).sum())
    if denom <= 0.0:
        raise RuntimeError("No online synchronous capacity found for AGC participation factors.")
    return stg_on * sn / denom


def initial_agc_aw_state() -> dict[str, int]:
    return {
        "freeze_active": 0,
        "freeze_on_streak": 0,
        "freeze_off_streak": 0,
        "freeze_dir": 0,
    }


def normalize_agc_aw_state(state: dict[str, object] | None) -> dict[str, int]:
    out = initial_agc_aw_state()
    if state is None:
        return out
    for key in out:
        if key in state:
            out[key] = int(state[key])
    return out


def backcalculate_ace_integral_partial(
    *,
    ace_integral: float,
    kp: float,
    ki: float,
    ace_sum: float,
    ace_raw: float,
    agc_request_sum_total: float,
    agc_applied_sum_total: float,
    blend: float = AGC_BACKCALC_BLEND,
) -> tuple[float, float]:
    """Blend the stored integral toward the executable aggregate AGC output."""
    if ki <= 0.0:
        return float(ace_integral), float(ace_raw)

    current_integral = float(ace_integral)
    current_ace_raw = float(ace_raw)
    if abs(agc_request_sum_total) <= AGC_RATIO_EPS:
        return current_integral, current_ace_raw

    applied_scale = agc_applied_sum_total / agc_request_sum_total
    if not np.isfinite(applied_scale):
        return current_integral, current_ace_raw

    applied_scale = float(np.clip(applied_scale, -1.0, 1.0))
    ace_raw_target = float(current_ace_raw * applied_scale)
    ace_integral_target = float(-(ace_raw_target + kp * ace_sum) / ki)

    alpha = float(np.clip(blend, 0.0, 1.0))
    ace_integral_blended = float(current_integral + alpha * (ace_integral_target - current_integral))
    ace_raw_blended = float(-(kp * ace_sum + ki * ace_integral_blended))
    return ace_integral_blended, ace_raw_blended


def prepare_system(
    dispatch_record: rdt.DispatchRecord,
    curve: pd.DataFrame,
    dyn_case: Path,
    dispatch_interval: int,
    init_mode: str,
    wind_prefixes: Iterable[str],
    solar_prefixes: Iterable[str],
    wind_pref_alpha: float = 1.0,
    solar_pref_alpha: float = 1.0,
) -> tuple[andes.system.System, dict[str, object]]:
    sa = andes.load(str(dyn_case), setup=False, no_output=True, default_config=True)
    sa.add("Output", dict(model="ACEc", varname="f"))
    sa.setup()

    link = rdt.build_andes_link(sa)
    pq_idx = sa.PQ.idx.v
    stg = sa.StaticGen.get_all_idxes()
    stg_w2t, stg_pv = rdt.pvd1_gen_subsets(sa, wind_prefixes, solar_prefixes)
    p0_w2t = sa.StaticGen.get(src="p0", attr="v", idx=stg_w2t)
    p0_pv = sa.StaticGen.get(src="p0", attr="v", idx=stg_pv)
    pvd1_w2t = sa.PVD1.find_idx(keys="gen", values=stg_w2t)
    pvd1_pv = sa.PVD1.find_idx(keys="gen", values=stg_pv)
    wind_pref_alpha = rdt.validate_pref_alpha(wind_pref_alpha, name="wind_pref_alpha")
    solar_pref_alpha = rdt.validate_pref_alpha(solar_pref_alpha, name="solar_pref_alpha")

    sap0 = sa.PQ.p0.v.copy()
    saq0 = sa.PQ.q0.v.copy()

    sa.StaticGen.set(src="p0", idx=dispatch_record.gen, attr="v", value=dispatch_record.pg)
    sa.Bus.set(src="v0", idx=dispatch_record.bus, attr="v", value=dispatch_record.vBus)
    sa.Bus.set(src="a0", idx=dispatch_record.bus, attr="v", value=dispatch_record.aBus)

    pv_bus = sa.PV.bus.v
    slack_bus = sa.Slack.bus.v
    v_pv = sa.Bus.get(src="v0", attr="v", idx=pv_bus)
    a_slack = sa.Bus.get(src="a0", attr="v", idx=slack_bus)
    sa.PV.set(src="v0", idx=sa.PV.idx.v, attr="v", value=v_pv)
    sa.Slack.set(src="a0", idx=sa.Slack.idx.v, attr="v", value=a_slack)

    sa.PQ.config.p2p = 1
    sa.PQ.config.q2q = 1
    sa.PQ.config.p2z = 0
    sa.PQ.config.q2z = 0
    sa.PQ.pq2z = 0
    sa.TDS.config.criteria = 0
    sa.TDS.config.no_tqdm = True

    init_load, init_wind, init_solar = rdt.resolve_initial_profile(
        curve=curve,
        dispatch_record=dispatch_record,
        duration_seconds=dispatch_interval,
        init_mode=init_mode,
    )
    init_wind_pavail = rdt.der_available_from_curve(init_wind, p0_w2t)
    init_solar_pavail = rdt.der_available_from_curve(init_solar, p0_pv)
    init_wind_pref = rdt.der_pref_from_available(init_wind_pavail, wind_pref_alpha)
    init_solar_pref = rdt.der_pref_from_available(init_solar_pavail, solar_pref_alpha)
    sa.PQ.set(src="p0", idx=pq_idx, attr="v", value=init_load * sap0)
    sa.PQ.set(src="q0", idx=pq_idx, attr="v", value=init_load * saq0)
    sa.StaticGen.set(src="p0", idx=stg_w2t, attr="v", value=init_wind_pref)
    sa.StaticGen.set(src="p0", idx=stg_pv, attr="v", value=init_solar_pref)
    if pvd1_w2t:
        sa.PVD1.set(src="pref0", idx=pvd1_w2t, attr="v", value=init_wind_pref)
        sa.PVD1.set(src="pavail0", idx=pvd1_w2t, attr="v", value=init_wind_pavail)
    if pvd1_pv:
        sa.PVD1.set(src="pref0", idx=pvd1_pv, attr="v", value=init_solar_pref)
        sa.PVD1.set(src="pavail0", idx=pvd1_pv, attr="v", value=init_solar_pavail)

    sa.PFlow.run()
    if sa.exit_code != 0:
        raise RuntimeError(f"PFlow failed with exit_code={sa.exit_code}")

    sa.TDS.init()
    if sa.exit_code != 0:
        raise RuntimeError(f"TDS init failed with exit_code={sa.exit_code}")

    pext_max = 999 * np.ones(sa.DG.n)
    if hasattr(sa, "ESD1") and sa.ESD1.n:
        ess_uid = sa.DG.idx2uid(sa.ESD1.idx.v)
        pext_max[ess_uid] = 999

    ctx: dict[str, object] = {
        "curve": curve,
        "link": link,
        "pq_idx": pq_idx,
        "sap0": sap0,
        "saq0": saq0,
        "stg": stg,
        "stg_w2t": stg_w2t,
        "stg_pv": stg_pv,
        "p0_w2t": p0_w2t,
        "p0_pv": p0_pv,
        "pvd1_w2t": pvd1_w2t,
        "pvd1_pv": pvd1_pv,
        "pext_max": pext_max,
        "wind_pref_alpha": float(wind_pref_alpha),
        "solar_pref_alpha": float(solar_pref_alpha),
    }
    return sa, ctx


def apply_second_dispatch_targets(
    sa: andes.system.System,
    link: pd.DataFrame,
    dispatch_record: rdt.DispatchRecord,
    apply_governor_targets: bool,
    apply_dg_targets: bool,
    duration_seconds: int | None = None,
    schedule_mode: str = "boundary_ramp",
    next_dispatch_record: rdt.DispatchRecord | None = None,
    basepoint_ramp_floor_frac_pmax_per_min: float = 0.005,
    basepoint_ramp_gap_factor: float = 1.25,
) -> dict[str, object]:
    pg_map = rdt.dispatch_pg_map(dispatch_record)
    transition: dict[str, object] = {"ramp_seconds": 0}

    gov_rows = link.dropna(subset=["gov_idx"])
    if apply_governor_targets and not gov_rows.empty:
        gov_idx = gov_rows["gov_idx"].tolist()
        pref_values = np.array([pg_map[int(gen)] for gen in gov_rows["stg_idx"]], dtype=float)
        transition["gov_idx"] = gov_idx
        pref_start = sa.TurbineGov.get(src="pref0", attr="v", idx=gov_idx)
        transition["gov_pref_start"] = pref_start
        transition["gov_pref_target"] = pref_values
        transition["governor_target_schedule"] = schedule_mode

        if schedule_mode == "midpoint_trajectory":
            if duration_seconds is None:
                raise ValueError("duration_seconds is required for midpoint_trajectory")

            if next_dispatch_record is None:
                pref_end = pref_values.copy()
            else:
                next_pg_map = rdt.dispatch_pg_map(next_dispatch_record)
                next_values = np.array([next_pg_map[int(gen)] for gen in gov_rows["stg_idx"]], dtype=float)
                pref_end = 0.5 * (pref_values + next_values)

            n_steps = int(duration_seconds)
            mid = max(1, n_steps // 2)
            last = max(1, n_steps - 1)
            pref_schedule = np.zeros((n_steps, len(gov_idx)), dtype=float)

            for step in range(n_steps):
                if step <= mid:
                    alpha = step / mid
                    pref_schedule[step] = pref_start + alpha * (pref_values - pref_start)
                else:
                    tail = max(1, last - mid)
                    alpha = (step - mid) / tail
                    pref_schedule[step] = pref_values + alpha * (pref_end - pref_values)

            transition["gov_pref_end"] = pref_end
            transition["gov_pref_schedule"] = pref_schedule
        elif schedule_mode == "ramp_limited_basepoint":
            if duration_seconds is None:
                raise ValueError("duration_seconds is required for ramp_limited_basepoint")

            stg_idx = gov_rows["stg_idx"].tolist()
            pmax = np.asarray(sa.StaticGen.get(src="pmax", attr="v", idx=stg_idx), dtype=float)
            dt_s = 1.0
            n_steps = int(duration_seconds)
            gap = np.abs(pref_values - pref_start)
            floor_rate = np.maximum(0.0, float(basepoint_ramp_floor_frac_pmax_per_min)) * pmax / 60.0
            gap_rate = np.maximum(0.0, float(basepoint_ramp_gap_factor)) * gap / max(1.0, float(n_steps - 1))
            ramp_rate = np.maximum(floor_rate, gap_rate)

            pref_schedule = np.zeros((n_steps, len(gov_idx)), dtype=float)
            pref_schedule[0] = pref_start
            current = np.asarray(pref_start, dtype=float).copy()
            for step in range(1, n_steps):
                delta = pref_values - current
                move = np.clip(delta, -ramp_rate * dt_s, ramp_rate * dt_s)
                current = current + move
                pref_schedule[step] = current

            transition["gov_pref_schedule"] = pref_schedule
            transition["gov_ramp_rate_mw_per_s"] = ramp_rate
            transition["gov_pref_end"] = pref_schedule[-1]

    # DG covers PVD1/ESD1 in this case. These units follow their curve / AGC path
    # and are no longer treated as dispatch-target devices at interval boundaries.
    _ = apply_dg_targets

    return transition


def activate_dispatch_target_transition(
    sa: andes.system.System,
    transition: dict[str, object] | None,
    step: int,
) -> None:
    if not transition:
        return

    ramp_seconds = int(transition.get("ramp_seconds", 0))
    if ramp_seconds <= 0:
        alpha = 1.0
    else:
        alpha = min(float(step) / float(ramp_seconds), 1.0)

    gov_idx = transition.get("gov_idx")
    gov_start = transition.get("gov_pref_start")
    gov_target = transition.get("gov_pref_target")
    gov_schedule = transition.get("gov_pref_schedule")
    if gov_idx is not None and gov_schedule is not None:
        schedule = np.asarray(gov_schedule, dtype=float)
        row = schedule[min(int(step), schedule.shape[0] - 1)]
        sa.TurbineGov.set(src="pref0", idx=gov_idx, attr="v", value=row)
    elif gov_idx is not None and gov_start is not None and gov_target is not None:
        gov_value = np.asarray(gov_start) + alpha * (np.asarray(gov_target) - np.asarray(gov_start))
        sa.TurbineGov.set(src="pref0", idx=gov_idx, attr="v", value=gov_value)


def total_governor_output_pe(sa: andes.system.System, gov_idx: list[int] | None = None) -> float:
    if gov_idx is None:
        gov_idx = list(sa.TurbineGov.idx.v) if hasattr(sa, "TurbineGov") and sa.TurbineGov.n else []
    if not gov_idx:
        return 0.0
    gov_syn = sa.TurbineGov.get(src="syn", attr="v", idx=gov_idx)
    pe = np.asarray(sa.SynGen.get(src="Pe", attr="v", idx=gov_syn), dtype=float)
    return float(pe.sum()) if pe.size else 0.0


def total_der_output_pe(sa: andes.system.System) -> float:
    total = 0.0
    for model_name in ("PVD1", "ESD1"):
        if not hasattr(sa, model_name):
            continue
        mdl = getattr(sa, model_name)
        if mdl.n == 0:
            continue
        total += float(np.sum(np.asarray(mdl.Ipout_y.v, dtype=float) * np.asarray(mdl.v.v, dtype=float)))
    return total


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _sum_series(values: np.ndarray) -> float:
    return float(values.sum()) if values.size else 0.0


def _sum_model_signal(model, src: str, idx: list[int] | np.ndarray | None = None) -> float:
    if model is None or model.n == 0 or not hasattr(model, src):
        return 0.0
    if idx is None:
        return _sum_series(np.asarray(getattr(model, src).v, dtype=float))
    idx_list = list(idx)
    if not idx_list:
        return 0.0
    return _sum_series(np.asarray(model.get(src=src, attr="v", idx=idx_list), dtype=float))


def _sum_model_output_pe(model, idx: list[int] | np.ndarray | None = None) -> float:
    if model is None or model.n == 0 or not hasattr(model, "Ipout_y") or not hasattr(model, "v"):
        return 0.0
    if idx is None:
        ipout = np.asarray(model.Ipout_y.v, dtype=float)
        voltage = np.asarray(model.v.v, dtype=float)
    else:
        idx_list = list(idx)
        if not idx_list:
            return 0.0
        ipout = np.asarray(model.get(src="Ipout_y", attr="v", idx=idx_list), dtype=float)
        voltage = np.asarray(model.get(src="v", attr="v", idx=idx_list), dtype=float)
    return _sum_series(ipout * voltage)


def _sum_governor_droop(sa: andes.system.System, gov_idx: list[int]) -> float:
    if not gov_idx:
        return 0.0
    db_y = np.asarray(sa.TurbineGov.get(src="DB_y", attr="v", idx=gov_idx), dtype=float)
    gain = np.asarray(sa.TurbineGov.get(src="gain", attr="v", idx=gov_idx), dtype=float)
    return _sum_series(-db_y * gain)


def _clip_with_headroom_and_ramp(
    desired: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    previous: np.ndarray,
    ramp_rate_mw_per_s: np.ndarray | None,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tol = 1e-12

    after_headroom = np.minimum(np.maximum(desired, lower), upper)
    headroom_mask = np.abs(after_headroom - desired) > tol

    if ramp_rate_mw_per_s is None:
        applied = after_headroom
        ramp_mask = np.zeros_like(headroom_mask, dtype=bool)
    else:
        delta_lim = np.maximum(0.0, np.asarray(ramp_rate_mw_per_s, dtype=float)) * float(dt_s)
        ramp_lower = previous - delta_lim
        ramp_upper = previous + delta_lim
        applied = np.minimum(np.maximum(after_headroom, ramp_lower), ramp_upper)
        ramp_mask = np.abs(applied - after_headroom) > tol

    saturation_dir = np.zeros_like(desired, dtype=int)
    saturation_dir[desired > applied + tol] = 1
    saturation_dir[desired < applied - tol] = -1

    return applied, headroom_mask, ramp_mask, saturation_dir


def apply_agc_dispatch_update(
    *,
    sa: andes.system.System,
    link: pd.DataFrame,
    bf: np.ndarray,
    ace_raw: float,
    pext_max: np.ndarray,
    agc_allocation_mode: str = AGC_ALLOCATION_HEADROOM,
    gov_output_ramp_frac_pmax_per_min: float = 0.0,
    dg_output_ramp_frac_pmax_per_min: float = 0.0,
    agc_anti_windup_mode: str = AGC_ANTI_WINDUP_OFF,
    prev_freeze_active: int = 0,
    prev_freeze_on_streak: int = 0,
    prev_freeze_off_streak: int = 0,
    prev_freeze_dir: int = 0,
    dt_s: float = 1.0,
) -> dict[str, float | int]:
    shares = rdt.compute_agc_allocation_shares(
        sa,
        link,
        bf,
        ace_raw=ace_raw,
        pext_max=pext_max,
        allocation_mode=agc_allocation_mode,
    )
    link["agov"] = ace_raw * shares * np.asarray(link["has_gov"], dtype=float)
    link["adg"] = ace_raw * shares * np.asarray(link["has_dg"], dtype=float)
    link["arg"] = 0.0

    agov_to_set = {gov: agov for gov, agov in zip(link["gov_idx"], link["agov"]) if pd.notna(gov)}
    adg_to_set = {dg: adg for dg, adg in zip(link["dg_idx"], link["adg"]) if pd.notna(dg)}

    last_agov_req_sum = float(sum(agov_to_set.values())) if agov_to_set else 0.0
    last_agov_req_abs_sum = float(sum(abs(val) for val in agov_to_set.values())) if agov_to_set else 0.0
    last_adg_req_sum = float(sum(adg_to_set.values())) if adg_to_set else 0.0
    last_adg_req_abs_sum = float(sum(abs(val) for val in adg_to_set.values())) if adg_to_set else 0.0

    gov_headroom_count = 0
    gov_ramp_count = 0
    dg_headroom_count = 0
    dg_ramp_count = 0
    gov_applied_sum = 0.0
    dg_applied_sum = 0.0

    if agov_to_set:
        gov_idx = list(agov_to_set.keys())
        paux0_raw = np.array(list(agov_to_set.values()), dtype=float)
        gov_syn = sa.TurbineGov.get(src="syn", attr="v", idx=gov_idx)
        gov_gen = sa.SynGen.get(src="gen", attr="v", idx=gov_syn)
        gov_pmax = np.asarray(sa.StaticGen.get(src="pmax", attr="v", idx=gov_gen), dtype=float)
        gov_pmin = np.asarray(sa.StaticGen.get(src="pmin", attr="v", idx=gov_gen), dtype=float)
        gov_pref0 = np.asarray(sa.TurbineGov.get(src="pref0", attr="v", idx=gov_idx), dtype=float)
        gov_prev = np.asarray(sa.TurbineGov.get(src="paux0", attr="v", idx=gov_idx), dtype=float)
        gov_up = np.maximum(0.0, gov_pmax - gov_pref0)
        gov_dn = np.minimum(0.0, gov_pmin - gov_pref0)
        gov_ramp_rate = (
            np.maximum(0.0, float(gov_output_ramp_frac_pmax_per_min)) * gov_pmax / 60.0
            if gov_output_ramp_frac_pmax_per_min > 0.0 else None
        )
        paux0, gov_headroom_mask, gov_ramp_mask, gov_sat_dir = _clip_with_headroom_and_ramp(
            paux0_raw,
            lower=gov_dn,
            upper=gov_up,
            previous=gov_prev,
            ramp_rate_mw_per_s=gov_ramp_rate,
            dt_s=dt_s,
        )
        sa.TurbineGov.set(src="paux0", idx=gov_idx, attr="v", value=paux0)
        gov_headroom_count = int(np.sum(gov_headroom_mask))
        gov_ramp_count = int(np.sum(gov_ramp_mask))
        gov_applied_sum = float(np.sum(paux0))

    if adg_to_set:
        dg_idx = list(adg_to_set.keys())
        pext0_raw = np.array(list(adg_to_set.values()), dtype=float)
        dg_prev = np.asarray(sa.DG.get(src="Pext0", attr="v", idx=dg_idx), dtype=float)
        dg_upper = np.asarray(pext_max[sa.DG.idx2uid(dg_idx)], dtype=float)
        dg_lower = np.full_like(dg_upper, -np.inf, dtype=float)
        dg_ramp_rate = None
        if dg_output_ramp_frac_pmax_per_min > 0.0:
            dg_gen = sa.DG.get(src="gen", attr="v", idx=dg_idx)
            dg_pmax = np.asarray(sa.StaticGen.get(src="pmax", attr="v", idx=dg_gen), dtype=float)
            dg_ramp_rate = np.maximum(0.0, float(dg_output_ramp_frac_pmax_per_min)) * dg_pmax / 60.0
        pext0, dg_headroom_mask, dg_ramp_mask, dg_sat_dir = _clip_with_headroom_and_ramp(
            pext0_raw,
            lower=dg_lower,
            upper=dg_upper,
            previous=dg_prev,
            ramp_rate_mw_per_s=dg_ramp_rate,
            dt_s=dt_s,
        )
        sa.DG.set(src="Pext0", idx=dg_idx, attr="v", value=pext0)
        dg_headroom_count = int(np.sum(dg_headroom_mask))
        dg_ramp_count = int(np.sum(dg_ramp_mask))
        dg_applied_sum = float(np.sum(pext0))

    saturation_active = int((gov_headroom_count + gov_ramp_count + dg_headroom_count + dg_ramp_count) > 0)
    agc_request_sum_total = float(last_agov_req_sum + last_adg_req_sum)
    agc_applied_sum_total = float(gov_applied_sum + dg_applied_sum)
    agc_clip_deficit_sum_total = float(agc_request_sum_total - agc_applied_sum_total)
    agc_saturation_ratio = float(
        abs(agc_clip_deficit_sum_total) / max(abs(agc_request_sum_total), AGC_RATIO_EPS)
    )
    same_dir_total = (
        abs(agc_request_sum_total) > AGC_RATIO_EPS
        and np.sign(agc_clip_deficit_sum_total) == np.sign(agc_request_sum_total)
    )
    request_dir = int(np.sign(agc_request_sum_total)) if abs(agc_request_sum_total) > AGC_RATIO_EPS else 0
    freeze_condition_active = int(
        agc_saturation_ratio >= AGC_FREEZE_RATIO_THRESHOLD and same_dir_total
    )
    release_condition_active = int(
        agc_saturation_ratio <= AGC_UNFREEZE_RATIO_THRESHOLD or not same_dir_total
    )
    direction_flip_release = int(
        bool(prev_freeze_active)
        and prev_freeze_dir != 0
        and request_dir != 0
        and request_dir != prev_freeze_dir
    )

    freeze_active = int(prev_freeze_active)
    freeze_on_streak = int(prev_freeze_on_streak)
    freeze_off_streak = int(prev_freeze_off_streak)
    freeze_dir = int(prev_freeze_dir)

    if freeze_active:
        if direction_flip_release:
            freeze_active = 0
            freeze_on_streak = 0
            freeze_off_streak = 0
            freeze_dir = 0
        else:
            freeze_off_streak = int(prev_freeze_off_streak + 1) if release_condition_active else 0
            if freeze_off_streak >= AGC_UNFREEZE_REQUIRED_STREAK:
                freeze_active = 0
                freeze_on_streak = 0
                freeze_off_streak = 0
                freeze_dir = 0
            else:
                freeze_active = 1
                freeze_on_streak = max(int(prev_freeze_on_streak), AGC_FREEZE_REQUIRED_STREAK)
                freeze_dir = int(prev_freeze_dir if prev_freeze_dir != 0 else request_dir)
    else:
        freeze_on_streak = int(prev_freeze_on_streak + 1) if freeze_condition_active else 0
        freeze_off_streak = 0
        if freeze_on_streak >= AGC_FREEZE_REQUIRED_STREAK:
            freeze_active = 1
            freeze_dir = request_dir
        else:
            freeze_active = 0
            freeze_dir = 0

    integrator_freeze = int(
        agc_anti_windup_mode == AGC_ANTI_WINDUP_FREEZE and freeze_active
    )

    return {
        "agov_request_sum": last_agov_req_sum,
        "agov_request_abs_sum": last_agov_req_abs_sum,
        "adg_request_sum": last_adg_req_sum,
        "adg_request_abs_sum": last_adg_req_abs_sum,
        "gov_applied_sum": gov_applied_sum,
        "dg_applied_sum": dg_applied_sum,
        "gov_headroom_saturated_count": gov_headroom_count,
        "gov_ramp_saturated_count": gov_ramp_count,
        "dg_headroom_saturated_count": dg_headroom_count,
        "dg_ramp_saturated_count": dg_ramp_count,
        "saturation_active": saturation_active,
        "freeze_condition_active": freeze_condition_active,
        "release_condition_active": release_condition_active,
        "direction_flip_release": direction_flip_release,
        "integrator_freeze_active": integrator_freeze,
        "agc_request_sum_total": agc_request_sum_total,
        "agc_applied_sum_total": agc_applied_sum_total,
        "agc_clip_deficit_sum_total": agc_clip_deficit_sum_total,
        "agc_saturation_ratio": agc_saturation_ratio,
        "agc_freeze_streak": freeze_on_streak,
        "agc_unfreeze_streak": freeze_off_streak,
        "agc_freeze_active": freeze_active,
        "agc_freeze_dir": freeze_dir,
    }


def run_segment(
    sa: andes.system.System,
    ctx: dict[str, object],
    start_offset: int,
    duration_seconds: int,
    agc_interval: int,
    kp: float,
    ki: float,
    bf: np.ndarray,
    agc_allocation_mode: str = AGC_ALLOCATION_HEADROOM,
    ace_integral: float = 0.0,
    ace_raw: float = 0.0,
    local_start: float = 0.0,
    include_initial: bool = True,
    dispatch_target_transition: dict[str, object] | None = None,
    gov_output_ramp_frac_pmax_per_min: float = 0.0,
    dg_output_ramp_frac_pmax_per_min: float = 0.0,
    agc_anti_windup_mode: str = AGC_ANTI_WINDUP_OFF,
    agc_aw_state: dict[str, int] | None = None,
    metrics_out: dict[str, float | int] | None = None,
    trace_out: dict[str, np.ndarray] | None = None,
    wind_pref_alpha: float = 1.0,
    solar_pref_alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
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
    wind_pref_alpha = rdt.validate_pref_alpha(wind_pref_alpha, name="wind_pref_alpha")
    solar_pref_alpha = rdt.validate_pref_alpha(solar_pref_alpha, name="solar_pref_alpha")

    local_t = []
    freq = []
    gov_paux0_sum_hist: list[float] = []
    dg_pext0_sum_hist: list[float] = []
    wind_pe_sum_hist: list[float] = []
    wind_pref_sum_hist: list[float] = []
    pv_pe_sum_hist: list[float] = []
    pv_pref_sum_hist: list[float] = []
    esd_pe_sum_hist: list[float] = []
    esd_pref_sum_hist: list[float] = []
    gov_droop_sum_hist: list[float] = []
    saturation_hist: list[int] = []
    freeze_hist: list[int] = []
    agc_request_sum_total_hist: list[float] = []
    agc_applied_sum_total_hist: list[float] = []
    agc_clip_deficit_sum_total_hist: list[float] = []
    agc_saturation_ratio_hist: list[float] = []
    agc_freeze_streak_hist: list[int] = []
    gov_all_idx = [idx for idx in link["gov_idx"].tolist() if pd.notna(idx)]
    dg_all_idx = [idx for idx in link["dg_idx"].tolist() if pd.notna(idx)]
    cycle_saturation_active = 0
    cycle_integrator_freeze_active = 0
    cycle_agc_request_sum_total = 0.0
    cycle_agc_applied_sum_total = 0.0
    cycle_agc_clip_deficit_sum_total = 0.0
    cycle_agc_saturation_ratio = 0.0
    cycle_agc_freeze_streak = 0
    cycle_agc_unfreeze_streak = 0
    agc_state = normalize_agc_aw_state(agc_aw_state)
    cycle_integrator_freeze_active = int(agc_state["freeze_active"])
    cycle_agc_freeze_streak = int(agc_state["freeze_on_streak"])
    cycle_agc_unfreeze_streak = int(agc_state["freeze_off_streak"])
    pvd_model = getattr(sa, "PVD1", None)
    esd_model = getattr(sa, "ESD1", None)
    if include_initial:
        local_t.append(float(local_start))
        freq.append(float((sa.ACEc.f.v[0] - 1.0) * sa.config.freq))
        gov_paux0 = (
            np.asarray(sa.TurbineGov.get(src="paux0", attr="v", idx=gov_all_idx), dtype=float)
            if gov_all_idx else np.asarray([])
        )
        dg_pext0 = (
            np.asarray(sa.DG.get(src="Pext0", attr="v", idx=dg_all_idx), dtype=float)
            if dg_all_idx else np.asarray([])
        )
        gov_paux0_sum_hist.append(float(gov_paux0.sum()) if gov_paux0.size else 0.0)
        dg_pext0_sum_hist.append(float(dg_pext0.sum()) if dg_pext0.size else 0.0)
        wind_pe_sum_hist.append(_sum_model_output_pe(pvd_model, pvd1_w2t))
        wind_pref_sum_hist.append(_sum_model_signal(pvd_model, "pref0", pvd1_w2t))
        pv_pe_sum_hist.append(_sum_model_output_pe(pvd_model, pvd1_pv))
        pv_pref_sum_hist.append(_sum_model_signal(pvd_model, "pref0", pvd1_pv))
        esd_pe_sum_hist.append(_sum_model_output_pe(esd_model))
        esd_pref_sum_hist.append(_sum_model_signal(esd_model, "pref0"))
        gov_droop_sum_hist.append(_sum_governor_droop(sa, gov_all_idx))
        saturation_hist.append(cycle_saturation_active)
        freeze_hist.append(cycle_integrator_freeze_active)
        agc_request_sum_total_hist.append(cycle_agc_request_sum_total)
        agc_applied_sum_total_hist.append(cycle_agc_applied_sum_total)
        agc_clip_deficit_sum_total_hist.append(cycle_agc_clip_deficit_sum_total)
        agc_saturation_ratio_hist.append(cycle_agc_saturation_ratio)
        agc_freeze_streak_hist.append(cycle_agc_freeze_streak)

    current_tf = float(sa.dae.t)
    for step in range(1, duration_seconds):
        activate_dispatch_target_transition(sa, dispatch_target_transition, step)

        if step % agc_interval == 0:
            agc_update_meta = apply_agc_dispatch_update(
                sa=sa,
                link=link,
                bf=bf,
                ace_raw=ace_raw,
                pext_max=pext_max,
                agc_allocation_mode=agc_allocation_mode,
                gov_output_ramp_frac_pmax_per_min=gov_output_ramp_frac_pmax_per_min,
                dg_output_ramp_frac_pmax_per_min=dg_output_ramp_frac_pmax_per_min,
                agc_anti_windup_mode=agc_anti_windup_mode,
                prev_freeze_active=agc_state["freeze_active"],
                prev_freeze_on_streak=agc_state["freeze_on_streak"],
                prev_freeze_off_streak=agc_state["freeze_off_streak"],
                prev_freeze_dir=agc_state["freeze_dir"],
            )
            cycle_saturation_active = int(agc_update_meta["saturation_active"])
            cycle_integrator_freeze_active = int(agc_update_meta["integrator_freeze_active"])
            cycle_agc_request_sum_total = float(agc_update_meta["agc_request_sum_total"])
            cycle_agc_applied_sum_total = float(agc_update_meta["agc_applied_sum_total"])
            cycle_agc_clip_deficit_sum_total = float(agc_update_meta["agc_clip_deficit_sum_total"])
            cycle_agc_saturation_ratio = float(agc_update_meta["agc_saturation_ratio"])
            cycle_agc_freeze_streak = int(agc_update_meta["agc_freeze_streak"])
            cycle_agc_unfreeze_streak = int(agc_update_meta["agc_unfreeze_streak"])
            agc_state["freeze_active"] = int(agc_update_meta["agc_freeze_active"])
            agc_state["freeze_on_streak"] = int(agc_update_meta["agc_freeze_streak"])
            agc_state["freeze_off_streak"] = int(agc_update_meta["agc_unfreeze_streak"])
            agc_state["freeze_dir"] = int(agc_update_meta["agc_freeze_dir"])

        kload = curve["Load"].iloc[start_offset + step]
        sa.PQ.set(src="Ppf", idx=sa.PQ.idx.v, attr="v", value=kload * sap0)
        sa.PQ.set(src="Qpf", idx=sa.PQ.idx.v, attr="v", value=kload * saq0)

        wind = curve["Wind"].iloc[start_offset + step]
        wind_pavail = rdt.der_available_from_curve(wind, p0_w2t)
        wind_pref = rdt.der_pref_from_available(wind_pavail, wind_pref_alpha)
        sa.PVD1.set(src="pref0", idx=pvd1_w2t, attr="v", value=wind_pref)
        sa.PVD1.set(src="pavail0", idx=pvd1_w2t, attr="v", value=wind_pavail)

        solar = curve["PV"].iloc[start_offset + step]
        solar_pavail = rdt.der_available_from_curve(solar, p0_pv)
        solar_pref = rdt.der_pref_from_available(solar_pavail, solar_pref_alpha)
        sa.PVD1.set(src="pref0", idx=pvd1_pv, attr="v", value=solar_pref)
        sa.PVD1.set(src="pavail0", idx=pvd1_pv, attr="v", value=solar_pavail)

        current_tf += 1.0
        sa.TDS.config.tf = current_tf
        sa.TDS.run()
        if sa.exit_code != 0:
            raise RuntimeError(f"TDS failed at local step={step} with exit_code={sa.exit_code}")

        local_t.append(float(local_start + step))
        freq.append(float((sa.ACEc.f.v[0] - 1.0) * sa.config.freq))
        gov_paux0 = (
            np.asarray(sa.TurbineGov.get(src="paux0", attr="v", idx=gov_all_idx), dtype=float)
            if gov_all_idx else np.asarray([])
        )
        dg_pext0 = (
            np.asarray(sa.DG.get(src="Pext0", attr="v", idx=dg_all_idx), dtype=float)
            if dg_all_idx else np.asarray([])
        )
        gov_paux0_sum_hist.append(float(gov_paux0.sum()) if gov_paux0.size else 0.0)
        dg_pext0_sum_hist.append(float(dg_pext0.sum()) if dg_pext0.size else 0.0)
        wind_pe_sum_hist.append(_sum_model_output_pe(pvd_model, pvd1_w2t))
        wind_pref_sum_hist.append(_sum_model_signal(pvd_model, "pref0", pvd1_w2t))
        pv_pe_sum_hist.append(_sum_model_output_pe(pvd_model, pvd1_pv))
        pv_pref_sum_hist.append(_sum_model_signal(pvd_model, "pref0", pvd1_pv))
        esd_pe_sum_hist.append(_sum_model_output_pe(esd_model))
        esd_pref_sum_hist.append(_sum_model_signal(esd_model, "pref0"))
        gov_droop_sum_hist.append(_sum_governor_droop(sa, gov_all_idx))
        saturation_hist.append(cycle_saturation_active)
        freeze_hist.append(cycle_integrator_freeze_active)
        agc_request_sum_total_hist.append(cycle_agc_request_sum_total)
        agc_applied_sum_total_hist.append(cycle_agc_applied_sum_total)
        agc_clip_deficit_sum_total_hist.append(cycle_agc_clip_deficit_sum_total)
        agc_saturation_ratio_hist.append(cycle_agc_saturation_ratio)
        agc_freeze_streak_hist.append(cycle_agc_freeze_streak)

        ace_sum = sa.ACEc.ace.v.sum()
        ace_raw = -(kp * ace_sum + ki * ace_integral)
        if cycle_integrator_freeze_active:
            ace_integral, ace_raw = backcalculate_ace_integral_partial(
                ace_integral=float(ace_integral),
                kp=kp,
                ki=ki,
                ace_sum=float(ace_sum),
                ace_raw=float(ace_raw),
                agc_request_sum_total=cycle_agc_request_sum_total,
                agc_applied_sum_total=cycle_agc_applied_sum_total,
            )
        else:
            ace_integral = ace_integral + ace_sum

    if metrics_out is not None:
        freq_arr = np.asarray(freq, dtype=float)
        freq_d1 = np.abs(np.diff(freq_arr))
        time_arr = np.asarray(local_t, dtype=float)
        wind_pe_arr = np.asarray(wind_pe_sum_hist, dtype=float)
        wind_pref_arr = np.asarray(wind_pref_sum_hist, dtype=float)
        pv_pe_arr = np.asarray(pv_pe_sum_hist, dtype=float)
        pv_pref_arr = np.asarray(pv_pref_sum_hist, dtype=float)
        esd_pe_arr = np.asarray(esd_pe_sum_hist, dtype=float)
        esd_pref_arr = np.asarray(esd_pref_sum_hist, dtype=float)
        gov_droop_arr = np.asarray(gov_droop_sum_hist, dtype=float)
        wind_effort = _trapz(np.abs(wind_pe_arr - wind_pref_arr), time_arr) if time_arr.size else 0.0
        pv_effort = _trapz(np.abs(pv_pe_arr - pv_pref_arr), time_arr) if time_arr.size else 0.0
        esd_throughput = _trapz(np.abs(esd_pe_arr - esd_pref_arr), time_arr) if time_arr.size else 0.0
        gov_droop_effort = _trapz(np.abs(gov_droop_arr), time_arr) if time_arr.size else 0.0
        metrics_out.update(
            {
                "wind_effort": float(wind_effort),
                "pv_effort": float(pv_effort),
                "pvd_effort": float(wind_effort + pv_effort),
                "esd_throughput": float(esd_throughput),
                "gov_droop_effort": float(gov_droop_effort),
                "gov_paux0_tv": float(np.sum(np.abs(np.diff(np.asarray(gov_paux0_sum_hist, dtype=float))))),
                "dg_pext0_tv": float(np.sum(np.abs(np.diff(np.asarray(dg_pext0_sum_hist, dtype=float))))),
                "freq_d1_abs_mean": float(freq_d1.mean()) if freq_d1.size else 0.0,
                "freq_d1_abs_p95": float(np.quantile(freq_d1, 0.95)) if freq_d1.size else 0.0,
                "saturation_fraction": float(np.mean(np.asarray(saturation_hist, dtype=float))) if saturation_hist else 0.0,
                "integrator_freeze_fraction": float(np.mean(np.asarray(freeze_hist, dtype=float))) if freeze_hist else 0.0,
                "saturation_step_count": int(np.sum(np.asarray(saturation_hist, dtype=int))) if saturation_hist else 0,
                "integrator_freeze_step_count": int(np.sum(np.asarray(freeze_hist, dtype=int))) if freeze_hist else 0,
                "agc_request_sum_total_abs_mean": float(np.mean(np.abs(np.asarray(agc_request_sum_total_hist, dtype=float)))),
                "agc_request_sum_total_abs_max": float(np.max(np.abs(np.asarray(agc_request_sum_total_hist, dtype=float)))),
                "agc_applied_sum_total_abs_mean": float(np.mean(np.abs(np.asarray(agc_applied_sum_total_hist, dtype=float)))),
                "agc_applied_sum_total_abs_max": float(np.max(np.abs(np.asarray(agc_applied_sum_total_hist, dtype=float)))),
                "agc_clip_deficit_sum_total_abs_mean": float(np.mean(np.abs(np.asarray(agc_clip_deficit_sum_total_hist, dtype=float)))),
                "agc_clip_deficit_sum_total_abs_max": float(np.max(np.abs(np.asarray(agc_clip_deficit_sum_total_hist, dtype=float)))),
                "agc_saturation_ratio_mean": float(np.mean(np.asarray(agc_saturation_ratio_hist, dtype=float))),
                "agc_saturation_ratio_max": float(np.max(np.asarray(agc_saturation_ratio_hist, dtype=float))),
                "agc_freeze_streak_max": int(np.max(np.asarray(agc_freeze_streak_hist, dtype=int))),
                "agc_freeze_streak_end": int(agc_freeze_streak_hist[-1]) if agc_freeze_streak_hist else 0,
                "agc_unfreeze_streak_end": int(cycle_agc_unfreeze_streak),
                "agc_freeze_active_end": int(agc_state["freeze_active"]),
                "agc_freeze_dir_end": int(agc_state["freeze_dir"]),
            }
        )
    if trace_out is not None:
        trace_out.update(
            {
                "time_s": np.asarray(local_t, dtype=float),
                "freq_dev_hz": np.asarray(freq, dtype=float),
                "gov_paux0_sum": np.asarray(gov_paux0_sum_hist, dtype=float),
                "dg_pext0_sum": np.asarray(dg_pext0_sum_hist, dtype=float),
                "wind_pe_sum": np.asarray(wind_pe_sum_hist, dtype=float),
                "wind_pref_sum": np.asarray(wind_pref_sum_hist, dtype=float),
                "pv_pe_sum": np.asarray(pv_pe_sum_hist, dtype=float),
                "pv_pref_sum": np.asarray(pv_pref_sum_hist, dtype=float),
                "esd_pe_sum": np.asarray(esd_pe_sum_hist, dtype=float),
                "esd_pref_sum": np.asarray(esd_pref_sum_hist, dtype=float),
                "gov_droop_sum": np.asarray(gov_droop_sum_hist, dtype=float),
                "saturation_active": np.asarray(saturation_hist, dtype=int),
                "integrator_freeze_active": np.asarray(freeze_hist, dtype=int),
                "agc_request_sum_total": np.asarray(agc_request_sum_total_hist, dtype=float),
                "agc_applied_sum_total": np.asarray(agc_applied_sum_total_hist, dtype=float),
                "agc_clip_deficit_sum_total": np.asarray(agc_clip_deficit_sum_total_hist, dtype=float),
                "agc_saturation_ratio": np.asarray(agc_saturation_ratio_hist, dtype=float),
                "agc_freeze_streak": np.asarray(agc_freeze_streak_hist, dtype=int),
            }
        )
    if agc_aw_state is not None:
        agc_aw_state.update(agc_state)
    return np.asarray(local_t), np.asarray(freq), float(ace_integral), float(ace_raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-dispatch-json", type=Path, required=True)
    parser.add_argument("--second-dispatch-json", type=Path, required=True)
    parser.add_argument("--first-cold-csv", type=Path, required=True)
    parser.add_argument("--second-cold-csv", type=Path, required=True)
    parser.add_argument("--kp", type=float, default=0.03)
    parser.add_argument("--ki", type=float, default=0.01)
    parser.add_argument("--agc-interval", type=int, default=4)
    parser.add_argument(
        "--agc-allocation-mode",
        choices=AGC_ALLOCATION_MODES,
        default=AGC_ALLOCATION_HEADROOM,
    )
    parser.add_argument("--agc-gov-output-ramp-frac-pmax-per-min", type=float, default=0.0)
    parser.add_argument("--agc-dg-output-ramp-frac-pmax-per-min", type=float, default=0.0)
    parser.add_argument("--disable-der-agc", action="store_true")
    parser.add_argument("--disable-pvd-agc", action="store_true")
    parser.add_argument("--disable-esd-agc", action="store_true")
    parser.add_argument(
        "--agc-anti-windup-mode",
        choices=(AGC_ANTI_WINDUP_OFF, AGC_ANTI_WINDUP_FREEZE),
        default=AGC_ANTI_WINDUP_OFF,
    )
    parser.add_argument("--dispatch-interval", type=int, default=900)
    parser.add_argument("--init-mode", choices=("dispatch", "first"), default="first")
    parser.add_argument("--resume-mode", choices=("memory", "snapshot"), default="memory")
    parser.add_argument("--apply-second-governor-targets", action="store_true")
    parser.add_argument(
        "--apply-second-dg-targets",
        action="store_true",
        help="Deprecated and ignored. DG/PVD1/ESD1 dispatch targets are not applied.",
    )
    parser.add_argument("--dispatch-target-ramp-seconds", type=int, default=0)
    parser.add_argument("--dyn-case", type=Path, default=rdt.DEFAULT_DYN_CASE)
    parser.add_argument("--stable-dyn-case", type=Path, default=rdt.DEFAULT_STABLE_DYN_CASE)
    parser.add_argument("--curve-file", type=Path, default=rdt.DEFAULT_CURVE_FILE)
    parser.add_argument("--results-dir", type=Path, default=rdt.RESULTS / "hotstart_compare")
    parser.add_argument("--label", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rdt.andes.config_logger(stream_level=30)

    first = rdt.DispatchRecord.from_json(args.first_dispatch_json)
    second = rdt.DispatchRecord.from_json(args.second_dispatch_json)
    label = args.label or f"{first.label}_{second.label}_hotstart"
    args.results_dir.mkdir(parents=True, exist_ok=True)

    curve = rdt.load_curve(args.curve_file)
    for record in (first, second):
        rdt.validate_curve_window(curve, record, args.dispatch_interval)

    dyn_case = rdt.adapt_dyn_case(args.dyn_case, args.stable_dyn_case)
    wind_prefixes = rdt.DEFAULT_WIND_PREFIXES
    solar_prefixes = rdt.DEFAULT_SOLAR_PREFIXES

    # First dispatch: regular cold start for hXdY.
    sa1, ctx1 = prepare_system(
        dispatch_record=first,
        curve=curve,
        dyn_case=dyn_case,
        dispatch_interval=args.dispatch_interval,
        init_mode=args.init_mode,
        wind_prefixes=wind_prefixes,
        solar_prefixes=solar_prefixes,
    )
    ctx1["link"] = rdt.configure_der_agc_participation(
        sa1,
        ctx1["link"],  # type: ignore[arg-type]
        enable_der_agc=not args.disable_der_agc,
        enable_pvd_agc=not args.disable_pvd_agc,
        enable_esd_agc=not args.disable_esd_agc,
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
        agc_allocation_mode=args.agc_allocation_mode,
        ace_integral=0.0,
        ace_raw=0.0,
        local_start=0.0,
        include_initial=True,
        gov_output_ramp_frac_pmax_per_min=args.agc_gov_output_ramp_frac_pmax_per_min,
        dg_output_ramp_frac_pmax_per_min=args.agc_dg_output_ramp_frac_pmax_per_min,
        agc_anti_windup_mode=args.agc_anti_windup_mode,
    )

    snapshot_path = args.results_dir / f"{label}_snapshot.pkl"
    if args.resume_mode == "snapshot":
        # Save the dynamic terminal state and external AGC state.
        sa1._deadband_hotstart_meta = {  # type: ignore[attr-defined]
            "ace_integral": ace_integral_end,
            "ace_raw": ace_raw_end,
        }
        save_ss(snapshot_path, sa1)
        sa2 = load_ss(snapshot_path)
        hcp.rehydrate_loaded_snapshot(sa2)
        hot_meta = getattr(sa2, "_deadband_hotstart_meta", {})
        ace_integral_hot = float(hot_meta.get("ace_integral", 0.0))
        ace_raw_hot = float(hot_meta.get("ace_raw", 0.0))
    else:
        sa2 = sa1
        ace_integral_hot = ace_integral_end
        ace_raw_hot = ace_raw_end

    ctx2 = ctx1.copy()
    ctx2["link"] = rdt.configure_der_agc_participation(
        sa2,
        rdt.build_andes_link(sa2),
        enable_der_agc=not args.disable_der_agc,
        enable_pvd_agc=not args.disable_pvd_agc,
        enable_esd_agc=not args.disable_esd_agc,
    )
    bf2 = compute_bf(sa2, second)
    transition = apply_second_dispatch_targets(
        sa2,
        ctx2["link"],  # type: ignore[arg-type]
        second,
        apply_governor_targets=args.apply_second_governor_targets,
        apply_dg_targets=args.apply_second_dg_targets,
    )
    transition["ramp_seconds"] = int(args.dispatch_target_ramp_seconds)
    if int(args.dispatch_target_ramp_seconds) <= 0:
        activate_dispatch_target_transition(sa2, transition, step=0)

    t2_hot, f2_hot, _, _ = run_segment(
        sa=sa2,
        ctx=ctx2,
        start_offset=dispatch_offset(second, args.dispatch_interval),
        duration_seconds=args.dispatch_interval,
        agc_interval=args.agc_interval,
        kp=args.kp,
        ki=args.ki,
        bf=bf2,
        agc_allocation_mode=args.agc_allocation_mode,
        ace_integral=ace_integral_hot,
        ace_raw=ace_raw_hot,
        local_start=float(args.dispatch_interval),
        include_initial=True,
        dispatch_target_transition=transition,
        gov_output_ramp_frac_pmax_per_min=args.agc_gov_output_ramp_frac_pmax_per_min,
        dg_output_ramp_frac_pmax_per_min=args.agc_dg_output_ramp_frac_pmax_per_min,
        agc_anti_windup_mode=args.agc_anti_windup_mode,
    )

    hot_df = pd.DataFrame({"time_s": np.concatenate([t1, t2_hot]), "freq_dev_hz": np.concatenate([f1, f2_hot])})
    hot_csv = args.results_dir / f"{label}_hotstart_frequency.csv"
    hot_df.to_csv(hot_csv, index=False)

    # Cold stitched traces from the existing per-dispatch runs for comparison.
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

    jump_cold = float(cold2["freq_dev_hz"].iloc[0] - cold1["freq_dev_hz"].iloc[-1])
    jump_hot = float(f2_hot[0] - f1[-1])
    step_hot = float(f2_hot[1] - f2_hot[0]) if len(f2_hot) > 1 else float("nan")

    summary = pd.DataFrame([{
        "label": label,
        "cold_end_first_hz": float(cold1["freq_dev_hz"].iloc[-1]),
        "cold_start_second_hz": float(cold2["freq_dev_hz"].iloc[0]),
        "cold_jump_hz": jump_cold,
        "hot_end_first_hz": float(f1[-1]),
        "hot_start_second_hz": float(f2_hot[0]),
        "hot_jump_hz": jump_hot,
        "hot_step_0_to_1_hz": step_hot,
        "hot_min_hz": float(hot_df["freq_dev_hz"].min()),
        "hot_max_hz": float(hot_df["freq_dev_hz"].max()),
    }])
    summary_csv = args.results_dir / f"{label}_hotstart_summary.csv"
    summary.to_csv(summary_csv, index=False)

    fig, axes = plt.subplots(2, 1, figsize=(15.5, 10.2), sharex=False)
    axes[0].plot(cold_x, cold_y, color="#b24c2a", linewidth=1.25, label="cold stitched")
    axes[0].plot(hot_df["time_s"], hot_df["freq_dev_hz"], color="#0f5c78", linewidth=1.4, label="hot-start second dispatch")
    axes[0].axvline(args.dispatch_interval, color="#666666", linestyle="--", linewidth=0.9)
    axes[0].axhline(0.0, color="#999999", linestyle=":", linewidth=0.8)
    axes[0].set_title(f"{first.label} -> {second.label}: cold stitched vs hot-start second dispatch")
    axes[0].set_ylabel("Frequency deviation [Hz]")
    axes[0].grid(True, alpha=0.22)
    axes[0].legend(loc="upper right")

    axes[1].plot(cold_x, cold_y, color="#b24c2a", linewidth=1.35, label="cold stitched")
    axes[1].plot(hot_df["time_s"], hot_df["freq_dev_hz"], color="#0f5c78", linewidth=1.45, label="hot-start")
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
        f"cold jump = {jump_cold:+.4f} Hz\nhot jump = {jump_hot:+.4f} Hz\nhot first step = {step_hot:+.4f} Hz",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )
    fig.tight_layout()
    plot_path = args.results_dir / f"{label}_hotstart_vs_cold.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    manifest = {
        "first_dispatch_json": str(args.first_dispatch_json),
        "second_dispatch_json": str(args.second_dispatch_json),
        "kp": args.kp,
        "ki": args.ki,
        "agc_interval": args.agc_interval,
        "agc_gov_output_ramp_frac_pmax_per_min": args.agc_gov_output_ramp_frac_pmax_per_min,
        "agc_dg_output_ramp_frac_pmax_per_min": args.agc_dg_output_ramp_frac_pmax_per_min,
        "agc_anti_windup_mode": args.agc_anti_windup_mode,
        "disable_der_agc": bool(args.disable_der_agc),
        "disable_pvd_agc": bool(args.disable_pvd_agc),
        "disable_esd_agc": bool(args.disable_esd_agc),
        "dispatch_interval": args.dispatch_interval,
        "init_mode": args.init_mode,
        "resume_mode": args.resume_mode,
        "apply_second_governor_targets": args.apply_second_governor_targets,
        "apply_second_dg_targets": args.apply_second_dg_targets,
        "snapshot_path": str(snapshot_path if args.resume_mode == "snapshot" else ""),
        "hot_csv": str(hot_csv),
        "summary_csv": str(summary_csv),
        "plot_path": str(plot_path),
    }
    (args.results_dir / f"{label}_hotstart_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"hot_csv={hot_csv}")
    print(f"summary_csv={summary_csv}")
    print(f"plot={plot_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
