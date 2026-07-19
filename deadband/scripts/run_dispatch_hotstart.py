#!/usr/bin/env python3
"""
Run one dispatch interval, optionally resuming from a disk checkpoint.

This is the atomic building block for parameter-specific checkpoint chains:

- cold start: no checkpoint, run one segment, save its terminal checkpoint
- hot start: load previous terminal checkpoint, apply the new dispatch targets,
  run one segment, save the new terminal checkpoint
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import run_dispatch_tds as rdt
from compare_dispatch_pair_hotstart import (
    AGC_ALLOCATION_HEADROOM,
    AGC_ANTI_WINDUP_FREEZE,
    AGC_ANTI_WINDUP_OFF,
    activate_dispatch_target_transition,
    apply_second_dispatch_targets,
    compute_bf,
    dispatch_offset,
    initial_agc_aw_state,
    prepare_system,
    run_segment,
)
import hotstart_checkpoint as hcp


COMPLEX_WARNING = getattr(getattr(np, "exceptions", object()), "ComplexWarning", None)


def summarize_series(t: np.ndarray, f_dev_hz: np.ndarray) -> dict[str, float | int]:
    imin = int(np.argmin(f_dev_hz))
    imax = int(np.argmax(f_dev_hz))
    abs_f = np.abs(f_dev_hz)
    return {
        "samples": int(len(t)),
        "t_end_s": float(t[-1]),
        "min_hz": float(f_dev_hz[imin]),
        "t_min_s": float(t[imin]),
        "max_hz": float(f_dev_hz[imax]),
        "t_max_s": float(t[imax]),
        "final_hz": float(f_dev_hz[-1]),
        "abs_mean_hz": float(np.mean(np.abs(f_dev_hz))),
        "rms_hz": float(np.sqrt(np.mean(np.square(f_dev_hz)))),
        "max_abs_hz": float(abs_f.max()),
        "final_abs_hz": float(abs_f[-1]),
        "share_abs_gt_0p036": float(np.mean(abs_f > 0.036)),
        "share_abs_gt_0p05": float(np.mean(abs_f > 0.05)),
        "zero_crossings": int(np.sum(np.signbit(f_dev_hz[1:]) != np.signbit(f_dev_hz[:-1]))),
    }


def storage_indices(sa) -> tuple[list[int], list[str]]:
    esd_idx = list(sa.ESD1.idx.v) if hasattr(sa, "ESD1") and sa.ESD1.n else []
    esd_gen = list(map(int, sa.ESD1.gen.v)) if hasattr(sa, "ESD1") and sa.ESD1.n else []
    return esd_gen, esd_idx


def storage_share(sa) -> dict[str, float]:
    esd_gen, _ = storage_indices(sa)
    stg = list(sa.StaticGen.get_all_idxes())
    pmax_all = np.asarray(sa.StaticGen.get(src="pmax", attr="v", idx=stg), dtype=float)
    sn_all = np.asarray(sa.StaticGen.get(src="Sn", attr="v", idx=stg), dtype=float)
    if not esd_gen:
        return {
            "pmax_sum": 0.0,
            "pmax_total": float(pmax_all.sum()),
            "pmax_share": 0.0,
            "sn_sum": 0.0,
            "sn_total": float(sn_all.sum()),
            "sn_share": 0.0,
        }

    pmax_esd = np.asarray(sa.StaticGen.get(src="pmax", attr="v", idx=esd_gen), dtype=float)
    sn_esd = np.asarray(sa.StaticGen.get(src="Sn", attr="v", idx=esd_gen), dtype=float)
    return {
        "pmax_sum": float(pmax_esd.sum()),
        "pmax_total": float(pmax_all.sum()),
        "pmax_share": float(pmax_esd.sum() / pmax_all.sum()),
        "sn_sum": float(sn_esd.sum()),
        "sn_total": float(sn_all.sum()),
        "sn_share": float(sn_esd.sum() / sn_all.sum()),
    }


def solve_scale_factor(current_share: float, target_share: float) -> float:
    if target_share <= 0.0:
        raise ValueError("target_share must be positive")
    if target_share >= 1.0:
        raise ValueError("target_share must be < 1")
    if current_share <= 0.0:
        raise ValueError("current_share must be positive")
    return float(target_share * (1.0 - current_share) / (current_share * (1.0 - target_share)))


def scale_storage_capacity(sa, factor: float) -> dict[str, float]:
    esd_gen, esd_idx = storage_indices(sa)
    if not esd_gen:
        raise RuntimeError("No ESD1 devices found")

    pv_idx = esd_gen

    for model, idx, fields in (
        (sa.StaticGen, esd_gen, ("Sn", "pmax", "pmin", "qmax", "qmin")),
        (sa.PV, pv_idx, ("Sn", "pmax", "pmin", "qmax", "qmin")),
        (sa.ESD1, esd_idx, ("Sn", "En", "pmx", "qmx", "qmn", "ialim")),
    ):
        for field in fields:
            if hasattr(model, field):
                values = np.asarray(model.get(src=field, attr="v", idx=idx), dtype=float)
                model.set(src=field, idx=idx, attr="v", value=values * factor)

    if hasattr(sa.ESD1, "pmin") and esd_idx:
        sa.ESD1.pmin.v[:] = -np.asarray(sa.ESD1.pmx.v, dtype=float)

    after = storage_share(sa)
    return {
        "factor": float(factor),
        "after_pmax_share": float(after["pmax_share"]),
        "after_sn_share": float(after["sn_share"]),
    }


def configure_all_der_deadband(
    sa,
    *,
    traditional_governor_deadband_hz: float | None,
    der_deadband_hz: float | None,
    der_base_ddn: float | None,
    pvd1_base_ddn: float | None,
    esd1_base_ddn: float | None,
    esd1_ddn_scale: float,
) -> dict[str, object]:
    meta: dict[str, object] = {
        "traditional_governor_deadband": [],
        "der_deadband_disabled": [],
        "configured_pvd1_count": 0,
        "configured_esd1_count": 0,
        "pvd1_ddn": None,
        "esd1_ddn": None,
    }

    if traditional_governor_deadband_hz is not None and traditional_governor_deadband_hz > 0.0:
        meta["traditional_governor_deadband"] = rdt.apply_traditional_governor_deadband(
            sa,
            float(traditional_governor_deadband_hz),
        )

    if der_deadband_hz is None or der_deadband_hz <= 0.0:
        return meta

    pvd1_ddn = pvd1_base_ddn if pvd1_base_ddn is not None else der_base_ddn
    esd1_ddn_base = esd1_base_ddn if esd1_base_ddn is not None else der_base_ddn
    if pvd1_ddn is None:
        raise ValueError("pvd1_base_ddn or der_base_ddn is required when DER deadband is enabled")
    if esd1_ddn_base is None:
        raise ValueError("esd1_base_ddn or der_base_ddn is required when DER deadband is enabled")

    meta["der_deadband_disabled"] = rdt.disable_der_frequency_deadband(sa)

    if hasattr(sa, "PVD1") and sa.PVD1.n:
        idx = list(sa.PVD1.idx.v)
        n = len(idx)
        sa.PVD1.set(src="fdbd", idx=idx, attr="v", value=np.full(n, -float(der_deadband_hz)))
        sa.PVD1.set(src="fdbdu", idx=idx, attr="v", value=np.full(n, float(der_deadband_hz)))
        sa.PVD1.set(src="ddn", idx=idx, attr="v", value=np.full(n, float(pvd1_ddn)))
        meta["configured_pvd1_count"] = n
        meta["pvd1_ddn"] = float(pvd1_ddn)

    if hasattr(sa, "ESD1") and sa.ESD1.n:
        idx = list(sa.ESD1.idx.v)
        n = len(idx)
        esd1_ddn = float(esd1_ddn_base) * float(esd1_ddn_scale)
        sa.ESD1.set(src="fdbd", idx=idx, attr="v", value=np.full(n, -float(der_deadband_hz)))
        sa.ESD1.set(src="fdbdu", idx=idx, attr="v", value=np.full(n, float(der_deadband_hz)))
        sa.ESD1.set(src="ddn", idx=idx, attr="v", value=np.full(n, esd1_ddn))
        meta["configured_esd1_count"] = n
        meta["esd1_ddn"] = esd1_ddn

    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatch-json", type=Path, default=None,
                        help="Existing dispatch JSON to replay through TDS.")
    parser.add_argument("--next-dispatch-json", type=Path, default=None,
                        help="Optional next dispatch JSON used to build a smooth governor target trajectory.")
    parser.add_argument("--hour", type=int, default=13,
                        help="Dispatch hour used when recomputing from AMS.")
    parser.add_argument("--dispatch", type=int, default=2,
                        help="Dispatch interval used when recomputing from AMS.")
    parser.add_argument("--label", type=str, default=None,
                        help="Output label. Defaults to the dispatch label.")
    parser.add_argument("--checkpoint-in", type=Path, default=None,
                        help="Checkpoint directory from the previous dispatch boundary.")
    parser.add_argument("--checkpoint-out", type=Path, default=None,
                        help="Explicit checkpoint directory for the current dispatch end state.")
    parser.add_argument("--checkpoints-dir", type=Path, default=rdt.RESULTS / "checkpoints")
    parser.add_argument("--results-dir", type=Path, default=rdt.RESULTS / "hotstart_segments")
    parser.add_argument("--opf-case", type=Path, default=rdt.DEFAULT_OPF_CASE)
    parser.add_argument("--dyn-case", type=Path, default=rdt.DEFAULT_DYN_CASE)
    parser.add_argument("--stable-dyn-case", type=Path, default=rdt.DEFAULT_STABLE_DYN_CASE)
    parser.add_argument("--curve-file", type=Path, default=rdt.DEFAULT_CURVE_FILE)
    parser.add_argument("--duration-seconds", type=int, default=900)
    parser.add_argument("--agc-interval", type=int, default=4)
    parser.add_argument("--kp", type=float, default=0.03)
    parser.add_argument("--ki", type=float, default=0.01)
    parser.add_argument("--wind-pref-alpha", type=float, default=1.0)
    parser.add_argument("--solar-pref-alpha", type=float, default=1.0)
    parser.add_argument(
        "--agc-allocation-mode",
        choices=rdt.AGC_ALLOCATION_MODES,
        default=AGC_ALLOCATION_HEADROOM,
        help="How to distribute AGC across online governor / DG participants.",
    )
    parser.add_argument(
        "--agc-gov-output-ramp-frac-pmax-per-min",
        type=float,
        default=0.0,
        help="Ramp limit for governor AGC output paux0, in pmax-fraction per minute. 0 disables output ramp limiting.",
    )
    parser.add_argument(
        "--agc-dg-output-ramp-frac-pmax-per-min",
        type=float,
        default=0.0,
        help="Ramp limit for DG AGC output Pext0, in pmax-fraction per minute. 0 disables output ramp limiting.",
    )
    parser.add_argument(
        "--agc-anti-windup-mode",
        choices=(AGC_ANTI_WINDUP_OFF, AGC_ANTI_WINDUP_FREEZE),
        default=AGC_ANTI_WINDUP_OFF,
        help="Anti-windup strategy for the external AGC PI integrator.",
    )
    parser.add_argument("--dispatch-target-ramp-seconds", type=int, default=0)
    parser.add_argument(
        "--governor-target-schedule",
        choices=("step", "boundary_ramp", "midpoint_trajectory", "ramp_limited_basepoint"),
        default="midpoint_trajectory",
        help="How to apply conventional generator dispatch targets when enabled.",
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
    parser.add_argument("--init-mode", choices=("dispatch", "first"), default="first")
    parser.add_argument("--wind-prefix", action="append", default=None)
    parser.add_argument("--solar-prefix", action="append", default=None)
    parser.add_argument(
        "--wind-deadband-hz",
        type=float,
        default=None,
        help="Override wind PVD1 frequency deadband in Hz while preserving ddn.",
    )
    parser.add_argument(
        "--solar-deadband-hz",
        type=float,
        default=None,
        help="Override solar PVD1 frequency deadband in Hz while preserving ddn.",
    )
    parser.add_argument(
        "--esd-deadband-hz",
        type=float,
        default=None,
        help="Override ESD1 frequency deadband in Hz while preserving ddn.",
    )
    parser.add_argument(
        "--traditional-governor-deadband-hz",
        type=float,
        default=None,
        help="Apply a symmetric +/- deadband in Hz to conventional governor models.",
    )
    parser.add_argument(
        "--traditional-governor-deadband-csv",
        type=Path,
        default=None,
        help="Apply a per-unit TGOV1NDB deadband scheme from CSV. Takes precedence over --traditional-governor-deadband-hz.",
    )
    parser.add_argument(
        "--der-deadband-hz",
        type=float,
        default=None,
        help="Apply a symmetric +/- deadband in Hz to all PVD1 / ESD1 devices.",
    )
    parser.add_argument(
        "--der-base-ddn",
        type=float,
        default=None,
        help="Base ddn used when enabling DER deadband. PVD1 uses this value directly.",
    )
    parser.add_argument(
        "--pvd1-base-ddn",
        type=float,
        default=None,
        help="Override PVD1 ddn when enabling DER deadband. Falls back to --der-base-ddn.",
    )
    parser.add_argument(
        "--esd1-base-ddn",
        type=float,
        default=None,
        help="Override ESD1 ddn when enabling DER deadband. Falls back to --der-base-ddn.",
    )
    parser.add_argument(
        "--pvd1-tfdb",
        type=float,
        default=None,
        help="Override PVD1 Tfdb lag on frequency droop output.",
    )
    parser.add_argument(
        "--esd1-tfdb",
        type=float,
        default=None,
        help="Override ESD1 Tfdb lag on frequency droop output.",
    )
    parser.add_argument(
        "--target-storage-share",
        type=float,
        default=None,
        help="Scale existing ESD1 capacity to this total StaticGen pmax share before running.",
    )
    parser.add_argument(
        "--scale-esd1-ddn-with-storage",
        dest="scale_esd1_ddn_with_storage",
        action="store_true",
        help="When storage is scaled, multiply ESD1 ddn by the same storage scale factor.",
    )
    parser.add_argument(
        "--no-scale-esd1-ddn-with-storage",
        dest="scale_esd1_ddn_with_storage",
        action="store_false",
    )
    parser.add_argument(
        "--disable-der-frequency-deadband",
        action="store_true",
        help="Explicitly disable PVD1/ESD1 frequency deadband / droop for this run.",
    )
    parser.add_argument(
        "--disable-der-agc",
        action="store_true",
        help="Do not allocate AGC Pext0 commands to PVD1 / ESD1 / other DG models.",
    )
    parser.add_argument(
        "--disable-pvd-agc",
        action="store_true",
        help="Do not allocate AGC Pext0 commands to wind/PV PVD1 devices.",
    )
    parser.add_argument(
        "--disable-esd-agc",
        action="store_true",
        help="Do not allocate AGC Pext0 commands to ESD1 storage devices.",
    )
    parser.add_argument(
        "--recompute-ace-raw-on-load",
        action="store_true",
        help="When resuming from a checkpoint, recompute the initial ACE control output with the current KP/KI.",
    )
    parser.add_argument(
        "--reset-ace-integral-on-load",
        action="store_true",
        help="When resuming from a checkpoint, reset the saved AGC integral state before the new run.",
    )
    parser.add_argument("--apply-governor-targets", dest="apply_governor_targets", action="store_true")
    parser.add_argument("--no-apply-governor-targets", dest="apply_governor_targets", action="store_false")
    parser.add_argument(
        "--apply-dg-targets",
        dest="apply_dg_targets",
        action="store_true",
        help="Deprecated and ignored. DG/PVD1/ESD1 dispatch targets are not applied.",
    )
    parser.add_argument(
        "--no-apply-dg-targets",
        dest="apply_dg_targets",
        action="store_false",
        help="Deprecated compatibility flag; DG/PVD1/ESD1 dispatch targets are never applied.",
    )
    parser.add_argument("--allow-signature-mismatch", action="store_true",
                        help="Do not fail when the checkpoint signature differs from the current settings.")
    parser.add_argument(
        "--save-agc-trace",
        action="store_true",
        help="Save a per-step AGC trace CSV for this single dispatch run.",
    )
    parser.add_argument("--no-save-checkpoint", dest="save_checkpoint", action="store_false")
    parser.add_argument("--save-plot", dest="save_plot", action="store_true")
    parser.add_argument("--no-save-plot", dest="save_plot", action="store_false")
    parser.set_defaults(
        apply_governor_targets=False,
        apply_dg_targets=False,
        scale_esd1_ddn_with_storage=False,
        save_checkpoint=True,
        save_plot=True,
    )
    return parser.parse_args()


def file_digest(path: Path | None) -> str | None:
    if path is None:
        return None
    path = Path(path).resolve()
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_dispatch_record(args: argparse.Namespace, curve: pd.DataFrame) -> tuple[rdt.DispatchRecord, Path | None]:
    if args.dispatch_json is not None:
        return rdt.DispatchRecord.from_json(args.dispatch_json), args.dispatch_json

    dispatch_record = rdt.compute_dispatch(
        args.hour,
        args.dispatch,
        curve,
        args.opf_case,
        args.duration_seconds,
        wind_pref_alpha=args.wind_pref_alpha,
        solar_pref_alpha=args.solar_pref_alpha,
    )
    return dispatch_record, None


def build_signature(
    args: argparse.Namespace,
    *,
    dyn_case: Path,
    wind_prefixes: tuple[str, ...],
    solar_prefixes: tuple[str, ...],
) -> dict[str, object]:
    dispatch_interval = getattr(args, "duration_seconds", None)
    if dispatch_interval is None:
        dispatch_interval = getattr(args, "dispatch_interval")
    dispatch_interval = int(dispatch_interval)
    return hcp.build_param_signature(
        kp=args.kp,
        ki=args.ki,
        agc_interval=args.agc_interval,
        init_mode=args.init_mode,
        dispatch_interval=dispatch_interval,
        curve_file=args.curve_file,
        dyn_case=args.dyn_case,
        stable_dyn_case=dyn_case,
        wind_prefixes=wind_prefixes,
        solar_prefixes=solar_prefixes,
        extra={
            "runner": "run_dispatch_hotstart",
            "agc_control_version": "aggregate_freeze_v4_headroom_alloc",
            "wind_pref_alpha": float(getattr(args, "wind_pref_alpha", 1.0)),
            "solar_pref_alpha": float(getattr(args, "solar_pref_alpha", 1.0)),
            "apply_governor_targets": bool(getattr(args, "apply_governor_targets", True)),
            "dispatch_target_scope": "governor_only",
            "governor_target_schedule": str(getattr(args, "governor_target_schedule", "midpoint_trajectory")),
            "dispatch_target_ramp_seconds": int(getattr(args, "dispatch_target_ramp_seconds", 0)),
            "agc_gov_output_ramp_frac_pmax_per_min": float(
                getattr(args, "agc_gov_output_ramp_frac_pmax_per_min", 0.0)
            ),
            "agc_dg_output_ramp_frac_pmax_per_min": float(
                getattr(args, "agc_dg_output_ramp_frac_pmax_per_min", 0.0)
            ),
            "agc_allocation_mode": str(getattr(args, "agc_allocation_mode", AGC_ALLOCATION_HEADROOM)),
            "agc_anti_windup_mode": str(getattr(args, "agc_anti_windup_mode", AGC_ANTI_WINDUP_OFF)),
            "traditional_governor_deadband_hz": (
                None if getattr(args, "traditional_governor_deadband_hz", None) is None
                else float(args.traditional_governor_deadband_hz)
            ),
            "traditional_governor_deadband_csv": (
                None
                if getattr(args, "traditional_governor_deadband_csv", None) is None
                else str(Path(args.traditional_governor_deadband_csv).resolve())
            ),
            "traditional_governor_deadband_csv_digest": file_digest(
                getattr(args, "traditional_governor_deadband_csv", None)
            ),
            "wind_deadband_hz": (
                None if getattr(args, "wind_deadband_hz", None) is None
                else float(args.wind_deadband_hz)
            ),
            "solar_deadband_hz": (
                None if getattr(args, "solar_deadband_hz", None) is None
                else float(args.solar_deadband_hz)
            ),
            "esd_deadband_hz": (
                None if getattr(args, "esd_deadband_hz", None) is None
                else float(args.esd_deadband_hz)
            ),
            "der_deadband_hz": (
                None if getattr(args, "der_deadband_hz", None) is None
                else float(args.der_deadband_hz)
            ),
            "der_base_ddn": (
                None if getattr(args, "der_base_ddn", None) is None
                else float(args.der_base_ddn)
            ),
            "pvd1_base_ddn": (
                None if getattr(args, "pvd1_base_ddn", None) is None
                else float(args.pvd1_base_ddn)
            ),
            "esd1_base_ddn": (
                None if getattr(args, "esd1_base_ddn", None) is None
                else float(args.esd1_base_ddn)
            ),
            "pvd1_tfdb": (
                None if getattr(args, "pvd1_tfdb", None) is None
                else float(args.pvd1_tfdb)
            ),
            "esd1_tfdb": (
                None if getattr(args, "esd1_tfdb", None) is None
                else float(args.esd1_tfdb)
            ),
            "target_storage_share": (
                None if getattr(args, "target_storage_share", None) is None
                else float(args.target_storage_share)
            ),
            "scale_esd1_ddn_with_storage": bool(getattr(args, "scale_esd1_ddn_with_storage", False)),
            "disable_der_frequency_deadband": bool(getattr(args, "disable_der_frequency_deadband", False)),
            "disable_der_agc": bool(getattr(args, "disable_der_agc", False)),
            "disable_pvd_agc": bool(getattr(args, "disable_pvd_agc", False)),
            "disable_esd_agc": bool(getattr(args, "disable_esd_agc", False)),
            "recompute_ace_raw_on_load": bool(getattr(args, "recompute_ace_raw_on_load", False)),
            "reset_ace_integral_on_load": bool(getattr(args, "reset_ace_integral_on_load", False)),
        },
    )


def main() -> None:
    args = parse_args()
    if COMPLEX_WARNING is not None:
        warnings.filterwarnings("ignore", category=COMPLEX_WARNING)
    rdt.andes.config_logger(stream_level=30)

    curve = rdt.load_curve(args.curve_file)
    dispatch_record, dispatch_json_source = load_dispatch_record(args, curve)
    next_dispatch_record = (
        rdt.DispatchRecord.from_json(args.next_dispatch_json)
        if args.next_dispatch_json is not None else None
    )
    if not dispatch_record.converged:
        raise RuntimeError(f"Dispatch {dispatch_record.label} did not converge")

    label = args.label or dispatch_record.label
    args.results_dir.mkdir(parents=True, exist_ok=True)

    dyn_case = rdt.adapt_dyn_case(args.dyn_case, args.stable_dyn_case)
    wind_prefixes = rdt.normalize_prefixes(args.wind_prefix, rdt.DEFAULT_WIND_PREFIXES)
    solar_prefixes = rdt.normalize_prefixes(args.solar_prefix, rdt.DEFAULT_SOLAR_PREFIXES)
    signature = build_signature(
        args,
        dyn_case=dyn_case,
        wind_prefixes=wind_prefixes,
        solar_prefixes=solar_prefixes,
    )
    signature_hash = hcp.param_hash(signature)
    signature_path = hcp.ensure_family_manifest(args.checkpoints_dir, signature)
    dispatch_target_transition = None

    if args.checkpoint_in is None:
        sa, ctx = prepare_system(
            dispatch_record=dispatch_record,
            curve=curve,
            dyn_case=dyn_case,
            dispatch_interval=args.duration_seconds,
            init_mode=args.init_mode,
            wind_prefixes=wind_prefixes,
            solar_prefixes=solar_prefixes,
            wind_pref_alpha=args.wind_pref_alpha,
            solar_pref_alpha=args.solar_pref_alpha,
        )
        ace_integral = 0.0
        ace_raw = 0.0
        agc_aw_state = initial_agc_aw_state()
        source_checkpoint = ""
        source_manifest = None
    else:
        sa, stored_ctx, agc_state, source_manifest = hcp.load_checkpoint(args.checkpoint_in)
        if not args.allow_signature_mismatch:
            hcp.validate_signature(signature, source_manifest["param_signature"])
        ctx = hcp.build_runtime_context(sa=sa, curve=curve, stored_ctx=stored_ctx)
        ace_integral = float(agc_state["ace_integral"])
        ace_raw = float(agc_state["ace_raw"])
        agc_aw_state = initial_agc_aw_state()
        for key in agc_aw_state:
            if key in agc_state:
                agc_aw_state[key] = int(agc_state[key])
        source_checkpoint = str(args.checkpoint_in)

        if args.reset_ace_integral_on_load:
            ace_integral = 0.0
            ace_raw = 0.0
            agc_aw_state = initial_agc_aw_state()

        if args.recompute_ace_raw_on_load:
            ace_sum = float(sa.ACEc.ace.v.sum())
            ace_raw = -(float(args.kp) * ace_sum + float(args.ki) * ace_integral)

    ctx["link"] = rdt.configure_der_agc_participation(
        sa,
        ctx["link"],  # type: ignore[arg-type]
        enable_der_agc=not args.disable_der_agc,
        enable_pvd_agc=not args.disable_pvd_agc,
        enable_esd_agc=not args.disable_esd_agc,
    )

    share_before = storage_share(sa)
    prior_storage_scale = source_manifest.get("storage_scale", {}) if source_manifest is not None else {}
    prior_target_share = source_manifest.get("target_storage_share") if source_manifest is not None else None
    if (
        args.target_storage_share is not None
        and prior_target_share is not None
        and abs(float(prior_target_share) - float(args.target_storage_share)) < 1e-12
    ):
        total_storage_scale_factor = float(prior_storage_scale.get("total_factor", prior_storage_scale.get("factor", 1.0)))
        base_pmax_share = float(prior_storage_scale.get("base_pmax_share", share_before["pmax_share"]))
        base_sn_share = float(prior_storage_scale.get("base_sn_share", share_before["sn_share"]))
    else:
        total_storage_scale_factor = 1.0
        base_pmax_share = float(share_before["pmax_share"])
        base_sn_share = float(share_before["sn_share"])

    storage_scale = {
        "applied_factor": 1.0,
        "total_factor": float(total_storage_scale_factor),
        "base_pmax_share": float(base_pmax_share),
        "base_sn_share": float(base_sn_share),
        "before_pmax_share": float(share_before["pmax_share"]),
        "before_sn_share": float(share_before["sn_share"]),
        "after_pmax_share": float(share_before["pmax_share"]),
        "after_sn_share": float(share_before["sn_share"]),
    }
    if args.target_storage_share is not None:
        current_share = float(share_before["pmax_share"])
        applied_factor = (
            1.0 if abs(float(args.target_storage_share) - current_share) < 1e-12
            else solve_scale_factor(current_share, float(args.target_storage_share))
        )
        if abs(applied_factor - 1.0) > 1e-12:
            scaled_meta = scale_storage_capacity(sa, applied_factor)
            storage_scale.update(
                {
                    "applied_factor": float(applied_factor),
                    "after_pmax_share": float(scaled_meta["after_pmax_share"]),
                    "after_sn_share": float(scaled_meta["after_sn_share"]),
                }
            )
            total_storage_scale_factor = float(applied_factor)
        else:
            total_storage_scale_factor = float(storage_scale["total_factor"])
            storage_scale.update(
                {
                    "applied_factor": 1.0,
                    "after_pmax_share": current_share,
                    "after_sn_share": float(share_before["sn_share"]),
                }
            )
        storage_scale["total_factor"] = float(total_storage_scale_factor)

    der_deadband_disabled: list[dict[str, object]] = []
    if args.disable_der_frequency_deadband and not (args.der_deadband_hz is not None and args.der_deadband_hz > 0.0):
        der_deadband_disabled = rdt.disable_der_frequency_deadband(sa)

    configured_deadband = {
        "traditional_governor_deadband": [],
        "der_deadband_disabled": [],
        "configured_pvd1_count": 0,
        "configured_esd1_count": 0,
        "pvd1_ddn": None,
        "esd1_ddn": None,
    }
    if args.der_deadband_hz is not None and args.der_deadband_hz > 0.0:
        configured_deadband = configure_all_der_deadband(
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
                float(storage_scale["total_factor"]) if args.scale_esd1_ddn_with_storage else 1.0
            ),
        )
        der_deadband_disabled = list(configured_deadband["der_deadband_disabled"])

    resource_deadband_overrides = rdt.apply_resource_deadband_overrides(
        sa,
        wind_prefixes=wind_prefixes,
        solar_prefixes=solar_prefixes,
        wind_deadband_hz=args.wind_deadband_hz,
        solar_deadband_hz=args.solar_deadband_hz,
        esd_deadband_hz=args.esd_deadband_hz,
    )

    if args.pvd1_tfdb is not None and hasattr(sa, "PVD1") and sa.PVD1.n:
        sa.PVD1.set(
            src="Tfdb",
            idx=sa.PVD1.idx.v,
            attr="v",
            value=np.full(sa.PVD1.n, float(args.pvd1_tfdb)),
        )
    if args.esd1_tfdb is not None and hasattr(sa, "ESD1") and sa.ESD1.n:
        sa.ESD1.set(
            src="Tfdb",
            idx=sa.ESD1.idx.v,
            attr="v",
            value=np.full(sa.ESD1.n, float(args.esd1_tfdb)),
        )

    traditional_governor_deadband: list[dict[str, object]] = list(configured_deadband["traditional_governor_deadband"])
    traditional_governor_deadband_source = ""
    if args.traditional_governor_deadband_csv is not None:
        traditional_governor_deadband = rdt.apply_traditional_governor_deadband_csv(
            sa,
            args.traditional_governor_deadband_csv,
            model_name="TGOV1NDB",
        )
        traditional_governor_deadband_source = str(Path(args.traditional_governor_deadband_csv).resolve())
    elif (
        not traditional_governor_deadband
        and args.traditional_governor_deadband_hz is not None
        and args.traditional_governor_deadband_hz > 0.0
    ):
        traditional_governor_deadband = rdt.apply_traditional_governor_deadband(
            sa,
            float(args.traditional_governor_deadband_hz),
        )
        traditional_governor_deadband_source = "uniform_hz"

    if args.apply_governor_targets:
        schedule_mode = args.governor_target_schedule
        build_duration = (
            args.duration_seconds
            if schedule_mode in ("midpoint_trajectory", "ramp_limited_basepoint")
            else None
        )
        dispatch_target_transition = apply_second_dispatch_targets(
            sa,
            ctx["link"],  # type: ignore[arg-type]
            dispatch_record,
            apply_governor_targets=True,
            apply_dg_targets=False,
            duration_seconds=build_duration,
            schedule_mode=schedule_mode,
            next_dispatch_record=next_dispatch_record if schedule_mode == "midpoint_trajectory" else None,
            basepoint_ramp_floor_frac_pmax_per_min=args.governor_basepoint_ramp_floor_frac_pmax_per_min,
            basepoint_ramp_gap_factor=args.governor_basepoint_ramp_gap_factor,
        )
        if schedule_mode == "boundary_ramp":
            dispatch_target_transition["ramp_seconds"] = int(args.dispatch_target_ramp_seconds)
        else:
            dispatch_target_transition["ramp_seconds"] = 0
        activate_dispatch_target_transition(sa, dispatch_target_transition, step=0)

    bf = compute_bf(sa, dispatch_record)
    agc_metrics: dict[str, float | int] = {}
    agc_trace: dict[str, np.ndarray] | None = {} if args.save_agc_trace else None
    agc_aw_state_start = dict(agc_aw_state)
    t, f_dev_hz, ace_integral_end, ace_raw_end = run_segment(
        sa=sa,
        ctx=ctx,
        start_offset=dispatch_offset(dispatch_record, args.duration_seconds),
        duration_seconds=args.duration_seconds,
        agc_interval=args.agc_interval,
        kp=args.kp,
        ki=args.ki,
        bf=bf,
        agc_allocation_mode=args.agc_allocation_mode,
        ace_integral=ace_integral,
        ace_raw=ace_raw,
        local_start=0.0,
        include_initial=True,
        dispatch_target_transition=dispatch_target_transition,
        gov_output_ramp_frac_pmax_per_min=args.agc_gov_output_ramp_frac_pmax_per_min,
        dg_output_ramp_frac_pmax_per_min=args.agc_dg_output_ramp_frac_pmax_per_min,
        agc_anti_windup_mode=args.agc_anti_windup_mode,
        agc_aw_state=agc_aw_state,
        metrics_out=agc_metrics,
        trace_out=agc_trace,
        wind_pref_alpha=args.wind_pref_alpha,
        solar_pref_alpha=args.solar_pref_alpha,
    )

    dispatch_json_path = rdt.write_dispatch_json(dispatch_record, args.results_dir, label=label)
    csv_path, png_path = rdt.save_outputs(
        t,
        f_dev_hz,
        dispatch_record,
        args.results_dir,
        label=label,
        save_plot=args.save_plot,
    )

    summary = {
        "label": label,
        "dispatch_label": dispatch_record.label,
        "hour": dispatch_record.hour,
        "dispatch": dispatch_record.dispatch,
        "dispatch_json": str(dispatch_json_path),
        "dispatch_json_source": str(dispatch_json_source) if dispatch_json_source is not None else "",
        "next_dispatch_json": str(args.next_dispatch_json) if args.next_dispatch_json is not None else "",
        "freq_csv": str(csv_path),
        "freq_png": str(png_path),
        "checkpoint_in": source_checkpoint,
        "signature_hash": signature_hash,
        "kp": args.kp,
        "ki": args.ki,
        "agc_interval": args.agc_interval,
        "wind_pref_alpha": float(args.wind_pref_alpha),
        "solar_pref_alpha": float(args.solar_pref_alpha),
        "agc_allocation_mode": str(args.agc_allocation_mode),
        "init_mode": args.init_mode,
        "apply_governor_targets": int(args.apply_governor_targets),
        "apply_dg_targets": 0,
        "governor_target_schedule": args.governor_target_schedule,
        "dispatch_target_ramp_seconds": int(args.dispatch_target_ramp_seconds),
        "agc_gov_output_ramp_frac_pmax_per_min": float(args.agc_gov_output_ramp_frac_pmax_per_min),
        "agc_dg_output_ramp_frac_pmax_per_min": float(args.agc_dg_output_ramp_frac_pmax_per_min),
        "agc_anti_windup_mode": str(args.agc_anti_windup_mode),
        "traditional_governor_deadband_hz": (
            "" if args.traditional_governor_deadband_hz is None else float(args.traditional_governor_deadband_hz)
        ),
        "traditional_governor_deadband_csv": traditional_governor_deadband_source,
        "wind_deadband_hz": "" if args.wind_deadband_hz is None else float(args.wind_deadband_hz),
        "solar_deadband_hz": "" if args.solar_deadband_hz is None else float(args.solar_deadband_hz),
        "esd_deadband_hz": "" if args.esd_deadband_hz is None else float(args.esd_deadband_hz),
        "der_deadband_hz": "" if args.der_deadband_hz is None else float(args.der_deadband_hz),
        "der_base_ddn": "" if args.der_base_ddn is None else float(args.der_base_ddn),
        "pvd1_base_ddn": "" if args.pvd1_base_ddn is None else float(args.pvd1_base_ddn),
        "esd1_base_ddn": "" if args.esd1_base_ddn is None else float(args.esd1_base_ddn),
        "pvd1_tfdb": "" if args.pvd1_tfdb is None else float(args.pvd1_tfdb),
        "esd1_tfdb": "" if args.esd1_tfdb is None else float(args.esd1_tfdb),
        "target_storage_share": "" if args.target_storage_share is None else float(args.target_storage_share),
        "achieved_storage_share": float(storage_scale["after_pmax_share"]),
        "storage_scale_factor": float(storage_scale["total_factor"]),
        "storage_scale_applied_factor": float(storage_scale["applied_factor"]),
        "scale_esd1_ddn_with_storage": int(args.scale_esd1_ddn_with_storage),
        "disable_der_frequency_deadband": int(args.disable_der_frequency_deadband),
        "disable_der_agc": int(args.disable_der_agc),
        "disable_pvd_agc": int(args.disable_pvd_agc),
        "disable_esd_agc": int(args.disable_esd_agc),
        "recompute_ace_raw_on_load": int(args.recompute_ace_raw_on_load),
        "reset_ace_integral_on_load": int(args.reset_ace_integral_on_load),
        "resume_mode": "checkpoint" if args.checkpoint_in is not None else "cold",
        "start_dae_t": float(source_manifest["end_dae_t"]) if source_manifest is not None else 0.0,
        "end_dae_t": float(sa.dae.t),
        "ace_integral_start": float(ace_integral),
        "ace_raw_start": float(ace_raw),
        "ace_integral_end": float(ace_integral_end),
        "ace_raw_end": float(ace_raw_end),
        "agc_freeze_active_start": int(agc_aw_state_start.get("freeze_active", 0)),
        "agc_freeze_on_streak_start": int(agc_aw_state_start.get("freeze_on_streak", 0)),
        "agc_freeze_off_streak_start": int(agc_aw_state_start.get("freeze_off_streak", 0)),
        "agc_freeze_dir_start": int(agc_aw_state_start.get("freeze_dir", 0)),
        "agc_freeze_active_end": int(agc_aw_state.get("freeze_active", 0)),
        "agc_freeze_on_streak_end": int(agc_aw_state.get("freeze_on_streak", 0)),
        "agc_freeze_off_streak_end": int(agc_aw_state.get("freeze_off_streak", 0)),
        "agc_freeze_dir_end": int(agc_aw_state.get("freeze_dir", 0)),
        "der_deadband_disabled_count": int(sum(item["count"] for item in der_deadband_disabled)),
        "traditional_governor_deadband_count": int(sum(item["count"] for item in traditional_governor_deadband)),
        "configured_pvd1_deadband_count": int(configured_deadband["configured_pvd1_count"]),
        "configured_esd1_deadband_count": int(configured_deadband["configured_esd1_count"]),
        "configured_pvd1_ddn": "" if configured_deadband["pvd1_ddn"] is None else float(configured_deadband["pvd1_ddn"]),
        "configured_esd1_ddn": "" if configured_deadband["esd1_ddn"] is None else float(configured_deadband["esd1_ddn"]),
        "configured_wind_pvd1_deadband_count": int(resource_deadband_overrides["configured_wind_pvd1_count"]),
        "configured_solar_pvd1_deadband_count": int(resource_deadband_overrides["configured_solar_pvd1_count"]),
        "configured_esd1_resource_deadband_count": int(resource_deadband_overrides["configured_esd1_count"]),
    }
    summary.update(summarize_series(t, f_dev_hz))
    summary.update(agc_metrics)
    if agc_trace is not None:
        trace_csv = args.results_dir / f"{label}_agc_trace.csv"
        pd.DataFrame(agc_trace).to_csv(trace_csv, index=False)
        summary["agc_trace_csv"] = str(trace_csv)
    summary_csv = args.results_dir / f"{label}_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    checkpoint_saved = ""
    if args.save_checkpoint:
        checkpoint_out = args.checkpoint_out or hcp.checkpoint_dir(args.checkpoints_dir, signature, dispatch_record.label)
        manifest = {
            "format": "deadband_hotstart_v1",
            "dispatch_label": dispatch_record.label,
            "hour": dispatch_record.hour,
            "dispatch": dispatch_record.dispatch,
            "dispatch_json": str(dispatch_json_path),
            "checkpoint_dir": str(checkpoint_out),
            "checkpoint_in": source_checkpoint,
            "curve_file": str(args.curve_file.resolve()),
            "dyn_case": str(args.dyn_case.resolve()),
            "stable_dyn_case": str(dyn_case.resolve()),
            "wind_prefixes": list(wind_prefixes),
            "solar_prefixes": list(solar_prefixes),
            "duration_seconds": int(args.duration_seconds),
            "agc_interval": int(args.agc_interval),
            "wind_pref_alpha": float(args.wind_pref_alpha),
            "solar_pref_alpha": float(args.solar_pref_alpha),
            "agc_allocation_mode": str(args.agc_allocation_mode),
            "agc_gov_output_ramp_frac_pmax_per_min": float(args.agc_gov_output_ramp_frac_pmax_per_min),
            "agc_dg_output_ramp_frac_pmax_per_min": float(args.agc_dg_output_ramp_frac_pmax_per_min),
            "agc_anti_windup_mode": str(args.agc_anti_windup_mode),
            "end_dae_t": float(sa.dae.t),
            "disable_der_frequency_deadband": bool(args.disable_der_frequency_deadband),
            "disable_der_agc": bool(args.disable_der_agc),
            "disable_pvd_agc": bool(args.disable_pvd_agc),
            "disable_esd_agc": bool(args.disable_esd_agc),
            "der_deadband_disabled": der_deadband_disabled,
            "traditional_governor_deadband_hz": (
                None if args.traditional_governor_deadband_hz is None else float(args.traditional_governor_deadband_hz)
            ),
            "traditional_governor_deadband_csv": traditional_governor_deadband_source or None,
            "wind_deadband_hz": None if args.wind_deadband_hz is None else float(args.wind_deadband_hz),
            "solar_deadband_hz": None if args.solar_deadband_hz is None else float(args.solar_deadband_hz),
            "esd_deadband_hz": None if args.esd_deadband_hz is None else float(args.esd_deadband_hz),
            "der_deadband_hz": None if args.der_deadband_hz is None else float(args.der_deadband_hz),
            "der_base_ddn": None if args.der_base_ddn is None else float(args.der_base_ddn),
            "pvd1_base_ddn": None if args.pvd1_base_ddn is None else float(args.pvd1_base_ddn),
            "esd1_base_ddn": None if args.esd1_base_ddn is None else float(args.esd1_base_ddn),
            "pvd1_tfdb": None if args.pvd1_tfdb is None else float(args.pvd1_tfdb),
            "esd1_tfdb": None if args.esd1_tfdb is None else float(args.esd1_tfdb),
            "target_storage_share": None if args.target_storage_share is None else float(args.target_storage_share),
            "storage_scale": storage_scale,
            "scale_esd1_ddn_with_storage": bool(args.scale_esd1_ddn_with_storage),
            "traditional_governor_deadband": traditional_governor_deadband,
            "configured_pvd1_deadband_count": int(configured_deadband["configured_pvd1_count"]),
            "configured_esd1_deadband_count": int(configured_deadband["configured_esd1_count"]),
            "configured_pvd1_ddn": configured_deadband["pvd1_ddn"],
            "configured_esd1_ddn": configured_deadband["esd1_ddn"],
            "resource_deadband_overrides": resource_deadband_overrides,
            "recompute_ace_raw_on_load": bool(args.recompute_ace_raw_on_load),
            "reset_ace_integral_on_load": bool(args.reset_ace_integral_on_load),
            "param_signature": signature,
            "param_signature_path": str(signature_path),
            "param_hash": signature_hash,
        }
        hcp.save_checkpoint(
            checkpoint_dir=checkpoint_out,
            sa=sa,
            ctx=ctx,
            ace_integral=ace_integral_end,
            ace_raw=ace_raw_end,
            agc_aw_state=agc_aw_state,
            manifest=manifest,
        )
        checkpoint_saved = str(checkpoint_out)

    print(f"dispatch_json={dispatch_json_path}")
    print(f"freq_csv={csv_path}")
    print(f"freq_plot={png_path}")
    print(f"summary_csv={summary_csv}")
    if checkpoint_saved:
        print(f"checkpoint_dir={checkpoint_saved}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
