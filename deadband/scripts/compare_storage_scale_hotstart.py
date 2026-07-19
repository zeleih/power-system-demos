#!/usr/bin/env python3
"""
Compare one hot-start dispatch under different storage penetration levels.

The comparison keeps:

- the same base checkpoint,
- the same dispatch trajectory,
- wind/PV/storage deadband enabled,
- conventional governor deadband enabled,
- AGC gains unchanged.

Only the existing ESD1 devices are scaled in-place to hit target storage shares.
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
    parser.add_argument("--warmup-dispatch-json", type=Path, required=True)
    parser.add_argument("--dispatch-json", type=Path, required=True)
    parser.add_argument("--next-dispatch-json", type=Path, required=True)
    parser.add_argument("--curve-file", type=Path, default=rdt.DEFAULT_CURVE_FILE)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--kp", type=float, default=0.03)
    parser.add_argument("--ki", type=float, default=0.003)
    parser.add_argument("--agc-interval", type=int, default=4)
    parser.add_argument("--duration-seconds", type=int, default=900)
    parser.add_argument("--traditional-governor-deadband-hz", type=float, default=0.036)
    parser.add_argument("--der-deadband-hz", type=float, default=0.036)
    parser.add_argument("--base-ddn", type=float, default=(1.0 / 3.0))
    parser.add_argument("--target-share", type=float, action="append", default=None,
                        help="Target storage pmax share as a fraction. Repeatable. Example: 0.022")
    return parser.parse_args()


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


def scale_storage_capacity(sa, factor: float) -> dict[str, object]:
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

    # ESD1 pmin is a ConstService derived from pmx; refresh it explicitly.
    if hasattr(sa.ESD1, "pmin") and esd_idx:
        sa.ESD1.pmin.v[:] = -np.asarray(sa.ESD1.pmx.v, dtype=float)

    after = storage_share(sa)
    return {
        "factor": float(factor),
        "after_pmax_share": after["pmax_share"],
        "after_sn_share": after["sn_share"],
    }


def configure_all_der_deadband(
    sa,
    *,
    traditional_governor_deadband_hz: float,
    der_deadband_hz: float,
    pvd1_ddn: float,
    esd1_ddn: float,
) -> dict[str, object]:
    meta: dict[str, object] = {}
    meta["der_deadband_disabled"] = rdt.disable_der_frequency_deadband(sa)
    meta["traditional_governor_deadband"] = rdt.apply_traditional_governor_deadband(
        sa,
        float(traditional_governor_deadband_hz),
    )

    if hasattr(sa, "PVD1") and sa.PVD1.n:
        idx = list(sa.PVD1.idx.v)
        n = len(idx)
        sa.PVD1.set(src="fdbd", idx=idx, attr="v", value=np.full(n, -float(der_deadband_hz)))
        sa.PVD1.set(src="fdbdu", idx=idx, attr="v", value=np.full(n, float(der_deadband_hz)))
        sa.PVD1.set(src="ddn", idx=idx, attr="v", value=np.full(n, float(pvd1_ddn)))

    if hasattr(sa, "ESD1") and sa.ESD1.n:
        idx = list(sa.ESD1.idx.v)
        n = len(idx)
        sa.ESD1.set(src="fdbd", idx=idx, attr="v", value=np.full(n, -float(der_deadband_hz)))
        sa.ESD1.set(src="fdbdu", idx=idx, attr="v", value=np.full(n, float(der_deadband_hz)))
        sa.ESD1.set(src="ddn", idx=idx, attr="v", value=np.full(n, float(esd1_ddn)))

    meta["pvd1_ddn"] = float(pvd1_ddn)
    meta["esd1_ddn"] = float(esd1_ddn)
    return meta


def run_segment_with_storage_trace(
    *,
    sa,
    ctx: dict[str, object],
    start_offset: int,
    duration_seconds: int,
    agc_interval: int,
    kp: float,
    ki: float,
    bf: np.ndarray,
    ace_integral: float,
    ace_raw: float,
    local_start: float = 0.0,
    include_initial: bool = True,
    dispatch_target_transition: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, float, float]:
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

    rows: list[dict[str, float]] = []

    def snapshot(ts: float) -> None:
        row = {
            "time_s": float(ts),
            "freq_dev_hz": float((sa.ACEc.f.v[0] - 1.0) * sa.config.freq),
            "esd1_pe_sum": 0.0,
        }
        if hasattr(sa, "ESD1") and sa.ESD1.n:
            pe = np.asarray(sa.ESD1.Ipout_y.v, dtype=float) * np.asarray(sa.ESD1.v.v, dtype=float)
            row["esd1_pe_sum"] = float(pe.sum())
            for idx, val in zip(sa.ESD1.idx.v, pe):
                row[f"esd1_{idx}_pe"] = float(val)
        rows.append(row)

    if include_initial:
        snapshot(local_start)

    current_tf = float(sa.dae.t)
    for step in range(1, duration_seconds):
        activate_dispatch_target_transition(sa, dispatch_target_transition, step)

        for col, has_col in (("agov", "has_gov"), ("adg", "has_dg"), ("arg", "has_rg")):
            link[col] = ace_raw * bf * link[has_col] * link["gammap"]

        if step % agc_interval == 0:
            agov_to_set = {gov: agov for gov, agov in zip(link["gov_idx"], link["agov"]) if pd.notna(gov)}
            if agov_to_set:
                gov_idx = list(agov_to_set.keys())
                paux0_raw = np.array(list(agov_to_set.values()))
                gov_syn = sa.TurbineGov.get(src="syn", attr="v", idx=gov_idx)
                gov_gen = sa.SynGen.get(src="gen", attr="v", idx=gov_syn)
                gov_pmax = sa.StaticGen.get(src="pmax", attr="v", idx=gov_gen)
                gov_pmin = sa.StaticGen.get(src="pmin", attr="v", idx=gov_gen)
                gov_pref0 = sa.TurbineGov.get(src="pref0", attr="v", idx=gov_idx)
                gov_up = np.maximum(0.0, gov_pmax - gov_pref0)
                gov_dn = np.minimum(0.0, gov_pmin - gov_pref0)
                paux0 = np.where(
                    paux0_raw >= 0.0,
                    np.minimum(paux0_raw, gov_up),
                    np.maximum(paux0_raw, gov_dn),
                )
                sa.TurbineGov.set(src="paux0", idx=gov_idx, attr="v", value=paux0)

            adg_to_set = {dg: adg for dg, adg in zip(link["dg_idx"], link["adg"]) if pd.notna(dg)}
            if adg_to_set:
                dg_idx = list(adg_to_set.keys())
                pext0_raw = np.array(list(adg_to_set.values()))
                dg_uids = sa.DG.idx2uid(dg_idx)
                pext0 = np.minimum(pext0_raw, pext_max[dg_uids])
                sa.DG.set(src="Pext0", idx=dg_idx, attr="v", value=pext0)

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

        snapshot(local_start + step)

        ace_sum = float(sa.ACEc.ace.v.sum())
        ace_raw = -(kp * ace_sum + ki * ace_integral)
        ace_integral = ace_integral + ace_sum

    return pd.DataFrame(rows), float(ace_integral), float(ace_raw)


def summarize_variant(label: str, df: pd.DataFrame, target_share: float, achieved_share: float, factor: float) -> dict[str, object]:
    f = df["freq_dev_hz"]
    a = f.abs()
    p = df["esd1_pe_sum"]
    return {
        "variant": label,
        "target_storage_share": float(target_share),
        "achieved_storage_share": float(achieved_share),
        "storage_scale_factor": float(factor),
        "samples": int(len(df)),
        "freq_min_hz": float(f.min()),
        "freq_max_hz": float(f.max()),
        "freq_abs_mean_hz": float(a.mean()),
        "freq_rms_hz": float(np.sqrt(np.mean(np.square(f)))),
        "share_abs_gt_0p036": float((a > 0.036).mean()),
        "share_abs_gt_0p05": float((a > 0.05).mean()),
        "esd1_pe_min": float(p.min()),
        "esd1_pe_max": float(p.max()),
        "esd1_pe_abs_mean": float(p.abs().mean()),
    }


def run_one_variant(
    *,
    checkpoint_in: Path,
    curve: pd.DataFrame,
    warmup_record: rdt.DispatchRecord,
    dispatch_record: rdt.DispatchRecord,
    next_dispatch_record: rdt.DispatchRecord,
    target_share: float,
    kp: float,
    ki: float,
    agc_interval: int,
    duration_seconds: int,
    traditional_governor_deadband_hz: float,
    der_deadband_hz: float,
    base_ddn: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    sa, stored_ctx, _agc_state, _manifest = hcp.load_checkpoint(checkpoint_in)
    ctx = hcp.build_runtime_context(sa=sa, curve=curve, stored_ctx=stored_ctx)

    share0 = storage_share(sa)
    current_share = float(share0["pmax_share"])
    factor = 1.0 if abs(target_share - current_share) < 1e-12 else solve_scale_factor(current_share, target_share)
    scale_meta = scale_storage_capacity(sa, factor) if abs(factor - 1.0) > 1e-12 else {
        "factor": 1.0,
        "after_pmax_share": current_share,
        "after_sn_share": float(share0["sn_share"]),
    }

    configure_meta = configure_all_der_deadband(
        sa,
        traditional_governor_deadband_hz=traditional_governor_deadband_hz,
        der_deadband_hz=der_deadband_hz,
        pvd1_ddn=base_ddn,
        esd1_ddn=base_ddn * factor,
    )

    ace_integral = 0.0
    ace_raw = -(kp * float(sa.ACEc.ace.v.sum()) + ki * ace_integral)

    warmup_transition = apply_second_dispatch_targets(
        sa,
        ctx["link"],  # type: ignore[arg-type]
        warmup_record,
        apply_governor_targets=True,
        apply_dg_targets=False,
        duration_seconds=duration_seconds,
        schedule_mode="midpoint_trajectory",
        next_dispatch_record=dispatch_record,
    )
    warmup_transition["ramp_seconds"] = 0
    activate_dispatch_target_transition(sa, warmup_transition, step=0)

    bf_warm = compute_bf(sa, warmup_record)
    _tw, _fw, ace_integral, ace_raw = run_segment(
        sa=sa,
        ctx=ctx,
        start_offset=dispatch_offset(warmup_record, duration_seconds),
        duration_seconds=duration_seconds,
        agc_interval=agc_interval,
        kp=kp,
        ki=ki,
        bf=bf_warm,
        ace_integral=ace_integral,
        ace_raw=ace_raw,
        local_start=0.0,
        include_initial=True,
        dispatch_target_transition=warmup_transition,
    )

    ctx["link"] = rdt.build_andes_link(sa)
    compare_transition = apply_second_dispatch_targets(
        sa,
        ctx["link"],  # type: ignore[arg-type]
        dispatch_record,
        apply_governor_targets=True,
        apply_dg_targets=False,
        duration_seconds=duration_seconds,
        schedule_mode="midpoint_trajectory",
        next_dispatch_record=next_dispatch_record,
    )
    compare_transition["ramp_seconds"] = 0
    activate_dispatch_target_transition(sa, compare_transition, step=0)

    bf_cmp = compute_bf(sa, dispatch_record)
    df, ace_integral_end, ace_raw_end = run_segment_with_storage_trace(
        sa=sa,
        ctx=ctx,
        start_offset=dispatch_offset(dispatch_record, duration_seconds),
        duration_seconds=duration_seconds,
        agc_interval=agc_interval,
        kp=kp,
        ki=ki,
        bf=bf_cmp,
        ace_integral=ace_integral,
        ace_raw=ace_raw,
        include_initial=True,
        dispatch_target_transition=compare_transition,
    )

    meta: dict[str, object] = {
        "target_storage_share": float(target_share),
        "current_storage_share": current_share,
        "storage_scale": scale_meta,
        "configure_meta": configure_meta,
        "ace_integral_end": float(ace_integral_end),
        "ace_raw_end": float(ace_raw_end),
    }
    return df, meta


def main() -> None:
    args = parse_args()
    rdt.andes.config_logger(stream_level=30)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    curve = rdt.load_curve(args.curve_file)
    warmup_record = rdt.DispatchRecord.from_json(args.warmup_dispatch_json)
    dispatch_record = rdt.DispatchRecord.from_json(args.dispatch_json)
    next_dispatch_record = rdt.DispatchRecord.from_json(args.next_dispatch_json)

    # Current case share is about 0.8637%.
    targets = args.target_share or [0.008636772895209975, 0.022, 0.05]

    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    config: dict[str, object] = {
        "checkpoint_in": str(args.checkpoint_in),
        "warmup_dispatch_json": str(args.warmup_dispatch_json),
        "dispatch_json": str(args.dispatch_json),
        "next_dispatch_json": str(args.next_dispatch_json),
        "curve_file": str(args.curve_file),
        "kp": float(args.kp),
        "ki": float(args.ki),
        "agc_interval": int(args.agc_interval),
        "duration_seconds": int(args.duration_seconds),
        "traditional_governor_deadband_hz": float(args.traditional_governor_deadband_hz),
        "der_deadband_hz": float(args.der_deadband_hz),
        "base_ddn": float(args.base_ddn),
        "variants": {},
    }

    for target in targets:
        label = f"share_{target * 100:.3f}pct".replace(".", "p")
        df, meta = run_one_variant(
            checkpoint_in=args.checkpoint_in,
            curve=curve,
            warmup_record=warmup_record,
            dispatch_record=dispatch_record,
            next_dispatch_record=next_dispatch_record,
            target_share=float(target),
            kp=float(args.kp),
            ki=float(args.ki),
            agc_interval=int(args.agc_interval),
            duration_seconds=int(args.duration_seconds),
            traditional_governor_deadband_hz=float(args.traditional_governor_deadband_hz),
            der_deadband_hz=float(args.der_deadband_hz),
            base_ddn=float(args.base_ddn),
        )
        df.insert(0, "variant", label)
        frames.append(df)
        summaries.append(
            summarize_variant(
                label,
                df,
                target_share=float(target),
                achieved_share=float(meta["storage_scale"]["after_pmax_share"]),  # type: ignore[index]
                factor=float(meta["storage_scale"]["factor"]),  # type: ignore[index]
            )
        )
        config["variants"][label] = meta
        df.to_csv(args.results_dir / f"{dispatch_record.label}_{label}.csv", index=False)

    combined = pd.concat(frames, ignore_index=True)
    combined_csv = args.results_dir / f"{dispatch_record.label}_storage_share_compare_all.csv"
    summary_csv = args.results_dir / f"{dispatch_record.label}_storage_share_compare_summary.csv"
    freq_png = args.results_dir / f"{dispatch_record.label}_storage_share_frequency_compare.png"
    pe_png = args.results_dir / f"{dispatch_record.label}_storage_share_output_compare.png"
    config_json = args.results_dir / f"{dispatch_record.label}_storage_share_compare_config.json"

    combined.to_csv(combined_csv, index=False)
    pd.DataFrame(summaries).to_csv(summary_csv, index=False)
    config_json.write_text(json.dumps(config, indent=2))

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    labels = list(combined["variant"].drop_duplicates())
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(labels)}

    plt.figure(figsize=(14.8, 5.6))
    for label, df in combined.groupby("variant"):
        plt.plot(df["time_s"], df["freq_dev_hz"], label=label, color=color_map[label], linewidth=2.0)
    plt.axhline(0.036, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
    plt.axhline(-0.036, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency deviation (Hz)")
    plt.title(f"{dispatch_record.label}: frequency under scaled storage share")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(freq_png, dpi=180)
    plt.close()

    plt.figure(figsize=(14.8, 5.6))
    for label, df in combined.groupby("variant"):
        plt.plot(df["time_s"], df["esd1_pe_sum"], label=label, color=color_map[label], linewidth=2.0)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("ESD1 active power output (pu on system base)")
    plt.title(f"{dispatch_record.label}: storage output under scaled storage share")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(pe_png, dpi=180)
    plt.close()

    print(f"summary_csv={summary_csv}")
    print(f"freq_png={freq_png}")
    print(f"pe_png={pe_png}")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
