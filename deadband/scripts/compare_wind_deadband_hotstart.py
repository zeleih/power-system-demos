#!/usr/bin/env python3
"""
Compare one hot-start chain with wind PVD1 deadband disabled/enabled.

This isolates the wind-frequency-response deadband effect while keeping:

- the same base checkpoint,
- the same AGC gains,
- the same governor target trajectory,
- traditional governor deadband unchanged,
- PV / storage DER deadband disabled in both variants.
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
    parser.add_argument("--wind-deadband-hz", type=float, default=0.036)
    parser.add_argument("--wind-ddn", type=float, default=(1.0 / 3.0))
    return parser.parse_args()


def configure_variant(
    sa,
    *,
    traditional_governor_deadband_hz: float,
    wind_deadband_on: bool,
    wind_deadband_hz: float,
    wind_ddn: float,
) -> dict[str, object]:
    meta: dict[str, object] = {}
    meta["der_deadband_disabled"] = rdt.disable_der_frequency_deadband(sa)
    meta["traditional_governor_deadband"] = rdt.apply_traditional_governor_deadband(
        sa,
        float(traditional_governor_deadband_hz),
    )

    stg_w2t, stg_pv = rdt.pvd1_gen_subsets(sa, rdt.DEFAULT_WIND_PREFIXES, rdt.DEFAULT_SOLAR_PREFIXES)
    pvd1_w2t = sa.PVD1.find_idx(keys="gen", values=stg_w2t)
    pvd1_pv = sa.PVD1.find_idx(keys="gen", values=stg_pv)

    if wind_deadband_on:
        n = len(pvd1_w2t)
        sa.PVD1.set(src="fdbd", idx=pvd1_w2t, attr="v", value=np.full(n, -float(wind_deadband_hz)))
        sa.PVD1.set(src="fdbdu", idx=pvd1_w2t, attr="v", value=np.full(n, float(wind_deadband_hz)))
        sa.PVD1.set(src="ddn", idx=pvd1_w2t, attr="v", value=np.full(n, float(wind_ddn)))

    meta["wind_deadband"] = {
        "enabled": int(wind_deadband_on),
        "wind_pvd1_count": int(len(pvd1_w2t)),
        "pv_pvd1_count": int(len(pvd1_pv)),
        "wind_deadband_hz": float(wind_deadband_hz),
        "wind_ddn": float(wind_ddn),
        "wind_pvd1_idx_sample": list(map(str, pvd1_w2t[:5])),
    }
    return meta


def run_segment_trace(
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
    wind_pvd1_idx: list[str],
    local_start: float = 0.0,
    include_initial: bool = True,
    dispatch_target_transition: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
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

    wind_uid = sa.PVD1.idx2uid(wind_pvd1_idx)

    local_t: list[float] = []
    freq: list[float] = []
    wind_db: list[float] = []

    def snap(ts: float) -> None:
        local_t.append(float(ts))
        freq.append(float((sa.ACEc.f.v[0] - 1.0) * sa.config.freq))
        db_vec = np.asarray(sa.PVD1.DB_y.v, dtype=float)
        wind_db.append(float(db_vec[wind_uid].sum()) if len(wind_uid) else 0.0)

    if include_initial:
        snap(local_start)

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

        snap(local_start + step)

        ace_sum = float(sa.ACEc.ace.v.sum())
        ace_raw = -(kp * ace_sum + ki * ace_integral)
        ace_integral = ace_integral + ace_sum

    return (
        np.asarray(local_t, dtype=float),
        np.asarray(freq, dtype=float),
        np.asarray(wind_db, dtype=float),
        float(ace_integral),
        float(ace_raw),
    )


def summarize(label: str, t: np.ndarray, freq: np.ndarray, wind_db: np.ndarray, deadband_hz: float) -> dict[str, object]:
    abs_f = np.abs(freq)
    return {
        "variant": label,
        "samples": int(len(t)),
        "min_hz": float(freq.min()),
        "max_hz": float(freq.max()),
        "final_hz": float(freq[-1]),
        "abs_mean_hz": float(abs_f.mean()),
        "rms_hz": float(np.sqrt(np.mean(np.square(freq)))),
        "share_abs_gt_deadband": float(np.mean(abs_f > float(deadband_hz))),
        "share_abs_gt_0p05": float(np.mean(abs_f > 0.05)),
        "wind_db_abs_max": float(np.abs(wind_db).max()),
        "wind_db_nonzero_samples": int((np.abs(wind_db) > 1e-9).sum()),
    }


def run_variant(
    *,
    checkpoint_in: Path,
    curve: pd.DataFrame,
    warmup_record: rdt.DispatchRecord,
    dispatch_record: rdt.DispatchRecord,
    next_dispatch_record: rdt.DispatchRecord,
    kp: float,
    ki: float,
    agc_interval: int,
    duration_seconds: int,
    wind_deadband_on: bool,
    wind_deadband_hz: float,
    wind_ddn: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    sa, stored_ctx, _agc_state, _manifest = hcp.load_checkpoint(checkpoint_in)
    ctx = hcp.build_runtime_context(sa=sa, curve=curve, stored_ctx=stored_ctx)

    meta = configure_variant(
        sa,
        traditional_governor_deadband_hz=float(args.traditional_governor_deadband_hz),
        wind_deadband_on=wind_deadband_on,
        wind_deadband_hz=wind_deadband_hz,
        wind_ddn=wind_ddn,
    )

    ace_integral = 0.0
    ace_sum0 = float(sa.ACEc.ace.v.sum())
    ace_raw = -(kp * ace_sum0 + ki * ace_integral)

    stg_w2t, _stg_pv = rdt.pvd1_gen_subsets(sa, rdt.DEFAULT_WIND_PREFIXES, rdt.DEFAULT_SOLAR_PREFIXES)
    wind_pvd1_idx = list(sa.PVD1.find_idx(keys="gen", values=stg_w2t))

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
    _tw, _fw, _dbw, ace_integral, ace_raw = run_segment_trace(
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
        wind_pvd1_idx=wind_pvd1_idx,
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
    t, freq, wind_db, ace_integral_end, ace_raw_end = run_segment_trace(
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
        wind_pvd1_idx=wind_pvd1_idx,
        include_initial=True,
        dispatch_target_transition=compare_transition,
    )

    frame = pd.DataFrame({
        "time_s": t,
        "freq_dev_hz": freq,
        "wind_db_sum": wind_db,
    })
    meta["ace_integral_end"] = float(ace_integral_end)
    meta["ace_raw_end"] = float(ace_raw_end)
    return frame, meta


def main() -> None:
    args = parse_args()
    rdt.andes.config_logger(stream_level=30)

    args.results_dir.mkdir(parents=True, exist_ok=True)

    curve = rdt.load_curve(args.curve_file)
    warmup_record = rdt.DispatchRecord.from_json(args.warmup_dispatch_json)
    dispatch_record = rdt.DispatchRecord.from_json(args.dispatch_json)
    next_dispatch_record = rdt.DispatchRecord.from_json(args.next_dispatch_json)

    variants = [
        ("wind_deadband_off", False),
        ("wind_deadband_on", True),
    ]

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
        "wind_deadband_hz": float(args.wind_deadband_hz),
        "wind_ddn": float(args.wind_ddn),
        "variants": {},
    }

    for label, enabled in variants:
        frame, meta = run_variant(
            checkpoint_in=args.checkpoint_in,
            curve=curve,
            warmup_record=warmup_record,
            dispatch_record=dispatch_record,
            next_dispatch_record=next_dispatch_record,
            kp=float(args.kp),
            ki=float(args.ki),
            agc_interval=int(args.agc_interval),
            duration_seconds=int(args.duration_seconds),
            wind_deadband_on=enabled,
            wind_deadband_hz=float(args.wind_deadband_hz),
            wind_ddn=float(args.wind_ddn),
        )
        frame.insert(0, "variant", label)
        frames.append(frame)
        summaries.append(summarize(label, frame["time_s"].to_numpy(), frame["freq_dev_hz"].to_numpy(), frame["wind_db_sum"].to_numpy(), args.wind_deadband_hz))
        config["variants"][label] = meta

        csv_path = args.results_dir / f"{dispatch_record.label}_{label}.csv"
        frame.to_csv(csv_path, index=False)

    combined = pd.concat(frames, ignore_index=True)
    combined_csv = args.results_dir / f"{dispatch_record.label}_wind_deadband_compare_all.csv"
    summary_csv = args.results_dir / f"{dispatch_record.label}_wind_deadband_compare_summary.csv"
    plot_png = args.results_dir / f"{dispatch_record.label}_wind_deadband_compare.png"
    config_json = args.results_dir / f"{dispatch_record.label}_wind_deadband_compare_config.json"

    combined.to_csv(combined_csv, index=False)
    pd.DataFrame(summaries).to_csv(summary_csv, index=False)
    config_json.write_text(json.dumps(config, indent=2))

    colors = {
        "wind_deadband_off": "#1f77b4",
        "wind_deadband_on": "#d62728",
    }
    labels = {
        "wind_deadband_off": "Wind deadband off",
        "wind_deadband_on": "Wind deadband on",
    }

    fig, axes = plt.subplots(2, 1, figsize=(14.5, 8.5), sharex=True)
    for variant, df in combined.groupby("variant"):
        axes[0].plot(df["time_s"], df["freq_dev_hz"], label=labels.get(variant, variant), color=colors.get(variant), linewidth=2.0)
        axes[1].plot(df["time_s"], df["wind_db_sum"], label=labels.get(variant, variant), color=colors.get(variant), linewidth=1.8)

    axes[0].axhline(args.wind_deadband_hz, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[0].axhline(-args.wind_deadband_hz, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[0].set_ylabel("Freq dev [Hz]")
    axes[0].set_title(f"{dispatch_record.label}: wind PVD1 deadband on/off, KP={args.kp}, KI={args.ki}")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[1].set_ylabel("Wind PVD1 DB_y sum")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(plot_png, dpi=200)
    plt.close(fig)

    print(f"plot_png={plot_png}")
    print(f"summary_csv={summary_csv}")
    print(f"combined_csv={combined_csv}")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
