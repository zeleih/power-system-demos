#!/usr/bin/env python3
"""
Compare one hot-start chain with one PVD1 subset deadband disabled/enabled.

Supported subsets:
- wind: PVD1 units classified by wind prefixes
- pv: PVD1 units classified by solar prefixes
- both: both wind and PV PVD1 units
- storage: all ESD1 units
- all: wind + PV + storage

All other DER deadband terms are disabled in both variants so the chosen subset
is the only renewable deadband channel being toggled.
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


def subset_label(subset: str) -> str:
    return {
        "wind": "Wind",
        "pv": "PV",
        "both": "Wind+PV",
        "storage": "Storage",
        "all": "Wind+PV+Storage",
    }.get(subset, subset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-in", type=Path, required=True)
    parser.add_argument("--warmup-dispatch-json", type=Path, required=True)
    parser.add_argument("--dispatch-json", type=Path, required=True)
    parser.add_argument("--next-dispatch-json", type=Path, required=True)
    parser.add_argument("--curve-file", type=Path, default=rdt.DEFAULT_CURVE_FILE)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--subset", choices=("wind", "pv", "both", "storage", "all"), required=True)
    parser.add_argument("--kp", type=float, default=0.03)
    parser.add_argument("--ki", type=float, default=0.003)
    parser.add_argument("--agc-interval", type=int, default=4)
    parser.add_argument("--duration-seconds", type=int, default=900)
    parser.add_argument("--traditional-governor-deadband-hz", type=float, default=0.036)
    parser.add_argument("--subset-deadband-hz", type=float, default=0.036)
    parser.add_argument("--subset-ddn", type=float, default=(1.0 / 3.0))
    return parser.parse_args()


def get_subset_indices(sa, subset: str) -> tuple[list[str], list[str], int]:
    stg_w2t, stg_pv = rdt.pvd1_gen_subsets(sa, rdt.DEFAULT_WIND_PREFIXES, rdt.DEFAULT_SOLAR_PREFIXES)
    if subset == "wind":
        pvd1_idx = list(sa.PVD1.find_idx(keys="gen", values=stg_w2t))
        esd1_idx: list[str] = []
        count = len(pvd1_idx)
    elif subset == "pv":
        pvd1_idx = list(sa.PVD1.find_idx(keys="gen", values=stg_pv))
        esd1_idx = []
        count = len(pvd1_idx)
    elif subset == "both":
        wind_idx = list(sa.PVD1.find_idx(keys="gen", values=stg_w2t))
        pv_idx = list(sa.PVD1.find_idx(keys="gen", values=stg_pv))
        pvd1_idx = list(dict.fromkeys([*wind_idx, *pv_idx]))
        esd1_idx = []
        count = len(pvd1_idx)
    elif subset == "storage":
        pvd1_idx = []
        esd1_idx = list(sa.ESD1.idx.v) if hasattr(sa, "ESD1") and sa.ESD1.n else []
        count = len(esd1_idx)
    else:
        wind_idx = list(sa.PVD1.find_idx(keys="gen", values=stg_w2t))
        pv_idx = list(sa.PVD1.find_idx(keys="gen", values=stg_pv))
        pvd1_idx = list(dict.fromkeys([*wind_idx, *pv_idx]))
        esd1_idx = list(sa.ESD1.idx.v) if hasattr(sa, "ESD1") and sa.ESD1.n else []
        count = len(pvd1_idx) + len(esd1_idx)
    return pvd1_idx, esd1_idx, count


def configure_variant(
    sa,
    *,
    subset: str,
    traditional_governor_deadband_hz: float,
    subset_deadband_on: bool,
    subset_deadband_hz: float,
    subset_ddn: float,
) -> dict[str, object]:
    meta: dict[str, object] = {}
    meta["der_deadband_disabled"] = rdt.disable_der_frequency_deadband(sa)
    meta["traditional_governor_deadband"] = rdt.apply_traditional_governor_deadband(
        sa,
        float(traditional_governor_deadband_hz),
    )

    pvd1_idx, esd1_idx, subset_count = get_subset_indices(sa, subset)
    if subset_deadband_on and pvd1_idx:
        n = len(pvd1_idx)
        sa.PVD1.set(src="fdbd", idx=pvd1_idx, attr="v", value=np.full(n, -float(subset_deadband_hz)))
        sa.PVD1.set(src="fdbdu", idx=pvd1_idx, attr="v", value=np.full(n, float(subset_deadband_hz)))
        sa.PVD1.set(src="ddn", idx=pvd1_idx, attr="v", value=np.full(n, float(subset_ddn)))
    if subset_deadband_on and esd1_idx:
        n = len(esd1_idx)
        sa.ESD1.set(src="fdbd", idx=esd1_idx, attr="v", value=np.full(n, -float(subset_deadband_hz)))
        sa.ESD1.set(src="fdbdu", idx=esd1_idx, attr="v", value=np.full(n, float(subset_deadband_hz)))
        sa.ESD1.set(src="ddn", idx=esd1_idx, attr="v", value=np.full(n, float(subset_ddn)))

    meta["subset_deadband"] = {
        "subset": subset,
        "enabled": int(subset_deadband_on),
        "subset_device_count": int(subset_count),
        "subset_deadband_hz": float(subset_deadband_hz),
        "subset_ddn": float(subset_ddn),
        "subset_pvd1_idx_sample": list(map(str, pvd1_idx[:5])),
        "subset_esd1_idx_sample": list(map(str, esd1_idx[:5])),
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
    subset_pvd1_idx: list[str],
    subset_esd1_idx: list[str],
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

    subset_pvd1_uid = sa.PVD1.idx2uid(subset_pvd1_idx) if subset_pvd1_idx else np.asarray([], dtype=int)
    subset_esd1_uid = sa.ESD1.idx2uid(subset_esd1_idx) if subset_esd1_idx else np.asarray([], dtype=int)

    local_t: list[float] = []
    freq: list[float] = []
    subset_db: list[float] = []

    def snap(ts: float) -> None:
        local_t.append(float(ts))
        freq.append(float((sa.ACEc.f.v[0] - 1.0) * sa.config.freq))
        db_total = 0.0
        if len(subset_pvd1_uid):
            db_total += float(np.asarray(sa.PVD1.DB_y.v, dtype=float)[subset_pvd1_uid].sum())
        if len(subset_esd1_uid):
            db_total += float(np.asarray(sa.ESD1.DB_y.v, dtype=float)[subset_esd1_uid].sum())
        subset_db.append(db_total)

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
        np.asarray(subset_db, dtype=float),
        float(ace_integral),
        float(ace_raw),
    )


def summarize(
    label: str,
    t: np.ndarray,
    freq: np.ndarray,
    subset_db: np.ndarray,
    deadband_hz: float,
    subset_label: str,
) -> dict[str, object]:
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
        f"{subset_label}_db_abs_max": float(np.abs(subset_db).max()),
        f"{subset_label}_db_nonzero_samples": int((np.abs(subset_db) > 1e-9).sum()),
    }


def run_variant(
    *,
    checkpoint_in: Path,
    curve: pd.DataFrame,
    warmup_record: rdt.DispatchRecord,
    dispatch_record: rdt.DispatchRecord,
    next_dispatch_record: rdt.DispatchRecord,
    subset: str,
    kp: float,
    ki: float,
    agc_interval: int,
    duration_seconds: int,
    traditional_governor_deadband_hz: float,
    subset_deadband_on: bool,
    subset_deadband_hz: float,
    subset_ddn: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    sa, stored_ctx, _agc_state, _manifest = hcp.load_checkpoint(checkpoint_in)
    ctx = hcp.build_runtime_context(sa=sa, curve=curve, stored_ctx=stored_ctx)

    meta = configure_variant(
        sa,
        subset=subset,
        traditional_governor_deadband_hz=float(traditional_governor_deadband_hz),
        subset_deadband_on=subset_deadband_on,
        subset_deadband_hz=subset_deadband_hz,
        subset_ddn=subset_ddn,
    )

    ace_integral = 0.0
    ace_sum0 = float(sa.ACEc.ace.v.sum())
    ace_raw = -(kp * ace_sum0 + ki * ace_integral)

    subset_pvd1_idx, subset_esd1_idx, _ = get_subset_indices(sa, subset)

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
        subset_pvd1_idx=subset_pvd1_idx,
        subset_esd1_idx=subset_esd1_idx,
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
    t, freq, subset_db, ace_integral_end, ace_raw_end = run_segment_trace(
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
        subset_pvd1_idx=subset_pvd1_idx,
        subset_esd1_idx=subset_esd1_idx,
        include_initial=True,
        dispatch_target_transition=compare_transition,
    )

    frame = pd.DataFrame({
        "time_s": t,
        "freq_dev_hz": freq,
        f"{subset}_db_sum": subset_db,
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
        (f"{args.subset}_deadband_off", False),
        (f"{args.subset}_deadband_on", True),
    ]

    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    config: dict[str, object] = {
        "checkpoint_in": str(args.checkpoint_in),
        "warmup_dispatch_json": str(args.warmup_dispatch_json),
        "dispatch_json": str(args.dispatch_json),
        "next_dispatch_json": str(args.next_dispatch_json),
        "curve_file": str(args.curve_file),
        "subset": args.subset,
        "kp": float(args.kp),
        "ki": float(args.ki),
        "agc_interval": int(args.agc_interval),
        "duration_seconds": int(args.duration_seconds),
        "traditional_governor_deadband_hz": float(args.traditional_governor_deadband_hz),
        "subset_deadband_hz": float(args.subset_deadband_hz),
        "subset_ddn": float(args.subset_ddn),
        "variants": {},
    }

    for label, enabled in variants:
        frame, meta = run_variant(
            checkpoint_in=args.checkpoint_in,
            curve=curve,
            warmup_record=warmup_record,
            dispatch_record=dispatch_record,
            next_dispatch_record=next_dispatch_record,
            subset=args.subset,
            kp=float(args.kp),
            ki=float(args.ki),
            agc_interval=int(args.agc_interval),
            duration_seconds=int(args.duration_seconds),
            traditional_governor_deadband_hz=float(args.traditional_governor_deadband_hz),
            subset_deadband_on=enabled,
            subset_deadband_hz=float(args.subset_deadband_hz),
            subset_ddn=float(args.subset_ddn),
        )
        frame.insert(0, "variant", label)
        frames.append(frame)
        summaries.append(
            summarize(
                label,
                frame["time_s"].to_numpy(),
                frame["freq_dev_hz"].to_numpy(),
                frame[f"{args.subset}_db_sum"].to_numpy(),
                float(args.subset_deadband_hz),
                args.subset,
            )
        )
        config["variants"][label] = meta
        csv_path = args.results_dir / f"{dispatch_record.label}_{label}.csv"
        frame.to_csv(csv_path, index=False)

    combined = pd.concat(frames, ignore_index=True)
    combined_csv = args.results_dir / f"{dispatch_record.label}_{args.subset}_deadband_compare_all.csv"
    summary_csv = args.results_dir / f"{dispatch_record.label}_{args.subset}_deadband_compare_summary.csv"
    plot_png = args.results_dir / f"{dispatch_record.label}_{args.subset}_deadband_compare.png"
    config_json = args.results_dir / f"{dispatch_record.label}_{args.subset}_deadband_compare_config.json"

    combined.to_csv(combined_csv, index=False)
    pd.DataFrame(summaries).to_csv(summary_csv, index=False)
    config_json.write_text(json.dumps(config, indent=2))

    colors = {
        f"{args.subset}_deadband_off": "#1f77b4",
        f"{args.subset}_deadband_on": "#d62728",
    }
    labels = {
        f"{args.subset}_deadband_off": f"{subset_label(args.subset)} deadband off",
        f"{args.subset}_deadband_on": f"{subset_label(args.subset)} deadband on",
    }

    fig, axes = plt.subplots(2, 1, figsize=(14.5, 8.5), sharex=True)
    for variant, df in combined.groupby("variant"):
        axes[0].plot(df["time_s"], df["freq_dev_hz"], label=labels.get(variant, variant), color=colors.get(variant), linewidth=2.0)
        axes[1].plot(df["time_s"], df[f"{args.subset}_db_sum"], label=labels.get(variant, variant), color=colors.get(variant), linewidth=1.8)

    axes[0].axhline(args.subset_deadband_hz, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[0].axhline(-args.subset_deadband_hz, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[0].set_ylabel("Freq dev [Hz]")
    axes[0].set_title(f"{dispatch_record.label}: {subset_label(args.subset)} DER deadband on/off, KP={args.kp}, KI={args.ki}")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[1].set_ylabel(f"{subset_label(args.subset)} DER DB_y sum")
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
