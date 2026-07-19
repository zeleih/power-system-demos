#!/usr/bin/env python3
"""
Run full-day validation for top phase-1 deadband candidates.

This consumes the ranked coarse-sweep candidates, launches 96-dispatch
hot-start runs for the top K eligible combinations, renders standard plots,
compares each candidate against the chosen baseline, and writes a short
recommendation memo.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DAY_SCRIPT = SCRIPT_DIR / "run_day_dispatch_hotstart.py"
PLOT_DAY_SCRIPT = SCRIPT_DIR / "plot_day_hotstart_results.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--baseline-results-dir", type=Path, required=True)
    parser.add_argument("--dispatch-dir", type=Path, required=True)
    parser.add_argument("--dyn-case", type=Path, required=True)
    parser.add_argument("--stable-dyn-case", type=Path, required=True)
    parser.add_argument("--curve-file", type=Path, required=True)
    parser.add_argument("--plot-python", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--dispatch-interval", type=int, default=900)
    parser.add_argument("--dispatches-per-hour", type=int, default=4)
    parser.add_argument("--agc-interval", type=int, default=4)
    parser.add_argument("--kp", type=float, default=0.1)
    parser.add_argument("--ki", type=float, default=0.002)
    parser.add_argument("--wind-pref-alpha", type=float, default=0.98)
    parser.add_argument("--solar-pref-alpha", type=float, default=0.98)
    parser.add_argument(
        "--agc-allocation-mode",
        choices=("fixed_capacity", "headroom_aware"),
        default="headroom_aware",
    )
    parser.add_argument(
        "--agc-anti-windup-mode",
        choices=("off", "freeze_on_saturation"),
        default="freeze_on_saturation",
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
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Ignore existing phase1_full_day_summary.csv and recompute all candidates.",
    )
    return parser.parse_args()


def load_day_samples(results_dir: Path) -> np.ndarray:
    summary = pd.read_csv(results_dir / "daily_hotstart_summary.csv")
    blocks = []
    for _, row in summary.iterrows():
        csv_path = Path(str(row["freq_csv"]))
        if not csv_path.exists():
            continue
        freq = pd.read_csv(csv_path)["freq_dev_hz"].to_numpy(dtype=float)
        if freq.size:
            blocks.append(freq)
    if not blocks:
        raise RuntimeError(f"No frequency samples found in {results_dir}")
    return np.concatenate(blocks)


DAY_COST_COLUMNS = (
    "wind_effort",
    "pv_effort",
    "pvd_effort",
    "esd_throughput",
    "gov_droop_effort",
)


def load_day_cost_metrics(results_dir: Path) -> dict[str, float]:
    summary = pd.read_csv(results_dir / "daily_hotstart_summary.csv")
    metrics: dict[str, float] = {}
    for col in DAY_COST_COLUMNS:
        if col in summary.columns:
            metrics[col] = float(pd.to_numeric(summary[col], errors="coerce").fillna(0.0).sum())
        else:
            metrics[col] = float("nan")
    return metrics


def compute_edge_metrics(samples: np.ndarray) -> dict[str, float]:
    abs_samples = np.abs(samples)
    pos_edge = np.mean((samples >= 0.032) & (samples <= 0.040))
    neg_edge = np.mean((samples <= -0.032) & (samples >= -0.040))
    return {
        "edge_mass_36": float(np.mean((abs_samples >= 0.032) & (abs_samples <= 0.040))),
        "edge_asymmetry_36": float(abs(pos_edge - neg_edge)),
        "edge_pos_36": float(pos_edge),
        "edge_neg_36": float(neg_edge),
    }


def load_baseline_metrics(results_dir: Path) -> dict[str, float]:
    stats = pd.read_csv(results_dir / "frequency_distribution_stats.csv").iloc[0].to_dict()
    samples = load_day_samples(results_dir)
    stats.update(compute_edge_metrics(samples))
    return {key: float(value) for key, value in stats.items() if isinstance(value, (int, float, np.floating))}


def candidate_outputs_ready(results_dir: Path) -> bool:
    required = [
        results_dir / "daily_hotstart_summary.csv",
        results_dir / "frequency_distribution_stats.csv",
        results_dir / "frequency_distribution.png",
        results_dir / "frequency_curves_all_96.png",
    ]
    if not all(path.exists() for path in required):
        return False
    summary = pd.read_csv(results_dir / "daily_hotstart_summary.csv", nrows=1)
    return all(col in summary.columns for col in DAY_COST_COLUMNS)


def run_candidate(args: argparse.Namespace, row: pd.Series) -> Path:
    combo_id = str(row["combo_id"])
    out_dir = args.results_root / combo_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if candidate_outputs_ready(out_dir):
        print(f"  reusing existing outputs for {combo_id}", flush=True)
        return out_dir

    cmd = [
        sys.executable,
        str(RUN_DAY_SCRIPT),
        "--dispatch-dir", str(args.dispatch_dir),
        "--results-dir", str(out_dir),
        "--dyn-case", str(args.dyn_case),
        "--stable-dyn-case", str(args.stable_dyn_case),
        "--curve-file", str(args.curve_file),
        "--dispatch-interval", str(args.dispatch_interval),
        "--dispatches-per-hour", str(args.dispatches_per_hour),
        "--agc-interval", str(args.agc_interval),
        "--kp", str(args.kp),
        "--ki", str(args.ki),
        "--wind-pref-alpha", str(args.wind_pref_alpha),
        "--solar-pref-alpha", str(args.solar_pref_alpha),
        "--agc-allocation-mode", args.agc_allocation_mode,
        "--agc-anti-windup-mode", args.agc_anti_windup_mode,
        "--agc-gov-output-ramp-frac-pmax-per-min", str(args.agc_gov_output_ramp_frac_pmax_per_min),
        "--agc-dg-output-ramp-frac-pmax-per-min", str(args.agc_dg_output_ramp_frac_pmax_per_min),
        "--init-mode", args.init_mode,
        "--governor-target-schedule", args.governor_target_schedule,
        "--governor-basepoint-ramp-floor-frac-pmax-per-min", str(args.governor_basepoint_ramp_floor_frac_pmax_per_min),
        "--governor-basepoint-ramp-gap-factor", str(args.governor_basepoint_ramp_gap_factor),
        "--apply-governor-targets",
        "--wind-deadband-hz", str(float(row["wind_deadband_hz"])),
        "--solar-deadband-hz", str(float(row["solar_deadband_hz"])),
        "--esd-deadband-hz", str(float(row["esd_deadband_hz"])),
        "--no-save-plot",
    ]
    if args.disable_der_agc:
        cmd.append("--disable-der-agc")
    if args.disable_pvd_agc:
        cmd.append("--disable-pvd-agc")
    if args.disable_esd_agc:
        cmd.append("--disable-esd-agc")

    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"full-day run failed for {combo_id}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    plot_cmd = [
        str(args.plot_python) if args.plot_python else sys.executable,
        str(PLOT_DAY_SCRIPT),
        "--results-dir", str(out_dir),
    ]
    plotted = subprocess.run(plot_cmd, capture_output=True, text=True)
    if plotted.returncode != 0:
        raise RuntimeError(
            f"plotting failed for {combo_id}\nstdout:\n{plotted.stdout}\nstderr:\n{plotted.stderr}"
        )

    return out_dir


def summarize_candidate(candidate: pd.Series, baseline: dict[str, float], result_dir: Path) -> dict[str, object]:
    stats = pd.read_csv(result_dir / "frequency_distribution_stats.csv").iloc[0].to_dict()
    samples = load_day_samples(result_dir)
    day_cost = load_day_cost_metrics(result_dir)
    edge = compute_edge_metrics(samples)
    row = candidate.to_dict()
    row.update({key: float(value) for key, value in stats.items() if isinstance(value, (int, float, np.floating))})
    row.update(edge)
    row.update(day_cost)
    row["result_dir"] = str(result_dir)
    row["delta_share_abs_gt_0p05"] = float(row["share_abs_gt_0p05"] - baseline["share_abs_gt_0p05"])
    row["delta_max_abs_hz"] = float(row["max_abs_hz"] - baseline["max_abs_hz"])
    row["delta_edge_mass_36"] = float(row["edge_mass_36"] - baseline["edge_mass_36"])
    row["delta_edge_asymmetry_36"] = float(row["edge_asymmetry_36"] - baseline["edge_asymmetry_36"])
    row["accepted"] = int(
        row["max_abs_hz"] <= baseline["max_abs_hz"]
        and row["share_abs_gt_0p05"] <= baseline["share_abs_gt_0p05"]
        and row["edge_mass_36"] < baseline["edge_mass_36"]
    )
    return row


def rank_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    ranked = summary.sort_values(
        by=[
            "accepted",
            "share_abs_gt_0p05",
            "max_abs_hz",
            "edge_mass_36",
            "edge_asymmetry_36",
            "esd_throughput",
            "wind_effort",
            "pv_effort",
            "gov_droop_effort",
        ],
        ascending=[False, True, True, True, True, True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def load_resume_state(results_root: Path, *, force_restart: bool) -> tuple[list[dict[str, object]], set[str]]:
    if force_restart:
        return [], set()

    summary_path = results_root / "phase1_full_day_summary.csv"
    if not summary_path.exists():
        return [], set()

    summary = pd.read_csv(summary_path)
    rows = summary.to_dict(orient="records")
    completed = {str(row["combo_id"]) for row in rows}
    return rows, completed


def write_outputs(
    *,
    args: argparse.Namespace,
    baseline: dict[str, float],
    rows: list[dict[str, object]],
    candidate_count: int,
) -> None:
    summary = pd.DataFrame(rows)
    summary.to_csv(args.results_root / "phase1_full_day_summary.csv", index=False)

    ranked = rank_summary(summary)
    ranked.to_csv(args.results_root / "phase1_full_day_ranked.csv", index=False)
    ranked.head(int(args.top_k)).to_csv(args.results_root / "phase1_full_day_top_candidates.csv", index=False)

    manifest = {
        "candidate_csv": str(args.candidate_csv.resolve()),
        "baseline_results_dir": str(args.baseline_results_dir.resolve()),
        "dispatch_dir": str(args.dispatch_dir.resolve()),
        "dyn_case": str(args.dyn_case.resolve()),
        "stable_dyn_case": str(args.stable_dyn_case.resolve()),
        "curve_file": str(args.curve_file.resolve()),
        "plot_python": str(args.plot_python.resolve()) if args.plot_python else "",
        "top_k": int(args.top_k),
        "kp": float(args.kp),
        "ki": float(args.ki),
        "wind_pref_alpha": float(args.wind_pref_alpha),
        "solar_pref_alpha": float(args.solar_pref_alpha),
        "agc_allocation_mode": str(args.agc_allocation_mode),
        "agc_anti_windup_mode": str(args.agc_anti_windup_mode),
        "disable_der_agc": bool(args.disable_der_agc),
        "disable_pvd_agc": bool(args.disable_pvd_agc),
        "disable_esd_agc": bool(args.disable_esd_agc),
        "force_restart": bool(args.force_restart),
    }
    (args.results_root / "phase1_full_day_manifest.json").write_text(json.dumps(manifest, indent=2))
    write_memo(args.results_root / "phase1_full_day_summary.md", args.baseline_results_dir, baseline, ranked)

    progress = {
        "candidate_count": int(candidate_count),
        "completed_candidate_count": int(len(rows)),
        "last_completed_combo_id": str(rows[-1]["combo_id"]) if rows else "",
        "accepted_candidate_count": int(ranked["accepted"].sum()) if not ranked.empty else 0,
        "top_candidate_ids": ranked.head(int(args.top_k))["combo_id"].tolist() if not ranked.empty else [],
    }
    (args.results_root / "phase1_full_day_progress.json").write_text(json.dumps(progress, indent=2))


def write_memo(out_path: Path, baseline_dir: Path, baseline: dict[str, float], ranked: pd.DataFrame) -> None:
    lines = [
        "# Phase-1 Deadband Full-Day Validation",
        "",
        "## Baseline",
        "",
        f"- baseline_dir: `{baseline_dir}`",
        f"- mean_abs_hz: {baseline['mean_abs_hz']:.5f}",
        f"- share(|f| > 0.036): {baseline['share_abs_gt_0p036']:.2%}",
        f"- share(|f| > 0.05): {baseline['share_abs_gt_0p05']:.2%}",
        f"- max_abs_hz: {baseline['max_abs_hz']:.5f}",
        f"- edge_mass_36: {baseline['edge_mass_36']:.2%}",
        f"- edge_asymmetry_36: {baseline['edge_asymmetry_36']:.2%}",
        "",
        "## Recommendation",
        "",
    ]
    if ranked.empty:
        lines.append("- no eligible candidate finished")
    else:
        best = ranked.iloc[0]
        lines.extend([
            f"- best candidate: `{best['combo_id']}`",
            f"- wind/pv/esd deadband: {best['wind_deadband_hz']:.3f} / {best['solar_deadband_hz']:.3f} / {best['esd_deadband_hz']:.3f} Hz",
            f"- mean_abs_hz: {best['mean_abs_hz']:.5f}",
            f"- share(|f| > 0.036): {best['share_abs_gt_0p036']:.2%}",
            f"- share(|f| > 0.05): {best['share_abs_gt_0p05']:.2%}",
            f"- max_abs_hz: {best['max_abs_hz']:.5f}",
            f"- edge_mass_36: {best['edge_mass_36']:.2%}",
            f"- edge_asymmetry_36: {best['edge_asymmetry_36']:.2%}",
            f"- esd_throughput: {best['esd_throughput']:.4f}",
            f"- wind_effort: {best['wind_effort']:.4f}",
            f"- pv_effort: {best['pv_effort']:.4f}",
            f"- gov_droop_effort: {best['gov_droop_effort']:.4f}",
            f"- result_dir: `{best['result_dir']}`",
            f"- distribution_plot: `{best['result_dir']}/frequency_distribution.png`",
            f"- curves_plot: `{best['result_dir']}/frequency_curves_all_96.png`",
        ])
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.results_root.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(args.candidate_csv)
    candidates = candidates[candidates["eligible"] == 1].copy().head(int(args.top_k))
    if candidates.empty:
        raise RuntimeError("No eligible candidates found in candidate CSV.")

    baseline = load_baseline_metrics(args.baseline_results_dir)
    rows, completed_combo_ids = load_resume_state(args.results_root, force_restart=bool(args.force_restart))
    if completed_combo_ids:
        print(f"[resume] loaded {len(rows)} completed candidates", flush=True)

    for pos, (_, candidate) in enumerate(candidates.iterrows(), start=1):
        combo_id = str(candidate["combo_id"])
        if combo_id in completed_combo_ids:
            print(f"[{pos}/{len(candidates)}] {combo_id} already complete, skipping", flush=True)
            continue

        print(f"[{pos}/{len(candidates)}] running {combo_id}", flush=True)
        result_dir = run_candidate(args, candidate)
        rows.append(summarize_candidate(candidate, baseline, result_dir))
        completed_combo_ids.add(combo_id)
        write_outputs(
            args=args,
            baseline=baseline,
            rows=rows,
            candidate_count=len(candidates),
        )

    write_outputs(
        args=args,
        baseline=baseline,
        rows=rows,
        candidate_count=len(candidates),
    )

    print(f"full_day_summary_csv={args.results_root / 'phase1_full_day_summary.csv'}")
    print(f"full_day_ranked_csv={args.results_root / 'phase1_full_day_ranked.csv'}")
    print(f"full_day_memo={args.results_root / 'phase1_full_day_summary.md'}")


if __name__ == "__main__":
    main()
