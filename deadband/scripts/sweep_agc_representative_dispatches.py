#!/usr/bin/env python3
"""
Sweep AGC parameters on a fixed set of representative cold-start dispatches.

The goal is to replace single-dispatch tuning with a reproducible multi-case
selection workflow:

- run the same AGC/grid configuration on each representative dispatch
- apply threshold filters (`final_abs_hz`, `max_abs_hz`)
- compute per-dispatch Pareto fronts over frequency-quality and oscillation metrics
- emit global candidates that stay Pareto-feasible across all representative cases
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DISPATCH_SCRIPT = SCRIPT_DIR / "run_dispatch_hotstart.py"
LABEL_RE = re.compile(r"^h(?P<hour>\d+)d(?P<dispatch>\d+)$")
FILE_RE = re.compile(r"^h(?P<hour>\d+)d(?P<dispatch>\d+)_dispatch\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatch-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--curve-file", type=Path, required=True)
    parser.add_argument("--dispatch-label", dest="dispatch_labels", action="append", required=True)
    parser.add_argument("--kp-list", type=float, nargs="+", required=True)
    parser.add_argument("--ki-list", type=float, nargs="+", required=True)
    parser.add_argument("--duration-seconds", type=int, default=900)
    parser.add_argument("--agc-interval", type=int, default=4)
    parser.add_argument("--init-mode", choices=("dispatch", "first"), default="first")
    parser.add_argument(
        "--governor-target-schedule",
        choices=("step", "boundary_ramp", "midpoint_trajectory", "ramp_limited_basepoint"),
        default="ramp_limited_basepoint",
    )
    parser.add_argument("--governor-basepoint-ramp-floor-frac-pmax-per-min", type=float, default=0.005)
    parser.add_argument("--governor-basepoint-ramp-gap-factor", type=float, default=1.25)
    parser.add_argument("--agc-gov-output-ramp-frac-pmax-per-min", type=float, default=0.0)
    parser.add_argument("--agc-dg-output-ramp-frac-pmax-per-min", type=float, default=0.0)
    parser.add_argument(
        "--agc-anti-windup-mode",
        choices=("off", "freeze_on_saturation"),
        default="off",
    )
    parser.add_argument("--traditional-governor-deadband-hz", type=float, default=None)
    parser.add_argument("--der-deadband-hz", type=float, default=None)
    parser.add_argument("--der-base-ddn", type=float, default=None)
    parser.add_argument("--target-storage-share", type=float, default=None)
    parser.add_argument("--scale-esd1-ddn-with-storage", action="store_true")
    parser.add_argument("--final-abs-limit", type=float, default=0.01)
    parser.add_argument("--peak-abs-limit", type=float, default=0.05)
    parser.add_argument("--save-plot", dest="save_plot", action="store_true")
    parser.add_argument("--no-save-plot", dest="save_plot", action="store_false")
    parser.set_defaults(save_plot=False)
    return parser.parse_args()


def fmt_token(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def parse_dispatch_label(label: str) -> tuple[int, int]:
    match = LABEL_RE.match(label)
    if match is None:
        raise ValueError(f"invalid dispatch label: {label}")
    return int(match.group("hour")), int(match.group("dispatch"))


def discover_dispatches(dispatch_dir: Path) -> dict[str, Path]:
    rows: list[tuple[int, int, str, Path]] = []
    for path in dispatch_dir.glob("h*d*_dispatch.json"):
        match = FILE_RE.match(path.name)
        if match is None:
            continue
        hour = int(match.group("hour"))
        dispatch = int(match.group("dispatch"))
        label = f"h{hour}d{dispatch}"
        rows.append((hour, dispatch, label, path))
    rows.sort()
    return {label: path for _, _, label, path in rows}


def next_dispatch_path(dispatch_files: dict[str, Path], label: str) -> Path:
    labels = list(dispatch_files.keys())
    if label not in dispatch_files:
        raise RuntimeError(f"dispatch JSON not found for {label}")
    pos = labels.index(label)
    if pos + 1 >= len(labels):
        raise RuntimeError(f"next dispatch JSON not found for {label}")
    return dispatch_files[labels[pos + 1]]


def pareto_front_mask(frame: pd.DataFrame, metric_cols: list[str]) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    values = frame[metric_cols].to_numpy(dtype=float)
    mask = [True] * len(frame)
    tol = 1e-12
    for i, row in enumerate(values):
        for j, other in enumerate(values):
            if i == j:
                continue
            if ((other <= row + tol).all() and (other < row - tol).any()):
                mask[i] = False
                break
    return pd.Series(mask, index=frame.index, dtype=bool)


def run_one(args: argparse.Namespace, dispatch_json: Path, next_dispatch_json: Path, kp: float, ki: float) -> pd.DataFrame:
    dispatch_label = dispatch_json.stem.replace("_dispatch", "")
    label = f"{dispatch_label}_kp{fmt_token(kp)}_ki{fmt_token(ki)}"
    cmd = [
        sys.executable,
        str(RUN_DISPATCH_SCRIPT),
        "--dispatch-json", str(dispatch_json),
        "--next-dispatch-json", str(next_dispatch_json),
        "--label", label,
        "--results-dir", str(args.results_dir),
        "--curve-file", str(args.curve_file),
        "--duration-seconds", str(args.duration_seconds),
        "--agc-interval", str(args.agc_interval),
        "--kp", str(kp),
        "--ki", str(ki),
        "--agc-gov-output-ramp-frac-pmax-per-min", str(args.agc_gov_output_ramp_frac_pmax_per_min),
        "--agc-dg-output-ramp-frac-pmax-per-min", str(args.agc_dg_output_ramp_frac_pmax_per_min),
        "--agc-anti-windup-mode", args.agc_anti_windup_mode,
        "--init-mode", args.init_mode,
        "--apply-governor-targets",
        "--governor-target-schedule", args.governor_target_schedule,
        "--governor-basepoint-ramp-floor-frac-pmax-per-min",
        str(args.governor_basepoint_ramp_floor_frac_pmax_per_min),
        "--governor-basepoint-ramp-gap-factor",
        str(args.governor_basepoint_ramp_gap_factor),
        "--no-save-checkpoint",
    ]
    if args.traditional_governor_deadband_hz is not None:
        cmd.extend(["--traditional-governor-deadband-hz", str(args.traditional_governor_deadband_hz)])
    if args.der_deadband_hz is not None:
        cmd.extend(["--der-deadband-hz", str(args.der_deadband_hz)])
    if args.der_base_ddn is not None:
        cmd.extend(["--der-base-ddn", str(args.der_base_ddn)])
    if args.target_storage_share is not None:
        cmd.extend(["--target-storage-share", str(args.target_storage_share)])
    if args.scale_esd1_ddn_with_storage:
        cmd.append("--scale-esd1-ddn-with-storage")
    else:
        cmd.append("--no-scale-esd1-ddn-with-storage")
    if args.save_plot:
        cmd.append("--save-plot")
    else:
        cmd.append("--no-save-plot")

    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"run failed for {dispatch_label} kp={kp} ki={ki}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    summary_csv = args.results_dir / f"{label}_summary.csv"
    freq_csv = args.results_dir / f"{label}_frequency.csv"
    if not summary_csv.exists():
        raise RuntimeError(f"missing summary for {label}")
    summary = pd.read_csv(summary_csv)
    if freq_csv.exists():
        freq = pd.read_csv(freq_csv)["freq_dev_hz"].to_numpy(dtype=float)
        abs_f = abs(freq)
        summary["share_abs_gt_0p036"] = float((abs_f > 0.036).mean())
        summary["share_abs_gt_0p05"] = float((abs_f > 0.05).mean())
        summary["zero_crossings"] = int(((freq[1:] < 0) != (freq[:-1] < 0)).sum()) if freq.size > 1 else 0
        summary["max_abs_hz"] = float(abs_f.max())
        summary["final_abs_hz"] = float(abs_f[-1])
    summary["dispatch_label"] = dispatch_label
    return summary


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    dispatch_files = discover_dispatches(args.dispatch_dir)
    for label in args.dispatch_labels:
        if label not in dispatch_files:
            raise RuntimeError(f"missing dispatch JSON for {label}")

    rows: list[dict[str, object]] = []
    total = len(args.dispatch_labels) * len(args.kp_list) * len(args.ki_list)
    run_no = 0

    for dispatch_label in args.dispatch_labels:
        dispatch_json = dispatch_files[dispatch_label]
        next_json = next_dispatch_path(dispatch_files, dispatch_label)
        for kp in args.kp_list:
            for ki in args.ki_list:
                run_no += 1
                print(f"[{run_no}/{total}] {dispatch_label} kp={kp:.4f} ki={ki:.4f}", flush=True)
                summary = run_one(args, dispatch_json, next_json, kp, ki).iloc[0].to_dict()
                summary["command_tv"] = float(summary.get("gov_paux0_tv", 0.0)) + float(summary.get("dg_pext0_tv", 0.0))
                summary["final_abs_hz"] = abs(float(summary["final_hz"]))
                summary["max_abs_hz"] = max(abs(float(summary["min_hz"])), abs(float(summary["max_hz"])))
                rows.append(summary)

    summary_df = pd.DataFrame(rows).sort_values(["dispatch_label", "kp", "ki"]).reset_index(drop=True)
    summary_df["passes_final_abs"] = summary_df["final_abs_hz"] <= float(args.final_abs_limit)
    summary_df["passes_peak_abs"] = summary_df["max_abs_hz"] <= float(args.peak_abs_limit)
    summary_df["passes_thresholds"] = summary_df["passes_final_abs"] & summary_df["passes_peak_abs"]
    summary_df["dispatch_pareto"] = 0

    metric_cols = [
        "share_abs_gt_0p036",
        "zero_crossings",
        "freq_d1_abs_mean",
        "command_tv",
        "saturation_fraction",
    ]
    for dispatch_label, group in summary_df.groupby("dispatch_label"):
        eligible = group[group["passes_thresholds"]].copy()
        if eligible.empty:
            continue
        mask = pareto_front_mask(eligible, metric_cols)
        summary_df.loc[mask.index[mask], "dispatch_pareto"] = 1

    global_rows: list[dict[str, object]] = []
    total_dispatches = len(args.dispatch_labels)
    for (kp, ki), group in summary_df.groupby(["kp", "ki"], sort=True):
        group = group.sort_values("dispatch_label")
        row = {
            "kp": float(kp),
            "ki": float(ki),
            "dispatch_count": int(len(group)),
            "all_pass_thresholds": bool(group["passes_thresholds"].all()),
            "all_dispatches_pareto": bool((group["dispatch_pareto"] == 1).all() and len(group) == total_dispatches),
            "share_abs_gt_0p036_mean": float(group["share_abs_gt_0p036"].mean()),
            "zero_crossings_mean": float(group["zero_crossings"].mean()),
            "freq_d1_abs_mean_mean": float(group["freq_d1_abs_mean"].mean()),
            "command_tv_mean": float(group["command_tv"].mean()),
            "saturation_fraction_mean": float(group["saturation_fraction"].mean()),
            "final_abs_hz_max": float(group["final_abs_hz"].max()),
            "max_abs_hz_max": float(group["max_abs_hz"].max()),
        }
        global_rows.append(row)

    global_df = pd.DataFrame(global_rows).sort_values(
        [
            "all_dispatches_pareto",
            "all_pass_thresholds",
            "share_abs_gt_0p036_mean",
            "freq_d1_abs_mean_mean",
            "command_tv_mean",
        ],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)

    summary_csv = args.results_dir / "representative_summary.csv"
    pareto_csv = args.results_dir / "representative_pareto.csv"
    global_csv = args.results_dir / "global_candidates.csv"
    config_json = args.results_dir / "representative_sweep_config.json"

    summary_df.to_csv(summary_csv, index=False)
    summary_df[summary_df["dispatch_pareto"] == 1].to_csv(pareto_csv, index=False)
    global_df.to_csv(global_csv, index=False)
    config_json.write_text(json.dumps(vars(args), indent=2, default=str))

    print(f"summary_csv={summary_csv}")
    print(f"pareto_csv={pareto_csv}")
    print(f"global_csv={global_csv}")
    print(f"config_json={config_json}")
    print(global_df.to_string(index=False))


if __name__ == "__main__":
    main()
