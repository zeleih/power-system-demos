#!/usr/bin/env python3
"""
Run one day of dispatches as independent cold-start segments.

Each dispatch interval is executed in a fresh Python process via
``run_dispatch_hotstart.py`` without loading a checkpoint. This keeps the
runtime configuration aligned with the current hotstart runner while ensuring
every segment starts from its own cold-start initialization.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

import run_dispatch_tds as rdt
from prepare_day_dispatches import enumerate_dispatches


RESULTS = rdt.RESULTS
DEFAULT_DISPATCHES_PER_HOUR = 4
DEFAULT_DISPATCH_INTERVAL = 900
SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DISPATCH_SCRIPT = SCRIPT_DIR / "run_dispatch_hotstart.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatch-dir", type=Path, default=RESULTS / "dispatches_day96")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--hour-start", type=int, default=0)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--dispatches-per-hour", type=int, default=DEFAULT_DISPATCHES_PER_HOUR)
    parser.add_argument("--dispatch-interval", type=int, default=DEFAULT_DISPATCH_INTERVAL)
    parser.add_argument("--opf-case", type=Path, default=rdt.DEFAULT_OPF_CASE)
    parser.add_argument("--dyn-case", type=Path, default=rdt.DEFAULT_DYN_CASE)
    parser.add_argument("--stable-dyn-case", type=Path, default=rdt.DEFAULT_STABLE_DYN_CASE)
    parser.add_argument("--curve-file", type=Path, default=rdt.DEFAULT_CURVE_FILE)
    parser.add_argument("--agc-interval", type=int, default=4)
    parser.add_argument("--kp", type=float, default=0.03)
    parser.add_argument("--ki", type=float, default=0.01)
    parser.add_argument("--agc-gov-output-ramp-frac-pmax-per-min", type=float, default=0.0)
    parser.add_argument("--agc-dg-output-ramp-frac-pmax-per-min", type=float, default=0.0)
    parser.add_argument(
        "--agc-anti-windup-mode",
        choices=("off", "freeze_on_saturation"),
        default="off",
    )
    parser.add_argument("--traditional-governor-deadband-hz", type=float, default=None)
    parser.add_argument("--traditional-governor-deadband-csv", type=Path, default=None)
    parser.add_argument("--der-deadband-hz", type=float, default=None)
    parser.add_argument("--der-base-ddn", type=float, default=None)
    parser.add_argument("--target-storage-share", type=float, default=None)
    parser.add_argument("--scale-esd1-ddn-with-storage", dest="scale_esd1_ddn_with_storage", action="store_true")
    parser.add_argument("--no-scale-esd1-ddn-with-storage", dest="scale_esd1_ddn_with_storage", action="store_false")
    parser.add_argument(
        "--governor-target-schedule",
        choices=("step", "boundary_ramp", "midpoint_trajectory", "ramp_limited_basepoint"),
        default="ramp_limited_basepoint",
    )
    parser.add_argument("--governor-basepoint-ramp-floor-frac-pmax-per-min", type=float, default=0.005)
    parser.add_argument("--governor-basepoint-ramp-gap-factor", type=float, default=1.25)
    parser.add_argument("--init-mode", choices=("dispatch", "first"), default="first")
    parser.add_argument("--wind-prefix", action="append", default=None)
    parser.add_argument("--solar-prefix", action="append", default=None)
    parser.add_argument("--apply-governor-targets", dest="apply_governor_targets", action="store_true")
    parser.add_argument("--no-apply-governor-targets", dest="apply_governor_targets", action="store_false")
    parser.add_argument("--save-plot", dest="save_plot", action="store_true")
    parser.add_argument("--no-save-plot", dest="save_plot", action="store_false")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.set_defaults(
        apply_governor_targets=True,
        scale_esd1_ddn_with_storage=False,
        save_plot=False,
    )
    return parser.parse_args()


def build_cmd(
    args: argparse.Namespace,
    *,
    label: str,
    dispatch_json: Path,
    wind_prefixes: tuple[str, ...],
    solar_prefixes: tuple[str, ...],
) -> list[str]:
    cmd = [
        sys.executable,
        str(RUN_DISPATCH_SCRIPT),
        "--dispatch-json", str(dispatch_json),
        "--label", label,
        "--results-dir", str(args.results_dir),
        "--opf-case", str(args.opf_case),
        "--dyn-case", str(args.dyn_case),
        "--stable-dyn-case", str(args.stable_dyn_case),
        "--curve-file", str(args.curve_file),
        "--duration-seconds", str(args.dispatch_interval),
        "--agc-interval", str(args.agc_interval),
        "--kp", str(args.kp),
        "--ki", str(args.ki),
        "--agc-gov-output-ramp-frac-pmax-per-min", str(args.agc_gov_output_ramp_frac_pmax_per_min),
        "--agc-dg-output-ramp-frac-pmax-per-min", str(args.agc_dg_output_ramp_frac_pmax_per_min),
        "--agc-anti-windup-mode", args.agc_anti_windup_mode,
        "--governor-target-schedule", args.governor_target_schedule,
        "--governor-basepoint-ramp-floor-frac-pmax-per-min",
        str(args.governor_basepoint_ramp_floor_frac_pmax_per_min),
        "--governor-basepoint-ramp-gap-factor",
        str(args.governor_basepoint_ramp_gap_factor),
        "--init-mode", args.init_mode,
        "--no-save-checkpoint",
    ]
    if args.traditional_governor_deadband_hz is not None:
        cmd.extend(["--traditional-governor-deadband-hz", str(args.traditional_governor_deadband_hz)])
    if args.traditional_governor_deadband_csv is not None:
        cmd.extend(["--traditional-governor-deadband-csv", str(args.traditional_governor_deadband_csv)])
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
    if args.apply_governor_targets:
        cmd.append("--apply-governor-targets")
    else:
        cmd.append("--no-apply-governor-targets")
    if args.save_plot:
        cmd.append("--save-plot")
    else:
        cmd.append("--no-save-plot")
    for prefix in wind_prefixes:
        cmd.extend(["--wind-prefix", prefix])
    for prefix in solar_prefixes:
        cmd.extend(["--solar-prefix", prefix])
    return cmd


def run_one(
    args: argparse.Namespace,
    *,
    label: str,
    dispatch_json: Path,
    wind_prefixes: tuple[str, ...],
    solar_prefixes: tuple[str, ...],
) -> dict[str, object]:
    cmd = build_cmd(
        args,
        label=label,
        dispatch_json=dispatch_json,
        wind_prefixes=wind_prefixes,
        solar_prefixes=solar_prefixes,
    )
    started = time.perf_counter()
    completed = subprocess.run(cmd, capture_output=True, text=True)
    elapsed_s = time.perf_counter() - started
    if completed.returncode != 0:
        return {
            "label": label,
            "ok": False,
            "elapsed_s": float(elapsed_s),
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }

    summary_csv = args.results_dir / f"{label}_summary.csv"
    row = pd.read_csv(summary_csv).iloc[0].to_dict()
    row["ok"] = True
    row["elapsed_s"] = float(elapsed_s)
    return row


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    wind_prefixes = rdt.normalize_prefixes(args.wind_prefix, rdt.DEFAULT_WIND_PREFIXES)
    solar_prefixes = rdt.normalize_prefixes(args.solar_prefix, rdt.DEFAULT_SOLAR_PREFIXES)
    tasks = enumerate_dispatches(args.hour_start, args.hours, args.dispatches_per_hour)

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    with cf.ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as executor:
        futures = {}
        for hour, dispatch in tasks:
            label = f"h{hour}d{dispatch}"
            dispatch_json = args.dispatch_dir / f"{label}_dispatch.json"
            if not dispatch_json.exists():
                raise RuntimeError(f"Missing dispatch JSON for {label}: {dispatch_json}")
            futures[executor.submit(
                run_one,
                args,
                label=label,
                dispatch_json=dispatch_json,
                wind_prefixes=wind_prefixes,
                solar_prefixes=solar_prefixes,
            )] = label

        total = len(futures)
        completed_count = 0
        for fut in cf.as_completed(futures):
            completed_count += 1
            item = fut.result()
            label = str(item["label"])
            if bool(item.get("ok")):
                rows.append(item)
                print(
                    f"[{completed_count}/{total}] {label} ok "
                    f"final_hz={float(item['final_hz']):+.6f} elapsed_s={float(item['elapsed_s']):.1f}",
                    flush=True,
                )
            else:
                failures.append(item)
                print(f"[{completed_count}/{total}] {label} FAILED", flush=True)

    if failures:
        failures_path = args.results_dir / "failures.json"
        failures_df = pd.DataFrame(failures)
        failures_df.to_json(failures_path, orient="records", indent=2, force_ascii=False)
        raise RuntimeError(f"{len(failures)} dispatches failed. See {failures_path}")

    summary = pd.DataFrame(rows).sort_values(["hour", "dispatch"]).reset_index(drop=True)
    summary_csv = args.results_dir / "daily_hotstart_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print(f"results_dir={args.results_dir}")
    print(f"summary_csv={summary_csv}")
    print(f"dispatches={len(summary)}")


if __name__ == "__main__":
    main()
