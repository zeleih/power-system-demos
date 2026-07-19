#!/usr/bin/env python3
"""
Prepare one day of dispatch JSON files without running TDS.

The output of this script is parameter-independent and can be reused by
different hot-start checkpoint families.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import run_dispatch_tds as rdt


RESULTS = rdt.RESULTS
DAY_SECONDS = 24 * 3600
DEFAULT_DISPATCHES_PER_HOUR = 4
DEFAULT_DISPATCH_INTERVAL = 900


def enumerate_dispatches(
    hour_start: int,
    hours: int,
    dispatches_per_hour: int,
) -> list[tuple[int, int]]:
    tasks: list[tuple[int, int]] = []
    for hour in range(hour_start, hour_start + hours):
        for dispatch in range(dispatches_per_hour):
            tasks.append((hour, dispatch))
    return tasks


def compute_one(
    hour: int,
    dispatch: int,
    curve_file: str,
    opf_case: str,
    dispatch_interval: int,
    wind_pref_alpha: float,
    solar_pref_alpha: float,
) -> dict[str, object]:
    curve = rdt.load_curve(Path(curve_file))
    record = rdt.compute_dispatch(
        hour=hour,
        dispatch=dispatch,
        curve=curve,
        opf_case=Path(opf_case),
        dispatch_interval=dispatch_interval,
        wind_pref_alpha=wind_pref_alpha,
        solar_pref_alpha=solar_pref_alpha,
    )
    return record.__dict__


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opf-case", type=Path, default=rdt.DEFAULT_OPF_CASE)
    parser.add_argument("--curve-file", type=Path, default=rdt.DEFAULT_CURVE_FILE)
    parser.add_argument("--results-dir", type=Path, default=RESULTS / "dispatches_day96")
    parser.add_argument("--hour-start", type=int, default=0)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--dispatches-per-hour", type=int, default=DEFAULT_DISPATCHES_PER_HOUR)
    parser.add_argument("--dispatch-interval", type=int, default=DEFAULT_DISPATCH_INTERVAL)
    parser.add_argument("--wind-pref-alpha", type=float, default=1.0)
    parser.add_argument("--solar-pref-alpha", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    tasks = enumerate_dispatches(args.hour_start, args.hours, args.dispatches_per_hour)
    total_expected = args.hours * args.dispatches_per_hour
    if args.dispatch_interval * total_expected > DAY_SECONDS:
        raise RuntimeError("Requested dispatch coverage exceeds 24 hours.")

    rows: list[dict[str, object]] = []
    pending: list[tuple[int, int]] = []

    for hour, dispatch in tasks:
        label = f"h{hour}d{dispatch}"
        path = args.results_dir / f"{label}_dispatch.json"
        if args.skip_existing and path.exists():
            record = rdt.DispatchRecord.from_json(path)
            rows.append({
                "hour": record.hour,
                "dispatch": record.dispatch,
                "label": record.label,
                "dispatch_json": str(path),
                "converged": int(record.converged),
                "obj": float(record.obj),
                "load": float(record.load),
                "wind": float(record.wind),
                "solar": float(record.solar),
                "status": "existing",
            })
        else:
            pending.append((hour, dispatch))

    if args.workers <= 1:
        for hour, dispatch in pending:
            payload = compute_one(
                hour=hour,
                dispatch=dispatch,
                curve_file=str(args.curve_file),
                opf_case=str(args.opf_case),
                dispatch_interval=args.dispatch_interval,
                wind_pref_alpha=args.wind_pref_alpha,
                solar_pref_alpha=args.solar_pref_alpha,
            )
            record = rdt.DispatchRecord(**payload)
            path = rdt.write_dispatch_json(record, args.results_dir, label=record.label)
            rows.append({
                "hour": record.hour,
                "dispatch": record.dispatch,
                "label": record.label,
                "dispatch_json": str(path),
                "converged": int(record.converged),
                "obj": float(record.obj),
                "load": float(record.load),
                "wind": float(record.wind),
                "solar": float(record.solar),
                "status": "new",
            })
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            future_map = {
                pool.submit(
                    compute_one,
                    hour,
                    dispatch,
                    str(args.curve_file),
                    str(args.opf_case),
                    args.dispatch_interval,
                    args.wind_pref_alpha,
                    args.solar_pref_alpha,
                ): (hour, dispatch)
                for hour, dispatch in pending
            }
            for future in as_completed(future_map):
                payload = future.result()
                record = rdt.DispatchRecord(**payload)
                path = rdt.write_dispatch_json(record, args.results_dir, label=record.label)
                rows.append({
                    "hour": record.hour,
                    "dispatch": record.dispatch,
                    "label": record.label,
                    "dispatch_json": str(path),
                    "converged": int(record.converged),
                    "obj": float(record.obj),
                    "load": float(record.load),
                    "wind": float(record.wind),
                    "solar": float(record.solar),
                    "status": "new",
                })

    summary = pd.DataFrame(rows).sort_values(["hour", "dispatch"]).reset_index(drop=True)
    summary_csv = args.results_dir / "dispatch_summary.csv"
    summary.to_csv(summary_csv, index=False)

    converged = int(summary["converged"].sum()) if not summary.empty else 0
    print(f"dispatch_dir={args.results_dir}")
    print(f"summary_csv={summary_csv}")
    print(f"dispatches={len(summary)}")
    print(f"converged={converged}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
