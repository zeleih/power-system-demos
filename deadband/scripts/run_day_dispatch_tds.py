#!/usr/bin/env python3
"""
Run one full day's 96 dispatch intervals through the deadband TDS workflow.

Outputs are organized into one directory containing:

- per-dispatch JSON/CSV files
- a summary CSV with one row per dispatch
- comparison plots across all successful dispatches
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_dispatch_tds as rdt


RESULTS = rdt.RESULTS
DAY_SECONDS = 24 * 3600
DEFAULT_DISPATCHES_PER_HOUR = 4
DEFAULT_DISPATCH_INTERVAL = 900
SERIES_PREFIX = "series"

_CTX: dict[str, Any] = {}


def format_token(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def build_default_results_dir(kp: float, ki: float, agc_interval: int) -> Path:
    stem = (
        f"day96_agc{agc_interval}_"
        f"kp{format_token(kp)}_"
        f"ki{format_token(ki)}"
    )
    return RESULTS / stem


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


def series_paths(out_dir: Path, label: str) -> tuple[Path, Path, Path]:
    dispatch_json = out_dir / f"{label}_dispatch.json"
    freq_csv = out_dir / f"{label}_frequency.csv"
    freq_png = out_dir / f"{label}_frequency.png"
    return dispatch_json, freq_csv, freq_png


def save_series_csv(t: np.ndarray, f_dev_hz: np.ndarray, out_dir: Path, label: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{label}_frequency.csv"
    pd.DataFrame({"time_s": t, "freq_dev_hz": f_dev_hz}).to_csv(path, index=False)
    return path


def summarize_series(t: np.ndarray, f_dev_hz: np.ndarray) -> dict[str, float | int]:
    imin = int(np.argmin(f_dev_hz))
    imax = int(np.argmax(f_dev_hz))
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
    }


def load_existing_summary(label: str, out_dir: Path) -> dict[str, Any]:
    dispatch_json, freq_csv, freq_png = series_paths(out_dir, label)
    dispatch_record = rdt.DispatchRecord.from_json(dispatch_json)
    df = pd.read_csv(freq_csv)
    t = df["time_s"].to_numpy(dtype=float)
    f_dev_hz = df["freq_dev_hz"].to_numpy(dtype=float)
    row = {
        "hour": dispatch_record.hour,
        "dispatch": dispatch_record.dispatch,
        "label": label,
        "success": 1,
        "error": "",
        "dispatch_json": str(dispatch_json),
        "freq_csv": str(freq_csv),
        "freq_png": str(freq_png if freq_png.exists() else ""),
        "dispatch_seconds": math.nan,
        "tds_seconds": math.nan,
        "total_seconds": math.nan,
        "init_mode_used": "",
        "retry_note": "",
    }
    row.update(summarize_series(t, f_dev_hz))
    return row


def init_worker(
    curve_file: str,
    opf_case: str,
    dyn_case: str,
    duration_seconds: int,
    agc_interval: int,
    kp: float,
    ki: float,
    init_mode: str,
    retry_init_mode: str | None,
    retry_early_fail_seconds: int,
    wind_prefixes: tuple[str, ...],
    solar_prefixes: tuple[str, ...],
    out_dir: str,
    save_series_plots: bool,
) -> None:
    _CTX["curve"] = rdt.load_curve(Path(curve_file))
    _CTX["opf_case"] = Path(opf_case)
    _CTX["dyn_case"] = Path(dyn_case)
    _CTX["duration_seconds"] = int(duration_seconds)
    _CTX["agc_interval"] = int(agc_interval)
    _CTX["kp"] = float(kp)
    _CTX["ki"] = float(ki)
    _CTX["init_mode"] = str(init_mode)
    _CTX["retry_init_mode"] = retry_init_mode
    _CTX["retry_early_fail_seconds"] = int(retry_early_fail_seconds)
    _CTX["wind_prefixes"] = tuple(wind_prefixes)
    _CTX["solar_prefixes"] = tuple(solar_prefixes)
    _CTX["out_dir"] = Path(out_dir)
    _CTX["save_series_plots"] = bool(save_series_plots)
    rdt.andes.config_logger(stream_level=40)

    try:
        import ams

        ams.config_logger(stream_level=50)
    except Exception:
        pass


def extract_fail_second(message: str) -> int | None:
    match = re.search(r"TDS failed at t=(\d+)s", message)
    return int(match.group(1)) if match else None


def should_retry(exc: Exception, init_mode_used: str) -> bool:
    retry_init_mode = _CTX["retry_init_mode"]
    if retry_init_mode is None or retry_init_mode == init_mode_used:
        return False

    message = str(exc)
    if "TDS init failed" in message:
        return True

    fail_second = extract_fail_second(message)
    if fail_second is None:
        return False

    return fail_second <= _CTX["retry_early_fail_seconds"]


def run_one(task: tuple[int, int]) -> dict[str, Any]:
    hour, dispatch = task
    label = f"h{hour}d{dispatch}"
    out_dir = _CTX["out_dir"]

    start_total = time.perf_counter()
    dispatch_start = time.perf_counter()
    dispatch_record = rdt.compute_dispatch(
        hour=hour,
        dispatch=dispatch,
        curve=_CTX["curve"],
        opf_case=_CTX["opf_case"],
        dispatch_interval=_CTX["duration_seconds"],
    )
    dispatch_seconds = time.perf_counter() - dispatch_start

    if not dispatch_record.converged:
        raise RuntimeError(f"Dispatch {label} did not converge")

    init_mode_used = _CTX["init_mode"]
    retry_note = ""
    tds_start = time.perf_counter()
    try:
        t, f_dev_hz = rdt.run_tds(
            dispatch_record=dispatch_record,
            curve=_CTX["curve"],
            dyn_case=_CTX["dyn_case"],
            duration_seconds=_CTX["duration_seconds"],
            agc_interval=_CTX["agc_interval"],
            kp=_CTX["kp"],
            ki=_CTX["ki"],
            wind_prefixes=_CTX["wind_prefixes"],
            solar_prefixes=_CTX["solar_prefixes"],
            init_mode=init_mode_used,
        )
    except Exception as exc:
        if not should_retry(exc, init_mode_used):
            raise

        retry_init_mode = _CTX["retry_init_mode"]
        init_mode_used = retry_init_mode
        retry_note = f"retry_after={exc}"
        t, f_dev_hz = rdt.run_tds(
            dispatch_record=dispatch_record,
            curve=_CTX["curve"],
            dyn_case=_CTX["dyn_case"],
            duration_seconds=_CTX["duration_seconds"],
            agc_interval=_CTX["agc_interval"],
            kp=_CTX["kp"],
            ki=_CTX["ki"],
            wind_prefixes=_CTX["wind_prefixes"],
            solar_prefixes=_CTX["solar_prefixes"],
            init_mode=init_mode_used,
        )

    tds_seconds = time.perf_counter() - tds_start

    dispatch_json = rdt.write_dispatch_json(dispatch_record, out_dir, label=label)
    if _CTX["save_series_plots"]:
        freq_csv, freq_png = rdt.save_outputs(t, f_dev_hz, dispatch_record, out_dir, label=label)
    else:
        freq_csv = save_series_csv(t, f_dev_hz, out_dir, label=label)
        _, _, freq_png = series_paths(out_dir, label)

    row = {
        "hour": hour,
        "dispatch": dispatch,
        "label": label,
        "success": 1,
        "error": "",
        "dispatch_json": str(dispatch_json),
        "freq_csv": str(freq_csv),
        "freq_png": str(freq_png if freq_png.exists() else ""),
        "dispatch_seconds": float(dispatch_seconds),
        "tds_seconds": float(tds_seconds),
        "total_seconds": float(time.perf_counter() - start_total),
        "init_mode_used": init_mode_used,
        "retry_note": retry_note,
    }
    row.update(summarize_series(t, f_dev_hz))
    return row


def load_series_grid(freq_csv: Path, t_grid: np.ndarray) -> np.ndarray:
    df = pd.read_csv(freq_csv)
    df = df.drop_duplicates(subset="time_s", keep="last").sort_values("time_s")
    t = df["time_s"].to_numpy(dtype=float)
    y = df["freq_dev_hz"].to_numpy(dtype=float)

    if len(t) == 0:
        return np.full_like(t_grid, np.nan, dtype=float)

    if len(t) == 1:
        out = np.full_like(t_grid, y[0], dtype=float)
        out[t_grid < t[0]] = np.nan
        return out

    left = y[0] if t_grid[0] >= t[0] else np.nan
    right = y[-1] if t_grid[-1] <= t[-1] else np.nan
    grid = np.interp(t_grid, t, y, left=np.nan, right=right)
    if np.isnan(left):
        grid[t_grid < t[0]] = np.nan
    return grid


def load_successful_grid(summary: pd.DataFrame, duration_seconds: int) -> tuple[np.ndarray, np.ndarray]:
    t_grid = np.arange(duration_seconds, dtype=float)
    rows = []
    for _, row in summary.iterrows():
        rows.append(load_series_grid(Path(row["freq_csv"]), t_grid))
    return t_grid, np.vstack(rows)


def make_overview_plot(fig_path: Path, summary: pd.DataFrame, duration_seconds: int) -> None:
    t_grid, grid = load_successful_grid(summary, duration_seconds)

    mean_curve = np.nanmean(grid, axis=0)
    median_curve = np.nanmedian(grid, axis=0)
    p10 = np.nanpercentile(grid, 10, axis=0)
    p90 = np.nanpercentile(grid, 90, axis=0)
    ymin = np.nanmin(grid, axis=0)
    ymax = np.nanmax(grid, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    cmap = plt.get_cmap("turbo")
    norm = plt.Normalize(summary["hour"].min(), summary["hour"].max())
    for _, row in summary.iterrows():
        color = cmap(norm(row["hour"]))
        df = pd.read_csv(row["freq_csv"])
        axes[0].plot(df["time_s"], df["freq_dev_hz"], color=color, alpha=0.32, linewidth=0.8)

    axes[0].plot(t_grid, mean_curve, color="#111111", linewidth=2.0, label="Mean across dispatches")
    axes[0].set_title("All 96 Dispatch Frequency Curves")
    axes[0].set_ylabel("Frequency deviation [Hz]")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, loc="upper right")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[0], pad=0.01)
    cbar.set_label("Hour of day")

    axes[1].fill_between(t_grid, ymin, ymax, color="#d9e5ec", alpha=0.9, label="Min-Max envelope")
    axes[1].fill_between(t_grid, p10, p90, color="#8fb6c9", alpha=0.95, label="10-90 percentile")
    axes[1].plot(t_grid, median_curve, color="#0f5c78", linewidth=1.8, label="Median")
    axes[1].plot(t_grid, mean_curve, color="#b85c38", linewidth=1.4, label="Mean")
    axes[1].axhline(0.0, color="#666666", linewidth=0.8, linestyle="--")
    axes[1].set_title("Daily Frequency Response Envelope")
    axes[1].set_xlabel("Time within dispatch [s]")
    axes[1].set_ylabel("Frequency deviation [Hz]")
    axes[1].set_xlim(0, duration_seconds - 1)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, ncol=4, loc="upper right")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def make_heatmap(fig_path: Path, summary: pd.DataFrame, duration_seconds: int) -> None:
    t_grid, grid = load_successful_grid(summary, duration_seconds)

    fig, ax = plt.subplots(figsize=(16, 11))
    vlim = np.nanmax(np.abs(grid))
    im = ax.imshow(
        grid,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-vlim,
        vmax=vlim,
        extent=[t_grid[0], t_grid[-1], 0, len(summary)],
    )
    for boundary in range(0, len(summary) + 1, DEFAULT_DISPATCHES_PER_HOUR):
        ax.axhline(boundary, color="white", linewidth=0.35, alpha=0.45)

    tick_pos = np.arange(0.5, len(summary), DEFAULT_DISPATCHES_PER_HOUR)
    tick_labels = [f"h{int(summary.iloc[i]['hour']):02d}" for i in range(0, len(summary), DEFAULT_DISPATCHES_PER_HOUR)]
    ax.set_yticks(tick_pos, tick_labels)
    ax.set_title("Daily Dispatch Frequency Heatmap")
    ax.set_xlabel("Time within dispatch [s]")
    ax.set_ylabel("Hour blocks (4 dispatches each)")
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Frequency deviation [Hz]")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def make_hourly_grid(fig_path: Path, summary: pd.DataFrame, duration_seconds: int) -> None:
    fig, axes = plt.subplots(6, 4, figsize=(18, 20), sharex=True, sharey=True)
    colors = ["#0f5c78", "#2a7f3f", "#b85c38", "#7a4ba0"]
    legend_added = False

    global_min = float(summary["min_hz"].min())
    global_max = float(summary["max_hz"].max())
    pad = 0.08 * max(abs(global_min), abs(global_max), 1e-4)

    for hour in range(24):
        ax = axes.flat[hour]
        hour_rows = summary[summary["hour"] == hour].sort_values("dispatch")
        for _, row in hour_rows.iterrows():
            dispatch_id = int(row["dispatch"])
            df = pd.read_csv(row["freq_csv"])
            ax.plot(
                df["time_s"],
                df["freq_dev_hz"],
                color=colors[dispatch_id % len(colors)],
                linewidth=1.1,
                label=f"d{dispatch_id}",
            )
        ax.axhline(0.0, color="#666666", linewidth=0.6, linestyle="--")
        ax.set_title(f"Hour {hour:02d}")
        ax.grid(True, alpha=0.22)
        if not legend_added and len(hour_rows) > 0:
            ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper right")
            legend_added = True

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [s]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Freq dev [Hz]")

    for ax in axes.flat:
        ax.set_xlim(0, duration_seconds - 1)
        ax.set_ylim(global_min - pad, global_max + pad)

    fig.suptitle("Daily Frequency Curves by Hour (4 dispatches per panel)", fontsize=16)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def make_rank_plot(fig_path: Path, summary: pd.DataFrame) -> None:
    ranked = summary.sort_values("abs_mean_hz", ascending=False).reset_index(drop=True)
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    x = np.arange(len(ranked))
    axes[0].bar(x, ranked["abs_mean_hz"], color="#0f5c78")
    axes[0].set_ylabel("Mean |freq dev| [Hz]")
    axes[0].set_title("Dispatch Ranking by Mean Absolute Frequency Deviation")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].plot(x, ranked["min_hz"], color="#b85c38", linewidth=1.2, label="Min")
    axes[1].plot(x, ranked["max_hz"], color="#2a7f3f", linewidth=1.2, label="Max")
    axes[1].axhline(0.0, color="#666666", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("Extreme freq dev [Hz]")
    axes[1].set_xlabel("Dispatches sorted by abs_mean_hz")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)

    xticks = np.arange(0, len(ranked), 4)
    axes[1].set_xticks(xticks, ranked.loc[xticks, "label"], rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hour-start", type=int, default=0)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--dispatches-per-hour", type=int, default=DEFAULT_DISPATCHES_PER_HOUR)
    parser.add_argument("--duration-seconds", type=int, default=DEFAULT_DISPATCH_INTERVAL)
    parser.add_argument("--agc-interval", type=int, default=4)
    parser.add_argument("--kp", type=float, default=0.03)
    parser.add_argument("--ki", type=float, default=0.01)
    parser.add_argument("--init-mode", choices=("dispatch", "first"),
                        default="first")
    parser.add_argument("--retry-init-mode", choices=("dispatch", "first"), default="dispatch",
                        help="Fallback init mode for very early TDS failures.")
    parser.add_argument("--retry-early-fail-seconds", type=int, default=10,
                        help="Retry only when the first TDS failure happens by this second.")
    parser.add_argument("--opf-case", type=Path, default=rdt.DEFAULT_OPF_CASE)
    parser.add_argument("--dyn-case", type=Path, default=rdt.DEFAULT_DYN_CASE)
    parser.add_argument("--stable-dyn-case", type=Path, default=rdt.DEFAULT_STABLE_DYN_CASE)
    parser.add_argument("--curve-file", type=Path, default=rdt.DEFAULT_CURVE_FILE)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--force", action="store_true",
                        help="Re-run dispatches even if JSON and CSV outputs already exist.")
    parser.add_argument("--save-series-plots", action="store_true",
                        help="Save individual PNG plots in addition to per-dispatch CSV files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    total_dispatches = args.hours * args.dispatches_per_hour
    expected_seconds = total_dispatches * args.duration_seconds
    if expected_seconds > DAY_SECONDS:
        raise ValueError(
            f"Requested window spans {expected_seconds} seconds, which exceeds one day."
        )

    args.results_dir = args.results_dir or build_default_results_dir(args.kp, args.ki, args.agc_interval)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    dyn_case = rdt.adapt_dyn_case(args.dyn_case, args.stable_dyn_case)
    tasks = enumerate_dispatches(args.hour_start, args.hours, args.dispatches_per_hour)

    manifest = {
        "hour_start": args.hour_start,
        "hours": args.hours,
        "dispatches_per_hour": args.dispatches_per_hour,
        "duration_seconds": args.duration_seconds,
        "agc_interval": args.agc_interval,
        "kp": args.kp,
        "ki": args.ki,
        "init_mode": args.init_mode,
        "retry_init_mode": args.retry_init_mode,
        "retry_early_fail_seconds": args.retry_early_fail_seconds,
        "opf_case": str(args.opf_case),
        "dyn_case": str(args.dyn_case),
        "stable_dyn_case": str(dyn_case),
        "curve_file": str(args.curve_file),
        "results_dir": str(args.results_dir),
        "jobs": args.jobs,
    }
    (args.results_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    rows: list[dict[str, Any]] = []
    pending: list[tuple[int, int]] = []

    for hour, dispatch in tasks:
        label = f"h{hour}d{dispatch}"
        dispatch_json, freq_csv, _ = series_paths(args.results_dir, label)
        if not args.force and dispatch_json.exists() and freq_csv.exists():
            rows.append(load_existing_summary(label, args.results_dir))
        else:
            pending.append((hour, dispatch))

    print(
        f"results_dir={args.results_dir}\n"
        f"total_dispatches={len(tasks)}\n"
        f"reused={len(rows)}\n"
        f"to_run={len(pending)}\n"
        f"jobs={args.jobs}"
    )

    if pending:
        with ProcessPoolExecutor(
            max_workers=args.jobs,
            initializer=init_worker,
            initargs=(
                str(args.curve_file),
                str(args.opf_case),
                str(dyn_case),
                args.duration_seconds,
                args.agc_interval,
                args.kp,
                args.ki,
                args.init_mode,
                args.retry_init_mode,
                args.retry_early_fail_seconds,
                rdt.DEFAULT_WIND_PREFIXES,
                rdt.DEFAULT_SOLAR_PREFIXES,
                str(args.results_dir),
                args.save_series_plots,
            ),
        ) as pool:
            future_map = {pool.submit(run_one, task): task for task in pending}
            done = len(rows)
            total = len(tasks)
            for future in as_completed(future_map):
                hour, dispatch = future_map[future]
                label = f"h{hour}d{dispatch}"
                done += 1
                try:
                    row = future.result()
                    rows.append(row)
                    print(
                        f"[{done}/{total}] {label} ok "
                        f"min={row['min_hz']:.4f} "
                        f"max={row['max_hz']:.4f} "
                        f"abs_mean={row['abs_mean_hz']:.4f} "
                        f"total_s={row['total_seconds']:.2f}"
                    )
                except Exception as exc:
                    row = {
                        "hour": hour,
                        "dispatch": dispatch,
                        "label": label,
                        "success": 0,
                        "error": str(exc),
                        "dispatch_json": "",
                        "freq_csv": "",
                        "freq_png": "",
                        "dispatch_seconds": math.nan,
                        "tds_seconds": math.nan,
                        "total_seconds": math.nan,
                        "samples": math.nan,
                        "t_end_s": math.nan,
                        "min_hz": math.nan,
                        "t_min_s": math.nan,
                        "max_hz": math.nan,
                        "t_max_s": math.nan,
                        "final_hz": math.nan,
                        "abs_mean_hz": math.nan,
                        "rms_hz": math.nan,
                    }
                    rows.append(row)
                    print(f"[{done}/{total}] {label} fail error={exc}")

    summary = pd.DataFrame(rows).sort_values(["hour", "dispatch"]).reset_index(drop=True)
    summary_csv = args.results_dir / "daily_summary.csv"
    summary.to_csv(summary_csv, index=False)

    success = summary[summary["success"] == 1].copy()
    if len(success) == 0:
        raise RuntimeError(f"No successful dispatches. See {summary_csv}")

    overview_png = args.results_dir / "daily_compare_overview.png"
    heatmap_png = args.results_dir / "daily_compare_heatmap.png"
    hourly_png = args.results_dir / "daily_compare_hourly_grid.png"
    rank_png = args.results_dir / "daily_compare_rankings.png"

    make_overview_plot(overview_png, success, args.duration_seconds)
    make_heatmap(heatmap_png, success, args.duration_seconds)
    make_hourly_grid(hourly_png, success, args.duration_seconds)
    make_rank_plot(rank_png, success)

    top_abs = success.sort_values("abs_mean_hz", ascending=False).head(12)
    top_abs.to_csv(args.results_dir / "top_abs_mean_dispatches.csv", index=False)

    print(f"summary_csv={summary_csv}")
    print(f"overview_png={overview_png}")
    print(f"heatmap_png={heatmap_png}")
    print(f"hourly_png={hourly_png}")
    print(f"rank_png={rank_png}")
    print(f"successes={len(success)}")
    print(f"failures={len(summary) - len(success)}")


if __name__ == "__main__":
    main()
