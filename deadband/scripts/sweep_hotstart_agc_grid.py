#!/usr/bin/env python3
"""
Sweep AGC PI gains for one dispatch interval.

The sweep can run in two modes:

- cold start: no checkpoint is loaded, so each KP/KI case starts from the same
  dispatch initialization profile
- hot start: a checkpoint is loaded first, so controller / machine states are
  inherited from the previous interval boundary

This is intended for local studies such as "how does AGC gain ratio change the
h5d2 deadband-dominated behavior?" while keeping the run mode explicit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
if "MPLCONFIGDIR" not in os.environ:
    _mpl_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "openandes-mpl"
    _mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DISPATCH_SCRIPT = SCRIPT_DIR / "run_dispatch_hotstart.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-in", type=Path, default=None)
    parser.add_argument("--dispatch-json", type=Path, required=True)
    parser.add_argument("--next-dispatch-json", type=Path, required=True)
    parser.add_argument("--curve-file", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--kp-list", type=float, nargs="+", required=True)
    parser.add_argument("--ki-list", type=float, nargs="+", required=True)
    parser.add_argument("--agc-interval", type=int, default=4)
    parser.add_argument("--init-mode", choices=("dispatch", "first"), default="first")
    parser.add_argument("--agc-gov-output-ramp-frac-pmax-per-min", type=float, default=0.0)
    parser.add_argument("--agc-dg-output-ramp-frac-pmax-per-min", type=float, default=0.0)
    parser.add_argument(
        "--agc-anti-windup-mode",
        choices=("off", "freeze_on_saturation"),
        default="off",
    )
    parser.add_argument(
        "--governor-target-schedule",
        choices=("step", "boundary_ramp", "midpoint_trajectory", "ramp_limited_basepoint"),
        default="midpoint_trajectory",
    )
    parser.add_argument("--governor-basepoint-ramp-floor-frac-pmax-per-min", type=float, default=0.005)
    parser.add_argument("--governor-basepoint-ramp-gap-factor", type=float, default=1.25)
    parser.add_argument("--traditional-governor-deadband-hz", type=float, default=None)
    parser.add_argument("--der-deadband-hz", type=float, default=None)
    parser.add_argument("--der-base-ddn", type=float, default=None)
    parser.add_argument("--target-storage-share", type=float, default=None)
    parser.add_argument("--scale-esd1-ddn-with-storage", action="store_true")
    parser.add_argument("--no-save-plot", dest="save_plot", action="store_false")
    parser.set_defaults(save_plot=False)
    return parser.parse_args()


def fmt_token(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def run_one(args: argparse.Namespace, kp: float, ki: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    label = f"{args.dispatch_json.stem.replace('_dispatch', '')}_kp{fmt_token(kp)}_ki{fmt_token(ki)}"
    cmd = [
        sys.executable,
        str(RUN_DISPATCH_SCRIPT),
        "--dispatch-json",
        str(args.dispatch_json),
        "--next-dispatch-json",
        str(args.next_dispatch_json),
        "--label",
        label,
        "--results-dir",
        str(args.results_dir),
        "--curve-file",
        str(args.curve_file),
        "--agc-interval",
        str(args.agc_interval),
        "--agc-gov-output-ramp-frac-pmax-per-min",
        str(args.agc_gov_output_ramp_frac_pmax_per_min),
        "--agc-dg-output-ramp-frac-pmax-per-min",
        str(args.agc_dg_output_ramp_frac_pmax_per_min),
        "--agc-anti-windup-mode",
        args.agc_anti_windup_mode,
        "--kp",
        str(kp),
        "--ki",
        str(ki),
        "--init-mode",
        args.init_mode,
        "--apply-governor-targets",
        "--governor-target-schedule",
        args.governor_target_schedule,
        "--governor-basepoint-ramp-floor-frac-pmax-per-min",
        str(args.governor_basepoint_ramp_floor_frac_pmax_per_min),
        "--governor-basepoint-ramp-gap-factor",
        str(args.governor_basepoint_ramp_gap_factor),
        "--no-save-checkpoint",
    ]
    if args.checkpoint_in is not None:
        cmd.extend([
            "--checkpoint-in",
            str(args.checkpoint_in),
            "--allow-signature-mismatch",
            "--recompute-ace-raw-on-load",
        ])
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
    if args.save_plot:
        cmd.append("--save-plot")
    else:
        cmd.append("--no-save-plot")

    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"run failed for kp={kp}, ki={ki}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    summary_csv = args.results_dir / f"{label}_summary.csv"
    freq_csv = args.results_dir / f"{label}_frequency.csv"
    if not summary_csv.exists() or not freq_csv.exists():
        raise RuntimeError(f"missing outputs for kp={kp}, ki={ki}")

    summary = pd.read_csv(summary_csv)
    freq = pd.read_csv(freq_csv)
    return summary, freq


def make_metric_heatmap(
    ax,
    summary: pd.DataFrame,
    metric: str,
    title: str,
    kp_vals: list[float],
    ki_vals: list[float],
    cmap: str = "viridis",
) -> None:
    pivot = summary.pivot(index="ki", columns="kp", values=metric).reindex(index=ki_vals, columns=kp_vals)
    data = pivot.to_numpy(dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
        if math.isclose(vmin, vmax):
            vmin -= 1e-9
            vmax += 1e-9
    im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(range(len(kp_vals)), [f"{v:.3f}" for v in kp_vals], rotation=45, ha="right")
    ax.set_yticks(range(len(ki_vals)), [f"{v:.3f}" for v in ki_vals])
    ax.set_xlabel("KP")
    ax.set_ylabel("KI")
    for i, ki in enumerate(ki_vals):
        for j, kp in enumerate(kp_vals):
            value = data[i, j]
            text = "nan" if np.isnan(value) else f"{value:.3f}"
            ax.text(j, i, text, ha="center", va="center", color="white", fontsize=7.5)
    plt.colorbar(im, ax=ax, shrink=0.82)


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    resume_mode = "hotstart" if args.checkpoint_in is not None else "cold"

    rows: list[dict[str, object]] = []
    total = len(args.kp_list) * len(args.ki_list)
    run_no = 0

    for kp in args.kp_list:
        for ki in args.ki_list:
            run_no += 1
            print(f"[{run_no}/{total}] kp={kp:.3f}, ki={ki:.3f}", flush=True)
            summary_df, freq_df = run_one(args, kp, ki)
            row = summary_df.iloc[0].to_dict()
            f = freq_df["freq_dev_hz"].to_numpy(dtype=float)
            df = np.diff(f)
            row["mean_abs_hz"] = float(np.mean(np.abs(f)))
            row["share_abs_gt_0p036"] = float(np.mean(np.abs(f) > 0.036))
            row["share_abs_gt_0p05"] = float(np.mean(np.abs(f) > 0.05))
            row["diff_abs_mean_hz"] = float(np.mean(np.abs(df))) if df.size else 0.0
            row["diff_std_hz"] = float(np.std(df)) if df.size else 0.0
            row["peak_abs_hz"] = float(np.max(np.abs(f)))
            rows.append(row)
            print(
                f"  final={float(row['final_hz']):+.4f} "
                f"mean|f|={float(row['mean_abs_hz']):.4f} "
                f"edge={float(row['share_abs_gt_0p036']):.2%} "
                f"noise={float(row['diff_abs_mean_hz']):.4f}",
                flush=True,
            )

    summary = pd.DataFrame(rows).sort_values(["kp", "ki"]).reset_index(drop=True)
    summary_csv = args.results_dir / "sweep_summary.csv"
    summary.to_csv(summary_csv, index=False)

    kp_vals = sorted(float(v) for v in summary["kp"].unique())
    ki_vals = sorted(float(v) for v in summary["ki"].unique())

    heatmap_png = args.results_dir / "sweep_heatmaps.png"
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
    make_metric_heatmap(axes[0, 0], summary, "mean_abs_hz", "Mean |f| [Hz]", kp_vals, ki_vals, cmap="viridis")
    make_metric_heatmap(axes[0, 1], summary, "share_abs_gt_0p036", "share(|f| > 0.036)", kp_vals, ki_vals, cmap="magma")
    make_metric_heatmap(axes[1, 0], summary, "diff_abs_mean_hz", "Mean |Δf| [Hz/sample]", kp_vals, ki_vals, cmap="plasma")
    make_metric_heatmap(axes[1, 1], summary, "final_hz", "Final f [Hz]", kp_vals, ki_vals, cmap="coolwarm")
    fig.suptitle(f"{resume_mode.capitalize()}-start AGC gain sweep", fontsize=14)
    fig.tight_layout()
    fig.savefig(heatmap_png, dpi=220)
    plt.close(fig)

    config = {
        "resume_mode": resume_mode,
        "checkpoint_in": "" if args.checkpoint_in is None else str(args.checkpoint_in),
        "dispatch_json": str(args.dispatch_json),
        "next_dispatch_json": str(args.next_dispatch_json),
        "curve_file": str(args.curve_file),
        "agc_interval": int(args.agc_interval),
        "init_mode": args.init_mode,
        "governor_target_schedule": args.governor_target_schedule,
        "governor_basepoint_ramp_floor_frac_pmax_per_min": args.governor_basepoint_ramp_floor_frac_pmax_per_min,
        "governor_basepoint_ramp_gap_factor": args.governor_basepoint_ramp_gap_factor,
        "agc_gov_output_ramp_frac_pmax_per_min": args.agc_gov_output_ramp_frac_pmax_per_min,
        "agc_dg_output_ramp_frac_pmax_per_min": args.agc_dg_output_ramp_frac_pmax_per_min,
        "agc_anti_windup_mode": args.agc_anti_windup_mode,
        "traditional_governor_deadband_hz": args.traditional_governor_deadband_hz,
        "der_deadband_hz": args.der_deadband_hz,
        "der_base_ddn": args.der_base_ddn,
        "target_storage_share": args.target_storage_share,
        "scale_esd1_ddn_with_storage": bool(args.scale_esd1_ddn_with_storage),
        "kp_list": list(map(float, args.kp_list)),
        "ki_list": list(map(float, args.ki_list)),
    }
    config_json = args.results_dir / "sweep_config.json"
    config_json.write_text(json.dumps(config, indent=2))

    print(f"summary_csv={summary_csv}")
    print(f"heatmap_png={heatmap_png}")
    print(f"config_json={config_json}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
