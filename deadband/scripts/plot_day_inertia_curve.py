#!/usr/bin/env python3
"""
Compute and plot dispatch-by-dispatch synchronous inertia metrics.

The raw dynamic data sheet stores ``M = 2H``. This script converts it to
``H`` and reports:

- ``sum_HSn_mva_s``: total synchronous stored kinetic energy proxy
- ``Heq_sync_s``: equivalent inertia constant on synchronous MVA base
- ``Heq_sys_s``: equivalent inertia constant on system MVA base
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "openandes-mpl"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_dispatch_tds as rdt


SYNC_SHEETS = ("GENROU", "GENROE", "GENSAL", "GENSAE", "GENCLS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--dyn-case", type=Path, default=rdt.DEFAULT_STABLE_DYN_CASE)
    parser.add_argument("--opf-case", type=Path, default=rdt.DEFAULT_OPF_CASE)
    parser.add_argument("--summary-name", type=str, default="daily_hotstart_summary.csv")
    parser.add_argument("--online-threshold", type=float, default=1e-4)
    return parser.parse_args()


def load_sync_meta(dyn_case: Path) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    for sheet in SYNC_SHEETS:
        try:
            df = pd.read_excel(dyn_case, sheet_name=sheet)
        except ValueError:
            continue
        if not {"u", "gen", "Sn", "M"}.issubset(df.columns):
            continue
        df = df[df["u"] == 1].copy()
        if df.empty:
            continue
        df["gen"] = df["gen"].astype(int)
        df["Sn"] = df["Sn"].astype(float)
        df["M"] = df["M"].astype(float)
        df["H"] = 0.5 * df["M"]
        df["model"] = sheet
        blocks.append(df[["gen", "Sn", "M", "H", "model"]])

    if not blocks:
        raise RuntimeError(f"No synchronous-machine sheets found in {dyn_case}")

    return pd.concat(blocks, ignore_index=True).drop_duplicates(subset=["gen"])


def load_system_mva(opf_case: Path) -> float:
    sp = rdt.make_sp(opf_case)
    return float(sp.config.mva)


def compute_metrics(
    summary: pd.DataFrame,
    sync_meta: pd.DataFrame,
    system_mva: float,
    online_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for _, row in summary.iterrows():
        label = str(row["label"])
        dispatch_json = Path(row["dispatch_json"])
        rec = rdt.DispatchRecord.from_json(dispatch_json)

        dispatch = pd.DataFrame({
            "gen": [int(x) for x in rec.gen],
            "pg_pu": [float(x) for x in rec.pg],
        })
        total_pg_pu = float(dispatch["pg_pu"].sum())

        df = sync_meta.merge(dispatch, on="gen", how="left").fillna({"pg_pu": 0.0})
        online = df["pg_pu"] > online_threshold

        sync_sn_online = float(df.loc[online, "Sn"].sum())
        sum_hsn = float((df.loc[online, "H"] * df.loc[online, "Sn"]).sum())
        sum_msn = float((df.loc[online, "M"] * df.loc[online, "Sn"]).sum())
        sync_pg_pu = float(df.loc[online, "pg_pu"].sum())

        rows.append({
            "label": label,
            "hour": int(rec.hour),
            "dispatch": int(rec.dispatch),
            "total_pg_pu": total_pg_pu,
            "total_pg_mw": total_pg_pu * system_mva,
            "sync_pg_pu": sync_pg_pu,
            "sync_pg_share": sync_pg_pu / total_pg_pu if total_pg_pu > 0.0 else np.nan,
            "sync_online_count": int(online.sum()),
            "sync_sn_online_mva": sync_sn_online,
            "sum_HSn_mva_s": sum_hsn,
            "sum_MSn_mva_s": sum_msn,
            "Heq_sync_s": sum_hsn / sync_sn_online if sync_sn_online > 0.0 else np.nan,
            "Heq_sys_s": sum_hsn / (system_mva * total_pg_pu) if total_pg_pu > 0.0 else np.nan,
        })

    return pd.DataFrame(rows).sort_values(["hour", "dispatch"]).reset_index(drop=True)


def add_generation_mix(results_dir: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    mix_csv = results_dir / "dispatch_generation_mix.csv"
    if not mix_csv.exists():
        return metrics
    mix = pd.read_csv(mix_csv)
    cols = [c for c in ("label", "vre_share", "nonsync_share", "conv_pg", "total_pg") if c in mix.columns]
    return metrics.merge(mix[cols], on="label", how="left")


def plot_metrics(results_dir: Path, df: pd.DataFrame) -> Path:
    x = np.arange(len(df))
    labels = df["label"].tolist()
    xticks = np.arange(0, len(df), 4)

    fig, axes = plt.subplots(3, 1, figsize=(17, 12), sharex=True)

    axes[0].plot(x, df["Heq_sys_s"], color="#0f5c78", linewidth=1.6)
    axes[0].set_ylabel("Heq_sys [s]")
    axes[0].set_title("System Equivalent Inertia Constant by Dispatch")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x, df["sum_HSn_mva_s"], color="#b24c2a", linewidth=1.6, label="Sum(H*Sn)")
    axes[1].set_ylabel("Sum(H*Sn) [MVA*s]")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, loc="upper right")

    axes[2].plot(x, df["sync_pg_share"], color="#2a7f3f", linewidth=1.5, label="Synchronous Pg share")
    if "vre_share" in df.columns:
        axes[2].plot(x, df["vre_share"], color="#7a4ba0", linewidth=1.3, label="VRE share")
    axes[2].set_ylabel("Share")
    axes[2].set_xlabel("Dispatch")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(frameon=False, loc="upper right")

    for ax in axes:
        for boundary in range(0, len(df), 4):
            ax.axvline(boundary - 0.5, color="#cccccc", linewidth=0.45, alpha=0.45)

    axes[2].set_xticks(xticks, [labels[i] for i in xticks], rotation=45, ha="right")
    fig.tight_layout()

    out = results_dir / "inertia_curve.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.results_dir / args.summary_name)
    sync_meta = load_sync_meta(args.dyn_case)
    system_mva = load_system_mva(args.opf_case)

    metrics = compute_metrics(summary, sync_meta, system_mva, args.online_threshold)
    metrics = add_generation_mix(args.results_dir, metrics)

    csv_path = args.results_dir / "dispatch_inertia_metrics.csv"
    metrics.to_csv(csv_path, index=False)
    png_path = plot_metrics(args.results_dir, metrics)

    print(f"system_mva={system_mva}")
    print(f"csv={csv_path}")
    print(f"png={png_path}")
    print(metrics[["sum_HSn_mva_s", "Heq_sync_s", "Heq_sys_s", "sync_online_count"]].describe().to_string())


if __name__ == "__main__":
    main()
