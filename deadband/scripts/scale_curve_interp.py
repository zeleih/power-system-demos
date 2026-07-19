#!/usr/bin/env python3
"""
Scale the deadband-demo interpolated curve without modifying the original file.

Typical use:

    python scale_curve_interp.py \
        --input cases/CurveInterp.csv \
        --output cases/CurveInterp_load105_wind115_pv120.csv \
        --load-scale 1.05 \
        --wind-scale 1.15 \
        --pv-scale 1.20

The script keeps the ``Time`` column intact, scales ``Load`` / ``Wind`` /
``PV`` independently, clips negative values to zero, and can optionally cap
columns from above when building a bounded scenario.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_COLUMNS = ("Load", "Wind", "PV")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--load-scale", type=float, default=1.0)
    parser.add_argument("--wind-scale", type=float, default=1.0)
    parser.add_argument("--pv-scale", type=float, default=1.0)
    parser.add_argument("--load-cap", type=float, default=None)
    parser.add_argument("--wind-cap", type=float, default=None)
    parser.add_argument("--pv-cap", type=float, default=None)
    parser.add_argument("--round-digits", type=int, default=12)
    return parser.parse_args()


def apply_scale(series: pd.Series, scale: float, cap: float | None, digits: int) -> pd.Series:
    out = series.astype(float) * float(scale)
    out = out.clip(lower=0.0)
    if cap is not None:
        out = out.clip(upper=float(cap))
    return out.round(digits)


def describe_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in DEFAULT_COLUMNS:
        rows.append({
            "column": col,
            "min": float(df[col].min()),
            "mean": float(df[col].mean()),
            "max": float(df[col].max()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)
    required = {"Time", *DEFAULT_COLUMNS}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    before = describe_columns(df)

    out = df.copy()
    out["Load"] = apply_scale(out["Load"], args.load_scale, args.load_cap, args.round_digits)
    out["Wind"] = apply_scale(out["Wind"], args.wind_scale, args.wind_cap, args.round_digits)
    out["PV"] = apply_scale(out["PV"], args.pv_scale, args.pv_cap, args.round_digits)

    after = describe_columns(out)
    compare = before.merge(after, on="column", suffixes=("_before", "_after"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"input={args.input}")
    print(f"output={args.output}")
    print(
        "scales="
        f"load:{args.load_scale:.4f}, "
        f"wind:{args.wind_scale:.4f}, "
        f"pv:{args.pv_scale:.4f}"
    )
    print(compare.to_string(index=False))


if __name__ == "__main__":
    main()
