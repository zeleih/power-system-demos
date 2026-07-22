from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = ROOT / "cases" / "cepri36"
sys.path.insert(0, str(ROOT / "src"))

from cepri36_gscr.identification import identify_symmetric_admittance
from cepri36_gscr.reference import load_reference_model
from cepri36_gscr.scenarios import frame_to_scenario


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Identify CEPRI36 gSCR from a PSASP-exported PMU CSV."
    )
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()

    model = load_reference_model(CASE_ROOT)
    data = frame_to_scenario(
        pd.read_csv(args.csv),
        model.bus_names,
        name=args.csv.stem,
    )
    result = identify_symmetric_admittance(
        data.voltage,
        data.current,
        model.capacities_pu,
    )
    summary = {
        "source": str(args.csv),
        "sample_count": result.sample_count,
        "analytical_identified_gscr": result.gscr,
        "direct_network_gscr": model.gscr,
        "absolute_error": abs(result.gscr - model.gscr),
        "residual_rmse_pu": result.residual_rmse,
        "real_parameter_rank": result.design_rank,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
