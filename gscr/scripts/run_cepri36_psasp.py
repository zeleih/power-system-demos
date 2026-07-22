from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = ROOT / "cases" / "cepri36"
sys.path.insert(0, str(ROOT / "src"))

from cepri36_gscr.identification import identify_symmetric_admittance
from cepri36_gscr.reference import load_archived_pmu, load_reference_model


if __name__ == "__main__":
    reference = load_reference_model(CASE_ROOT)
    scenarios = {}
    for name, relative_path in {
        "fault_bus30": "outputs/pmu/fault_bus30_pmu.csv",
        "load_bus50": "outputs/pmu/load_bus50_pmu.csv",
    }.items():
        data = load_archived_pmu(CASE_ROOT, relative_path)
        result = identify_symmetric_admittance(
            data.voltage,
            data.current,
            reference.capacities_pu,
        )
        scenarios[name] = {
            "sample_count": result.sample_count,
            "identified_gscr": result.gscr,
            "absolute_error_to_reference": abs(result.gscr - reference.gscr),
            "residual_rmse_pu": result.residual_rmse,
            "real_parameter_rank": result.design_rank,
        }
    print(
        json.dumps(
            {
                "mode": "archived PSASP-compatible PMU verification",
                "direct_reference_gscr": reference.gscr,
                "scenarios": scenarios,
                "raw_psasp_records_included": False,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
