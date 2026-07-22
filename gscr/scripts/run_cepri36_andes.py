from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = ROOT / "cases" / "cepri36"
sys.path.insert(0, str(ROOT / "src"))

from cepri36_gscr.andes_experiments import run_andes_experiments
from cepri36_gscr.identification import identify_symmetric_admittance
from cepri36_gscr.reference import load_archived_pmu, load_reference_model


if __name__ == "__main__":
    raw = CASE_ROOT / "data" / "raw_psasp"
    if (raw / "LF.L1").is_file():
        result = run_andes_experiments(CASE_ROOT)
    else:
        reference = load_reference_model(CASE_ROOT)
        archived = json.loads(
            (CASE_ROOT / "outputs" / "andes" / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        direct_by_scenario = {
            row["scenario"]: row["andes_direct_gscr"]
            for row in archived["scenarios"]
        }
        scenarios = {}
        for name in direct_by_scenario:
            data = load_archived_pmu(
                CASE_ROOT,
                f"outputs/andes/pmu/{name}_pmu.csv",
            )
            identified = identify_symmetric_admittance(
                data.voltage,
                data.current,
                reference.capacities_pu,
            )
            direct = direct_by_scenario[name]
            scenarios[name] = {
                "sample_count": identified.sample_count,
                "andes_direct_gscr": direct,
                "identified_gscr": identified.gscr,
                "absolute_error_to_andes_direct": abs(identified.gscr - direct),
                "residual_rmse_pu": identified.residual_rmse,
                "real_parameter_rank": identified.design_rank,
            }
        result = {
            "mode": "archived ANDES PMU verification",
            "engine_used_for_archive": archived["engine"],
            "scenarios": scenarios,
            "full_tds_regeneration": False,
            "regeneration_note": (
                "Supply authorized local PSASP records under "
                "cases/cepri36/data/raw_psasp to rebuild and rerun ANDES TDS."
            ),
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))
