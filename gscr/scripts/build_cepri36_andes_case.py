from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = ROOT / "cases" / "cepri36"
sys.path.insert(0, str(ROOT / "src"))

from cepri36_gscr.andes_case import build_andes_workbook
from cepri36_gscr.psasp import load_psasp_case


if __name__ == "__main__":
    case = load_psasp_case(CASE_ROOT / "data" / "raw_psasp")
    result = build_andes_workbook(
        case,
        CASE_ROOT / "data" / "andes" / "CEPRI36_andes.xlsx",
    )
    print(
        json.dumps(
            {
                "workbook": str(result.workbook),
                "model_counts": result.model_counts,
                "assumptions": result.assumptions,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
