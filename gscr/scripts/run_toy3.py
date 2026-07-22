from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gscr_demo import PMURun, generalized_scr, identify_symmetric_admittance


def main() -> None:
    conductance = np.array(
        [[0.30, -0.05, -0.02], [-0.05, 0.25, -0.04], [-0.02, -0.04, 0.20]]
    )
    susceptance = np.array(
        [[8.0, -2.0, -1.0], [-2.0, 7.0, -1.5], [-1.0, -1.5, 6.0]]
    )
    true_y = conductance - 1j * susceptance
    capacities_pu = np.array([1.5, 1.0, 0.8])

    rng = np.random.default_rng(20250701)
    voltage = rng.standard_normal((160, 3)) + 1j * rng.standard_normal((160, 3))
    current = voltage @ true_y.T
    identified = identify_symmetric_admittance([PMURun(voltage, current)])

    direct_gscr = generalized_scr(true_y, capacities_pu).value
    identified_gscr = generalized_scr(identified.y_hat, capacities_pu).value
    result = {
        "port_count": 3,
        "increment_count": identified.sample_increment_count,
        "voltage_rank": identified.voltage_rank,
        "relative_y_error": float(
            np.linalg.norm(identified.y_hat - true_y) / np.linalg.norm(true_y)
        ),
        "direct_gscr": direct_gscr,
        "identified_gscr": identified_gscr,
        "gscr_absolute_error": abs(identified_gscr - direct_gscr),
    }

    output = ROOT / "results" / "reference" / "toy3_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
