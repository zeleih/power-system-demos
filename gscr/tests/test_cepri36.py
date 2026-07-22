from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = ROOT / "cases" / "cepri36"
sys.path.insert(0, str(ROOT / "src"))

from cepri36_gscr.identification import identify_symmetric_admittance
from cepri36_gscr.reference import load_archived_pmu, load_reference_model


class CEPRI36Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_reference_model(CASE_ROOT)

    def test_published_reference_model(self) -> None:
        self.assertEqual(self.model.bus_names, [f"BUS{k}" for k in range(1, 9)])
        self.assertTrue(np.all(self.model.capacities_pu > 0.0))
        relative_asymmetry = np.linalg.norm(
            self.model.reduced_y - self.model.reduced_y.T
        ) / np.linalg.norm(self.model.reduced_y)
        self.assertLess(relative_asymmetry, 1e-12)

    def test_direct_and_identified_gscr(self) -> None:
        self.assertAlmostEqual(self.model.gscr, 0.1714094976, places=9)
        sampled = load_archived_pmu(
            CASE_ROOT,
            "outputs/pmu/fault_bus30_pmu.csv",
        )
        identified = identify_symmetric_admittance(
            sampled.voltage,
            sampled.current,
            self.model.capacities_pu,
        )
        relative_y_error = np.linalg.norm(
            identified.y_hat - self.model.reduced_y
        ) / np.linalg.norm(self.model.reduced_y)
        self.assertEqual(identified.design_rank, 72)
        self.assertLess(relative_y_error, 1e-9)
        self.assertLess(abs(identified.gscr - self.model.gscr), 1e-9)


if __name__ == "__main__":
    unittest.main()
