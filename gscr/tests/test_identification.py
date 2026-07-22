from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gscr_demo import PMURun, generalized_scr, identify_symmetric_admittance


class AnalyticalIdentificationTests(unittest.TestCase):
    def test_known_symmetric_three_port_network(self) -> None:
        true_y = np.array(
            [
                [0.30 - 8.0j, -0.05 + 2.0j, -0.02 + 1.0j],
                [-0.05 + 2.0j, 0.25 - 7.0j, -0.04 + 1.5j],
                [-0.02 + 1.0j, -0.04 + 1.5j, 0.20 - 6.0j],
            ]
        )
        capacities = np.array([1.5, 1.0, 0.8])
        rng = np.random.default_rng(20250701)
        voltage = rng.standard_normal((160, 3)) + 1j * rng.standard_normal((160, 3))
        current = voltage @ true_y.T

        result = identify_symmetric_admittance([PMURun(voltage, current)])
        self.assertEqual(result.voltage_rank, 3)
        self.assertEqual(result.parameter_rank_real, 12)
        np.testing.assert_allclose(result.y_hat, true_y, atol=1e-11, rtol=1e-11)
        self.assertAlmostEqual(
            generalized_scr(result.y_hat, capacities).value,
            generalized_scr(true_y, capacities).value,
            places=12,
        )

    def test_multiple_batches_are_accumulated(self) -> None:
        rng = np.random.default_rng(17)
        raw = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        true_y = 0.5 * (raw + raw.T)
        runs = []
        for _ in range(3):
            voltage = rng.standard_normal((60, 4)) + 1j * rng.standard_normal((60, 4))
            runs.append(PMURun(voltage, voltage @ true_y.T))

        result = identify_symmetric_admittance(iter(runs))
        self.assertEqual(result.sample_increment_count, 177)
        np.testing.assert_allclose(result.y_hat, true_y, atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
