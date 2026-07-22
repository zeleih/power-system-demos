from __future__ import annotations

from pathlib import Path
import sys
import unittest

import andes


ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = ROOT / "cases" / "il200"
sys.path.insert(0, str(ROOT / "src"))

from il200_gscr.network import build_direct_network, generalized_scr


class IL200Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        system = andes.load(
            str(CASE_ROOT / "data" / "IL200_ffr.xlsx"),
            setup=False,
            no_output=True,
            pycode_path=str(CASE_ROOT / ".andes" / "pycode"),
        )
        if not system.setup() or not system.PFlow.run():
            raise RuntimeError("failed to initialize IL200")
        cls.system = system

    def test_final_ports_and_standard_gscr(self) -> None:
        direct = build_direct_network(
            self.system,
            include_loads=False,
            sg_reactance="xd2",
        )
        self.assertEqual(len(direct.sg_buses), 38)
        self.assertEqual(len(direct.ibr_buses), 11)
        self.assertEqual(
            direct.ibr_buses.tolist(),
            [65, 104, 105, 114, 115, 125, 126, 127, 135, 136, 147],
        )
        result = generalized_scr(direct.ibr_y, direct.ibr_capacities_pu)
        self.assertAlmostEqual(result.value, 0.8109979585, places=8)


if __name__ == "__main__":
    unittest.main()
