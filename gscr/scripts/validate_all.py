from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    toy = load(ROOT / "results" / "reference" / "toy3_summary.json")
    cepri = load(ROOT / "cases" / "cepri36" / "outputs" / "summary.json")
    cepri_andes = load(
        ROOT / "cases" / "cepri36" / "outputs" / "andes" / "summary.json"
    )
    il200 = load(ROOT / "cases" / "il200" / "outputs" / "summary.json")

    bus30 = next(
        row for row in cepri_andes["scenarios"] if row["scenario"] == "fault_bus30"
    )
    checks = {
        "toy Y recovery": toy["relative_y_error"] < 1e-12,
        "toy gSCR recovery": toy["gscr_absolute_error"] < 1e-12,
        "CEPRI36 direct benchmark": abs(cepri["rebuilt_network_gscr"] - 0.1714094976)
        < 1e-9,
        "CEPRI36 analytical recovery": abs(
            cepri["clean_fault_identified_mean"] - cepri["rebuilt_network_gscr"]
        )
        < 1e-9,
        "CEPRI36 ANDES Bus30": bus30["absolute_error_to_andes_direct"] < 1e-4,
        "IL200 standard benchmark": abs(
            il200["standard_short_circuit_gscr_xd2_without_load"] - 0.8109979585
        )
        < 1e-8,
        "IL200 analytical identification": il200["pmu_identified_absolute_error"]
        < 1e-3,
        "IL200 final IBR ports": il200["final_gscr_ports"]["count"] == 11,
    }
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
