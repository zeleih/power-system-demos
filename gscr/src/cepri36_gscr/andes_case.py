"""Build and run an auditable ANDES version of the CEPRI36 case."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .model import build_passive_ybus, infer_hvdc_terminal_equivalents
from .psasp import PSASPCase


SYSTEM_BASE_MVA = 100.0
NOMINAL_FREQUENCY_HZ = 50.0


@dataclass(frozen=True)
class AndesCaseBuild:
    workbook: Path
    model_counts: dict[str, int]
    assumptions: dict[str, object]


def _records(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in path.read_text(encoding="ascii", errors="replace").splitlines():
        if line.strip():
            row = next(csv.reader([line], skipinitialspace=True))
            while row and not row[-1].strip():
                row.pop()
            rows.append(row)
    return rows


def _generator_dispatch(case: PSASPCase) -> dict[int, complex]:
    return {
        int(row[0]): complex(float(row[1]), float(row[2]))
        for row in _records(case.source_dir / "LF.LP5")
        if len(row) >= 3
    }


def _with_uid(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "uid", np.arange(len(result), dtype=int))
    return result


def _safe_base_kv(case: PSASPCase, bus_index: int) -> float:
    base_kv = case.buses[bus_index - 1].base_kv
    return base_kv if base_kv > 0 else 1.0


def build_andes_workbook(
    case: PSASPCase,
    output_path: str | Path,
    *,
    classical_inertia_m: float = 8.0,
    classical_damping: float = 1.0,
) -> AndesCaseBuild:
    """Translate the PSASP steady-state records to an ANDES native workbook.

    PSASP stores line charging as the shunt susceptance at each end, while
    ANDES stores total charging in ``Line.b``. PSASP's transformer tap is on
    the second terminal; ANDES places it on ``bus1``, so transformer terminals
    are intentionally reversed.
    """

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    active_buses = [bus for bus in case.buses if not bus.name.upper().startswith("NULL")]
    bus_rows = []
    for bus in active_buses:
        voltage = case.solved_voltage[bus.index - 1]
        bus_rows.append(
            {
                "idx": bus.index,
                "u": 1,
                "name": bus.name,
                "Vn": _safe_base_kv(case, bus.index),
                "vmax": 1.2,
                "vmin": 0.8,
                "v0": abs(voltage),
                "a0": float(np.angle(voltage)),
                "area": bus.area if bus.area > 0 else None,
                "zone": None,
                "owner": None,
            }
        )

    area_ids = sorted({bus.area for bus in active_buses if bus.area > 0})
    area_rows = [{"idx": area, "u": 1, "name": f"AREA{area}"} for area in area_ids]

    line_rows = []
    shunt_rows = []
    for branch in case.branches:
        if branch.from_bus == branch.to_bus:
            impedance = complex(branch.resistance_pu, branch.reactance_pu)
            if 1e-12 < abs(impedance) < 1e5:
                admittance = 1.0 / impedance
                shunt_rows.append(
                    {
                        "idx": f"SH_{branch.record_id}",
                        "u": 1,
                        "name": branch.name,
                        "bus": branch.from_bus,
                        "Sn": SYSTEM_BASE_MVA,
                        "Vn": _safe_base_kv(case, branch.from_bus),
                        "g": admittance.real,
                        "b": admittance.imag,
                        "fn": NOMINAL_FREQUENCY_HZ,
                    }
                )
            continue

        line_rows.append(
            {
                "idx": f"AC_{branch.record_id}",
                "u": 1,
                "name": branch.name,
                "bus1": branch.from_bus,
                "bus2": branch.to_bus,
                "Sn": SYSTEM_BASE_MVA,
                "fn": NOMINAL_FREQUENCY_HZ,
                "Vn1": _safe_base_kv(case, branch.from_bus),
                "Vn2": _safe_base_kv(case, branch.to_bus),
                "r": branch.resistance_pu,
                "x": branch.reactance_pu,
                "b": 2.0 * branch.charging_each_end_pu,
                "g": 0.0,
                "b1": 0.0,
                "g1": 0.0,
                "b2": 0.0,
                "g2": 0.0,
                "trans": 0,
                "tap": 1.0,
                "phi": 0.0,
            }
        )

    for transformer in case.transformers:
        line_rows.append(
            {
                "idx": f"TR_{transformer.record_id}",
                "u": 1,
                "name": transformer.name,
                "bus1": transformer.to_bus,
                "bus2": transformer.from_bus,
                "Sn": SYSTEM_BASE_MVA,
                "fn": NOMINAL_FREQUENCY_HZ,
                "Vn1": _safe_base_kv(case, transformer.to_bus),
                "Vn2": _safe_base_kv(case, transformer.from_bus),
                "r": transformer.resistance_pu,
                "x": transformer.reactance_pu,
                "b": 0.0,
                "g": 0.0,
                "b1": 0.0,
                "g1": 0.0,
                "b2": 0.0,
                "g2": 0.0,
                "trans": 1,
                "tap": transformer.tap_second_side,
                "phi": 0.0,
            }
        )

    pq_rows = [
        {
            "idx": f"LOAD_{load.load_id}",
            "u": 1,
            "name": load.name,
            "bus": load.bus,
            "Vn": _safe_base_kv(case, load.bus),
            "p0": load.p_pu,
            "q0": load.q_pu,
            "vmax": 1.2,
            "vmin": 0.8,
            "owner": None,
        }
        for load in case.loads
    ]

    hvdc_equivalents, _ = infer_hvdc_terminal_equivalents(case, build_passive_ybus(case))
    for bus_index, load_power in hvdc_equivalents.items():
        pq_rows.append(
            {
                "idx": f"HVDC_EQ_{bus_index}",
                "u": 1,
                "name": f"HVDC terminal equivalent at {case.buses[bus_index - 1].name}",
                "bus": bus_index,
                "Vn": _safe_base_kv(case, bus_index),
                "p0": load_power.real,
                "q0": load_power.imag,
                "vmax": 1.2,
                "vmin": 0.8,
                "owner": None,
            }
        )

    dispatch = _generator_dispatch(case)
    generator_by_name = {generator.name.upper(): generator for generator in case.generators}
    pv_rows = []
    slack_rows = []
    gencls_rows = []
    for ordinal in range(1, 9):
        generator = generator_by_name[f"BUS{ordinal}"]
        bus = generator.bus
        operating_power = dispatch[bus]
        voltage = case.solved_voltage[bus - 1]
        static_idx = f"G{ordinal}"
        common = {
            "idx": static_idx,
            "u": 1,
            "name": generator.name,
            "Sn": generator.capacity_mva,
            "Vn": _safe_base_kv(case, bus),
            "bus": bus,
            "busr": None,
            "p0": operating_power.real,
            "q0": operating_power.imag,
            "pmax": generator.capacity_mva / SYSTEM_BASE_MVA,
            "pmin": -generator.capacity_mva / SYSTEM_BASE_MVA,
            "qmax": generator.capacity_mva / SYSTEM_BASE_MVA,
            "qmin": -generator.capacity_mva / SYSTEM_BASE_MVA,
            "v0": abs(voltage),
            "vmax": 1.4,
            "vmin": 0.6,
            "ra": 0.0,
            "xs": 0.25,
        }
        if ordinal == 1:
            slack_rows.append({**common, "a0": float(np.angle(voltage))})
        else:
            pv_rows.append(common)

        gencls_rows.append(
            {
                "idx": f"GENCLS_{ordinal}",
                "u": 1,
                "name": f"GENCLS {generator.name}",
                "bus": bus,
                "gen": static_idx,
                "coi": None,
                "coi2": None,
                "Sn": generator.capacity_mva,
                "Vn": _safe_base_kv(case, bus),
                "fn": NOMINAL_FREQUENCY_HZ,
                "D": classical_damping,
                "M": classical_inertia_m,
                "ra": 0.0,
                "xl": 0.15,
                "xd1": 0.25,
                "kp": 0.0,
                "kw": 0.0,
                "S10": 0.0,
                "S12": 1.0,
                "gammap": 1.0,
                "gammaq": 1.0,
            }
        )

    sheets = {
        "Bus": _with_uid(pd.DataFrame(bus_rows)),
        "PQ": _with_uid(pd.DataFrame(pq_rows)),
        "PV": _with_uid(pd.DataFrame(pv_rows)),
        "Slack": _with_uid(pd.DataFrame(slack_rows)),
        "Shunt": _with_uid(pd.DataFrame(shunt_rows)),
        "Line": _with_uid(pd.DataFrame(line_rows)),
        "Area": _with_uid(pd.DataFrame(area_rows)),
        "GENCLS": _with_uid(pd.DataFrame(gencls_rows)),
    }
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)

    return AndesCaseBuild(
        workbook=output,
        model_counts={name: len(frame) for name, frame in sheets.items()},
        assumptions={
            "system_base_mva": SYSTEM_BASE_MVA,
            "nominal_frequency_hz": NOMINAL_FREQUENCY_HZ,
            "dynamic_generator_model": "GENCLS",
            "classical_inertia_m": classical_inertia_m,
            "classical_damping": classical_damping,
            "hvdc_model": "steady-state two-terminal PQ equivalent",
            "tds_load_model": "ANDES PQ default constant-impedance conversion",
        },
    )
