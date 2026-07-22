"""Minimal, auditable parser for the PSASP text records used by CEPRI36v7."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _clean(value: str) -> str:
    return value.strip().strip("'").strip()


def _records(path: Path) -> list[list[str]]:
    records: list[list[str]] = []
    for line in path.read_text(encoding="ascii", errors="replace").splitlines():
        if line.strip():
            record = next(csv.reader([line], skipinitialspace=True))
            while record and not record[-1].strip():
                record.pop()
            records.append(record)
    return records


@dataclass(frozen=True)
class Bus:
    index: int
    name: str
    base_kv: float
    area: int


@dataclass(frozen=True)
class ACBranch:
    from_bus: int
    to_bus: int
    record_id: int
    resistance_pu: float
    reactance_pu: float
    charging_each_end_pu: float
    name: str


@dataclass(frozen=True)
class Transformer:
    from_bus: int
    to_bus: int
    record_id: int
    resistance_pu: float
    reactance_pu: float
    tap_second_side: float
    name: str


@dataclass(frozen=True)
class Load:
    bus: int
    load_id: int
    p_pu: float
    q_pu: float
    name: str


@dataclass(frozen=True)
class Generator:
    bus: int
    ordinal: int
    capacity_mva: float
    rated_mw: float
    name: str


@dataclass(frozen=True)
class HVDCLink:
    from_bus: int
    to_bus: int
    link_id: int
    name: str


@dataclass
class PSASPCase:
    source_dir: Path
    buses: list[Bus]
    branches: list[ACBranch]
    transformers: list[Transformer]
    loads: list[Load]
    generators: list[Generator]
    hvdc: HVDCLink | None
    solved_voltage: np.ndarray

    @property
    def bus_names(self) -> list[str]:
        return [bus.name for bus in self.buses]

    @property
    def active_bus_indices(self) -> list[int]:
        return [bus.index for bus in self.buses if not bus.name.upper().startswith("NULL")]

    @property
    def retained_bus_indices(self) -> list[int]:
        by_name = {generator.name.upper(): generator.bus for generator in self.generators}
        return [by_name[f"BUS{k}"] for k in range(1, 9)]

    @property
    def retained_bus_names(self) -> list[str]:
        return [f"BUS{k}" for k in range(1, 9)]

    def index_of(self, name: str) -> int:
        target = name.upper()
        for bus in self.buses:
            if bus.name.upper() == target:
                return bus.index
        raise KeyError(f"Unknown PSASP bus: {name}")


def load_psasp_case(source_dir: str | Path) -> PSASPCase:
    """Load only the documented records needed for the reproduction."""

    source = Path(source_dir)
    buses = [
        Bus(index=i, name=_clean(row[0]), base_kv=float(row[1]), area=int(row[2]))
        for i, row in enumerate(_records(source / "LF.L1"), start=1)
    ]

    branches = [
        ACBranch(
            from_bus=int(row[1]),
            to_bus=int(row[2]),
            record_id=int(row[3]),
            resistance_pu=float(row[4]),
            reactance_pu=float(row[5]),
            charging_each_end_pu=float(row[6]),
            name=_clean(row[-1]),
        )
        for row in _records(source / "LF.L2")
        if int(row[0]) == 1
    ]

    transformers = [
        Transformer(
            from_bus=abs(int(row[1])),
            to_bus=int(row[2]),
            record_id=int(row[3]),
            resistance_pu=float(row[4]),
            reactance_pu=float(row[5]),
            tap_second_side=float(row[6]),
            name=_clean(row[-1]),
        )
        for row in _records(source / "LF.L3")
        if int(row[0]) == 1
    ]

    loads = [
        Load(
            bus=int(row[1]),
            load_id=int(row[2]),
            p_pu=float(row[4]),
            q_pu=float(row[5]),
            name=_clean(row[-1]),
        )
        for row in _records(source / "LF.L6")
        if int(row[0]) == 1
    ]

    generators = [
        Generator(
            bus=int(row[1]),
            ordinal=int(row[3]),
            capacity_mva=float(row[-3]),
            rated_mw=float(row[-2]),
            name=_clean(row[-1]),
        )
        for row in _records(source / "ST.S5")
        if int(row[0]) == 1
    ]

    voltage = np.zeros(len(buses), dtype=complex)
    for row in _records(source / "LF.LP1"):
        if len(row) < 3:
            continue
        try:
            bus = int(row[0])
        except ValueError:
            continue
        magnitude = float(row[1])
        angle_rad = np.deg2rad(float(row[2]))
        voltage[bus - 1] = magnitude * np.exp(1j * angle_rad)

    hvdc_records = _records(source / "LF.L4") if (source / "LF.L4").exists() else []
    hvdc = None
    if hvdc_records and len(hvdc_records[0]) >= 4:
        row = hvdc_records[0]
        hvdc = HVDCLink(
            from_bus=int(row[1]),
            to_bus=int(row[2]),
            link_id=int(row[3]),
            name=_clean(row[-1]),
        )

    return PSASPCase(
        source_dir=source,
        buses=buses,
        branches=branches,
        transformers=transformers,
        loads=loads,
        generators=generators,
        hvdc=hvdc,
        solved_voltage=voltage,
    )
