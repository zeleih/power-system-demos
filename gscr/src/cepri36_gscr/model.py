"""Network reconstruction, Kron reduction and direct gSCR calculation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .psasp import PSASPCase


@dataclass
class NetworkModel:
    full_y: np.ndarray
    reduced_y: np.ndarray
    retained_indices_zero_based: np.ndarray
    internal_indices_zero_based: np.ndarray
    capacities_pu: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def reduced_b(self) -> np.ndarray:
        return -self.reduced_y.imag

    @property
    def eigenvalues(self) -> np.ndarray:
        return generalized_scr(self.reduced_b, self.capacities_pu)[1]

    @property
    def gscr(self) -> float:
        return generalized_scr(self.reduced_b, self.capacities_pu)[0]


def build_passive_ybus(case: PSASPCase) -> np.ndarray:
    """Build the PSASP passive Ybus.

    LF.L2's charging value is the susceptance placed at *each* end. For LF.L3,
    the off-nominal tap is on the second side. Both conventions are verified
    against the stored PSASP branch-flow solution.
    """

    n = len(case.buses)
    ybus = np.zeros((n, n), dtype=complex)

    for branch in case.branches:
        i = branch.from_bus - 1
        j = branch.to_bus - 1
        z = complex(branch.resistance_pu, branch.reactance_pu)
        if i == j:
            if 1e-12 < abs(z) < 1e5:
                ybus[i, i] += 1.0 / z
            continue
        series_y = 1.0 / z
        charging_y = 1j * branch.charging_each_end_pu
        ybus[i, i] += series_y + charging_y
        ybus[j, j] += series_y + charging_y
        ybus[i, j] -= series_y
        ybus[j, i] -= series_y

    for transformer in case.transformers:
        i = transformer.from_bus - 1
        j = transformer.to_bus - 1
        series_y = 1.0 / complex(transformer.resistance_pu, transformer.reactance_pu)
        tap = transformer.tap_second_side
        ybus[i, i] += series_y
        ybus[j, j] += series_y / tap**2
        ybus[i, j] -= series_y / tap
        ybus[j, i] -= series_y / tap

    return ybus


def add_constant_impedance_loads(case: PSASPCase, ybus: np.ndarray) -> np.ndarray:
    result = ybus.copy()
    for load in case.loads:
        i = load.bus - 1
        voltage_sq = abs(case.solved_voltage[i]) ** 2
        if voltage_sq == 0:
            voltage_sq = 1.0
        result[i, i] += complex(load.p_pu, -load.q_pu) / voltage_sq
    return result


def infer_hvdc_terminal_equivalents(
    case: PSASPCase, passive_ybus: np.ndarray
) -> tuple[dict[int, complex], np.ndarray]:
    """Infer steady-state HVDC end powers from the solved AC nodal balance.

    The two CEPRI36 HVDC terminal buses have no other listed injection, so their
    passive-network residual is the converter injection. The returned complex
    values use the load convention P+jQ (positive means consumption).
    """

    if case.hvdc is None:
        return {}, np.zeros_like(passive_ybus)
    voltage = case.solved_voltage
    injection = voltage * np.conj(passive_ybus @ voltage)
    additions = np.zeros_like(passive_ybus)
    terminal_loads: dict[int, complex] = {}
    for bus in (case.hvdc.from_bus, case.hvdc.to_bus):
        i = bus - 1
        load_power = -injection[i]
        terminal_loads[bus] = load_power
        voltage_sq = abs(voltage[i]) ** 2
        additions[i, i] = np.conj(load_power) / voltage_sq
    return terminal_loads, additions


def kron_reduce(
    ybus: np.ndarray,
    retained_indices_zero_based: np.ndarray,
    internal_indices_zero_based: np.ndarray,
) -> np.ndarray:
    b = retained_indices_zero_based
    i = internal_indices_zero_based
    ybb = ybus[np.ix_(b, b)]
    ybi = ybus[np.ix_(b, i)]
    yii = ybus[np.ix_(i, i)]
    yib = ybus[np.ix_(i, b)]
    return ybb - ybi @ np.linalg.solve(yii, yib)


def generalized_scr(
    susceptance: np.ndarray, capacities_pu: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return gSCR, all generalized eigenvalues and normalized eigenvectors."""

    inv_sqrt = np.diag(1.0 / np.sqrt(capacities_pu))
    normalized = inv_sqrt @ susceptance @ inv_sqrt
    normalized = 0.5 * (normalized + normalized.T)
    values, vectors = np.linalg.eigh(normalized)
    return float(values[0]), values, vectors


def build_network_model(
    case: PSASPCase,
    *,
    include_loads: bool = True,
    include_hvdc: bool = True,
    system_base_mva: float = 100.0,
) -> NetworkModel:
    passive = build_passive_ybus(case)
    full_y = add_constant_impedance_loads(case, passive) if include_loads else passive.copy()
    hvdc_loads: dict[int, complex] = {}
    if include_hvdc:
        hvdc_loads, hvdc_additions = infer_hvdc_terminal_equivalents(case, passive)
        full_y += hvdc_additions

    retained = np.array([index - 1 for index in case.retained_bus_indices], dtype=int)
    active = np.array([index - 1 for index in case.active_bus_indices], dtype=int)
    retained_set = set(retained.tolist())
    internal = np.array([index for index in active if index not in retained_set], dtype=int)
    reduced = kron_reduce(full_y, retained, internal)

    generator_by_name = {g.name.upper(): g for g in case.generators}
    capacities = np.array(
        [generator_by_name[f"BUS{k}"].capacity_mva / system_base_mva for k in range(1, 9)],
        dtype=float,
    )

    return NetworkModel(
        full_y=full_y,
        reduced_y=reduced,
        retained_indices_zero_based=retained,
        internal_indices_zero_based=internal,
        capacities_pu=capacities,
        metadata={
            "system_base_mva": system_base_mva,
            "include_loads": include_loads,
            "include_hvdc": include_hvdc,
            "hvdc_terminal_loads_pu": hvdc_loads,
        },
    )
