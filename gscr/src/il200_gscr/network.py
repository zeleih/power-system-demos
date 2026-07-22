"""Network reduction and IBR-port gSCR calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from andes.linsolvers.scipy import spmatrix_to_csc
from scipy.linalg import eigvalsh, eigh


@dataclass(frozen=True)
class GSCRResult:
    value: float
    eigenvalues: np.ndarray
    critical_mode: np.ndarray
    participation: np.ndarray
    normalized_susceptance: np.ndarray


@dataclass(frozen=True)
class DirectNetwork:
    full_y: np.ndarray
    source_port_y: np.ndarray
    ibr_y: np.ndarray
    sg_norton_y: np.ndarray
    source_port_buses: np.ndarray
    sg_buses: np.ndarray
    ibr_buses: np.ndarray
    sg_model_positions: np.ndarray
    ibr_model_positions: np.ndarray
    ibr_capacities_mva: np.ndarray
    ibr_capacities_pu: np.ndarray
    include_loads: bool
    sg_reactance: str
    direct_reduction_relative_error: float


def kron_reduce(ybus: np.ndarray, keep: np.ndarray | list[int]) -> np.ndarray:
    """Kron-reduce a nodal admittance matrix to zero-based ``keep`` indices."""

    ybus = np.asarray(ybus, dtype=complex)
    keep = np.asarray(keep, dtype=int)
    if ybus.ndim != 2 or ybus.shape[0] != ybus.shape[1]:
        raise ValueError("ybus must be square")
    if len(np.unique(keep)) != len(keep):
        raise ValueError("keep indices must be unique")
    drop = np.setdiff1d(np.arange(ybus.shape[0]), keep)
    if not len(drop):
        return ybus.copy()
    return ybus[np.ix_(keep, keep)] - ybus[np.ix_(keep, drop)] @ np.linalg.solve(
        ybus[np.ix_(drop, drop)], ybus[np.ix_(drop, keep)]
    )


def terminate_synchronous_ports(
    source_port_y: np.ndarray,
    sg_norton_y: np.ndarray,
    n_synchronous: int,
) -> np.ndarray:
    """Terminate synchronous-generator terminal ports and retain IBR ports.

    Source ports must be ordered as synchronous machines followed by IBRs.
    Internal synchronous-machine voltage increments are suppressed while their
    Norton admittances are retained.
    """

    source_port_y = np.asarray(source_port_y, dtype=complex)
    sg_norton_y = np.asarray(sg_norton_y, dtype=complex)
    if source_port_y.shape[0] != source_port_y.shape[1]:
        raise ValueError("source_port_y must be square")
    if sg_norton_y.shape != (n_synchronous, n_synchronous):
        raise ValueError("sg_norton_y has incompatible dimensions")
    yss = source_port_y[:n_synchronous, :n_synchronous]
    ysr = source_port_y[:n_synchronous, n_synchronous:]
    yrs = source_port_y[n_synchronous:, :n_synchronous]
    yrr = source_port_y[n_synchronous:, n_synchronous:]
    return yrr - yrs @ np.linalg.solve(yss + sg_norton_y, ysr)


def generalized_scr(y_ibr: np.ndarray, capacities_pu: np.ndarray) -> GSCRResult:
    """Calculate classical capacity-normalized gSCR at IBR terminals."""

    y_ibr = np.asarray(y_ibr, dtype=complex)
    capacities_pu = np.asarray(capacities_pu, dtype=float)
    if y_ibr.shape != (len(capacities_pu), len(capacities_pu)):
        raise ValueError("capacity vector must match y_ibr")
    if np.any(capacities_pu <= 0):
        raise ValueError("all IBR capacities must be positive")

    susceptance = -0.5 * (y_ibr.imag + y_ibr.imag.T)
    scale = np.diag(1.0 / np.sqrt(capacities_pu))
    normalized = scale @ susceptance @ scale
    eigenvalues, eigenvectors = eigh(normalized)
    critical = eigenvectors[:, 0]
    participation = np.square(np.abs(critical))
    participation /= participation.sum()
    return GSCRResult(
        value=float(eigenvalues[0]),
        eigenvalues=eigenvalues,
        critical_mode=critical,
        participation=participation,
        normalized_susceptance=normalized,
    )


def _active_positions(model) -> np.ndarray:
    return np.flatnonzero(np.asarray(model.u.v, dtype=float) > 0.5)


def _constant_impedance_load_y(system, bus_position: dict[int, int]) -> np.ndarray:
    """Return the load admittance used by ANDES default TDS PQ conversion."""

    yload = np.zeros((system.Bus.n, system.Bus.n), dtype=complex)
    for row in _active_positions(system.PQ):
        bus = int(system.PQ.bus.v[row])
        position = bus_position[bus]
        voltage = float(system.Bus.v.v[position])
        vmin = float(system.PQ.vmin.v[row])
        vmax = float(system.PQ.vmax.v[row])
        conversion_voltage = float(np.clip(voltage, vmin, vmax))
        p = float(system.PQ.p0.v[row])
        q = float(system.PQ.q0.v[row])
        yload[position, position] += complex(p, -q) / conversion_voltage**2
    return yload


def _generator_norton_y(system, positions: np.ndarray, reactance: str) -> np.ndarray:
    if reactance not in {"xd1", "xd2"}:
        raise ValueError("reactance must be 'xd1' or 'xd2'")
    sn = np.asarray(system.GENROU.Sn.v, dtype=float)[positions]
    ra = np.asarray(system.GENROU.ra.v, dtype=float)[positions]
    x = np.asarray(getattr(system.GENROU, reactance).v, dtype=float)[positions]
    machine_base_admittance = 1.0 / (ra + 1j * x)
    system_base_admittance = (sn / float(system.config.mva)) * machine_base_admittance
    return np.diag(system_base_admittance)


def build_direct_network(
    system,
    *,
    include_loads: bool,
    sg_reactance: str = "xd2",
) -> DirectNetwork:
    """Build the 49-source-port and final 11-IBR-port equivalents."""

    bus_position = {int(bus): row for row, bus in enumerate(system.Bus.idx.v)}
    full_y = spmatrix_to_csc(system.build_ybus()).toarray()
    if include_loads:
        full_y = full_y + _constant_impedance_load_y(system, bus_position)

    sg_positions = _active_positions(system.GENROU)
    ibr_positions = _active_positions(system.REGCA1)
    sg_buses = np.asarray(system.GENROU.bus.v, dtype=int)[sg_positions]
    ibr_buses = np.asarray(system.REGCA1.bus.v, dtype=int)[ibr_positions]
    source_buses = np.concatenate([sg_buses, ibr_buses])
    if len(np.unique(source_buses)) != len(source_buses):
        raise ValueError("this implementation expects one active source per terminal bus")
    source_indices = np.asarray([bus_position[int(bus)] for bus in source_buses])
    ibr_indices = np.asarray([bus_position[int(bus)] for bus in ibr_buses])

    source_port_y = kron_reduce(full_y, source_indices)
    sg_norton_y = _generator_norton_y(system, sg_positions, sg_reactance)
    ibr_y = terminate_synchronous_ports(source_port_y, sg_norton_y, len(sg_buses))

    augmented = full_y.copy()
    for source_position, bus in enumerate(sg_buses):
        augmented[bus_position[int(bus)], bus_position[int(bus)]] += sg_norton_y[
            source_position, source_position
        ]
    direct_ibr_y = kron_reduce(augmented, ibr_indices)
    relative_error = float(
        np.linalg.norm(ibr_y - direct_ibr_y) / max(np.linalg.norm(direct_ibr_y), 1e-15)
    )

    capacities_mva = np.asarray(system.REGCA1.Sn.v, dtype=float)[ibr_positions]
    return DirectNetwork(
        full_y=full_y,
        source_port_y=source_port_y,
        ibr_y=ibr_y,
        sg_norton_y=sg_norton_y,
        source_port_buses=source_buses,
        sg_buses=sg_buses,
        ibr_buses=ibr_buses,
        sg_model_positions=sg_positions,
        ibr_model_positions=ibr_positions,
        ibr_capacities_mva=capacities_mva,
        ibr_capacities_pu=capacities_mva / float(system.config.mva),
        include_loads=include_loads,
        sg_reactance=sg_reactance,
        direct_reduction_relative_error=relative_error,
    )
