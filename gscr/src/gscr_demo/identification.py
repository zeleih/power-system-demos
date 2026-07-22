"""Non-iterative analytical identification of a complex-symmetric admittance.

The implementation follows the accumulated least-squares equations in
Han et al. (2025).  PMU batches contribute fixed-size sufficient statistics;
the independent conductance and susceptance entries are solved once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.sparse import bmat, coo_matrix, eye
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class PMURun:
    """One PMU record with arrays shaped ``[time, retained_port]``."""

    voltage: np.ndarray
    current: np.ndarray


@dataclass(frozen=True)
class IdentificationResult:
    """Result and observability diagnostics for the analytical solution."""

    y_hat: np.ndarray
    residual_rmse: float
    voltage_rank: int
    voltage_condition: float
    sample_increment_count: int
    port_count: int
    independent_complex_parameter_count: int
    parameter_rank_real: int
    estimator: str = "paper_analytic"
    solver_iterations: int = 1
    solver_stop_code: int = 0


def _pairs(port_count: int) -> list[tuple[int, int]]:
    return [
        (row, column)
        for row in range(port_count)
        for column in range(row, port_count)
    ]


def _unpack_symmetric(
    theta: np.ndarray,
    pairs: list[tuple[int, int]],
    port_count: int,
) -> np.ndarray:
    admittance = np.zeros((port_count, port_count), dtype=complex)
    for value, (row, column) in zip(theta, pairs):
        admittance[row, column] = value
        admittance[column, row] = value
    return admittance


def _normal_equations(
    covariance: np.ndarray,
    cross: np.ndarray,
) -> tuple[object, np.ndarray, list[tuple[int, int]]]:
    """Build equations for the independent entries of a symmetric matrix."""

    port_count = covariance.shape[0]
    pairs = _pairs(port_count)
    pair_index = np.empty((port_count, port_count), dtype=int)
    for index, (row, column) in enumerate(pairs):
        pair_index[row, column] = index
        pair_index[column, row] = index

    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    right_hand_side = np.zeros(len(pairs), dtype=complex)
    base = np.arange(port_count)

    for output in range(port_count):
        indices = pair_index[output, :]
        rows.append(np.repeat(indices, port_count))
        columns.append(np.tile(indices, port_count))
        values.append(covariance.reshape(-1))
        np.add.at(right_hand_side, indices, cross[base, output])

    gram = coo_matrix(
        (
            np.concatenate(values),
            (np.concatenate(rows), np.concatenate(columns)),
        ),
        shape=(len(pairs), len(pairs)),
        dtype=np.complex128,
    ).tocsc()
    gram = 0.5 * (gram + gram.getH())
    return gram, right_hand_side, pairs


def identify_symmetric_admittance(
    runs: Iterable,
    *,
    noise_std: float = 0.0,
    seed: int = 20260722,
    ridge: float = 0.0,
) -> IdentificationResult:
    """Identify ``Y = Y.T`` from one or more PMU voltage/current records.

    Each record contributes to fixed-size accumulated matrices.  Historical
    time samples do not need to be retained after their contribution has been
    added.  ``noise_std`` is provided for repeatable sensitivity experiments
    and is zero in the reference reproductions.
    """

    rng = np.random.default_rng(seed)
    covariance: np.ndarray | None = None
    cross: np.ndarray | None = None
    current_energy = 0.0
    increment_count = 0
    port_count = 0

    for run in runs:
        voltage = np.asarray(run.voltage, dtype=complex).copy()
        current = np.asarray(run.current, dtype=complex).copy()
        if voltage.shape != current.shape or voltage.ndim != 2:
            raise ValueError(
                "each PMU run must contain equally shaped [time, port] arrays"
            )
        if voltage.shape[0] < 2:
            raise ValueError("each PMU run must contain at least two samples")

        if noise_std > 0:
            scale = noise_std / np.sqrt(2.0)
            voltage += scale * (
                rng.standard_normal(voltage.shape)
                + 1j * rng.standard_normal(voltage.shape)
            )
            current += scale * (
                rng.standard_normal(current.shape)
                + 1j * rng.standard_normal(current.shape)
            )

        delta_voltage = np.diff(voltage, axis=0)
        delta_current = np.diff(current, axis=0)
        if covariance is None:
            port_count = delta_voltage.shape[1]
            covariance = np.zeros((port_count, port_count), dtype=complex)
            cross = np.zeros((port_count, port_count), dtype=complex)
        elif delta_voltage.shape[1] != port_count:
            raise ValueError("all PMU runs must use the same retained ports")

        covariance += delta_voltage.conj().T @ delta_voltage
        cross += delta_voltage.conj().T @ delta_current
        current_energy += float(np.sum(np.abs(delta_current) ** 2))
        increment_count += len(delta_voltage)

    if covariance is None or cross is None:
        raise ValueError("at least one PMU run is required")

    gram, right_hand_side, pairs = _normal_equations(covariance, cross)

    # Real G/B block form of the paper's analytical equations.
    real_gram = bmat(
        [[gram.real, -gram.imag], [gram.imag, gram.real]],
        format="csc",
    )
    if ridge > 0:
        real_gram = real_gram + ridge * eye(real_gram.shape[0], format="csc")
    real_right_hand_side = np.concatenate(
        [right_hand_side.real, right_hand_side.imag]
    )
    real_theta = spsolve(real_gram, real_right_hand_side)
    independent_count = len(pairs)
    theta = (
        real_theta[:independent_count]
        + 1j * real_theta[independent_count:]
    )
    y_hat = _unpack_symmetric(theta, pairs, port_count)

    # Evaluate the least-squares objective from the same accumulated terms.
    residual_energy = (
        current_energy
        - 2.0 * float(np.real(np.vdot(theta, right_hand_side)))
        + float(np.real(np.vdot(theta, gram @ theta)))
    )
    residual_energy = max(residual_energy, 0.0)

    covariance = 0.5 * (covariance + covariance.conj().T)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)[::-1]
    if eigenvalues[0] == 0.0:
        voltage_rank = 0
        voltage_condition = float("inf")
    else:
        tolerance = (
            np.finfo(float).eps
            * max(increment_count, port_count)
            * eigenvalues[0]
        )
        voltage_rank = int(np.sum(eigenvalues > tolerance))
        voltage_condition = (
            float(np.sqrt(eigenvalues[0] / eigenvalues[-1]))
            if voltage_rank == port_count
            else float("inf")
        )

    nullity = port_count - voltage_rank
    complex_parameter_rank = independent_count - nullity * (nullity + 1) // 2

    return IdentificationResult(
        y_hat=y_hat,
        residual_rmse=float(
            np.sqrt(residual_energy / (increment_count * port_count))
        ),
        voltage_rank=voltage_rank,
        voltage_condition=voltage_condition,
        sample_increment_count=increment_count,
        port_count=port_count,
        independent_complex_parameter_count=independent_count,
        parameter_rank_real=2 * complex_parameter_rank,
    )
