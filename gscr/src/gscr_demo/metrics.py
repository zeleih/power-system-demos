"""Small, simulator-independent helpers for gSCR calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GSCRResult:
    value: float
    eigenvalues: np.ndarray
    critical_mode: np.ndarray


def generalized_scr(
    admittance: np.ndarray,
    capacities_pu: np.ndarray,
) -> GSCRResult:
    """Calculate rated-capacity-normalized gSCR from a reduced admittance."""

    susceptance = -np.asarray(admittance).imag
    capacities = np.asarray(capacities_pu, dtype=float)
    inverse_sqrt = np.diag(1.0 / np.sqrt(capacities))
    normalized = inverse_sqrt @ susceptance @ inverse_sqrt
    normalized = 0.5 * (normalized + normalized.T)
    values, vectors = np.linalg.eigh(normalized)
    return GSCRResult(
        value=float(values[0]),
        eigenvalues=values,
        critical_mode=vectors[:, 0],
    )
