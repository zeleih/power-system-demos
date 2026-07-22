"""CEPRI36 adapter for the paper's analytical PMU identification method."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gscr_demo.identification import PMURun, identify_symmetric_admittance as identify_y

from .model import generalized_scr


@dataclass(frozen=True)
class IdentificationResult:
    y_hat: np.ndarray
    gscr: float
    eigenvalues: np.ndarray
    residual_rmse: float
    design_rank: int
    design_condition: float
    sample_count: int


def identify_symmetric_admittance(
    voltage: np.ndarray,
    current: np.ndarray,
    capacities_pu: np.ndarray,
    *,
    ridge: float = 0.0,
) -> IdentificationResult:
    """Identify the reduced eight-port admittance and its gSCR."""

    analytical = identify_y(
        [PMURun(voltage=voltage, current=current)],
        ridge=ridge,
    )
    gscr, eigenvalues, _ = generalized_scr(-analytical.y_hat.imag, capacities_pu)
    return IdentificationResult(
        y_hat=analytical.y_hat,
        gscr=gscr,
        eigenvalues=eigenvalues,
        residual_rmse=analytical.residual_rmse,
        design_rank=analytical.parameter_rank_real,
        design_condition=analytical.voltage_condition,
        sample_count=voltage.shape[0],
    )
