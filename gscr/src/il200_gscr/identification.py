"""IL200 adapter for the paper's analytical PMU identification method."""

from __future__ import annotations

from typing import Iterable

from gscr_demo.identification import IdentificationResult, identify_symmetric_admittance


def identify_port_admittance(
    runs: Iterable,
    *,
    noise_std: float = 0.0,
    seed: int = 20260722,
    ridge: float = 0.0,
) -> IdentificationResult:
    return identify_symmetric_admittance(
        runs,
        noise_std=noise_std,
        seed=seed,
        ridge=ridge,
    )


__all__ = ["IdentificationResult", "identify_port_admittance"]
