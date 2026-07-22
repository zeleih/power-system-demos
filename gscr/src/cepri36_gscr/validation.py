"""Independent checks against the PSASP solved branch-flow files."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .psasp import PSASPCase


def _records(path: Path) -> list[list[str]]:
    records = []
    for line in path.read_text(encoding="ascii", errors="replace").splitlines():
        if not line.strip():
            continue
        record = next(csv.reader([line], skipinitialspace=True))
        while record and not record[-1].strip():
            record.pop()
        records.append(record)
    return records


def validate_branch_conventions(case: PSASPCase) -> dict[str, float]:
    voltage = case.solved_voltage
    line_errors: list[float] = []
    lp2 = _records(case.source_dir / "LF.LP2")
    for branch, output in zip(case.branches, lp2):
        if branch.from_bus == branch.to_bus:
            continue
        i = branch.from_bus - 1
        j = branch.to_bus - 1
        series_y = 1.0 / complex(branch.resistance_pu, branch.reactance_pu)
        current_i = (voltage[i] - voltage[j]) * series_y + 1j * branch.charging_each_end_pu * voltage[i]
        current_j = (voltage[j] - voltage[i]) * series_y + 1j * branch.charging_each_end_pu * voltage[j]
        power_i = voltage[i] * np.conj(current_i)
        power_j = voltage[j] * np.conj(current_j)
        predicted = np.array([power_i.real, power_i.imag, -power_j.real, -power_j.imag])
        observed = np.array([float(value) for value in output[3:7]])
        line_errors.extend((predicted - observed).tolist())

    transformer_errors: list[float] = []
    lp3 = _records(case.source_dir / "LF.LP3")
    for transformer, output in zip(case.transformers, lp3):
        if "_2w" not in transformer.name.lower():
            continue
        i = transformer.from_bus - 1
        j = transformer.to_bus - 1
        series_y = 1.0 / complex(transformer.resistance_pu, transformer.reactance_pu)
        tap = transformer.tap_second_side
        current_i = series_y * voltage[i] - series_y / tap * voltage[j]
        current_j = series_y / tap**2 * voltage[j] - series_y / tap * voltage[i]
        power_i = voltage[i] * np.conj(current_i)
        power_j = voltage[j] * np.conj(current_j)
        predicted = np.array([power_i.real, power_i.imag, -power_j.real, -power_j.imag])
        observed = np.array([float(value) for value in output[3:7]])
        transformer_errors.extend((predicted - observed).tolist())

    return {
        "ac_line_rmse_pu": float(np.sqrt(np.mean(np.square(line_errors)))),
        "ac_line_max_abs_error_pu": float(np.max(np.abs(line_errors))),
        "two_winding_transformer_rmse_pu": float(np.sqrt(np.mean(np.square(transformer_errors)))),
        "two_winding_transformer_max_abs_error_pu": float(np.max(np.abs(transformer_errors))),
    }
