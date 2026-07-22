"""Load the publishable CEPRI36 reference port model and archived PMU data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .model import generalized_scr
from .scenarios import ScenarioData, frame_to_scenario


@dataclass(frozen=True)
class CEPRI36Reference:
    bus_names: list[str]
    reduced_y: np.ndarray
    capacities_pu: np.ndarray

    @property
    def gscr(self) -> float:
        return generalized_scr(-self.reduced_y.imag, self.capacities_pu)[0]


def load_reference_model(case_root: str | Path) -> CEPRI36Reference:
    """Load the eight-port matrix and capacities published with the demo."""

    root = Path(case_root)
    tables = root / "outputs" / "tables"
    real = pd.read_csv(tables / "reduced_y_real.csv", index_col=0)
    imag = pd.read_csv(tables / "reduced_y_imag.csv", index_col=0)
    capacities = pd.read_csv(tables / "generator_capacities.csv")
    bus_names = capacities["bus"].astype(str).tolist()
    if real.index.tolist() != bus_names or real.columns.tolist() != bus_names:
        raise ValueError("CEPRI36 real-admittance labels do not match the port list")
    if imag.index.tolist() != bus_names or imag.columns.tolist() != bus_names:
        raise ValueError("CEPRI36 imaginary-admittance labels do not match the port list")
    return CEPRI36Reference(
        bus_names=bus_names,
        reduced_y=real.to_numpy(float) + 1j * imag.to_numpy(float),
        capacities_pu=capacities["capacity_pu_on_100_mva"].to_numpy(float),
    )


def load_archived_pmu(
    case_root: str | Path,
    relative_path: str | Path,
) -> ScenarioData:
    """Load one archived CEPRI36 PMU CSV using the common eight-port schema."""

    root = Path(case_root)
    reference = load_reference_model(root)
    path = root / relative_path
    return frame_to_scenario(
        pd.read_csv(path),
        reference.bus_names,
        name=path.stem,
    )
