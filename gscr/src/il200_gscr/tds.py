"""ANDES TDS disturbance runs and ideal PMU phasor extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np

from .network import DirectNetwork, build_direct_network


@dataclass(frozen=True)
class FaultRun:
    fault_bus: int
    time_s: np.ndarray
    voltage: np.ndarray
    current: np.ndarray
    ibr_voltage: np.ndarray
    kcl_rmse: float
    kcl_max_error: float
    max_ibr_voltage_deviation: float
    source_port_buses: np.ndarray


def _source_current(system, direct: DirectNetwork, source_voltage: np.ndarray) -> np.ndarray:
    sg = direct.sg_model_positions
    ibr = direct.ibr_model_positions
    psg = system.dae.ts.y[:, np.asarray(system.GENROU.Pe.a)[sg]]
    qsg = system.dae.ts.y[:, np.asarray(system.GENROU.Qe.a)[sg]]
    pibr = system.dae.ts.y[:, np.asarray(system.REGCA1.Pe.a)[ibr]]
    qibr = system.dae.ts.y[:, np.asarray(system.REGCA1.Qe.a)[ibr]]
    power = np.column_stack([psg + 1j * qsg, pibr + 1j * qibr])
    return np.conj(power / source_voltage)


def run_fault(
    case_path: str | Path,
    pycode_path: str | Path,
    fault_bus: int,
    *,
    fault_start_s: float = 1.0,
    fault_clear_s: float = 1.05,
    fault_x_pu: float = 0.05,
    final_time_s: float = 4.0,
    time_step_s: float = 0.01,
) -> FaultRun:
    import andes

    system = andes.load(
        str(case_path),
        setup=False,
        no_output=True,
        pycode_path=str(pycode_path),
    )
    system.add(
        "Fault",
        idx=f"GSCR_FAULT_BUS_{fault_bus}",
        name=f"gSCR excitation fault at Bus {fault_bus}",
        bus=fault_bus,
        tf=fault_start_s,
        tc=fault_clear_s,
        xf=fault_x_pu,
        rf=0.0,
    )
    if not system.setup():
        raise RuntimeError(f"ANDES setup failed for fault at Bus {fault_bus}")
    if not system.PFlow.run():
        raise RuntimeError(f"ANDES power flow failed for fault at Bus {fault_bus}")
    direct = build_direct_network(system, include_loads=True, sg_reactance="xd2")

    system.TDS.config.tf = final_time_s
    system.TDS.config.fixt = 1
    system.TDS.config.tstep = time_step_s
    system.TDS.config.no_tqdm = 1
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Casting complex values to real")
        if not system.TDS.run():
            raise RuntimeError(f"ANDES TDS failed for fault at Bus {fault_bus}")

    time = np.asarray(system.dae.ts.t, dtype=float)
    magnitude = system.dae.ts.y[:, system.Bus.v.a]
    angle = system.dae.ts.y[:, system.Bus.a.a]
    voltage_all = magnitude * np.exp(1j * angle)
    bus_position = {int(bus): row for row, bus in enumerate(system.Bus.idx.v)}
    source_indices = np.asarray(
        [bus_position[int(bus)] for bus in direct.source_port_buses], dtype=int
    )
    ibr_indices = np.asarray([bus_position[int(bus)] for bus in direct.ibr_buses], dtype=int)
    source_voltage = voltage_all[:, source_indices]
    source_current = _source_current(system, direct, source_voltage)
    network_current = (voltage_all @ direct.full_y.T)[:, source_indices]

    analysis_start = fault_clear_s + time_step_s
    selected = (time >= analysis_start - 1e-12) & (time <= final_time_s + 1e-12)
    kcl_error = source_current[selected] - network_current[selected]
    ibr_voltage = voltage_all[selected][:, ibr_indices]
    initial_ibr_magnitude = np.abs(ibr_voltage[0])
    max_voltage_deviation = float(
        np.max(np.abs(np.abs(ibr_voltage) - initial_ibr_magnitude[None, :]))
    )
    return FaultRun(
        fault_bus=fault_bus,
        time_s=time[selected],
        voltage=source_voltage[selected],
        current=source_current[selected],
        ibr_voltage=ibr_voltage,
        kcl_rmse=float(np.sqrt(np.mean(np.abs(kcl_error) ** 2))),
        kcl_max_error=float(np.max(np.abs(kcl_error))),
        max_ibr_voltage_deviation=max_voltage_deviation,
        source_port_buses=direct.source_port_buses,
    )
