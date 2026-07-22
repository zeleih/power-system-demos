"""Native ANDES time-domain scenarios for the CEPRI36 gSCR study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .andes_case import NOMINAL_FREQUENCY_HZ, SYSTEM_BASE_MVA
from .identification import IdentificationResult, identify_symmetric_admittance
from .model import NetworkModel, build_network_model, generalized_scr, kron_reduce
from .psasp import PSASPCase
from .scenarios import ScenarioData


SUPPORTED_SCENARIOS = (
    "fault_bus30",
    "fault_bus25",
    "fault_line_bus9_bus23",
    "load_bus50",
)


@dataclass
class AndesScenarioRun:
    name: str
    raw_time_s: np.ndarray
    raw_voltage: np.ndarray
    raw_current: np.ndarray
    analysis_start_s: float
    analysis_duration_s: float
    reduced_y: np.ndarray
    direct_gscr: float
    capacities_pu: np.ndarray
    pflow_voltage_rmse: float
    pflow_voltage_max_error: float
    port_current_kcl_rmse: float
    tds_steps: int
    metadata: dict[str, object]

    def sample(self, interval_s: float = 0.1, duration_s: float | None = None) -> ScenarioData:
        if duration_s is None:
            duration_s = self.analysis_duration_s
        stop = min(self.analysis_start_s + duration_s, float(self.raw_time_s[-1]))
        target = np.arange(
            self.analysis_start_s,
            stop + 0.5 * interval_s,
            interval_s,
        )
        target = target[target <= self.raw_time_s[-1] + 1e-12]

        def interpolate(values: np.ndarray) -> np.ndarray:
            return np.column_stack(
                [
                    np.interp(target, self.raw_time_s, values[:, column].real)
                    + 1j * np.interp(target, self.raw_time_s, values[:, column].imag)
                    for column in range(values.shape[1])
                ]
            )

        return ScenarioData(
            name=self.name,
            time_s=target,
            voltage=interpolate(self.raw_voltage),
            current=interpolate(self.raw_current),
            metadata={
                **self.metadata,
                "source": "ANDES 2.0.0 TDS",
                "analysis_start_s": self.analysis_start_s,
                "sample_interval_s": interval_s,
                "window_s": float(target[-1] - target[0]) if len(target) > 1 else 0.0,
            },
        )

    def identify(
        self,
        interval_s: float = 0.1,
        duration_s: float | None = None,
    ) -> IdentificationResult:
        sampled = self.sample(interval_s, duration_s)
        return identify_symmetric_admittance(
            sampled.voltage,
            sampled.current,
            self.capacities_pu,
        )


def _add_ideal_pmu_models(system, case: PSASPCase) -> None:
    for ordinal, bus in enumerate(case.retained_bus_indices, start=1):
        system.add(
            "PMU",
            idx=f"PMU_BUS{ordinal}",
            name=f"PMU BUS{ordinal}",
            bus=bus,
            Ta=0.02,
            Tv=0.02,
        )


def _add_bus_fault(system, case: PSASPCase, bus_name: str) -> tuple[float, dict[str, object]]:
    bus = case.index_of(bus_name)
    system.add(
        "Fault",
        idx=f"FAULT_{bus_name}",
        name=f"Three-phase fault at {bus_name}",
        bus=bus,
        tf=1.0,
        tc=1.05,
        xf=0.05,
        rf=0.0,
    )
    return 1.06, {"event": "three-phase bus fault", "event_bus": bus_name}


def _add_midpoint_line_fault(system, case: PSASPCase) -> tuple[float, dict[str, object]]:
    bus9 = case.index_of("BUS9")
    bus23 = case.index_of("BUS23")
    branch = next(
        branch
        for branch in case.branches
        if {branch.from_bus, branch.to_bus} == {bus9, bus23}
        and branch.from_bus != branch.to_bus
    )
    system.Line.set("u", f"AC_{branch.record_id}", 0)

    midpoint = "MID_BUS9_BUS23"
    voltage_mid = 0.5 * (
        case.solved_voltage[branch.from_bus - 1]
        + case.solved_voltage[branch.to_bus - 1]
    )
    base_kv = case.buses[branch.from_bus - 1].base_kv
    system.add(
        "Bus",
        idx=midpoint,
        u=1,
        name=midpoint,
        Vn=base_kv,
        v0=abs(voltage_mid),
        a0=float(np.angle(voltage_mid)),
        vmax=1.2,
        vmin=0.8,
    )
    line_common = {
        "u": 1,
        "Sn": SYSTEM_BASE_MVA,
        "fn": NOMINAL_FREQUENCY_HZ,
        "Vn1": base_kv,
        "Vn2": base_kv,
        "r": branch.resistance_pu / 2.0,
        "x": branch.reactance_pu / 2.0,
        "b": 0.0,
        "g": 0.0,
        "trans": 0,
        "tap": 1.0,
        "phi": 0.0,
    }
    system.add(
        "Line",
        idx="LINE_BUS9_MID",
        name="BUS9 to line midpoint",
        bus1=branch.from_bus,
        bus2=midpoint,
        b1=branch.charging_each_end_pu,
        g1=0.0,
        b2=0.0,
        g2=0.0,
        **line_common,
    )
    system.add(
        "Line",
        idx="LINE_MID_BUS23",
        name="Line midpoint to BUS23",
        bus1=midpoint,
        bus2=branch.to_bus,
        b1=0.0,
        g1=0.0,
        b2=branch.charging_each_end_pu,
        g2=0.0,
        **line_common,
    )
    system.add(
        "Fault",
        idx="FAULT_LINE_BUS9_BUS23",
        name="Three-phase fault at BUS9-BUS23 midpoint",
        bus=midpoint,
        tf=1.0,
        tc=1.05,
        xf=0.05,
        rf=0.0,
    )
    return 1.06, {
        "event": "three-phase line midpoint fault",
        "event_line": "BUS9-BUS23",
        "split_original_line": f"AC_{branch.record_id}",
    }


def _add_bus50_load_pulse(system, case: PSASPCase) -> tuple[float, dict[str, object]]:
    bus = case.index_of("BUS50")
    system.add(
        "PQ",
        idx="LOAD_PULSE_BUS50",
        name="BUS50 excitation load",
        bus=bus,
        Vn=case.buses[bus - 1].base_kv,
        p0=0.08,
        q0=0.03,
        vmax=1.2,
        vmin=0.8,
    )
    system.add(
        "Toggle",
        idx="BUS50_PULSE_OFF",
        name="BUS50 load pulse off",
        model="PQ",
        dev="LOAD_PULSE_BUS50",
        t=1.0,
    )
    system.add(
        "Toggle",
        idx="BUS50_PULSE_ON",
        name="BUS50 load pulse restore",
        model="PQ",
        dev="LOAD_PULSE_BUS50",
        t=1.25,
    )
    return 1.26, {
        "event": "BUS50 load pulse",
        "pulse_p_pu": 0.08,
        "pulse_q_pu": 0.03,
        "pulse_off_s": 1.0,
        "pulse_restore_s": 1.25,
    }


def _scenario_network(
    case: PSASPCase,
    base_model: NetworkModel,
    scenario: str,
    solved_voltage_by_bus: dict[int, float],
) -> tuple[np.ndarray, float]:
    full_y = base_model.full_y.copy()
    if scenario == "load_bus50":
        bus = case.index_of("BUS50")
        voltage_sq = solved_voltage_by_bus[bus] ** 2
        full_y[bus - 1, bus - 1] += np.conj(complex(0.08, 0.03)) / voltage_sq
    reduced = kron_reduce(
        full_y,
        base_model.retained_indices_zero_based,
        base_model.internal_indices_zero_based,
    )
    direct_gscr, _, _ = generalized_scr(-reduced.imag, base_model.capacities_pu)
    return reduced, direct_gscr


def _phasor_matrix(system, case: PSASPCase) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(system.dae.ts.t, dtype=float)
    voltage_all = np.zeros((len(times), len(case.buses)), dtype=complex)
    bus_positions = np.asarray(system.Bus.idx.v, dtype=object)
    for column, bus_idx in enumerate(bus_positions):
        if isinstance(bus_idx, (int, np.integer)):
            magnitude = system.dae.ts.y[:, system.Bus.v.a[column]]
            angle = system.dae.ts.y[:, system.Bus.a.a[column]]
            voltage_all[:, int(bus_idx) - 1] = magnitude * np.exp(1j * angle)
    retained_voltage = voltage_all[:, np.asarray(case.retained_bus_indices) - 1]
    return voltage_all, retained_voltage


def _generator_current(system, retained_voltage: np.ndarray, case: PSASPCase) -> np.ndarray:
    power = (
        system.dae.ts.y[:, system.GENCLS.Pe.a]
        + 1j * system.dae.ts.y[:, system.GENCLS.Qe.a]
    )
    generator_bus_order = list(system.GENCLS.bus.v)
    order = [generator_bus_order.index(bus) for bus in case.retained_bus_indices]
    power = power[:, order]
    return np.conj(power / retained_voltage)


def run_andes_scenario(
    case: PSASPCase,
    workbook: str | Path,
    pycode_path: str | Path,
    scenario: str,
    *,
    final_time_s: float = 11.5,
    integration_step_s: float = 0.01,
) -> AndesScenarioRun:
    if scenario not in SUPPORTED_SCENARIOS:
        raise ValueError(f"Unsupported ANDES scenario: {scenario}")

    import andes

    base_model = build_network_model(case)
    system = andes.load(
        str(workbook),
        setup=False,
        pycode_path=str(pycode_path),
    )
    _add_ideal_pmu_models(system, case)

    if scenario == "fault_bus30":
        analysis_start_s, event_metadata = _add_bus_fault(system, case, "BUS30")
    elif scenario == "fault_bus25":
        analysis_start_s, event_metadata = _add_bus_fault(system, case, "BUS25")
    elif scenario == "fault_line_bus9_bus23":
        analysis_start_s, event_metadata = _add_midpoint_line_fault(system, case)
    else:
        analysis_start_s, event_metadata = _add_bus50_load_pulse(system, case)

    if not system.setup():
        raise RuntimeError(f"ANDES setup failed for {scenario}")
    if not system.PFlow.run():
        raise RuntimeError(f"ANDES power flow failed for {scenario}")

    solved_voltage_by_bus = {
        int(bus): float(voltage)
        for bus, voltage in zip(system.Bus.idx.v, system.Bus.v.v)
        if isinstance(bus, (int, np.integer))
    }
    reference = np.abs(case.solved_voltage)
    comparable = np.asarray(case.active_bus_indices) - 1
    andes_voltage = np.asarray([solved_voltage_by_bus[index] for index in case.active_bus_indices])
    voltage_error = andes_voltage - reference[comparable]

    reduced_y, direct_gscr = _scenario_network(
        case,
        base_model,
        scenario,
        solved_voltage_by_bus,
    )

    system.TDS.config.tf = final_time_s
    system.TDS.config.fixt = 1
    system.TDS.config.tstep = integration_step_s
    system.TDS.config.no_tqdm = 1
    if not system.TDS.run():
        raise RuntimeError(f"ANDES TDS failed for {scenario}")

    voltage_all, retained_voltage = _phasor_matrix(system, case)
    generator_current = _generator_current(system, retained_voltage, case)
    times = np.asarray(system.dae.ts.t, dtype=float)

    full_y = base_model.full_y.copy()
    if scenario == "load_bus50":
        bus = case.index_of("BUS50")
        voltage_sq = solved_voltage_by_bus[bus] ** 2
        full_y[bus - 1, bus - 1] += np.conj(complex(0.08, 0.03)) / voltage_sq
    network_current = (voltage_all @ full_y.T)[:, base_model.retained_indices_zero_based]
    post_event = times >= analysis_start_s
    kcl_rmse = float(
        np.sqrt(np.mean(np.abs(generator_current[post_event] - network_current[post_event]) ** 2))
    )

    return AndesScenarioRun(
        name=scenario,
        raw_time_s=times,
        raw_voltage=retained_voltage,
        raw_current=generator_current,
        analysis_start_s=analysis_start_s,
        analysis_duration_s=min(10.0, final_time_s - analysis_start_s),
        reduced_y=reduced_y,
        direct_gscr=direct_gscr,
        capacities_pu=base_model.capacities_pu,
        pflow_voltage_rmse=float(np.sqrt(np.mean(voltage_error**2))),
        pflow_voltage_max_error=float(np.max(np.abs(voltage_error))),
        port_current_kcl_rmse=kcl_rmse,
        tds_steps=len(times),
        metadata={
            **event_metadata,
            "andes_version": andes.__version__,
            "integration_step_s": integration_step_s,
            "final_time_s": final_time_s,
            "pmu_bus_names": case.retained_bus_names,
            "current_direction": "generator injection into the external AC network",
            "current_source": "GENCLS Pe/Qe converted with S=U*conj(I)",
        },
    )
