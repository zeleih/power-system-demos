"""Network-consistent PMU surrogate data for the paper's disturbance studies."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .model import NetworkModel, generalized_scr
from .psasp import PSASPCase


@dataclass
class ScenarioData:
    name: str
    time_s: np.ndarray
    voltage: np.ndarray
    current: np.ndarray
    metadata: dict[str, object]


def bus50_profile(time_s: np.ndarray) -> np.ndarray:
    """A reproducible representative profile; exact paper breakpoints are not required."""

    return np.interp(time_s, [0.0, 0.6, 4.5, 5.1, time_s[-1]], [0.0, 1.0, 1.0, 0.0, 0.0])


def _event_source(case: PSASPCase, event: str, active: np.ndarray) -> np.ndarray:
    lookup = {full_index: position for position, full_index in enumerate(active)}
    source = np.zeros(len(active), dtype=complex)

    def add(bus_name: str, value: complex) -> None:
        full_index = case.index_of(bus_name) - 1
        source[lookup[full_index]] += value

    if event == "fault_bus30":
        add("BUS30", 1.0)
    elif event == "fault_bus25":
        add("BUS25", 1.0)
    elif event == "fault_line_bus9_bus23":
        add("BUS9", 0.5)
        add("BUS23", 0.5)
    elif event == "load_bus50":
        add("BUS50", 1.0)
    else:
        raise ValueError(f"Unknown event: {event}")
    return source


def _modal_voltage_response(
    case: PSASPCase,
    model: NetworkModel,
    event: str,
    time_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    active = np.concatenate([model.retained_indices_zero_based, model.internal_indices_zero_based])
    source = _event_source(case, event, active)
    y_active = model.full_y[np.ix_(active, active)].copy()

    # A generator Thevenin grounding is used only to derive an event-location
    # sensitivity vector; the identified current is still computed from Ybar.
    for retained_position in range(len(model.retained_indices_zero_based)):
        y_active[retained_position, retained_position] += -4j
    signature_full = np.linalg.solve(y_active, source)
    signature = signature_full[: len(model.retained_indices_zero_based)]

    _, _, eigenvectors = generalized_scr(model.reduced_b, model.capacities_pu)
    inv_sqrt_capacity = np.diag(1.0 / np.sqrt(model.capacities_pu))
    shapes = inv_sqrt_capacity @ eigenvectors
    shapes /= np.maximum(np.max(np.abs(shapes), axis=0, keepdims=True), 1e-12)
    projection = shapes.T @ signature
    weights = np.abs(projection)
    weights = 0.18 + 0.82 * weights / max(float(weights.max()), 1e-12)
    phases = np.exp(1j * np.angle(projection + 1e-12))

    frequencies = np.array([0.36, 0.55, 0.78, 1.03, 1.31, 1.68, 2.12, 2.65])
    damping = np.array([0.10, 0.13, 0.17, 0.21, 0.26, 0.31, 0.37, 0.44])
    response = np.zeros((len(time_s), 8), dtype=complex)
    gate = 1.0 - np.exp(-time_s / 0.025)
    profile = bus50_profile(time_s)
    for mode in range(8):
        phase = 0.31 * mode
        oscillation = np.sin(2 * np.pi * frequencies[mode] * time_s + phase) - np.sin(phase)
        if event == "load_bus50":
            waveform = profile * (0.45 + 0.55 * np.cos(2 * np.pi * frequencies[mode] * time_s + phase))
            waveform += 0.16 * gate * np.exp(-damping[mode] * time_s) * oscillation
        else:
            waveform = gate * np.exp(-damping[mode] * time_s) * oscillation
        response += np.outer(waveform, weights[mode] * phases[mode] * shapes[:, mode])

    target_peak = 0.014 if event == "load_bus50" else 0.060
    response *= target_peak / max(float(np.max(np.abs(response))), 1e-12)
    return response, source


def generate_scenario(
    case: PSASPCase,
    model: NetworkModel,
    event: str,
    *,
    duration_s: float = 10.0,
    integration_step_s: float = 0.01,
    internal_injection_scale: float = 0.0,
    measurement_noise_power: float = 0.0,
    noise_mode: str = "independent_pmu",
    random_seed: int = 202507,
) -> ScenarioData:
    """Generate auditable PMU data around the PSASP solved operating point.

    internal_injection_scale optionally represents unmeasured dynamic devices at
    eliminated buses. It is zero in the clean algorithm-verification runs.
    """

    time_s = np.arange(0.0, duration_s + 0.5 * integration_step_s, integration_step_s)
    delta_voltage, internal_source = _modal_voltage_response(case, model, event, time_s)
    u0 = case.solved_voltage[model.retained_indices_zero_based]
    voltage = u0[None, :] + delta_voltage
    current = voltage @ model.reduced_y.T

    if internal_injection_scale != 0:
        active = np.concatenate([model.retained_indices_zero_based, model.internal_indices_zero_based])
        n_retained = len(model.retained_indices_zero_based)
        y_active = model.full_y[np.ix_(active, active)]
        ybi = y_active[:n_retained, n_retained:]
        yii = y_active[n_retained:, n_retained:]
        source_internal = internal_source[n_retained:]
        transfer = ybi @ np.linalg.solve(yii, source_internal)
        gate = 1.0 - np.exp(-time_s / 0.03)
        if event == "load_bus50":
            waveform = bus50_profile(time_s)
        else:
            waveform = gate * np.exp(-0.45 * time_s) * np.sin(2 * np.pi * 0.83 * time_s)
        current += internal_injection_scale * np.outer(waveform, transfer)

    if measurement_noise_power > 0:
        rng = np.random.default_rng(random_seed)
        sigma = np.sqrt(measurement_noise_power / 2.0)
        voltage_noise = sigma * (
            rng.standard_normal(voltage.shape) + 1j * rng.standard_normal(voltage.shape)
        )
        if noise_mode == "independent_pmu":
            voltage = voltage + voltage_noise
            current = current + sigma * (
                rng.standard_normal(current.shape) + 1j * rng.standard_normal(current.shape)
            )
        elif noise_mode == "network_consistent_process":
            # This is useful as a sensitivity bound, but is not independent PMU
            # measurement noise: the current component follows the network law.
            voltage = voltage + voltage_noise
            current = current + voltage_noise @ model.reduced_y.T
        else:
            raise ValueError(f"Unknown noise mode: {noise_mode}")

    return ScenarioData(
        name=event,
        time_s=time_s,
        voltage=voltage,
        current=current,
        metadata={
            "duration_s": duration_s,
            "integration_step_s": integration_step_s,
            "internal_injection_scale": internal_injection_scale,
            "measurement_noise_power": measurement_noise_power,
            "noise_mode": noise_mode,
            "random_seed": random_seed,
        },
    )


def resample_scenario(
    data: ScenarioData, sample_interval_s: float, *, duration_s: float | None = None
) -> ScenarioData:
    if duration_s is None:
        duration_s = float(data.time_s[-1])
    target = np.arange(0.0, duration_s + 0.5 * sample_interval_s, sample_interval_s)
    indices = np.searchsorted(data.time_s, target)
    indices = np.clip(indices, 0, len(data.time_s) - 1)
    return replace(
        data,
        time_s=data.time_s[indices],
        voltage=data.voltage[indices],
        current=data.current[indices],
        metadata={**data.metadata, "sample_interval_s": sample_interval_s, "window_s": duration_s},
    )


def scenario_to_frame(data: ScenarioData, bus_names: list[str]) -> pd.DataFrame:
    columns: dict[str, np.ndarray] = {"time_s": data.time_s}
    for position, name in enumerate(bus_names):
        columns[f"U_{name}_re"] = data.voltage[:, position].real
        columns[f"U_{name}_im"] = data.voltage[:, position].imag
        columns[f"I_{name}_re"] = data.current[:, position].real
        columns[f"I_{name}_im"] = data.current[:, position].imag
    return pd.DataFrame(columns)


def frame_to_scenario(frame: pd.DataFrame, bus_names: list[str], name: str = "csv") -> ScenarioData:
    voltage = np.column_stack(
        [frame[f"U_{bus}_re"].to_numpy() + 1j * frame[f"U_{bus}_im"].to_numpy() for bus in bus_names]
    )
    current = np.column_stack(
        [frame[f"I_{bus}_re"].to_numpy() + 1j * frame[f"I_{bus}_im"].to_numpy() for bus in bus_names]
    )
    return ScenarioData(name, frame["time_s"].to_numpy(), voltage, current, {"source": "csv"})
