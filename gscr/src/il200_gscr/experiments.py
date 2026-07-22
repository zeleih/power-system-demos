"""End-to-end IL200 IBR-port gSCR reproduction and artifact generation."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".mplconfig")
)

import andes
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .identification import identify_port_admittance
from .network import build_direct_network, generalized_scr, terminate_synchronous_ports
from .tds import run_fault


FAULT_BUSES = (20, 45, 80, 110, 140, 175, 200)


def _load_static_system(case_path: Path, pycode_path: Path):
    system = andes.load(
        str(case_path), setup=False, no_output=True, pycode_path=str(pycode_path)
    )
    if not system.setup() or not system.PFlow.run():
        raise RuntimeError("failed to initialize the IL200 base case")
    return system


def _matrix_csv(matrix: np.ndarray, path: Path) -> None:
    labels = [f"Bus{bus}" for bus in (65, 104, 105, 114, 115, 125, 126, 127, 135, 136, 147)]
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        path, encoding="utf-8-sig"
    )


def run_experiments(project_root: str | Path) -> dict[str, object]:
    root = Path(project_root).resolve()
    case_path = root / "data" / "IL200_ffr.xlsx"
    pycode_path = root / ".andes" / "pycode"
    output = root / "outputs"
    tables = output / "tables"
    figures = output / "figures"
    matrices = output / "matrices"
    pmu = output / "pmu"
    for directory in (tables, figures, matrices, pmu, pycode_path):
        directory.mkdir(parents=True, exist_ok=True)

    system = _load_static_system(case_path, pycode_path)
    direct_cases = {}
    benchmark_rows = []
    for reactance in ("xd2", "xd1"):
        for include_loads in (False, True):
            direct = build_direct_network(
                system, include_loads=include_loads, sg_reactance=reactance
            )
            key = f"{reactance}_{'with_tds_z_load' if include_loads else 'without_load'}"
            result = generalized_scr(direct.ibr_y, direct.ibr_capacities_pu)
            direct_cases[key] = (direct, result)
            benchmark_rows.append(
                {
                    "case": key,
                    "synchronous_reactance": reactance,
                    "constant_impedance_loads_included": include_loads,
                    "gscr": result.value,
                    "minimum_eigenvalue": result.eigenvalues[0],
                    "maximum_eigenvalue": result.eigenvalues[-1],
                    "termination_vs_direct_kron_relative_error": direct.direct_reduction_relative_error,
                }
            )
    benchmark_table = pd.DataFrame(benchmark_rows)
    benchmark_table.to_csv(
        tables / "direct_benchmarks.csv", index=False, encoding="utf-8-sig"
    )

    matched_direct, matched_gscr = direct_cases["xd2_with_tds_z_load"]
    standard_direct, standard_gscr = direct_cases["xd2_without_load"]

    source_rows = []
    for row in matched_direct.sg_model_positions:
        source_rows.append(
            {
                "port_role": "network_support_synchronous_machine",
                "model": str(system.GENROU.idx.v[row]),
                "bus": int(system.GENROU.bus.v[row]),
                "capacity_mva": float(system.GENROU.Sn.v[row]),
                "xd1_machine_base_pu": float(system.GENROU.xd1.v[row]),
                "xd2_machine_base_pu": float(system.GENROU.xd2.v[row]),
                "included_in_ibr_capacity_matrix": False,
            }
        )
    for row in matched_direct.ibr_model_positions:
        source_rows.append(
            {
                "port_role": "final_ibr_evaluation_port",
                "model": str(system.REGCA1.idx.v[row]),
                "bus": int(system.REGCA1.bus.v[row]),
                "capacity_mva": float(system.REGCA1.Sn.v[row]),
                "xd1_machine_base_pu": np.nan,
                "xd2_machine_base_pu": np.nan,
                "included_in_ibr_capacity_matrix": True,
            }
        )
    pd.DataFrame(source_rows).to_csv(
        tables / "source_ports.csv", index=False, encoding="utf-8-sig"
    )

    runs = []
    fault_rows = []
    for fault_bus in FAULT_BUSES:
        fault = run_fault(case_path, pycode_path, fault_bus)
        runs.append(fault)
        np.savez_compressed(
            pmu / f"fault_bus_{fault_bus}.npz",
            time_s=fault.time_s,
            voltage=fault.voltage,
            current=fault.current,
            source_port_buses=fault.source_port_buses,
        )
        fault_rows.append(
            {
                "fault_bus": fault_bus,
                "sample_count": len(fault.time_s),
                "kcl_rmse_pu": fault.kcl_rmse,
                "kcl_max_error_pu": fault.kcl_max_error,
                "max_ibr_voltage_deviation_pu": fault.max_ibr_voltage_deviation,
            }
        )
    fault_table = pd.DataFrame(fault_rows)
    fault_table.to_csv(tables / "fault_runs.csv", index=False, encoding="utf-8-sig")

    identified = identify_port_admittance(runs)
    identified_ibr_y = terminate_synchronous_ports(
        identified.y_hat,
        matched_direct.sg_norton_y,
        len(matched_direct.sg_buses),
    )
    identified_gscr = generalized_scr(
        identified_ibr_y, matched_direct.ibr_capacities_pu
    )
    source_y_relative_error = float(
        np.linalg.norm(identified.y_hat - matched_direct.source_port_y)
        / np.linalg.norm(matched_direct.source_port_y)
    )
    ibr_y_relative_error = float(
        np.linalg.norm(identified_ibr_y - matched_direct.ibr_y)
        / np.linalg.norm(matched_direct.ibr_y)
    )

    identification_table = pd.DataFrame(
        [
            {
                "estimator": identified.estimator,
                "measurement_port_count": identified.port_count,
                "final_ibr_port_count": len(matched_direct.ibr_buses),
                "synchronous_port_count_terminated": len(matched_direct.sg_buses),
                "increment_count": identified.sample_increment_count,
                "voltage_rank": identified.voltage_rank,
                "voltage_condition": identified.voltage_condition,
                "residual_rmse_pu": identified.residual_rmse,
                "solver_iterations": identified.solver_iterations,
                "solver_stop_code": identified.solver_stop_code,
                "theoretical_tds_matched_gscr": matched_gscr.value,
                "identified_gscr": identified_gscr.value,
                "gscr_absolute_error": abs(identified_gscr.value - matched_gscr.value),
                "source_port_y_relative_error": source_y_relative_error,
                "ibr_y_relative_error": ibr_y_relative_error,
            }
        ]
    )
    identification_table.to_csv(
        tables / "identification_summary.csv", index=False, encoding="utf-8-sig"
    )

    participation = pd.DataFrame(
        {
            "bus": matched_direct.ibr_buses,
            "capacity_mva": matched_direct.ibr_capacities_mva,
            "critical_mode_participation": matched_gscr.participation,
            "critical_mode_component": matched_gscr.critical_mode,
        }
    ).sort_values("critical_mode_participation", ascending=False)
    participation.to_csv(
        tables / "critical_mode_participation.csv", index=False, encoding="utf-8-sig"
    )

    _matrix_csv(matched_direct.ibr_y.real, matrices / "theoretical_y_ibr_real.csv")
    _matrix_csv(matched_direct.ibr_y.imag, matrices / "theoretical_y_ibr_imag.csv")
    _matrix_csv(identified_ibr_y.real, matrices / "identified_y_ibr_real.csv")
    _matrix_csv(identified_ibr_y.imag, matrices / "identified_y_ibr_imag.csv")

    fig, axis = plt.subplots(figsize=(8.4, 4.5))
    plot_benchmarks = benchmark_table.copy()
    labels = ["Xd''\nno load", "Xd''\nTDS Z load", "Xd'\nno load", "Xd'\nTDS Z load"]
    ordering = [
        "xd2_without_load",
        "xd2_with_tds_z_load",
        "xd1_without_load",
        "xd1_with_tds_z_load",
    ]
    values = [
        float(plot_benchmarks.loc[plot_benchmarks["case"] == item, "gscr"].iloc[0])
        for item in ordering
    ]
    x = np.arange(len(values))
    axis.bar(x, values, color="#4C78A8", label="Direct network")
    axis.scatter(
        [1], [identified_gscr.value], color="#E45756", s=75, zorder=3, label="PMU identified"
    )
    axis.set_xticks(x, labels)
    axis.set_ylabel("gSCR")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "gscr_benchmarks.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.4, 4.5))
    ordered_participation = participation.sort_values("bus")
    axis.bar(
        ordered_participation["bus"].astype(str),
        ordered_participation["critical_mode_participation"],
        color="#F58518",
    )
    axis.set_xlabel("IBR bus")
    axis.set_ylabel("Critical-mode participation")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "critical_mode_participation.png", dpi=180)
    plt.close(fig)

    reference_fault = runs[3]
    elapsed = reference_fault.time_s - reference_fault.time_s[0]
    initial = np.abs(reference_fault.ibr_voltage[0])
    fig, axis = plt.subplots(figsize=(8.4, 4.8))
    for column, bus in enumerate(matched_direct.ibr_buses):
        axis.plot(
            elapsed,
            np.abs(reference_fault.ibr_voltage[:, column]) - initial[column],
            label=f"Bus{bus}",
        )
    axis.set_xlabel("Time after fault clearing (s)")
    axis.set_ylabel("IBR terminal voltage deviation (pu)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "fault_bus_110_ibr_voltage.png", dpi=180)
    plt.close(fig)

    summary = {
        "engine": f"ANDES {andes.__version__}",
        "case": str(case_path.relative_to(root)),
        "system_base_mva": float(system.config.mva),
        "bus_count": int(system.Bus.n),
        "branch_count": int(system.Line.n),
        "measurement_ports": {
            "count": int(len(matched_direct.source_port_buses)),
            "synchronous_machine_count": int(len(matched_direct.sg_buses)),
            "ibr_count": int(len(matched_direct.ibr_buses)),
        },
        "final_gscr_ports": {
            "count": int(len(matched_direct.ibr_buses)),
            "buses": matched_direct.ibr_buses.tolist(),
            "capacities_mva": matched_direct.ibr_capacities_mva.tolist(),
        },
        "standard_short_circuit_gscr_xd2_without_load": standard_gscr.value,
        "tds_matched_theoretical_gscr_xd2_with_constant_z_load": matched_gscr.value,
        "pmu_identified_gscr": identified_gscr.value,
        "pmu_identified_absolute_error": abs(identified_gscr.value - matched_gscr.value),
        "source_port_y_relative_error": source_y_relative_error,
        "ibr_y_relative_error": ibr_y_relative_error,
        "voltage_increment_rank": identified.voltage_rank,
        "voltage_increment_condition": identified.voltage_condition,
        "identification_residual_rmse_pu": identified.residual_rmse,
        "fault_buses": list(FAULT_BUSES),
        "maximum_kcl_rmse_pu": float(fault_table["kcl_rmse_pu"].max()),
        "validation_thresholds": {
            "full_voltage_rank": 49,
            "maximum_voltage_condition": 5e4,
            "maximum_gscr_absolute_error": 1e-3,
            "maximum_source_port_y_relative_error": 1e-2,
            "maximum_ibr_y_relative_error": 5e-3,
            "maximum_kcl_rmse_pu": 3e-4,
            "maximum_termination_identity_relative_error": 1e-10,
        },
        "interpretation": (
            "All 49 active-source terminals are used only for passive-network identification. "
            "The 38 synchronous-machine terminals are then terminated with machine-base xd2 "
            "Norton admittances converted to the 100-MVA system base. The final capacity-normalized "
            "gSCR is calculated exclusively at the 11 REGCA1 IBR terminals."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
