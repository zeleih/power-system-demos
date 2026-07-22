"""End-to-end ANDES reproduction and artifact generation."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".mplconfig")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .andes_case import build_andes_workbook
from .andes_reproduction import SUPPORTED_SCENARIOS, run_andes_scenario
from .experiments import PAPER_VALUES
from .identification import identify_symmetric_admittance
from .psasp import load_psasp_case
from .scenarios import scenario_to_frame


def run_andes_experiments(project_root: str | Path) -> dict[str, object]:
    root = Path(project_root).resolve()
    raw = root / "data" / "raw_psasp"
    workbook = root / "data" / "andes" / "CEPRI36_andes.xlsx"
    pycode = root / ".andes" / "pycode"
    output = root / "outputs" / "andes"
    tables = output / "tables"
    figures = output / "figures"
    pmu = output / "pmu"
    for directory in (tables, figures, pmu):
        directory.mkdir(parents=True, exist_ok=True)

    case = load_psasp_case(raw)
    build = build_andes_workbook(case, workbook)

    runs = {}
    scenario_rows = []
    for scenario in SUPPORTED_SCENARIOS:
        run = run_andes_scenario(
            case,
            workbook,
            pycode,
            scenario,
            final_time_s=11.5,
            integration_step_s=0.01,
        )
        sampled = run.sample(0.1, 10.0)
        identified = identify_symmetric_admittance(
            sampled.voltage,
            sampled.current,
            run.capacities_pu,
        )
        runs[scenario] = run
        scenario_to_frame(sampled, case.retained_bus_names).to_csv(
            pmu / f"{scenario}_pmu.csv",
            index=False,
            encoding="utf-8-sig",
        )
        scenario_rows.append(
            {
                "scenario": scenario,
                "paper_identified": PAPER_VALUES[scenario],
                "andes_direct_gscr": run.direct_gscr,
                "andes_identified_gscr": identified.gscr,
                "absolute_error_to_andes_direct": abs(identified.gscr - run.direct_gscr),
                "relative_error_percent": 100.0
                * abs(identified.gscr - run.direct_gscr)
                / abs(run.direct_gscr),
                "residual_rmse": identified.residual_rmse,
                "design_rank": identified.design_rank,
                "design_condition": identified.design_condition,
                "port_current_kcl_rmse": run.port_current_kcl_rmse,
                "pflow_voltage_rmse": run.pflow_voltage_rmse,
                "pflow_voltage_max_error": run.pflow_voltage_max_error,
                "tds_steps": run.tds_steps,
                "sample_count": identified.sample_count,
            }
        )

    scenario_table = pd.DataFrame(scenario_rows)
    scenario_table.to_csv(tables / "scenario_results.csv", index=False, encoding="utf-8-sig")

    benchmark_run = runs["fault_bus30"]
    sampling_rows = []
    for interval in (0.1, 0.2, 0.5, 0.8, 1.0):
        result = benchmark_run.identify(interval, 10.0)
        sampling_rows.append(
            {
                "sample_interval_s": interval,
                "sample_count": result.sample_count,
                "andes_direct_gscr": benchmark_run.direct_gscr,
                "identified_gscr": result.gscr,
                "absolute_error": abs(result.gscr - benchmark_run.direct_gscr),
                "design_rank": result.design_rank,
                "condition_number": result.design_condition,
                "residual_rmse": result.residual_rmse,
            }
        )
    sampling_table = pd.DataFrame(sampling_rows)
    sampling_table.to_csv(tables / "sampling_interval.csv", index=False, encoding="utf-8-sig")

    window_rows = []
    for window in (10.0, 8.0, 6.0, 4.0):
        result = benchmark_run.identify(0.1, window)
        window_rows.append(
            {
                "window_s": window,
                "sample_count": result.sample_count,
                "andes_direct_gscr": benchmark_run.direct_gscr,
                "identified_gscr": result.gscr,
                "absolute_error": abs(result.gscr - benchmark_run.direct_gscr),
                "design_rank": result.design_rank,
                "condition_number": result.design_condition,
                "residual_rmse": result.residual_rmse,
            }
        )
    window_table = pd.DataFrame(window_rows)
    window_table.to_csv(tables / "window_length.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8.0, 4.2))
    x = np.arange(len(scenario_table))
    width = 0.25
    plt.bar(x - width, scenario_table["paper_identified"], width, label="Paper identified")
    plt.bar(x, scenario_table["andes_direct_gscr"], width, label="ANDES direct")
    plt.bar(x + width, scenario_table["andes_identified_gscr"], width, label="ANDES identified")
    plt.xticks(x, ["Bus30", "Bus25", "Line 9-23", "Bus50 load"])
    plt.ylabel("gSCR")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(figures / "scenario_gscr_comparison.png", dpi=180)
    plt.close()

    bus30 = benchmark_run.sample(0.02, 4.0)
    elapsed = bus30.time_s - bus30.time_s[0]
    initial_magnitude = np.abs(bus30.voltage[0])
    unwrapped_angle = np.unwrap(np.angle(bus30.voltage), axis=0)
    relative_angle = unwrapped_angle - unwrapped_angle[:, [0]]
    relative_angle -= relative_angle[[0], :]
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.2), sharex=True)
    for position, bus_name in enumerate(case.retained_bus_names):
        axes[0].plot(
            elapsed,
            np.abs(bus30.voltage[:, position]) - initial_magnitude[position],
            label=bus_name,
        )
        axes[1].plot(elapsed, np.rad2deg(relative_angle[:, position]), label=bus_name)
    axes[0].set_ylabel("Voltage magnitude deviation (pu)")
    axes[1].set_ylabel("Angle deviation relative to BUS1 (deg)")
    axes[1].set_xlabel("Time after fault clearing (s)")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "fault_bus30_andes_response.png", dpi=180)
    plt.close(fig)

    summary = {
        "engine": "ANDES 2.0.0",
        "case_workbook": str(workbook.relative_to(root)),
        "model_counts": build.model_counts,
        "model_assumptions": build.assumptions,
        "paper_actual_gscr": PAPER_VALUES["actual_gscr"],
        "psasp_rebuilt_direct_gscr": runs["fault_bus30"].direct_gscr,
        "scenarios": scenario_table.to_dict(orient="records"),
        "sampling": sampling_table.to_dict(orient="records"),
        "windows": window_table.to_dict(orient="records"),
        "validation_thresholds": {
            "pflow_voltage_max_error_pu": 2.0e-6,
            "port_current_kcl_rmse_pu": 1.0e-5,
            "clean_fault_identification_absolute_error": 1.0e-4,
            "required_real_design_rank": 72,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary
