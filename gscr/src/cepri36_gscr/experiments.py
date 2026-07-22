"""End-to-end reproducibility workflow and artifact generation."""

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

from .identification import identify_symmetric_admittance
from .model import build_network_model
from .psasp import load_psasp_case
from .scenarios import (
    bus50_profile,
    generate_scenario,
    resample_scenario,
    scenario_to_frame,
)
from .validation import validate_branch_conventions


PAPER_VALUES = {
    "actual_gscr": 0.1701,
    "fault_bus30": 0.1639,
    "fault_bus25": 0.1630,
    "fault_line_bus9_bus23": 0.1633,
    "load_bus50": 0.1650,
}


def _write_matrix(matrix: np.ndarray, names: list[str], path: Path) -> None:
    pd.DataFrame(matrix, index=names, columns=names).to_csv(path, encoding="utf-8-sig")


def _identify(data, model):
    return identify_symmetric_admittance(data.voltage, data.current, model.capacities_pu)


def run_experiments(project_root: str | Path) -> dict[str, object]:
    root = Path(project_root).resolve()
    raw = root / "data" / "raw_psasp"
    output = root / "outputs"
    figures = output / "figures"
    tables = output / "tables"
    pmu = output / "pmu"
    for directory in (figures, tables, pmu):
        directory.mkdir(parents=True, exist_ok=True)

    case = load_psasp_case(raw)
    validation = validate_branch_conventions(case)
    passive_model = build_network_model(case, include_loads=False, include_hvdc=False)
    load_model = build_network_model(case, include_loads=True, include_hvdc=False)
    model = build_network_model(case, include_loads=True, include_hvdc=True)

    generator_by_name = {generator.name.upper(): generator for generator in case.generators}
    capacities = pd.DataFrame(
        [
            {
                "bus": f"BUS{k}",
                "psasp_bus_index": generator_by_name[f"BUS{k}"].bus,
                "capacity_mva": generator_by_name[f"BUS{k}"].capacity_mva,
                "rated_mw": generator_by_name[f"BUS{k}"].rated_mw,
                "capacity_pu_on_100_mva": generator_by_name[f"BUS{k}"].capacity_mva / 100.0,
            }
            for k in range(1, 9)
        ]
    )
    capacities.to_csv(tables / "generator_capacities.csv", index=False, encoding="utf-8-sig")

    benchmark = pd.DataFrame(
        [
            {"model": "passive branches only", "gscr": passive_model.gscr},
            {"model": "constant-Z loads", "gscr": load_model.gscr},
            {"model": "constant-Z loads + HVDC terminal equivalents", "gscr": model.gscr},
            {"model": "paper reported actual", "gscr": PAPER_VALUES["actual_gscr"]},
        ]
    )
    benchmark["difference_from_paper"] = benchmark["gscr"] - PAPER_VALUES["actual_gscr"]
    benchmark.to_csv(tables / "network_benchmark.csv", index=False, encoding="utf-8-sig")

    validation_frame = pd.DataFrame([validation])
    validation_frame.to_csv(tables / "psasp_branch_validation.csv", index=False, encoding="utf-8-sig")
    eigenvalues = pd.DataFrame(
        {
            "order": np.arange(1, len(model.eigenvalues) + 1),
            "direct_eigenvalue": model.eigenvalues,
        }
    )
    eigenvalues.to_csv(tables / "direct_eigenvalues.csv", index=False, encoding="utf-8-sig")
    _write_matrix(model.reduced_y.real, case.retained_bus_names, tables / "reduced_y_real.csv")
    _write_matrix(model.reduced_y.imag, case.retained_bus_names, tables / "reduced_y_imag.csv")

    fault_rows: list[dict[str, object]] = []
    fault_data: dict[str, object] = {}
    for offset, event in enumerate(("fault_bus30", "fault_bus25", "fault_line_bus9_bus23")):
        high_resolution = generate_scenario(case, model, event, random_seed=202507 + offset)
        sampled = resample_scenario(high_resolution, 0.1, duration_s=10.0)
        result = _identify(sampled, model)
        fault_data[event] = sampled
        fault_rows.append(
            {
                "scenario": event,
                "paper_actual": PAPER_VALUES["actual_gscr"],
                "paper_identified": PAPER_VALUES[event],
                "reproduced_direct": model.gscr,
                "reproduced_identified": result.gscr,
                "absolute_error_to_rebuilt_network": abs(result.gscr - model.gscr),
                "residual_rmse": result.residual_rmse,
                "design_rank": result.design_rank,
                "design_condition": result.design_condition,
            }
        )
    fault_table = pd.DataFrame(fault_rows)
    fault_table.to_csv(tables / "fault_scenarios.csv", index=False, encoding="utf-8-sig")

    load_high_resolution = generate_scenario(case, model, "load_bus50", random_seed=202601)
    load_sampled = resample_scenario(load_high_resolution, 0.1, duration_s=10.0)
    load_result = _identify(load_sampled, model)
    load_table = pd.DataFrame(
        [
            {
                "scenario": "load_bus50",
                "paper_actual": PAPER_VALUES["actual_gscr"],
                "paper_identified": PAPER_VALUES["load_bus50"],
                "reproduced_direct": model.gscr,
                "reproduced_identified": load_result.gscr,
                "absolute_error_to_rebuilt_network": abs(load_result.gscr - model.gscr),
                "residual_rmse": load_result.residual_rmse,
                "design_rank": load_result.design_rank,
                "design_condition": load_result.design_condition,
            }
        ]
    )
    load_table.to_csv(tables / "load_scenario.csv", index=False, encoding="utf-8-sig")

    # A fixed, very small acquisition noise makes subsampling/window comparisons
    # non-degenerate without changing the network benchmark.
    robust_source = generate_scenario(
        case,
        model,
        "load_bus50",
        measurement_noise_power=2.5e-10,
        random_seed=202602,
    )
    sampling_rows = []
    for interval in (0.1, 0.2, 0.5, 0.8, 1.0):
        sampled = resample_scenario(robust_source, interval, duration_s=10.0)
        result = _identify(sampled, model)
        sampling_rows.append(
            {
                "sample_interval_s": interval,
                "sample_count": result.sample_count,
                "identified_gscr": result.gscr,
                "absolute_error": abs(result.gscr - model.gscr),
                "design_rank": result.design_rank,
                "condition_number": result.design_condition,
            }
        )
    sampling_table = pd.DataFrame(sampling_rows)
    sampling_table.to_csv(tables / "sampling_interval.csv", index=False, encoding="utf-8-sig")

    window_rows = []
    for window in (10.0, 8.0, 6.0, 4.0):
        sampled = resample_scenario(robust_source, 0.1, duration_s=window)
        result = _identify(sampled, model)
        window_rows.append(
            {
                "window_s": window,
                "sample_count": result.sample_count,
                "identified_gscr": result.gscr,
                "absolute_error": abs(result.gscr - model.gscr),
                "design_rank": result.design_rank,
                "condition_number": result.design_condition,
            }
        )
    window_table = pd.DataFrame(window_rows)
    window_table.to_csv(tables / "window_length.csv", index=False, encoding="utf-8-sig")

    scenario_to_frame(fault_data["fault_bus30"], case.retained_bus_names).to_csv(
        pmu / "fault_bus30_pmu.csv", index=False, encoding="utf-8-sig"
    )
    scenario_to_frame(load_sampled, case.retained_bus_names).to_csv(
        pmu / "load_bus50_pmu.csv", index=False, encoding="utf-8-sig"
    )

    plt.figure(figsize=(7.2, 3.8))
    plt.plot(load_high_resolution.time_s, bus50_profile(load_high_resolution.time_s), linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Bus50 load change")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures / "bus50_load_profile.png", dpi=180)
    plt.close()

    bus30 = fault_data["fault_bus30"]
    u0 = bus30.voltage[0]
    i0 = bus30.current[0]
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True)
    for position, bus in enumerate(case.retained_bus_names):
        axes[0].plot(bus30.time_s, np.abs(bus30.voltage[:, position] - u0[position]), label=bus)
        axes[1].plot(bus30.time_s, np.abs(bus30.current[:, position] - i0[position]), label=bus)
    axes[0].set_ylabel("|ΔU| (p.u.)")
    axes[1].set_ylabel("|ΔI| (p.u.)")
    axes[1].set_xlabel("Time (s)")
    axes[0].grid(alpha=0.2)
    axes[1].grid(alpha=0.2)
    axes[0].legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "fault_bus30_phasor_changes.png", dpi=180)
    plt.close(fig)

    plt.figure(figsize=(7.5, 4.0))
    comparison = pd.concat(
        [
            fault_table[["scenario", "paper_identified", "reproduced_identified"]],
            load_table[["scenario", "paper_identified", "reproduced_identified"]],
        ],
        ignore_index=True,
    )
    x = np.arange(len(comparison))
    plt.bar(x - 0.18, comparison["paper_identified"], width=0.36, label="Paper")
    plt.bar(x + 0.18, comparison["reproduced_identified"], width=0.36, label="Independent reproduction")
    plt.axhline(model.gscr, color="black", linestyle="--", linewidth=1, label="Rebuilt-network gSCR")
    plt.xticks(x, ["Bus30", "Bus25", "Line 9-23", "Bus50 load"])
    plt.ylabel("gSCR")
    plt.ylim(0.155, 0.177)
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(figures / "gscr_comparison.png", dpi=180)
    plt.close()

    summary = {
        "rebuilt_network_gscr": model.gscr,
        "paper_actual_gscr": PAPER_VALUES["actual_gscr"],
        "relative_difference_percent": 100.0 * (model.gscr / PAPER_VALUES["actual_gscr"] - 1.0),
        "psasp_validation": validation,
        "retained_buses": case.retained_bus_names,
        "capacity_mva": capacities["capacity_mva"].tolist(),
        "clean_fault_identified_mean": float(fault_table["reproduced_identified"].mean()),
        "clean_load_identified": float(load_result.gscr),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "case": case,
        "model": model,
        "summary": summary,
        "capacities": capacities,
        "benchmark": benchmark,
        "validation": validation_frame,
        "faults": fault_table,
        "load": load_table,
        "sampling": sampling_table,
        "windows": window_table,
        "eigenvalues": eigenvalues,
    }
