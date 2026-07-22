from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "analytical_gscr_walkthrough.ipynb"


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python"}
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Analytical PMU-Based gSCR Identification\n\n"
            "This tutorial implements the accumulated, non-iterative method in "
            "[Han et al. (2025)](https://doi.org/10.1049/gtd2.70026)."
        ),
        nbf.v4.new_markdown_cell(
            "## Goal\n\n"
            "Use a known three-port complex-symmetric network to verify the full "
            "PMU-increment → analytical admittance → gSCR workflow."
        ),
        nbf.v4.new_markdown_cell(
            "## Setup\n\n"
            "The example is deterministic and does not require ANDES or PSASP. "
            "The larger cases use the same identifier."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import sys\n\n"
            "import numpy as np\n"
            "import pandas as pd\n\n"
            "ROOT = Path.cwd().resolve()\n"
            "if ROOT.name == 'notebooks':\n"
            "    ROOT = ROOT.parent\n"
            "sys.path.insert(0, str(ROOT / 'src'))\n\n"
            "from gscr_demo import PMURun, generalized_scr, identify_symmetric_admittance"
        ),
        nbf.v4.new_markdown_cell(
            "## Steps\n\n### 1. Define a symmetric network and port capacities"
        ),
        nbf.v4.new_code_cell(
            "conductance = np.array([\n"
            "    [0.30, -0.05, -0.02],\n"
            "    [-0.05, 0.25, -0.04],\n"
            "    [-0.02, -0.04, 0.20],\n"
            "])\n"
            "susceptance = np.array([\n"
            "    [8.0, -2.0, -1.0],\n"
            "    [-2.0, 7.0, -1.5],\n"
            "    [-1.0, -1.5, 6.0],\n"
            "])\n"
            "true_y = conductance - 1j * susceptance\n"
            "capacities_pu = np.array([1.5, 1.0, 0.8])\n"
            "pd.DataFrame(true_y, index=['P1', 'P2', 'P3'], columns=['P1', 'P2', 'P3'])"
        ),
        nbf.v4.new_markdown_cell(
            "### 2. Generate PMU phasors\n\n"
            "Current is calculated with the same retained-port reference direction "
            "used by the identification model."
        ),
        nbf.v4.new_code_cell(
            "rng = np.random.default_rng(20250701)\n"
            "voltage = rng.standard_normal((160, 3)) + 1j * rng.standard_normal((160, 3))\n"
            "current = voltage @ true_y.T\n"
            "voltage.shape, current.shape"
        ),
        nbf.v4.new_markdown_cell(
            "### 3. Accumulate the analytical equations and solve once"
        ),
        nbf.v4.new_code_cell(
            "identified = identify_symmetric_admittance([PMURun(voltage, current)])\n"
            "identified.y_hat"
        ),
        nbf.v4.new_markdown_cell("### 4. Calculate and compare gSCR"),
        nbf.v4.new_code_cell(
            "direct_gscr = generalized_scr(true_y, capacities_pu).value\n"
            "identified_gscr = generalized_scr(identified.y_hat, capacities_pu).value\n"
            "relative_y_error = np.linalg.norm(identified.y_hat - true_y) / np.linalg.norm(true_y)\n\n"
            "pd.DataFrame([{\n"
            "    'direct_gscr': direct_gscr,\n"
            "    'identified_gscr': identified_gscr,\n"
            "    'absolute_gscr_error': abs(identified_gscr - direct_gscr),\n"
            "    'relative_y_error': relative_y_error,\n"
            "    'voltage_rank': identified.voltage_rank,\n"
            "    'real_parameter_rank': identified.parameter_rank_real,\n"
            "}])"
        ),
        nbf.v4.new_markdown_cell("## Checks"),
        nbf.v4.new_code_cell(
            "assert identified.voltage_rank == 3\n"
            "assert identified.parameter_rank_real == 12\n"
            "assert relative_y_error < 1e-12\n"
            "assert abs(identified_gscr - direct_gscr) < 1e-12\n"
            "print('All analytical identification checks passed.')"
        ),
        nbf.v4.new_markdown_cell(
            "## Next Steps\n\n"
            "Run `scripts/run_cepri36_psasp.py` for the PSASP-record workflow, "
            "`scripts/run_cepri36_andes.py` for the open ANDES reconstruction, "
            "and `scripts/run_il200.py` for the mixed synchronous-machine/IBR extension. "
            "See the case READMEs for port definitions and reference values."
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
