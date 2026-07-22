# Analytical PMU-Based gSCR Identification

[中文说明](README_zh.md)

This case reproduces and extends the analytical generalized short-circuit
ratio (gSCR) identification method proposed in:

> Z. Han, P. Ju, H. Li, and Y. Liu, "Analytical Identification Method of
> Generalized Short-Circuit Ratio Using Phasor Measurement Units," *IET
> Generation, Transmission & Distribution*, vol. 19, no. 1, e70026, 2025.
> <https://doi.org/10.1049/gtd2.70026>

Only the publisher link and citation metadata are included; the article PDF is
not redistributed here.

## What is included

- a three-port simulator-independent example for learning and testing;
- a CEPRI/EPRI 36-bus reproduction with two workflows:
  - archived PSASP-compatible PMU verification and a PSASP PMU-CSV import interface;
  - an independently converted ANDES case with archived time-domain PMU data;
- an ANDES IL200 extension for a mixed synchronous-machine/IBR system;
- the paper's accumulated, non-iterative analytical identification code;
- reference PMU data, matrices, tables, figures, and machine-readable summaries.

The project reproduces the method and comparable results. It does not claim a
sample-by-sample reproduction of the original unpublished PSASP waveforms.

## Method

For retained ports, consecutive PMU increments satisfy

\[
\Delta I_k = \bar{Y}\,\Delta U_k.
\]

The independent conductance and susceptance entries of the complex-symmetric
matrix \(\bar{Y}=G+jB\) are obtained by minimizing the accumulated squared
residual. Each PMU batch contributes only to fixed-size matrices,

\[
C_U=\sum_k \Delta U_k^{\mathrm H}\Delta U_k,\qquad
C_{UI}=\sum_k \Delta U_k^{\mathrm H}\Delta I_k,
\]

after which the real \(G/B\) analytical equations are solved once. See
[docs/method.md](docs/method.md).

## Reference results

| Case | Direct/reference gSCR | Identified gSCR | Interpretation |
|---|---:|---:|---|
| Toy three-port | 3.59367518 | 3.59367518 | Known analytical network |
| CEPRI36 published PSASP reference | 0.17140950 | 0.17140950 | Eight aggregated IBR ports |
| CEPRI36 ANDES, Bus30 fault | 0.17140950 | 0.17140985 | Eight aggregated IBR ports |
| IL200, standard short-circuit convention | 0.81099796 | — | No-load, synchronous support retained |
| IL200, ANDES TDS matched convention | 0.89356187 | 0.89356446 | Constant-impedance loads included |

The paper reports `0.1701` as the CEPRI36 direct value. The reconstructed
network value differs by about `0.77%`.

## Quick start

```bash
python -m venv .venv
python -m pip install -e ".[notebook]"
python scripts/run_toy3.py
python scripts/build_notebook.py
python scripts/execute_notebook.py
python scripts/run_cepri36_psasp.py
python scripts/run_cepri36_andes.py
python scripts/run_il200.py
python scripts/validate_all.py
python scripts/build_manifest.py
python -m unittest discover -s tests -v
```

The default CEPRI36 commands verify the archived PMU data without requiring raw
PSASP records. Users with authorized local records can place them under
`cases/cepri36/data/raw_psasp/` to trigger full ANDES time-domain regeneration.
The IL200 command regenerates its ANDES time-domain PMU records and therefore
takes longer. Generated files are written below `cases/*/outputs/`.

## Case map

- [CEPRI36 PSASP and ANDES workflows](cases/cepri36/README.md)
- [Executable method walkthrough](notebooks/analytical_gscr_walkthrough.ipynb)
- [IL200 mixed-source workflow](cases/il200/README.md)
- [Port definitions](docs/port_definition.md)
- [Results and validation](docs/results.md)
- [PSASP PMU export format](docs/psasp_export.md)
- [Data and software provenance](THIRD_PARTY.md)
- [Reference artifact hashes](results/reference/artifact_manifest.json)
