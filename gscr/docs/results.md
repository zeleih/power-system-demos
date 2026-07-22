# Reference results and validation

## CEPRI36 PSASP-reference workflow

| Quantity | Value |
|---|---:|
| Paper direct gSCR | 0.1701 |
| Reconstructed-network gSCR | 0.1714094976 |
| Difference from paper | 0.7698% |
| Mean identified gSCR, clean fault records | 0.1714094976 |
| Identified gSCR, load record | 0.1714094976 |
| Real independent-parameter rank | 72/72 |

The published matrix and archived PMU records preserve the direct and identified
benchmarks. The raw PSASP load-flow records used during preparation are not
redistributed.

## CEPRI36 ANDES workflow

| Scenario | ANDES direct | Analytical identification | Absolute error |
|---|---:|---:|---:|
| Bus30 fault | 0.1714094976 | 0.1714098519 | 3.54e-7 |
| Bus25 fault | 0.1714094976 | 0.1714098579 | 3.60e-7 |
| Bus9–Bus23 midpoint fault | 0.1714094976 | 0.1714099770 | 4.79e-7 |
| Bus50 load variation | 0.1722054217 | 0.1728156898 | 6.10e-4 |

The ANDES power-flow voltage maximum error relative to the saved PSASP solution
is below `8.2e-7 pu` for the fault cases. Port-current KCL RMSE is below
`3.4e-6 pu`.

## IL200 workflow

| Quantity | Value |
|---|---:|
| Bus count | 200 |
| Branch count | 245 |
| PMU measurement ports | 49 |
| Synchronous-machine ports terminated | 38 |
| Final IBR evaluation ports | 11 |
| Standard no-load direct gSCR | 0.8109979585 |
| TDS constant-Z matched direct gSCR | 0.8935618720 |
| PMU analytical gSCR | 0.8935644622 |
| gSCR absolute error | 2.59e-6 |
| 49-port admittance relative error | 0.2355% |
| 11-port admittance relative error | 0.0649% |
| Voltage-increment rank | 49/49 |

Machine-readable results are written to:

- `cases/cepri36/outputs/summary.json`;
- `cases/cepri36/outputs/andes/summary.json`;
- `cases/il200/outputs/summary.json`.
