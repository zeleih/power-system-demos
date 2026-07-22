# CEPRI36: PSASP and ANDES reproductions

This case has two reproducible paths sharing the same eight-port analytical
identifier.

## PSASP path

- `scripts/run_cepri36_psasp.py` re-identifies the archived, PSASP-compatible
  PMU records against the published eight-port reference matrix;
- `scripts/identify_cepri36_psasp_export.py` processes actual PMU phasors
  exported from PSASP.

Raw PSASP example records are not redistributed because their redistribution
terms have not been confirmed. Users who are permitted to use them locally can
place them under `data/raw_psasp/`; that path is ignored by Git.

## ANDES path

- `data/andes/CEPRI36_andes.xlsx` is the converted ANDES workbook;
- the archived PMU records and result tables allow direct verification of the
  four time-domain scenarios;
- `scripts/build_cepri36_andes_case.py` and `scripts/run_cepri36_andes.py`
  support full regeneration when the local PSASP source records are supplied.

The conversion uses eight `GENCLS` units, steady-state two-terminal PQ
equivalents for the original HVDC terminals, and ANDES constant-impedance load
conversion during TDS.
