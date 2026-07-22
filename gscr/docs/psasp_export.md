# PSASP PMU export interface

The repository does not automate the proprietary PSASP graphical interface.
Instead, `scripts/identify_cepri36_psasp_export.py` accepts an exported CSV with
Bus1–Bus8 terminal voltage and current phasors.

Required columns are:

```text
time_s,
U_BUS1_re,U_BUS1_im,I_BUS1_re,I_BUS1_im,
U_BUS2_re,U_BUS2_im,I_BUS2_re,I_BUS2_im,
...
U_BUS8_re,U_BUS8_im,I_BUS8_re,I_BUS8_im
```

The reference current direction is from each retained port into the external
network. All voltage and current phasors must use one common synchronous
reference frame.

Run:

```bash
python scripts/identify_cepri36_psasp_export.py path/to/psasp_pmu.csv
```

The script reports the analytical identified gSCR, the reconstructed-network
reference, residual RMSE, and real parameter rank.

`scripts/run_cepri36_psasp.py` provides a publishable fallback that identifies
the archived PSASP-compatible PMU records against the included reference
eight-port matrix. The archived records are teaching/reproduction data and are
not presented as the unpublished PSASP transient waveforms used in the article.

The proprietary PSASP graphical workflow is not automated. Raw PSASP example
records are also not redistributed. If local redistribution/use terms permit,
place the expected records under `cases/cepri36/data/raw_psasp/` to enable the
full record-reconstruction and ANDES-regeneration modules.
