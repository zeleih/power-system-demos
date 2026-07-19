# Deadband-Edge Frequency Accumulation — Reproduction Package

Simulation code, case files, archived inputs, and reference outputs for the paper:

> **Deadband-Edge Frequency Accumulation in Converter-Rich Power Systems:
> Mechanism and Heterogeneous Deadband Design**

The study identifies a deadband-edge frequency accumulation phenomenon — wind,
PV, and storage sharing one deadband boundary engage/disengage synchronously
and leave a boundary-dwelling shoulder in the day-long frequency-deviation
distribution — and proposes a safety-first heterogeneous deadband design
(storage 15 mHz ≤ PV 25 mHz ≤ wind 36 mHz) evaluated through 96-dispatch
sequential hot-start simulation of the Illinois 200-bus system on ANDES/AMS.

Every quantitative claim in the paper can be regenerated from this repository.
See **[REPRODUCE.md](REPRODUCE.md)** for the step-by-step pipeline.

## Repository layout

```
env/                  Environment setup: pinned requirements, setup_env.sh, and
                      the exact patches applied to ANDES/AMS dev versions
cases/                ANDES/AMS case files (dynamic case, aligned OPF case,
                      unified daily curve CurveInterp.csv)
data/dispatches_day96 The 96 archived OPF dispatch JSONs used by all paper runs
                      (bit-reproducible from cases/, see tools/smoke_check.sh)
scripts/              Simulation drivers: dispatch prep, hot-start day runner,
                      Stage-1 sweep, full-day ranking, ablation, load-step,
                      inertia screen, hot-start equivalence checker
paper/latex/figures/  Figure-generation scripts (fig3–fig11 of the paper)
results/              Reference outputs from the paper runs (CSV summaries and
                      traces; large per-run binaries and checkpoints excluded)
tools/                smoke_check.sh + check_dispatch_regen.py
```

## Quick start

```bash
bash env/setup_env.sh            # venv + pinned deps + patched ANDES/AMS
source .venv/bin/activate
bash tools/smoke_check.sh        # verifies patches + dispatch determinism
```

The smoke check regenerates the hour-0 OPF dispatches from `cases/` and
compares them against `data/dispatches_day96/`; on the original environment
the match is exact (max |diff| = 0.0).

## Verified reproduction status

The full reviewer path was executed on a clean machine state (fresh clone,
fresh venv created by `env/setup_env.sh`, CURENT repositories cloned from
GitHub at the pinned commits, patches applied):

1. **Environment build** — `setup_env.sh` completes; patched PVD1/ESD1
   deadband extensions (`fdbd`/`fdbdu`/`Tfdb`) present.
2. **OPF dispatch determinism** — regenerated hour-0 dispatches match the
   archived `data/dispatches_day96/` with max |diff| = 0.0 (pg, pd, vBus, obj).
3. **Dynamic simulation determinism** — the hot-start window h11d2→h11d3
   (uniform and 36/25/15 deadbands, nominal inertia) rerun in the fresh
   environment matches the archived paper traces bit-for-bit:
   max |diff| = 0.0 for frequency deviation, all three droop aggregates,
   engaged-unit counts, and storage SOC over all 900 samples.
4. **Multi-interval spot-check** (four independent re-runs, all bitwise
   exact, max |diff| = 0.0):
   - inertia-screen windows h7d3→h8d2 and h20d3→h21d0 (uniform + 36/25/15;
     2 × 57,600 trace values each, byte-identical CSVs);
   - ESD-only ablation point 15 mHz on windows h2d3→h3d0 and h15d1→h15d2
     (all 18 window metrics identical to the archived ablation CSV);
   - full-day **baseline replay**: hour 12 (h12d0–h12d3) resumed from the
     archived `end_h11d3` checkpoint reproduces the archived per-second
     frequency traces byte-for-byte, confirming the checkpoint chain.

Cross-platform note: bitwise identity is expected on macOS/arm64 with the
pinned versions. On a different OS/BLAS stack, floating-point differences at
machine-epsilon scale may appear; all paper-level statistics are insensitive
to these.

## Environment

| Component | Version |
|---|---|
| Python | 3.12 (3.12.13 used for archived results) |
| ANDES | [CURENT/andes](https://github.com/CURENT/andes) @ `eda5163c9` + `env/patches/andes-eda5163c9-deadband.patch` |
| AMS (ltbams) | [CURENT/ams](https://github.com/CURENT/ams) @ `38325a1c` + `env/patches/ams-38325a1c-compat.patch` |
| CVXPY / kvxopt / PYPOWER | 1.6.5 / 1.3.3.1 / 5.1.19 |
| Convex solvers (pinned) | clarabel 0.11.1, scs 3.2.11, osqp 1.1.1, highspy 1.14.0 |
| NumPy / SciPy / pandas / matplotlib | 2.4.4 / 1.17.1 / 3.0.2 / 3.10.8 |

The ANDES patch adds to PVD1/ESD1: a symmetric upper deadband (`fdbdu`), an
optional first-order lag on the droop output (`Tfdb`), and an explicit
available-power limit (`pavail0`). The AMS patch contains compatibility shims
for the ANDES 2.0 API. Full freeze: `env/requirements-frozen.txt`.

## Configuration ↔ paper cross-check

All settings below are encoded in the case files and runner defaults, recorded
in the run manifests under `results/*/`, and were verified against the
manuscript:

| Paper statement | Where encoded |
|---|---|
| Illinois 200-bus; 25 SG (TGOV1NDB), 22 PVD1 (12 wind WT_, 10 PV PV_), 2 ESD1 | `cases/IL200_dyn_db2_stable_tgov1ndb_restore0_17_*_tip100.xlsx` |
| Governor deadband tiers 0/17/36 mHz = 21/1/3 units, 3.316/7.300/1.473 pu (pmax) | same case file (`TGOV1NDB.dbU`, `Sn = 1.2×pmax` except the 730 MVA unit) |
| DER deadband baseline ±36 mHz, ddn 0.333 (PVD1) / 1.667 (ESD1), tip=tiq=1.0 s | same case file (`fdbd/fdbdu/ddn/tip` device-base inputs) |
| Available-power factor γ = 0.98 | `--wind-pref-alpha 0.98 --solar-pref-alpha 0.98` |
| AGC: KP=0.1, KI=0.002, 4 s, headroom-aware, freeze-on-saturation, DER AGC off | runner flags, recorded in `results/*/phase1_*manifest.json` |
| 96 dispatch intervals × 900 s; ramp-limited governor basepoints | `--dispatch-interval 900`, `--governor-target-schedule ramp_limited_basepoint` |
| Design grids ESD {15,20,25,30} / PV {25,30,36,42} / wind {36,45,55,65} mHz; 56 ordered candidates | `sweep_deadband_phase1_windows.py` arguments (manifest) |
| Safety screens max\|Δf\|≤0.10 Hz, share>0.05 Hz ≤5%; lexicographic ranking | sweep thresholds + `run_deadband_phase1_day.py::rank_summary` |
| ESD-only ablation {10,...,36} mHz; load steps ±5% @ t=120 s; inertia ×{0.5,1.0,1.25} | dedicated runners, see REPRODUCE.md steps 6–8 |

## Reference outputs

`results/` ships the summary CSVs and traces behind every table and figure of
the paper (Tables I–III, Figs. 3–11), so figures can be rebuilt without
re-running the day-long simulations:

```bash
cd paper/latex/figures
python plot_fig3_baseline_dist.py   # ... through plot_fig11_inertia_screen.py
python ../../../scripts/plot_minimal_model.py
```

## License and citation

This package is released under **GPL-3.0** (see `LICENSE`): the patches in
`env/patches/` are derivative works of ANDES and AMS, which are GPL-licensed
open-source projects of [CURENT LTB](https://github.com/CURENT). Please cite
the paper when using this package — see `CITATION.cff`. The Illinois 200-bus
synthetic network is from Birchfield *et al.*, IEEE Trans. Power Syst., 2017.
