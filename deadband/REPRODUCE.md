# End-to-End Reproduction Pipeline

This document reproduces every quantitative result in the paper from scratch.
All commands run from the repository root with the venv active
(`source .venv/bin/activate`). `$PY` denotes `python`.

Wall-clock guidance: each 900-s dispatch interval simulates in roughly
1–3 min on a laptop. A full 96-interval day is several hours; Stage 1
(56 candidates × 5 windows) and the full-day top-6 validation are the long
steps. Use the archived inputs/outputs in `data/` and `results/` to skip or
spot-check any stage.

Common variables used below:

```bash
DYN=cases/IL200_dyn_db2_stable_tgov1ndb_restore0_17_pvd005_esd001_storage5_slack47_pmax730_sn730_tip100.xlsx
OPF=cases/IL200_opf2_aligned_to_dyn_swap0_17_pvd005_esd001_storage5_slack47_pmax730_sn730.xlsx
CURVE=cases/CurveInterp.csv
DISPATCHES=data/dispatches_day96          # archived; or regenerate in step 1
AGC_FLAGS="--agc-interval 4 --kp 0.1 --ki 0.002 \
  --agc-allocation-mode headroom_aware --agc-anti-windup-mode freeze_on_saturation \
  --disable-pvd-agc --disable-esd-agc"
ALPHA_FLAGS="--wind-pref-alpha 0.98 --solar-pref-alpha 0.98"
GOV_FLAGS="--init-mode first --governor-target-schedule ramp_limited_basepoint \
  --governor-basepoint-ramp-floor-frac-pmax-per-min 0.005 \
  --governor-basepoint-ramp-gap-factor 1.25"
```

These values reproduce the run manifests stored under `results/*/`
(`phase1_sweep_manifest.json`, `phase1_full_day_manifest.json`), which record
the exact configuration of the paper runs.

## Quick verification path (~15–25 min, recommended first)

Before committing to the multi-hour full pipeline, the following three
commands establish that the environment reproduces the paper's simulations
end to end:

```bash
bash env/setup_env.sh && source .venv/bin/activate
bash tools/smoke_check.sh        # patched models + OPF dispatch determinism
# one real hot-start dynamic window, uniform vs. best, nominal inertia:
python scripts/run_inertia_sensitivity_windows.py \
  --dispatch-dir data/dispatches_day96 \
  --results-dir results/_quick_verify \
  --dyn-case $DYN --stable-dyn-case $DYN --curve-file $CURVE \
  --inertia-multiplier 1.0 --window h11d2,h11d3 \
  $AGC_FLAGS $ALPHA_FLAGS $GOV_FLAGS --dispatch-interval 900
python - <<'EOF'
import pandas as pd, numpy as np
for case in ("uniform", "best"):
    a = pd.read_csv(f"results/_quick_verify/H100_{case}_h11d2_h11d3_trace.csv")
    b = pd.read_csv(f"results/inertia_multiscenario_screen_20260503/H100_{case}_h11d2_h11d3_trace.csv")
    d = float(np.abs(a["freq_dev_hz"].to_numpy() - b["freq_dev_hz"].to_numpy()).max())
    print(f"{case}: max |freq diff| vs archived reference = {d:.3e} Hz")
EOF
```

Expected: machine-precision agreement (`~1e-16` Hz or exactly 0) on the same
platform/pins; small numerical differences (sub-mHz) on a different OS/BLAS
are possible and do not change any paper-level statistic.

## 0. Environment + smoke check

```bash
bash env/setup_env.sh
source .venv/bin/activate
bash tools/smoke_check.sh
```

The smoke check asserts (i) the patched PVD1/ESD1 deadband extensions are
present, and (ii) regenerated hour-0 OPF dispatches match
`data/dispatches_day96` numerically (exact on the original environment).

## 1. Generate the 96 OPF dispatches (Paper §II-A)

```bash
$PY scripts/prepare_day_dispatches.py \
  --opf-case $OPF --curve-file $CURVE \
  --results-dir results/dispatches_day96_regen \
  --hour-start 0 --hours 24 --dispatches-per-hour 4 --dispatch-interval 900 \
  $ALPHA_FLAGS --workers 4
```

Outputs 96 `h{H}d{D}_dispatch.json`. These should equal
`data/dispatches_day96/` (deterministic convex OPF; verified bitwise on the
original machine). Either directory can serve as `$DISPATCHES` below.

## 2. Baseline full-day hot-start run (Paper §IV-A, Table I/II baseline, Fig. 3)

```bash
$PY scripts/run_day_dispatch_hotstart.py \
  --dispatch-dir $DISPATCHES \
  --results-dir results/phase1_baseline_full_day_tip100_alpha098_disable_pvd_agc_disable_esd_agc_kp0p1_ki0p002/wind036_pv036_esd036 \
  --checkpoints-dir results/phase1_baseline_full_day_tip100_alpha098_disable_pvd_agc_disable_esd_agc_kp0p1_ki0p002/checkpoints/base \
  --hour-start 0 --hours 24 --dispatches-per-hour 4 --dispatch-interval 900 \
  --dyn-case $DYN --stable-dyn-case $DYN --curve-file $CURVE \
  $AGC_FLAGS $ALPHA_FLAGS $GOV_FLAGS --apply-governor-targets \
  --wind-deadband-hz 0.036 --solar-deadband-hz 0.036 --esd-deadband-hz 0.036
```

`--apply-governor-targets` is required here: this runner defaults to *not*
applying OPF basepoints to the governors, whereas the paper baseline (and the
sweep/full-day runners in steps 4–5, which hard-code it) runs with governor
targets applied. Verified by checkpoint-replaying the archived baseline:
hour 12 replayed from `end_h11d3` is byte-identical to the archived
`h12d*_frequency.csv` only with this flag set.

Check against the reference: `frequency_distribution_stats.csv` must give
mean|Δf| = 0.02409 Hz, share>36 mHz = 23.10 %, share>50 mHz = 2.87 %,
max|Δf| = 0.09334 Hz (Table II baseline row).

> Checkpoint-replay note: checkpoints embed a parameter signature that
> includes the *resolved absolute paths* of the case/curve files and the
> governor-target setting. Replaying a checkpoint produced elsewhere
> therefore requires matching paths (or the `--allow-signature-mismatch`
> escape hatch of `scripts/run_dispatch_hotstart.py`). Checkpoints you
> generate locally in step 2 replay without any of this.

## 3. Hot-start equivalence validation (Paper §II-F, "machine precision")

After step 2 produced checkpoints, pick any interior boundary
(e.g. h5d1→h5d2) and compare a checkpoint-resumed run against the
continuously-run reference:

```bash
$PY scripts/compare_dispatch_pair_midpoint_continuous.py \
  --checkpoint-in  <checkpoints-dir>/end_h5d0 \
  --first-dispatch-json  $DISPATCHES/h5d1_dispatch.json \
  --second-dispatch-json $DISPATCHES/h5d2_dispatch.json \
  --third-dispatch-json  $DISPATCHES/h5d3_dispatch.json \
  --first-hotstart-csv  <baseline-results-dir>/h5d1_frequency.csv \
  --second-hotstart-csv <baseline-results-dir>/h5d2_frequency.csv \
  --curve-file $CURVE --results-dir results/_equivalence_check \
  --kp 0.1 --ki 0.002 --agc-interval 4 --dispatch-interval 900
```

Expected: `diff_max_abs_hz` at the 1e-16 Hz level.

## 4. Stage-1 representative-window screening (Paper §III-F, 56→48→top-6)

```bash
$PY scripts/sweep_deadband_phase1_windows.py \
  --dispatch-dir $DISPATCHES \
  --results-dir results/phase1_deadband_tip100_alpha098_disable_pvd_agc_disable_esd_agc_kp0p1_ki0p002 \
  --dyn-case $DYN --stable-dyn-case $DYN --curve-file $CURVE \
  --windows h2d3_h3d0 h7d3_h8d2 h11d2_h11d3 h15d1_h15d2 h20d3_h21d0 \
  --wind-deadband-list 0.036 0.045 0.055 0.065 \
  --solar-deadband-list 0.025 0.030 0.036 0.042 \
  --esd-deadband-list 0.015 0.020 0.025 0.030 \
  --max-abs-hz-threshold 0.10 --share-abs-gt-0p05-threshold 0.05 \
  $AGC_FLAGS $ALPHA_FLAGS $GOV_FLAGS --dispatch-interval 900
```

Only ordered combinations (ESD ≤ PV ≤ wind) are simulated: 56 of 64.
Expected: 48 eligible after the safety screen; `phase1_top_candidates.csv`
lists the top-6 forwarded to Stage 2. The shipped manifest
`results/phase1_deadband_*/phase1_sweep_manifest.json` records this exact
configuration.

## 5. Full-day top-6 validation + lexicographic ranking (Paper §IV-E, Table II, Figs. 5–8)

```bash
$PY scripts/run_deadband_phase1_day.py \
  --candidate-csv results/phase1_deadband_tip100_alpha098_disable_pvd_agc_disable_esd_agc_kp0p1_ki0p002/phase1_top_candidates.csv \
  --results-root results/phase1_full_day_tip100_alpha098_disable_pvd_agc_disable_esd_agc_kp0p1_ki0p002 \
  --baseline-results-dir results/phase1_baseline_full_day_tip100_alpha098_disable_pvd_agc_disable_esd_agc_kp0p1_ki0p002/wind036_pv036_esd036 \
  --dispatch-dir $DISPATCHES --dyn-case $DYN --stable-dyn-case $DYN --curve-file $CURVE \
  --top-k 6 $AGC_FLAGS $ALPHA_FLAGS $GOV_FLAGS
```

Expected (`phase1_full_day_summary.csv` / `_ranked.csv`, cf. Table II): best
candidate 36/25/15 mHz with mean|Δf| = 0.02113 Hz, share>36 mHz = 11.88 %,
share>50 mHz = 1.31 %, max|Δf| = 0.08949 Hz, EM36 = 12.32 %.

## 6. ESD-only ablation (Paper §IV-E "ESD-Only Ablation", Fig. 10-file/Fig. 7-paper)

```bash
$PY scripts/sweep_deadband_phase1_windows.py \
  --dispatch-dir $DISPATCHES \
  --results-dir results/ablation_esd_only_stage1 \
  --dyn-case $DYN --stable-dyn-case $DYN --curve-file $CURVE \
  --windows h2d3_h3d0 h7d3_h8d2 h11d2_h11d3 h15d1_h15d2 h20d3_h21d0 \
  --wind-deadband-list 0.036 --solar-deadband-list 0.036 \
  --esd-deadband-list 0.010 0.015 0.020 0.025 0.030 0.036 \
  --max-abs-hz-threshold 0.10 --share-abs-gt-0p05-threshold 0.05 \
  $AGC_FLAGS $ALPHA_FLAGS $GOV_FLAGS --dispatch-interval 900
```

Reference: `results/ablation_esd_only_stage1_20260503/phase1_combo_summary.csv`
(36→15 mHz cuts share>36 mHz 35.47→23.91 %, share>50 mHz 4.49→1.87 %).

## 7. Load-step stress check (Paper Table III)

```bash
for STEP in 0.05 -0.05; do
$PY scripts/run_load_step_event_check.py \
  --dispatch-dir $DISPATCHES --results-dir results/load_step_event_check \
  --dyn-case $DYN --stable-dyn-case $DYN --curve-file $CURVE \
  --window h11d2_h11d3 --event-time-s 120 --load-step-frac $STEP \
  $AGC_FLAGS $ALPHA_FLAGS $GOV_FLAGS --dispatch-interval 900
done
```

The runner evaluates both the uniform 36/36/36 baseline and the selected
36/25/15 candidate. Reference: `results/load_step_event_check_20260503/`.

## 8. Bounded-inertia robustness screen (Paper §IV-F, Fig. 4 + Fig. 10-paper)

```bash
$PY scripts/run_inertia_sensitivity_windows.py \
  --dispatch-dir $DISPATCHES --results-dir results/inertia_multiscenario_screen \
  --dyn-case $DYN --stable-dyn-case $DYN --curve-file $CURVE \
  $AGC_FLAGS $ALPHA_FLAGS $GOV_FLAGS --dispatch-interval 900
```

Runs uniform-vs-best on h20d3_h21d0 / h11d2_h11d3 / h7d3_h8d2 at synchronous
inertia scalings ×{0.5, 1.0, 1.25} (script defaults; see `--help`). The
nominal-inertia traces double as the mechanism-figure data: the archived
`H100_best_h11d2_h11d3_trace.csv` is bit-identical to the dedicated run in
`results/fig4_mechanism_best_h11d2_h11d3/`. Aggregate expectations (paper):
mean|Δf| reduction 14.1/14.7/14.2 %, tail-share reduction 69.0–77.9 %.

## 9. Figures (paper Figs. 3–11)

All figure scripts read the `results/` paths above by default and write
PDF+PNG next to themselves:

```bash
cd paper/latex/figures
python plot_fig3_baseline_dist.py
python plot_fig4_mechanism.py
python plot_fig5_pareto.py
python plot_fig6_dist_compare.py
python plot_fig7_panorama.py
python plot_fig8_cost_tradeoff.py
python plot_fig10_esd_ablation.py
python plot_fig11_inertia_screen.py
cd ../../..
python scripts/plot_minimal_model.py    # minimal-model figure (Langevin density)
```

(`fig1`/`fig2` of the paper are TikZ diagrams in the manuscript itself.)

## Numbers-to-files map

| Paper item | Reference file |
|---|---|
| Table II (all rows) | `results/phase1_full_day_*/phase1_full_day_summary.csv` |
| Baseline stats (§IV-A) | `results/phase1_baseline_full_day_*/wind036_pv036_esd036/frequency_distribution_stats.csv` |
| Stage-1 56→48→6 | `results/phase1_deadband_*/phase1_sweep_manifest.json` + step-4 outputs |
| ESD ablation numbers | `results/ablation_esd_only_stage1_20260503/phase1_combo_summary.csv` |
| Table III load steps | `results/load_step_event_check_20260503/load_step_event_summary.csv` |
| Inertia screen (9 cells) | `results/inertia_multiscenario_screen_20260503/H*_trace.csv` |
| Mechanism stats (244/257, 39.4 %, 10.6 %, 88.7 %) | `H100_{uniform,best}_h11d2_h11d3_trace.csv` engaged-count columns |
