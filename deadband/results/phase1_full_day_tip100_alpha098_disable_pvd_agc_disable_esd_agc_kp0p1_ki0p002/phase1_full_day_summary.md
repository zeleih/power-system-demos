# Phase-1 Deadband Full-Day Validation

## Baseline

- baseline_dir: `/Applications/openandes/demo/demo/deadband/results/day96_hotstart_casefile_tgov1ndb_restore0_17_tip100_alpha098_disable_pvd_agc_disable_esd_agc_kp0p1_ki0p002_aw_headroom_applygov`
- mean_abs_hz: 0.02409
- share(|f| > 0.036): 23.10%
- share(|f| > 0.05): 2.87%
- max_abs_hz: 0.09334
- edge_mass_36: 16.90%
- edge_asymmetry_36: 0.84%

## Recommendation

- best candidate: `wind036_pv025_esd015`
- wind/pv/esd deadband: 0.036 / 0.025 / 0.015 Hz
- mean_abs_hz: 0.02113
- share(|f| > 0.036): 11.88%
- share(|f| > 0.05): 1.31%
- max_abs_hz: 0.08949
- edge_mass_36: 12.32%
- edge_asymmetry_36: 0.75%
- result_dir: `/Applications/openandes/demo/demo/deadband/results/phase1_full_day_tip100_alpha098_disable_pvd_agc_disable_esd_agc_kp0p1_ki0p002/wind036_pv025_esd015`
- distribution_plot: `/Applications/openandes/demo/demo/deadband/results/phase1_full_day_tip100_alpha098_disable_pvd_agc_disable_esd_agc_kp0p1_ki0p002/wind036_pv025_esd015/frequency_distribution.png`
- curves_plot: `/Applications/openandes/demo/demo/deadband/results/phase1_full_day_tip100_alpha098_disable_pvd_agc_disable_esd_agc_kp0p1_ki0p002/wind036_pv025_esd015/frequency_curves_all_96.png`
