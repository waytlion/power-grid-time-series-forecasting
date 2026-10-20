# CLAUDE.md — Thesis_Repo

## Purpose
My thesis pipeline (100% mine, free to rewrite): OPF ground-truth generation, load forecasters, exact + 2-step
(AC/DC) arms, cross-arm aggregation, thesis plots/tables. The ML arms live in `~/gridfm-graphkit-dev`.
Thesis context + global rules: `~/CLAUDE.md`. Results/rationale: `docs/findings.md`.
Full pipeline, step by step: `readme.txt` (all commands assume CWD = Thesis_Repo root).

## Map
```
phase1_generation/       phase 1a: realistic load profiles → datakit AC-OPF ground truth (configs/phase1_config.yaml)
phase1_baseline/         phase 1b: forecasters. run_benchmark_temporal.py (XGB + Optuna, SARIMA, SNaive, TinyTGT);
                         src/{data_processing,evaluation,models,splitting}.py; experiments/ = ablation drivers + CSVs
exp1/generate_metrics/
  compare.py             exact + 2-step arms → results/<run>/<forecaster>/<case>_metrics.csv (AC + DC in one CSV, 'AC '/'DC ' prefix)
  aggregate_results.py   all arms from their native locations (compare.py tree + mlruns eval CSVs) → one tidy long CSV
  metrics.py, loaders.py, config.py   shared metric helpers / datakit loaders / paths
  expost_pf.py           ex-post AC-PF (pypowsybl, distributed slack): PF-solved dispatch = rebalanced dispatch + feasibility
  corrected_gap.py       rebalanced opt gap + Q3 shares from the expost_pf outputs (AC/DC 2-step, surrogate, E2E)
  dispatch_diag.py       diagnostics: Pg error by unit class (2-step vs E2E), Pg vs total demand, cross-bus Pd error corr
exp1/ml_surrogate/build_eval_set.py   forecast profiles → graphkit HeteroData eval sets (only Pd_forecast + Qd_derived in bus.x)
exp1/thesis_plots/       ALL thesis figures/tables (make_latex_tables.py, plot_*.py, run_residual_pipeline_h1.py,
                         extract_residuals_2step.py) → thesis_plots/figures/
exp1/results/            compare.py output; _archiv/ = superseded runs
scripts/                 sbatch + helpers: phase1a_*, phase1b.sbatch, phase1c*.sbatch, phase1c_transform_forecasts.py,
                         phase1c_surrogate_eval.sbatch, surrogate_smoke.sbatch, state_dump_eval.sbatch, expost_pf.sbatch,
                         backup_pack.sh + backup_manifest.txt (tiered tar of results/mlruns/configs, see ~/CLAUDE.md)
tests/                   phase1b_1c/ regression vs golden CSVs; scripts/phase1c_preflight.py, check_true_residual.py
```

## Run & validate
```bash
sbatch scripts/phase1a_run_datakit.sh                    # 1a: AC-OPF ground truth (barnard CPU)
sbatch scripts/phase1b.sbatch                            # 1b: forecasters
CASE=case500_goc HORIZON=6 sbatch scripts/phase1c.sbatch   # 1c: 2-step AC/DC solves on forecasts (env: MODELS,
                                                         # METHODS, DATA_IN_PARQUET, PRECOMPUTED_DIR, RESULTS_DIR)
CASE_DIR=case500_goc HORIZON=6 sbatch scripts/phase1c_step3.sbatch   # compare.py only (default case118 h1)
CASE=case14_ieee SEEDS="0 1 2" sbatch scripts/phase1c_surrogate_eval.sbatch   # 1c: surrogate eval (env: HORIZONS, FORECASTERS, SEEDS)
sbatch scripts/surrogate_smoke.sbatch                    # smoke: surrogate, seed 0, h=1, true forecaster
sbatch tests/phase1b_1c/phase1b_1c_test.sbatch           # regression: case14, 10% data, vs tests/phase1b_1c/golden/
python tests/scripts/phase1c_preflight.py                # before phase1c eval runs
python exp1/generate_metrics/aggregate_results.py ...    # tidy cross-arm CSV; then thesis_plots/make_latex_tables.py
sbatch scripts/backup_pack.sh core                       # backup tar after every final run (from login1.barnard)
```

## Rules & traps
- /home quota 50 GB (exceeding it locks sbatch until the nightly check): bulk/scratch output goes to `<data>`, e.g.
  precomputed profiles → `<data>/temp_precomputed_profiles/` (phase1c*, surrogate_smoke).
- **aggregate_results identity = paths + run names only** (no MLflow tags):
  - exact/2-step: `<results-root>/<run>/<forecaster>/<case>_metrics.csv`; ONE run generation per
    (case, horizon, forecaster). Archive superseded dirs under `exp1/results/_archiv/` (any `_`-prefixed dir is skipped).
  - ML arms: E2E `base_e2e_<case>_h<H>_s<S>`, surrogate `surrogate_<case>_<fc>_h<H>_s<S>` (`_ml_identity`, anchored
    both ends); other names are skipped. Only canonical CSVs (marker metric "Mean Optimality Gap") are ingested.
  - Output: one tidy row per (case, horizon, optimizer, forecaster, metric), aggregated over seed only.
  - Duplicate run names in one experiment get averaged as extra seeds → archive old runs first.
- Timing: "per sample" = per SCENARIO (one dispatch decision) in every arm. Details: `docs/findings.md`.
- `timing_benchmark.csv` has a schema guard: if you change timing columns, archive the old file first.
- Plots: vector PDF for line/bar plots (pdflatex + Overleaf); PNG only for raster figures.
- `--sarima-fit-hours` default is 26280 (3yr) but all reported results used 17520 (2yr).
- case500 is `case500_goc` on disk / as network key; phase1a needs SCALE ≈ 1.22 for it.
- Check sum(PMIN) vs scenario demand before blaming the solver when a scale sweep fails.
- Barnard jobs must be submitted by me from login1.barnard (no ssh key from capella).
