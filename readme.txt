
# FULL PIPELINE WORKFLOW
#
# This file documents the end-to-end pipeline.
# All commands assume CWD = Thesis_Repo root.
# gridfm-datakit must be installed: pip install -e ../gridfm-datakit-fork

# =============================================================================
# PHASE 1a : Generate OPF Ground Truth Data 
# =============================================================================

! Execute Once For Each Network Case!
! ADJUST PATHS MANUALLY!

1. Preprocess realistic load profiles into datakit format
        -> INPUT: 
                - realistic load profiles .parquet (from Marcus)
                - Cluster: "/data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1a/data_in/df_load_bus_2019-2021.parquet"
                - local: D:\Data\studium\Master\MA_Code\data\phase_1a\data_in
        -> EXECUTE:
                - Cluster: python phase1_generation/preprocessing/generate_precomputed_profile.py
                - Local:   phase1_generation/preprocessing/generate_precomputed_profile.ipynb
        -> OUTPUT: phase1_generation/preprocessing/f"{CASE_NAME.strip()}_3yr_precomputed_load_profiles.csv"

2. Run datakit to solve AC-OPF (on cluster — too computationally intensive for local)
        -> SET PARAMS IN <phase1_generation/configs/phase1_config.yaml>
                -- scenario_file, data_dir, scenarios
                -- num_processes, large_chunk_size
        -> EXECUTE: sbatch scripts/phase1a_run_datakit.sh
        -> OUTPUT: 
                - Cluster: /data/horse/ws/tibo990i-thesis_data/2002-2023/phase_1a/data_out/

# =============================================================================
# PHASE 1b: Temporal Baseline Forecasting
# =============================================================================
! Execute Once For Each Network Case, Forecast Horizon !
! ADJUST PATHS MANUALLY!

3. Run baseline models (SNaive, SARIMA, XGBoost, TGT) on Leipzig
        -> SET PARAMS: in <phase1b.sbatch>
                --data-path
                --output-path
                --forecast_horizon
        -> EXECUTE: cd phase1_baseline -> sbatch run_benchmark_temporal.sbatch

# ==========================================================================
# PHASE 1c: Two-Step OPF Evaluation
===========================================================================

! Execute Once For Each Network Case, Forecast Horizon !
! ADJUST PATHS MANUALLY!

4. Evaluate baseline forecasts via full Two-Step OPF approach
        -> CONFIG: scripts/phase1c_eval.sbatch
                - CASE
                - HORIZON
        -> EXECUTE: sbatch scripts/phase1c.sbatch
        -> THIS SCRIPT SEQUENTIALLY:
             a) Transforms baseline predictions into datakit inputs via scripts/phase1c_transform_forecasts.py
             b) Runs datakit loop over each model sequentially via scripts/phase1c_run_datakit_batch.py
             c) Computes & compares OPF metrics via exp1/generate_metrics/compare.py
        -> OUTPUT: exp1/results/

# =============================================================================
# PHASE 1c (ML SURROGATE arm): replace the exact OPF solve with a trained GNN
# =============================================================================
! Train ONCE per network (horizon-independent). Eval per network (all horizons). !
! ADJUST PATHS MANUALLY!

5. Train the surrogate GNN  (gridfm-graphkit, GPU/capella)
        -> learns true-load -> OPF dispatch
        -> EXECUTE: sbatch train_surrogate.sh        (3 seeds via job array)
        -> OUTPUT: checkpoints in MLflow, run names  surrogate_<case>_s<seed>

6. Evaluate surrogate on forecasts (all horizons) -> metrics
        -> CONFIG: scripts/phase1c_surrogate_eval.sbatch
                   (CASE, NETWORK, HORIZONS, FORECASTERS, SEEDS, paths)
        -> EXECUTE: sbatch scripts/phase1c_surrogate_eval.sbatch
        -> THIS SCRIPT:
             a) preprocess: phase1c_transform_forecasts.py
                          + exp1/ml_surrogate/build_eval_set.py   (forecasts -> graphkit eval sets)
             b) gridfm evaluate (OptimalPowerFlowTwoStep) per (seed, horizon, forecaster)
        -> OUTPUT: graphkit metric CSVs in MLflow,
                   run names surrogate_<case>_<forecaster>_h<H>_s<seed>

# =============================================================================
# PHASE 1c (E2E arm): one ST-GNN, load history -> dispatch (forecast+optimize fused)
# =============================================================================
! Train one model per (horizon, seed); metrics come from training's test step. 

7. Train the E2E ST-GNN  (gridfm-graphkit, GPU)
        -> CONFIG: examples/config/.../HGNS_ST_ForecastOPF_h<H>.yaml
        -> EXECUTE: gridfm train ...   (manually per horizon x seed, same experiment)
        -> OUTPUT: graphkit metric CSVs in MLflow (per-horizon gap in <net>_forecast.csv)

# -----------------------------------------------------------------------------
# METRICS: all three arms report a comparable set (gap, residuals, RMSE,
#   feasibility, inference time). Exact arm via compare.py; the two ML arms via
#   graphkit's own test metrics (validated identical to compare.py).
# -----------------------------------------------------------------------------

7b. Combine all arms into one combined CSV (aggregated over seeds for ML-surrogate/E2E)
        -> EXECUTE:
           python exp1/generate_metrics/aggregate_results.py --results-root exp1/results \
             --mlruns-dir ~/gridfm-graphkit-dev/mlruns \
             --timing-csv phase1_baseline/output/2019-2021/timing_benchmark.csv \
             --datakit-out-dir /data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1c/data_out \
             --out exp1/results/combined/case118_combined.csv
        -> RULES (if violated the script still runs but results are silently wrong):
             - exp1/results must hold one run generation per (case,horizon,forecaster)
             - MLflow eval run names MUST match exactly (arm = first token, s<S> = last):
               surrogate_{case}_{forecaster}_h{H}_s{S}  /  e2e_{case}_h{H}_s{S}
        -> OUTPUT: <combined>.csv, one row per (case,horizon,optimizer,forecaster,metric).

7c. create final plots for thesis:
        - EXECUTE: python exp1/thesis_plots/make_latex_tables.py
                

# =============================================================================
# PHASE 2: Per-bus power-balance residuals + error-distribution figures (results ch.)
# =============================================================================
! EXECUTED ONCE FOR EVERY (NETWORK, power-balance-residual-space) ! (HORIZON IS FIXED AT 1)

8. Extract 2-step AC/DC residuals (offline, CPU, ~1 min)
        -> EXECUTE:
           python exp1/thesis_plots/extract_residuals_2step.py \
             --opf-base-dir <phase1c data_out>/case118_ieee_horizon_1_3yr_HPO_xgb \
             --forecaster xgb --load realized --out-dir <dir>
           (repeat with --load forecast)
        -> OUTPUT: <dir>/residuals_xgb_{ac,dc}_{realized,forecast}.parquet
                   [scenario, bus, res_p_mw, res_q_mvar]

9. Surrogate / E2E residuals: byproduct of the arm's eval (step 6) / train test-pass (step 7).
        -> OUTPUT (in the run's artifacts/test/):
           <dataset>_perbus_residuals_{realized,forecast}.parquet

10. Build the results figures + stats CSVs
        -> EXECUTE:
           python exp1/thesis_plots/plot_residual_hist_h1.py \
             --arm "AC-2step"       <residuals_xgb_ac_realized.parquet> \
             --arm "DC-2step"       <residuals_xgb_dc_realized.parquet> \
             --arm "Surrogate-2step" <run>/artifacts/test/..._perbus_residuals_realized.parquet \
             --arm "E2E"            <run>/artifacts/test/..._perbus_residuals_realized.parquet \
             --out-dir exp1/thesis_plots/figures
        -> OUTPUT (exp1/thesis_plots/figures/):
             residual_hist_h1.png    bus-level signed hist (MW) + violin, 1 MW heuristic line
             net_imbalance_h1.png    scenario-level signed net (sum over buses) per arm
             residual_stats.csv      per arm x P/Q: bias, MAE, |res| quantiles, share>1 MW
             net_imbalance_stats.csv per arm: net bias, quantiles, mean|net|
        -> DESIGN: absolute MW (no peak-load normalization, no eps-floor, all buses);
                   1 MW line = practitioner negligibility heuristic (pers. comm.), not a rule.


===========================================================================
# NOTES
==========================================================================
- phase1a (generate_precomputed_profile.py): Needs Scale 1.22 for case500!
- If Forecast Horizon > 1 -> The number of scenarios/timesteps predicted is less.
        - Example: if T=100 and test indices are 85..99:
                -> h=1 starts: 85..99 -> 15 starts
                -> h=6 starts: 85..94 -> 10 starts
- Qd is not forecasted by baseline models -> derived by applying scaling factor to Pd
- I believe Datakit on cluster cannot be run in parallel -> Julia makes probs.
--> mby this is also only a problem, when they operate on same compute node, meaning if every datakit job has own compute node, they might not have problems parallelizing
- timing_benchmark.csv: run_benchmark_temporal.py  raises err if columns don't match the file's existing header -> archive the old file first if add/change timing cols