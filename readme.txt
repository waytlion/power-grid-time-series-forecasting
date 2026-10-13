
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
                [--xgb_input_all_opf_var]  (OPTIONAL toggle: if set, XGB uses ALL historical OPF vars
                                            Pd,Qd,Qg,Vm,Va as input lags instead of Pd-only.
                                            Target stays Pd either way. Writes to the SAME output paths
                                            as baseline)
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
                - !!ATTENTION: This currently skips training models, when the model already exists 
        -> OUTPUT: checkpoints in MLflow, run names  surrogate_<case>_s<seed>

5b. Evaluate surrogate on forecasts (all horizons) -> metrics
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
! Train one model per (case, horizon, seed)! 

6. Train the E2E ST-GNN 
        -> CONFIG: sbatch train.sh 
                - CASE, HORIZON, SEED, CONFIG
        -> EXECUTE: sbatch train.sh 
        -> OUTPUT: MLflow artifacts 

# -----------------------------------------------------------------------------
# METRICS: all three arms report a comparable set (gap, residuals, RMSE,
#   feasibility, inference time). Exact arm via compare.py; the two ML arms via
#   graphkit's own test metrics (identical to compare.py).
# -----------------------------------------------------------------------------

7. Combine all arms into one combined CSV (aggregated over seeds for ML-surrogate/E2E)
        -> EXECUTE:
           python exp1/generate_metrics/aggregate_results.py --results-root exp1/results \
             --mlruns-dir /data/horse/ws/tibo990i-thesis_data/mlruns/375695290152400764 \
             /data/horse/ws/tibo990i-thesis_data/mlruns/108250444742230565 \
             --timing-csv phase1_baseline/output/2019-2021/timing_benchmark.csv \
             --datakit-out-dir /data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1c/data_out \
             --out exp1/results/combined/case118_combined.csv
        -> RULES (if violated the script still runs but results are silently wrong):
             - exp1/results must hold one run generation per (case,horizon,forecaster)
             - MLflow eval run names MUST match exactly:
               surrogate_{case}_{forecaster}_h{H}_s{S}  /  base_e2e_{case}_h{H}_s{S}
             - a retrain under the same identity (e.g. base_e2e_case500_h1_s0 -> base_e2e_case500_goc_h1_s0)
               is averaged in as an extra seed: archive the superseded run dirs first

7a. Ex-post AC-PF (pypowsybl OpenLoadFlow, distributed slack ∝ Pmax): rebalanced dispatch P̃ + feasibility (barnard CPU)
        arms: label (sanity), exact (AC-OPF), dc (DC-OPF dispatch + case-file generator voltage setpoints), surrogate, e2e
        -> EXECUTE (from login1.barnard):
           for c in case14_ieee case118_ieee case500_goc; do CASE=$c sbatch scripts/expost_pf.sbatch; done
           (one arm only: CASE=case500_goc ARMS=e2e sbatch scripts/expost_pf.sbatch)
        -> OUTPUT: /data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1c/expost_pf/<case>/h<H>/<arm>[_<fc>][_s<seed>].parquet
           + <case>/run_config_<job>.json (load-flow settings, OLF defaults)
           (ML arms: newest run with a state dump per run name; case118 h1 label + AC arm: ~4 min on 2 CPUs)

7c. Rebalanced opt gap from the 7a outputs -> exp1/results/combined/corrected_gap.csv
        -> EXECUTE: python exp1/generate_metrics/corrected_gap.py [--cases case14_ieee ...]
           (γ_reb = (C(P̃) - C(P*)) / C(P*), R = (C(P̃) - C(P̂)) / C(P*), both as % of C(P*); also thermal
            overload share, insufficient headroom, diverged / no_dispatch counts;
            fails if a recomputed raw gap != logged gap)

7b. create final plots for thesis:
        - EXECUTE: python exp1/thesis_plots/make_latex_tables.py
                

# =============================================================================
# PHASE 2: Per-bus power-balance residuals + error-distribution figures (results ch.)
# =============================================================================
! EXECUTED ONCE FOR EVERY (NETWORK, power-balance-residual-space) ! (HORIZON IS FIXED AT 1)

8. Surrogate / E2E residuals: byproduct of the arm's eval (step 6) / train test-pass (step 7).
        -> OUTPUT (in the run's artifacts/test/):
           <dataset>_perbus_residuals_{realized,forecast}.parquet

9. Extract 2-step AC/DC residuals + build the results figures + latex tables
        -> EXECUTE:
           python exp1/thesis_plots/run_residual_pipeline_h1.py \
             --opf-base-dir /data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1c/data_out/case118_ieee_horizon_1 \
             --forecaster xgb \
             --dataset case118_ieee \
             --surrogate-realized /data/horse/ws/tibo990i-thesis_data/mlruns/108250444742230565/cc051bb63d474f79abab6372ffde3e2b/artifacts/test/case118_ieee_perbus_residuals_realized.parquet \
             --e2e-realized /data/horse/ws/tibo990i-thesis_data/mlruns/375695290152400764/f2f12a35c5804902bb625d5afbc97f10/artifacts/test/case118_ieee_perbus_residuals_realized.parquet \
             --out-dir exp1/thesis_plots/figures \
             --format pdf    
        -> OUTPUT (exp1/thesis_plots/figures/):
             residuals_xgb_{ac,dc}_{realized,forecast}.parquet  2-step per-bus residuals
             residual_hist_h1.png    bus-level signed hist (MW) + violin, 1 MW heuristic line
             net_imbalance_h1.png    scenario-level signed net (sum over buses) per arm
             error_metrics.tex       per arm x P/Q: bias, MAE, |res| quantiles, share>1 MW
             net_imbalance.tex per arm: net bias, quantiles, mean|net|
        -> DESIGN: absolute MW (no peak-load normalization, no eps-floor, all buses);
                   1 MW line = practitioner negligibility heuristic (pers. comm.), not a rule.

===========================================================================
# SMOKE TESTS
==========================================================================
RUN ML SURROGATE:
       - sbatch Thesis_Repo/scripts/surrogate_smoke.sbatch   # seed 0, h=1, true forecaster


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