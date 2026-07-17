
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
! Train one model per (horizon, seed); metrics come from training's test step. !

7. Train the E2E ST-GNN  (gridfm-graphkit, GPU)
        -> CONFIG: examples/config/.../HGNS_ST_ForecastOPF_h<H>.yaml
        -> EXECUTE: gridfm train ...   (manually per horizon x seed, same experiment)
        -> OUTPUT: graphkit metric CSVs in MLflow (per-horizon gap in <net>_forecast.csv)

# -----------------------------------------------------------------------------
# METRICS: all three arms report a comparable set (gap, residuals, RMSE,
#   feasibility, inference time). Exact arm via compare.py; the two ML arms via
#   graphkit's own test metrics (validated identical to compare.py).
# -----------------------------------------------------------------------------


===========================================================================
# Results table i want in my thesis
==========================================================================
Questions:


-metrics
-> opt. gap
-> Constraint Violations:
a) active and reactive physics residualds 
b) Description: For the following inequality constraints i want to compute: total violation rate (fraction of buses over limit/all buses), scenario violation rate (fraction of scenarios with a least one bus over limit/all scenarios), mean and max violation. Mean violation is over ALL samples/nodes (incl. non-violating), one-sided and two-sided: mean(max(0, x − upper) + max(0, lower − x)) for two-sided limits (Vm, Pg, Qg, angle); thermal is upper-only: mean(max(0, |S| − rate_a)). Compute in the constraint's physical unit (pu for Vm, MVA for thermal, MW/MVAr for generators). NOTE: mean-over-violating-only is derivable as mean_all / viol_rate, so not stored separately.
- Vm limit violations
- For generators: Pg/Qg limit violations 
- thermal limit violations: The two ends are separate constraints — |S_from| ≤ rate_a and |S_to| ≤ rate_a — and they differ by the line's losses. If either exceeds the rating, the line is overloaded and the state is infeasible; Use max(violation_from, violation_to) per line for computing the violation statistics
- voltage angle difference limit violations
-> training and inference time per sample

REVIEW:
- confirm computed metrics (predicted vs ground truth) are in the same unit
- drop ac_rmse_pf/pt_branch (redundant for thermal limit)

Notes:
- following metrics are is in the ieee_metrics.csv and prob not important for thesis for now: -> MAE or RMSE for all "predicted" variables, which i believe are: Pd, Qd, Pg, Vm, Va (MAE is over all samples/nodes, signed-symmetric, and measured against ground truth. Being under a limit contributes to MAE.)

TABLES: 
- Open for dicsussion: which names to use for metrics (could use from compare.py or from E2E arm, not sure which ones are better)
- 
===========================================================================
# NOTES
==========================================================================
- If Forecast Horizon > 1 -> The number of scenarios/timesteps predicted is less.
        - Example: if T=100 and test indices are 85..99:
                -> h=1 starts: 85..99 -> 15 starts
                -> h=6 starts: 85..94 -> 10 starts
- Qd is not forecasted by baseline models -> derived by applying scaling factor to Pd
- I believe Datakit on cluster cannot be run in parallel -> Julia makes probs.
--> mby this is also only a problem, when they operate on same compute node, meaning if every datakit job has own compute node, they might not have problems parallelizing