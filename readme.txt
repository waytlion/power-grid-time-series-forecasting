
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
                - Cluster: /data/horse/ws/tibo990i-thesis_data/phase_1a/data_in/df_load_bus_2019-2021.parquet
                - local: D:\Data\studium\Master\MA_Code\data\phase_1a\data_in
        -> EXECUTE:
                - Cluster: python phase1_generation/preprocessing/generate_precomputed_profile.py
                - Local:   phase1_generation/preprocessing/generate_precomputed_profile.ipynb
        -> OUTPUT: phase1_generation/preprocessing/f"{CASE_NAME.strip()}_3yr_precomputed_load_profiles.csv"

2. Run datakit to solve AC-OPF (on cluster — too computationally intensive for local)
        -> CONFIG: phase1_generation/configs/phase1_config.yaml
        -> EXECUTE: sbatch scripts/phase1a_run_datakit.sh
        -> OUTPUT: 
                - Cluster: /data/horse/ws/tibo990i-thesis_data/data_out/3yr_2019-2021/phase_1a/data_out/case14_ieee

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