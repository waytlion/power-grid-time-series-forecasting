
# FULL PIPELINE WORKFLOW
#
# This file documents the end-to-end pipeline.
# All commands assume CWD = Thesis_Repo root.
# gridfm-datakit must be installed: pip install -e ../gridfm-datakit-fork


# =============================================================================
### Configuring the Pipeline (Changing Horizons or Networks)
# =============================================================================
To run a different network (e.g., Case 14) or forecast horizon (e.g., 6 or 24),  change variables in the top sections of the sbatch files:

**1. Update `phase1_baseline/run_benchmark_temporal.sbatch` (Phase 1b):**
- Update `--data-path` to the raw data input.
- Update `--output-path` naming (e.g., `case14_ieee_horizon6.parquet`).
- Add `--forecast-horizon 6` to the python arguments.

**2. Update `scripts/phase1c_eval.sbatch` (Phase 1c):**
Change the variables at the very top of the script:
```bash
CASE="case14"               # The datakit case name
CASE_DIR="case14_ieee"      # Data export directory naming
HORIZON="6"                 # Horizon size (1, 6, 24)
DATAKIT_BASE_YAML="exp1/configs/case14_generate_opf_for_forecast_cluster.yaml"
```

# =============================================================================
# PHASE 1a: Generate OPF Ground Truth Data
# =============================================================================

1. Preprocess realistic load profiles into datakit format
        -> INPUT: realistic load profiles .parquet (from Marcus)
        -> EXECUTE:
                - Cluster: python phase1_generation/preprocessing/generate_precomputed_profile.py
                - Local:   phase1_generation/preprocessing/generate_precomputed_profile.ipynb
        -> OUTPUT: phase1_generation/preprocessing/*_precomputed_load_profiles.csv

2. Run datakit to solve AC-OPF (on cluster — too computationally intensive for local)
        -> CONFIG: phase1_generation/configs/phase1_config.yaml
        -> EXECUTE: sbatch scripts/cluster_leipzig.sh
        -> OUTPUT: data/data_out/

# =============================================================================
# PHASE 1b: Temporal Baseline Forecasting
# =============================================================================

3. Run baseline models (SNaive, SARIMA, XGBoost, TGT) on Leipzig
        -> SET PARAMS: in <run_benchmark_temporal.sbatch>
                --data-path
                --output-path
                --forecast_horizon
        -> EXECUTE: cd phase1_baseline && sbatch run_benchmark_temporal.sbatch

# =============================================================================
# PHASE 1c: Two-Step OPF Evaluation
# =============================================================================

4. Evaluate baseline forecasts via full Two-Step OPF approach
        -> CONFIG: scripts/phase1c_eval.sbatch
        -> EXECUTE: sbatch scripts/phase1c_eval.sbatch
        -> THIS SCRIPT SEQUENTIALLY:
             a) Transforms baseline predictions into datakit inputs via scripts/transform_forecasts.py
             b) Runs datakit loop over each model sequentially via scripts/run_datakit_batch.py
             c) Computes & compares OPF metrics via exp1/generate_metrics/compare.py
        -> OUTPUT: exp1/results/

# ALL PHASES AUTOMATED (One-Click)
        -> EXECUTE: bash scripts/submit_pipeline.sh
        -> This uses SLURM job dependencies to chain Phase 1a -> 1b -> 1c.
- If Forecast Horizon > 1 -> The number of scenarios/timesteps predicted is less.
        - Example: if T=100 and test indices are 85..99:
                -> h=1 starts: 85..99 -> 15 starts
                -> h=6 starts: 85..94 -> 10 starts
- Qd is not forecasted by baseline models -> derived by applying scaling factor to Pd
- Datakit on cluster cannot be run in parallel -> Julia makes probs