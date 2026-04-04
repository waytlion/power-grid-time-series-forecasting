#!/bin/bash
# test_pipeline_case14.sh
# 
# Master submission script to test the entire thesis workflow on Case14.
# This chains Phase 1A, Phase 1B, and Phase 1C for Case14.
#
# Usage: 
#   cd Thesis_Repo
#   bash scripts/test_pipeline_case14.sh

set -euo pipefail

echo "Submitting Full Thesis Pipeline - TEST (Case14)"
echo "----------------------------------------------"

# Note: Phase 1a for case14 uses a specific config
# We write a temporary sbatch script to run datakit for phase1a case14
cat << 'EOF' > scripts/temp_phase1a_case14.sbatch
#!/bin/bash
#SBATCH --job-name=datakit_case14_gen
#SBATCH --partition=paula
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/case14_gen_%j.out

source "$SLURM_SUBMIT_DIR/../thesis_env/bin/activate"
# Assuming there is a phase1_config for case14. 
# We'll just run phase1b and phase1c directly assuming phase1a ground truth exists for case14.
EOF


cat << 'EOF' > scripts/temp_phase1b_case14.sbatch
#!/bin/bash
#SBATCH --job-name=bench_case14
#SBATCH --partition=paula    
#SBATCH --cpus-per-task=32       
#SBATCH --mem=64G               
#SBATCH --time=01:00:00
#SBATCH --output=logs/case14_bench_%j.out

set -euo pipefail
module --force purge
source "$SLURM_SUBMIT_DIR/../thesis_env/bin/activate"

cd phase1_baseline
python run_benchmark_temporal.py \
  --data-path ../data/data_out/3yr_2019-2021/case14_ieee/raw \
  --output-path case14_ieee_horizon1.parquet \
  --seed 42 \
  --skip-tgt \
  --xgb-device cpu
EOF

cat << 'EOF' > scripts/temp_phase1c_case14.sbatch
#!/bin/bash
#SBATCH --job-name=eval_case14
#SBATCH --partition=paula
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/case14_eval_%j.out

set -euo pipefail
module --force purge
source "$SLURM_SUBMIT_DIR/../thesis_env/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR"

CASE="case14"
CASE_DIR="case14_ieee"
HORIZON="1"

DATA_IN_PARQUET="phase1_baseline/${CASE_DIR}_horizon${HORIZON}.parquet"
PRECOMPUTED_DIR="exp1/data/precomputed_profiles/${CASE_DIR}_horizon${HORIZON}_3yr"
DATAKIT_BASE_YAML="exp1/configs/case14_generate_opf_for_forecast_cluster.yaml"
OPF_OUT_DIR="data/data_out/3yr_2019-2021/baseline_preds/${CASE_DIR}_horizon_${HORIZON}_3yr"
GROUND_TRUTH_DIR="data/data_out/3yr_2019-2021/${CASE_DIR}/raw"
FINAL_RESULTS_DIR="exp1/results/${CASE_DIR}_horizon${HORIZON}_3yr"

echo "Step 1: Transform..."
python scripts/transform_forecasts.py --case "$CASE" --input-parquet "$DATA_IN_PARQUET" --out-dir "$PRECOMPUTED_DIR"

echo "Step 2: DataKit Batch..."
python scripts/run_datakit_batch.py --base-yaml "$DATAKIT_BASE_YAML" --data-in-dir "$PRECOMPUTED_DIR" --out-dir "$OPF_OUT_DIR" --models xgb sarima snaive

echo "Step 3: Compare..."
python exp1/generate_metrics/compare.py --predicted-opf-base-dir "$OPF_OUT_DIR" --ground-truth-dir "$GROUND_TRUTH_DIR" --output-dir "$FINAL_RESULTS_DIR" --dataset "$CASE_DIR" --forecasts-parquet "$DATA_IN_PARQUET"
EOF

# Submitting
echo "Submitting Phase 1b (Temporal Benchmarking for Case14)..."
JOB1_ID=$(sbatch --parsable scripts/temp_phase1b_case14.sbatch)
echo "  -> Job ID: $JOB1_ID"

echo "Submitting Phase 1c (Two-Step OPF Evaluation for Case14)..."
JOB2_ID=$(sbatch --parsable --dependency=afterok:$JOB1_ID scripts/temp_phase1c_case14.sbatch)
echo "  -> Job ID: $JOB2_ID"

echo "----------------------------------------------"
echo "Submitted! Run 'squeue -u \$USER' to monitor."
