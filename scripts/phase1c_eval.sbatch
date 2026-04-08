#!/bin/bash
#SBATCH --job-name=phase1c_eval
#SBATCH --partition=paula
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/phase1c_eval_%j.out
#SBATCH --error=logs/phase1c_eval_%j.err

set -euo pipefail

# --- Configuration ---
CASE="case14"
HORIZON="1"
CASE_DIR="${CASE}_ieee"

VENV_PATH="$SLURM_SUBMIT_DIR/../thesis_env"
DATA_IN_PARQUET="phase1_baseline/output/${CASE_DIR}_horizon${HORIZON}.parquet"
PRECOMPUTED_DIR="exp1/temp_precomputed_profiles/${CASE_DIR}_horizon${HORIZON}_3yr"
DATAKIT_BASE_YAML="exp1/configs/base_yaml_phase_1c.yaml"
OPF_OUT_DIR="~/data/horse/ws/tibo990i-thesis_data/data_out/3yr_2019-2021/phase_1c/data_out/${CASE_DIR}_horizon_${HORIZON}_3yr"
GROUND_TRUTH_DIR="~/data/horse/ws/tibo990i-thesis_data/data_out/3yr_2019-2021/phase_1a/data_out/${CASE_DIR}/raw" # from phase1a
FINAL_RESULTS_DIR="exp1/results/${CASE_DIR}_horizon${HORIZON}_3yr"
# ---------------------

module purge
module load Anaconda3

source "$VENV_PATH/bin/activate"
mkdir -p logs
export PYTHONPATH="$SLURM_SUBMIT_DIR"

echo "=========================================="
echo "Phase 1c: Two-Step OPF Evaluation Started"
echo "$(date)"
echo "=========================================="

echo "Step 1: Transform baseline predictions..."
python scripts/transform_forecasts.py \
    --case "$CASE" \
    --input-parquet "$DATA_IN_PARQUET" \
    --out-dir "$PRECOMPUTED_DIR"

echo ""
echo "Step 2: Generate OPF results for predictions via DataKit..."
python scripts/run_datakit_batch.py \
    --base-yaml "$DATAKIT_BASE_YAML" \
    --data-in-dir "$PRECOMPUTED_DIR" \
    --out-dir "$OPF_OUT_DIR" \
    --models xgb sarima snaive tgt "true"

echo ""
echo "Step 3: Compare results with Ground Truth and generate metrics..."
python exp1/generate_metrics/compare.py \
    --predicted-opf-base-dir "$OPF_OUT_DIR" \
    --ground-truth-dir "$GROUND_TRUTH_DIR" \
    --output-dir "$FINAL_RESULTS_DIR" \
    --dataset "$CASE_DIR" \
    --forecasts-parquet "$DATA_IN_PARQUET"

echo "=========================================="
echo "Phase 1c Completed: $(date)"
echo "Metrics saved to: $FINAL_RESULTS_DIR"
echo "=========================================="
