#!/bin/bash
#SBATCH --job-name=phase1c_step3
#SBATCH --partition=alpha
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1 
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=logs/phase1c_eval_step3_%j.out
#SBATCH --error=logs/phase1c_eval_step3_%j.err

set -euo pipefail

# --- Configuration ---
CASE_DIR="case118_ieee"
HORIZON="1"


VENV_PATH="$SLURM_SUBMIT_DIR/../thesis_env"
DATA_IN_PARQUET="phase1_baseline/output/2019-2021/${CASE_DIR}_horizon${HORIZON}.parquet"
OPF_OUT_DIR="/data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1c/data_out/${CASE_DIR}_horizon_${HORIZON}"
GROUND_TRUTH_DIR="/data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1a/data_out/${CASE_DIR}/raw"
FINAL_RESULTS_DIR="exp1/results/${CASE_DIR}_horizon${HORIZON}"
# ---------------------

module purge
module load GCC/14.2.0

source "$VENV_PATH/bin/activate"
mkdir -p logs
export PYTHONPATH="$SLURM_SUBMIT_DIR"

echo "=========================================="
echo "Phase 1c: Step 3 (Metrics) Started"
echo "$(date)"
echo "=========================================="

echo "Step 3: Compare results with Ground Truth and generate metrics..."
python exp1/generate_metrics/compare.py \
    --predicted-opf-base-dir "$OPF_OUT_DIR" \
    --ground-truth-dir "$GROUND_TRUTH_DIR" \
    --output-dir "$FINAL_RESULTS_DIR" \
    --dataset "$CASE_DIR" \
    --forecasts-parquet "$DATA_IN_PARQUET"

echo "=========================================="
echo "Phase 1c Step 3 Completed: $(date)"
echo "Metrics saved to: $FINAL_RESULTS_DIR"
echo "=========================================="
