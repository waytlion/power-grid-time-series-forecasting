#!/bin/bash
#SBATCH --job-name=phase1c_eval_xgb_multi
#SBATCH --partition=barnard
#SBATCH --nodes=1 --exclusive
#SBATCH --cpus-per-task=26
#SBATCH --mem=250G
#SBATCH --time=2:00:00
#SBATCH --output=logs/phase1c_eval_case14_h24_multi%j.out
#SBATCH --error=logs/phase1c_eval_case14_h24_multi%j.err

set -euo pipefail

export LD_PRELOAD="${LD_PRELOAD-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"

# --- Configuration ---
CASE="case14"
HORIZON="24"
CASE_DIR="${CASE}_ieee"

VENV_PATH="$SLURM_SUBMIT_DIR/../thesis_env"
DATA_IN_PARQUET="phase1_baseline/output/2019-2021_multi_output_XGBoost/case14_ieee_horizon24_xgb.parquet"
PRECOMPUTED_DIR="exp1/temp_precomputed_profiles/${CASE_DIR}_horizon${HORIZON}_multi_output_XGBoost"
DATAKIT_BASE_YAML="exp1/configs/base_yaml_phase_1c.yaml"
OPF_OUT_DIR="/data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1c/data_out/${CASE_DIR}_horizon_${HORIZON}_multi_output_XGBoost"
GROUND_TRUTH_DIR="/data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1a/data_out/${CASE_DIR}/raw" # from phase1a
FINAL_RESULTS_DIR="exp1/results/${CASE_DIR}_horizon${HORIZON}_multi_output_XGBoost"
# ---------------------


set +u
module --force purge
module load release/2026 GCC/14.2.0
set -u

# 1. Prevent the Julia-Python bridge from checking/resolving dependencies
export PYTHON_JULIAPKG_OFFLINE=yes
# 2. Prevent Julia from attempting to precompile code during execution
export JULIA_PKG_PRECOMPILE_AUTO=0
# 3. Ensure underlying BLAS/OpenMP libraries don't over-subscribe threads
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Always invoke the runtime preflight script
if ! /usr/bin/env python3 "$SLURM_SUBMIT_DIR/tests/scripts/phase1c_preflight.py" --case "$CASE"; then
    echo "Preflight failed; aborting job." >&2
    exit 1
fi

source "$VENV_PATH/bin/activate"
mkdir -p logs
export PYTHONPATH="$SLURM_SUBMIT_DIR"

echo "=========================================="
echo "Phase 1c: Two-Step OPF Evaluation Started"
echo "$(date)"
echo "=========================================="

echo "Step 1: Transform baseline predictions..."
python scripts/phase1c_transform_forecasts.py \
    --case "$CASE" \
    --input-parquet "$DATA_IN_PARQUET" \
    --out-dir "$PRECOMPUTED_DIR"

echo ""
echo "Step 2: Generate OPF results for predictions via DataKit..."
python scripts/phase1c_run_datakit_batch.py \
    --base-yaml "$DATAKIT_BASE_YAML" \
    --network-name "$CASE_DIR" \
    --num-processes "${SLURM_CPUS_PER_TASK:-1}" \
    --data-in-dir "$PRECOMPUTED_DIR" \
    --out-dir "$OPF_OUT_DIR" \
    --models xgb "true"

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
