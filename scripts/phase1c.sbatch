#!/bin/bash
#SBATCH --job-name=phase1c_eval
#SBATCH --partition=barnard
#SBATCH --nodes=1 --exclusive
#SBATCH --cpus-per-task=104
#SBATCH --mem=500G
#SBATCH --time=10:00:00
#SBATCH --output=logs/phase1c_%j.out
#SBATCH --error=logs/phase1c_%j.err

set -euo pipefail

# Some module files reference LD_PRELOAD under `set -u`; define it so the
# environment setup does not fail before the job reaches Python.
export LD_PRELOAD="${LD_PRELOAD-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"

# --- Configuration ---
# Override via env, e.g. CASE=case14_ieee HORIZON=1 MODELS=xgb METHODS=xgb RESULTS_DIR=... sbatch -c 8 --mem=64G scripts/phase1c.sbatch
CASE="${CASE:-case118_ieee}"
HORIZON="${HORIZON:-24}"
MODELS=(${MODELS:-xgb sarima snaive tgt true})      # datakit OPF
METHODS=(${METHODS:-xgb snaive tgt sarima})         # compare.py (default = config.FORECAST_METHODS)

VENV_PATH="$SLURM_SUBMIT_DIR/../thesis_env"
DATA_IN_PARQUET="${DATA_IN_PARQUET:-phase1_baseline/output/2019-2021/${CASE}_horizon${HORIZON}.parquet}"
PRECOMPUTED_DIR="${PRECOMPUTED_DIR:-/data/horse/ws/tibo990i-thesis_data/temp_precomputed_profiles/${CASE}_horizon${HORIZON}_3yr}"   # scratch, off /home (quota)
DATAKIT_BASE_YAML="exp1/configs/base_yaml_phase_1c.yaml"
OPF_OUT_DIR="/data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1c/data_out/${CASE}_horizon_${HORIZON}"
GROUND_TRUTH_DIR="/data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1a/data_out/${CASE}/raw" # from phase1a
FINAL_RESULTS_DIR="${RESULTS_DIR:-exp1/results/${CASE}_horizon${HORIZON}_3yr_HPO_xgb}"
# ---------------------


set +u
module --force purge
module load release/2026 GCC/14.2.0
set -u

# 1. Prevent the Julia-Python bridge from checking/resolving dependencies
export PYTHON_JULIAPKG_OFFLINE=yes
# 2. Prevent Julia from attempting to precompile code during execution
# This avoids creating .ji files and associated locks in the shared cache
export JULIA_PKG_PRECOMPILE_AUTO=0
# 3. Ensure underlying BLAS/OpenMP libraries don't over-subscribe threads
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Always invoke the runtime preflight script placed in tests/scripts
if ! /usr/bin/env python3 "$SLURM_SUBMIT_DIR/tests/scripts/phase1c_preflight.py" --case "$CASE"; then
    echo "Preflight failed; aborting job." >&2
    exit 1
fi


start_time=$(date +%s)
source "$VENV_PATH/bin/activate"
mkdir -p logs
export PYTHONPATH="$SLURM_SUBMIT_DIR"

echo "=========================================="
echo "Phase 1c: Two-Step OPF Evaluation Started"
echo "$(date)"
echo "=========================================="
echo "Configuration:"
echo "  CASE: $CASE"
echo "  HORIZON: $HORIZON"
echo ""

echo "Step 1: Transform baseline predictions..."
python scripts/phase1c_transform_forecasts.py \
    --case "$CASE" \
    --input-parquet "$DATA_IN_PARQUET" \
    --out-dir "$PRECOMPUTED_DIR"

echo ""
echo "Step 2: Generate OPF results for predictions via DataKit..."
python scripts/phase1c_run_datakit_batch.py \
    --base-yaml "$DATAKIT_BASE_YAML" \
    --network-name "$CASE" \
    --num-processes "${SLURM_CPUS_PER_TASK:-1}" \
    --data-in-dir "$PRECOMPUTED_DIR" \
    --out-dir "$OPF_OUT_DIR" \
    --models "${MODELS[@]}"

echo ""
echo "Step 3: Compare results with Ground Truth and generate metrics..."
python exp1/generate_metrics/compare.py \
    --predicted-opf-base-dir "$OPF_OUT_DIR" \
    --ground-truth-dir "$GROUND_TRUTH_DIR" \
    --output-dir "$FINAL_RESULTS_DIR" \
    --dataset "$CASE" \
    --forecasts-parquet "$DATA_IN_PARQUET" \
    --methods "${METHODS[@]}"

echo "=========================================="
echo "Phase 1c Completed: $(date)"
echo "Metrics saved to: $FINAL_RESULTS_DIR"
echo "=========================================="

end_time=$(date +%s)
elapsed=$((end_time - start_time))
elapsed_min=$((elapsed / 60))
elapsed_sec=$((elapsed % 60))
echo "Total script runtime: ${elapsed}s (${elapsed_min}m ${elapsed_sec}s)"
