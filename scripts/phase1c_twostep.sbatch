#!/bin/bash
#SBATCH --job-name=phase1c_twostep
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=08:00:00
# Logs in 'logs/' (must exist).
#SBATCH --output=logs/phase1c_twostep_%j.out
#SBATCH --error=logs/phase1c_twostep_%j.err

# 2-step ML-surrogate OPF arm of Phase 1c (capella/GPU). Cheap: forward passes only.
# Self-contained: (1) transform forecasts -> per-model CSVs (same inputs the datakit
# arm consumes, for fairness), (2) build eval sets + evaluate per (seed, model).

# --- Configuration ---
CASE="case118"
NETWORK="case118_ieee"
HORIZON="24"
DATA_IN_PARQUET="phase1_baseline/output/2019-2021/${NETWORK}_horizon${HORIZON}.parquet"
PRECOMPUTED_DIR="exp1/temp_precomputed_profiles/${NETWORK}_horizon${HORIZON}_3yr"
GT_DIR="/data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1a/data_out/${NETWORK}/raw"
EVAL_CONFIG="$HOME/gridfm-graphkit-dev/examples/config/gnn_heterogeneous_gns/twostep_opf_case118_eval.yaml"
EVAL_SET_DIR="/data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1c/twostep_eval_sets/${NETWORK}_horizon${HORIZON}"
MLRUNS="/data/horse/ws/tibo990i-thesis_data/mlruns"
VENV_PATH="$HOME/thesis_env"
# ---------------------

module --force purge
source "$VENV_PATH/bin/activate"
mkdir -p logs

# Step 1: forecasts parquet -> per-model load CSVs (shared format with the datakit arm)
echo "Step 1: transform forecasts..."
python scripts/phase1c_transform_forecasts.py \
    --case "$CASE" \
    --input-parquet "$DATA_IN_PARQUET" \
    --out-dir "$PRECOMPUTED_DIR"

# Step 2: build eval sets + evaluate the surrogate per (seed, model)
echo "Step 2: build eval sets + evaluate..."
python scripts/phase1c_run_twostep_eval.py \
    --case "$CASE" \
    --network-name "$NETWORK" \
    --horizon "$HORIZON" \
    --data-in-dir "$PRECOMPUTED_DIR" \
    --ground-truth-dir "$GT_DIR" \
    --eval-config "$EVAL_CONFIG" \
    --eval-set-dir "$EVAL_SET_DIR" \
    --mlruns-dir "$MLRUNS" \
    --models xgb sarima snaive tgt true \
    --seeds 0 1 2
