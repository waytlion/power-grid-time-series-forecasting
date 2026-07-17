#!/bin/bash
#SBATCH --job-name=surrogate_eval
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=logs/surrogate_eval_%j.out
#SBATCH --error=logs/surrogate_eval_%j.err

set -euo pipefail
export LD_PRELOAD="${LD_PRELOAD-}"

# ============================ Configuration ============================
CASE="case118_ieee"
NETWORK=${CASE%_*}  # strips "_ieee" suffix
HORIZONS=(1 24)                        
FORECASTERS=(xgb sarima snaive true)
SEEDS=(0 1 2)

THESIS_DIR="$SLURM_SUBMIT_DIR"             # Thesis_Repo root
VENV_PATH="$THESIS_DIR/../thesis_env"
GRAPHKIT_DIR="$THESIS_DIR/../gridfm-graphkit-dev"
EVAL_CONFIG="$GRAPHKIT_DIR/examples/config/gnn_heterogeneous_gns/twostep_opf_case118_eval.yaml"
LOG_DIR="/data/horse/ws/tibo990i-thesis_data/mlruns"          # MLflow store (surrogate checkpoints + eval metrics)
GROUND_TRUTH_DIR="/data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1a/data_out/${CASE}/raw"
EVAL_BASE="/data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1c/eval_sets/${CASE}"   # converter output (converting into format, needed for the surrogate)
# Per-horizon forecast parquet (phase1b output). Adjust the pattern to your files.
FORECAST_PARQUET() { echo "$THESIS_DIR/phase1_baseline/output/2019-2021/${CASE}_horizon${1}_HPO_xgb.parquet"; }
EXP_NAME="surrogate_eval"
# ======================================================================

set +u; module --force purge; set -u
source "$VENV_PATH/bin/activate"
mkdir -p logs
export PYTHONPATH="$THESIS_DIR"

# ---- 1. Preprocess: forecasts -> graphkit eval sets (once per horizon) ----
for H in "${HORIZONS[@]}"; do
    PARQUET="$(FORECAST_PARQUET "$H")"
    PRECOMP="$THESIS_DIR/exp1/temp_precomputed_profiles/${CASE}_horizon${H}"
    OUT_H="${EVAL_BASE}/h${H}"
    [ -d "${OUT_H}/${FORECASTERS[0]}/${CASE}/raw" ] && { echo "h${H} eval sets exist, skip preprocess"; continue; }
    echo "### preprocess h${H}"
    python scripts/phase1c_transform_forecasts.py --case "$CASE" --input-parquet "$PARQUET" --out-dir "$PRECOMP"
    python exp1/ml_surrogate/build_eval_set.py --data-in-dir "$PRECOMP" --ground-truth-dir "$GROUND_TRUTH_DIR" \
        --out-dir "$OUT_H" --network-name "$CASE" --models "${FORECASTERS[@]}"
done

# ---- 2. Evaluate: one graphkit eval per (seed, horizon, forecaster) ----
for SEED in "${SEEDS[@]}"; do
    # resolve this seed's trained surrogate checkpoint by run name (filesystem glob)
    RUN_NAME="surrogate_${NETWORK}_s${SEED}"
    read -r CKPT STATS < <(python - <<PY
import glob, os
LOG_DIR, RUN_NAME = "${LOG_DIR}", "${RUN_NAME}"
cands = []
for tf in glob.glob(os.path.join(LOG_DIR, "*", "*", "tags", "mlflow.runName")):
    try:
        if open(tf).read().strip() == RUN_NAME:
            rd = os.path.dirname(os.path.dirname(tf)); cands.append((os.path.getmtime(rd), rd))
    except OSError:
        pass
assert cands, f"No surrogate run named {RUN_NAME}"
art = os.path.join(max(cands)[1], "artifacts")
print(os.path.join(art, "model", "best_model_state_dict.pt"), os.path.join(art, "stats", "normalizer_stats.pt"))
PY
)
    echo "seed ${SEED}: CKPT=$CKPT"
    for H in "${HORIZONS[@]}"; do
        for M in "${FORECASTERS[@]}"; do
            EVAL_SET="${EVAL_BASE}/h${H}/${M}"
            [ -d "${EVAL_SET}/${CASE}/raw" ] || { echo "SKIP missing ${EVAL_SET}"; continue; }
            python -u -m gridfm_graphkit evaluate \
                --config "$EVAL_CONFIG" --model_path "$CKPT" --normalizer_stats "$STATS" \
                --data_path "$EVAL_SET" --log_dir "$LOG_DIR" --exp_name "$EXP_NAME" \
                --run_name "surrogate_${CASE}_${M}_h${H}_s${SEED}"
        done
    done
done
