#!/bin/bash
#SBATCH --job-name=2-step-surrogate-eval
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=logs/2-step-surrogate-eval_%j.out
#SBATCH --error=logs/2-step-surrogate-eval_%j.err

set -euo pipefail
export LD_PRELOAD="${LD_PRELOAD-}"

# ============================ Configuration ============================
CASE="${CASE:-case500_goc}"  # override: sbatch --export=ALL,CASE=...
NETWORK=${CASE%_*}  # strips "_ieee" suffix
HORIZONS=(${HORIZONS:-1 6 24})             # env overrides: HORIZONS="24" FORECASTERS="sarima" SEEDS="0 1 2"
FORECASTERS=(${FORECASTERS:-xgb sarima snaive true})
SEEDS=(${SEEDS:-0})

THESIS_DIR="$SLURM_SUBMIT_DIR"             # Thesis_Repo root
VENV_PATH="$THESIS_DIR/../thesis_env"
GRAPHKIT_DIR="$THESIS_DIR/../gridfm-graphkit-dev"
EVAL_CONFIG="$GRAPHKIT_DIR/examples/config/gnn_heterogeneous_gns/twostep_opf_case118_eval.yaml"
LOG_DIR="/data/horse/ws/tibo990i-thesis_data/mlruns"          # MLflow store (surrogate checkpoints + eval metrics)
GROUND_TRUTH_DIR="/data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1a/data_out/${CASE}/raw"
EVAL_BASE="/data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1c/eval_sets/${CASE}"   # converter output (converting into format, needed for the surrogate)
# Per-horizon forecast parquet (phase1b output). Adjust the pattern to your files.
FORECAST_PARQUET() { echo "$THESIS_DIR/phase1_baseline/output/2019-2021/${CASE}_horizon${1}.parquet"; }
EXP_NAME="2-step-surrogate-eval"
# ======================================================================

set +u; module --force purge; set -u
source "$VENV_PATH/bin/activate"
mkdir -p logs
export PYTHONPATH="$THESIS_DIR"

# ---- 1. Preprocessing: forecasts -> graphkit eval sets (once per horizon) ----
for H in "${HORIZONS[@]}"; do
    PARQUET="$(FORECAST_PARQUET "$H")"
    PRECOMP="$THESIS_DIR/exp1/temp_precomputed_profiles/${CASE}_horizon${H}"
    OUT_H="${EVAL_BASE}/h${H}"
    MISSING=(); for M in "${FORECASTERS[@]}"; do [ -d "${OUT_H}/${M}/${CASE}/raw" ] || MISSING+=("$M"); done
    [ ${#MISSING[@]} -eq 0 ] && { echo "h${H} eval sets exist, skip preprocess"; continue; }
    echo "### preprocess h${H}: ${MISSING[*]}"
    python scripts/phase1c_transform_forecasts.py --case "$CASE" --input-parquet "$PARQUET" --out-dir "$PRECOMP"
    for M in "${MISSING[@]}"; do  # one process per model: case500 h24 OOMs at 128G when built together
        python exp1/ml_surrogate/build_eval_set.py --data-in-dir "$PRECOMP" --ground-truth-dir "$GROUND_TRUTH_DIR" \
            --out-dir "$OUT_H" --network-name "$CASE" --models "$M"
    done
done

# ---- 2. get the mlflow checkpoint (newest, if multiple same run names exist) ----
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

# ---- 3. Evaluate: one graphkit eval per (seed, horizon, forecaster) ----
# run eval setting diff seeds, horizons, network in base yaml
    echo "seed ${SEED}: CKPT=$CKPT"
    for H in "${HORIZONS[@]}"; do
        for M in "${FORECASTERS[@]}"; do   
            # derive a per-case config: only data.networks differs from the case118 base
            CASE_CONFIG=$(mktemp --suffix=.yaml)
            sed "s/^  - case118_ieee\$/  - ${CASE}/" "$EVAL_CONFIG" > "$CASE_CONFIG"
            EVAL_SET="${EVAL_BASE}/h${H}/${M}"
            [ -d "${EVAL_SET}/${CASE}/raw" ] || { echo "SKIP missing ${EVAL_SET}"; continue; }
            python -u -m gridfm_graphkit evaluate \
                --config "$CASE_CONFIG" --model_path "$CKPT" --normalizer_stats "$STATS" \
                --data_path "$EVAL_SET" --log_dir "$LOG_DIR" --exp_name "$EXP_NAME" \
                --run_name "surrogate_${CASE}_${M}_h${H}_s${SEED}"
        done
    done
done
