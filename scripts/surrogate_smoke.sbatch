#!/bin/bash
#SBATCH --job-name=surr_smoke
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH --output=logs/surr_smoke_%j.out
#SBATCH --error=logs/surr_smoke_%j.err

# Minimal single-eval smoke test of the surrogate arm's canonical metrics.csv
# (seed 0, horizon 1, forecaster 'true'). Validates the new constraint-violation
# quintet + canonical scalar rows on real data.
set -euo pipefail
export LD_PRELOAD="${LD_PRELOAD-}"

CASE="case118_ieee"
H=1; M="true"; SEED=0

THESIS_DIR="$SLURM_SUBMIT_DIR"
VENV_PATH="$THESIS_DIR/../thesis_env"
GRAPHKIT_DIR="$THESIS_DIR/../gridfm-graphkit-dev"
EVAL_CONFIG="$GRAPHKIT_DIR/examples/config/gnn_heterogeneous_gns/twostep_opf_case118_eval.yaml"
LOG_DIR="/data/horse/ws/tibo990i-thesis_data/mlruns"
GROUND_TRUTH_DIR="/data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1a/data_out/${CASE}/raw"
EVAL_BASE="/data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1c/eval_sets/${CASE}"
PRECOMP="$THESIS_DIR/exp1/temp_precomputed_profiles/${CASE}_horizon${H}"
OUT_H="${EVAL_BASE}/h${H}"
PARQUET="$THESIS_DIR/phase1_baseline/output/2019-2021/${CASE}_horizon${H}_HPO_xgb.parquet"

set +u; module --force purge; set -u
source "$VENV_PATH/bin/activate"
mkdir -p logs
export PYTHONPATH="$THESIS_DIR"

echo "### preprocess h${H} model=${M}"
[ -f "${PRECOMP}/${M}.csv" ] || python scripts/phase1c_transform_forecasts.py \
    --case "$CASE" --input-parquet "$PARQUET" --out-dir "$PRECOMP"
python exp1/ml_surrogate/build_eval_set.py \
    --data-in-dir "$PRECOMP" --ground-truth-dir "$GROUND_TRUTH_DIR" \
    --out-dir "$OUT_H" --network-name "$CASE" --models "$M"

# Resolve trained checkpoint (run name uses short case: surrogate_case118_s<seed>)
RUN_NAME="surrogate_case118_s${SEED}"
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
print(os.path.join(art, "model", "best_model_state_dict.pt"),
      os.path.join(art, "stats", "normalizer_stats.pt"))
PY
)
echo "CKPT=$CKPT"
echo "STATS=$STATS"

echo "### evaluate surrogate (seed ${SEED}, h${H}, ${M})"
python -u -m gridfm_graphkit evaluate \
    --config "$EVAL_CONFIG" --model_path "$CKPT" --normalizer_stats "$STATS" \
    --data_path "${OUT_H}/${M}" --log_dir "$LOG_DIR" --exp_name "surrogate_smoke" \
    --run_name "surrogate_smoke_${CASE}_${M}_h${H}_s${SEED}"

echo "### DONE"
