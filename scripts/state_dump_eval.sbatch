#!/bin/bash
#SBATCH --job-name=state-dump-eval
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=logs/state-dump-eval_%j.out
#SBATCH --error=logs/state-dump-eval_%j.err

# Eval-only rerun of trained surrogate + E2E checkpoints -> per-bus/per-gen pred vs true dumps
# (<dataset>_state_{bus,gen}.parquet in artifacts/test/) for plot_state_error_dist.py.
# Only needed for runs trained/evaluated BEFORE the dump existed; new runs write it themselves.
# Separate EXP_NAME keeps the aggregate_results mlruns dirs untouched.
# Gate: each new run's metrics must reproduce the original run's logged metrics.
set -euo pipefail
export LD_PRELOAD="${LD_PRELOAD-}"

# override from env, e.g. CASES=case500_goc HORIZONS="6 24" FORECASTERS= sbatch ... (empty FORECASTERS = E2E only)
CASES=(${CASES:-case14_ieee case118_ieee case500_goc})
HORIZONS=(${HORIZONS:-1 6 24})
FORECASTERS=(${FORECASTERS-xgb sarima snaive})  # surrogate only; E2E has no forecaster
SEED=0
GRAPHKIT_DIR="$HOME/gridfm-graphkit-dev"
LOG_DIR="/data/horse/ws/tibo990i-thesis_data/mlruns"
EXP_NAME="state-dump"
E2E_EXP="375695290152400764"   # final_thesis_runs = the E2E experiment aggregate_results reads
SURR_CONFIG="$GRAPHKIT_DIR/examples/config/gnn_heterogeneous_gns/twostep_opf_case118_eval.yaml"
E2E_DATA="/data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1a/data_out"

set +u; module --force purge; set -u
source "$HOME/thesis_env/bin/activate"
mkdir -p logs

# newest checkpointed run with this name (optionally within one experiment) -> "<ckpt> <normalizer_stats>"
resolve() { python - "$LOG_DIR" "$1" "${2:-*}" <<'PY'
import glob, os, sys
log_dir, name, exp = sys.argv[1:]
runs = [os.path.dirname(os.path.dirname(t)) for t in glob.glob(f"{log_dir}/{exp}/*/tags/mlflow.runName")
        if open(t).read().strip() == name]
runs = [r for r in runs if all(os.path.exists(f"{r}/artifacts/{p}") for p in ("model/best_model_state_dict.pt", "test"))]
assert runs, f"no checkpointed run named {name}"
r = max(runs, key=os.path.getmtime)
print(f"{r}/artifacts/model/best_model_state_dict.pt {r}/artifacts/stats/normalizer_stats.pt")
PY
}

# new (has state dump) vs newest original run of the same name: print metrics that differ > 0.1 %
gate() { python - "$LOG_DIR" "$1" "$2" "${3:-*}" <<'PY'
import glob, os, sys
import pandas as pd
log_dir, name, ds, old_exp = sys.argv[1:]
runs = [os.path.dirname(os.path.dirname(t)) for t in glob.glob(f"{log_dir}/*/*/tags/mlflow.runName")
        if open(t).read().strip() == name and os.path.exists(f"{os.path.dirname(os.path.dirname(t))}/artifacts/test/{ds}_metrics.csv")]
new = [r for r in runs if os.path.exists(f"{r}/artifacts/test/{ds}_state_bus.parquet")]
canonical = lambda r: "Mean Optimality Gap" in open(f"{r}/artifacts/test/{ds}_metrics.csv").read()  # aggregate_results marker
# "old" must be a pre-existing canonical run NOT in old_exp's state-dump pass -- pin to old_exp (the
# arm's real experiment, e.g. final_thesis_runs for E2E) so a same-named ablation run elsewhere can't win
old = [r for r in runs if r not in new and canonical(r) and (old_exp == "*" or f"/mlruns/{old_exp}/" in r)]
if not (new and old):
    sys.exit(f"GATE {name}: missing new ({len(new)}) or original ({len(old)}) run")
read = lambda r: pd.read_csv(f"{r}/artifacts/test/{ds}_metrics.csv").set_index("Metric")["Value"]
a, b = read(max(new, key=os.path.getmtime)), read(max(old, key=os.path.getmtime))
a, b = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
keep = [m for m in a.index.intersection(b.index) if not any(s in m.lower() for s in ("time", "infer", "_s"))]
rel = ((a[keep] - b[keep]).abs() / b[keep].abs().clip(lower=1e-9)).dropna()
bad = rel[rel > 1e-3]
print(f"GATE {name}: {len(keep)} metrics compared, {len(bad)} differ > 0.1%" + ("" if bad.empty else f"\n{bad}"))
PY
}

cd "$GRAPHKIT_DIR"
echo "graphkit: $(git rev-parse --short HEAD) ($(git branch --show-current)), dirty files: $(git status --short | wc -l)"

for CASE in "${CASES[@]}"; do
    NETWORK=${CASE%_*}                       # case500_goc -> case500 (surrogate/E2E run names)
    CASE_CONFIG=$(mktemp --suffix=.yaml)
    sed "s/^  - case118_ieee\$/  - ${CASE}/" "$SURR_CONFIG" > "$CASE_CONFIG"
    read -r SURR_CKPT SURR_STATS < <(resolve "surrogate_${NETWORK}_s${SEED}")
    for H in "${HORIZONS[@]}"; do
        for FC in "${FORECASTERS[@]}"; do
            RUN="surrogate_${CASE}_${FC}_h${H}_s${SEED}"
            EVAL_SET="/data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1c/eval_sets/${CASE}/h${H}/${FC}"
            [ -d "$EVAL_SET/${CASE}/raw" ] || { echo "SKIP missing $EVAL_SET"; continue; }
            echo "### $RUN  CKPT=$SURR_CKPT"
            python -u -m gridfm_graphkit evaluate --config "$CASE_CONFIG" --model_path "$SURR_CKPT" \
                --normalizer_stats "$SURR_STATS" --log_dir "$LOG_DIR" --exp_name "$EXP_NAME" --run_name "$RUN" \
                --data_path "$EVAL_SET"
            gate "$RUN" "$CASE" || true
        done

        RUN="base_e2e_${NETWORK}_h${H}_s${SEED}"
        read -r CKPT STATS < <(resolve "$RUN" "$E2E_EXP")
        echo "### $RUN  CKPT=$CKPT"
        python -u -m gridfm_graphkit evaluate --config "examples/config/final_thesis_runs/base_${CASE%_ieee}_h${H}.yaml" \
            --model_path "$CKPT" --normalizer_stats "$STATS" --data_path "$E2E_DATA" \
            --log_dir "$LOG_DIR" --exp_name "$EXP_NAME" --run_name "$RUN"
        gate "$RUN" "$CASE" "$E2E_EXP" || true
    done
done
