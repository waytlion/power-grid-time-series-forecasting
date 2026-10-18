#!/bin/bash
#SBATCH --job-name=e2e_pd_opf
#SBATCH --partition=barnard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=104
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=logs/e2e_pd_opf_%j.out
#SBATCH --error=logs/e2e_pd_opf_%j.err

# E2E Pd → AC-OPF (exp1/generate_metrics/e2e_pd_opf.py): convert → phase1c chain unchanged (transform → datakit
# AC-OPF → compare.py) → table. Writes only e2e_pd paths:
#   forecasts/precomputed  <DATA>/e2e_pd_opf/{forecasts,precomputed}/
#   OPF                    <DATA>/2019-2021/phase_1c/data_out/<case>_horizon_<H>/e2e_pd/
#   metrics                exp1/results/_e2e_pd/<case>_horizon<H>/e2e_pd/   (`_` → aggregate_results skips it)
# Usage (from login1.barnard, Thesis_Repo root):
#   CASE=case500_goc sbatch scripts/e2e_pd_opf.sbatch
#   CASE=case14_ieee sbatch -c 8 --mem=64G scripts/e2e_pd_opf.sbatch      # case14: < 12 processes (phase1c rule)
#   CASE=case118_ieee HORIZONS=24 sbatch -t 12:00:00 scripts/e2e_pd_opf.sbatch   # h24 (≈ 94k OPF solves)
set -euo pipefail
export LD_PRELOAD="${LD_PRELOAD-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"

CASE="${CASE:?set CASE=case14_ieee|case118_ieee|case500_goc}"
HORIZONS=(${HORIZONS:-1 6})
DATA=/data/horse/ws/tibo990i-thesis_data
GT_DIR="$DATA/2019-2021/phase_1a/data_out/${CASE}/raw"

set +u; module --force purge; module load release/2026 GCC/14.2.0; set -u
export PYTHON_JULIAPKG_OFFLINE=yes JULIA_PKG_PRECOMPILE_AUTO=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 tests/scripts/phase1c_preflight.py --case "$CASE"
source "$SLURM_SUBMIT_DIR/../thesis_env/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR"
mkdir -p logs

python -u exp1/generate_metrics/e2e_pd_opf.py convert --case "$CASE" --horizons "${HORIZONS[@]}"
for H in "${HORIZONS[@]}"; do
    FC="$DATA/e2e_pd_opf/forecasts/${CASE}_horizon${H}.parquet"
    PRE="$DATA/e2e_pd_opf/precomputed/${CASE}_horizon${H}"
    OPF="$DATA/2019-2021/phase_1c/data_out/${CASE}_horizon_${H}"
    echo "### $CASE h$H $(date)"
    python scripts/phase1c_transform_forecasts.py --case "$CASE" --input-parquet "$FC" --out-dir "$PRE"
    python scripts/phase1c_run_datakit_batch.py --base-yaml exp1/configs/base_yaml_phase_1c.yaml \
        --network-name "$CASE" --num-processes "$SLURM_CPUS_PER_TASK" --data-in-dir "$PRE" --out-dir "$OPF" \
        --models e2e_pd
    python exp1/generate_metrics/compare.py --predicted-opf-base-dir "$OPF" --ground-truth-dir "$GT_DIR" \
        --output-dir "exp1/results/_e2e_pd/${CASE}_horizon${H}" --dataset "$CASE" --forecasts-parquet "$FC" \
        --methods e2e_pd
done
python -u exp1/generate_metrics/e2e_pd_opf.py table --case "$CASE" --horizons "${HORIZONS[@]}"
echo "### done $(date)"
