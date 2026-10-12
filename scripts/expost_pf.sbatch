#!/bin/bash
#SBATCH --job-name=expost_pf
#SBATCH --partition=barnard
#SBATCH --nodes=1 --exclusive
#SBATCH --cpus-per-task=104
#SBATCH --mem=500G
#SBATCH --time=06:00:00
#SBATCH --output=logs/expost_pf_%j.out
#SBATCH --error=logs/expost_pf_%j.err

# Ex-post AC-PF feasibility per arm (exp1/generate_metrics/expost_pf.py), one case per job.
# Submit from Thesis_Repo root; override from env, e.g.
#   CASE=case500_goc sbatch scripts/expost_pf.sbatch
#   CASE=case118_ieee HORIZONS=6 LIMIT=200 sbatch scripts/expost_pf.sbatch     # smoke test
# Output: /data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1c/expost_pf/<case>/h<H>/<arm>[_<fc>][_s<seed>].parquet
set -euo pipefail
export LD_PRELOAD="${LD_PRELOAD-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"

CASE=${CASE:-case118_ieee}
HORIZONS=${HORIZONS:-"1 6 24"}
ARMS=${ARMS:-"label exact surrogate e2e"}
FORECASTERS=${FORECASTERS:-"true xgb sarima snaive"}
SEEDS=${SEEDS:-0}
LIMIT=${LIMIT:-}

set +u
module --force purge
module load release/2026 GCC/14.2.0
set -u
export PYTHON_JULIAPKG_OFFLINE=yes JULIA_PKG_PRECOMPILE_AUTO=0  # no dependency resolution / precompile locks
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

source "$SLURM_SUBMIT_DIR/../thesis_env/bin/activate"
mkdir -p logs

python -u exp1/generate_metrics/expost_pf.py --case "$CASE" --horizons $HORIZONS --arms $ARMS \
    --forecasters $FORECASTERS --seeds $SEEDS --workers "$SLURM_CPUS_PER_TASK" ${LIMIT:+--limit $LIMIT}
