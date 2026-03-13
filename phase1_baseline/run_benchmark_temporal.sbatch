#!/bin/bash
#SBATCH --job-name=benchmark_temporal
#SBATCH --nodes=1
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/benchmark_temporal_%j.out
#SBATCH --error=logs/benchmark_temporal_%j.err

set -euo pipefail # makes failing fast

module --force purge
source ~/Thesis_Repo/.venv/bin/activate

# Set OpenMP threads to match requested CPUs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Default cluster data path (change if needed)
DATA="/data/horse/ws/tibo990i-thesis_data/data_out/3yr_2019-2021/case118_ieee/raw"
export DATA

# Determine the directory where sbatch was invoked
REPO_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
echo "Starting job from submit dir: ${REPO_DIR}"

# Locate the script reliably whether sbatch was run from repo root or phase1_baseline/
if [[ -f "${REPO_DIR}/phase1_baseline/run_benchmark_temporal.py" ]]; then
  SCRIPT_PATH="${REPO_DIR}/phase1_baseline/run_benchmark_temporal.py"
elif [[ -f "${REPO_DIR}/run_benchmark_temporal.py" ]]; then
  SCRIPT_PATH="${REPO_DIR}/run_benchmark_temporal.py"
else
  echo "Error: cannot find run_benchmark_temporal.py under ${REPO_DIR}" >&2
  exit 1
fi

mkdir -p "${REPO_DIR}/phase1_baseline/logs"

python "${SCRIPT_PATH}" \
  --data-path "${DATA}" \
  --seed 42 \