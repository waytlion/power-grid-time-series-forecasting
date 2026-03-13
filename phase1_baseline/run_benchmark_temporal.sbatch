#!/bin/bash
#SBATCH --job-name=benchmark_temporal
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/benchmark_temporal_%j.out
#SBATCH --error=logs/benchmark_temporal_%j.err

set -euo pipefail #makes failing fast 

module purge
source ~/Thesis_Repo/.venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)
cd "${REPO_DIR}"

mkdir -p "${SCRIPT_DIR}/logs"

python phase1_baseline/run_benchmark_temporal.py \
  --seed 42 \
