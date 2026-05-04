#!/bin/bash
#SBATCH --job-name=benchmark_temporal
#SBATCH --partition=alpha    
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --nodes=1
#SBATCH --mem=120G               
#SBATCH --time=26:00:00
#SBATCH --output=logs/case118_ieee_horizon1_%j.out
#SBATCH --error=logs/case118_ieee_horizon1_%j.err

set -euo pipefail

module --force purge

source "$SLURM_SUBMIT_DIR/../thesis_env/bin/activate"

python phase1_baseline/run_benchmark_temporal.py \
  --data-path  /data/horse/ws/tibo990i-thesis_data/2002-2025/phase_1a/data_out/case118_ieee/raw \
  --output-path  phase1_baseline/output/2002-2025/case118_ieee_horizon1.parquet \
  --seed 42 \
  --skip-tgt \
  --xgb-device cuda \
  --n-jobs "${SLURM_CPUS_PER_TASK:-1}"

