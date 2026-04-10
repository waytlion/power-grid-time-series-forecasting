#!/bin/bash
#SBATCH --job-name=benchmark_temporal
#SBATCH --partition=barnard    
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G               
#SBATCH --time=06:00:00
#SBATCH --output=logs/case118_ieee_horizon1_%j.out
#SBATCH --error=logs/case118_ieee_horizon1_%j.err

set -euo pipefail

module --force purge

source "$SLURM_SUBMIT_DIR/../thesis_env/bin/activate"

python phase1_baseline/run_benchmark_temporal.py \
  --data-path  /data/horse/ws/tibo990i-thesis_data/data_out/3yr_2019-2021/phase_1a/data_out/case118_ieee/raw \
  --output-path  phase1_baseline/output/case118_ieee_horizon1.parquet \
  --seed 42 \
  --skip-tgt \
  --xgb-device cpu 
