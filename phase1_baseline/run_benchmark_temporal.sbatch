#!/bin/bash
#SBATCH --job-name=benchmark_temporal
#SBATCH --partition=barnard    
#SBATCH --cpus-per-task=64       
#SBATCH --mem=128G               
#SBATCH --time=02:00:00
#SBATCH --output=logs/case118_ieee_horizon1_%j.out
#SBATCH --error=logs/case118_ieee_horizon1_%j.err

set -euo pipefail

module --force purge

source ~/Thesis_Repo/.venv/bin/activate

python run_benchmark_temporal.py \
  --data-path /data/horse/ws/tibo990i-thesis_data/data_out/3yr_2019-2021/case118_ieee/raw \
  --seed 42 \
  --skip-tgt \
  --xgb-device cpu \
  --output-path case118_ieee_horizon1.parquet