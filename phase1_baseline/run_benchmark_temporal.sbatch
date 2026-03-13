#!/bin/bash
#SBATCH --job-name=benchmark_temporal
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/benchmark_temporal_%j.out
#SBATCH --error=logs/benchmark_temporal_%j.err

set -euo pipefail

module --force purge
source ~/Thesis_Repo/.venv/bin/activate

python run_benchmark_temporal.py \
  --data-path /data/horse/ws/tibo990i-thesis_data/data_out/3yr_2019-2021/case118_ieee/raw \
  --seed 42