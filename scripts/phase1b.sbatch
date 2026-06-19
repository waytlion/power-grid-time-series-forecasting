#!/bin/bash
#SBATCH --job-name=benchmark_temporal
#SBATCH --partition=capella    
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=28
#SBATCH --nodes=1
#SBATCH --mem=350G               
#SBATCH --time=26:00:00
#SBATCH --output=logs/case500_goc_horizon24%j.out
#SBATCH --error=logs/case500_goc_horizon24%j.err

set -euo pipefail

module --force purge

source "$SLURM_SUBMIT_DIR/../thesis_env/bin/activate"

start_time=$(date +%s)

python phase1_baseline/run_benchmark_temporal.py \
  --data-path  /data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1a/data_out/case500_goc/raw \
  --output-path  phase1_baseline/output/2019-2021/case500_goc_horizon24_HPO_xgb.parquet \
  --timing-output phase1_baseline/output/2019-2021/timing_benchmark.csv \
  --forecast-horizon 24 \
  --seed 42 \
  --skip-tgt \
  #--skip-sarima \
  --xgb-device cuda \
  --n-jobs "${SLURM_CPUS_PER_TASK:-1}"

end_time=$(date +%s)
elapsed=$((end_time - start_time))
elapsed_min=$((elapsed / 60))
elapsed_sec=$((elapsed % 60))
echo "Total script runtime: ${elapsed}s (${elapsed_min}m ${elapsed_sec}s)"
