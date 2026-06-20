#!/bin/bash
#SBATCH --job-name=test_xgb_multi
#SBATCH --partition=alpha    
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --nodes=1
#SBATCH --mem=120G               
#SBATCH --time=4:00:00
#SBATCH --output=logs/case118_multi_out_h24%j.out
#SBATCH --error=logs/case118_multi_out_h24%j.err

set -euo pipefail

module --force purge

source "$SLURM_SUBMIT_DIR/../thesis_env/bin/activate"

start_time=$(date +%s)

# Create output dir if it doesn't exist
mkdir -p phase1_baseline/output/2019-2021_multi_output_XGBoost/

python phase1_baseline/run_benchmark_temporal.py \
  --data-path  /data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1a/data_out/case118_ieee/raw \
  --output-path  phase1_baseline/output/2019-2021_multi_output_XGBoost/case118_ieee_horizon24_xgb.parquet \
  --timing-output phase1_baseline/output/2019-2021_multi_output_XGBoost/timing_benchmark.csv \
  --forecast-horizon 24 \
  --seed 42 \
  --skip-tgt \
  --skip-sarima \
  --xgb-device cuda \
  --n-jobs "${SLURM_CPUS_PER_TASK:-1}"

end_time=$(date +%s)
elapsed=$((end_time - start_time))
elapsed_min=$((elapsed / 60))
elapsed_sec=$((elapsed % 60))
echo "Total script runtime: ${elapsed}s (${elapsed_min}m ${elapsed_sec}s)"
