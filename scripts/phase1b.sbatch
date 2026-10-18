#!/bin/bash
#SBATCH --job-name=benchmark_temporal
#SBATCH --partition=capella    
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=28
#SBATCH --nodes=1
#SBATCH --mem=350G               
#SBATCH --time=26:00:00
# Generic fallback (SBATCH directives can't interpolate CASE/HORIZON — they're
# parsed before the script runs). Real per-run logs are set below once vars exist.
#SBATCH --output=logs/phase1b_%j.out
#SBATCH --error=logs/phase1b_%j.err

# --- Configuration --- (env overrides; extra sbatch args are passed through to the python script)
# e.g. CASE=case14_ieee HORIZON=1 OUT=<parquet> TIMING=<csv> TAG=_A1 sbatch scripts/phase1b.sbatch --skip-sarima
CASE="${CASE:-case118_ieee}"
HORIZON="${HORIZON:-24}"
OUT="${OUT:-phase1_baseline/output/2019-2021/${CASE}_horizon${HORIZON}.parquet}"
TIMING="${TIMING:-phase1_baseline/output/2019-2021/timing_benchmark.csv}"

set -euo pipefail

# Route logs to a name that ALWAYS matches the real case/horizon (avoids the stale
# hard-coded log-name trap). Redirect here where CASE/HORIZON are known.
mkdir -p logs
exec  > "logs/${CASE}_horizon${HORIZON}${TAG:-}_j${SLURM_JOB_ID}.out" \
     2> "logs/${CASE}_horizon${HORIZON}${TAG:-}_j${SLURM_JOB_ID}.err"

module --force purge

source "$SLURM_SUBMIT_DIR/../thesis_env/bin/activate"

start_time=$(date +%s)
python phase1_baseline/run_benchmark_temporal.py \
  --data-path  /data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1a/data_out/${CASE}/raw \
  --output-path "${OUT}" \
  --timing-output "${TIMING}" \
  --forecast-horizon ${HORIZON} \
  --seed 42 \
  --skip-tgt \
  --xgb-device cuda \
  --n-jobs "${SLURM_CPUS_PER_TASK:-1}" \
  "$@"

end_time=$(date +%s)
elapsed=$((end_time - start_time))
elapsed_min=$((elapsed / 60))
elapsed_sec=$((elapsed % 60))
echo "Total script runtime: ${elapsed}s (${elapsed_min}m ${elapsed_sec}s)"
