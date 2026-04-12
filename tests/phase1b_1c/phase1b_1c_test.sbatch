#!/bin/bash
#SBATCH --job-name=phase1b_1c_test
#SBATCH --partition=barnard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/phase1b_1c_test_%j.out
#SBATCH --error=logs/phase1b_1c_test_%j.err

# =============================================================================
#  Phase1b → Phase1c Regression Smoke Test
# =============================================================================
#
#  USAGE
#  ─────
#  # First run – generate the golden baseline file:
#    sbatch tests/phase1b_1c/phase1b_1c_test.sbatch --generate-golden
#
#  # Every subsequent run – compare against the golden file:
#    sbatch tests/phase1b_1c/phase1b_1c_test.sbatch
#
#  # Re-run from a specific step (e.g. after DataKit already finished):
#    sbatch tests/phase1b_1c/phase1b_1c_test.sbatch --skip-to-step 4
#
#  You can also run the Python script directly on a login/compute node:
#    python tests/phase1b_1c/run_phase1b_1c_test.py [--generate-golden] [--num-processes N]
#
#  RESOURCES
#  ─────────
#  --cpus-per-task=8 : passed to DataKit as num_processes (8 parallel Ipopt solves).
#  --mem=64G         : generous for case14 + XGBoost; reduce if quota-limited.
#  --time=02:00:00   : worst-case estimate; usually finishes in 20–40 min.
# =============================================================================

set -euo pipefail

module purge
module load GCC/14.2.0

# Julia / Python bridge settings (mirrors phase1c.sbatch)
export PYTHON_JULIAPKG_OFFLINE=yes
export JULIA_PKG_PRECOMPILE_AUTO=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

VENV_PATH="$SLURM_SUBMIT_DIR/../thesis_env"
source "${VENV_PATH}/bin/activate"

mkdir -p logs

export PYTHONPATH="$SLURM_SUBMIT_DIR"

START_TIME=$(date +%s)

echo "=========================================="
echo "  Smoke Test Started: $(date)"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Node     : $SLURMD_NODENAME"
echo "  CPUs     : $SLURM_CPUS_PER_TASK"
echo "=========================================="

# Forward any extra CLI arguments (e.g. --generate-golden, --skip-to-step N)
# directly to the Python orchestrator.
python tests/phase1b_1c/run_phase1b_1c_test.py \
    --num-processes "${SLURM_CPUS_PER_TASK}" \
    "$@"

ELAPSED=$(( $(date +%s) - START_TIME ))
echo "=========================================="
echo "  Smoke Test Finished: $(date)"
echo "  Wall time : $((ELAPSED / 60))m $((ELAPSED % 60))s"
echo "=========================================="
