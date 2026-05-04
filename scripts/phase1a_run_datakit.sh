#!/bin/bash
#SBATCH --job-name=gridfm_datakit_gen
#SBATCH --partition=barnard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=52
#SBATCH --mem=500G
#SBATCH --time=08:00:00

#SBATCH --output=logs/datakit_gen_%j.out
#SBATCH --error=logs/datakit_gen_%j.err

# --- Configuration ---
VENV_PATH="$SLURM_SUBMIT_DIR/../thesis_env"
#! Set the config path relative to Thesis_Repo root:
# CONFIG="phase1_generation/configs/phase1_config.yaml"
CONFIG="phase1_generation/configs/phase1_config.yaml"
# ---------------------

module purge
module load GCC/14.2.0
# 1. Prevent the Julia-Python bridge from checking/resolving dependencies
export PYTHON_JULIAPKG_OFFLINE=yes
# 2. Prevent Julia from attempting to precompile code during execution
# This avoids creating .ji files and associated locks in the shared cache
export JULIA_PKG_PRECOMPILE_AUTO=0
# 3. Ensure underlying BLAS/OpenMP libraries don't over-subscribe threads
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

source $VENV_PATH/bin/activate

mkdir -p logs

# Execute the datakit generation pipeline
# NOTE: gridfm-datakit must be installed in the venv: pip install -e ../gridfm-datakit-fork
srun python -m gridfm_datakit.cli generate $CONFIG
