#!/bin/bash
#SBATCH --job-name=gridfm_datakit_gen
#SBATCH --partition=paula
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=02:00:00

#SBATCH --output=logs/datakit_gen_%j.out
#SBATCH --error=logs/datakit_gen_%j.err

# --- Configuration ---
VENV_PATH="$SLURM_SUBMIT_DIR/../thesis_env"
#! Set the config path relative to Thesis_Repo root:
# CONFIG="phase1_generation/configs/phase1_config.yaml"
CONFIG="exp1/configs/cluster_leipzig_opf_for_forecast.yaml"
# ---------------------

module purge
module load Anaconda3

source $VENV_PATH/bin/activate

mkdir -p logs

# Execute the datakit generation pipeline
# NOTE: gridfm-datakit must be installed in the venv: pip install -e ../gridfm-datakit-fork
srun python -m gridfm_datakit.cli generate $CONFIG
