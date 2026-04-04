#!/bin/bash
#SBATCH --partition=barnard
#SBATCH --cpus-per-task=64
#SBATCH --mem=450G
#SBATCH --time=02:00:00
#SBATCH --job-name=datakit_case118
#SBATCH --output=logs/datakit_gen_%j.out
#SBATCH --error=logs/datakit_gen_%j.err

VENV_PATH="$HOME/Thesis_Repo/datakit_env"
CONFIG="exp1/configs/case118_generate_opf_for_forecast.yaml"

module reset
module load GCC/14.2.0

source $VENV_PATH/bin/activate

mkdir -p logs

echo "=========================================="
echo "Config: $CONFIG"
echo "Start: $(date)"
echo "=========================================="

# NOTE: gridfm-datakit must be installed in the venv: pip install -e ../gridfm-datakit-fork
srun gridfm_datakit generate "$CONFIG"

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
