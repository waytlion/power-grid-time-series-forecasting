#!/bin/bash
# submit_pipeline.sh
# 
# Master submission script to automate the entire thesis workflow.
# This chains Phase 1A, Phase 1B, and Phase 1C using SLURM dependencies.
#
# Usage: 
#   cd Thesis_Repo
#   bash scripts/submit_pipeline.sh

set -euo pipefail

echo "Submitting Full Thesis Pipeline"
echo "--------------------------------"

# Step 1: Subscribe Phase 1a (OPF Ground Truth Generation)
echo "Submitting Phase 1a (Ground Truth Definition)..."
JOB1_ID=$(sbatch --parsable scripts/cluster_leipzig.sh)
echo "  -> Job ID: $JOB1_ID"

# Step 2: Subscribe Phase 1b (Temporal Benchmarks)
# Only runs after Job 1 completes successfully
echo "Submitting Phase 1b (Temporal Benchmarking)..."
cd phase1_baseline
JOB2_ID=$(sbatch --parsable --dependency=afterok:$JOB1_ID run_benchmark_temporal.sbatch)
echo "  -> Job ID: $JOB2_ID"
cd ..

# Step 3: Subscribe Phase 1c (Evaluation and Metrics)
# Only runs after Job 2 completes successfully
echo "Submitting Phase 1c (Two-Step OPF Evaluation)..."
JOB3_ID=$(sbatch --parsable --dependency=afterok:$JOB2_ID scripts/phase1c_eval.sbatch)
echo "  -> Job ID: $JOB3_ID"

echo "--------------------------------"
echo "Pipeline submitted successfully."
echo "Use 'squeue -u \$USER' to monitor the jobs."
