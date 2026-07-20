#!/bin/bash
#SBATCH --job-name=gridfm_scale_sweep
#SBATCH --partition=barnard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=52
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --output=logs/scale_sweep_%j.out
#SBATCH --error=logs/scale_sweep_%j.err

# --- Configuration ---
REPO_ROOT="$SLURM_SUBMIT_DIR"
VENV_PATH="$REPO_ROOT/../thesis_env"
GENERATE_SCRIPT="$REPO_ROOT/phase1_generation/preprocessing/generate_precomputed_profile.py"
DATAKIT_SCRIPT="$REPO_ROOT/scripts/phase1a_run_datakit.sh"
OUTPUT_DIR="/data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1a/data_out/case500_goc/raw"

# SCALE values to try.
# Derived from data: at SCALE=0.2, per-scenario total P demand for case500_goc
# ranged [1240.6, 3437.2] MW, while the in-service generator fleet requires
# sum(PMIN)=7525.7 MW and allows at most sum(PMAX)=23304.0 MW. Since demand
# scales linearly with SCALE, every scenario's demand falls within
# [sum(PMIN), sum(PMAX)] only for SCALE in [1.2132, 1.3560]. Below 1.2132,
# scenarios are guaranteed infeasible (demand can't reach the must-run floor).
SCALES=(1.29 1.25 1.33 1.22)

# --- Functions ---
set_scale() {
    local scale=$1
    sed -i "s/^SCALE = .*/SCALE = $scale/" "$GENERATE_SCRIPT"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Set SCALE = $scale"
}

run_generation() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running generate_precomputed_profile.py..."
    source $VENV_PATH/bin/activate
    python "$GENERATE_SCRIPT"
    if [ $? -ne 0 ]; then
        echo "ERROR: generate_precomputed_profile.py failed"
        return 1
    fi
    return 0
}

run_datakit() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running phase1a_run_datakit.sh..."
    bash "$DATAKIT_SCRIPT"
    if [ $? -ne 0 ]; then
        echo "ERROR: phase1a_run_datakit.sh failed"
        return 1
    fi
    return 0
}

check_convergence() {
    if [ ! -f "$OUTPUT_DIR/error.log" ]; then
        echo "ERROR: error.log not found at $OUTPUT_DIR/error.log"
        return 1
    fi

    # error.log always contains a 2-line "New generation started" header even
    # when no scenarios fail, so count actual failure lines, not raw wc -l.
    local error_count=$(grep -c "Caught an exception" "$OUTPUT_DIR/error.log")
    if [ "$error_count" -eq 0 ]; then
        return 0  # Success: 0 errors
    else
        return 1  # Failure: errors exist
    fi
}

# --- Main Loop ---
echo "================================"
echo "SCALE Sweep Starting"
echo "================================"

mkdir -p logs

for SCALE in "${SCALES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Testing SCALE = $SCALE"
    echo "=========================================="

    # Set SCALE in the generation script
    set_scale "$SCALE"

    # Run generation
    if ! run_generation; then
        echo "Generation failed for SCALE=$SCALE, skipping..."
        continue
    fi

    # Run datakit pipeline
    if ! run_datakit; then
        echo "Datakit pipeline failed for SCALE=$SCALE, skipping..."
        continue
    fi

    # Check convergence
    if check_convergence; then
        echo ""
        echo "================================"
        echo "SUCCESS! All scenarios converged"
        echo "Optimal SCALE = $SCALE"
        echo "================================"

        # Log the result
        echo "Optimal SCALE found: $SCALE" >> logs/scale_sweep_result.txt
        echo "Timestamp: $(date +'%Y-%m-%d %H:%M:%S')" >> logs/scale_sweep_result.txt

        exit 0
    else
        error_count=$(grep -c "Caught an exception" "$OUTPUT_DIR/error.log")
        echo "SCALE=$SCALE: $error_count scenarios failed. Trying next..."
    fi
done

# If we get here, no SCALE value achieved 100% convergence
echo ""
echo "ERROR: No SCALE value achieved 100% convergence"
echo "Tested SCALES: ${SCALES[@]}"
echo "Recommended: adjust SCALES array or check grid/configuration"
exit 1
