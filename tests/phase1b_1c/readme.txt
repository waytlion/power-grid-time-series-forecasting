# =============================================================================
# Phase1b → Phase1c Regression Smoke Test
# =============================================================================
# This suite tests phase 1b and 1c end-to-end on case_14 with 10% of the data. 
# It compares the results of the end-to-end run against a committed "golden file"
# (verified baseline metrics). If the new metrics deviate from the golden file 
# beyond a defined tolerance, the test prints the difference and fails (exit code 1).
# 
# CONFIGURATION & TOLERANCES
# ──────────────────────────
# Settings, paths, and allowed error tolerances (e.g. REL_TOL = 5%) are defined
# centrally in `tests/phase1b_1c/config.py`.
#
# ARTIFACTS & DIRECTORIES
# ───────────────────────
# - Intermediate outputs (parquets, CSVs, DataKit OPF results) go to `tests/phase1b_1c/tmp/` (git-ignored)
# - The baseline CSV is saved in `tests/phase1b_1c/golden/` (MUST BE COMMITTED!)
#
# USAGE
# ─────
# First run – generate the golden baseline file:
#   sbatch tests/phase1b_1c/phase1b_1c_test.sbatch --generate-golden
#
# Every subsequent run – compare against the golden file:
#   sbatch tests/phase1b_1c/phase1b_1c_test.sbatch
#
# Re-run from a specific step (e.g. after DataKit already finished):
#   sbatch tests/phase1b_1c/phase1b_1c_test.sbatch --skip-to-step 4
#
# You can also run the Python script directly on an interactive/compute node:
#   python tests/phase1b_1c/run_phase1b_1c_test.py [--generate-golden] [--num-processes N]
#
# TROUBLESHOOTING
# ───────────────
# If the test FAILS due to an intentional code change (e.g., fixing a calculation), 
# you must re-run with `--generate-golden` to overwrite the baseline, and then 
# commit the updated golden file.
# =============================================================================
