"""
Smoke test configuration.

All tunable constants for the phase1b → phase1c regression test live here.
Change these if you want to adjust the test scope or tolerances.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths  (relative to Thesis_Repo root)
# ---------------------------------------------------------------------------

# Phase1a ground-truth OPF data for case14
GROUND_TRUTH_DIR = Path(
    "/data/horse/ws/tibo990i-thesis_data/data_out"
    "/3yr_2019-2021/phase_1a/data_out/case14_ieee/raw"
)

# Phase1a raw load data (input to phase1b)
DATA_PATH = Path(
    "/data/horse/ws/tibo990i-thesis_data/data_out"
    "/3yr_2019-2021/phase_1a/data_out/case14_ieee/raw"
)

DATAKIT_BASE_YAML = Path("exp1/configs/base_yaml_phase_1c.yaml")

# Smoke test workspace (all intermediate files, ignored by git)
TMP_DIR           = Path("tests/phase1b_1c/tmp")

FORECASTS_PARQUET = TMP_DIR / "forecasts.parquet"
PROFILES_DIR      = TMP_DIR / "profiles"
OPF_OUT_DIR       = TMP_DIR / "opf_out"
RESULTS_DIR       = TMP_DIR / "results"

# Golden file (COMMITTED to git)
GOLDEN_DIR  = Path("tests/phase1b_1c/golden")
GOLDEN_FILE = GOLDEN_DIR / "case14_h6_baseline.csv"

# ---------------------------------------------------------------------------
# Phase 1b settings
# ---------------------------------------------------------------------------

CASE          = "case14"
CASE_DIR      = "case14_ieee"
HORIZON       = 6
SEED          = 42

# Minimum viable subset for the test to have a non-empty evaluation set.
#
# Math (get_temporal_splits: 70/15/15, INPUT_WINDOW=336, HORIZON=6):
#   usable_test_origins = 0.15 × SUBSET_PERCENT × 26_280 − 336 − 6
#   0.10 → 394 − 342 = 52 origins  (minimum; ≈ 936 OPF solves total)
#   0.15 → 591 − 342 = 249 origins
#   0.30 → 1183 − 342 = 841 origins
#
# 0.10 gives 52 evaluation origins × 6 × 3 models = 936 AC-OPF solves on
# case14 (14 buses).  That finishes in a few minutes and is more than enough
# to catch regression bugs.  Do NOT go below 0.09 or the test set vanishes.
SUBSET_PERCENT = 0.10

# All evaluation origins produced by SUBSET_PERCENT are used — no extra
# slicing needed when the subset is already small.

MODELS = ["xgb", "sarima", "snaive"]

# ---------------------------------------------------------------------------
# Phase 1c (DataKit / Ipopt) settings
# ---------------------------------------------------------------------------

# Number of parallel Ipopt processes.
# Override via --num-processes CLI arg or SLURM_CPUS_PER_TASK env var.
DEFAULT_NUM_PROCESSES = 1

# ---------------------------------------------------------------------------
# Regression tolerances
# ---------------------------------------------------------------------------
# A metric X fails if:   |X_new - X_golden| / X_golden > REL_TOL
#   (or, for optimality_gap_pct, absolute diff > ABS_TOL_GAP_PCT)
#
# 5 % relative is intentionally lenient: it catches logic bugs (wrong sign,
# wrong scenario mapping, wrong P→Q ratio) but ignores Ipopt micro-noise
# and floating-point order differences between platforms.

REL_TOL = 0.05          # 5 % relative tolerance for all metrics
ABS_TOL_GAP_PCT = 0.05  # 5 percentage-point absolute tolerance for optimality gap

# Metrics checked against the golden file (columns of comparison_summary.csv)
CHECKED_METRICS = [
    "mae_pd",
    "rmse_pg_gen",
    "rmse_vm",
    "rmse_va",
    "optimality_gap_pct",
    "res_p",
    "res_q",
]
