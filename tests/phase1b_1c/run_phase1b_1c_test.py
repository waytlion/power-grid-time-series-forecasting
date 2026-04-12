"""
run_phase1b_1c_test.py
──────────────────────
Main orchestrator for the phase1b → phase1c regression test.

The script runs all 5 pipeline steps in order and then compares the
resulting metrics against a committed golden file. It is designed to run
both locally and inside a SLURM job.

Usage
-----
  # First run: generate golden baseline
  python tests/phase1b_1c/run_phase1b_1c_test.py --generate-golden

  # Every subsequent run: compare against golden
  python tests/phase1b_1c/run_phase1b_1c_test.py

Steps
-----
  1. Phase 1b   – run_benchmark_temporal.py (XGB + SARIMA + snaive, no TGT)
                   with --use-subset --subset-percent 0.10 --seed 42
  2. Transform  – phase1c_transform_forecasts.py (P → Q ratio, CSV prep)
  3. DataKit    – phase1c_run_datakit_batch.py (AC-OPF via Julia/Ipopt)
  4. Metrics    – compare.py (comparison_summary.csv)
  5. Assert     – compare_against_golden.py (regression check)
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Repo root & path setup ────────────────────────────────────────────────────
_FILE = Path(__file__).resolve()
_REPO_ROOT = _FILE.parents[2]          # tests/phase1b_1c/run_phase1b_1c_test.py → repo root
sys.path.insert(0, str(_REPO_ROOT / "tests" / "phase1b_1c"))

from config import (  # noqa: E402
    CASE,
    CASE_DIR,
    HORIZON,
    SEED,
    SUBSET_PERCENT,
    MODELS,
    DEFAULT_NUM_PROCESSES,
    DATA_PATH,
    GROUND_TRUTH_DIR,
    DATAKIT_BASE_YAML,
    TMP_DIR,
    FORECASTS_PARQUET,
    PROFILES_DIR,
    OPF_OUT_DIR,
    RESULTS_DIR,
    GOLDEN_DIR,
    GOLDEN_FILE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOG = logging.getLogger("phase1b_1c_test")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _run(cmd: list, cwd: Optional[Path] = None, step_name: str = "") -> None:
    """Run a subprocess; raise on non-zero exit."""
    cwd = cwd or _REPO_ROOT
    LOG.info("  $ %s", " ".join(str(x) for x in cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        LOG.error("Step '%s' failed (exit %d).", step_name, result.returncode)
        sys.exit(result.returncode)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline steps
# ──────────────────────────────────────────────────────────────────────────────

def step1_phase1b(python: str) -> None:
    """Run benchmark temporal (phase 1b) with a small subset for speed."""
    LOG.info("=" * 60)
    LOG.info("STEP 1 – Phase 1b: forecast benchmark (XGB + SARIMA + snaive)")
    LOG.info("=" * 60)
    FORECASTS_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    _run(
        [
            python,
            str(_REPO_ROOT / "phase1_baseline" / "run_benchmark_temporal.py"),
            "--data-path",     str(DATA_PATH),
            "--output-path",   str(FORECASTS_PARQUET),
            "--forecast-horizon", str(HORIZON),
            "--seed",          str(SEED),
            "--use-subset",
            "--subset-percent", str(SUBSET_PERCENT),
            "--skip-tgt",
            "--xgb-device",   "cpu",
        ],
        step_name="phase1b",
    )


def step2_transform(python: str) -> None:
    """Convert predicted loads to DataKit-ready CSV profiles."""
    LOG.info("=" * 60)
    LOG.info("STEP 2 – Transform forecasts → DataKit CSV profiles")
    LOG.info("=" * 60)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    _run(
        [
            python,
            str(_REPO_ROOT / "scripts" / "phase1c_transform_forecasts.py"),
            "--case",           CASE,
            "--input-parquet",  str(FORECASTS_PARQUET),
            "--out-dir",        str(PROFILES_DIR),
        ],
        step_name="transform_forecasts",
    )


def step3_datakit(python: str, num_processes: int) -> None:
    """Run AC-OPF via DataKit / Julia / Ipopt for each forecast model."""
    LOG.info("=" * 60)
    LOG.info(
        "STEP 3 – DataKit AC-OPF  (models: %s, processes: %d)",
        ", ".join(MODELS),
        num_processes,
    )
    LOG.info("=" * 60)
    OPF_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Count flattened scenarios from a representative CSV
    first_csv = PROFILES_DIR / f"{MODELS[0]}.csv"
    if not first_csv.exists():
        LOG.error("Expected CSV not found: %s", first_csv)
        sys.exit(1)
    scenario_count = int(pd.read_csv(first_csv, usecols=["load_scenario"])["load_scenario"].max() + 1)
    LOG.info("Auto-detected %d flattened scenarios from %s", scenario_count, first_csv.name)

    _run(
        [
            python,
            str(_REPO_ROOT / "scripts" / "phase1c_run_datakit_batch.py"),
            "--base-yaml",       str(DATAKIT_BASE_YAML),
            "--network-name",    CASE_DIR,
            "--num-processes",   str(num_processes),
            "--data-in-dir",     str(PROFILES_DIR),
            "--out-dir",         str(OPF_OUT_DIR),
            "--scenarios",       str(scenario_count),
            "--models",          *MODELS,
        ],
        step_name="datakit_batch",
    )


def step4_metrics(python: str) -> None:
    """Run compare.py to generate comparison_summary.csv."""
    LOG.info("=" * 60)
    LOG.info("STEP 4 – Compute metrics (compare.py)")
    LOG.info("=" * 60)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # compare.py uses sys.path relative imports from exp1/generate_metrics/
    compare_script = _REPO_ROOT / "exp1" / "generate_metrics" / "compare.py"

    _run(
        [
            python,
            str(compare_script),
            "--predicted-opf-base-dir", str((_REPO_ROOT / OPF_OUT_DIR).resolve()),
            "--ground-truth-dir",       str(GROUND_TRUTH_DIR.resolve()),
            "--output-dir",             str((_REPO_ROOT / RESULTS_DIR).resolve()),
            "--dataset",                CASE_DIR,
            "--forecasts-parquet",      str((_REPO_ROOT / FORECASTS_PARQUET).resolve()),
            "--methods",                *MODELS,
        ],
        # compare.py uses bare relative imports (config, loaders, metrics),
        # so it must be run from its own directory for Python to find them.
        # All path *arguments* above are resolved to absolute to avoid cwd confusion.
        cwd=_REPO_ROOT / "exp1" / "generate_metrics",
        step_name="compare_metrics",
    )


def step5_assert(python: str, generate_golden: bool) -> None:
    """Compare results against golden file (or write golden if first run)."""
    LOG.info("=" * 60)
    LOG.info(
        "STEP 5 – Regression check (%s)",
        "GENERATING GOLDEN" if generate_golden else "comparing against golden",
    )
    LOG.info("=" * 60)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        python,
        str(_FILE.parent / "compare_against_golden.py"),
        "--results-dir", str(RESULTS_DIR),
        "--golden",      str(GOLDEN_FILE),
    ]
    if generate_golden:
        cmd.append("--generate-golden")

    _run(cmd, step_name="assert_golden")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase1b → Phase1c regression test."
    )
    parser.add_argument(
        "--generate-golden",
        action="store_true",
        help="Write current results as the new golden file instead of comparing.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help=(
            "Number of parallel Ipopt processes for DataKit. "
            "When submitted via SLURM this is set automatically from "
            "$SLURM_CPUS_PER_TASK — you never need to pass it there. "
            "Only specify this when running the script directly on a "
            f"compute node; otherwise the default of {DEFAULT_NUM_PROCESSES} is used."
        ),
    )
    parser.add_argument(
        "--skip-to-step",
        type=int,
        default=1,
        choices=range(1, 6),
        metavar="N",
        help=(
            "Skip to step N (1=phase1b, 2=transform, "
            "3=datakit, 4=metrics, 5=assert). "
            "Useful for re-running a failed step without restarting."
        ),
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Path to Python interpreter. Defaults to the current interpreter.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    # Resolve number of parallel processes
    num_processes = args.num_processes
    if num_processes is None:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        num_processes = int(slurm_cpus) if slurm_cpus else DEFAULT_NUM_PROCESSES
    LOG.info("Using %d parallel Ipopt process(es).", num_processes)

    python = args.python
    skip_to = args.skip_to_step

    LOG.info("▶  Smoke test started (skip_to_step=%d, generate_golden=%s)",
             skip_to, args.generate_golden)
    LOG.info("   Repo root  : %s", _REPO_ROOT)
    LOG.info("   Case       : %s  horizon=%d  seed=%d", CASE_DIR, HORIZON, SEED)
    LOG.info("   Subset     : %.0f%%  (≒52 usable eval origins on 3yr case14)", SUBSET_PERCENT * 100)

    if skip_to <= 1:
        step1_phase1b(python)
    if skip_to <= 2:
        step2_transform(python)
    if skip_to <= 3:
        step3_datakit(python, num_processes)
    if skip_to <= 4:
        step4_metrics(python)
    if skip_to <= 5:
        step5_assert(python, generate_golden=args.generate_golden)

    LOG.info("▶  Smoke test finished.")


if __name__ == "__main__":
    main()
