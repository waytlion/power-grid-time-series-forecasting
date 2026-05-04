"""
compare_against_golden.py
─────────────────────────
Compare a fresh comparison_summary.csv against the committed golden file.

Exit codes:
  0 – all metrics within tolerance
  1 – one or more metrics outside tolerance (or missing)
  2 – usage / IO error

Usage
-----
  # First run: generate the golden file
  python tests/phase1b_1c/compare_against_golden.py \\
      --results-dir tests/phase1b_1c/tmp/results \\
      --golden      tests/phase1b_1c/golden/case14_h6_baseline.csv \\
      --generate-golden

  # Subsequent runs: compare against golden
  python tests/phase1b_1c/compare_against_golden.py \\
      --results-dir tests/phase1b_1c/tmp/results \\
      --golden      tests/phase1b_1c/golden/case14_h6_baseline.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Allow running from any working directory when inside the repo
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "tests" / "phase1b_1c"))
from config import (  # noqa: E402
    CHECKED_METRICS,
    REL_TOL,
    ABS_TOL_GAP_PCT,
    GOLDEN_FILE,
    RESULTS_DIR,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"ERROR  File not found: {path}", file=sys.stderr)
        sys.exit(2)
    # Auto-detect delimiter so both comma- and tab-separated summaries are supported.
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df


def _check_metric(
    method: str,
    metric: str,
    new_val: float,
    golden_val: float,
) -> tuple[bool, str]:
    """Return (passed, message)."""
    if metric == "optimality_gap_pct":
        diff = abs(new_val - golden_val)
        ok = diff <= ABS_TOL_GAP_PCT
        msg = (
            f"  {'PASS' if ok else 'FAIL'}  [{method}] {metric}: "
            f"golden={golden_val:.6f}%  new={new_val:.6f}%  "
            f"|Δ|={diff:.4f}pp  tol={ABS_TOL_GAP_PCT}pp (absolute)"
        )
    else:
        if golden_val == 0.0:
            # Avoid division by zero; use absolute diff instead
            ok = abs(new_val - golden_val) < 1e-9
            msg = (
                f"  {'PASS' if ok else 'FAIL'}  [{method}] {metric}: "
                f"golden=0  new={new_val:.6e}  (exact match required for zero golden)"
            )
        else:
            rel = abs(new_val - golden_val) / abs(golden_val)
            ok = rel <= REL_TOL
            msg = (
                f"  {'PASS' if ok else 'FAIL'}  [{method}] {metric}: "
                f"golden={golden_val:.6f}  new={new_val:.6f}  "
                f"|Δ|/golden={rel:.4%}  tol={REL_TOL:.4%} (relative)"
            )
    return ok, msg


def _normalize_summary_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize summary columns across schema versions.

    New schema keeps `rmse_pg_gen` and renamed several columns.
    We map aliases to canonical names used by CHECKED_METRICS.
    """
    out = df.copy()
    alias_map = {
        "Mean optimality gap (%)": "mean_optimality_gap_pct",
        "Avg. active res. (MW)": "avg_active_res_mw",
        "Avg. reactive res. (MVar)": "avg_reactive_res_mvar",
    }
    out = out.rename(columns=alias_map)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare phase1b→1c test results against the golden file."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing the fresh comparison_summary.csv",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=GOLDEN_FILE,
        help="Path to the committed golden CSV file",
    )
    parser.add_argument(
        "--generate-golden",
        action="store_true",
        help="Write current results as the new golden file (first-run bootstrap)",
    )
    args = parser.parse_args(argv)

    summary_path = args.results_dir / "comparison_summary.csv"
    new_df = _normalize_summary_schema(_load_summary(summary_path))

    # ── Generate-golden mode ───────────────────────────────────────────────
    if args.generate_golden:
        args.golden.parent.mkdir(parents=True, exist_ok=True)
        new_df.to_csv(args.golden, index=False)
        print(f"\n✅  Golden file written to: {args.golden}")
        print("    Commit this file to git to lock in the baseline.\n")
        print(new_df.to_string(index=False))
        sys.exit(0)

    # ── Comparison mode ────────────────────────────────────────────────────
    golden_df = _normalize_summary_schema(_load_summary(args.golden))

    required_cols = ["method", *CHECKED_METRICS]
    missing_new = [c for c in required_cols if c not in new_df.columns]
    missing_golden = [c for c in required_cols if c not in golden_df.columns]
    if missing_new or missing_golden:
        if missing_new:
            print(
                "ERROR  New results are missing required columns: "
                f"{missing_new}",
                file=sys.stderr,
            )
        if missing_golden:
            print(
                "ERROR  Golden file is missing required columns: "
                f"{missing_golden}",
                file=sys.stderr,
            )
        print(
            "HINT   Schema likely changed. Re-generate golden with --generate-golden.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  Smoke Test: Regression Check")
    print(f"  Golden : {args.golden}")
    print(f"  New    : {summary_path}")
    print("=" * 70)

    all_pass = True
    results_table = []

    for method in new_df["method"].unique():
        new_row = new_df[new_df["method"] == method]
        golden_rows = golden_df[golden_df["method"] == method]

        if golden_rows.empty:
            print(f"\nWARN  Method '{method}' not found in golden file – skipping.")
            continue

        golden_row = golden_rows.iloc[0]

        for metric in CHECKED_METRICS:
            if metric not in new_row.columns:
                print(f"WARN  Column '{metric}' missing in new results – skipping.")
                continue
            if metric not in golden_row.index:
                print(f"WARN  Column '{metric}' missing in golden file – skipping.")
                continue

            new_val    = float(new_row[metric].iloc[0])
            golden_val = float(golden_row[metric])

            ok, msg = _check_metric(method, metric, new_val, golden_val)
            print(msg)
            if not ok:
                all_pass = False

            results_table.append(
                {
                    "method": method,
                    "metric": metric,
                    "golden": golden_val,
                    "new": new_val,
                    "pass": ok,
                }
            )

    print("\n" + "=" * 70)
    if all_pass:
        print("  ✅  ALL METRICS WITHIN TOLERANCE  – regression test PASSED")
    else:
        n_fail = sum(1 for r in results_table if not r["pass"])
        print(f"  ❌  {n_fail} METRIC(S) OUTSIDE TOLERANCE – regression test FAILED")
    print("=" * 70 + "\n")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
