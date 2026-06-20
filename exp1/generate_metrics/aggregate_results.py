"""
Aggregate Phase 1c results across all three arms into one comparison table.

Sources -> one tidy long table (approach, case, horizon, model, seed, metric, value):
  - ML arms (surrogate, e2e): MLflow runs, found BY TAGS across all experiments
    (robust to the experiment-name race), metrics read from
    artifacts/test/<network>_metrics.csv.
  - exact arm (datakit): compare.py CSVs under exp1/results/ (approach=exact, no seed).

Then reduce over seed -> mean ± std for the stochastic ML arms; pass-through for exact.

This replaces the manual "MLflow UI -> download -> hand-copy" workflow.
"""

import argparse
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

TAG_KEYS = ["approach", "case", "horizon", "model", "seed", "split"]


def _read_tags(run_dir):
    out = {}
    tagdir = os.path.join(run_dir, "tags")
    for k in TAG_KEYS:
        p = os.path.join(tagdir, k)
        if os.path.exists(p):
            out[k] = open(p).read().strip()
    return out


def collect_mlflow(mlruns_dir, approach=None):
    """Tidy rows from MLflow runs' test metrics CSVs, matched by tags.

    approach: optional filter (e.g. 'surrogate' or 'e2e'); None = any tagged run.
    """
    rows = []
    for run in glob(os.path.join(mlruns_dir, "*", "*", "")):
        tags = _read_tags(run)
        if "approach" not in tags:
            continue
        if approach is not None and tags["approach"] != approach:
            continue
        base = {k: tags.get(k) for k in TAG_KEYS}
        # quality metrics from the test CSV
        for csv in glob(os.path.join(run, "artifacts", "test", "*_metrics.csv")):
            df = pd.read_csv(csv)
            if not {"Metric", "Value"}.issubset(df.columns):
                continue
            for _, r in df.iterrows():
                try:
                    val = float(r["Value"])
                except (ValueError, TypeError):
                    continue
                rows.append({**base, "metric": str(r["Metric"]).strip(), "value": val})
        # MLflow-logged metrics (e.g. infer_time_*) from the metrics/ dir
        mdir = os.path.join(run, "metrics")
        if os.path.isdir(mdir):
            for mf in os.listdir(mdir):
                try:
                    last = open(os.path.join(mdir, mf)).read().strip().splitlines()[-1]
                    val = float(last.split()[1])  # "<ts> <value> <step>"
                except (ValueError, IndexError, OSError):
                    continue
                rows.append({**base, "metric": mf, "value": val})
    return pd.DataFrame(rows)


def collect_compare_csvs(results_dir, case, horizon):
    """Tidy rows from the datakit/exact arm's compare.py per-method metrics CSVs.

    Layout: <results_dir>/<method>/<dataset>_metrics.csv (Metric,Value). approach=exact.
    NOTE: metric names differ from the graphkit CSVs; a name-normalization map is
    applied so the arms line up (extend as needed once real datakit results exist).
    """
    rows = []
    for csv in glob(os.path.join(results_dir, "*", "*_metrics.csv")):
        method = Path(csv).parent.name
        df = pd.read_csv(csv)
        if not {"Metric", "Value"}.issubset(df.columns):
            continue
        for _, r in df.iterrows():
            try:
                val = float(r["Value"])
            except (ValueError, TypeError):
                continue
            rows.append({"approach": "exact", "case": case, "horizon": str(horizon),
                         "model": method, "seed": None, "split": None,
                         "metric": _normalize_metric(str(r["Metric"]).strip()), "value": val})
    return pd.DataFrame(rows)


# graphkit metric name <- compare.py metric name (extend once datakit results are on hand)
_METRIC_ALIASES = {
    "Mean Optimality Gap": "Mean optimality gap (%)",
    "Generator Pg RMSE": "RMSE PG generators (MW)",
    "Residual P (MAE)": "Avg. active res. (MW)",
    "Residual Q (MAE)": "Avg. reactive res. (MVar)",
}


def _normalize_metric(name):
    return _METRIC_ALIASES.get(name, name)


def reduce_over_seed(tidy):
    """mean ± std over seed for ML arms; pass-through for exact (seed is NaN)."""
    keys = ["approach", "case", "horizon", "model", "metric"]
    g = tidy.groupby(keys, dropna=False)["value"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["std"] = out["std"].fillna(0.0)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Aggregate Phase 1c results across arms.")
    p.add_argument("--mlruns-dir", type=Path, required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--horizon", required=True)
    p.add_argument("--compare-results-dir", type=Path, default=None,
                   help="exp1/results/<...> dir with datakit compare.py CSVs (optional)")
    p.add_argument("--out", type=Path, required=True, help="output tidy CSV")
    args = p.parse_args(argv)

    frames = [collect_mlflow(str(args.mlruns_dir))]
    if args.compare_results_dir is not None and args.compare_results_dir.exists():
        frames.append(collect_compare_csvs(str(args.compare_results_dir), args.case, args.horizon))
    tidy = pd.concat([f for f in frames if not f.empty], ignore_index=True)

    # restrict to this case/horizon for the comparison
    tidy = tidy[(tidy["case"] == args.case) & (tidy["horizon"] == str(args.horizon))]
    reduced = reduce_over_seed(tidy)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    reduced.to_csv(args.out, index=False)
    print(f"Wrote {len(reduced)} (approach,model,metric) rows -> {args.out}")
    # quick pivot preview: optimality gap mean±std per (approach, model)
    gap = reduced[reduced["metric"].str.contains("ptimality gap", na=False)]
    if not gap.empty:
        print("\nOptimality gap (%) mean ± std over seed:")
        for _, r in gap.iterrows():
            print(f"  {r['approach']:9s} {str(r['model']):8s} {r['mean']:.3f} ± {r['std']:.3f}  (n={int(r['count'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
