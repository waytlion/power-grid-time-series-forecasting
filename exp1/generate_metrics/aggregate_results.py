"""
Aggregate Phase 1c results across all arms into one comparison table.

Tidy long schema: case, horizon, optimizer, forecaster, seed, metric, value
  optimizer  in {ac, dc, surrogate, e2e}
  forecaster in {xgb, sarima, snaive, tgt, true, none}   (none = E2E)

Quality metrics come from compare.py-format CSVs (score_ml_arm for the ML arms,
compare.py for the exact arm) under <results-root>/<arm>/h<H>/<forecaster>/[s<S>]/.
The exact CSV carries AC and DC together (DC-prefixed) and is split into the two
optimizers here.

Timing stays in each producer's native location and is combined here:
  - forecast infer/train          <- timing_benchmark.csv
  - AC/DC solve                   <- datakit runtime_data.parquet (ac/dc per scenario)
  - surrogate/E2E forward + train <- MLflow (matched by tags)

Emitted timing metrics (per case,horizon,optimizer,forecaster,seed):
  opf_step_infer_s       per-dispatch optimizer time (solve / forward)
  full_pipeline_infer_s  forecast + opf-step per dispatch (E2E = forward only)
  infer_total_min        absolute total inference over the eval set
  train_time_min         one-time training (forecaster + model); datakit = 0

Then reduce over seed -> mean +/- std for the stochastic ML arms.
"""

import argparse
import os
import re
from glob import glob
from pathlib import Path

import pandas as pd

FORECASTERS = {"xgb", "sarima", "snaive", "tgt", "true"}
TIMED_FORECASTERS = ("xgb", "sarima")  # measured in timing_benchmark.csv
ZERO_FORECASTERS = ("snaive", "true")  # lookup baseline / perfect-info ref -> no forecast cost


# ---------------------------------------------------------------------------
# Path / MLflow parsing helpers
# ---------------------------------------------------------------------------

def _scan_path(csv, root):
    """Infer (arm, horizon, forecaster, seed, case) from a results path."""
    parts = os.path.relpath(csv, root).split(os.sep)
    arm = parts[0]
    horizon = next((p[1:] for p in parts if re.fullmatch(r"h\d+", p)), None)
    seed = next((p[1:] for p in parts if re.fullmatch(r"s\d+", p)), None)
    forecaster = next((p for p in parts if p in FORECASTERS), "none")
    case = Path(csv).name.replace("_metrics.csv", "")
    return arm, horizon, forecaster, seed, case


def _read_tags(run_dir):
    out = {}
    d = os.path.join(run_dir, "tags")
    if os.path.isdir(d):
        for k in os.listdir(d):
            try:
                out[k] = open(os.path.join(d, k)).read().strip()
            except OSError:
                pass
    return out


def _read_metrics(run_dir):
    """Last logged value of each MLflow metric (file format: '<ts> <value> <step>')."""
    out = {}
    d = os.path.join(run_dir, "metrics")
    if os.path.isdir(d):
        for k in os.listdir(d):
            try:
                out[k] = float(open(os.path.join(d, k)).read().strip().splitlines()[-1].split()[1])
            except (OSError, ValueError, IndexError):
                pass
    return out


# ---------------------------------------------------------------------------
# Quality metrics (all arms, compare.py format)
# ---------------------------------------------------------------------------

def collect_quality(results_root):
    """Tidy quality rows from every compare.py-format CSV under results_root.

    The exact arm's CSV holds AC + DC metrics together; DC-prefixed names go to
    optimizer=dc (prefix stripped so AC/DC share metric names), the rest to ac.
    """
    rows = []
    for csv in glob(os.path.join(results_root, "**", "*_metrics.csv"), recursive=True):
        arm, horizon, forecaster, seed, case = _scan_path(csv, results_root)
        df = pd.read_csv(csv)
        if not {"Metric", "Value"}.issubset(df.columns):
            continue
        for _, r in df.iterrows():
            try:
                val = float(r["Value"])
            except (ValueError, TypeError):
                continue
            name = str(r["Metric"]).strip()
            if arm == "exact":
                optimizer, metric = ("dc", name[3:]) if name.startswith("DC ") else ("ac", name)
            else:
                optimizer, metric = arm, name
            rows.append({"case": case, "horizon": horizon, "optimizer": optimizer,
                         "forecaster": forecaster, "seed": seed, "metric": metric, "value": val})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Timing sources (native locations)
# ---------------------------------------------------------------------------

def collect_forecast_timing(timing_csv):
    """-> {(case, horizon, forecaster): {'infer_s', 'train_min'}} from timing_benchmark.csv."""
    out = {}
    if not timing_csv or not os.path.exists(timing_csv):
        return out
    t = pd.read_csv(timing_csv)
    for _, r in t.iterrows():
        case, h = str(r["case"]), str(int(r["forecast_horizon"]))
        tsteps = float(r.get("test_timesteps", 0) or 0)
        for m in TIMED_FORECASTERS:
            per_scen = r.get(f"{m}_infer_per_scenario")
            if per_scen is None or pd.isna(per_scen):  # CSVs written before the source fix
                it = float(r.get(f"{m}_infer_time", 0) or 0)
                per_scen = it / tsteps if tsteps else 0.0
            out[(case, h, m)] = {"infer_s": float(per_scen),
                                 "train_min": float(r.get(f"{m}_train_time", 0) or 0) / 60.0}
        for m in ZERO_FORECASTERS:
            out[(case, h, m)] = {"infer_s": 0.0, "train_min": 0.0}
    return out


def collect_datakit_timing(datakit_root, network):
    """-> {(horizon, optimizer, forecaster): {'per_dispatch_s', 'total_min'}}.

    Layout: <root>/.../<forecaster>/<network>/raw/runtime_data.parquet (partitioned),
    per-scenario 'ac'/'dc' columns; horizon parsed from the dir name.
    """
    out = {}
    if not datakit_root:
        return out
    for p in glob(os.path.join(datakit_root, "**", network, "raw", "runtime_data.parquet"), recursive=True):
        h = re.search(r"horizon_(\d+)", p)
        if not h:
            continue
        horizon, forecaster = h.group(1), Path(p).parents[2].name
        df = pd.read_parquet(p)
        for opt in ("ac", "dc"):
            if opt not in df.columns:
                continue
            v = df[opt].dropna()
            if len(v):
                out[(horizon, opt, forecaster)] = {"per_dispatch_s": float(v.mean()),
                                                   "total_min": float(v.sum()) / 60.0}
    return out


def collect_mlflow_timing(mlruns_dir):
    """-> (surrogate, e2e) dicts of forward + train timing, matched by tags.

    Surrogate timing is forecaster- and horizon-independent (one model, one forward),
    so it is keyed by (case, seed) and broadcast across the grid downstream.
    """
    surrogate, e2e = {}, {}
    if not mlruns_dir:
        return surrogate, e2e
    for run in glob(os.path.join(mlruns_dir, "*", "*", "")):
        tags = _read_tags(run)
        optimizer = tags.get("optimizer") or tags.get("approach")
        if optimizer not in ("surrogate", "e2e"):
            continue
        m = _read_metrics(run)
        rec = {"per_dispatch_s": m.get("infer_time_per_sample_s"),
               "total_min": (m.get("infer_time_total_s", 0) or 0) / 60.0,
               "train_min": m.get("train_time_min")}
        if optimizer == "surrogate":
            surrogate[(tags.get("case"), tags.get("seed"))] = rec
        else:
            e2e[(tags.get("case"), tags.get("horizon"), tags.get("seed"))] = rec
    return surrogate, e2e


# ---------------------------------------------------------------------------
# Combine timing onto the quality grid
# ---------------------------------------------------------------------------

def build_timing_rows(quality, forecast_t, datakit_t, surrogate_t, e2e_t):
    """Emit timing metrics for each (case,horizon,optimizer,forecaster,seed) in the grid."""
    rows = []
    keys = quality[["case", "horizon", "optimizer", "forecaster", "seed"]].drop_duplicates()
    for _, k in keys.iterrows():
        case, h, opt, fc, seed = (k["case"], k["horizon"], k["optimizer"], k["forecaster"], k["seed"])

        opf_s = opf_total = None
        model_train = 0.0
        if opt in ("ac", "dc"):
            d = datakit_t.get((h, opt, fc))
            if d:
                opf_s, opf_total = d["per_dispatch_s"], d["total_min"]
        elif opt == "surrogate":
            r = surrogate_t.get((case, seed))  # broadcast across forecaster x horizon
            if r:
                opf_s, opf_total, model_train = r["per_dispatch_s"], r["total_min"], (r["train_min"] or 0.0)
        elif opt == "e2e":
            r = e2e_t.get((case, h, seed))
            if r:
                opf_s, opf_total, model_train = r["per_dispatch_s"], r["total_min"], (r["train_min"] or 0.0)

        ft = forecast_t.get((case, h, fc), {"infer_s": 0.0, "train_min": 0.0})
        fc_infer = 0.0 if opt == "e2e" else ft["infer_s"]
        fc_train = 0.0 if opt == "e2e" else ft["train_min"]

        def emit(metric, val):
            if val is not None:
                rows.append({"case": case, "horizon": h, "optimizer": opt, "forecaster": fc,
                             "seed": seed, "metric": metric, "value": val})

        emit("opf_step_infer_s", opf_s)
        if opf_s is not None:
            emit("full_pipeline_infer_s", opf_s + fc_infer)
        emit("infer_total_min", opf_total)
        emit("train_time_min", fc_train + model_train)
    return pd.DataFrame(rows)


def reduce_over_seed(tidy):
    """mean +/- std over seed; deterministic arms (seed=None) pass through as n=1."""
    keys = ["case", "horizon", "optimizer", "forecaster", "metric"]
    out = tidy.groupby(keys, dropna=False)["value"].agg(["mean", "std", "count"]).reset_index()
    out["std"] = out["std"].fillna(0.0)
    return out


def comparison_table(reduced):
    """Headline table: average the per-horizon means over horizon, per arm.

    One row per (case, optimizer, forecaster, metric) = the average-over-horizons value
    (E2E already carries one averaged row; the 2-step arms average over all lead times).
    The per-horizon detail stays in the tidy CSV for plots.
    """
    keys = ["case", "optimizer", "forecaster", "metric"]
    out = reduced.groupby(keys, dropna=False)["mean"].mean().reset_index()
    return out.rename(columns={"mean": "value"})


def main(argv=None):
    p = argparse.ArgumentParser(description="Aggregate Phase 1c results across arms.")
    p.add_argument("--results-root", type=Path, required=True,
                   help="<root>/<arm>/h<H>/<forecaster>/[s<S>]/<case>_metrics.csv")
    p.add_argument("--mlruns-dir", type=Path, default=None, help="MLflow store for surrogate/E2E timing")
    p.add_argument("--timing-csv", type=Path, default=None, help="forecaster timing_benchmark.csv")
    p.add_argument("--datakit-out-dir", type=Path, default=None, help="datakit OPF output base")
    p.add_argument("--network", default=None, help="datakit network dir name, e.g. case118_ieee")
    p.add_argument("--out", type=Path, required=True, help="output tidy CSV")
    args = p.parse_args(argv)

    quality = collect_quality(str(args.results_root))
    if quality.empty:
        raise SystemExit(f"No *_metrics.csv found under {args.results_root}")

    forecast_t = collect_forecast_timing(str(args.timing_csv) if args.timing_csv else None)
    datakit_t = collect_datakit_timing(str(args.datakit_out_dir) if args.datakit_out_dir else None, args.network)
    surrogate_t, e2e_t = collect_mlflow_timing(str(args.mlruns_dir) if args.mlruns_dir else None)
    timing = build_timing_rows(quality, forecast_t, datakit_t, surrogate_t, e2e_t)

    tidy = pd.concat([quality, timing], ignore_index=True)
    reduced = reduce_over_seed(tidy)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    reduced.to_csv(args.out, index=False)
    # Headline comparison table: averaged over horizon (sibling file).
    table = comparison_table(reduced)
    table_path = args.out.with_name(args.out.stem + "_table.csv")
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(reduced)} tidy rows -> {args.out}")
    print(f"Wrote {len(table)} averaged-over-horizon rows -> {table_path}")

    gap = table[table["metric"].str.contains("optimality gap", case=False, na=False)
                & table["metric"].str.contains("Mean", na=False)]
    if not gap.empty:
        print("\nMean optimality gap (%), averaged over horizon, by (optimizer, forecaster):")
        for _, r in gap.iterrows():
            print(f"  {r['optimizer']:9s} {str(r['forecaster']):7s}  {r['value']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
