"""
Aggregate Phase 1c results across all arms into one comparison table.

Tidy long schema: case, horizon, optimizer, forecaster, seed, metric, value
  optimizer  in {ac, dc, surrogate, e2e}
  forecaster in {xgb, sarima, snaive, true, none}   (none = E2E)

Each arm is read from its NATIVE location (no unified copy tree):
  - exact + 2-step (ac/dc)  <- compare.py output  results/<run>/<forecaster>/<case>_metrics.csv
                               (one CSV holds AC + DC together; prefixes split the two optimizers,
                                horizon from <run> name, forecaster from the sub-dir, seed=None)
  - surrogate / e2e         <- graphkit eval  <experiment_dir>/*/artifacts/test/<case>_metrics.csv
                               (--mlruns-dir points straight at the mlflow EXPERIMENT dir(s)
                                <mlruns_root>/<experiment_id>, whose immediate children are runs;
                                identity from the run NAME, no tags:
                                {arm}_{case}_{forecaster}_h{H}_s{S} / base_e2e_{case}_h{H}_s{S})
Only canonical-schema CSVs are ingested (marker metric "Mean Optimality Gap"); older-format
files are skipped so mixed-vintage stores can be pointed at directly.

Timing stays in each producer's native location and is combined onto the quality grid:
  - forecast infer/train          <- timing_benchmark.csv
  - AC/DC solve                   <- datakit runtime_data.parquet (per-scenario 'ac' and 'dc' cols)
  - surrogate/E2E forward + train <- rows inside the eval metrics.csv
Emitted per (case,horizon,optimizer,forecaster,seed): opf_step_infer_s, opf_step_infer_total_min,
full_pipeline_infer_s, full_pipeline_infer_total_min, train_time_min.
All per-sample values are per scenario (= one dispatch decision); totals are over the same scenarios.

Then reduce over seed -> mean for the stochastic ML arms.
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
CANON_MARKER = "Mean Optimality Gap"   # present in every current-schema CSV; absent in old ones

# timing rows carried inside an ML eval CSV (kept out of the quality metrics)
_INFER_PER, _INFER_TOT, _TRAIN_TOT = (
    "Inference time model-only per sample (s)",
    "Inference time model-only (s)",
    "Training time model-only (s)",
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _horizon(name):
    """Horizon integer from a dir name like case118_ieee_horizon24_3yr / horizon_24."""
    m = re.search(r"horizon_?(\d+)", name)
    return m.group(1) if m else None


def _case(csv):
    return Path(csv).name.replace("_metrics.csv", "")


def _long(csv):
    """compare-format CSV -> long df [metric, value] (numeric), or None if not canonical."""
    df = pd.read_csv(csv)
    if not {"Metric", "Value"}.issubset(df.columns):
        return None
    out = pd.DataFrame({"metric": df["Metric"].astype(str).str.strip(),
                        "value": pd.to_numeric(df["Value"], errors="coerce")})
    # marker is bare in ML eval CSVs but AC/DC-prefixed in compare.py CSVs (stripped later);
    # test both forms so 2-step files aren't mistaken for old-schema and dropped.
    if CANON_MARKER not in set(out["metric"].str.replace(r"^(AC|DC) ", "", regex=True)):
        return None
    return out.dropna(subset=["value"]).reset_index(drop=True)


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


# ---------------------------------------------------------------------------
# Quality: exact + 2-step (compare.py output)
# ---------------------------------------------------------------------------

def collect_exact(results_root):
    """Tidy ac/dc rows from every canonical compare.py CSV under results_root.

    Layout <results_root>/<run>/<forecaster>/<case>_metrics.csv. One CSV holds AC + DC;
    the 'AC '/'DC ' prefix picks the optimizer (and is stripped so both share metric names),
    un-prefixed metrics (forecast error, residuals) belong to the AC-OPF arm.
    """
    frames = []
    for csv in glob(os.path.join(results_root, "**", "*_metrics.csv"), recursive=True):
        # leading-underscore dirs (e.g. _archiv) hold superseded runs - excluded by convention.
        if any(part.startswith("_") for part in Path(csv).relative_to(results_root).parts):
            continue
        forecaster, run = Path(csv).parent.name, Path(csv).parent.parent.name
        if forecaster not in FORECASTERS:
            continue
        long = _long(csv)
        if long is None:
            continue
        long["optimizer"] = "ac"
        long.loc[long["metric"].str.startswith("DC "), "optimizer"] = "dc"
        long["metric"] = long["metric"].str.replace(r"^(AC|DC) ", "", regex=True)
        frames.append(long.assign(case=_case(csv), horizon=_horizon(run),
                                  forecaster=forecaster, seed=None))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Quality + timing: surrogate / e2e (graphkit eval CSVs, identity from run name)
# ---------------------------------------------------------------------------

def _ml_identity(run_name):
    """(optimizer, forecaster, horizon, seed) parsed from the run name (no tags).

    surrogate_case118_ieee_xgb_h24_s0: h<H>/s<S> tokens give horizon/seed, and the token
    before h<H> is the forecaster. E2E runs (final_thesis_runs experiment) carry a "base_"
    prefix ahead of the arm token: base_e2e_case118_h24_s0.

    Anchored on the first and last token (arm first, s<S> last) so variant runs that merely
    contain the right keywords - e2e_case118_h24_s0 (missing the base_ prefix),
    e2e_case118_h24_s0_tcn_linear (extra trailing tokens) - are rejected instead of silently
    merged into the real seed grid.
    """
    toks = run_name.split("_")
    if len(toks) >= 3 and toks[0] == "base" and toks[1] == "e2e" and re.fullmatch(r"s\d+", toks[-1]):
        optimizer, toks = "e2e", toks[1:]
    elif len(toks) >= 2 and toks[0] in ("surrogate", "twostep") and re.fullmatch(r"s\d+", toks[-1]):
        optimizer = "surrogate"
    else:
        return None, None, None, None
    horizon = next((t[1:] for t in toks if re.fullmatch(r"h\d+", t)), None)
    seed = toks[-1][1:]
    if optimizer == "e2e":
        forecaster = "none"
    else:
        i = next((j for j, t in enumerate(toks) if re.fullmatch(r"h\d+", t)), 0)
        forecaster = toks[i - 1] if i > 0 else "none"
    return optimizer, forecaster, horizon, seed


def collect_ml(experiment_dirs):
    """-> (quality_df, ml_timing) for surrogate/e2e runs across one or more mlflow experiments.

    Each dir is an mlflow EXPERIMENT dir (<mlruns_root>/<experiment_id>) whose immediate
    children are runs - the caller picks the experiment(s), the script does not search for
    them. ml_timing[(case, horizon, optimizer, forecaster, seed)] = {opf_s, total_min,
    train_min}, read straight from the eval CSV's timing rows.
    """
    frames, timing = [], {}
    for root in experiment_dirs:
        for csv in glob(os.path.join(root, "*", "artifacts", "test", "*_metrics.csv")):
            run_dir = Path(csv).parents[2]
            run_name = _read_tags(run_dir).get("mlflow.runName", "")
            opt, fc, h, seed = _ml_identity(run_name)
            if opt not in ("surrogate", "e2e"):
                continue
            long = _long(csv)
            if long is None:
                continue
            case = _case(csv)
            frames.append(long[~long["metric"].isin((_INFER_PER, _INFER_TOT, _TRAIN_TOT))].assign(
                case=case, horizon=h, optimizer=opt, forecaster=fc, seed=seed))
            v = dict(zip(long["metric"], long["value"]))
            timing[(case, h, opt, fc, seed)] = {
                "opf_s": v.get(_INFER_PER),
                "total_min": v[_INFER_TOT] / 60.0 if _INFER_TOT in v else None,
                "train_min": v.get(_TRAIN_TOT, 0.0) / 60.0,
            }
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), timing


# ---------------------------------------------------------------------------
# Timing sources for the exact / 2-step arm
# ---------------------------------------------------------------------------

def collect_forecast_timing(timing_csv):
    """-> {(case, horizon, forecaster): {'infer_s', 'train_min'}} from timing_benchmark.csv."""
    out = {}
    if not timing_csv or not os.path.exists(timing_csv):
        return out
    for _, r in pd.read_csv(timing_csv).iterrows():
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


def collect_datakit_solve(datakit_root):
    """-> {(case, horizon, forecaster, optimizer): {'per_dispatch_s', 'total_min'}} from
    runtime_data.parquet. optimizer in {'ac', 'dc'}: both solves are timed per scenario in
    their own columns ('ac', 'dc').

    Layout <root>/<run>/<forecaster>/<network>/raw/runtime_data.parquet;
    horizon from <run>, case from <network> (same string as the quality-CSV case).
    """
    out = {}
    if not datakit_root:
        return out
    # fixed depth: a recursive glob would descend into the partitioned parquet dirs and blow up.
    for p in glob(os.path.join(datakit_root, "*", "*", "*", "raw", "runtime_data.parquet")):
        case, forecaster, horizon = (Path(p).parents[1].name, Path(p).parents[2].name,
                                     _horizon(Path(p).parents[3].name))
        if forecaster not in FORECASTERS or horizon is None:
            continue
        df = pd.read_parquet(p)
        for opt in ("ac", "dc"):
            if opt not in df.columns:
                continue
            v = df[opt].dropna()
            if len(v):
                out[(case, horizon, forecaster, opt)] = {"per_dispatch_s": float(v.mean()),
                                                         "total_min": float(v.sum()) / 60.0}
    return out


# ---------------------------------------------------------------------------
# Combine timing onto the quality grid
# ---------------------------------------------------------------------------

def build_timing_rows(quality, forecast_t, datakit_t, ml_timing):
    """Emit timing metrics for each (case,horizon,optimizer,forecaster,seed) in the grid."""
    rows = []
    keys = quality[["case", "horizon", "optimizer", "forecaster", "seed"]].drop_duplicates()
    for _, k in keys.iterrows():
        case, h, opt, fc, seed = (k["case"], k["horizon"], k["optimizer"], k["forecaster"], k["seed"])

        opf_s = opf_total = None
        model_train = 0.0
        if opt in ("ac", "dc"):
            d = datakit_t.get((case, h, fc, opt))
            if d:
                opf_s, opf_total = d["per_dispatch_s"], d["total_min"]
        elif opt in ("surrogate", "e2e"):
            r = ml_timing.get((case, h, opt, fc, seed))
            if r:
                opf_s, opf_total, model_train = r["opf_s"], r["total_min"], (r["train_min"] or 0.0)

        ft = forecast_t.get((case, h, fc), {"infer_s": 0.0, "train_min": 0.0})
        fc_infer = 0.0 if opt == "e2e" else ft["infer_s"]
        fc_train = 0.0 if opt == "e2e" else ft["train_min"]

        def emit(metric, val):
            if val is not None:
                rows.append({"case": case, "horizon": h, "optimizer": opt, "forecaster": fc,
                             "seed": seed, "metric": metric, "value": val})

        emit("opf_step_infer_s", opf_s)
        emit("opf_step_infer_total_min", opf_total)
        if opf_s is not None:
            full_s = opf_s + fc_infer
            emit("full_pipeline_infer_s", full_s)
            if opf_total is not None:
                # forecast cost over the same n = opf_total / opf_s dispatches as the OPF step
                emit("full_pipeline_infer_total_min", opf_total * full_s / opf_s)
        emit("train_time_min", fc_train + model_train)
    return pd.DataFrame(rows)


def reduce_over_seed(tidy):
    """mean over seed; deterministic arms (seed=None) pass through as n=1."""
    keys = ["case", "horizon", "optimizer", "forecaster", "metric"]
    return tidy.groupby(keys, dropna=False)["value"].agg(["mean", "count"]).reset_index()


def main(argv=None):
    p = argparse.ArgumentParser(description="Aggregate Phase 1c results across arms.")
    p.add_argument("--results-root", type=Path, required=True,
                   help="compare.py output base: <root>/<run>/<forecaster>/<case>_metrics.csv")
    p.add_argument("--mlruns-dir", type=Path, nargs="*", default=[],
                   help="one or more mlflow EXPERIMENT dirs (<mlruns_root>/<experiment_id>) whose "
                        "children are surrogate/E2E eval runs; used as-is, not searched")
    p.add_argument("--timing-csv", type=Path, default=None, help="forecaster timing_benchmark.csv")
    p.add_argument("--datakit-out-dir", type=Path, default=None, help="phase_1c datakit output base")
    p.add_argument("--out", type=Path, required=True, help="output tidy CSV")
    args = p.parse_args(argv)

    ml_quality, ml_timing = collect_ml([str(d) for d in args.mlruns_dir])
    quality = pd.concat([collect_exact(str(args.results_root)), ml_quality], ignore_index=True)
    if quality.empty:
        raise SystemExit(f"No canonical *_metrics.csv found under {args.results_root} or the mlruns stores")

    forecast_t = collect_forecast_timing(str(args.timing_csv) if args.timing_csv else None)
    datakit_t = collect_datakit_solve(str(args.datakit_out_dir) if args.datakit_out_dir else None)
    timing = build_timing_rows(quality, forecast_t, datakit_t, ml_timing)

    reduced = reduce_over_seed(pd.concat([quality, timing], ignore_index=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    reduced.to_csv(args.out, index=False)

    print(f"Arms found (optimizer, forecaster): {sorted(set(zip(quality.optimizer, quality.forecaster)))}")
    print(f"Wrote {len(reduced)} tidy rows -> {args.out}")

    gap = reduced[reduced["metric"] == "Mean Optimality Gap"].sort_values(
        ["case", "horizon", "optimizer", "forecaster"])
    if not gap.empty:
        print("\nMean optimality gap (%) by (case, horizon, optimizer, forecaster):")
        for _, r in gap.iterrows():
            print(f"  {str(r['case']):14s} h{str(r['horizon']):4s} {r['optimizer']:9s} "
                  f"{str(r['forecaster']):7s}  {r['mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
