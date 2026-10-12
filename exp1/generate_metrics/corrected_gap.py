"""Rebalanced optimality gap, exact per scenario -> tidy CSV (combined schema).

Raw gap (C(Pg_pred) - C(Pg*)) / C(Pg*) is biased by the energy VOLUME the forecast misses: under-supply looks
cheap, over-supply expensive. Rebalance each scenario to the optimum's total generation, then re-cost:
- dP_s = sum Pg_pred - sum Pg*  (MW; < 0 = short vs the optimum, which meets realized demand + losses)
- merit order (metrics.merit_order_rebalance): shortfall to the cheapest units up to Pmax, surplus off the most
  expensive down to Pmin (ranked by c1 + 2 c2 Pg_pred); remainder -> slack, counted as violation
- rebalanced gap = (C(Pg_reb) - C(Pg*)) / C(Pg*);  rebalancing cost R_s = (C(Pg_reb) - C(Pg_pred)) / C(Pg*)
- gaps reported as mean |.| (as the raw gap), R_s signed; each arm on its own test set
- same rebalancing as the ex-post AC-PF pre-step (expost_pf.py)

Sources (per-generator Pg, pred + optimum, same scenario):
- AC/DC 2-step: datakit gen_data (forecaster arm p_mw / p_mw_dc vs `true` arm p_mw)
- surrogate/E2E: <case>_state_gen.parquet dumps; newest run per (identity, seed) wins (retrains supersede old runs)
Gate: the raw |gap| recomputed here must reproduce the logged Mean Optimality Gap.
"""
import argparse
import os
from functools import lru_cache
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from aggregate_results import _ml_identity, _read_tags
from metrics import merit_order_rebalance

KEY = ["case", "horizon", "optimizer", "forecaster"]
COEF = ["min_p_mw", "max_p_mw", "cp0_eur", "cp1_eur_per_mw", "cp2_eur_per_mw2", "is_slack_gen"]
STATE_DUMP = "_state_gen.parquet"
TOL = 0.1  # MW; slack remainder counted as violation (= expost_pf.TOL_GEN)


def gen_table(data_out, case, h, arm, cols):
    g = pd.read_parquet(Path(data_out) / f"{case}_horizon_{h}" / arm / case / "raw" / "gen_data.parquet",
                        columns=["load_scenario_idx", "idx", "in_service", *cols])
    return g[g["in_service"] == 1].astype({"idx": int})


@lru_cache
def coefs(data_out, case, h):
    """Static per-gen limits/costs (identical across scenarios), indexed by gen idx."""
    return gen_table(data_out, case, h, "true", COEF).groupby("idx")[COEF].first()


def gap_metrics(d, coef):
    """d: [scenario, gen, pg_pred, pg_true] (MW), gen = datakit idx; coef: static per-gen limits/costs -> metric dict"""
    d = d.sort_values(["scenario", "gen"])
    S, G = d["scenario"].nunique(), len(coef)
    assert len(d) == S * G and (d["gen"].to_numpy()[:G] == coef.index).all(), "incomplete scenario x gen grid"
    pg, pt = (d[c].to_numpy().reshape(S, G) for c in ("pg_pred", "pg_true"))
    pmin, pmax, c0, c1, c2, slack = (coef[c].to_numpy() for c in COEF)
    cost = lambda p: (c0 + c1 * p + c2 * p ** 2).sum(1)
    delta = pt.sum(1) - pg.sum(1)
    pr, rest = merit_order_rebalance(pg, delta, pmin, pmax, c1, c2, slack == 1)
    assert np.allclose(pr.sum(1), pt.sum(1)), "rebalanced total != optimum total"
    cp, cr, ct = cost(pg), cost(pr), cost(pt)
    pct = lambda x: x / ct * 100
    return {
        "Optimality Gap (recomputed)": np.abs(pct(cp - ct)).mean(),
        "Rebalanced Gap": np.abs(pct(cr - ct)).mean(),
        "Rebalancing Cost": pct(cr - cp).mean(),
        "Mean dPg (MW)": -delta.mean(),
        "Mean |dPg| (MW)": np.abs(delta).mean(),
        "Mean |dPg| (%)": (np.abs(delta) / pt.sum(1) * 100).mean(),
        "Slack Overflow (%)": (rest > TOL).mean() * 100,
        "Slack Overflow (n)": int((rest > TOL).sum()),
    }


def two_step(data_out, case, h, fc):
    """-> {optimizer: metrics} for the AC/DC arms of one forecaster."""
    pred = gen_table(data_out, case, h, fc, ["p_mw", "p_mw_dc"])
    true = gen_table(data_out, case, h, "true", ["p_mw"])
    m = pred.merge(true, on=["load_scenario_idx", "idx"], suffixes=("", "_true"))
    return {opt: gap_metrics(pd.DataFrame({"scenario": m["load_scenario_idx"], "gen": m["idx"],
                                           "pg_pred": m[col], "pg_true": m["p_mw_true"]}),
                               coefs(data_out, case, h))
            for opt, col in (("ac", "p_mw"), ("dc", "p_mw_dc"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", type=Path, default=Path("exp1/results/combined/case118_combined.csv"))
    ap.add_argument("--mlruns-dir", nargs="+",
                    default=[f"/data/horse/ws/tibo990i-thesis_data/mlruns/{e}"
                             for e in ("760726881304984634", "375695290152400764")],  # state-dump, final_thesis_runs
                    help="experiment(s) holding the *_state_gen.parquet dumps")
    ap.add_argument("--seeds", nargs="+", default=["0"])
    ap.add_argument("--data-out", default="/data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1c/data_out")
    ap.add_argument("--cases", nargs="+", default=["case118_ieee"])
    ap.add_argument("--out", type=Path, default=Path("exp1/results/combined/corrected_gap.csv"))
    args = ap.parse_args()

    comb = pd.read_csv(args.combined)
    logged = comb[comb["metric"] == "Mean Optimality Gap"].set_index(KEY)["mean"]
    rows = []  # (key, metrics, logged gap of the same run)

    for case in args.cases:
        for base in sorted(glob(f"{args.data_out}/{case}_horizon_*")):
            h = int(base.rsplit("_", 1)[1])
            for fc in ("sarima", "snaive", "xgb"):
                if Path(base, fc).exists():
                    for opt, met in two_step(args.data_out, case, h, fc).items():
                        key = (case, h, opt, fc)
                        rows.append((key, met, logged.get(key)))

    newest = {}  # (case, h, opt, fc, seed) -> (dump, run)
    for root in args.mlruns_dir:
        for p in glob(f"{root}/*/artifacts/test/*{STATE_DUMP}"):
            case, run = Path(p).name.replace(STATE_DUMP, ""), Path(p).parents[2]
            opt, fc, h, seed = _ml_identity(_read_tags(run).get("mlflow.runName", ""))
            k = (case, int(h or 0), opt, fc, seed)
            if opt and case in args.cases and seed in args.seeds and (
                    k not in newest or os.path.getmtime(run) > os.path.getmtime(newest[k][1])):
                newest[k] = (p, run)
    for (case, h, opt, fc, _), (p, run) in newest.items():
        m = pd.read_csv(run / "artifacts" / "test" / f"{case}_metrics.csv")
        coef, d = coefs(args.data_out, case, h), pd.read_parquet(p)
        d["gen"] = coef.index[d["gen"]]  # dump numbers in-service gens 0..N-1 -> datakit idx
        rows.append(((case, h, opt, fc), gap_metrics(d, coef),
                     float(m.loc[m["Metric"] == "Mean Optimality Gap", "Value"].iloc[0])))

    out = []
    for key, met, ref in rows:
        if ref is None:  # arm absent from the combined CSV
            continue
        if abs(met["Optimality Gap (recomputed)"] - ref) > 1e-3:
            raise ValueError(f"{key}: recomputed gap {met['Optimality Gap (recomputed)']:.4f} != logged {ref:.4f}")
        out += [dict(zip(KEY, key), metric=k, mean=v) for k, v in met.items()]
    out = pd.DataFrame(out).groupby(KEY + ["metric"], as_index=False)["mean"].agg(["mean", "count"])  # over seeds
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(rows)} arms, all reproduce the logged gap)")


if __name__ == "__main__":
    main()
