"""Rebalanced optimality gap from the ex-post AC-PF (expost_pf.py) -> tidy CSV (combined schema).

Raw gap (C(P̂) - C(P*)) / C(P*) is biased by the energy VOLUME the forecast misses: under-supply looks
cheap, over-supply expensive. The ex-post AC-PF closes the imbalance on the true load (distributed slack ∝ Pmax,
losses included); its PF-solved dispatch P̃ is the rebalanced dispatch, re-costed:
- rebalanced gap γ_reb = (C(P̃) - C(P*)) / C(P*);  rebalancing cost R = (C(P̃) - C(P̂)) / C(P*)
- gaps (raw + rebalanced) reported as mean |.| and signed mean (< 0 = cheaper than optimum);
  R split into under-supply (R > 0) and over-supply (R < 0) contributions to its mean
- γ_reb, R over converged samples, incl. insufficient headroom (remainder on the REF unit beyond its limit, priced on
  its cost curve = lower bound)
- Q3: thermal overload share over converged; insufficient headroom (remainder > TOL_P on the slack) and
  diverged (no solution = infeasible) / no_dispatch counted per arm, case, horizon
- ΔP = ΣP̂ - ΣP* (MW, < 0 = short)
- arms: exact -> "ac", surrogate, e2e (seed per file); DC 2-step has no ex-post AC-PF -> not covered
Gate: the raw |gap| recomputed from the ex-post costs must reproduce the logged Mean Optimality Gap
(exact: combined CSV; ML: the run's metrics CSV).
"""
import argparse
from pathlib import Path

import pandas as pd

KEY = ["case", "horizon", "optimizer", "forecaster"]
OPT = {"exact": "ac", "surrogate": "surrogate", "e2e": "e2e"}
EXPOST = Path("/data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1c/expost_pf")


def gap_metrics(d):
    """d: one ex-post AC-PF output (one row per sample) -> metric dict"""
    pct = lambda x: x / d.cost_true * 100
    c = d[d.status == "converged"]
    disp = d[d.status != "no_dispatch"]  # PF attempted
    raw = pct(d.cost_pred - d.cost_true).dropna()  # no_dispatch rows carry no cost
    reb, r = pct(c.cost_pf - c.cost_true), pct(c.cost_pf - c.cost_pred)
    dp = (d.pg_pred_mw - d.pg_true_mw).dropna()
    return {
        "Optimality Gap (recomputed)": raw.abs().mean(),
        "Signed Optimality Gap": raw.mean(),
        "Rebalanced Gap": reb.abs().mean(),
        "Signed Rebalanced Gap": reb.mean(),
        "Thermal Overload (%)": (c.thermal_n > 0).mean() * 100,
        "Insufficient Headroom (%)": c.slack_mismatch_flag.mean() * 100,
        "Insufficient Headroom (n)": int(c.slack_mismatch_flag.sum()),
        "Pred. Pg out of limits (%)": (disp.pg_pred_out_n > 0).mean() * 100,
        "Diverged (n)": int((d.status == "diverged").sum()),
        "No Dispatch (n)": int((d.status == "no_dispatch").sum()),
        "Rebalancing Cost (under-supply)": r.clip(lower=0).mean(),
        "Rebalancing Cost (over-supply)": r.clip(upper=0).mean(),
        "Mean dPg (MW)": dp.mean(),
        "Mean |dPg| (MW)": dp.abs().mean(),
        "Mean |dPg| (%)": (dp.abs() / d.pg_true_mw * 100).mean(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", type=Path, default=Path("exp1/results/combined/case118_combined.csv"))
    ap.add_argument("--expost-dir", type=Path, default=EXPOST)
    ap.add_argument("--cases", nargs="+", default=["case118_ieee"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--out", type=Path, default=Path("exp1/results/combined/corrected_gap.csv"))
    args = ap.parse_args()

    comb = pd.read_csv(args.combined)
    logged = comb[comb["metric"] == "Mean Optimality Gap"].set_index(KEY)["mean"]
    out = []
    for case in args.cases:
        for p in sorted(args.expost_dir.glob(f"{case}/h*/*.parquet")):
            d = pd.read_parquet(p)
            arm, fc, seed = d.arm.iloc[0], d.forecaster.iloc[0] or "none", d.seed.iloc[0]
            if arm not in OPT or fc == "true" or "_limit" in p.stem or (pd.notna(seed) and int(seed) not in args.seeds):
                continue
            key = (case, int(d.horizon.iloc[0]), OPT[arm], fc)
            if arm == "exact":
                ref = logged.get(key)
            else:
                m = pd.read_csv(Path(d.run.iloc[0]) / "artifacts" / "test" / f"{case}_metrics.csv")
                ref = float(m.loc[m["Metric"] == "Mean Optimality Gap", "Value"].iloc[0])
            if ref is None:  # arm absent from the combined CSV
                continue
            met = gap_metrics(d)
            if abs(met["Optimality Gap (recomputed)"] - ref) > 1e-3:
                raise ValueError(f"{p}: recomputed gap {met['Optimality Gap (recomputed)']:.4f} != logged {ref:.4f}")
            out += [dict(zip(KEY, key), metric=k, mean=v) for k, v in met.items()]
    out = pd.DataFrame(out).groupby(KEY + ["metric"], as_index=False)["mean"].agg(["mean", "count"])  # over seeds
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({out.groupby(KEY).ngroups} arms, all reproduce the logged gap)")


if __name__ == "__main__":
    main()
