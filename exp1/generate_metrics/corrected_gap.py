"""Rebalanced optimality gap from the ex-post AC-PF (expost_pf.py) -> tidy CSV (combined schema).

Raw gap (C(P̂) - C(P*)) / C(P*) is biased by the energy VOLUME the forecast misses: under-supply looks
cheap, over-supply expensive. The ex-post AC-PF closes the imbalance on the true load (distributed slack ∝ Pmax,
losses included); its PF-solved dispatch P̃ is the rebalanced dispatch, re-costed:
- rebalanced gap γ_reb = (C(P̃) - C(P*)) / C(P*);  rebalancing cost R = (C(P̃) - C(P̂)) / C(P*)
- gaps (raw + rebalanced) reported as mean |.| and signed mean (< 0 = cheaper than optimum);
  R split into under-supply (R > 0) and over-supply (R < 0) contributions to its mean
- no-refund gap γ_noref = (max(C(P̃), C(P̂)) - C(P*)) / C(P*): ramp-down earns nothing back (pay the committed
  dispatch on over-supply); per-sample max == per-unit max since distributed slack moves all units one way
- γ_reb, R over converged samples, incl. insufficient headroom (remainder on the REF unit beyond its limit, priced on
  its cost curve = lower bound)
- Q3: thermal overload share over converged; insufficient headroom (remainder > TOL_P on the slack) and
  diverged (no solution = infeasible) / no_dispatch counted per arm, case, horizon
- Q3 per constraint group (thermal, PQ voltage, Qg), over converged samples, element sets as in expost_pf
  (in-service lines, PQ buses, non-REF gen buses): share of samples / of elements with a violation, worst violation
  per violating sample (mean), violation per violating element (mean; needs the *_sum columns, expost_pf ≥ 2026-11-01)
  - thermal magnitudes in % of the rating above 100 %
- ΔP = ΣP̂ - ΣP* (MW, < 0 = short)
- arms: exact -> "ac", dc (DC-OPF dispatch, case-file generator voltage setpoints), surrogate, e2e (seed per file)
Gate: the raw |gap| recomputed from the ex-post costs must reproduce the logged Mean Optimality Gap
(exact/dc: combined CSV; ML: the run's metrics CSV).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from gridfm_datakit.network import load_net_from_pglib
from gridfm_datakit.utils.idx_bus import BUS_TYPE, PQ
from gridfm_datakit.utils.idx_gen import GEN_BUS

KEY = ["case", "horizon", "optimizer", "forecaster"]
OPT = {"exact": "ac", "dc": "dc", "surrogate": "surrogate", "e2e": "e2e"}
EXPOST = Path("/data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1c/expost_pf")


def element_counts(case):
    """size of the element sets expost_pf checks: in-service lines, PQ buses, non-REF gen buses"""
    net = load_net_from_pglib(case)
    gb = np.unique(net.gens[net.idx_gens_in_service, GEN_BUS].astype(int))
    return dict(thermal=len(net.idx_branches_in_service), vm=int((net.buses[:, BUS_TYPE] == PQ).sum()),
                qg=len(gb) - int(net.ref_bus_idx in gb))


def violation_metrics(name, unit, n, worst, total, n_el):
    """one constraint group over converged samples: n = violating elements per sample, worst = largest violation
    per sample, total = violation summed over the violating elements (None for pre-2026-11-01 outputs)"""
    assert (n <= n_el).all(), f"{name}: more violating elements than the set holds"
    bad = n > 0
    out = {f"{name} (%)": bad.mean() * 100,
           f"{name} (% elements)": n.sum() / (len(n) * n_el) * 100,
           f"{name} Worst ({unit})": worst[bad].mean(),
           f"{name} Worst P99 ({unit})": np.percentile(worst.where(bad, 0), 99)}  # all samples, none = 0
    if total is not None:
        out[f"{name} Mean ({unit})"] = total.sum() / n.sum() if n.sum() else np.nan
    return out


def gap_metrics(d, n_el):
    """d: one ex-post AC-PF output (one row per sample), n_el: element_counts -> metric dict"""
    pct = lambda x: x / d.cost_true * 100
    c = d[d.status == "converged"]
    disp = d[d.status != "no_dispatch"]  # PF attempted
    raw = pct(d.cost_pred - d.cost_true).dropna()  # no_dispatch rows carry no cost
    reb, r = pct(c.cost_pf - c.cost_true), pct(c.cost_pf - c.cost_pred)
    dp = (d.pg_pred_mw - d.pg_true_mw).dropna()
    col = lambda name, scale=1: c[name] * scale if name in c else None
    q3 = {**violation_metrics("Thermal Overload", "% of rating", c.thermal_n, (c.thermal_max_loading - 1) * 100,
                              col("thermal_sum", 100), n_el["thermal"]),
          **violation_metrics("Vm Violation", "p.u.", c.vm_n, c.vm_max_viol, col("vm_sum"), n_el["vm"]),
          **violation_metrics("Qg Violation", "MVAr", c.qg_n, c.qg_max_viol_mvar, col("qg_sum_mvar"), n_el["qg"])}
    return {
        "Optimality Gap (recomputed)": raw.abs().mean(),
        "Signed Optimality Gap": raw.mean(),
        "Rebalanced Gap": reb.abs().mean(),
        "Signed Rebalanced Gap": reb.mean(),
        "No-refund Gap": pct(c[["cost_pf", "cost_pred"]].max(axis=1) - c.cost_true).abs().mean(),
        **q3,
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
        n_el = element_counts(case)
        print(f"[{case}] element sets: {n_el}")
        for p in sorted(args.expost_dir.glob(f"{case}/h*/*.parquet")):
            d = pd.read_parquet(p)
            arm, fc, seed = d.arm.iloc[0], d.forecaster.iloc[0] or "none", d.seed.iloc[0]
            if arm not in OPT or fc == "true" or "_limit" in p.stem or (pd.notna(seed) and int(seed) not in args.seeds):
                continue
            key = (case, int(d.horizon.iloc[0]), OPT[arm], fc)
            if arm in ("exact", "dc"):
                ref = logged.get(key)
            else:
                m = pd.read_csv(Path(d.run.iloc[0]) / "artifacts" / "test" / f"{case}_metrics.csv")
                ref = float(m.loc[m["Metric"] == "Mean Optimality Gap", "Value"].iloc[0])
            if ref is None:  # arm absent from the combined CSV
                continue
            met = gap_metrics(d, n_el)
            if abs(met["Optimality Gap (recomputed)"] - ref) > 1e-3:
                raise ValueError(f"{p}: recomputed gap {met['Optimality Gap (recomputed)']:.4f} != logged {ref:.4f}")
            out += [dict(zip(KEY, key), metric=k, mean=v) for k, v in met.items()]
    out = pd.DataFrame(out).groupby(KEY + ["metric"], as_index=False)["mean"].agg(["mean", "count"])  # over seeds
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({out.groupby(KEY).ngroups} arms, all reproduce the logged gap)")


if __name__ == "__main__":
    main()
