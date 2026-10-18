"""
E2E Pd → AC-OPF: does E2E's Pg advantage come from its output target? Same E2E run, Pd head fed through AC-OPF.

What?
    convert  E2E test-dump pd_pred → phase-1b forecast parquet [load_scenario_idx = origin = target − k, bus_id,
             horizon_step, true, e2e_pd] → phase1c chain unchanged (transform → datakit AC-OPF → compare.py)
    proxy    Phase A: plug-in f(Σ pd_pred) + hedged via dispatch_diag.step4 (f = GT dispatch at nearest D) vs E2E Pg
             output + AC-xgb / AC-sarima rows (dispatch_diag step4_jensen.csv); valid on case14 / case500 only
    table    Phase B: dispatch_diag.step1 split (Pg MAE / RMSE / raw gap / |ΣΔPg| / misplaced MW / summed-Pd error) for
             E2E Pg output and E2E Pd → AC-OPF (+ AC-xgb / AC-sarima rows from dispatch_diag step1_split.csv);
             step4 on the real OPF dispatch = Phase A proxy check
Why?
    - same model, same information, same training run; only the output differs
    - (ii) loses most of E2E's edge → output target / objective alignment; keeps it → architecture / information
How?
    - E2E runs pinned via dispatch_diag.E2E_RUNS; dump row → target hour via expost_pf.load_ml
    - zero-load buses keep E2E's pd_pred (main arm "as is"); Σ pd_pred there reported (convert, table)
    - gates: dump pd_true == GT Pd; compare.py MAE Pd (e2e_pd) == E2E logged MAE Pd (rel 1e-3); step1 vs logged Pg
    - out: <DATA>/e2e_pd_opf/{forecasts/<case>_horizon<H>.parquet, phaseA_<case>.csv, phaseB_<case>_h<Hs>.csv,
      phaseB_step4_<case>_h<Hs>.csv}

Usage (Thesis_Repo root, barnard): Phase B = scripts/e2e_pd_opf.sbatch; Phase A:
    sbatch -p barnard -c 8 --mem=128G -t 01:00:00 -o logs/e2e_pd_proxy_%j.out --wrap "module --force purge; \
        module load release/2026 GCC/14.2.0; source ~/thesis_env/bin/activate; for c in case14_ieee case500_goc; \
        do python -u exp1/generate_metrics/e2e_pd_opf.py proxy --case \$c; done"
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import expost_pf
from dispatch_diag import OUT as DIAG, SEP, e2e_run, logged, step1, step4, truth
from expost_pf import DATA, _dense, load_exact, load_ml, load_truth

ARM = "e2e_pd"
OUT = DATA / "e2e_pd_opf"
FC_DIR = OUT / "forecasts"  # mirrors expost_pf.FORECASTS layout: <case>_horizon<H>.parquet
RESULTS = Path(__file__).resolve().parents[2] / "exp1/results/_e2e_pd"  # `_` → aggregate_results skips it
SPLIT_COLS = ["H", "arm", "n_samples", "pg_mae", "pg_rmse", "gap_raw", "total_abs_mw", "misplaced_mw", "dD_abs_mw",
              "dD_bias_mw", "logged_mae", "logged_rmse", "logged_gap"]


def e2e(case, H):
    """pinned E2E run → load_ml dict + pd_pred [S, n_bus] in the same scenario order"""
    run = e2e_run(case, H)
    x = load_ml("e2e", case, H, run)
    b = pq.read_table(run / f"artifacts/test/{case}_state_bus.parquet",
                      columns=["scenario", "bus", "pd_pred", "pd_true"]).to_pandas()
    _, (x["pd_pred"], pd_true) = _dense(b, "scenario", "bus", "pd_pred", "pd_true")
    assert np.array_equal(pd_true, x["pd_true"]), "pd_pred row order != load_ml"
    return run, x


def load_opf(case, H):
    """expost_pf.load_exact on the e2e_pd OPF run; flat order from our forecast parquet (FORECASTS swapped)"""
    old, expost_pf.FORECASTS = expost_pf.FORECASTS, FC_DIR
    try:
        return load_exact(case, H, ARM)
    finally:
        expost_pf.FORECASTS = old


def convert(case, horizons):
    FC_DIR.mkdir(parents=True, exist_ok=True)
    for H in horizons:
        run, x = e2e(case, H)
        hours, t = load_truth(case, x["target"])
        i = np.searchsorted(hours, x["target"])
        assert np.array_equal(hours[i], x["target"]), "target hours missing in phase1a labels"
        dev = np.abs(x["pd_true"] - t["pd"][i]).max()
        zero = (t["pd"] == 0).all(0)
        zs = x["pd_pred"][:, zero].sum(1)
        print(f"[{case} h{H}] {run.parent.name}/{run.name}: {len(i)} samples, max |pd_true − GT| {dev:.2e} MW; "
              f"{zero.sum()} zero-load buses: Σ pd_pred mean {zs.mean():+.2f}, mean |.| {np.abs(zs).mean():.2f} MW")
        assert dev < 1e-2, "E2E pd_true != phase1a GT at the target hour"
        n_s, n_b = x["pd_pred"].shape
        df = pd.DataFrame(dict(load_scenario_idx=np.repeat(x["target"] - x["k"], n_b), bus_id=np.tile(np.arange(n_b), n_s),
                               horizon_step=np.repeat(x["k"], n_b), true=x["pd_true"].ravel(), e2e_pd=x["pd_pred"].ravel()))
        assert not df.duplicated(["load_scenario_idx", "horizon_step", "bus_id"]).any()
        out = FC_DIR / f"{case}_horizon{H}.parquet"
        df.to_parquet(out, index=False)
        print(f"  wrote {out} ({len(df)} rows)")


def proxy(case, horizons):
    gt = truth(case)
    rename = {"ac_mae": "e2e_pg_mae", "ac_rmse": "e2e_pg_rmse", "plug_vs_ac_rmse": "plug_vs_e2e_pg_rmse"}
    rows = []
    for H in horizons:
        _, x = e2e(case, H)
        r = step4(gt, H, ARM, x["pd_pred"].sum(1), x["k"], x["target"], x["pg"])
        rows.append({rename.get(c, c): v for c, v in r.items()})
    ref = pd.read_csv(DIAG / case / "step4_jensen.csv")
    df = pd.concat([pd.DataFrame(rows), ref[ref.H.isin(horizons)]], ignore_index=True).sort_values(["H", "fc"])
    df.to_csv(OUT / f"phaseA_{case}.csv", index=False)
    cols = ["H", "fc", "n_samples", "e2e_pg_mae", "e2e_pg_rmse", "ac_mae", "ac_rmse", "plug_mae", "plug_rmse",
            "hedge_mae", "hedge_rmse", "plug_vs_e2e_pg_rmse", "plug_vs_ac_rmse"]
    with pd.option_context("display.width", 250, "display.float_format", "{:.4g}".format):
        print(f"\n[{case}] Phase A proxy\n{df[cols].to_string(index=False)}")


def table(case, horizons):
    gt = truth(case)
    units = pd.read_csv(DIAG / case / "step2_units.csv")  # unit classes from dispatch_diag step 2
    assert np.array_equal(units.idx.to_numpy(), gt["idx"])
    zero = (gt["pd"] == 0).all(0)
    splits, jensen = [], []
    for H in horizons:
        run, x = e2e(case, H)
        e_pd = x["pd_pred"] - x["pd_true"]
        _, s = step1(gt, units, H, "e2e", x["pg"], x["target"], e_pd.sum(1), logged(case, H, "e2e", run))
        s["pd_zero_bus_sum_mw"] = x["pd_pred"][:, zero].sum(1).mean()
        splits.append(s)

        y = load_opf(case, H)
        kx, ky = x["target"] * SEP + x["k"], y["target"] * SEP + y["k"]
        o = np.argsort(kx)
        j = o[np.searchsorted(kx[o], ky)]
        assert np.array_equal(kx[j], ky), "OPF sample not in the E2E dump"
        m = pd.read_csv(RESULTS / f"{case}_horizon{H}/{ARM}/{case}_metrics.csv").set_index("Metric")["Value"]
        gate = tuple(float(m[f"AC {k}"]) for k in ("Generator Pg MAE", "Generator Pg RMSE", "Mean Optimality Gap"))
        _, s = step1(gt, units, H, "e2e_pd_ac", y["pg"], y["target"], e_pd[j].sum(1), gate)
        e2e_mae_pd = float(pd.read_csv(run / f"artifacts/test/{case}_metrics.csv").set_index("Metric")["Value"]["MAE Pd"])
        s.update(n_missing=len(y["missing"][0]), mae_pd_compare=float(m["MAE Pd"]), mae_pd_e2e_logged=e2e_mae_pd,
                 pd_gate_ok=bool(np.isclose(float(m["MAE Pd"]), e2e_mae_pd, rtol=1e-3)))
        print(f"  gate MAE Pd h{H}: compare {s['mae_pd_compare']:.5f} vs E2E logged {e2e_mae_pd:.5f} "
              f"-> {'ok' if s['pd_gate_ok'] else 'MISMATCH'} ({s['n_missing']} OPF failures)")
        splits.append(s)
        jensen.append(step4(gt, H, ARM, x["pd_pred"][j].sum(1), y["k"], y["target"], y["pg"]))

    ref = pd.read_csv(DIAG / case / "step1_split.csv")
    df = pd.concat([pd.DataFrame(splits), ref[ref.H.isin(horizons) & ref.arm.str.startswith("ac_")]],
                   ignore_index=True).sort_values(["H", "arm"])
    tag = f"{case}_h{'-'.join(map(str, horizons))}"
    df.to_csv(OUT / f"phaseB_{tag}.csv", index=False)
    pd.DataFrame(jensen).to_csv(OUT / f"phaseB_step4_{tag}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4g}".format):
        extra = ["n_missing", "mae_pd_compare", "mae_pd_e2e_logged", "pd_gate_ok", "pd_zero_bus_sum_mw"]
        print(f"\n[{case}] Phase B split\n{df[SPLIT_COLS + extra].to_string(index=False)}")
        print(f"\n[{case}] step4 on the real OPF dispatch (ac_* = E2E Pd → AC-OPF)\n{pd.DataFrame(jensen).to_string(index=False)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["convert", "proxy", "table"])
    p.add_argument("--case", required=True)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 6])
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dict(convert=convert, proxy=proxy, table=table)[a.step](a.case, a.horizons)


if __name__ == "__main__":
    main()
