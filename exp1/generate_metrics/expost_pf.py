"""
Ex-post AC power flow: is each arm's dispatch operable once the true load arrives?

What?
    - per test sample (realized target hour, horizon step k), per arm:
        1. rebalance: close Δ = ΣPg_true - ΣPg_pred (true OPF, incl. losses) by merit order (metrics.merit_order_rebalance,
           same rule as the rebalanced gap); remainder beyond headroom -> slack (rebal_slack_mw)
        2. AC-PF (PowerModels compute_ac_pf, single slack, NO Q limits):
           Pg' at non-slack gens, predicted Vm at gen buses, true Pd/Qd everywhere
        3. limits on the PF solution: convergence, thermal, PQ voltage, bus-level Qg, Pg' headroom, ΔP_slack
Why?
    - replaces realized-space residuals (dispatch frozen, true load inserted), which are no physical operating point
How?
    - arms: exact (phase1c datakit OPF on forecast load), surrogate / e2e (graphkit state dumps),
      label (phase1a OPF labels -> sanity check: must give zero violations, ΔP_slack ~ 0)
    - samples aligned on the target hour (load_scenario_idx), never on row order (ML test order is shuffled)
    - Qg per bus vs summed limits: PF fixes only the bus total, its per-gen split on multi-gen buses is arbitrary
    - PF path reused from datakit (Network setters + run_pf(fast=True) + pf_post_processing), one Julia per worker
    - run on barnard via scripts/expost_pf.sbatch

Usage (from Thesis_Repo root, inside a barnard job):
    python exp1/generate_metrics/expost_pf.py --case case118_ieee --horizons 6 --arms label exact surrogate e2e --limit 200
"""

import argparse
import glob
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from compare import _build_predicted_scenario_map
from metrics import merit_order_rebalance
from gridfm_datakit.network import load_net_from_pglib
from gridfm_datakit.process.process_network import init_julia, pf_post_processing
from gridfm_datakit.process.solvers import run_pf
from gridfm_datakit.utils.column_names import BUS_COLUMNS, GEN_COLUMNS, BRANCH_COLUMNS
from gridfm_datakit.utils.idx_gen import GEN_BUS, PMIN, PMAX

DATA = Path("/data/horse/ws/tibo990i-thesis_data")
PHASE1A = DATA / "2019-2021/phase_1a/data_out"
PHASE1C = DATA / "2019-2021/phase_1c"
MLRUNS = DATA / "mlruns"
FORECASTS = Path(__file__).resolve().parents[2] / "phase1_baseline/output/2019-2021"
OUT = PHASE1C / "expost_pf"

TOL_GEN = 0.1   # MW / MVAr (= 1e-3 p.u. at baseMVA 100)
TOL_REL = 1e-3  # voltage + thermal limits: 0.1 %
B, G, BR = ({c: i for i, c in enumerate(cols)} for cols in (BUS_COLUMNS, GEN_COLUMNS, BRANCH_COLUMNS))


# ---------------------------------------------------------------- data

def _read(path, cols, flt=None):
    """(hive-partitioned) parquet -> DataFrame"""
    return ds.dataset(path, format="parquet", partitioning="hive").to_table(columns=cols, filter=flt).to_pandas()


def _dense(df, key, idx, *vals):
    """long [key, idx, vals] -> sorted int keys + one [n_key, n_idx] array per val; asserts a complete grid"""
    df = df.sort_values([key, idx], kind="stable")
    keys = df[key].unique().astype(np.int64)
    n = df[idx].nunique()
    assert len(df) == len(keys) * n, f"incomplete grid over {key} x {idx}"
    return keys, [df[v].to_numpy(np.float64).reshape(len(keys), n) for v in vals]


def flat_targets(case, H):
    """2-step flat scenario -> (target hour, step k); same flattening as phase1c transform + compare.py"""
    fc = _read(FORECASTS / f"{case}_horizon{H}.parquet", ["load_scenario_idx", "horizon_step"], ds.field("bus_id") == 0)
    target, step = _build_predicted_scenario_map(fc)
    return np.array([target[i] for i in range(len(target))]), np.array([step[i] for i in range(len(step))])


def load_truth(case, hours):
    """phase1a OPF labels (on the true load) at the given hours"""
    raw = PHASE1A / case / "raw"
    lsi = ds.field("load_scenario_idx")
    flt = (lsi >= float(hours.min())) & (lsi <= float(hours.max()))
    b = _read(raw / "bus_data.parquet", ["load_scenario_idx", "bus", "Pd", "Qd", "Vm"], flt)
    g = _read(raw / "gen_data.parquet",
              ["load_scenario_idx", "idx", "p_mw", "in_service", "cp1_eur_per_mw", "cp2_eur_per_mw2"], flt)
    hb, (pd_, qd, vm) = _dense(b, "load_scenario_idx", "bus", "Pd", "Qd", "Vm")
    hg, (pg, c1, c2) = _dense(g[g.in_service == 1], "load_scenario_idx", "idx", "p_mw", "cp1_eur_per_mw", "cp2_eur_per_mw2")
    assert np.array_equal(hb, hg)
    return hb, dict(pd=pd_, qd=qd, vm=vm, pg=pg, c1=c1[0], c2=c2[0])  # costs static across hours


def load_exact(case, H, fc):
    """phase1c datakit AC-OPF on the forecast load; its load_scenario_idx is the 2-step flat scenario"""
    raw = PHASE1C / f"data_out/{case}_horizon_{H}/{fc}/{case}/raw"
    b = _read(raw / "bus_data.parquet", ["load_scenario_idx", "bus", "Vm", "Pd"])
    g = _read(raw / "gen_data.parquet", ["load_scenario_idx", "idx", "p_mw", "in_service"])
    flat, (vm, pd_fc) = _dense(b, "load_scenario_idx", "bus", "Vm", "Pd")
    flat_g, (pg,) = _dense(g[g.in_service == 1], "load_scenario_idx", "idx", "p_mw")
    assert np.array_equal(flat, flat_g)
    target, k = flat_targets(case, H)
    miss = np.setdiff1d(np.arange(len(target)), flat)  # OPF failed on the forecast load -> no dispatch
    pred = dict(target=target[flat], k=k[flat], pg=pg, vm=vm, missing=(target[miss], k[miss]))
    if fc == "true":  # forecast == truth -> alignment check against phase1a
        pred["pd_true"] = pd_fc
    return pred


def resolve_run(names, case):
    """newest mlruns run named any of <names> that carries a state dump"""
    runs = [Path(t).parents[1] for t in glob.glob(f"{MLRUNS}/*/*/tags/mlflow.runName")
            if open(t).read().strip() in names
            and (Path(t).parents[1] / f"artifacts/test/{case}_state_gen.parquet").exists()]
    return max(runs, key=os.path.getmtime) if runs else None


def load_ml(arm, case, H, run):
    """graphkit state dump (physical units); row s follows the shuffled test order in scenario_splits.json"""
    test = np.array(json.load(open(run / f"artifacts/stats/{case}_scenario_splits.json"))["test"])
    b = pq.read_table(run / f"artifacts/test/{case}_state_bus.parquet", columns=["scenario", "bus", "vm_pred", "pd_true"])
    g = pq.read_table(run / f"artifacts/test/{case}_state_gen.parquet", columns=["scenario", "gen", "pg_pred"])
    s, (vm, pd_true) = _dense(b.to_pandas(), "scenario", "bus", "vm_pred", "pd_true")
    s_g, (pg,) = _dense(g.to_pandas(), "scenario", "gen", "pg_pred")
    assert np.array_equal(s, s_g)
    if arm == "surrogate":  # eval-set scenario == 2-step flat scenario
        assert len(test) == len(s)
        target, k = (a[test[s]] for a in flat_targets(case, H))
    else:  # window w: inputs [w, w+W), targets w+W+k; file index -> hour via load_scenarios.pt
        assert len(test) * H == len(s)
        assert int((run / "params/args/data/forecast_horizon").read_text()) == H
        W = int((run / "params/args/data/temporal_window").read_text())
        import torch  # lazy: Pool workers fork before this -> juliacall never sees torch (segfault warning)
        hours = torch.load(PHASE1A / case / "processed_temporal/load_scenarios.pt").numpy().astype(np.int64)
        k = s % H
        target = hours[test[s // H] + W + k]
    return dict(target=target, k=k, pg=pg, vm=vm, pd_true=pd_true)


# ---------------------------------------------------------------- AC-PF (worker side)

_NET = _JL = None


def _init_worker(case):
    global _NET, _JL
    _NET = load_net_from_pglib(case)
    _JL = init_julia(max_iter=300)


def _solve(task):
    """one sample: true load + rebalanced dispatch + Vm setpoints -> AC-PF -> limit checks on the PF solution"""
    pd_, qd, pg, vm = task
    _NET.Pd, _NET.Qd, _NET.Pg_gen, _NET.Vm = pd_, qd, pg, vm  # Vm used at PV/REF only, slack Pg' ignored by PF
    try:
        res = run_pf(_NET, _JL, fast=True)
    except RuntimeError:  # raised on non-convergence (NLsolve ftol 1e-8 p.u.)
        return None
    out = pf_post_processing(0, _NET, res, None, False)  # MW / MVAr / p.u.
    bus = out["bus"]
    gen = out["gen"][_NET.idx_gens_in_service]
    br = out["branch"][_NET.idx_branches_in_service]

    # thermal: max(|S_from|, |S_to|) / rate_a
    s = np.maximum(np.hypot(br[:, BR["pf"]], br[:, BR["qf"]]), np.hypot(br[:, BR["pt"]], br[:, BR["qt"]]))
    loading = s / br[:, BR["rate_a"]]

    # voltage at PQ buses (gen-bus Vm is a fixed setpoint)
    pq_ = bus[:, B["PQ"]] == 1
    v, vmin, vmax = bus[pq_, B["Vm"]], bus[pq_, B["min_vm_pu"]], bus[pq_, B["max_vm_pu"]]
    v_viol = np.maximum(v - vmax, 0) + np.maximum(vmin - v, 0)
    v_bad = (v > vmax * (1 + TOL_REL)) | (v < vmin * (1 - TOL_REL))

    # Qg per non-slack gen bus vs summed limits
    gb = gen[:, G["bus"]].astype(int)
    n = len(bus)
    q, qmin, qmax = (np.bincount(gb, gen[:, G[c]], n) for c in ("q_mvar", "min_q_mvar", "max_q_mvar"))
    gen_bus = np.bincount(gb, minlength=n) > 0
    gen_bus[_NET.ref_bus_idx] = False
    q_viol = (np.maximum(q - qmax, 0) + np.maximum(qmin - q, 0))[gen_bus]

    # slack: PF-solved Pg vs its rebalanced setpoint and its limits
    ref = gb == _NET.ref_bus_idx
    p_ref = gen[ref, G["p_mw"]].sum()
    slack_viol = max(p_ref - gen[ref, G["max_p_mw"]].sum(), 0) + max(gen[ref, G["min_p_mw"]].sum() - p_ref, 0)

    return dict(
        dp_slack_mw=p_ref - pg[ref].sum(),
        slack_viol_mw=slack_viol,
        thermal_max_loading=loading.max(),
        thermal_n=int((loading > 1 + TOL_REL).sum()),
        vm_max_viol=v_viol.max(initial=0.0),
        vm_n=int(v_bad.sum()),
        qg_max_viol_mvar=q_viol.max(initial=0.0),
        qg_n=int((q_viol > TOL_GEN).sum()),
    )


# ---------------------------------------------------------------- driver

def evaluate(pool, net, case, pred, limit=None):
    """align with truth, rebalance (step 1), solve PF (steps 2-3) -> one row per sample"""
    order = np.lexsort((pred["k"], pred["target"]))
    if limit:  # smoke test: evenly spread subset
        order = order[np.linspace(0, len(order) - 1, min(limit, len(order))).astype(int)]
    target, k, pg, vm = pred["target"][order], pred["k"][order], pred["pg"][order], pred["vm"][order]

    hours, gt = load_truth(case, target)
    i = np.searchsorted(hours, target)
    assert np.array_equal(hours[i], target), "target hours missing in phase1a labels"
    if "pd_true" in pred:
        err = np.abs(pred["pd_true"][order] - gt["pd"][i]).max()
        assert err < 1e-3, f"true load != phase1a labels (max {err:.3g} MW): misaligned or stale processed data"

    # step 1: close the net imbalance by merit order, Δ = ΣPg_true - ΣPg_pred
    on = net.idx_gens_in_service
    slack = net.gens[on, GEN_BUS] == net.ref_bus_idx
    delta = gt["pg"][i].sum(1) - pg.sum(1)
    pg_s, rest = merit_order_rebalance(pg, delta, net.gens[on, PMIN], net.gens[on, PMAX], gt["c1"], gt["c2"], slack)
    head = (np.maximum(pg_s - net.gens[on, PMAX], 0) + np.maximum(net.gens[on, PMIN] - pg_s, 0))[:, ~slack]

    tasks = [(gt["pd"][j], gt["qd"][j], pg_s[r], vm[r]) for r, j in enumerate(i)]
    res = pool.map(_solve, tasks, chunksize=16)

    df = pd.DataFrame(dict(
        target=target, k=k,
        status=np.where([r is not None for r in res], "converged", "diverged"),
        delta_mw=delta, rebal_slack_mw=rest, headroom_max_mw=head.max(1), headroom_n=(head > TOL_GEN).sum(1),
    ))
    df = df.join(pd.DataFrame([r or {} for r in res]))
    if "missing" in pred and len(pred["missing"][0]) and not limit:
        miss = pd.DataFrame(dict(target=pred["missing"][0], k=pred["missing"][1], status="no_dispatch"))
        df = pd.concat([df, miss], ignore_index=True)
    return df


def summarize(tag, df):
    c = df[df.status == "converged"]
    share = lambda col: (c[col] > 0).mean()
    print(f"[{tag}] n={len(df)} converged={len(c) / len(df):.4f} no_dispatch={(df.status == 'no_dispatch').sum()} | "
          f"viol share thermal={share('thermal_n'):.4f} vm={share('vm_n'):.4f} qg={share('qg_n'):.4f} "
          f"headroom={(df.headroom_n > 0).mean():.4f} slack={(c.slack_viol_mw > TOL_GEN).mean():.4f} "
          f"rebal_slack={(df.rebal_slack_mw > TOL_GEN).mean():.4f} | "
          f"mean |Δ|={df.delta_mw.abs().mean():.3f} MW, mean |ΔP_slack|={c.dp_slack_mw.abs().mean():.3f} MW", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", required=True)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 24])
    p.add_argument("--arms", nargs="+", default=["label", "exact", "surrogate", "e2e"])
    p.add_argument("--forecasters", nargs="+", default=["true", "xgb", "sarima", "snaive"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    p.add_argument("--limit", type=int, default=None, help="smoke test: N samples per config")
    a = p.parse_args()

    # (arm, horizon, forecaster, seed) configs; label = phase1a labels on the 2-step h1 test hours
    configs = [("label", 1, None, None)] if "label" in a.arms else []
    for H in a.horizons:
        configs += [("exact", H, fc, None) for fc in a.forecasters if "exact" in a.arms]
        configs += [("surrogate", H, fc, s) for fc in a.forecasters for s in a.seeds if "surrogate" in a.arms]
        configs += [("e2e", H, None, s) for s in a.seeds if "e2e" in a.arms]

    net = load_net_from_pglib(a.case)
    with Pool(a.workers, initializer=_init_worker, initargs=(a.case,)) as pool:
        for arm, H, fc, seed in configs:
            tag = "_".join(str(x) for x in (arm, fc, f"s{seed}" if seed is not None else None) if x)
            out = OUT / a.case / f"h{H}" / f"{tag}{f'_limit{a.limit}' if a.limit else ''}.parquet"
            try:
                if arm == "label":
                    hours = np.unique(flat_targets(a.case, 1)[0])
                    _, gt = load_truth(a.case, hours)
                    pred = dict(target=hours, k=np.zeros_like(hours), pg=gt["pg"], vm=gt["vm"])
                elif arm == "exact":
                    pred = load_exact(a.case, H, fc)
                else:
                    # E2E: case500 retrain uses the full case name (base_e2e_case500_goc_h*), older runs the short one
                    names = ([f"surrogate_{a.case}_{fc}_h{H}_s{seed}"] if arm == "surrogate" else
                             [f"base_e2e_{n}_h{H}_s{seed}" for n in (a.case, a.case.split('_')[0])])
                    run = resolve_run(names, a.case)
                    if run is None:
                        raise FileNotFoundError(f"no run with state dump named {names}")
                    print(f"[{a.case} h{H} {tag}] run {run}", flush=True)
                    pred = load_ml(arm, a.case, H, run)
                df = evaluate(pool, net, a.case, pred, a.limit)
            except (FileNotFoundError, AssertionError) as e:  # missing inputs / failed alignment check
                print(f"SKIP [{a.case} h{H} {tag}]: {e!r}", flush=True)
                continue
            for pos, (col, val) in enumerate(dict(arm=arm, case=a.case, horizon=H, forecaster=fc, seed=seed).items()):
                df.insert(pos, col, val)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out, index=False)
            summarize(f"{a.case} h{H} {tag}", df)


if __name__ == "__main__":
    main()
