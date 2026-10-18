"""
Ex-post AC power flow: is each arm's dispatch operable once the true load arrives, and what does it cost?

What?
    - per test sample (realized target hour, horizon step k), per arm:
        1. AC-PF (pypowsybl OpenLoadFlow, distributed slack ∝ Pmax, NO Q limits): predicted dispatch P̂ as target P,
           predicted Vm at gen buses, true Pd/Qd everywhere; the imbalance (load error + losses) is spread over the units
           -> PF-solved dispatch P̃ = rebalanced dispatch, single source for cost and feasibility
        2. limits on the PF solution: convergence, thermal, PQ voltage, bus-level Qg, units at/beyond P limits,
           active power mismatch left on the slack bus
        3. costs C(P̂), C(P̃), C(P*) per sample -> γ_reb, R in corrected_gap.py
Why?
    - replaces realized-space residuals (dispatch frozen, true load inserted), which are no physical operating point
How?
    - arms: exact (phase1c datakit AC-OPF on forecast load), dc (same run's DC-OPF dispatch; no voltages -> case-file
      generator setpoints VG at PV/REF buses), surrogate / e2e (graphkit state dumps),
      label (phase1a OPF labels -> sanity check: must give zero violations, P̃ = P*)
    - samples aligned on the target hour (load_scenario_idx), never on row order (ML test order is shuffled)
    - Qg per bus vs summed limits: PF fixes only the bus total, its per-gen split on multi-gen buses is arbitrary
    - datakit Network -> pypowsybl via gridfm_datakit.powsybl (one network per worker, updated in place);
      slack = case-file REF bus; LF_PARAMS + OLF provider defaults logged to <OUT>/<case>/run_config_<job>.json
    - insufficient headroom: remainder left on the REF unit beyond its limit (LEAVE_ON_SLACK_BUS) -> stays in the cost
      metric (priced on its cost curve = lower bound), flagged + counted as P-limit violation
    - status: converged | diverged (NR, no solution -> infeasible) | no_dispatch (2-step: OPF failed on the forecast)
    - run on barnard via scripts/expost_pf.sbatch

Usage (from Thesis_Repo root, inside a barnard job):
    python exp1/generate_metrics/expost_pf.py --case case118_ieee --horizons 6 --arms label exact surrogate e2e --limit 200
"""

import argparse
import glob
import json
import os
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pypowsybl as pp
import pypowsybl.loadflow as lf

from compare import _build_predicted_scenario_map
from gridfm_datakit.network import load_net_from_pglib
from gridfm_datakit.powsybl.convert import to_powsybl, update_powsybl
from gridfm_datakit.powsybl.preprocess import get_pf_res
from gridfm_datakit.process.process_network import pf_post_processing
from gridfm_datakit.utils.column_names import BUS_COLUMNS, GEN_COLUMNS, BRANCH_COLUMNS
from gridfm_datakit.utils.idx_bus import VM
from gridfm_datakit.utils.idx_gen import GEN_BUS, PMIN, PMAX, VG

DATA = Path("/data/horse/ws/tibo990i-thesis_data")
PHASE1A = DATA / "2019-2021/phase_1a/data_out"
PHASE1C = DATA / "2019-2021/phase_1c"
MLRUNS = DATA / "mlruns"
FORECASTS = Path(__file__).resolve().parents[2] / "phase1_baseline/output/2019-2021"
OUT = PHASE1C / "expost_pf"

TOL_GEN = 0.1   # MVAr (= 1e-3 p.u. at baseMVA 100)
TOL_P = 0.01    # MW: P-limit checks = slack distribution stop (slackBusPMaxMismatch)
TOL_REL = 1e-3  # voltage + thermal limits: 0.1 %
B, G, BR = ({c: i for i, c in enumerate(cols)} for cols in (BUS_COLUMNS, GEN_COLUMNS, BRANCH_COLUMNS))

LF_PARAMS = lf.Parameters(
    distributed_slack=True,
    balance_type=lf.BalanceType.PROPORTIONAL_TO_GENERATION_P_MAX,
    use_reactive_limits=False,
    read_slack_bus=True,  # slackTerminal of the case-file REF bus (MATPOWER import)
    write_slack_bus=True,
    twt_split_shunt_admittance=True,  # MATPOWER pi model; without it case118 labels miss by 1.8e-3 p.u. (bus 86)
    provider_parameters={
        "generatorsWithZeroMwTargetAreNotStarted": "false",  # units at P̂ = 0 take part too
        "slackBusPMaxMismatch": str(TOL_P),  # residual left on the slack bus < TOL_P (default 1 MW)
        # insufficient headroom: remainder stays on the REF unit (beyond its limit) instead of a failed solve
        "slackDistributionFailureBehavior": "LEAVE_ON_SLACK_BUS",
    },
)
OLF_DEFAULTS = lf.get_provider_parameters()["default"]  # logged; useActiveLimits=true kept


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
              ["load_scenario_idx", "idx", "p_mw", "in_service", "cp0_eur", "cp1_eur_per_mw", "cp2_eur_per_mw2"], flt)
    hb, (pd_, qd, vm) = _dense(b, "load_scenario_idx", "bus", "Pd", "Qd", "Vm")
    hg, (pg, c0, c1, c2) = _dense(g[g.in_service == 1], "load_scenario_idx", "idx",
                                  "p_mw", "cp0_eur", "cp1_eur_per_mw", "cp2_eur_per_mw2")
    assert np.array_equal(hb, hg)
    return hb, dict(pd=pd_, qd=qd, vm=vm, pg=pg, c0=c0[0], c1=c1[0], c2=c2[0])  # costs static across hours


def case_vm(case):
    """case-file bus voltages, generator setpoint VG at generator buses"""
    net = load_net_from_pglib(case)
    on = net.idx_gens_in_service
    vm = net.buses[:, VM].copy()
    vm[net.gens[on, GEN_BUS].astype(int)] = net.gens[on, VG]
    return vm


def load_exact(case, H, fc, dc=False):
    """phase1c datakit OPF on the forecast load; its load_scenario_idx is the 2-step flat scenario
    dc: DC-OPF dispatch of the same run, Vm = case-file setpoints (DC-OPF has no voltages)"""
    raw = PHASE1C / f"data_out/{case}_horizon_{H}/{fc}/{case}/raw"
    col = "p_mw_dc" if dc else "p_mw"
    b = _read(raw / "bus_data.parquet", ["load_scenario_idx", "bus", "Vm", "Pd"])
    g = _read(raw / "gen_data.parquet", ["load_scenario_idx", "idx", col, "in_service"])
    flat, (vm, pd_fc) = _dense(b, "load_scenario_idx", "bus", "Vm", "Pd")
    flat_g, (pg,) = _dense(g[g.in_service == 1], "load_scenario_idx", "idx", col)
    assert np.array_equal(flat, flat_g)
    if dc:
        vm = np.broadcast_to(case_vm(case), vm.shape)
    target, k = flat_targets(case, H)
    miss = np.setdiff1d(np.arange(len(target)), flat)  # OPF failed on the forecast load -> no dispatch
    pred = dict(target=target[flat], k=k[flat], pg=pg, vm=vm, missing=(target[miss], k[miss]))
    if fc == "true":  # forecast == truth -> alignment check against phase1a
        pred["pd_true"] = pd_fc
    return pred


RUN_EXPERIMENTS = ("state-dump", "final_thesis_runs", "final_e2e_qgmse", "2-step-surrogate-eval")  # thesis models only; ablation exps reuse run names


def resolve_run(names, case):
    """newest run named any of <names> with a state dump, within RUN_EXPERIMENTS"""
    exps = [m.parent for m in MLRUNS.glob("*/meta.yaml")
            if any(line.strip() in [f"name: {e}" for e in RUN_EXPERIMENTS] for line in m.read_text().splitlines())]
    runs = [Path(t).parents[1] for e in exps for t in glob.glob(f"{e}/*/tags/mlflow.runName")
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
        import torch  # lazy: only E2E needs it
        hours = torch.load(PHASE1A / case / "processed_temporal/load_scenarios.pt").numpy().astype(np.int64)
        k = s % H
        target = hours[test[s // H] + W + k]
    return dict(target=target, k=k, pg=pg, vm=vm, pd_true=pd_true)


# ---------------------------------------------------------------- AC-PF (worker side)

_NET = _PP = _MAP = _REF = None


def _init_worker(case):
    global _NET, _PP, _MAP, _REF
    _NET = load_net_from_pglib(case)
    conv = to_powsybl(_NET)
    _PP, _MAP = conv.pp_net, conv.mapping_p2g
    _REF = next(b for b, i in _MAP.bus.items() if i == _NET.ref_bus_idx)


def _solve(task):
    """one sample: true load + predicted dispatch + Vm setpoints -> AC-PF (distributed slack) -> limit checks"""
    pd_, qd, pg, vm = task
    on = _NET.idx_gens_in_service
    lim = np.clip(pg, _NET.gens[on, PMIN], _NET.gens[on, PMAX])
    dev = np.abs(pg - lim)  # P̂ outside [Pmin, Pmax]: OLF holds it there, the unit drops out of the distribution
    pg = np.where(dev <= TOL_P, lim, pg)  # solver noise -> clipped, unit still takes part
    info = dict(pg_pred_out_n=int((dev > TOL_P).sum()), pg_pred_out_max_mw=dev.max())
    _NET.Pd, _NET.Qd, _NET.Pg_gen, _NET.Vm = pd_, qd, pg, vm
    update_powsybl(_PP, _NET, _MAP)  # in place: loads, target P, target V (uniform voltage init -> no carry-over)
    r = lf.run_ac(_PP, LF_PARAMS)
    slack = r[0].slack_bus_results[0]
    assert slack.id == _REF, f"slack bus {slack.id} != case-file REF bus {_REF}"
    info["slack_mismatch_mw"] = slack.active_power_mismatch  # left on the slack bus: > TOL_P = insufficient headroom
    try:
        res = get_pf_res(_PP, 0.0, r, _MAP)  # ValueError unless converged; mismatch added to the REF gen's Pg
    except (ValueError, RuntimeError):  # NR not converged: no solution
        return dict(status="diverged", **info)
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

    # P limits on the PF-solved dispatch; "at" only for dispatchable units (Pmax > Pmin)
    p, pmin, pmax = (gen[:, G[c]] for c in ("p_mw", "min_p_mw", "max_p_mw"))
    at_lim = (pmax - pmin > TOL_P) & ((p >= pmax - TOL_P) | (p <= pmin + TOL_P))
    beyond = np.maximum(p - pmax, 0) + np.maximum(pmin - p, 0)

    return dict(
        status="converged",
        **info,
        pg_pf=p,
        thermal_max_loading=loading.max(),
        thermal_n=int((loading > 1 + TOL_REL).sum()),
        vm_max_viol=v_viol.max(initial=0.0),
        vm_n=int(v_bad.sum()),
        qg_max_viol_mvar=q_viol.max(initial=0.0),
        qg_n=int((q_viol > TOL_GEN).sum()),
        pg_lim_n=int(at_lim.sum()),
        pg_beyond_n=int((beyond > TOL_P).sum()),
    )


# ---------------------------------------------------------------- driver

def evaluate(pool, case, pred, limit=None):
    """align with truth, solve PF with distributed slack, cost P̂ / P̃ / P* -> one row per sample"""
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

    res = pool.map(_solve, [(gt["pd"][j], gt["qd"][j], pg[r], vm[r]) for r, j in enumerate(i)], chunksize=16)
    pg_pf = np.array([r.pop("pg_pf", np.full(pg.shape[1], np.nan)) for r in res])
    cost = lambda p: (gt["c0"] + gt["c1"] * p + gt["c2"] * p ** 2).sum(1)

    df = pd.DataFrame(dict(
        target=target, k=k,
        pg_pred_mw=pg.sum(1), pg_pf_mw=pg_pf.sum(1), pg_true_mw=gt["pg"][i].sum(1),
        cost_pred=cost(pg), cost_pf=cost(pg_pf), cost_true=cost(gt["pg"][i]),
    ))
    df = df.join(pd.DataFrame(res))
    if "missing" in pred and len(pred["missing"][0]) and not limit:
        miss = pd.DataFrame(dict(target=pred["missing"][0], k=pred["missing"][1], status="no_dispatch"))
        df = pd.concat([df, miss], ignore_index=True)
    df["slack_mismatch_flag"] = df.slack_mismatch_mw.abs() > TOL_P
    return df


def summarize(tag, df):
    c = df[df.status == "converged"]
    share = lambda col: (c[col] > 0).mean()
    gap = (c.cost_pf - c.cost_true) / c.cost_true * 100
    print(f"[{tag}] n={len(df)} " + " ".join(f"{s}={n}" for s, n in df.status.value_counts().items()) + " | "
          f"viol share thermal={share('thermal_n'):.4f} vm={share('vm_n'):.4f} qg={share('qg_n'):.4f} "
          f"pg_beyond={share('pg_beyond_n'):.4f} headroom={c.slack_mismatch_flag.mean():.4f} "
          f"pred_out={(df.pg_pred_out_n > 0).mean():.4f} | "
          f"γ_reb signed={gap.mean():+.4f} % |γ_reb|={gap.abs().mean():.4f} % | "
          f"mean |ΣP̃-ΣP̂|={(c.pg_pf_mw - c.pg_pred_mw).abs().mean():.2f} MW", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", required=True)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 24])
    p.add_argument("--arms", nargs="+", default=["label", "exact", "dc", "surrogate", "e2e"])
    p.add_argument("--forecasters", nargs="+", default=["true", "xgb", "sarima", "snaive"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    p.add_argument("--limit", type=int, default=None, help="smoke test: N samples per config")
    p.add_argument("--out-dir", type=Path, default=OUT)
    a = p.parse_args()

    # (arm, horizon, forecaster, seed) configs; label = phase1a labels on the 2-step h1 test hours
    configs = [("label", 1, None, None)] if "label" in a.arms else []
    for H in a.horizons:
        configs += [(arm, H, fc, None) for arm in ("exact", "dc") for fc in a.forecasters if arm in a.arms]
        configs += [("surrogate", H, fc, s) for fc in a.forecasters for s in a.seeds if "surrogate" in a.arms]
        configs += [("e2e", H, None, s) for s in a.seeds if "e2e" in a.arms]

    cfg = a.out_dir / a.case / f"run_config_{os.environ.get('SLURM_JOB_ID', 'local')}.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps(dict(args={**vars(a), "out_dir": str(a.out_dir)}, pypowsybl=pp.__version__,
                                   load_flow=repr(LF_PARAMS), olf_provider_defaults=OLF_DEFAULTS.to_dict(),
                                   tol_gen=TOL_GEN, tol_p=TOL_P, tol_rel=TOL_REL), indent=1))

    load_net_from_pglib(a.case)  # download + correct the grid once, before workers race on the cache
    # spawn: fresh pypowsybl runtime per worker instead of a forked copy of the parent's
    with get_context("spawn").Pool(a.workers, initializer=_init_worker, initargs=(a.case,)) as pool:
        for arm, H, fc, seed in configs:
            tag = "_".join(str(x) for x in (arm, fc, f"s{seed}" if seed is not None else None) if x)
            out = a.out_dir / a.case / f"h{H}" / f"{tag}{f'_limit{a.limit}' if a.limit else ''}.parquet"
            try:
                if arm == "label":
                    hours = np.unique(flat_targets(a.case, 1)[0])
                    _, gt = load_truth(a.case, hours)
                    pred = dict(target=hours, k=np.zeros_like(hours), pg=gt["pg"], vm=gt["vm"])
                elif arm in ("exact", "dc"):
                    pred = load_exact(a.case, H, fc, dc=arm == "dc")
                else:
                    # E2E: case500 retrain uses the full case name (base_e2e_case500_goc_h*), older runs the short one
                    names = ([f"surrogate_{a.case}_{fc}_h{H}_s{seed}"] if arm == "surrogate" else
                             [f"base_e2e_{n}_h{H}_s{seed}" for n in (a.case, a.case.split('_')[0])])
                    run = resolve_run(names, a.case)
                    if run is None:
                        raise FileNotFoundError(f"no run with state dump named {names}")
                    print(f"[{a.case} h{H} {tag}] run {run}", flush=True)
                    pred = load_ml(arm, a.case, H, run)
                df = evaluate(pool, a.case, pred, a.limit)
            except (FileNotFoundError, AssertionError) as e:  # missing inputs / failed alignment check
                print(f"SKIP [{a.case} h{H} {tag}]: {e!r}", flush=True)
                continue
            meta = dict(arm=arm, case=a.case, horizon=H, forecaster=fc, seed=seed,
                        run=str(run) if arm in ("surrogate", "e2e") else None)  # run -> logged-gap gate
            for pos, (col, val) in enumerate(meta.items()):
                df.insert(pos, col, val)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out, index=False)
            summarize(f"{a.case} h{H} {tag}", df)


if __name__ == "__main__":
    main()
