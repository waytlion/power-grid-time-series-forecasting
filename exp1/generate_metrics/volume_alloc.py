"""Pg volume / allocation split (MW) + loss shift for all arms -> tidy CSV (table 8 allocation column).
- per forecast timestep, e = P̂ - P* per generator: volume |Σe|, allocation Σ|e| - |Σe| (split of Q1 step 5, before
  rebalancing)
- loss shift = ΣP̃ - ΣP* (ex-post PF, converged) = actual - optimal losses (same load); measured, not in the thesis
- inputs via expost_pf loaders (same alignment); gate: mean volume == ex-post mean |ΣP̂ - ΣP*|
Usage (from Thesis_Repo root, inside a barnard job, ~40 min on 4 CPUs):
    python exp1/generate_metrics/volume_alloc.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from expost_pf import PHASE1C, load_exact, load_ml, load_truth, resolve_run

EXPOST = PHASE1C / "expost_pf_v4_qlim"
CASES = ["case14_ieee", "case118_ieee", "case500_goc"]
FCS = ["xgb", "xgb_tot", "sarima", "snaive"]
OPT = {"exact": "ac", "dc": "dc", "surrogate": "surrogate", "e2e": "e2e"}


def configs(seeds):
    """(arm, forecaster, seed, expost_pf tag)"""
    for arm in ("exact", "dc"):
        for fc in FCS:
            yield arm, fc, None, f"{arm}_{fc}"
    for fc in FCS:
        for s in seeds:
            yield "surrogate", fc, s, f"surrogate_{fc}_s{s}"
    for s in seeds:
        yield "e2e", None, s, f"e2e_s{s}"


def predict(arm, case, H, fc, seed):
    if arm in ("exact", "dc"):
        return load_exact(case, H, fc, dc=arm == "dc")
    names = ([f"surrogate_{case}_{fc}_h{H}_s{seed}"] if arm == "surrogate" else
             [f"base_e2e_{n}_h{H}_s{seed}" for n in (case, case.split("_")[0])])
    run = resolve_run(names, case)
    if run is None:
        raise FileNotFoundError(names)
    return load_ml(arm, case, H, run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("exp1/results/combined/volume_alloc.csv"))
    ap.add_argument("--cases", nargs="+", default=CASES)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    rows = []
    for case in a.cases:
        for H in (1, 6, 24):
            for arm, fc, seed, tag in configs(a.seeds):
                xp = EXPOST / case / f"h{H}" / f"{tag}.parquet"
                if not xp.exists():
                    continue
                try:
                    pred = predict(arm, case, H, fc, seed)
                except FileNotFoundError as e:
                    print(f"SKIP {case} h{H} {tag}: {e!r}", flush=True)
                    continue
                hours, gt = load_truth(case, pred["target"])
                i = np.searchsorted(hours, pred["target"])
                assert np.array_equal(hours[i], pred["target"]), f"{tag}: target hours missing"
                e = pred["pg"] - gt["pg"][i]
                vol = np.abs(e.sum(1))
                alloc = np.abs(e).sum(1) - vol
                x = pd.read_parquet(xp, columns=["status", "pg_pred_mw", "pg_pf_mw", "pg_true_mw"])
                gate = (x.pg_pred_mw - x.pg_true_mw).dropna().abs().mean()
                assert abs(vol.mean() - gate) < 1e-3, f"{tag}: volume {vol.mean():.4f} != ex-post {gate:.4f}"
                c = x[x.status == "converged"]
                loss = c.pg_pf_mw - c.pg_true_mw
                rows.append(dict(case=case, horizon=H, optimizer=OPT[arm], forecaster=fc or "none", seed=seed,
                                 n=len(e), volume_mw=vol.mean(), alloc_mw=alloc.mean(), signed_mw=e.sum(1).mean(),
                                 abs_sum_mw=np.abs(e).sum(1).mean(), loss_shift_mw=loss.mean(),
                                 loss_shift_abs_mw=loss.abs().mean(), n_converged=len(c)))
                print(f"{case} h{H} {tag}: n={len(e)} vol={vol.mean():.2f} alloc={alloc.mean():.2f} "
                      f"loss={loss.mean():+.2f} MW", flush=True)
            pd.DataFrame(rows).to_csv(a.out, index=False)  # partial results survive a crash
    print(f"Wrote {a.out} ({len(rows)} configs)")


if __name__ == "__main__":
    main()
