"""
Dispatch-structure diagnostics: why E2E barely beats 2-step on Pg (case500: worse Pg MAE, same RMSE, better gap).

What?
    1. Pg error split per arm (AC 2-step per forecaster, E2E s0), test samples:
        - per unit group: pinned / switching / free (GT share of test hours at Pmin|Pmax), linear / quadratic cost,
          and both crossed; mae_contrib sums to the pooled Pg MAE
        - per sample: total |Σ_g e_g| vs misplaced (Σ_g |e_g| − |Σ_g e_g|) / 2 MW; total-demand error ΔD
        - 2-step only: regime flips = units whose state (Pmin / Pmax / free) differs from GT
        - gap split: raw |C(P̂) − C*| = allocation C(P̂) − C(P_T) + volume C(P_T) − C*, P_T = GT dispatch of the hour whose
          total ΣPg* is nearest ΣP̂ (OPF split of the predicted total; valid since Pg ≈ f(D), step 2);
          Pg error of P_T = "this arm's total, OPF's split"
    2. Pg vs total demand D = Σ_i Pd_i (GT; all hours and test hours):
        - R²_g of a leave-one-out local-linear fit Pg_g ~ f(D); pooled R² = share of Pg variance explained by D alone
        - share of the unexplained variance in congested hours (≥ 1 line at its limit)
        - regimes: free units per hour, distinct unit-state / binding-line patterns, hourly switch rate
        - pg_vs_D.png: Pg against D for the 3 largest-variance and the 3 least-explained units
    3. Cross-bus Pd forecast errors (2-step forecasters + E2E Pd head, load buses, test samples):
        - same-hour correlation, PC1 share, cancellation ratio Var(Σ_i e_i) / Σ_i Var(e_i); raw Pd as contrast
        - correlation vs grid hop distance
        - H=1: R² of e_i,t on e_i,t−1, and the gain when the other buses' mean error at t−1 is added;
          R² of the total error Σ_i e_i,t on Σ_i e_i,t−1 (persistence of the shared component)
        - survive = |Σ_i e_i| / Σ_i |e_i| (share of the per-bus error left after summing; 1 = all same sign)
    4. Jensen ceiling of the 2-step (AC, per forecaster): f(D) = GT dispatch at the nearest total demand
        - plug-in f(D̂) (≈ AC 2-step) vs hedged mean_q f(D̂ + ε_q), ε_q = quantiles of the forecaster's own test D error
          per horizon step (centered = curvature only; raw = + bias correction)
        - hedged − plug-in = what E[OPF(Pd) | h] ≠ OPF(E[Pd | h]) can buy with this forecaster's information
        - Δ = mean (hedged − plug-in)²; cross = MSE_plug − MSE_hedged − Δ (= 2 E[(Y − hedged)(hedged − plug-in)],
          0 iff hedged is the conditional mean)
        - D̂ conditional-mean test: share of Var(D − D̂) explained by group means of D̂ decile / target hour / step
          (≈ 0 for E[D | h], since every grouping is a function of h)
Why?
    - pinned units: AC-OPF lands on the limit, the sigmoid-bounded E2E never quite does -> MAE penalty, ~no cost impact?
    - Pg ≈ f(D) -> where the load error sits barely matters; f is kinked -> curvature gain sized in step 4
    - XGB/SARIMA see only their own bus -> correlated errors = the information E2E could add
    - thesis ceiling argument rests on a non-affine OPF map; step 4 sizes it on these grids
How?
    - loaders + target-hour alignment from expost_pf; E2E runs pinned by (experiment, run name), see E2E_RUNS
    - gates: recomputed pooled Pg MAE/RMSE vs the logged metrics CSVs; E2E dumped pg_true / pd_true vs GT
    - out: <PHASE1C>/dispatch_diag/<case>/{step1_groups,step1_split,step2_units,step2_summary,step3_corr,step3_hop,
      step4_jensen}.csv

Usage (from Thesis_Repo root, barnard):
    sbatch -p barnard -c 8 --mem=128G -t 02:00:00 -o logs/dispatch_diag_%j.out --wrap "module --force purge; \
        module load release/2026 GCC/14.2.0; source ~/thesis_env/bin/activate; \
        python -u exp1/generate_metrics/dispatch_diag.py --case case500_goc"
"""

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr

from expost_pf import FORECASTS, MLRUNS, PHASE1A, PHASE1C, TOL_P, TOL_REL, _dense, _read, load_exact, load_ml, load_truth

REPO = Path(__file__).resolve().parents[2]
OUT = PHASE1C / "dispatch_diag"
E2E_RUNS = {  # final base E2E (exp final_e2e_qgmse; state dumps written in-run)
    "case14_ieee": ("final_e2e_qgmse", "base_e2e_case14_h{H}_s0"),
    "case118_ieee": ("final_e2e_qgmse", "base_e2e_case118_h{H}_s0"),
    "case500_goc": ("final_e2e_qgmse", "base_e2e_case500_goc_h{H}_s0"),
}
PIN = 0.99  # unit class by share of test hours at a P limit: pinned ≥ PIN, free ≤ 1 − PIN, else switching
SEP = 1000  # flat sample key = origin * SEP + step (H < SEP)
SURFACE, INK, MUTED, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"


# ---------------------------------------------------------------- data

def truth(case):
    """GT OPF labels, all hours: Pd, D, Pg, unit limits + cost type, binding lines, bus hop distances"""
    hb, t = load_truth(case, np.array([0, 10**7]))
    raw = PHASE1A / case / "raw"
    g = _read(raw / "gen_data.parquet", ["idx", "min_p_mw", "max_p_mw", "in_service"],
              ds.field("load_scenario_idx") == float(hb[0]))
    g = g[g.in_service == 1].sort_values("idx")
    assert len(g) == t["pg"].shape[1]
    br = _read(raw / "branch_data.parquet",
               ["load_scenario_idx", "idx", "from_bus", "to_bus", "pf", "qf", "pt", "qt", "rate_a", "br_status"])
    on = (br.br_status == 1) & (br.rate_a > 0)
    s = np.maximum(np.hypot(br.pf, br.qf), np.hypot(br.pt, br.qt))
    br["loading"] = np.where(on, s / br.rate_a.where(on, 1.0), 0.0)
    hl, (loading,) = _dense(br, "load_scenario_idx", "idx", "loading")
    assert np.array_equal(hb, hl)
    e = br[(br.load_scenario_idx == hb[0]) & (br.br_status == 1)]
    n_bus = t["pd"].shape[1]
    adj = coo_matrix((np.ones(len(e)), (e.from_bus.astype(int), e.to_bus.astype(int))), shape=(n_bus, n_bus))
    pos = np.full(hb.max() + 1, -1)
    pos[hb] = np.arange(len(hb))
    return dict(hours=hb, pos=pos, pd=t["pd"], D=t["pd"].sum(1), pg=t["pg"], idx=g.idx.to_numpy().astype(int),
                pmin=g.min_p_mw.to_numpy(), pmax=g.max_p_mw.to_numpy(), linear=t["c2"] == 0,
                c0=t["c0"], c1=t["c1"], c2=t["c2"],
                binding=loading >= 1 - TOL_REL, hops=shortest_path(adj, directed=False, unweighted=True))


def rows_of(gt, hours):
    r = gt["pos"][hours]
    assert (r >= 0).all(), "sample target hour missing from GT"
    return r


def forecasts(case, H, cols, root=FORECASTS):
    """2-step forecast parquet -> flat keys (sorted = 2-step flat order) + one [S, n_bus] array per col"""
    df = pd.read_parquet(root / f"{case}_horizon{H}.parquet",
                         columns=["load_scenario_idx", "bus_id", "horizon_step", *cols])
    df["key"] = df.load_scenario_idx.astype(np.int64) * SEP + df.horizon_step.astype(np.int64)
    key, arrs = _dense(df, "key", "bus_id", *cols)
    return key, dict(zip(cols, arrs))


def e2e_run(case, H):
    """run dir by (experiment name, run name); must be unique"""
    exp, name = E2E_RUNS[case][0], E2E_RUNS[case][1].format(H=H)
    ids = [Path(m).parent for m in glob.glob(f"{MLRUNS}/*/meta.yaml")
           if f"name: {exp}" in open(m).read().splitlines()]
    runs = [Path(t).parents[1] for i in ids for t in glob.glob(f"{i}/*/tags/mlflow.runName")
            if open(t).read().strip() == name]
    assert len(runs) == 1, f"{exp}/{name}: {runs}"
    return runs[0]


def logged(case, H, arm, run=None):
    """(Pg MAE, Pg RMSE, gap %) from the logged metrics CSV (E2E run / compare.py); NaN unless exactly one file"""
    if arm == "e2e":
        fs, pre = [run / f"artifacts/test/{case}_metrics.csv"], ""
    else:
        fc = arm.split("_", 1)[1]
        fs = [f for pat in (f"{case}_horizon{H}", f"{case}_horizon{H}_*")
              for f in glob.glob(str(REPO / f"exp1/results/{pat}/{fc}/{case}_metrics.csv"))]
        pre = "AC "
    if len(fs) != 1:
        print(f"  gate {arm} h{H}: {len(fs)} logged CSVs {fs}")
        return np.nan, np.nan, np.nan
    m = pd.read_csv(fs[0]).set_index("Metric")["Value"]
    return tuple(float(m[f"{pre}{k}"]) for k in ("Generator Pg MAE", "Generator Pg RMSE", "Mean Optimality Gap"))


# ---------------------------------------------------------------- step 1

def unit_state(pg, pmin, pmax):
    """0 free, 1 at Pmin, 2 at Pmax (within TOL_P)"""
    return np.where(pg <= pmin + TOL_P, 1, np.where(pg >= pmax - TOL_P, 2, 0)).astype(np.int8)


def cost(gt, pg):
    return (gt["c0"] + gt["c1"] * pg + gt["c2"] * pg ** 2).sum(1)


def step1(gt, units, H, arm, pg, target, dD, gate):
    """Pg error by unit group + total vs misplaced MW for one arm"""
    true = gt["pg"][rows_of(gt, target)]
    e = pg - true
    ae = np.abs(e)
    ctype = np.where(units.linear, "linear", "quadratic")
    groups = []
    for by, lab in (("cls", units.cls.to_numpy()), ("cost", ctype), ("cls_cost", (units.cls + "/" + ctype).to_numpy())):
        for v in np.unique(lab):
            m = lab == v
            groups.append(dict(H=H, arm=arm, by=by, group=v, n_units=int(m.sum()), mae=ae[:, m].mean(),
                               rmse=np.sqrt((e[:, m] ** 2).mean()), mae_contrib=ae[:, m].sum() / e.size))
    E = e.sum(1)
    flips = (unit_state(pg, gt["pmin"], gt["pmax"]) != unit_state(true, gt["pmin"], gt["pmax"]))
    two_step = arm != "e2e"  # E2E never lands exactly on a limit -> flips meaningless
    split = dict(H=H, arm=arm, n_samples=len(e), pg_mae=ae.mean(), pg_rmse=np.sqrt((e ** 2).mean()),
                 logged_mae=gate[0], logged_rmse=gate[1],
                 total_abs_mw=np.abs(E).mean(), misplaced_mw=((ae.sum(1) - np.abs(E)) / 2).mean(),
                 total_bias_mw=E.mean(), dD_abs_mw=np.abs(dD).mean(), dD_bias_mw=dD.mean(),
                 flip_share=flips.any(1).mean() if two_step else np.nan,
                 flips_per_sample=flips.sum(1).mean() if two_step else np.nan)
    T = gt["pg"].sum(1)
    o = np.argsort(T)
    j = np.clip(np.searchsorted(T[o], pg.sum(1)), 1, len(o) - 1)
    j = o[np.where(np.abs(T[o][j - 1] - pg.sum(1)) < np.abs(T[o][j] - pg.sum(1)), j - 1, j)]
    c_star, c_hat, c_t = cost(gt, true), cost(gt, pg), cost(gt, gt["pg"][j])
    alloc = (c_hat - c_t) / c_star * 100
    eT = gt["pg"][j] - true
    split.update(pg_mae_opfsplit=np.abs(eT).mean(), pg_rmse_opfsplit=np.sqrt((eT ** 2).mean()), gap_raw=np.mean(np.abs(c_hat - c_star) / c_star) * 100, logged_gap=gate[2],
                 gap_alloc=alloc.mean(), gap_alloc_neg_share=(alloc < 0).mean(),
                 gap_volume=np.mean(np.abs(c_t - c_star) / c_star) * 100, nn_total_err_mw=np.abs(T[j] - pg.sum(1)).mean())
    ok = np.isclose(split["pg_mae"], gate[0], rtol=1e-3) and np.isclose(split["pg_rmse"], gate[1], rtol=1e-3)
    print(f"  gate {arm} h{H}: Pg MAE {split['pg_mae']:.4f} (logged {gate[0]:.4f}), "
          f"RMSE {split['pg_rmse']:.4f} (logged {gate[1]:.4f}) -> {'ok' if ok else 'MISMATCH'}")
    return groups, split


# ---------------------------------------------------------------- step 2

def loo_fit(x, Y, half):
    """leave-one-out local-linear fit of each column of Y on x over `half` ranks of x on each side"""
    o = np.argsort(x, kind="stable")
    xs, Ys = x[o] - x.mean(), Y[o] - Y.mean(0)
    n = len(xs)
    i = np.arange(n)
    lo, hi = np.maximum(i - half, 0), np.minimum(i + half + 1, n)

    def wsum(a):  # window sum without the point itself
        c = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, 0)])
        return c[hi] - c[lo] - a

    m, sx, sxx = wsum(np.ones(n)), wsum(xs), wsum(xs * xs)
    sy, sxy = wsum(Ys), wsum(xs[:, None] * Ys)
    den = m * sxx - sx ** 2
    b = (m[:, None] * sxy - sx[:, None] * sy) / np.where(den > 0, den, np.inf)[:, None]
    fit = np.empty_like(Ys)
    fit[o] = (sy - b * sx[:, None]) / m[:, None] + b * xs[:, None] + Y.mean(0)
    return fit


def step2(gt, test):
    """per-unit table (classes from test hours, R² on D) + summary per period"""
    st = unit_state(gt["pg"], gt["pmin"], gt["pmax"])
    cong = gt["binding"].any(1)
    units = pd.DataFrame(dict(idx=gt["idx"], linear=gt["linear"], pmin=gt["pmin"], pmax=gt["pmax"],
                              at_min=(st[test] == 1).mean(0), at_max=(st[test] == 2).mean(0), std_pg=gt["pg"].std(0)))
    pin = units.at_min + units.at_max
    units["cls"] = np.select([pin >= PIN, pin <= 1 - PIN], ["pinned", "free"], "switching")
    rows = []
    for name, m in (("all", np.ones(len(cong), bool)), ("test", test)):
        pg = gt["pg"][m]
        res = pg - loo_fit(gt["D"][m], pg, half=max(12, int(m.sum()) // 400))
        ss_res, ss_tot = (res ** 2).sum(0), ((pg - pg.mean(0)) ** 2).sum(0)
        units[f"r2_{name}"] = 1 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan)
        moving = ss_tot / m.sum() > 0.01  # std > 0.1 MW
        s, b = st[m], gt["binding"][m]
        consec = np.diff(gt["hours"][m]) == 1
        cnt = np.sort(np.unique(s, axis=0, return_counts=True)[1])[::-1]
        rows.append(dict(
            period=name, n_hours=int(m.sum()), n_units=len(units), n_units_moving=int(moving.sum()),
            r2_pooled=1 - ss_res.sum() / ss_tot.sum(),
            share_moving_r2_099=(units[f"r2_{name}"][moving] >= 0.99).mean(),
            share_moving_r2_09=(units[f"r2_{name}"][moving] >= 0.9).mean(),
            congested_hours=cong[m].mean(), unexplained_in_congested=(res[cong[m]] ** 2).sum() / ss_res.sum(),
            free_units_mean=(s == 0).sum(1).mean(), free_units_max=int((s == 0).sum(1).max()),
            unit_patterns=len(cnt), unit_top10_cover=cnt[:10].sum() / cnt.sum(),
            unit_switch_rate=(s[1:] != s[:-1]).any(1)[consec].mean(),
            binding_lines_mean=b.sum(1).mean(), line_patterns=len(np.unique(b, axis=0)),
            line_switch_rate=(b[1:] != b[:-1]).any(1)[consec].mean()))
    return units, pd.DataFrame(rows), cong


def plot_units(gt, units, cong, path, case):
    """Pg vs D (all hours) for the 3 largest-variance and the 3 least-explained moving units"""
    mv = units[units.std_pg > 0.1]
    top = list(mv.nlargest(3, "std_pg").index)
    unexp = ((1 - mv.r2_all) * mv.std_pg ** 2).sort_values(ascending=False)
    top += [u for u in unexp.index if u not in top][:3]
    fit = loo_fit(gt["D"], gt["pg"][:, top], half=max(12, len(gt["D"]) // 400))
    o = np.argsort(gt["D"])
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5), facecolor=SURFACE, sharex=True)
    for k, (ax, u) in enumerate(zip(axs.flat, top)):
        ax.set_facecolor(SURFACE)
        for m, c, lab in ((~cong, BLUE, "no line at its limit"), (cong, ORANGE, "≥ 1 line at its limit")):
            ax.scatter(gt["D"][m], gt["pg"][m, u], s=2, c=c, alpha=0.35, lw=0, label=lab, rasterized=True)
        ax.plot(gt["D"][o], fit[o, k], c=INK, lw=1, label="local-linear fit f(D)")
        for lim in (units.pmin[u], units.pmax[u]):
            ax.axhline(lim, c=MUTED, lw=0.8, ls="--")
        ax.set_title(f"unit {units.idx[u]} ({'linear' if units.linear[u] else 'quadratic'} cost), "
                     f"R² = {units.r2_all[u]:.3f}", fontsize=10, color=INK)
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(MUTED)
    for ax in axs[1]:
        ax.set_xlabel("total demand D (MW)", color=INK)
    for ax in axs[:, 0]:
        ax.set_ylabel("Pg (MW)", color=INK)
    h, lab = axs.flat[0].get_legend_handles_labels()
    fig.suptitle(f"{case}: unit output vs total demand, all hours (dashed: Pmin / Pmax)\n"
                 "top: largest variance; bottom: least explained by D", color=INK, y=0.995)
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=3, frameon=False, markerscale=4,
               labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- step 3

def corr_stats(X):
    """X [S, n] -> mean off-diagonal correlation, PC1 / PC1-3 variance share, correlation matrix"""
    Z = (X - X.mean(0)) / X.std(0)
    C = Z.T @ Z / len(Z)
    ev = np.linalg.eigvalsh(C)[::-1] / len(C)
    return dict(mean_corr=C[~np.eye(len(C), dtype=bool)].mean(), pc1=ev[0], pc1_3=ev[:3].sum()), C


def colcorr(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))


def step3(gt, H, fc, true, pred, key, load):
    """cross-bus error correlation of one forecaster at one horizon"""
    e, x = (pred - true)[:, load], true[:, load]
    se, C = corr_stats(e)
    sr, Cr = corr_stats(x)
    dD = e.sum(1)
    row = dict(H=H, fc=fc, n_load=int(load.sum()), n_samples=len(e), **se,
               survive=np.abs(e.sum(1)).mean() / np.abs(e).sum(1).mean(),
               cancel_ratio=dD.var() / e.var(0).sum(), dD_mae_mw=np.abs(dD).mean(), dD_rmse_mw=np.sqrt((dD ** 2).mean()),
               raw_mean_corr=sr["mean_corr"], raw_pc1=sr["pc1"])
    if H == 1:  # lagged: does the other buses' last error help beyond the own last error?
        Z = (e - e.mean(0)) / e.std(0)
        consec = np.diff(key // SEP) == 1
        n = Z.shape[1]
        y, x1 = Z[1:][consec], Z[:-1][consec]
        x2 = (x1.sum(1, keepdims=True) - x1) / (n - 1)
        r1, r2, r12 = colcorr(y, x1), colcorr(y, x2), colcorr(x1, x2)
        both = (r1 ** 2 + r2 ** 2 - 2 * r1 * r2 * r12) / (1 - r12 ** 2)
        row.update(lag_r2_own=(r1 ** 2).mean(), lag_r2_others=(r2 ** 2).mean(), lag_r2_both=both.mean(),
                   lag_share_gain_001=(both - r1 ** 2 > 0.01).mean(),
                   lag_r2_total=np.corrcoef(dD[1:][consec], dD[:-1][consec])[0, 1] ** 2)
    iu = np.triu_indices(int(load.sum()), 1)
    d = gt["hops"][np.ix_(load, load)][iu]
    f = np.isfinite(d)
    hop = pd.DataFrame(dict(hops=np.minimum(d[f], 10).astype(int), corr_err=C[iu][f], corr_raw=Cr[iu][f]))
    row.update(spearman_hop_err=spearmanr(hop.hops, hop.corr_err)[0], spearman_hop_raw=spearmanr(hop.hops, hop.corr_raw)[0])
    hop = hop.groupby("hops").agg(n_pairs=("corr_err", "size"), corr_err=("corr_err", "mean"),
                                  corr_raw=("corr_raw", "mean")).reset_index()
    hop.insert(0, "fc", fc)
    hop.insert(0, "H", H)
    return row, hop


# ---------------------------------------------------------------- step 4

def between_share(x, groups):
    """share of Var(x) explained by its group means"""
    return pd.Series(x).groupby(np.asarray(groups)).transform("mean").var() / x.var()


def step4(gt, H, fc, D_hat, k, target, pg_ac, n_q=41):
    """plug-in vs hedged dispatch from the GT curve f(D); pg_ac = actual AC 2-step dispatch (proxy check)"""
    o = np.argsort(gt["D"])
    Ds = gt["D"][o]

    def f(d):  # GT dispatch of the hour with the nearest total demand
        j = np.clip(np.searchsorted(Ds, d), 1, len(Ds) - 1)
        return gt["pg"][o[np.where(np.abs(Ds[j - 1] - d) < np.abs(Ds[j] - d), j - 1, j)]]

    r = rows_of(gt, target)
    true, eps = gt["pg"][r], gt["D"][r] - D_hat
    plug = f(D_hat)
    hedge, hedge_b = np.empty_like(plug), np.empty_like(plug)
    for kk in np.unique(k):
        m = k == kk
        q = np.quantile(eps[m], (np.arange(n_q) + 0.5) / n_q)
        hedge[m] = sum(f(D_hat[m] + e) for e in q - eps[m].mean()) / n_q
        hedge_b[m] = sum(f(D_hat[m] + e) for e in q) / n_q
    row = dict(H=H, fc=fc, n_samples=len(true))
    for name, p in (("ac", pg_ac), ("plug", plug), ("hedge", hedge), ("hedge_bias", hedge_b)):
        row.update({f"{name}_mae": np.abs(p - true).mean(), f"{name}_rmse": np.sqrt(((p - true) ** 2).mean())})
    delta = ((hedge - plug) ** 2).mean()
    row.update(plug_vs_ac_rmse=np.sqrt(((plug - pg_ac) ** 2).mean()), delta_ms=delta,
               cross_ms=((plug - true) ** 2).mean() - ((hedge - true) ** 2).mean() - delta,
               eps_bias=eps.mean(), eps_std=eps.std(),
               **{f"eps_r2_{n}": between_share(eps, g) for n, g in
                  (("dhat_decile", pd.qcut(D_hat, 10, labels=False)), ("hour", target % 24), ("step", k))})
    return row


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", required=True, choices=list(E2E_RUNS))
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 24])
    p.add_argument("--forecasters", nargs="+", default=["xgb", "sarima"])
    p.add_argument("--out-dir", type=Path, default=OUT)
    a = p.parse_args()
    out = a.out_dir / a.case
    out.mkdir(parents=True, exist_ok=True)

    gt = truth(a.case)
    h1 = pd.read_parquet(FORECASTS / f"{a.case}_horizon1.parquet", columns=["load_scenario_idx", "horizon_step"])
    test = np.zeros(len(gt["hours"]), bool)
    test[rows_of(gt, np.unique(h1.load_scenario_idx + h1.horizon_step).astype(np.int64))] = True
    units, s2, cong = step2(gt, test)
    plot_units(gt, units, cong, out / "pg_vs_D.png", a.case)
    print(f"[{a.case}] step 2: {len(gt['hours'])} hours, {test.sum()} test hours\n{s2.T.to_string()}")
    print(pd.crosstab(units.cls, np.where(units.linear, "linear", "quadratic")).to_string())

    groups, splits, corr, hops, jensen = [], [], [], [], []
    for H in a.horizons:
        key, f = forecasts(a.case, H, ["true", *a.forecasters])
        target = key // SEP + key % SEP
        dev = np.abs(f["true"] - gt["pd"][rows_of(gt, target)]).max()
        print(f"  h{H}: max |forecast 'true' − GT Pd| {dev:.2e} MW")
        assert dev < 1e-2, "forecast parquet not built from this GT"
        load = f["true"].std(0) > 0
        for fc in a.forecasters:
            r, h = step3(gt, H, fc, f["true"], f[fc], key, load)
            corr.append(r)
            hops.append(h)
            x = load_exact(a.case, H, fc)
            dD = (f[fc] - f["true"]).sum(1)
            g, s = step1(gt, units, H, f"ac_{fc}", x["pg"], x["target"], dD, logged(a.case, H, f"ac_{fc}"))
            groups += g
            splits.append(s)
            kx = (x["target"] - x["k"]) * SEP + x["k"]
            rows = np.searchsorted(key, kx)
            assert np.array_equal(key[rows], kx)
            jensen.append(step4(gt, H, fc, f[fc][rows].sum(1), x["k"], x["target"], x["pg"]))
        del f
        run = e2e_run(a.case, H)
        x = load_ml("e2e", a.case, H, run)
        _, (pd_pred,) = _dense(pq.read_table(run / f"artifacts/test/{a.case}_state_bus.parquet",
                                             columns=["scenario", "bus", "pd_pred"]).to_pandas(), "scenario", "bus", "pd_pred")
        _, (pg_true,) = _dense(pq.read_table(run / f"artifacts/test/{a.case}_state_gen.parquet",
                                             columns=["scenario", "gen", "pg_true"]).to_pandas(), "scenario", "gen", "pg_true")
        r = rows_of(gt, x["target"])
        print(f"  E2E h{H} {run.parent.name}/{run.name}: max |pd_true − GT| {np.abs(x['pd_true'] - gt['pd'][r]).max():.2e}, "
              f"max |pg_true − GT| {np.abs(pg_true - gt['pg'][r]).max():.2e} MW")
        g, s = step1(gt, units, H, "e2e", x["pg"], x["target"], (pd_pred - x["pd_true"]).sum(1),
                     logged(a.case, H, "e2e", run))
        groups += g
        splits.append(s)
        k0 = (x["target"] - x["k"]) * SEP + x["k"]  # origin * SEP + step, like the 2-step keys
        o = np.argsort(k0, kind="stable")
        r, h = step3(gt, H, "e2e", x["pd_true"][o], pd_pred[o], k0[o], load)
        corr.append(r)
        hops.append(h)

    tables = dict(step1_groups=pd.DataFrame(groups), step1_split=pd.DataFrame(splits), step2_units=units,
                  step2_summary=s2, step3_corr=pd.DataFrame(corr), step3_hop=pd.concat(hops),
                  step4_jensen=pd.DataFrame(jensen))
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.4g}".format):
        for name in ("step1_split", "step3_corr", "step4_jensen"):
            print(f"\n[{a.case}] {name}\n{tables[name].to_string(index=False)}")
        g = tables["step1_groups"]
        print(f"\n[{a.case}] step1 mae_contrib by unit class\n"
              f"{g[g.by == 'cls_cost'].pivot_table(index=['H', 'arm'], columns='group', values='mae_contrib').to_string()}")
        print(f"\n[{a.case}] step1 MAE by unit class\n"
              f"{g[g.by == 'cls_cost'].pivot_table(index=['H', 'arm'], columns='group', values='mae').to_string()}")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
