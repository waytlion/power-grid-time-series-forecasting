"""Signed error distributions (pred - OPF on true load): plot_rmse.py grid layout, step
histograms per cell like figures/residual_hist_h24.pdf (one line per arm, density, focused x-view).
  1) state_dist_<case>: rows = Pg (per generator), Vm, Va; cols = horizon; one histogram per arm
     Pg excludes synchronous condensers (Pmax = 0: can never inject active power, error identically ~0)
  2) state_dist_pd:     rows = case; cols = horizon; 2-step forecast (best forecaster, shared by AC/DC/surrogate,
     grey) vs E2E
  3) dist_pd_pg_case500 (Q1 main text): rows = Pd, Pg (all arms); linear density, x-view = 99th pct of |err| of the
     main comparison (2-step forecast / AC, E2E); every arm's "% in view" in the legend
  4) dist_pd_pg_vm_va_<case>_full (appendix): rows = Pd, Pg, Vm, Va; whole range, log density
  density always w.r.t. the full sample (area in view = share in view: clipping the view drops no data)
  Pd: load buses only (true Pd > LOAD_TOL); zero-load buses (Pd = 0 by definition) excluded
  legend per panel: arm + RMSE + share of samples in view; density is w.r.t. the full sample
  (area in view = that share), x-view = see hist_panel.
  arms + forecaster per (case, horizon, metric) = exactly those plot_rmse.py shows (best by RMSE)
  2-step AC/DC  <- datakit phase_1c raw parquets (forecaster arm vs `true` arm)
  Surrogate/E2E <- <case>_state_{bus,gen}.parquet (scripts/state_dump_eval.sbatch; new runs write it)
  runs pinned: E2E via dispatch_diag.e2e_run, surrogate via expost_pf.resolve_run (newest-by-name over all
  experiments picked stale / ablation runs)
Run from repo root on a compute node (case500 h24 raw parquets are large).
"""
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from colors import ARM_COLORS, OPT_LABELS, OPT_ORDER
from plot_rmse import (h_label, CASE_LABELS, CASE_ORDER, HORIZONS, best_forecaster_per_metric,
                       best_forecaster_pd, load, save)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generate_metrics"))
from dispatch_diag import e2e_run  # noqa: E402  final E2E run per (case, horizon)
from expost_pf import resolve_run  # noqa: E402  surrogate run within the thesis experiments

DATA_OUT = Path("/data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1c/data_out")
LOAD_TOL = 1e-3  # MW; datakit stores zero-load buses as ~1e-7 noise (dumps: exact 0), real loads > 0.1 MW
VARS = [("pg", "Generator Pg RMSE", "$P_g$ error ($P_{max}>0$)", "MW"),
        ("vm", "Vm RMSE", "$V_m$ error", "p.u."),
        ("va", "Va RMSE", "$V_a$ error", "deg")]


def datakit_raw(case, h, arm):
    d = DATA_OUT / f"{case}_horizon_{h}" / arm / case / "raw"
    if not d.exists():  # e.g. case14 h1 xgb only exists in the archived run generation
        d = DATA_OUT / "archiv" / f"{case}_horizon_{h}_3yr" / arm / case / "raw"
        print(f"  note: {case} h{h} {arm} read from {d}")
    return d


@lru_cache(None)
def active_gens(case, h):
    """gen idx with Pmax > 0 (static); ML dumps order gens by the same idx (verified)"""
    g = pd.read_parquet(datakit_raw(case, h, "true") / "gen_data.parquet", columns=["idx", "max_p_mw"])
    pmax = g.groupby("idx")["max_p_mw"].first()
    return np.flatnonzero(pmax.to_numpy() > 0)


@lru_cache(None)
def two_step(case, h, fc):
    """-> {(var, optimizer): signed error array} for 2-step AC/DC with forecaster fc"""
    raw = lambda arm, f, cols: pd.read_parquet(datakit_raw(case, h, arm) / f, columns=cols)
    kg, kb = ["load_scenario_idx", "idx"], ["load_scenario_idx", "bus"]
    g = raw(fc, "gen_data.parquet", kg + ["p_mw", "p_mw_dc"]).merge(
        raw("true", "gen_data.parquet", kg + ["p_mw"]), on=kg, suffixes=("", "_t"))
    g = g[g["idx"].isin(np.sort(g["idx"].unique())[active_gens(case, h)])]
    b = raw(fc, "bus_data.parquet", kb + ["Pd", "Vm", "Va", "Va_dc"]).merge(
        raw("true", "bus_data.parquet", kb + ["Pd", "Vm", "Va"]), on=kb, suffixes=("", "_t"))
    ld = b["bus"].isin(b.loc[b.Pd_t.abs() > LOAD_TOL, "bus"].unique())  # load buses
    return {("pd", "2step"): (b.Pd - b.Pd_t)[ld].to_numpy(),
            ("pg", "ac"): (g.p_mw - g.p_mw_t).to_numpy(), ("pg", "dc"): (g.p_mw_dc - g.p_mw_t).to_numpy(),
            ("vm", "ac"): (b.Vm - b.Vm_t).to_numpy(),
            ("va", "ac"): (b.Va - b.Va_t).to_numpy(), ("va", "dc"): (b.Va_dc - b.Va_t).to_numpy()}


@lru_cache(None)
def ml(case, h, opt, fc):
    """-> {var: signed error array}; surrogate: newest state dump of its canonical run name within
    expost_pf.RUN_EXPERIMENTS (None if not dumped yet), E2E: pinned final run"""
    run = e2e_run(case, h) if opt == "e2e" else resolve_run([f"surrogate_{case}_{fc}_h{h}_s0"], case)
    if run is None:
        print(f"  missing state dump: surrogate_{case}_{fc}_h{h}_s0 -> run scripts/state_dump_eval.sbatch")
        return None
    d = run / "artifacts" / "test"
    b, g = pd.read_parquet(f"{d}/{case}_state_bus.parquet"), pd.read_parquet(f"{d}/{case}_state_gen.parquet")
    g = g[g["gen"].isin(active_gens(case, h))]
    ld = b["bus"].isin(b.loc[b.pd_true.abs() > LOAD_TOL, "bus"].unique())  # load buses
    return {"pd": (b.pd_pred - b.pd_true)[ld].to_numpy(), "pg": (g.pg_pred - g.pg_true).to_numpy(),
            "vm": (b.vm_pred - b.vm_true).to_numpy(), "va": (b.va_deg_pred - b.va_deg_true).to_numpy()}


def err(case, h, opt, fc, var):
    if opt in ("ac", "dc"):
        return two_step(case, h, fc).get((var, opt))
    arm = ml(case, h, opt, fc)
    return arm and arm[var]


def hist_panel(ax, series, unit, focus=None, logy=False, nbins=61):
    """series: list of (label, array, color). Step histogram, density, x-view = LOWER median (the
    smaller of the two middle values on an even count, not their average) across arms of each arm's
    own 75th-pct |err|. Two deliberate choices over the naive "pooled 99th pct" view:
      - per-arm, not pooled: a single heavy-tailed arm (DC) must not set the scale for everyone else,
        or every other arm's curve goes flat at the center.
      - LOWER median of p75 (not the statistical median of p90): plain np.median on an even-length
        list averages the two middle values, which still gets dragged wide by any one arm that's
        heavier-tailed without being the single worst outlier (e.g. Surrogate on Pg) -- that average
        was still showing a dominant central spike with little else visible. Taking the smaller of
        the two middle values, at the tighter p75, keeps the view on the bulk of the TYPICAL arms and
        reveals real shape differences between them (e.g. Surrogate/E2E's distinct humps on Pg).
    Any arm's tail that runs past the view just gets cut off at the edge; its RMSE is still in the legend.
    - focus: fixed half-width of the x-view (skips the automatic view); logy: log density axis"""
    # Escalate the percentile tier only if needed: some arms (e.g. an exact solver matching the
    # true dispatch almost everywhere, only deviating in a long tail) have >75% of |err| at
    # effectively 0, which collapses the lower-median to 0 -- a degenerate zero-width bin range
    # that silently empties every OTHER arm's view (0/0 in the density normalization). Requiring
    # only >=1 point in view is nearly vacuous (any arm with even one near-zero observation among
    # millions trivially passes), so require a REAL fraction of each arm's data in view, not just
    # a stray point -- keep the tightest percentile tier that still clears that bar for everyone.
    MIN_FRAC_IN_VIEW = 0.05
    for pct in (75, 80, 85, 90, 95, 99, 99.9) if focus is None else ():
        p = sorted(np.percentile(np.abs(v), pct) for _, v, _ in series)
        cand = p[(len(p) - 1) // 2]
        if cand > 0 and all(np.mean(np.abs(v) <= cand) >= MIN_FRAC_IN_VIEW for _, v, _ in series):
            focus = cand
            break
    focus = focus or 1.0
    bins = np.linspace(-focus, focus, nbins)
    bw = bins[1] - bins[0]
    for label, v, color in series:
        rmse = np.sqrt(np.mean(v ** 2))
        fmt = f"{rmse:.4f}" if unit == "p.u." else f"{rmse:.2f}"
        x = v[np.abs(v) <= focus]
        # density w.r.t. the FULL sample (area in view = fraction in view), not density=True,
        # which renormalizes the clipped subset to area 1 and inflates arms with big hidden tails
        ax.hist(x, bins=bins, weights=np.full(len(x), 1.0 / (len(v) * bw)), histtype="step", lw=1.4,
                color=color, label=f"{label} (RMSE {fmt}, {len(x) / len(v):.0%} in view)")
    ax.axvline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_xlim(-focus, focus)
    if logy:
        ax.set_yscale("log")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * (hi / lo) ** 0.6)  # same headroom on the log axis
    else:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.6)  # headroom: legend sits above the central peak
    ax.legend(frameon=False, fontsize=5.5, loc="upper right", handlelength=1.2, borderaxespad=0.2)


def arm_series(best, case, h, var):
    """-> [(label, signed error, color)] per optimizer, forecaster from best (best_forecaster_per_metric rows)"""
    fc = best[(best["case"] == case) & (best["horizon"] == h)].set_index("optimizer")["forecaster"]
    series = [(OPT_LABELS[o], err(case, h, o, fc[o], var), ARM_COLORS[o]) for o in OPT_ORDER if o in fc.index]
    return [(l, v, c) for l, v, c in series if v is not None]


def pd_series(data, case, h):
    """-> [(label, signed Pd error, color)]: 2-step forecast (best forecaster, shared by all 2-step arms) vs E2E"""
    fc = data[(data["case"] == case) & (data["horizon"] == h)].set_index("series")["forecaster"]
    series = [("2-step forecast", two_step(case, h, fc["2-step (best forecaster)"])[("pd", "2step")], "0.45"),
              ("E2E", err(case, h, "e2e", "none", "pd"), ARM_COLORS["e2e"])]
    return [(l, v, c) for l, v, c in series if v is not None]


def plot_grid(df):
    best = {metric: best_forecaster_per_metric(df, metric) for _, metric, _, _ in VARS}
    for case in CASE_ORDER:
        fig, axes = plt.subplots(len(VARS), len(HORIZONS), figsize=(11, 7.5), squeeze=False)
        for r, (var, metric, label, unit) in enumerate(VARS):
            for c, h in enumerate(HORIZONS):
                series = arm_series(best[metric], case, h, var)
                ax = axes[r][c]
                if series:
                    hist_panel(ax, series, unit)
                if r == 0:
                    ax.set_title(f"{h_label(h)}")
                if c == 0:
                    ax.set_ylabel("Density")
                ax.set_xlabel(f"{label}  [{unit}]")
        fig.suptitle(f"Signed error distribution -- {CASE_LABELS[case]} (2-step: best forecaster per metric). "
                     "$P_g$: synchronous condensers ($P_{max}=0$) excluded", y=1.01)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        save(fig, f"state_dist_{CASE_LABELS[case]}")


def plot_pd(df):
    data = best_forecaster_pd(df)
    fig, axes = plt.subplots(len(CASE_ORDER), len(HORIZONS), figsize=(11, 7.5), squeeze=False)
    for r, case in enumerate(CASE_ORDER):
        for c, h in enumerate(HORIZONS):
            series = pd_series(data, case, h)
            ax = axes[r][c]
            if series:
                hist_panel(ax, series, "MW")
            if r == 0:
                ax.set_title(f"{h_label(h)}")
            if c == 0:
                ax.set_ylabel(f"{CASE_LABELS[case]}\nDensity")
            ax.set_xlabel("$P_d$ error [MW]")
    fig.suptitle("$P_d$ forecast error distribution by grid size and horizon", y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "state_dist_pd")


MAIN_ARMS = ("2-step forecast", OPT_LABELS["ac"], "E2E")  # main comparison: sets the x-view of the clipped figure


def plot_dist(df, case, variables, full):
    """rows = variables (subset of pd, pg, vm, va), cols = horizon
    - full=False: linear density, x-view = 99th pct of |err| of MAIN_ARMS (other arms: "% in view" in the legend)
    - full=True: whole range, log density"""
    best = {v: best_forecaster_per_metric(df, m) for v, m, _, _ in VARS}
    rows = {"pd": ("$P_d$ error (load buses)", "MW", lambda h: pd_series(best_forecaster_pd(df), case, h))}
    rows |= {v: (label, unit, lambda h, v=v: arm_series(best[v], case, h, v)) for v, _, label, unit in VARS}
    fig, axes = plt.subplots(len(variables), len(HORIZONS), figsize=(11, 2.7 * len(variables)), squeeze=False)
    for r, var in enumerate(variables):
        label, unit, series_of = rows[var]
        for c, h in enumerate(HORIZONS):
            series, ax = series_of(h), axes[r][c]
            a = {l: np.abs(v) for l, v, _ in series}
            focus = max(x.max() for x in a.values()) if full else \
                max(np.percentile(x, 99) for l, x in a.items() if l in MAIN_ARMS)
            hist_panel(ax, series, unit, focus=focus, logy=full, nbins=121 if full else 61)
            if r == 0:
                ax.set_title(h_label(h))
            if c == 0:
                ax.set_ylabel("Density" + (" (log)" if full else ""))
            ax.set_xlabel(f"{label}  [{unit}]")
    fig.tight_layout()
    save(fig, f"dist_{'_'.join(variables)}_{CASE_LABELS[case]}" + ("_full" if full else ""))


def main():
    df = load()
    plot_grid(df)
    plot_pd(df)
    plot_dist(df, "case500_goc", ("pd", "pg"), full=False)  # main text (Q1)
    for case in CASE_ORDER:  # appendix
        plot_dist(df, case, ("pd", "pg", "vm", "va"), full=True)


if __name__ == "__main__":
    main()
