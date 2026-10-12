"""Signed error distributions (pred - OPF on true load): plot_rmse.py grid layout, step
histograms per cell like figures/residual_hist_h24.pdf (one line per arm, density, focused x-view).
  1) state_dist_<case>: rows = Pg (per generator), Vm, Va; cols = horizon; one histogram per arm
     Pg excludes synchronous condensers (Pmax = 0: can never inject active power, error identically ~0)
  2) state_dist_pd:     rows = case; cols = horizon; 2-step (best forecaster) vs E2E
  legend per panel: arm + RMSE + share of samples in view; density is w.r.t. the full sample
  (area in view = that share), x-view = see hist_panel.
  arms + forecaster per (case, horizon, metric) = exactly those plot_rmse.py shows (best by RMSE)
  2-step AC/DC  <- datakit phase_1c raw parquets (forecaster arm vs `true` arm)
  Surrogate/E2E <- <case>_state_{bus,gen}.parquet (scripts/state_dump_eval.sbatch; new runs write it)
Run from repo root on a compute node (case500 h24 raw parquets are large).
"""
import glob
import os
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from colors import ARM_COLORS, OPT_LABELS, OPT_ORDER
from plot_rmse import (CASE_LABELS, CASE_ORDER, HORIZONS, best_forecaster_per_metric,
                       best_forecaster_pd, load, save)

DATA_OUT = Path("/data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1c/data_out")
MLRUNS = "/data/horse/ws/tibo990i-thesis_data/mlruns"
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
    return {("pd", "2step"): (b.Pd - b.Pd_t).to_numpy(),
            ("pg", "ac"): (g.p_mw - g.p_mw_t).to_numpy(), ("pg", "dc"): (g.p_mw_dc - g.p_mw_t).to_numpy(),
            ("vm", "ac"): (b.Vm - b.Vm_t).to_numpy(),
            ("va", "ac"): (b.Va - b.Va_t).to_numpy(), ("va", "dc"): (b.Va_dc - b.Va_t).to_numpy()}


@lru_cache(None)
def ml(case, h, opt, fc):
    """-> {var: signed error array} from the newest state dump of the canonical run name, None if not dumped yet"""
    net = CASE_LABELS[case]
    name = f"surrogate_{case}_{fc}_h{h}_s0" if opt == "surrogate" else f"base_e2e_{net}_h{h}_s0"
    dirs = [os.path.join(os.path.dirname(os.path.dirname(t)), "artifacts", "test")
            for t in glob.glob(f"{MLRUNS}/*/*/tags/mlflow.runName") if open(t).read().strip() == name]
    dirs = [d for d in dirs if os.path.exists(f"{d}/{case}_state_bus.parquet")]
    if not dirs:
        print(f"  missing state dump: {name} -> run scripts/state_dump_eval.sbatch")
        return None
    d = max(dirs, key=os.path.getmtime)
    b, g = pd.read_parquet(f"{d}/{case}_state_bus.parquet"), pd.read_parquet(f"{d}/{case}_state_gen.parquet")
    g = g[g["gen"].isin(active_gens(case, h))]
    return {"pd": (b.pd_pred - b.pd_true).to_numpy(), "pg": (g.pg_pred - g.pg_true).to_numpy(),
            "vm": (b.vm_pred - b.vm_true).to_numpy(), "va": (b.va_deg_pred - b.va_deg_true).to_numpy()}


def err(case, h, opt, fc, var):
    if opt in ("ac", "dc"):
        return two_step(case, h, fc).get((var, opt))
    arm = ml(case, h, opt, fc)
    return arm and arm[var]


def hist_panel(ax, series, unit):
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
    Any arm's tail that runs past the view just gets cut off at the edge; its RMSE is still in the legend."""
    # Escalate the percentile tier only if needed: some arms (e.g. an exact solver matching the
    # true dispatch almost everywhere, only deviating in a long tail) have >75% of |err| at
    # effectively 0, which collapses the lower-median to 0 -- a degenerate zero-width bin range
    # that silently empties every OTHER arm's view (0/0 in the density normalization). Requiring
    # only >=1 point in view is nearly vacuous (any arm with even one near-zero observation among
    # millions trivially passes), so require a REAL fraction of each arm's data in view, not just
    # a stray point -- keep the tightest percentile tier that still clears that bar for everyone.
    MIN_FRAC_IN_VIEW = 0.05
    focus = 0.0
    for pct in (75, 80, 85, 90, 95, 99, 99.9):
        p = sorted(np.percentile(np.abs(v), pct) for _, v, _ in series)
        cand = p[(len(p) - 1) // 2]
        if cand > 0 and all(np.mean(np.abs(v) <= cand) >= MIN_FRAC_IN_VIEW for _, v, _ in series):
            focus = cand
            break
    focus = focus or 1.0
    bins = np.linspace(-focus, focus, 61)
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
    ax.legend(frameon=False, fontsize=6)


def plot_grid(df):
    best = {metric: best_forecaster_per_metric(df, metric) for _, metric, _, _ in VARS}
    for case in CASE_ORDER:
        fig, axes = plt.subplots(len(VARS), len(HORIZONS), figsize=(11, 7.5), squeeze=False)
        for r, (var, metric, label, unit) in enumerate(VARS):
            for c, h in enumerate(HORIZONS):
                sel = best[metric][(best[metric]["case"] == case) & (best[metric]["horizon"] == h)]
                fc = sel.set_index("optimizer")["forecaster"]
                series = [(OPT_LABELS[o], err(case, h, o, fc[o], var), ARM_COLORS[o])
                          for o in OPT_ORDER if o in fc.index]
                series = [(l, v, c2) for l, v, c2 in series if v is not None]
                ax = axes[r][c]
                if series:
                    hist_panel(ax, series, unit)
                if r == 0:
                    ax.set_title(f"$t+{h}$")
                if c == 0:
                    ax.set_ylabel("Density")
                ax.set_xlabel(f"{label}  [{unit}]")
        fig.suptitle(f"Signed error distribution -- {CASE_LABELS[case]} (2-step: best forecaster per metric). "
                     "$P_g$: synchronous condensers ($P_{max}=0$) excluded", y=1.01)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        save(fig, f"state_dist_{CASE_LABELS[case]}")


def plot_pd(df):
    data = best_forecaster_pd(df)
    series_colors = {"2-step (best forecaster)": ARM_COLORS["ac"], "E2E": ARM_COLORS["e2e"]}
    fig, axes = plt.subplots(len(CASE_ORDER), len(HORIZONS), figsize=(11, 7.5), squeeze=False)
    for r, case in enumerate(CASE_ORDER):
        for c, h in enumerate(HORIZONS):
            series = []
            for name, color in series_colors.items():
                fc = data[(data["case"] == case) & (data["series"] == name) & (data["horizon"] == h)]["forecaster"]
                if fc.empty:
                    continue
                v = two_step(case, h, fc.iloc[0])[("pd", "2step")] if name.startswith("2-step") \
                    else err(case, h, "e2e", "none", "pd")
                if v is not None:
                    series.append((name, v, color))
            ax = axes[r][c]
            if series:
                hist_panel(ax, series, "MW")
            if r == 0:
                ax.set_title(f"$t+{h}$")
            if c == 0:
                ax.set_ylabel(f"{CASE_LABELS[case]}\nDensity")
            ax.set_xlabel("$P_d$ error [MW]")
    fig.suptitle("$P_d$ forecast error distribution by grid size and horizon", y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "state_dist_pd")


def main():
    df = load()
    plot_grid(df)
    plot_pd(df)


if __name__ == "__main__":
    main()
