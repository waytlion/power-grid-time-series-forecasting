"""One-shot: Q2 figures, run from repo root.
  - gap_slide: rebalanced gap, cols = case, x = horizon, one line per optimizer, best forecaster per metric
    (same rule + layout as rmse_slide); raw gap left out (under-supply looks cheap before rebalancing)
  - supply_error_h<H>: per forecast timestep, supply error ΔP = ΣP̂ − ΣP* (what rebalancing corrects), step
    histograms per arm (best forecaster by rebalanced gap, as table 8; ML seeds pooled); density over all
    timesteps, view = 0.5–99.5 % of all arms pooled (clipping the view drops no data)
    - needs the per-timestep ex-post outputs (cluster: expost_pf_v3; override EXPOST_DIR) -> skipped if absent
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from colors import ARM_COLORS, OPT_ORDER
from plot_rmse import (CASE_LABELS, CASE_ORDER, LS, SLIDE_RC, best_forecaster_per_metric, finish_slide, h_label,
                       keep_forecasters, optimizer_handles, optimizer_lines, save, style_slide_axes)

IN = "exp1/results/combined/corrected_gap.csv"
EXPOST = Path(os.environ.get("EXPOST_DIR", "/data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1c/expost_pf_v3"))
FILES = {"ac": "exact_{}", "dc": "dc_{}", "surrogate": "surrogate_{}_s*", "e2e": "e2e_s*"}  # expost_pf tags
SUPPLY_H = 6


def plot_gap(df):
    best = best_forecaster_per_metric(df, "Rebalanced Gap")
    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(1, len(CASE_ORDER), figsize=(10, 3.4), squeeze=False)
        for c, case in enumerate(CASE_ORDER):
            optimizer_lines(axes[0][c], best[best["case"] == case], tag=c == 0)
            axes[0][c].set_title(CASE_LABELS[case])
        axes[0][0].set_ylabel("Rebalanced gap [%], log")
        style_slide_axes(axes)
        finish_slide(fig, optimizer_handles(), "gap_slide")


def supply_error(case, h, opt, fc):
    """per-forecast-timestep ΔP = ΣP̂ − ΣP* [MW] of one arm, ML seeds pooled; no-dispatch timesteps dropped"""
    files = sorted(EXPOST.glob(f"{case}/h{h}/{FILES[opt].format(fc)}.parquet"))
    assert files, f"no ex-post output for {case} h{h} {opt} {fc}"
    d = pd.concat([pd.read_parquet(f, columns=["pg_pred_mw", "pg_true_mw"]) for f in files])
    return (d["pg_pred_mw"] - d["pg_true_mw"]).dropna().to_numpy()


def plot_supply_error(df, h=SUPPLY_H):
    best = best_forecaster_per_metric(df, "Rebalanced Gap").set_index(["case", "horizon", "optimizer"])["forecaster"]
    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(1, len(CASE_ORDER), figsize=(10, 3.6))
        for ax, case in zip(axes, CASE_ORDER):
            err = {o: supply_error(case, h, o, best[(case, h, o)]) for o in OPT_ORDER}
            lo, hi = np.percentile(np.concatenate(list(err.values())), [0.5, 99.5])
            bins = np.linspace(lo, hi, 61)
            for o, x in err.items():
                ax.hist(x, bins, weights=np.full(len(x), 1 / (len(x) * (bins[1] - bins[0]))), histtype="step",
                        color=ARM_COLORS[o], ls=LS[o], lw=1.6)
            ax.axvline(0, color="0.6", lw=0.8)
            ax.set_yticks([])
            ax.set_title(f"{CASE_LABELS[case]}, {h_label(h)}")
            ax.set_xlabel("supply error $\\Sigma \\hat{P}_g - \\Sigma P_g^*$ [MW]")
        axes[0].set_ylabel("density")
        fig.legend(handles=optimizer_handles(), loc="upper center", ncol=len(OPT_ORDER), frameon=False,
                   bbox_to_anchor=(0.5, 1.04))
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        save(fig, f"supply_error_h{h}")


def main():
    df = keep_forecasters(pd.read_csv(IN))
    plot_gap(df)
    if EXPOST.exists():
        plot_supply_error(df)
    else:
        print(f"skip supply_error: {EXPOST} not found (per-timestep ex-post outputs live on the cluster)")


if __name__ == "__main__":
    main()
