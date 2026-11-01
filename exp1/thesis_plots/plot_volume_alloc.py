"""One-shot: dispatch_diag step1 outputs -> Q1 "why" slides, run from repo root.
  - input: exp1/results/_dispatch_diag/<case>/step1_{split.csv,samples.parquet}
    (copied from <data>/2019-2021/phase_1c/dispatch_diag_final)
  - arms: AC with the best forecaster by Pg RMSE (same rule as rmse_slide) vs E2E (seed 0); cols = case
  1) volume_alloc_slide: per test sample, mean MW, x = horizon
     - Σ_g |ΔPg_g| = volume |Σ_g ΔPg_g| + allocation 2 · misplaced (exact split per sample)
     - stacked bar per arm: volume (solid) + allocation (hatched); ◆ = summed load forecast error |Σ_i ΔPd_i|
  2) total_scatter_slide: per test sample at SCATTER_H, signed Σ_i ΔPd_i (x) vs Σ_g ΔPg_g (y), y = x dashed
     - 2-step lies on y ≈ x by construction (ΣPg = ΣPd + losses); slope / r of y on x per arm and horizon
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from colors import ARM_COLORS
from plot_rmse import CASE_LABELS, CASE_ORDER, HORIZONS, SLIDE_RC, X_OF, save

IN_DIR = Path("exp1/results/_dispatch_diag")
W = 0.34  # bar width
SCATTER_H = 1  # E2E decoupling is strongest at t+1; slopes for all horizons go in the panel text


def load():
    """best AC arm per horizon (lowest Pg RMSE) + E2E, per available case"""
    out = {}
    for case in CASE_ORDER:
        f = IN_DIR / case / "step1_split.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f)
        ac = s[s["arm"].str.startswith("ac_")]
        best = ac.loc[ac.groupby("H")["pg_rmse"].idxmin()].assign(opt="ac")
        out[case] = pd.concat([best, s[s["arm"] == "e2e"].assign(opt="e2e")])
    return out


def plot_bars(data):
    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(1, len(data), figsize=(10, 3.9), squeeze=False)
        for ax, (case, d) in zip(axes[0], data.items()):
            for opt, dx in (("ac", -W / 2 - 0.02), ("e2e", W / 2 + 0.02)):
                r = d[d["opt"] == opt].set_index("H").reindex(HORIZONS)
                x = [X_OF[h] + dx for h in HORIZONS]
                ax.bar(x, r["total_abs_mw"], W, color=ARM_COLORS[opt])
                ax.bar(x, 2 * r["misplaced_mw"], W, bottom=r["total_abs_mw"], color="white",
                       edgecolor=ARM_COLORS[opt], hatch="////", lw=0.8)
                ax.plot(x, r["dD_abs_mw"], "D", color="0.15", ms=6, zorder=3)
            top = max((d["total_abs_mw"] + 2 * d["misplaced_mw"]).max(), d["dD_abs_mw"].max())
            ax.set_ylim(0, top * 1.08)
            ax.set_title(CASE_LABELS[case])
            ax.set_xticks(range(len(HORIZONS)))
            ax.set_xticklabels([f"$t+{h}$" for h in HORIZONS])
            ax.grid(axis="y", color="0.9", lw=0.6)
            ax.set_axisbelow(True)
        axes[0][0].set_ylabel("mean error per sample [MW]")

        handles = [plt.Rectangle((0, 0), 1, 1, color=ARM_COLORS["ac"], label="AC: volume"),
                   plt.Rectangle((0, 0), 1, 1, color=ARM_COLORS["e2e"], label="E2E: volume"),
                   plt.Rectangle((0, 0), 1, 1, fc="white", ec="0.3", hatch="////", label="allocation"),
                   plt.Line2D([], [], ls="", marker="D", color="0.15", label="summed $P_d$ error")]
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.04))
        fig.text(0.99, -0.02, "volume $|\\Sigma_g \\Delta P_g|$, allocation $\\Sigma_g|\\Delta P_g| - |\\Sigma_g \\Delta P_g|$, "
                 "summed $P_d$ error $|\\Sigma_i \\Delta P_{d,i}|$; AC: best forecaster by $P_g$ RMSE; E2E: seed 0",
                 ha="right", fontsize=8, color="0.35")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        save(fig, "volume_alloc_slide")


def plot_scatter(data):
    labels = {"ac": "AC", "e2e": "E2E"}
    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(1, len(data), figsize=(10, 4.0), squeeze=False)
        for ax, (case, d) in zip(axes[0], data.items()):
            smp = pd.read_parquet(IN_DIR / case / "step1_samples.parquet")
            fits, lim = [], 0.0
            for opt in ("ac", "e2e"):
                fit = []
                for h in HORIZONS:
                    arm = d[(d["opt"] == opt) & (d["H"] == h)]["arm"].iloc[0]
                    s = smp[(smp["H"] == h) & (smp["arm"] == arm)]
                    fit.append(np.polyfit(s["dD_mw"], s["total_mw"], 1)[0])
                    if h == SCATTER_H:
                        ax.scatter(s["dD_mw"], s["total_mw"], s=4, alpha=0.25, color=ARM_COLORS[opt], lw=0,
                                   rasterized=True)
                        r = np.corrcoef(s["dD_mw"], s["total_mw"])[0, 1]
                        lim = max(lim, np.percentile(np.abs(s[["dD_mw", "total_mw"]]), 99.5))
                fits.append(f"{labels[opt]}: r = {r:.2f}, slope " + " / ".join(f"{v:.2f}" for v in fit))
            ax.plot([-lim, lim], [-lim, lim], "--", color="0.4", lw=1)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.text(0.03, 0.97, "\n".join(fits), transform=ax.transAxes, va="top", fontsize=8.5,
                    bbox=dict(fc="white", ec="none", alpha=0.85))
            ax.set_title(f"{CASE_LABELS[case]}, $t+{SCATTER_H}$")
            ax.set_xlabel("summed $P_d$ error $\\Sigma_i \\Delta P_{d,i}$ [MW]")
        axes[0][0].set_ylabel("total $P_g$ error $\\Sigma_g \\Delta P_g$ [MW]")
        handles = [plt.Line2D([], [], ls="", marker="o", color=ARM_COLORS[o], label=labels[o]) for o in labels]
        handles.append(plt.Line2D([], [], ls="--", color="0.4", label="$y = x$"))
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.04))
        fig.text(0.99, -0.02, f"one point per test sample ($t+{SCATTER_H}$); slope of $y$ on $x$ for "
                 f"$t+{HORIZONS[0]}$ / $t+{HORIZONS[1]}$ / $t+{HORIZONS[2]}$; AC: best forecaster by $P_g$ RMSE; "
                 "E2E: seed 0", ha="right", fontsize=8, color="0.35")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        save(fig, "total_scatter_slide")


def main():
    data = load()
    plot_bars(data)
    plot_scatter(data)


if __name__ == "__main__":
    main()
