"""One-shot: control arm -> control_slide.{pdf,svg}, run from repo root.
  - same E2E model (seed 0), same inputs, two routes to Pg: its Pg output vs its Pd output -> AC-OPF (e2e_pd_opf.py)
  - cols = case, x = horizon, Pg RMSE; % per horizon = Pg output vs Pd route
  - inputs: exp1/results/_e2e_pd/<case>_horizon<H>/e2e_pd/<case>_metrics.csv (Pd route),
    exp1/results/_dispatch_diag/<case>/step1_split.csv (E2E s0, recomputed == logged)
  - case500 t+24: 5.5 % OPF failures dropped from the Pd route
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from colors import ARM_COLORS
from plot_rmse import h_label, CASE_LABELS, CASE_ORDER, HORIZONS, SLIDE_RC, X_OF, save, style_slide_axes

PD_ROUTE = Path("exp1/results/_e2e_pd")
SPLIT = Path("exp1/results/_dispatch_diag")
ROUTE, OUTPUT = "E2E $P_d \\rightarrow$ AC-OPF", "E2E $P_g$ output"
SERIES = {ROUTE: ("--", "white"), OUTPUT: ("-", ARM_COLORS["e2e"])}  # name: (line style, marker face)


def load(case):
    """-> {series: [Pg RMSE per horizon]}"""
    s = pd.read_csv(SPLIT / case / "step1_split.csv").set_index(["H", "arm"])["pg_rmse"]
    route = [pd.read_csv(PD_ROUTE / f"{case}_horizon{h}" / "e2e_pd" / f"{case}_metrics.csv")
             .set_index("Metric").loc["AC Generator Pg RMSE", "Value"] for h in HORIZONS]
    return {ROUTE: route, OUTPUT: [s[h]["e2e"] for h in HORIZONS]}


def main():
    cases = [c for c in CASE_ORDER if (SPLIT / c / "step1_split.csv").exists()]
    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(1, len(cases), figsize=(10, 3.8), squeeze=False)
        style_slide_axes(axes)
        for ax, case in zip(axes[0], cases):
            d = load(case)
            for name, (ls, face) in SERIES.items():
                ax.plot([X_OF[h] for h in HORIZONS], d[name], ls=ls, marker="o", color=ARM_COLORS["e2e"], mfc=face,
                        lw=2, ms=6)
            ax.set_xticklabels([f"{h_label(h)}\n{100 * (a / b - 1):+.0f} %"
                                for h, a, b in zip(HORIZONS, d[OUTPUT], d[ROUTE])])
            ax.set_ylim(0, max(max(v) for v in d.values()) * 1.12)
            ax.set_title(CASE_LABELS[case])
        axes[0][0].set_ylabel("$P_g$ RMSE [MW]")
        handles = [plt.Line2D([], [], color=ARM_COLORS["e2e"], ls=ls, marker="o", mfc=f, lw=2, label=n)
                   for n, (ls, f) in SERIES.items()]
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.04))
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        save(fig, "control_slide")


if __name__ == "__main__":
    main()
