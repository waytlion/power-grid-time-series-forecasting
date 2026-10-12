"""Inference time for one 24 h dispatch plan vs grid size.

- 2-step (XGB forecaster): one multi-output XGB call (whole 24-step forecast) + H OPF-step passes.
  Both measured at h=24 (per-scenario OPF step = unaffected by the E2E timing fix).
- E2E: one forward pass emits the whole horizon -> its h=1 per-sample time (h=24 per-window
  measurement agrees within ~5%, but its CSV unit changes once E2E is re-evaluated).
Reads the tidy aggregate_results CSV; writes figures/infer_time_h24.{svg,pdf}.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from colors import ARM_COLORS, OPT_LABELS, OPT_ORDER, THESIS_RC

H = 24
IN = Path("exp1/results/combined/case118_combined.csv")
OUT = Path("exp1/thesis_plots/figures/infer_time_h24.svg")

df = pd.read_csv(IN)
df["n_bus"] = df["case"].map(lambda c: int(re.search(r"case(\d+)", c).group(1)))
t = df.pivot_table(index=["n_bus", "optimizer", "forecaster", "horizon"], columns="metric", values="mean")

twostep = t.xs((H, "xgb"), level=("horizon", "forecaster"))
plan = (twostep["full_pipeline_infer_s"] - twostep["opf_step_infer_s"]  # 24-step XGB forecast
        + H * twostep["opf_step_infer_s"]).unstack("optimizer")
plan["e2e"] = t.xs((1, "e2e", "none"), level=("horizon", "optimizer", "forecaster"))["opf_step_infer_s"]

plt.rcParams.update(THESIS_RC)
fig, ax = plt.subplots(figsize=(4.2, 3.0))
for opt in OPT_ORDER:
    s = plan[opt].dropna()
    ax.plot(s.index, s.values, marker="o", ms=5, lw=1.5, color=ARM_COLORS[opt], label=OPT_LABELS[opt])
    ax.annotate(OPT_LABELS[opt], (s.index[-1], s.values[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=8)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(plan.index, [str(n) for n in plan.index])
ax.minorticks_off()
ax.set_xlim(plan.index.min() / 1.3, plan.index.max() * 1.9)
ax.set_xlabel("Number of buses")
ax.set_ylabel(f"Inference time per {H} h plan (s)")
ax.grid(axis="y", which="major", lw=0.4, alpha=0.4)
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
print(plan.round(4).to_string())
