"""Per-generator signed dispatch error (pred - AC truth) for 2-step AC vs DC, top-K generators by DC RMSE.
Shows WHERE the pooled Generator Pg RMSE comes from. Run from repo root:
  python exp1/thesis_plots/plot_dpg_by_gen.py [--opf-base-dir <phase1c data_out>/case118_ieee_horizon_1]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from colors import ARM_COLORS, THESIS_RC

BASE = "/data/horse/ws/tibo990i-thesis_data/2019-2021/phase_1c/data_out/case118_ieee_horizon_1"

ap = argparse.ArgumentParser()
ap.add_argument("--opf-base-dir", type=Path, default=Path(BASE))
ap.add_argument("--forecaster", default="xgb")
ap.add_argument("--dataset", default="case118_ieee")
ap.add_argument("--top", type=int, default=8)
ap.add_argument("--out", type=Path, default=Path("exp1/thesis_plots/figures/dpg_by_gen.pdf"))
args = ap.parse_args()

raw = lambda arm: args.opf_base_dir / arm / args.dataset / "raw" / "gen_data.parquet"
cols = ["load_scenario_idx", "idx", "bus", "p_mw", "p_mw_dc"]
pred = pd.read_parquet(raw(args.forecaster), columns=cols)
true = pd.read_parquet(raw("true"), columns=["load_scenario_idx", "idx", "p_mw"])
m = pred.merge(true, on=["load_scenario_idx", "idx"], suffixes=("", "_true"))
m["AC"] = m["p_mw"] - m["p_mw_true"]
m["DC"] = m["p_mw_dc"] - m["p_mw_true"]

rmse_dc = m.groupby("idx")["DC"].apply(lambda x: np.sqrt((x ** 2).mean()))
top = rmse_dc.nlargest(args.top).index
bus = m.groupby("idx")["bus"].first()
print(f"pooled RMSE  AC={np.sqrt((m['AC']**2).mean()):.2f}  DC={np.sqrt((m['DC']**2).mean()):.2f}  "
      f"top-{args.top} share of DC sum-sq: {(m[m.idx.isin(top)]['DC']**2).sum() / (m['DC']**2).sum():.1%}")

plt.rcParams.update(THESIS_RC)
fig, ax = plt.subplots(figsize=(9, 4))
w = 0.35
for j, arm in enumerate(("AC", "DC")):
    data = [m.loc[m.idx == i, arm].to_numpy() for i in top]
    bp = ax.boxplot(data, positions=np.arange(len(top)) + (j - 0.5) * w * 1.1, widths=w, whis=(5, 95),
                    showfliers=False, patch_artist=True, medianprops=dict(color="k"))
    for box in bp["boxes"]:
        box.set(facecolor=ARM_COLORS[arm.lower()], alpha=0.85)
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(range(len(top)))
ax.set_xticklabels([f"gen {int(i)}\n(bus {int(bus[i])})" for i in top], fontsize=8)
ax.set_ylabel(r"$P_g^{\mathrm{pred}} - P_g^{\mathrm{AC\ truth}}$  [MW]")
ax.grid(axis="y", color="0.9", lw=0.6)
ax.set_axisbelow(True)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ARM_COLORS[a.lower()]) for a in ("AC", "DC")],
          labels=["AC", "DC"], frameon=False, ncol=2, loc="upper right")
ax.set_title(f"Signed per-generator dispatch error, top {args.top} by DC RMSE ({args.dataset.replace('_', ' ')}, {args.forecaster}; box = IQR, whiskers = 5-95%)")
fig.tight_layout()
args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, bbox_inches="tight")
print(f"wrote {args.out}")
