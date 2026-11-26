"""Physics-loss ablation (E2E, case118, H=24, 3 seeds each) -> table rows, run from repo root.
  - in: results/_ablation_nophys/ ('_' -> not aggregated): metrics_<arm>_s<S>.csv (run metrics), corrected_gap.py
    on the Q-limit ex-post: cg_{ref,ablation}.csv (seeds pooled), seeds/corrgap_qlim_s<S>.csv (base) and
    cg_ablation_s<S>.csv
  - table13_nophys_rows.tex: mean ± std over seeds; P99 over pooled seeds (no std)
"""
import pandas as pd

IN = "exp1/results/_ablation_nophys"
OUT = "exp1/results/combined/table13_nophys_rows.tex"
SEEDS = (0, 1, 2)
ARMS = {  # arm (= table column order) -> per-seed corrected_gap CSV, pooled corrected_gap suffix
    "base": ("exp1/results/combined/seeds/corrgap_qlim_s{}.csv", "ref"),
    "nophys": (f"{IN}/cg_ablation_s{{}}.csv", "ablation"),
}
ROWS = [  # source, metric, row label, display scale, decimals
    ("run", "Generator Pg RMSE", "RMSE $P_g$ [MW]", 1, 2),
    ("run", "Vm RMSE", "RMSE $|V|$ [$10^{-3}$\\,p.u.]", 1e3, 2),
    ("run", "Va RMSE", "RMSE $\\theta$ [deg]", 1, 2),
    ("cg", "Rebalanced Gap", "Rebalanced gap [\\%]", 1, 2),
    ("cg", "Vm Violation (%)", "$|V|$ violation [\\% samples]", 1, 1),
    ("pooled", "Vm Violation P99 (p.u.)", "$|V|$ violation, P99 [$10^{-3}$\\,p.u.]", 1e3, 1),
    ("cg", "Thermal Overload (%)", "Thermal overload [\\% samples]", 1, 1),
    ("cg", "Thermal Overload (% elements)", "Thermal overload [\\% elements]", 1, 2),
]


def select(path):
    """case118 H=24 E2E rows of a corrected_gap CSV"""
    d = pd.read_csv(path)
    return d[(d["case"] == "case118_ieee") & (d["horizon"] == 24) & (d["optimizer"] == "e2e")]


def per_seed(arm, source, metric):
    if source == "run":
        return pd.Series([pd.read_csv(f"{IN}/metrics_{arm}_s{s}.csv").set_index("Metric").loc[metric, "Value"]
                          for s in SEEDS], dtype=float)
    return pd.Series([select(ARMS[arm][0].format(s)).set_index("metric").loc[metric, "mean"] for s in SEEDS])


pooled = {a: select(f"{IN}/cg_{v[1]}.csv").set_index("metric")["mean"] for a, v in ARMS.items()}

lines = []
for source, metric, label, scale, dec in ROWS:
    cells = []
    for arm in ARMS:
        if source == "pooled":
            cells.append(f"{pooled[arm][metric] * scale:.{dec}f}")
        else:
            v = per_seed(arm, source, metric) * scale
            cells.append(f"{v.mean():.{dec}f} $\\pm$ {v.std():.{dec}f}")
            print(f"{arm:7s} {metric}: " + ", ".join(f"{x:.4g}" for x in v))
    lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    if metric == "Va RMSE" or metric == "Rebalanced Gap":
        lines.append("\\midrule")
with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {OUT}")
