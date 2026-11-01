"""One-shot: corrected_gap.csv -> Q3 table slides, one per constraint group, run from repo root.
  - after the ex-post AC-PF; cell = share of test samples with >= 1 violation (mean worst violation over the
    violating samples); shaded = lowest share per row
  - forecaster = lowest rebalanced gap per (case, horizon, optimizer): same operating state as gap_slide row 1
  - feasibility_slide (thermal): case14 omitted (0 everywhere); diverged PF / insufficient headroom / Pg out of
    limits are 0 everywhere (asserted) -> footnote, with the 2-step OPF failures (no dispatch)
"""
import matplotlib.pyplot as plt
import pandas as pd

from colors import ARM_COLORS, OPT_LABELS, OPT_ORDER
from plot_rmse import CASE_LABELS, CASE_ORDER, HORIZONS, SLIDE_RC, best_forecaster_per_metric, save

IN = "exp1/results/combined/corrected_gap.csv"
ZERO = ["Diverged (n)", "Insufficient Headroom (n)", "Pred. Pg out of limits (%)"]
KEY = ["case", "horizon", "optimizer", "forecaster"]
GROUPS = [  # slide, constraint, share metric, worst metric, worst unit, worst format, cases
    ("feasibility_slide", "thermal overload", "Thermal Overload (%)", "Thermal Overload Worst (% of rating)",
     "% of rating above 100 %", "{:.1f}", ["case118_ieee", "case500_goc"]),
    ("feasibility_vm_slide", "PQ-bus voltage violation", "Vm Violation (%)", "Vm Violation Worst (p.u.)",
     "p.u.", "{:.4f}", CASE_ORDER),
    ("feasibility_qg_slide", "generator $Q_g$ violation", "Qg Violation (%)", "Qg Violation Worst (MVAr)",
     "MVAr", "{:.1f}", CASE_ORDER),
]


def table_slide(on, name, constraint, share_m, worst_m, unit, fmt, cases, note=""):
    piv = lambda m: on[on["metric"] == m].pivot_table(index=["case", "horizon"], columns="optimizer",
                                                       values="mean", dropna=False)
    share, worst = piv(share_m), piv(worst_m).reindex(columns=OPT_ORDER)
    rows = [(c, h) for c in cases for h in HORIZONS]
    xs = [0.33 + 0.18 * i for i in range(len(OPT_ORDER))]
    top, step, gap = 0.80, 0.62 / len(rows), 0.035
    y_of = lambda i: top - step * i - gap * (i // len(HORIZONS))  # gap between the case blocks
    with plt.rc_context(SLIDE_RC):
        fig, ax = plt.subplots(figsize=(7.8, 2.2 + 0.32 * len(rows)))
        ax.axis("off")
        for x, o in zip(xs, OPT_ORDER):
            ax.plot([x - 0.04, x + 0.04], [0.985, 0.985], color=ARM_COLORS[o], lw=3, transform=ax.transAxes)
            ax.text(x, 0.92, OPT_LABELS[o], ha="center", transform=ax.transAxes)
        ax.plot([0, 1], [0.875, 0.875], color="0.6", lw=0.8, transform=ax.transAxes)
        for i, (c, h) in enumerate(rows):
            s, w = share.loc[(c, h)], worst.loc[(c, h)]
            ax.text(0.01, y_of(i), f"{CASE_LABELS[c]}   $t+{h}$", transform=ax.transAxes)
            for x, o in zip(xs, OPT_ORDER):
                txt = f"{s[o]:.1f} %" + (f"  ({fmt.format(w[o])})" if pd.notna(w[o]) else "")
                best = dict(boxstyle="round,pad=0.3", fc="0.9", ec="none") if s[o] == s.min() else None
                ax.text(x, y_of(i), txt, ha="center", transform=ax.transAxes, bbox=best, fontsize=10)
        ax.text(0.01, 0.0, f"Share of test samples with $\\geq$1 {constraint} after the ex-post AC power flow; "
                f"brackets = mean worst violation over the violating samples [{unit}]; shaded = lowest share.{note}",
                transform=ax.transAxes, fontsize=8, color="0.35", va="bottom", wrap=True)
        save(fig, name)


def main():
    df = pd.read_csv(IN)
    assert df[df["metric"].isin(ZERO)]["mean"].eq(0).all(), "footnote no longer true"
    on = df.merge(best_forecaster_per_metric(df, "Rebalanced Gap")[KEY], on=KEY)
    no_disp = on[on["metric"] == "No Dispatch (n)"].pivot_table(index=["case", "horizon"], columns="optimizer",
                                                                values="mean")
    assert on[(on["metric"] == "Thermal Overload (%)") & (on["case"] == "case14_ieee")]["mean"].eq(0).all()
    fails = no_disp[no_disp.gt(0).any(axis=1)]
    fail_txt = "; ".join(f"{CASE_LABELS[c]} $t+{h}$: {int(r.max())}" for (c, h), r in fails.iterrows()) or "none"
    thermal_note = ("\ncase14: 0 % everywhere. Diverged PFs, insufficient headroom, $P_g$ out of limits: 0 everywhere. "
                    f"No dispatch (2-step OPF failed): {fail_txt}.")
    for g in GROUPS:
        table_slide(on, *g, note=thermal_note if g[0] == "feasibility_slide" else "")


if __name__ == "__main__":
    main()
