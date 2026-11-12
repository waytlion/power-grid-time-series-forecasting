"""One-shot: corrected_gap.csv -> Q3 table slides, one per constraint group, run from repo root.
  - after the ex-post AC-PF; cell = share of elements with a violation (mean violation per violating element),
    grey line = share of test samples with >= 1 violation; shaded = lowest element share per row
  - needs the per-element sums (expost_pf ≥ 2026-11-01 -> corrected_gap "... Mean (unit)" metrics)
  - forecaster = lowest rebalanced gap per (case, horizon, optimizer): same operating state as gap_slide row 1
  - feasibility_slide (thermal): case14 omitted (0 everywhere)
  - zero-checks (diverged PF, insufficient headroom, Pg out of limits, case14 thermal) asserted here, reported as a
    sentence in the results text (MA-tex results.tex Q3), not on the figure
"""
import matplotlib.pyplot as plt
import pandas as pd

from colors import ARM_COLORS, OPT_LABELS, OPT_ORDER
from plot_rmse import (CASE_LABELS, CASE_ORDER, HORIZONS, SLIDE_RC, best_forecaster_per_metric, h_label,
                       keep_forecasters, save)

IN = "exp1/results/combined/corrected_gap.csv"
ZERO = ["Diverged (n)", "Insufficient Headroom (n)", "Pred. Pg out of limits (%)"]
KEY = ["case", "horizon", "optimizer", "forecaster"]
GROUPS = [  # slide, constraint, share metric, worst metric, worst unit, worst format, cases
    ("feasibility_slide", "Thermal Overload", "lines", "% of rating", "% of rating above 100 %", "{:.1f}",
     ["case118_ieee", "case500_goc"]),
    ("feasibility_vm_slide", "Vm Violation", "PQ buses", "p.u.", "p.u.", "{:.4f}", CASE_ORDER),
    ("feasibility_qg_slide", "Qg Violation", "non-REF gen buses", "MVAr", "MVAr", "{:.1f}", CASE_ORDER),
]


def pct(v, floor, fmt):
    """'0 %' / '< floor %' for tiny nonzero values / formatted"""
    return "0 %" if v == 0 else f"< {floor:g} %" if v < floor else fmt.format(v) + " %"


def table_slide(on, name, group, elements, metric_unit, unit, fmt, cases):
    """cell: share of elements with a violation (mean violation per violating element); grey: share of samples"""
    piv = lambda m: on[on["metric"] == m].pivot_table(index=["case", "horizon"], columns="optimizer",
                                                       values="mean", dropna=False).reindex(columns=OPT_ORDER)
    elem, mean, samp = piv(f"{group} (% elements)"), piv(f"{group} Mean ({metric_unit})"), piv(f"{group} (%)")
    rows = [(c, h) for c in cases for h in HORIZONS]
    xs = [0.31 + 0.19 * i for i in range(len(OPT_ORDER))]
    top, step, gap = 0.80, 0.64 / len(rows), 0.035
    y_of = lambda i: top - step * i - gap * (i // len(HORIZONS))  # gap between the case blocks
    with plt.rc_context(SLIDE_RC):
        fig, ax = plt.subplots(figsize=(7.8, 2.4 + 0.45 * len(rows)))
        ax.axis("off")
        for x, o in zip(xs, OPT_ORDER):
            ax.plot([x - 0.04, x + 0.04], [0.985, 0.985], color=ARM_COLORS[o], lw=3, transform=ax.transAxes)
            ax.text(x, 0.92, OPT_LABELS[o], ha="center", transform=ax.transAxes)
        ax.plot([0, 1], [0.875, 0.875], color="0.6", lw=0.8, transform=ax.transAxes)
        for i, (c, h) in enumerate(rows):
            e, m, s = elem.loc[(c, h)], mean.loc[(c, h)], samp.loc[(c, h)]
            ax.text(0.01, y_of(i), f"{CASE_LABELS[c]}   {h_label(h)}", transform=ax.transAxes)
            for x, o in zip(xs, OPT_ORDER):
                txt = pct(e[o], 0.001, "{:.2g}") + (f"  ({fmt.format(m[o])})" if pd.notna(m[o]) else "")
                best = dict(boxstyle="round,pad=0.3", fc="0.9", ec="none") if e[o] == e.min() else None
                ax.text(x, y_of(i), txt, ha="center", transform=ax.transAxes, bbox=best, fontsize=10)
                ax.text(x, y_of(i) - step * 0.42, pct(s[o], 1, "{:.0f}") + " of samples", ha="center",
                        transform=ax.transAxes,
                        fontsize=7.5, color="0.45")
        ax.text(0.01, 0.0, f"After the ex-post AC power flow: share of {elements} with a violation; brackets = mean "
                f"violation per violating element [{unit}]; grey = share of test samples with $\\geq$1 violation; "
                "shaded = lowest element share.", transform=ax.transAxes, fontsize=8, color="0.35", va="bottom",
                wrap=True)
        save(fig, name)


def main():
    df = keep_forecasters(pd.read_csv(IN))
    assert df[df["metric"].isin(ZERO)]["mean"].eq(0).all(), "results-text zero-check no longer true"
    assert df[(df["metric"] == "Thermal Overload (%)") & (df["case"] == "case14_ieee")]["mean"].eq(0).all()
    on = df.merge(best_forecaster_per_metric(df, "Rebalanced Gap")[KEY], on=KEY)
    for g in GROUPS:
        table_slide(on, *g)


if __name__ == "__main__":
    main()
