"""One-shot: corrected_gap.csv -> Q3 table slides, one per constraint group, run from repo root.
  - after the ex-post AC-PF; cell = share of elements with a violation (mean violation per violating element),
    grey line = share of test samples with >= 1 violation; shaded = lowest element share per row
  - needs the per-element sums (expost_pf ≥ 2026-11-01 -> corrected_gap "... Mean (unit)" metrics)
  - forecaster = lowest rebalanced gap per (case, horizon, optimizer): same operating state as gap_slide row 1
  - feasibility_slide (thermal): case14 omitted (0 everywhere)
  - zero-checks (diverged PF, insufficient headroom, Pg out of limits, case14 thermal) asserted here, reported as a
    sentence in the results text (MA-tex results.tex Q3), not on the figure
  - violation_ccdf_h<H>: exceedance curve of the violation per violating element (seeds pooled, corrected_gap.py
    --ccdf-out); rows = constraint group, cols = case, line = arm; median / P99 where a curve crosses 0.5 / 0.01
"""
import matplotlib.pyplot as plt
import pandas as pd

from colors import ARM_COLORS, OPT_LABELS, OPT_ORDER
from plot_rmse import (CASE_LABELS, CASE_ORDER, HORIZONS, LS, SLIDE_RC, best_forecaster_per_metric, h_label,
                       keep_forecasters, log_ticks, optimizer_handles, save)

IN = "exp1/results/combined/corrected_gap.csv"
CCDF_IN = "exp1/results/combined/violation_ccdf.csv"
CCDF_ROWS = [  # element group, x label, display scale of the stored violation
    ("thermal", "loading above rating [%]", 1),
    ("vm", "$V_m$ violation [$10^{-3}$ p.u.]", 1e3),
    ("qg", "$Q_g$ violation [MVAr]", 1),
]
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


def plot_ccdf(cc, h):
    """exceedance curves per violating element at horizon h; cc = violation_ccdf rows of the best forecasters"""
    fig, axes = plt.subplots(len(CCDF_ROWS), len(CASE_ORDER), figsize=(9, 7.5), sharey=True)
    for r, (g, xlabel, scale) in enumerate(CCDF_ROWS):
        for c, case in enumerate(CASE_ORDER):
            ax, sub = axes[r][c], cc[(cc["case"] == case) & (cc["horizon"] == h) & (cc["group"] == g)]
            if r == 0:
                ax.set_title(CASE_LABELS[case])
            if sub.empty:  # no arm with >= MIN_EL violating elements
                ax.text(0.5, 0.5, "no or too few violations", ha="center", transform=ax.transAxes, color="0.45")
                ax.set_xticks([])
                continue
            for o in OPT_ORDER:
                s = sub[sub["optimizer"] == o]
                ax.plot(s["viol"] * scale, s["exceed"], color=ARM_COLORS[o], ls=LS[o], lw=1.5)
            ax.set(xscale="log", yscale="log", ylim=(1e-4, 1.5), xlabel=xlabel)
            log_ticks(ax.xaxis)
            for y, label in ((0.5, "median"), (0.01, "P99")):
                ax.axhline(y, color="0.6", ls=":", lw=0.8)
                if c == len(CASE_ORDER) - 1:
                    ax.text(1.02, y, label, transform=ax.get_yaxis_transform(), va="center", fontsize=8, color="0.45")
            ax.grid(color="0.92", lw=0.5)
        axes[r][0].set_ylabel("share of violating\nelements above $x$")
    fig.legend(handles=optimizer_handles(), loc="upper center", ncol=len(OPT_ORDER), frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, f"violation_ccdf_h{h}")


def main():
    df = keep_forecasters(pd.read_csv(IN))
    assert df[df["metric"].isin(ZERO)]["mean"].eq(0).all(), "results-text zero-check no longer true"
    assert df[(df["metric"] == "Thermal Overload (%)") & (df["case"] == "case14_ieee")]["mean"].eq(0).all()
    best = best_forecaster_per_metric(df, "Rebalanced Gap")[KEY]
    on = df.merge(best, on=KEY)
    for g in GROUPS:
        table_slide(on, *g)
    cc = pd.read_csv(CCDF_IN).merge(best, on=KEY)
    for h in HORIZONS:
        plot_ccdf(cc, h)


if __name__ == "__main__":
    main()
