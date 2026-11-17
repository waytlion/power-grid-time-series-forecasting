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
  - violation_<kind> over H, same grid as violation_ccdf, log y -> zeros = hollow markers on a "0" row
    - expected: violation per element and sample (= % elements x mean, each averaged over seeds)
    - el: % elements violated per sample
    - mean: mean violation per violating element (only with >= 100 violating elements, as P99 / table 9)
  - violation_box_<case>: distribution of the violation per violating element, cols = H (box P25-P75, median, mean
    diamond, whiskers P1-P99 interpolated on the violation_ccdf levels, cross = max from corrected_gap);
    right: % elements violated per sample
  - ex-post variant: plot_rmse.VARIANT; Q limits on -> no Qg row (0 by construction), |V| over all buses, diverged PF
    not zero-checked (case500)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter

from colors import ARM_COLORS, OPT_LABELS, OPT_ORDER
from plot_rmse import (CASE_LABELS, CASE_ORDER, CCDF_IN, CORR_IN as IN, HORIZONS, LS, QLIM, SLIDE_RC,
                       best_forecaster_per_metric, h_label, keep_forecasters, log_ticks, optimizer_handles, save)

VM_BUS = ("bus", "buses") if QLIM else ("PQ bus", "PQ buses")  # Q limits on: |V| checked at all buses
CCDF_ROWS = [  # element group, x label, display scale of the stored violation
    ("thermal", "loading above rating [%]", 1),
    ("vm", "$|V|$ violation [$10^{-3}$ p.u.]", 1e3),
    ("qg", "$Q_{\\mathrm{g}}$ violation [MVAr]", 1),
][:2 if QLIM else 3]
BOX_Q = {"whislo": 0.99, "q1": 0.75, "med": 0.5, "q3": 0.25, "whishi": 0.01}  # bxp key -> exceedance level
H_ROWS = [  # constraint, stored unit, display scale, element (singular, plural), violation, display unit
    ("Thermal Overload", "% of rating", 1, "line", "lines", "thermal overload", "% of rating"),
    ("Vm Violation", "p.u.", 1e3, *VM_BUS, "$|V|$ violation", "$10^{-3}$ p.u."),
    ("Qg Violation", "MVAr", 1, "gen. bus", "gen. buses", "$Q_{\\mathrm{g}}$ violation", "MVAr"),
][:len(CCDF_ROWS)]
ROW_H = 2.5  # figure height per constraint row [in]
H_YLABEL = {  # kind -> y label (element, elements, violation, unit)
    "expected": "{2} per {0}\nand sample [{3}]",
    "el": "{1} violated\nper sample [%]",
    "mean": "{2} per\nviolating {0} [{3}]",
}
DODGE = dict(zip(OPT_ORDER, (-0.09, -0.03, 0.03, 0.09)))  # x offset of the zero markers (several arms at one H)
ZERO = ["Diverged (n)"] * (not QLIM) + ["Insufficient Headroom (n)", "Pred. Pg out of limits (%)"]
KEY = ["case", "horizon", "optimizer", "forecaster"]
GROUPS = [  # slide, constraint, share metric, worst metric, worst unit, worst format, cases
    ("feasibility_slide", "Thermal Overload", "lines", "% of rating", "% of rating above 100 %", "{:.1f}",
     ["case118_ieee", "case500_goc"]),
    ("feasibility_vm_slide", "Vm Violation", VM_BUS[1], "p.u.", "p.u.", "{:.4f}", CASE_ORDER),
    ("feasibility_qg_slide", "Qg Violation", "non-REF gen buses", "MVAr", "MVAr", "{:.1f}", CASE_ORDER),
][:len(CCDF_ROWS)]


def pct(v, floor, fmt):
    """'0 %' / '< floor %' for tiny nonzero values / formatted"""
    return "0 %" if v == 0 else f"$<$ {floor:g} %" if v < floor else fmt.format(v) + " %"


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
    fig, axes = plt.subplots(len(CCDF_ROWS), len(CASE_ORDER), figsize=(9, ROW_H * len(CCDF_ROWS)), sharey=True)
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


def box_stats(s):
    """one arm's exceedance curve (violation_ccdf rows) -> bxp quantiles"""
    s = s.sort_values("exceed")
    q = lambda e: np.interp(np.log10(e), np.log10(s["exceed"]), s["viol"])
    return {k: q(e) for k, e in BOX_Q.items()}


def plot_box(on, cc, case):
    """violation distribution per violating element of one case, cols = H; on / cc = corrected_gap / violation_ccdf
    rows of the best forecasters"""
    at = on[on["case"] == case].set_index(["horizon", "optimizer", "metric"])["mean"]
    y_of = {o: len(OPT_ORDER) - 1 - i for i, o in enumerate(OPT_ORDER)}  # first arm on top
    fig, axes = plt.subplots(len(H_ROWS), len(HORIZONS), figsize=(9, ROW_H * len(H_ROWS)), sharey=True, sharex="row")
    for r, ((g, xlabel, scale), (group, unit, *_)) in enumerate(zip(CCDF_ROWS, H_ROWS)):
        for c, h in enumerate(HORIZONS):
            ax, sub = axes[r][c], cc[(cc["case"] == case) & (cc["horizon"] == h) & (cc["group"] == g)]
            if r == 0:
                ax.set_title(h_label(h))
            ax.grid(axis="x", color="0.92", lw=0.5)
            ax.set(xscale="log", xlabel=xlabel)
            el = {o: at[(h, o, f"{group} (% elements)")] for o in OPT_ORDER}
            if not any(el.values()):
                ax.text(0.5, 0.5, "no violations", ha="center", transform=ax.transAxes, color="0.45")
                continue
            ax.text(1.02, len(OPT_ORDER) - 0.5, "% el.", transform=ax.get_yaxis_transform(), fontsize=7.5,
                    color="0.45")
            for o, y in y_of.items():
                ax.text(1.02, y, pct(el[o], 0.001, "{:.2g}"), transform=ax.get_yaxis_transform(), va="center",
                        fontsize=7.5)
                s = sub[sub["optimizer"] == o]
                if s.empty:
                    ax.text(0.5, y, "none" if el[o] == 0 else "$<$ 100 violating elements", ha="center", va="center",
                            transform=ax.get_yaxis_transform(), fontsize=7.5, color="0.45")
                    continue
                col = ARM_COLORS[o]
                stat = lambda k: at[(h, o, f"{group} {k} ({unit})")] * scale
                ax.bxp([box_stats(s.assign(viol=s["viol"] * scale)) | {"mean": stat("Mean"), "fliers": [stat("Max")]}],
                       positions=[y], orientation="horizontal", widths=0.55, showmeans=True, patch_artist=True,
                       boxprops=dict(facecolor=col, edgecolor=col, alpha=0.4), whiskerprops=dict(color=col),
                       capprops=dict(color=col), medianprops=dict(color="0.15", lw=1.5),
                       meanprops=dict(marker="D", ms=4.5, mfc="white", mec="0.15"),
                       flierprops=dict(marker="x", ms=5, mec=col))
        log_ticks(axes[r][-1].xaxis)
    axes[0][0].set(yticks=list(y_of.values()), yticklabels=[OPT_LABELS[o] for o in y_of], ylim=(-0.6, 3.8))
    fig.text(0.5, 0.0, "Per violating element: box P25 to P75, line = median, diamond = mean, whiskers P1 to P99, "
             "cross = max; % el. = mean share of elements violated per sample",
             ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, f"violation_box_{CASE_LABELS[case]}")


def plot_h_grid(on, kind):
    """violation metric `kind` (key of H_YLABEL) over H; on = corrected_gap rows of the best forecasters"""
    piv = lambda m: on[on["metric"] == m].pivot_table(index=["case", "horizon"], columns="optimizer", values="mean")
    fig, axes = plt.subplots(len(H_ROWS), len(CASE_ORDER), figsize=(9, ROW_H * len(H_ROWS)), sharex=True, sharey="row")
    x = range(len(HORIZONS))
    for r, (group, unit, scale, *names) in enumerate(H_ROWS):
        el = piv(f"{group} (% elements)")
        mean = piv(f"{group} Mean ({unit})").reindex_like(el) * scale  # NaN without violations
        ev = {"el": el, "expected": (el / 100 * mean).where(el > 0, 0),
              "mean": mean.where(piv(f"{group} P99 ({unit})").reindex_like(el).notna())}[kind]
        floor = ev[ev > 0].min().min() / 10 ** 0.8  # zero row: hollow markers below the data (log axis)
        has_zero = False
        for c, case in enumerate(CASE_ORDER):
            ax, e = axes[r][c], ev.loc[case].reindex(index=HORIZONS, columns=OPT_ORDER)
            if r == 0:
                ax.set_title(CASE_LABELS[case])
            ax.grid(color="0.92", lw=0.5)
            if not (e > 0).any().any():
                ax.text(0.5, 0.5, "no or too few violations" if kind == "mean" else "no violations", ha="center",
                        transform=ax.transAxes, color="0.45")
                continue
            for o in OPT_ORDER:
                ax.plot(x, e[o].where(e[o] > 0), color=ARM_COLORS[o], ls=LS[o], marker="o", lw=1.5, ms=5)
                z = [i for i in x if e[o].iloc[i] == 0]
                ax.plot([i + DODGE[o] for i in z], [floor] * len(z), ls="none", marker="o", ms=5, mfc="none",
                        mec=ARM_COLORS[o])
                has_zero |= bool(z)
        ax.set(yscale="log", xticks=x, xticklabels=[h_label(h) for h in HORIZONS])
        log_ticks(ax.yaxis)
        if has_zero:
            sep = floor * 10 ** 0.4
            for a in axes[r]:
                a.axhline(sep, color="0.75", lw=0.6)
            ax.set_ylim(bottom=floor / 10 ** 0.3)
            ax.yaxis.set_major_locator(FixedLocator([t for t in ax.yaxis.get_majorticklocs() if t > sep] + [floor]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _, f=floor: "0" if v == f else f"{v:g}"))
        axes[r][0].set_ylabel(H_YLABEL[kind].format(*names))
    fig.legend(handles=optimizer_handles(), loc="upper center", ncol=len(OPT_ORDER), frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, f"violation_{kind}")


def main():
    df = keep_forecasters(pd.read_csv(IN))
    assert df[df["metric"].isin(ZERO)]["mean"].eq(0).all(), "results-text zero-check no longer true"
    assert df[(df["metric"] == "Thermal Overload (%)") & (df["case"] == "case14_ieee")]["mean"].eq(0).all()
    best = best_forecaster_per_metric(df, "Rebalanced Gap")[KEY]
    on = df.merge(best, on=KEY)
    for g in GROUPS:
        table_slide(on, *g)
    for kind in H_YLABEL:
        plot_h_grid(on, kind)
    cc = pd.read_csv(CCDF_IN).merge(best, on=KEY)
    for h in HORIZONS:
        plot_ccdf(cc, h)
    for case in CASE_ORDER:
        plot_box(on, cc, case)


if __name__ == "__main__":
    main()
