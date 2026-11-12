"""One-shot: combined CSV -> RMSE figures. No config, just run it (from repo root).

  1) rmse_bars_<case>.pdf (one per case14 / case118 / case500): grid of barplots,
     rows = variable (Pg, Vm, Va), cols = horizon (1, 6, 24), bars = arm
     (AC / DC / Surrogate / E2E). 2-step arms use their best forecaster PER
     METRIC -- e.g. 2-step AC's XGBoost forecast is shown for Pg RMSE only if it
     beats SARIMA's Pg RMSE, independent of which forecaster wins for Vm or Va
     (unlike make_latex_tables.py, which fixes one forecaster per optimizer
     across all metrics, chosen by optimality gap).

  2) rmse_pd_lines.pdf: RMSE of Pd vs horizon, one lineplot row per case (grid
     size takes the row slot here since there's only one variable). 2-step arms
     (AC/DC/Surrogate) all dispatch the SAME predicted Pd, so they collapse to a
     single "2-step (best forecaster)" line instead of one line per optimizer.

  3) rmse_slide.pdf: one-slide Q1 summary, cols = case, x = horizon.
     - row 1: Pd RMSE, 2-step best forecaster vs E2E; E2E vs 2-step in % per horizon
     - row 2: Pg RMSE per optimizer (log y), best forecaster per metric; E2E vs AC in % per horizon
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

from colors import ARM_COLORS, OPT_LABELS, OPT_ORDER, THESIS_RC

IN = "exp1/results/combined/case118_combined.csv"
OUT_DIR = "exp1/thesis_plots/figures"

CASE_LABELS = {"case14_ieee": "case14", "case118_ieee": "case118", "case500_goc": "case500"}
CASE_ORDER = ["case14_ieee", "case118_ieee", "case500_goc"]
HORIZONS = [1, 6, 24]

BAR_VARS = [
    ("Generator Pg RMSE", "$P_g$ RMSE", "MW"),
    ("Vm RMSE", "$V_m$ RMSE", "p.u."),
    ("Va RMSE", "$V_a$ RMSE", "deg"),
]
PD_METRIC = "RMSE Pd"

plt.rcParams.update(THESIS_RC)


def load():
    return keep_forecasters(pd.read_csv(IN))


def h_label(h):
    """horizon label: H = h means all lead times 1..h evaluated together (not only h steps ahead)"""
    return f"$H={h}$"


EXCLUDED_FC = ("true", "xgb")  # oracle (no realistic counterpart); plain XGB, replaced by xgb_tot (labelled "XGB")


def keep_forecasters(df):
    """2-step forecaster set of every figure / table: drops EXCLUDED_FC"""
    return df[~df["forecaster"].isin(EXCLUDED_FC)]


def best_forecaster_per_metric(df, metric):
    """One row per (case, horizon, optimizer): the forecaster with the lowest RMSE for THIS
    metric specifically. E2E has a single forecaster ("none") so it passes through unchanged."""
    sub = df[df["metric"] == metric]
    idx = sub.groupby(["case", "horizon", "optimizer"])["mean"].idxmin()
    return sub.loc[idx]


def save(fig, name):
    # pdf = LaTeX, svg = PowerPoint (native vector)
    for ext in ("pdf", "svg"):
        fig.savefig(f"{OUT_DIR}/{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR}/{name}.{{pdf,svg}}")


def plot_bar_grid(df):
    best = {metric: best_forecaster_per_metric(df, metric) for metric, _, _ in BAR_VARS}

    for case in CASE_ORDER:
        fig, axes = plt.subplots(len(BAR_VARS), len(HORIZONS), figsize=(9, 7), sharey="row")
        row_top = [0.0] * len(BAR_VARS)
        for r, (metric, mlabel, unit) in enumerate(BAR_VARS):
            sub_case = best[metric][best[metric]["case"] == case]
            for c, h in enumerate(HORIZONS):
                ax = axes[r][c]
                sub = sub_case[sub_case["horizon"] == h]
                opts = [o for o in OPT_ORDER if o in set(sub["optimizer"])]
                vals = [sub.loc[sub["optimizer"] == o, "mean"].iloc[0] for o in opts]
                if vals:
                    row_top[r] = max(row_top[r], max(vals))
                bars = ax.bar(range(len(opts)), vals, color=[ARM_COLORS[o] for o in opts], width=0.65)
                ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals],
                             padding=2, fontsize=6.5, fontweight="bold")
                ax.set_xticks(range(len(opts)))
                ax.set_xticklabels([OPT_LABELS[o] for o in opts], rotation=20, ha="right", fontsize=7.5)
                ax.grid(axis="y", color="0.9", lw=0.6)
                ax.set_axisbelow(True)
                if r == 0:
                    ax.set_title(f"{h_label(h)}")
                if c == 0:
                    ax.set_ylabel(f"{mlabel}\n[{unit}]")
        for r in range(len(BAR_VARS)):
            axes[r][0].set_ylim(0, (row_top[r] or 1.0) * 1.18)

        fig.suptitle(f"RMSE by variable and horizon -- {CASE_LABELS[case]} "
                     "(2-step: best forecaster per metric)", y=1.01)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        save(fig, f"rmse_bars_{CASE_LABELS[case]}")


def best_forecaster_pd(df):
    """Pd RMSE depends only on the forecaster, not the downstream optimizer -- 2-step arms
    (AC/DC/Surrogate) forecast the SAME Pd, so pool them per (case, horizon, forecaster) and
    keep only the best forecaster's value, one row per (case, horizon)."""
    sub = df[(df["metric"] == PD_METRIC) & (df["optimizer"] != "e2e")]
    per_fc = sub.groupby(["case", "horizon", "forecaster"])["mean"].mean().reset_index()
    idx = per_fc.groupby(["case", "horizon"])["mean"].idxmin()
    best = per_fc.loc[idx].copy()
    best["series"] = "2-step (best forecaster)"

    e2e = df[(df["metric"] == PD_METRIC) & (df["optimizer"] == "e2e")].copy()
    e2e["series"] = "E2E"
    cols = ["case", "horizon", "forecaster", "mean", "series"]
    return pd.concat([best[cols], e2e[cols]], ignore_index=True)


def plot_pd_lines(df):
    data = best_forecaster_pd(df)
    series_colors = {"2-step (best forecaster)": ARM_COLORS["ac"], "E2E": ARM_COLORS["e2e"]}

    fig, axes = plt.subplots(len(CASE_ORDER), 1, figsize=(5.5, 8), sharex=True)
    row_top = 0.0
    for r, case in enumerate(CASE_ORDER):
        ax = axes[r]
        for series, color in series_colors.items():
            sub = data[(data["case"] == case) & (data["series"] == series)].sort_values("horizon")
            if sub.empty:
                continue
            row_top = max(row_top, sub["mean"].max())
            ax.plot(sub["horizon"], sub["mean"], "-o", color=color, ms=5, lw=2, label=series)
        ax.set_xticks(HORIZONS)
        ax.set_xticklabels([f"{h_label(h)}" for h in HORIZONS])
        ax.set_ylabel(f"{CASE_LABELS[case]}\n$P_d$ RMSE [MW]")
        ax.grid(axis="y", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
    for ax in axes:
        ax.set_ylim(0, (row_top or 1.0) * 1.1)
    axes[-1].set_xlabel("Forecast horizon")

    handles = [plt.Line2D([], [], color=c, marker="o", lw=2, label=s) for s, c in series_colors.items()]
    fig.suptitle("$P_d$ forecast RMSE by grid size and horizon", y=1.0)
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "rmse_pd_lines")


# ---- slide helpers (shared with plot_gap.py): cols = case, x = horizon at equal spacing
SLIDE_RC = {"font.size": 11, "axes.titlesize": 12}
X_OF = {h: i for i, h in enumerate(HORIZONS)}
LS = {o: "--" if o == "surrogate" else "-" for o in OPT_ORDER}  # surrogate ≈ AC on case14/118 -> keep AC visible


def pct_row(ax, e2e, ref, tag=None):
    """E2E vs ref in % per horizon along the bottom of ax (text ink, identity stays with the marks)
    - e2e, ref: mean per horizon; tag: row label (first column only)"""
    for h in HORIZONS:
        ax.text(X_OF[h], 0.04, f"{100 * (e2e[h] / ref[h] - 1):+.0f} %", transform=ax.get_xaxis_transform(),
                ha="center", fontsize=9.5)
    if tag:
        ax.text(-0.3, 0.12, tag, transform=ax.get_xaxis_transform(), fontsize=9, color="0.35")


def optimizer_lines(ax, sub, tag=False):
    """one line per optimizer (log y) + E2E vs AC in % per horizon; sub = best_forecaster_per_metric rows of one case
    - tag: print the 'E2E vs AC' row label (first column)"""
    sub = sub.sort_values("horizon")
    for o in OPT_ORDER:
        s = sub[sub["optimizer"] == o]
        ax.plot(s["horizon"].map(X_OF), s["mean"], marker="o", ls=LS[o], color=ARM_COLORS[o], lw=2, ms=6)
    ax.set_yscale("log")
    ax.set_ylim(sub["mean"].min() / 3, sub["mean"].max() * 1.6)
    ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    by_h = lambda o: sub[sub["optimizer"] == o].set_index("horizon")["mean"]
    pct_row(ax, by_h("e2e"), by_h("ac"), f"E2E vs {OPT_LABELS['ac']}" if tag else None)


def style_slide_axes(axes):
    for ax in axes.flat:
        ax.grid(axis="y", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xticks(range(len(HORIZONS)))
        ax.set_xticklabels([f"{h_label(h)}" for h in HORIZONS])
        ax.set_xlim(-0.3, len(HORIZONS) - 0.7)


def optimizer_handles():
    return [plt.Line2D([], [], color=ARM_COLORS[o], marker="o", ls=LS[o], lw=2, label=OPT_LABELS[o])
            for o in OPT_ORDER]


def finish_slide(fig, handles, name):
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.99, -0.01, "2-step: best forecaster per metric; means over seeds", ha="right", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, name)


def plot_slide(df):
    pd_data = best_forecaster_pd(df)
    pg = best_forecaster_per_metric(df, BAR_VARS[0][0])
    fc_color = "0.45"  # 2-step forecast is shared by all optimizers -> neutral, not an arm hue

    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(2, len(CASE_ORDER), figsize=(10, 5.8), sharex=True)
        for c, case in enumerate(CASE_ORDER):
            ax = axes[0][c]
            by_h = {}
            for series, color in (("2-step (best forecaster)", fc_color), ("E2E", ARM_COLORS["e2e"])):
                s = pd_data[(pd_data["case"] == case) & (pd_data["series"] == series)].sort_values("horizon")
                ax.plot(s["horizon"].map(X_OF), s["mean"], "-o", color=color, lw=2, ms=6)
                by_h[series] = s.set_index("horizon")["mean"]
            pct_row(ax, by_h["E2E"], by_h["2-step (best forecaster)"], "E2E vs 2-step" if c == 0 else None)
            ax.set_title(CASE_LABELS[case])
            top = ax.get_ylim()[1]
            ax.set_ylim(-0.25 * top, top)  # margin below 0 for the % row; axis itself starts at 0
            ax.set_yticks([t for t in ax.get_yticks() if 0 <= t <= top])
            ax.spines["left"].set_bounds(0, top)
            optimizer_lines(axes[1][c], pg[pg["case"] == case], tag=c == 0)
        style_slide_axes(axes)
        axes[0][0].set_ylabel("$P_d$ RMSE [MW]")
        axes[1][0].set_ylabel("$P_g$ RMSE [MW], log")
        handles = [plt.Line2D([], [], color=fc_color, marker="o", lw=2, label="2-step forecast (best)")]
        finish_slide(fig, handles + optimizer_handles(), "rmse_slide")


def plot_vmva_slide(df):
    """same layout as rmse_slide for the voltage variables; DC has no voltage magnitudes (no Vm line)"""
    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(2, len(CASE_ORDER), figsize=(10, 5.8), sharex=True)
        for r, (metric, label, unit) in enumerate(BAR_VARS[1:]):
            best = best_forecaster_per_metric(df, metric)
            for c, case in enumerate(CASE_ORDER):
                optimizer_lines(axes[r][c], best[best["case"] == case], tag=c == 0)
            axes[r][0].set_ylabel(f"{label} [{unit}], log")
        for c, case in enumerate(CASE_ORDER):
            axes[0][c].set_title(CASE_LABELS[case])
        style_slide_axes(axes)
        finish_slide(fig, optimizer_handles(), "rmse_vmva_slide")


def main():
    df = load()
    plot_bar_grid(df)
    plot_pd_lines(df)
    plot_slide(df)
    plot_vmva_slide(df)


if __name__ == "__main__":
    main()
