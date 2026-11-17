# Table 1: core metrics (opt gap, active/reactive res) across grid sizes at horizon = 6,
#          arms filtered to the best forecaster per optimizer (by opt gap)
# Table 2: full performance metrics (opt gap, MAE Pd/Pg/Vm/Va, active/reactive res),
#          case118 fixed, over all horizons, best forecaster per optimizer
# Table 3: case118, horizon = 24 fixed. inequality constraints, mean value only
# Table 4: horizon = 1 fixed, all cases, timing (training + inference)
# Table 5 (appendix): full performance metrics across grid sizes over all horizons
# Table 6: core metrics (as Table 1), case118 fixed, over all horizons, best forecaster per optimizer
# Table 7: raw + rebalanced opt gap (ex-post AC-PF, distributed slack), case118, all horizons, best forecaster
# Table 8 (appendix, Q2): gap + supply, all grids and horizons
# Table 9 (Q3): rebalanced gap + limit violations after the ex-post AC-PF, all grids, horizon = 6 (frequency + magnitude)
# Table 10 / 11 (appendix, Q3): violation frequency / magnitude, all grids and horizons
# Table 12 (appendix, Q1): RMSE of the predicted state, all grids and horizons (forecaster rule of rmse_slide)

"""One-shot: combined CSV (+ corrected_gap CSV) -> 12 LaTeX tables (metrics as columns, models as rows). No config, just run it.
Ex-post variant (Q limits on / off) of tables 7-11: plot_rmse.VARIANT (env EXPOST_VARIANT)."""
import math

import pandas as pd

from colors import OPT_LABELS as ARM_LABELS
from plot_rmse import CORR_IN, QLIM, best_forecaster_pd, best_forecaster_per_metric, h_label, keep_forecasters

IN = "exp1/results/combined/case118_combined.csv"
OUT_DIR = "exp1/results/combined"

CASE_LABELS = {"case14_ieee": "case14", "case118_ieee": "case118", "case500_goc": "case500"}
CASE_ORDER = {"case14": 0, "case118": 1, "case500": 2}  # by grid size, not lexical
OPT_LABELS = {"ac": "AC", "dc": "DC", "surrogate": "Surrogate", "e2e": "E2E"}
FC_LABELS = {"snaive": "SNaive", "xgb_tot": "XGB", "true": "True", "none": "None", "sarima": "SARIMA"}
OPT_ORDER = {"ac": 0, "dc": 1, "surrogate": 2, "e2e": 3}
FC_ORDER = {"snaive": 0, "sarima": 1, "xgb_tot": 3, "true": 4, "none": 5}

PERF_METRICS = {
    "Mean Optimality Gap": "Opt. Gap \\% (mean)",
    "Residual P (MAE)": "Res. P (MAE)",
    "Residual Q (MAE)": "Res. Q (MAE)",
    "MAE Pd": "MAE $P_d$",
    "Generator Pg MAE": "MAE $P_g$",
    "Vm MAE": "MAE $V_m$",
    "Va MAE": "MAE $V_a$",
}
# Core comparison metrics: optimality gap + power-balance residuals (realized loads)
CORE_METRICS = {
    "Mean Optimality Gap": "Opt. Gap \\% (mean)",
    "Residual P (MAE)": "Res. P (MAE)",
    "Residual Q (MAE)": "Res. Q (MAE)",
}
# rebalanced opt gap (per scenario): PF-solved dispatch of the ex-post AC-PF (distributed slack), re-costed.
# ML arms: mean over seeds (corrected_gap.py aggregates the per-seed ex-post PF runs)
CORR_GAP_METRICS = {
    "Optimality Gap (recomputed)": "Opt. Gap \\%",
    "Signed Optimality Gap": "Signed Opt. Gap \\%",
    "Mean dPg (MW)": "$\\Delta P$ (MW)",
    "Mean |dPg| (%)": "$|\\Delta P|$ \\%",
    "Rebalancing Cost (under-supply)": "$R^{+}$ \\%",
    "Rebalancing Cost (over-supply)": "$R^{-}$ \\%",
    "Rebalanced Gap": "Rebal. Gap \\%",
    "Signed Rebalanced Gap": "Signed Rebal. Gap \\%",
    "Thermal Overload (%)": "Thermal \\%",
}
INEQ_METRICS = {
    "pg viol. mean": "$P_g$ viol.",
    "qg viol. mean": "$Q_g$ viol.",
    "vm viol. mean": "$V_m$ viol.",
    "angle viol. mean": "Angle viol.",
    "thermal viol. mean": "Thermal viol.",
}
# full pipeline (forecast + OPF step): total over the whole eval run and per scenario
TIMING_METRICS = {
    "train_time_min": "Train (min, total)",
    "full_pipeline_infer_total_min": "Infer (min, total)",
    "full_pipeline_infer_s": "Infer/sample (s)",
}

df = pd.concat([pd.read_csv(IN), pd.read_csv(CORR_IN)], ignore_index=True)
df = keep_forecasters(df)
vm = df[df["metric"].str.match(r"Vm Violation (Mean|Median|P99) \(p\.u\.\)")]
df = pd.concat([df, vm.assign(metric=vm["metric"].str.replace("(p.u.)", "(1e-3 p.u.)", regex=False),
                              mean=vm["mean"] * 1e3)], ignore_index=True)  # 3-decimal tables flatten p.u. values
# Q3 mean per violating element: same rule as the quantiles (>= corrected_gap.MIN_EL elements, seeds pooled; P99 NaN below)
p99 = df[df["metric"].str.contains(" P99 (", regex=False)].dropna(subset=["mean"])
enough = set(zip(p99["case"], p99["horizon"], p99["optimizer"], p99["forecaster"],
                 p99["metric"].str.replace(" P99 (", " Mean (", regex=False)))
key = zip(df["case"], df["horizon"], df["optimizer"], df["forecaster"], df["metric"])
df = df[~df["metric"].str.match(r"(Thermal Overload|Vm Violation|Qg Violation) Mean \(")
        | pd.Series([k in enough for k in key], index=df.index)]
df["model"] =df["forecaster"].map(FC_LABELS) + "-" + df["optimizer"].map(OPT_LABELS)
df.loc[df["optimizer"] == "e2e", "model"] = "E2E"
df["case_label"] = df["case"].map(CASE_LABELS)
df["model_order"] = df["optimizer"].map(OPT_ORDER) * 10 + df["forecaster"].map(FC_ORDER)


def filter_best_forecaster(d, metric="Mean Optimality Gap"):
    """Keep the best forecaster per (case, horizon, optimizer), chosen by lowest `metric` (default: mean optimality
    gap) -- one row per arm (AC / DC / Surrogate / E2E). Best is picked per horizon so a forecaster
    missing a hard horizon can't win on a partial-coverage average. E2E has a single forecaster
    (none) so it passes through unchanged."""
    gap = d[d["metric"] == metric]
    best = gap.loc[gap.groupby(["case", "horizon", "optimizer"])["mean"].idxmin()]
    keep = set(zip(best["case"], best["horizon"], best["optimizer"], best["forecaster"]))
    key = zip(d["case"], d["horizon"], d["optimizer"], d["forecaster"])
    return d[pd.Series([k in keep for k in key], index=d.index)]


def write_table(sub, metrics, out_path, caption, label, group_col=None, group_labels=None,
                sub_group_col=None, sub_group_labels=None, longtable=False, fmt=None, header_groups=None,
                with_std=False, bold_best=False):
    """group_col: optional column ('case_label' or 'horizon') whose distinct values get their
    own \\multicolumn section header, so it's clear where each case/horizon's rows start.
    sub_group_col: optional second-level column, emitted as an indented sub-header within
    each group_col section (e.g. case as major section, horizon as minor).
    longtable=True emits a longtable that breaks across pages and repeats the header, instead
    of a resizebox'd float (which can't page-break and so gets cut off when too tall).
    Longtable in \\footnotesize with narrow column padding (no resizebox possible).
    Requires \\usepackage{longtable} (+ booktabs) in the preamble.
    fmt: value -> str (default: rounded to 3 decimals).
    header_groups: [(title, n_columns), ...] over the metric columns -> extra header row with \\cmidrule
    with_std: append ' ± std' where a std over seeds exists (> 1 seed); needs fmt
    bold_best: lowest value per row block + column in bold (ties at the printed precision all bold; skipped when
      all rows print the same)"""
    sub = sub[sub["metric"].isin(metrics)].dropna(subset=["mean"]).copy()  # undefined (e.g. no violation) -> "--"
    sub["val"] = sub["mean"].map(fmt) if fmt else sub["mean"].round(3).astype(str)
    if with_std:
        has = sub["std"].notna()
        sub.loc[has, "val"] += " $\\pm$ " + sub.loc[has, "std"].map(fmt)

    cols = list(metrics)
    n_cols = 1 + len(cols)
    col_fmt = "l" + "r" * len(cols)
    header = "Model & " + " & ".join(metrics.values()) + " \\\\"
    if header_groups:
        starts = [2 + sum(n for _, n in header_groups[:i]) for i in range(len(header_groups))]
        header = (" & " + " & ".join(f"\\multicolumn{{{n}}}{{c}}{{{t}}}" for t, n in header_groups) + " \\\\\n"
                  + " ".join(f"\\cmidrule(lr){{{s}-{s + n - 1}}}" for s, (_, n) in zip(starts, header_groups))
                  + "\n" + header)

    if longtable:
        lines = [
            "\\begingroup\\footnotesize\\setlength{\\tabcolsep}{3pt}",  # no resizebox in a longtable -> shrink to fit
            f"\\begin{{longtable}}{{{col_fmt}}}",
            f"\\caption{{{caption}}}\\label{{{label}}}\\\\",
            "\\toprule", header, "\\midrule", "\\endfirsthead",
            f"\\multicolumn{{{n_cols}}}{{c}}{{\\tablename\\ \\thetable{{}} -- continued}} \\\\",
            "\\toprule", header, "\\midrule", "\\endhead",
            "\\midrule",
            f"\\multicolumn{{{n_cols}}}{{r}}{{\\textit{{continued on next page}}}} \\\\",
            "\\endfoot",
            "\\bottomrule", "\\endlastfoot",
        ]
    else:
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            # shrink only when wider than the text block; never upscale a narrow table
            # (upscaling blows the font up and the table spills the page height)
            "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%",
            f"\\begin{{tabular}}{{{col_fmt}}}",
            "\\toprule", header, "\\midrule",
        ]

    def _sorted_groups(block, col):
        # explicit case order (by grid size); else numeric-aware so horizons read 1, 6, 24
        vals = list(block[col].dropna().unique())
        if all(v in CASE_ORDER for v in vals):
            return sorted(vals, key=CASE_ORDER.get)
        try:
            return sorted(vals, key=lambda v: float(v))
        except (TypeError, ValueError):
            return sorted(vals)

    def emit_rows(block):
        if bold_best:
            block = block.copy()
            for _, g in block.groupby("metric"):
                if g["val"].nunique() > 1:
                    best = g.loc[g["mean"].idxmin(), "val"]
                    block.loc[g.index[g["val"] == best], "val"] = f"\\textbf{{{best}}}"
        piv = block.pivot_table(index=["model", "model_order"], columns="metric",
                                 values="val", aggfunc="first")
        piv = piv.reindex(columns=cols).fillna("--")
        piv = piv.reset_index().sort_values("model_order")
        for _, r in piv.iterrows():
            lines.append(r["model"] + " & " + " & ".join(r[c] for c in cols) + " \\\\")

    def emit_section(block):
        if sub_group_col is None:
            emit_rows(block)
            return
        for j, sg in enumerate(_sorted_groups(block, sub_group_col)):
            if j > 0:
                lines.append("\\addlinespace[2pt]")
            sgl = sub_group_labels(sg) if sub_group_labels else str(sg)
            lines.append(f"\\multicolumn{{{n_cols}}}{{@{{}}l}}{{\\quad {sgl}}} \\\\[2pt]")
            emit_rows(block[block[sub_group_col] == sg])

    if group_col is None:
        emit_section(sub)
    else:
        groups = _sorted_groups(sub, group_col)
        for i, g in enumerate(groups):
            if i > 0:
                lines.append("\\midrule")
            gl = group_labels(g) if group_labels else str(g)
            lines.append(f"\\multicolumn{{{n_cols}}}{{@{{}}l}}{{\\textit{{{gl}}}}} \\\\[2pt]")
            emit_section(sub[sub[group_col] == g])

    if longtable:
        lines += ["\\end{longtable}", "\\endgroup", ""]
    else:
        lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")


# Table 1: core metrics across grid sizes at horizon = 6, best 2-step forecaster per optimizer
write_table(
    filter_best_forecaster(df[df["horizon"] == 6]),
    CORE_METRICS,
    f"{OUT_DIR}/table1_core_grid_h6.tex",
    "Optimality gap and power-balance residuals across grid sizes "
    "(horizon $H=6$; best 2-step forecaster per optimizer).",
    "tab:core_grid_h6",
    group_col="case_label",
    group_labels=lambda g: g,
)

# Table 2: full performance metrics across horizons, case118 fixed, best forecaster per optimizer
write_table(
    filter_best_forecaster(df[df["case"] == "case118_ieee"]),
    PERF_METRICS,
    f"{OUT_DIR}/table2_perf_case118_horizons.tex",
    "Performance metrics across forecast horizons (case118; best forecaster per optimizer).",
    "tab:perf_case118_horizons",
    group_col="horizon",
    group_labels=lambda h: f"Horizon {h_label(int(h))}",
)

# Table 3: mean inequality constraint violations, case118 & horizon=24 fixed (DC excluded --
# DC has no inequality constraints in its formulation, so its rows are meaningless here)
write_table(
    df[(df["case"] == "case118_ieee") & (df["horizon"] == 24) & (df["optimizer"] != "dc")],
    INEQ_METRICS,
    f"{OUT_DIR}/table3_ineq_case118_h24.tex",
    "Mean inequality constraint violations (case118, horizon = 24).",
    "tab:ineq_case118_h24",
)

# Table 4: training/inference timing across grid sizes, horizon=1 fixed
# longtable: paginates instead of overflowing the page height (spills vertically as a float).
write_table(
    df[df["horizon"] == 1],
    TIMING_METRICS,
    f"{OUT_DIR}/table4_timing_h1.tex",
    "Training and inference timing across grid sizes (horizon = 1).",
    "tab:timing_h1",
    group_col="case_label",
    group_labels=lambda g: g,
    longtable=True,
)

# Table 5 (appendix): full performance metrics across grid sizes, all forecast horizons
# (case = major section, horizon = minor sub-header within each case)
write_table(
    df,
    PERF_METRICS,
    f"{OUT_DIR}/table5_perf_grid_all_horizons.tex",
    "Performance metrics across grid sizes and forecast horizons.",
    "tab:perf_grid_all_horizons",
    group_col="case_label",
    group_labels=lambda g: g,
    sub_group_col="horizon",
    sub_group_labels=lambda h: f"Horizon {h_label(int(h))}",
    longtable=True,  # spans grid x all horizons -> too tall for one page; paginate
)

# Table 6: core metrics (as Table 1) across horizons, case118 fixed, best forecaster per optimizer
write_table(
    filter_best_forecaster(df[df["case"] == "case118_ieee"]),
    CORE_METRICS,
    f"{OUT_DIR}/table6_core_case118_horizons.tex",
    "Optimality gap and power-balance residuals across forecast horizons "
    "(case118; best 2-step forecaster per optimizer).",
    "tab:core_case118_horizons",
    group_col="horizon",
    group_labels=lambda h: f"Horizon {h_label(int(h))}",
)

# Table 7: raw + rebalanced opt gap (ex-post AC-PF, distributed slack), case118, all horizons
write_table(
    filter_best_forecaster(df[df["case"] == "case118_ieee"]),
    CORR_GAP_METRICS,
    f"{OUT_DIR}/table7_corr_gap_case118_horizons.tex",
    "Raw and rebalanced optimality gap across forecast horizons (case118; best 2-step forecaster "
    "per optimizer; ML arms: mean over seeds). Rebal.: AC power flow on the realised load with the predicted "
    "dispatch as setpoints; the mismatch (forecast error and losses) is distributed over all in-service "
    "units in proportion to $P_{\\max}$ within their limits (distributed slack), and the PF-solved dispatch "
    "is re-costed. " + ("Generator reactive limits are enforced. " if QLIM else "") + "DC-OPF has no voltages: its dispatch runs with the case-file generator voltage setpoints. "
    "$R^{+}$/$R^{-}$: under-/over-supply contributions to the mean rebalancing cost (sum = mean). "
    "Thermal: share of samples with a line overload.",
    "tab:corr_gap_case118_horizons",
    group_col="horizon",
    group_labels=lambda h: f"Horizon {h_label(int(h))}",
)

# Table 8 (appendix, Q2): gaps + supply error across grids and horizons; forecaster per arm = lowest rebalanced gap
# (same operating state as gap_slide)
write_table(
    filter_best_forecaster(df, "Rebalanced Gap"),
    {"Optimality Gap (recomputed)": "raw", "Rebalanced Gap": "rebal.",
     "Mean dPg (MW)": "$\\Delta P$", "Mean |dPg| (MW)": "$|\\Delta P|$"},
    f"{OUT_DIR}/table8_gap_supply_grid_all_horizons.tex",
    "Raw and rebalanced optimality gap and the supply error "
    "$\\Delta P = \\sum_g \\hat P_g - \\sum_g P^{*}_g$ "
    "(signed and absolute mean, MW) across grids and horizons. 2-step: forecaster with the lowest rebalanced gap per "
    "optimizer; ML arms: mean over seeds.",
    "tab:gap_supply_grid_all_horizons",
    group_col="case_label",
    group_labels=lambda g: g,
    sub_group_col="horizon",
    sub_group_labels=lambda h: f"Horizon {h_label(int(h))}",
    longtable=True,
    header_groups=[("Optimality gap (\\%)", 2), ("Supply error (MW)", 2)],
)


# Q3: limit violations after the ex-post AC-PF; forecaster per arm = lowest rebalanced gap (same operating state as Q2)
Q3_GROUPS = {  # constraint group -> (header, magnitude unit; None = frequency only)
    "Thermal Overload": ("Thermal (lines)", "% of rating"),
    # Q limits on: Qg violation 0 by construction -> the shortfall shows as units at their limit (lost voltage control)
    **({"Vm Violation": ("$|V|$ (buses)", "1e-3 p.u."),
        "Lost Voltage Control": ("Lost volt. control (gen. buses)", None)} if QLIM else
       {"Vm Violation": ("$|V|$ (PQ buses)", "1e-3 p.u."),
        "Qg Violation": ("$Q_{\\mathrm{g}}$ (generator buses)", "MVAr")}),
}
EL_SETS = ("lines, all buses, generator buses that can control voltage" if QLIM else
           "lines, PQ buses, generator buses without the reference bus")
QLIM_NOTE = ("Generator reactive limits enforced: a unit at its limit stops controlling its bus voltage (lost volt. "
             "control: voltage off its setpoint). Diverged: no power-flow solution (infeasible; share of test samples, "
             "not in the other columns). ")
Q3_COLS = {  # column kind -> (metric template (group, unit), label, caption term)
    "smp": ("{} (%)", "\\% smp.", "share of test samples with at least one violation (\\% smp.)"),
    "el": ("{} (% elements)", "\\% el.", f"mean share of elements violated per sample (\\% el.; {EL_SETS})"),
    "mean": ("{} Mean ({})", "mean", "mean"),
    "median": ("{} Median ({})", "median", "median"),
    "p99": ("{} P99 ({})", "P99", "99th percentile (P99)"),
}
SEVERITY = ("mean", "median", "p99")  # statistics of the violation over the violating elements
GAP_REB = "$\\gamma^{\\mathrm{reb}}$"  # = MA-tex \gapreb (tables_to_svg has no notation.tex)


def q3_fmt(v):
    """exact 0 -> '0'; < 0.1: 2 significant digits (tiny shares stay visible); else 2 decimals"""
    if v == 0:
        return "0"
    return f"{v:.{1 - math.floor(math.log10(abs(v)))}f}" if abs(v) < 0.1 else f"{v:.2f}"


def q3_table(d, kinds, name, title, label, gap=False, **kw):
    """one Q3 table: per constraint group the columns `kinds` (keys of Q3_COLS); gap: rebalanced gap first;
    Q limits on: + diverged share in frequency tables"""
    cols = {g: [k for k in kinds if unit or k not in SEVERITY] for g, (_, unit) in Q3_GROUPS.items()}
    div = QLIM and "smp" in kinds
    metrics = {"Rebalanced Gap": "\\%"} if gap else {}
    metrics |= {Q3_COLS[k][0].format(g, Q3_GROUPS[g][1]): Q3_COLS[k][1] for g, ks in cols.items() for k in ks}
    metrics |= {"Diverged (%)": "\\% smp."} if div else {}
    terms = [Q3_COLS[k][2] for k in kinds if k not in SEVERITY]
    if sev := [Q3_COLS[k][2] for k in kinds if k in SEVERITY]:
        terms.append(", ".join(sev) + " of the violation over all violating elements (thermal in \\% of the rating "
                     "above 100\\,\\%, $|V|$ in $10^{-3}$\\,p.u." + ("" if QLIM else ", $Q_{\\mathrm{g}}$ in MVAr")
                     + "; only with at least 100 violating elements, ML seeds pooled)")
    block = "grid, horizon" if "sub_group_col" in kw else "grid"
    caption = (("Rebalanced optimality gap and limit" if gap else "Limit")
               + f" violations after the ex-post AC power flow{title}. Lower is better for every metric; bold: best "
               f"arm per {block} and column. " + (f"{GAP_REB}: rebalanced optimality gap (\\%). " if gap else "")
               + (QLIM_NOTE if QLIM else "")
               + "Per constraint group: " + "; ".join(terms) + ". 0 = no violation"
               + ("; -- = fewer than 100 violating elements" if sev else "")
               + ". 2-step: forecaster with the lowest rebalanced gap per optimizer; ML arms: mean over seeds"
               + (" (quantiles: pooled seeds)" if sev else "")
               + ". DC: generator voltage setpoints from the case file (1.0\\,p.u.), since DC-OPF has no voltages.")
    write_table(filter_best_forecaster(d, "Rebalanced Gap"), metrics, f"{OUT_DIR}/{name}.tex", caption, label,
                group_col="case_label", group_labels=lambda g: g, fmt=q3_fmt, bold_best=True,
                header_groups=[(GAP_REB, 1)] * gap + [(Q3_GROUPS[g][0], len(ks)) for g, ks in cols.items() if ks]
                + [("Diverged", 1)] * div, **kw)


ALL_H = dict(sub_group_col="horizon", sub_group_labels=lambda h: f"Horizon {h_label(int(h))}", longtable=True)
# Table 9 (main text, Q3): H = 6, rebalanced gap + frequency + severity
q3_table(df[df["horizon"] == 6], ("smp", "el", "mean", "p99"), "table9_violations_h6", f" ({h_label(6)})",
         "tab:violations_h6", gap=True)
# Tables 10 + 11 (appendix, Q3): all horizons, frequency / severity (one table would be too wide)
q3_table(df, ("smp", "el"), "table10_violations_share_all_horizons", ": frequency",
         "tab:violations_share_all_horizons", **ALL_H)
q3_table(df, ("mean", "median", "p99"), "table11_violations_magnitude_all_horizons", ": magnitude",
         "tab:violations_magnitude_all_horizons", **ALL_H)

# Table 12 (appendix, Q1): RMSE per arm, all grids and horizons; 2-step forecaster per variable as in rmse_slide
# - Pd: the shared 2-step forecast (best forecaster, same for all 2-step arms) repeated per 2-step row
# - reference row: 2-step AC with the SNaive load forecast (load-driven error of a naive forecast, Q1 voltages)
Q1_RMSE = {"RMSE Pd": "$P_d$ [MW]", "Generator Pg RMSE": "$P_g$ [MW]", "Vm RMSE": "$|V|$ [$10^{-3}$\\,p.u.]",
           "Va RMSE": "$\\theta$ [deg]"}
fc = best_forecaster_pd(df)
pd_e2e = best_forecaster_per_metric(df, "RMSE Pd").query("optimizer == 'e2e'")  # keeps its seed std
q1 = pd.concat([fc[fc["series"] != "E2E"].assign(optimizer=o, metric="RMSE Pd") for o in ("ac", "dc", "surrogate")]
               + [pd_e2e] + [best_forecaster_per_metric(df, m) for m in list(Q1_RMSE)[1:]], ignore_index=True)
q1 = q1.assign(model=q1["optimizer"].map(ARM_LABELS), model_order=q1["optimizer"].map(OPT_ORDER))
sn = df[(df["optimizer"] == "ac") & (df["forecaster"] == "snaive") & df["metric"].isin(Q1_RMSE)]
q1 = pd.concat([q1, sn.assign(model="2-step AC (SNaive)", model_order=OPT_ORDER["ac"] + 0.5)], ignore_index=True)
q1.loc[q1["metric"] == "Vm RMSE", ["mean", "std"]] *= 1e3
q1["case_label"] = q1["case"].map(CASE_LABELS)
write_table(q1, Q1_RMSE, f"{OUT_DIR}/table12_rmse_all_horizons.tex",
            "RMSE of the predicted operating state per arm across grids and horizons. 2-step: best forecaster per "
            "variable; $P_d$ is the forecast shared by all 2-step arms. 2-step AC (SNaive): 2-step AC with the "
            "seasonal naive load forecast, as a reference. ML arms: mean $\\pm$ std over three seeds (E2E "
            "on case500: two seeds at $H=1$, one seed at $H=6$ and $H=24$, no std). -- = not predicted (DC-OPF has no "
            "voltage magnitudes).",
            "tab:rmse_all_horizons", group_col="case_label", group_labels=lambda g: g, fmt=lambda v: f"{v:.2f}",
            with_std=True, **ALL_H)
