# Table 1: core metrics (opt gap, active/reactive res) across grid sizes at horizon = 6,
#          arms filtered to the best forecaster per optimizer (by opt gap)
# Table 2: full performance metrics (opt gap, MAE Pd/Pg/Vm/Va, active/reactive res),
#          case118 fixed, over all horizons, best forecaster per optimizer
# Table 3: case118, horizon = 24 fixed. inequality constraints, mean value only
# Table 4: horizon = 1 fixed, all cases, timing (training + inference)
# Table 5 (appendix): full performance metrics across grid sizes over all horizons
# Table 6: core metrics (as Table 1), case118 fixed, over all horizons, best forecaster per optimizer
# Table 7: raw + rebalanced opt gap (ex-post AC-PF, distributed slack), case118, all horizons, best forecaster

"""One-shot: combined CSV (+ corrected_gap.csv) -> 7 LaTeX tables (metrics as columns, models as rows). No config, just run it."""
import pandas as pd

IN = "exp1/results/combined/case118_combined.csv"
CORR_IN = "exp1/results/combined/corrected_gap.csv"  # exp1/generate_metrics/corrected_gap.py
OUT_DIR = "exp1/results/combined"

CASE_LABELS = {"case14_ieee": "case14", "case118_ieee": "case118", "case500_goc": "case500"}
CASE_ORDER = {"case14": 0, "case118": 1, "case500": 2}  # by grid size, not lexical
OPT_LABELS = {"ac": "AC", "dc": "DC", "surrogate": "Surrogate", "e2e": "E2E"}
FC_LABELS = {"snaive": "SNaive", "xgb": "XGB", "true": "True", "none": "None", "sarima": "SARIMA"}
OPT_ORDER = {"ac": 0, "dc": 1, "surrogate": 2, "e2e": 3}
FC_ORDER = {"snaive": 0, "sarima": 1, "xgb": 2, "true": 3, "none": 4}

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
# ML arms seed 0 (same dumps as the raw gap column)
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
df = df[df["forecaster"] != "true"]  # drop the perfect-info oracle arm (no realistic counterpart)
df["model"] = df["forecaster"].map(FC_LABELS) + "-" + df["optimizer"].map(OPT_LABELS)
df.loc[df["optimizer"] == "e2e", "model"] = "E2E"
df["case_label"] = df["case"].map(CASE_LABELS)
df["model_order"] = df["optimizer"].map(OPT_ORDER) * 10 + df["forecaster"].map(FC_ORDER)


def filter_best_forecaster(d):
    """Keep the best forecaster per (case, horizon, optimizer), chosen by lowest mean optimality
    gap -- one row per arm (AC / DC / Surrogate / E2E). Best is picked per horizon so a forecaster
    missing a hard horizon can't win on a partial-coverage average. E2E has a single forecaster
    (none) so it passes through unchanged."""
    gap = d[d["metric"] == "Mean Optimality Gap"]
    best = gap.loc[gap.groupby(["case", "horizon", "optimizer"])["mean"].idxmin()]
    keep = set(zip(best["case"], best["horizon"], best["optimizer"], best["forecaster"]))
    key = zip(d["case"], d["horizon"], d["optimizer"], d["forecaster"])
    return d[pd.Series([k in keep for k in key], index=d.index)]


def write_table(sub, metrics, out_path, caption, label, group_col=None, group_labels=None,
                sub_group_col=None, sub_group_labels=None, longtable=False):
    """group_col: optional column ('case_label' or 'horizon') whose distinct values get their
    own \\multicolumn section header, so it's clear where each case/horizon's rows start.
    sub_group_col: optional second-level column, emitted as an indented sub-header within
    each group_col section (e.g. case as major section, horizon as minor).
    longtable=True emits a longtable that breaks across pages and repeats the header, instead
    of a resizebox'd float (which can't page-break and so gets cut off when too tall).
    Requires \\usepackage{longtable} (+ booktabs) in the preamble."""
    sub = sub[sub["metric"].isin(metrics)].copy()
    sub["val"] = sub["mean"].round(3).astype(str)
    sub["metric_label"] = sub["metric"].map(metrics)

    cols = list(metrics.values())
    n_cols = 1 + len(cols)
    col_fmt = "l" + "r" * len(cols)
    header = "Model & " + " & ".join(cols) + " \\\\"

    if longtable:
        lines = [
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
        piv = block.pivot_table(index=["model", "model_order"], columns="metric_label",
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
        lines += ["\\end{longtable}", ""]
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
    "(horizon $t+6$; best 2-step forecaster per optimizer).",
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
    group_labels=lambda h: f"Horizon $t+{int(h)}$",
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
    sub_group_labels=lambda h: f"Horizon $t+{int(h)}$",
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
    group_labels=lambda h: f"Horizon $t+{int(h)}$",
)

# Table 7: raw + rebalanced opt gap (ex-post AC-PF, distributed slack), case118, all horizons
write_table(
    filter_best_forecaster(df[df["case"] == "case118_ieee"]),
    CORR_GAP_METRICS,
    f"{OUT_DIR}/table7_corr_gap_case118_horizons.tex",
    "Raw and rebalanced optimality gap across forecast horizons (case118; best 2-step forecaster "
    "per optimizer; ML arms seed 0). Rebal.: AC power flow on the realised load with the predicted "
    "dispatch as setpoints; the mismatch (forecast error and losses) is distributed over all in-service "
    "units in proportion to $P_{\\max}$ within their limits (distributed slack), and the PF-solved dispatch "
    "is re-costed. DC-OPF has no voltages: its dispatch runs with the case-file generator voltage setpoints. "
    "$R^{+}$/$R^{-}$: under-/over-supply contributions to the mean rebalancing cost (sum = mean). "
    "Thermal: share of samples with a line overload.",
    "tab:corr_gap_case118_horizons",
    group_col="horizon",
    group_labels=lambda h: f"Horizon $t+{int(h)}$",
)
