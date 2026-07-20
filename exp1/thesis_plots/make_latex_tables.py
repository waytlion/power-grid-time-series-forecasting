# Table 1:
#- performance metrics:
#-> mean opt gap,  mae_pd, mae_pg
#-> equality constraints with realized loads (active/reactive res),
#- over  different grid sizes for horizon = 24

# Table 2: same as Table 1, but case118 fixed and now over all horizons

# table 3: case118, and horizon = 24 fixed. metrics: inequality constraints, only the mean value pls.

# table 4:  horizon = 1 fixed, include all cases, metric: timing (training and inference)

"""One-shot: combined CSV -> 4 LaTeX tables (metrics as columns, models as rows). No config, just run it."""
import pandas as pd

IN = "exp1/results/combined/case118_combined.csv"
OUT_DIR = "exp1/results/combined"

CASE_LABELS = {"case118_ieee": "case118", "case500_goc": "case500"}
OPT_LABELS = {"ac": "AC", "dc": "DC", "surrogate": "Surrogate", "e2e": "E2E"}
FC_LABELS = {"snaive": "SNaive", "xgb": "XGB", "true": "True", "none": "None", "sarima": "SARIMA"}
OPT_ORDER = {"ac": 0, "dc": 1, "surrogate": 2, "e2e": 3}
FC_ORDER = {"snaive": 0, "sarima": 1, "xgb": 2, "true": 3, "none": 4}
STD_ARMS = {"surrogate", "e2e"}  # stochastic across seeds -> keep +/- std; ac/dc are deterministic

PERF_METRICS = {
    "Mean Optimality Gap": "Opt. Gap (mean)",
    "MAE Pd": "MAE $P_d$",
    "Generator Pg MAE": "MAE $P_g$",
    "Residual P (MAE)": "Res. P (MAE)",
    "Residual Q (MAE)": "Res. Q (MAE)",
}
INEQ_METRICS = {
    "pg viol. mean": "$P_g$ viol.",
    "qg viol. mean": "$Q_g$ viol.",
    "vm viol. mean": "$V_m$ viol.",
    "angle viol. mean": "Angle viol.",
    "thermal viol. mean": "Thermal viol.",
}
# infer_total_min / train_time_min are totals over the whole eval run (not per-sample);
# full_pipeline_infer_s is the per-sample (per-scenario) inference cost.
TIMING_METRICS = {
    "train_time_min": "Train (min, total)",
    "infer_total_min": "Infer (min, total)",
    "full_pipeline_infer_s": "Infer/sample (s)",
}

df = pd.read_csv(IN)
df["model"] = df["forecaster"].map(FC_LABELS) + "-" + df["optimizer"].map(OPT_LABELS)
df.loc[df["optimizer"] == "e2e", "model"] = "E2E"
df["case_label"] = df["case"].map(CASE_LABELS)
df["model_order"] = df["optimizer"].map(OPT_ORDER) * 10 + df["forecaster"].map(FC_ORDER)


def _format_val(row, mean_only):
    if mean_only or row["optimizer"] not in STD_ARMS:
        return f"{round(row['mean'], 3)}"
    return f"{round(row['mean'], 3)} $\\pm$ {round(row['std'], 3)}"


def write_table(sub, metrics, out_path, caption, label, group_col=None, group_labels=None, mean_only=False):
    """group_col: optional column ('case_label' or 'horizon') whose distinct values get their
    own \\multicolumn section header, so it's clear where each case/horizon's rows start."""
    sub = sub[sub["metric"].isin(metrics)].copy()
    sub["val"] = sub.apply(_format_val, axis=1, mean_only=mean_only)
    sub["metric_label"] = sub["metric"].map(metrics)

    cols = list(metrics.values())
    n_cols = 1 + len(cols)
    col_fmt = "l" + "r" * len(cols)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        "\\toprule",
        "Model & " + " & ".join(cols) + " \\\\",
        "\\midrule",
    ]

    def emit_rows(block):
        piv = block.pivot_table(index=["model", "model_order"], columns="metric_label",
                                 values="val", aggfunc="first")
        piv = piv.reindex(columns=cols).fillna("--")
        piv = piv.reset_index().sort_values("model_order")
        for _, r in piv.iterrows():
            lines.append(r["model"] + " & " + " & ".join(r[c] for c in cols) + " \\\\")

    if group_col is None:
        emit_rows(sub)
    else:
        groups = sorted(sub[group_col].dropna().unique())
        for i, g in enumerate(groups):
            if i > 0:
                lines.append("\\midrule")
            gl = group_labels(g) if group_labels else str(g)
            lines.append(f"\\multicolumn{{{n_cols}}}{{@{{}}l}}{{\\textit{{{gl}}}}} \\\\[2pt]")
            emit_rows(sub[sub[group_col] == g])

    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")


# Table 1: performance metrics across grid sizes, horizon=24 fixed
write_table(
    df[df["horizon"] == 24],
    PERF_METRICS,
    f"{OUT_DIR}/table1_perf_grid_h24.tex",
    "Performance metrics across grid sizes (horizon = 24).",
    "tab:perf_grid_h24",
    group_col="case_label",
    group_labels=lambda g: g,
)

# Table 2: performance metrics across horizons, case118 fixed
write_table(
    df[df["case"] == "case118_ieee"],
    PERF_METRICS,
    f"{OUT_DIR}/table2_perf_case118_horizons.tex",
    "Performance metrics across forecast horizons (case118).",
    "tab:perf_case118_horizons",
    group_col="horizon",
    group_labels=lambda h: f"Horizon $t+{h}$",
)

# Table 3: mean inequality constraint violations, case118 & horizon=24 fixed
write_table(
    df[(df["case"] == "case118_ieee") & (df["horizon"] == 24)],
    INEQ_METRICS,
    f"{OUT_DIR}/table3_ineq_case118_h24.tex",
    "Mean inequality constraint violations (case118, horizon = 24).",
    "tab:ineq_case118_h24",
    mean_only=True,
)

# Table 4: training/inference timing across grid sizes, horizon=1 fixed
# NOTE: at horizon=1 the timing source has no forecaster-training entries, so
# "Train (min, total)" reads 0 for every arm here -- that's a data gap upstream
# (timing_benchmark.csv has no h=1 rows), not a table bug. Non-zero at horizon=24.
write_table(
    df[df["horizon"] == 1],
    TIMING_METRICS,
    f"{OUT_DIR}/table4_timing_h1.tex",
    "Training and inference timing across grid sizes (horizon = 1).",
    "tab:timing_h1",
    group_col="case_label",
    group_labels=lambda g: g,
)
