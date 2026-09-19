"""
Main CLI script for comparing two-step OPF approach against ground truth.

Usage (from Thesis_Repo root):
    python exp1/generate_metrics/compare.py \
        --predicted-opf-base-dir exp1/data/data_out/case118_horizon_1_3yr \
        --ground-truth-dir data/data_out/3yr_2019-2021/case118_ieee/raw \
        --output-dir exp1/results/case118_horizon1_3yr2019-2021 \
        --dataset case118_ieee \
        --forecasts-parquet exp1/data/data_in/XX.parquet
        
    Or with defaults:
    python exp1/generate_metrics/compare.py  # Uses default paths
    
    Compare specific methods only:
    python exp1/generate_metrics/compare.py --methods xgb sarima
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from config import FORECAST_METHODS, OUTPUT_TEMPLATES, FORECASTS_PARQUET, FORECAST_SEASONALITY
from loaders import (
    load_forecasts,
    load_datakit_bus,
    load_datakit_gen,
    load_datakit_branch,
    align_opf_results,
    align_branch_results,
    has_dc_columns,
)
from metrics import (
    compute_rmse_by_bus_type,
    compute_generator_rmse,
    compute_cost_metrics,
    compute_forecast_metrics_table,
    compute_algebraic_power_residuals,
    compute_dc_rmse_by_bus_type,
    compute_dc_generator_rmse,
    compute_dc_cost_metrics,
    compute_dc_branch_rmse,
    compute_dc_algebraic_power_residuals,
    compute_ac_branch_rmse,
    compute_feasibility_metrics,
)


# Canonical physical units per constraint type (used for both AC and DC rows).
_CT_UNITS = {"vm": "p.u.", "pg": "MW", "qg": "MVAr", "thermal": "MVA", "angle": "deg"}


def _pool_by_feature(df, col, squared):
    """Collapse a per-bus-type error column into one value per feature, 
      squared=True  (RMSE): sqrt(sum(n * col^2) / sum(n))   -- pool MSE then sqrt
      squared=False (MAE):       sum(n * col)   / sum(n)"""
    out = {}
    for feat, grp in df.groupby("feature"):
        w, v = grp["n"].to_numpy(), grp[col].to_numpy()
        pooled = (w * v ** 2).sum() / w.sum() if squared else (w * v).sum() / w.sum()
        out[feat] = float(np.sqrt(pooled)) if squared else float(pooled)
    return out

# Slim debug summary = a PURE PROJECTION of the exhaustive metrics file. Each entry
# is a verbatim "Metric" name from metrics_rows; the summary never recomputes.
# To add a column here it must first exist in the exhaustive file.
SUMMARY_METRICS = [
    # Accuracy block in the ML arms' canonical order (Pd/Qd un-prefixed: shared
    # forecaster metrics; everything else is AC-computed -> "AC " prefix).
    "AC Mean Optimality Gap",
    "AC Residual P (MAE)", "AC Residual Q (MAE)",
    "MAE Pd", "RMSE Pd",
    "MAE Qd", "RMSE Qd",
    "AC Vm MAE", "AC Vm RMSE",
    "AC Va MAE", "AC Va RMSE",
    "AC Pg bus MAE", "AC Pg bus RMSE",
    "AC Qg MAE", "AC Qg RMSE",
    "AC Generator Pg MAE", "AC Generator Pg RMSE",
    "AC vm viol. rate", "AC pg viol. rate", "AC qg viol. rate",
    "AC thermal viol. rate", "AC angle viol. rate",
    "AC fully feasible scenario viol. rate",
    "AC Solver total time (s)", "AC Solver time per scenario (s)",
    # DC headline
    "DC Mean Optimality Gap",
    "DC vm viol. rate", "DC pg viol. rate",
    "DC thermal viol. rate", "DC angle viol. rate",
    "DC fully feasible scenario viol. rate",
    "DC Solver total time (s)", "DC Solver time per scenario (s)",
]


def _violation_rows(feas: dict, prefix: str) -> list[dict]:
    """Build the canonical violation quintet rows from a compute_feasibility_metrics
    output dict. ``prefix`` is "" (AC) or "DC "; feasibility keys use ""/"dc_"."""
    fp = "dc_" if prefix == "DC " else ""
    rows = []
    for ct, unit in _CT_UNITS.items():
        if f"{fp}{ct}_viol_max" not in feas:
            continue  # e.g. qg is absent in DC mode
        rows.extend([
            {"Metric": f"{prefix}{ct} viol. count", "Value": feas[f"{fp}{ct}_viol_count"], "Unit": "count"},
            {"Metric": f"{prefix}{ct} viol. mean", "Value": feas[f"{fp}{ct}_viol_mean"], "Unit": unit},
            {"Metric": f"{prefix}{ct} viol. max", "Value": feas[f"{fp}{ct}_viol_max"], "Unit": unit},
            {"Metric": f"{prefix}{ct} viol. rate", "Value": feas[f"{fp}{ct}_viol_rate"], "Unit": "frac"},
            {"Metric": f"{prefix}{ct} scenario viol. rate", "Value": feas[f"{fp}{ct}_scenario_viol_rate"], "Unit": "frac"},
        ])
    rows.append(
        {"Metric": f"{prefix}fully feasible scenario viol. rate",
         "Value": feas[f"{fp}fully_scenario_viol_rate"], "Unit": "frac"}
    )
    return rows


def _build_predicted_scenario_map(forecasts_df: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    """
    Map predicted OPF flattened scenario indices to both:
    1. Ground-truth scenario indices (target_load_scenario_idx)
    2. Horizon steps (if present)
    """
    if "horizon_step" in forecasts_df.columns:
        keys = (
            forecasts_df[["load_scenario_idx", "horizon_step"]]
            .drop_duplicates()
            .copy()
        )
        keys["load_scenario_idx"] = pd.to_numeric(
            keys["load_scenario_idx"],
            errors="raise",
        ).astype(int)
        keys["horizon_step"] = pd.to_numeric(
            keys["horizon_step"],
            errors="raise",
        ).astype(int)

        keys = keys.sort_values(["load_scenario_idx", "horizon_step"], kind="stable").reset_index(drop=True)
        keys["pred_flat_idx"] = keys.index.astype(int)
        keys["target_load_scenario_idx"] = (
            keys["load_scenario_idx"] + keys["horizon_step"]
        ).astype(int)
        
        target_map = dict(zip(keys["pred_flat_idx"].tolist(), keys["target_load_scenario_idx"].tolist()))
        horizon_map = dict(zip(keys["pred_flat_idx"].tolist(), keys["horizon_step"].tolist()))
        return target_map, horizon_map

    forecast_scenarios = sorted(pd.to_numeric(forecasts_df["load_scenario_idx"], errors="raise").astype(int).unique())
    target_map = dict(enumerate(forecast_scenarios))
    return target_map, {}


def compare_single_method(
    method: str,
    forecasts_df: pd.DataFrame,
    ground_truth_dir: Path,
    predicted_opf_dir: Path,
    output_dir: Path,
    dataset: str,
    pred_scenario_map: dict[int, int],
    true_bus: pd.DataFrame,
    true_gen: pd.DataFrame,
    true_branch: pd.DataFrame,
    pred_horizon_map: dict[int, int] = None,
) -> dict:
    """
    Run complete comparison for a single forecast method.
    
    Returns:
        Dict with summary metrics for aggregation.
    """
    print(f"\n{'='*60}")
    print(f"Processing method: {method}")
    print(f"{'='*60}")
    
    method_output_dir = output_dir / method
    method_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load OPF results
    print("Loading OPF results...")
    pred_bus = load_datakit_bus(predicted_opf_dir)
    pred_gen = load_datakit_gen(predicted_opf_dir)
    pred_branch = load_datakit_branch(predicted_opf_dir)
    
    # Preserve original flattened indices for graph contiguity
    pred_bus["pred_flat_idx"] = pred_bus["load_scenario_idx"].copy()
    pred_gen["pred_flat_idx"] = pred_gen["load_scenario_idx"].copy()
    pred_branch["pred_flat_idx"] = pred_branch["load_scenario_idx"].copy()

    # Attach horizon steps if map provided
    if pred_horizon_map:
        pred_bus["horizon_step"] = pred_bus["pred_flat_idx"].map(pred_horizon_map)
        pred_gen["horizon_step"] = pred_gen["pred_flat_idx"].map(pred_horizon_map)
        pred_branch["horizon_step"] = pred_branch["pred_flat_idx"].map(pred_horizon_map)

    # Remap predicted OPF scenario indices (0-based) to match ground-truth indices
    pred_bus["load_scenario_idx"] = pred_bus["load_scenario_idx"].map(pred_scenario_map)
    pred_gen["load_scenario_idx"] = pred_gen["load_scenario_idx"].map(pred_scenario_map)
    pred_branch["load_scenario_idx"] = pred_branch["load_scenario_idx"].map(pred_scenario_map)

    if pred_bus["load_scenario_idx"].isna().any() or pred_gen["load_scenario_idx"].isna().any() or pred_branch["load_scenario_idx"].isna().any():
        raise ValueError(
            "Predicted OPF scenario mapping produced NaN values. "
            "Forecast parquet and OPF output likely use incompatible flattened scenario indexing."
        )

    pred_bus["load_scenario_idx"] = pred_bus["load_scenario_idx"].astype(int)
    pred_gen["load_scenario_idx"] = pred_gen["load_scenario_idx"].astype(int)
    pred_branch["load_scenario_idx"] = pred_branch["load_scenario_idx"].astype(int)

    # 2. Align OPF results
    print("Aligning OPF results...")
    bus_aligned, gen_aligned = align_opf_results(pred_bus, true_bus, pred_gen, true_gen)
    
    # Check if DC columns are present before deleting
    dc_available = has_dc_columns(pred_bus, pred_gen)

    # Free bus/gen prediction memory early
    del pred_bus, pred_gen
    import gc
    gc.collect()

    branch_aligned = align_branch_results(pred_branch, true_branch)
    print(f"  Aligned {len(bus_aligned)} bus observations across {bus_aligned['load_scenario_idx'].nunique()} scenarios")
    print(f"  Aligned {len(gen_aligned)} generator observations")
    print(f"  Aligned {len(branch_aligned)} branch observations")
    
    # Free branch prediction memory
    del pred_branch
    gc.collect()
    
    # 3. Compute RMSE by bus type
    print("Computing RMSE by bus type...")
    rmse_df = compute_rmse_by_bus_type(bus_aligned, ["Vm", "Va", "Pg", "Qg"])
    # Collapse per-bus-type -> per-feature, weighted by bus count 
    rmse_summary = _pool_by_feature(rmse_df, "rmse", squared=True)
    mae_summary = _pool_by_feature(rmse_df, "mae", squared=False)
    # Pd/Qd load forecast error (forecast vs true load, pooled over all buses --
    # non-load buses are Pd=Qd=0 on both sides, so they count in the denominator
    # with zero error, matching how the surrogate/E2E arms compute this metric).
    pd_err = bus_aligned["Pd_pred"] - bus_aligned["Pd_true"]
    mae_pd = float(pd_err.abs().mean())
    rmse_pd = float(np.sqrt((pd_err ** 2).mean()))
    qd_err = bus_aligned["Qd_pred"] - bus_aligned["Qd_true"]
    mae_qd = float(qd_err.abs().mean())
    rmse_qd = float(np.sqrt((qd_err ** 2).mean()))

    
    # 4. Compute generator RMSE
    print("Computing generator-level RMSE...")
    gen_rmse = compute_generator_rmse(gen_aligned)
    mae_pg_gen = float(np.abs(gen_aligned["p_mw_pred"] - gen_aligned["p_mw_true"]).mean())
    
    # 5. Compute cost metrics
    print("Computing cost/optimality gap...")
    cost_metrics = compute_cost_metrics(gen_aligned)
    
    # 5.5 Compute AC Power Residuals
    print("Computing exact AC algebraic power residuals...")
    residuals = compute_algebraic_power_residuals(bus_aligned, true_branch)
    
    # 5.6 Compute AC branch flow RMSE
    print("Computing AC branch flow RMSE...")
    ac_branch_metrics = compute_ac_branch_rmse(branch_aligned)

    # 5.7 Feasibility / constraint-violation statistics (canonical quintet per type)
    print("Computing AC constraint violations...")
    ac_feas = compute_feasibility_metrics(bus_aligned, gen_aligned, branch_aligned, mode="ac")

    # 6. Build the exhaustive metrics rows (AC). This file is the SINGLE SOURCE OF
    #    TRUTH; comparison_summary is a pure projection of these rows (see
    #    generate_comparison_summary). Add a new metric here, never in the summary.
    #    Naming: metrics each optimizer computes SEPARATELY carry an "AC "/"DC "
    #    prefix (aggregate_results strips it -> same metric name, different
    #    optimizer). Only the forecaster-shared load metrics (Pd, Qd) are
    #    un-prefixed, since they are identical across arms.
    #    Order matches the ML arms' canonical_scalar_rows.
    nan = float("nan")
    metrics_rows = [
        {"Metric": "Mean Ground Truth Cost", "Value": cost_metrics["mean_true_cost"], "Unit": "EUR"},
        {"Metric": "AC Mean Optimality Gap", "Value": cost_metrics["mean_optimality_gap_pct"], "Unit": "%"},
        {"Metric": "AC Residual P (MAE)", "Value": residuals["mae_residual_p_mw"], "Unit": "MW"},
        {"Metric": "AC Residual Q (MAE)", "Value": residuals["mae_residual_q_mvar"], "Unit": "MVAr"},
        {"Metric": "MAE Pd", "Value": mae_pd, "Unit": "MW"},
        {"Metric": "RMSE Pd", "Value": rmse_pd, "Unit": "MW"},
        {"Metric": "MAE Qd", "Value": mae_qd, "Unit": "MVAr"},
        {"Metric": "RMSE Qd", "Value": rmse_qd, "Unit": "MVAr"},
        {"Metric": "AC Vm MAE", "Value": mae_summary.get("Vm", nan), "Unit": "p.u."},
        {"Metric": "AC Vm RMSE", "Value": rmse_summary.get("Vm", nan), "Unit": "p.u."},
        {"Metric": "AC Va MAE", "Value": mae_summary.get("Va", nan), "Unit": "deg"},
        {"Metric": "AC Va RMSE", "Value": rmse_summary.get("Va", nan), "Unit": "deg"},
        {"Metric": "AC Pg bus MAE", "Value": mae_summary.get("Pg", nan), "Unit": "MW"},
        {"Metric": "AC Pg bus RMSE", "Value": rmse_summary.get("Pg", nan), "Unit": "MW"},
        {"Metric": "AC Qg MAE", "Value": mae_summary.get("Qg", nan), "Unit": "MVAr"},
        {"Metric": "AC Qg RMSE", "Value": rmse_summary.get("Qg", nan), "Unit": "MVAr"},
        {"Metric": "AC Generator Pg MAE", "Value": mae_pg_gen, "Unit": "MW"},
        {"Metric": "AC Generator Pg RMSE", "Value": gen_rmse, "Unit": "MW"},
        # AC-arm extras with no ML counterpart, kept after the canonical block.
        {"Metric": "AC Median Optimality Gap", "Value": cost_metrics["median_optimality_gap_pct"], "Unit": "%"},
        {"Metric": "AC Max Optimality Gap", "Value": cost_metrics["max_optimality_gap_pct"], "Unit": "%"},
        # Branch flow RMSE kept in the exhaustive file but NOT projected to the
        # summary (redundant accuracy metric — flow error is captured on the
        # feasibility side via thermal violations).
        {"Metric": "AC Branch pf RMSE", "Value": ac_branch_metrics["rmse_pf"], "Unit": "MW"},
        {"Metric": "AC Branch pt RMSE", "Value": ac_branch_metrics["rmse_pt"], "Unit": "MW"},
    ]
    metrics_rows.extend(_violation_rows(ac_feas, "AC "))

    # Return summary for aggregation (main() also reads horizon_gaps for the
    # forecast table). Summary column values are pulled from metrics_rows below.
    res = {"method": method}
    if "horizon_gaps" in cost_metrics:
        res["horizon_gaps"] = cost_metrics["horizon_gaps"]
        res["mean_optimality_gap_pct"] = cost_metrics["mean_optimality_gap_pct"]

    # -----------------------------------------------------------------------
    # 7. DC-OPF analysis (if DC columns are present in predicted data)
    # -----------------------------------------------------------------------
    if dc_available:
        print("\n  --- DC-OPF Analysis (DC predictions vs AC ground truth) ---")

        # 8a. DC Bus RMSE (Va_dc, Pg_dc, Vm_dc=1.0 vs AC truth)
        print("  Computing DC RMSE by bus type...")
        dc_rmse_df = compute_dc_rmse_by_bus_type(bus_aligned)
        # Append DC rows to the same RMSE table as AC
        rmse_df = pd.concat([rmse_df, dc_rmse_df], ignore_index=True)
        dc_rmse_summary = _pool_by_feature(dc_rmse_df, "rmse", squared=True)
        dc_mae_summary = _pool_by_feature(dc_rmse_df, "mae", squared=False)

        # 8b. DC Generator RMSE
        print("  Computing DC generator-level RMSE...")
        dc_gen_rmse = compute_dc_generator_rmse(gen_aligned)
        dc_mae_pg_gen = float(np.abs(gen_aligned["p_mw_dc"] - gen_aligned["p_mw_true"]).mean())

        # 8c. DC Cost / Optimality Gap
        print("  Computing DC cost/optimality gap...")
        dc_cost_metrics = compute_dc_cost_metrics(gen_aligned)

        # 8d. DC Branch Flow RMSE
        print("  Computing DC branch flow RMSE...")
        dc_branch_metrics = compute_dc_branch_rmse(branch_aligned)

        # 8e. DC Algebraic Residuals (using AC equations)
        print("  Computing DC exact algebraic residuals (AC equations)...")
        dc_residuals = compute_dc_algebraic_power_residuals(bus_aligned, true_branch)

        # 8f. DC Constraint Violations (canonical quintet per type)
        print("  Computing DC constraint violations...")
        dc_feas = compute_feasibility_metrics(bus_aligned, gen_aligned, branch_aligned, mode="dc")

        # Append DC metric rows in the same  order as the AC/ML arms
        # (DC has no reactive quantities, so no Qd/Qg rows; Vm is a deviation-
        # from-flat-1.0 metric, kept under its own name).
        metrics_rows.extend([
            {"Metric": "DC Mean Optimality Gap", "Value": dc_cost_metrics["dc_mean_optimality_gap_pct"], "Unit": "%"},
            {"Metric": "DC Residual P (MAE)", "Value": dc_residuals["dc_mae_residual_p_mw"], "Unit": "MW"},
            {"Metric": "DC Residual Q (MAE)", "Value": dc_residuals["dc_mae_residual_q_mvar"], "Unit": "MVAr"},
            {"Metric": "DC Vm Deviation MAE (1.0 vs AC)", "Value": dc_mae_summary.get("Vm_dc", nan), "Unit": "p.u."},
            {"Metric": "DC Vm Deviation RMSE (1.0 vs AC)", "Value": dc_rmse_summary.get("Vm_dc", nan), "Unit": "p.u."},
            {"Metric": "DC Va MAE", "Value": dc_mae_summary.get("Va_dc", nan), "Unit": "deg"},
            {"Metric": "DC Va RMSE", "Value": dc_rmse_summary.get("Va_dc", nan), "Unit": "deg"},
            {"Metric": "DC Pg bus MAE", "Value": dc_mae_summary.get("dc_Pg_bus", nan), "Unit": "MW"},
            {"Metric": "DC Pg bus RMSE", "Value": dc_rmse_summary.get("dc_Pg_bus", nan), "Unit": "MW"},
            {"Metric": "DC Generator Pg MAE", "Value": dc_mae_pg_gen, "Unit": "MW"},
            {"Metric": "DC Generator Pg RMSE", "Value": dc_gen_rmse, "Unit": "MW"},
            {"Metric": "DC Median Optimality Gap", "Value": dc_cost_metrics["dc_median_optimality_gap_pct"], "Unit": "%"},
            {"Metric": "DC Max Optimality Gap", "Value": dc_cost_metrics["dc_max_optimality_gap_pct"], "Unit": "%"},
            {"Metric": "DC Branch pf RMSE", "Value": dc_branch_metrics["dc_rmse_pf"], "Unit": "MW"},
            {"Metric": "DC Branch pt RMSE", "Value": dc_branch_metrics["dc_rmse_pt"], "Unit": "MW"},
        ])
        metrics_rows.extend(_violation_rows(dc_feas, "DC "))

        if "dc_horizon_gaps" in dc_cost_metrics:
            res["dc_horizon_gaps"] = dc_cost_metrics["dc_horizon_gaps"]
        res["dc_mean_optimality_gap_pct"] = dc_cost_metrics["dc_mean_optimality_gap_pct"]

        print(f"  DC Vm deviation RMSE: {dc_rmse_summary.get('Vm_dc', float('nan')):.4f} p.u.")
        print(f"  DC Va RMSE: {dc_rmse_summary.get('Va_dc', float('nan')):.4f} deg")
        print(f"  DC Gen Pg RMSE: {dc_gen_rmse:.4f} MW")
        print(f"  DC Mean Opt. Gap: {dc_cost_metrics['dc_mean_optimality_gap_pct']:.4f}%")
        print(f"  DC Branch pf RMSE: {dc_branch_metrics['dc_rmse_pf']:.4f} MW")
        print(f"  DC Residual P MAE: {dc_residuals['dc_mae_residual_p_mw']:.4f} MW")
        print(f"  DC Residual Q MAE: {dc_residuals['dc_mae_residual_q_mvar']:.4f} MVar")
    else:
        print("\n  DC columns not found in predicted data — skipping DC analysis.")

    # 8. Save RMSE table (AC + DC combined)
    rmse_path = method_output_dir / OUTPUT_TEMPLATES["rmse"].format(dataset=dataset)
    rmse_df.to_csv(rmse_path, index=False)
    print(f"  Saved RMSE table: {rmse_path}")

    # 8.5 Solver timing — per-scenario AC/DC solve_time from DataKit's runtime_data.parquet.
    #   - columns: "ac" (always), "dc" (present when include_dc_res); values in seconds.
    #   - AC/DC kept separate, following the "AC "/"DC " metric-naming convention above.
    runtime_path = predicted_opf_dir / "runtime_data.parquet"
    if runtime_path.exists():
        _rt = pd.read_parquet(runtime_path)
        metrics_rows.extend([
            {"Metric": "AC Solver total time (s)", "Value": float(_rt["ac"].sum()), "Unit": "s"},
            {"Metric": "AC Solver time per scenario (s)", "Value": float(_rt["ac"].mean()), "Unit": "s/scenario"},
        ])
        if "dc" in _rt.columns and _rt["dc"].notna().any():
            metrics_rows.extend([
                {"Metric": "DC Solver total time (s)", "Value": float(_rt["dc"].sum()), "Unit": "s"},
                {"Metric": "DC Solver time per scenario (s)", "Value": float(_rt["dc"].mean()), "Unit": "s/scenario"},
            ])
        metrics_rows.append(
            {"Metric": "Solver n_scenarios", "Value": int(len(_rt)), "Unit": "count"},
        )

    # 9. Save metrics summary (AC + DC combined) — the SINGLE SOURCE OF TRUTH.
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = method_output_dir / OUTPUT_TEMPLATES["metrics"].format(dataset=dataset)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Saved metrics: {metrics_path}")

    # 10. Project the slim summary row from the exhaustive rows (no recomputation).
    name2val = {r["Metric"]: r["Value"] for r in metrics_rows}
    for m in SUMMARY_METRICS:
        res[m] = name2val.get(m, float("nan"))

    return res


def generate_comparison_summary(summaries: list, output_dir: Path, dataset: str):
    """Slim debug summary — a PURE PROJECTION of each method's exhaustive metrics.

    Columns are ``method`` + the ``SUMMARY_METRICS`` names (verbatim from the
    exhaustive file). No recomputation, no arithmetic, no rounding: the values
    are exactly those in each method's <net>_metrics.csv, so the two files can
    never disagree.
    """
    summary_df = pd.DataFrame(summaries)

    # Keep only method + the projected summary metrics that any method produced.
    cols = ["method"] + [m for m in SUMMARY_METRICS if m in summary_df.columns]
    summary_df = summary_df.reindex(columns=cols)

    summary_path = output_dir / OUTPUT_TEMPLATES["summary"].format(dataset=dataset)
    summary_df.to_csv(summary_path, index=False, sep="\t")

    print(f"\n{'='*60}")
    print("Comparison Summary (projection of exhaustive metrics)")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    print(f"\nSaved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare two-step OPF approach vs ground truth")
    parser.add_argument(
        "--ground-truth-dir", 
        type=Path, 
        default=Path("data/data_out/3yr_2019-2021/case14_ieee/raw"),
        help="Path to ground-truth OPF parquet directory"
    )
    parser.add_argument(
        "--predicted-opf-base-dir", 
        type=Path, 
        default=Path("exp1/data/data_out"),
        help="Base directory containing {method}/case14_ieee/raw subdirs"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path("exp1/results"),
        help="Output directory for comparison results"
    )
    parser.add_argument("--dataset", type=str, default="case14_ieee",
                        help="Dataset name for output file naming")
    parser.add_argument("--methods", nargs="+", default=FORECAST_METHODS,
                        help="Forecast methods to compare (default: all)")
    parser.add_argument(
        "--forecasts-parquet",
        type=Path,
        default=FORECASTS_PARQUET,
        help="Path to forecasts parquet file",
    )
    parser.add_argument(
        "--forecast-seasonality",
        type=int,
        default=FORECAST_SEASONALITY,
        help="Seasonality used for MASE/MSSE scaling",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.ground_truth_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {args.ground_truth_dir}")
    
    # Load forecasts once (shared across all methods)
    print("Loading forecasts...")
    forecasts_df = load_forecasts(args.forecasts_parquet)
    print(f"Loaded {len(forecasts_df)} forecast observations for {forecasts_df['load_scenario_idx'].nunique()} scenarios")

    available_methods = [method for method in args.methods if method in forecasts_df.columns]
    missing_methods = [method for method in args.methods if method not in forecasts_df.columns]
    if missing_methods:
        print(f"Skipping unavailable forecast methods in parquet: {missing_methods}")
    if not available_methods:
        raise ValueError("None of the requested forecast methods are present in forecasts parquet")

    # Update scenario and horizon mapping
    pred_scenario_map, pred_horizon_map = _build_predicted_scenario_map(forecasts_df)
    target_scenarios = list(pred_scenario_map.values())

    # 0. Load ground truth branch data (static topology), bus and generator data once, filtered to target scenarios
    print("Loading ground truth datasets once...")
    true_branch = load_datakit_branch(args.ground_truth_dir)
    true_branch = true_branch[true_branch["load_scenario_idx"].isin(target_scenarios)].copy()
    
    if not true_branch.empty:
        first_scen = true_branch["load_scenario_idx"].iloc[0]
        static_branch = true_branch[true_branch["load_scenario_idx"] == first_scen]
        unlimited_mask = (static_branch["rate_a"] <= 0.0) | (static_branch["rate_a"].isna())
        num_unlimited = unlimited_mask.sum()
        print(f"Ground Truth static topology: {len(static_branch)} total branches, {num_unlimited} branches without capacity limits.")
        if num_unlimited > 0:
            unlimited_idxs = static_branch.loc[unlimited_mask, "idx"].tolist()
            print(f"  Line indices without capacity limits: {unlimited_idxs}")

    
    true_bus = load_datakit_bus(args.ground_truth_dir)
    true_bus = true_bus[true_bus["load_scenario_idx"].isin(target_scenarios)].copy()
    
    true_gen = load_datakit_gen(args.ground_truth_dir)
    true_gen = true_gen[true_gen["load_scenario_idx"].isin(target_scenarios)].copy()

    # Prepare combined forecast metrics table (all selected methods)
    # We will finalize and save it AFTER metrics are collected for the OPF gap.
    forecast_table = compute_forecast_metrics_table(
        forecasts_df=forecasts_df,
        methods=available_methods,
        seasonality=args.forecast_seasonality,
    )
    
    # Process each method
    summaries = []
    horizon_gap_column = "Opt. Gap (%)"

    for method in available_methods:
        predicted_opf_dir = args.predicted_opf_base_dir / method / args.dataset / "raw"
        
        if not predicted_opf_dir.exists():
            print(f"\n  Skipping {method}: OPF results not found at {predicted_opf_dir}")
            continue
        
        try:
            summary = compare_single_method(
                method=method,
                forecasts_df=forecasts_df,
                ground_truth_dir=args.ground_truth_dir,
                predicted_opf_dir=predicted_opf_dir,
                output_dir=args.output_dir,
                dataset=args.dataset,
                pred_scenario_map=pred_scenario_map,
                true_bus=true_bus,
                true_gen=true_gen,
                true_branch=true_branch,
                pred_horizon_map=pred_horizon_map,
            )
            summaries.append(summary)

            # If optimality gap is present, update the forecast_table
            if "horizon_gaps" in summary:
                # Add overall global gap
                forecast_table.loc[
                    (forecast_table["Model"] == method) & (forecast_table["Horizon"] == "GLOBAL"), 
                    horizon_gap_column
                ] = summary["mean_optimality_gap_pct"]

                # Add per-horizon gaps
                min_h = min(pred_horizon_map.values()) if pred_horizon_map else 0
                for h_step, gap_val in summary["horizon_gaps"].items():
                    h_label = f"t+{h_step + 1}" if min_h == 0 else f"t+{h_step}"
                    forecast_table.loc[
                        (forecast_table["Model"] == method) & (forecast_table["Horizon"] == h_label), 
                        horizon_gap_column
                    ] = gap_val

            # DC optimality gap — append to the same forecast table
            dc_gap_column = "DC Opt. Gap (%)"
            if "dc_horizon_gaps" in summary:
                forecast_table.loc[
                    (forecast_table["Model"] == method) & (forecast_table["Horizon"] == "GLOBAL"),
                    dc_gap_column
                ] = summary["dc_mean_optimality_gap_pct"]

                min_h = min(pred_horizon_map.values()) if pred_horizon_map else 0
                for h_step, gap_val in summary["dc_horizon_gaps"].items():
                    h_label = f"t+{h_step + 1}" if min_h == 0 else f"t+{h_step}"
                    forecast_table.loc[
                        (forecast_table["Model"] == method) & (forecast_table["Horizon"] == h_label),
                        dc_gap_column
                    ] = gap_val
            elif "dc_mean_optimality_gap_pct" in summary:
                # Single-step horizon: only global DC gap
                forecast_table.loc[
                    (forecast_table["Model"] == method) & (forecast_table["Horizon"] == "GLOBAL"),
                    dc_gap_column
                ] = summary["dc_mean_optimality_gap_pct"]

        except Exception as e:
            print(f"\nERROR processing {method}: {e}")
            raise
    
    # Save the updated forecast table
    forecast_table_path = args.output_dir / OUTPUT_TEMPLATES["forecast"].format(dataset=args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    forecast_table.to_csv(forecast_table_path, index=False)
    print(f"Saved forecast metrics table with optimality gaps: {forecast_table_path}")

    # Generate comparison summary
    if summaries:
        generate_comparison_summary(summaries, args.output_dir, args.dataset)
    else:
        print("\nERROR No methods successfully processed.")


if __name__ == "__main__":
    main()
