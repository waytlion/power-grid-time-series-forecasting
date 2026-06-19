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
    prepare_load_forecast_comparison,
    align_opf_results,
    align_branch_results,
    has_dc_columns,
)
from metrics import (
    compute_mae,
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
    compute_violations,
)


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
    
    # 1. Load forecast MAE (Pd only, since forecasts.parquet has active power only)
    print("Computing load forecast MAE...")
    forecast_comparison = prepare_load_forecast_comparison(forecasts_df, method)
    # Rename for compute_mae which expects {feature}_pred and {feature}_true
    forecast_for_mae = forecast_comparison.rename(columns={"pred": "Pd_pred", "true": "Pd_true"})
    mae_pd = compute_mae(forecast_for_mae, ["Pd"])["Pd"]
    rmse_pd = float(np.sqrt(((forecast_for_mae["Pd_pred"] - forecast_for_mae["Pd_true"]) ** 2).mean()))
    
    # Save forecast MAE
    forecast_mae_df = pd.DataFrame([{
        "Feature": "Pd",
        "MAE": mae_pd,
        "Unit": "MW",
    }])
    forecast_mae_path = method_output_dir / OUTPUT_TEMPLATES["forecast_mae"].format(dataset=dataset)
    forecast_mae_df.to_csv(forecast_mae_path, index=False)
    print(f"  MAE Pd: {mae_pd:.4f} MW")
    
    # 2. Load OPF results
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

    # 3. Align OPF results
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
    
    # 4. Compute RMSE by bus type
    print("Computing RMSE by bus type...")
    rmse_df = compute_rmse_by_bus_type(bus_aligned, ["Vm", "Va", "Pg", "Qg"])
    # Extract key RMSE values for summary (aggregate across bus types)
    rmse_summary = rmse_df.groupby("feature")["rmse"].mean().to_dict()
    
    
    # 5. Compute generator RMSE
    print("Computing generator-level RMSE...")
    gen_rmse = compute_generator_rmse(gen_aligned)
    mae_pg_gen = float(np.abs(gen_aligned["p_mw_pred"] - gen_aligned["p_mw_true"]).mean())
    
    # 6. Compute cost metrics
    print("Computing cost/optimality gap...")
    cost_metrics = compute_cost_metrics(gen_aligned)
    
    # 6.5 Compute AC Power Residuals
    print("Computing exact AC algebraic power residuals...")
    residuals = compute_algebraic_power_residuals(bus_aligned, true_branch)
    
    # 6.6 Compute AC branch flow RMSE
    print("Computing AC branch flow RMSE...")
    ac_branch_metrics = compute_ac_branch_rmse(branch_aligned)

    # 6.7 Compute AC constraint violations
    print("Computing AC constraint violations...")
    ac_viols = compute_violations(bus_aligned, gen_aligned, branch_aligned, mode="ac")

    # To compute these exactly for the AC prediction:
    rate_a_vals = branch_aligned["rate_a_true"].values
    valid_rate_vals = (rate_a_vals > 0.0) & (~np.isnan(rate_a_vals))
    
    pf_pred_vals = branch_aligned["pf_pred"].values
    qf_pred_vals = branch_aligned["qf_pred"].values
    pt_pred_vals = branch_aligned["pt_pred"].values
    qt_pred_vals = branch_aligned["qt_pred"].values
    
    s_from_vals = np.sqrt(pf_pred_vals**2 + qf_pred_vals**2)
    s_to_vals = np.sqrt(pt_pred_vals**2 + qt_pred_vals**2)
    
    viol_from_vals = np.zeros_like(s_from_vals)
    viol_to_vals = np.zeros_like(s_to_vals)
    
    viol_from_vals[valid_rate_vals] = np.maximum(0.0, s_from_vals[valid_rate_vals] - rate_a_vals[valid_rate_vals])
    viol_to_vals[valid_rate_vals] = np.maximum(0.0, s_to_vals[valid_rate_vals] - rate_a_vals[valid_rate_vals])
    
    mean_branch_thermal_violation_from = float(viol_from_vals[valid_rate_vals].mean()) if valid_rate_vals.any() else 0.0
    mean_branch_thermal_violation_to = float(viol_to_vals[valid_rate_vals].mean()) if valid_rate_vals.any() else 0.0
    
    # For angle difference:
    from_col_name = "from_bus_true" if "from_bus_true" in branch_aligned.columns else "from_bus"
    to_col_name = "to_bus_true" if "to_bus_true" in branch_aligned.columns else "to_bus"
    
    # Determine the unique scenario alignment column
    align_col = "pred_flat_idx" if "pred_flat_idx" in branch_aligned.columns and "pred_flat_idx" in bus_aligned.columns else "load_scenario_idx"

    # Merge va_col chunk by chunk to keep peak memory low
    unique_scenarios = sorted(branch_aligned[align_col].unique())
    chunk_size = 1000
    branch_merged_chunks = []
    for j in range(0, len(unique_scenarios), chunk_size):
        chunk_sc = unique_scenarios[j : j + chunk_size]
        chunk_branch = branch_aligned[branch_aligned[align_col].isin(chunk_sc)]
        chunk_bus = bus_aligned[bus_aligned[align_col].isin(chunk_sc)]
        
        chunk_merged = chunk_branch[[align_col, "load_scenario_idx", from_col_name, to_col_name, "ang_min_true", "ang_max_true"]].merge(
            chunk_bus[[align_col, "bus", "Va_pred"]],
            left_on=[align_col, from_col_name],
            right_on=[align_col, "bus"],
            how="left"
        ).rename(columns={"Va_pred": "va_from"})
        
        chunk_merged = chunk_merged.merge(
            chunk_bus[[align_col, "bus", "Va_pred"]],
            left_on=[align_col, to_col_name],
            right_on=[align_col, "bus"],
            how="left"
        ).rename(columns={"Va_pred": "va_to"})
        
        branch_merged_chunks.append(chunk_merged)
        
    branch_merged_vals = pd.concat(branch_merged_chunks, ignore_index=True)
    
    ang_diff_vals = branch_merged_vals["va_from"].values - branch_merged_vals["va_to"].values
    
    ang_min_vals = branch_aligned["ang_min_true"].values
    ang_max_vals = branch_aligned["ang_max_true"].values
    
    valid_ang_vals = (~np.isnan(ang_min_vals)) & (~np.isnan(ang_max_vals)) & (ang_min_vals > -359.0) & (ang_max_vals < 359.0)
    
    ang_viols_vals = np.zeros_like(ang_diff_vals)
    under_ang_vals = np.maximum(0.0, ang_min_vals - ang_diff_vals)
    over_ang_vals = np.maximum(0.0, ang_diff_vals - ang_max_vals)
    ang_viols_vals[valid_ang_vals] = under_ang_vals[valid_ang_vals] + over_ang_vals[valid_ang_vals]
    
    mean_branch_angle_difference_violation_deg = float(ang_viols_vals[valid_ang_vals].mean()) if valid_ang_vals.any() else 0.0
    
    # 7. Build metrics rows (AC)
    metrics_rows = [{
        "Metric": "Generator Pg RMSE",
        "Value": gen_rmse,
        "Unit": "MW",
    }, {
        "Metric": "Mean Optimality Gap",
        "Value": cost_metrics["mean_optimality_gap_pct"],
        "Unit": "%",
    }, {
        "Metric": "Median Optimality Gap",
        "Value": cost_metrics["median_optimality_gap_pct"],
        "Unit": "%",
    }, {
        "Metric": "Max Optimality Gap",
        "Value": cost_metrics["max_optimality_gap_pct"],
        "Unit": "%",
    }, {
        "Metric": "Residual P (MAE)",
        "Value": residuals["mae_residual_p_mw"],
        "Unit": "MW",
    }, {
        "Metric": "Residual Q (MAE)",
        "Value": residuals["mae_residual_q_mvar"],
        "Unit": "MVA",
    }, {
        "Metric": "AC Branch pf RMSE",
        "Value": ac_branch_metrics["rmse_pf"],
        "Unit": "MW",
    }, {
        "Metric": "AC Branch pt RMSE",
        "Value": ac_branch_metrics["rmse_pt"],
        "Unit": "MW",
    }, {
        "Metric": "AC Vm Violations Count",
        "Value": ac_viols["vm"]["count"],
        "Unit": "count",
    }, {
        "Metric": "AC Vm Violations Mean",
        "Value": ac_viols["vm"]["mean_magnitude"],
        "Unit": "p.u.",
    }, {
        "Metric": "AC Pg Violations Count",
        "Value": ac_viols["pg"]["count"],
        "Unit": "count",
    }, {
        "Metric": "AC Pg Violations Mean",
        "Value": ac_viols["pg"]["mean_magnitude"],
        "Unit": "MW",
    }, {
        "Metric": "AC Thermal Violations Count",
        "Value": ac_viols["thermal"]["count"],
        "Unit": "count",
    }, {
        "Metric": "AC Thermal Violations Mean",
        "Value": ac_viols["thermal"]["mean_magnitude"],
        "Unit": "MVA",
    }, {
        "Metric": "AC Angle Violations Count",
        "Value": ac_viols["angle"]["count"],
        "Unit": "count",
    }, {
        "Metric": "AC Angle Violations Mean",
        "Value": ac_viols["angle"]["mean_magnitude"],
        "Unit": "deg",
    }]
    
    # Return summary for aggregation
    res = {
        "method": method,
        "mae_pd": mae_pd,
        "rmse_pd": rmse_pd,
        "ac_rmse_vm": rmse_summary.get("Vm", float("nan")),
        "ac_rmse_va": rmse_summary.get("Va", float("nan")),
        "ac_rmse_pg_bus": rmse_summary.get("Pg", float("nan")),
        "mae_pg": mae_pg_gen,
        "rmse_pg_gen": gen_rmse,  # Generator-level
        "ac_rmse_pf_branch": ac_branch_metrics["rmse_pf"],
        "ac_rmse_pt_branch": ac_branch_metrics["rmse_pt"],
        "avg_active_res_mw": residuals["mae_residual_p_mw"],
        "avg_reactive_res_mvar": residuals["mae_residual_q_mvar"],
        "mean_optimality_gap_pct": cost_metrics["mean_optimality_gap_pct"],
        "mean_branch_thermal_violation_from_mva": mean_branch_thermal_violation_from,
        "mean_branch_thermal_violation_to_mva": mean_branch_thermal_violation_to,
        "mean_branch_angle_difference_violation_rad": mean_branch_angle_difference_violation_deg,
        "ac_violations_vm_count": ac_viols["vm"]["count"],
        "ac_violations_pg_count": ac_viols["pg"]["count"],
        "ac_violations_thermal_count": ac_viols["thermal"]["count"],
        "ac_violations_angle_count": ac_viols["angle"]["count"],
    }
    if "horizon_gaps" in cost_metrics:
        res["horizon_gaps"] = cost_metrics["horizon_gaps"]

    # -----------------------------------------------------------------------
    # 8. DC-OPF analysis (if DC columns are present in predicted data)
    # -----------------------------------------------------------------------
    if dc_available:
        print("\n  --- DC-OPF Analysis (DC predictions vs AC ground truth) ---")

        # 8a. DC Bus RMSE (Va_dc, Pg_dc, Vm_dc=1.0 vs AC truth)
        print("  Computing DC RMSE by bus type...")
        dc_rmse_df = compute_dc_rmse_by_bus_type(bus_aligned)
        # Append DC rows to the same RMSE table as AC
        rmse_df = pd.concat([rmse_df, dc_rmse_df], ignore_index=True)
        dc_rmse_summary = dc_rmse_df.groupby("feature")["rmse"].mean().to_dict()

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

        # 8f. DC Constraint Violations
        print("  Computing DC constraint violations...")
        dc_viols = compute_violations(bus_aligned, gen_aligned, branch_aligned, mode="dc")

        # Append DC metric rows
        metrics_rows.extend([{
            "Metric": "DC Vm Deviation RMSE (1.0 vs AC)",
            "Value": dc_rmse_summary.get("Vm_dc", float("nan")),
            "Unit": "p.u.",
        }, {
            "Metric": "DC Va RMSE",
            "Value": dc_rmse_summary.get("Va_dc", float("nan")),
            "Unit": "deg",
        }, {
            "Metric": "DC Pg Bus RMSE",
            "Value": dc_rmse_summary.get("dc_Pg_bus", float("nan")),
            "Unit": "MW",
        }, {
            "Metric": "DC Generator Pg RMSE",
            "Value": dc_gen_rmse,
            "Unit": "MW",
        }, {
            "Metric": "DC Mean Optimality Gap",
            "Value": dc_cost_metrics["dc_mean_optimality_gap_pct"],
            "Unit": "%",
        }, {
            "Metric": "DC Median Optimality Gap",
            "Value": dc_cost_metrics["dc_median_optimality_gap_pct"],
            "Unit": "%",
        }, {
            "Metric": "DC Max Optimality Gap",
            "Value": dc_cost_metrics["dc_max_optimality_gap_pct"],
            "Unit": "%",
        }, {
            "Metric": "DC Branch pf RMSE",
            "Value": dc_branch_metrics["dc_rmse_pf"],
            "Unit": "MW",
        }, {
            "Metric": "DC Branch pt RMSE",
            "Value": dc_branch_metrics["dc_rmse_pt"],
            "Unit": "MW",
        }, {
            "Metric": "DC Residual P (MAE)",
            "Value": dc_residuals["dc_mae_residual_p_mw"],
            "Unit": "MW",
        }, {
            "Metric": "DC Residual Q (MAE)",
            "Value": dc_residuals["dc_mae_residual_q_mvar"],
            "Unit": "MVA",
        }, {
            "Metric": "DC Vm Violations Count",
            "Value": dc_viols["vm"]["count"],
            "Unit": "count",
        }, {
            "Metric": "DC Vm Violations Mean",
            "Value": dc_viols["vm"]["mean_magnitude"],
            "Unit": "p.u.",
        }, {
            "Metric": "DC Pg Violations Count",
            "Value": dc_viols["pg"]["count"],
            "Unit": "count",
        }, {
            "Metric": "DC Pg Violations Mean",
            "Value": dc_viols["pg"]["mean_magnitude"],
            "Unit": "MW",
        }, {
            "Metric": "DC Thermal Violations Count",
            "Value": dc_viols["thermal"]["count"],
            "Unit": "count",
        }, {
            "Metric": "DC Thermal Violations Mean",
            "Value": dc_viols["thermal"]["mean_magnitude"],
            "Unit": "MVA",
        }, {
            "Metric": "DC Angle Violations Count",
            "Value": dc_viols["angle"]["count"],
            "Unit": "count",
        }, {
            "Metric": "DC Angle Violations Mean",
            "Value": dc_viols["angle"]["mean_magnitude"],
            "Unit": "deg",
        }])

        # Extend summary dict with DC metrics
        res.update({
            "dc_rmse_vm": dc_rmse_summary.get("Vm_dc", float("nan")),
            "dc_rmse_va": dc_rmse_summary.get("Va_dc", float("nan")),
            "dc_rmse_pg_bus": dc_rmse_summary.get("dc_Pg_bus", float("nan")),
            "dc_mae_pg_gen": dc_mae_pg_gen,
            "dc_rmse_pg_gen": dc_gen_rmse,
            "dc_mean_optimality_gap_pct": dc_cost_metrics["dc_mean_optimality_gap_pct"],
            "dc_rmse_pf_branch": dc_branch_metrics["dc_rmse_pf"],
            "dc_rmse_pt_branch": dc_branch_metrics["dc_rmse_pt"],
            "dc_mae_residual_p_mw": dc_residuals["dc_mae_residual_p_mw"],
            "dc_mae_residual_q_mvar": dc_residuals["dc_mae_residual_q_mvar"],
            "dc_violations_vm_count": dc_viols["vm"]["count"],
            "dc_violations_pg_count": dc_viols["pg"]["count"],
            "dc_violations_thermal_count": dc_viols["thermal"]["count"],
            "dc_violations_angle_count": dc_viols["angle"]["count"],
        })
        if "dc_horizon_gaps" in dc_cost_metrics:
            res["dc_horizon_gaps"] = dc_cost_metrics["dc_horizon_gaps"]

        print(f"  DC Vm deviation RMSE: {dc_rmse_summary.get('Vm_dc', float('nan')):.4f} p.u.")
        print(f"  DC Va RMSE: {dc_rmse_summary.get('Va_dc', float('nan')):.4f} deg")
        print(f"  DC Gen Pg RMSE: {dc_gen_rmse:.4f} MW")
        print(f"  DC Mean Opt. Gap: {dc_cost_metrics['dc_mean_optimality_gap_pct']:.4f}%")
        print(f"  DC Branch pf RMSE: {dc_branch_metrics['dc_rmse_pf']:.4f} MW")
        print(f"  DC Residual P MAE: {dc_residuals['dc_mae_residual_p_mw']:.4f} MW")
        print(f"  DC Residual Q MAE: {dc_residuals['dc_mae_residual_q_mvar']:.4f} MVar")
    else:
        print("\n  DC columns not found in predicted data — skipping DC analysis.")

    # 9. Save RMSE table (AC + DC combined)
    rmse_path = method_output_dir / OUTPUT_TEMPLATES["rmse"].format(dataset=dataset)
    rmse_df.to_csv(rmse_path, index=False)
    print(f"  Saved RMSE table: {rmse_path}")

    # 10. Save metrics summary (AC + DC combined)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = method_output_dir / OUTPUT_TEMPLATES["metrics"].format(dataset=dataset)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Saved metrics: {metrics_path}")

    return res


def generate_comparison_summary(summaries: list, output_dir: Path, dataset: str):
    """Generate side-by-side comparison table across all methods (AC + DC)."""
    summary_df = pd.DataFrame(summaries)

    # Define column order: AC columns first, then DC columns
    ac_columns = [
        "method",
        "mae_pd",
        "rmse_pd",
        "ac_rmse_vm",
        "ac_rmse_va",
        "ac_rmse_pg_bus",
        "mae_pg",
        "rmse_pg_gen",
        "ac_rmse_pf_branch",
        "ac_rmse_pt_branch",
        "avg_active_res_mw",
        "avg_reactive_res_mvar",
        "mean_optimality_gap_pct",
        "mean_branch_thermal_violation_from_mva",
        "mean_branch_thermal_violation_to_mva",
        "mean_branch_angle_difference_violation_rad",
        "ac_violations_vm_count",
        "ac_violations_pg_count",
        "ac_violations_thermal_count",
        "ac_violations_angle_count",
    ]
    dc_columns = [
        "dc_rmse_vm",
        "dc_rmse_va",
        "dc_rmse_pg_bus",
        "dc_mae_pg_gen",
        "dc_rmse_pg_gen",
        "dc_rmse_pf_branch",
        "dc_rmse_pt_branch",
        "dc_mean_optimality_gap_pct",
        "dc_mae_residual_p_mw",
        "dc_mae_residual_q_mvar",
        "dc_violations_vm_count",
        "dc_violations_pg_count",
        "dc_violations_thermal_count",
        "dc_violations_angle_count",
    ]
    # Only include DC columns if any method produced them
    available_dc_cols = [c for c in dc_columns if c in summary_df.columns]
    all_columns = ac_columns + available_dc_cols
    summary_df = summary_df.reindex(columns=all_columns)

    rename_map = {
        "ac_rmse_vm": "AC Vm RMSE (p.u.)",
        "ac_rmse_va": "AC Va RMSE (deg)",
        "ac_rmse_pg_bus": "AC Pg bus RMSE (MW)",
        "ac_rmse_pf_branch": "AC pf branch RMSE (MW)",
        "ac_rmse_pt_branch": "AC pt branch RMSE (MW)",
        "avg_active_res_mw": "AC Avg. active res. (MW)",
        "avg_reactive_res_mvar": "AC Avg. reactive res. (MVar)",
        "mean_optimality_gap_pct": "AC Mean opt. gap (%)",
        "mean_branch_thermal_violation_from_mva": "AC Mean branch thermal viol. from (MVA)",
        "mean_branch_thermal_violation_to_mva": "AC Mean branch thermal viol. to (MVA)",
        "mean_branch_angle_difference_violation_rad": "AC Mean branch angle diff. viol. (deg)",
        "ac_violations_vm_count": "AC Vm viol. count",
        "ac_violations_pg_count": "AC Pg viol. count",
        "ac_violations_thermal_count": "AC Thermal viol. count",
        "ac_violations_angle_count": "AC Angle viol. count",
        "dc_rmse_vm": "DC Vm dev. RMSE (p.u.)",
        "dc_rmse_va": "DC Va RMSE (deg)",
        "dc_rmse_pg_bus": "DC Pg bus RMSE (MW)",
        "dc_mae_pg_gen": "DC Pg gen MAE (MW)",
        "dc_rmse_pg_gen": "DC Pg gen RMSE (MW)",
        "dc_mean_optimality_gap_pct": "DC Mean opt. gap (%)",
        "dc_rmse_pf_branch": "DC pf branch RMSE (MW)",
        "dc_rmse_pt_branch": "DC pt branch RMSE (MW)",
        "dc_mae_residual_p_mw": "DC Avg. active res. (MW)",
        "dc_mae_residual_q_mvar": "DC Avg. reactive res. (MVar)",
        "dc_violations_vm_count": "DC Vm viol. count",
        "dc_violations_pg_count": "DC Pg viol. count",
        "dc_violations_thermal_count": "DC Thermal viol. count",
        "dc_violations_angle_count": "DC Angle viol. count",
    }
    summary_df = summary_df.rename(columns=rename_map)

    numeric_cols = summary_df.select_dtypes(include=["number"]).columns
    summary_df[numeric_cols] = summary_df[numeric_cols].round(4)

    summary_path = output_dir / OUTPUT_TEMPLATES["summary"].format(dataset=dataset)
    summary_df.to_csv(summary_path, index=False, sep="\t")
    
    print(f"\n{'='*60}")
    print("Comparison Summary (AC + DC)")
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
