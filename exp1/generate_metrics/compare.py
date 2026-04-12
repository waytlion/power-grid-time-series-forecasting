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
from config import FORECAST_METHODS, OUTPUT_TEMPLATES, FORECASTS_PARQUET, FORECAST_SEASONALITY
from loaders import (
    load_forecasts,
    load_datakit_bus,
    load_datakit_gen,
    load_datakit_branch,
    prepare_load_forecast_comparison,
    align_opf_results,
)
from metrics import (
    compute_mae,
    compute_rmse_by_bus_type,
    compute_generator_rmse,
    compute_cost_metrics,
    compute_forecast_metrics_table,
    compute_algebraic_power_residuals,
)


def _build_predicted_scenario_map(forecasts_df: pd.DataFrame) -> dict[int, int]:
    """Map predicted OPF flattened scenario indices to ground-truth scenario indices."""
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
        return dict(
            zip(
                keys["pred_flat_idx"].tolist(),
                keys["target_load_scenario_idx"].tolist(),
            )
        )

    forecast_scenarios = sorted(pd.to_numeric(forecasts_df["load_scenario_idx"], errors="raise").astype(int).unique())
    return dict(enumerate(forecast_scenarios))


def compare_single_method(
    method: str,
    forecasts_df: pd.DataFrame,
    ground_truth_dir: Path,
    predicted_opf_dir: Path,
    output_dir: Path,
    dataset: str,
    pred_scenario_map: dict[int, int],
    true_branch: pd.DataFrame,
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
    true_bus = load_datakit_bus(ground_truth_dir)
    pred_gen = load_datakit_gen(predicted_opf_dir)
    true_gen = load_datakit_gen(ground_truth_dir)
    
    # Preserve original flattened indices for graph contiguity
    pred_bus["pred_flat_idx"] = pred_bus["load_scenario_idx"].copy()
    pred_gen["pred_flat_idx"] = pred_gen["load_scenario_idx"].copy()

    # Remap predicted OPF scenario indices (0-based) to match ground-truth indices
    pred_bus["load_scenario_idx"] = pred_bus["load_scenario_idx"].map(pred_scenario_map)
    pred_gen["load_scenario_idx"] = pred_gen["load_scenario_idx"].map(pred_scenario_map)

    if pred_bus["load_scenario_idx"].isna().any() or pred_gen["load_scenario_idx"].isna().any():
        raise ValueError(
            "Predicted OPF scenario mapping produced NaN values. "
            "Forecast parquet and OPF output likely use incompatible flattened scenario indexing."
        )

    pred_bus["load_scenario_idx"] = pred_bus["load_scenario_idx"].astype(int)
    pred_gen["load_scenario_idx"] = pred_gen["load_scenario_idx"].astype(int)

    # 3. Align OPF results
    print("Aligning OPF results...")
    bus_aligned, gen_aligned = align_opf_results(pred_bus, true_bus, pred_gen, true_gen)
    print(f"  Aligned {len(bus_aligned)} bus observations across {bus_aligned['load_scenario_idx'].nunique()} scenarios")
    print(f"  Aligned {len(gen_aligned)} generator observations")
    
    # 4. Compute RMSE by bus type
    print("Computing RMSE by bus type...")
    rmse_df = compute_rmse_by_bus_type(bus_aligned, ["Vm", "Va", "Pg", "Qg"])
    rmse_path = method_output_dir / OUTPUT_TEMPLATES["rmse"].format(dataset=dataset)
    rmse_df.to_csv(rmse_path, index=False)
    print(f"  Saved RMSE table: {rmse_path}")
    # Extract key RMSE values for summary (aggregate across bus types)
    rmse_summary = rmse_df.groupby("feature")["rmse"].mean().to_dict()
    
    
    # 5. Compute generator RMSE
    print("Computing generator-level RMSE...")
    gen_rmse = compute_generator_rmse(gen_aligned)
    
    # 6. Compute cost metrics
    print("Computing cost/optimality gap...")
    cost_metrics = compute_cost_metrics(gen_aligned)
    
    # 6.5 Compute AC Power Residuals
    print("Computing exact AC algebraic power residuals...")
    residuals = compute_algebraic_power_residuals(bus_aligned, true_branch)
    
    # 7. Save metrics summary
    metrics_df = pd.DataFrame([{
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
    }])
    
    metrics_path = method_output_dir / OUTPUT_TEMPLATES["metrics"].format(dataset=dataset)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Saved metrics: {metrics_path}")
    
    # Return summary for aggregation
    return {
        "method": method,
        "mae_pd": mae_pd,
        "rmse_vm": rmse_summary.get("Vm", float("nan")),
        "rmse_va": rmse_summary.get("Va", float("nan")),
        "rmse_pg_bus": rmse_summary.get("Pg", float("nan")),  # Bus-level aggregated
        "rmse_pg_gen": gen_rmse,  # Generator-level
        "optimality_gap_pct": cost_metrics["mean_optimality_gap_pct"],
        "res_p": residuals["mae_residual_p_mw"],
        "res_q": residuals["mae_residual_q_mvar"],
    }


def generate_comparison_summary(summaries: list, output_dir: Path, dataset: str):
    """Generate side-by-side comparison table across all methods."""
    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / OUTPUT_TEMPLATES["summary"].format(dataset=dataset)
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n{'='*60}")
    print("Comparison Summary")
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

    pred_scenario_map = _build_predicted_scenario_map(forecasts_df)

    # 0. Load ground truth branch data (static topology) once
    print("Loading static branch topology...")
    true_branch = load_datakit_branch(args.ground_truth_dir)

    # Save combined forecast metrics table (all selected methods)
    forecast_table = compute_forecast_metrics_table(
        forecasts_df=forecasts_df,
        methods=available_methods,
        seasonality=args.forecast_seasonality,
    )
    forecast_table_path = args.output_dir / OUTPUT_TEMPLATES["forecast"].format(dataset=args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    forecast_table.to_csv(forecast_table_path, index=False)
    print(f"Saved forecast metrics table: {forecast_table_path}")
    
    # Process each method
    summaries = []
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
                true_branch=true_branch,
            )
            summaries.append(summary)
        except Exception as e:
            print(f"\nERROR processing {method}: {e}")
            raise
    
    # Generate comparison summary
    if summaries:
        generate_comparison_summary(summaries, args.output_dir, args.dataset)
    else:
        print("\nERROR No methods successfully processed.")


if __name__ == "__main__":
    main()
