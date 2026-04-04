"""
Data loading and alignment utilities for two-step OPF comparison.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
import config 
from config import PARQUET_FILES, FORECAST_METHODS, FORECASTS_PARQUET, BUS_COLUMNS, GEN_COLUMNS


def load_forecasts(forecasts_parquet: Path = FORECASTS_PARQUET) -> pd.DataFrame:
    """
    Load all forecast methods from unified parquet file.
    Uses parquet column names directly: load_scenario_idx, bus_id, true, xgb, snaive, tgt, sarima.
    
    Returns:
        DataFrame with forecast columns (parquet names unchanged).
    """
    df = pd.read_parquet(forecasts_parquet)
    
    # Validate core columns needed by comparison pipeline
    required_cols = {"load_scenario_idx", "bus_id", "true"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in forecasts.parquet: {missing_cols}")
    
    return df


def load_datakit_bus(parquet_dir: Path) -> pd.DataFrame:
    """Load bus data from datakit parquet output."""
    path = parquet_dir / PARQUET_FILES["bus"]
    df = pd.read_parquet(path)
    
    # Validate expected columns exist
    missing_cols = set(BUS_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in bus_data.parquet: {missing_cols}")
    
    return df


def load_datakit_gen(parquet_dir: Path) -> pd.DataFrame:
    """Load generator data from datakit parquet output."""
    path = parquet_dir / PARQUET_FILES["gen"]
    df = pd.read_parquet(path)
    
    # Validate expected columns exist
    missing_cols = set(GEN_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in gen_data.parquet: {missing_cols}")
    
    return df


def prepare_load_forecast_comparison(
    forecasts_df: pd.DataFrame, method: str
) -> pd.DataFrame:
    """
    Prepare load forecast data for a single method.
    
    Args:
        forecasts_df: Full forecasts dataframe with all methods.
        method: Forecast method name (e.g., 'xgb').
    
    Returns:
        DataFrame with forecast comparison columns
    """
    if method not in FORECAST_METHODS:
        raise ValueError(f"Unknown method '{method}'. Available: {FORECAST_METHODS}")
    
    # Select columns and rename to standard names for comparison
    return forecasts_df[["load_scenario_idx", "bus_id", method, "true"]].rename(
        columns={"load_scenario_idx": "scenario", "bus_id": "bus", method: "pred"}
    )


def align_opf_results(
    pred_bus: pd.DataFrame,
    true_bus: pd.DataFrame,
    pred_gen: pd.DataFrame,
    true_gen: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align predicted and ground-truth OPF results.
    
    Returns:
        (bus_merged, gen_merged): Aligned dataframes with _pred and _true suffixes.
    """
    # Align bus data on (load_scenario_idx, bus)
    bus_merged = pred_bus.merge(
        true_bus,
        on=["load_scenario_idx", "bus"],
        how="inner",
        suffixes=("_pred", "_true"),
    )
    
    # Align generator data on (load_scenario_idx, idx)
    gen_merged = pred_gen.merge(
        true_gen,
        on=["load_scenario_idx", "idx"],
        how="inner",
        suffixes=("_pred", "_true"),
    )
    
    # Validate that all predicted scenarios exist in ground truth
    pred_scenarios = set(pred_bus["load_scenario_idx"].unique())
    true_scenarios = set(true_bus["load_scenario_idx"].unique())
    
    missing_from_true = pred_scenarios - true_scenarios
    if missing_from_true:
        raise ValueError(
            f"Predicted scenarios not found in ground truth: {sorted(missing_from_true)[:5]}"
        )
    
    return bus_merged, gen_merged
