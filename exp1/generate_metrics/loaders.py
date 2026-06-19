"""
Data loading and alignment utilities for two-step OPF comparison.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
import config 
from config import (
    PARQUET_FILES, FORECAST_METHODS, FORECASTS_PARQUET,
    BUS_COLUMNS, GEN_COLUMNS,
    DC_BUS_COLUMNS, DC_GEN_COLUMNS, DC_BRANCH_COLUMNS,
)


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
    
    keep = [
        "load_scenario_idx", "bus", "Pd", "Qd", "Pg", "Qg", "Vm", "Va", 
        "PQ", "PV", "REF", "min_vm_pu", "max_vm_pu", "GS", "BS", "Va_dc", "Pg_dc"
    ]
    existing = [c for c in keep if c in df.columns]
    return df[existing].copy()


def load_datakit_gen(parquet_dir: Path) -> pd.DataFrame:
    """Load generator data from datakit parquet output."""
    path = parquet_dir / PARQUET_FILES["gen"]
    df = pd.read_parquet(path)
    
    # Validate expected columns exist
    missing_cols = set(GEN_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in gen_data.parquet: {missing_cols}")
    
    keep = [
        "load_scenario_idx", "idx", "bus", "p_mw", "q_mvar", "min_p_mw", "max_p_mw", 
        "cp0_eur", "cp1_eur_per_mw", "cp2_eur_per_mw2", "p_mw_dc"
    ]
    existing = [c for c in keep if c in df.columns]
    return df[existing].copy()


def load_datakit_branch(parquet_dir: Path) -> pd.DataFrame:
    """Load branch topology data from datakit parquet output."""
    path = parquet_dir / PARQUET_FILES["branch"]
    df = pd.read_parquet(path)
    keep = [
        "load_scenario_idx", "idx", "from_bus", "to_bus", "pf", "qf", "pt", "qt", 
        "Yff_r", "Yff_i", "Yft_r", "Yft_i", "Ytf_r", "Ytf_i", "Ytt_r", "Ytt_i", 
        "ang_min", "ang_max", "rate_a", "pf_dc", "pt_dc"
    ]
    existing = [c for c in keep if c in df.columns]
    return df[existing].copy()


def has_dc_columns(bus_df: pd.DataFrame, gen_df: pd.DataFrame) -> bool:
    """Check whether datakit output contains DC-OPF result columns."""
    bus_has_dc = all(col in bus_df.columns for col in DC_BUS_COLUMNS)
    gen_has_dc = all(col in gen_df.columns for col in DC_GEN_COLUMNS)
    return bus_has_dc and gen_has_dc


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


def align_branch_results(
    pred_branch: pd.DataFrame,
    true_branch: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align predicted and ground-truth branch data.

    Returns:
        branch_merged with _pred and _true suffixes.
        DC columns (pf_dc, pt_dc) from pred are kept without suffix.
    """
    branch_merged = pred_branch.merge(
        true_branch,
        on=["load_scenario_idx", "idx"],
        how="inner",
        suffixes=("_pred", "_true"),
    )
    return branch_merged

