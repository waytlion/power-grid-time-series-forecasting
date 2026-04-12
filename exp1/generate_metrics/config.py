"""
Configuration for two-step OPF comparison pipeline.
"""

from pathlib import Path

# Paths (relative to Thesis_Repo root)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # Thesis_Repo/
FORECASTS_PARQUET = _REPO_ROOT / "exp1" / "data" / "data_in" / "benchmark_results.parquet"
FORECAST_SEASONALITY = 48

# Forecast methods available in forecasts.parquet
FORECAST_METHODS = ["xgb", "snaive", "tgt", "sarima"]

# Parquet column names (single source of truth - uses parquet names directly)
FORECAST_COLUMNS = ["load_scenario_idx", "bus_id", "true", "xgb", "snaive", "tgt", "sarima"]
BUS_COLUMNS = ["load_scenario_idx", "bus", "Pd", "Qd", "Pg", "Qg", "Vm", "Va", "PQ", "PV", "REF"]
GEN_COLUMNS = ["load_scenario_idx", "idx", "bus", "p_mw", "q_mvar", "cp0_eur", "cp1_eur_per_mw", "cp2_eur_per_mw2"]

# NOTE: Pg in bus_data.parquet is assumed to be pre-aggregated per bus. Verify with colleague if needed.
# NOTE: All units assumed as-labeled (MW, MVar, p.u., rad, EUR). No conversion applied.

PARQUET_FILES = {
    "bus": "bus_data.parquet",
    "gen": "gen_data.parquet",
    "branch": "branch_data.parquet",
}

OUTPUT_TEMPLATES = {
    "forecast_mae": "{dataset}_forecast_MAE.csv",
    "forecast": "{dataset}_forecast.csv",
    "rmse": "{dataset}_RMSE.csv",
    "metrics": "{dataset}_metrics.csv",
    "summary": "comparison_summary.csv",
}
