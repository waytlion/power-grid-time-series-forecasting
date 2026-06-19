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

BUS_COLUMNS = ["load_scenario_idx", "bus", "Pd", "Qd", "Pg", "Qg", "Vm", "Va", "PQ", "PV", "REF"]
DC_BUS_COLUMNS = ["Va_dc", "Pg_dc"]

GEN_COLUMNS = ["load_scenario_idx", "idx", "bus", "p_mw", "q_mvar", "cp0_eur", "cp1_eur_per_mw", "cp2_eur_per_mw2"]
DC_GEN_COLUMNS = ["p_mw_dc"]

BRANCH_COLUMNS = ["load_scenario_idx", "idx", "from_bus", "to_bus", "pf", "qf", "pt", "qt"]
DC_BRANCH_COLUMNS = ["pf_dc", "pt_dc"]

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

