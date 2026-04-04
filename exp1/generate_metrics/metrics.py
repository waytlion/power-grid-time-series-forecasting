"""
Metric computation functions matching GraphKit's ForecastOPFTask.
"""

import pandas as pd
import numpy as np
from typing import Dict


_EPS = 1e-8


def compute_mae(df: pd.DataFrame, features: list) -> Dict[str, float]:
    """
    Compute MAE for specified features.
    
    Args:
        df: DataFrame with {feature}_pred and {feature}_true columns.
        features: List of feature names (e.g., ['pd', 'qd']).
    
    Returns:
        Dict mapping feature names to MAE values.
    """
    return {
        feat: np.abs(df[f"{feat}_pred"] - df[f"{feat}_true"]).mean()
        for feat in features
    }


def compute_rmse_by_bus_type(
    bus_df: pd.DataFrame, features: list
) -> pd.DataFrame:
    """
    Compute RMSE for specified features, split by bus type (PQ/PV/REF).
    
    Args:
        bus_df: Aligned bus dataframe with PQ/PV/REF flags and pred/true columns.
        features: List of feature names (using parquet names, e.g., ['Vm', 'Va', 'Pg', 'Qg']).
    
    Returns:
        DataFrame with columns [bus_type, feature, rmse].
    """
    results = []
    
    # Bus type flags in parquet: PQ, PV, REF
    for bus_type in ["PQ", "PV", "REF"]:
        # Filter to buses of this type (flag == 1 or True)
        mask = bus_df[f"{bus_type}_true"].astype(bool)
        subset = bus_df[mask]
        
        if len(subset) == 0:
            continue  # No buses of this type
        
        for feat in features:
            squared_error = (subset[f"{feat}_pred"] - subset[f"{feat}_true"]) ** 2
            rmse = np.sqrt(squared_error.mean())
            results.append({
                "bus_type": bus_type,
                "feature": feat,
                "rmse": rmse,
            })
    
    return pd.DataFrame(results)


def compute_generator_rmse(gen_df: pd.DataFrame) -> float:
    """Compute RMSE for generator active power. Uses parquet column name: p_mw."""
    squared_error = (gen_df["p_mw_pred"] - gen_df["p_mw_true"]) ** 2
    return np.sqrt(squared_error.mean())


def compute_cost_metrics(gen_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute total generation cost and optimality gap.
    
    Formula: cost = cp0 + cp1 * pg + cp2 * pg^2
    
    Returns:
        Dict with 'mean_optimality_gap_pct' and related statistics.
    """
    # Compute cost per generator. Uses parquet names: p_mw, cp0_eur, cp1_eur_per_mw, cp2_eur_per_mw2
    for suffix in ["pred", "true"]:
        pg = gen_df[f"p_mw_{suffix}"]
        gen_df[f"cost_{suffix}"] = (
            gen_df[f"cp0_eur_{suffix}"]
            + gen_df[f"cp1_eur_per_mw_{suffix}"] * pg
            + gen_df[f"cp2_eur_per_mw2_{suffix}"] * pg ** 2
        )
    
    # Aggregate cost per scenario
    cost_per_scenario = gen_df.groupby("load_scenario_idx").agg({
        "cost_pred": "sum",
        "cost_true": "sum",
    })
    
    # Compute optimality gap (%)
    cost_per_scenario["gap_pct"] = (
        np.abs(cost_per_scenario["cost_pred"] - cost_per_scenario["cost_true"])
        / cost_per_scenario["cost_true"]
        * 100
    )
    
    return {
        "mean_optimality_gap_pct": cost_per_scenario["gap_pct"].mean(),
        "median_optimality_gap_pct": cost_per_scenario["gap_pct"].median(),
        "max_optimality_gap_pct": cost_per_scenario["gap_pct"].max(),
    }


def _attach_seasonal_naive_baseline(
    df: pd.DataFrame,
    seasonality: int,
    horizon_col: str | None,
) -> pd.DataFrame:
    """Attach naive baseline per row using value from 48 origins earlier.

    Matches ST-GNN behavior: for a given forecast origin, take one past value and
    repeat it across all future horizon steps.
    """
    out = df.copy()
    out["load_scenario_idx"] = pd.to_numeric(out["load_scenario_idx"], errors="raise").astype(int)
    out["bus_id"] = pd.to_numeric(out["bus_id"], errors="raise").astype(int)
    out["true"] = pd.to_numeric(out["true"], errors="coerce")

    if horizon_col:
        out[horizon_col] = pd.to_numeric(out[horizon_col], errors="raise").astype(int)
        min_horizon = int(out[horizon_col].min())
        base = out[out[horizon_col] == min_horizon][["load_scenario_idx", "bus_id", "true"]].copy()
    else:
        base = out[["load_scenario_idx", "bus_id", "true"]].copy()

    base = base.sort_values(["bus_id", "load_scenario_idx"], kind="stable")
    base["naive"] = base.groupby("bus_id", sort=False)["true"].shift(seasonality)
    base = base[["load_scenario_idx", "bus_id", "naive"]]

    out = out.merge(base, on=["load_scenario_idx", "bus_id"], how="left")
    return out


def _compute_basic_errors(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float]:
    """Return (rmse, mae, wmape, mse)."""
    error = y_pred - y_true
    abs_error = np.abs(error)
    sq_error = error ** 2

    rmse = float(np.sqrt(sq_error.mean()))
    mae = float(abs_error.mean())
    mse = float(sq_error.mean())

    denom = float(np.abs(y_true).sum())
    wmape = float(abs_error.sum() / (denom + _EPS))
    return rmse, mae, wmape, mse


def compute_forecast_metrics_table(
    forecasts_df: pd.DataFrame,
    methods: list[str],
    seasonality: int = 48,
) -> pd.DataFrame:
    """Compute forecast metrics per model for each horizon and a GLOBAL row."""
    required = {"load_scenario_idx", "bus_id", "true"}
    missing = required - set(forecasts_df.columns)
    if missing:
        raise ValueError(f"Missing required forecast columns: {sorted(missing)}")

    horizon_col = "horizon_step" if "horizon_step" in forecasts_df.columns else None
    work_df = forecasts_df.copy()
    work_df = _attach_seasonal_naive_baseline(work_df, seasonality=seasonality, horizon_col=horizon_col)

    horizon_values = (
        sorted(pd.to_numeric(work_df[horizon_col], errors="coerce").dropna().astype(int).unique().tolist())
        if horizon_col
        else [0]
    )

    if not horizon_values:
        raise ValueError("No valid horizon values found for forecast metric computation")

    min_horizon = min(horizon_values)

    def _horizon_label(h: int) -> str:
        return f"t+{h + 1}" if min_horizon == 0 else f"t+{h}"

    rows = []
    for method in methods:
        if method not in work_df.columns:
            raise ValueError(f"Method column '{method}' not found in forecasts dataframe")

        for horizon in horizon_values:
            subset = (
                work_df[work_df[horizon_col] == horizon]
                if horizon_col
                else work_df
            )

            y_true = pd.to_numeric(subset["true"], errors="coerce").to_numpy(dtype=float)
            y_pred = pd.to_numeric(subset[method], errors="coerce").to_numpy(dtype=float)
            y_naive = pd.to_numeric(subset["naive"], errors="coerce").to_numpy(dtype=float)

            valid = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true = y_true[valid]
            y_pred = y_pred[valid]

            if y_true.size == 0:
                rmse = mae = wmape = mse = float("nan")
            else:
                rmse, mae, wmape, mse = _compute_basic_errors(y_true, y_pred)

            valid_scaled = np.isfinite(pd.to_numeric(subset["true"], errors="coerce").to_numpy(dtype=float)) & np.isfinite(pd.to_numeric(subset[method], errors="coerce").to_numpy(dtype=float)) & np.isfinite(y_naive)
            if valid_scaled.any():
                y_true_s = pd.to_numeric(subset["true"], errors="coerce").to_numpy(dtype=float)[valid_scaled]
                y_pred_s = pd.to_numeric(subset[method], errors="coerce").to_numpy(dtype=float)[valid_scaled]
                y_naive_s = y_naive[valid_scaled]

                mae_scaled = float(np.abs(y_pred_s - y_true_s).mean())
                mse_scaled = float(((y_pred_s - y_true_s) ** 2).mean())
                mae_naive = float(np.abs(y_naive_s - y_true_s).mean())
                mse_naive = float(((y_naive_s - y_true_s) ** 2).mean())

                mase = float(mae_scaled / (mae_naive + _EPS))
                msse = float(mse_scaled / (mse_naive + _EPS))
            else:
                mase = float("nan")
                msse = float("nan")

            rows.append(
                {
                    "Model": method,
                    "Horizon": _horizon_label(int(horizon)),
                    "Pd (MW) - RMSE": rmse,
                    "Pd (MW) - MAE": mae,
                    "Pd (MW) - wMAPE": wmape,
                    "Pd (MW) - MASE": mase,
                    "Pd (MW) - MSSE": msse,
                }
            )

        y_true_global = pd.to_numeric(work_df["true"], errors="coerce").to_numpy(dtype=float)
        y_pred_global = pd.to_numeric(work_df[method], errors="coerce").to_numpy(dtype=float)
        y_naive_global = pd.to_numeric(work_df["naive"], errors="coerce").to_numpy(dtype=float)
        valid_global = np.isfinite(y_true_global) & np.isfinite(y_pred_global)
        y_true_global = y_true_global[valid_global]
        y_pred_global = y_pred_global[valid_global]

        if y_true_global.size == 0:
            rmse_g = mae_g = wmape_g = mse_g = float("nan")
        else:
            rmse_g, mae_g, wmape_g, mse_g = _compute_basic_errors(y_true_global, y_pred_global)

        valid_scaled_global = np.isfinite(pd.to_numeric(work_df["true"], errors="coerce").to_numpy(dtype=float)) & np.isfinite(pd.to_numeric(work_df[method], errors="coerce").to_numpy(dtype=float)) & np.isfinite(y_naive_global)
        if valid_scaled_global.any():
            y_true_sg = pd.to_numeric(work_df["true"], errors="coerce").to_numpy(dtype=float)[valid_scaled_global]
            y_pred_sg = pd.to_numeric(work_df[method], errors="coerce").to_numpy(dtype=float)[valid_scaled_global]
            y_naive_sg = y_naive_global[valid_scaled_global]

            mae_scaled_g = float(np.abs(y_pred_sg - y_true_sg).mean())
            mse_scaled_g = float(((y_pred_sg - y_true_sg) ** 2).mean())
            mae_naive_g = float(np.abs(y_naive_sg - y_true_sg).mean())
            mse_naive_g = float(((y_naive_sg - y_true_sg) ** 2).mean())

            mase_g = float(mae_scaled_g / (mae_naive_g + _EPS))
            msse_g = float(mse_scaled_g / (mse_naive_g + _EPS))
        else:
            mase_g = float("nan")
            msse_g = float("nan")

        rows.append(
            {
                "Model": method,
                "Horizon": "GLOBAL",
                "Pd (MW) - RMSE": rmse_g,
                "Pd (MW) - MAE": mae_g,
                "Pd (MW) - wMAPE": wmape_g,
                "Pd (MW) - MASE": mase_g,
                "Pd (MW) - MSSE": msse_g,
            }
        )

    out = pd.DataFrame(rows)
    horizon_order = {f"t+{h + 1}" if min_horizon == 0 else f"t+{h}": i for i, h in enumerate(horizon_values)}
    horizon_order["GLOBAL"] = len(horizon_values)
    out["_h_order"] = out["Horizon"].map(horizon_order)
    out = out.sort_values(["Model", "_h_order"], kind="stable").drop(columns=["_h_order"]).reset_index(drop=True)
    return out
