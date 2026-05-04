"""
Metric computation functions matching GraphKit's ForecastOPFTask.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Dict

# Provide access to gridfm-graphkit-dev
_kit_path = Path(__file__).resolve().parents[3] / "gridfm-graphkit-dev"
if str(_kit_path) not in sys.path:
    sys.path.insert(0, str(_kit_path))

try:
    from gridfm_graphkit.datasets.globals import (
        VM_OUT, VA_OUT, PG_OUT, QG_OUT,
        PD_H, QD_H, GS, BS,
        YFF_TT_R, YFF_TT_I, YFT_TF_R, YFT_TF_I
    )
    from gridfm_graphkit.models.utils import (
        ComputeBranchFlow,
        ComputeNodeInjection,
        ComputeNodeResiduals
    )
except ImportError as e:
    raise ImportError(
        f"Failed to load physics modules from {str(_kit_path)}. "
        f"Ensure gridfm-graphkit-dev is located side-by-side with Thesis_Repo."
    ) from e

import torch


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


def format_2step_tensors(
    predictions_dict: Dict[str, np.ndarray],
    true_data_dict: Dict[str, np.ndarray],
    topology_dict: Dict[str, np.ndarray],
    num_nodes: int,
    num_scenarios: int
):
    """
    Map flat arrays from datakit into exactly scaled PyG disjoint graphs.
    """
    S = num_scenarios
    N = num_nodes
    num_nodes_total = S * N
    
    # 1. Bus data
    bus_data_pred = torch.zeros((num_nodes_total, max(VM_OUT, VA_OUT, PG_OUT, QG_OUT) + 1), dtype=torch.float32)
    bus_data_pred[:, VM_OUT] = torch.tensor(predictions_dict["VM_OUT"], dtype=torch.float32)
    bus_data_pred[:, VA_OUT] = torch.tensor(predictions_dict["VA_OUT"], dtype=torch.float32)
    bus_data_pred[:, PG_OUT] = torch.tensor(predictions_dict["PG_OUT"], dtype=torch.float32)
    bus_data_pred[:, QG_OUT] = torch.tensor(predictions_dict["QG_OUT"], dtype=torch.float32)

    bus_data_orig = torch.zeros((num_nodes_total, max(PD_H, QD_H, GS, BS) + 1), dtype=torch.float32)
    bus_data_orig[:, PD_H] = torch.tensor(true_data_dict["PD_H"], dtype=torch.float32)
    bus_data_orig[:, QD_H] = torch.tensor(true_data_dict["QD_H"], dtype=torch.float32)
    bus_data_orig[:, GS] = torch.tensor(true_data_dict["GS"], dtype=torch.float32)
    bus_data_orig[:, BS] = torch.tensor(true_data_dict["BS"], dtype=torch.float32)

    # 2. Topology expansion
    base_from = torch.tensor(topology_dict["from_bus"], dtype=torch.long)
    base_to = torch.tensor(topology_dict["to_bus"], dtype=torch.long)
    E = len(base_from)
    
    # Batch shift logic:
    offsets = torch.arange(S, dtype=torch.long).unsqueeze(1) * N  # [S, 1]
    
    from_bus_global = (base_from.unsqueeze(0) + offsets).flatten() # [S * E]
    to_bus_global = (base_to.unsqueeze(0) + offsets).flatten()     # [S * E]
    
    # Admittances (repeat identical topology S times)
    Yff_r = torch.tensor(topology_dict["Yff_r"], dtype=torch.float32).repeat(S)
    Yff_i = torch.tensor(topology_dict["Yff_i"], dtype=torch.float32).repeat(S)
    Yft_r = torch.tensor(topology_dict["Yft_r"], dtype=torch.float32).repeat(S)
    Yft_i = torch.tensor(topology_dict["Yft_i"], dtype=torch.float32).repeat(S)
    
    Ytf_r = torch.tensor(topology_dict["Ytf_r"], dtype=torch.float32).repeat(S)
    Ytf_i = torch.tensor(topology_dict["Ytf_i"], dtype=torch.float32).repeat(S)
    Ytt_r = torch.tensor(topology_dict["Ytt_r"], dtype=torch.float32).repeat(S)
    Ytt_i = torch.tensor(topology_dict["Ytt_i"], dtype=torch.float32).repeat(S)

    # Bi-directional Forward Edge mapping (from_bus -> to_bus)
    edge_index_fwd = torch.stack([from_bus_global, to_bus_global], dim=0)
    edge_attr_fwd = torch.zeros((E * S, max(YFF_TT_R, YFF_TT_I, YFT_TF_R, YFT_TF_I) + 1), dtype=torch.float32)
    edge_attr_fwd[:, YFF_TT_R] = Yff_r
    edge_attr_fwd[:, YFF_TT_I] = Yff_i
    edge_attr_fwd[:, YFT_TF_R] = Yft_r
    edge_attr_fwd[:, YFT_TF_I] = Yft_i

    # Bi-directional Reverse Edge mapping (to_bus -> from_bus)
    edge_index_rev = torch.stack([to_bus_global, from_bus_global], dim=0)
    edge_attr_rev = torch.zeros((E * S, max(YFF_TT_R, YFF_TT_I, YFT_TF_R, YFT_TF_I) + 1), dtype=torch.float32)
    # Important: the physics equations view 'from' from the first element of edge_index.
    # Therefore, the reversed perspective requires mapping Ytt to Yff.
    edge_attr_rev[:, YFF_TT_R] = Ytt_r
    edge_attr_rev[:, YFF_TT_I] = Ytt_i
    edge_attr_rev[:, YFT_TF_R] = Ytf_r
    edge_attr_rev[:, YFT_TF_I] = Ytf_i
    
    edge_index = torch.cat([edge_index_fwd, edge_index_rev], dim=1)
    edge_attr = torch.cat([edge_attr_fwd, edge_attr_rev], dim=0)
    
    return bus_data_pred, bus_data_orig, edge_index, edge_attr


def compute_algebraic_power_residuals(
    bus_aligned: pd.DataFrame, 
    topology_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Offline calculation mapping DataFrames -> Tensors and returning exactly computed P/Q residuals
    via purely algebraic PhysicsModules natively on CPU.
    """
    if bus_aligned.empty:
        return {"mae_residual_p_mw": float("nan"), "mae_residual_q_mvar": float("nan")}
        
    num_scenarios = bus_aligned["pred_flat_idx"].nunique()
    num_nodes = len(bus_aligned) // num_scenarios
    
    # Sort so arrays map natively 0 -> N_total elements logically (grouped by prediction instance!)
    bus_aligned = bus_aligned.sort_values(["pred_flat_idx", "bus"]).reset_index(drop=True)
    
    # Base MVA for per-unit scaling
    # GridFM physics modules natively evaluate residuals in per-unit (p.u.) space.
    # While DataKit exports Y-bus and shunts (GS/BS) in p.u., active/reactive 
    # generation (Pg/Qg) and demand (Pd/Qd) are exported in absolute physical units (MW/MVAR).
    baseMVA = 100.0
    
    pred_dict = {
        "VM_OUT": bus_aligned["Vm_pred"].values,
        # DataKit exports voltage angles natively in degrees. PyTorch trigonometric 
        # operations (used in ComputeBranchFlow) mathematically require radians.
        "VA_OUT": bus_aligned["Va_pred"].values * (np.pi / 180.0),
        "PG_OUT": bus_aligned["Pg_pred"].values / baseMVA,
        "QG_OUT": bus_aligned["Qg_pred"].values / baseMVA,
    }
    
    true_dict = {
        "PD_H": bus_aligned["Pd_true"].values / baseMVA,
        "QD_H": bus_aligned["Qd_true"].values / baseMVA,
        # GS and BS are already structured as per-unit admittances by DataKit.
        "GS": bus_aligned["GS_true"].values,
        "BS": bus_aligned["BS_true"].values,
    }
    
    # Obtain static base topology from any one scenario (e.g. index 0).
    base_topology = topology_df[topology_df['load_scenario_idx'] == topology_df['load_scenario_idx'].iloc[0]]
    topo_dict = {
        "from_bus": base_topology["from_bus"].values,
        "to_bus": base_topology["to_bus"].values,
        "Yff_r": base_topology["Yff_r"].values,
        "Yff_i": base_topology["Yff_i"].values,
        "Yft_r": base_topology["Yft_r"].values,
        "Yft_i": base_topology["Yft_i"].values,
        "Ytf_r": base_topology["Ytf_r"].values,
        "Ytf_i": base_topology["Ytf_i"].values,
        "Ytt_r": base_topology["Ytt_r"].values,
        "Ytt_i": base_topology["Ytt_i"].values,
    }
    
    bus_pred, bus_orig, edge_index, edge_attr = format_2step_tensors(
        pred_dict, true_dict, topo_dict, num_nodes, num_scenarios
    )
    
    branch_flow = ComputeBranchFlow()
    node_inj = ComputeNodeInjection()
    node_res = ComputeNodeResiduals()
    
    with torch.no_grad():
        Pft, Qft = branch_flow(bus_pred, edge_index, edge_attr)
        P_in, Q_in = node_inj(Pft, Qft, edge_index, num_nodes * num_scenarios)
        res_P, res_Q = node_res(P_in, Q_in, bus_pred, bus_orig)
        
    return {
        # Rescale the evaluated per-unit residuals back into MW/MVAR for reporting
        "mae_residual_p_mw": float(torch.abs(res_P).mean()) * baseMVA,
        "mae_residual_q_mvar": float(torch.abs(res_Q).mean()) * baseMVA,
    }


def _attach_seasonal_naive_baseline(
    df: pd.DataFrame,
    seasonality: int,
    horizon_col: str | None,
) -> pd.DataFrame:
    """Attach seasonal naive baseline per row using y_{t+h-seasonality}.

    For each row (origin t, horizon h), the baseline prediction is the observed
    value at target index (t+h) shifted by `seasonality`.
    """
    out = df.copy()
    out["load_scenario_idx"] = pd.to_numeric(out["load_scenario_idx"], errors="raise").astype(int)
    out["bus_id"] = pd.to_numeric(out["bus_id"], errors="raise").astype(int)
    out["true"] = pd.to_numeric(out["true"], errors="coerce")

    if horizon_col:
        out[horizon_col] = pd.to_numeric(out[horizon_col], errors="raise").astype(int)
        out["target_load_scenario_idx"] = out["load_scenario_idx"] + out[horizon_col]
    else:
        out["target_load_scenario_idx"] = out["load_scenario_idx"]

    # Build true time series by bus and target time index, then shift by seasonality.
    base = (
        out[["bus_id", "target_load_scenario_idx", "true"]]
        .drop_duplicates(subset=["bus_id", "target_load_scenario_idx"], keep="first")
        .sort_values(["bus_id", "target_load_scenario_idx"], kind="stable")
    )
    base["naive"] = base.groupby("bus_id", sort=False)["true"].shift(seasonality)

    out = out.merge(
        base[["bus_id", "target_load_scenario_idx", "naive"]],
        on=["bus_id", "target_load_scenario_idx"],
        how="left",
    )
    out = out.drop(columns=["target_load_scenario_idx"])
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
