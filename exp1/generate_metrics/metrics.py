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
        If 'horizon_step' is present in gen_df, also returns 'horizon_gaps' dict.
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
    group_cols = ["load_scenario_idx"]
    if "horizon_step" in gen_df.columns:
        group_cols.append("horizon_step")

    cost_per_scenario = gen_df.groupby(group_cols).agg({
        "cost_pred": "sum",
        "cost_true": "sum",
    }).reset_index()
    
    # Compute optimality gap (%)
    cost_per_scenario["gap_pct"] = (
        np.abs(cost_per_scenario["cost_pred"] - cost_per_scenario["cost_true"])
        / cost_per_scenario["cost_true"]
        * 100
    )
    
    results = {
        "mean_optimality_gap_pct": cost_per_scenario["gap_pct"].mean(),
        "median_optimality_gap_pct": cost_per_scenario["gap_pct"].median(),
        "max_optimality_gap_pct": cost_per_scenario["gap_pct"].max(),
    }

    if "horizon_step" in gen_df.columns:
        # Compute mean gap per horizon
        horizon_gaps = cost_per_scenario.groupby("horizon_step")["gap_pct"].mean().to_dict()
        results["horizon_gaps"] = horizon_gaps

    return results


# ---------------------------------------------------------------------------
# DC-OPF metric functions (DC predictions vs AC ground truth)
# ---------------------------------------------------------------------------

def compute_dc_rmse_by_bus_type(bus_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RMSE for DC-OPF results vs AC ground truth, split by bus type.

    DC-OPF yields Va_dc and Pg_dc; voltage magnitude is fixed at 1.0 p.u.
    All are compared against the AC ground-truth columns (Va_true, Pg_true, Vm_true).

    Returns:
        DataFrame with columns [bus_type, feature, rmse].
    """
    results = []

    for bus_type in ["PQ", "PV", "REF"]:
        mask = bus_df[f"{bus_type}_true"].astype(bool)
        subset = bus_df[mask]

        if len(subset) == 0:
            continue

        # Vm: DC assumes flat voltage at 1.0 p.u. — deviation from AC truth
        rmse_vm = float(np.sqrt(((1.0 - subset["Vm_true"]) ** 2).mean()))
        results.append({"bus_type": bus_type, "feature": "Vm_dc", "rmse": rmse_vm})

        # Va: DC voltage angle vs AC ground truth
        rmse_va = float(np.sqrt(((subset["Va_dc"] - subset["Va_true"]) ** 2).mean()))
        results.append({"bus_type": bus_type, "feature": "Va_dc", "rmse": rmse_va})

        # Pg: DC bus-level active generation vs AC ground truth
        rmse_pg = float(np.sqrt(((subset["Pg_dc"] - subset["Pg_true"]) ** 2).mean()))
        results.append({"bus_type": bus_type, "feature": "dc_Pg_bus", "rmse": rmse_pg})

    return pd.DataFrame(results)


def compute_dc_generator_rmse(gen_df: pd.DataFrame) -> float:
    """Compute RMSE for DC generator active power vs AC ground truth p_mw."""
    squared_error = (gen_df["p_mw_dc"] - gen_df["p_mw_true"]) ** 2
    return float(np.sqrt(squared_error.mean()))


def compute_dc_cost_metrics(gen_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute DC-OPF generation cost and optimality gap vs AC ground truth.

    Uses p_mw_dc with the predicted cost coefficients (same generators, same
    coefficients — only the dispatch differs between AC and DC).

    Returns:
        Dict with 'dc_mean_optimality_gap_pct' and related statistics.
        If 'horizon_step' is present, also returns 'dc_horizon_gaps' dict.
    """
    pg_dc = gen_df["p_mw_dc"]
    gen_df = gen_df.copy()
    gen_df["cost_dc"] = (
        gen_df["cp0_eur_pred"]
        + gen_df["cp1_eur_per_mw_pred"] * pg_dc
        + gen_df["cp2_eur_per_mw2_pred"] * pg_dc ** 2
    )

    # Reuse the already-computed AC true cost column if present, otherwise compute
    if "cost_true" not in gen_df.columns:
        pg_true = gen_df["p_mw_true"]
        gen_df["cost_true"] = (
            gen_df["cp0_eur_true"]
            + gen_df["cp1_eur_per_mw_true"] * pg_true
            + gen_df["cp2_eur_per_mw2_true"] * pg_true ** 2
        )

    group_cols = ["load_scenario_idx"]
    if "horizon_step" in gen_df.columns:
        group_cols.append("horizon_step")

    cost_per_scenario = gen_df.groupby(group_cols).agg({
        "cost_dc": "sum",
        "cost_true": "sum",
    }).reset_index()

    cost_per_scenario["gap_pct"] = (
        np.abs(cost_per_scenario["cost_dc"] - cost_per_scenario["cost_true"])
        / cost_per_scenario["cost_true"]
        * 100
    )

    results = {
        "dc_mean_optimality_gap_pct": float(cost_per_scenario["gap_pct"].mean()),
        "dc_median_optimality_gap_pct": float(cost_per_scenario["gap_pct"].median()),
        "dc_max_optimality_gap_pct": float(cost_per_scenario["gap_pct"].max()),
    }

    if "horizon_step" in gen_df.columns:
        horizon_gaps = cost_per_scenario.groupby("horizon_step")["gap_pct"].mean().to_dict()
        results["dc_horizon_gaps"] = horizon_gaps

    return results


def compute_dc_branch_rmse(branch_aligned: pd.DataFrame) -> Dict[str, float]:
    """
    Compute RMSE for DC branch flows vs AC ground truth.

    Compares pf_dc, pt_dc (from DC-OPF) against pf_true, pt_true (from AC ground truth).

    Returns:
        Dict with 'dc_rmse_pf' and 'dc_rmse_pt'.
    """
    rmse_pf = float(np.sqrt(((branch_aligned["pf_dc"] - branch_aligned["pf_true"]) ** 2).mean()))
    rmse_pt = float(np.sqrt(((branch_aligned["pt_dc"] - branch_aligned["pt_true"]) ** 2).mean()))
    return {
        "dc_rmse_pf": rmse_pf,
        "dc_rmse_pt": rmse_pt,
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
    topology_df: pd.DataFrame,
    chunk_size: int = 500
) -> Dict[str, float]:
    """
    Offline calculation mapping DataFrames -> Tensors and returning exactly computed P/Q residuals
    via purely algebraic PhysicsModules natively on CPU, processed in chunks to keep memory usage low.
    """
    if bus_aligned.empty:
        return {"mae_residual_p_mw": float("nan"), "mae_residual_q_mvar": float("nan")}
        
    unique_scenarios = sorted(bus_aligned["pred_flat_idx"].unique())
    num_nodes = len(bus_aligned) // len(unique_scenarios)
    
    # Sort so arrays map natively 0 -> N_total elements logically (grouped by prediction instance!)
    bus_aligned = bus_aligned.sort_values(["pred_flat_idx", "bus"]).reset_index(drop=True)
    
    baseMVA = 100.0
    
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
    
    branch_flow = ComputeBranchFlow()
    node_inj = ComputeNodeInjection()
    node_res = ComputeNodeResiduals()
    
    total_abs_res_P = 0.0
    total_abs_res_Q = 0.0
    total_count = 0
    
    for i in range(0, len(unique_scenarios), chunk_size):
        chunk_sc = unique_scenarios[i : i + chunk_size]
        chunk_num_sc = len(chunk_sc)
        chunk_df = bus_aligned[bus_aligned["pred_flat_idx"].isin(chunk_sc)]
        
        pred_dict = {
            "VM_OUT": chunk_df["Vm_pred"].values,
            "VA_OUT": chunk_df["Va_pred"].values * (np.pi / 180.0),
            "PG_OUT": chunk_df["Pg_pred"].values / baseMVA,
            "QG_OUT": chunk_df["Qg_pred"].values / baseMVA,
        }
        
        true_dict = {
            "PD_H": chunk_df["Pd_true"].values / baseMVA,
            "QD_H": chunk_df["Qd_true"].values / baseMVA,
            "GS": chunk_df["GS_true"].values,
            "BS": chunk_df["BS_true"].values,
        }
        
        bus_pred, bus_orig, edge_index, edge_attr = format_2step_tensors(
            pred_dict, true_dict, topo_dict, num_nodes, chunk_num_sc
        )
        
        with torch.no_grad():
            Pft, Qft = branch_flow(bus_pred, edge_index, edge_attr)
            P_in, Q_in = node_inj(Pft, Qft, edge_index, num_nodes * chunk_num_sc)
            res_P, res_Q = node_res(P_in, Q_in, bus_pred, bus_orig)
            
            total_abs_res_P += float(torch.abs(res_P).sum())
            total_abs_res_Q += float(torch.abs(res_Q).sum())
            total_count += res_P.numel()
            
    return {
        "mae_residual_p_mw": (total_abs_res_P / total_count) * baseMVA,
        "mae_residual_q_mvar": (total_abs_res_Q / total_count) * baseMVA,
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


def compute_dc_algebraic_power_residuals(
    bus_aligned: pd.DataFrame, 
    topology_df: pd.DataFrame,
    chunk_size: int = 500
) -> Dict[str, float]:
    """
    Offline calculation mapping dataframes-> tensors 
    --> returning computed P/Q residuals
    for using AC-PhysicsModules from graphkit, processed in chunks to keep memory usage low.
    DC-OPF uses: Vm = 1.0, Va = Va_dc, Pg = Pg_dc, Qg = 0.
    """
    if bus_aligned.empty or "Va_dc" not in bus_aligned.columns or "Pg_dc" not in bus_aligned.columns:
        return {"dc_mae_residual_p_mw": float("nan"), "dc_mae_residual_q_mvar": float("nan")}
        
    unique_scenarios = sorted(bus_aligned["pred_flat_idx"].unique())
    num_nodes = len(bus_aligned) // len(unique_scenarios)
    
    # Sort so arrays map  0 -> N_total elements logically (grouped by prediction instance!)
    bus_aligned = bus_aligned.sort_values(["pred_flat_idx", "bus"]).reset_index(drop=True)
    
    baseMVA = 100.0
    
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
    
    branch_flow = ComputeBranchFlow()
    node_inj = ComputeNodeInjection()
    node_res = ComputeNodeResiduals()
    
    total_abs_res_P = 0.0
    total_abs_res_Q = 0.0
    total_count = 0
    
    for i in range(0, len(unique_scenarios), chunk_size):
        chunk_sc = unique_scenarios[i : i + chunk_size]
        chunk_num_sc = len(chunk_sc)
        chunk_df = bus_aligned[bus_aligned["pred_flat_idx"].isin(chunk_sc)]
        
        pred_dict = {
            "VM_OUT": np.ones_like(chunk_df["Va_dc"].values),
            # Convert Va_dc to radians
            "VA_OUT": chunk_df["Va_dc"].values * (np.pi / 180.0),
            "PG_OUT": chunk_df["Pg_dc"].values / baseMVA,
            "QG_OUT": np.zeros_like(chunk_df["Pg_dc"].values),
        }
        
        true_dict = {
            "PD_H": chunk_df["Pd_true"].values / baseMVA,
            "QD_H": chunk_df["Qd_true"].values / baseMVA,
            "GS": chunk_df["GS_true"].values,
            "BS": chunk_df["BS_true"].values,
        }
        
        bus_pred, bus_orig, edge_index, edge_attr = format_2step_tensors(
            pred_dict, true_dict, topo_dict, num_nodes, chunk_num_sc
        )
        
        with torch.no_grad():
            Pft, Qft = branch_flow(bus_pred, edge_index, edge_attr)
            P_in, Q_in = node_inj(Pft, Qft, edge_index, num_nodes * chunk_num_sc)
            res_P, res_Q = node_res(P_in, Q_in, bus_pred, bus_orig)
            
            total_abs_res_P += float(torch.abs(res_P).sum())
            total_abs_res_Q += float(torch.abs(res_Q).sum())
            total_count += res_P.numel()
            
    return {
        "dc_mae_residual_p_mw": (total_abs_res_P / total_count) * baseMVA,
        "dc_mae_residual_q_mvar": (total_abs_res_Q / total_count) * baseMVA,
    }


def compute_ac_branch_rmse(branch_aligned: pd.DataFrame) -> Dict[str, float]:
    """
    Compute RMSE for AC predicted branch flows vs AC ground truth.
    Compares pf_pred, pt_pred against pf_true, pt_true.
    """
    if branch_aligned.empty or "pf_pred" not in branch_aligned.columns:
        return {"rmse_pf": float("nan"), "rmse_pt": float("nan")}
    rmse_pf = float(np.sqrt(((branch_aligned["pf_pred"] - branch_aligned["pf_true"]) ** 2).mean()))
    rmse_pt = float(np.sqrt(((branch_aligned["pt_pred"] - branch_aligned["pt_true"]) ** 2).mean()))
    return {
        "rmse_pf": rmse_pf,
        "rmse_pt": rmse_pt,
    }


def compute_violations(
    bus_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    mode: str = "ac",
    tol: float = 1e-5
) -> Dict[str, Dict[str, float]]:
    """
    Compute count and average magnitude of constraint violations for AC or DC OPF results.
    """
    results = {}
    
    # 1. Voltage Magnitude (Vm) violations
    if mode == "ac":
        vm_vals = bus_df["Vm_pred"].values
    else: # dc
        vm_vals = np.ones(len(bus_df)) # flat 1.0 p.u.
        
    min_vm = bus_df["min_vm_pu_true"].values
    max_vm = bus_df["max_vm_pu_true"].values
    
    under_vm = np.maximum(0.0, min_vm - vm_vals)
    over_vm = np.maximum(0.0, vm_vals - max_vm)
    vm_viols = under_vm + over_vm
    
    vm_viol_mask = vm_viols > tol
    vm_count = int(vm_viol_mask.sum())
    vm_mean = float(vm_viols.mean())
    
    results["vm"] = {"count": vm_count, "mean_magnitude": vm_mean}
    
    # 2. Generator Active Power (Pg) violations
    if mode == "ac":
        pg_vals = gen_df["p_mw_pred"].values
    else: # dc
        pg_vals = gen_df["p_mw_dc"].values
        
    min_pg = gen_df["min_p_mw_true"].values
    max_pg = gen_df["max_p_mw_true"].values
    
    under_pg = np.maximum(0.0, min_pg - pg_vals)
    over_pg = np.maximum(0.0, pg_vals - max_pg)
    pg_viols = under_pg + over_pg
    
    pg_viol_mask = pg_viols > tol
    pg_count = int(pg_viol_mask.sum())
    pg_mean = float(pg_viols.mean())
    
    results["pg"] = {"count": pg_count, "mean_magnitude": pg_mean}
    
    # 3. Thermal (S_ij) violations
    rate_a = branch_df["rate_a_true"].values
    valid_rate = (rate_a > 0.0) & (~np.isnan(rate_a))
    
    if mode == "ac":
        pf_pred = branch_df["pf_pred"].values
        qf_pred = branch_df["qf_pred"].values
        pt_pred = branch_df["pt_pred"].values
        qt_pred = branch_df["qt_pred"].values
        
        s_from = np.sqrt(pf_pred**2 + qf_pred**2)
        s_to = np.sqrt(pt_pred**2 + qt_pred**2)
    else: # dc
        pf_dc = branch_df["pf_dc"].values
        pt_dc = branch_df["pt_dc"].values
        
        s_from = np.abs(pf_dc)
        s_to = np.abs(pt_dc)
        
    s_max = np.maximum(s_from, s_to)
    
    thermal_viols = np.zeros_like(s_max)
    thermal_viols[valid_rate] = np.maximum(0.0, s_max[valid_rate] - rate_a[valid_rate])
    
    thermal_viol_mask = thermal_viols > tol
    thermal_count = int(thermal_viol_mask.sum())
    thermal_mean = float(thermal_viols[valid_rate].mean()) if valid_rate.any() else 0.0
    
    results["thermal"] = {"count": thermal_count, "mean_magnitude": thermal_mean}
    
    # 4. Angle Difference violations
    from_col = "from_bus_true" if "from_bus_true" in branch_df.columns else "from_bus"
    to_col = "to_bus_true" if "to_bus_true" in branch_df.columns else "to_bus"
    
    va_col = "Va_pred" if mode == "ac" else "Va_dc"
    
    if va_col in bus_df.columns:
        # Determine the unique scenario alignment column
        align_col = "pred_flat_idx" if "pred_flat_idx" in branch_df.columns and "pred_flat_idx" in bus_df.columns else "load_scenario_idx"
        
        # Merge va_col chunk by chunk to keep peak memory low
        unique_scenarios = sorted(branch_df[align_col].unique())
        chunk_size = 1000
        branch_merged_chunks = []
        for j in range(0, len(unique_scenarios), chunk_size):
            chunk_sc = unique_scenarios[j : j + chunk_size]
            chunk_branch = branch_df[branch_df[align_col].isin(chunk_sc)]
            chunk_bus = bus_df[bus_df[align_col].isin(chunk_sc)]
            
            chunk_merged = chunk_branch[[align_col, "load_scenario_idx", from_col, to_col, "ang_min_true", "ang_max_true"]].merge(
                chunk_bus[[align_col, "bus", va_col]],
                left_on=[align_col, from_col],
                right_on=[align_col, "bus"],
                how="left"
            ).rename(columns={va_col: "va_from"})
            
            chunk_merged = chunk_merged.merge(
                chunk_bus[[align_col, "bus", va_col]],
                left_on=[align_col, to_col],
                right_on=[align_col, "bus"],
                how="left"
            ).rename(columns={va_col: "va_to"})
            
            branch_merged_chunks.append(chunk_merged)
            
        branch_merged = pd.concat(branch_merged_chunks, ignore_index=True)
        ang_diff = branch_merged["va_from"].values - branch_merged["va_to"].values
        
        ang_min = branch_df["ang_min_true"].values
        ang_max = branch_df["ang_max_true"].values
        
        valid_ang = (~np.isnan(ang_min)) & (~np.isnan(ang_max)) & (ang_min > -359.0) & (ang_max < 359.0)
        
        ang_viols = np.zeros_like(ang_diff)
        under_ang = np.maximum(0.0, ang_min - ang_diff)
        over_ang = np.maximum(0.0, ang_diff - ang_max)
        ang_viols[valid_ang] = under_ang[valid_ang] + over_ang[valid_ang]
        
        ang_viol_mask = ang_viols > tol
        ang_count = int(ang_viol_mask.sum())
        ang_mean = float(ang_viols[valid_ang].mean()) if valid_ang.any() else 0.0
    else:
        ang_count = 0
        ang_mean = 0.0
        
    results["angle"] = {"count": ang_count, "mean_magnitude": ang_mean}
    
    return results
