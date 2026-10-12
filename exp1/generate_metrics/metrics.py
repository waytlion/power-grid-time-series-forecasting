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


def compute_rmse_by_bus_type(
    bus_df: pd.DataFrame, features: list
) -> pd.DataFrame:
    """
    Compute RMSE for specified features, split by bus type (PQ/PV/REF).
    
    Args:
        bus_df: Aligned bus dataframe with PQ/PV/REF flags and pred/true columns.
        features: List of feature names (using parquet names, e.g., ['Vm', 'Va', 'Pg', 'Qg']).
    
    Returns:
        DataFrame with columns [bus_type, feature, rmse, mae, n] (n = bus count
        for that type, so callers can pool per-bus-equal not per-type-equal).
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
            error = subset[f"{feat}_pred"] - subset[f"{feat}_true"]
            results.append({
                "bus_type": bus_type,
                "feature": feat,
                "rmse": np.sqrt((error ** 2).mean()),
                "mae": error.abs().mean(),
                "n": len(subset),
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
        "mean_true_cost": cost_per_scenario["cost_true"].mean(),
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
        DataFrame with columns [bus_type, feature, rmse, mae, n] (n = bus count
        for that type, so callers can pool per-bus-equal not per-type-equal).
    """
    results = []

    for bus_type in ["PQ", "PV", "REF"]:
        mask = bus_df[f"{bus_type}_true"].astype(bool)
        subset = bus_df[mask]

        if len(subset) == 0:
            continue

        n = len(subset)
        # Vm: DC assumes flat voltage at 1.0 p.u. — deviation from AC truth
        dev_vm = 1.0 - subset["Vm_true"]
        results.append({"bus_type": bus_type, "feature": "Vm_dc",
                        "rmse": float(np.sqrt((dev_vm ** 2).mean())), "mae": float(dev_vm.abs().mean()), "n": n})

        # Va: DC voltage angle vs AC ground truth
        err_va = subset["Va_dc"] - subset["Va_true"]
        results.append({"bus_type": bus_type, "feature": "Va_dc",
                        "rmse": float(np.sqrt((err_va ** 2).mean())), "mae": float(err_va.abs().mean()), "n": n})

        # Pg: DC bus-level active generation vs AC ground truth
        err_pg = subset["Pg_dc"] - subset["Pg_true"]
        results.append({"bus_type": bus_type, "feature": "dc_Pg_bus",
                        "rmse": float(np.sqrt((err_pg ** 2).mean())), "mae": float(err_pg.abs().mean()), "n": n})

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


def merit_order_rebalance(pg, delta, pmin, pmax, c1, c2, slack):
    """
    Close each scenario's generation imbalance by merit order on the predicted dispatch.

    - pg [S, G] MW; delta [S] MW (> 0 = add generation); pmin, pmax, c1, c2, slack (bool) [G]
    - rank by marginal cost at the predicted output, c1 + 2 c2 pg (fixed, not re-ranked while filling)
    - shortfall: cheapest unit first, up to Pmax; surplus: most expensive first, down to Pmin
    - remainder beyond total headroom -> first slack unit, outside its limits (= violation)
    -> (rebalanced pg [S, G], remainder put on the slack [S] MW)
    """
    up = (delta > 0)[:, None]
    room = np.where(up, pmax - pg, pg - pmin).clip(min=0)
    mc = c1 + 2 * c2 * pg
    order = np.argsort(np.where(up, mc, -mc), axis=1, kind="stable")
    room_o = np.take_along_axis(room, order, axis=1)
    need = np.abs(delta)[:, None]
    step_o = np.clip(need - (np.cumsum(room_o, axis=1) - room_o), 0, room_o)
    step = np.empty_like(step_o)
    np.put_along_axis(step, order, step_o, axis=1)
    rest = (need[:, 0] - step.sum(1)).clip(min=0)
    step[:, np.flatnonzero(slack)[0]] += rest
    return pg + np.sign(delta)[:, None] * step, rest


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


def extract_per_bus_residuals(
    bus_aligned: pd.DataFrame,
    topology_df: pd.DataFrame,
    mode: str = "ac",
    chunk_size: int = 500,
) -> pd.DataFrame:
    """Per-bus SIGNED power-balance residuals in physical units, for AC or DC dispatch.

    Same physics/tensor path as compute_(dc_)algebraic_power_residuals, but keeps
    the per-bus value instead of collapsing to a MAE. res_P/res_Q are computed in
    p.u. and returned in MW/MVAr (x baseMVA). Row order is preserved, so each output
    row maps 1:1 to a (pred_flat_idx, bus) observation.

    - mode="ac": dispatch = (Vm_pred, Va_pred, Pg_pred, Qg_pred)
    - mode="dc": dispatch = (Vm=1, Va_dc, Pg_dc, Qg=0)  [DC-OPF scored via AC equations]

    Returns tidy DataFrame: [pred_flat_idx, bus, res_p_mw, res_q_mvar].
    """
    if mode not in ("ac", "dc"):
        raise ValueError(f"mode must be 'ac' or 'dc', got {mode!r}")
    need = {"ac": ["Vm_pred", "Va_pred", "Pg_pred", "Qg_pred"],
            "dc": ["Va_dc", "Pg_dc"]}[mode]
    if bus_aligned.empty or any(c not in bus_aligned.columns for c in need):
        return pd.DataFrame(columns=["pred_flat_idx", "bus", "res_p_mw", "res_q_mvar"])

    unique_scenarios = sorted(bus_aligned["pred_flat_idx"].unique())
    num_nodes = len(bus_aligned) // len(unique_scenarios)

    # Sort so flattened node arrays map 0 -> N_total grouped by prediction instance.
    bus_aligned = bus_aligned.sort_values(["pred_flat_idx", "bus"]).reset_index(drop=True)

    baseMVA = 100.0

    base_topology = topology_df[topology_df["load_scenario_idx"] == topology_df["load_scenario_idx"].iloc[0]]
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

    out = []
    for i in range(0, len(unique_scenarios), chunk_size):
        chunk_sc = unique_scenarios[i : i + chunk_size]
        chunk_num_sc = len(chunk_sc)
        chunk_df = bus_aligned[bus_aligned["pred_flat_idx"].isin(chunk_sc)]

        if mode == "ac":
            pred_dict = {
                "VM_OUT": chunk_df["Vm_pred"].values,
                "VA_OUT": chunk_df["Va_pred"].values * (np.pi / 180.0),
                "PG_OUT": chunk_df["Pg_pred"].values / baseMVA,
                "QG_OUT": chunk_df["Qg_pred"].values / baseMVA,
            }
        else:  # dc: Vm=1, Va=Va_dc, Pg=Pg_dc, Qg=0
            pred_dict = {
                "VM_OUT": np.ones_like(chunk_df["Va_dc"].values),
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

        rec = chunk_df[["pred_flat_idx", "bus"]].copy()
        rec["res_p_mw"] = res_P.numpy() * baseMVA
        rec["res_q_mvar"] = res_Q.numpy() * baseMVA
        out.append(rec)

    return pd.concat(out, ignore_index=True)


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


def _angle_diff_per_branch(bus_df: pd.DataFrame, branch_df: pd.DataFrame, va_col: str) -> np.ndarray:
    """Voltage-angle difference (va_from - va_to) per branch row, aligned to branch_df."""
    from_col = "from_bus_true" if "from_bus_true" in branch_df.columns else "from_bus"
    to_col = "to_bus_true" if "to_bus_true" in branch_df.columns else "to_bus"
    align_col = (
        "pred_flat_idx"
        if "pred_flat_idx" in branch_df.columns and "pred_flat_idx" in bus_df.columns
        else "load_scenario_idx"
    )
    scenarios = sorted(branch_df[align_col].unique())
    merged_chunks = []
    for j in range(0, len(scenarios), 1000):
        chunk_sc = scenarios[j : j + 1000]
        cb = branch_df[branch_df[align_col].isin(chunk_sc)]
        cu = bus_df[bus_df[align_col].isin(chunk_sc)]
        m = cb[[align_col, from_col, to_col]].merge(
            cu[[align_col, "bus", va_col]], left_on=[align_col, from_col],
            right_on=[align_col, "bus"], how="left",
        ).rename(columns={va_col: "va_from"})
        m = m.merge(
            cu[[align_col, "bus", va_col]], left_on=[align_col, to_col],
            right_on=[align_col, "bus"], how="left",
        ).rename(columns={va_col: "va_to"})
        merged_chunks.append(m)
    merged = pd.concat(merged_chunks, ignore_index=True)
    return merged["va_from"].values - merged["va_to"].values


def compute_feasibility_metrics(
    bus_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    mode: str = "ac",
    tol: float = 1e-5,
) -> Dict[str, float]:
    """Feasibility statistics standard in the learning-to-OPF literature, per
    constraint type (vm, pg, qg, thermal, angle) and overall.

    Canonical quintet per constraint type:
      - ``{t}_viol_count``: number of violating elements
      - ``{t}_viol_mean``: mean violation magnitude over ALL elements (one-sided)
      - ``{t}_viol_max``: max violation magnitude
      - ``{t}_viol_rate``: violating elements / total
      - ``{t}_scenario_viol_rate``: scenarios with >=1 violation / total scenarios
    ``fully_scenario_viol_rate`` is the fraction of scenarios with any violation of
    any type. Keys are prefixed ``dc_`` for DC mode.

    Units: vm [p.u.], pg [MW], qg [MVAr], thermal [MVA], angle [deg].
    """
    prefix = "" if mode == "ac" else "dc_"
    scen_bus = "pred_flat_idx" if "pred_flat_idx" in bus_df.columns else "load_scenario_idx"
    scen_gen = "pred_flat_idx" if "pred_flat_idx" in gen_df.columns else "load_scenario_idx"
    scen_br = "pred_flat_idx" if "pred_flat_idx" in branch_df.columns else "load_scenario_idx"

    # (magnitude per element, scenario per element) for each constraint type
    per_type: Dict[str, tuple] = {}

    # Voltage magnitude (DC: flat 1.0 p.u.)
    vm = np.ones(len(bus_df)) if mode == "dc" else bus_df["Vm_pred"].values
    vm_mag = (
        np.maximum(0.0, bus_df["min_vm_pu_true"].values - vm)
        + np.maximum(0.0, vm - bus_df["max_vm_pu_true"].values)
    )
    per_type["vm"] = (vm_mag, bus_df[scen_bus].values)

    # Generator active power limits
    pg = gen_df["p_mw_dc"].values if mode == "dc" else gen_df["p_mw_pred"].values
    pg_mag = (
        np.maximum(0.0, gen_df["min_p_mw_true"].values - pg)
        + np.maximum(0.0, pg - gen_df["max_p_mw_true"].values)
    )
    per_type["pg"] = (pg_mag, gen_df[scen_gen].values)

    # Generator reactive power limits, aggregated to the bus (predicted Qg is
    # bus-level). Checked only at generator buses (inner join) — reactive limits
    # do not apply at pure load buses, matching graphkit's PV/REF convention.
    # DC-OPF carries no reactive power, so Qg violations are omitted.
    if mode == "ac" and "Qg_pred" in bus_df.columns and "min_q_mvar_true" in gen_df.columns:
        # After align_opf_results merges on ["load_scenario_idx", "idx"], the "bus"
        # column from each side gets a suffix → use "bus_true" if present, else "bus".
        bus_col_gen = "bus_true" if "bus_true" in gen_df.columns else "bus"
        q_lim = (
            gen_df.groupby([scen_gen, bus_col_gen])[["min_q_mvar_true", "max_q_mvar_true"]]
            .sum()
            .reset_index()
            .rename(columns={bus_col_gen: "bus"})
        )
        bus_cols = list(dict.fromkeys([scen_bus, "bus", "Qg_pred"]))
        b = bus_df[bus_cols].merge(q_lim, left_on=[scen_bus, "bus"], right_on=[scen_gen, "bus"], how="inner")
        qg = b["Qg_pred"].values
        qg_mag = (
            np.maximum(0.0, b["min_q_mvar_true"].values - qg)
            + np.maximum(0.0, qg - b["max_q_mvar_true"].values)
        )
        per_type["qg"] = (qg_mag, b[scen_bus].values)

    # Branch thermal limits (apparent power vs rate_a)
    rate_a = branch_df["rate_a_true"].values
    valid_rate = (rate_a > 0.0) & (~np.isnan(rate_a))
    if mode == "ac":
        s_from = np.sqrt(branch_df["pf_pred"].values ** 2 + branch_df["qf_pred"].values ** 2)
        s_to = np.sqrt(branch_df["pt_pred"].values ** 2 + branch_df["qt_pred"].values ** 2)
    else:
        s_from = np.abs(branch_df["pf_dc"].values)
        s_to = np.abs(branch_df["pt_dc"].values)
    thermal_mag = np.zeros(len(branch_df))
    thermal_mag[valid_rate] = np.maximum(0.0, np.maximum(s_from, s_to)[valid_rate] - rate_a[valid_rate])
    per_type["thermal"] = (thermal_mag[valid_rate], branch_df[scen_br].values[valid_rate])

    # Branch angle-difference limits
    va_col = "Va_pred" if mode == "ac" else "Va_dc"
    if va_col in bus_df.columns:
        ang_diff = _angle_diff_per_branch(bus_df, branch_df, va_col)
        ang_min = branch_df["ang_min_true"].values
        ang_max = branch_df["ang_max_true"].values
        valid_ang = (~np.isnan(ang_min)) & (~np.isnan(ang_max)) & (ang_min > -359.0) & (ang_max < 359.0)
        ang_mag = np.zeros(len(branch_df))
        ang_mag[valid_ang] = (
            np.maximum(0.0, ang_min - ang_diff)[valid_ang]
            + np.maximum(0.0, ang_diff - ang_max)[valid_ang]
        )
        per_type["angle"] = (ang_mag[valid_ang], branch_df[scen_br].values[valid_ang])

    all_scenarios = set(bus_df[scen_bus].unique())
    n_scen = len(all_scenarios) or 1
    infeasible_any = set()
    out: Dict[str, float] = {}
    for t, (mag, sc) in per_type.items():
        viol = mag > tol
        n_total = len(mag)
        # Canonical quintet per constraint type. viol_mean is over ALL elements
        # (incl. non-violating), one-sided/two-sided per _mag construction above.
        out[f"{prefix}{t}_viol_count"] = int(viol.sum())
        out[f"{prefix}{t}_viol_mean"] = float(mag.mean()) if n_total else 0.0
        out[f"{prefix}{t}_viol_max"] = float(mag.max()) if n_total else 0.0
        out[f"{prefix}{t}_viol_rate"] = float(viol.sum() / n_total) if n_total else 0.0
        infeasible_sc = set(np.asarray(sc)[viol].tolist())
        out[f"{prefix}{t}_scenario_viol_rate"] = len(infeasible_sc) / n_scen
        infeasible_any |= infeasible_sc
    out[f"{prefix}fully_scenario_viol_rate"] = len(infeasible_any) / n_scen
    return out
