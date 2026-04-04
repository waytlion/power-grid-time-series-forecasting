"""
Converts predicted loads (from baseline metrics) into precomputed load profiles
suitable for gridfm-datakit AC-OPF evaluation.

Usage:
    python scripts/transform_forecasts.py \
        --case 118 \
        --input-parquet phase1_baseline/case118_ieee_horizon1.parquet \
        --out-dir exp1/data/precomputed_profiles/case118_ieee_horizon1

This script derives the reactive power (Q) based on the specific load buses' P/Q
ratio in the matpower case, and creates a separate CSV for each forecasting model
found in the parquet file.
"""

import argparse
import os
import sys
from pathlib import Path
from importlib import resources

import numpy as np
import pandas as pd
from matpowercaseframes import CaseFrames

def get_ieee_base(case_n: int):
    """Return (P_base, Q_base) arrays for `pglib_opf_case{case_n}_...m`."""
    # Handle naming conventions
    case_str = str(case_n).strip()
    if case_str.endswith(("_ieee", "_goc", "_api", "_sad")):
        filename = f"pglib_opf_case{case_str}.m"
    else:
        filename = f"pglib_opf_case{case_str}_ieee.m"
        
    m_path = Path(str(resources.files("gridfm_datakit.grids").joinpath(filename)))
    
    if not m_path.is_file():
        # Fallback to download (similar to phase1 generation)
        import requests
        url = f"https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master/{filename}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        m_path.parent.mkdir(parents=True, exist_ok=True)
        m_path.write_bytes(response.content)
        print(f"Downloaded {filename}")
        
    cf = CaseFrames(str(m_path))
    P_base = cf.bus["PD"].to_numpy(dtype=float)
    Q_base = cf.bus["QD"].to_numpy(dtype=float)
    return P_base, Q_base


def main():
    parser = argparse.ArgumentParser(description="Transform forecast output into DataKit inputs.")
    parser.add_argument("--case", type=str, required=True, help="IEEE case name (e.g., 118 or case118_ieee)")
    parser.add_argument("--input-parquet", type=Path, required=True, help="Input parquet containing forecasts")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to save precomputed CSVs")
    parser.add_argument("--id-col", type=str, default="load_scenario_idx", help="Name of the scenario ID column")
    
    args = parser.parse_args()

    # Load Parquet
    if not args.input_parquet.exists():
        raise FileNotFoundError(f"Input parquet not found: {args.input_parquet}")

    df = pd.read_parquet(args.input_parquet)

    id_col = args.id_col
    if id_col not in df.columns:
        raise ValueError(f"Missing required id column: {id_col}")

    # Ensure horizon dimension is explicit for stable flattening
    if "horizon_step" in df.columns:
        df["_horizon_step"] = pd.to_numeric(df["horizon_step"], errors="raise").astype(int)
    else:
        df["_horizon_step"] = 0

    # Get base P/Q and compute Q/P ratio
    P_base, Q_base = get_ieee_base(args.case)
    P_base = np.asarray(P_base, dtype=float)
    Q_base = np.asarray(Q_base, dtype=float)

    ratio = np.zeros_like(P_base, dtype=float)
    mask = P_base != 0.0
    ratio[mask] = Q_base[mask] / P_base[mask]

    # DataKit expects load_scenario to be contiguous 0..N-1
    # Flatten by (original scenario idx, horizon step)
    df[id_col] = pd.to_numeric(df[id_col], errors="raise").astype(int)
    rows_before = len(df)
    df["_load_scenario"] = df.groupby([id_col, "_horizon_step"], sort=True).ngroup()
    if len(df) != rows_before:
        raise RuntimeError("Unexpected row-count change while building flattened scenarios")

    # Ensure `load` is the 0-based bus index used by the Matpower case
    bus_raw = pd.to_numeric(df["bus_id"], errors="raise").astype(int)
    if bus_raw.min() == 1 and bus_raw.max() == len(P_base):
        df["_load_idx"] = bus_raw - 1
    else:
        df["_load_idx"] = bus_raw

    # Identify models
    exclude = {id_col, "bus_id", "horizon_step", "_horizon_step", "_load_scenario", "_load_idx"}
    model_cols = [c for c in df.columns if c not in exclude]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    load_idx = pd.to_numeric(df["_load_idx"], errors="raise").to_numpy(dtype=int)
    valid_mask = (load_idx >= 0) & (load_idx < len(ratio))
    if not valid_mask.all():
        bad = int((~valid_mask).sum())
        raise ValueError(f"Found {bad} invalid load indices outside 0..{len(ratio)-1}")

    n_buses = len(ratio)
    n_flat_scenarios = int(df["_load_scenario"].nunique())
    expected_rows = n_buses * n_flat_scenarios

    if int(df["_load_scenario"].min()) != 0 or int(df["_load_scenario"].max()) != n_flat_scenarios - 1:
        raise ValueError("Flattened load_scenario is not contiguous 0..S-1")

    print(f"Diagnostics: rows={len(df)}, orig_scenarios={df[id_col].nunique()}, horizons={df['_horizon_step'].nunique()}, buses={n_buses}")
    print(f"Flattened scenario count: {n_flat_scenarios}")
    print(f"Expected rows check: {n_buses} x {n_flat_scenarios} = {expected_rows}; actual rows={len(df)}")

    for col in model_cols:
        p_arr = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        q_arr = p_arr * ratio[load_idx]
        out = pd.DataFrame({
            "load_scenario": df["_load_scenario"].astype(int),
            "load_scenario_idx": df[id_col].astype(int),
            "horizon_step": df["_horizon_step"].astype(int),
            "load": df["_load_idx"].astype(int),
            "p_mw": p_arr,
            "q_mvar": q_arr,
        })

        dup_count = int(out.duplicated(subset=["load_scenario", "load"]).sum())
        if dup_count > 0:
            raise ValueError(f"{col}: detected {dup_count} duplicates")

        if len(out) != expected_rows:
            raise ValueError(f"{col}: expected {expected_rows} rows, got {len(out)}")

        out_path = args.out_dir / f"{col}.csv"
        out.to_csv(out_path, index=False)
        print(f"Wrote {out_path} | scenarios={n_flat_scenarios} | rows={len(out)}")


if __name__ == "__main__":
    main()
