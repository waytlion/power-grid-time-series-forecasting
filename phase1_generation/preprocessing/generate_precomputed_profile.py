"""Single-script version of generate_precomputed_profile.ipynb with timing logs.

Generates precomputed load profile CSVs for gridfm-datakit from realistic load
profiles. The output CSV can be passed to datakit via the `precomputed_profile`
load generator in the config YAML.

Requirements:
    pip install matpowercaseframes pandas numpy requests
    pip install -e <path-to-gridfm-datakit-fork>   # for grid file lookup
"""

import os
from importlib import resources
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import requests
from matpowercaseframes import CaseFrames


# --- 1. Configuration ---
CASE_NAME = "case500_goc"  # Options: case14, case30, case57, case118, case300, case500_goc, case2383, case2746wp, etc.
LOAD_PROFILE_PATH = "D:/Data/studium/Master/MA_Code/data/updated_load_profiles/df_load_bus_2019-2021.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent  # Output alongside this script
OUTPUT_FILE = OUTPUT_DIR / f"{CASE_NAME.strip()}_3yr_precomputed_load_profiles.csv"
RANDOM_SEED = 42


def get_ieee_base_loads(case_name: str) -> pd.DataFrame:
    """Return base P (MW) and Q (MVar) for all buses of the selected case."""
    t0 = perf_counter()
    normalized_case = case_name.strip()
    if normalized_case.endswith(("_ieee", "_goc", "_api", "_sad")):
        grid_name = normalized_case
    else:
        grid_name = f"{normalized_case}_ieee"
    filename = f"pglib_opf_{grid_name}.m"

    pglib_path = Path(str(resources.files("gridfm_datakit.grids").joinpath(filename)))

    # Download once if missing, then parse directly with CaseFrames (no Julia dependency).
    if not pglib_path.is_file():
        url = f"https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master/{filename}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        pglib_path.parent.mkdir(parents=True, exist_ok=True)
        pglib_path.write_bytes(response.content)
        print(f"Downloaded {filename}")

    cf = CaseFrames(str(pglib_path))
    bus = cf.bus
    base_df = pd.DataFrame(
        {
            "bus_idx": np.arange(len(bus)),
            "P_base": bus["PD"].values,
            "Q_base": bus["QD"].values,
        }
    )
    print(f"[timing] Load base case (.m parse): {perf_counter() - t0:.3f}s")
    return base_df


def main() -> None:
    total_start = perf_counter()

    # --- 4. Load and Prepare Data ---
    t0 = perf_counter()
    ieee_base = get_ieee_base_loads(CASE_NAME)
    n_buses = int((ieee_base["P_base"] != 0).sum())  # count of actual load buses

    # Load or generate sample data
    if Path(LOAD_PROFILE_PATH).exists():
        df = pd.read_parquet(LOAD_PROFILE_PATH)
        print(f"Loaded parquet from {LOAD_PROFILE_PATH}")
    else:
        # Generate synthetic load profile data (1 year * 365 days * 24 hours = 8760 time steps)
        print(f"Warning: {LOAD_PROFILE_PATH} not found. Generating synthetic load data.")
        np.random.seed(RANDOM_SEED)
        n_buses_sample = max(100, n_buses)  # Ensure enough profiles for large PGLib cases
        n_timestamps = 8760  # 1 year of hourly data
        timestamps = pd.date_range("2019-01-01", periods=n_timestamps, freq="h")
        buses = np.arange(1, n_buses_sample + 1)

        # Vectorized synthetic load generation with daily + seasonal pattern and per-bus noise.
        t = np.arange(n_timestamps)
        daily = np.sin(2 * np.pi * (t % 24) / 24)
        seasonal = np.sin(2 * np.pi * (t / 24) / 365)
        base = 50 + 30 * daily + 20 * seasonal
        noise = np.random.normal(0, 5, size=(n_timestamps, n_buses_sample))
        load_matrix = np.maximum(base[:, None] + noise, 0)
        df = (
            pd.DataFrame(load_matrix, index=timestamps, columns=buses)
            .stack()
            .reset_index(name="load_corrected")
            .rename(columns={"level_0": "timestamp", "level_1": "bus"})
        )

    # Sample N_BUSES
    wide_df = df.pivot(index="timestamp", columns="bus", values="load_corrected").sort_index()
    sampled_df = wide_df.sample(n=n_buses, axis=1, random_state=RANDOM_SEED)

    # Normalize each sampled bus profile by its own max
    norm_profiles = (sampled_df / sampled_df.max()).values
    n_scenarios = len(norm_profiles)

    print(f"Time steps: {n_scenarios},\\n normalised Shape: {norm_profiles.shape}")
    print("Sampled profiles head:")
    print(sampled_df.head())
    print(f"[timing] Load and prepare data: {perf_counter() - t0:.3f}s")

    # --- 5. Generate Output ---
    t0 = perf_counter()
    ieee = ieee_base.sort_values("bus_idx")
    n_buses_case = len(ieee)
    load_mask = (ieee["P_base"] != 0).values

    p_mat = np.zeros((n_scenarios, n_buses_case))
    q_mat = np.zeros((n_scenarios, n_buses_case))
    p_mat[:, load_mask] = norm_profiles * ieee.loc[load_mask, "P_base"].values
    q_mat[:, load_mask] = norm_profiles * ieee.loc[load_mask, "Q_base"].values

    out_df = pd.DataFrame(
        {
            "load_scenario": np.repeat(np.arange(n_scenarios), n_buses_case),
            "load": np.tile(ieee["bus_idx"].values, n_scenarios),
            "p_mw": p_mat.flatten(),
            "q_mvar": q_mat.flatten(),
        }
    )

    out_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Output_df {len(out_df)} rows.")
    print(f"Expected rows: {n_scenarios * n_buses_case}")
    print("Output head:")
    print(out_df.head())
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"[timing] Generate and save output: {perf_counter() - t0:.3f}s")

    print(f"[timing] Total runtime: {perf_counter() - total_start:.3f}s")


if __name__ == "__main__":
    main()
