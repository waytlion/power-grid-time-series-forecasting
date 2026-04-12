# This script tests, if the power residual computation, done in Thesis_Repo/Exp1/generate_metrics is done correctly.
# correctly means, that the power residuals are 0, if we compute them based on true data vs true data.

import sys
from pathlib import Path

repo_root = Path("/home/tibo990i/Thesis_Repo")
sys.path.insert(0, str(repo_root / "exp1" / "generate_metrics"))

from loaders import load_datakit_bus, load_datakit_branch
from metrics import compute_algebraic_power_residuals
import pandas as pd

gt_dir = Path("/data/horse/ws/tibo990i-thesis_data/data_out/3yr_2019-2021/phase_1a/data_out/case118_ieee/raw")

print("Loading Ground Truth Data...")
true_bus = load_datakit_bus(gt_dir)
true_branch = load_datakit_branch(gt_dir)

# Duplicate Truth into the format 'compute_algebraic_power_residuals' expects
bus_aligned = true_bus.copy()
bus_aligned = bus_aligned.rename(columns={
    "Vm": "Vm_pred",
    "Va": "Va_pred",
    "Pg": "Pg_pred",
    "Qg": "Qg_pred",
    "Pd": "Pd_true",
    "Qd": "Qd_true",
    "GS": "GS_true",
    "BS": "BS_true",
})
bus_aligned["pred_flat_idx"] = bus_aligned["load_scenario_idx"]

print("Computing Algebraic Residuals...")
res = compute_algebraic_power_residuals(bus_aligned, true_branch)

print(f"\nRESULTS:")
print(f"Absolute Residual P (MAE): {res['mae_residual_p_mw']:.6f} MW")
print(f"Absolute Residual Q (MAE): {res['mae_residual_q_mvar']:.6f} MVAR")
