import argparse
import gc
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from src.data_processing import generate_cyclical_features, scale_data_selectively
from src.evaluation import (
    RollingDataset,
    get_scaling_factors,
    prepare_xgb_data,
    print_metrics,
    run_evaluation,
)
from src.models import GlobalFitLocalApplySARIMA, TinyTGT
from src.splitting import get_temporal_splits

CONFIG = {
    "START_DATE": "2019-01-01",
    "FREQUENCY": "h",
    "DATA_PATH": "../../data/data_out/Three_Years_2019-2021/case118_ieee/raw",
    "USE_SUBSET": False,
    "SUBSET_PERCENT": 0.01,
    "USE_ONLY_LOAD": True,
    "BINARY_ADJACENCY": True,
    "BATCH_SIZE": 32,
    "EPOCHS": 10,
    "INPUT_WINDOW": 168,
    "FORECAST_HORIZON": 1,
    "EVAL_HOUR": None,
    "SNAIVE_LAG": 48,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run temporal benchmark.")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--use-subset", action="store_true")
    parser.add_argument("--subset-percent", type=float, default=CONFIG["SUBSET_PERCENT"])
    parser.add_argument("--epochs", type=int, default=CONFIG["EPOCHS"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["BATCH_SIZE"])
    parser.add_argument("--input-window", type=int, default=CONFIG["INPUT_WINDOW"])
    parser.add_argument("--forecast-horizon", type=int, default=CONFIG["FORECAST_HORIZON"])
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--xgb-device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--output-path", default="benchmark_results.parquet")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = CONFIG.copy()
    if args.data_path is not None:
        config["DATA_PATH"] = args.data_path
    config["USE_SUBSET"] = bool(args.use_subset)
    config["SUBSET_PERCENT"] = float(args.subset_percent)
    config["EPOCHS"] = int(args.epochs)
    config["BATCH_SIZE"] = int(args.batch_size)
    config["INPUT_WINDOW"] = int(args.input_window)
    config["FORECAST_HORIZON"] = int(args.forecast_horizon)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.xgb_device:
        xgb_device = args.xgb_device
    else:
        xgb_device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.seed is not None:
        set_seed(args.seed)
        print(f"Seed set to {args.seed}")

    print("Device:", device)

    data_source = "cli" if args.data_path is not None else "default"
    resolved_data_path = Path(config["DATA_PATH"]).expanduser().resolve()
    print(f"Using DATA_PATH (source={data_source}): {resolved_data_path}")

    data_path = resolved_data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    bus_df = pd.read_parquet(data_path / "bus_data.parquet")
    branch_df = pd.read_parquet(data_path / "branch_data.parquet")

    structure = branch_df[branch_df["load_scenario_idx"] == 0].sort_values("idx")
    edge_index = list(structure[["from_bus", "to_bus"]].itertuples(index=False, name=None))

    m = len(edge_index)
    n = bus_df["bus"].nunique()
    print(f"Topology Loaded: n={n} Nodes, m={m} Edges")

    loads_pivot = bus_df.pivot(index="load_scenario_idx", columns="bus", values="Pd").fillna(0)
    flows_pivot = branch_df.pivot(index="load_scenario_idx", columns="idx", values="pf").fillna(0)

    loads_matrix = loads_pivot.values.astype(np.float32)
    flows_matrix = flows_pivot.values.astype(np.float32)
    print(f"shapes: Loads: {loads_matrix.shape}, Flows: {flows_matrix.shape}")

    if config["USE_SUBSET"]:
        subset_size = int(len(loads_matrix) * config["SUBSET_PERCENT"])
        loads_matrix = loads_matrix[:subset_size]
        flows_matrix = flows_matrix[:subset_size]
        print(f"Using {subset_size} samples ({config['SUBSET_PERCENT']*100:.1f}%)")

    A = np.zeros((n, n), dtype=bool)
    for i, j in edge_index:
        A[i, j] = True
        A[j, i] = True
    for i in range(n):
        A[i, i] = True
    A_mask = torch.from_numpy(A).to(device)

    T = loads_matrix.shape[0]
    time_features_global = generate_cyclical_features(
        T, config["START_DATE"], config["FREQUENCY"]
    )
    print(f"Time Features generated. Shape: {time_features_global.shape}")

    load_tensor = loads_matrix[:, :, np.newaxis]
    time_tensor_expanded = np.broadcast_to(
        time_features_global[:, np.newaxis, :],
        (time_features_global.shape[0], loads_matrix.shape[1], time_features_global.shape[1]),
    )
    X = np.concatenate([load_tensor, time_tensor_expanded], axis=2)
    print(f"Final Input Tensor Shape: {X.shape} (Time, Buses, Features)")

    train_idx, val_idx, test_idx = get_temporal_splits(T)
    print(f"Train Hours: {len(train_idx)}")
    print(f"Val Hours:   {len(val_idx)}")
    print(f"Test Hours:  {len(test_idx)}")

    scaled_tensor, scaler = scale_data_selectively(X, train_idx)
    print(f"Scaled Mean (Load): {scaled_tensor[:,:,0].mean():.4f}")
    print(
        f"Scaled Mean (Hour_Sin): {scaled_tensor[:,:,1].mean():.4f} (Should be near 0 but unscaled)"
    )

    train_dataset = RollingDataset(
        scaled_tensor, train_idx, config["INPUT_WINDOW"], config["FORECAST_HORIZON"]
    )
    val_dataset = RollingDataset(
        scaled_tensor, val_idx, config["INPUT_WINDOW"], config["FORECAST_HORIZON"]
    )

    generator = None
    if args.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        generator=generator,
    )

    x_batch, y_batch = next(iter(train_loader))
    print("Batch Shapes:")
    print(
        f"X (Input):  {x_batch.shape}  -> [Batch, {config['INPUT_WINDOW']}, 14, 7]"
    )
    print(
        f"Y (Target): {y_batch.shape}  -> [Batch, {config['FORECAST_HORIZON']}, 14, 7]"
    )

    X_train_xgb, Y_train_xgb = prepare_xgb_data(
        scaled_tensor,
        train_idx,
        config["INPUT_WINDOW"],
        config["FORECAST_HORIZON"],
        step_size=1,
    )
    print(f"XGB Train Shape: {X_train_xgb.shape}")

    xgb_estimator = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        objective="reg:squarederror",
        device=xgb_device,
        tree_method="hist",
        n_jobs=4,
        random_state=args.seed,
    )

    xgb_model = MultiOutputRegressor(xgb_estimator, n_jobs=1)

    print("Training Global XGBoost...")
    xgb_model.fit(X_train_xgb, Y_train_xgb)
    print("XGBoost Training Complete.")

    torch.cuda.empty_cache()

    tgt_model = TinyTGT(
        n_nodes=14,
        d_model=64,
        n_heads=4,
        in_feat=7,
        out_steps=config["FORECAST_HORIZON"],
    ).to(device)
    opt = optim.Adam(tgt_model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for ep in range(1, config["EPOCHS"] + 1):
        tgt_model.train()
        train_loss, batch_count = 0.0, 0

        for xb, yb in train_loader:
            xb = xb.float().to(device)
            yb = yb.float().to(device)

            opt.zero_grad()
            yhat = tgt_model(xb, A_mask)
            loss = loss_fn(yhat, yb[..., 0])
            loss.backward()
            opt.step()

            train_loss += loss.item()
            batch_count += 1

        tgt_model.eval()
        val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.float().to(device)
                y_true = yb[..., 0].float().to(device)
                yhat = tgt_model(xb, A_mask)
                val_loss += loss_fn(yhat, y_true).item()
                val_batches += 1

        print(
            f"Epoch {ep}: Train MSE={train_loss/batch_count:.6f} | Val MSE={val_loss/val_batches:.6f}"
        )

    sarima_model = GlobalFitLocalApplySARIMA(order=(2, 0, 0), seasonal_order=(1, 0, 1, 24))
    sarima_model.fit(scaled_tensor.astype(np.float32), train_idx, None)

    denom_mae, denom_mse = get_scaling_factors(scaled_tensor, train_idx, scaler, m=24)
    print(f"Scaling Baseline (Train): MAE={denom_mae:.4f}, MSE={denom_mse:.4f}")

    results, starts = run_evaluation(
        scaled_tensor,
        test_idx,
        xgb_model,
        tgt_model,
        sarima_model,
        A_mask,
        scaler,
        config,
        device=device,
    )

    print_metrics(results, denom_mae, denom_mse)

    n_evals, n_buses, n_horizon = results["true"].shape
    scenario_indices = loads_pivot.index.values

    data_list = []
    for eval_idx, t in enumerate(starts):
        scenario_id = scenario_indices[t]
        for bus_id in range(n_buses):
            for h in range(n_horizon):
                row = {
                    "load_scenario_idx": scenario_id,
                    "bus_id": bus_id,
                    "horizon_step": h,
                    "true": results["true"][eval_idx, bus_id, h],
                    "xgb": results["xgb"][eval_idx, bus_id, h],
                    "snaive": results["snaive"][eval_idx, bus_id, h],
                    "tgt": results["tgt"][eval_idx, bus_id, h],
                    "sarima": results["sarima"][eval_idx, bus_id, h],
                }
                data_list.append(row)

    df_results = pd.DataFrame(data_list)
    df_results.to_parquet(args.output_path, index=False, compression="snappy")
    print(f"Results saved to {args.output_path}")
    print(f"Shape: {df_results.shape}")
    print(df_results.head())

    gc.collect()


if __name__ == "__main__":
    main()
