import argparse
import gc
import logging
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
    "BATCH_SIZE": 64,
    "EPOCHS": 10,
    "INPUT_WINDOW": 336,
    "FORECAST_HORIZON": 1,
    "EVAL_HOUR": None,
    "SNAIVE_LAG": 48,
}

LOGGER = logging.getLogger(__name__)


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
    parser.add_argument("--skip-tgt", action="store_true", help="Skip TinyTGT training")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
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
        LOGGER.info("Seed set to %s", args.seed)

    LOGGER.info("Device: %s", device)

    data_source = "cli" if args.data_path is not None else "default"
    resolved_data_path = Path(config["DATA_PATH"]).expanduser().resolve()
    LOGGER.info("Using DATA_PATH (source=%s): %s", data_source, resolved_data_path)

    data_path = resolved_data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    bus_df = pd.read_parquet(data_path / "bus_data.parquet")
    branch_df = pd.read_parquet(data_path / "branch_data.parquet")

    structure = branch_df[branch_df["load_scenario_idx"] == 0].sort_values("idx")
    edge_index = list(structure[["from_bus", "to_bus"]].itertuples(index=False, name=None))

    m = len(edge_index)
    n = bus_df["bus"].nunique()
    LOGGER.info("Topology loaded: n=%s nodes, m=%s edges", n, m)

    loads_pivot = bus_df.pivot(index="load_scenario_idx", columns="bus", values="Pd").fillna(0)
    flows_pivot = branch_df.pivot(index="load_scenario_idx", columns="idx", values="pf").fillna(0)

    loads_matrix = loads_pivot.values.astype(np.float32)
    flows_matrix = flows_pivot.values.astype(np.float32)
    LOGGER.info("Matrix shapes: loads=%s, flows=%s", loads_matrix.shape, flows_matrix.shape)

    if config["USE_SUBSET"]:
        subset_size = int(len(loads_matrix) * config["SUBSET_PERCENT"])
        loads_matrix = loads_matrix[:subset_size]
        flows_matrix = flows_matrix[:subset_size]
        LOGGER.info("Using subset: %s samples (%.1f%%)", subset_size, config["SUBSET_PERCENT"] * 100)

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
    LOGGER.info("Time features generated with shape=%s", time_features_global.shape)

    load_tensor = loads_matrix[:, :, np.newaxis]
    time_tensor_expanded = np.broadcast_to(
        time_features_global[:, np.newaxis, :],
        (time_features_global.shape[0], loads_matrix.shape[1], time_features_global.shape[1]),
    )
    X = np.concatenate([load_tensor, time_tensor_expanded], axis=2)
    LOGGER.info("Final input tensor shape=%s (time, buses, features)", X.shape)

    train_idx, val_idx, test_idx = get_temporal_splits(T)
    LOGGER.info("Split sizes: train=%s, val=%s, test=%s", len(train_idx), len(val_idx), len(test_idx))

    scaled_tensor, scaler = scale_data_selectively(X, train_idx)
    LOGGER.info("Scaled mean (load)=%.4f", scaled_tensor[:, :, 0].mean())
    LOGGER.info(
        "Scaled mean (hour_sin)=%.4f (expected near 0 but unscaled)",
        scaled_tensor[:, :, 1].mean(),
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
    LOGGER.info("Batch shapes: X=%s, Y=%s", x_batch.shape, y_batch.shape)

    X_train_xgb, Y_train_xgb = prepare_xgb_data(
        scaled_tensor,
        train_idx,
        config["INPUT_WINDOW"],
        config["FORECAST_HORIZON"],
        step_size=1,
    )
    LOGGER.info("XGB training data shape=%s", X_train_xgb.shape)

    xgb_estimator = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,              # Randomly sample 80% of data per tree (prevents overfitting)
        colsample_bytree=0.8,       # Randomly sample 80% of features per tree
        objective="reg:squarederror",
        device=xgb_device,
        tree_method="hist",
        n_jobs=32,
        random_state=args.seed,
    )

    xgb_model = MultiOutputRegressor(xgb_estimator, n_jobs=32)

    LOGGER.info("Training global XGBoost model")
    xgb_model.fit(X_train_xgb, Y_train_xgb)
    LOGGER.info("XGBoost training complete")

    torch.cuda.empty_cache()

    tgt_model = None
    if not args.skip_tgt:
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

            LOGGER.info(
                "Epoch %s: train_mse=%.6f | val_mse=%.6f",
                ep,
                train_loss / batch_count,
                val_loss / val_batches,
            )
    else:
        LOGGER.info("Skipping TinyTGT training (--skip-tgt flag set)")

    sarima_model = GlobalFitLocalApplySARIMA(order=(2, 0, 0), seasonal_order=(1, 0, 1, 24))
    sarima_model.fit(scaled_tensor.astype(np.float32), train_idx, None)

    denom_mae, denom_mse = get_scaling_factors(scaled_tensor, train_idx, scaler, m=24)
    LOGGER.info("Scaling baseline (train): MAE=%.4f, MSE=%.4f", denom_mae, denom_mse)

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
                }
                if "tgt" in results:
                    row["tgt"] = results["tgt"][eval_idx, bus_id, h]
                row["sarima"] = results["sarima"][eval_idx, bus_id, h]
                data_list.append(row)

    df_results = pd.DataFrame(data_list)
    df_results.to_parquet(args.output_path, index=False, compression="snappy")
    LOGGER.info("Results saved to %s", args.output_path)
    LOGGER.info("Result shape=%s", df_results.shape)
    LOGGER.info("Preview:\n%s", df_results.head().to_string(index=False))

    print_metrics(results, denom_mae, denom_mse)

    gc.collect()


if __name__ == "__main__":
    main()
