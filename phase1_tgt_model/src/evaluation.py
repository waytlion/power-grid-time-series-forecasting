import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from numpy.lib.stride_tricks import sliding_window_view

class RollingDataset(Dataset):
    def __init__(self, data, split_indices, input_window, forecast_horizon):
        """
        A Torch Dataset for rolling window forecasting.

        Args:
            data: Scaled Tensor [T, N, F]
            split_indices: Array of valid start times (e.g. train_idx)
            input_window: Int (e.g 168)
            forecast_horizon: Int (e.g 33)
        """
        self.data = data
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        
        # Create a boolean mask for the global timeline
        valid_mask = np.zeros(self.data.shape[0], dtype=bool)
        valid_mask[split_indices] = True
        # Apply boundary constraints
        valid_mask[:input_window] = False  # Start: Must be >= input_window 
        if forecast_horizon > 0: 
            valid_mask[-forecast_horizon:] = False # End: Must have space for 33h horizon
        self.valid_starts = np.where(valid_mask)[0] # [valid t indices]

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        t = self.valid_starts[idx] # t = time step where Prediction STARTS (Forecast Hour 0
        x = self.data[t - self.input_window : t] # Input X: History [t-168 ... t-1] 
        y = self.data[t : t + self.forecast_horizon] # Target Y: Future [t ... t+33-1] 
        
        return torch.from_numpy(x), torch.from_numpy(y)

def prepare_xgb_data(full_data_scaled, indices, input_window, forecast_horizon, step_size):
    """
    Args:
        preparation of data for XGBoost (Flattening 3D -> 2D).
    Returns:
        X: (N_samples, n_features) float32 array.
           Features: [Lag_1...Lag_W, TimeFeat_1...TimeFeat_K, Bus_ID]
        Y: (N_samples, forecast_horizon) float32 array.
    """
    T, N_buses, F = full_data_scaled.shape
    n_time_feats = F - 1
    valid_indices = [t for t in indices if t >= input_window and t <= T - forecast_horizon]
    sampled_indices = valid_indices[::step_size]  #*  Downsampling for RAM
    
    n_samples = len(sampled_indices) * N_buses
    # input_window lags + time features + 1 bus id
    n_features = input_window + n_time_feats + 1
    print(f"Allocating RAM for {n_samples} samples...")
    
    X = np.empty((n_samples, n_features), dtype=np.float32)
    Y = np.empty((n_samples, forecast_horizon), dtype=np.float32)
    row_idx = 0
    for t in tqdm(sampled_indices, desc="Building XGB Dataset"):
        time_feats = full_data_scaled[t, 0, 1:]
        
        for bus in range(N_buses):
            lags = full_data_scaled[t-input_window:t, bus, 0]
            
            X[row_idx, :input_window] = lags
            X[row_idx, input_window : input_window+n_time_feats] = time_feats
            X[row_idx, input_window+n_time_feats] = bus
            
            Y[row_idx, :] = full_data_scaled[t : t+forecast_horizon, bus, 0]
            
            row_idx += 1
            
    return X, Y



def get_scaling_factors(full_data, train_idx, scaler, m=24):
    """
    Berechnet den Fehler der Seasonal Naive Methode (Lag m) auf dem TRAINING-Set.
    Das ist die Basis (Nenner) für MASE und MSSE.
    """
    # Training Load isolieren & denormalisieren
    train_load_scaled = full_data[train_idx, :, 0]
    sigma, mu = np.sqrt(scaler.var_[0]), scaler.mean_[0]
    train_mw = train_load_scaled * sigma + mu
    
    # Seasonal Naive Error (In-Sample): Vergleiche t mit t-m
    # Wir flachklopfen alles für einen globalen Skalierungsfaktor
    y_true = train_mw[m:].flatten()
    y_naive = train_mw[:-m].flatten()
    
    denom_mae = np.mean(np.abs(y_true - y_naive))
    denom_mse = np.mean((y_true - y_naive)**2)
    
    return denom_mae, denom_mse

def run_evaluation(full_data, test_idx, xgb, tgt, sarima, mask, scaler, cfg, device='cpu'):
    results = {"true": [], "xgb": [], "snaive": [], "tgt": [], "sarima": []}
    
    T, N, F = full_data.shape
    win, hor = cfg["INPUT_WINDOW"], cfg["FORECAST_HORIZON"]
    n_time_feats = F - 1
    
    starts = [t for t in test_idx if t % 24 == cfg["EVAL_HOUR"] and t >= win and t <= T - hor]
    
    tgt.eval()
    sigma, mu = np.sqrt(scaler.var_[0]), scaler.mean_[0]
    
    print(f"Evaluating {len(starts)} days...")
    
    for t in tqdm(starts):
        # XGB Input
        xgb_in = []
        for b in range(N):
            xgb_in.append(np.concatenate([
                full_data[t-win:t, b, 0],
                full_data[t, b, 1:1+n_time_feats],
                [b]
            ]))
        X_xgb = np.array(xgb_in)
        # TGT Input
        X_tgt = torch.from_numpy(full_data[t-win:t]).unsqueeze(0).float().to(device)

        # 1. XGB Predict
        pred_xgb = xgb.predict(X_xgb)
        
        # 2. SNaive Predict
        lag = cfg["SNAIVE_LAG"]
        pred_naive = full_data[t-lag : t-lag+hor, :, 0].T
        
        # 3. SARIMA Predict (Using Pre-fitted, Rolling Apply)
        pred_sarima = []
        for b in range(N):
            # We pass SCALED history to match the global fit domain
            hist_scaled = full_data[t-win:t, b, 0]
            # Predict
            fc_scaled = sarima.predict(hist_scaled, bus_idx=b, horizon=hor)
            # Denormalize output
            pred_sarima.append(fc_scaled * sigma + mu)
            
        pred_sarima = np.array(pred_sarima)
        
        with torch.no_grad():
            pred_tgt = tgt(X_tgt, mask).cpu().numpy().squeeze(0).T

        # Inverse Scale & Slice Target Day (9-33)
        s, e = 9, 33
        Y_true = full_data[t:t+hor, :, 0].T
        
        results["xgb"].append((pred_xgb[:, s:e] * sigma) + mu)
        results["snaive"].append((pred_naive[:, s:e] * sigma) + mu)
        results["tgt"].append((pred_tgt[:, s:e] * sigma) + mu)
        results["sarima"].append(pred_sarima[:, s:e])
        results["true"].append((Y_true[:, s:e] * sigma) + mu)

    return {k: np.array(v) for k, v in results.items()}

def print_metrics(res, scale_mae, scale_mse):
    """calc RMSE, MAE, MAPE, MASE, MSSE."""
    truth = res["true"].flatten()
    
    print("\n=== FINAL BENCHMARK RESULTS ===")
    header = f"{'Model':<10} | {'RMSE':<10} | {'MAE':<10} | {'MAPE':<10} | {'MASE':<10} | {'MSSE':<10}"
    print(header)
    print("-" * len(header))
    
    metrics_store = {}
    
    for model_name in ["snaive", "xgb", "sarima", "tgt"]:
        if len(res[model_name]) == 0: continue
        pred = res[model_name].flatten()
        
        rmse = np.sqrt(mean_squared_error(truth, pred))
        mae = mean_absolute_error(truth, pred)
        mask = np.abs(truth) > 0.05  #! MAPE calc only for true vals != 0
        if mask.sum() > 0: 
            mape = np.mean(np.abs((truth[mask] - pred[mask]) / truth[mask])) * 100.0
        else:
            mape = float('nan')
        
        # Scaled (Metrik / Training_Naive_Error)
        mase = mae / scale_mae
        msse = (rmse**2) / scale_mse
        
        metrics_store[model_name] = rmse
        
        name_display = "TinyTGT" if model_name == "tgt" else model_name.upper()
        print(f"{name_display:<10} | {rmse:.4f}     | {mae:.4f}     | {mape:.2f}%     | {mase:.4f}     | {msse:.4f}")
    
    print("-" * len(header))
