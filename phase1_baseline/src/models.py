import logging
import torch
import torch.nn as nn
import math
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from joblib import Parallel, delayed
import os

LOGGER = logging.getLogger(__name__)

def _fit_single_bus(b, bus_data, order, seasonal_order):
    try:
        model = SARIMAX(
            bus_data,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted_model = model.fit(disp=False, method='powell')
        # Store ONLY the fitted parameter values (~5 floats), NOT the full
        # SARIMAXResults object which stores T × state_dim² Kalman matrices
        # (~200-300 MB per bus in float64). At inference time we reconstruct
        # a fresh model from params + the current history window.
        return b, fitted_model.params.copy()
    except Exception as e:
        LOGGER.warning("Fit failed for bus %s: %s", b, e)
        return b, None
    
class SNaiveModel:
    def __init__(self, lag=48, forecast_horizon=33):
        self.lag = lag
        self.forecast_horizon = forecast_horizon

    def predict(self, history):
        if len(history) < self.lag:
            raise ValueError("History shorter than lag!")
        return history[-self.lag : -self.lag + self.forecast_horizon]

# --- TGT Models ---
class SpatialAttention(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__(); self.h=n_heads; self.dk=d_model//n_heads
        self.q=nn.Linear(d_model,d_model); self.k=nn.Linear(d_model,d_model); self.v=nn.Linear(d_model,d_model); self.o=nn.Linear(d_model,d_model)
    def forward(self, x, mask):
        B,N,D = x.shape; H=self.h; dk=self.dk
        q=self.q(x).view(B,N,H,dk).transpose(1,2)
        k=self.k(x).view(B,N,H,dk).transpose(1,2)
        v=self.v(x).view(B,N,H,dk).transpose(1,2)
        s=(q@k.transpose(-2,-1))/math.sqrt(dk)
        m = mask.unsqueeze(0).unsqueeze(0).expand(B,H,N,N)
        s = s.masked_fill(~m, float('-inf'))
        a = torch.softmax(s, dim=-1)
        out = a@v
        out = out.transpose(1,2).reshape(B,N,D)
        return self.o(out)

class TemporalBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff  = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))
        self.ln2 = nn.LayerNorm(d_model)
    def forward(self, x):
        y,_ = self.attn(x,x,x); x=self.ln1(x+y)
        y = self.ff(x); x=self.ln2(x+y); return x

class TinyTGT(nn.Module):
    def __init__(self, n_nodes, d_model=64, n_heads=4, n_temporal_layers=2, in_feat=7, out_steps=1):
        super().__init__()
        self.n_nodes = n_nodes
        
        self.enc = nn.Linear(in_feat, d_model)
        
        self.spat = SpatialAttention(d_model, n_heads)
        self.temporal = nn.ModuleList([TemporalBlock(d_model, n_heads) for _ in range(n_temporal_layers)])
        
        self.node_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_steps) 
        )

    def forward(self, x, mask):
        B, T, N, F = x.shape
        x_emb = self.enc(x.reshape(B*T*N, F)).view(B, T, N, -1)
        x_spat = self.spat(x_emb.view(B*T, N, -1), mask).view(B, T, N, -1)
        x_temp = x_spat.permute(0,2,1,3).contiguous().view(B*N, T, -1)
        
        for layer in self.temporal:
            x_temp = layer(x_temp)
            
        hT = x_temp[:, -1, :] 
        out = self.node_decoder(hT) 
        return out.view(B, N, -1).transpose(1, 2)

# --- SARIMA ---
class GlobalFitLocalApplySARIMA:
    def __init__(self, order=(2, 0, 0), seasonal_order=(1, 0, 1, 24), n_jobs=-1):
        self.order = order
        self.seasonal_order = seasonal_order
        self.n_jobs = n_jobs
        self.bus_params = {} # Stores the fitted result wrapper per bus
        self.failure_count = 0 
        self.call_count = 0

    def fit(self, full_data, train_indices, max_fit_hours=None):
        """
        Fit one SARIMAX model per bus using a window of training data.

        max_fit_hours: If None, use all training data.
                       If set (e.g. 17_520 = 2 years), use the LAST max_fit_hours
                       timesteps of train_indices so that the model captures the
                       most recent seasonal patterns rather than very old data.
        """
        if max_fit_hours is None:
            train_block_idx = train_indices
        else:
            # Take the tail of the sorted training indices (most recent data)
            sorted_idx = np.sort(train_indices)
            train_block_idx = sorted_idx[-max_fit_hours:]
            LOGGER.info(
                "SARIMA fit capped at %s hours: using indices [%s … %s]",
                len(train_block_idx),
                train_block_idx[0],
                train_block_idx[-1],
            )
        
        dataset_subset = full_data[train_block_idx, :, 0]  # [T_sub, N_buses]
        
        # joblib's 'verbose=10' acts as built-in progress bar (similar to tqdm)
        LOGGER.info("Firing up parallel SARIMAX fitting on %s buses", full_data.shape[1])
        
        results = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(_fit_single_bus)(
                b, 
                dataset_subset[:, b], 
                self.order, 
                self.seasonal_order
            ) for b in range(full_data.shape[1])
        )

        # Reconstruct  bus_params dictionary/list from the parallel results
        for b, fitted_model in results:
            self.bus_params[b] = fitted_model

    def predict(self, history, bus_idx, horizon=33):
        """
        Reconstruct a SARIMAX model from the frozen params and forecast.
        `bus_params[bus_idx]` is now a small params array, not a Results object.
        """
        self.call_count += 1
        params = self.bus_params.get(bus_idx)

        if params is None:
            self.failure_count += 1
            return self._naive_fallback(history, horizon)

        try:
            model = SARIMAX(
                history,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.filter(params)
            return res.forecast(steps=horizon)
        except Exception:
            self.failure_count += 1
            return self._naive_fallback(history, horizon)

    def _naive_fallback(self, history, horizon):
        return np.resize(history[-24:], horizon)
