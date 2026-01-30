import torch
import torch.nn as nn
import math
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm

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
    def __init__(self, n_nodes, d_model=64, n_heads=4, n_temporal_layers=2, in_feat=7, out_steps=33):
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
    def __init__(self, order=(2, 0, 0), seasonal_order=(1, 0, 1, 24)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.bus_params = {} # Stores the fitted result wrapper per bus
        self.failure_count = 0 
        self.call_count = 0

    def fit(self, full_data, train_indices, max_fit_hours=None):
        """
        max_fit_hours: If None, use all training data. 
                    If set (e.g., 672), limit to that many hours 
                    -> This is implemented for interleaved splitting and
                    should be removed once we dont need interleaved splitting anymore.
        """
        if max_fit_hours is None:
            # Use all training indices (for temporal split)
            train_block_idx = train_indices
        else:
            # Use limited contiguous block (for interleaved split)
            train_limit = max_fit_hours
            valid_train_range = train_indices[train_indices < train_limit]
            train_block_idx = valid_train_range if len(valid_train_range) > 0 else train_indices[:train_limit]
        
        dataset_subset = full_data[train_block_idx, :, 0]
        dataset_subset = full_data[train_block_idx, :, 0] # [T_sub, N_buses]
        
        for b in tqdm(range(full_data.shape[1]), desc="Fitting Bus Models"):
            try:
                # Fit on the training block
                model = SARIMAX(
                    dataset_subset[:, b],
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                # 'powell' is robust; disp=False hides logs
                self.bus_params[b] = model.fit(disp=False, method='powell')
            except Exception as e:
                print(f" Fit failed for Bus {b}: {e}")
                self.bus_params[b] = None

    def predict(self, history, bus_idx, horizon=33):
        """
        Uses pre-learned parameters to filter the new 'history' window and forecast.
        """
        self.call_count += 1
        model_res = self.bus_params.get(bus_idx)
        
        if model_res is None:
            self.failure_count += 1
            return self._naive_fallback(history, horizon)
        
        try:
            # .apply() updates the state (Kalman Filter) with the new observation window
            # using the frozen parameters from training.
            new_res = model_res.apply(history)
            return new_res.forecast(steps=horizon)
        except:
            self.failure_count += 1
            return self._naive_fallback(history, horizon)

    def _naive_fallback(self, history, horizon):
        return np.resize(history[-24:], horizon)
