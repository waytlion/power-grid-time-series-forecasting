import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def generate_cyclical_features(T, start_date, frequency):
    """
    Generates global temporal features 
    Returns: matrix: [T, 6].
    """
    dates = pd.date_range(start=start_date, periods=T, freq=frequency)
    df_time = pd.DataFrame({'date': dates})
    hour = df_time['date'].dt.hour
    dow  = df_time['date'].dt.dayofweek
    doy  = df_time['date'].dt.dayofyear
    
    # Sin/Cos Encoding 
    df_time['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df_time['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)   
    df_time['dow_sin'] = np.sin(2 * np.pi * dow / 7.0)
    df_time['dow_cos'] = np.cos(2 * np.pi * dow / 7.0)
    df_time['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
    df_time['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
    
    feature_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos']
    return df_time[feature_cols].values.astype(np.float32)

def scale_data_selectively(full_data, train_indices):
    """
    StandardScaler scales feature 0 (active Power P) in whole dataset
    """
    scaler = StandardScaler()
    scaler.fit(full_data[train_indices, :, 0].reshape(-1, 1)) # Fit on feature 0 using train time steps
    full_data_scaled = full_data.copy()

    full_data_scaled[:, :, 0] = scaler.transform(
        full_data[:, :, 0].reshape(-1, 1)
    ).reshape(full_data.shape[0], full_data.shape[1])

    return full_data_scaled.astype(np.float32), scaler
