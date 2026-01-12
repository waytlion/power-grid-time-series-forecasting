#* Load real load profiles -> create aggregated load profile
# -> can be used for data gen via gridfm-datakit

import pandas as pd
import numpy as np

# 1. Load data
try:
    df = pd.read_parquet(".\phase1_generation\preprocessing\df_load_buses.parquet")
except Exception as e:
    print(f"Error loading parquet: {e}")
    exit()

# 2. Agg 
# Group by time and sum the load col
system_profile = df.groupby("timestamp")["load"].sum()

# 3. Normalization
# Scale - peak is 1.0
peak_value = system_profile.max()
normalized_profile = system_profile / peak_value

# 4. Save to CSV
output_filename = "thesis_profile.csv"
normalized_profile.to_csv(output_filename, index=False, header=False)

print(f"Saved normalized profile to '{output_filename}'")
print(f"File has {len(normalized_profile)} time steps.")
print(f"Format: Single column, no header.")
print(f"Normalized Peak: {normalized_profile.max():.2f}")
print("\n--- 🔍 Time Order Verification ---")
print(f"Start Time: {system_profile.index[0]}")
print(f"End Time:   {system_profile.index[-1]}")
