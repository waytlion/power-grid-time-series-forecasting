import pandas as pd

# Load the file
try:
    df = pd.read_parquet("df_load_buses.parquet")
    
    print("--- 1. DATA SHAPE (Rows, Columns) ---")
    print(df.shape)
    print("\n--- 2. COLUMN DATA TYPES ---")
    print(df.dtypes)
    
    print("\n--- 3. FIRST 50 ROWS ---")
    print(df.head(50))

except Exception as e:
    print(f"Error reading file: {e}")

#save file
df.head(10000).to_csv("df_load_buses.csv", index=False)