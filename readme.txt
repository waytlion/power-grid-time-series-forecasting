# WORKFLOW
1. copy datakit output data to bernard cluster
        -> SOURCE: from local / cluster Leipzig
        -> DESTINATION: /data/horse/ws/tibo990i-thesis_data/data_out/3yr_2019-2021/caseXY (see --data-path in <run_benchmark_temporal.sbatch)

2. run baseline models
        -> SET PARAMS: in <run_benchmark_temporal.sbatch>
                --data-path 
                --output-path
                --forecast_horizon

3. Transfer results to local

# IMPORTANT Notes
- If Forecast Horizon > 1 -> The number of scenarios/timesteps predicted  is less.
        - Example: if T=100 and test indices are 85..99:
                -> h=1 starts: 85..99 → 15 starts
                -> h=6 starts: 85..94 → 10 starts (Cannot predict 6 timesteps into the future at t > 94)
# Side-Info
- interleaved is outdated and unneccessary since we have more years of data
- original_TGT_case14_reference: Hendriks orignial TGT code

# Test run_benchmark_temporal.py locally
python run_benchmark_temporal.py --use-subset --subset-percent 0.01 --skip-tgt --xgb-device cpu --seed 42 


# Run local
python run_benchmark_temporal.py `
  --data-path ..\data\data_out\3yr_2019-2021\case14_ieee\raw `
  --output-path case14_ieee_horizon6.parquet `
  --seed 42 `
  --skip-tgt `
  --xgb-device cpu 