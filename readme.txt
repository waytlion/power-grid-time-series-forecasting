# Info
- phase1/ 02_benchmark_temporal.ipynb is the important notebook
- interleaved is outdated and unneccessary since we have more years of data
- original_TGT_case14_reference: Hendriks orignial TGT code

# Test run_benchmark_temporal.py locally
python run_benchmark_temporal.py --use-subset --subset-percent 0.01 --skip-tgt --xgb-device cpu --seed 42 