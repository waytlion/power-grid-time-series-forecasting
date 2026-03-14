# WORKFLOW
1. copy datakit output data to bernard cluster
        -> SOURCE: from local / cluster Leipzig
        -> DESTINATION: see --data-path in <run_benchmark_temporal.sbatch

2. run baseline models
        -> SET PARAMS: in <run_benchmark_temporal.sbatch>
                --data-path 
                --output-path

3. Transfer results to local

# Info
- phase1/ 02_benchmark_temporal.ipynb is the important notebook
- interleaved is outdated and unneccessary since we have more years of data
- original_TGT_case14_reference: Hendriks orignial TGT code

# Test run_benchmark_temporal.py locally
python run_benchmark_temporal.py --use-subset --subset-percent 0.01 --skip-tgt --xgb-device cpu --seed 42 