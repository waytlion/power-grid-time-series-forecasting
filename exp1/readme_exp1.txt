# WORKFLOW — Exp1: Two-Step OPF Evaluation

This step evaluates baseline forecasting methods by solving AC-OPF for
the predicted loads and comparing against ground-truth OPF results.

## Prerequisites
- gridfm-datakit installed: `pip install -e ../gridfm-datakit-fork`
- Baseline prediction parquets from phase1_baseline (on barnard cluster)
- Ground-truth OPF data from phase1_generation

## Steps

1. Load baseline predictions from cluster "barnard" to local
      -> scp tibo990i@login1.barnard.hpc.tu-dresden.de:/home/tibo990i/Thesis_Repo/phase1_baseline/*parquet exp1/data/data_in

2. On Local: Transform predictions into datakit input format ("precomputed_profiles")
      -> EXECUTE: exp1/generate_opf_inputs/exp1_step2_transform_benchmark_to_datakit.ipynb
      -> OUTPUT:  exp1/data/precomputed_profiles/
      -> OPTIONAL: Load precomputed_profiles .csv files to google drive

3. On Cluster Leipzig: Run datakit on baseline predictions
      -> EXPORT PRECOMPUTED_PROFILES TO CLUSTER:
            scp -r exp1/data/precomputed_profiles/case118_ieee_horizon24_3yr/ og98ohex@export01.sc.uni-leipzig.de:~/Thesis_Repo/exp1/data/precomputed_profiles/
      -> CONFIG 3 PARAMS IN: exp1/configs/cluster_leipzig_opf_for_forecast.yaml
            - scenarios
            - scenario_file
            - data_dir
      -> EXECUTE: sbatch scripts/cluster_leipzig.sh
      -> IMPORT AC-OPF RESULTS TO LOCAL:
            scp -r og98ohex@export01.sc.uni-leipzig.de:~/data_out/baseline_preds/... exp1/data/data_out/...

4. On Local: Compare AC-OPF results derived from predictions with ground truth
      -> READ AND EXECUTE: python exp1/generate_metrics/compare.py
      -> OUTPUT: exp1/results/

## NOTES
1. Qd is not forecasted by baseline models -> derived by applying scaling factor to Pd
2. datakit on cluster cannot be run in parallel -> Julia makes probs

## Abstract Description
Input: read in the predicted loads (which were predicted in phase1_baseline)

Pipeline:
1. For each model: Take the predicted loads and transform into format for datakit OPF
   -> Add q_mvar derived via bus-specific scaling factor
2. Solve OPF for predicted loads (on cluster)
3. Compute the same metrics as graphkit does

Output:
- For each model, output the metrics of the two-step approach
