# WORKFLOW — Phase 1: OPF Ground Truth Generation

This step generates AC-OPF ground truth data by combining realistic load profiles
with IEEE test cases and solving via gridfm-datakit (Julia/PowerModels.jl + IPOPT).

## Prerequisites
- gridfm-datakit installed: `pip install -e ../gridfm-datakit-fork`
- Realistic load profiles parquet (from Marcus)

## Steps

1. Generate datakit input (precomputed load profiles)
        -> INPUT: realistic load profiles .parquet
        -> EXECUTE:
                - Cluster: python phase1_generation/preprocessing/generate_precomputed_profile.py
                - Local:   phase1_generation/preprocessing/generate_precomputed_profile.ipynb
        -> OUTPUT: phase1_generation/preprocessing/*_precomputed_load_profiles.csv

2. Run datakit (on cluster — too computationally intensive for local)
        -> EXECUTE: gridfm_datakit generate phase1_generation/configs/phase1_config.yaml
        -> OUTPUT: data/data_out/

3. Generate Baseline Predictions
        -> TRANSFER: data to cluster barnard
        -> RUN: phase1_baseline/ (see readme.txt)
