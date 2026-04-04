# Grid-Aware Power Forecasting Benchmarks

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Regression-2F7ED8)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![statsmodels](https://img.shields.io/badge/statsmodels-Time%20Series-5A5A5A)](https://www.statsmodels.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## What This Project Does

This repository is the main orchestration repo for a master thesis on grid-aware
power forecasting. It benchmarks temporal forecasting approaches for electrical
grid load scenarios and evaluates their downstream impact on AC Optimal Power Flow.

### Pipeline Overview

```
Phase 1a: phase1_generation/   →  Generate OPF ground truth data via gridfm-datakit
Phase 1b: phase1_baseline/     →  Train & evaluate temporal forecasting baselines
Phase 1c: exp1/                →  Evaluate baseline forecasts via two-step OPF
```

The spatio-temporal GNN forecasting model is developed in the separate
[gridfm-graphkit](../gridfm-graphkit) repository.

## External Dependencies (Forked Repos)

This repo imports two forked libraries as editable installs:

| Repo | Purpose | Install |
|---|---|---|
| `gridfm-datakit-fork` | AC-OPF data generation via Julia/PowerModels.jl | `pip install -e ../gridfm-datakit-fork` |
| `gridfm-graphkit` | GNN-based OPF proxy model | `pip install -e ../gridfm-graphkit` |

Both forks are kept **clean** — all custom thesis code lives in this repo.

## Tech Stack

- Python, NumPy, pandas, pyarrow
- scikit-learn, statsmodels (SARIMAX), XGBoost
- PyTorch, torch-geometric, torch-scatter
- gridfm-datakit (Julia/PowerModels.jl + IPOPT)
- joblib, tqdm

## Prerequisites

- Python 3.11+
- Access to prepared parquet input data (bus and branch data)
- HPC cluster access for OPF computation (**Leipzig University Cluster "Paula" / "Polaris"**)
- Optional: CUDA-compatible GPU locally for deep learning experiments

## Installation

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e ../gridfm-datakit-fork   # OPF data generation
```

## Usage: Full Pipeline Automation

The entire 3-phase pipeline runs on the Leipzig cluster without any manual data transfers between clusters or local machines.

```bash
# From the root of Thesis_Repo
bash scripts/submit_pipeline.sh
```

This master script uses SLURM dependencies (`--dependency=afterok`) to sequentially chain:
1. `scripts/cluster_leipzig.sh` (Phase 1a)
2. `phase1_baseline/run_benchmark_temporal.sbatch` (Phase 1b)
3. `scripts/phase1c_eval.sbatch` (Phase 1c)



### Manual Usage per Phase

If you prefer to run phases individually:

```bash
# Preprocess load profiles
python phase1_generation/preprocessing/generate_precomputed_profile.py

# Run datakit on cluster
sbatch scripts/cluster_leipzig.sh
```

### Phase 1b: Temporal Baseline Benchmarks

```bash
cd phase1_baseline
sbatch run_benchmark_temporal.sbatch
```

### Phase 1c: Two-Step OPF Evaluation

```bash
sbatch scripts/phase1c_eval.sbatch
```

## Repository Layout

```
Thesis_Repo/
├── phase1_generation/       # OPF ground truth generation pipeline
│   ├── preprocessing/       #   Load profile preprocessing scripts
│   └── configs/             #   DataKit YAML configs
├── phase1_baseline/         # Temporal forecasting benchmarks
│   ├── src/                 #   Forecasting model implementations
│   └── run_benchmark_temporal.py
├── exp1/                    # Two-step OPF evaluation pipeline
│   ├── configs/             #   DataKit YAML configs for forecast OPF
│   ├── generate_opf_inputs/ #   Transform forecasts → datakit format
│   └── generate_metrics/    #   Compare predicted vs ground-truth OPF
├── scripts/                 # Cluster submission scripts
├── data/                    # Shared data directory (gitignored)
└── requirements.txt
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
