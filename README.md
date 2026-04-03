# Grid-Aware Power Forecasting Benchmarks

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Regression-2F7ED8)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![statsmodels](https://img.shields.io/badge/statsmodels-Time%20Series-5A5A5A)](https://www.statsmodels.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## What This Project Does

This repository benchmarks forecasting approaches for electrical grid load scenarios and develops a next-step spatio-temporal graph neural network pipeline.

- Phase 1 (`phase1_baseline`): temporal benchmark of SNaive, SARIMA, XGBoost, and TinyTGT on bus-level load forecasting.

The goal is to compare strong statistical and ML baselines, then extend toward topology-aware deep learning models for power system forecasting.

## Tech Stack

- Python
- NumPy, pandas, pyarrow
- scikit-learn
- statsmodels (SARIMAX)
- XGBoost
- PyTorch
- torch-geometric, torch-scatter
- joblib, tqdm

## Prerequisites

- Python 3.11+ (recommended)
- Access to prepared parquet input data (bus and branch data)
- Optional: CUDA-compatible GPU for faster deep learning and XGBoost training

## Installation

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Usage

Run the temporal baseline benchmark locally:

```bash
cd phase1_baseline
python run_benchmark_temporal.py \
  --data-path ..\data\data_out\3yr_2019-2021\case14_ieee\raw \
  --output-path case14_ieee_horizon1.parquet \
  --seed 42 \
  --skip-tgt \
  --xgb-device cpu
```

Run on a SLURM cluster with the provided batch script:

```bash
cd phase1_baseline
sbatch run_benchmark_temporal.sbatch
```

## Repository Layout

- `phase1_baseline/`: baseline experiments, notebooks, benchmark script, and core forecasting modules.
- `requirements.txt`: Python dependencies.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
