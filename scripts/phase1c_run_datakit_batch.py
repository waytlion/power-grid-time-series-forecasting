"""
Batch runner for executing DataKit sequentially for multiple forecast models.
Since Julia may crash if threaded incorrectly, this ensures DataKit OPF jobs 
are executed strictly sequentially per-model within the same SLURM job.

Usage:
    python scripts/run_datakit_batch.py \
        --base-yaml exp1/configs/case118_generate_opf_for_forecast.yaml \
        --scenarios 23622 \
        --data-in-dir exp1/data/precomputed_profiles/case118_ieee_horizon6_3yr \
        --out-dir data/data_out/3yr_2019-2021/baseline_preds/case118_horizon_6_3yr \
        --models xgb sarima snaive tgt
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run DataKit over multiple models sequentially.")
    parser.add_argument("--base-yaml", type=Path, required=True, help="Base datakit YAML config file")
    parser.add_argument("--data-in-dir", type=Path, required=True, help="Directory containing <model>.csv files")
    parser.add_argument("--out-dir", type=Path, required=True, help="Base directory to save output data")
    parser.add_argument("--models", nargs="+", default=["xgb", "sarima", "snaive", "tgt", "true"], help="Models to evaluate")
    parser.add_argument("--scenarios", type=int, default=None, help="Number of scenarios (auto-detected if omitted)")
    parser.add_argument("--network-name", type=str, default=None, help="Override config network.name from base YAML")
    
    args = parser.parse_args()

    # Load base yaml
    with open(args.base_yaml, 'r') as f:
        config = yaml.safe_load(f)


    config['network']['name'] = args.network_name

    temp_yaml_path = args.base_yaml.parent / f"temp_batch_{args.base_yaml.name}"

    try:
        for model in args.models:
            model_csv = args.data_in_dir / f"{model}.csv"
            
            if not model_csv.exists():
                print(f"Warning: Skipping {model}, file missing: {model_csv}")
                continue
                
            model_out_dir = args.out_dir / model

            print(f"\n{'='*60}")
            print(f"Running DataKit for model: {model}")
            print(f"Output to: {model_out_dir}")
            print(f"{'='*60}\n")

            # Update config for this model
            config['load']['scenario_file'] = str(model_csv)
            config['settings']['data_dir'] = str(model_out_dir)

            # Auto-detect scenarios if not provided
            if args.scenarios is not None:
                config['load']['scenarios'] = args.scenarios
            else:
                import pandas as pd
                print(f"Auto-detecting scenario count from {model_csv.name}...")
                df = pd.read_csv(model_csv, usecols=["load_scenario"])
                scenarios = int(df["load_scenario"].max() + 1)
                config['load']['scenarios'] = scenarios
                print(f"-> Detected {scenarios} scenarios.")

            # Write temporary YAML
            with open(temp_yaml_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            # Execute datakit CLI
            cmd = [sys.executable, "-m", "gridfm_datakit.cli", "generate", str(temp_yaml_path)]
            
            # Using Popen to stream output rather than capture it
            result = subprocess.run(cmd)
            
            if result.returncode != 0:
                print(f"Error: DataKit failed for model {model} (return code {result.returncode})", file=sys.stderr)
                sys.exit(result.returncode)

    finally:
        # Cleanup temp yaml
        if temp_yaml_path.exists():
            temp_yaml_path.unlink()
            
    print("\nBatch generation complete.")


if __name__ == "__main__":
    main()
