"""
Run the 2-step ML-surrogate OPF arm of Phase 1c.

Steps:
  1. Build per-model eval sets from forecasts (phase1c_build_twostep_eval_set.py).
  2. Resolve each surrogate seed's checkpoint + normalizer stats from MLflow,
     matched BY TAGS across all experiments (robust to the experiment-name race).
  3. For each (seed, model): run `gridfm_graphkit evaluate` (OptimalPowerFlowTwoStep),
     tagged approach/case/horizon/model/seed for the downstream aggregator.

Eval sets are seed-independent (forecast input + true-future labels), so they are
built once and reused across seeds (the graphkit dataset caches the processed graphs).

Mirrors phase1c_run_datakit_batch.py (the exact-OPF arm).
"""

import argparse
import os
import subprocess
import sys
from glob import glob
from pathlib import Path


def resolve_surrogate_run(mlruns_dir, case, seed):
    """Find a surrogate training run's (checkpoint, normalizer_stats) by tags.

    Searches ALL experiments (the array job can split a sweep across duplicate
    experiment ids), matching tags approach=surrogate, case, seed.
    """
    for run in glob(os.path.join(mlruns_dir, "*", "*", "")):
        tagdir = os.path.join(run, "tags")

        def tag(name):
            p = os.path.join(tagdir, name)
            return open(p).read().strip() if os.path.exists(p) else None

        if tag("approach") == "surrogate" and tag("case") == case and tag("seed") == str(seed):
            ckpt = os.path.join(run, "artifacts", "model", "best_model_state_dict.pt")
            stats = os.path.join(run, "artifacts", "stats", "normalizer_stats.pt")
            if os.path.exists(ckpt) and os.path.exists(stats):
                return ckpt, stats
    return None, None


def build_parser():
    p = argparse.ArgumentParser(description="Run 2-step surrogate OPF eval (Phase 1c).")
    p.add_argument("--case", required=True, help="e.g. case118")
    p.add_argument("--network-name", required=True, help="e.g. case118_ieee")
    p.add_argument("--horizon", required=True, help="forecast horizon (tag value), e.g. 24")
    p.add_argument("--data-in-dir", type=Path, required=True, help="dir of <model>.csv forecasts")
    p.add_argument("--ground-truth-dir", type=Path, required=True, help="Phase 1a raw/ dir")
    p.add_argument("--eval-config", type=Path, required=True, help="twostep_opf eval yaml (abs path)")
    p.add_argument("--eval-set-dir", type=Path, required=True, help="where the converter writes eval sets")
    p.add_argument("--mlruns-dir", type=Path, required=True, help="mlruns to resolve surrogate checkpoints from")
    p.add_argument("--log-dir", type=Path, default=None, help="mlruns to log eval runs to (default: --mlruns-dir)")
    p.add_argument("--eval-exp-name", default="twostep_eval", help="MLflow experiment for eval runs")
    p.add_argument("--models", nargs="+", default=["xgb", "sarima", "snaive", "tgt", "true"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--converter", type=Path,
                   default=Path(__file__).with_name("phase1c_build_twostep_eval_set.py"))
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    py = sys.executable
    log_dir = args.log_dir or args.mlruns_dir

    # 1. Build eval sets (once; reused across seeds)
    print("=== Building eval sets ===")
    subprocess.run([
        py, str(args.converter),
        "--data-in-dir", str(args.data_in_dir),
        "--ground-truth-dir", str(args.ground_truth_dir),
        "--out-dir", str(args.eval_set_dir),
        "--network-name", args.network_name,
        "--models", *args.models,
    ], check=True)

    # 2+3. Per seed: resolve checkpoint; per model: evaluate
    failures = []
    for seed in args.seeds:
        ckpt, stats = resolve_surrogate_run(str(args.mlruns_dir), args.case, seed)
        if ckpt is None:
            print(f"WARN seed={seed}: no surrogate run found (tags approach=surrogate "
                  f"case={args.case} seed={seed}) — skipping.")
            failures.append(("seed", seed))
            continue
        print(f"\n=== seed={seed} -> {ckpt} ===")
        for model in args.models:
            data_path = args.eval_set_dir / model
            raw = data_path / args.network_name / "raw"
            if not raw.exists():
                print(f"  skip {model}: no eval set at {raw}")
                continue
            run_name = f"twostep_{args.case}_{model}_h{args.horizon}_s{seed}"
            print(f"  evaluate model={model} -> run {run_name}")
            r = subprocess.run([
                py, "-u", "-m", "gridfm_graphkit", "evaluate",
                "--config", str(args.eval_config),
                "--model_path", ckpt,
                "--normalizer_stats", stats,
                "--data_path", str(data_path),
                "--log_dir", str(log_dir),
                "--exp_name", args.eval_exp_name,
                "--run_name", run_name,
                "--tags",
                "approach=surrogate", f"case={args.case}", f"horizon={args.horizon}",
                f"model={model}", f"seed={seed}", "split=twostep_eval",
            ])
            if r.returncode != 0:
                print(f"  ERROR evaluate failed: {run_name} (rc={r.returncode})", file=sys.stderr)
                failures.append((model, seed))

    print("\nDone." if not failures else f"\nDone with {len(failures)} failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
