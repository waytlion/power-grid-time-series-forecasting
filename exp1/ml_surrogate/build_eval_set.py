"""
Build gridfm-graphkit eval datasets for the 2-step ML-surrogate OPF arm (Phase 1c).

For each forecast model, emit a graphkit raw/ dir consumed by the
`OptimalPowerFlowTwoStep` task. Per (flattened) scenario:
  - model INPUT load  -> bus.Pd / bus.Qd        (the forecast)
  - true-future label -> bus.Qg/Vm/Va, gen.p_mw (Phase 1a OPF at the realized step)
  - true realized load-> bus.Pd_true / bus.Qd_true (for the residual-vs-true metric)
  - static network    -> copied from Phase 1a (admittances, limits, costs, bus types)

Mapping: target_scenario = load_scenario_idx (origin) + horizon_step. Phase 1a is
indexed by scenario == load_scenario_idx (1:1, timestep-aligned), so the true-future
values are looked up directly at the target scenario. Horizon-agnostic: the horizon
is whichever forecast config produced the input CSVs (set as a tag at eval time).

Inputs (mirrors phase1c_run_datakit_batch.py):
  --data-in-dir : dir of <model>.csv from phase1c_transform_forecasts.py
                  (cols: load_scenario, load_scenario_idx, horizon_step, load, p_mw, q_mvar)
  --ground-truth-dir : Phase 1a raw/ (bus_data/gen_data/branch_data.parquet)
  --out-dir     : <out>/<model>/<network>/raw/{bus,gen,branch}_data.parquet
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# graphkit dataset reads these (process() also pulls min_q_mvar/max_q_mvar from gen)
BUS_OUT_COLS = [
    "scenario", "load_scenario_idx", "bus",
    "Pd", "Qd", "Pd_true", "Qd_true",          # forecast load + true realized load
    "Qg", "Vm", "Va",                          # true-future dispatch (label)
    "PQ", "PV", "REF", "min_vm_pu", "max_vm_pu", "GS", "BS", "vn_kv",  # static
]
GEN_COPY_COLS = [
    "idx", "bus", "p_mw", "q_mvar", "min_p_mw", "max_p_mw",
    "min_q_mvar", "max_q_mvar", "cp0_eur", "cp1_eur_per_mw", "cp2_eur_per_mw2",
    "in_service",
]
BRANCH_COPY_COLS = [
    "idx", "from_bus", "to_bus", "pf", "qf", "pt", "qt",
    "Yff_r", "Yff_i", "Yft_r", "Yft_i", "Ytf_r", "Ytf_i", "Ytt_r", "Ytt_i",
    "tap", "ang_min", "ang_max", "rate_a", "br_status",
]


def build_parser():
    p = argparse.ArgumentParser(description="Build 2-step surrogate eval datasets.")
    p.add_argument("--data-in-dir", type=Path, required=True,
                   help="Dir of <model>.csv forecasts (phase1c_transform_forecasts.py output)")
    p.add_argument("--ground-truth-dir", type=Path, required=True,
                   help="Phase 1a raw/ dir (bus_data/gen_data/branch_data.parquet)")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output base: <out>/<model>/<network>/raw/")
    p.add_argument("--network-name", type=str, required=True, help="e.g. case118_ieee")
    p.add_argument("--models", nargs="+", default=["xgb", "sarima", "snaive", "tgt", "true"])
    p.add_argument("--limit-scenarios", type=int, default=None,
                   help="Process only the first N flattened scenarios (debug).")
    return p


def build_bus(fc, gt_bus):
    """fc: forecast CSV rows; gt_bus: Phase 1a bus_data. -> eval bus_data."""
    fc = fc.copy()
    fc["target"] = fc["load_scenario_idx"].astype(int) + fc["horizon_step"].astype(int)
    m = fc.merge(
        gt_bus,
        left_on=["target", "load"], right_on=["scenario", "bus"],
        how="left", suffixes=("", "_gt"),
    )
    if m["Pd"].isna().any():
        raise ValueError("Forecast (target, bus) not found in ground truth bus_data.")
    out = pd.DataFrame({
        "scenario": m["load_scenario"].astype(int),   # flattened, contiguous 0..S-1
        "load_scenario_idx": m["target"].astype(int),  # realized target timestep
        "bus": m["load"].astype(int),
        "Pd": m["p_mw"].astype(float),                 # forecast load (model input)
        "Qd": m["q_mvar"].astype(float),
        "Pd_true": m["Pd"].astype(float),              # true realized load (gt)
        "Qd_true": m["Qd"].astype(float),
        "Qg": m["Qg"], "Vm": m["Vm"], "Va": m["Va"],   # true-future dispatch (label)
        "PQ": m["PQ"], "PV": m["PV"], "REF": m["REF"],
        "min_vm_pu": m["min_vm_pu"], "max_vm_pu": m["max_vm_pu"],
        "GS": m["GS"], "BS": m["BS"], "vn_kv": m["vn_kv"],
    })
    return out[BUS_OUT_COLS]


def replicate_by_target(sc_map, gt_df, copy_cols):
    """Replicate Phase 1a gen/branch rows per flattened scenario sharing a target."""
    m = sc_map.merge(gt_df, left_on="target", right_on="scenario", how="left", suffixes=("", "_gt"))
    out = m[["load_scenario", "target"] + copy_cols].rename(
        columns={"load_scenario": "scenario", "target": "load_scenario_idx"}
    )
    return out


def main(argv=None):
    args = build_parser().parse_args(argv)
    gt = args.ground_truth_dir
    gt_bus = pd.read_parquet(gt / "bus_data.parquet")
    gt_gen = pd.read_parquet(gt / "gen_data.parquet")
    gt_branch = pd.read_parquet(gt / "branch_data.parquet")

    for model in args.models:
        csv = args.data_in_dir / f"{model}.csv"
        if not csv.exists():
            print(f"Skip {model}: missing {csv}")
            continue
        print(f"\n=== {model} ===")
        fc = pd.read_csv(csv)
        if args.limit_scenarios is not None:
            keep = fc["load_scenario"] < args.limit_scenarios
            fc = fc[keep]

        # flattened scenario -> target timestep (one row per scenario)
        sc_map = fc[["load_scenario", "load_scenario_idx", "horizon_step"]].drop_duplicates("load_scenario").copy()
        sc_map["target"] = sc_map["load_scenario_idx"].astype(int) + sc_map["horizon_step"].astype(int)
        sc_map = sc_map[["load_scenario", "target"]]

        bus = build_bus(fc, gt_bus)
        gen = replicate_by_target(sc_map, gt_gen, GEN_COPY_COLS)
        branch = replicate_by_target(sc_map, gt_branch, BRANCH_COPY_COLS)

        # graphkit requires contiguous scenario 0..S-1
        n = bus["scenario"].nunique()
        assert bus["scenario"].min() == 0 and bus["scenario"].max() == n - 1, "non-contiguous scenario"

        out_raw = args.out_dir / model / args.network_name / "raw"
        out_raw.mkdir(parents=True, exist_ok=True)
        bus.to_parquet(out_raw / "bus_data.parquet", index=False)
        gen.to_parquet(out_raw / "gen_data.parquet", index=False)
        branch.to_parquet(out_raw / "branch_data.parquet", index=False)
        print(f"  scenarios={n} | bus={len(bus)} gen={len(gen)} branch={len(branch)} -> {out_raw}")

    print("\nDone.")


if __name__ == "__main__":
    main()
