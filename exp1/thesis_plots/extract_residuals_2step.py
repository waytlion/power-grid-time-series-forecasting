"""
Extract per-bus SIGNED power-balance residuals (MW) for the 2-step AC/DC arms at
a single horizon, in either the REALIZED or the FORECAST load space.

  --load realized (default): dispatch scored vs the TRUE realized load
      dispatch  <- forecaster arm (e.g. xgb): Vm/Va/Pg/Qg [+ Va_dc/Pg_dc]
      realized load <- `true` arm (OPF solved on true loads): Pd/Qd/GS/BS
      Both arms share identical flat indexing, so a plain merge on
      (load_scenario_idx, bus) aligns dispatch with its realization.
      -> operational feasibility (intrinsic infeasibility + forecast error).

  --load forecast: dispatch scored vs the arm's OWN forecast load (self-score)
      Both dispatch AND load come from the forecaster arm; no `true` arm needed.
      -> intrinsic modeling infeasibility (AC ~ 0; DC = losses ignored). This is
      the clean modeling-error component (no MAE-subtraction subtlety).

Output: one tidy parquet per arm -> [scenario, bus, res_p_mw, res_q_mvar, arm].
Normalization by per-bus peak load, REF exclusion and eps-floor happen in the plot script.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

_GEN_METRICS = Path(__file__).resolve().parents[1] / "generate_metrics"
if str(_GEN_METRICS) not in sys.path:
    sys.path.insert(0, str(_GEN_METRICS))

from loaders import load_datakit_bus, load_datakit_branch  # noqa: E402
from metrics import extract_per_bus_residuals  # noqa: E402


def build_bus_aligned(opf_base_dir: Path, forecaster: str, dataset: str,
                      load_ref: str = "realized") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the aligned dispatch+load frame the residual core expects.

    load_ref="realized": dispatch (forecaster arm) vs true-arm load, merged on (scenario, bus).
    load_ref="forecast": dispatch vs the forecaster arm's OWN load (self-score, no true arm).
    """
    pred_dir = opf_base_dir / forecaster / dataset / "raw"
    if not pred_dir.exists():
        raise FileNotFoundError(f"OPF arm dir not found: {pred_dir}")
    pred_bus = load_datakit_bus(pred_dir)      # Vm/Va/Pg/Qg (+ Va_dc/Pg_dc), Pd/Qd = forecast
    branch = load_datakit_branch(pred_dir)     # static topology (admittances; identical across arms)

    if load_ref == "realized":
        true_dir = opf_base_dir / "true" / dataset / "raw"
        if not true_dir.exists():
            raise FileNotFoundError(f"OPF `true` arm dir not found: {true_dir}")
        true_bus = load_datakit_bus(true_dir)  # Pd/Qd = realized loads, GS/BS static
        bus_aligned = pred_bus.merge(true_bus, on=["load_scenario_idx", "bus"], suffixes=("_pred", "_true"))
        if len(bus_aligned) != len(pred_bus):
            raise ValueError(
                f"Merge lost rows: pred={len(pred_bus)} aligned={len(bus_aligned)} "
                "(forecaster and true arms are not on the same flat index set)."
            )
        # DC dispatch: use the FORECASTER arm's DC solve (true arm also carries DC cols,
        # so the merge suffixes both); expose the pred side as unsuffixed Va_dc/Pg_dc.
        for c in ("Va_dc", "Pg_dc"):
            if f"{c}_pred" in bus_aligned.columns:
                bus_aligned = bus_aligned.rename(columns={f"{c}_pred": c})
    elif load_ref == "forecast":
        # Self-score: dispatch AND load both from the forecaster arm. Map its columns
        # into the _pred/_true convention the residual core reads (Va_dc/Pg_dc kept as-is).
        bus_aligned = pred_bus.rename(columns={
            "Vm": "Vm_pred", "Va": "Va_pred", "Pg": "Pg_pred", "Qg": "Qg_pred",
            "Pd": "Pd_true", "Qd": "Qd_true", "GS": "GS_true", "BS": "BS_true",
        })
    else:
        raise ValueError(f"load_ref must be 'realized' or 'forecast', got {load_ref!r}")

    # flat index == scenario index (arms already share it; no remap needed)
    bus_aligned["pred_flat_idx"] = bus_aligned["load_scenario_idx"]
    return bus_aligned, branch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opf-base-dir", type=Path, required=True,
                    help="phase1c data_out dir holding <forecaster>/<dataset>/raw and true/...")
    ap.add_argument("--forecaster", default="xgb")
    ap.add_argument("--dataset", default="case118_ieee")
    ap.add_argument("--load", choices=["realized", "forecast"], default="realized",
                    help="score dispatch vs true realized load (default) or vs the arm's own forecast load")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bus_aligned, branch = build_bus_aligned(args.opf_base_dir, args.forecaster, args.dataset, args.load)
    n_scen = bus_aligned["pred_flat_idx"].nunique()
    n_bus = bus_aligned["bus"].nunique()
    print(f"[{args.load} space] Aligned {len(bus_aligned)} obs = {n_scen} scenarios x {n_bus} buses")

    has_dc = {"Va_dc", "Pg_dc"}.issubset(bus_aligned.columns)
    modes = [("ac", "AC-2step")] + ([("dc", "DC-2step")] if has_dc else [])
    if not has_dc:
        print("  NOTE: no Va_dc/Pg_dc columns -> skipping DC arm (re-run phase1c with include_dc_res).")

    for mode, arm in modes:
        res = extract_per_bus_residuals(bus_aligned, branch, mode=mode)
        res = res.rename(columns={"pred_flat_idx": "scenario"})
        res["arm"] = arm
        out = args.out_dir / f"residuals_{args.forecaster}_{mode}_{args.load}.parquet"
        res.to_parquet(out, index=False)
        rp = res["res_p_mw"]
        print(f"  {arm}: {len(res)} rows -> {out}")
        print(f"      res_p_mw  mean={rp.mean():+.4f}  MAE={rp.abs().mean():.4f}  "
              f"min={rp.min():+.3f}  max={rp.max():+.3f}")


if __name__ == "__main__":
    main()
