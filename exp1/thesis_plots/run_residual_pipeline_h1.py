"""
One-shot per-network per-horizon residual pipeline (results ch.): readme.txt PHASE 2 steps 8+10,
run once per (network, load-space) with a single command.

Step 9 (surrogate/E2E per-bus dumps) is a byproduct of that arm's own eval/train pass, not
something this script produces -- point --surrogate-realized/--e2e-realized at the resulting
<dataset>_perbus_residuals_realized.parquet files.

Extracts 2-step AC/DC residuals in both load spaces (realized -> fed into the figures/stats;
forecast -> dumped alongside for the modeling-error appendix), then builds the results figures
+ stats CSVs from the realized-space parquets (AC/DC/Surrogate/E2E).
"""

import argparse
import sys
from pathlib import Path

_GEN_METRICS = Path(__file__).resolve().parents[1] / "generate_metrics"
if str(_GEN_METRICS) not in sys.path:
    sys.path.insert(0, str(_GEN_METRICS))

from metrics import extract_per_bus_residuals  # noqa: E402

from extract_residuals_2step import build_bus_aligned  # noqa: E402
from plot_residual_hist_h1 import (  # noqa: E402
    load_arm, residual_stats, residual_stats_latex, net_per_scenario, net_stats,
    net_stats_latex, plot_bus_level, plot_net,
)


def extract_2step(opf_base_dir: Path, forecaster: str, dataset: str, out_dir: Path) -> dict:
    """-> {"AC-2step": [df], ["DC-2step": [df]]} in the realized load space; also dumps the
    forecast-space parquets to out_dir for the modeling-error appendix."""
    realized = {}
    for load in ("realized", "forecast"):
        bus_aligned, branch = build_bus_aligned(opf_base_dir, forecaster, dataset, load)
        has_dc = {"Va_dc", "Pg_dc"}.issubset(bus_aligned.columns)
        modes = [("ac", "AC-2step")] + ([("dc", "DC-2step")] if has_dc else [])
        for mode, arm in modes:
            res = extract_per_bus_residuals(bus_aligned, branch, mode=mode).rename(
                columns={"pred_flat_idx": "scenario"})
            res["arm"] = arm
            out = out_dir / f"residuals_{forecaster}_{mode}_{load}.parquet"
            res.to_parquet(out, index=False)
            print(f"  [{load}] {arm}: {len(res)} rows -> {out}")
            if load == "realized":
                realized[arm] = [res]  # seed-free arm -> single "seed"
    return realized


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opf-base-dir", type=Path, required=True,
                    help="phase1c data_out dir, e.g. .../case118_ieee_horizon_1_3yr_HPO_xgb")
    ap.add_argument("--forecaster", default="xgb")
    ap.add_argument("--dataset", default="case118_ieee")
    ap.add_argument("--surrogate-realized", type=Path, nargs="+", required=True,
                    help="one per seed (pooled): <surrogate run>/artifacts/test/<dataset>_perbus_residuals_realized.parquet")
    ap.add_argument("--e2e-realized", type=Path, nargs="+", required=True,
                    help="one per seed (pooled): <e2e run>/artifacts/test/<dataset>_perbus_residuals_realized.parquet")
    ap.add_argument("--horizon", type=int, default=1, help="forecast horizon (figure titles + file names)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--focus", type=float, default=10.0, help="bus-level histogram x-view half-range [MW]")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = extract_2step(args.opf_base_dir, args.forecaster, args.dataset, args.out_dir)
    frames["Surrogate-2step"] = load_arm(*args.surrogate_realized)
    frames["E2E"] = load_arm(*args.e2e_realized)
    for arm, seeds in frames.items():
        print(f"{arm}: {len(seeds)} seed(s), {seeds[0]['scenario'].nunique()} scenarios x {seeds[0]['bus'].nunique()} buses")

    case_label = args.dataset.split("_")[0]  # case500_goc -> case500

    stats = residual_stats(frames)
    residual_stats_latex(stats, args.out_dir / "residual_bus_level.tex")
    print("\nBus-level stats:")
    print(stats.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    plot_bus_level(frames, args.focus, args.out_dir / f"residual_hist_h{args.horizon}.pdf", case_label, args.horizon)

    nets = net_per_scenario(frames)
    nstats = net_stats(nets)
    net_stats_latex(nstats, args.out_dir / "residual_system_net.tex")
    print("\nNet-imbalance stats (signed sum over all buses per scenario):")
    print(nstats.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    plot_net(nets, args.out_dir / f"net_imbalance_h{args.horizon}.pdf", args.horizon)

    print(f"\nDone -> {args.out_dir}")


if __name__ == "__main__":
    main()
