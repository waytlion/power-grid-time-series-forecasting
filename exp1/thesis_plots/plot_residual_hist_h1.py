"""
Results-chapter figures: per-bus power-balance residuals at h=1, all arms, physical MW.
- design settled w/ TSO input + review cycles: absolute MW units, NO normalization,
  NO bus exclusions (all buses incl. REF/PV; PQ-only = optional appendix sensitivity)
- 1 MW / 1 MVAr line = practitioner negligibility heuristic (pers. comm.), NOT a decision rule

Produces:
  - residual_hist_h1.png    : bus-level signed residual histogram (focused) + full-range violin
  - net_imbalance_h1.png    : scenario-level signed NET imbalance (sum over buses) per arm
  - residual_stats.csv      : per arm x P/Q: bias, MAE, abs-quantiles, share above 1 MW|MVAr
  - net_imbalance_stats.csv : per arm: net bias, signed quantiles, mean |net|

Arm parquet schema: [scenario, bus, res_p_mw, res_q_mvar, ...] (2-step extractor or graphkit dump).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REF_MW = 1.0    # active negligibility heuristic (pers. comm.: <~1 MW bus-level needs no corrective action)
REF_MVAR = 1.0  # reactive analogue


def load_arm(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    need = {"scenario", "bus", "res_p_mw", "res_q_mvar"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path}: missing columns {need - set(df.columns)}")
    return df[list(need)]


def residual_stats(frames: dict) -> pd.DataFrame:
    """Per arm x quantity: signed bias, MAE, |res| quantiles, share above the heuristic line."""
    rows = []
    for arm, df in frames.items():
        for qty, col, ref in (("P", "res_p_mw", REF_MW), ("Q", "res_q_mvar", REF_MVAR)):
            v = df[col]
            a = v.abs()
            rows.append({
                "arm": arm, "quantity": qty, "unit": "MW" if qty == "P" else "MVAr",
                "bias_signed": v.mean(), "mae": a.mean(),
                "absP50": a.quantile(0.50), "absP90": a.quantile(0.90),
                "absP95": a.quantile(0.95), "absP99": a.quantile(0.99),
                f"share_above_{ref:g}": 100.0 * (a > ref).mean(),
            })
    return pd.DataFrame(rows)


def net_per_scenario(frames: dict) -> dict:
    """Signed net imbalance per scenario (sum over ALL buses) -> {arm: Series}."""
    return {arm: df.groupby("scenario")["res_p_mw"].sum() for arm, df in frames.items()}


def net_stats(nets: dict) -> pd.DataFrame:
    rows = []
    for arm, s in nets.items():
        rows.append({
            "arm": arm, "unit": "MW",
            "net_mean_signed": s.mean(), "net_P5": s.quantile(0.05),
            "net_P50": s.quantile(0.50), "net_P95": s.quantile(0.95),
            "net_mean_abs": s.abs().mean(),
        })
    return pd.DataFrame(rows)


def plot_bus_level(frames: dict, focus: float, out_png: Path):
    """(L) signed histogram focused to +/-focus MW, share>1MW in legend; (R) full-range violin."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2.3, 1.0]})

    bins = np.linspace(-focus, focus, 81)
    for arm, df in frames.items():
        v = df["res_p_mw"].to_numpy()
        share = 100.0 * np.mean(np.abs(v) > REF_MW)
        axL.hist(v[(v >= -focus) & (v <= focus)], bins=bins, histtype="step", density=True,
                 linewidth=1.6, label=f"{arm}  ({share:.1f}% > {REF_MW:g} MW)")
    for t in (-REF_MW, REF_MW):
        axL.axvline(t, color="k", ls="--", lw=1.0, alpha=0.7)
    axL.axvline(REF_MW, color="none", label=f"±{REF_MW:g} MW negligibility heuristic (pers. comm.)")
    axL.set_xlim(-focus, focus)
    axL.set_xlabel("Signed nodal active-power residual  [MW]")
    axL.set_ylabel("Density")
    axL.set_title(f"Bus-level residuals  (x-view ±{focus:g} MW; full tails in violin →)")
    axL.legend(frameon=False, fontsize=9)

    labels = list(frames.keys())
    parts = axR.violinplot([frames[a]["res_p_mw"].to_numpy() for a in labels],
                           showextrema=True, showmedians=True, widths=0.9)
    for pc in parts["bodies"]:
        pc.set_alpha(0.55)
    for t in (-REF_MW, REF_MW):
        axR.axhline(t, color="k", ls="--", lw=0.8, alpha=0.6)
    axR.set_xticks(range(1, len(labels) + 1))
    axR.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axR.set_ylabel("Residual [MW]  (full range)")
    axR.set_title("Full distribution")

    fig.suptitle("Power-balance residuals at h=1 (case118, per bus, realized load, all buses)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved bus-level figure -> {out_png}")


def plot_net(nets: dict, out_png: Path):
    """Scenario-level signed net imbalance histogram per arm. No threshold lines (by design:
    frequency control responds to this sum; no documented acceptance threshold exists)."""
    lo = min(s.min() for s in nets.values())
    hi = max(s.max() for s in nets.values())
    bins = np.linspace(lo, hi, 101)

    fig, ax = plt.subplots(figsize=(9, 5))
    for arm, s in nets.items():
        ax.hist(s.to_numpy(), bins=bins, histtype="step", density=True, linewidth=1.6,
                label=f"{arm}  (mean {s.mean():+.1f} MW)")
    ax.axvline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("Signed NET imbalance per scenario  [MW]  (sum over all buses)")
    ax.set_ylabel("Density")
    ax.set_title("System net imbalance at h=1 — balancing burden per scenario")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved net-imbalance figure -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", nargs=2, metavar=("LABEL", "PARQUET"),
                    required=True, help="arm label + per-bus residual parquet (repeatable)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--focus", type=float, default=10.0, help="bus-level histogram x-view half-range [MW]")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = {label: load_arm(Path(path)) for label, path in args.arm}
    for arm, df in frames.items():
        print(f"{arm}: {df['scenario'].nunique()} scenarios x {df['bus'].nunique()} buses")

    # bus-level stats + figure
    stats = residual_stats(frames)
    stats.to_csv(args.out_dir / "residual_stats.csv", index=False)
    print("\nBus-level stats:")
    print(stats.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    plot_bus_level(frames, args.focus, args.out_dir / "residual_hist_h1.png")

    # scenario-level net + figure (DC sanity: net ~ -losses, shifted copy of AC arm's net)
    nets = net_per_scenario(frames)
    nstats = net_stats(nets)
    nstats.to_csv(args.out_dir / "net_imbalance_stats.csv", index=False)
    print("\nNet-imbalance stats (signed sum over all buses per scenario):")
    print(nstats.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    plot_net(nets, args.out_dir / "net_imbalance_h1.png")


if __name__ == "__main__":
    main()
