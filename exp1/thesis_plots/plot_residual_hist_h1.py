"""
Results-chapter figures: per-bus power-balance residuals at h=1, all arms, physical MW.
- design settled w/ TSO input + review cycles: absolute MW units, NO normalization,
  NO bus exclusions (all buses incl. REF/PV; PQ-only = optional appendix sensitivity)
- 1 MW / 1 MVAr line = practitioner negligibility heuristic (pers. comm.), NOT a decision rule

Produces:
  - residual_hist_h1.{pdf,svg}: bus-level signed residual histogram (focused) + full-range violin
  - net_imbalance_h1.{pdf,svg}: scenario-level signed NET imbalance (sum over buses) per arm
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

from colors import build_color_map, THESIS_RC

REF_MW = 1.0    # active negligibility heuristic (pers. comm.: <~1 MW bus-level needs no corrective action)
REF_MVAR = 1.0  # reactive analogue


def load_arm(*paths: Path) -> list:
    """One dump per seed -> list of per-seed frames (seeds stay separate: stats/plots are mean +- std)."""
    need = {"scenario", "bus", "res_p_mw", "res_q_mvar"}
    frames = []
    for path in paths:
        df = pd.read_parquet(path)
        if not need.issubset(df.columns):
            raise ValueError(f"{path}: missing columns {need - set(df.columns)}")
        frames.append(df[list(need)])
    return frames


def _mean_std(rows: list, keys: list) -> pd.DataFrame:
    """Per-seed stat rows -> mean columns + `<col>_std` columns (std over seeds, 0 for a single seed)."""
    g = pd.DataFrame(rows).groupby(keys, sort=False)
    return g.mean().join(g.std(ddof=1).fillna(0.0).add_suffix("_std")).reset_index()


def residual_stats(frames: dict) -> pd.DataFrame:
    """Per arm x quantity, mean +- std over seeds: signed bias, MAE, |res| quantiles, share above the heuristic line."""
    rows = []
    for arm, seeds in frames.items():
        for df in seeds:
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
    return _mean_std(rows, ["arm", "quantity", "unit"])


def net_per_scenario(frames: dict) -> dict:
    """Signed net imbalance per scenario (sum over ALL buses) -> {arm: [Series per seed]}."""
    return {arm: [df.groupby("scenario")["res_p_mw"].sum() for df in seeds] for arm, seeds in frames.items()}


def net_stats(nets: dict) -> pd.DataFrame:
    rows = [{
        "arm": arm, "unit": "MW",
        "net_mean_signed": s.mean(), "net_P5": s.quantile(0.05),
        "net_P50": s.quantile(0.50), "net_P95": s.quantile(0.95),
        "net_mean_abs": s.abs().mean(),
    } for arm, seeds in nets.items() for s in seeds]
    return _mean_std(rows, ["arm", "unit"])


# ---------------------------------------------------------------------------
# LaTeX table writers (booktabs). Emitted straight from the stats frames, arm
# order preserved from the frames dict (AC / DC / Surrogate / E2E).
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    """3-decimal; snap negative zero to +0.000 so tables read clean."""
    return f"{0.0 if abs(v) < 5e-4 else v:.3f}"


def _cell(r, col: str) -> str:
    """mean $\\pm$ std over seeds (plain mean when std is 0, e.g. seed-free 2-step arms)."""
    return _fmt(r[col]).strip() + (rf" $\pm$ {_fmt(r[col + '_std']).strip()}" if r[col + "_std"] > 0 else "")


def net_stats_latex(nstats: pd.DataFrame, out_path: Path):
    """tab:net_imbalance -- one row per arm: signed net mean, P5/median/P95, mean |net|."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Distribution of net active power imbalance (MW). "
        r"Mean absolute imbalance is reported in the last column. Mean $\pm$ std over seeds.}",
        r"\label{tab:net_imbalance}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        "Method &", "Mean &", "P5 &", "Median &", "P95 &", r"Mean Abs. \\",
        r"\midrule",
    ]
    for _, r in nstats.iterrows():
        cells = " & ".join(_cell(r, c) for c in
                           ("net_mean_signed", "net_P5", "net_P50", "net_P95", "net_mean_abs"))
        lines.append(f"{r['arm']:<15} & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out_path.write_text("\n".join(lines))
    print(f"Wrote net-imbalance LaTeX -> {out_path}")


def residual_stats_latex(stats: pd.DataFrame, out_path: Path):
    """tab:error_metrics -- two rows per arm (P, Q): bias, MAE, abs-percentiles, share>ref.
    A \\midrule separates each arm's P/Q pair."""
    share_col = next(c for c in stats.columns if c.startswith("share_above_") and not c.endswith("_std"))
    metric_cols = ("bias_signed", "mae", "absP50", "absP90", "absP95", "absP99", share_col)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Prediction error statistics for active ($P$) and reactive ($Q$) power. "
        r"Lower is better for all metrics except signed bias. "
        r"Percentiles refer to absolute error. Mean $\pm$ std over seeds.}",
        r"\label{tab:error_metrics}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        "Method & Qty &", "Bias &", "MAE &", "P50 &", "P90 &",
        "P95 &", "P99 &", r"$>1$ (\%) \\",
        r"\midrule",
    ]
    arms = list(dict.fromkeys(stats["arm"]))  # frames-dict order
    for i, arm in enumerate(arms):
        if i > 0:
            lines.append(r"\midrule")
        for _, r in stats[stats["arm"] == arm].iterrows():
            cells = " & ".join(_cell(r, c) for c in metric_cols)
            lines.append(f"{arm:<15} & ${r['quantity']}$ & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out_path.write_text("\n".join(lines))
    print(f"Wrote error-metrics LaTeX -> {out_path}")


def _hist_band(ax, seeds: list, bins, color, label: str):
    """Per-seed density histograms -> mean step line + std band across seeds."""
    dens = np.array([np.histogram(x, bins=bins, density=True)[0] for x in seeds])
    m, sd = dens.mean(0), dens.std(0, ddof=1) if len(seeds) > 1 else 0.0
    ax.stairs(m, bins, color=color, lw=1.6, label=label)
    ax.stairs(m + sd, bins, baseline=m - sd, fill=True, color=color, alpha=0.25, lw=0)


def _mean_std_str(x, fmt: str) -> str:
    return f"{np.mean(x):{fmt}}" + (f" ± {np.std(x, ddof=1):{fmt}}" if len(x) > 1 else "")


def plot_bus_level(frames: dict, focus: float, out_path: Path, case_label: str = "case500", horizon: int = 1):
    """(L) signed histogram focused to +/-focus MW, share>1MW in legend; (R) full-range violin."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2.3, 1.0]})

    colors = build_color_map(frames.keys())
    bins = np.linspace(-focus, focus, 81)
    for arm, seeds in frames.items():
        vs = [df["res_p_mw"].to_numpy() for df in seeds]
        share = _mean_std_str([100.0 * np.mean(np.abs(v) > REF_MW) for v in vs], ".1f")
        _hist_band(axL, [v[(v >= -focus) & (v <= focus)] for v in vs], bins, colors[arm],
                   f"{arm}  ({share}% > {REF_MW:g} MW)")
    for t in (-REF_MW, REF_MW):
        axL.axvline(t, color="k", ls="--", lw=1.0, alpha=0.7)
    axL.axvline(REF_MW, color="none", label=f"±{REF_MW:g} MW negligibility heuristic (pers. comm.)")
    axL.set_xlim(-focus, focus)
    axL.set_xlabel("Signed nodal active-power residual  [MW]")
    axL.set_ylabel("Density")
    axL.set_title(f"Bus-level residuals  (x-view ±{focus:g} MW; full tails in violin →)")
    axL.legend(frameon=False, fontsize=9)

    labels = list(frames.keys())
    parts = axR.violinplot([pd.concat(frames[a])["res_p_mw"].to_numpy() for a in labels],  # seeds pooled (shape only)
                           showextrema=True, showmedians=True, widths=0.9)
    for pc, arm in zip(parts["bodies"], labels):
        pc.set_facecolor(colors[arm])
        pc.set_alpha(0.55)
    for t in (-REF_MW, REF_MW):
        axR.axhline(t, color="k", ls="--", lw=0.8, alpha=0.6)
    axR.set_xticks(range(1, len(labels) + 1))
    axR.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axR.set_ylabel("Residual [MW]  (full range)")
    axR.set_title("Full distribution")

    fig.suptitle(f"Power-balance residuals at h={horizon} ({case_label}, per bus, realized load, all buses)", y=1.02)
    fig.tight_layout()
    for ext in (".pdf", ".svg"):  # pdf = LaTeX, svg = PowerPoint
        fig.savefig(out_path.with_suffix(ext), bbox_inches="tight")
    print(f"Saved bus-level figure -> {out_path}")


def plot_net(nets: dict, out_path: Path, horizon: int = 1):
    """Scenario-level signed net imbalance histogram per arm. No threshold lines (by design:
    frequency control responds to this sum; no documented acceptance threshold exists)."""
    lo = min(s.min() for seeds in nets.values() for s in seeds)
    hi = max(s.max() for seeds in nets.values() for s in seeds)
    bins = np.linspace(lo, hi, 101)
    colors = build_color_map(nets.keys())

    fig, ax = plt.subplots(figsize=(9, 5))
    for arm, seeds in nets.items():
        _hist_band(ax, [s.to_numpy() for s in seeds], bins, colors[arm],
                   f"{arm}  (mean {_mean_std_str([s.mean() for s in seeds], '+.1f')} MW)")
    ax.axvline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("Signed NET imbalance per scenario  [MW]  (sum over all buses)")
    ax.set_ylabel("Density")
    ax.set_title(f"System net imbalance at h={horizon} — balancing burden per scenario")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for ext in (".pdf", ".svg"):  # pdf = LaTeX, svg = PowerPoint
        fig.savefig(out_path.with_suffix(ext), bbox_inches="tight")
    print(f"Saved net-imbalance figure -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", nargs=2, metavar=("LABEL", "PARQUET"),
                    required=True, help="arm label + per-bus residual parquet (repeatable)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--focus", type=float, default=10.0, help="bus-level histogram x-view half-range [MW]")
    ap.add_argument("--case-label", default="case500", help="case name shown in the figure title")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(THESIS_RC)

    frames = {label: load_arm(Path(path)) for label, path in args.arm}
    for arm, seeds in frames.items():
        print(f"{arm}: {len(seeds)} seed(s), {seeds[0]['scenario'].nunique()} scenarios x {seeds[0]['bus'].nunique()} buses")

    # bus-level stats + figure
    stats = residual_stats(frames)
    residual_stats_latex(stats, args.out_dir / "error_metrics.tex")
    print("\nBus-level stats:")
    print(stats.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    plot_bus_level(frames, args.focus, args.out_dir / "residual_hist_h1.pdf", args.case_label)

    # scenario-level net + figure (DC sanity: net ~ -losses, shifted copy of AC arm's net)
    nets = net_per_scenario(frames)
    nstats = net_stats(nets)
    net_stats_latex(nstats, args.out_dir / "net_imbalance.tex")
    print("\nNet-imbalance stats (signed sum over all buses per scenario):")
    print(nstats.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    plot_net(nets, args.out_dir / "net_imbalance_h1.pdf")


if __name__ == "__main__":
    main()
