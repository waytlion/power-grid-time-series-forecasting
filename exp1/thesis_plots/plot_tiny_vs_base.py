"""

Results Chapter: Tiny vs base E2E comparison figure: 3x3 small-multiples grid.

  rows    = metric (opt gap %, active residual MW, reactive residual MVAr)
  cols    = network (14 / 118 / 500 bus)
  x-axis  = forecast horizon (h1, h6, h24), per-panel y-scale (absolute units)
  base    = single-seed line (s0)
  tiny    = mean over seeds s0..sN + shaded min-max band

Reads mlflow eval CSVs straight from the experiment dir (identity from run name,
network from the *_metrics.csv filename so case500 / case500_goc unify).
Run: python plot_tiny_vs_base.py --exp-dir <mlruns_root>/<experiment_id> --out <path.pdf>
"""

import argparse
import re
from collections import defaultdict
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# metric CSV rows -> (short label, unit) for the three quantities of interest
METRICS = [
    ("Mean Optimality Gap", "Optimality gap", "%"),
    ("Residual P (MAE)", "Active-power residual", "MW"),
    ("Residual Q (MAE)", "Reactive-power residual", "MVAr"),
]
# network display order + pretty title, keyed on the CSV case string
NETWORKS = [("case14_ieee", "14-bus"), ("case118_ieee", "118-bus"), ("case500_goc", "500-bus")]
HORIZONS = [1, 6, 24]
COLORS = {"base": "#0072B2", "tiny": "#D55E00"}   # Okabe-Ito CVD-safe pair


def collect(exp_dir):
    """-> {(network_case, model, horizon): {metric_row: [values over seeds]}}."""
    data = defaultdict(lambda: defaultdict(list))
    for csv in glob(str(Path(exp_dir) / "*" / "artifacts" / "test" / "*_metrics.csv")):
        run_dir = Path(csv).parents[2]
        name_f = run_dir / "tags" / "mlflow.runName"
        if not name_f.exists():
            continue
        run = name_f.read_text().strip()
        # only the two clean E2E arms; drop ablations (_no_temporal) and split-ratio variants
        m = re.fullmatch(r"(base|tiny)_e2e_.+_h(\d+)_s(\d+)", run)
        if not m or "_no_temporal" in run or run.startswith("10_10_80"):
            continue
        model, horizon = m.group(1), int(m.group(2))
        network = Path(csv).name.replace("_metrics.csv", "")   # case500 & case500_goc -> case500_goc
        df = pd.read_csv(csv)
        vals = dict(zip(df["Metric"].astype(str).str.strip(),
                        pd.to_numeric(df["Value"], errors="coerce")))
        for row, _, _ in METRICS:
            if row in vals and pd.notna(vals[row]):
                data[(network, model, horizon)][row].append(vals[row])
    return data


def series(data, network, model, metric_row):
    """(x positions, mean, lo, hi) across horizons; x = true horizon value (to scale)."""
    xs, mean, lo, hi = [], [], [], []
    for h in HORIZONS:
        v = data.get((network, model, h), {}).get(metric_row)
        if not v:
            continue
        xs.append(h)
        mean.append(float(np.mean(v)))
        lo.append(float(np.min(v)))
        hi.append(float(np.max(v)))
    return xs, mean, lo, hi


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", required=True, help="mlflow experiment dir (children are runs)")
    p.add_argument("--out", type=Path, default=Path("tiny_vs_base.pdf"))
    args = p.parse_args(argv)

    data = collect(args.exp_dir)

    # Computer Modern to match the article-class thesis body font (no LaTeX needed;
    # cmr10 ships with matplotlib and is embedded into the SVG for \includesvg).
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150,
                         "font.family": "serif", "font.serif": ["cmr10", "Computer Modern Roman"],
                         "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
                         "axes.unicode_minus": False})
    nrow, ncol = len(METRICS), len(NETWORKS)
    # sharey='row' -> one common y-scale per metric across the three networks
    fig, axes = plt.subplots(nrow, ncol, figsize=(9, 7.5), sharex=True, sharey="row")

    row_top = [0.0] * len(METRICS)   # shared per-metric y ceiling (incl. band max)
    for r, (row, mlabel, unit) in enumerate(METRICS):
        for c, (net, ntitle) in enumerate(NETWORKS):
            ax = axes[r][c]
            for model in ("base", "tiny"):
                xs, mean, lo, hi = series(data, net, model, row)
                if not xs:
                    continue
                row_top[r] = max(row_top[r], max(hi))
                col = COLORS[model]
                ax.plot(xs, mean, "-o", color=col, ms=5, lw=2,
                        label=model, zorder=3)
                if model == "tiny" and any(h > l for h, l in zip(hi, lo)):
                    ax.fill_between(xs, lo, hi, color=col, alpha=0.18, lw=0, zorder=1)
            if r == 0:
                ax.set_title(ntitle)
            if c == 0:
                ax.set_ylabel(f"{mlabel}\n[{unit}]")
            # horizon to scale: h1/h6 sit close, h24 far (true numeric x)
            ax.set_xticks(HORIZONS)
            ax.set_xticklabels([f"h{h}" for h in HORIZONS])
            ax.set_xlim(0, max(HORIZONS) + 1)
            ax.grid(axis="y", color="0.9", lw=0.6)
            ax.set_axisbelow(True)
    # shared y per metric row, anchored at 0 (headroom for the top marker/band)
    for r in range(nrow):
        axes[r][0].set_ylim(0, (row_top[r] or 1.0) * 1.08)
    for c in range(ncol):
        axes[-1][c].set_xlabel("Forecast horizon")

    handles = [plt.Line2D([], [], color=COLORS[m], marker="o", lw=2, label=lab)
               for m, lab in (("base", "base"), ("tiny", "tiny (mean, min-max band)"))]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    # .svg for \includesvg, .png for quick preview
    for ext in (".svg", ".png"):
        fig.savefig(args.out.with_suffix(ext), bbox_inches="tight")
    print(f"wrote {args.out} (+ .svg, .png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
