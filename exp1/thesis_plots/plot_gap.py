"""One-shot: corrected_gap.csv -> gap_slide.{pdf,svg} (Q2), run from repo root.
  - rows = rebalanced gap (over-supply refunded) / no-refund gap (over-supply paid): they bracket the real cost
  - cols = case, x = horizon, one line per optimizer, best forecaster per metric (same rule + layout as rmse_slide)
  - raw gap left out: under-supply looks cheap before rebalancing (biased toward under-supplying arms)
"""
import pandas as pd
import matplotlib.pyplot as plt

from plot_rmse import (CASE_LABELS, CASE_ORDER, SLIDE_RC, best_forecaster_per_metric, finish_slide,
                       optimizer_handles, optimizer_lines, style_slide_axes)

IN = "exp1/results/combined/corrected_gap.csv"
ROWS = [("Rebalanced Gap", "Rebalanced gap [%], log"), ("No-refund Gap", "No-refund gap [%], log")]


def main():
    df = pd.read_csv(IN)
    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(len(ROWS), len(CASE_ORDER), figsize=(10, 5.8), sharex=True)
        for r, (metric, ylabel) in enumerate(ROWS):
            best = best_forecaster_per_metric(df, metric)
            for c, case in enumerate(CASE_ORDER):
                optimizer_lines(axes[r][c], best[best["case"] == case], tag=c == 0)
            axes[r][0].set_ylabel(ylabel)
        for c, case in enumerate(CASE_ORDER):
            axes[0][c].set_title(CASE_LABELS[case])
        style_slide_axes(axes)
        finish_slide(fig, optimizer_handles(), "gap_slide")


if __name__ == "__main__":
    main()
