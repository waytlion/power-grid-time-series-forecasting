"""One-shot: dispatch_diag step1 outputs -> Q1 "why" slides, run from repo root.
  - input: exp1/results/_dispatch_diag/<case>/step1_{split.csv,samples.parquet}
    (copied from <data>/2019-2021/phase_1c/dispatch_diag_final)
  - arms: AC with the best forecaster by Pg RMSE (same rule as rmse_slide) vs E2E (seed 0); cols = case
  a) violin_slide: per test sample, RMSE across buses (Pd) / generators (Pg), AC vs E2E; violins = lowest 99 % of
     samples (long upper tail would flatten the bodies), ◆ = overall RMSE over all samples (= rmse_slide, E2E seed 0)
     - per sample, not per bus: per-bus/-gen RMSE would be dominated by zero-load buses and pinned units (AC exact 0)
  b) corr_slide: load-bus Pd error correlation matrix (2-step, CORR_CASE at CORR_H)
  0) totals_slide: RMSE over test samples of each arm's own summed Pd error (hatched) vs summed Pg error (solid),
     AC (best forecaster) vs E2E; AC: Pg = Pd + loss change (+2–9 %), E2E: Pg below its own Pd
  0b) control_totals_slide: control arm, input exp1/results/_e2e_pd/phaseB_<case>_h1-6-24.csv (e2e_pd_opf.py table):
     E2E's summed Pd error (hatched) vs the summed Pg error of its Pd → AC-OPF route and of its own Pg output
     (identical Pd)
  1) volume_alloc_example: 3 generators (true 10 MW each) -> Σ_g |ΔPg_g| = volume + allocation, schematic
  2) volume_alloc_slide (absolute errors, the split is exact for |.| only): per test sample, mean MW, x = horizon
     - Σ_g |ΔPg_g| = volume |Σ_g ΔPg_g| + allocation Σ_g |ΔPg_g| − |Σ_g ΔPg_g| (exact split per sample)
     - stacked bar per arm: volume (solid) + allocation (hatched)
  3) total_scatter_slide: per test sample at SCATTER_H, signed Σ_i ΔPd_i (x) vs Σ_g ΔPg_g (y), y = x dashed
     - 2-step lies on y ≈ x by construction (ΣPg = ΣPd + losses)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from colors import ARM_COLORS, OPT_LABELS
from plot_rmse import h_label, CASE_LABELS, CASE_ORDER, EXCLUDED_FC, HORIZONS, SLIDE_RC, X_OF, save

IN_DIR = Path("exp1/results/_dispatch_diag")
W = 0.34  # bar width
SCATTER_H = 1  # E2E decoupling is strongest at t+1
CORR_CASE, CORR_H = "case118_ieee", 6
INK = "0.35"  # schematic: no arm hue
LABELS = {o: OPT_LABELS[o] for o in ("ac", "e2e")}


def load():
    """best AC arm per horizon (lowest Pg RMSE) + E2E, per available case"""
    out = {}
    for case in CASE_ORDER:
        f = IN_DIR / case / "step1_split.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f)
        ac = s[s["arm"].str.startswith("ac_") & ~s["arm"].str[3:].isin(EXCLUDED_FC)]
        best = ac.loc[ac.groupby("H")["pg_rmse"].idxmin()].assign(opt="ac")
        out[case] = pd.concat([best, s[s["arm"] == "e2e"].assign(opt="e2e")])
    return out


def arm_of(d, opt, h):
    return d[(d["opt"] == opt) & (d["H"] == h)]["arm"].iloc[0]


def plot_violins(data, q=100, log=False, name="violin_slide"):
    """q: % of samples in the violin bodies (lowest q %); log: violins on log10 values (heavy upper tail)"""
    f = (lambda v: np.log10(np.clip(v, 1e-3, None))) if log else (lambda v: v)
    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(2, len(data), figsize=(10, 6.2), squeeze=False)
        for c, (case, d) in enumerate(data.items()):
            smp = pd.read_parquet(IN_DIR / case / "step1_samples.parquet")
            for r, col in enumerate(("pd_rmse", "pg_rmse")):
                ax, lo, hi = axes[r][c], np.inf, -np.inf
                for opt, dx in (("ac", -0.18), ("e2e", 0.18)):
                    vals = [smp[(smp["H"] == h) & (smp["arm"] == arm_of(d, opt, h))][col].to_numpy() for h in HORIZONS]
                    pos = [X_OF[h] + dx for h in HORIZONS]
                    body = [f(v[v <= np.percentile(v, q)]) for v in vals]
                    vp = ax.violinplot(body, pos, widths=0.32, showmedians=True, showextrema=False)
                    for poly in vp["bodies"]:
                        poly.set(facecolor=ARM_COLORS[opt], edgecolor=ARM_COLORS[opt], alpha=0.6)
                    vp["cmedians"].set(color="0.15", lw=1.2)
                    ax.scatter(pos, [f(np.sqrt((v ** 2).mean())) for v in vals], marker="D", s=22, color="white",
                               edgecolor="0.15", zorder=3)
                    lo, hi = min(lo, *(b.min() for b in body)), max(hi, *(b.max() for b in body))
                if log:
                    ticks = np.arange(np.floor(lo), np.ceil(hi) + 1)
                    ax.set_yticks(ticks)
                    ax.set_yticklabels([f"{10 ** t:g}" for t in ticks])
                    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
                else:
                    ax.set_ylim(0, hi * 1.05)
                ax.set_xticks(range(len(HORIZONS)))
                ax.set_xticklabels([f"{h_label(h)}" for h in HORIZONS])
                ax.grid(axis="y", color="0.9", lw=0.6)
                ax.set_axisbelow(True)
            axes[0][c].set_title(CASE_LABELS[case])
        scale = ", log" if log else ""
        axes[0][0].set_ylabel(f"$P_d$ RMSE across buses\nper test sample [MW{scale}]")
        axes[1][0].set_ylabel(f"$P_g$ RMSE across generators\nper test sample [MW{scale}]")
        share = f" ({q} % of samples)" if q < 100 else ""
        handles = [plt.Rectangle((0, 0), 1, 1, color=ARM_COLORS[o], alpha=0.6, label=LABELS[o] + share) for o in LABELS]
        handles += [plt.Line2D([], [], color="0.15", lw=1.2, label="median"),
                    plt.Line2D([], [], ls="", marker="D", mfc="white", mec="0.15", label="overall RMSE")]
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.03))
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save(fig, name)


def plot_corr(data):
    mats = np.load(IN_DIR / CORR_CASE / "step3_corr_mats.npz")
    fc = arm_of(data[CORR_CASE], "ac", CORR_H)[3:]
    with plt.rc_context(SLIDE_RC):
        fig, a = plt.subplots(figsize=(5.2, 4.4))
        im = a.imshow(mats[f"{fc}_h{CORR_H}"], cmap="RdBu_r", vmin=-1, vmax=1)
        a.set_xticks([])
        a.set_yticks([])
        a.set_xlabel("load bus")
        a.set_ylabel("load bus")
        a.set_title(f"2-step $P_d$ error correlation\n{CASE_LABELS[CORR_CASE]}, {h_label(CORR_H)}")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
        fig.tight_layout()
        save(fig, "corr_slide")


def plot_example():
    true, pred = np.array([10.0, 10.0, 10.0]), np.array([12.0, 9.0, 11.0])
    err = pred - true
    vol, tot = abs(err.sum()), np.abs(err).sum()
    with plt.rc_context(SLIDE_RC):
        fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw=dict(width_ratios=[3, 2]))
        x = np.arange(len(pred))
        a.bar(x, pred, 0.55, color="0.8", label="forecast")
        for i, (t, p, e) in enumerate(zip(true, pred, err)):
            a.plot([i - 0.33, i + 0.33], [t, t], color="0.1", lw=2, label="true" if i == 0 else None)
            a.annotate("", (i, p), (i, t), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
            a.text(i + 0.32, (t + p) / 2, f"{e:+.0f}", va="center", fontsize=12)
        a.set_xticks(x)
        a.set_xticklabels([f"G{i + 1}" for i in x])
        a.set_ylim(0, 14)
        a.set_ylabel("$P_g$ [MW]")
        a.legend(loc="upper right", frameon=False)
        a.text(0.02, 0.97, f"$\\Sigma_g \\Delta P_g$ = {err.sum():+.0f} MW", transform=a.transAxes, va="top")

        b.bar(0, vol, 0.5, color=INK)
        b.bar(0, tot - vol, 0.5, bottom=vol, color="white", edgecolor=INK, hatch="////", lw=0.8)
        b.text(0.32, vol / 2, f"volume $|\\Sigma_g \\Delta P_g|$ = {vol:.0f} MW", va="center")
        b.text(0.32, vol + (tot - vol) / 2, f"allocation $\\Sigma_g |\\Delta P_g| - |\\Sigma_g \\Delta P_g|$ = "
               f"{tot - vol:.0f} MW", va="center")
        b.text(0, tot + 0.15, f"$\\Sigma_g |\\Delta P_g|$ = {tot:.0f} MW", ha="center", va="bottom")
        b.set_xlim(-0.4, 2.2)
        b.set_ylim(0, tot * 1.25)
        b.set_xticks([])
        b.set_ylabel("[MW]")
        fig.tight_layout()
        save(fig, "volume_alloc_example")


def plot_totals(data):
    w = W / 2
    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(1, len(data), figsize=(10, 3.9), squeeze=False)
        for ax, (case, d) in zip(axes[0], data.items()):
            for opt, dx in (("ac", -W / 2 - 0.02), ("e2e", W / 2 + 0.02)):
                r = d[d["opt"] == opt].set_index("H").reindex(HORIZONS)
                x = np.array([X_OF[h] + dx for h in HORIZONS])
                ax.bar(x - w / 2, r["dD_rmse_mw"], w, facecolor="white", edgecolor=ARM_COLORS[opt], hatch="////",
                       lw=0.8)
                ax.bar(x + w / 2, r["total_rmse_mw"], w, color=ARM_COLORS[opt])
            ax.set_title(CASE_LABELS[case])
            ax.set_xticks(range(len(HORIZONS)))
            ax.set_xticklabels([f"{h_label(h)}" for h in HORIZONS])
            ax.grid(axis="y", color="0.9", lw=0.6)
            ax.set_axisbelow(True)
        axes[0][0].set_ylabel("RMSE of summed error [MW]")
        handles = [plt.Rectangle((0, 0), 1, 1, color=ARM_COLORS[o], label=LABELS[o]) for o in LABELS]
        handles += [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="0.3", hatch="////",
                                  label="summed $P_d$ error"),
                    plt.Rectangle((0, 0), 1, 1, color="0.45", label="summed $P_g$ error")]
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.04))
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        save(fig, "totals_slide")


def plot_control_totals():
    """bars per horizon: E2E summed Pd error | Pg via AC-OPF on that Pd | E2E Pg output (route's samples for Pd)"""
    files = {c: Path(f"exp1/results/_e2e_pd/phaseB_{c}_h1-6-24.csv") for c in CASE_ORDER}
    files = {c: f for c, f in files.items() if f.exists()}
    w = 0.26
    bars = [("e2e_pd_ac", "dD_rmse_mw", dict(facecolor="white", edgecolor=ARM_COLORS["e2e"], hatch="////", lw=0.8),
             "E2E summed $P_d$ error"),
            ("e2e_pd_ac", "total_rmse_mw", dict(color=ARM_COLORS["ac"]),
             "summed $P_g$ error: E2E $P_d \\rightarrow$ AC-OPF"),
            ("e2e", "total_rmse_mw", dict(color=ARM_COLORS["e2e"]), "summed $P_g$ error: E2E $P_g$ output")]
    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(1, len(files), figsize=(10, 3.9), squeeze=False)
        for ax, (case, f) in zip(axes[0], files.items()):
            s = pd.read_csv(f).set_index(["arm", "H"])
            for i, (arm, col, style, _) in enumerate(bars):
                ax.bar([X_OF[h] + (i - 1) * w for h in HORIZONS], [s.loc[(arm, h), col] for h in HORIZONS], w * 0.92,
                       **style)
            ax.set_title(CASE_LABELS[case])
            ax.set_xticks(range(len(HORIZONS)))
            ax.set_xticklabels([f"{h_label(h)}" for h in HORIZONS])
            ax.grid(axis="y", color="0.9", lw=0.6)
            ax.set_axisbelow(True)
        axes[0][0].set_ylabel("RMSE of summed error [MW]")
        handles = [plt.Rectangle((0, 0), 1, 1, **style, label=label) for _, _, style, label in bars]
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.04))
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        save(fig, "control_totals_slide")


def plot_bars(data):
    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(1, len(data), figsize=(10, 3.9), squeeze=False)
        for ax, (case, d) in zip(axes[0], data.items()):
            for opt, dx in (("ac", -W / 2 - 0.02), ("e2e", W / 2 + 0.02)):
                r = d[d["opt"] == opt].set_index("H").reindex(HORIZONS)
                x = [X_OF[h] + dx for h in HORIZONS]
                ax.bar(x, r["total_abs_mw"], W, color=ARM_COLORS[opt])
                ax.bar(x, 2 * r["misplaced_mw"], W, bottom=r["total_abs_mw"], color="white",
                       edgecolor=ARM_COLORS[opt], hatch="////", lw=0.8)
            ax.set_ylim(0, (d["total_abs_mw"] + 2 * d["misplaced_mw"]).max() * 1.08)
            ax.set_title(CASE_LABELS[case])
            ax.set_xticks(range(len(HORIZONS)))
            ax.set_xticklabels([f"{h_label(h)}" for h in HORIZONS])
            ax.grid(axis="y", color="0.9", lw=0.6)
            ax.set_axisbelow(True)
        axes[0][0].set_ylabel("mean error per sample [MW]")

        handles = [plt.Rectangle((0, 0), 1, 1, color=ARM_COLORS["ac"], label=f"{LABELS['ac']}: volume"),
                   plt.Rectangle((0, 0), 1, 1, color=ARM_COLORS["e2e"], label="E2E: volume"),
                   plt.Rectangle((0, 0), 1, 1, fc="white", ec="0.3", hatch="////", label="allocation")]
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.04))
        fig.text(0.99, -0.02, "volume $|\\Sigma_g \\Delta P_g|$, "
                 "allocation $\\Sigma_g|\\Delta P_g| - |\\Sigma_g \\Delta P_g|$; "
                 f"{LABELS['ac']}: best forecaster by $P_g$ RMSE; E2E: seed 0", ha="right", fontsize=8, color="0.35")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        save(fig, "volume_alloc_slide")


def plot_scatter(data):
    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(1, len(data), figsize=(10, 4.0), squeeze=False)
        for ax, (case, d) in zip(axes[0], data.items()):
            smp = pd.read_parquet(IN_DIR / case / "step1_samples.parquet")
            lim = 0.0
            for opt in LABELS:
                arm = d[(d["opt"] == opt) & (d["H"] == SCATTER_H)]["arm"].iloc[0]
                s = smp[(smp["H"] == SCATTER_H) & (smp["arm"] == arm)]
                ax.scatter(s["dD_mw"], s["total_mw"], s=4, alpha=0.25, color=ARM_COLORS[opt], lw=0, rasterized=True)
                lim = max(lim, np.percentile(np.abs(s[["dD_mw", "total_mw"]]), 99.5))
            ax.plot([-lim, lim], [-lim, lim], "--", color="0.4", lw=1)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.set_title(f"{CASE_LABELS[case]}, {h_label(SCATTER_H)}")
            ax.set_xlabel("summed $P_d$ error $\\Sigma_i \\Delta P_{d,i}$ [MW]")
        axes[0][0].set_ylabel("summed $P_g$ error $\\Sigma_g \\Delta P_g$ [MW]")
        handles = [plt.Line2D([], [], ls="", marker="o", color=ARM_COLORS[o], label=LABELS[o]) for o in LABELS]
        handles.append(plt.Line2D([], [], ls="--", color="0.4", label="$y = x$"))
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.04))
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        save(fig, "total_scatter_slide")


def main():
    plot_example()
    data = load()
    plot_violins(data)
    plot_violins(data, log=True, name="violin_slide_log")
    plot_corr(data)
    plot_totals(data)
    plot_control_totals()
    plot_bars(data)
    plot_scatter(data)


if __name__ == "__main__":
    main()
