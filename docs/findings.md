# Findings — Thesis_Repo (data gen, forecasters, exact/2-step arms, aggregation)

Measured results, retractions, and implementation rationale. Dated; newest facts win.
RETRACTED entries stay so they are not re-proposed. ML-arm findings: `~/gridfm-graphkit-dev/docs/findings.md`.

---

## 1. Cross-arm methodology (state these; they cut against the ML arms)

- **Splits**: forecasters + E2E base = 70/15/15; surrogate + E2E tiny = 80/10/10. Chronological — `temporal_split`
  is checked BEFORE `split_by_load_scenario_idx` in the datamodule branch.
  - 2026-11-13 (audit F1.1, decision mine): surrogate runs up to 2026-10-17 = 80/10/10 (val hours 21024–23651 overlap the
    test period 22338–26279; checkpoint selection only). Surrogate config switched to 70/15/15 (= forecasters: train
    0–18395, val 18396–22337); retrain + re-eval pending, thesis text already describes 70/15/15.
- **HPO**: graphkit has none (no optuna/ray); the GNN val split only drives ReduceLROnPlateau + best-checkpoint
  selection. XGB gets 20 Optuna trials on val RMSE.
- Horizon > 1 → fewer forecast starts (test idx 85..99: h=1 → 15 starts, h=6 → 10).
- Qd is never forecast by the baselines: derived from Pd via a fixed per-bus ratio (`phase1c_transform_forecasts.py`).

## 2. Timing semantics (2026-09-23)

- "Per sample" = per SCENARIO (one dispatch decision) in every arm.
  - AC/DC: datakit per-scenario solve time (`runtime_data.parquet`, 'ac'/'dc' columns) → mean / sum.
  - Surrogate/E2E: model forward only (CUDA-synced); divisor `_infer_samples` (surrogate num_graphs; E2E B × n).
    E2E CSVs before 2026-09-23 divided by windows → inflated ×h at h>1 (h=1 fine); needs E2E test-only re-eval.
  - Forecaster: `*_infer_per_scenario` = forecaster time / number of test windows — not yet verified to equal per
    dispatch at h>1.
- aggregate_results emits `opf_step_*` and `full_pipeline_*` (forecast + OPF), each `_infer_s` and
  `_infer_total_min`; table4 uses `full_pipeline_*` for both columns. Timing is batched GPU throughput (ML) vs
  sequential CPU per-solve (AC/DC).
- Surrogate GNN training time is NOT logged → surrogate "Train" column = forecaster training time only.
- Case for timing joins comes from the `<network>` path component, so several networks under one
  `--datakit-out-dir` aggregate correctly.
- 2026-11-14, H=1 timing per dispatch decision (combined CSV of 2026-11-14; ML = batched GPU throughput, AC/DC =
  sequential CPU solves; range over the load forecasts fed to the OPF), case14 / case118 / case500:
  - OPF step: AC 9 ms / 0.10–0.14 s / 0.79–1.08 s; DC 2 / 12–14 / 55–73 ms; surrogate forward 0.6–2.1 / 0.9–1.9 /
    1.4 ms. E2E incl. its forecast: 2.0 / 14.6 / 47.2 ms → vs one AC solve ≈ 4–5× / 7–9× / 17–23× faster (grows with
    grid size); comparable to DC, slower than the surrogate's forward pass.
  - Forecast step (full pipeline − OPF step): XGB 16 / 3 / 4 ms, SARIMA 72 / 397 / 2634 ms (dominates its pipelines).
  - Training (min): E2E 69 / 393 / 1527; XGB 0.9 / 5.9 / 20; SARIMA 0.4 / 1.2 / 4.3; surrogate not logged.
  - Hardware (sbatch configs + ZIH compendium): AC/DC solves on Barnard (exclusive node, 2× Intel Xeon Platinum 8470,
    104 cores; phase1c.sbatch); E2E train + eval on Capella with 4× H100-SXM5 (graphkit train.sh, --gres=gpu:4);
    surrogate train + eval on 1× H100 (train_surrogate.sh, phase1c_surrogate_eval.sbatch, --gres=gpu:1).

## 3. Forecasters (phase 1b)

- **XGB full-OPF-state lags** (`--xgb_input_all_opf_var`): Pd, Qd, Qg, Vm, Va lags instead of Pd-only; target stays
  Pd. case118 h1, fixed params: RMSE 0.903 → 0.732 MW (~−19%), MAE ~−5%. TinyTGT skipped under the flag (fixed
  `in_feat`); SARIMA/SNaive unaffected. Writes to the same output paths as baseline (overwrites if run sequentially).
  Train vs inference feature vectors verified byte-identical. SARIMAX variant deliberately skipped (would need unknown
  future OPF outputs as exog).
- **XGB calendar features** (`--no-time-feats`, case118 h24, 2026-08-11): with feats RMSE 2.199 / MAE 1.081, without
  2.239 (+1.8%) / 1.064 (−1.6%). h=1: dropping helps both; from h~3 RMSE worsens without them (+2.9% by h20-22), MAE
  flat-to-better → feats damp long-horizon tail errors. Cost ~19-23% more XGB time. Caveat: each arm ran its own
  20-trial HPO → rerun with shared params / seeds before citing.
- **`--sarima-fit-hours` default changed 17520 → 26280**: all reported results used 17520 ("2.0 years" in logs).
  26280 ≥ the ~18.4k-hour training split → new default = all training data. Re-run or write "most recent two years".
- **phase1_baseline dead code**: `SNaiveModel` and `GlobalFitLocalApplySARIMA.predict()` are never called. Live paths:
  inline `pred_naive = full_data[t-lag : t-lag+hor, :, 0].T` (lag 48) and `_infer_single_bus` in evaluation.py. Real
  SARIMA fallback is post-hoc (NaN → SNaive at aggregation); it never fired (0 "Fit failed for bus", 0
  ConvergenceWarnings across case14/118/500 at h1).
- Known bug (2026-07-20, on main): xgboost 3.2.0 returns 1-D pred for h=1 single-output → `pred_xgb[:, s:e]` in
  run_evaluation (~l.277) IndexErrors at h=1. Status unverified since.
- **XGB + total-load lags** (`--xgb_total_load_lags`, 2026-10-14, jobs 4417734-63, `<data>/xgb_totlag/report.md`):
  one extra lag channel D_t = bus-mean Pd (≡ ΣPd, same scaler), broadcast to every bus; target stays per-bus Pd.
  Train vs inference features verified byte-identical (job 4417613). Arms: A1 = + D lag, A0's fixed params
  (isolates the feature); A2 = + D lag, own 20-trial HPO seed 42. A0r (refit/rerun, seed 42) reproduces the
  original h1 logs exactly on all 3 cases (byte-level RMSE/MAE match); h6/h24 original Optuna logs are gone, so
  A0r there is itself a fresh seeded run (becomes the fixed-param source for A1).
  - Per-bus Pd RMSE −5 to −18% vs A0 (largest at h1: case118 −16%, case500 A2 −18%).
  - Total-demand (ΣΔPd) MAE vs A0: **case118 h1 −17%, h6 −11%** (clears the −10% trigger); case500 h1 A2 −15%
    (clears it too); case500 h6/h24 and case14 all H stay under 10% (case14 h24 even +5% with A2 — HPO
    overfit on a tiny net). case118 has no valid Pg proxy (findings §9: proxy noise > curvature estimate there)
    but shows the same pattern on the total-error gap-closure analog: 27–39% at every H.
  - Gap closure to E2E, plug-in Pg-RMSE proxy (case14/case500 only, findings §9): case500 h1 A2 **157%**
    (A2's proxy Pg RMSE 3.12 MW < E2E's own Pg RMSE 3.28 MW — total-load info alone surpasses E2E's total-demand
    edge there) and case500 h24 A1 26%; case14 stays ≤17% everywhere (small net, Pg ≈ f(D) affine → total-error
    cuts barely move Pg, findings §9).
  - A1 == A2 at 6 of 9 case×H (Optuna's own search landed on A0r's trial) → HPO only matters at case500 h1
    (A2 clearly beats A1's reused params there), not a formality elsewhere.
  - Verdict: **trigger hit** on both legs (−10% total-demand at case118 h1/h6 + case500 h1; ≥25% proxy-Pg
    closure at case500 h1/h24). Recommend phase 1c for case118 (all H) and case500 (all H, A1 and A2); case14
    optional. Phase 1c NOT started — user's call (see ~/CLAUDE.md "Current mission", open decision now answered).
- **Information-sufficiency check: is total load (PC1) enough?** (2026-10-16, jobs 39760557-59,
  `<data>/xgb_totlag/pc_check.md`): PCA of per-bus Pd (train-fit, scores at lags origin/origin−1/origin−24);
  regress A2's (and A0's) errors on PC1 vs PC1-3 vs PC1-5, OOS R² (fit first half of test, score second half vs
  train-mean baseline). Verdict rule: gain of PC2-5 over PC1 ≤0.01 (per bus) / ≤0.02 (total) → sufficient.
  - **Not uniformly sufficient**: case14 h1 fails clearly (gain_bus 0.029–0.057, gain_tot 0.031–0.060, both
    PC1-3 and PC1-5, same pattern in A0) — real extra signal beyond total load there. case118 h1 (per-bus,
    0.018–0.024) and a few case500/case14 h6/h24 cells sit at/just past the bar.
  - **Caveat (the check itself is noisy, not just the signal)**: many cells are negative, some large
    (case118 h24 PC1-5 gain_bus −0.20) — 15-feature OOS fits overfitting on a short test-half, not evidence the
    extra PCs hurt. Only the case14 h1 result is unambiguous; elsewhere read as inconclusive, not "confirmed
    sufficient". Diagnostic only (uses the test split) — not a basis for model/feature selection.
  - Net: total-load lags (A2) capture most but not all exploitable cross-bus structure; the leftover is modest
    and case-dependent. Doesn't change the phase 1c recommendation above.

## 4. Data generation (phase 1a)

- **case500_goc scale sweep never converged** (2026-07-18): in-service gens have sum(PMIN) = 7,526 MW; at SCALE=0.2
  demand ranged 1,241-3,437 MW → infeasible regardless of solver. Lower SCALE makes it worse. Necessary-condition
  window SCALE ∈ [1.2132, 1.3560] (demand between sum(PMIN) and sum(PMAX)=23,304 MW); not sufficient (tight
  thermal/voltage limits). Also fixed: `local error_count=...` outside a function in `phase1a_sweep_scale.sbatch`.
- **3yr (2019-2021) and 24yr (2002-2025) datasets are different realizations**: 0/300 sampled 3yr load vectors match
  any 24yr scenario; 3yr ~30% hotter (mean 2263 vs 1732 MW). Likely a data-gen scaling/config difference — unchecked.

## 5. DC-OPF on case118 (Pg/Va RMSE outlier)

- case118 DC Pg RMSE ~4.9× AC (46.7 vs 9.6 MW), Va RMSE up to ~70×; case14/case500 only ~1.1×.
  `Generator Pg RMSE` pools raw MW over all (gen, scenario) pairs, so a few units dominate.
- RETRACTED (2026-09-22): "DC degeneracy impossible because pglib costs are quadratic". Raw data (2026-09-23): c2 = 0
  for ALL case118 gens → DC-OPF is an LP; marginal units' c1 within ~5%.
- 2026-09-23, h1 xgb: top-8 gens = 99.9% of DC sum-sq; errors systematic (gen39 +212 MW, gen36 −143, gen4 −47, gen24
  −58, gen29 slack −36). Mean total DC−true = −53 MW (lossless shortfall, only part of it).
- Va is NOT a reference-offset artifact: REF angle 0 in AC and DC; RMSE 7.41 → 7.18 after de-meaning.
- 2026-10-04 cross-case (h6 xgb): losses similar everywhere (case14 3.7%, case118 2.2%, case500 1.7% of gen) → not
  the explanation. Amplifier = cost structure: case14 one dispatchable unit; case118 linear costs with gen36 vs gen39
  c1 0.02% apart (~500-640 MW units) → ~200 MW shifts; case500 88/224 quadratic gens → smooth.
- Check 4 (2026-10-04, case118 h1): line 127 (IEEE 77-82, 141 MW) binds in 78% of DC scenarios, never in AC (mean 38%
  loading). gen36 DC−AC = −182 MW when bound, −8 otherwise. DC-only binding lines: case14 0, case500 0/733, case118 1.
  DC Pg RMSE (Pmax>0): case118 83.5 bound / 50.3 not; case500 8.9; case14 6.6.
- Not a bug: DC on TRUE load gives Pg RMSE 76.9 MW (intrinsic model); DC self-consistent (ΣPg = ΣPd, all flows ≤
  rate). PTDF on AC's own dispatch puts line 127 at 61 MW → AC dispatch is DC-feasible; DC deviates by objective
  (maxes cheap remote 36/39 until line 127 saturates). Cost impact small (gap DC 3.7% vs AC 0.8% at h1).
- Open: whether AC avoids the 36/39 shift because of locational losses or V/Q limits.
- Script: `exp1/thesis_plots/plot_dpg_by_gen.py`. Raw parquets:
  `<data>/2019-2021/phase_1c/data_out/<case>_horizon_<H>/<forecaster|true>/<case>/raw/{gen,bus}_data.parquet`.

## 6. Residual figures (phase 2)

- 2-step validated (case118 h1 xgb): AC forecast-space ≈ 0; AC realized MAE == forecast MAE (0.31 MW); DC modeling
  bias −0.89 MW/bus (≈ system losses when summed). Exact-AC realized residual = per-bus forecast error.
- Design (per TSO + colleague input): absolute MW, no peak-load normalization, no ε-floor, all buses; 1 MW line =
  practitioner heuristic (PSSE ~1 MW/1 MVAr ≈ 0.01 p.u.), not a rule; quantile-first; per-scenario net imbalance.

## 7. Ex-post AC-PF feasibility (2026-10-10, `exp1/generate_metrics/expost_pf.py`)

- Alignment (verified to float32): exact arm flat scenario s → origin 22338 + s//H, step s%H. Surrogate: dump s →
  eval-set `test[s]` (`artifacts/stats/<case>_scenario_splits.json`), then same flattening. E2E: lsi =
  `test[s//H] + 48 + s%H` (test order shuffled).
- (PowerModels path, replaced 2026-10-12 by pypowsybl, see below) Datakit PF path (Network + pf_preprocessing setters +
  `run_pf(fast=True)`) reproduces GT labels to 1e-6 on
  case14/case118. case500: per-gen Qg differs 22-31 MVAr (PowerModels `_assign_qg!` split) → evaluate Qg at bus level
  against summed limits.
- SUPERSEDED (same day, distributed slack below): 2026-10-12: step 1 changed from proportional pre-scaling to merit-order rebalancing (`metrics.merit_order_rebalance`,
  same rule as §8). The full run of 2026-10-11 (jobs 39467094-6) used proportional scaling → outdated, rerun needed.
- SUPERSEDED (2026-10-12, no oracle Δ any more): Design: Δ = ΣPg_true − ΣPg_pred (oracle scalar = realized system
  requirement); no warm-start retry; run_pf error = non-converged; each arm uses its own timestamps.
- Cost (PowerModels): ~4 ms (case14), 20-100 ms (case118), 80 ms (case500) per PF; ~2.6M solves ≈ 50 CPU-h.
- Julia (now only for the one-time grid correction in `load_net_from_pglib`) needs `module load release/2026 GCC/14.2.0` + `LD_PRELOAD=/software/genoa/r2026/GCCcore/14.2.0/lib64/libstdc++.so.6`.
- Smoke job 39459686 (case118 h6, 200 samples): label + exact_true have 0 violations, ΔP_slack 0 → pipeline valid.
  Qg max violation per sample: exact_xgb ~1 MVAr, surrogate ~88, e2e ~55. Thermal share (~47%) dominated by lines at
  ~1.002 loading → report magnitudes, not share alone.

- 2026-10-12: distributed-slack AC-PF replaces step 1 + PowerModels (pypowsybl 1.14.0 / OpenLoadFlow):
  - Per sample: predicted dispatch P̂ as target P, predicted Vm as generator target_v, true Pd/Qd. The imbalance
    (forecast error + losses) is distributed ∝ Pmax. The PF-solved Pg (slack output included) is the rebalanced
    dispatch P̃, used for both cost (§8) and feasibility. No oracle Δ.
  - Settings (logged to `expost_pf/<case>/run_config_<job>.json`): distributed_slack=True,
    balance_type=PROPORTIONAL_TO_GENERATION_P_MAX, use_reactive_limits=False, slack = case-file REF bus
    (read_slack_bus via the MATPOWER slackTerminal, asserted per solve), twt_split_shunt_admittance=True.
  - Converter: `gridfm_datakit/powsybl/` copied into the datakit fork from upstream a48ad50. Two upstream bugs are
    fixed in the fork:
    (1) `build_p2g_maps` mapped elements by enumeration order. pypowsybl groups buses by substation and lists lines
        before transformers, so generators, branches and loads were mismatched (case118: 179/186 branches wrong,
        250 MW of load never set). Now matched on bus-breaker IDs `BUS-<n>`.
    (2) `update_powsybl` wrote bus v_mag but not generator target_v, so PV setpoints stayed at case-file values
        (|ΔVm| up to 0.10 p.u.).
  - Validation, one label sample (GT dispatch on GT load) vs the PowerModels labels: case14 Vm 6e-10 p.u. /
    Pg 5e-7 MW; case118 Vm 3e-8 p.u. / Pg 2e-4 MW. case118 needs twt_split_shunt_admittance=True; without it the
    error is 1.8e-3 p.u. at bus 86 (transformer 86-87 carries charging B; MATPOWER splits it pi-wise).
    Units that take part get identical ΔP/Pmax (0.013623, case118, −2 % dispatch).
  - OLF defaults, NOT changed:
    (a) limits: useActiveLimits=true caps units at Pmin/Pmax during distribution (−30 % dispatch: 0 units beyond).
        generatorsWithZeroMwTargetAreNotStarted=true excludes units with P̂ ≈ 0 from distribution (P̂ = 1e-5 MW
        counts as zero).
        CORRECTED 2026-10-13: the case14 collapse to single slack is not caused by this parameter. Gen 2 has
        P̂ = −9e-7 MW < Pmin in every sample, so OLF drops it either way (see the 2026-10-13 entry).
        plausibleActivePowerLimit=10000 MW excludes larger units (case118 max Pmax 1182).
    (b) failure: slackDistributionFailureBehavior=FAIL. Status "Outer loop failed: Failed to distribute ..." →
        counted as `slack_not_distributed`. Alternatives: THROW, LEAVE_ON_SLACK_BUS, DISTRIBUTE_ON_REFERENCE_GENERATOR.
        slackBusPMaxMismatch=1 MW: distribution stops below this, and the residual on the slack bus is added to the
        REF unit's Pg.
  - Metrics: ΔP_slack violation replaced by `slack_mismatch_mw` (left on the slack bus) + flag (|·| > 1 MW);
    `pg_lim_n` = dispatchable units at/beyond P limits, `pg_beyond_n` = beyond (> 0.1 MW). Status is one of
    converged / diverged / slack_not_distributed / no_dispatch, counted per arm, case, horizon in corrected_gap.
  - Sanity run: case118 h1, label + AC 2-step (true/xgb/sarima/snaive), n = 3942 each; capella srun, 2 CPUs,
    3 m 49 s in total.
    - label and exact_true: 0 violations, γ_reb 0, ΣP̃ = ΣP̂.
    - Forecast arms: 100 % converged, 0 slack_not_distributed, slack flag 0 %.
    - Thermal violation share 44 / 43 / 50 % (xgb / sarima / snaive).
    - `pg_beyond` share 21 / 25 / 30 % is an ARTIFACT. In 200/200 checked xgb samples it is the REF unit (bus 69)
      with P̂_ref = 0: the unit is "not started", so the < 1 MW residual lands on it as P < Pmin = 0 (−0.2…−0.9 MW).
      CORRECTED 2026-10-13: P̂_REF is 5e-6…2e-5 MW (IPOPT at Pmin = 0, 75 % of xgb samples), not exactly 0.

- 2026-10-13: OLF variants, h1 AC 2-step, all samples. case118 xgb n = 3942; case14 sarima n = 3942 (case14 has no
  xgb phase1c output). Single process, ~14 ms/solve (case118) and ~8 ms/solve (case14).
  - slackBusPMaxMismatch 1 → 0.01 MW:
    - case118: the `pg_beyond` artifact goes away (21.1 % → 0 %, max 1.0 → 0.01 MW); mean |residual| 0.32 → 0.003 MW.
    - γ_reb barely changes: mean |Δ| 0.002 pp, max 0.019 pp. Thermal and Qg are unchanged; runtime +1.5 %.
    - case14: signed γ_reb −0.0037 → −0.0000 %, share < 0 76 % → 51 %. In case14 the 1 MW residual is a visible part
      of |γ_reb| (0.004 → 0.0008 %).
  - generatorsWithZeroMwTargetAreNotStarted → false (case118):
    - Units at P̂ ≈ 0 now take part: units moved per sample 6.5 → 8.8; the REF share of the distributed ΔP 11 → 17 %.
    - signed γ_reb +0.003 → +0.061 %; share γ_reb < 0 49 → 10 %; |γ_reb| 0.069 → 0.064 %; per sample up to 1.15 pp.
    - Thermal share 44 → 35 %, p99 loading 1.026 → 1.012. The γ_reb < 0 ↔ thermal link weakens: thermal in 78 % of
      γ_reb < 0 vs 31 % of γ_reb ≥ 0 (was 90 vs 0.1 %).
    - It does NOT remove the `pg_beyond` artifact (still 21 %): in surplus samples the REF unit at Pmin still
      receives the negative residual.
    - case14: no effect. Gen 2 has P̂ < Pmin and is dropped anyway, so distributed slack equals single slack.
  - Both changes together: the effects add up (case118 |γ_reb| 0.065, signed +0.063, artifact 0 %).

- 2026-10-13: settings for the full run (these replace the OLF defaults above):
  - generatorsWithZeroMwTargetAreNotStarted=false: all in-service units take part, independent of the forecast.
  - P̂ outside [Pmin, Pmax] by ≤ TOL_P is clipped to the limit. Reason: OLF keeps any unit whose setpoint is outside
    its limits at that setpoint, and drops it from the distribution; voltage control is kept. Verified with
    case118 Pmax+20 / Pmin−5. Without the clip, case14 gen 2 (P̂ = −9e-7 MW in every sample) never takes part,
    i.e. single slack; clipped to 0, it takes part (dp/Pmax 0.0148). Larger violations stay as predicted and are
    counted (`pg_pred_out_n`, `pg_beyond_n`).
  - slackBusPMaxMismatch = TOL_P = 0.01 MW, also the tolerance of every P-limit check.
  - slackDistributionFailureBehavior=LEAVE_ON_SLACK_BUS (was FAIL). Insufficient headroom stays in the sample: the
    remainder lands on the REF unit beyond its limit, priced on its cost curve (a lower bound), flagged
    (`slack_mismatch_flag`) and counted as a P-limit violation. case14 load ×2: Converged, 12.3 MW left, REF 352 MW >
    Pmax 340. Diverged PFs have no solution; they are counted and treated as infeasible in Q3.
    useActiveLimits=true kept.
  - DC arm: DC-OPF dispatch (`p_mw_dc`) with case-file generator setpoints VG at PV/REF buses (DC-OPF has no voltages).
    - 2026-11-09, consequence for Q3: case14 case-file VG = 1.0 p.u. at all 5 units (bus limits 0.94–1.06) → DC's PQ
      voltages sit mid-band (0 Vm violations, all H) while its units hold 1.0 p.u. whatever the reactive need ($Q_g$
      violated at 45 % of generator buses, 100 % of samples, H=6). DC's Vm / Qg columns reflect the case-file
      setpoints, not the DC dispatch. AC uses its OPF Vm setpoints (cost-optimal, typically near Vmax for lower losses
      → PQ buses close to the bound; not checked on our data).
    - 2026-11-14: 1.0 p.u. = the flat voltage profile DC-OPF itself assumes → DC's Vm / Qg results are a consequence
      of the DC approximation, not an evaluation artifact; case-file setpoints kept (training-mean schedule dropped).
  - Reporting (corrected_gap): thermal overload share, insufficient headroom (% and n), diverged / no_dispatch
    counts. A violation-free γ_reb was tested (numbers below) and dropped the same day: not reported.
- 2026-10-13: case118 h1 test, all arms, final settings. n = 3942 each (E2E 3934); capella srun, 2 CPUs, 5 m 46 s.
  - All arms: 100 % converged, 0 insufficient headroom, 0 diverged, 0 predicted Pg outside limits (> 0.01 MW),
    `pg_beyond` 0 % (the artifact is gone). label and exact_true: γ_reb 0, 100 % violation-free.
  - Gate: the raw gap is reproduced for 9 arms (AC, DC, surrogate, E2E).
  - With all units taking part ∝ Pmax (cost-blind), γ_reb ≥ 0 almost always: share γ_reb < 0 is AC 2–4 %,
    ML ≤ 1.6 %, DC 0 %. Those samples coincide with thermal overloads (AC xgb 93 %, all other arms 100 %).
  - Columns: |γ_reb| (signed within 0.004) / thermal overload % / violation-free %:
    - AC: sarima 0.102 / 36.7 / 9.2; xgb 0.137 / 40.5 / 1.2; snaive 1.103 / 48.7 / 0.03
    - DC: xgb 2.296 / 99.9 / 0
    - surrogate: xgb 0.310 / 51.4 / 0; snaive 1.203 / 52.4 / 0
    - E2E: 0.101 / 57.8 / 0
  - Violation-free is (almost) empty because of Qg. Median Qg violation: AC 0.3–4 MVAr, DC 113, surrogate 88,
    E2E 61 MVAr. AC at a 1 MVAr tolerance would be 48–57 % violation-free. DC and ML stay at 0 % even at 20 MVAr,
    so γ_reb (violation-free) is undefined for DC, surrogate and E2E.
- 2026-10-13: `resolve_run` picks the newest run (dir mtime) with that name across ALL experiments. case118 h24 →
  `e2e_derived_qgmse` be27fb31… (ablation), not the final model's dump `state-dump` cb29a9a9…. h1/h6 unaffected.
  Pin the experiment before the full run (`dispatch_diag.E2E_RUNS` does).
- 2026-11-16: **Q limits on** (`--q-limits`, supervisor request: operators' PF enforces Q limits, PV→PQ at the limit).
  Smoke barnard 41857341–44: H6, seed 0, n = 640 per arm, case118 + case500, on vs off on the same samples.
  - Qg violations 0 in every arm (limits enforced). case118 label + exact_true: 0 violations, 0 buses lost control.
  - case500 artefact: bus 291 (2 units, Q range 0.22 MVAr). OLF never lets a unit with Q range < 1 MVAr control
    voltage when Q limits are on (`reactiveRangeCheckMode`, every mode tests against 1 MVAr). Its |V| drifts
    (label: 1.100 → 1.092 p.u.) in 98 % of label samples. Vm violations at bus 291 ≤ 0.3 % of violating elements
    in every arm. Label floor: 3 / 640 samples with a Vm violation (buses 392/393, max 0.0034 p.u.).
    `vctl_lost_n` now counts only buses with a unit ≥ 1 MVAr range (code after bb35f9f).
  - γ_reb |Δ| ≤ 0.012 pp, thermal share within ±1.1 pp; except case500 DC (diverged samples drop out): γ_reb
    −0.05 to −0.12 pp, thermal −1.2 to −3.9 pp. Wall time per config +3–5 % → full run ≈ v3.
  - Vm share of samples, off → on: case118 AC 0.2–5.6 → 3.4–24 %, E2E 0 → 2.7 %, surrogate 0 → 0.5–0.6 %, DC 0 → 0;
    case500 AC 33–52 → 42–53 %, DC 0 → 47–52 %, E2E 68 → 69 %, surrogate 26–27 → 29–30 %.
  - Diverged (case500, on): DC 13–19 / 640, surrogate 2–4 / 640; AC + E2E 0; case118 0. All 0 with limits off.
  - Lost voltage control, share of samples (case118): AC 26–27 % (snaive 68 %), surrogate 50–83 %, E2E 69 %, DC 100 %.
- 2026-11-16: **Q limits on, FULL RUN** (fa08475; barnard 41857873–81, case × H, seeds 0–2, same 117 arms as v3;
  corrgap 41862959, gates ok) → `phase_1c/expost_pf_v4_qlim`, `exp1/results/combined/corrected_gap_qlim.csv` +
  `violation_ccdf_qlim.csv`. v3 (limits off) untouched. Thesis selection (best forecaster by v3 γ_reb), v3 → Q limits:
  - γ_reb: |Δ| ≤ 0.02 pp for AC / surrogate / E2E; DC lower: case14 −0.34 to −0.36 pp, case118 −0.01, case500 −0.06 to −0.11.
  - Thermal share: within ±0.4 pp except DC (case118 −0.4 to −0.6 pp, case500 6.3–7.3 → 2.5–5.6 %). Qg share 0 everywhere.
  - |V| violation, % of samples (H1 / H6 / H24): case14 AC 0→3.0 / 1.2→18.3 / 2.9→34.6, E2E 0→3.2 / 0→9.2 / 0→6.5,
    surrogate 0→0 / 0→1.5 / 0→4.5, DC 0; case118 AC 0→2.5 / 0.2→3.8 / 1.1→10.8, E2E 0.2→3.4 / 0.1→2.6 / 0→0.3,
    surrogate 0→0.6, DC 0; case500 AC 7→24 / 39→47 / 51→54, E2E 68→69 / 70→71 / 64→65, surrogate 16→23, DC 0→51–52.
  - |V| magnitude, mean per violating element: ≤ 0.009 p.u. except case500 surrogate 0.013–0.015, DC 0.025–0.027;
    P99 case500 DC 0.144–0.162, surrogate 0.076–0.082, E2E 0.055–0.065, AC 0.015–0.052.
  - Diverged, case500 only (% of samples, H1 / H6 / H24): DC 2.5 / 2.4 / 2.5, surrogate 0.6 / 0.7 / 0.9,
    E2E 0.03 / 0.13 / 0.19, AC 0 / 0.01 / 0.03. case14 / case118: 0.
  - Ranking E2E vs AC on |V|: E2E fewer violating samples on case14 / case118 at H6 + H24, slightly more at H1
    (3.2 vs 3.0 %, 3.4 vs 2.5 %), more on case500 at every H (gap narrows at H6 + H24).
  - 2026-11-17, **lost voltage control** (gen bus with a unit ≥ 1 MVAr Q range, |V| off its setpoint = units at their
    Q limit; corrgap 41954076, cb8522d), % samples (% buses), H1 / H6 / H24:
    - case14: AC 3 / 19 / 38 (1.0 / 6.4 / 12.3), E2E 38 / 32 / 34 (10.7 / 10.6 / 11.3), surrogate 17 / 25 / 38, DC 100 (50)
    - case118: AC 0.3 / 26 / 52 (0.01 / 0.95 / 2.4), E2E 65 / 67 / 86 (3.0 / 3.3 / 5.1), surrogate 51 / 63 / 75, DC 100 (25)
    - case500: AC 21 / 60 / 79 (3.6 / 10.3 / 15.3), E2E 94 / 96 / 98 (10.8 / 14.7 / 16.8), surrogate 100 (27–28), DC 100 (56–57)
    - E2E above AC in every cell except case14 H24 (34 vs 38 % samples, 11.3 vs 12.3 % buses); gap shrinks with H
      (bus share case118 300× / 3.5× / 2.2×, case500 3.0× / 1.4× / 1.1×). DC worst everywhere → the Qg ranking of v3
      (DC worst, AC best) carries over. DC's reactive shortfall stays inside the |V| band on case14 / case118 only.
  - Thesis outputs switched to Q limits on (2026-11-17, `plot_rmse.VARIANT`, `EXPOST_VARIANT=v3` → limits off); v3
    snapshot committed cb8522d. supply_error_h6 unchanged (ΔP from P̂; best forecaster identical in every cell). DC: case14 / case118 show no |V| or Qg violation any more (case-file
    VG 1.0 mid-band; reactive shortfall absorbed by voltage drift within the band).

## 8. Rebalanced optimality gap (2026-10-12, `exp1/generate_metrics/corrected_gap.py`)

- Why: in forecast-then-optimize the raw gap mostly measures energy volume (forecast error): under-supply looks
  cheap, over-supply expensive; `abs()` makes both positive. Signed raw gap cancels (AC h1: |γ| 0.82 % vs signed +0.02 %).
- 2026-10-12: rebalancing = ex-post AC-PF with distributed slack (§7). Merit order removed.
  P̃ = PF-solved Pg (slack output included) is the single source for cost and feasibility.
  γ_reb = (C(P̃) − C(P*)) / C(P*); R = C(P̃) − C(P̂), reported as % of C(P*). γ_reb and R are over converged
  samples. Gate unchanged: the raw gap from the ex-post costs reproduces the logged gap (3 AC h1 arms, |Δ| < 1e-3).
  DC 2-step has no ex-post AC-PF (no Vm), so it is no longer covered.
- SUPERSEDED (merit order, 2026-10-12): Method, per scenario: Δ = ΣPg* − ΣPg_pred closed by merit order on the predicted dispatch (ranked by c1 + 2c2·Pg_pred;
  shortfall → cheapest up to Pmax, surplus → most expensive down to Pmin; remainder → slack = violation), then
  γ_reb = (C(Pg_reb) − C*)/C*, reported as mean |γ_reb|; rebalancing cost R = (C(Pg_reb) − C(Pg_pred))/C*, signed.
  Gate: raw gap recomputed from the per-gen dumps reproduces the logged gap (all 28 case118 arms, |Δ| < 1e-3).
- Discarded (same session): single λ (mean marginal cost of non-binding units, ≈22.3 €/MW) with lower/upper surplus
  bounds. Ex-post `mean|γ| + λ·shortfall` double-counts (AC h1 upper 1.15 vs exact 0.40). The allocation ranking was
  λ-sensitive (λ_s ×1.08 flips AC vs E2E), and "share below optimum" was pricing noise (AC 42–47 % at ε = 0.01 %).
- SUPERSEDED (merit order): case118, best forecaster per optimizer, ML arms seed 0 (Table 7). Columns: raw |γ| / |γ_reb| / R / ΔP mean (MW).
  - h1: AC 0.822/0.210/−0.23/+0.3; DC 3.729/2.039/+1.67/−53.4; surrogate 1.461/0.222/−0.90/+9.0; E2E 0.553/0.164/−0.05/−2.6
  - h6: AC 3.035/0.646/−0.93; DC 4.490/2.058/+1.45; surrogate 3.208/0.539/−1.46; E2E 1.772/0.458/−0.34
  - h24: AC 4.987/0.891/−1.50; DC 5.804/2.082/+1.12; surrogate 5.043/0.803/−1.82; E2E 4.295/0.931/−1.07
  - Slack overflow 0 % everywhere (enough headroom). Rebalancing removes 70–85 % of the raw gap for AC/surrogate/E2E.
  - After rebalancing: E2E is best at h1/h6, the surrogate at h24. DC is about 2 % at every horizon: it is ~53 MW short
    (missing losses) and its mix ignores AC limits.
- (merit order) 2026-10-12 (same data): signed γ_reb ≈ −|γ_reb| for EVERY arm (AC h1 −0.21 vs |0.21|, h24 −0.887 vs 0.891), i.e.
  almost every rebalanced dispatch is cheaper than the AC-OPF optimum. That includes AC, whose dispatch is optimal
  for its forecast. The merit order ignores the network (congestion, voltage, losses) that keeps C* up, so
  |γ_reb| mostly measures this network-blind undercut, which grows with |ΔP|. It is not allocation quality, and a
  negative signed gap does not single out bad arms. R split (contribution to mean R, sum = mean): R⁻ dominates for
  every arm, e.g. AC h24 R⁺ +1.44 / R⁻ −2.94, E2E +1.34 / −2.41. Insufficient headroom 0 % for all case118 arms.
- (merit order) Caveat: R < 0 on average even when mean ΔP ≈ 0 (AC). Up-steps buy the cheapest units and down-steps remove the most
  expensive ones, so rebalancing is optimistic in both directions and the network is ignored. The ex-post AC-PF
  (§7) checks operability of the same rebalanced dispatch.
- case14 / case500: earlier λ-based results (2026-10-10) showed a surrogate allocation penalty (case14 ~0.59 %,
  case500 1.25–1.42 %) not present in case118. Not yet recomputed with merit order. case500 E2E is invalid until the
  retrain (stale processed_temporal).
- 2026-10-12, distributed slack, case118 h1, AC 2-step arm, n = 3942 per forecaster (sanity run, §7).
  Columns: raw |γ| / |γ_reb| / signed γ_reb / R⁺ / R⁻ / share γ_reb < 0; merit order in brackets.
  - xgb:    1.127 / 0.069 / +0.003 / +0.496 / −0.564 / 48.8 %   [0.275 / −0.273 / +0.288 / −0.632 / 94.3 %]
  - sarima: 0.822 / 0.049 / +0.000 / +0.375 / −0.399 / 51.0 %   [0.210 / −0.210 / +0.212 / −0.447 / 96.5 %]
  - snaive: 9.235 / 0.548 / +0.085 / +3.806 / −4.892 / 49.5 %   [1.336 / −1.334 / +2.955 / −5.460 / 97.3 %]
  - Signed γ_reb quantiles (xgb): p1 −0.23, p25 −0.05, p50 +0.002, p75 +0.05, p99 +0.25 %.
    Per-sample correlation with the merit order: 0.72 / 0.71 / 0.49.
  - γ_reb < 0 ⇔ under-supply: corr(γ_reb, ΣP̂ − ΣP*) = 0.91 / 0.93 / 0.89.
  - γ_reb < 0 coincides with Q3 thermal violations. A thermal overload occurs in 90 / 84 / 99.5 % of the γ_reb < 0
    samples vs 0.1 / 0.1 / 0.7 % of the γ_reb ≥ 0 samples. Median max loading is 1.006 vs 0.995 (xgb); for snaive,
    46 % of the γ_reb < 0 samples exceed 1.05.
  - Qg violations do not discriminate (83–100 % in both groups, median 0.2–4 MVAr). Voltage violations appear
    only for snaive γ_reb ≥ 0 (13 %).
  - Reading: with distributed slack, a negative γ_reb marks dispatch that is cheaper than C* only by overloading
    lines the OPF kept binding. The network-blind undercut of the merit order is gone (signed mean ≈ 0).
  - Full rerun (all cases / horizons / arms) pending; Table 7 numbers above are outdated.
- 2026-10-13: participation rule matters (case118 h1 AC 2-step, |γ_reb| / signed γ_reb):
  - merit order: sarima 0.210 / −0.210; xgb 0.275 / −0.273
  - DS, OLF defaults: sarima 0.049 / +0.000; xgb 0.069 / +0.003
  - DS, all in-service units ∝ Pmax (final): sarima 0.102 / +0.102; xgb 0.137 / +0.136
  - The merit order is network-blind (optimistic, γ_reb < 0). DS over all units is cost-blind (γ_reb ≥ 0 unless a
    limit is violated, as OPF optimality on the true load predicts). Final run uses the last setting (§7).


- 2026-10-22: **FULL RUN, final models, all cases × H, seed 0** (ex-post PF romeo jobs 9118247-49, distributed slack
  final settings §7; corrected_gap job 4449353: 108 arms, all reproduce the logged gap; `exp1/results/combined/corrected_gap.csv`).
  Rebalanced gap |γ_reb| %, AC sarima / AC xgb / AC xgb_tot / DC xgb_tot / surrogate xgb / E2E:
  - case14 h1 0.114/0.198/0.212/1.70/0.332/0.095; h6 0.444/0.446/0.452/1.82/0.567/0.263; h24 0.828/0.702/0.723/1.99/0.819/0.704
  - case118 h1 0.102/0.137/0.112/2.28/0.296/0.083; h6 0.408/0.367/0.327/2.35/0.486/0.214; h24 0.789/0.602/0.572/2.51/0.694/0.484
  - case500 h1 0.277/0.333/0.283/1.47/1.217/0.265; h6 1.079/0.858/0.842/1.79/1.463/0.664; h24 2.016/1.505/1.414/2.23/1.869/1.231
  - → E2E lowest rebalanced gap in all 9 cells (case14 h24 ≈ tie with AC xgb 0.702).
  - Thermal overload % (same order): case14 0 everywhere; case118 h1 36.7/40.5/36.3/99.9/58.0/46.9, h6 47.1/45.1/42.3/96.4/
    57.0/55.7, h24 50.2/44.5/44.3/89.5/56.6/55.5; case500 h1 0.7/6.2/5.7/6.3/5.0/2.8, h6 3.7/5.7/5.6/6.6/5.2/3.5, h24
    6.1/6.0/5.7/7.3/6.3/4.0 → E2E overloads lines more than AC on case118 (+10–11 pp vs xgb_tot), fewer on case500.
  - 2026-10-24 UPDATE, seeds 0–2 (romeo ex-post 9126824-26 + 9164986-88; corrected_gap job 4461076, 117 arms, all gates ok).
    E2E 3 seeds on case14/118, 1 on case500; surrogate 3 seeds; AC/DC deterministic. Rebalanced gap %, E2E vs best 2-step:
    case14 0.085/0.306/0.724 vs 0.114/0.444/0.702; case118 0.082/0.221/0.473 vs 0.102/0.327/0.572 → E2E lowest in 8 of 9
    cells; case14 h24 now a narrow loss to AC xgb (0.724 vs 0.702; s0 alone was 0.704). Surrogate xgb_tot ≈ surrogate xgb
    (case500 1.29/1.56/1.91). Thermal overload %: E2E case118 43.5/51.8/57.7 vs AC xgb_tot 36.3/42.3/44.3 (E2E worse,
    gap grows with H); case500: surrogate fewest (1.7–2.7 %), E2E 2.8/3.5/4.0, AC xgb_tot 5.6–5.7.
  - 2026-10-28: **No-refund gap** γ_noref = (max(C(P̃), C(P̂)) − C*)/C* (over-supply pays the committed dispatch; corrgap
    job 4466077, 117 arms, gates ok). %, AC sarima / AC xgb / AC xgb_tot / surrogate xgb_tot / E2E:
    case14 h1 0.53/0.85/0.74/1.00/0.36, h6 2.27/2.19/1.97/2.23/1.39, h24 4.69/3.64/3.77/4.07/3.23;
    case118 h1 0.49/0.68/0.59/0.98/0.34, h6 2.08/1.87/1.71/2.06/1.02, h24 4.40/3.14/3.08/3.37/2.84;
    case500 h1 0.49/0.57/0.48/2.73/0.45, h6 2.00/1.53/1.51/3.28/1.25, h24 3.97/2.80/2.68/4.15/2.57.
    → E2E lowest in all 9 cells (incl. case14 h24, where the rebalanced gap was a narrow loss); γ_noref ≈ 2–5× γ_reb,
    i.e. most of the raw gap's over-supply cost survives when ramp-down is not refunded.
  - 2026-10-28 FINAL: E2E case500 h1 now 2 seeds (s2 cancelled; romeo ex-post 9205436, corrgap 4482007, 117 arms, gates ok):
    rebalanced 0.256 %, no-refund 0.445 %, thermal 2.82 % (s0 alone 0.265 / 0.447 / 2.82) → conclusions unchanged.
  - 2026-11-01: **Q3 per constraint group** (corrgap job 4506484 on the same ex-post outputs, 117 arms, gates ok, all 1872
    previous values identical). Element sets: lines / PQ buses / non-REF gen buses = case14 20/9/4, case118 186/64/53,
    case500 728/387/112. Forecaster = best by rebalanced gap. Share of samples % (mean worst over violating samples):
    - thermal, AC / DC / surrogate / E2E: case118 h1 36.7 (0.3) / 99.9 (4.0) / 53.5 (0.9) / 43.5 (0.4), h24 44.3 (1.7) /
      89.5 (4.5) / 52.2 (1.9) / 57.7 (2.1) [% of rating above 100 %]; ≈ 0.2–0.3 % of lines (the one binding line)
      → case118 overloads are frequent but marginal; case500 ≤ 7.3 % of samples.
    - Qg: ≈ 90–100 % of samples for every arm on case118/500 (AC too) → share not informative; worst MVAr AC 0.5–12.8,
      E2E 3.0–17.2, surrogate case500 55–56, DC 113 (case118) / 391 (case500, case-file voltage setpoints).
    - PQ voltage: E2E case500 64–70 % of samples (AC 7–51, surrogate 17), but worst ≤ 0.006 p.u. and ≤ 0.5 % of PQ buses;
      DC 0 everywhere.
    - Mean per violating element: expost_pf rerun with *_sum columns (romeo 9238374-76 → `phase_1c/expost_pf_v2`,
      corrgap 9238386, 117 arms, gates ok; all 2808 previous values bit-identical). Element share % (mean per element):
      thermal case118 AC 0.20–0.24 (0.3–1.7), E2E 0.23–0.31 (0.4–2.1), DC 0.89–0.97 (3.6–3.9) [% of rating]; Qg AC 6–21
      (0.3–3.5), E2E 10–22 (0.4–4.3), surrogate case500 26 (13), DC 25–47 (10–61) [MVAr]; Vm ≤ 0.57 (≤ 0.0045 p.u.).
      → AC lowest on most Q3 cells; E2E close to AC except case500 Vm; DC far worst on thermal + Qg.
  - 2026-11-09: **tail per constraint** = P99 over all converged samples of the per-sample worst violation (0 if none),
    "Worst P99" metrics (romeo 9457317 on expost_pf_v2, 117 arms, gates ok, all 3159 previous values identical).
    Best forecaster by rebalanced gap; AC / E2E:
    - Qg [MVAr]: H1 case14 0.6 / 1.2, case118 2.4 / 8.3, case500 13.7 / 29.5; H24 3.4 / 3.4, 12.9 / 12.0, 62.8 / 63.5
      → E2E's tail ≈ AC's at H24, up to 3.5× at H1 (AC's tail grows with H, E2E's less). Surrogate case500 ~160,
      DC 37 / 125 / 470 (case-file voltage setpoints), both ≈ constant over H.
    - Vm case500 [1e-3 p.u.]: AC 2.5 → 8.8 → 13.6, E2E 10.4 → 12.4 → 14.4 (H1 → H6 → H24); elsewhere ≤ 2.4.
    - Thermal case118 [% of rating above 100]: AC 1.0 / 3.7 / 5.5, E2E 1.6 / 3.7 / 8.0, DC 6.9–9.4; case500 ≤ 3.1.
    - SUPERSEDED 2026-11-14 (metric dropped, user chose per-element severity, entry below); numbers stand as measured.
  - 2026-11-14: **severity per violating element** (expost_pf_v3: romeo 9482941–43, per-sample columns bit-identical to v2
    in a 200-sample smoke; corrgap 9482950, 117 arms, gates ok, all 3159 previous values identical). Median / P99 over
    violating elements, ML seeds pooled, only with ≥ 100 elements. H6, best forecaster by rebalanced gap,
    AC / E2E / surrogate / DC:
    - Qg [MVAr] median case14 0.42 / 0.59 / 0.39 / 8.40, case118 0.68 / 1.42 / 1.22 / 15.5, case500 0.84 / 1.60 / 7.28
      / 29.2; P99 case14 3.29 / 2.55 / 2.90 / 35.4, case118 5.43 / 6.84 / 7.61 / 118, case500 18.8 / 22.9 / 106 / 390
      → E2E: median 1.4–2.1× AC, P99 ≈ AC (−22 … +26 %); DC P99 11–22× AC; surrogate case500 P99 5.6× AC.
    - Thermal case118 [% above rating] median 0.76 / 0.74 / 1.02 / 3.58, P99 4.53 / 4.19 / 4.77 / 7.64; case500 P99
      3.39 / 2.26 / 5.22 / 5.12 (≤ 7 % of samples).
    - Vm case500 [1e-3 p.u.] median 2.09 / 3.25 / 3.87 / –, P99 10.6 / 11.8 / 12.1 / – (DC: none); case14 AC only
      (median 1.79, P99 6.68); case118 < 100 violating elements in every arm.
  - 2026-11-16: **H=6 is not representative for E2E vs 2-step AC** (same data, all H; quantiles interpolated on the
    violation_ccdf levels, max = P99.99 / top element; `violation_box_h<H>`). E2E / AC:
    - Holds at all H: DC worst on Qg (median 8–29 MVAr, flat over H) and case118 thermal; surrogate case500 Qg (P99
      ≈ 106 MVAr, flat) and thermal (median 2.9–3.9 %, above DC); E2E median > AC for Qg (1.4–4.9×) and case500 |V|
      (1.3–2.0×); case118 thermal % el E2E > AC (0.23 / 0.28 / 0.31 vs 0.20 / 0.23 / 0.24, H1 / H6 / H24).
    - Tail depends on H: Qg P99 1.5–3.8× at H1, 0.77–1.26× at H6, 0.97–1.11× at H24; case500 |V| P99 2.8× at H1,
      ≈ 1 at H6 / H24; case118 thermal P99 1.5× / 0.92× / 1.5×, max 3.3 vs 1.9 / 8.0 vs 7.4 / 18.8 vs 10.0 % (E2E H24 =
      largest thermal overload of all arms, DC 11.9); case500 thermal E2E < AC at H6 / H24 (P99 2.3 vs 3.4, 3.5 vs 7.2).
    - Qg % el E2E > AC at H1 / H6, below AC at H24 on case14 (14.9 vs 17.4) and case118 (10.9 vs 11.4).
    → AC's violations grow with the forecast error (H); E2E starts higher (H-independent part) and grows less, so the
      E2E–AC difference shrinks with H. Exception: case118 thermal widens again at H24. H6 statements on DC / surrogate
      generalise; on E2E vs AC they must be given per H.
- 2026-11-14, gap significance E2E vs 2-step AC (best forecaster by the metric, treated as fixed): corrected_gap per
  seed (romeo 9483961, `--seeds 0/1/2` on expost_pf_v3; per-seed means == committed corrected_gap.csv), one-sample
  t-test, n = 3, df = 2. Rebalanced gap E2E / AC (%): case118 0.082 / 0.102 (p 0.057), 0.221 / 0.327 (p 0.001),
  0.473 / 0.572 (p 0.005); case14 0.085 / 0.114 (p 0.073), 0.306 / 0.444 (p 0.030), 0.724 / 0.723 (p 0.96)
  → significant in 3/6 (H=6 both, case118 H=24); H=1 lower but n.s. (seed std 0.009 / 0.014 vs gaps ≈ 0.1 %).
  No-refund gap: significant in 6/6 (−8 … −40 %, p ≤ 0.025). case500: 2 / 1 / 1 seeds, not testable.
- 2026-11-17, **seed summary instead of t-tests** (user decision: no tests at n = 3; barnard 41954220: per-seed
  corrected_gap on expost_pf_v4_qlim + per-seed Pg RMSE from exp 764191360493391788; seed means == aggregated means).
  E2E vs 2-step AC (best forecaster per metric): relative difference / difference in E2E seed std / all seeds better.
  - Pg RMSE: case14 −36 / −29 / −7 % (12 / 19 / 3 std), case118 −5 / −24 % (5 / 28 std), all seeds better;
    case118 H24 +1.6 % (0.45 std, seeds 8.58–9.20 vs 8.77 → within spread); case500 H1 +5.8 % (both seeds worse),
    H6 −17 %, H24 −11 % (one seed).
  - Rebalanced gap (qlim): case14 −26 / −31 % (2.1 / 3.3 std), case118 −18 / −32 / −17 % (2.2 / 15 / 7.8 std), all
    seeds better; case14 H24 −0.2 % (0.05 std, seeds 0.70–0.76 vs 0.72 → tie); case500 −7.4 / −21 / −13 % (all seeds
    better; 2 / 1 / 1 seeds). Range over the 8 non-tie cells 7–32 % (v3 text said 8–32 %).
  - No-refund gap (qlim): all seeds better in all 9 cells; case14 / case118 3.6–17 std, −8 … −40 %.
- 2026-11-15, rebalancing volume = mean |ΔP| = |ΣP̂ − ΣP*| (table-8 selection, best forecaster by rebalanced gap):
  E2E lowest of all arms in 9/9 cells; vs 2-step AC case14 0.7 / 2.9 / 6.9 vs 1.1 / 4.7 / 7.9 MW, case118 11.0 /
  33.9 / 81.4 vs 14.9 / 50.6 / 89.0, case500 66.1 / 187.6 / 379.3 vs 71.9 / 235.5 / 420.9 (H1 / H6 / H24).
- 2026-11-16, why no-refund favours E2E (same selection, corrected_gap.csv): the added term is the over-supply cost,
  driven by mean max(ΔP, 0) = (mean |ΔP| + mean ΔP) / 2, not by mean ΔP. E2E vs AC (MW): case118 5.4 / 16.1 / 46.7 vs
  7.6 / 26.9 / 46.8, case14 0.39 / 1.49 / 3.30 vs 0.56 / 2.37 / 3.70, case500 32.2 / 95.2 / 213.3 vs 36.3 / 112.9 /
  211.2 → E2E smaller in 8/9, exception case500 H24 (bias +47.3 vs +1.5 MW) = the one cell where its over-supply
  refund |R⁻| is larger (1.34 vs 1.27 %). |mean ΔP| E2E < AC in only 3/9 → the mean is not the reason.

## 9. Dispatch structure: why E2E barely beats 2-step on Pg (2026-10-13, `exp1/generate_metrics/dispatch_diag.py`)

**Current state (2026-10-14) — read this first; dated details below.** Test split, E2E seed 0 only.
- Data regime: Pg ≈ f(total demand D): R² 0.999999 / 0.997 / 0.9998 (case14/118/500). Binding lines rare or
  constant. Loads share one factor (PC1 90–96 %), no grid-distance structure (profiles randomly assigned) →
  OPF forecasting ≈ total-demand forecasting.
- Corrected Pg (case500 2-step CSVs were stale; compare.py rerun verified): E2E has the best Pg RMSE on every case ×
  horizon; Pg MAE worse only on case500 (h1 +57 % vs sarima, h6/h24 +7/+8 % vs xgb).
- E2E's RMSE edge = (1) curvature Δ (ceiling term; hedging under MSE): case14 ≈ 0, case500 4.5–9 % (f(D) proxy,
  cross term small), case118 unknown (proxy too coarse) + (2) better system total: E2E |ΣΔPg| lowest everywhere,
  below its own summed Pd error; per-bus forecaster errors are correlated (survive 0.58–0.92) and add up in D.
- E2E's cost: allocation error (misplaced MW case500 23–48 vs 2-step ≈ 0.1) → worse MAE; hedging also raises MAE
  (median commutes with monotone f → no ceiling on MAE). Raw gap ≈ volume term; allocation costs 0.02–0.12 pp.
- Forecaster premise: SARIMA ≈ conditional mean; XGB is not (total error 51–58 % predictable from last hour, h1).
- Open: case118 Δ via OPF on load scenarios; information-parity control (XGB + total-load lags); E2E seeds;
  Qg-MSE consistency (case500 E2E trained with it, case14/118 without).

- RESOLVED 2026-10-14 (compare.py rerun h1/h6/h24; aggregate 2026-10-24 gates ok): **case500 2-step Pg metrics in `exp1/results/` were stale** (CSVs 2026-08-12/19, before in-service fix 45aaf0e):
  `AC Generator Pg MAE/RMSE` pool 224 units incl. 53 out-of-service (zero error) → MAE × 171/224, RMSE × √(171/224)
  exactly (h1 xgb 0.523 → logged 0.399). E2E pools the 171 in-service units → "E2E worse on case500 Pg" was mostly this.
  - `AC pg viol. rate` 0.244 ≈ 53/224 = 0.237 → out-of-service units counted as P violations. VERIFIED 2026-10-14:
    compare.py rerun (h1/h6) gives viol. rate 0.0099 and Pg MAE/RMSE == dispatch_diag recomputation (0.5229/3.567).
  - Rerun compare.py for case500 (all H) before aggregating. case500 h24 xgb has two result dirs
    (`case500_goc_horizon24/`, `case500_goc_horizon24_3yr_HPO_xgb/`) → breaks one generation per (case, H, fc).
  - case118: recomputed == logged for all 9 arm × H (54 units, all in service).
- Corrected Pg, test samples, E2E s0 (n = 3934 / 23604 / 94344), MAE / RMSE MW:
  - case500: h1 AC xgb 0.523/3.57, sarima 0.420/3.38, E2E 0.661/3.28; h6 1.410/8.41, 1.758/9.86, 1.506/6.84;
    h24 2.565/13.03, 3.568/16.26, 2.777/11.36 → E2E best RMSE at every H; MAE worse (h1 +57 % vs sarima, h6/h24
    +7/+8 % vs xgb).
  - case118: E2E best on both at every H (MAE h1 0.261 vs 0.281, h6 0.743 vs 1.087, h24 1.690 vs 1.776).
- **Pg is (almost) a function of total demand D = ΣPd** (GT, 26280 h, leave-one-out local-linear fit per unit):
  pooled R² case500 0.9998 (test 0.99992), case118 0.9967 (test 0.9987). Moving units with R² ≥ 0.99: 95 / 36 %,
  ≥ 0.9: 98 / 91 % (all hours). The rest sits in rare peak hours (case500 D > 19.5 GW, congested).
  - case500: 171 units (65 pinned, 106 switching, 0 free; 111 linear, 60 quadratic); ~10.5 marginal units/h (max 28);
    410 unit-state patterns, a unit changes state in 86 % of hours; ≥ 1 binding line in 7 % of hours (test 11 %).
  - case118: 54 linear units (46 pinned, 7 switching, 1 free); ~4.6 marginal/h; one line binds in 99.9 % of hours
    (constant regime, 5 line patterns).
  - → a forecast error moves all units in the same direction; where the error sits barely matters.
- **2-step error is all volume; E2E adds allocation error** (per sample, MW: total |ΣΔPg| / misplaced (Σ|ΔPg| − |ΣΔPg|)/2,
  h1 / h6 / h24):
  - case500: AC xgb 89/0.11, 241/0.11, 438/0.10; sarima 72/0.02, 301/0.07, 610/0.07; E2E 67/23, 188/35, 379/48.
  - case118: AC xgb 21/0.6, 57/0.9, 94/1.0; sarima 15/0.1, 63/0.5, 128/0.5; E2E 11/1.6, 34/3.1, 82/4.8.
  - E2E has the lowest total error at every H (case500 −7/−22/−13 % vs best 2-step; case118 −27/−40/−13 %), also below
    its own summed Pd forecast error (case500 88/233/421; case118 19/44/147, bias +126 at h24).
  - E2E MAE deficit on case500 vs xgb (mae_contrib): pinned linear units +0.06–0.07 MW at every H (sigmoid never at the
    limit; ≈ half the h1 deficit); switching linear units +0.07/+0.06/+0.11 → learned D→Pg mapping (fine merit-order
    staircase) misplaces MW. Model effect, not data.
    CORRECTED (2026-10-13, step 4): hedging under MSE also raises MAE; sigmoid vs hedging share not separated.
  - 2-step regime flips (≥ 1 unit state ≠ GT): case500 55–88 % of samples (1.6–12 units), case118 5–34 %.
- **Load forecast errors share one factor and add up; grid distance carries nothing** (test; n_load 281 / 99):
  - raw Pd: mean pairwise corr 0.89 / 0.92, PC1 90 / 92 % (shared daily cycle).
  - error corr XGB 0.34–0.47 / 0.36–0.51 (h1 → h24), SARIMA 0.69–0.76. Var(Σe) / ΣVar(e): XGB 29–99 / 8–27,
    SARIMA 135–142 / 35–37 (1 = independent) → bus errors do not cancel in D.
  - Spearman(hop distance, corr) 0.00–0.03 for errors and raw loads (profiles were assigned to buses at random).
  - h1: own last error explains R² 0.37 / 0.42 of XGB's next error (SARIMA 0.08); other buses' mean last error adds
    +0.008 → other buses' history adds ~nothing beyond the own.
- RETRACTED (2026-10-13, same day, step 4 below): "E2E's only lever is the system total" and "decision-aware gains are
  capped by Pg ≈ f(D)". f is 1-D but kinked → the curvature (Jensen) gain exists in D. Still valid: location /
  spatial gains are capped (Pg ≈ f(D), random load placement). HYPOTHESIS kept: the Pg head forecasts the aggregate
  directly and skips the correlated per-bus error sum. Untested control: 2-step reconciled to a direct total forecast.
- 2026-10-13 (same script, rerun): **gap split + E2E Pd errors**
  - Gate: recomputed raw gap == logged for all arm × H (|Δ| ≤ 0.0004 pp) → the stale case500 cp0 barely moves the gap.
  - Raw gap = volume |C(P_T) − C*| + allocation C(P̂) − C(P_T), P_T = GT dispatch of the hour with the nearest total
    (lookup error ≈ 0.27 MW). case500, raw / volume / allocation %, h1 / h6 / h24:
    AC xgb 0.661/0.660/−0.001, 1.783/1.783/−0.001, 3.255/3.255/−0.001; E2E 0.494/0.496/+0.017, 1.395/1.393/+0.050,
    2.833/2.812/+0.116 → E2E's misplaced MW cost 0.02–0.12 pp, its smaller volume error saves 0.17–0.44 pp.
    case118: allocation ≈ 0.17–0.19 pp for EVERY arm (AC too) → the nearest-total reference is too coarse there
    (location effects, R² 0.997); only the case500 split is clean.
  - E2E Pd head errors are as correlated as XGB's: mean corr case500 0.33/0.36/0.45, case118 0.19/0.29/0.46;
    Var(Σe)/ΣVar(e) 88/88/101 and 16/20/28. Summed Pd error (load buses) not better than XGB: case500 97/250/413 vs
    88/236/428 MW; case118 19/46/135 vs 21/56/92. Yet E2E ΣPg error is the lowest (above) → Pg head ≠ Pd head + losses.
  - Total-error persistence, h1, R²(Σe_t ~ Σe_t−1): XGB 0.51 / 0.58, SARIMA 0.07 / 0.07, E2E 0.25 / 0.30
    (case500 / case118) → XGB's system-level error is half predictable from the last hour's.
  - E2E summed Pd error over ALL buses (case500 h1 88 MW) ≠ over load buses (97 MW) → E2E predicts nonzero Pd at
    zero-load buses (not checked further).
- 2026-10-13: **curvature (Jensen) ceiling, survive ratio, "E2E total + OPF split"** (dispatch_diag step 3/4, step 1)
  - survive = |Σe| / Σ|e| per sample (1/√n ≈ 0.06 if independent): case500 XGB 0.58/0.65/0.71, SARIMA 0.83/0.83/0.84,
    E2E Pd 0.61/0.67/0.70; case118 XGB 0.59/0.67/0.73, SARIMA 0.84/0.84/0.87, E2E Pd 0.44/0.56/0.77 (h1/h6/h24).
  - Ceiling, case500 (AC arm, f(D) = GT dispatch at nearest D; plug-in f(D̂) reproduces the AC Pg RMSE to 3 decimals,
    RMSE(plug-in − AC dispatch) 0.42–0.48 MW). Hedged = mean over 41 quantiles of the forecaster's own centered test
    D error per step (unconditional; bias correction adds nothing). Pg RMSE plug-in → hedged:
    xgb 3.57 → 3.35 / 8.41 → 7.81 / 13.03 → 11.91 (−6 / −7 / −9 %); sarima 3.40 → 3.25 / 9.86 → 9.21 / 16.26 → 14.78.
    MAE rises: xgb 0.53 → 0.58 / 1.41 → 1.64 / 2.57 → 3.07 → better RMSE + worse MAE = signature of a conditional-mean
    predictor on a kinked map, the same pattern as E2E.
  - E2E RMSE 3.28 / 6.84 / 11.36 vs best plug-in 3.38 / 8.41 / 13.03 and best hedged 3.25 / 7.81 / 11.91 → h1 gain ≈
    hedging alone; h6/h24 E2E beats hedged 2-step by 12 / 5 % (better total).
  - case118: proxy too coarse (RMSE plug-in − AC 2.9–3.3 MW; plug-in 3.81 vs AC 2.16 at h1) → no ceiling estimate there.
  - "E2E total + OPF split" (P_T, step 1), case500 Pg MAE / RMSE: 0.40/2.89, 1.10/6.86, 2.22/12.00 vs E2E 0.66/3.28,
    1.51/6.84, 2.78/11.36 and best 2-step 0.42/3.38, 1.41/8.41, 2.57/13.03 → beats all arms on MAE at every H; RMSE
    best at h1, worse than E2E at h24 (the plug-in split drops the hedging).
- 2026-10-14: **Δ check + case14** (dispatch_diag step 4; case14 E2E = `state-dump` runs, SARIMA only: no xgb at h1).
  - Δ = mean (hedged − plug-in)² vs MSE_plug − MSE_hedged (equal iff hedged = conditional mean, cross term 0), MW²:
    case500 xgb 1.37/1.52, 8.25/9.66, 23.9/27.8; sarima 0.84/1.01, 13.3/12.5, 46.6/45.8 (h1/h6/h24) → cross term
    2–14 % of the MSE drop → hedged ≈ conditional mean (given the unconditional error spread).
    Proxy noise (RMSE² plug-in − AC dispatch) 0.18–0.23 MW² → up to ~25 % of Δ at h1 may be lookup smoothing, h6/h24 robust.
  - case14: 1 of 5 units moves, R² (Pg ~ f(D)) 0.999999, no binding line → Δ 0.000–0.011 MW², gain 0.0–0.1 %:
    affine dispatch → no curvature gain, as the ceiling argument predicts. E2E still beats AC-sarima on Pg RMSE
    0.43/1.95/4.35 vs 0.70/3.12/5.77 (−39/−38/−25 %), entirely via the total (|ΣΔPg| 0.69/2.85/6.89 vs 1.11/4.67/9.44 MW;
    misplaced ≤ 0.02 MW); E2E's own ΣPd error is NOT better at h1 (1.17 vs 1.04).
  - case118: proxy noise 8.6–10.9 MW² > Δ estimate 4.3–8.2 → no valid curvature estimate (needs OPF on load scenarios).
  - D̂ conditional-mean test: share of Var(D − D̂) explained by D̂ decile / target hour / step ≤ 0.02 (all cases,
    xgb + sarima); bias ≤ 10 MW vs std 21–581 → unbiased given calendar and level. NOT a conditional mean w.r.t. the full
    history: XGB's total error is 51–58 % predictable from the last hour's (SARIMA 7–8 %, §9 above).
- 2026-10-17: **E2E Pd → AC-OPF, case500 h1/h6** (`e2e_pd_opf.py`, job 39799885; E2E = `final_e2e_qgmse`
  `base_e2e_case500_goc_h{H}_s0`; `<data>/e2e_pd_opf/phaseB_*case500_goc_h1-6.csv`). Same model + inputs, only the
  output differs (Pd head → AC-OPF vs Pg head). Gates ok: pd_true vs GT ≤ 2.3e-5 MW; MAE Pd compare == E2E logged
  (rel ≤ 5e-4); OPF failures 0 / 11 (h1 / h6). n = 3934 / 23593.
  - Pg MAE / RMSE MW, h1: E2E Pd → AC 0.527/3.62 vs E2E 0.661/3.28, AC xgb 0.523/3.57, AC sarima 0.420/3.38;
    h6: 1.391/8.29 vs 1.506/6.84, 1.410/8.41, 1.758/9.86 → the Pd route lands at plain-XGB level; Pg head −10 / −18 %.
  - Total |ΣΔPg| MW: E2E Pd → AC 89.7 / 237 vs E2E 66.7 / 188; misplaced 0.16 / 0.25 (2-step-like) vs 23 / 35.
    Summed Pd bias +12.6 / +90.8 MW (h6 Pd head clearly biased).
  - Hedged (step 4 on the real OPF dispatch): RMSE 3.39 / 7.61 (bias-corrected 3.34 / 7.15) → E2E's Pg head still beats
    hedged own-Pd 2-step at h6 (−10 %), ≈ hedging at h1 (−3 %).
  - → E2E's Pg advantage comes from the output target/loss, not model class or inputs (confirms §9 hypothesis
    "Pg head ≠ Pd head + losses"). h24 + case14/118 pending (case14/118 wait for final_e2e_qgmse retrains).
  - 2026-10-19, case500 h24 (job 39854226): 5218 / 94344 OPF failures (5.5 %) → MAE-Pd gate MISMATCH (1.433 vs 1.446,
    compare drops failed samples); Pg gates ok. n = 89126: E2E Pd → AC 2.48/12.87 vs E2E 2.78/11.36, AC xgb 2.57/13.03
    (Pg head −12 %); hedged 11.92, bias-corrected 11.53 (E2E −1.5 %). Failed samples excluded → subset may flatter Pd route.
    E2E Pd at the 219 zero-load buses sums to −68 MW (mean |.| 92 MW) at h24.
  - 2026-10-19, case14 h1/h6/h24 (job 39854258, all gates ok, 0 OPF failures, n = 3934/23604/94344):
    E2E Pd → AC 0.246/0.726, 0.677/2.205, 1.626/4.789 vs E2E 0.141/0.428, 0.573/1.965, 1.412/4.421, AC sarima
    0.222/0.703, 0.933/3.124, 1.889/5.773 → Pg head −41 / −11 / −8 % RMSE; total |ΣΔPg| 1.23/3.38/8.13 vs 0.70/2.86/7.03.
    Hedging gains nothing (0.725/2.206/4.782, affine dispatch) → case14 advantage = output target only.
  - 2026-10-22, case118 h1/h6/h24 (romeo job 9118246, all gates ok, 0 OPF failures, n = 3934/23604/94344; E2E s0):
    Pg MAE/RMSE E2E Pd → AC 0.406/2.04, 1.107/5.61, 1.743/8.95 vs E2E 0.268/1.51, 0.733/4.08, 1.711/9.20, AC xgb
    0.420/2.16, 1.087/5.82, 1.776/9.21 → Pg head −26 / −27 % at h1/h6 but +3 % WORSE at h24 (E2E s0 is the worst of
    3 seeds there: 8.58–9.20). Total |ΣΔPg| 20.4/57.5/91.2 vs E2E 11.0/33.5/82.9; misplaced 0.78/1.16/1.47 vs
    1.74/3.05/4.78 → at h24 E2E's allocation error cancels its total-error edge. Raw gap still lower for E2E
    (4.38 vs 4.90 %). → output-target advantage holds on Pg RMSE everywhere except case118 h24.
- 2026-10-19: **Cross-arm Pg RMSE / gap with XGB+D (`xgb_tot`) and final E2E (Qg MSE)** — PRELIMINARY (read from per-run
  CSVs, not aggregate_results; E2E case118 h6 n = 2 seeds, case500 n = 1; others 3 seeds; surrogate 3 seeds).
  Pg RMSE MW, AC xgb / AC xgb_tot / AC sarima / E2E mean [range]:
  - case14 h1 1.27/1.18/0.70/0.45 [0.43–0.47]; h6 2.98/2.81/3.12/1.99 [1.96–2.04]; h24 4.51/4.67/5.77/4.33 [4.22–4.42]
  - case118 h1 2.16/1.80/1.58/1.50 [1.48–1.51]; h6 5.82/5.35/6.80/4.11 [4.08–4.13]; h24 9.21/8.77/12.51/8.91 [8.58–9.20]
  - case500 h1 3.57/3.11/3.38/3.28; h6 8.41/8.25/9.86/6.84; h24 13.03/12.77/16.26/11.36
  - → CORRECTS §9 summary "E2E best Pg RMSE on every case × horizon": vs XGB+D, E2E is worse at case500 h1 (3.28 vs 3.11,
    n = 1 seed) and case118 h24 (mean 8.91 vs 8.77, inside E2E's seed range). E2E keeps the lowest mean opt gap in every
    cell (e.g. case500 h1 0.494 vs xgb_tot 0.557 / sarima 0.521; case118 h24 4.31 vs 4.77).
  - E2E seed spread (max−min)/mean: 1–9 % Pg RMSE (case118 h24 7 %, case14 h1 8 %).
  - 2026-10-28: case500 h1 E2E s1 (job 4457402, 200 epochs): Pg RMSE 3.292 / gap 0.486 vs s0 3.279 / 0.494 → seed spread
    < 2 %; E2E's Pg RMSE deficit vs AC xgb_tot (3.11) at case500 h1 holds over 2 seeds; gap advantage holds too.
  - Surrogate case500 after the Qg fix: Pg RMSE 9.6–11.7 (h1, 3 seeds) vs pre-fix s0 8.76; gap 2.1–3.1 vs 2.74 → the
    Pg regression holds across seeds (graphkit findings §3); case14/118 surrogate unchanged or better.
- 2026-11-01: **final totals** (dispatch_diag rerun on final_e2e_qgmse s0 + XGB+D, capella 4506411/12, romeo 9238378
  for case500; out `phase_1c/dispatch_diag_final`, per-sample `step1_samples.parquet`; all Pg gates recomputed == logged).
  - Slope / r of per-sample ΣΔPg on ΣΔPd (h1 / h6 / h24): AC (all forecasters) 1.02–1.09, r = 1.00 (Pg error = load
    error by construction); E2E case14 0.45 / 0.81 / 0.84 (r 0.71–0.86), case118 0.37 / 0.69 / 0.82 (r 0.63–0.85),
    case500 0.53 / 0.72 / 0.75 (r 0.67–0.82) → E2E's Pg output damps its own summed Pd error, most at h1.
  - case500 volume / misplaced MW: AC xgb_tot 75.0/0.09, 235.5/0.11, 420.9/0.11; E2E 66.7/23.1, 187.6/35.0, 379.3/47.7
    → volume win, but 2 × misplaced (46–95 MW) exceeds it: E2E's Σ|ΔPg| is the larger one (its worse case500 Pg MAE).
  - Volume |ΣΔPg| / misplaced MW, case118 h1/h6/h24: AC sarima 14.9/0.1, AC xgb_tot 50.6/0.9, 89.0/1.1; E2E 11.0/1.7,
    33.5/3.1, 82.9/4.8 vs E2E's own |ΣΔPd| 19.8/56.1/89.0 → E2E wins on volume, pays a small allocation cost.
  - 2026-11-02, control route on identical Pd (e2e_pd_opf.py table h1/6/24, romeo 9238639-41, Pg gates ok; case500
    h24 MAE-Pd gate mismatch = known 5218 OPF failures): RMSE of summed error MW, E2E ΣPd / Pd → AC-OPF ΣPg / E2E ΣPg:
    case14 1.5/1.6/1.0, 4.6/4.9/4.4, 10.0/10.7/9.9; case118 25.6/26.3/15.0, 73.3/75.1/50.8, 120.5/123.6/119.0;
    case500 114.7/116.9/91.3, 321.0/327.3/277.9, 575.4/586.5/532.5 → the route passes the summed Pd error on 1:1
    (+2–7 %, losses), the Pg output damps it (−1 to −41 %), fading with H.
  - 2026-11-08, **Pd RMSE domain**: logged RMSE Pd (compare.py, E2E/surrogate logs, dispatch_diag) pools ALL buses,
    zero-load buses count with their error (2-step: exactly 0; E2E: small nonzero predictions). Load buses only (true
    Pd > 1e-3 MW; datakit stores zero-load buses as ~1e-7 MW noise, dumps as exact 0; 3 / 19 / 219 zero-load buses),
    2-step forecast vs E2E s0, h1/h6/h24: case14 0.24/1.05/1.68 vs 0.31/0.85/1.63; case118 0.36/1.58/2.36 vs
    0.71/1.52/2.46; case500 0.52/2.29/3.51 vs 0.81/2.14/3.44 → on load buses E2E is worse only at h1 (all grids)
    and case118 h24; at case500 h6/h24 the all-bus "E2E slightly worse" comes from its zero-load-bus predictions.
  - 2026-11-14, **MAE ≠ RMSE ranking** (combined CSV, E2E vs best 2-step; Pd: best forecaster per metric, Pg: 2-step AC,
    best forecaster per metric; ML arms mean over seeds, E2E case500 1 seed). E2E vs 2-step in %, h1/h6/h24:
    - Pd RMSE case14 +48/−12/−2, case118 +135/−4/+7, case500 +76/+5/+1; Pd MAE +145/+19/+6, +253/+30/+20,
      +146/+39/+14 → E2E worse in MAE in all 9 cells (all-bus pooling; zero-load buses weigh more in MAE).
    - Pg RMSE case14 −36/−29/−7, case118 −5/−24/+2, case500 +6/−17/−11; Pg MAE −34/−35/−13, −5/−24/0, +57/+9/+13
      → case14/118 same as RMSE; case500 E2E worse in MAE at every h. Consistent with allocation (§9 2026-11-09):
      2-step AC puts its error on few marginal units (large, rare → RMSE), E2E spreads small errors over many units (MAE).
    - Pg pooling includes units with Pmax = 0 / pinned units: zero error in every arm (all bound Pg to [Pmin, Pmax]) →
      same factor on every arm's RMSE/MAE, ratios unchanged (not checked numerically).
  - 2026-11-08, **2-step ΣPg error > ΣPd error = network losses**: exact OPF power balance → per sample
    ΣΔPg − ΣΔPd = ΔL (change in losses incl. shunt consumption). Slope of ΔL on ΣΔPd (marginal loss factor):
    case14 0.060–0.091 (R² 0.88–0.93), case118 0.020–0.025 (R² 0.73–0.85), case500 0.016–0.020 (R² 0.43–0.73);
    RMSE(ΣΔPg)/RMSE(ΣΔPd) = 1 + slope to 3 decimals for all 27 AC arm × case × H → the excess is entirely the
    load-driven loss change; ordering matches the average loss shares (§5: 3.7 / 2.2 / 1.7 %).
  - 2026-11-02, per-sample RMSE (step1_samples pd_rmse / pg_rmse; recomputed == logged for all arms, E2E = s0): Pg p90
    lower for E2E in all 9 cells (e.g. case118 h6 6.2 vs 8.3 MW), median lower in 5/9 (higher at case118/500 h1)
    → RETRACTED 2026-11-09 (interpretation only, numbers stand): "E2E's Pg gain sits in the bad hours", see next
    entry. Summed ΣPd error RMSE ÷ independent-error value (sqrt cancel_ratio): 1.5–10, same for AC and E2E Pd.
  - 2026-11-09, per-sample Pg RMSE redone with the current forecaster set (plain XGB excluded; 2-step AC = lowest
    pooled Pg RMSE; E2E s0; n = 3934–94344 per cell; rank-matched quantiles of the squared per-sample error):
    - p90 lower for E2E in 9/9 cells, median in 6/9 (higher: case118 h1, case500 h1; case118 h24 equal), p99 higher
      at case14 h24 (14.9 vs 14.1 MW) and case118 h24 (31.7 vs 27.2).
    - Not driven by a few outliers: in the cells where E2E clearly wins (case14 h1/h6, case118 h6, case500 h6/h24)
      the top 10 % of samples give 49–69 % of the MSE gap and the top 1 % 12–25 %, about their share of AC's own MSE
      (52–64 % / 14–20 %, heavy tail). E2E's quantile function lies below AC's at 99–100 % of ranks on case14 h1/h6
      and case118 h6 (whole distribution shifts), at 51–62 % on case500 h6/h24 (gain in the upper half).
  - 2026-11-09, volume vs allocation with the same arms (per sample: volume |Σ_g ΔPg|, allocation 2·misplaced =
    Σ_g|ΔPg| − |Σ_g ΔPg|; mean MW, 2-step AC / E2E): E2E volume lower in 9/9 cells (e.g. case118 h6 50.6 / 33.5).
    Allocation: case14 ≈ 0 for both; case118 0.2–2.1 / 3.5–9.6; case500 0.2 / 46–96 (2-step AC ≈ 0: the OPF puts
    the whole volume error on the marginal units, all with the same sign). E2E's Pg RMSE is higher exactly in the
    cells where its extra allocation error outweighs the volume gain: case118 h24 (8.77 / 9.20), case500 h1
    (3.11 / 3.28). MAE-type split, so it explains the RMSE ranking, not an exact RMSE decomposition.
  - 2026-11-14, RMSE vs MAE verdict (logged metrics, seed means, best forecaster per metric, E2E vs best 2-step arm,
    9 cells, % = E2E / 2-step − 1): Pg RMSE / MAE case500 h1 +6 / +58 (2 seeds, see next entry), h6 −17 / +9,
    h24 −11 / +13 (flip at h6, h24);
    case14/118 same sign (e.g. case118 h6 −24 / −24). Pd: E2E worse under MAE in 9/9 (RMSE better in 3/9: case14 h6/h24,
    case118 h6; e.g. case118 h1 +135 / +253). Vm same verdict 9/9, Va 8/9 (flip case500 h24).
    Hypothesis (unverified): 2-step AC has exactly zero error on units pinned at limits (and zero-load buses for Pd,
    pooled in the logged metrics) and concentrates its error on few marginal units; E2E spreads small errors over all
    → MAE counts the many small errors, RMSE the few large ones. Check: share of units / buses with |error| > tol.
  - 2026-11-14, combined CSV re-aggregated with seed std (romeo 9483097, command now in CLAUDE.md): 6314/6367 rows
    bit-identical; the other 53 = E2E case500 h1, which now has 2 seeds (old CSV of 2026-10-24: 1, s1 finished later;
    corrected_gap already used 2). Shift < 2 %: Pg RMSE 3.279 → 3.286 ± 0.009, Pd RMSE 0.680 → 0.692 ± 0.017,
    mean opt. gap 0.494 → 0.490 %. Seeds: E2E + surrogate 3 everywhere, except E2E case500 2 / 1 / 1 (h1 / h6 / h24).
  - 2026-11-14, E2E's Pd output as a 2-step load forecaster (control route E2E Pd → AC-OPF, seed 0, `_e2e_pd/phaseB_*`)
    vs best 2-step AC (Pg RMSE, xgb_tot / sarima): the route would be the best 2-step AC only in case14 h6 (2.21 vs
    2.81 MW); E2E's Pg output (1.99, seed mean) still beats it (−10 % instead of −29 %). Other 8 cells: route worse
    than the best 2-step AC (e.g. case118 h6 5.61 vs 5.35, case500 h24 12.87 vs 12.77) → story unchanged.
  - 2026-11-14, forecaster choice in rmse_slide (best per variable AND per optimizer): the Pg-best 2-step AC forecaster
    differs from the Pd-best (lowest per-bus Pd RMSE) in 3/9 cells (case14 h6, case118 h6, case500 h1: Pd SARIMA, Pg
    xgb_tot). In 9/9 cells the Pg-best (sarima vs xgb_tot, `_dispatch_diag/step1_split`) is the one with the smaller
    summed-load error RMSE (e.g. case118 h6 dD 71.9 vs 92.6 MW → Pg 5.35 vs 6.80) → Pg follows the total load error,
    not the per-bus Pd error (supports the Q1 mechanism).
  - 2026-11-14, summed-load-error RMSE E2E (s0) vs best 2-step (`step1_split` dD_rmse_mw): E2E smaller in 2/9 cells,
    case14 h6 (4.6 vs 5.8 MW) and case500 h6 (321 vs 335); larger or equal elsewhere. There, E2E's Pg gain could stem
    from its load forecast; the control rules it out (route Pg 2.21 vs E2E 1.97, 8.29 vs 6.84 MW).
  - 2026-11-14, why E2E's cost advantage (raw / rebalanced / no-refund gap lowest in 9/9 cells, rebalanced case14 h24
    a tie 0.724 vs 0.723 %) is more consistent than its Pg RMSE advantage (7/9): raw-gap split (`step1_split`, proxy =
    true optimal dispatch of the hour with the nearest total) AC / E2E: volume cost gap_volume 0.56–5.63 / 0.50–5.00 %,
    E2E lower in 9/9; allocation cost gap_alloc ≤ 0.19 / ≤ 0.12 % although E2E misallocates far more MW (case500
    46–96 vs 0.2 MW) → cost is dominated by the volume error; E2E's misallocations are cheap (between similarly priced
    units), while Pg RMSE weights every misplaced MW alike. Proxy is noisy (AC gap_alloc 0.18 % on case118 at ≈ 0 MW).
    Rebalancing ∝ Pmax is cost-blind → its cost grows with |volume| → smaller E2E volume error = cheaper correction.
  - 2026-11-15, voltages: two error sources (load-driven vs learning the physics). RMSE H1 / H6 / H24, 2-step AC vs E2E
    (best forecaster per metric, seed means): |V| [1e-3 p.u.] case14 0.30/1.04/1.72 vs 2.29/3.70/6.42, case118
    0.27/0.89/1.40 vs 4.12/4.46/7.35, case500 0.79/2.41/3.67 vs 4.91/4.92/6.04; θ [deg] case14 0.07/0.29/0.49 vs
    0.18/0.41/0.65, case118 0.09/0.36/0.59 vs 0.43/0.55/0.73, case500 0.45/1.31/1.99 vs 1.24/1.82/1.96. 2-step AC grows
    ×4.4–6.7 H1→H24 (load error ×6.6–7.1; Qd ∝ Pd, so |V| follows the load), E2E only ×1.2–3.5 → E2E's voltage error
    is dominated by learning the physics (already 2.3–4.9e-3 p.u. at H1), 2-step AC's by the (small) load error.
    Surrogate |V| +166 … +4094 % vs 2-step AC.
  - 2026-11-16, naive baselines on the final GT test hours (≥ 22338 of 26280, all buses pooled; romeo 9491728,
    `<data>/notes/vm_baseline_2026-11-16.py`): per-bus train mean / persistence (value at the origin) H1 / H6 / H24.
    |V| [1e-3 p.u.] case14 4.65 / 1.32 / 3.69 / 4.90, case118 5.01 / 1.07 / 3.33 / 4.60, case500 13.60 / 3.11 / 9.38 /
    13.21; θ [deg] case14 2.06 / 0.37 / 1.27 / 1.87, case118 2.42 / 0.54 / 1.74 / 2.47, case500 3.18 / 1.62 / 3.34 / 3.90.
    → E2E |V| worse than persistence at H1 on all grids (2.3 / 4.1 / 4.9 vs 1.3 / 1.1 / 3.1) and worse than the
    constant train mean at H24 on case14 / case118 (6.4 / 7.4 vs 4.7 / 5.0); case500 E2E beats both at H6 / H24.
    θ: E2E below both baselines on all grids and H. 2-step AC below both everywhere. |V| varies by only 4.7–13.6e-3
    p.u. around its per-bus mean (gen buses more than PQ: 5.3–19.6 vs 3.4–11.3). Supersedes the local-GT numbers
    (memory voltage-section-checks; case14/118 train-mean identical).
  - 2026-11-16, same vs 2-step AC with the SNaive load forecast (combined CSV, ≈ constant over H): |V| [1e-3 p.u.]
    case14 2.58, case118 2.49, case500 6.07–6.09; θ [deg] 0.80 / 1.06 / 2.90. E2E |V| above it on case118 at all H
    (4.12 / 4.46 / 7.35) and on case14 at H6 / H24 (3.70 / 6.42), below it on case14 H1 (2.29) and case500 (4.91 /
    4.92 / 6.04, H24 ≈ tie). E2E θ below it on all grids and H.
  - 2026-11-16, loss scale (graphkit `HeteroDataMVANormalizer` family: Pd/Qd/Pg/Qg / baseMVA, Vm stays p.u., Va
    deg → rad; no per-variable standardisation): variance around the per-bus mean (train-mean RMSE², above) |V|
    2.2e-5 / 2.5e-5 / 1.8e-4 p.u.², θ 1.3e-3 / 1.8e-3 / 3.1e-3 rad² → θ carries ~59 / 71 / 17× the loss scale of |V|
    (case14 / 118 / 500) at equal relative error. case500 (lowest ratio) is where E2E |V| beats SNaive → consistent with
    "|V| under-weighted in the loss"; not a controlled test (no loss-scaling ablation).
  - 2026-11-14, Pg significance E2E vs 2-step AC (deterministic): one-sample t-test over E2E seeds, n = 3, df = 2,
    significant at 5 % if |diff| > 4.30/√3 = 2.48 seed std. case14 / case118: diff = 12.3 / 18.9 / 3.3 and 5.2 / 28.4 /
    0.5 std (h1 / h6 / h24) → all significant except case118 h24 (+1.6 %, p = 0.52). case500: 2 / 1 / 1 seeds, not
    testable. 2-step AC value treated as fixed (SARIMA deterministic; XGB HPO + training run once).
    Covers training randomness only, not test-set sampling (no block bootstrap).
  - Bug fixed on the way: 2-step ΔD was summed over all forecast samples, Pg over OPF-solved ones (case500 h6/h24
    sarima: 6 no_dispatch) → ΔD now on the solved samples only; affects dD_abs_mw there by ≤ 6/23604 samples.
