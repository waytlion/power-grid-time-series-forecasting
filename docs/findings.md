# Findings — Thesis_Repo (data gen, forecasters, exact/2-step arms, aggregation)

Measured results, retractions, and implementation rationale. Dated; newest facts win.
RETRACTED entries stay so they are not re-proposed. ML-arm findings: `~/gridfm-graphkit-dev/docs/findings.md`.

---

## 1. Cross-arm methodology (state these; they cut against the ML arms)

- **Splits**: forecasters + E2E base = 70/15/15; surrogate + E2E tiny = 80/10/10. Chronological — `temporal_split`
  is checked BEFORE `split_by_load_scenario_idx` in the datamodule branch.
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

