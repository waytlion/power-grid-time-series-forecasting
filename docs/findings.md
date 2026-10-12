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
- Datakit PF path (Network + pf_preprocessing setters + `run_pf(fast=True)`) reproduces GT labels to 1e-6 on
  case14/case118. case500: per-gen Qg differs 22-31 MVAr (PowerModels `_assign_qg!` split) → evaluate Qg at bus level
  against summed limits.
- 2026-10-12: step 1 changed from proportional pre-scaling to merit-order rebalancing (`metrics.merit_order_rebalance`,
  same rule as §8). The full run of 2026-10-11 (jobs 39467094-6) used proportional scaling → outdated, rerun needed.
- Design: Δ = ΣPg_true − ΣPg_pred (oracle scalar = realized system requirement); no warm-start retry; run_pf error =
  non-converged; each arm uses its own timestamps.
- Cost: ~4 ms (case14), 20-100 ms (case118), 80 ms (case500) per PF; ~2.6M solves ≈ 50 CPU-h.
- Julia needs `module load release/2026 GCC/14.2.0` + `LD_PRELOAD=/software/genoa/r2026/GCCcore/14.2.0/lib64/libstdc++.so.6`.
- Smoke job 39459686 (case118 h6, 200 samples): label + exact_true have 0 violations, ΔP_slack 0 → pipeline valid.
  Qg max violation per sample: exact_xgb ~1 MVAr, surrogate ~88, e2e ~55. Thermal share (~47%) dominated by lines at
  ~1.002 loading → report magnitudes, not share alone.

## 8. Rebalanced optimality gap (2026-10-12, `exp1/generate_metrics/corrected_gap.py`)

- Why: in forecast-then-optimize the raw gap mostly measures energy volume (forecast error): under-supply looks
  cheap, over-supply expensive; `abs()` makes both positive. Signed raw gap cancels (AC h1: |γ| 0.82 % vs signed +0.02 %).
- Method, per scenario: Δ = ΣPg* − ΣPg_pred closed by merit order on the predicted dispatch (ranked by c1 + 2c2·Pg_pred;
  shortfall → cheapest up to Pmax, surplus → most expensive down to Pmin; remainder → slack = violation), then
  γ_reb = (C(Pg_reb) − C*)/C*, reported as mean |γ_reb|; rebalancing cost R = (C(Pg_reb) − C(Pg_pred))/C*, signed.
  Gate: raw gap recomputed from the per-gen dumps reproduces the logged gap (all 28 case118 arms, |Δ| < 1e-3).
- Discarded (same session): single λ (mean marginal cost of non-binding units, ≈22.3 €/MW) with lower/upper surplus
  bounds. Ex-post `mean|γ| + λ·shortfall` double-counts (AC h1 upper 1.15 vs exact 0.40). The allocation ranking was
  λ-sensitive (λ_s ×1.08 flips AC vs E2E), and "share below optimum" was pricing noise (AC 42–47 % at ε = 0.01 %).
- case118, best forecaster per optimizer, ML arms seed 0 (Table 7). Columns: raw |γ| / |γ_reb| / R / ΔP mean (MW).
  - h1: AC 0.822/0.210/−0.23/+0.3; DC 3.729/2.039/+1.67/−53.4; surrogate 1.461/0.222/−0.90/+9.0; E2E 0.553/0.164/−0.05/−2.6
  - h6: AC 3.035/0.646/−0.93; DC 4.490/2.058/+1.45; surrogate 3.208/0.539/−1.46; E2E 1.772/0.458/−0.34
  - h24: AC 4.987/0.891/−1.50; DC 5.804/2.082/+1.12; surrogate 5.043/0.803/−1.82; E2E 4.295/0.931/−1.07
  - Slack overflow 0 % everywhere (enough headroom). Rebalancing removes 70–85 % of the raw gap for AC/surrogate/E2E.
  - After rebalancing: E2E is best at h1/h6, the surrogate at h24. DC is about 2 % at every horizon: it is ~53 MW short
    (missing losses) and its mix ignores AC limits.
- Caveat: R < 0 on average even when mean ΔP ≈ 0 (AC). Up-steps buy the cheapest units and down-steps remove the most
  expensive ones, so rebalancing is optimistic in both directions and the network is ignored. The ex-post AC-PF
  (§7) checks operability of the same rebalanced dispatch.
- case14 / case500: earlier λ-based results (2026-10-10) showed a surrogate allocation penalty (case14 ~0.59 %,
  case500 1.25–1.42 %) not present in case118. Not yet recomputed with merit order. case500 E2E is invalid until the
  retrain (stale processed_temporal).
