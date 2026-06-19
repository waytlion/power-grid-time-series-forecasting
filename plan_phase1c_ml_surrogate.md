# Plan: ML Surrogate for the AC-OPF Step in Phase 1c

**Status: DESIGN PROPOSAL — architecture settled through review, NOT yet approved to code.**

This document is the output of an extended planning + code-reading discussion between the
thesis author and an assistant. It is meant to be handed to another agent/session as
complete context. The architecture has been pressure-tested against the actual code in
both repos and revised twice; the remaining open items are in Section 11. Do not start
coding before reading Sections 6 (residual-convention finding) and 11 (open items) and
confirming with the user.

**Major revision history (so a future agent understands why earlier notes may conflict):**
- v1 proposed adapting the surrogate's predictions into datakit-format parquets and
  reusing Thesis_Repo's `compare.py` (an "adapter" arm).
- v2 (THIS VERSION) **replaces the adapter** with: evaluate the surrogate **inside
  gridfm-graphkit** using its own (inherited) OPF metric code, emitting MLflow artifacts —
  exactly the way the End-to-End (E2E) model is already evaluated. This was triggered by
  the author's observation that the surrogate's task (`OptimalPowerFlowTask`) and the E2E
  task (`ST_ForecastOPFTask`) share metric code, and confirmed by finding that
  `compare.py` itself imports the *same* physics layers. The adapter, the
  `predict_step`, and the generator-`idx` remapping are all eliminated.

---

## 1. Background / motivation

The thesis compares paradigms for look-ahead AC-OPF. Two already exist:

- **`Thesis_Repo`** — the **2-step / forecast-driven** pipeline: baseline temporal models
  (SNaive, SARIMA, XGBoost, TGT — Phase 1b) forecast future loads, then those forecasts are
  fed into `gridfm-datakit`'s **exact AC-OPF solver** (Phase 1c) to obtain the dispatch.
  See `Thesis_Repo/readme.txt`.
- **`gridfm-graphkit-dev`** — the **1-step / End-to-End (E2E)** model (`ST_ForecastOPF`
  task, `ST_GNN_heterogeneous`): a single GNN predicts future optimal operating states
  directly from historical trajectories. Trained/evaluated via `train.sh`; its metrics are
  written as MLflow artifacts.

**This plan adds a third arm:** replace the *exact* AC-OPF solve in Phase 1c with a trained
**ML surrogate** for AC-OPF (a single-snapshot GNN — like the E2E model but with no
temporal/forecast dimension; it only learns "load state → optimal dispatch"). The thesis
then compares, on the *same* Phase 1b forecasts:

1. **E2E** (1-step ML) — exists.
2. **2-step + exact OPF** (datakit) — exists, unchanged.
3. **2-step + surrogate OPF** — **new, this plan.**

Motivation is the efficiency/feasibility/quality tradeoff: exact AC-OPF takes hours; a
surrogate forward pass takes seconds. Note (from the author): both ML approaches (E2E and
surrogate) are **approximate AC-OPF and do NOT produce DC results** — DC analysis is
relevant only to the datakit arm and is out of scope for the ML-vs-ML comparison.

---

## 2. What the surrogate is, concretely

- Config: `gridfm-graphkit-dev/examples/config/gnn_heterogeneous_gns/HGNS_OPF_datakit_case118.yaml`
- Task (TRAINING): `OptimalPowerFlow` (`OptimalPowerFlowTask`, `gridfm_graphkit/tasks/opf_task.py`) — **stock, unchanged.**
- Task (EVALUATION on forecasts): a **thin new subclass** of `OptimalPowerFlowTask` — see D2/Section 6.
- Model: `GNS_heterogeneous`, single snapshot.
- Trained like the E2E model (a `train.sh`-style script).

### 2.1 Why valid surrogate inputs need no exact OPF solution (verified)
`AddOPFHeteroMask` (`gridfm_graphkit/datasets/masking.py:106-156`) overwrites the dispatch
fields (`Vm`, `Va`, `Qg` by bus type; `Pg` for all gens) with `mask_value` (0.0) before the
model sees them. `Pd`/`Qd` (load) are never masked. So the only *real* inputs are the load +
static network parameters (bus-type flags, limits, shunts, `vn_kv`, gen limits/costs, branch
admittances). For a new forecasted-load scenario these come from (a) the forecast and (b)
static columns copyable from any Phase 1a ground-truth scenario. **No exact OPF solve is
needed to build inputs.** (Cross-checked against datakit `process_network.py:124`.)

---

## 3. The central architectural decision (v2)

**Evaluate BOTH ML approaches (E2E and surrogate) inside gridfm-graphkit, producing MLflow
artifacts via the same inherited OPF metric code. Keep the exact solver (datakit) in
Thesis_Repo's `compare.py`. Unify all three at an aggregation step.**

Why this is correct and low-risk (all verified by reading code):

- `_compute_opf_metrics` (optimality gap, power-balance residuals, thermal/angle/Qg
  violations, per-bus-type RMSEs) is defined in `OptimalPowerFlowTask`
  (`opf_task.py:58-287`) and **inherited unchanged** by `ST_ForecastOPFTask`, whose
  `test_step` builds OPF-format tensors and delegates to it
  (`st_forecast_opf_task.py:444`). ⇒ Surrogate and E2E compute metrics through the *same
  method*.
- `compare.py`'s `metrics.py` **imports the same physics layers** — `ComputeBranchFlow`,
  `ComputeNodeInjection`, `ComputeNodeResiduals` from `gridfm_graphkit.models.utils`
  (`metrics.py:22-24`, used at `:373-375`). ⇒ The datakit arm's residual/branch-flow
  definitions are literally the same code as graphkit's. The three arms share metric
  definitions; only the *implementations of orchestration* differ.
- This eliminates the v1 adapter entirely: no datakit-format reconstruction, no
  `predict_step`, no generator-`idx` remapping (graphkit indexes generators and computes
  branch flows internally).

Trade-off accepted: the surrogate's metrics are MLflow artifacts (E2E's world), not
`compare.py` CSVs (datakit's world). But a CSV↔MLflow reconciliation **already exists**
because the author already compares E2E (MLflow) against 2-step (CSV); the surrogate simply
joins the MLflow side. The unification is handled once, by the aggregator (D7). Net split is
cleaner: **both ML models → graphkit/MLflow; exact solver → datakit/compare.py.**

---

## 4. Decisions and rationale

### D1 — Third arm, exact-solve arm unchanged
Add the surrogate alongside the datakit exact-OPF arm (which stays the ground-truth-quality
baseline) and the E2E arm. User confirmed.

### D2 — Evaluate via a THIN new task subclass (replaces v1's `predict_step`)
**Training uses the stock `OptimalPowerFlowTask`** (no new task). **Evaluation on
forecasts uses a thin subclass** that overrides `test_step` for exactly one purpose: the
residual load-reference fix (Section 6). It reuses 100% of `_compute_opf_metrics`. We use
the `evaluate`/`test` path (not `predict`), because the labeled-eval-dataset (D5b) gives
graphkit true-future labels, so `test_step` computes all metrics directly and writes them as
MLflow CSV artifacts. `predict_step` is never needed and stays unimplemented.

### D3 — Split TRAIN and EVALUATE; train once per case, standalone
- **Train:** a standalone `train_surrogate.sh` in gridfm-graphkit (a copy of `train.sh`
  pinned to the OPF config + chronological split + MLflow tags), run **once per case** on
  capella (GPU). Produces a checkpoint + normalizer stats as MLflow artifacts. Deliberately
  a copy, NOT a reuse of `train.sh`, because the author actively edits `train.sh` for E2E
  experiments and the surrogate's entry must be stable.
- **Evaluate:** driven from Thesis_Repo Phase 1c (D6), per (case × horizon × baseline
  model), as a subprocess into `gridfm_graphkit evaluate`.
Rationale: training is expensive and amortized across all horizons/models; embedding it in
the Phase 1c loop would retrain wastefully. This mirrors how the author already trains the
E2E model standalone.

### D4 — Parameterize for multi-case/horizon from the start
Mirror `scripts/phase1c.sbatch` (`CASE`/`HORIZON` vars). The full study spans **3 grid
cases** (TBD; case118 is the first validation case) × **3 forecast horizons (1, 6, 24)** ×
**~5 baseline forecasters** (xgb, sarima, snaive, tgt, true) × **3 approaches** (exact,
surrogate, E2E). First validation pass: case118, then generalize. See D9 for the seed
dimension and the resulting run matrix.

### D5a — Leakage guard: chronological split for training (hoist existing code)
`OptimalPowerFlowTask` uses `LitGridHeteroDataModule`, which only wires the two *random*
splits. `split_dataset_by_time` (`utils.py:84+`) already exists and is already used by
`LitGridHeteroForecastDataModule._split_dataset` behind a `temporal_split: true` flag
(`hetero_powergrid_forecast_datamodule.py:34-70`). **Decision:** hoist that 3-branch
`_split_dataset` (temporal → load-scenario-id → random) into the base
`LitGridHeteroDataModule`; delete the now-duplicate override in the forecast datamodule
(inherits unchanged). Purely additive — configs without `temporal_split` are byte-identical.
The surrogate's TRAIN config sets `temporal_split: true` with cutoff aligned to Phase 1b/1c's
test window (value TBD, §11.3). Prevents the surrogate training on scenarios Phase 1c scores.

### D5b — A new "labeled eval dataset" (the main new Thesis_Repo artifact)
Per (case × horizon × baseline model), build gridfm-graphkit-format graph parquets where:
- `bus.Pd`, `bus.Qd` = **forecast** load (what the model sees as input),
- `bus.Qg`, `bus.Vm`, `bus.Va` and `gen.p_mw` = **true-future OPF solution** at the realized
  target timestep (these become the label `y` via `process()`'s `y = x[:, :5]`),
- static columns (bus type, limits, shunts, `vn_kv`, gen limits/costs, branch
  admittances/ratings/angles) = **copied from a Phase 1a ground-truth scenario** (sources the
  admittance matrix from datakit itself — no reimplementation),
- an **extra attribute carrying the true-future load** (`true_Pd`/`true_Qd`), used by the D2
  task for the residual fix (Section 6),
- `scenario` ids **contiguous 0..N-1** (required by `process()` and `normalizer.fit()`).
Built from the **same `PRECOMPUTED_DIR/<model>.csv` that the datakit arm consumes**, so both
2-step arms run on byte-identical forecasted loads (fairness guarantee). Masked dispatch
fields can hold any finite value (they're zeroed post-normalization — §11.2/resolved); we use
the true-future values so the same columns double as the label.

### D5c — Evaluate on the full external eval set
The datamodule always splits train/val/test, and `evaluate` tests only the test split. For a
pure forecast eval set we want metrics on ALL scenarios. Add a small additive datamodule
option (e.g. `eval_full_as_test: true` → assign everything to the test split). Zero-code
fallback if we defer this: a config with `test_ratio≈0.99, val_ratio≈0.0` (the unused ~1%
"train" split is simply ignored at eval since we load a checkpoint).

### D6 — Orchestration: training decoupled; Phase 1c drives eval; independent tracks
- **Track A (barnard/CPU, unchanged):** `transform_forecasts → datakit exact solve →
  compare.py → CSV`.
- **Track B (capella/GPU, NEW):** `build labeled eval set → [gridfm_graphkit evaluate] →
  MLflow CSV artifacts`. Cheap once trained (forward passes only).
- **Aggregate:** tag-driven collector → unified 3-way table.
No SLURM dependency-chaining (rejected in v1 for the same reasons: extra failure modes for a
single-user watched pipeline). Independent, idempotent, manually-invoked steps. The
expensive costs (datakit hours on barnard; surrogate *training* hours on capella) are both
one-time/decoupled; per-Phase-1c work is cheap on both sides.

### D7 — Unification via tagging convention + one aggregator (fixes the MLflow pain)
Root cause of the author's current manual "MLflow UI → download → hand-copy into summary
CSV" grind: runs aren't machine-findable. The fix is to make every run self-describing and
let one script reduce everything to a tidy table. **Keep the MLflow concepts distinct:**
- **TAGS** (categorical identifiers, used to *filter/group* runs):
  `approach` ∈ {exact, surrogate, e2e}, `case`, `horizon` ∈ {1,6,24}, `model` (the baseline
  forecaster feeding the OPF step ∈ {xgb,sarima,snaive,tgt,true}; "integrated"/N-A for E2E),
  `seed` ∈ {0,1,2}, `split`, and `hardware`/`partition` (capella-GPU vs barnard-CPU — needed
  to read the timing metrics fairly, since the arms run on different hardware). Queryable via
  `mlflow.search_runs(filter_string="tags.approach='surrogate' and tags.case='case118'")`.
- **PARAMS** (config context): `num_train_samples`, `num_eval_samples`, `batch_size`, etc.
  (needed for the per-sample timing normalization in D10).
- **METRICS** (measured values, what we compare): the OPF quality metrics **and the timing
  metrics from D10** (training/inference time, absolute + per-sample). Timing is a metric,
  **not** a tag.
- **Aggregator script** (`Thesis_Repo/exp1/generate_metrics/aggregate_results.py`, new): uses
  `mlflow.search_runs` to find tagged runs, reads each run's `artifacts/test/{ds}_metrics.csv`
  (plain files on the shared FS at `mlruns/{exp_id}/{run_id}/artifacts/test/`), parses into a
  tidy long table **`(approach, case, horizon, model, seed, metric, value)`**, and **ingests
  the datakit arm's `compare.py` CSVs into the same schema** (with `seed` = NA for the
  deterministic exact arm). It then **reduces over `seed`**: mean ± std for the stochastic ML
  arms (surrogate, e2e), pass-through for exact. The result pivots into the 3-way comparison.
  This also replaces the author's existing manual E2E-vs-2step reconciliation. Seeds are the
  reason this matters: in the manual workflow 3 seeds triples the hand-copying; in the tidy
  table `seed` is just one more group-by column → N× compute, ~1× human effort.
- Optional nice-to-have (shared-code change): flip the OPF task's
  `self.log(..., logger=False)` → `logger=True` so metrics also land in the MLflow tracking
  store and `search_runs` returns them as columns (no CSV parsing). The artifact-CSV reader
  works without this.

### D8 — Code ownership
**gridfm-graphkit** (owns model/task/training/eval framework):
- new thin eval-task subclass of `OptimalPowerFlowTask` (D2/Section 6),
- hoist `_split_dataset` into base `LitGridHeteroDataModule`; delete forecast override (D5a),
- small `eval_full_as_test` datamodule option (D5c),
- `train_surrogate.sh` (D3) and an eval entry, both setting MLflow tags + params (D7),
- timing capture for train + inference as MLflow metrics (D10) — via the Lightning `Timer`
  callback already present in `get_training_callbacks` and a wall-clock around `evaluate`.
**Thesis_Repo** (owns baseline forecasters + orchestration + comparison):
- `scripts/phase1c_build_surrogate_eval_set.py` — the labeled-eval-set converter (D5b),
- `scripts/phase1c_surrogate.sbatch` — Track B (capella), subprocess into
  `gridfm_graphkit evaluate` (D6); seed-swept (D9),
- datakit solve-time capture in `phase1c_run_datakit_batch.py` (D10),
- `exp1/generate_metrics/aggregate_results.py` — tagging-aware, seed-reducing 3-way
  aggregator (D7),
- `readme.txt` — document Track B in the existing Phase 1a/1b/1c style.

### D9 — Experimental rigor: 3 seeds for ML arms, deterministic exact arm, and the run matrix
- **Seeds:** the surrogate and E2E are stochastic trained GNNs → run **3 seeds each**, report
  **mean ± std**. The datakit exact arm (AC and DC) is a **deterministic** solver → **1 run,
  no seeds** (seed-averaging it would be meaningless). This keeps the two ML methods on an
  equal, defensible footing while not over-running the solver. (3 chosen by the author.)
- **KEY efficiency point — the surrogate is horizon-independent.** It maps a load snapshot →
  dispatch and does not care how far ahead that load was forecast. So **one trained surrogate
  per (case, seed) serves all 3 horizons and all baseline forecasters.** Training cost is
  therefore only **3 cases × 3 seeds = 9 surrogate trainings**, each reused across
  3 horizons × ~5 forecasters. (Contrast: the E2E model is horizon-aware; how many E2E
  trainings depends on whether one h=24 model covers h=1/6/24 or you train per horizon — the
  author's call.)
- **Run matrix (per the D4 axes):**
  - Surrogate: 9 trainings (one-time) + 3×3×5×3 = **135 cheap eval runs** (forward passes).
  - E2E: 9 or 27 trainings (author's choice) + its own evals.
  - Exact (datakit): 3×3×5 = **45 deterministic solve-batches**, no seeds.
  The expensive, one-time cost is the trainings (9 surrogate + E2E's); per-Phase-1c work is
  cheap. This is why the surrogate's seed burden is much smaller than it first sounds.

### D10 — Timing capture (separate instrumentation work-item; metrics, not tags)
Reported **absolute and per-sample**. Per the author, **inference time is the headline
metric; training time may be excluded from the thesis** (capture it cheaply if free, but do
not over-invest).

- **PRIMARY — inference / "time to produce a dispatch decision".** The apples-to-apples
  comparison is at the **OPF step**, where the three arms are directly equivalent:
  **datakit AC/DC solve time ≡ surrogate forward-pass time ≡ E2E forward-pass time.** This is
  where the surrogate's value proposition lives (hours → milliseconds). Log all three under
  the **same metric name** (`infer_time_total_s`, `infer_time_per_sample_s`) so the aggregator
  compares them as one column. (Optional second level — *full-pipeline* inference for the
  2-step-vs-E2E end-to-end view = forecaster inference + OPF-step time for the 2-step arms vs a
  single pass for E2E; the forecaster term cancels in the exact-vs-surrogate OPF-step
  comparison, so it only matters for the vs-E2E view.)
- **SECONDARY / OPTIONAL — training time** (one-time, amortized; the author may omit it).
  Note the correct semantics: the OPF **solver** has no training; the **2-step arm's** training
  cost is the **forecaster** fit (XGBoost/SARIMA/TGT — owned by Phase 1b), and the
  surrogate-2-step additionally includes the **surrogate** training. The E2E arm's is the
  single E2E model training. So if training time IS reported, the 2-step number is forecaster
  (+ surrogate for that variant), not zero.

Capture mechanism (small, separate instrumentation item — **no** new Lightning Task class;
rides on callbacks + wall-clock):
- ML inference time (PRIMARY): wall-clock around the `evaluate` forward passes →
  `infer_time_total_s`, `infer_time_per_sample_s` as MLflow metrics.
- Datakit solve time (PRIMARY): wall-clock per scenario/batch in
  `phase1c_run_datakit_batch.py` — but first check whether datakit already emits per-scenario
  solve time via `enable_solver_logs` (§11.9); reuse if so. Same metric names as the ML arms.
- ML training time (OPTIONAL): the Lightning `Timer` callback already wired in
  `get_training_callbacks` (`cli.py`) makes this nearly free → `train_time_total_s`,
  `train_time_per_sample_s`.
**Per-sample denominators differ by arm** (surrogate/E2E sample = one graph/scenario forward
pass; datakit sample = one OPF solve). They are comparable as "time per dispatch decision,"
but the aggregator must divide each arm's total by **its own** scenario count — a silent
skew otherwise.
**Fairness caveat (record, don't hide):** arms run on different hardware (surrogate/E2E
capella-GPU; datakit barnard-CPU). The honest, operationally meaningful comparison is
"as-deployed," but the **`hardware`/`partition` tag (D7) must be recorded** so the GPU-vs-CPU
caveat is explicit in the write-up.

---

## 5. (reserved — see Section 6 for the residual convention)

---

## 6. Critical finding: the power-balance residual load-reference (justifies D2)
The three arms must compute the power-balance residual the same way to be comparable.
- **Exact arm** (`compare.py compute_algebraic_power_residuals`): residual = predicted
  dispatch vs **`Pd_true`/`Qd_true`** — the *true* realized load (`metrics.py:394-395`).
- **E2E arm**: `_compute_opf_metrics` is invoked with `target_batch` as `batch`, whose load
  is the **true-future** load (`st_forecast_opf_task.py:444`) → residual vs true load.
- **Plain OPF task**: residual uses `batch.x_dict["bus"]` = the **input** load
  (`opf_task.py:149-154`). For the surrogate the input is the *forecast* → residual vs
  forecast (intrinsic feasibility, *excludes* forecast error).

⇒ To match the other two arms (operational feasibility vs the realized load), the surrogate
must compute its residual vs the **true-future** load while still feeding the **forecast** to
the model. The stock OPF task can't separate these (it uses `x` for both). **Hence the thin
D2 task:** override `test_step` to substitute the true-future load (carried as the D5b extra
attribute) into the load slots before calling `_compute_opf_metrics`. One small override; all
metric math reused. Note: optimality gap and all violation metrics are cost/limit-based with
no load reference, so they are unaffected and already consistent — **only the residual needs
this fix.**

---

## 7. End-to-end shape
```
ONE-TIME, per case (author runs, checks MLflow convergence):
  gridfm-graphkit:  train_surrogate.sh
    → OptimalPowerFlow task, chronological split, tags{approach=surrogate,case,split}
    → checkpoint + normalizer stats (MLflow artifacts)

PHASE 1c  (Thesis_Repo orchestration, per case × horizon):
  Step 0 (cheap, once): scripts/phase1c_transform_forecasts.py   [UNCHANGED]
        → PRECOMPUTED_DIR/<model>.csv   (shared by both 2-step arms)

  Track A — barnard/CPU  [UNCHANGED]:
        phase1c_run_datakit_batch.py → datakit exact solve → OPF_OUT/exact/<model>/...
        compare.py → exp1/results/.../exact/   (CSV)

  Track B — capella/GPU  [NEW]:
        scripts/phase1c_build_surrogate_eval_set.py
            reads PRECOMPUTED_DIR/<model>.csv + Phase 1a truth
            → labeled eval graphs (forecast input, true-future label, true-load attr)
        scripts/phase1c_surrogate.sbatch  →  subprocess:
            gridfm_graphkit evaluate --config <opf eval cfg> --model_path <ckpt>
                --normalizer_stats <stats> --data_path <eval set>
                (thin eval task; tags{approach=surrogate,case,horizon,model})
            → MLflow artifacts test/{ds}_metrics.csv, test/{ds}_RMSE.csv

  Aggregate (cheap):  exp1/generate_metrics/aggregate_results.py
        mlflow.search_runs(tags) → read MLflow test CSVs   (surrogate + E2E)
        + ingest compare.py CSVs                            (datakit arm)
        → unified tidy table → 3-way comparison
```

---

## 8. Implementation checklist (once approved)
Validate each step before the next (tests and/or smoke run). Do not batch.

1. [graphkit] Hoist `_split_dataset` into base `LitGridHeteroDataModule`; delete forecast
   override (D5a). Run `tests/test_data_module.py` — confirm no change without `temporal_split`.
2. [graphkit] Add `eval_full_as_test` datamodule option (D5c) — or decide on the
   `test_ratio≈0.99` fallback.
3. [graphkit] Write the thin eval-task subclass (D2/Section 6): override `test_step` to
   substitute the true-future load into the residual computation; everything else inherited.
   Register it (e.g. task name `OptimalPowerFlowForecastEval`).
4. [graphkit] `train_surrogate.sh` (D3) + an eval entry/config; set MLflow tags + params
   incl. `seed` (D7) and capture timing metrics (D10). Add `temporal_split: true` + aligned
   ratios to the surrogate train config (cutoff TBD §11.3).
5. [cluster] Train the surrogate as a **3-seed sweep** (case118 first, then 3 cases → 9
   trainings total; D9). Multi-hour GPU; start early. Confirm convergence in MLflow; note
   checkpoint + normalizer-stats paths per seed. Remember: one surrogate per (case, seed)
   serves ALL horizons.
6. [Thesis_Repo] `scripts/phase1c_build_surrogate_eval_set.py` (D5b). Carry the true-future
   load attribute; emit contiguous scenarios; copy static cols from Phase 1a.
7. [Thesis_Repo] `scripts/phase1c_surrogate.sbatch` (Track B, capella) — subprocess into
   `gridfm_graphkit evaluate` with checkpoint + normalizer stats as top-of-script vars;
   seed-swept (D9); records inference timing (D10). Also add datakit solve-timing to
   `phase1c_run_datakit_batch.py` (Track A) so all three arms report timing.
8. [Thesis_Repo] Smoke run Track B for case118, ONE model (e.g. `xgb`); sanity-check the
   MLflow `{ds}_metrics.csv` (residual vs true load; opt gap finite/plausible).
9. [validation] One-time numerical check: run the surrogate's metrics (graphkit) and
   `compare.py` on an *identical labeled set*; confirm optimality gap and residual agree
   numerically (definitions are shared code, so they should — this catches plumbing bugs).
10. [Thesis_Repo] `exp1/generate_metrics/aggregate_results.py` (D7): tag-driven collection,
    **seed reduction (mean ± std)**, timing + quality metrics unified across all 3 arms.
11. [Thesis_Repo] Update `readme.txt` (Track B section).
12. Full run: scale to 3 cases × 3 horizons (1,6,24) × ~5 forecasters × 3 ML seeds (D9); all
    three arms; unified seed-reduced 3-way table with quality + timing metrics.

---

## 9. Things explicitly NOT changed
- `scripts/phase1c_transform_forecasts.py`, `phase1c_run_datakit_batch.py`,
  `phase1c.sbatch` exact-OPF behavior — unchanged.
- `exp1/generate_metrics/compare.py`, `loaders.py`, `metrics.py` — unchanged (the surrogate
  no longer routes through them; the datakit arm still does).
- Any graphkit config not setting `temporal_split`/`eval_full_as_test` — unchanged behavior.
- The E2E model and its training/eval — untouched; comparison only.
- `predict_step` for any OPF task — stays unimplemented (we use `evaluate`, not `predict`).

---

## 10. Investigation status (resolved findings)
- **§ Premise (graphkit metrics ≡ compare.py metrics): VALIDATED at code level.**
  `_compute_opf_metrics` is shared by surrogate/E2E (inheritance) and `compare.py` imports
  the same physics layers. A one-time *numerical* spot-check is still worth doing (checklist 9).
- **§ Placeholder safety + transform order: RESOLVED.** Normalization runs BEFORE masking
  (`HeteroGridDatasetDisk.get()`), so masked dispatch fields are zeroed after normalization →
  raw placeholder values are irrelevant. The normalizer's only fitted stat is `baseMVA` =
  95th pct of non-zero {Pd,Qd,Pg,Qg} (`normalizers.py:152-167`) → at eval we **MUST** pass
  `--normalizer_stats` from training, else it refits wrong (placeholder-zero Pg/Qg excluded).
  Contiguous-`scenario` assertion confirmed in `process()` and `fit()`.
- **§ Generator `idx` alignment: now MOOT.** With the graphkit/MLflow approach there is no
  datakit-format adapter, so graphkit indexes generators internally and no cross-repo `idx`
  remapping is needed. (It was the v1 plan's top risk; v2 removes it.)

---

## 11. Open uncertainties — resolve or accept as risk BEFORE coding
- **11.1 Architecture confidence.** v2 is reasoned from code, not yet from a real run.
  Re-confirm after checklist step 8/9. The author has repeatedly (and rightly) said "not 100%
  sure this is the best way" — keep that posture.
- **11.2 Static-column constancy.** D5b copies static columns from one Phase 1a scenario,
  assuming they're constant across scenarios. The generation config has all perturbations
  `none`, but re-check against the ACTUAL generated case118 dataset before relying on it.
- **11.3 Chronological cutoff value.** Read Phase 1b's real case118 train/test boundary out of
  its config/outputs and set the surrogate's `temporal_split` ratios so the surrogate's test
  window starts at/before it (no leakage).
- **11.4 Capella env for Track B.** The new `phase1c_surrogate.sbatch` runs on capella and
  shells into `gridfm_graphkit evaluate`; confirm module/venv setup (the `thesis_env` used by
  `train.sh`) and that it reads/writes the shared `/data/horse/ws/...` tree (datakit Track A
  is barnard). Not yet end-to-end verified for this new job.
- **11.5 `eval_full_as_test` vs `test_ratio≈0.99`.** Decide whether to add the clean
  datamodule option (recommended) or accept the zero-code fallback that ignores ~1% of eval
  scenarios.
- **11.6 Metric-set coverage.** `compare.py` computes a few extras the graphkit OPF task does
  not surface directly (Vm-limit violations, branch-flow RMSE vs true flows, median/max opt
  gap). The shared core (opt gap, P/Q residuals, thermal/angle violations, gen RMSE) is what
  the 3-way comparison rests on and is fully covered. Forecast accuracy (Pd MAE) for both
  2-step arms comes from Phase 1b, not from the OPF metrics. Decide if any extra is needed for
  the surrogate arm; if so, add it to the thin eval-task's `on_test_end` (it already overrides
  CSV layout in the E2E task as precedent).
- **11.7 Surrogate quality is itself a research question.** Build infrastructure to MEASURE
  it; a poor surrogate is a valid finding, not a plan failure.
- **11.8 Grid case selection.** The study spans 3 cases (D4); case118 is decided as first,
  the other 2 are TBD by the author. The converter/scripts are parameterized by `CASE` so this
  is a config choice, not a code change — but Phase 1a ground truth must exist for each.
- **11.9 Datakit solve-timing source.** Before instrumenting (D10), check whether datakit
  already emits per-scenario solve time in its solver logs (`enable_solver_logs: true` in the
  Phase 1a/1c config) — reuse it if so, rather than adding wall-clock.
- **Decided (no longer open):** seed count = **3** for the ML arms; exact arm deterministic
  (D9). Horizons = **1, 6, 24** (D4).

---

## 12. Summary for a future agent
The plan: train a stock single-snapshot OPF GNN as an AC-OPF surrogate (chronological split,
standalone `train_surrogate.sh`, once per case); in Phase 1c, build a **labeled eval dataset**
(forecast input + true-future label + true-load attribute) from the same forecasts the
datakit arm uses, then evaluate via a **thin `OptimalPowerFlowTask` subclass** whose only new
logic is computing the power-balance residual against the **true-future** load (Section 6) —
all other metrics are inherited and identical to the E2E model's. Metrics land as MLflow
artifacts; a **tag-driven aggregator** unifies surrogate (MLflow) + E2E (MLflow) + datakit
(`compare.py` CSV) into one 3-way table, replacing the author's manual UI-download workflow.
No adapter, no `predict_step`, no generator-`idx` remapping. **Experimental design (D9/D10):**
3 seeds for the stochastic ML arms (surrogate, E2E) → mean ± std; the datakit exact arm is
deterministic → no seeds. The surrogate is **horizon-independent** (one model per (case,seed)
serves all of horizons 1/6/24 and all forecasters), so only **9 surrogate trainings** total.
Timing is captured as **metrics** (not tags): **inference is the headline** — datakit solve
time ≡ surrogate ≡ E2E forward pass, the OPF-step "time per dispatch decision" (training time
is optional, may be excluded; the 2-step's training is the forecaster fit). `seed` and
`hardware` are **tags**; the aggregator reduces over `seed`. Biggest
correctness levers: pass `--normalizer_stats` at eval (§10), the residual load-reference fix
(Section 6), and the chronological split (D5a, §11.3). Read Sections 6 and 11 first; confirm
with the user before coding.
