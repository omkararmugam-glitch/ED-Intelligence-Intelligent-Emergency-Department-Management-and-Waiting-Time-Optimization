# Phase 1c checkpoint — explainability and fairness

## Gates

| Gate | Result |
| --- | --- |
| Full test suite | **198 passed** |
| Pipeline end to end on synthetic | **8 stages, 15.92 s** |
| `tests/test_leakage.py` | 12 passed |
| No random splits in `src/` | 9 patterns tested, 0 found |
| Every tunable from config | asserted by `test_project_invariants.py` |

## What was built

| Requirement | Module | Status |
| --- | --- | --- |
| 1. Global SHAP summary + top-feature CSV + local explanations | `src/explain/shap_report.py` | done |
| 2. Subgroup MAE / FNR / FPR with bootstrap CIs, graceful skips | `src/explain/fairness.py` | done |
| 3. Test that fairness runs with an optional column removed | `tests/test_explain.py` | done (4 variants) |

New pipeline stages: `explain`, `fairness`. 19 new tests in `tests/test_explain.py`.

## 1. SHAP

Explained model: **base+queue / log1p**
(the cell the Phase 1b ablation selected), TreeExplainer over
4000 test-block rows, capped by `explain.shap_max_samples`.

**Units: `log1p(minutes)`.** The winning model was fitted on a log target, so
SHAP values live in log space and do **not** convert additively to minutes. Every table,
figure axis and JSON field carries the unit label rather than implying minutes.

Top 10 features by mean |SHAP|:

| rank | feature | mean_abs_shap | mean_shap | share_of_attribution | is_queue_feature |
| --- | --- | --- | --- | --- | --- |
| 1 | n_waiting | 0.5593 | 0.0786 | 0.4196 | True |
| 2 | triage_level | 0.2616 | 0.0069 | 0.1962 | False |
| 3 | minutes_since_midnight | 0.1338 | 0.0044 | 0.1004 | False |
| 4 | n_waiting_same_priority | 0.1083 | -0.0035 | 0.0812 | True |
| 5 | n_waiting_higher_priority | 0.0965 | 0.0021 | 0.0724 | True |
| 6 | n_in_treatment | 0.0489 | 0.0004 | 0.0367 | True |
| 7 | hour_cos | 0.0315 | -0.0023 | 0.0236 | False |
| 8 | arrivals_last_1h | 0.0231 | 0.0044 | 0.0173 | True |
| 9 | day_of_year | 0.0159 | -0.0071 | 0.0119 | False |
| 10 | mean_wait_last_10_seen | 0.0109 | 0.004 | 0.0082 | True |

Queue-state features carry **64.0%** of total attribution. That is
consistent with the Phase 1b ablation and carries the same caveat: the generator
produces waits by simulating a queue, so this is an upper bound on what queue features
would contribute in a real department.

Local explanations: 4 worked patients chosen to span the
prediction range (not sampled at random, so a short wait and a long wait are both
illustrated). Example — the highest-prediction patient:

| patient_id | triage_level | observed_wait_minutes | feature | feature_value | shap_contribution |
| --- | --- | --- | --- | --- | --- |
| P0049390 | 5 | 168.9 | n_waiting | 21 | 1.276 |
| P0049390 | 5 | 168.9 | n_waiting_higher_priority | 20 | 0.4243 |
| P0049390 | 5 | 168.9 | triage_level | 5 | 0.2474 |
| P0049390 | 5 | 168.9 | minutes_since_midnight | 1241 | 0.1064 |
| P0049390 | 5 | 168.9 | n_waiting_same_priority | 1 | -0.0299 |
| P0049390 | 5 | 168.9 | hour_cos | 0.5 | 0.0288 |

Figures: `shap_importance.png`, `shap_summary.png`, `shap_local_examples.png`.

## 2. Fairness (research gap G2)

Overall on the test block (n = 7284): FNR 0.1193, FPR
0.0872, recall 0.8807, MAE 14.19 min.

Attributes evaluated: sex, arrival_mode, triage_level, age_band.
Skipped (absent): none.

| attribute | group | n | n_positive | fnr | fnr_gap | fnr_gap_ci_low | fnr_gap_ci_high | fnr_underpowered | mae | mae_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sex | female | 3706 | 1440 | 0.1208 | 0.0015 | -0.0192 | 0.0229 | False | 13.71 | -0.4867 |
| sex | male | 3578 | 1384 | 0.1178 | -0.0015 | -0.021 | 0.0189 | False | 14.7 | 0.504 |
| arrival_mode | ambulance | 1467 | 373 | 0.1635 | 0.0442 | 0.003 | 0.0858 | False | 11 | -3.192 |
| arrival_mode | other | 155 | 63 | 0.0794 | -0.0399 | -0.1044 | 0.0299 | False | 13.41 | -0.7802 |
| arrival_mode | walk_in | 5662 | 2388 | 0.1135 | -0.0058 | -0.0217 | 0.0122 | False | 15.04 | 0.8483 |
| triage_level | 1 | 159 | 2 | 1 | 0.8807 | 0.8672 | 0.8931 | True | 3.596 | -10.6 |
| triage_level | 2 | 899 | 29 | 1 | 0.8807 | 0.8696 | 0.8916 | False | 4.618 | -9.576 |
| triage_level | 3 | 2981 | 1172 | 0.1126 | -0.0067 | -0.0286 | 0.0172 | False | 15.13 | 0.9392 |
| triage_level | 4 | 2361 | 1171 | 0.1085 | -0.0108 | -0.032 | 0.0105 | False | 16.45 | 2.259 |
| triage_level | 5 | 884 | 450 | 0.1044 | -0.0149 | -0.0423 | 0.016 | False | 16.64 | 2.445 |
| age_band | 0-17 | 878 | 411 | 0.1533 | 0.034 | -0.0013 | 0.0724 | False | 16.16 | 1.965 |
| age_band | 18-39 | 2382 | 988 | 0.0921 | -0.0272 | -0.0493 | -0.007 | False | 14.58 | 0.3854 |
| age_band | 40-64 | 2876 | 1096 | 0.1159 | -0.0034 | -0.0265 | 0.0181 | False | 14.22 | 0.0283 |
| age_band | 65-79 | 820 | 246 | 0.1707 | 0.0514 | 0.007 | 0.0998 | False | 12.28 | -1.918 |
| age_band | 80+ | 328 | 83 | 0.1687 | 0.0494 | -0.0338 | 0.1302 | False | 10.68 | -3.513 |

### The finding that matters

**The classifier never warns a high-acuity patient who will wait a long time.**
ESI 2 has an FNR of **1.0000** — every one of its 29
prolonged-wait cases was missed — against an overall FNR of 0.1193. ESI 1
shows the same pattern but rests on only
2 positive cases and is
reported as under-powered rather than as a finding.

This is a direct consequence of how the threshold was chosen. The `min_recall`
objective optimises recall **pooled across all patients**, and ESI 1-2 patients are
3.2% positive against
roughly 39-51% for ESI 3-5. A single global threshold therefore buys its recall almost
entirely from the low-acuity majority and abandons the high-acuity minority. Pooled
recall of 0.881 looks healthy and conceals a subgroup at 0.0.

Clinically this is the wrong way round: a long wait matters *most* for an ESI 2 patient.
Mitigation (per-acuity thresholds, or an acuity-aware objective) belongs in future work;
Phase 1c's job was to detect and report it, which it does.

### Smaller, genuine gaps

- `arrival_mode = ambulance`: FNR gap +0.0442
  (CI +0.0030 to
  +0.0858) — but ambulance
  arrivals are mostly ESI 1-2, so this largely restates the acuity effect rather than an
  independent one.
- `age_band = 65-79`: FNR gap +0.0514; older
  patients skew higher-acuity, so the same confound applies.
- `sex`: gaps of +0.0015 and
  -0.0015, both with intervals spanning zero.
  **This is a null result by construction**, not a discovered absence of bias: the
  generator draws sex independently of everything else, so there is no disparity for the
  module to find. It demonstrates the code works; it says nothing about real EDs.

### A correctness fix made during this phase

The first run flagged ESI 1 as a finding with a tight bootstrap interval, off **two**
positive cases. The group-size guard used total rows, but **an FNR is estimated from
positives alone**, so a large group can carry almost no information about it. Added
`n_positive`, `n_negative`, `fnr_underpowered` and `fairness.min_positives_for_fnr`
(default 20). Under-powered groups are reported and logged, but never presented as
findings — and the log says explicitly that a tight interval over 2 cases reflects
resampling the same two cases, not reliability.

## Deviations from CLAUDE.md

None new. The Phase 1b deviations still stand (pyarrow; split keeps all rows;
`ablation` aliases `train_wait`).

One near-miss worth recording: `tabulate` was installed to render Markdown tables, then
**uninstalled** and replaced with `src/utils/report.py`, because it was not on the
approved dependency list. `requirements.txt` is unchanged.

## Known limitations

- SHAP attribution describes **this model**, not causation. `n_waiting` leading the
  ranking means the model leans on it, not that queue length causes waits — though in
  the generator it does, which is exactly why synthetic evidence cannot settle it.
- SHAP values are in log space for the winning model and are not additively convertible
  to minutes.
- Fairness subgroups are confounded with acuity. `arrival_mode` and `age_band` gaps
  largely track ESI composition; no adjusted (within-acuity) analysis is computed yet.
- The ESI 1-2 FNR failure is **detected but not mitigated**. Per-class thresholds are
  deferred to future work, as the project brief directs.
- Bootstrap intervals use 400 resamples
  (config: `fairness.bootstrap_resamples`) and are percentile intervals, which are
  adequate here but not bias-corrected.
