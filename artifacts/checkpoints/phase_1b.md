# Phase 1b checkpoint — wait-time models, ablation, conformal

## Gates

| Gate | Result |
| --- | --- |
| Full test suite | **179 passed** |
| Pipeline end to end on synthetic | **6 stages, 12.59 s** |
| `tests/test_leakage.py` | **12 passed** |
| Deliberate leak detected then reverted | see `phase_1a_leakcheck.md` |
| No random splits anywhere in `src/` | **9 forbidden patterns tested, 0 found** |
| Every tunable from `configs/default.yaml` | asserted by `test_project_invariants.py` |

## What was built

| Requirement | Module | Status |
| --- | --- | --- |
| 1. Four-block time split | `src/data/split.py` | done |
| 2. Rolling-mean-by-triage baseline | `src/models/baselines.py` | done |
| 3. Regressor, raw vs log1p target | `src/models/wait_time.py` | done |
| 4. Prolonged-wait classifier | `src/models/wait_time.py` | done |
| 5. Ablation table with caveat | `src/models/train_wait_stage.py` | done |
| 6. Mondrian split conformal | `src/models/conformal.py` | done |
| 7. Seen-only bias report | `src/models/wait_time.py` | done |
| 8. Regression-cohort exclusions | `src/models/wait_time.py` | done |
| 9. Tests for all of the above | `tests/test_models.py` (25 tests) | done |

## 1. The four blocks

| block | n_rows | share | start | end | regression cohort | excluded |
| --- | --- | --- | --- | --- | --- | --- |
| train | 29918 | 59.8% | 2024-01-01 | 2024-07-29 | 28912 | 1006 |
| val | 7548 | 15.1% | 2024-07-30 | 2024-09-25 | 7401 | 147 |
| calib | 4914 | 9.8% | 2024-09-26 | 2024-10-30 | 4719 | 195 |
| test | 7620 | 15.2% | 2024-10-31 | 2024-12-20 | 7284 | 336 |

Cuts: calib_end = 2024-10-31, train_end = 2024-07-30, val_end = 2024-09-26.

`calib` is separated from `val` deliberately: split conformal is only valid when the
calibration residuals are exchangeable with the test residuals, which fails if the
model was early-stopped on the same rows.

## 2-3, 5. Ablation (test block, n = 7284)

| feature_set | target | prediction_kind | estimates | mae_minutes | rmse_minutes | bias_minutes | best_iteration |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | raw | raw | conditional median (L1 objective on the minute scale) | 32.74 | 57.71 | -29.02 | 57 |
| base | log1p | log1p_median | conditional median (expm1 back-transform, no spread correction) | 33.17 | 58.47 | -29.84 | 60 |
| base | log1p | log1p_smearing | conditional mean (Duan smearing over empirical residuals) | 30.49 | 50.58 | -16.21 | 60 |
| base+queue | raw | raw | conditional median (L1 objective on the minute scale) | 14.2 | 24.14 | -6.051 | 130 |
| base+queue | log1p | log1p_median | conditional median (expm1 back-transform, no spread correction) | 14.19 | 24.15 | -6.467 | 113 |
| base+queue | log1p | log1p_smearing | conditional mean (Duan smearing over empirical residuals) | 14.41 | 22.82 | 1.031 | 113 |

Baseline (rolling mean of last 20 observed waits at the same ESI
level): **MAE 38.71 min, RMSE 56.99 min**.

Best cell by test MAE: **base+queue / log1p / log1p_median**
— MAE 14.19 min, RMSE 24.15 min.

**Raw vs log1p.** The two *median* estimators are indistinguishable on MAE
(14.20
vs 14.19),
while Duan smearing — the only *mean* estimator — loses slightly on MAE
(14.41)
and wins on RMSE
(22.82
vs 24.14).
That is the expected ordering: MAE rewards a conditional median, RMSE a conditional mean.

**Caveat (also on line 1 of `ablation.csv`):** CAVEAT: on synthetic data this comparison overstates the value of queue-state features. The generator produces waits BY SIMULATING A QUEUE, so congestion at arrival is close to a sufficient statistic for the wait (measured congestion-to-wait correlation r = 0.75). A real ED has measurement error, unmodelled resource constraints and clinician discretion that weaken this link. Treat the direction of the effect as supported and the magnitude as an upper bound.

## 4. Prolonged-wait classifier

| feature_set | threshold | threshold_source | pr_auc | recall | precision | fnr | fpr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0.1799 | val | 0.7343 | 0.7727 | 0.6488 | 0.2273 | 0.2648 |
| base+queue | 0.3642 | val | 0.9509 | 0.8807 | 0.8647 | 0.1193 | 0.0872 |

Threshold rule: `highest threshold with recall >= 0.80 (achieved 0.800)`, applied on the **val** block
only. Base generalises from 0.80 recall on val down to
0.773 on test;
base+queue holds at 0.881.

## 6. Mondrian conformal intervals (90% nominal)

| triage_level | n | nominal_coverage | empirical_coverage | coverage_gap | mean_width_minutes | median_width_minutes |
| --- | --- | --- | --- | --- | --- | --- |
| overall | 7284 | 0.9 | 0.8814 | -0.0186 | 52.84 | 51.32 |
| 1 | 159 | 0.9 | 0.8616 | -0.0384 | 13.49 | 13.32 |
| 2 | 899 | 0.9 | 0.901 | 0.001 | 17.54 | 17.92 |
| 3 | 2981 | 0.9 | 0.8839 | -0.0161 | 57.16 | 52.55 |
| 4 | 2361 | 0.9 | 0.8789 | -0.0211 | 60.45 | 58.4 |
| 5 | 884 | 0.9 | 0.8631 | -0.0369 | 60.88 | 59.39 |

Calibrated on the `calib` block
(n = 4719). Per-level half-widths:
ESI 1 = 9.0 min, ESI 2 = 10.1 min, ESI 3 = 35.7 min, ESI 4 = 36.3 min, ESI 5 = 35.9 min.

**Coverage runs 1.9 pp below nominal, and the cause is
instructive rather than a bug.** Split conformal assumes calibration and test residuals
are exchangeable. The blocks are consecutive in time and the synthetic year carries a
winter demand ramp, so the test block is busier than the calibration block and the
intervals come out slightly narrow. The baseline confirms it independently: its MAE
degrades 24.9 →
31.9 →
38.7 min across val → calib →
test. The gap is inside the configured tolerance of
0.04.

## 7. Seen-only bias

| triage_level | n_seen | n_abandoned | abandoned_share | mean_wait_seen_only | mean_wait_with_abandon_lower_bound | bias_minutes | bias_percent |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1004 | 0 | 0 | 5.39 | 5.39 | 0 | 0 |
| 2 | 5962 | 0 | 0 | 9.45 | 9.45 | 0 | 0 |
| 3 | 19599 | 438 | 0.0219 | 41.71 | 44.17 | 2.46 | 5.9 |
| 4 | 15799 | 755 | 0.0456 | 45.03 | 48.81 | 3.78 | 8.39 |
| 5 | 5961 | 466 | 0.0725 | 47.57 | 52 | 4.43 | 9.31 |

Overall: mean wait **38.78 min** among seen patients
only, against **41.79 min** once the
1659 abandoners enter at their time-to-abandon.

**Direction: downward.** A model trained only on seen patients under-predicts, because
the excluded patients are the ones who waited longest. **Size: at least
3.01 min (7.8%)** —
a lower bound, not an estimate, because time-to-abandon is itself only a lower bound on
how long those patients would have waited. The bias grows with decreasing acuity:
ESI 1 +0.0%, ESI 2 +0.0%, ESI 3 +5.9%, ESI 4 +8.4%, ESI 5 +9.3%.

## 8. Regression-cohort exclusions

| block | reason | rows removed |
| --- | --- | --- |
| train | left without being seen | 1006 |
| val | left without being seen | 147 |
| calib | left without being seen | 195 |
| test | departure censored by the data window | 9 |
| test | left without being seen | 311 |
| test | still waiting at the data cutoff | 16 |

The split keeps all four kinds of row; each model stage selects its own cohort. Phase 2's
survival model needs exactly the rows the regressor drops.

## Artifacts written

- `artifacts/tables/`: `ablation.csv`, `classifier_summary.csv`, `conformal_coverage.csv`, `seen_only_bias.csv`
- `artifacts/figures/`: `ablation_error.png`, `conformal_coverage.png`, `seen_only_bias.png`, `prediction_quality.png`
- `artifacts/models/`: 6 regressors/classifiers, the baseline, the conformal calibrator
- `artifacts/metrics/wait_models.json`

## Deviations from CLAUDE.md

1. **`pyarrow` added to the stack** (approved in the Phase 0 review). Intermediate
   tables are Parquet; `profile_*.csv` and the result tables stay CSV for readability.
2. **The `split` stage no longer pre-filters to patients with an observed wait.** All
   validated rows go into the blocks and each model stage selects its own cohort. This
   was required by requirement 8 (exclusions must be counted where they happen) and is
   needed by the Phase 2 survival model.
3. **`ablation` is registered as an alias of `train_wait`**, so `--stages ablation`
   reproduces the table standalone. A full pipeline run therefore repeats that work
   (~6 s). Flagged rather than hidden; a caching split is possible if wanted.

## Known limitations

- The ablation overstates the value of queue features on synthetic data; the generator
  produces waits by simulating a queue (congestion-to-wait r = 0.75). Direction
  supported, magnitude an upper bound.
- Conformal coverage is below nominal for the non-stationarity reason above. On real
  data with drift the same effect would appear, and more strongly.
- The classifier's recall target is met on `val` by construction and need not hold on
  `test`; for `base` it does not (0.80 → 0.773).
- Models are fitted on fully-observed visits only, so they inherit the downward bias
  quantified in section 7. Phase 2's survival module treats the censoring properly.
