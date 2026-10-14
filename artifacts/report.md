# ED Intelligence - results report

**Decision support only.** Every figure below is an estimate for a human to weigh. Nothing in this system triages a patient or makes an autonomous clinical decision.

Every number here is read from a file a pipeline stage wrote; nothing is re-estimated in this report and nothing is entered by hand. Where a sentence compares two of those numbers -- a difference, an interval width, a percentage change -- it is computed from them and from nothing else.

- Data source: **real**
- Stages run: load, validate, split, features, train_wait, ablation, explain, fairness, forecast, survival, competing_risks, calibrate_sim, scenarios, sensitivity, staffing_search, report
- Total pipeline time: **529.3 s**

## Research gaps addressed

| Gap | Problem | Where it is addressed |
| --- | --- | --- |
| **G1** | Forecasting, wait prediction and simulation are built in isolation. | The arrival forecast drives the simulation, which drives the staffing search. |
| **G2** | Subgroup fairness is rarely evaluated. | Subgroup MAE, FNR and FPR with bootstrap intervals, and an explicit finding about high-acuity patients. |
| **G3** | Queue-state features are assumed to help but rarely isolated. | A controlled `base` vs `base+queue` ablation, with its caveat attached. |

## Data and validation

- **57,500 visits** loaded, **57,179** valid, **321** dropped.
- **55,132** have an observed wait and can train the wait-time model.
- **2,046** left without being seen and **1** were still waiting when the extract closed. These are different things and are never pooled: one is an event, the other is censoring.
- Blocks: **train** 34,269 (59.9%), **val** 8,507 (14.9%), **calib** 5,783 (10.1%), **test** 8,620 (15.1%), cut forward in time and non-overlapping.

### Observed waits by acuity

| triage_level | n_arrived | n_seen | n_lwbs | mean | median | p90 | share_over_30min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1139 | 1139 | 0 | 4.5 | 3 | 12 | 0.0035 |
| 2 | 7304 | 7304 | 0 | 5.1 | 3 | 13 | 0.0107 |
| 3 | 23201 | 22817 | 383 | 39.1 | 14 | 113 | 0.3924 |
| 4 | 18467 | 17438 | 1029 | 40.5 | 29 | 104.3 | 0.4909 |
| 5 | 7068 | 6434 | 634 | 45 | 40 | 100 | 0.5398 |
#### How far this dataset sits from the one the models were tuned on

| name | statistic | difference | threshold | flagged |
| --- | --- | --- | --- | --- |
| arrival-hour shape | total variation distance | 0.0003 | 0.1 | False |
| acuity mix | total variation distance | 0.0001 | 0.05 | False |
| median wait | relative change | 0 | 0.25 | False |
| LWBS rate | absolute change | 0 | 0.02 | False |

This dataset looks like the synthetic data the models were tuned on, across all 4 comparisons. That is reassuring about the shape of the data and says nothing about whether the models are any good on it -- they have still never been validated on real data.

## Wait-time prediction

Best model: **base+queue features, raw target** - test MAE **11.52 min**, RMSE 19.74 min.

Baseline (rolling mean of the last 20 observed waits at the same acuity): MAE **34.23 min**, RMSE 48.90 min.

### Ablation: do queue-state features help? (research gap G3)

| feature_set | target | prediction_kind | mae_minutes | rmse_minutes |
| --- | --- | --- | --- | --- |
| base | raw | raw | 33.93 | 53.07 |
| base | log1p | log1p_median | 33.84 | 53.47 |
| base | log1p | log1p_smearing | 53.58 | 93.8 |
| base+queue | raw | raw | 11.52 | 19.74 |
| base+queue | log1p | log1p_median | 11.71 | 19.82 |
| base+queue | log1p | log1p_smearing | 15.44 | 24.86 |

> CAVEAT: on synthetic data this comparison overstates the value of queue-state features. The generator produces waits BY SIMULATING A QUEUE, so congestion at arrival is close to a sufficient statistic for the wait (measured congestion-to-wait correlation r = 0.75). A real ED has measurement error, unmodelled resource constraints and clinician discretion that weaken this link. Treat the direction of the effect as supported and the magnitude as an upper bound.

### Prolonged-wait classifier

| feature_set | threshold | threshold_source | pr_auc | recall | precision | fnr |
| --- | --- | --- | --- | --- | --- | --- |
| base | 0.4471 | val | 0.7109 | 0.721 | 0.7074 | 0.279 |
| base+queue | 0.6982 | val | 0.9823 | 0.8755 | 0.9577 | 0.1245 |

The decision threshold is chosen on the validation block and frozen; it never sees the test block.

### Conformal prediction intervals

**Rolling recalibration is the result to read.** Empirical coverage **0.8875** against a nominal 0.90 (gap -0.0125), at a mean interval width of 46.5 minutes. Each patient's interval is calibrated from the residuals of same-acuity patients **already seen when that patient arrived**, so it tracks drift instead of assuming it away.

Static Mondrian calibration is kept as the comparison: coverage **0.8490** (gap -0.0510) at 39.0 minutes. It calibrates once on a fixed block and assumes the test residuals are exchangeable with it; under temporal drift they are not, and coverage falls. The rolling intervals are wider, which is the honest price of actually covering: a narrow interval that misses one time in 7 is worse than a wider one that does what it claims.

| scope | method | n | nominal_coverage | empirical_coverage | coverage_gap | mean_width_minutes | median_width_minutes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | static | 8198 | 0.9 | 0.849 | -0.051 | 39.04 | 43.8 |
| overall | rolling | 8198 | 0.9 | 0.8875 | -0.0125 | 46.49 | 50.26 |
| ESI 1 | static | 188 | 0.9 | 0.8404 | -0.0596 | 10.65 | 11.35 |
| ESI 1 | rolling | 188 | 0.9 | 0.8777 | -0.0223 | 11.84 | 12.34 |
| ESI 2 | static | 1171 | 0.9 | 0.8523 | -0.0477 | 10.5 | 11.11 |
| ESI 2 | rolling | 1171 | 0.9 | 0.8898 | -0.0102 | 13.24 | 13.64 |
| ESI 3 | static | 3345 | 0.9 | 0.8508 | -0.0492 | 44.96 | 55.96 |
| ESI 3 | rolling | 3345 | 0.9 | 0.8876 | -0.0124 | 53.23 | 55.47 |
| ESI 4 | static | 2555 | 0.9 | 0.8329 | -0.0671 | 41.78 | 50.42 |
| ESI 4 | rolling | 2555 | 0.9 | 0.8834 | -0.0166 | 52.45 | 56.25 |
| ESI 5 | static | 939 | 0.9 | 0.8839 | -0.0161 | 51.81 | 61.61 |
| ESI 5 | rolling | 939 | 0.9 | 0.8978 | -0.0022 | 54.66 | 62.74 |

#### Static coverage per acuity level

| triage_level | n | nominal_coverage | empirical_coverage | coverage_gap | mean_width_minutes | median_width_minutes |
| --- | --- | --- | --- | --- | --- | --- |
| overall | 8198 | 0.9 | 0.849 | -0.051 | 39.04 | 43.8 |
| 1 | 188 | 0.9 | 0.8404 | -0.0596 | 10.65 | 11.35 |
| 2 | 1171 | 0.9 | 0.8523 | -0.0477 | 10.5 | 11.11 |
| 3 | 3345 | 0.9 | 0.8508 | -0.0492 | 44.96 | 55.96 |
| 4 | 2555 | 0.9 | 0.8329 | -0.0671 | 41.78 | 50.42 |
| 5 | 939 | 0.9 | 0.8839 | -0.0161 | 51.81 | 61.61 |

Calibrating within each acuity level does not make the static guarantee hold per level: the worst per-level gap is **-0.0671** at ESI 4, where empirical coverage is 0.8329 against a nominal 0.90.

#### Per-acuity decision thresholds

| scope | rule | threshold | n | n_positive | recall | recall_ci_low | recall_ci_high | fnr | fnr_ci_low | fnr_ci_high | precision | threshold_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | pooled | 0.6982 | 8198 | 3695 | 0.8755 | 0.864 | 0.8862 | 0.1245 | 0.1143 | 0.1361 | 0.9577 | pooled |
| overall | per_triage |  | 8198 | 3695 | 0.8747 | 0.864 | 0.8856 | 0.1253 | 0.1151 | 0.1377 | 0.9573 | mixed |
| ESI 1 | pooled | 0.6982 | 188 | 1 | 0 | 0 | 0 | 1 | 1 | 1 |  | too few positives; using the pooled threshold |
| ESI 1 | per_triage | 0.6982 | 188 | 1 | 0 | 0 | 0 | 1 | 1 | 1 |  | too few positives; using the pooled threshold |
| ESI 2 | pooled | 0.6982 | 1171 | 29 | 0 | 0 | 0 | 1 | 1 | 1 |  | too few positives; using the pooled threshold |
| ESI 2 | per_triage | 0.6982 | 1171 | 29 | 0 | 0 | 0 | 1 | 1 | 1 |  | too few positives; using the pooled threshold |
| ESI 3 | pooled | 0.6982 | 3345 | 1620 | 0.8932 | 0.879 | 0.9077 | 0.1068 | 0.0929 | 0.1226 | 0.9513 | fitted on the validation block |
| ESI 3 | per_triage | 0.7143 | 3345 | 1620 | 0.8846 | 0.8681 | 0.8987 | 0.1154 | 0.1022 | 0.1325 | 0.9528 | fitted on the validation block |
| ESI 4 | pooled | 0.6982 | 2555 | 1465 | 0.8655 | 0.8468 | 0.8818 | 0.1345 | 0.117 | 0.1531 | 0.965 | fitted on the validation block |
| ESI 4 | per_triage | 0.6677 | 2555 | 1465 | 0.8765 | 0.8593 | 0.893 | 0.1235 | 0.1068 | 0.1427 | 0.9625 | fitted on the validation block |
| ESI 5 | pooled | 0.6982 | 939 | 580 | 0.8966 | 0.8699 | 0.9202 | 0.1034 | 0.0791 | 0.1283 | 0.9576 | fitted on the validation block |
| ESI 5 | per_triage | 0.7175 | 939 | 580 | 0.8879 | 0.8628 | 0.9124 | 0.1121 | 0.0865 | 0.1378 | 0.9572 | fitted on the validation block |

A single pooled threshold optimises recall across all patients, and prolonged waits are overwhelmingly an ESI 3-5 phenomenon, so the pooled rule can abandon the few high-acuity patients who do face a long wait. Thresholds are therefore chosen per acuity on the validation block, never on test.

Own thresholds were fitted for ESI 3, ESI 4, ESI 5. Pooled recall **0.8755** against per-acuity **0.8747** on the test block, a change of -0.0008 against a 95% interval 0.0222 wide. **There was no meaningful improvement**: the change is smaller than the interval around either estimate, so the per-acuity rule moves recall within noise.

**ESI 1, ESI 2 could not be fixed.** Each has too few positive cases on the validation block to choose a threshold from, so they keep the pooled one. That is a data limit, not a tuning failure: no cut-off manufactures signal from a handful of patients, and a threshold fitted to that many cases would be noise dressed as a decision. Read `n_positive` before reading any false-negative rate in the table above.

#### Can a prolonged wait be predicted for a high-acuity patient at all?

Four approaches on the same split, the same queue features and the same bootstrap procedure, scored once on the test block:

| approach | scope | n | n_positive | val_positives | threshold | threshold_source | recall | recall_ci_low | recall_ci_high | fnr | fnr_ci_low | fnr_ci_high | precision | reportable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | ESI 1 | 188 | 1 | 0 | 0.4471 | too few positives; using the pooled threshold | 0 | 0 | 0 | 1 | 1 | 1 |  | False |
| pooled | ESI 1-2 pooled | 1359 | 30 | 10 |  | mixed | 0 | 0 | 0 | 1 | 1 | 1 |  | True |
| pooled | ESI 2 | 1171 | 29 | 10 | 0.4471 | too few positives; using the pooled threshold | 0 | 0 | 0 | 1 | 1 | 1 |  | True |
| pooled | overall | 8198 | 3695 |  |  | mixed | 0.7291 | 0.7158 | 0.7418 | 0.2709 | 0.2575 | 0.2842 | 0.7125 | True |
| acuity_weighted | ESI 1 | 188 | 1 | 0 | 0.4293 | too few positives; using the pooled threshold | 0 | 0 | 0 | 1 | 1 | 1 |  | False |
| acuity_weighted | ESI 1-2 pooled | 1359 | 30 | 10 |  | mixed | 0 | 0 | 0 | 1 | 1 | 1 |  | True |
| acuity_weighted | ESI 2 | 1171 | 29 | 10 | 0.4293 | too few positives; using the pooled threshold | 0 | 0 | 0 | 1 | 1 | 1 |  | True |
| acuity_weighted | overall | 8198 | 3695 |  |  | mixed | 0.7172 | 0.7043 | 0.7319 | 0.2828 | 0.2668 | 0.2972 | 0.7106 | True |
| acuity_target | ESI 1 | 188 | 76 | 43 | 0.2917 | fitted on the validation block | 0.75 | 0.6363 | 0.8442 | 0.25 | 0.1448 | 0.3626 | 0.4385 | True |
| acuity_target | ESI 1-2 pooled | 1359 | 189 | 126 |  | mixed | 0.8413 | 0.7873 | 0.894 | 0.1587 | 0.1081 | 0.2086 | 0.153 | True |
| acuity_target | ESI 2 | 1171 | 113 | 83 | 0.1 | fitted on the validation block | 0.9027 | 0.8468 | 0.9485 | 0.0973 | 0.0438 | 0.1532 | 0.1122 | True |
| acuity_target | overall | 8198 | 2753 |  |  | mixed | 0.7396 | 0.7229 | 0.756 | 0.2604 | 0.2443 | 0.2773 | 0.451 | True |

- **`high_acuity_only` was not fitted.** NOT FITTED: 40 positive case(s) in train and 10 in val, against a minimum of 20. A model fitted on that many positives cannot be evaluated, and a threshold chosen from them would be noise.
- **ESI 1: a defensible result exists** under `acuity_target`. The best of them, `acuity_target`, reaches recall 0.750 over 76 positive case(s), with a false-negative rate of 0.250 (95% CI 0.145 to 0.363).
- **ESI 2: a defensible result exists** under `pooled`, `acuity_weighted`, `acuity_target`. The best of them, `acuity_target`, reaches recall 0.903 over 113 positive case(s), with a false-negative rate of 0.097 (95% CI 0.044 to 0.153).

The acuity-specific target used here is ESI 1: 5 min, ESI 2: 15 min, ESI 3: 30 min, ESI 4: 60 min, ESI 5: 120 min. The per-acuity target waits are ASSUMED ILLUSTRATIVE VALUES set in features.acuity_target_wait_minutes, chosen to be plausible and editable. They are NOT clinical standards and carry no endorsement. Changing them changes which patients count as positive and therefore every number in this table.

![ablation_error.png](figures/ablation_error.png)

## Subgroup performance (G2)

Overall on the test block (n = 8,198): FNR **0.1245**, FPR 0.0318, recall 0.8755, MAE 11.52 min.

| attribute | group | n | n_positive | fnr | fnr_gap | fnr_gap_ci_low | fnr_gap_ci_high | fnr_underpowered | mae_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sex | female | 4238 | 1896 | 0.1371 | 0.0126 | -0.0077 | 0.033 | False | 0.1 |
| sex | male | 3960 | 1799 | 0.1112 | -0.0133 | -0.0294 | 0.0053 | False | -0.107 |
| arrival_mode | ambulance | 1416 | 406 | 0.1429 | 0.0184 | -0.0181 | 0.0577 | False | -2.427 |
| arrival_mode | other | 505 | 226 | 0.1283 | 0.0038 | -0.0355 | 0.0488 | False | 0.2569 |
| arrival_mode | walk_in | 6277 | 3063 | 0.1218 | -0.0027 | -0.0191 | 0.0119 | False | 0.5269 |
| triage_level | 1 | 188 | 1 | 1 | 0.8755 | 0.8657 | 0.885 | True | -7.922 |
| triage_level | 2 | 1171 | 29 | 1 | 0.8755 | 0.8639 | 0.8853 | False | -7.974 |
| triage_level | 3 | 3345 | 1620 | 0.1068 | -0.0177 | -0.0351 | -0.0019 | False | 1.771 |
| triage_level | 4 | 2555 | 1465 | 0.1345 | 0.01 | -0.0095 | 0.03 | False | 1.503 |
| triage_level | 5 | 939 | 580 | 0.1034 | -0.0211 | -0.047 | 0.003 | False | 1.132 |
| age_band | 0-17 | 1472 | 653 | 0.1256 | 0.0011 | -0.0266 | 0.032 | False | -0.1054 |
| age_band | 18-39 | 1934 | 877 | 0.1243 | -0.0002 | -0.0237 | 0.0262 | False | -0.3157 |
| age_band | 40-64 | 2717 | 1226 | 0.1297 | 0.0052 | -0.0194 | 0.0239 | False | -0.1871 |
| age_band | 65-79 | 1414 | 650 | 0.1246 | 0.0001 | -0.025 | 0.0252 | False | 0.7237 |
| age_band | 80+ | 479 | 206 | 0.1019 | -0.0226 | -0.0609 | 0.0194 | False | -0.1418 |

**Finding: the classifier never warns a high-acuity patient who will wait a long time.** triage_level = 2 has an FNR of **1.0000** over 29 positive cases, against an overall 0.1245. The `min_recall` objective optimises recall pooled across all patients, and high-acuity patients are a small fraction of the positives, so a single global threshold buys its recall from the low-acuity majority. Clinically this is the wrong way round. Detecting and reporting it is what this section is for.

**Mitigation was tried, and the artifacts show what it bought.** Per-acuity thresholds were fitted on the validation block for ESI 3, ESI 4, ESI 5 - the levels with enough positive cases - and scored against the pooled rule in the per-acuity threshold table above. Pooled recall **0.8755** against per-acuity **0.8747** on the test block, a change of -0.0008 against a 95% interval 0.0222 wide. **There was no meaningful improvement**: the change is smaller than the interval around either estimate, so the per-acuity rule moves recall within noise. They could not be applied to ESI 1 (n_positive = 1) and ESI 2 (n_positive = 29) at all: too few positive cases to choose a cut-off from, which is exactly where this finding lives. So the failure reported here is not waiting on a mitigation nobody wrote. It needs an approach that works with very few high-acuity positives - an acuity-aware objective, or more data at those levels.

1 subgroup(s) have too few positive cases to judge their false-negative rate and are marked rather than reported as findings. An FNR is estimated from positives alone, so a large group can still carry almost no information about it.

![fairness_gaps.png](figures/fairness_gaps.png)

## Arrival forecasting

Backtested over 288 hours in 4 rolling-origin folds. Against the seasonal-naive baseline the model is **20.93% better on MAE overall**, but that reverses at the busiest hours (>= 11.0 arrivals/hour, 60 of 288 hours): peak MAE **4.364** against the baseline's 4.100, **6.43% worse**. The two are reported together because the peak is when a staffing decision bites, and the overall figure alone would hide it.

| segment | model | mae | rmse | mape | smape | wape |
| --- | --- | --- | --- | --- | --- | --- |
| overall | lightgbm | 2.356 | 3.06 | 45.73 | 43.21 | 34.06 |
| overall | seasonal_naive | 2.979 | 3.827 | 60.91 | 51.2 | 43.07 |
| peak_hours | lightgbm | 4.364 | 4.994 | 33.12 | 41.67 | 34.31 |
| peak_hours | seasonal_naive | 4.1 | 5.083 | 30.8 | 38.3 | 32.24 |
| off_peak_hours | lightgbm | 1.827 | 2.295 | 49.11 | 43.62 | 33.9 |
| off_peak_hours | seasonal_naive | 2.684 | 3.421 | 68.98 | 54.6 | 49.8 |

MAPE is reported for comparability with the literature but is undefined at zero-count hours; sMAPE and WAPE are defined everywhere and are the ones to trust.

### Conformal intervals by hour-of-day band

| hour_band | n | nominal_coverage | empirical_coverage | coverage_gap | mean_width_arrivals |
| --- | --- | --- | --- | --- | --- |
| overall | 288 | 0.9 | 0.8924 | -0.0076 | 8.918 |
| afternoon | 72 | 0.9 | 0.9028 | 0.0028 | 11.1 |
| evening | 72 | 0.9 | 0.9306 | 0.0306 | 9.879 |
| morning | 72 | 0.9 | 0.8056 | -0.0944 | 8.839 |
| night | 72 | 0.9 | 0.9306 | 0.0306 | 5.859 |

![forecast_backtest.png](figures/forecast_backtest.png)

#### A longer backtest, because four folds could not settle the peak question

The default backtest above uses 4 folds. This study re-runs the comparison over **26 weekly origins** (1872 scored hours), splits them in time, tunes every constant on the 13 validation origins and scores the 13 test origins once.

| block | variant | scope | n_hours | mae | wape | bias | seasonal_naive_mae | mae_vs_naive | mae_vs_naive_ci_low | mae_vs_naive_ci_high | beats_naive | difference_is_significant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation | peak_corrected | overall | 936 | 2.11 | 34.57 | -0.3899 | 2.871 | -0.7605 | -0.8983 | -0.6216 | True | True |
| validation | same_hour_lags | overall | 936 | 2.118 | 34.71 | -0.205 | 2.871 | -0.7523 | -0.8992 | -0.6063 | True | True |
| validation | blend | overall | 936 | 2.121 | 34.75 | -0.0964 | 2.871 | -0.7498 | -0.8881 | -0.6155 | True | True |
| validation | poisson | overall | 936 | 2.158 | 35.36 | -0.1099 | 2.871 | -0.7126 | -0.8623 | -0.5728 | True | True |
| validation | tweedie | overall | 936 | 2.342 | 38.37 | -0.4343 | 2.871 | -0.5288 | -0.6823 | -0.3691 | True | True |
| validation | blend | peak | 248 | 2.788 | 25.79 | -2.479 | 3.835 | -1.047 | -1.343 | -0.751 | True | True |
| validation | poisson | peak | 248 | 2.823 | 26.12 | -2.437 | 3.835 | -1.011 | -1.328 | -0.668 | True | True |
| validation | same_hour_lags | peak | 248 | 2.973 | 27.5 | -2.566 | 3.835 | -0.8618 | -1.165 | -0.5419 | True | True |
| validation | peak_corrected | peak | 248 | 3.11 | 28.76 | -3.022 | 3.835 | -0.7252 | -1.044 | -0.4135 | True | True |
| validation | tweedie | peak | 248 | 3.289 | 30.42 | -2.788 | 3.835 | -0.5458 | -0.8903 | -0.1911 | True | True |
| test | blend | overall | 936 | 2.278 | 35.43 | -0.3663 | 2.864 | -0.5862 | -0.7236 | -0.4509 | True | True |
| test | same_hour_lags | overall | 936 | 2.285 | 35.53 | -0.456 | 2.864 | -0.5792 | -0.7315 | -0.4232 | True | True |
| test | peak_corrected | overall | 936 | 2.301 | 35.78 | -0.7025 | 2.864 | -0.5637 | -0.7172 | -0.4061 | True | True |
| test | poisson | overall | 936 | 2.347 | 36.5 | -0.4353 | 2.864 | -0.5172 | -0.6752 | -0.3636 | True | True |
| test | tweedie | overall | 936 | 2.434 | 37.85 | -0.7315 | 2.864 | -0.43 | -0.5903 | -0.2718 | True | True |
| test | blend | peak | 263 | 3.243 | 28.68 | -3.008 | 3.669 | -0.4266 | -0.7286 | -0.1434 | True | True |
| test | poisson | peak | 263 | 3.376 | 29.86 | -3.104 | 3.669 | -0.2932 | -0.6202 | 0.0162 | True | False |
| test | same_hour_lags | peak | 263 | 3.397 | 30.05 | -3.1 | 3.669 | -0.2726 | -0.612 | 0.0647 | True | False |
| test | tweedie | peak | 263 | 3.638 | 32.18 | -3.291 | 3.669 | -0.0316 | -0.3519 | 0.307 | True | False |
| test | peak_corrected | peak | 263 | 3.657 | 32.35 | -3.606 | 3.669 | -0.0118 | -0.3182 | 0.2917 | True | False |

Selected on the **validation** origins at peak hours: **`blend`** (MAE 2.7880 against seasonal-naive 3.8347).

**On the test origins the selected forecaster beats seasonal-naive at peak hours**: MAE 3.2426 against 3.6692, a difference of -0.4266 with a 95% interval of -0.7286 to -0.1434, which excludes zero. The peak-hour weakness visible in the four-fold backtest does not survive the larger sample.

**Rolling conformal intervals, by hour-of-day band.** Each interval is sized from the residuals of forecasts issued at *earlier* origins in the same band, so it uses only what had already been observed when the forecast was made.

| variant | block | scope | n | nominal_coverage | empirical_coverage | coverage_gap | within_tolerance | tolerance | mean_width |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blend | test | overall | 936 | 0.9 | 0.891 | -0.009 | True | 0.04 | 8.838 |
| blend | test | afternoon | 234 | 0.9 | 0.8932 | -0.0068 | True | 0.04 | 11.39 |
| blend | test | evening | 234 | 0.9 | 0.9017 | 0.0017 | True | 0.04 | 9.487 |
| blend | test | morning | 234 | 0.9 | 0.8803 | -0.0197 | True | 0.04 | 9.146 |
| blend | test | night | 234 | 0.9 | 0.8889 | -0.0111 | True | 0.04 | 5.328 |
| poisson | test | overall | 936 | 0.9 | 0.8835 | -0.0165 | True | 0.04 | 8.915 |
| poisson | test | afternoon | 234 | 0.9 | 0.8803 | -0.0197 | True | 0.04 | 11.56 |
| poisson | test | evening | 234 | 0.9 | 0.8932 | -0.0068 | True | 0.04 | 9.496 |
| poisson | test | morning | 234 | 0.9 | 0.8761 | -0.0239 | True | 0.04 | 9.182 |
| poisson | test | night | 234 | 0.9 | 0.8846 | -0.0154 | True | 0.04 | 5.423 |

Every band on the test origins is within 0.04 of nominal 0.90, which is the target.

This study does not change the shipped forecaster. artifacts/ reproduces a documented forecast MAE of 2.356 from the default four-fold backtest, and adopting a variant here would move it. Adoption is a separate, explicit change: see tests/test_headline_regression.py.

## Competing-risks survival analysis

Events: **55,132 seen**, **2,046 abandoned**, **1 censored** at the data cutoff.

Abandonment is a *competing event*, not censoring. Treating it as censoring assumes the patient stayed at risk of being seen when in fact they had left, which overstates the probability of being seen.

### How much the usual shortcut costs

| triage_level | horizon_minutes | km_prob_seen | aalen_johansen_prob_seen | km_overstatement | km_overstatement_percent |
| --- | --- | --- | --- | --- | --- |
| 1 | 60 | 1 | 1 | 0 | 0 |
| 2 | 60 | 0.9996 | 0.9997 | -0.0001 | -0.01 |
| 3 | 60 | 0.7243 | 0.7217 | 0.0026 | 0.36 |
| 4 | 60 | 0.7057 | 0.6985 | 0.0072 | 1.03 |
| 5 | 60 | 0.5347 | 0.5287 | 0.006 | 1.13 |
| 1 | 120 | 1 | 1 | 0 | 0 |
| 2 | 120 | 1 | 1 | 0 | 0 |
| 3 | 120 | 0.9051 | 0.8995 | 0.0056 | 0.62 |
| 4 | 120 | 0.9051 | 0.8798 | 0.0253 | 2.88 |
| 5 | 120 | 0.9053 | 0.8554 | 0.0499 | 5.83 |

Kaplan-Meier overstates the probability of being seen by up to **0.0499 (5.83%)**, at ESI 5 and 120 minutes. The bias is zero where nobody abandons and grows with both decreasing acuity and elapsed time.

### Cause-specific Cox models

| cause | covariate | hazard_ratio | hr_lower_95 | hr_upper_95 | significant_at_5pct |
| --- | --- | --- | --- | --- | --- |
| seen | triage_level | 0.7284 | 0.7212 | 0.7357 | True |
| seen | age | 1 | 0.9998 | 1 | False |
| seen | is_ambulance | 1.06 | 1.037 | 1.084 | True |
| seen | n_waiting | 0.8384 | 0.8367 | 0.84 | True |
| seen | hour_sin | 1.268 | 1.251 | 1.286 | True |
| seen | hour_cos | 1.013 | 1.002 | 1.025 | True |
| abandoned | triage_level | 1.164 | 1.135 | 1.194 | True |
| abandoned | age | 0.9999 | 0.9989 | 1.001 | False |
| abandoned | is_ambulance | 0.9356 | 0.8784 | 0.9965 | True |
| abandoned | n_waiting | 0.9985 | 0.9952 | 1.002 | False |
| abandoned | hour_sin | 0.9945 | 0.9586 | 1.032 | False |
| abandoned | hour_cos | 0.9938 | 0.9613 | 1.027 | False |

Fitting both causes is the point: a covariate can raise one hazard while lowering the other, and a single model would show only one of them.

#### Does the proportional-hazards assumption hold?

| cause | covariate | test | test_statistic | p_value | violates_proportional_hazards | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| seen | age | Schoenfeld residuals (rank time-transform) | 1.459 | 0.2271 | False | no evidence against proportional hazards |
| seen | hour_cos | Schoenfeld residuals (rank time-transform) | 602.3 | 5.228e-133 | True | effect changes over the wait; the hazard ratio is a time-average only |
| seen | hour_sin | Schoenfeld residuals (rank time-transform) | 719 | 2.231e-158 | True | effect changes over the wait; the hazard ratio is a time-average only |
| seen | is_ambulance | Schoenfeld residuals (rank time-transform) | 3.536 | 0.06004 | False | no evidence against proportional hazards |
| seen | n_waiting | Schoenfeld residuals (rank time-transform) | 1366 | 6.3e-299 | True | effect changes over the wait; the hazard ratio is a time-average only |
| seen | triage_level | Schoenfeld residuals (rank time-transform) | 12.39 | 0.0004307 | True | effect changes over the wait; the hazard ratio is a time-average only |
| abandoned | age | Schoenfeld residuals (rank time-transform) | 0.0004 | 0.9846 | False | no evidence against proportional hazards |
| abandoned | hour_cos | Schoenfeld residuals (rank time-transform) | 0.0072 | 0.9322 | False | no evidence against proportional hazards |
| abandoned | hour_sin | Schoenfeld residuals (rank time-transform) | 0.0566 | 0.8119 | False | no evidence against proportional hazards |
| abandoned | is_ambulance | Schoenfeld residuals (rank time-transform) | 0.6747 | 0.4114 | False | no evidence against proportional hazards |
| abandoned | n_waiting | Schoenfeld residuals (rank time-transform) | 0.1924 | 0.6609 | False | no evidence against proportional hazards |
| abandoned | triage_level | Schoenfeld residuals (rank time-transform) | 4.6 | 0.03197 | True | effect changes over the wait; the hazard ratio is a time-average only |

**5 of 12 covariate tests reject proportional hazards** at p < 0.05: `hour_cos` (seen, p = 5.23e-133), `hour_sin` (seen, p = 2.23e-158), `n_waiting` (seen, p = 6.3e-299), `triage_level` (seen, p = 0.000431), `triage_level` (abandoned, p = 0.032).

A Cox hazard ratio is one number for the whole of the waiting period. That is only a complete description when the covariate's effect is constant over time, which is what the proportional-hazards assumption asserts. Where the test below rejects it, the reported hazard ratio is still a valid TIME-AVERAGED summary -- weighted towards the times when most patients are still waiting -- but it is not the effect at any particular minute, and it should not be read as one. A covariate whose effect strengthens or fades as the wait goes on will show a ratio somewhere between its early and late values.

## Simulation and staffing (G1)

**Validation PASSED.** The simulation is checked against the observed department before any scenario is allowed to run; scenarios refuse to execute on a model that failed, unless explicitly overridden.

| metric | key | observed | simulated | absolute_error | relative_error | tolerance | absolute_floor | within_tolerance | passed_on | used_for_fitting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean wait | mean_wait_minutes | 35.01 | 32.54 | -2.476 | 0.0707 | 0.2 | 3 | True | relative | True |
| p90 wait | p90_wait_minutes | 100 | 89.21 | -10.79 | 0.1079 | 0.25 | 3 | True | relative | False |
| mean LOS | mean_los_minutes | 160.6 | 152 | -8.598 | 0.0535 | 0.2 | 3 | True | relative | False |
| abandonment rate | abandonment_rate | 0.0358 | 0.0298 | -0.006 | 0.1668 | 0.5 | 0.01 | True | relative | True |
| mean wait ESI 1 | mean_wait_esi1 | 4.547 | 5.242 | 0.6954 | 0.1529 | 0.2 | 3 | True | relative | False |
| mean wait ESI 2 | mean_wait_esi2 | 5.141 | 5.652 | 0.5117 | 0.0995 | 0.2 | 3 | True | relative | False |
| mean wait ESI 3 | mean_wait_esi3 | 39.08 | 34.91 | -4.172 | 0.1068 | 0.2 | 3 | True | relative | False |
| mean wait ESI 4 | mean_wait_esi4 | 40.5 | 38.1 | -2.398 | 0.0592 | 0.2 | 3 | True | relative | False |
| mean wait ESI 5 | mean_wait_esi5 | 45.01 | 44.16 | -0.8493 | 0.0189 | 0.2 | 3 | True | relative | False |
| abandonment rate ESI 1 | abandonment_rate_esi1 | 0 | 0 | 0 |  | 0.5 | 0.01 | True | absolute floor | False |
| abandonment rate ESI 2 | abandonment_rate_esi2 | 0 | 0 | 0 |  | 0.5 | 0.01 | True | absolute floor | False |
| abandonment rate ESI 3 | abandonment_rate_esi3 | 0.0165 | 0.0184 | 0.0019 | 0.1159 | 0.5 | 0.01 | True | relative | False |
| abandonment rate ESI 4 | abandonment_rate_esi4 | 0.0557 | 0.0384 | -0.0174 | 0.3116 | 0.5 | 0.01 | True | relative | False |
| abandonment rate ESI 5 | abandonment_rate_esi5 | 0.0897 | 0.0802 | -0.0095 | 0.1058 | 0.5 | 0.01 | True | relative | False |

**2 metrics** (mean wait, abandonment rate) are used to fit the unobservable treatment/boarding split and therefore excluded from the gate; every other row is out-of-sample validation.

### Scenarios

| scenario | n_replications | mean_wait_minutes | mean_wait_minutes_ci_low | mean_wait_minutes_ci_high | p90_wait_esi3 | abandonment_rate |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 30 | 33.86 | 32.42 | 35.31 | 100.9 | 0.0342 |
| plus_one_doctor_peak | 30 | 20.76 | 19.62 | 21.91 | 72.84 | 0.0166 |
| plus_one_nurse_peak | 30 | 26.85 | 25.5 | 28.2 | 84.09 | 0.0216 |
| fast_track_esi45 | 30 | 8.118 | 7.537 | 8.7 | 27.88 | 0.0012 |
| surge_10pct | 30 | 47.73 | 46.3 | 49.16 | 127.4 | 0.0664 |
| surge_20pct | 30 | 60.75 | 59.02 | 62.48 | 146.7 | 0.1062 |

| scenario | difference_vs_baseline | ci_low | ci_high | method | significant |
| --- | --- | --- | --- | --- | --- |
| plus_one_doctor_peak | -13.1 | -13.98 | -12.22 | paired (common random numbers) | True |
| plus_one_nurse_peak | -7.015 | -7.452 | -6.579 | paired (common random numbers) | True |
| fast_track_esi45 | -25.74 | -26.84 | -24.64 | paired (common random numbers) | True |
| surge_10pct | 13.87 | 11.84 | 15.9 | unpaired (scenario changes demand) | True |
| surge_20pct | 26.89 | 24.64 | 29.14 | unpaired (scenario changes demand) | True |

Staffing scenarios are compared with **common random numbers**: replication *i* of each faces the same arrivals and service draws, so the shared noise cancels in the difference. A demand scenario necessarily draws a different stream and is reported as unpaired -- visible in the `method` column, and in the width of its interval.

### Sensitivity: does the ranking survive being wrong about the inputs?

| scenario | rank_at_point_estimate | best_rank | worst_rank | rank_is_stable | max_rank_shift |
| --- | --- | --- | --- | --- | --- |
| fast_track_esi45 | 1 | 1 | 1 | True | 0 |
| plus_one_doctor_peak | 2 | 2 | 2 | True | 0 |
| plus_one_nurse_peak | 3 | 3 | 3 | True | 0 |
| baseline | 4 | 4 | 4 | True | 0 |
| surge_10pct | 5 | 5 | 5 | True | 0 |
| surge_20pct | 6 | 6 | 6 | True | 0 |

**`fast_track_esi45` is rank 1 across all 19 perturbed worlds, spanning acuity_mix, aging_floor, aging_rate, arrival_rate, clinician_time_fraction, patience, point_estimate, service_time.**

#### How wrong could this be?

The scenarios were re-run in **19 perturbed worlds** across 8 families of input: `acuity_mix`, `aging_floor`, `aging_rate`, `arrival_rate`, `clinician_time_fraction`, `patience`, `point_estimate`, `service_time`. Each world moves exactly one input, so an effect belongs to that input alone.

**The scenario ranking is stable.** No scenario changed position on mean wait in any of those worlds.

One world in the grid is a **noise control**: it sets the priority-aging floor to the value already configured, so it changes nothing about the department and differs only in its random stream. Whatever it moves is Monte Carlo noise at this replication count: abandonment_rate 0.0048, mean_wait_minutes 3.546, p90_wait_esi3 4.796. Any effect below that is not a finding, and the table marks it.

Largest effects, excluding the noise control:

- **walkout rate (LWBS)**: `clinician_share_x1.2` (+0.0797), `service_x1.2` (+0.0746), `arrivals_x1.2` (+0.0681)
- **mean wait (min)**: `clinician_share_x1.2` (+27.6), `service_x1.2` (+25.9), `arrivals_x1.2` (+24.9)
- **p90 wait, ESI 3 (min)**: `service_x0.8` (-74.2), `clinician_share_x0.8` (-71.3), `arrivals_x0.8` (-64.9)

![sensitivity_tornado.png](figures/sensitivity_tornado.png)

The aging floor is a clinical safety constraint, not a tuning knob, so the grid only contains values that keep ESI 1-2 protected. 2.5 (the configured value) lets an aged low-acuity patient overtake a fresher ESI 3 or 4 but never a freshly arrived ESI 1 or 2; 3.0 stops ESI 3 ageing at all. 2.0 is NOT tested: it lets aged patients overtake ESI 2, whose mean wait then triples (9.4 -> 31 min). Measuring an unsafe policy to three decimal places does not make it available.

### Staffing search on forecast demand (research gap G1)

Target: p90 wait for ESI 3 <= 90 min and mean abandonment <= 4.0%, judged on the upper 95% confidence bound.

**Cheapest feasible plan: `day=3, evening=3, night_early=2, night_late=2` at cost 49,280** per 168-hour week, achieving p90 ESI 3 wait 77.2 min and abandonment 0.0183. 72 of 144 plans were feasible.

Feasibility is decided before cost, and judged on the upper confidence bound: a plan must be *shown* to meet the target, not merely estimated to.

## Robustness across datasets

The whole pipeline was re-run on **10 regenerated dataset(s)**, each produced by the original generator with nothing changed but its random seed, and each writing to its own run folder so none of them can touch the baseline artifacts. Every figure below is read from those runs.

Seed 42 is the reference row: at 365 days and scale 9.7 it *is* `data/raw/ed_v2.csv`, so it must reproduce the headline numbers, and the study stops if it does not.

| seed | n_rows | wait_mae_model | wait_mae_baseline | conformal_static_overall | conformal_rolling_overall | forecast_mae | forecast_mae_seasonal_naive | sim_validation_passed | sim_worst_relative_error | cheapest_feasible_plan | cheapest_feasible_cost | n_feasible_plans |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 56671 | 9.988 | 27.15 | 0.9121 | 0.9067 | 2.253 | 2.608 | True | 0.3716 | day=3, evening=3, night_early=1, night_late=2 | 4.389e+04 | 80 |
| 2 | 56778 | 10.92 | 32.97 | 0.8541 | 0.8854 | 2.315 | 2.969 | True | 0.3759 | day=3, evening=3, night_early=2, night_late=2 | 4.928e+04 | 75 |
| 3 | 57115 | 11.51 | 34.79 | 0.8657 | 0.8955 | 2.537 | 3.08 | True | 0.2772 | day=3, evening=3, night_early=2, night_late=1 | 4.851e+04 | 73 |
| 4 | 57129 | 10.69 | 32.8 | 0.8625 | 0.885 | 2.486 | 2.816 | True | 0.2105 | day=3, evening=4, night_early=1, night_late=1 | 4.928e+04 | 71 |
| 5 | 58180 | 12.05 | 35.85 | 0.879 | 0.8962 | 2.88 | 3.233 | True | 0.2189 | day=3, evening=4, night_early=2, night_late=1 | 5.467e+04 | 61 |
| 6 | 57258 | 10.95 | 34.28 | 0.8788 | 0.8965 | 2.397 | 3.01 | True | 0.3158 | day=3, evening=3, night_early=1, night_late=3 | 4.466e+04 | 83 |
| 7 | 57037 | 11.02 | 31.6 | 0.8951 | 0.8968 | 2.657 | 3.021 | True | 0.279 | day=3, evening=3, night_early=1, night_late=2 | 4.389e+04 | 80 |
| 8 | 57562 | 10.14 | 30.33 | 0.9082 | 0.9 | 2.292 | 2.941 | False | 0.7193 | refused (simulation failed validation) |  |  |
| 9 | 57094 | 9.92 | 31.4 | 0.8801 | 0.8946 | 2.588 | 3.035 | True | 0.3351 | day=3, evening=4, night_early=1, night_late=3 | 5.082e+04 | 64 |
| 10 | 57148 | 9.229 | 28.56 | 0.9426 | 0.9137 | 2.329 | 2.667 | False | 0.327 | refused (simulation failed validation) |  |  |
| 42 | 57500 | 11.52 | 34.23 | 0.849 | 0.8875 | 2.356 | 2.979 | True | 0.3116 | day=3, evening=3, night_early=2, night_late=2 | 4.928e+04 | 72 |

**The simulation validation gate passed on 8 of 10.** On the rest it refused to issue a staffing plan, which is what it is for: a plan from a model that cannot reproduce its own dataset looks exactly as confident as a correct one.
- **Seed 8:** ESI 3 abandonment against a 1.6% base rate -- a 1.2 percentage-point miss, which is 72% in relative terms and clinically trivial. A **rare-event tolerance artefact**: the absolute floor exists for exactly this case and is set just low enough to let it through.
- **Seed 10:** ESI 3 mean wait **8 minutes low** against an observed 34.4 minutes. A **genuine fit miss**, not a small-sample effect, and the gate is right to refuse it.

No tolerance was widened to make any of these pass. Loosening a gate until it stops refusing is how a staffing recommendation from an unvalidated model reaches a reader.

**The staffing method transfers; the specific roster does not.** Across the 8 seed(s) that produced a plan the cheapest cost ranged **43,890 to 54,670** over **7 distinct roster(s)**, with the day shift at 3 in every one. The cheapest plan this report quotes is therefore the answer for **this dataset**, not a recommendation for a department.

**What held across the seeds.**

- The wait model beat its queue-position baseline on **10 of 10** seeds.
- The arrivals forecast beat seasonal-naive on **10 of 10** seeds.
- Rolling conformal coverage was **3.1x more stable** than a static split (sd 0.0086 against 0.0270). The defensible claim is not that rolling is always closer to nominal but that it lands in the same place whatever dataset arrives.

**Spread across the seeds.** Mean, standard deviation and range for every quantity the study measured.

| metric | n_seeds | mean | std | min | max | spread |
| --- | --- | --- | --- | --- | --- | --- |
| wait_mae_model | 10 | 10.64 | 0.8329 | 9.229 | 12.05 | 2.818 |
| wait_mae_baseline | 10 | 31.97 | 2.75 | 27.15 | 35.85 | 8.698 |
| ablation_gap_minutes | 10 | 20.58 | 2.775 | 15.65 | 24.29 | 8.641 |
| classifier_pr_auc | 10 | 0.9732 | 0.0054 | 0.9632 | 0.9779 | 0.0147 |
| classifier_recall | 10 | 0.849 | 0.0178 | 0.8112 | 0.8757 | 0.0645 |
| conformal_static_overall | 10 | 0.8878 | 0.027 | 0.8541 | 0.9426 | 0.0885 |
| conformal_rolling_overall | 10 | 0.897 | 0.0086 | 0.885 | 0.9137 | 0.0287 |
| conformal_static_esi1 | 10 | 0.9006 | 0.0328 | 0.8403 | 0.9503 | 0.11 |
| conformal_rolling_esi1 | 10 | 0.8931 | 0.0273 | 0.8547 | 0.9503 | 0.0956 |
| conformal_static_esi2 | 10 | 0.8916 | 0.0243 | 0.8509 | 0.9222 | 0.0713 |
| conformal_rolling_esi2 | 10 | 0.8996 | 0.0077 | 0.8854 | 0.9094 | 0.024 |
| conformal_static_esi3 | 10 | 0.8879 | 0.032 | 0.8539 | 0.9556 | 0.1017 |
| conformal_rolling_esi3 | 10 | 0.8962 | 0.0102 | 0.881 | 0.9172 | 0.0362 |
| conformal_static_esi4 | 10 | 0.8873 | 0.0296 | 0.8462 | 0.9444 | 0.0982 |
| conformal_rolling_esi4 | 10 | 0.8972 | 0.009 | 0.8887 | 0.9148 | 0.0261 |
| conformal_static_esi5 | 10 | 0.8824 | 0.0333 | 0.8225 | 0.9236 | 0.1011 |
| conformal_rolling_esi5 | 10 | 0.8972 | 0.0127 | 0.8742 | 0.915 | 0.0408 |
| forecast_mae | 10 | 2.474 | 0.1971 | 2.253 | 2.88 | 0.6269 |
| forecast_mae_seasonal_naive | 10 | 2.938 | 0.1906 | 2.608 | 3.233 | 0.625 |
| forecast_mae_peak | 10 | 3.84 | 0.7934 | 2.989 | 5.688 | 2.699 |
| forecast_mae_peak_seasonal_naive | 10 | 4.171 | 0.5529 | 3.266 | 5.311 | 2.045 |
| sim_worst_relative_error | 10 | 0.343 | 0.1437 | 0.2105 | 0.7193 | 0.5088 |
| cheapest_feasible_cost | 8 | 4.812e+04 | 3795 | 4.389e+04 | 5.467e+04 | 1.078e+04 |
| n_feasible_plans | 8 | 73.38 | 7.837 | 61 | 83 | 22 |
| n_rows | 10 | 5.72e+04 | 422.4 | 5.667e+04 | 5.818e+04 | 1509 |
| wall_seconds | 10 | 433.5 | 126.7 | 206.4 | 628.3 | 421.9 |

All of these datasets come from one generator. Agreement across its seeds shows the pipeline is not over-fitted to a single draw of it; it recovers the generator, and is not evidence about any real emergency department.

## Limitations

- **Synthetic data.** Every number in this report comes from data generated by a queue
  whose structure the models are then asked to recover. That validates the methods; it
  is not evidence about any real emergency department.
- **Single site, single configuration.** No external validation, no cross-site
  comparison, no temporal validation beyond the held-out test block.
- **Censoring bias.** The wait-time models train only on patients who were seen.
  Patients who left without being seen waited longest, so the models under-predict; the
  survival section quantifies a lower bound on that bias.
- **The ablation overstates queue features.** The generator creates waits by simulating
  a queue, so congestion at arrival is close to a sufficient statistic. The direction is
  supported; the magnitude is an upper bound.
- **The staffing recommendation is partly circular.** The roster that generated the
  data is also the cheapest plan the search finds feasible. The simulation is calibrated
  to reproduce a department running that roster, so it is close to guaranteed that the
  roster looks adequate and that anything leaner does not. The finding "the current
  roster is the minimum viable staffing" is therefore reinforced by how the data was
  made, and would need a department whose roster changed during the observation period
  to be tested properly.
- **Priority-aging assumptions.** The queue applies a specific aging rule with a safety
  floor. Both the generator and the simulation use it, so the simulation reproduces the
  generator partly by construction. A real department's prioritisation differs and the
  aging floor in particular is a policy choice, not a measurement.
- **The treatment/boarding split is not identifiable** from arrival, contact
  and departure timestamps alone. It is fitted to 2 aggregate metrics
  (mean wait, abandonment rate), which are therefore excluded from the
  validation gate; a site with real treatment-end timestamps should use them.
- **The abandonment gate is loose.** The abandonment rows are checked
  against a relative tolerance of 0.50,
  where the wait and length-of-stay rows use 0.20, 0.25.
  Abandonment is a small count and a tight relative band would fail on
  noise alone, so a simulated rate wrong by 50% still passes.
  The largest relative abandonment error in the table is **0.3116**
  (abandonment rate ESI 4), and it passes.
  Read those rows as a sanity check, not as evidence that the simulation
  reproduces who leaves.
- **Unmodelled resources.** Nurses, beds, imaging and laboratory turnaround are not
  modelled. Doctors are the only hard constraint.
- **Fairness subgroups are confounded with acuity.** Arrival-mode and age-band gaps
  largely track acuity composition; no within-acuity adjusted analysis is computed.
- **Conformal coverage assumes exchangeability**, which time-ordered blocks only
  approximate. Empirical coverage is reported rather than assumed.
- **The simulation validation gate passed on 8 of 10 regenerated datasets**; it
  refused on seed 8 (rare-event tolerance) and seed 10 (ESI 3 mean-wait miss).
- **The cheapest roster varies across datasets** (43,890-54,670 over 7 distinct
  rosters on the 8 seeds that produced one); the method transfers, the specific
  roster does not.
- **Decision support only.** Nothing here triages a patient, allocates care, or makes an
  autonomous clinical decision.

## Future work

- **Validation on real data.** The single most valuable next step: run the adapter
  against a real extract and re-measure everything. Every number in this report should
  be treated as provisional until then.
- **An objective that works with very few high-acuity positives.** Per-acuity
  thresholds are already fitted and measured (ESI 3, ESI 4, ESI 5), and
  they changed pooled recall by -0.0008 against a 95% interval
  0.0222 wide.
  ESI 1 (n_positive = 1) and ESI 2 (n_positive = 29) could not be fitted at all.
  So thresholding is not the open problem: an acuity-aware training
  objective, or more positive cases at those levels, is.
- **High-acuity prolonged-wait prediction.** Four approaches were tried and measured: the pooled classifier, acuity sample weights, a separate ESI 1-2 model, and an acuity-specific target.
  ESI 1 became evaluable under `acuity_target`.
  ESI 2 became evaluable under `acuity_target`, `acuity_weighted`, `pooled`.
  So the open problem is no longer whether it can be done at all for ESI 1, ESI 2.
  What remains is choosing the target with clinicians rather than assuming it: the per-acuity targets used here are illustrative, and every rate above moves when they move.
- **Deep forecasters: not currently justified.** The peak-hour weakness that motivated this item does not survive a larger sample. Over 26 rolling origins the selected forecaster (`blend`) beats seasonal-naive at peak hours, 3.243 against 3.669 MAE, with a 95% interval on the difference of -0.729 to -0.143 that excludes zero. A deep architecture remains interesting, but it no longer has a measured deficit to fix.
- **A reinforcement learning staffing policy that works.** One was built and it lost. A
  tabular Q-learning agent over the validated simulator cost **17,832 more** per
  evaluation
  week than the fixed roster on a pre-declared 60-episode budget (95% CI 14,663 to
  21,002), with more ESI 1-2 breaches than acting at random. A larger training budget is
  the obvious suspect and is a hypothesis, not a defence; it was not retrained until it
  won. It is also circular -- the agent learns a simulator calibrated to a generator --
  so a win would be evidence about the method and nothing else.
- **Calibrating the richer resource model.** Nurses, beds, imaging turnaround and
  bed-holding boarding exist as explicit constrained resources behind
  `simulation.resource_extensions`, off by default. They cannot be calibrated here: the
  data has no bed, nurse or imaging column, so those parameters are assumed and the
  validation gate cannot test them. Their ranking is usable; their magnitudes are not.
- **Competing-risks prediction on real data.** Built: a discrete-time model for the
  cumulative incidence of leaving without being seen at 30, 60 and 120 minutes, with
  isotonic recalibration and an Aalen-Johansen comparison. What it recovers is the
  generator's patience distribution, which was chosen rather than observed.
