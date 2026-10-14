# Phase 2 checkpoint — forecasting and survival

## Gates

| Gate | Result |
| --- | --- |
| Full test suite | **242 passed** |
| Pipeline end to end on synthetic | 10 stages (timing in the Phase 4 checkpoint) |
| `tests/test_leakage.py` | 12 passed |
| No random splits in `src/` | 9 patterns tested, 0 found |
| Every tunable from config | asserted by `test_project_invariants.py` |

## What was built

| Requirement | Module | Status |
| --- | --- | --- |
| 1. Hourly counts, seasonal-naive baseline, LightGBM, rolling origin, MAE/RMSE/MAPE (+sMAPE, WAPE) | `src/models/forecast.py` | done |
| 2. Conformal intervals with coverage by hour-of-day band | `src/models/forecast.py` | done |
| 3. `forecast(horizon_hours, as_of)` using only pre-`as_of` data, with a leakage test | `src/models/forecast.py` | done |
| 4. Peak-hour error reported separately | `src/models/forecast.py` | done |
| 5. Competing-risks survival: KM, Aalen-Johansen, cause-specific Cox | `src/models/survival.py` | done |

New stages: `forecast`, `survival`. 42 new tests across `tests/test_forecast.py` (23)
and `tests/test_survival.py` (19).

## 1. Forecast accuracy (288 hours over 4 rolling-origin folds)

| segment | model | mae | rmse | mape | smape | wape | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | lightgbm | 2.22 | 2.933 | 48.62 | 43.87 | 35.31 | 288 |
| overall | seasonal_naive | 2.764 | 3.666 | 62.59 | 52.51 | 43.95 | 288 |
| peak_hours | lightgbm | 3.894 | 4.581 | 30.62 | 38.46 | 31.92 | 60 |
| peak_hours | seasonal_naive | 4.367 | 5.17 | 34.81 | 44.84 | 35.79 | 60 |
| off_peak_hours | lightgbm | 1.78 | 2.312 | 53.49 | 45.3 | 37.6 | 228 |
| off_peak_hours | seasonal_naive | 2.342 | 3.153 | 70.1 | 54.55 | 49.49 | 228 |

**The model beats the seasonal-naive baseline by 19.67% on MAE**
(2.220 against 2.764).

MAPE is reported for comparability with the literature but is undefined at zero-count
hours; 6 scored hours had no arrivals at all. sMAPE
and WAPE are defined everywhere and are the ones to trust: WAPE
35.31% against 43.95%.

### By lead time

| horizon_band | n | model_mae | naive_mae | model_wape | naive_wape |
| --- | --- | --- | --- | --- | --- |
| 1-24h | 96 | 2.057 | 2.312 | 33.63 | 37.82 |
| 25-48h | 96 | 2.135 | 2.646 | 33.83 | 41.91 |
| 49-72h | 96 | 2.468 | 3.333 | 38.34 | 51.78 |

Error grows with lead time, as a direct multi-horizon forecaster should: the model's
information about the recent past goes stale while the calendar signal does not. The
seasonal-naive baseline is horizon-independent by construction — it is always the value
168 hours earlier — so its variation across bands reflects which hours landed in each
band, not forecast difficulty.

### Peak hours (top 20% of demand, >= 10.0 arrivals/hour)

Peak-hour MAE **3.894** against baseline 4.367
over 60 hours. Absolute error is naturally larger at the peak
(3.894) than off-peak (1.780), but
*relative* error is lower (WAPE 31.92% against
37.60%): busy hours are more predictable in
proportional terms, which is the useful direction for staffing.

## 2. Conformal intervals by hour-of-day band

| hour_band | n | nominal_coverage | empirical_coverage | coverage_gap | mean_width_arrivals |
| --- | --- | --- | --- | --- | --- |
| overall | 288 | 0.9 | 0.8993 | -0.0007 | 8.824 |
| afternoon | 72 | 0.9 | 0.8889 | -0.0111 | 10.77 |
| evening | 72 | 0.9 | 0.9028 | 0.0028 | 8.915 |
| morning | 72 | 0.9 | 0.9306 | 0.0306 | 10.85 |
| night | 72 | 0.9 | 0.875 | -0.025 | 4.76 |

Overall coverage **0.8993** against a nominal
0.90 — a gap of -0.0007, which is
essentially exact. This is notably better than the wait-time model's conformal coverage
(which under-covered by ~2pp), because each backtest fold calibrates on its own
immediately-preceding window rather than on one fixed block months earlier, so the
exchangeability assumption holds far better.

Mondrian banding earns its place: night intervals are
**4.76 arrivals**
wide against
**10.85** in the
morning. A single pooled interval would be simultaneously too wide at 3 a.m. and too
narrow at the peak — backwards for staffing.

## 3. Leakage control

`forecast(horizon_hours, as_of)` truncates the series at `as_of` before building a
single feature. Every lag is anchored to `as_of` rather than to the target timestamp,
which is the design decision that makes this honest: the usual shortcut of anchoring
`lag_1` to the target silently assumes you know the hour before each forecast hour,
which is false for every horizon beyond one.

Two tests enforce it:

- `test_forecast_uses_only_data_before_as_of` — a forecast built from the full series
  must be **bit-identical** to one built from a copy truncated at `as_of`.
- `test_corrupting_the_future_cannot_change_a_forecast` — overwriting every future
  observation with 9999 must not move the output.

A third, `test_seasonal_lag_is_dropped_once_it_would_need_the_future`, checks that the
168-hour seasonal lag goes missing rather than being guessed once the horizon exceeds
it.

## 5. Competing-risks survival analysis

Events: 48325 seen, 1659 abandoned,
16 still waiting at the cutoff.
Administrative censoring at 720 minutes.

### Kaplan-Meier time-to-seen (abandonment treated as censoring)

| triage_level | n | n_seen | median_minutes_to_seen | prob_unseen_at_60min | prob_unseen_at_120min |
| --- | --- | --- | --- | --- | --- |
| 1 | 1004 | 1004 | 2 | 0 | 0 |
| 2 | 5963 | 5962 | 6.7 | 0.002 | 0 |
| 3 | 20042 | 19599 | 17.18 | 0.2484 | 0.1152 |
| 4 | 16562 | 15799 | 23.88 | 0.2956 | 0.1226 |
| 5 | 6429 | 5961 | 27.7 | 0.3856 | 0.1233 |

### Aalen-Johansen cumulative incidence (competing risks handled correctly)

| triage_level | cause | n_events | incidence_at_60min | incidence_at_120min | incidence_at_240min |
| --- | --- | --- | --- | --- | --- |
| 1 | seen | 1004 | 1 | 1 | 1 |
| 2 | seen | 5962 | 0.998 | 1 | 1 |
| 3 | seen | 19599 | 0.7516 | 0.8827 | 0.9677 |
| 3 | abandoned | 438 | 0.0005 | 0.0076 | 0.0197 |
| 4 | seen | 15799 | 0.7037 | 0.8686 | 0.9463 |
| 4 | abandoned | 755 | 0.0045 | 0.023 | 0.0439 |
| 5 | seen | 5961 | 0.6135 | 0.8533 | 0.9236 |
| 5 | abandoned | 466 | 0.0094 | 0.0472 | 0.071 |

### How much the usual shortcut costs

| triage_level | horizon_minutes | km_prob_seen | aalen_johansen_prob_seen | km_overstatement | km_overstatement_percent |
| --- | --- | --- | --- | --- | --- |
| 1 | 60 | 1 | 1 | 0 | 0 |
| 2 | 60 | 0.998 | 0.998 | 0 | 0 |
| 3 | 60 | 0.7516 | 0.7516 | 0 | 0 |
| 4 | 60 | 0.7044 | 0.7037 | 0.0007 | 0.1 |
| 5 | 60 | 0.6144 | 0.6135 | 0.0009 | 0.15 |
| 1 | 120 | 1 | 1 | 0 | 0 |
| 2 | 120 | 1 | 1 | 0 | 0 |
| 3 | 120 | 0.8848 | 0.8827 | 0.0021 | 0.24 |
| 4 | 120 | 0.8774 | 0.8686 | 0.0088 | 1.01 |
| 5 | 120 | 0.8767 | 0.8533 | 0.0234 | 2.74 |

**Treating abandonment as censoring overstates the probability of being seen by up to
0.0234 (2.74%), at ESI
5 and 120 minutes.** The bias
is exactly zero for ESI 1–2, who essentially never abandon, and grows with both
decreasing acuity and elapsed time — the signature of a competing risk. The direction is
guaranteed: KM can only ever overstate, because censoring an abandonment pretends the
patient remained at risk of being seen when they had left.

### Cause-specific Cox models

| cause | covariate | hazard_ratio | hr_lower_95 | hr_upper_95 | significant_at_5pct |
| --- | --- | --- | --- | --- | --- |
| seen | triage_level | 0.6583 | 0.6505 | 0.6661 | True |
| seen | age | 1.001 | 1.001 | 1.002 | True |
| seen | is_ambulance | 1.178 | 1.152 | 1.205 | True |
| seen | n_waiting | 0.8593 | 0.8574 | 0.8611 | True |
| seen | hour_sin | 1.511 | 1.49 | 1.532 | True |
| seen | hour_cos | 0.8537 | 0.8421 | 0.8655 | True |
| abandoned | triage_level | 1.118 | 1.088 | 1.149 | True |
| abandoned | age | 0.9986 | 0.9975 | 0.9998 | True |
| abandoned | is_ambulance | 0.9484 | 0.8906 | 1.01 | False |
| abandoned | n_waiting | 0.9978 | 0.9941 | 1.002 | False |
| abandoned | hour_sin | 1.001 | 0.9634 | 1.041 | False |
| abandoned | hour_cos | 1.002 | 0.9651 | 1.041 | False |

**This is why both hazards are fitted.** `triage_level` has a hazard ratio of
0.658
for being seen and
1.118
for abandoning: each step down in acuity cuts the rate of being seen by about a third
*and* raises the rate of giving up. A single model would have shown one of those and
hidden the other.

`n_waiting` behaves instructively too: HR
0.859
for being seen (congestion at arrival materially slows you down) but
0.998
and **not significant** for abandoning. In the generator, abandonment is driven by each
patient's own patience rather than directly by queue length — congestion causes
abandonment only through the wait it produces. The cause-specific split recovers that
structure.

## Deviations from CLAUDE.md

1. **A `survival` stage was added to `STAGE_ORDER`**, which the original 13-stage list
   did not contain. The Phase 2 brief requires `survival.py`, and Phase 4's stage list
   names it, so the registry now holds 14 stages.
2. Carried forward from earlier phases: `pyarrow` (approved); the `split` stage keeps
   all rows; `ablation` aliases `train_wait`.

No new dependency was added. `lifelines` was already approved and is used for
KaplanMeierFitter, AalenJohansenFitter and CoxPHFitter.

## Known limitations

- **Forecast horizon is capped by the seasonal lag.** Beyond 168 hours the seasonal
  feature is unavailable and accuracy would fall sharply. The configured horizon is 72.
- **The forecast stage skips entirely on short extracts.** It needs
  `max(lags, rolling_windows) + 168` hours of history before `train_end`; a 10-day
  pilot dataset gets a logged explanation and `forecast.json` with `ran: false`, not a
  crash. This path is tested.
- **Aalen-Johansen resolves tied event times by random jitter** (a lifelines behaviour).
  Timestamps are second-quantised, so ties are common; the resulting estimates move by
  under 0.002, which is why the bias test asserts a directional inequality beyond that
  noise rather than exact non-negativity.
- **Cox proportional-hazards assumptions are not tested.** With ~48k events the
  proportionality assumption almost certainly fails somewhere; the hazard ratios should
  be read as time-averaged summaries, not as constants.
- **The Cox models use a penalizer of 0.1** for numerical stability, so coefficients
  are slightly shrunk toward zero.
- All of this is synthetic data generated by a queue whose structure the models are then
  asked to recover. The methods are validated; the numbers are not evidence about any
  real department.
