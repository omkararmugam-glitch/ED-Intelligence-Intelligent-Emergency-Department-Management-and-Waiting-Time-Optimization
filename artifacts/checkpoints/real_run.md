# Real-data run — `data/raw/ed_v2.csv`

**The dataset is synthetic, calibrated to published ED benchmarks. It is not real
patient data.** 57,500 visits, 2024-01-01 to 2024-12-30, with deliberate defects
injected to exercise validation.

Every number below was read from a file in `artifacts/`. Nothing is recomputed or
entered by hand.

- Pipeline: **15 stages in 174.8 s**, no stage skipped
- Test suite: **326 passed**

## 1. Validation counts

| category | detail | count | share_of_input |
| --- | --- | --- | --- |
| input | rows read from source | 57500 | 1 |
| outcome | rows valid after checks | 57179 | 0.9944 |
| outcome | rows dropped | 321 | 0.005583 |
| outcome | usable for the wait-time model | 55132 | 0.9588 |
| censoring | never seen: left without being seen | 2046 | 0.03558 |
| censoring | never seen: still waiting at the data cutoff | 1 | 1.7e-05 |
| dropped | triage_level missing or outside 1-5 | 206 | 0.003583 |
| dropped | seen_time before arrival_time (negative wait) | 115 | 0.002 |
| dropped | missing patient_id | 0 | 0 |
| dropped | missing or unparseable arrival_time | 0 | 0 |
| dropped | departure_time before arrival_time (negative LOS) | 0 | 0 |
| dropped | seen_time after departure_time | 0 | 0 |
| dropped | duplicate patient_id | 0 | 0 |
| optional column | age: present | 1 | 1.7e-05 |
| optional column | sex: present | 1 | 1.7e-05 |
| optional column | arrival_mode: present | 1 | 1.7e-05 |
| optional column | chief_complaint: present | 1 | 1.7e-05 |

The two injected defects were found at exactly the documented scale and **0.56% of rows
were dropped**, not the whole file:

- **115 rows** with `arrival_time`/`seen_time` swapped, detected as a negative wait.
- **206 rows** with a missing acuity (0.36%).

Everything else was retained. Patients with no `seen_time` are *flagged*, not dropped:
**2,046 left without being seen** and **1** was
still waiting when the window closed. The missing optional values (age, arrival mode,
chief complaint) do not invalidate a row — they are left missing and the models handle
them.

All 10 columns were sourced. Observed waits by acuity:

| triage_level | n_arrived | n_seen | n_lwbs | mean | median | p90 | share_over_30min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1139 | 1139 | 0 | 4.5 | 3 | 12 | 0.0035 |
| 2 | 7304 | 7304 | 0 | 5.1 | 3 | 13 | 0.0107 |
| 3 | 23201 | 22817 | 383 | 39.1 | 14 | 113 | 0.3924 |
| 4 | 18467 | 17438 | 1029 | 40.5 | 29 | 104.3 | 0.4909 |
| 5 | 7068 | 6434 | 634 | 45 | 40 | 100 | 0.5398 |

## 2. Wait-time prediction

Best model: **base+queue / raw / raw** —
test MAE **11.52 min**, RMSE 19.74 min.

Baseline (rolling mean of the last 20 observed waits at the same
acuity): MAE **34.23 min**, RMSE 48.90 min. The model
cuts MAE by **66.3%**.

### Ablation: base vs base+queue (G3)

| feature_set | target | prediction_kind | mae_minutes | rmse_minutes |
| --- | --- | --- | --- | --- |
| base | raw | raw | 33.93 | 53.07 |
| base | log1p | log1p_median | 33.84 | 53.47 |
| base | log1p | log1p_smearing | 53.58 | 93.8 |
| base+queue | raw | raw | 11.52 | 19.74 |
| base+queue | log1p | log1p_median | 11.71 | 19.82 |
| base+queue | log1p | log1p_smearing | 15.44 | 24.86 |

Queue-state features cut test MAE from **33.84**
to **11.52 min**, a
66% reduction.

> CAVEAT: on synthetic data this comparison overstates the value of queue-state features. The generator produces waits BY SIMULATING A QUEUE, so congestion at arrival is close to a sufficient statistic for the wait (measured congestion-to-wait correlation r = 0.75). A real ED has measurement error, unmodelled resource constraints and clinician discretion that weaken this link. Treat the direction of the effect as supported and the magnitude as an upper bound.

### Classifier

| feature_set | threshold | pr_auc | recall | precision | fnr | fpr |
| --- | --- | --- | --- | --- | --- | --- |
| base | 0.4471 | 0.7109 | 0.721 | 0.7074 | 0.279 | 0.2447 |
| base+queue | 0.6982 | 0.9823 | 0.8755 | 0.9577 | 0.1245 | 0.0318 |

Per acuity, from the fairness table (the `n_positive` column is what an FNR is actually
estimated from):

| triage_level | n | n_positive | fnr | recall | precision | mae |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 188 | 1 | 1 | 0 |  | 3.602 |
| 2 | 1171 | 29 | 1 | 0 |  | 3.55 |
| 3 | 3345 | 1620 | 0.1068 | 0.8932 | 0.9513 | 13.29 |
| 4 | 2555 | 1465 | 0.1345 | 0.8655 | 0.965 | 13.03 |
| 5 | 939 | 580 | 0.1034 | 0.8966 | 0.9576 | 12.66 |

### Conformal intervals, overall and per acuity

| triage_level | n | nominal_coverage | empirical_coverage | coverage_gap | mean_width_minutes | median_width_minutes |
| --- | --- | --- | --- | --- | --- | --- |
| overall | 8198 | 0.9 | 0.849 | -0.051 | 39.04 | 43.8 |
| 1 | 188 | 0.9 | 0.8404 | -0.0596 | 10.65 | 11.35 |
| 2 | 1171 | 0.9 | 0.8523 | -0.0477 | 10.5 | 11.11 |
| 3 | 3345 | 0.9 | 0.8508 | -0.0492 | 44.96 | 55.96 |
| 4 | 2555 | 0.9 | 0.8329 | -0.0671 | 41.78 | 50.42 |
| 5 | 939 | 0.9 | 0.8839 | -0.0161 | 51.81 | 61.61 |

**Coverage is 0.8490 against a nominal
0.90 — a gap of -0.0510, which
exceeds the configured tolerance of 0.04.** This is a real finding about this dataset,
not a code fault. Split conformal assumes the calibration and test residuals are
exchangeable, and in this extract they are not: mean wait by block runs **train 35.7,
val 29.1, calib 29.2, test 42.3 minutes**. The test period is 45% busier than the
calibration period, so intervals calibrated on the quiet stretch are too narrow for it.
Every acuity level under-covers by a similar margin, which is the signature of a shift
in the whole distribution rather than a problem with one group.

## 3. Arrival forecasting

Model MAE **2.356** against seasonal-naive
**2.979** arrivals/hour —
**20.9% better** — over 288 hours in
4 rolling-origin folds. WAPE 34.06% against
43.07%.

Peak hours (top 20% of demand, >= 11.0 arrivals/hour): MAE
**4.364** against baseline
4.100.

Interval coverage **0.8924** against nominal
0.90 — close, and much better than the wait-time
intervals because each fold calibrates on its own immediately preceding window rather
than on one fixed block months earlier.

## 4. Competing-risks survival

Events: **55,132 seen**, **2,046
abandoned**, **1 censored** at the cutoff.

How much the usual shortcut costs — treating abandonment as censoring:

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

Cause-specific Cox hazard ratios:

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

## 5. Simulation calibration and validation

The department's roster is a **config input**, as a hospital would supply it:
`staffing_roster` gives 2 doctors 00:00-07:00, 3 from 07:00-23:00, 2 from 23:00-24:00 —
**64 doctor-hours a day**.

**Validation PASSED: all 12 held-out checks within tolerance.** Tolerances were not
changed; five per-acuity abandonment checks were *added* to the gate.

| metric | observed | simulated | relative_error | tolerance | within_tolerance | used_for_fitting |
| --- | --- | --- | --- | --- | --- | --- |
| mean wait | 35.01 | 32.54 | 0.0707 | 0.2 | True | True |
| p90 wait | 100 | 89.21 | 0.1079 | 0.25 | True | False |
| mean LOS | 160.6 | 152 | 0.0535 | 0.2 | True | False |
| abandonment rate | 0.0358 | 0.0298 | 0.1668 | 0.5 | True | True |
| mean wait ESI 1 | 4.547 | 5.242 | 0.1529 | 0.2 | True | False |
| mean wait ESI 2 | 5.141 | 5.652 | 0.0995 | 0.2 | True | False |
| mean wait ESI 3 | 39.08 | 34.91 | 0.1068 | 0.2 | True | False |
| mean wait ESI 4 | 40.5 | 38.1 | 0.0592 | 0.2 | True | False |
| mean wait ESI 5 | 45.01 | 44.16 | 0.0189 | 0.2 | True | False |
| abandonment rate ESI 1 | 0 | 0 |  | 0.5 | True | False |
| abandonment rate ESI 2 | 0 | 0 |  | 0.5 | True | False |
| abandonment rate ESI 3 | 0.0165 | 0.0184 | 0.1159 | 0.5 | True | False |
| abandonment rate ESI 4 | 0.0557 | 0.0384 | 0.3116 | 0.5 | True | False |
| abandonment rate ESI 5 | 0.0897 | 0.0802 | 0.1058 | 0.5 | True | False |

### The identification problem, and what had to change

The data gives arrival, first contact and departure, never when treatment ended, so
`departure - seen` is clinician time plus tests, results and boarding. **Median
post-contact time is 99 minutes against a roster of 64 doctor-hours a day for 157
patients: if all of it were doctor occupancy the department would need 4.95x the
capacity it has.**

The previous calibration subtracted a constant boarding offset from treatment. That
cannot work here — subtracting enough to make ESI 5 feasible drives ESI 1 negative — so
it was replaced by a **multiplicative clinician-time fraction**, fitted at
**0.2072**: a doctor is occupied
for about 21% of a patient's post-contact time. The split preserves
`treatment + boarding = post_contact` exactly, so length of stay stays an honest
held-out metric (simulated 152.0
against observed 160.6 min).

Patience is fitted from the data: the shape per acuity comes from the observed
time-to-abandon, and the scale (fitted at x2.23) is fitted against the observed abandonment rate, which
corrects the censoring in that sample. ESI 1 and 2 record **zero** abandonments in this
extract, so they keep their configured patience; they are seen in
4.5 and
5.1 minutes on average and never
reach the point of leaving.

Two metrics (mean wait, overall abandonment) were consumed by the fit and are excluded
from the gate. The other 12 — p90 wait, LOS, five per-acuity waits, five per-acuity
abandonment rates — are genuine out-of-sample validation.

## 6. Scenarios, sensitivity and staffing

| scenario | mean_wait_minutes | mean_wait_minutes_ci_low | mean_wait_minutes_ci_high | p90_wait_esi3 | abandonment_rate | doctor_utilisation |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 33.86 | 32.42 | 35.31 | 100.9 | 0.0342 | 0.814 |
| plus_one_doctor_peak | 20.76 | 19.62 | 21.91 | 72.84 | 0.0166 | 0.737 |
| plus_one_nurse_peak | 26.85 | 25.5 | 28.2 | 84.09 | 0.0216 | 0.7835 |
| fast_track_esi45 | 8.118 | 7.537 | 8.7 | 27.88 | 0.0012 | 0.6127 |
| surge_10pct | 47.73 | 46.3 | 49.16 | 127.4 | 0.0664 | 0.8677 |
| surge_20pct | 60.75 | 59.02 | 62.48 | 146.7 | 0.1062 | 0.8992 |

| scenario | difference_vs_baseline | ci_low | ci_high | method | significant |
| --- | --- | --- | --- | --- | --- |
| plus_one_doctor_peak | -13.1 | -13.98 | -12.22 | paired (common random numbers) | True |
| plus_one_nurse_peak | -7.015 | -7.452 | -6.579 | paired (common random numbers) | True |
| fast_track_esi45 | -25.74 | -26.84 | -24.64 | paired (common random numbers) | True |
| surge_10pct | 13.87 | 11.84 | 15.9 | unpaired (scenario changes demand) | True |
| surge_20pct | 26.89 | 24.64 | 29.14 | unpaired (scenario changes demand) | True |

Sensitivity (+/-10% and +/-20% on service times and arrival rates):

| scenario | rank_at_point_estimate | best_rank | worst_rank | rank_is_stable | max_rank_shift |
| --- | --- | --- | --- | --- | --- |
| fast_track_esi45 | 1 | 1 | 1 | True | 0 |
| plus_one_doctor_peak | 2 | 2 | 2 | True | 0 |
| plus_one_nurse_peak | 3 | 3 | 3 | True | 0 |
| baseline | 4 | 4 | 4 | True | 0 |
| surge_10pct | 5 | 5 | 5 | True | 0 |
| surge_20pct | 6 | 6 | 6 | True | 0 |

### Staffing search

Target: p90 wait for ESI 3 <= 90 min and mean abandonment <= 4.0%, judged on the upper 95% confidence bound. Load: **forecast (72 hours, repeated 5x to 336h)**.
**72 of 144 plans were feasible.**

**Cheapest feasible plan: `day=3, evening=3, night_early=2, night_late=2` at cost 49,280** per
168-hour week — p90 ESI 3 wait
77.2 min (upper bound
88.8 against a 90-minute target) and abandonment
0.0183.

That plan **is the current roster**. The useful finding is what happens below it — every
cheaper plan fails, and not marginally:

| plan | cost | p90_wait_esi3 | abandonment_rate | doctor_utilisation |
| --- | --- | --- | --- | --- |
| day=2, evening=2, night_early=1, night_late=1 | 3.08e+04 | 196.2 | 0.2357 | 0.9703 |
| day=2, evening=2, night_early=1, night_late=2 | 3.157e+04 | 188.6 | 0.2197 | 0.97 |
| day=2, evening=2, night_early=1, night_late=3 | 3.234e+04 | 180.4 | 0.2001 | 0.9707 |

Cutting any shift to 2 doctors pushes p90 ESI 3 wait past 170 minutes and abandonment
above 20%. **The existing roster is the minimum viable staffing for this demand**, and
it meets the target with little margin.

## 7. How this differs from the earlier 50k synthetic run

| | 50k generated run | ed_v2 real-data run |
| --- | --- | --- |
| Visits | 50,000 | 57,500 |
| Rows dropped | 0 | 321 (0.56%) |
| Doctors | 4 night / 8 day / 5 evening | 2 overnight / 3 day / 3 evening |
| Observed mean wait | 38.8 min | 35.0 min |
| Best wait MAE | 14.19 min | 11.52 min |
| Baseline MAE | 38.71 min | 34.23 min |
| Best target transform | log1p / median | raw / raw |
| Classifier PR-AUC (base+queue) | 0.9509 | 0.9823 |
| Classifier FNR | 0.1193 | 0.1245 |
| Conformal coverage | 0.8814 (gap -0.019) | 0.8490 (gap -0.0510) |
| Forecast vs naive | 19.7% better | 20.9% better |
| Abandonment | 3.32% | 3.58% |
| Sim validation | passed, 8 held-out checks | passed, 12 held-out checks |

What actually changed, and why:

- **The wait model got better in absolute terms** (MAE 11.52 vs 14.19)
  **and far better relative to its baseline** (66%
  vs 63%). ed_v2's waits are more strongly driven by queue state, so the queue features
  carry more.
- **The raw target now wins.** On the generated data the log target tied; here the raw
  target is best and `log1p_smearing` is markedly worse
  (15.44 min),
  because ed_v2's wait distribution has a heavier right tail and the smearing correction
  over-inflates the mean.
- **Conformal coverage got worse** (-0.0510 against -0.019)
  because this dataset has a genuinely busier test period, as shown in section 2.
- **ESI 1 and 2 never abandon here**, where the generated data had them abandoning
  occasionally. They are seen in under 6 minutes.
- **The staffing conclusion inverted.** The generated run found the roster had slack;
  here the roster is the floor and nothing cheaper works.

## 8. Deviations, assumptions and unresolved problems

### Changes made to get this to run

1. **`configs/column_map.yaml` rewritten** as an identity mapping for ed_v2, with
   `sex` mapped F/M -> female/male because `src/features/build.py` derives `is_female`
   from the literal string and the fairness report groups on the same values.
2. **Adapter now coerces numeric columns when no `value_map` is given**, so 1, 1.0, "1"
   and "1.0" are all accepted. Without this the float acuity column would not have
   mapped.
3. **Adapter warns when a `value_map` matches under 5% of rows**, and **fails when a
   schema column is unsourced while the raw file holds a column of that exact name**.
   Both are silent failure modes that look like data problems much later.
4. **`staffing_roster` added to the config** as the roster input, replacing the implicit
   3-shift structure. `StaffingPlan` reads it and still honours the old keys.
5. **Calibration rewritten** from a constant boarding offset to a multiplicative
   clinician-time fraction, and from alternating to **nested** fitting. Alternating
   could not converge: whichever parameter was fitted last broke the other, leaving
   either mean wait or abandonment badly off.
6. **Staffing search now extends the forecast to the simulation horizon.** It was
   evaluating plans over 72 forecast hours minus a 24-hour warm-up — 48 scored hours
   from an empty department, which never reaches steady state. p90 ESI 3 came out at
   61 minutes against 103 for the same roster over 336 hours, and **all 144 plans
   passed**. This was a genuine bug that made the search meaningless.
7. **Search grid extended below the current roster.** Its floor was the roster itself,
   so "cheapest feasible" could only ever return the status quo.

### Assumptions

- **The clinician-time fraction is fitted, not measured.** 21% of post-contact time is
  the value that reproduces observed mean wait under this roster. A site with real
  treatment-end timestamps should use them instead of fitting it.
- **Tiling the forecast** to fill the simulation horizon assumes the 72-hour forecast
  window is representative of the weeks the roster will run.
- **Patience scale** is fitted to the overall abandonment rate, which assumes the
  censoring mechanism is the same across acuity levels.

### Unresolved

- **Conformal coverage is outside tolerance** (-0.0510 against
  0.04) because of real temporal drift. Not fixed here: the honest options are rolling
  re-calibration on a recent window, or weighting the calibration set toward recent
  rows. Both change the method, so I have not applied them unasked.
- **The staffing plan is sized for the forecast window**, not the year, and the forecast
  period is slightly quieter than the annual mean (6.17 vs 6.53 arrivals/hour).
- **The high-acuity classifier failure persists.** ESI 1-2 have very few prolonged-wait
  cases here, so the per-acuity FNR for those groups rests on few positives; see the
  `n_positive` column in section 2.
- **`C:\dev\ed-project` is a stale copy** of this project (no `scripts/`, still pins
  `shap`, 295 tests). All work in this run was done in the OneDrive tree. The two should
  be reconciled before anyone runs the copy.


---

# Addendum — rolling conformal, per-acuity thresholds, scenarios

Added after the original real-data run. Every number is read from a file in `artifacts/`.

## Rolling conformal calibration

The original run reported conformal coverage of
0.8490 against a nominal 0.90 — a gap of
-0.0510, outside the configured 0.04 tolerance — and
left it unresolved because the fix changes the method. It is now implemented as a second
variant, with the static result kept as the comparison baseline.

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

Coverage moves from **0.8490** to
**0.8875**, from outside the tolerance to
inside it, and every acuity level improves. Mean interval width rises from
39.0 to
46.5 minutes, which is what honest coverage
costs on a drifting series.

The mechanism: for each patient, calibrate on the absolute residuals of same-acuity
patients whose `seen_time` precedes that patient's `arrival_time`, within a 28-day window —
widening, then pooling across acuity, then falling back to the static half-width when a
window is too thin. 8,062 of the 8,198 test rows used the primary window.

The admissibility rule is `seen_time < arrival_time`, not block membership: a patient
arriving in December may use the residual of a patient seen in November even though both
are in the test block. A deployed system keeps learning from outcomes as they land, and
withholding them would measure something no deployment would do. Restricting the pool to
earlier blocks was tried first and barely helped (0.8556).

## Per-acuity decision thresholds

| scope | rule | threshold | n_positive | recall | fnr | fnr_ci_low | fnr_ci_high | precision | threshold_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | pooled | 0.6982 | 3695 | 0.8755 | 0.1245 | 0.1143 | 0.1361 | 0.9577 | pooled |
| overall | per_triage |  | 3695 | 0.8747 | 0.1253 | 0.1151 | 0.1377 | 0.9573 | mixed |
| ESI 1 | pooled | 0.6982 | 1 | 0 | 1 | 1 | 1 |  | too few positives; using the pooled threshold |
| ESI 1 | per_triage | 0.6982 | 1 | 0 | 1 | 1 | 1 |  | too few positives; using the pooled threshold |
| ESI 2 | pooled | 0.6982 | 29 | 0 | 1 | 1 | 1 |  | too few positives; using the pooled threshold |
| ESI 2 | per_triage | 0.6982 | 29 | 0 | 1 | 1 | 1 |  | too few positives; using the pooled threshold |
| ESI 3 | pooled | 0.6982 | 1620 | 0.8932 | 0.1068 | 0.0929 | 0.1226 | 0.9513 | fitted on the validation block |
| ESI 3 | per_triage | 0.7143 | 1620 | 0.8846 | 0.1154 | 0.1022 | 0.1325 | 0.9528 | fitted on the validation block |
| ESI 4 | pooled | 0.6982 | 1465 | 0.8655 | 0.1345 | 0.117 | 0.1531 | 0.965 | fitted on the validation block |
| ESI 4 | per_triage | 0.6677 | 1465 | 0.8765 | 0.1235 | 0.1068 | 0.1427 | 0.9625 | fitted on the validation block |
| ESI 5 | pooled | 0.6982 | 580 | 0.8966 | 0.1034 | 0.0791 | 0.1283 | 0.9576 | fitted on the validation block |
| ESI 5 | per_triage | 0.7175 | 580 | 0.8879 | 0.1121 | 0.0865 | 0.1378 | 0.9572 | fitted on the validation block |

**ESI 1 and 2 still cannot be fixed.** ESI 1 has 1 prolonged-wait case on the test block
and 0 on the validation block used for selection; ESI 2 has 29 and 10. Both fall below the
configured minimum of 25 positives and keep the pooled threshold, which the
`threshold_source` column records.

The two fail for different reasons, and only one is hopeless:

- **ESI 1 has no signal.** Its single positive case scores 0.0130, *below* the 95th
  percentile of its own negative cases (0.0309). No cut-off separates it from noise.
- **ESI 2 has signal.** A threshold near 0.05 would catch 22 of its 29 cases — but at
  precision 0.250, three false alarms per true catch, and it cannot be chosen honestly
  from 10 validation positives.

Where thresholds were fitted (ESI 3-5) the effect is within noise: ESI 4 improves, ESI 3
and 5 worsen slightly, every confidence interval overlaps, and overall FNR is essentially
unchanged. Per-acuity thresholds are the right mechanism; they are not where this model's
problem lies.

## Scenarios

| scenario | n_replications | mean_wait_minutes | mean_wait_minutes_ci_low | mean_wait_minutes_ci_high | p90_wait_esi3 | abandonment_rate | doctor_utilisation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 30 | 33.86 | 32.42 | 35.31 | 100.9 | 0.0342 | 0.814 |
| plus_one_doctor_peak | 30 | 20.76 | 19.62 | 21.91 | 72.84 | 0.0166 | 0.737 |
| plus_one_nurse_peak | 30 | 26.85 | 25.5 | 28.2 | 84.09 | 0.0216 | 0.7835 |
| fast_track_esi45 | 30 | 8.118 | 7.537 | 8.7 | 27.88 | 0.0012 | 0.6127 |
| surge_10pct | 30 | 47.73 | 46.3 | 49.16 | 127.4 | 0.0664 | 0.8677 |
| surge_20pct | 30 | 60.75 | 59.02 | 62.48 | 146.7 | 0.1062 | 0.8992 |

Paired differences against baseline:

| scenario | difference_vs_baseline | ci_low | ci_high | method | significant |
| --- | --- | --- | --- | --- | --- |
| plus_one_doctor_peak | -13.1 | -13.98 | -12.22 | paired (common random numbers) | True |
| plus_one_nurse_peak | -7.015 | -7.452 | -6.579 | paired (common random numbers) | True |
| fast_track_esi45 | -25.74 | -26.84 | -24.64 | paired (common random numbers) | True |
| surge_10pct | 13.87 | 11.84 | 15.9 | unpaired (scenario changes demand) | True |
| surge_20pct | 26.89 | 24.64 | 29.14 | unpaired (scenario changes demand) | True |

Staffing scenarios use common random numbers, so replication *i* of each faces the same
arrivals and service draws and the shared noise cancels in the difference. Demand
scenarios necessarily draw a different stream and are reported unpaired — visible in both
the `method` column and the width of the interval.

## Additional limitation: the staffing recommendation is circular

The roster that generated `ed_v2.csv` is also the cheapest plan the search finds feasible.
The simulation is calibrated to reproduce a department running that roster, so it is close
to guaranteed that the roster looks adequate and that anything leaner does not. The finding
"the current roster is the minimum viable staffing" is reinforced by how the data was made,
and would need a department whose roster changed during the observation period to be
tested properly.

## Serving

The API and dashboard are documented in `artifacts/checkpoints/serving.md`, including a
500-error bug that the live run found and the test suite had missed.
