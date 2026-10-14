# External real-data validation

The first test of anything in this repository against data it did not generate. Two
real datasets, both obtained successfully; nothing was blocked.

Date: **2026-10-14**. No git. No modelling code, config, threshold or result changed.
**Every headline number unchanged** — wait MAE 11.52 vs 34.23, conformal 0.8875 vs
0.849, forecast 2.356 vs 2.979, plan 3/3/2/2 at 49,280, 72 of 144; headline
regression 10/10 and number audit 78 claims / 0 mismatches after the work.

All outputs are under `artifacts/external_validation/` and `data/external/`. The one
extra dependency (`pyreadr==0.5.3`) is pinned in `requirements-extra.txt` and is
imported by nothing in the pipeline, the API, the dashboard or the test suite.

---

## Part 1 — UIHC hourly arrivals (real, forecasting only)

**Obtained.** No `Rscript` on this machine, so the dataset came out of the fastTS
**source tarball** from CRAN rather than being exported by R: only
`data/uihc_ed_arrivals.RData` and `DESCRIPTION` were extracted, and nothing in the
tarball was executed. Read with `pyreadr` and written to
`data/external/uihc_ed_arrivals.csv`.

| | |
|---|---|
| Rows | **41,640** hourly |
| Range | 2013-07-01 00:00 to 2018-03-31 23:00 |
| Licence | GPL (>= 3), package by Ryan Andrew Peterson |
| Tarball SHA256 | `bee1648b…45324f8` |
| CSV SHA256 | `ee268ffe…22a939d8` |

**Validation before fitting:** 0 missing hours, 0 duplicates, 0 missing counts, 0
negative counts, all integral, mean 6.628/hour, 918 zero-arrival hours, no
daylight-saving artefacts. Cleaner than the synthetic data, which has deliberate
missingness injected.

### Result, 26 rolling origins, 1,872 scored hours

| | MAE | RMSE | sMAPE | WAPE |
|---|---|---|---|---|
| **This project's forecaster** | **2.025** | **2.611** | **37.7%** | **31.5%** |
| Seasonal-naive (168 h) | 2.751 | 3.602 | 51.4% | 42.9% |
| Mean of same hour, previous 4 weeks | 2.215 | 2.875 | 40.9% | 34.5% |

Peak (top 20% of hours, 10+ arrivals): model **3.041**, naive 3.951, four-week 3.240.
Off peak: model **1.760**, naive 2.438, four-week 1.948. It wins everywhere.

Conformal coverage **0.898** against a 0.90 target, against 0.8875 on synthetic —
the intervals are *better* calibrated on the real series than on the data they were
designed against.

**Nothing was retuned.** Same model class, same lags including 168 h, same
rolling-origin design, same metrics, read from `configs/default.yaml`. A test asserts
the script sets no hyperparameter of its own.

### A reading that did not survive more data

The configured 4-origin run (288 hours) showed the model **tying** the four-week
baseline off peak — 1.8743 against 1.8762. That was reported before the wider run
finished, and it is wrong: over 1,872 hours the model is 9.7% better off peak. A
288-hour artefact. Both runs are kept, in `uihc/` and `uihc_26folds/`.

### A defect the real data found

The project's nine configured holidays are all **2024** dates. On 2013–2018 data
`is_holiday` is constant, and the feature builder says so in a warning it already
had. **The model beat both baselines with that feature accidentally disabled.**
Fixing it means editing `configs/default.yaml`, which this work may not touch, so it
is reported and left.

### What the real data has that the synthetic lacks

840 game-day hours, Christmas, New Year, Thanksgiving and Independence Day
indicators — none of which the generator has any concept of — plus five years of
trend and drift, and nothing imputed.

### Temperature, handled honestly

Same-hour `temp` is **not** a feature: it is not known when a forecast is issued, and
using it would be predicting the past. Lagged by the 72-hour horizon it correlates
**0.314** with arrivals but **−0.029** with the model's residuals — the calendar and
seasonal-lag features have already absorbed what it would say.

---

## Part 2 — MIMIC-IV-ED demo (adapter check only)

**Obtained** by `scripts\fetch_mimic_ed_demo.ps1`: 97,782 bytes, the archive's own
`SHA256SUMS.txt` verified **7 of 7**, ODbL v1.0 with the attribution recorded.

**No model was fitted and no performance is reported.** 222 stays. A test asserts the
script contains no fit, no predict and no scoring function at all.

### The column map, written blind, checked against reality

Every column name in `configs/column_map_mimic_ed.yaml` — written from the published
schema with no data in hand — **exists in the real tables**, and the adapter mapped
all 222 stays through its own `adapt_real_dataset`.

**One mapping was wrong.** `LEFT AGAINST MEDICAL ADVICE` was mapped to `lwbs`. It is
not: leaving against advice means a clinician saw and advised the patient. Mapping it
to `lwbs` would have inflated the abandonment rate with patients who had already been
assessed — and abandonment is one of the quantities the simulation's validation gate
is judged on. Corrected to `other`, with a test pinning it.

Dispositions as applied: admitted 150, discharged 60, transferred 5, other 4
(2 against-advice + 2 OTHER), lwbs 3 (2 left-without-being-seen + 1 eloped).

### Profile

Acuity: ESI 1 → 18, 2 → 97, 3 → 90, 4 → 2, 5 → **0**, not recorded 15. **This is not
a department's acuity mix** — a real one is dominated by ESI 3–5 — because the demo is
a sample of patients, not a window of operation.

Length of stay (real, not a proxy): median **350.5 min**, IQR 253–526, p90 943.

### The proxy wait, measured for the first time

MIMIC-IV-ED has **no first-provider-contact timestamp**. With
`first_vitalsign_charttime` the adapter derived `seen_time` for **206 of 222 stays
(92.8%)** and emitted its own warning. Every figure labelled **proxy wait (not true
door-to-provider)**.

| Proxy | Coverage | Median | p90 |
|---|---|---|---|
| First `vitalsign.charttime` | 206 (92.8%) | **2.0 min** | 190.5 min |
| First `pyxis.charttime` | 176 (79.3%) | **118.0 min** | 306.6 min |

**The two bracket the truth from opposite sides, exactly as
`docs/real_data_readiness.md` predicted from the schema alone.** Among the 167 stays
with both, the dispense is later in **80.2%** of cases by a median of **76 minutes**.
A 2-minute median for first vitals confirms it is recorded at triage — essentially the
arrival time — so using it as door-to-provider would report almost everyone as seen
immediately.

### Date shifting, and what it rules out

MIMIC-IV shifts dates **per patient**. Intervals within a stay survive; two patients
are not on a common clock. The demo makes it visible: **222 stays spanning 32,548
days — 89 years — at 0.01 stays per day.**

| Can be validated on MIMIC-IV-ED | Cannot be, with or without credentials |
|---|---|
| The adapter and the column mapping | **Queue-state features** — counts over concurrent patients, and concurrency does not survive per-patient shifting |
| Per-patient features: acuity, arrival mode, time of day, length of stay | **The simulation's validation gate**, which needs a real window of operation |
| Disposition and acuity distributions | **The staffing search**, scored by that simulation |
| The proxy-wait distribution, labelled | **Door-to-provider wait itself** — the timestamp does not exist |

**Credentialed access removes the sample-size problem and nothing else on that list.**
This matters for the project's central claim: queue-state features are what make the
wait model beat its baseline threefold, and MIMIC-IV-ED cannot test them at all.

---

## Part 3 — Docs and tests

`README.md` and `SUBMISSION.md` both gained an **External validation** section with
the results, the limits, and the two sources credited with their licences. The
section ends by stating that **no real-hospital performance is claimed for anything
but the arrival forecast**: no patient-level model touched real data.

`tests/test_external_validation.py` adds **25 tests**. Besides the arithmetic, they
guard the honesty of the exercise: that the UIHC run reads the project's own config
and sets no hyperparameter itself, that same-hour temperature is never a feature, that
the MIMIC script contains no fit or score at all, that every proxy is named in words,
that the external data is excluded from the package, and that `pyreadr` never reaches
`requirements.txt`.

### A real defect the tests found

`test_the_external_data_is_excluded_from_the_package` failed on first run:
`scripts\make_submission.ps1` did **not** exclude `data\external`, so the zip would
have redistributed the GPL UIHC series and the ODbL MIMIC demo inside this project's
package. Fixed, with the exclusion verified in the staging check like the others.
