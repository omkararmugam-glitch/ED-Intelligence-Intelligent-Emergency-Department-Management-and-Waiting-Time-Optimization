# UIHC hourly arrivals: this project's forecaster on real data

The first test of anything in this repository against data it did not generate.

**Dataset.** `uihc_ed_arrivals` from the R package **fastTS** (GPL >= 3): 41,640
hourly arrival counts from the University of Iowa Hospitals and Clinics emergency
department, 2013-07-01 to 2018-03-31. Provenance, licence and checksums in
`data/external/README.md`. Produced by `scripts\external_validation_uihc.py`.

**What this can show.** Whether the forecaster, **unchanged**, beats the baselines it
is supposed to beat on a real hourly series. That is a claim about the method.

**What it cannot show.** Anything about this project's wait model, its simulation or
its staffing search: those need per-patient records with timestamps, and this is a
count series. It is also one department over one period, and says nothing about any
other hospital or any other year.

Nothing was retuned. Same model class, same lag set including the 168-hour seasonal
lag, same rolling-origin design, same metrics, same conformal calibration, all read
from `configs/default.yaml`.

---

## 1. The data, checked before anything was fitted

| Check | Result |
|---|---|
| Rows | **41,640** |
| Range | 2013-07-01 00:00 to 2018-03-31 23:00 |
| Expected hours in that range | 41,640 |
| **Missing hours** | **0** |
| **Duplicate hours** | **0** |
| Missing counts | 0 |
| Negative counts | 0 |
| All counts integral | yes |
| Mean arrivals/hour | 6.628 |
| Busiest single hour | 31 |
| Zero-arrival hours | 918 |

A forecaster will happily fit a series with a silent gap and produce confident
nonsense, because a gap corrupts every lag spanning it. This series has none: no
daylight-saving artefacts, no duplicated hour, nothing imputed. It is cleaner than the
synthetic data in one respect — the generator deliberately injects missing values, and
this real extract has none.

---

## 2. The result

Two backtests. The configured design is 4 rolling origins; the second run uses 26
origins over the same series, which is the same design with more of the five years
used. **The 26-origin run is the one to read** — 1,872 scored hours against 288.

### 26 origins, 1,872 hours scored

| | MAE | RMSE | sMAPE | WAPE |
|---|---|---|---|---|
| **This project's forecaster** | **2.025** | **2.611** | **37.7%** | **31.5%** |
| Seasonal-naive (168 h) | 2.751 | 3.602 | 51.4% | 42.9% |
| Mean of same hour, previous 4 weeks | 2.215 | 2.875 | 40.9% | 34.5% |

The model is **26.4% better than seasonal-naive** and **8.6% better than the
four-week mean** on every metric.

### Peak and off-peak

Peak is the top 20% of hours by actual arrivals, the project's own definition
(`forecast.peak_quantile`), which here is 10 or more arrivals in an hour.

| | Model | Seasonal-naive | 4-week mean |
|---|---|---|---|
| **Peak** (MAE) | **3.041** | 3.951 | 3.240 |
| **Off-peak** (MAE) | **1.760** | 2.438 | 1.948 |

It wins in both, by 6.1% at peak and 9.7% off peak against the harder baseline.

### By hour band

| Band | Model MAE | Coverage |
|---|---|---|
| Night (00–06) | 1.333 | 0.897 |
| Morning (06–12) | 2.063 | 0.883 |
| Afternoon (12–18) | 2.581 | 0.904 |
| Evening (18–24) | 2.123 | 0.908 |

### Conformal coverage

**0.898 overall against a 0.90 target.** On the synthetic data the same method
under-covers at 0.8875. The intervals are *better* calibrated on the real series than
on the data they were designed against, which is the opposite of what one would
expect and is worth stating plainly rather than explaining away: the real series is
long enough and regular enough that the exchangeability the method assumes is closer
to true than it is on a single synthetic year.

### A correction worth recording

The 4-origin run — the configured default, 288 hours — showed the model **tying** the
four-week baseline off peak: 1.8743 against 1.8762, a 0.1% difference. That reading
does not survive more data: over 1,872 hours the model is 9.7% better off peak. The
tie was a 288-hour artefact, and it is recorded here because it was reported before
the wider run finished.

| | 4 origins (288 h) | 26 origins (1,872 h) |
|---|---|---|
| Model MAE | 1.979 | 2.025 |
| Seasonal-naive | 2.722 | 2.751 |
| 4-week mean | 2.135 | 2.215 |
| Off-peak gap to 4-week | **+0.1% (a tie)** | **+9.7%** |
| Coverage | 0.9167 | 0.898 |

---

## 3. Figure

`uihc_forecast_vs_baselines.png` — the last 14 days of the backtest: actual, forecast
with its interval band, and both baselines.

---

## 4. Comparison with the synthetic results

| | Synthetic (`ed_v2.csv`) | Real (UIHC) |
|---|---|---|
| Forecast MAE | 2.356 | 2.025 |
| Seasonal-naive MAE | 2.979 | 2.751 |
| **Model's advantage over seasonal-naive** | **20.9%** | **26.4%** |
| Second baseline (4-week mean) | not measured | 2.215 |
| Conformal coverage (target 0.90) | 0.8875 | 0.898 |
| Mean arrivals/hour | 6.5 | 6.628 |

**The absolute MAEs are not comparable** — different departments, different years,
different arrival rates — and the near-identical mean arrival rate is a coincidence of
scaling, not evidence of anything. **The comparable quantity is the gap to the
baseline**, and that gap is slightly *larger* on real data: 26.4% against 20.9%.

The honest reading: the forecasting method transfers. It was not overfitted to the
generator, and the one real series available to test it on does not contradict any
claim this project makes about forecasting.

---

## 5. What the real data has that the synthetic data does not

**A dead holiday feature, which is a defect this run found.** The project's nine
configured holidays are all 2024 dates, for the synthetic year. On 2013–2018 data
`is_holiday` is constant, and the feature builder says so:

> `None of the 9 configured holidays fall inside the data range … is_holiday is
> constant and will be useless. Update features.holidays.`

So the model beat both baselines **with its holiday feature switched off by
accident**. Fixing it would require editing `configs/default.yaml`, which this work
is not permitted to touch, so it is reported and left: a real deployment would need
its own holiday calendar, and the warning that catches it already exists.

**Event days the generator has no concept of.** The dataset ships indicators the
synthetic data has no equivalent for: Christmas and Boxing Day, New Year's Eve and
Day, Thanksgiving and the day after, Independence Day, and **840 game-day hours** —
home football fixtures, which move ED demand in a university town. The project's
feature builder does not know about any of them; they were available to the
forecaster only in so far as the calendar features happen to encode the date.

**A multi-year trend and drift.** Five years of real operation contain growth,
service changes and policy changes. One synthetic year generated from fixed
parameters contains none, which is why the synthetic conformal split "gets lucky"
in a way the real series does not.

**Nothing imputed.** 918 genuinely quiet hours with zero arrivals, and no missing
values at all.

---

## 6. The temperature question, answered honestly

The dataset ships `temp` for the same hour. **It was not used as a feature.** A
forecaster issuing a 72-hour forecast does not know the temperature in hour 72;
feeding it in would be predicting the past and would make the model look better than
it is.

It was tested only as a clearly labelled side experiment, **lagged by the forecast
horizon** so the value would genuinely have been available when the forecast was
issued:

| | Correlation |
|---|---|
| Lagged temperature vs arrivals | **0.314** |
| Lagged temperature vs the model's residuals | **−0.029** |

Temperature does relate to arrivals — warmer hours are busier — but it tells the model
**almost nothing it does not already know**: the correlation with what the model got
wrong is essentially zero. The calendar and seasonal-lag features have already absorbed
it. A forecast-weather feature might still help, but that is a different experiment
requiring forecast weather rather than recorded weather, and this does not test it.

---

## 7. What this run does and does not establish

**Establishes:** the arrival forecaster, unchanged and with one of its features
accidentally disabled, beats seasonal-naive by 26.4% and a four-week same-hour mean
by 8.6% on 1,872 hours of real ED arrivals, and its intervals cover at 0.898 against
a 0.90 target.

**Does not establish:** anything about waits, patients, triage, simulation or
staffing. No patient-level model touched this data, because this data has no patients
in it. And one department over one period is one data point about transfer, not a
general claim.
