# Serving checkpoint — API, dashboard and packaging

Everything below was run on this machine. Numbers come from the recorded session, not
from expectation.

- **Tests: 396 passed** (`python -m pytest -q`), plus 2 regression tests added after the
  sweep found a live bug, so the suite now stands at 398.
- **API and dashboard: started locally and exercised end to end.**
- **Docker: NOT built and NOT run.** See "What was not verified".

## How they were started

```powershell
# terminal 1
.\scripts\run_api.ps1          # uvicorn api.main:app on http://localhost:8000

# terminal 2
.\scripts\run_app.ps1          # streamlit on http://localhost:8501
```

For this session both were launched directly so output could be captured:

```
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
python -m streamlit run app/dashboard.py --server.port 8501 --server.address 127.0.0.1 --server.headless true
```

Startup log:

```
api | Ready: model base+queue-raw-b098128b, data real-57179rows-20261005
Uvicorn running on http://127.0.0.1:8000
streamlit | You can now view your Streamlit app in your browser. URL: http://127.0.0.1:8501
```

## Endpoint sweep

```
## health
$ curl -X GET http://127.0.0.1:8000/health
-> 200 in 4 ms
   status=ok models=True forecaster=True sim_calibrated=True
   model_version=base+queue-raw-b098128b  data_version=real-57179rows-20261005
## predict-wait (full queue state)
$ curl -X POST http://127.0.0.1:8000/predict-wait
-> 200 in 50 ms
   93.6 min  [65.7, 121.6] at 90%
   prolonged probability 0.986, flag=True
   top factors: [('n_waiting_same_priority', 30.32), ('n_waiting', 19.63), ('n_waiting_higher_priority', 5.29)]
## predict-wait (minimal queue state)
$ curl -X POST http://127.0.0.1:8000/predict-wait
-> 200 in 40 ms
   19.1 min  [0.0, 44.4]
## forecast
$ curl -X GET http://127.0.0.1:8000/forecast?horizon_hours=24
-> 200 in 101 ms
   24 hourly points from 2024-12-30T23:00:00, 90% intervals
   first: 2024-12-30T23:00:00  yhat=5.54  [0.82, 10.25]
   mean 6.51 arrivals/hour
## simulate (baseline)
$ curl -X POST http://127.0.0.1:8000/simulate
-> 200 in 551 ms
   mean wait 35.37 min  CI [33.59, 37.14]
   p90 ESI 3 102.2 min, abandonment 0.0394, utilisation 0.822
## simulate (fast track)
$ curl -X POST http://127.0.0.1:8000/simulate
-> 200 in 505 ms
   mean wait 8.18 min  CI [7.16, 9.21]
## simulate (replication cap)
$ curl -X POST http://127.0.0.1:8000/simulate
-> 200 in 1350 ms
   requested 50, ran 10; capped notice present: True
## staffing-recommendation
$ curl -X GET http://127.0.0.1:8000/staffing-recommendation
-> 200 in 25 ms
   feasible=True  plan={'day': 3, 'evening': 3, 'night_early': 2, 'night_late': 2}  cost=49,280
   p90 ESI 3 77.2 min, 72/144 plans feasible
## error handling
   triage_level = 0         POST -> 422
   age = 200                POST -> 422
   horizon_hours = 999      GET  -> 422
   unknown scenario         POST -> 400
   unknown route            GET  -> 404
   GET on a POST route      GET  -> 405
## summary
   200      4 ms  health
   200     50 ms  predict-wait (full queue state)
   200     40 ms  predict-wait (minimal queue state)
   200    101 ms  forecast
   200    551 ms  simulate (baseline)
   200    505 ms  simulate (fast track)
   200   1350 ms  simulate (replication cap)
   200     25 ms  staffing-recommendation
   422         -  triage_level = 0
   422         -  age = 200
   422         -  horizon_hours = 999
   400         -  unknown scenario
   404         -  unknown route
   405         -  GET on a POST route
```

The dashboard was checked separately:

```
GET http://127.0.0.1:8501/_stcore/health  -> 200 "ok"
GET http://127.0.0.1:8501/                -> 200, 1837 bytes, title "Streamlit"
```

and its own `call_api` helper was exercised against the live API, including the
unreachable case, which returns the friendly message rather than raising:

```
call_api(/health)                 -> OK
call_api(/staffing-recommendation)-> OK
call_api(/predict-wait POST)      -> OK, 19.1 min
unreachable API                   -> "Cannot reach the API at http://127.0.0.1:9. Start it with ..."
```

## A bug the live run found that the tests did not

`POST /predict-wait` returned **500** whenever the caller omitted the optional queue
fields:

```
ValueError: Out of range float values are not JSON compliant: nan
```

A feature the caller does not supply is `NaN` in the design matrix. That NaN reached the
explanation's `value` field, and NaN is not valid JSON, so encoding the response raised
and turned a perfectly good prediction into a server error.

Every existing test supplied the full queue state, so none of them hit it — but the
dashboard's own form sends exactly that shape when the user does not know the recent-wait
history, so a real user would have hit it immediately. Fixed by serialising non-finite
floats as `null`, which also says the right thing: the value is *absent*, not zero. Two
regression tests were added, one of which asserts the whole response survives
`json.dumps(..., allow_nan=False)` — the constraint a browser or proxy actually enforces.

## Part A — rolling conformal calibration

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

Static calibrates once on a fixed block and assumes the test residuals are exchangeable
with it. Under drift they are not: coverage was
**0.8490** against a nominal 0.90, a gap of
-0.0510, outside the configured 0.04 tolerance.

Rolling recalibrates per patient from same-acuity residuals **already observed when that
patient arrived**, which moves coverage to
**0.8875**
(gap -0.0125) — inside tolerance. Every acuity level
improved. Mean width rises from 39.0 to
46.5 minutes, which is the honest price: a
narrow interval that misses one time in seven is worse than a wider one that does what it
claims.

One implementation note that mattered. The first version drew only on the train,
validation and calibration blocks and barely helped (0.8556). The pool now spans **every**
block including test, because admissibility is decided by `seen_time < arrival_time`, not
by block membership: a patient arriving in December may use the residual of a patient seen
in November. Withholding those would measure something no deployment would ever do.
`tests/test_rolling_conformal.py` asserts the boundary directly, including that a residual
observed at exactly the arrival instant is excluded.

## Part B — per-acuity thresholds

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

**ESI 1 and 2 still cannot be fixed, and the reason is data, not tuning.**

- ESI 1 has **1** prolonged-wait case in the whole test block, and **0** in the validation
  block used for selection.
- ESI 2 has **29** on test and **10** on validation — below the configured minimum of 25.

Both keep the pooled threshold, and the table says so in `threshold_source`.

Diagnostic run on the test block, to establish whether a threshold *could* help:

| | ESI 1 | ESI 2 |
| --- | --- | --- |
| positive-case scores, max | 0.0130 | 0.1993 |
| negative-case scores, 95th percentile | 0.0309 | 0.0544 |
| best achievable at threshold 0.10 | recall 0.000 | recall 0.345, precision 0.400 |
| best achievable at threshold 0.05 | recall 0.000, 2 false alarms | recall 0.759, precision 0.250 |

So the two levels fail for different reasons, and only one of them is hopeless:

- **ESI 1 has no signal at all.** Its single positive case scores *below* the 95th
  percentile of its negatives, so no cut-off separates it from noise.
- **ESI 2 does have signal.** A threshold near 0.05 would catch 22 of 29 cases — but at
  precision 0.250, which is three false alarms for every true catch, and it cannot be
  chosen honestly from 10 validation positives. Lowering `min_positives_per_level`, or
  selecting thresholds on validation and calibration combined, would make it fittable.
  Neither was applied: both change the method, and the second spends a block reserved for
  conformal calibration.

Where thresholds *were* fitted (ESI 3-5) the effect is small and within noise: ESI 4's FNR
improves 0.1345 ->
0.1235
while ESI 3 and 5 worsen slightly, and every confidence interval overlaps. Overall FNR is
essentially unchanged
(0.1245 ->
0.1253).
Per-acuity thresholds are the right mechanism; they are simply not where this model's
problem lies.

## Parts D, E, F — what was built

| Part | Files |
| --- | --- |
| D: API | `api/main.py`, `api/schemas.py`, `api/version.py`, `tests/test_api.py` (43 tests) |
| E: dashboard | `app/dashboard.py` — 7 pages |
| F: packaging | `requirements-serving.txt`, `Dockerfile.api`, `Dockerfile.app`, `docker-compose.yml`, `.dockerignore`, `scripts/run_api.ps1`, `scripts/run_app.ps1` |

Design points worth recording:

- **No mutable global state.** The bundle is loaded once into `app.state` at startup and
  only read. Nothing in the request path trains, refits or writes.
- **`/staffing-recommendation` reads the precomputed artifact.** The search evaluates 144
  plans over many replications; re-running it per request would take minutes.
- **`/simulate` is capped** at `serving.simulate.max_replications` (10). A discrete-event
  run is real work, and an uncapped endpoint is a denial of service wearing a lab coat.
  A capped request says so in its caveats rather than quietly returning a wider interval
  than the caller asked for.
- **No patient identifiers in logs.** Requests log method, path, status, duration, acuity
  and queue depth. `test_no_patient_identifier_is_logged` asserts the arrival timestamp
  never appears.
- **Every prediction response carries** the decision-support notice, the model version and
  the data version, both derived from the artifacts so they cannot drift from what is
  loaded.

## What was not verified

**Docker images were never built or run.** The CLI is present but the daemon is not:

```
$ docker --version
Docker version 29.7.2, build a7dcaa6

$ docker info
failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine;
check if the path is correct and if the daemon is running

$ docker compose config
exit 0          # the compose file is valid and resolves both services

$ docker compose config --services
api
dashboard
```

So the compose file and both Dockerfiles are **syntactically valid and resolve**, and
nothing more than that has been established. The images are unbuilt, the healthchecks
have never executed, and the read-only artifact mounts have never been exercised. Treat
the Docker path as unverified. The PowerShell scripts are the supported path on this
machine and have been run end to end.

Also not verified: the dashboard's pages were confirmed to *serve* (HTTP 200, Streamlit
health `ok`, no errors in its log) and its API-calling helper was exercised directly, but
the rendered widgets were not driven through a browser.
