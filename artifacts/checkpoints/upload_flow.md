# Upload workflow — end-to-end run

Upload an ED visits CSV through the dashboard or the API, and the whole pipeline runs
on it with no manual commands. This is the record of one real run.

Every number below was read from the running service's own responses
(`GET /jobs/{id}` and `GET /jobs/{id}/results`), not recomputed here.

**The dataset is synthetic** — `data/raw/ed_v2.csv`, 57,500 visits generated and
calibrated to published benchmarks. It is not real patient data.

## How it was run

```powershell
.\scripts\run_api.ps1          # terminal 1, bound to 127.0.0.1:8000
.\scripts\run_app.ps1          # terminal 2, bound to 127.0.0.1:8501
.\scripts\smoke_upload.ps1 -Keep -TimeoutMinutes 90
```

The smoke script posts the file to `POST /jobs` as multipart form data over HTTP,
polls `GET /jobs/{id}` until the job finishes, and prints what it reads back.

| | |
| --- | --- |
| File | `data/raw/ed_v2.csv`, 5.89 MB, 57,500 rows |
| Job id | `d3600d104f37` |
| Options | depth `full`, simulation on, config's demo roster |
| Mapping | automatic — every column matched its schema name exactly |
| **Total wall-clock** | **207.3 s** |
| Outcome | succeeded, 14 of 15 stages ran |

## Stage timings

| Stage | State | Seconds |
| --- | --- | ---: |
| load | done | 0.24 |
| validate | done | 0.33 |
| drift | done | 0.15 |
| split | done | 0.14 |
| features | done | 0.35 |
| train_wait | done | 12.11 |
| score | skipped | — |
| explain | done | 2.15 |
| fairness | done | 2.52 |
| forecast | done | 42.85 |
| survival | done | 3.03 |
| calibrate_sim | done | 24.08 |
| scenarios | done | 26.82 |
| staffing_search | done | 90.56 |
| report | done | 0.05 |

`score` is skipped outside scoring-only mode, and says so: *"Scoring against the
baseline models runs only in scoring-only mode; this run trained models on your data
instead."* The staffing search dominates the run at 90.6 s, because it simulates 144
candidate plans.

`sensitivity` is not part of the upload workflow at all. It re-runs every scenario
under four demand and service multipliers — offline research work, not something to
put behind an upload button.

## Headline results

| Measure | Value |
| --- | ---: |
| Rows read | 57,500 |
| Rows accepted | 57,179 |
| Rows dropped | 321 |
| Wait-time MAE | 11.52 min |
| Rolling-mean baseline MAE | 34.23 min |
| Improvement over baseline | 66.3% |
| Conformal coverage (nominal 90%) | 84.9% |
| Forecast MAE | 2.36 arrivals/h |
| Forecast improvement over seasonal-naive | 20.93% |
| Forecast interval coverage | 89.2% |
| Features with a large distribution shift | 0 of 8 |
| Simulation validated | yes |
| Downloadable files produced | 60 |

Rows were dropped for two reasons only: 206 with `triage_level` missing or outside
1–5, and 115 where `seen_time` preceded `arrival_time`. No column was more than half
empty.

Fairness compared 15 subgroups and flagged none. Survival recorded 55,132 patients
seen, 2,046 who left without being seen, and 1 still waiting at the cutoff.

The staffing search evaluated 144 plans, found 72 that met the target, and returned
the cheapest: cost 49,280, p90 wait for ESI 3 of 77.2 min, abandonment 1.83%.

**The roster was the config's demo roster**, and the run says so on every staffing
result. The warning returned with the job reads: *"This used the demo roster from the
config, which describes the synthetic ed_v2 dataset and not your department.
Simulation and staffing results depend on it entirely, so enter your own roster
before acting on them."*

## The uploaded run reproduced the baseline exactly

Uploading the same file the baseline pipeline was run on produced the same numbers to
every reported digit — MAE 11.52, baseline MAE 34.23, coverage 0.849, forecast MAE
2.36, improvement 20.93%. The comparison panel showed the two columns identical.

That is the result to want: the upload path and the command-line pipeline are the
same code reading the same config with the same seed, so they cannot disagree. A
difference here would have meant the upload path was quietly doing something else.

## Isolation held

The run wrote only to `runs/d3600d104f37/`. After it finished, the newest file in
`artifacts/` was dated 2026-10-05 21:18 and the newest in `data/processed/` 2026-10-05
20:52 — both from the previous baseline run, neither touched by this one, which ran on
2026-10-06 at 23:43.

Two tests enforce this rather than leaving it to inspection:
`test_a_run_does_not_touch_the_baseline_artifacts` hashes every baseline file before
and after a real run, and `test_uploading_never_changes_what_predict_wait_answers`
compares a `/predict-wait` response field for field either side of one.

## What the run warned about

Three notes about inferred timestamp formats, one per time column — correct, since the
upload did not state a format. Stating `-DatetimeFormat "%Y-%m-%d %H:%M:%S"` removes
them. Plus the demo-roster warning quoted above.

## Not covered by this run

- **Browser interaction.** This exercised the API. What a person sees in the
  dashboard — progress moving, a download opening, the mapping form being usable — is
  in `docs/upload_manual_test_checklist.md` and was not performed here.
- **A failing simulation validation.** The demo roster fits this dataset, so
  calibration passed. The refusal path is covered by a test asserting the invariant
  (a scenario result implies a passed validation) and by the dashboard test that
  renders the refusal, but no real dataset in this repository makes it fire.
- **Concurrent users.** One job at a time is enforced; several people uploading at
  once was not tried.
