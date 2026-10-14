# Phase 4 checkpoint — pipeline, report, serving-readiness

## Gates

| Gate | Result |
| --- | --- |
| Full test suite | **309 passed** (5 min 16 s) |
| Pipeline end to end on 50,000 synthetic visits | **15 stages, 143.6 s** |
| `tests/test_leakage.py` | 12 passed |
| No random splits in `src/` | 9 patterns tested, 0 found |
| Every tunable from config | asserted by `test_project_invariants.py` |
| Nothing left unimplemented | `stages_not_yet_implemented` is empty |

## What was built

| Requirement | Where | Status |
| --- | --- | --- |
| 1. Every stage wired, idempotent, individually runnable, `--data`/`--stages`, Parquet cache | `src/pipeline.py` | done |
| 2. Report stage with limitations and future work | `src/report.py` | done |
| 3. `src/serving/` with four pure pydantic-typed functions | `src/serving/api.py` | done |
| 4. README: setup, running, real datasets, structure, limitations | `README.md` | done |
| 5. End-to-end smoke test plus a timed 50k run | `tests/test_pipeline.py` | done |

## Full-run timing (50,000 visits)

| stage | seconds |
| --- | --- |
| load | 0.4 |
| validate | 0.22 |
| split | 0.17 |
| features | 0.26 |
| train_wait | 6.52 |
| ablation | 6.64 |
| explain | 1.84 |
| fairness | 1.82 |
| forecast | 32.24 |
| survival | 4.69 |
| calibrate_sim | 2.11 |
| scenarios | 14.4 |
| sensitivity | 68.48 |
| staffing_search | 3.76 |
| report | 0.02 |

Total **143.6 s**. The two slowest stages are `sensitivity`
(re-runs every scenario in nine perturbed worlds) and `forecast` (refits the model at
each rolling origin).

## Headline results, read from the artifact files

| Result | Value |
| --- | --- |
| Best wait model | base+queue / log1p / log1p_median |
| Test MAE / RMSE | 14.19 / 24.15 min |
| Baseline MAE | 38.71 min |
| Conformal coverage (90% nominal) | 0.8814 |
| Classifier PR-AUC / recall / FNR | 0.9509 / 0.8807 / 0.1193 |
| Forecast MAE vs seasonal-naive | 2.220 vs 2.764 (19.7% better) |
| Forecast interval coverage | 0.8993 |
| Fairness: overall FNR | 0.1193 |
| Survival: abandonment events | 1,659 |
| Cheapest feasible staffing plan | `day=7, eve=5, night=3` at 92,400 |
| Plans feasible | 36 of 48 |

## Serving-readiness

`src/serving/api.py` exposes four functions, each loading from `artifacts/models/`,
retraining nothing and holding no global state:

- `predict_wait(patient, queue_state)` -> point, conformal interval, prolonged
  probability, top SHAP contributions, caveats
- `forecast_arrivals(horizon_hours, as_of)` -> the documented
  `DataFrame[timestamp, yhat, lower, upper]`
- `run_scenario(name, overrides)` -> KPIs with confidence intervals
- `recommend_staffing(target)` -> cheapest feasible plan, or an explicit "none"

Requests and responses are pydantic models, so a FastAPI layer can adopt them as its
schema unchanged. Two design points worth noting:

- **Queue state is an explicit input.** Only the live system knows how many patients are
  waiting, so the library takes it rather than guessing, and the response says when it
  was missing.
- **The validation gate reaches the serving layer.** `run_scenario` and
  `recommend_staffing` refuse on an unvalidated simulation exactly as the pipeline does.
  This was a bug found by the serving tests: the gate was in the pipeline but not in the
  library a future API would call.

## Deviations from CLAUDE.md

1. **Three stages were added to the original 13** — `survival` (Phase 2),
   `sensitivity` (Phase 3) — making 15. Both are named in the project brief.
2. **New modules beyond the original layout:** `src/utils/priority.py` (so the
   generator and the DES cannot drift), `src/utils/report.py` (Markdown tables without
   a `tabulate` dependency), `src/report.py`, `src/serving/`,
   `src/models/train_wait_stage.py`, `src/models/figures.py`,
   `src/models/forecast_figures.py`, `src/simulation/sim_figures.py`.
3. **`pyarrow` added** (approved in review). **`tabulate` was installed and then
   removed** when it turned out not to be on the approved list.
4. **The `split` stage keeps all rows**; each model stage selects and logs its own
   cohort.
5. **`ablation` aliases `train_wait`**, so a full run repeats that work (~6.6 s).

## Known limitations

Carried forward from every earlier phase and consolidated in `artifacts/report.md`:
synthetic data throughout, the ablation's inflated effect, censoring bias, the
high-acuity classifier failure, the unidentifiable treatment/boarding split, unmodelled
nurses and beds, conformal exchangeability, and the forecast-window-sized staffing plan.

Specific to this phase:

- **The report is only as complete as the artifacts present.** Missing files produce an
  explicit "not available" section rather than a silently shorter report; this is tested.
- **The smoke test runs on a dataset too small to forecast.** That is deliberate: it
  checks the stages compose and that forecasting degrades with an explanation rather
  than crashing.
- **No API, UI or container.** Out of scope by instruction; `src/serving/` is the seam.
