# CLAUDE.md -- working rules for this repository

Decision support only. Nothing in this repository makes an autonomous clinical
decision, triages a patient, or alters care. Every output is an estimate for a human
to weigh. Keep that framing in docstrings, the report and the README.

## Research gaps this project closes

Keep these visible in code comments, artifact names and the final report.

| Gap | Problem | How this repo closes it |
| --- | --- | --- |
| **G1** | Forecasting, wait prediction and simulation are published as separate work. | One pipeline: `forecast()` output feeds the DES, which feeds `staffing_search.py`. |
| **G2** | Fairness is rarely evaluated in ED prediction papers. | `explain/fairness.py` reports subgroup MAE, FNR and FPR with gaps vs overall. |
| **G3** | Queue-state features are known to help but are rarely isolated. | `base` vs `base+queue` ablation written to `artifacts/tables/ablation.csv`. |

## Non-negotiable ML rules

1. **No leakage.** Wait-time features may use only what is knowable at the patient's
   `arrival_time`. `seen_time`, `departure_time`, `wait_minutes`, `los_minutes`,
   `disposition`, `is_lwbs` and `prolonged_wait` are on the blocklist in `schema.yaml`,
   and `src.data.schema.assert_no_leakage` is called before every model fit.
2. **Queue state is strictly backward-looking.** Every queue feature is computed from
   *other* patients' timestamps strictly before this patient's `arrival_time`. A
   patient who arrives later must never change an earlier patient's features -- there
   is a test for exactly this.
3. **Time-based splits only.** `train < val < test` by arrival date. There is no random
   split mode, and `split.mode` accepts only `fraction` or `dates`.
4. **Every model has a baseline.** Wait time: rolling mean of the last *k* patients'
   waits at the same triage level. Arrivals: seasonal-naive (lag 168 h). A model that
   does not beat its baseline is reported as such, not quietly dropped.
5. **Report MAE and RMSE** (plus MAPE for arrivals), and **recall and FNR** for the
   prolonged-wait classifier. A false negative -- telling a patient they will be seen
   soon when they will not -- is the costly error.
6. **Fix every seed.** One `seed` in the config; modules derive child streams through
   `src.utils.seed.derive_seed`. Save every model, metrics JSON and figure under
   `artifacts/`.

## Code conventions

- Python 3.11. Type hints and docstrings on every public function.
- `logging` via `src.utils.logging.get_logger(__name__)`. Never `print`.
- No magic numbers in code: everything tunable lives in `configs/default.yaml`.
- Real-dataset column knowledge lives **only** in `configs/column_map.yaml` and
  `src/data/adapter.py`.
- Pure functions and explicit classes; no hidden global state. Models, calibrators and
  parameter bundles are saved as artifacts so a FastAPI layer can load them later.
- Small, readable, tested functions over cleverness.
- **Never fabricate results.** Any metric in a README, report or summary must come from
  actually running the code.
- No placeholder code and no silent TODOs. Deferred work is named explicitly in the
  README's "Not built yet" section and in `STAGE_ORDER` in `src/pipeline.py`.

## Scope

**In scope now:** data contract, synthetic generator, real-data adapter, features,
wait-time models, explainability, fairness, conformal intervals, arrival forecasting,
SimPy simulation, staffing search, one-command pipeline, the REST API, the Streamlit
dashboard, the upload workflow, Docker, tests.

**Deliberately not built:** authentication, TLS and rate limiting on the API. That is
why both services bind to `127.0.0.1` by default and why the upload workflow caps
file size, rows, queue depth and run time. Do not change those defaults to `0.0.0.0`;
they are environment variables so a container can widen them without editing a file.

**Rules the upload workflow adds:**

1. **Never guess a required column.** Detection is exact name, then the same name
   ignoring case and punctuation, then a documented alias in
   `uploads.column_aliases`. Nothing after that. Anything ambiguous or missing stops
   the job and asks, because a wrong guess trains a plausible model on the wrong
   field.
2. **A stage that cannot run says why, in English.** Not a traceback, not a failed
   job, and never a number computed from data that does not support it.
3. **A job writes only to its own run folder.** `src.pipeline.redirect_to_run` is the
   single implementation; two tests enforce it by hashing `artifacts/` across a real
   run and by comparing `/predict-wait` either side of one.
4. **Scenario and staffing results are refused when the simulation fails validation,
   with no override in the UI.** A recommendation from a model that cannot reproduce
   the department looks exactly as confident as a correct one.
5. **The uploaded file is never logged and never echoed.** Logs carry counts, not
   rows. The file lives in the run folder and is deleted on request and on a timer.

**Out of scope entirely (future work):** reinforcement learning, deep learning
(LSTM/TFT).

**Dependencies:** ask before adding anything not in `requirements.txt`.
`python-multipart` was added for the upload endpoint: FastAPI raises at import time
when a route declares `File`/`Form` without it, so the whole API depends on it.

## Phases

Work proceeds in phases. After each phase: stop, summarise what was built, show real
test results, and wait for approval.

- **Phase 0 -- Foundation. Done.** Repo skeleton, `requirements.txt`, `Makefile`,
  configs, `schema.yaml` and `schema.py` validation, `synthetic.py`, `adapter.py` with
  a sample `column_map.yaml`, `split.py`, utils, and tests for schema, split and
  synthetic data. A `pipeline.py` stage registry was built here too, so the foundation
  stages are runnable; Phase 4 wires the remaining stages into it.
- **Phase 1 -- Features and wait-time models. Done.** Calendar and queue-state features,
  baselines, LightGBM regressor and classifier, ablation table, conformal intervals,
  SHAP report, fairness report, plus the leakage and conformal tests. Delivered in
  three checkpoints: **1a** features + leakage tests, **1b** models + baselines +
  ablation + conformal, **1c** SHAP + fairness.
  **1a and 1b are done.** Agreed requirements for 1b, all implemented:
  - Compare a raw `wait_minutes` target against a log-transformed one, and report MAE
    for both **on the original minute scale** (back-transform before scoring; remember
    that `exp(mean of logs)` is a median, so state which estimator is used).
  - Report the **direction and rough size of the bias** from training only on patients
    who were seen. Patients who left without being seen are the longest waiters, so
    excluding them biases predictions low; quantify it rather than only noting it.
- **Phase 2 -- Forecasting. Done.** Seasonal-naive baseline, LightGBM forecaster with
  rolling-origin evaluation, conformal intervals, and the optional survival module.
- **Phase 3 -- Simulation. Done.** SimPy DES, calibration and validation against the data,
  scenarios, and staffing search fed by `forecast()` output.
- **Phase 4 -- Pipeline and report. Done.** `pipeline.py` CLI wiring every stage, the report
  stage, README, and the end-to-end smoke test.

### Generator policy: priority aging

The queue uses priority aging: effective priority is
`max(triage - waited/aging_minutes_per_level, aging_floor)`. The floor is a **clinical
safety constraint, not a tuning knob**. At 2.5 an aged low-acuity patient may overtake
a fresher ESI 3 or 4 but never a freshly arrived ESI 1 or 2. Measured alternatives:

| Floor | Consequence |
| --- | --- |
| 3.0 | ESI 3 cannot age at all; ESI 3 and ESI 5 wait tails invert. |
| **2.5** | **Chosen.** ESI 1-2 protected; ESI 3-5 tails converge by design. |
| 2.0 | Aged patients overtake ESI 2, whose mean wait triples (9.4 -> 31 min). Unsafe. |

Relieving low-acuity starvation necessarily means someone fresher is overtaken, so the
ESI 3-5 p90 waits converge and ESI 3's can exceed ESI 5's. Mean waits still order
correctly by acuity. Never "fix" that convergence by raising the floor without
re-measuring ESI 2.

### Honesty constraints carried forward

Two results this project will produce are shaped by the synthetic generator and must
never be presented as findings about real emergency departments:

- **The G3 ablation will favour queue features heavily**, because the generator creates
  waits from a queue (measured congestion-to-wait correlation r = 0.75). State this
  alongside the ablation table.
- **The G2 fairness report will show near-zero gaps for `sex`**, which is generated
  independently of outcome, and `age`/`arrival_mode` gaps that are largely acuity
  effects. Report it as a correct module with a null finding by construction.
