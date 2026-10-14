# ED Intelligence

Emergency-department arrival forecasting, per-patient wait-time prediction and staffing
simulation, connected in one pipeline.

**Decision support only.** Every output is an estimate for a clinician or manager to
weigh. Nothing here triages a patient or makes an autonomous clinical decision.

## Quick start

**On a new machine, double-click `Setup.cmd` first** — it finds Python 3.12, creates
`.venv` and installs the pinned dependencies. A few minutes, once. Skip it if `.venv`
already exists; if you skip it and it was needed, the launcher says so.

**Then double-click `Start-Demo.cmd`.**

It checks the environment, starts the API and the dashboard, waits for both and opens
the browser. Nothing else is needed and no terminal is involved.

**Double-click `Stop-Demo.cmd`** to shut it down.

```
Dashboard: http://localhost:8501   API docs: http://localhost:8000/docs
```

The dashboard opens on the Command center, which shows the headline results without a
click. Both services bind to `127.0.0.1` and are reachable from this machine only.

A few things it deliberately will not do. If something already answers on port 8501 --
which is what the Docker stack looks like from outside -- it opens the browser and
starts nothing. If port 8000 is held by something that is not a healthy ED API, it says
which port, who holds it and why, and stops: it never kills a process to free a port.
If `artifacts\` holds no trained models it names the one command that produces them
(`.\scripts\run_pipeline.ps1`) and stops, rather than quietly starting a few minutes of
training. `Stop-Demo.cmd` stops only what `Start-Demo.cmd` recorded in
`runs\demo.pids`, after checking each process is still the one that was recorded, and
never touches Docker.

Logs are in `runs\logs\api.log` and `runs\logs\app.log`.

## What it is for

Three things are normally studied separately: how many patients will arrive, how long a
given patient will wait, and what staffing would fix it. This repository links them, and
makes three usually-missing checks first-class.

| Gap | What is missing in the literature | What this repo does |
| --- | --- | --- |
| **G1** | Forecasting, wait prediction and simulation are built in isolation. | `forecast()` output drives the discrete-event simulation, which drives the staffing search. |
| **G2** | Subgroup fairness is rarely evaluated. | Subgroup MAE, FNR and FPR with bootstrap intervals — which surfaced a real failure for high-acuity patients. |
| **G3** | Queue-state features are assumed to help but rarely isolated. | A controlled `base` vs `base+queue` ablation, with its caveat attached to the table. |

## Headline results

On `data/raw/ed_v2.csv` (57,500 synthetic visits, 365 days), each against the
baseline it is measured against. `tests/test_headline_regression.py` pins all of
them, so none can drift unnoticed.

| Result | This system | Baseline | Margin |
|---|---|---|---|
| Wait-time MAE | **11.52 min** | 34.23 min — queue position x mean service time | **66.3% lower** |
| Interval coverage (0.90 target) | **0.8875** rolling conformal | 0.849 static split | **3.9 pp closer to nominal** |
| Arrivals forecast MAE | **2.356 /h** | 2.979 /h — seasonal-naive (last week, same hour) | **20.9% lower** |
| Cheapest feasible roster | **3 / 3 / 2 / 2 at 49,280** per week | 72 of 144 candidate plans were feasible | feasibility judged on the upper 95% bound |

The data is synthetic, so these are measurements of the method rather than of any
department. Across ten regenerated datasets the wait model beat its baseline on all
ten and the forecast beat seasonal-naive on all ten, while the cheapest roster varied
over seven different plans — the method transfers, the specific roster does not. See
**Known limitations** below and the **External validation** section for the one part
of this that has been tested on real data.

## Status

All phases complete. **1418 tests pass, 13 skipped.** The pipeline has **16 stages**,
and the most recent full run of all of them on the 57,500-visit `ed_v2.csv` took
**269 seconds** (measured in the container, with the trainer profile).

| # | Stage | | # | Stage |
| --- | --- | --- | --- | --- |
| 1 | `load` | | 9 | `forecast` |
| 2 | `validate` | | 10 | `survival` |
| 3 | `split` | | 11 | `competing_risks` |
| 4 | `features` | | 12 | `calibrate_sim` |
| 5 | `train_wait` | | 13 | `scenarios` |
| 6 | `ablation` | | 14 | `sensitivity` |
| 7 | `explain` | | 15 | `staffing_search` |
| 8 | `fairness` | | 16 | `report` |

Every stage is independently runnable and idempotent; `-Stages` resumes or repeats any
part of it.

Uploading the same file through the API runs the same stages and reproduces the
command-line numbers exactly. Per-stage timings from that run are in
`artifacts/checkpoints/upload_flow.md`, which records 14 stages because it predates
`competing_risks` and the upload path's optional skips.

The 13 skips are conditional rules in `tests/test_dashboard_imports.py` that do not
apply to the modules they skip — a helper with no `render()`, and pages that draw no
charts. None can be enabled; each activates by itself when a module gains a chart.

Per-phase write-ups, with exact test results and metrics read from the artifact files,
are in `artifacts/checkpoints/`. The most recent is `days_work.md`.

## Setup

Requires **Python 3.11 or 3.12**. Every pin ships prebuilt wheels for both.

```bash
make setup          # creates .venv and installs pinned dependencies
make test           # runs the test suite
make run            # runs the whole pipeline on synthetic data
```

> The `make` targets need GNU Make and a POSIX shell (Git Bash, WSL, macOS, Linux)
> because `clean` uses `rm` and `find`. **On Windows, use the PowerShell scripts
> below instead** -- `make` is not required.

## Manual start (Windows, PowerShell)

The commands `Start-Demo.cmd` wraps. Use them when you want the two services in their
own terminals, with their output in front of you.

`make` is not available on a stock Windows install, so `scripts\` holds PowerShell
equivalents of every target. They locate `.venv` themselves, so nothing needs
activating.

```powershell
.\scripts\run_api.ps1               # terminal 1: the API on 127.0.0.1:8000
.\scripts\run_app.ps1               # terminal 2: the dashboard on 127.0.0.1:8501
```

```powershell
.\scripts\setup.ps1                 # create .venv (Python 3.12), wheels-only install
.\scripts\run_tests.ps1             # the whole suite
.\scripts\run_tests.ps1 -Fast       # skip the slow-marked tests
.\scripts\run_tests.ps1 -Path tests\test_simulation.py
.\scripts\run_pipeline.ps1          # the whole pipeline on synthetic data
.\scripts\run_pipeline.ps1 -Stages load,validate,split
.\scripts\run_pipeline.ps1 -Data real
.\scripts\run_pipeline.ps1 -ListStages
```

The research studies have helpers too. None of them is part of the pipeline, and
each prints what its result is and is not before it prints the result:

```powershell
.\scripts\run_robustness_study.ps1                  # seeds 1-10 with the true generator (~20 min, resumable)
.\scripts\run_robustness_study.ps1 -AggregateOnly   # rebuild the tables from the seeds already on disk
.\scripts\run_scenario_lab.ps1                      # uncalibrated bed / nurse / imaging what-ifs
.\scripts\run_staffing_rl.ps1                       # experimental staffing agent (it loses; read the verdict)
```

If PowerShell blocks the scripts (`running scripts is disabled on this system`), either
allow them for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

...or skip the scripts entirely and call the interpreter directly, which always works:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt
.\.venv\Scripts\Activate.ps1       # optional; then plain `python` is the venv's
python -m pytest
python -m pytest -m "not slow"
python -m src.pipeline run --data synthetic
python -m src.pipeline list-stages

# keep a run's outputs away from the baseline artifacts/
python -m src.pipeline run --data real --run-name my_extract
python -m src.pipeline run --data real --artifacts-dir artifacts_experiment
```

`--run-name` puts everything, intermediates included, under `runs/<name>/`;
`--artifacts-dir` moves only the results and leaves the intermediates shared. The
upload workflow uses the same redirection function, which is why an uploaded job can
never write to `artifacts/`.

`python -m pytest` works directly from the project root: `tests/conftest.py` puts the
repository root on `sys.path`, so no editable install or `PYTHONPATH` is needed.

### Running the API and the dashboard

The service loads the trained artifacts and serves them over HTTP. It never trains, and
never re-runs the staffing search per request.

```powershell
# terminal 1 -- the API
.\scripts\run_api.ps1          # prints http://localhost:8000/docs

# terminal 2 -- the dashboard
.\scripts\run_app.ps1          # prints http://localhost:8501
```

Both scripts find `.venv` themselves and print the URLs that actually open. **Both
bind to `127.0.0.1` by default**: authentication is optional and off unless you set
`ED_API_KEY`, the API accepts file uploads, and both handle patient-shaped data, so
neither is published to the local network unless you ask for it:

```powershell
$env:ED_API_HOST = "0.0.0.0"; .\scripts\run_api.ps1   # publish deliberately
.\scripts\run_api.ps1 -BindHost 0.0.0.0 -Port 8080    # or per run
```

#### The dashboard's pages

Written for an ED operations manager or charge nurse rather than a developer:
plain language on every main view, technical detail behind "Technical details"
expanders and on the About page. Every page carries a **Demo data (synthetic)**
badge, because the data is generated and a dashboard that looks like a hospital
feed has to keep saying that it is not one.

| Group | Page | What it answers |
|---|---|---|
| Operations | **Command center** | Where the department stands: six figures with their change against the same hour last week, the arrivals forecast against its seasonal-naive baseline, the acuity mix, waits by acuity against the 30-minute target, a day-by-hour congestion heatmap, and the alerts with the rule that fired each one |
| Operations | **Patient flow board** | Who was waiting at any moment in the recorded period, in the department's own priority order, with each patient's wait so far, their estimated remaining wait and their long-wait risk |
| Planning | **Demand forecast** | Expected arrivals per hour at 24 / 48 / 72 h, with ranges, a CSV download, and the measured error against "just repeat last week" |
| Planning | **Staffing planner** | The cheapest roster that still meets the waiting target, how many of the 144 candidates were feasible, why the finding is partly circular, and what the same search produced on other datasets |
| Planning | **Scenario lab** | Whether a change (an extra doctor at peak, a fast track, a surge) would actually help, with confidence intervals and a plain verdict, using the simulation that passed its validation check |
| Clinical intelligence | **Wait and risk prediction** | How long one patient is likely to wait, as a range with the factors driving it, and the chance they leave before being seen at 30, 60 and 120 minutes |
| Clinical intelligence | **Quality and equity** | How accurate the estimates were on held-out data, where the ranges are too narrow, and where the long-wait warning does not work |
| Clinical intelligence | **Research results** | Which conclusions survive across generator seeds, the risk of leaving before being seen, the uncalibrated resource scenarios, and the experimental staffing agent |
| Administration | **Data upload** | Upload an extract and run the whole analysis on it, as a guided four-step flow |
| Administration | **System status** | Service health, model and data versions, the model card, every alert rule with its threshold, the upload limits, and a collapsed **Assumptions and limitations** section holding the roster, circularity and robustness caveats |
| Administration | **About and limitations** | Purpose, the synthetic data, how to read a range, known limitations, and a Diagnostics expander with the endpoint, model identifier and retention |

Every page carries a slim persistent banner reading **"Synthetic data. Decision
support only, not for clinical use."**, and the top bar names the fictional
institution (Northfield General Hospital — Emergency Department) beside a **Replay
time** chip, the model and data versions, and a `SYNTHETIC` environment chip. The
replay time is labelled as replay for a reason: nothing in this app updates by
itself, and a dashboard fed by a recorded extract that implied otherwise would be
the most misleading thing here.

### A five-minute demo path

1. **Command center** — the six figures, then the forecast against its baseline.
   Note the median door-to-doctor card turning red against the 30-minute target.
2. **Patient flow board** — press *Busiest hour*, then read the queue in priority
   order: an aged ESI 4 sits above a fresher one, and a new ESI 2 above both.
3. **Staffing planner** — the cheapest roster, 72 of 144 feasible, the circularity
   warning, and the caveat that the same search gave 7 different rosters on 10 other
   datasets.
4. **Scenario lab** — run "an extra doctor at peak" and read the verdict with its
   interval.
5. **Quality and equity** — where the model is weakest, including the high-acuity
   failure it reports about itself.

**Shift replay** sits on the Command center, the Patient flow board and the Wait and risk prediction page. Choose a date
and hour from the held-out period and the dashboard reconstructs what the service
would have known at that moment — the forecast it would have issued, and the queue
state computed only from patients who had already arrived — then shows what
actually happened afterwards, in a separate panel labelled as hindsight. Nothing
in this app is a live feed.

#### Taking screenshots

The theme is in `.streamlit/config.toml`. To capture both appearances:

```powershell
.\scripts\run_app.ps1
# light: the committed config is base = "light"
# dark:  edit .streamlit/config.toml -> base = "dark", reload the browser
```

Capture at 1280×800 for the laptop layout and at about 800px wide for the
tablet layout, where the KPI cards wrap. Use the browser's own device toolbar
rather than resizing the window, so the width is reproducible. The pages worth
capturing are Command center (light and dark), the Patient flow board at a busy
hour, and Data upload mid-run with the stepper visible.
`docs/dashboard_manual_test_checklist.md` is the list a human tester works
through, and it names the same views.

The parameter is `-BindHost`, not `-Host`, because `$Host` is a PowerShell automatic
variable. A bind address is not a URL -- nothing can open `http://0.0.0.0:8000` -- so
the banner prints `localhost` whatever the service is bound to, and warns when the
bind is wider than loopback.

`run_api.ps1` refuses to start, with the command to run, if
`artifacts\metrics\wait_models.json` is missing; `run_app.ps1` warns but still starts
when the API is down, so the artifact-backed pages remain readable.

The `/v1` routes are **read-only additions**: they compute nothing new, write
nothing, and exist so the dashboard asks the service for a figure instead of
re-deriving it. The unversioned routes are unchanged, and a contract test in
`tests/test_analytics_api.py` asserts their paths and response keys.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | liveness, which artifacts loaded, model and data version |
| `POST /predict-wait` | one patient's wait: point, interval, prolonged probability, top factors |
| `POST /predict-risk` | the chance one patient leaves without being seen, by 30 / 60 / 120 minutes |
| `GET /forecast?horizon_hours=N` | hourly arrivals with conformal intervals |
| `POST /simulate` | a staffing scenario's KPIs with confidence intervals |
| `GET /staffing-recommendation` | the precomputed cheapest feasible plan |
| `POST /jobs` | upload a CSV and run the whole pipeline on it |
| `GET /jobs/{id}` | status, current stage, per-stage state and timing |
| `GET /jobs/{id}/results` | headline metrics, skip reasons, file listing |
| `GET /jobs/{id}/report` | that run's report, as markdown |
| `GET /jobs/{id}/artifacts/{name}` | one table, figure or metrics file |
| `POST /jobs/{id}/cancel` | stop a queued or running job |
| `DELETE /jobs/{id}` | delete the run folder and the uploaded file |
| `GET /jobs/-/defaults` | the limits, the default roster, the privacy notice |
| `GET /v1/analytics/overview` | the Command center's figures and alerts, as at one moment |
| `GET /v1/analytics/flow-hourly` | arrivals and the reconstructed waiting-room census, hour by hour |
| `GET /v1/analytics/acuity-summary` | median and 90th-percentile wait, walkouts and share over target, per acuity |
| `GET /v1/board` | who was waiting at a moment, in priority order, paginated |
| `GET /v1/meta` | versions, the data window, row counts, the roster and every alert rule |

Configuration:

| Variable | Default | Used by |
| --- | --- | --- |
| `ED_API_HOST` | `127.0.0.1` | API bind address |
| `ED_API_PORT` | `8000` | API port |
| `ED_APP_HOST` | `127.0.0.1` | dashboard bind address |
| `ED_APP_PORT` | `8501` | dashboard port |
| `ED_API_URL` | `http://localhost:8000` | which API the dashboard calls |
| `ED_API_TIMEOUT` | `60` | dashboard |
| `ED_ARTIFACTS_DIR` | `artifacts` | dashboard quality page |
| `ED_PROCESSED_DIR` | `data/processed` | dashboard Shift replay |

`/simulate` runs a real discrete-event model, so replications are capped by
`serving.simulate.max_replications` in the config. When a request asks for more, the
response says it was capped rather than silently returning a narrower interval than the
caller expected.

### Upload a file and analyse it

The dashboard's **Data upload** page takes an ED visits CSV and runs the whole
pipeline on it with no manual commands: read and map the columns, validate, split,
build features, train the wait-time models, calibrate the intervals, explain, check
fairness, forecast, survival, competing risks, calibrate and validate the simulation,
run the scenarios, search for a staffing plan, write the report. Progress is live, results are tabbed, and
everything is downloadable.

The same thing from the command line, against the running API:

```powershell
.\scripts\smoke_upload.ps1                          # data\raw\ed_v2.csv, full depth
.\scripts\smoke_upload.ps1 -Depth quick -SkipSimulation
.\scripts\smoke_upload.ps1 -Path C:\data\extract.csv -Keep
```

**Columns are detected, never guessed.** Three steps in order: the schema's own name,
the same name ignoring case and punctuation (`Arrival Time` is `arrival_time`), then a
documented alias from `uploads.column_aliases` in the config (`ESI`, `acuity` and
`triage` all mean `triage_level`). Nothing after that -- no edit distance, no "closest
match". If two columns compete for one required field, or none matches, the run stops
and the page asks, with a preview of the first rows. A wrong guess on a required column
is not a visible error; it is a plausible model trained on the wrong field.

Also handled: comma, semicolon, tab and pipe separators; a UTF-8 byte-order mark
(Excel writes one); numeric acuity (`1`, `1.0`, `"1"` alike); text acuity via the
configured label map only; disposition codes through their own map, with every
unrecognised value counted and shown; and extracts that store durations instead of
`seen_time`/`departure_time`. A value map that matches under 5% of rows is a loud
warning, because that failure looks like bad data and is actually a bad mapping.

**The simulation needs your roster.** Doctors are the only hard capacity constraint, so
every wait, scenario and staffing figure depends on the roster you enter. The form
starts from the config's roster and says plainly that it describes the demo dataset;
every staffing result is labelled with the roster it assumed. If you do not have one,
tick **Skip the simulation**.

**Stages degrade rather than fail.** Too few days to split by time, under about 60 days
(no forecast), too few patients who left without being seen (no survival), missing
optional columns (fairness reports on what exists) -- each produces a SKIPPED stage with
a reason in plain English rather than a failed job. A simulation that fails validation
against your data causes the scenarios and the staffing search to be **refused**, with
the validation table shown; there is no override in the UI, because a staffing
recommendation from a model that cannot reproduce your department looks exactly as
confident as a correct one.

**Scoring-only mode** skips training entirely and runs the baseline models over your
data instead, reporting error, fairness and drift. The models were fitted elsewhere, so
the drift table is what tells you whether the numbers transfer.

Limits, all in `uploads.limits` in the config:

| Limit | Default |
| --- | --- |
| Upload size | 50 MB |
| Rows | 500,000 |
| Jobs queued behind the running one | 4 |
| Per-stage timeout | 30 minutes |
| Total job timeout | 90 minutes |
| Run folders kept on disk | 10 (oldest deleted) |
| Retention before automatic deletion | 24 hours |

**Privacy.** Upload only synthetic or fully de-identified data; the page says so before
the file picker. The file is written to `runs/<job_id>/` and nowhere else, is never
copied and never logged -- logs carry counts, not rows -- and the whole folder is
deleted when you press **Delete now**, and automatically after the retention window.
`runs/` is git-ignored. Model files are never served for download: a joblib runs code
when it is loaded.

**Isolation.** A job rewrites every path in its config into its own run folder, so
nothing it does can touch the baseline `artifacts/` the serving endpoints loaded at
startup. A test asserts this by hashing every baseline artifact before and after a real
run, and another checks that `/predict-wait` answers identically either side of an
upload.

One job runs at a time, in a **separate worker process**. That is what makes Cancel and
the timeouts real: a Python thread cannot be interrupted inside LightGBM's fit or a
SimPy loop, but a process can be terminated.

#### Troubleshooting

| What you see | What it means |
| --- | --- |
| "Cannot reach the API at ..." | The API is not running. Start `.\scripts\run_api.ps1` in another terminal. |
| The mapping form appears | A required column was ambiguous or missing. Pick the right column; the preview shows what is in each. |
| "no column could be matched to ..." | That required field is not in your file under any known name. Rename it, or choose it in the form. |
| "Read as semicolon-separated" | The separator was detected. If that is wrong, the preview will look wrong too. |
| Forecast tab says "needs about 60 days" | The extract is too short for weekly seasonality plus an honest backtest. |
| Scenarios "refused" | The simulation could not reproduce your data. Read the validation table; usually the roster is wrong. |
| "the job passed its ... limit" | Try **Quick** analysis, or a shorter date range. |
| "That file is N MB, over the 50 MB limit" | Upload a shorter date range, or drop columns you do not need. |
| 429 / "the queue is full" | Another analysis is queued. Wait, or cancel one. |
| Job vanished | Retention deleted it. Runs are kept 24 hours by default. |

### Docker

`docker-compose.yml` runs both services with `artifacts/` mounted **read-only** — the
pipeline writes them, the services only read.

```powershell
docker compose build
docker compose up -d
docker compose ps          # both should report healthy
docker compose logs -f api
docker compose down
```

> **Not verified on this machine.** The Docker CLI is present (29.7.2) but the daemon is
> not running, so `docker compose config` was used to validate the file (it exits 0 and
> resolves both services) and **the images were never built or run**. Treat the Docker
> path as unverified until someone runs it with a live daemon. The PowerShell scripts
> above are the supported path here and have been run end to end.

### Run with Docker

Two containers: the API (which also runs upload jobs) and the dashboard. Local
use on one machine, for a demo. No cloud, no registry, no Kubernetes, and still
no authentication — the ports are published on `127.0.0.1` only.

**Prerequisites.** Docker Desktop for Windows with the **WSL 2** backend enabled
(Settings → General → *Use the WSL 2 based engine*). Check it with
`wsl --status` and `docker info`. Work from a local path such as
`C:\dev\ed-project`: the scripts refuse to run from a OneDrive or Dropbox copy,
because a sync client rewrites files underneath the containers and Docker cannot
dependably bind-mount cloud placeholder files.

**Quick start** — three commands:

```powershell
Copy-Item .env.example .env         # optional: every value has a safe default
.\scripts\docker_train.ps1          # produce artifacts\ (a few minutes)
.\scripts\docker_up.ps1 -Open       # build, start, wait for health, open the browser
```

Or double-click `start_docker.bat`, which runs `docker_up.ps1` for you.

`docker_up.ps1` checks the engine, checks that 8000 and 8501 are free, builds,
starts, waits for *both* healthchecks, and only then prints the URLs — so a URL
it prints is one that answers:

| | |
|---|---|
| Dashboard | <http://localhost:8501> |
| API docs | <http://localhost:8000/docs> |
| API health | <http://localhost:8000/health> |

The other scripts:

```powershell
.\scripts\docker_status.ps1              # containers, health, model version, runs on disk
.\scripts\docker_logs.ps1                # follow both; -Service api, -NoFollow
.\scripts\docker_train.ps1 -RestartServices   # retrain, then reload the models
.\scripts\docker_down.ps1                # stop; -RemoveData also drops the runs volume
.\scripts\docker_clean.ps1               # remove THIS project's containers/images/volume
```

`docker_clean.ps1` names the four objects this project created and removes only
those, after a prompt. It never runs `docker system prune`, which would delete
other people's images and volumes.

#### Where data and artifacts live

| Path | In the container | Mode | Why |
|---|---|---|---|
| `.\artifacts` | `/app/artifacts` | **read-only** for api and app | the pipeline writes them, the services only read them |
| `.\data\processed` | `/app/data/processed` | read-only | Shift replay reads the validated visits |
| `.\configs` | `/app/configs` | read-only | one config, not a copy baked into an image |
| `.\data\raw` | `/app/data/raw` | read-only, **trainer only** | the extract is an input; nothing may modify it |
| `ed-intelligence-runs` volume | `/app/runs` | read-write | uploaded files and their run folders |

Nothing is baked into an image: no models, no extract, no `.env`. The images
carry code only, so a retrain does not need a rebuild. Uploaded files live in the
named volume and nowhere else — `docker compose down -v`, or
`.\scripts\docker_down.ps1 -RemoveData`, destroys them. The root filesystem of
both containers is read-only, and each runs as a non-root user with a fixed uid
(10001 for the API, 10002 for the dashboard).

#### Training inside Docker

```powershell
.\scripts\docker_train.ps1                  # python -m src.pipeline run --data real
docker compose --profile train run --rm trainer    # the same thing, by hand
```

The `trainer` service is in the `train` profile, so `docker compose up` never
starts it. It is the only container with write access to `.\artifacts`, and it
exits when the pipeline finishes. The API loads artifacts once, at startup, so
pass `-RestartServices` (or run `docker_up.ps1 -NoBuild`) to serve what was just
trained.

If artifacts are missing the stack still starts and says so, rather than
crash-looping: `/health` returns `models_loaded: false` with a `how_to_fix`
sentence, every prediction endpoint answers **503** with the training command in
the message, and the dashboard shows a card carrying the same command.

#### Exposing the API deliberately

Both ports are published on loopback. To reach the API from another machine, set
the publish address and understand what you are doing:

```powershell
# .env
ED_API_PUBLISH_HOST=0.0.0.0
```

**There is no TLS and no rate limiting**, and `POST /jobs` accepts a file upload
that runs the whole pipeline. Anything that can reach the port can submit work and
read every previous job's results. Put a reverse proxy with authentication in front
of it, or keep it on loopback and use SSH port forwarding.

##### Optional API key

There is one thin lock available. Set `ED_API_KEY` and every endpoint except
`/health` and the documentation requires a matching `X-API-Key` header:

```powershell
# .env  -- both containers read this one value
ED_API_KEY=choose-a-long-random-string
```

```bash
curl -H "X-API-Key: choose-a-long-random-string" http://127.0.0.1:8000/forecast
```

The dashboard reads the same variable and sends the header for you, so nothing else
needs configuring. `GET /health` reports `authentication_required` so you can tell
the mode without reading the server's environment, and the key itself is never
logged or echoed in an error.

Leave it unset and the service behaves exactly as it always has — that is still the
default.

What this is: enough that something which finds the port cannot simply query the
service. What it is **not**: an access-control system. There is one key for all
callers, no rotation, no revocation, no per-caller audit and no rate limiting, and
without TLS the header crosses the network in clear text. A leaked key is a full
compromise until the value is changed. It does not replace the reverse proxy above.

#### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `error during connect ... docker_engine` / "engine not reachable" | Docker Desktop is not running | Start Docker Desktop, wait for the whale icon to settle. If it never does: `wsl --status`, then Troubleshoot → Restart |
| `Port 8000 is already in use` | a non-Docker `run_api.ps1` is running, or another app | Close that window, or set `ED_API_PORT` / `ED_APP_PORT` in `.env` |
| API container never becomes healthy | `artifacts\` is empty or not shared | `docker compose exec api ls /app/artifacts/metrics`; if empty, run `docker_train.ps1`; check Docker Desktop → Resources → File sharing includes the drive |
| First build takes minutes | it pulls `python:3.12-slim` and installs the wheels | Expected once. Later builds reuse the dependency layer unless `requirements-*.txt` changed |
| Mounted `artifacts\` looks empty, or files are half-written | the repository is on OneDrive/Dropbox | Move it to a local path; the scripts refuse to run from a synced folder |
| `exec /usr/local/bin/...: no such file or directory` | CRLF line endings reached a file Linux executes | `.gitattributes` pins LF for `*.sh`, `Dockerfile`, `*.yml`, `*.yaml`, `requirements*.txt`; re-checkout with `git add --renormalize .` |
| `no space left on device`, or a build that dies late | Docker's disk is full | `docker system df`, then `.\scripts\docker_clean.ps1` and `docker builder prune` |
| Docker Desktop will not start, or `libgomp` errors | antivirus or Windows Application Control blocking the engine or a compiled library | Ask IT to allow Docker Desktop and WSL 2; the non-Docker path (`run_api.ps1`) is the fallback |
| Dashboard pages are empty but the API is healthy | the dashboard is pointed at the wrong API | Inside the network it must be `ED_API_BASE_URL=http://api:8000`, never `localhost` |

#### What is different from the non-Docker run

Measured, not assumed — on this machine, the same pipeline run on the host and in
the `trainer` container produced **no reported metric that differs**: the largest
difference anywhere in the metrics JSON was `2.3e-13` relative (a Cox p-value),
and nothing differs at six decimal places. Two consecutive `trainer` runs were
byte-identical across all twelve metrics files.

What *does* differ:

- **Line endings in the artifact JSON** — LF from the container, CRLF from a
  Windows host run. Content-identical; it only matters if you hash the files.
- **Recorded paths** inside `fairness.json` and `shap.json` are the figure paths
  as the writer saw them (`/app/artifacts/...` versus `C:\...`).
- **Timings** (`features.json` records `seconds` and `rows_per_second`).
- **Line-level floating point** in `sim_calibration.json` and `survival.json`,
  at the 13th significant figure — a different libm/BLAS, below every rounding
  the report applies.
- **Where uploaded files go**: a container volume rather than `.\runs`, so
  `docker compose down -v` is how you delete them.
- **Reload**: the host scripts can run uvicorn with `--reload`; the demo stack
  does not. Use `docker-compose.dev.yml` for that.

### Dependencies

`requirements.txt` is the full stack, for running the pipeline and the tests.
`requirements-serving.txt` is the runtime subset the API and dashboard need: it drops
`pytest`, `matplotlib` and `lifelines`, and keeps `lightgbm` (the saved models are
LightGBM boosters, and explanations use its native TreeSHAP) and `simpy` (`/simulate`
runs the real model).

The Docker images split that subset again, because the two services need
different things:

| File | Image | Holds |
|---|---|---|
| `requirements-api.txt` | `ed-intelligence-api` (1.05 GB) | the **full** stack minus `pytest`: an upload job runs the whole pipeline inside this container, so `lightgbm`, `lifelines`, `simpy` and `matplotlib` all have to be present |
| `requirements-app.txt` | `ed-intelligence-app` (753 MB) | `streamlit`, `requests`, `pandas`, `numpy`, `pyarrow` and nothing else — the dashboard fits no model and reads no `.joblib` |

Every pin is identical across the three files. `libgomp1` is the one OS package
either image installs: LightGBM links against it and `import lightgbm` fails on
`python:3.12-slim` without it. Neither image contains a compiler — every pinned
wheel installs without one, which is checked by a test and by the build itself.

### Why there is no `shap` dependency

`shap` pulls in `numba`, whose compiled DLL is blocked by Windows Application Control on
some managed machines, making the package unimportable. It is not needed: **LightGBM
implements TreeSHAP natively**, and `model.predict(X, pred_contrib=True)` returns the
same exact Shapley values that `shap.TreeExplainer` would -- the `shap` package calls
into this very routine for LightGBM models. The plots are drawn with matplotlib and the
output files and table names are unchanged. A test asserts the defining property:
contributions plus the base value sum exactly to the raw prediction.

### If one library is unusable

Heavy libraries are imported by the stage that needs them, not at module import time, so
`python -m src.pipeline --help` and `list-stages` still work when `lightgbm`, `simpy`,
`lifelines` or `matplotlib` is missing or blocked. One unusable library disables one
stage rather than the whole CLI.

## How to run

```bash
# whole pipeline on generated data
python -m src.pipeline run --config configs/default.yaml --data synthetic

# whole pipeline on your own extract (see "Adding a real dataset")
python -m src.pipeline run --config configs/default.yaml --data real

# a subset of stages -- each reads its inputs from disk, so any stage runs alone
python -m src.pipeline run --stages load,validate,split
python -m src.pipeline run --stages forecast
python -m src.pipeline list-stages
```

Every stage is deterministic given `seed` in the config, reads its inputs from disk and
writes its outputs to disk, so re-running a stage overwrites its outputs with identical
content. Intermediate tables are Parquet, which round-trips dtypes; result tables are
CSV for readability.

## Project structure

```
ed-project/
├── CLAUDE.md                 working rules: leakage, baselines, seeds, scope
├── README.md
├── SUBMISSION.md             what to run first, the results, and what they are not
├── Start-Demo.cmd            double-click: starts everything and opens the browser
├── Stop-Demo.cmd             double-click: stops only what Start-Demo started
├── start_docker.bat          double-click: the containerised demo
├── requirements.txt          pinned, Python 3.11
├── Makefile                  setup / test / synthetic / run / clean
├── pytest.ini
├── schema.yaml               the data contract -- the only source of column truth
├── configs/
│   ├── default.yaml          every path, seed, threshold and hyperparameter
│   ├── column_map.yaml       real-dataset column -> schema column mapping
│   └── column_map_mimic_ed.yaml   MIMIC-IV-ED template, unverified by design
├── data/{raw,processed,synthetic}/
├── src/
│   ├── notices.py   shared constants, imports nothing -- breaks an import cycle
│   ├── data/        schema.py, synthetic.py, adapter.py, split.py, shift_check.py
│   ├── features/    calendar.py, queue_state.py, build.py
│   ├── models/      wait_time.py, baselines.py, conformal.py, rolling_conformal.py,
│   │                forecast.py, forecast_study.py, survival.py,
│   │                competing_risks.py, high_acuity.py, triage_thresholds.py,
│   │                train_wait_stage.py, figures.py, forecast_figures.py
│   ├── explain/     shap_report.py, fairness.py
│   ├── simulation/  des.py, calibrate.py, scenarios.py, sensitivity.py,
│   │                staffing_search.py, scenario_lab.py, staffing_rl.py,
│   │                sim_figures.py
│   ├── serving/     api.py -- pure functions over the saved artifacts
│   ├── jobs/        the upload workflow: settings.py, mapping.py, roster.py,
│   │                preconditions.py, storage.py, models.py, runner.py,
│   │                engine.py, worker.py, analysis.py, results.py
│   ├── utils/       io.py, logging.py, seed.py, metrics.py, priority.py, report.py
│   ├── pipeline.py  CLI orchestrator and stage registry
│   └── report.py    assembles artifacts/report.md
├── api/             main.py, schemas.py, jobs.py, job_schemas.py, version.py
├── app/             the dashboard
│   ├── dashboard.py   entry script: set_page_config, sidebar, dispatch
│   ├── config.py      constants, imports no Streamlit at all
│   ├── theme.py       palette, spacing, components, number formats
│   ├── api_client.py  the one HTTP client
│   ├── artifacts.py   reads the pipeline's saved tables
│   ├── replay.py      leakage-safe historical reconstruction
│   ├── upload_page.py the guided upload flow
│   └── views/         one module per page, each exposing render()
├── scripts/         PowerShell helpers, the research drivers, and
│   └── original_generator/generate_ed_data.py   the true data generator
├── docs/            real_data_readiness.md, demo_day_checklist.md,
│                    container_scan.md, two manual-test checklists
├── notebooks/
├── runs/            uploaded analyses, seed studies and demo logs; git-ignored
├── tests/           1418 tests
└── artifacts/       models/ metrics/ figures/ tables/ checkpoints/ report.md
```

## The data contract

`schema.yaml` is the single source of truth. Nothing outside `src/data/` hard-codes a
column name.

**Required:** `patient_id`, `arrival_time`, `triage_level` (ESI 1-5), `seen_time`
(nullable), `departure_time` (nullable), `disposition`.

**Optional, used when present:** `age`, `sex`, `arrival_mode`, `chief_complaint`. Drop
any of them and the pipeline still runs.

**Derived by the loader:** `wait_minutes`, `los_minutes`, `is_lwbs`,
`still_waiting_at_cutoff`.

A missing *required column* is a hard error. A bad *row* is dropped or repaired, never
silently: every drop is attributed to one reason and counted in
`artifacts/metrics/validation.json`.

**The two reasons a patient has no wait are kept strictly apart.** One who gave up and
left is an *event*; one still waiting when the extract closed is *censored*. Pooling
them biases every time-to-event estimate, and the survival module quantifies by how
much.

## Adding a real dataset

All dataset-specific knowledge lives in `configs/column_map.yaml` and
`src/data/adapter.py`.

1. **Put the extract where the config expects it** — default `data/raw/ed_visits.csv`.
   CSV, TSV and gzipped text are supported (no Excel, to avoid an extra dependency).

2. **Map each schema column to yours:**

   ```yaml
   columns:
     patient_id:   {source: YourVisitIdColumn}
     arrival_time: {source: YourArrivalTimestamp, datetime_format: "%Y-%m-%d %H:%M:%S"}
   ```

   Leave `datetime_format: null` to infer it; set it explicitly to avoid ambiguous
   day/month parsing.

3. **Recode coded values with `value_map`**, and give a `default`. Matching ignores case
   and surrounding whitespace:

   ```yaml
     triage_level:
       source: AcuityText
       value_map: {Resuscitation: 1, Emergent: 2, Urgent: 3, "Less Urgent": 4, "Non-Urgent": 5}
     disposition:
       source: OutcomeCode
       value_map: {Home: discharged, Inpatient: admitted, LWBS: lwbs}
       default: other
   ```

4. **Delete the block for any optional column you do not have.**

5. **If your extract stores durations rather than timestamps**, enable the fallback and
   the adapter reconstructs them from `arrival_time`:

   ```yaml
   duration_fallback:
     enabled: true
     wait_minutes_source: DoorToDoc_Min
     los_minutes_source: LOS_Min
     units: minutes        # or hours
   ```

6. **Run load and validate first, and read the report:**

   ```bash
   python -m src.pipeline run --data real --stages load,validate
   cat artifacts/metrics/validation.json
   ```

   A large drop count almost always means a mapping or a date format is wrong, not that
   your data is bad.

7. **Consider `split.mode: dates`** so results stay comparable across data refreshes.

Real extracts are never committed; `data/raw/` is git-ignored apart from its `.gitkeep`.

## Serving

`src/serving/` exposes four pure functions over the saved artifacts. They load models
from `artifacts/models/`, retrain nothing, and hold no global state, so a FastAPI layer
can wrap them without refactoring.

```python
from src.serving import load_bundle, predict_wait, PatientRecord, QueueState

bundle = load_bundle()
prediction = predict_wait(
    PatientRecord(arrival_time=..., triage_level=3, age=54),
    QueueState(n_waiting=12, n_in_treatment=7, mean_wait_last_k_seen=38.0),
    bundle,
)
# -> point, lower, upper, prolonged_prob, top_shap_features, caveats
```

Also `forecast_arrivals(horizon, as_of)`, `run_scenario(name, overrides)` and
`recommend_staffing(target)`. Queue state is an explicit input because only the live
system knows it. Every response carries the caveats needed to read it, including the
high-acuity false-negative warning from the fairness analysis.

## Tests

```bash
make test            # 309 tests
make test-fast       # skips the few with a multi-second budget
```

Notable guarantees the suite enforces:

- **Leakage.** Four classes of leak are tested: reading a later patient, reading a later
  patient's outcome, depending on later rows existing, and reading *this* patient's own
  outcome. The last was added after a deliberately introduced leak passed all the
  others — see `artifacts/checkpoints/phase_1a_leakcheck.md`.
- **No random splits.** Nine forbidden constructs are grepped out of `src/`, and the
  splitter rejects any mode but `fraction` and `dates`.
- **Config completeness.** No hard-coded seeds or thresholds in `src/`.
- **Forecast honesty.** A forecast built from the full series must be bit-identical to
  one built from a series truncated at `as_of`.
- **Simulation sanity.** More servers reduce waits; doubling arrivals increases them; a
  fixed seed reproduces exactly; the DES reproduces the generator's KPIs; the staffing
  search never returns an infeasible plan as "best".

## External validation

Two real datasets were obtained to test this project against data it did not
generate. Neither ships in `dist\ed-project-submission.zip`; both are one command
to fetch, and `data/external/README.md` records source, licence, retrieval date and
checksums.

### Real hourly ED arrivals (forecasting)

41,640 hourly arrival counts from the University of Iowa Hospitals and Clinics
emergency department, 2013-2018, from the R package **fastTS** (GPL >= 3). The
arrival forecaster was run on it **unchanged** -- same model class, same lag set
including the 168-hour seasonal lag, same rolling-origin design, same metrics, all
read from `configs/default.yaml`.

Over 26 rolling origins and 1,872 scored hours:

| | MAE | RMSE | sMAPE | WAPE |
|---|---|---|---|---|
| **This project's forecaster** | **2.025** | **2.611** | **37.7%** | **31.5%** |
| Seasonal-naive (168 h) | 2.751 | 3.602 | 51.4% | 42.9% |
| Mean of same hour, previous 4 weeks | 2.215 | 2.875 | 40.9% | 34.5% |

It wins at peak (3.04 against 3.95 and 3.24) and off peak (1.76 against 2.44 and
1.95), and its intervals cover at **0.898** against a 0.90 target -- better
calibrated on the real series than on the synthetic one, which under-covers at
0.8875. The gap to seasonal-naive is **26.4%** on real data against 20.9% on
synthetic, so the forecasting method transfers.

The run also found a defect: the project's nine configured holidays are all 2024
dates, so on 2013-2018 data `is_holiday` is constant and useless. **The model beat
both baselines with that feature accidentally disabled.** Same-hour temperature was
*not* used -- it is not known when a forecast is issued -- and a lagged version
correlates 0.31 with arrivals but only -0.03 with the model's residuals.

Full write-up: `artifacts/external_validation/uihc/report.md`.

### MIMIC-IV-ED demo (adapter check only)

The openly published demo subset, 222 ED stays, ODbL with attribution. **No model was
fitted on it and no model performance is reported from it**: 222 stays is far too
few, and a figure from it would be quoted later as though it were not.

What it did establish:

- The column map written from MIMIC's published schema, without any data in hand,
  maps the **real** columns. The adapter read all 222 stays through its own
  `adapt_real_dataset`.
- **One mapping was wrong and is now corrected.** `LEFT AGAINST MEDICAL ADVICE` was
  mapped to left-without-being-seen. Leaving against advice means a clinician saw and
  advised the patient, so it would have inflated the abandonment rate -- one of the
  quantities the simulation's validation gate is judged on.
- The two wait proxies bracket the truth from opposite sides, now measured rather
  than argued: first vitals median **2.0 min** (recorded at triage, so essentially
  the arrival time), first medication dispense median **118 min**, the dispense later
  in **80%** of stays by a median of 76 minutes. Every such figure is labelled
  **proxy wait (not true door-to-provider)**.
- **The queue state cannot be reconstructed from MIMIC-IV-ED, with or without
  credentials.** Dates are shifted per patient, which the demo makes visible: 222
  stays spanning 32,548 days at 0.01 stays per day. So the queue-state features --
  the ones that make the wait model beat its baseline threefold -- and the
  simulation's validation gate cannot be checked on it at all. Credentialed access
  would fix the sample size and nothing else on that list.

Full write-up: `artifacts/external_validation/mimic_demo/report.md`.

### What this does not claim

**No real-hospital performance is claimed for anything but the arrival forecast.** No
patient-level model touched real data: the UIHC series has no patients in it, and the
MIMIC demo was profiled rather than modelled. The wait model, the conformal
intervals on waits, the simulation and the staffing search remain validated only
against synthetic data.

### External data sources

- **`uihc_ed_arrivals`** -- University of Iowa Hospitals and Clinics ED hourly
  arrivals, from the R package **fastTS**, licensed **GPL (>= 3)**.
  Peterson RA. *fastTS: Fast Time Series Modeling for Seasonal Series with Exogenous
  Variables.* R package version 1.0.3. <https://cran.r-project.org/package=fastTS>
- **MIMIC-IV-ED demo v2.2** -- licensed **ODbL v1.0**, attribution required.
  Johnson A, Bulgarelli L, Pollard T, Celi LA, Mark R, Horng S. *MIMIC-IV-ED Demo*
  (version 2.2). PhysioNet (2023). <https://doi.org/10.13026/jd97-se30>.
  Also cite Johnson AEW et al., *MIMIC-IV, a freely accessible electronic health
  record dataset*, Scientific Data 10, 1 (2023), and Goldberger AL et al.,
  *PhysioBank, PhysioToolkit, and PhysioNet*, Circulation 101(23):e215-e220 (2000).

Neither dataset is redistributed in this project's package. `data/external/README.md`
records how each was obtained, with checksums.

## Known limitations

- **Synthetic data.** Every number comes from data generated by a queue whose structure
  the models are then asked to recover. That validates the methods; it is not evidence
  about any real department.
- **The ablation overstates queue features.** The generator creates waits from a queue,
  so congestion is close to a sufficient statistic. Direction supported, magnitude an
  upper bound.
- **Censoring bias.** Wait models train only on patients who were seen, who waited less
  than those who gave up. The bias is downward and at least 7.8%.
- **The classifier fails high-acuity patients.** ESI 2 has a false-negative rate of
  1.000: a single global threshold optimising pooled recall buys it from the low-acuity
  majority. Detected and reported, not yet mitigated.
- **The treatment/boarding split is not identifiable** from arrival, contact and
  departure timestamps. It is fitted to one aggregate.
- **Nurses, beds, imaging and lab turnaround are not calibrated.** Doctors are the only
  hard constraint in the validated model. They can be switched on as explicit resources
  behind `simulation.resource_extensions`, off by default, but the data has no bed,
  nurse or imaging column, so those parameters are assumed and the validation gate
  cannot test them. Read the ranking of levers, not the minutes.
- **The validation gate has a known blind spot.** It compares marginal KPIs — per-acuity
  mean and p90 wait, length of stay, abandonment — so a permutation of waits *within*
  each acuity preserves every one of them exactly and passes. A dataset whose waits are
  completely independent of the queue would clear the gate.
  `test_within_acuity_decoupling_slips_past_the_gate` pins this and is written to fail
  loudly if the gate is ever strengthened.
- **The gate refused a staffing plan on 2 of 10 regenerated datasets.** It passed on
  8 of 10. Seed 8 was a rare-event tolerance artefact — ESI 3 abandonment missed a 1.6%
  base rate by 1.2 percentage points. Seed 10 was a genuine 8-minute miss on ESI 3 mean
  wait. No tolerance was loosened to make either pass.
- **The cheapest roster is dataset-specific.** Across the 8 seeds that produced a plan
  the cost spread 43,890 to 54,670 over 7 distinct rosters, with the day shift at 3 in
  every one. The method transfers; 3/3/2/2 at 49,280 is the reference dataset's answer,
  not a recommendation.
- **Conformal coverage assumes exchangeability**, which time-ordered blocks only
  approximate. Wait-time intervals under-cover by about 2 points for this reason.
- **Forecast horizon is capped** by the 168-hour seasonal lag; the staffing plan is
  sized for the forecast window, not the year.

## Not built

Deliberately out of scope, named here rather than hidden:

- **Real authentication, TLS and rate limiting on the API.** There is now an
  optional shared-secret key (`ED_API_KEY`, see "Optional API key" above), and that
  is all it is: one key for everyone, no rotation, no revocation, no per-caller
  audit, no rate limiting and no TLS. It is why both services still bind to
  `127.0.0.1` by default and why the upload workflow caps file size, rows, queue
  depth and run time. Do not expose this to a network that matters without putting
  something in front of it.
- **A durable job store.** Jobs live in the API process's memory plus their run
  folders, so a restart forgets the list (the folders, and their reports, survive).
  Fine for one analyst on one machine; not a multi-user service.
- **Per-user separation.** One job runs at a time and every job is visible to anyone
  who can reach the service. There is no notion of who uploaded what.
- **Deep forecasters.** No LSTM, Transformer or TFT. LightGBM, linear models and simple
  statistics only, which is enough for hourly counts with a 168-hour seasonal lag.
- **A reinforcement-learning staffing policy that works.** One was built and it is a
  **negative result**, reported rather than tuned away. On the pre-declared 60-episode
  budget the learned policy cost **+17,832 more** than the fixed roster (95% CI 14,663
  to 21,002) and produced **more ESI 1-2 breaches than acting at random**. The likely
  cause is the budget — roughly five visits per table cell, far short of convergence —
  but that is a hypothesis, not a defence, and it was not retrained until it won. It is
  also circular: the agent learns a simulator calibrated to a generator, so a win would
  be evidence about the method and nothing else.
- **An objective for scarce high-acuity positives.** The per-acuity threshold
  mitigation *is* built (`src/models/triage_thresholds.py`): thresholds are fitted for
  the levels with enough positive cases, and `artifacts/tables/triage_thresholds.csv`
  scores them against the pooled rule. It changes pooled recall by less than its own
  confidence interval, and ESI 1-2 have too few positives to fit at all. What is not
  built is the thing that would actually help there: an acuity-aware training
  objective, or more high-acuity positive cases.
#   E D - I n t e l l i g e n c e .  
 