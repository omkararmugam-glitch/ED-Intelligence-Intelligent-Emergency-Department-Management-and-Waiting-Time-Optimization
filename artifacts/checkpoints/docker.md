# Containerisation — verification record

Written for: whoever has to trust, repeat or debug this containerisation.

Every command below was run on the machine named in *Environment*, and every
output is pasted from that run. Where something was **not** verified it says so
in *Not verified* at the end rather than being left to inference.

## Environment

| | |
|---|---|
| Host | Windows 11 Home Single Language 10.0.26300, PowerShell 5.1 |
| Docker | client and engine 29.7.2, Docker Desktop 4.90.0 (238679), WSL 2 backend |
| Docker VM | 32 CPUs, 7.9 GB (`docker info`: `NCPU 32`, `MemTotal 7903875072`) |
| Project path | `C:\dev\ed-project` (local disk, deliberately not the OneDrive copy) |
| Base image | `python:3.12-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f` |
| Python in image | 3.12.15 (GCC 14.2.0) |

## Files

| File | What it is |
|---|---|
| `Dockerfile` | One file, two targets: `api` and `app`, each with its own dependency stage |
| `docker-compose.yml` | The demo stack: `api`, `app`, and `trainer` behind the `train` profile |
| `docker-compose.dev.yml` | Development overlay (bind mounts, reload). Explicitly named — see note below |
| `requirements-api.txt` | Full stack minus pytest: upload jobs run the whole pipeline in the api container |
| `requirements-app.txt` | streamlit, requests, pandas, numpy, pyarrow. Nothing else |
| `.dockerignore` | Excludes `.venv/`, `runs/`, `data/`, `artifacts/`, `tests/`, `.env`, caches, docs |
| `.gitattributes` | LF for everything a Linux container reads; CRLF for `.ps1`/`.bat` |
| `.env.example` | Every variable, with the safe default. `.env` is gitignored |
| `scripts\_docker_common.ps1` | Shared: project root, synced-folder refusal, engine check, port checks, health wait |
| `scripts\docker_{up,down,logs,status,train,clean}.ps1` | The six scripts |
| `start_docker.bat` | Double-click entry point (CRLF, verified with `file`) |
| `tests/test_docker.py` | 48 tests over the compose file, the Dockerfile, `.dockerignore`, `.gitattributes`, `.env.example` |

**Why `docker-compose.dev.yml` and not `docker-compose.override.yml`.** Compose
loads an override file automatically and silently. Shipping the development
overlay under that name would mean every `docker compose up` — including the one
on demo day — ran with bind-mounted code and auto-reload. The overlay therefore
has its own name and must be asked for with `-f`. `.gitignore` excludes
`docker-compose.override.yml` so a developer who wants the automatic behaviour
can copy it locally, and `tests/test_docker.py` asserts the file is absent.

## 1. Build

```
> docker compose config --quiet
(exit 0, no output)

> docker compose build --no-cache          # both targets, base image already pulled
real    1m4.274s

> docker compose build                     # warm cache
real    0m1.451s
```

Image sizes and layer counts:

```
> docker image ls --format '{{.Repository}}:{{.Tag}}|{{.Size}}'
ed-intelligence-api:latest|1.05GB
ed-intelligence-app:latest|753MB

> docker image inspect <image> --format '{{len .RootFS.Layers}}'
api: 12 layers      app: 9 layers       base python:3.12-slim: 43 MB
```

The api image is large because an upload job trains models inside it: lightgbm,
scikit-learn, lifelines, matplotlib, pyarrow and simpy are all required. The app
image carries none of them.

No compiler in either image, and none needed — every pinned wheel installed on
`python:3.12-slim` without `build-essential`:

```
> docker run --rm ed-intelligence-api:latest python -c "import lightgbm, lifelines, simpy, matplotlib, fastapi, sklearn, pyarrow; print('imports ok', lightgbm.__version__, lifelines.__version__)"
imports ok 4.5.0 0.29.0
```

`libgomp1` is the one OS package installed, because `import lightgbm` fails on
slim without OpenMP.

## 2. Start and health

```
> docker compose up -d
 Container ed-api Started / Waiting / Healthy
 Container ed-dashboard Started

> docker compose ps
NAME           STATUS                    PORTS
ed-api         Up 11 seconds (healthy)   127.0.0.1:8000->8000/tcp
ed-dashboard   Up 5 seconds (healthy)    127.0.0.1:8501->8501/tcp
```

Note the published addresses: `127.0.0.1:8000->8000/tcp`, not `0.0.0.0:8000`.

```
> curl -s http://127.0.0.1:8000/health
{"status":"ok","models_loaded":true,"forecaster_loaded":true,"simulation_calibrated":true,
 "model_version":"base+queue-raw-b098128b","data_version":"real-57179rows-20261008",
 "uptime_seconds":9.3,"how_to_fix":null}

> curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/docs       -> 200
> curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8501/_stcore/health -> 200
```

`GET /forecast?horizon_hours=24`:

```
points: 24
first:  {"timestamp": "2024-12-30T23:00:00", "yhat": 5.5353, "lower": 0.8205, "upper": 10.2501}
model:  base+queue-raw-b098128b
```

`POST /predict-wait` **with** the optional queue fields:

```
point_estimate_minutes 101.91   lower 73.93   upper 129.88
prolonged_probability  0.9863   prolonged_flag true
top factors: n_waiting_same_priority +33.7, n_waiting +25.2, n_waiting_higher_priority +5.5
```

`POST /predict-wait` with the optional queue fields **omitted** (only
`n_waiting` given):

```
point 30.51   interval 2.53 - 58.48
caveat: No recent-wait history was supplied, so that feature was missing; the
        estimate is less reliable than it would be with it
```

`queue_state` itself is required by the schema; the *fields inside it* are
optional, and the response names each one it had to do without.

### The dashboard, driven through the container network

Rendered with Streamlit's `AppTest` **inside the dashboard image**, on the
compose network, with `ED_API_BASE_URL=http://api:8000` and the read-only
artifacts mount:

```
dashboard resolved API_URL = http://api:8000
  ok    Command center  (23792 chars of text)
  ok    Demand forecast  (20714 chars)
  ok    Wait estimator  (19809 chars)
  ok    What-if scenarios  (21417 chars)
  ok    Staffing planner  (27049 chars)
  ok    Data and analysis  (20955 chars)
  ok    Model quality and fairness  (32199 chars)
  ok    About and limitations  (35717 chars)
failures: 0
```

All eight pages render with no exception. The same sweep run from the host
against the published port resolved `http://127.0.0.1:8000` and also passed,
which is what shows the base URL is taken from the environment rather than
hardcoded.

## 3. Upload, end to end, through the containerised API

```
> curl -s -X POST http://127.0.0.1:8000/jobs -F "file=@data/raw/ed_v2.csv" -F "depth=full"
{"job_id":"c38225bfeb3b","status":"queued","needs_mapping":false, ...}
```

Polled to completion (about 3 minutes), then `GET /jobs/{id}/results`:

```
rows_in 57500   rows_valid 57179   rows_dropped 321
wait_mae_minutes 11.52   wait_baseline_mae_minutes 34.23   wait_improvement_percent 66.3
forecast_mae 2.36        forecast_improvement_percent 20.93
simulation_validated true
```

The four headline numbers from the non-Docker run, checked against the
containerised run's own artifacts:

| Number | Expected | Container | Source |
|---|---|---|---|
| Wait MAE (min) | 11.52 | **11.52** | `/jobs/{id}/results` |
| Rolling conformal coverage | 0.8875 | **0.8875** | `artifacts/tables/conformal_static_vs_rolling.csv` |
| Forecast MAE | 2.356 | **2.36** (2.3557) | `/jobs/{id}/results`, `forecast.json` |
| Cheapest plan cost | 49,280 | **49280.0** | `metrics/staffing_search.json` |

Report download and deletion:

```
> curl -s http://127.0.0.1:8000/jobs/c38225bfeb3b/report -o report.md -w "%{http_code} %{size_download}"
200 25340            # 347 lines, first line "# ED Intelligence - results report"

> docker compose exec api ls -la /app/runs
drwxr-xr-x 4 appuser appuser 4096 c38225bfeb3b

> curl -s -X DELETE http://127.0.0.1:8000/jobs/c38225bfeb3b
{"job_id":"c38225bfeb3b","deleted":true,
 "detail":"The run folder and the uploaded file have been deleted."}

> docker compose exec api ls -la /app/runs
total 8                                  # the folder is gone
```

## 4. Cancellation leaves no orphan worker

While a job was running:

```
> docker top ed-api
UID    PID   PPID  CMD
10001  1004  980   /sbin/docker-init -- sh -c exec uvicorn api.main:app --host "${ED_API_HOST:-0.0.0.0}" ...
10001  1038  1004  /usr/local/bin/python3.12 /usr/local/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
10001  2202  1038  /usr/local/bin/python3.12 -m src.jobs.worker /app/runs/ea50e7869619
```

PID 1 is `docker-init` (`init: true`), uvicorn is its direct child because the
`CMD` uses `exec`, and the worker is a child of uvicorn.

After `POST /jobs/{id}/cancel`:

```
status: cancelled | "Cancelled at your request."
current_stage at cancellation: survival

> docker top ed-api
10001  1004  980   /sbin/docker-init -- sh -c exec uvicorn ...
10001  1038  1004  /usr/local/bin/python3.12 /usr/local/bin/uvicorn ...
```

The worker is gone; no orphan remains.

## 5. Restart, stop/start, and persistence

```
> docker compose restart          real 0m1.206s   # both healthy within 12s
> docker compose stop             real 0m1.231s   # fast = SIGTERM reached the servers
> netstat -ano | grep -E ":8000|:8501"
ports 8000 and 8501 are free
> docker compose start            real 0m6.326s   # both healthy
> docker compose exec api cat /app/runs/marker.txt
persisted                                          # the volume survived
> docker compose down
> netstat ...                      ports 8000 and 8501 are free
> docker volume ls | grep ed-intelligence
local  ed-intelligence-runs                        # a plain down keeps the data
```

`docker compose stop` finishing in ~1.2s rather than hitting a 10-second kill
timeout is the evidence that signals reach the process: `init: true` plus `exec`
in the CMD.

**Unplanned, and worth recording:** mid-session Docker Desktop exited on its own
(`wsl --list --running` reported no distributions). After restarting Docker
Desktop, `restart: unless-stopped` brought both containers back with no
intervention; both reported healthy 18 seconds later, the models reloaded, and
the runs volume still held its marker. That is the demo-resilience behaviour the
restart policy is there for, tested by accident rather than by design.

## 6. Missing artifacts: a friendly 503, not a crash loop

Run with an empty temporary directory mounted at `/app/artifacts`:

```
> docker run -d --name ed-api-missing --init --read-only --tmpfs /tmp:size=256m \
    -p 127.0.0.1:8010:8000 -v <empty>:/app/artifacts:ro ed-intelligence-api:latest

> curl -s http://127.0.0.1:8010/health
{"status":"degraded","models_loaded":false,"forecaster_loaded":false,
 "simulation_calibrated":false,"model_version":"unknown","data_version":"unknown",
 "uptime_seconds":11.3,
 "how_to_fix":"No trained artifacts were found. Produce them with: python -m
   src.pipeline run --config configs/default.yaml --data real (in Docker:
   docker compose --profile train run --rm trainer). Then restart this service."}

> curl -s -X POST http://127.0.0.1:8010/predict-wait -d '{...}'      HTTP 503
{"detail":"Models are not loaded. No trained artifacts were found. Produce them
  with: python -m src.pipeline run --config configs/default.yaml --data real
  (in Docker: docker compose --profile train run --rm trainer). Then restart
  this service."}

> docker ps --filter name=ed-api-missing --format "{{.Status}}"
Up 20 seconds (healthy)            # degraded, but alive and answering
```

**This required a code change.** `api/main.py` previously re-raised
`FileNotFoundError` from `load_bundle()` in its lifespan, so a container with an
empty artifacts mount crash-looped. A crash loop tells an operator nothing: they
see a restarting container, not the one sentence that fixes it. The service now
starts degraded, `/health` carries `how_to_fix`, and every prediction endpoint
answers 503 with the same sentence. A present-but-unreadable artifact still
raises, because that is a real fault and not a missing setup step.

The dashboard card, rendered in the app image against that degraded API:

```
dashboard resolved API_URL = http://ed-api-missing:8000
  ok    Command center ... no-models card        (all 8 pages, failures: 0)

CODE: No trained artifacts were found. Produce them with: python -m src.pipeline
      run --config configs/default.yaml --data real (in Docker: docker compose
      --profile train run --rm trainer). Then restart this service.
```

The card takes its remedy from `/health`, so the dashboard cannot drift from
what the API would tell any other caller.

## 7. Trainer profile, and reproducibility

```
> docker compose --profile train run --rm trainer
...
Pipeline finished: 15 stage(s) in 164.78s
real 2m46.739s                       # run 1

> docker compose --profile train run --rm trainer
real ~2m45s                          # run 2
```

Headline numbers from the Docker-trained artifacts:

```
wait MAE             11.524
forecast MAE         2.3557
cheapest plan cost   49280.0
rolling coverage     0.8875
```

Run 1 against run 2, all twelve metrics files:

```
IDENTICAL across the two trainer runs: 12
  wait_models.json  forecast.json  staffing_search.json  fairness.json
  validation.json   split.json     sim_calibration.json  scenarios.json
  survival.json     sensitivity.json  shap.json  features.json
DIFFERENT: 0
```

### Does Docker change the numbers?

The same pipeline was then run on the host into a separate artifacts directory
and compared field by field against the container's output:

```
largest relative float difference: 2.274e-13  at survival.json/cox[4]/p_value
values that differ when rounded to 6 decimals: 0
```

So: **no reported metric changes.** What does differ, and why:

| Difference | Cause |
|---|---|
| Line endings in the JSON (CRLF on Windows, LF in the container) | `save_json` writes text mode; content identical |
| `figures[...]` paths in `fairness.json`, `shap.json` | the writer records the path it saw (`/app/artifacts/...` vs `C:\...`) |
| `seconds`, `rows_per_second` in `features.json` | wall-clock timings |
| 13th significant figure in `sim_calibration.json`, `survival.json` | different libm/BLAS; below every rounding the report applies |

## 8. Non-root, and the read-only mounts

```
> docker compose exec api id
uid=10001(appuser) gid=10001(appuser) groups=10001(appuser)
> docker compose exec app id
uid=10002(appuser) gid=10002(appuser) groups=10002(appuser)

> docker compose exec api touch /app/artifacts/SHOULD_FAIL
touch: cannot touch '/app/artifacts/SHOULD_FAIL': Read-only file system   (exit 1)
> docker compose exec api touch /app/SHOULD_FAIL
touch: cannot touch '/app/SHOULD_FAIL': Read-only file system             (exit 1)
> docker compose exec api touch /app/runs/.writable
write ok                                                                  (the volume)
> docker compose exec app touch /app/artifacts/SHOULD_FAIL
touch: cannot touch '/app/artifacts/SHOULD_FAIL': Read-only file system   (exit 1)
```

`read_only: true` is in effect for **both** services, with `tmpfs` for `/tmp`
and, for the dashboard, for `/home/appuser` — Streamlit writes into `$HOME` on
first run and exits if it cannot, so a tmpfs home is what lets the root
filesystem stay immutable. Both were verified to serve traffic under this
configuration; nothing had to be dropped.

### Retention sweep, in the container

Exercised with the environment override, which also tests that the override is
read at all:

```
> docker compose exec -e ED_UPLOAD_RETENTION_HOURS=0.0001 api python -c "..."
runs_dir           : /app/runs
retention_hours    : 0.0001 (env override of 24)
before sweep       : ['ea50e7869619']
deleted            : ['ea50e7869619']
after sweep        : []
```

`prune_runs` scans the filesystem rather than the engine's in-memory job list,
so it also removes folders orphaned by a container restart — which is exactly
what had happened to that one.

## 9. The scripts

Each was run, not just written.

```
> .\scripts\docker_up.ps1
  Built in 1.1s.  Both containers are healthy (12.7s).
  Dashboard: http://localhost:8501   API docs: http://localhost:8000/docs
  UP exit=0 elapsed=16.5s

> .\scripts\docker_status.ps1
  ed-api / ed-dashboard healthy, 127.0.0.1 ports
  Models loaded: yes | base+queue-raw-f864e5aa | real-57179rows-20261008
  Uploaded runs: (listed, with total size)      STATUS exit=0

> .\scripts\docker_logs.ps1 -NoFollow -Service api -Tail 5      LOGS exit=0
> .\scripts\docker_down.ps1
  Port 8000 is free. Port 8501 is free.                         DOWN exit=0
> .\scripts\docker_train.ps1 -RestartServices
  Trained in 186.2s. Report: C:\dev\ed-project\artifacts\report.md
  "Nothing is running, so there was nothing to restart."        TRAIN exit=0
> .\scripts\docker_clean.ps1 -KeepData -Force
  removed both images, kept ed-intelligence-runs                CLEAN exit=0
> .\scripts\docker_up.ps1        # rebuilt from removed images
  Both containers are healthy (13s)                             UP exit=0
```

### Two bugs the scripts had, found by running them

1. **"Build failed" after a successful build.** `Invoke-EdCompose` returned
   compose's output *and* its exit code, so `$code = Invoke-EdCompose build`
   captured an array of log lines and `$code -ne 0` was true. Fixed by piping
   compose's output to `Out-Host` inside the helper, so the return value is a
   single integer.
2. **`docker compose up -d` ran in the foreground.** The helper took
   `[Parameter(ValueFromRemainingArguments)]`, and PowerShell reads a
   dash-prefixed token as a parameter name of the *function* before it becomes a
   value, so `-d` was swallowed. The symptom was a script that appeared to hang
   while streaming logs. Fixed by taking an explicit array:
   `Invoke-EdCompose @("up", "-d")`. Every call site was converted, including the
   two that were splatting with `@`, which has the same problem.

Both are the kind of bug that only running the script finds, which is why the
scripts were run rather than reviewed.

## 10. Image scanning

`docker scout` is available (CLI 1.24.0). It initially failed to reach the
engine; after the Docker Desktop restart it indexed both images. `trivy` is not
installed on this machine.

```
> docker scout quickview ed-intelligence-api:latest
 Target     │  ed-intelligence-api:latest  │    0C     9H    10M    31L
 Base image │  python:3.12-slim            │    0C     1H     6M    27L
Policy status FAILED (3/7)   Health score D (33%)
  v Default non-root user
  v No outdated base images
  v No unapproved base images
  ! Fixable critical or high vulnerabilities found     0C 8H 0M 0L
  ! High-profile vulnerabilities found                 0C 0H 1M 0L
  ! Copyleft licensed packages found                   366 packages
  ! Required supply chain attestations missing         2 deviations

> docker scout quickview ed-intelligence-app:latest
 Target     │  ed-intelligence-app:latest  │    0C    15H    14M    28L
Policy status FAILED (4/7)   Health score C (56%)
```

The fixable high-severity findings, with the versions that fix them:

| Image | Package | Pinned | Fixed in | Highs |
|---|---|---|---|---|
| api | `starlette` (via fastapi 0.115.6) | 0.41.3 | 0.49.1 / 1.1.0 / 1.3.1 | 3 |
| api | `python-multipart` | 0.0.20 | 0.0.22 / 0.0.27 / 0.0.30 | 3 |
| api | `lightgbm` | 4.5.0 | 4.6.0 | 1 |
| api | `pyarrow` | 17.0.0 | 23.0.1 | 1 |
| app | `pillow` (via streamlit) | 11.3.0 | — (13 advisories) | 13 |
| app | `pyarrow` | 17.0.0 | 23.0.1 | 1 |

**No pins were changed.** Those versions are deliberate and documented in
`requirements.txt` (every pin ships a cp311/cp312 wheel so a clean install needs
no compiler, and `autograd-gamma` is held at 0.4.2 for exactly that reason), and
the project rule is to ask before touching dependencies. Bumping them is a
decision for the maintainer, and it needs a test run: `starlette` and
`python-multipart` are transitive through FastAPI, so they move with it.

Note also that `0 critical` and `Default non-root user` passing are the two
results that matter most for a locally-published demo with no network exposure.

## 11. Python tests

`tests/test_docker.py` adds 48 tests that parse the compose file, the Dockerfile,
`.dockerignore`, `.gitattributes` and `.env.example` and assert:

- no host port published on anything but loopback, and the container side *does*
  bind `0.0.0.0` (the mirror-image mistake)
- `USER` is a non-root numeric uid in both targets
- `HEALTHCHECK` present in both targets, probing with urllib and not curl, plus
  the compose healthcheck that `service_healthy` actually reads
- `ED_API_BASE_URL` names the compose service, not localhost, *and* `app/config.py`
  really reads that variable
- artifacts mounted read-only for both services; the runs volume is the only
  writable mount; the trainer is the only service that may write artifacts, and
  `data/raw` is read-only even for it
- `.dockerignore` excludes `.venv/`, `runs/`, `logs/`, `data/`, `artifacts/`,
  `tests/`, `__pycache__/`, `.env`
- no `build-essential` or `gcc`; base image pinned by digest; no `COPY` of
  `data/`, `artifacts/`, `runs/` or a bare `.`
- `exec` in both CMDs and `init: true` in compose, so signals and reaping work
- `.gitattributes` pins LF, and the compose files and Dockerfile contain no CRLF
- `.env` gitignored, `.env.example` committed, no secret-shaped assignments, and
  every variable it declares is one the code actually reads
- `docker-compose.override.yml` is absent, so the dev overlay is never applied
  silently
- logs bounded, and `read_only: true` with a writable `/tmp` for both services

Two further tests were added to `tests/test_local_run_scripts.py`, which owns the
"the scripts and the code agree about where the API is" invariant: one asserts
the dashboard's resolved default with a clean environment, the other that
`ED_API_BASE_URL` wins over `ED_API_URL` and that `ED_API_URL` still works alone.

```
> python -m pytest -q
994 passed, 11 skipped, 115 warnings in 1083.59s (0:18:03)
exit=0
```

Up from 945 before this work: 48 in `tests/test_docker.py`, plus the two above,
minus one that was rewritten rather than added.

## Not verified

- **Other machines and other Windows versions.** Everything here is one host:
  Windows 11 Home 10.0.26300, Docker Desktop 4.90.0, WSL 2. Hyper-V backend,
  Windows 10, Server, Linux and macOS hosts were not tested.
- **Performance under load.** One upload job at a time, one browser. No
  concurrency, no sustained load, and the resource limits (`cpus`, `memory`)
  were set from the VM's size rather than measured against a load test.
- **`trivy`** is not installed here, so only `docker scout` results are reported.
- **Scout's own availability is fragile.** It failed with "Failed to connect to
  Docker Engine" before the Docker Desktop restart and worked after. Treat a
  scout failure as a scout problem before assuming anything about the image.
- **The vulnerability fixes themselves.** The upgrade paths in section 10 are
  what scout reports; none were applied or tested.
- **Multi-architecture images.** Built and run on `linux/amd64` only. No arm64
  build, so this is untested on Apple silicon.
- **A registry.** Images are local; nothing was pushed, and no registry
  authentication or signing was exercised.
- **The dev overlay was validated but not lived in.** `docker compose -f
  docker-compose.yml -f docker-compose.dev.yml config` passes, but a working
  session with `--reload` editing files was not run.
- **`start_docker.bat` was not double-clicked.** Its contents and CRLF line
  endings were verified, and it does nothing but call `docker_up.ps1`, which was
  run repeatedly — but the double-click path itself was not exercised.
- **The synced-folder refusal was not triggered from a real OneDrive checkout.**
  The logic was exercised directly — the helper was dot-sourced and its project
  root pointed at `C:\Users\Omkar\OneDrive\Documents\ed-project`, and
  `Assert-EdNotSynced` refused with exit 1 and the explanation — but there is no
  actual OneDrive copy of this repository to run the real script from.
