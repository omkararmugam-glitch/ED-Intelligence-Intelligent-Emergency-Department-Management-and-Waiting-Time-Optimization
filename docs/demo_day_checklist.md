# Demo day checklist

Written for: whoever is standing up and showing this, on the day.

For the half hour before a demo. It assumes nothing is running and that something
will go wrong. Two paths are given, in order: **Docker**, which starts both
services with one command, and the **non-Docker scripts**, which are the fallback
when the engine will not cooperate. Both serve the same models from the same
`artifacts\` directory, so switching between them changes no number on screen.

Allow 30 minutes the first time, 10 minutes once you have done it before.
The first Docker build takes a few minutes; do not start it five minutes before
the demo.

## The hour before (do this early)

- [ ] Working copy is at a **local** path — `C:\dev\ed-project`, not OneDrive.
      The Docker scripts refuse to run from a synced folder, and for good
      reason: a sync client rewriting files under a running container produces
      failures that look like bugs in the analysis.
- [ ] `artifacts\metrics\wait_models.json` exists. If not, train first:
      `.\scripts\docker_train.ps1` (or `.\scripts\run_pipeline.ps1`).
- [ ] `data\raw\ed_v2.csv` exists, if you plan to demonstrate the upload flow.
- [ ] Laptop on mains power, sleep disabled, notifications silenced.
- [ ] Browser zoom at 100%, one window, no unrelated tabs.

## Path 0 — the one-click launcher (simplest)

Double-click **`Start-Demo.cmd`** in the project root. Double-click
**`Stop-Demo.cmd`** when you are done.

It is the two terminals of Path B, run hidden with their output going to
`runs\logs\api.log` and `runs\logs\app.log`, plus the waiting and the browser.
Use it when you want the demo up with no typing; use Path A or B when you want to
watch the services' output live, which is what you want if something is misbehaving.

- [ ] It prints `API healthy, models loaded`, then `Dashboard serving pages`, then
      the two URLs. It does not print a URL before both answer.
- [ ] The browser opens on the Command center with the figures already on screen.
- [ ] If it refuses, the message names what to fix. It will not start a demo on top
      of a port something else is using, and it will not kill whatever is using it.
- [ ] If Docker is already up it notices and only opens the browser, so it is safe
      to double-click without checking first.

`Stop-Demo.cmd` stops only what the launcher recorded in `runs\demo.pids`, and
checks each process is still the one recorded before stopping it. It never stops
containers; if Docker is up it says so and gives you `docker compose down`.

## Path A — Docker (preferred)

```powershell
cd C:\dev\ed-project
.\scripts\docker_up.ps1 -Open
```

Or double-click `start_docker.bat`.

- [ ] Docker Desktop is running before you start — the whale icon is still, not
      animating. `docker info` answers.
- [ ] The script reports **both containers healthy** and prints the URLs. It will
      not print a URL that does not answer, so if it printed them, they work.
- [ ] <http://localhost:8501> shows the dashboard, sidebar status **Connected**.
- [ ] <http://localhost:8000/docs> shows the API docs.
- [ ] `.\scripts\docker_status.ps1` shows **Models loaded: yes** and a model
      version string.

Keep a second terminal open on `.\scripts\docker_logs.ps1`. If a page misbehaves
mid-demo, the answer is usually already on that screen.

### If Docker will not start

Decide fast and move on — do not debug WSL in front of an audience.

| What you see | Do this |
|---|---|
| "The Docker engine is not reachable" | Start Docker Desktop, wait for the icon to settle (up to a minute). One retry only, then switch to Path B. |
| "Port 8000 is already in use" | Something non-Docker is already serving. That is fine — skip to Path B, it is probably already running. |
| Build fails, or "no space left on device" | `docker system df`, then `.\scripts\docker_clean.ps1`. If that is not quick, switch to Path B. |
| API never becomes healthy | `docker compose exec api ls /app/artifacts/metrics` — if empty, the mount or the training is the problem. Switch to Path B, which reads `artifacts\` directly. |

## Path B — the non-Docker fallback

No containers, no engine, no mounts. Two terminals, both in the project root:

```powershell
# Terminal 1 — the analysis service
.\scripts\run_api.ps1

# Terminal 2 — the dashboard
.\scripts\run_app.ps1
```

- [ ] Terminal 1 prints <http://localhost:8000/docs> and stays running.
- [ ] Terminal 2 prints <http://localhost:8501> and opens the dashboard.
- [ ] Sidebar status says **Connected**.

Stop both with Ctrl+C. If Docker is also running, these will collide on the
ports — run one path or the other, not both.

## Smoke test, either path (3 minutes)

The same checks work against either, because the URLs are the same.

- [ ] **Command center** loads in seconds. Four KPI cards with numbers, not dashes.
- [ ] **Demand forecast** draws the arrivals chart with a shaded interval.
- [ ] **Wait estimator**: pick ESI 3 and a busy queue; a range comes back, with
      factors underneath.
- [ ] **What-if scenarios**: run the baseline; a verdict with an interval appears.
- [ ] **Staffing planner** names a cheapest plan *and* the sentence about the
      finding being partly circular. Do not skip that sentence in the demo.
- [ ] **Model quality and fairness**: the high-acuity false-negative finding is
      visible, and so is the per-acuity threshold table underneath it.
- [ ] Every page carries the **Demo data (synthetic)** badge.

## If you are demonstrating the upload flow

- [ ] Use `data\raw\ed_v2.csv`. It takes about three minutes end to end, so start
      it and talk over it rather than waiting in silence.
- [ ] Say out loud that the file is synthetic. The privacy notice on the page says
      so too; read it.
- [ ] Afterwards, delete the job in the UI. Under Docker that also removes it from
      the runs volume; `.\scripts\docker_down.ps1 -RemoveData` is the belt-and-braces
      version.

## Things to say, so nobody has to ask

- It is **decision support**. Every figure is an estimate for a person to weigh.
  Nothing here triages a patient.
- The data is **synthetic**, calibrated to published benchmarks. Numbers are a
  demonstration of method, not evidence about any real department.
- The staffing finding is **partly circular**: the roster that generated the data
  is the one the search calls cheapest-feasible.
- The long-wait warning **fails ESI 1–2** on this data. Per-acuity thresholds were
  fitted and changed pooled recall by less than its own confidence interval; ESI 1
  and 2 have 1 and 29 positive cases, too few to fit at all. That is a finding, not
  an oversight — and it is in the report.
- There is **no authentication**, which is why both ports are published on
  `127.0.0.1` only.

## After the demo

```powershell
.\scripts\docker_down.ps1            # keeps the runs volume
.\scripts\docker_down.ps1 -RemoveData   # also deletes uploaded files
```

- [ ] If anyone uploaded a file, remove it: `-RemoveData`, or delete the job in
      the UI. Do not leave someone's extract on the machine.
- [ ] Ports 8000 and 8501 are free again (`docker_down.ps1` reports this).
