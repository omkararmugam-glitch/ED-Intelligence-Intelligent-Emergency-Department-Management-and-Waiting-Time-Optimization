# Dashboard redesign — verification record

What was run, and what came back. Every number below was read from test output or
from a running service; none is estimated.

Date of this run: 2026-10-08. Windows 11, PowerShell, `.venv` (Python 3.12.10,
Streamlit 1.41.1).

## Test suite

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

```
930 passed, 11 skipped, 117 warnings in 1246.24s (0:20:46)
exit code 0
```

Before this work the same command reported `572 passed`. The 11 skips are
conditional and expected: 5 views draw no Altair charts so the two chart rules
skip them, 5 more for the chart-caption rule, and `_shared.py` is a helper rather
than a page so the `render()` rule skips it.

Dashboard-specific files alone:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_dashboard_imports.py `
  tests\test_dashboard_navigation.py tests\test_dashboard_language.py `
  tests\test_wait_estimator.py tests\test_replay_leakage.py `
  tests\test_upload_dashboard.py tests\test_local_run_scripts.py `
  tests\test_api_client.py -q
```

```
425 passed, 11 skipped, 1 warning in 50.50s
```

## Live services

Port 8000 was already occupied by a pre-existing API, so a second instance was
started on 8001 and left under this session's control.

```powershell
.\scripts\run_api.ps1 -Port 8001
.\scripts\run_app.ps1 -Port 8502 -ApiUrl http://localhost:8001
```

API health:

```
{"status":"ok","models_loaded":true,"forecaster_loaded":true,
 "simulation_calibrated":true,"model_version":"base+queue-raw-b098128b",
 "data_version":"real-57179rows-20261005","uptime_seconds":20.1}
```

Dashboard:

```
GET http://localhost:8502/               -> http 200
GET http://localhost:8502/_stcore/health -> ok [http 200]
```

Launcher banner, verbatim:

```
API reachable at http://localhost:8001

ED Intelligence dashboard
  Dashboard: http://localhost:8502
  Using API: http://localhost:8001   (docs at http://localhost:8001/docs)

  Bound to 127.0.0.1:8502
  Reachable from this machine only. Set ED_APP_HOST=0.0.0.0 to publish it.
```

No tracebacks, errors or warnings in the dashboard's stderr for the whole session.

## Every page against the live API

Driven through Streamlit's own script runner (`AppTest.from_file` on the real
`app/dashboard.py`), navigating with the real sidebar radios. No mocks.

| Page | Result | Elements | KPI cards | Tables |
|---|---|---|---|---|
| Command center | OK | 16 | 4 | 0 |
| Demand forecast | OK | 21 | 5 | 3 |
| Wait estimator | OK | 7 | 0 | 0 |
| What-if scenarios | OK | 9 | 0 | 0 |
| Staffing planner | OK | 18 | 8 | 1 |
| Data and analysis | OK | 18 | 0 | 1 |
| Model quality and fairness | OK | 31 | 9 | 6 |
| About and limitations | OK | 23 | 0 | 0 |

`FAILURES: 0`. Wait estimator and What-if scenarios show no cards before
submission by design — neither answers a question nobody has asked yet.

Interactions exercised live:

| Interaction | Result |
|---|---|
| Forecast horizon 24 h | OK, 5 cards |
| Forecast horizon 48 h | OK, 5 cards |
| Forecast horizon 72 h | OK, 5 cards |
| Shift replay on, Command center | OK, cards 4 → 7 (comparison panel appeared) |
| Shift replay labelled "historical" | True |
| Shift replay on, Wait estimator, then submit | OK, 6 cards |
| Wait estimate from the default form | OK, 3 cards, range rendered |
| What-if scenarios, real simulation run | OK, 1 comparison table |

## A real upload, end to end on the live service

12,000 rows trimmed from `data/raw/ed_v2.csv`, pushed to the running service and
driven through the dashboard's results view.

```powershell
head -12001 data/raw/ed_v2.csv > upload_sample.csv   # via git bash
curl -X POST http://localhost:8001/jobs -F "file=@upload_sample.csv" `
     -F "depth=quick" -F "skip_simulation=true"
```

```
job 5cae7af24b40 accepted, mapping ready, 12000 rows, no blocking issues
  5s: running  forecast  9/15
 15s: succeeded          15/15
```

The results view then rendered against that real job:

```
[ OK ] results view rendered for a real job
       tabs=7 metrics=19 tables=9
       tabs: ['Data quality', 'Wait times', 'Forecast', 'Fairness',
              'Patients who left', 'Simulation and staffing', 'Report']
[ OK ] no banned term or statistical jargon in the real results view
```

**This found a crash that every test had missed.** The Report tab raised
`RuntimeError: Invalid binary data format: <class 'dict'>`. `validation.json` is
served by the API as a *file* but carries `application/json`, so the client
sniffed the content type, decoded the bytes to a dict, and `st.download_button`
rejected it. The canned test payload was plain `b"{}"`, which gets
`application/octet-stream`, so the stub never reproduced the real headers. Fixed
by giving the client an explicit raw-bytes path and routing every artifact and
report fetch through it; `tests/test_api_client.py` now pins it with the real
content type, and reintroducing the bug fails
`test_every_artifact_helper_asks_for_raw_bytes`.

The job and its uploaded file were deleted afterwards (`DELETE /jobs/{id}` →
`deleted: true`), and `GET /jobs` came back empty.

## Replay leakage, confirmed server-side

The service's own log, which is the part that cannot be faked by the dashboard:

```
forecast: horizon=48h as_of=2024-12-30 23:00:00   <- normal mode, end of history
forecast: horizon=24h as_of=2024-12-30 23:00:00
forecast: horizon=72h as_of=2024-12-30 23:00:00
forecast: horizon=48h as_of=2024-11-08 17:00:00   <- replay, inside the held-out period
```

The last line is the point: with replay on, the dashboard asks for a forecast
*issued at* the replayed moment, and the service computes it from data before that
instant only. The queue-state half is covered by `tests/test_replay_leakage.py`,
which asserts truncation equivalence — computing the snapshot from the full table
and from the table cut off at T must give identical results — across five moments
and three acuity levels, plus a stronger variant that scrambles the future and
requires the past to be unchanged.

Endpoint hit counts for the session:

```
5 GET /forecast          4 GET /jobs/-/defaults   4 GET /health
2 POST /simulate         2 POST /predict-wait     2 GET /staffing-recommendation
2 GET /jobs
```

## Degraded-service behaviour

Every page was also driven with the service answering badly. All eight pages
render without an exception under each of: an empty object, an empty list, a
`null` body, 404, 422, 500, 502, a refused connection, and a timeout — 8 pages ×
9 conditions. Found and fixed two real crashes doing this (see the report).

With no service reachable at all:

| Page | Result |
|---|---|
| all 8 | render, each showing a friendly card with a Try again button |

Sidebar shows **Service unavailable**; with the service up it shows **Connected**.

## Contrast, measured

Computed from the palette in `app/theme.py` against the light base in
`.streamlit/config.toml` (`#fbfcfd`) and Streamlit's dark base (`#0e1117`).
These colours are used for status pills, chart marks and bold numbers, so the
applicable WCAG AA threshold is **3.0** (graphical objects and large text)
rather than 4.5.

| Colour | On light | On dark | Verdict |
|---|---|---|---|
| BLUE `#2a78d6` | 4.30 | 4.28 | both ≥ 3.0 |
| TEAL `#0e8a8a` | 4.07 | 4.52 | both ≥ 3.0 |
| AMBER `#c77700` | 3.37 | 5.46 | both ≥ 3.0 |
| RED `#c0392b` | 5.29 | 3.47 | both ≥ 3.0 |
| GREEN `#1b8a5a` | 4.23 | 4.35 | both ≥ 3.0 |
| GREY `#6b7280` | 4.71 | 3.91 | both ≥ 3.0 |
| INK_SOFT `#8a9099` | 3.13 | 5.88 | both ≥ 3.0 |

Body text (`#1c2430` on the light base) is 15.21:1, well past the 4.5 AA body
threshold.

This is an arithmetic check on the colour values, not a look at a screen. It says
the palette is usable on either base; it does not say the pages look good on
either, which nobody has checked.

## What was NOT verified

- **Human browser interaction.** Nobody opened Chrome and clicked through. The
  script runner executes the same code path Streamlit executes, but drag-and-drop
  file upload, the 2-second progress fragment polling over a live websocket,
  hover tooltips, and actual visual layout are untested.
  `docs/dashboard_manual_test_checklist.md` is the list for whoever does that.
- **Other browsers.** Only the code path was exercised; no browser engine was.
- **Screen readers.** Chart captions, widget labels and the icon-plus-word status
  pills are written for one, and a static test requires every chart to carry a
  description, but no screen reader has been run against this app.
- **Dark mode appearance.** The CSS draws accent surfaces as translucent washes
  over the theme background specifically so both bases work, and
  `.streamlit/config.toml` documents how to switch, but nobody has looked at the
  dark base.
- **Small phone screens.** The responsive rules target laptop and tablet widths.
  Below roughly 700px is out of scope and untested.
- **Printing.**
- **The upload flow driven by hand in a browser.** A real 12,000-row CSV was put
  through the live service and its results view rendered (see above), but the
  file was posted with `curl` rather than dropped on the upload widget, and
  nobody watched the progress view tick over a websocket.
- **Colour-vision simulation.** Status always carries an icon and a word as well
  as a colour, and the replay comparison distinguishes its two series by dash
  pattern too, but no simulator was run.
