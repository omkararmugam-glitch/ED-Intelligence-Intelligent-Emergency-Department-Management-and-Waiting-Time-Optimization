# Upload workflow: manual test checklist

The automated suite covers the API, the engine and the page's rendering. It cannot
cover what a person sees in a browser: whether the progress actually moves, whether
a message is legible, whether a download opens. This checklist is that part.

Work through it against a real browser after any change to `app/upload_page.py`,
`api/jobs.py` or `src/jobs/`.

This is the **deep** list for the upload workflow alone.
`docs/dashboard_manual_test_checklist.md` is the broad one covering every page,
theming, keyboard use and failure behaviour, with a shorter upload section. Use
that one after a change to the dashboard generally, and this one after a change
to the upload flow in particular.

**Upload only synthetic or fully de-identified data.** `data/raw/ed_v2.csv` is
synthetic data calibrated to published benchmarks and is the right file for every
step below.

> ## Status: NOT PERFORMED
>
> **Nothing in this file has been run.** It was written and last extended on
> 2026-10-10 by an automated session that has no browser and cannot see a screen.
> Every box below is unticked because none of them has been checked, not because
> something failed.
>
> The automated suite covers the API, the job engine and the page's rendering
> through Streamlit's `AppTest` — 1,000-plus tests, all passing. None of that is
> evidence about what a person sees: whether the progress bar actually moves,
> whether a message is legible at the size it renders, whether a download opens in
> the application the reader expects. **A human has to do the run below.**

## 0. The short run (about five minutes)

Do this one after any change to the upload page. It is the smallest sequence that
would catch a page that is broken rather than merely imperfect; the numbered
sections after it are the thorough version.

```powershell
.\scripts\run_api.ps1          # terminal 1
.\scripts\run_app.ps1          # terminal 2
```

| # | Click this | This should appear |
|---|---|---|
| 1 | Open `http://localhost:8501` | The dashboard loads. The sidebar says **API ok** and shows a model version. |
| 2 | **Data and analysis** in the sidebar | The four-step flow, step 1 highlighted. The privacy notice is **above** the file picker. |
| 3 | Pick `data\raw\ed_v2.csv` | The file name and its size appear. The roster editor appears below, with an amber warning that the default roster describes the demo dataset. |
| 4 | **Start analysis** | The page switches to progress within a second or two. It must not sit on "uploading". |
| 5 | Watch for ~3 minutes | Stages tick off **in order**, each gaining an elapsed time. The bar advances. The page still scrolls while it works. |
| 6 | Wait for the end | The page switches to results **by itself**. No manual refresh. |
| 7 | Read the headline row | Rows accepted, wait MAE, interval coverage, and whether the simulation validated. The decision-support banner is above them. |
| 8 | Open the **Simulation and staffing** tab | The validation verdict is **above** the scenario table. The roster you entered is quoted back. |
| 9 | Open the **Report** tab, press the download | `report.md` saves and opens in a text editor. |
| 10 | Press **Delete this upload** | The upload goes, and `runs\` no longer holds that job folder. |

**Stop and investigate if** any step shows a Python traceback, an empty tab with no
explanation, a progress bar that never moves, or a number with no units.

## 0b. The refusal path (about four minutes)

This exercises the gate that `tests/test_validation_gate.py` proves fires. Build an
extract whose waits are detached from the queue, so the simulation cannot reproduce
it:

```powershell
.\.venv\Scripts\python.exe -c @'
import sys; sys.path[:0] = ['.', 'tests']
import upload_fixtures as f
d = f.decouple_waits_from_queue(f.slice_days(120), scope='across_acuity')
open('decoupled.csv','wb').write(f.to_csv_bytes(d))
'@
```

| # | Click this | This should appear |
|---|---|---|
| 1 | Upload `decoupled.csv`, leave the simulation **on**, **Start analysis** | The job runs normally through the model stages. |
| 2 | Wait for it to finish | The job **completes** — it does not fail. The stage list shows *Scenarios* and *Staffing search* as **skipped**, each with a reason in English. |
| 3 | Open **Simulation and staffing** | A red refusal box saying validation **failed** and that scenario and staffing results are refused. |
| 4 | Read below it | The per-metric validation table, and a table of the checks that failed, naming `mean wait ESI 1` and `mean wait ESI 2`. |
| 5 | Look for a way to override | **There must not be one.** No checkbox, no "show anyway", no expander that reveals scenario numbers. |

**Stop and investigate if** the job fails instead of completing, a traceback reaches
the screen, the simulation tab is empty rather than refusing, or any scenario or
staffing number is displayed at all.

## 0c. With the optional API key set (about two minutes)

Only if `ED_API_KEY` is being used. Set it in `.env`, restart both services.

| # | Click this | This should appear |
|---|---|---|
| 1 | Open the dashboard | The sidebar still says **API ok** — the dashboard reads the same variable and authenticates for you. |
| 2 | Run the short run above | No change in behaviour anywhere. |
| 3 | In a terminal: `curl http://127.0.0.1:8000/forecast` | `401` and a message naming the `X-API-Key` header. The key itself must not appear in it. |
| 4 | `curl http://127.0.0.1:8000/health` | `200`, with `"authentication_required": true`. |
| 5 | Unset `ED_API_KEY`, restart, repeat step 3 | A normal answer. The service is open again. |


## Setup

```powershell
.\scripts\run_api.ps1          # terminal 1
.\scripts\run_app.ps1          # terminal 2
```

- [ ] The API banner prints `http://localhost:8000/docs`, and that link opens.
- [ ] The dashboard banner prints `http://localhost:8501`, and that link opens.
- [ ] Both say they are bound to `127.0.0.1`.
- [ ] The sidebar shows **API ok** and the model version.
- [ ] The sidebar carries the "synthetic or fully de-identified" line.

## 1. The happy path

- [ ] Open **Data and analysis**. The four-step flow is shown at the top, with
      step 1 highlighted.
- [ ] The privacy notice is visible before the file picker, not below it.
- [ ] The file size and row limits are stated on the uploader.
- [ ] Upload `data\raw\ed_v2.csv`. The file name and size appear.
- [ ] The roster editor shows the four demo shifts with an amber warning that the
      default describes the demo dataset.
- [ ] "Doctor-hours per day" updates when a doctor count is edited.
- [ ] Press **Start analysis**. The page switches to progress within a second or two
      — it does not hang on the upload.
- [ ] The stage checklist ticks through in order. Elapsed seconds appear per stage.
- [ ] The progress bar advances. The page stays responsive while it does.
- [ ] When the job finishes the page switches to results by itself, without a manual
      refresh.

## 2. Results

- [ ] The decision-support banner is above the numbers.
- [ ] The headline row shows rows accepted, wait MAE, interval coverage and whether
      the simulation validated.
- [ ] The comparison table shows your run beside the baseline dataset.
- [ ] **Data quality**: drop reasons add up to the dropped count; optional columns
      are listed; the drift verdict reads sensibly.
- [ ] **Wait times**: the typical error, the baseline's error and the improvement
      are all shown. The feature comparison table renders, and the caveat about it
      overstating the effect is in a Technical details expander, not on the surface.
- [ ] **Forecast**: the chart draws and the error-by-horizon table renders.
- [ ] **Fairness**: subgroups render; any flagged group is in red above the table.
- [ ] **Patients who left**: the event counts and the two ways of counting render.
      The statistical method names (Kaplan-Meier, Aalen-Johansen, Cox) appear only
      inside Technical details.
- [ ] **Simulation and staffing**: the validation verdict is **above** the scenario
      table, not below it. The roster you entered is quoted.
- [ ] **Report**: a plain-language note says what the document is, the download is
      on the surface, and the report itself is inside a Technical details expander
      with a scrolling box.
- [ ] Every tab has a "What these numbers mean" expander that opens.

## 3. Downloads

- [ ] `report.md` downloads and opens in a text editor.
- [ ] The ZIP downloads, opens, and contains `tables/` and `figures/`.
- [ ] The validation JSON downloads.
- [ ] The file list at the bottom matches what actually downloaded.

## 4. Mapping that must be confirmed

Make a copy of the extract with an ambiguous header:

```powershell
$f = Import-Csv data\raw\ed_v2.csv | Select-Object -First 5000
$f | Select-Object @{n='pid';e={$_.patient_id}}, @{n='arrival';e={$_.arrival_time}},
    @{n='registration_time';e={$_.arrival_time}}, @{n='esi';e={$_.triage_level}},
    @{n='seen_time';e={$_.seen_time}}, @{n='departure_time';e={$_.departure_time}},
    @{n='disposition';e={$_.disposition}} | Export-Csv ambiguous.csv -NoTypeInformation
```

- [ ] Uploading it shows the mapping form, not a running job.
- [ ] The blocking reason names `arrival` and `registration_time`.
- [ ] The preview shows the first rows as parsed.
- [ ] There is one dropdown per schema field, pre-filled where detection was sure.
- [ ] **Confirm mapping and start** is disabled until every required field has a
      column.
- [ ] Choosing `registration_time` and confirming starts the job.
- [ ] **Cancel and start over** returns to the upload form and leaves no job behind
      (check `runs\`).

## 5. Options

- [ ] **Quick** skips explain, survival and the simulation, each with a reason in its
      tab that reads as English, not as a code path.
- [ ] **Skip the simulation** hides the roster editor and skips the three simulation
      stages, saying the roster is why.
- [ ] **Scoring-only** skips training, shows the "trained on the baseline dataset"
      note, and shows the drift table beside the error figures.
- [ ] Entering a timestamp format removes the day/month warning from the next run.

## 6. Degraded inputs

Build a one-month extract and upload it:

```powershell
$cut = (Get-Date "2024-02-01")
Import-Csv data\raw\ed_v2.csv |
  Where-Object { [datetime]$_.arrival_time -lt $cut } |
  Export-Csv one_month.csv -NoTypeInformation
```

- [ ] The job succeeds.
- [ ] The Forecast tab says forecasting needs about 60 days and states what the file
      covers.
- [ ] **Patients who left** either runs or says there were too few to analyse.
- [ ] No tab is silently empty.

## 7. Things that should be refused

- [ ] An empty file: a clear message, no job created.
- [ ] A header-only file: says there are no data rows.
- [ ] A `.xlsx`: says to save as CSV.
- [ ] A file over 50 MB: refused by size, with the limit named.
- [ ] A file missing `triage_level`: the mapping form names the missing column.
- [ ] A roster with a gap (delete the night shift): the error names the uncovered
      hours and **Start analysis** stays disabled.
- [ ] A roster with a negative doctor count: refused.

## 8. Interruptions

- [ ] Start a full run and press **Cancel this analysis**. It reaches `cancelled`
      within a few seconds, not at the end of the current stage.
- [ ] Refresh the browser mid-run. The same job resumes — the id is in the URL.
- [ ] Copy the URL into a second tab. Both tabs follow the same job.
- [ ] Press **New analysis** mid-run: the form returns, and the job keeps running
      (paste the old URL back to confirm).
- [ ] Stop the API (Ctrl+C in terminal 1) while the results are open, then refresh.
      The page says it cannot reach the API and names the script to start. No
      traceback appears anywhere on screen.
- [ ] Restart the API. The page recovers on the next refresh.

## 9. Queueing

- [ ] Start a full run, then upload a second file. The second says it is queued and
      explains that one runs at a time.
- [ ] Queue up to the limit (4 by default) and upload once more: the message says
      the queue is full, and suggests cancelling one.

## 10. Privacy

- [ ] After a run, `runs\<job_id>\upload.csv` exists.
- [ ] **Delete now** removes the whole `runs\<job_id>\` folder, immediately.
- [ ] After deletion the page offers a fresh start rather than erroring.
- [ ] `artifacts\pipeline.log` and the API's console output contain **no** patient
      rows, ids, names or timestamps from the uploaded file.
- [ ] `runs\<job_id>\pipeline.log` likewise contains counts, not rows.
- [ ] `git status` shows no `runs/` content staged — the folder is ignored.

## 11. Isolation

Before and after a run:

```powershell
Get-FileHash artifacts\models\*.joblib | Format-Table Hash, Path
```

- [ ] Every hash is unchanged after an upload completes.
- [ ] `/predict-wait` returns the same numbers before and after.

## What this checklist does not cover

- Screen readers and keyboard-only navigation.
- Browsers other than the one you used.
- Behaviour on a touch device.
- Concurrent users on the same service.
