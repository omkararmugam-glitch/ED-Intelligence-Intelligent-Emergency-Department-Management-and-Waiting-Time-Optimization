# Dashboard manual test checklist

For a human tester with a browser. The automated suite covers rendering, error
handling, wording and replay leakage; it cannot check that the thing *looks* right,
that a keyboard can reach everything, or that a screen reader makes sense of it.
That is what this list is for.

Allow about 30 minutes. Tick each box or write down what you saw instead.

This is the **broad** list: every page once. For the upload workflow in depth --
every column-mapping case, every stage that can skip, retention and cancellation
-- use `docs/upload_manual_test_checklist.md` instead.

## Before you start

Two terminals, both in `C:\dev\ed-project`:

```powershell
# Terminal 1 — the analysis service
.\scripts\run_api.ps1

# Terminal 2 — the dashboard
.\scripts\run_app.ps1
```

Open <http://localhost:8501>. If the API is on another port, use
`.\scripts\run_app.ps1 -ApiUrl http://localhost:8001`.

- [ ] The sidebar status line says **Connected**.
- [ ] Every page shows a **Demo data (synthetic)** badge.

## 1. First impressions

- [ ] The Command center loads in a few seconds, not tens of seconds.
- [ ] Four KPI cards across the top, each with a number and a short label.
- [ ] Hovering the **?** on each card gives an explanation in plain words.
- [ ] The congestion card's tooltip states the rule used (the 80th-percentile rule).
- [ ] The arrivals chart has a shaded band and amber triangles on the busy hours.
- [ ] Nothing on the page uses the words "real-time" or "live" except to deny them.

## 2. Language and audience

Walk every page. On each one:

- [ ] No job ids, file paths, `.parquet`, `artifacts/`, model hashes or stage names
      like `train_wait` appear **outside** a "Technical details" expander.
- [ ] Every number has a unit: minutes read as "34 min", shares as "12.3%", counts
      with thousands separators.
- [ ] Every chart has a "Chart description:" caption underneath it.
- [ ] Nothing reads as if written for a developer.

## 3. Colour, contrast and the colourblind check

- [ ] Status pills always show an **icon and a word**, not just a colour.
- [ ] On the replay comparison chart, the two lines differ by **dash pattern** as
      well as colour.
- [ ] Switch your OS to dark mode and reload. Text stays readable, cards and
      callouts still have visible edges, charts' axes are still legible.
- [ ] Switch back to light mode and reload. Same check.
- [ ] If you have a colourblindness simulator, check that "On target" and
      "Worth a look" are still distinguishable (they should be, by icon).

## 4. Keyboard only

Put the mouse away.

- [ ] <kbd>Tab</kbd> reaches every sidebar navigation option.
- [ ] <kbd>Space</kbd> / arrow keys change the selected page.
- [ ] The focus ring is clearly visible on every control you land on.
- [ ] You can complete the whole Wait estimator form and submit it by keyboard.
- [ ] You can open and close a "Technical details" expander by keyboard.

## 5. Each page

### Demand forecast
- [ ] Switching 24 / 48 / 72 hours changes the chart and the table.
- [ ] "How reliable is this?" shows two error figures with the naive baseline
      beside each.
- [ ] The peak-hour caution is present and says the model is **not** better than
      repeating last week at peak.
- [ ] "Download as CSV" produces a file whose headings are readable words.

### Wait estimator
- [ ] Acuity options read as descriptions, not bare numbers.
- [ ] The result leads with a **range**, with the point estimate in smaller text.
- [ ] Leave "I don't know the recent average wait" ticked → the result says the
      estimate is **less precise**.
- [ ] The factor chart's labels are plain English, not column names.

### What-if scenarios
- [ ] Nothing runs until you press the button.
- [ ] A progress bar names the scenario being simulated.
- [ ] Each scenario gets a one-sentence verdict with a reliability judgement.
- [ ] "What these scenarios assume" states that only doctors are constrained.

### Staffing planner
- [ ] The circularity warning is visible **without** opening anything.
- [ ] The roster appears as a grid of shifts, one card each.

### Data and analysis
- [ ] A four-step stepper is at the top and highlights the current step.
- [ ] Upload a small CSV (`data/raw/ed_v2.csv` trimmed to a few thousand rows).
- [ ] Progress shows stage names in plain words ("Learning wait-time patterns"),
      elapsed time, and a "roughly N min to go" estimate.
- [ ] **Cancel** stops the run and the page says so without an error.
- [ ] Re-upload, let it finish, and check the results tabs all render.
- [ ] "Compare with baseline" shows your run beside the reference run.
- [ ] **Delete** removes the run and the uploaded file.
- [ ] Refresh mid-run: the page resumes the same analysis.

### Model quality and fairness
- [ ] The accuracy statement is a sentence with the numbers in it.
- [ ] The interval section says the ranges are slightly **narrower** than advertised.
- [ ] The long-wait warning section says it does **not** work for urgency 1-2.
- [ ] The fairness table marks small groups as unreliable.

### About and limitations
- [ ] Limitations list is present and unflattering.
- [ ] Diagnostics expander shows the endpoint, model identifier and retention.
- [ ] This is the **only** page where the endpoint URL appears.

## 6. Failure behaviour

- [ ] Stop the API (<kbd>Ctrl</kbd>+<kbd>C</kbd> in terminal 1). Reload the
      dashboard.
- [ ] The sidebar says **Service unavailable**.
- [ ] Each page shows a friendly card, a **Try again** button, and the technical
      reason only inside an expander.
- [ ] No traceback appears anywhere on screen.
- [ ] Restart the API, press **Try again** → the page recovers.

## 7. Shift replay

- [ ] On the Command center, open "Shift replay" and turn it on.
- [ ] It says "Replay of historical synthetic data".
- [ ] Pick a date and hour. The page shows the forecast issued at that moment.
- [ ] "What actually happened next" appears **below**, labelled as hindsight.
- [ ] Do the same on the Wait estimator: the department fields fill from the past
      and the comparison panel appears after you submit.
- [ ] Nothing anywhere suggests this is a live feed.

## 8. Window sizes

- [ ] At 1280×800 (laptop): no horizontal scrolling on any page.
- [ ] At about 1024px wide (tablet landscape): KPI cards wrap rather than squash.
- [ ] At about 800px wide: cards are two-up, tables scroll sideways inside their
      own box, the page itself does not scroll sideways.

## What this checklist does not cover

- Screen-reader output. The captions and labels are written for it, but no
  screen reader has been run against this app.
- Browsers other than the one you are using.
- Phone-sized screens (under about 700px). The layout is built for laptop and
  tablet; a phone is out of scope.
- Printing.
