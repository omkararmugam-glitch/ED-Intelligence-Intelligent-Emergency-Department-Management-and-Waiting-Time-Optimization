# Frontend redesign and additive backend

A clinical-operations redesign of the dashboard, plus read-only API endpoints to
feed it. Presentation layer and additions only: **no modelling code, config,
threshold, artifact or result is changed**, and the headline numbers are re-checked
at every milestone gate.

The institution on screen is fictional throughout: **Northfield General Hospital -
Emergency Department**. Every page carries the standing banner *"Synthetic data.
Decision support only, not for clinical use."*

Each milestone below records its own gate: the full test suite, `Start-Demo.cmd`,
and `tests/test_headline_regression.py`. A milestone that could not be finished
cleanly is reverted and reported rather than left half-built.

---

## Milestone 1 - Design system

**Gate: PASSED.** 1286 passed, 11 skipped, exit 0 (13m 25s). `Start-Demo.cmd` came
up with models loaded and the dashboard serving. Headline regression 10/10.

Test count moved from 1246 to **1286**: `tests/test_design_system.py` adds 34, and
six existing parametrized tests gained cases because `app/components.py` and
`app/icons.py` are new modules under `app/` that the dashboard rules apply to. No
existing test was modified.

### What was built

| File | What it is |
|---|---|
| `app/static/app.css` | The single stylesheet, 755 lines, injected once by `theme.inject_css()` |
| `app/icons.py` | 25 inline SVG icons, no colour of their own, nothing fetched |
| `app/components.py` | The component surface a page imports |
| `tests/test_design_system.py` | 34 tests over the palette, the signals and the offline promise |

**Palette.** Deep navy shell (`#14243a`, white text on it at about 13:1), teal
accent (`#0f7a7a`), white cards on a cool light grey (`#f4f6f9`), and a dark-theme
block that re-tunes the tokens so no component needs a dark variant of its own.

The accent is deliberately **not** one of the semantic colours. If teal also meant
"good", every focused control would look like a passing check and a reader would
have to work out which of the two things the colour was saying. Critical, warning,
ok and info are separate tokens, each used for exactly one meaning.

**Triage chips** follow the convention a clinician already knows -- 1 red, 2 orange,
3 yellow, 4 green, 5 blue -- and each prints its number *and* its word, "ESI 2
Emergent". The colour is the third copy of the signal, not the only one, so the chip
survives greyscale, deuteranopia and a reader who has never been taught the scheme.
ESI 3's yellow cannot carry white text at AA, so that one chip uses dark ink; the
exception lives in the data rather than in the component.

**Type scale** has five steps with one job each (page title, section title, card
label, big number, caption) and tabular numerals on every figure, including inside
Streamlit's own metric, so a column of numbers lines up and a changing KPI does not
jitter as its digits change width.

### Decisions

**`st.metric` was kept rather than replaced.** The widget brings Streamlit's own
screen-reader behaviour, and a dashboard test counts metrics to check the Command
center leads with its numbers. So the styling comes to the widget in
`static/app.css` instead of the widget being replaced by a `<div>`.

**Inline SVG rather than `:material/...:`.** Streamlit does bundle Material Symbols
locally, so the shortcode would also work offline -- but it is only expanded in
Streamlit's own markdown and label slots, not inside the HTML this project emits for
cards, chips and table rows. Half the icons on a page would have rendered and half
would have shown as literal `:material/x:` text.

**The palette exists twice, and that is checked.** Altair cannot read a stylesheet
and CSS cannot read Python, so the colours are Python constants for charts and custom
properties for everything else. `test_design_system.py` asserts the two agree token
by token, including all five ESI colours, because repetition that nothing checks is
repetition that drifts.

### What failed on the way

Three of the new tests failed on first run, all three being the test's fault:

1. The "nothing is fetched" check matched the stylesheet's own comment saying
   *"No @import"*. Fixed by stripping CSS comments first -- a test that cannot tell
   a rule from a sentence about the rules is not checking anything.
2. The icon check banned the substring `http` and hit
   `xmlns="http://www.w3.org/2000/svg"`, which is an XML namespace identifier and
   not a request. Now it looks for `src=`, `href=` and `url(`.
3. The compact-chip check forbade the acuity word, which legitimately stays in the
   chip's `title` so a dense table still offers the full label on hover. Now it
   asserts on the visible span and requires the tooltip to keep the word.

### Known gap carried into Milestone 2

**Per-item icons in the sidebar were not delivered.** Navigation is one `st.radio`
per group; a radio's option text *is* its value; and `format_func` changes what
`widget.options` returns -- which is exactly what
`test_the_sidebar_offers_exactly_the_pages_the_script_defines` compares and what
every parametrized navigation and language test passes to `set_value()`. Streamlit's
widget labels support Images but not material shortcodes, so even a data-URI icon
could only arrive through `format_func`. Decorating the labels would have meant
rewriting roughly four hundred test cases to carry presentation in their navigation
calls, which is a worse trade than a missing icon.

Instead each navigation **group** carries its icon in the caption, and each **page**
carries its icon in the header -- both plain HTML that renders reliably. The visual
intent is served; the literal requirement is not met, and is recorded here rather
than quietly dropped.

---

## Milestone 2 - App shell

**Gate: PASSED.** 1351 passed, 13 skipped, exit 0 (14m 17s). `Start-Demo.cmd` came
up with models loaded and the dashboard serving; the shell renders with no
exception. Headline regression 10/10.

Test count 1286 -> **1351**. Nothing was added: the growth is existing parametrized
tests gaining cases for the two new pages (eleven pages instead of nine, across
several per-page rules). Skips 11 -> 13 because `system_status.py` draws no Altair
charts, so the two conditional chart rules skip it exactly as they already skip
`about.py`.

### The navigation

| Group | Pages |
|---|---|
| **Operations** | Command center, Patient flow board *(new)* |
| **Planning** | Demand forecast, Staffing planner, Scenario lab |
| **Clinical intelligence** | Wait and risk prediction, Quality and equity, Research results |
| **Administration** | Data upload, System status *(new)*, About and limitations |

Four labels were renamed to the clinical names: Wait estimator -> Wait and risk
prediction, Model quality and fairness -> Quality and equity, Data and analysis ->
Data upload, What-if scenarios -> Scenario lab. "About and limitations" is reachable
under Administration rather than in a group of its own; the brief did not place it,
and every existing page stays reachable as required.

The top bar carries the hospital name and department, a **Replay time** chip, the
model and data version chips, and a `SYNTHETIC` environment chip. The default page
is still Command center.

### New modules

| File | What it is |
|---|---|
| `app/alerts.py` | The five alert rules, as data, with their thresholds and statements |
| `app/views/system_status.py` | Service health, versions, model card, alert rules, upload limits |
| `app/views/flow_board.py` | The patient flow board |

`app/alerts.py` holds the rules rather than scattering `if` statements through a
page, because an alert whose rule is hidden is indistinguishable from a guess. The
Command center cannot show an alert the System status page does not explain, since
both read the same dictionary. Every threshold in it is a **presentation**
threshold -- what is worth putting in front of a reader -- and none of them feeds a
model, a simulation or a staffing search, so changing one cannot move a published
figure. `evaluate()` returns an empty list when nothing is known, so an empty alerts
panel is never ambiguous between "all clear" and "not computed".

The flow board was built as the real page rather than a placeholder, since shipping
a stub in one milestone and replacing it in the next is wasted work. It reuses
`app/replay.py`, whose `knowledge_at` and `for_patient` already truncate the visits
table at the cutoff, so it is leakage-safe by construction; Milestone 4 adds the
test that holds it to that, plus any refinement.

### Decisions

**No continuous play on the board.** The controls step an hour at a time and jump to
the busiest hour. A page that advanced itself on a timer would make a recorded
extract look like a ticking feed, which is what
`test_no_main_view_claims_to_be_live` exists to prevent. The reader moves the clock.

**Estimates are fetched for the visible page only.** A busy hour leaves two hundred
people waiting, and asking the service for two hundred individual estimates to fill
one screen would make the page unusable. The board paginates at 25 rows and says so
in its own caption.

**The aging rule and the roster are mirrored, not read.** The dashboard image
installs `requirements-app.txt`, which has no PyYAML in it, so a dashboard that
opened the analysis config would crash the container. `AGING_MINUTES_PER_LEVEL` and
`AGING_FLOOR` therefore live in `app/config.py` with a test asserting they still
equal the config's values, and the roster comes from `sim_calibration.json`.

### What the renames cost, and what failed

Only **three** tests broke on the renames, because almost everything navigates
dynamically from the script's own page list. Those three hardcoded labels were
updated, along with `_shared.ANALYSIS_PAGE` -- which is exactly the indirection its
own comment had anticipated.

One failure was mine: the System status page tripped the "no Python `None` on a
page" guard with a sentence that legitimately **began** with the word "None". The
sentence was reworded rather than the guard weakened.

---

## Milestone 3 - Command center

**Gate: PASSED.** 1351 passed, 13 skipped, exit 0 (17m 19s). `Start-Demo.cmd` came
up with models loaded and the dashboard serving (HTTP 200). Headline regression
10/10. Test count unchanged at 1351: no test was added or removed.

The page was **extended rather than rewritten**. `_congestion` and
`_summary_markdown` are called directly by
`test_the_command_center_summary_is_a_readable_handover_note`, so both keep their
signatures and the handover note keeps working.

### Row 1 - where the department stands

Six cards, each naming what its delta is measured against. Rendered against the
running service:

| Card | Value | Status |
|---|---|---|
| Patients waiting now | 3 | good, with a 12-hour sparkline |
| In treatment | 25 | never coloured as bad |
| Longest wait | 28 min | good; act above 90 |
| Median door-to-doctor, Tue | 40 min | **act** -- above the 30-minute target |
| Left without being seen | 2.3% | good -- within 4.0% |
| Congestion risk, next 6 h | Low | from the existing forecast model |

Deltas are **against the same hour last week**, because an emergency department's
load is strongly weekly: 18:00 Friday against 17:00 Friday says almost nothing.
When a week earlier falls outside the extract the card reads "vs overall average"
rather than silently changing meaning -- a delta whose baseline is unstated is
worse than no delta.

"In treatment" is deliberately never coloured as bad. More patients in treatment is
the department working, and a card that reddened as the department got through its
queue would teach the reader to ignore the colours.

### Rows 2 to 4

**Row 2** is the arrivals forecast with its interval band, amber busy-hour markers
and a new **seasonal-naive line** -- what repeating the same hour last week would
have predicted -- so the baseline the forecast is measured against is visible
rather than described. It is computed from arrivals a week before each forecast
hour, which already existed when the forecast was issued, so drawing it adds no
hindsight. Beside it, the waiting room's **acuity mix** as a ring in the ESI
palette.

**Row 3** is waiting time by acuity, middle and 90th percentile side by side
against a dashed **30-minute target line**, and the congestion **heatmap** of day of
week against hour. The two together answer "is the wait acceptable" and "when does
it stop being acceptable".

**Row 4** is the alerts panel, driven by `app/alerts.py`. Each alert shows its
severity pill, the rule that fired and the dataset moment, and every rule is listed
on the System status page.

### Two defects found and fixed

**Identity columns were empty everywhere.** `replay.load_visits()` requests five
columns; `patient_id`, `age`, `arrival_mode` and `is_lwbs` are in the parquet but
were never read, so every board row read "unknown" and the left-without-being-seen
card read "no data". Fixed with a new `replay.load_visits_detailed()` rather than by
widening the shared `REQUIRED_COLUMNS`, because `load_visits` returns None when a
requested column is missing -- widening it would turn a cosmetic gap in somebody's
uploaded extract into a blank dashboard.

**The hero page opened on a row of zeros.** The default moment was the extract's
last timestamp, where the waiting room is empty by construction: everyone who
arrived earlier has been seen, and the final arrival is excluded by the strict
before-the-cutoff rule. The default is now the **midpoint of the held-out window**,
which is representative rather than flattering -- it is not chosen for being busy,
and the replay control still reaches any other instant.

### And one I caused

A 745-second dashboard test run and an order-dependent failure on an unrelated
page, both from the Milestone 1 test that swapped `components._html` for a capture
list and then **reloaded the module** to undo it. A reload installs a new module
object in `sys.modules` while every page that had already run
`from app import components` keeps the old one -- still holding the capture list in
place of `st.markdown`. Replaced with save and restore of the single attribute. The
same combination of test files now runs in **75 seconds** and passes, and the
reason is written into the helper's docstring so the shortcut is not taken again.

---

## Milestone 4 - Patient flow board

**Gate: PASSED.** 1364 passed, 13 skipped, exit 0. `Start-Demo.cmd` came up from a
stopped state with models loaded and the dashboard serving; the board renders with
census and doctors-on-duty. Headline regression 10/10.

Test count 1351 -> **1364**: `tests/test_flow_board.py` adds 13.

The run took 33 minutes rather than the usual 14 because the demo from the previous
gate was still running and competing for the CPU. Nothing failed; the wall time is
contention, not a regression.

The board itself shipped in Milestone 2, so this milestone's deliverable is the
leakage proof the brief asks for, plus the two defects Milestone 3 found in it
(empty identity columns, and the end-of-extract default).

### The test that matters

`test_truncating_the_future_does_not_change_the_board` does exactly what was
specified: it builds the board from the whole extract, builds it again from an
extract that physically stops at the cutoff, and requires the two to be identical
in membership, waits and ordering. A second test applies the same treatment to the
queue-state payload sent for each estimate, because an estimate's inputs have to be
as leakage-safe as the row it sits on.

The fixture is hand-built rather than loaded from the extract, so it states its own
answer: ten patients covering waiting, in treatment, discharged, left without being
seen, and four who arrive **after** the cutoff and must never appear anywhere.

### What else is pinned

- **The ordering is the department's rule, not the clock.** An ESI 4 who has waited
  two hours comes before a fresher ESI 4; an ESI 2 who arrived ten minutes ago comes
  before both.
- **Aging cannot overtake a fresh high-acuity arrival.** A two-day wait still sits at
  the 2.5 floor, which is a clinical safety constraint rather than a tuning knob.
- **The board ages at exactly the simulator's rate**, so it cannot show a queue the
  staffing plan does not believe in.
- A missing triage level and a missing arrival mode are normal (the generator drops
  0.4% and 3% on purpose) and neither breaks a row or prints "nan".
- The page size stays small, because the board asks the service for one estimate per
  visible row and that is only affordable while the page is a screenful.

Verified against the real extract at the default moment: three waiting, ordered
2.88 / 3.32 / 4.44 by effective priority, with their own generated identifiers and
arrival modes.

### One interpretation worth stating

The brief says to use the existing feature builder. The board goes through
`replay.for_patient`, which is this project's existing leakage-safe queue-state path
and produces exactly the payload the service expects. `src/features` is not
reachable from the dashboard image -- `requirements-app.txt` has neither PyYAML nor
LightGBM -- so importing it would have broken the container rather than improved
correctness.

### Still not delivered

Continuous play, as recorded in Milestone 2: the controls step and jump, they do not
animate. A page that advanced itself on a timer would make a recorded extract look
like a ticking feed.

---

## Milestone 5 - Remaining pages

**Gate: PASSED on the second run.** 1364 passed, 13 skipped, exit 0 (16m 06s).
`Start-Demo.cmd` up with models loaded, dashboard serving HTTP 200, headline
regression 10/10. Test count unchanged at 1364.

### What changed on every page

All seven remaining pages now draw `components.page_header` with their breadcrumb,
their icon and the standing disclaimer, so the shell reads the same whichever page
a reader lands on. Titles follow the new navigation. Every page already had the
loading, empty and error states -- `_shared.load` has handled all three since the
upload work -- and `showErrorDetails = "none"` in `.streamlit/config.toml` keeps a
traceback off the screen by configuration rather than by discipline.

### Staffing planner - two real gaps

The **72-of-144 feasibility summary** was buried inside a "How this was worked out"
expander. It is now in the section header, where a reader sees it without clicking.

The **robustness caveat was missing entirely**, which was the more serious of the
two: the page recommended a roster and said nothing about how far that roster
travels. It now reads its figures from the seed study rather than quoting typed-in
numbers -- the method transfers, the gate **passed on 8 of 10** regenerated datasets
and refused a roster on the rest, and where it did recommend one the cost spanned
**43,890 to 54,670 across 7 different rosters** with the day shift identical in
every one. It closes on the point that matters: read the shape of the answer, not
the roster.

### Wait and risk prediction - a feature, not a restyle

The page estimated the wait and never the walkout risk. `api_client.predict_risk`
was added (the `/predict-risk` endpoint already existed) and the page now shows the
chance of leaving before being seen at 30, 60 and 120 minutes as cards with status
pills.

The two belong together: "about two hours" and "roughly one in twenty of these
patients walks out" are the same situation described twice, and the second is the
one that prompts a decision. An acuity outside the calibrated range gets the
service's own notice and an instruction not to act on the figure. A failed call is a
quiet absence rather than an error card, because the waiting estimate above is the
page's answer and context that did not load should not look like breakage.

### Scenario lab - an ambiguity the rename created

Renaming "What-if scenarios" to "Scenario lab" put two different things under one
name: this page's scenarios run through the simulation that **passed** its
validation check, while the bed, nurse and imaging experiments on Research results
have assumed parameters and carry a not-calibrated banner. The page now says which
kind it is and where the other kind lives, instead of leaving a reader to wonder why
one set carries a warning and the other does not.

### Data upload

No change was needed for the brief's items: `st.file_uploader` is already a
drag-and-drop zone, and the four-step stepper with per-stage progress was already
there from the upload work.

### The gate failure, which was mine and was not Milestone 5's

The first gate run failed on `test_no_patient_identifier_reaches_the_log`, a test
written in an earlier session with a latent flaw: it searched the whole log for the
string `"35"`, the age in its fixture, and matched the service's own request
duration, `-> 200 in 35.7 ms`. **No age was ever logged.** It had passed on faster
timings and tripped now by luck.

Reverting Milestone 5 would have been the wrong call -- the milestone's work was
clean and 485 dashboard tests had already passed on it -- so the test was fixed
instead. It now strips durations, request ids and httpx's own lines before
asserting, and still checks the positive half, that acuity and queue depth **are**
logged, so it has not been weakened into vacuity. Two further mistakes of mine on
the way: a heredoc turned `"\n"` into a real newline and broke collection, and the
replacement assertion `"age" not in log` matched the "age" inside `triage=4`. The
final form uses `\bage\s*=`, which cannot match `triage=`. Verified stable over five
consecutive runs.

A privacy test that cries wolf gets muted, which is worse than not having one, so
the reason is written into the test's docstring.

---

## Milestone 6 - Backend, additive

**Gate: PASSED.** 1391 passed, 13 skipped, exit 0 (15m 49s). `Start-Demo.cmd` up,
all five new routes answering in the running service, `/health` unchanged in shape,
headline regression 10/10. Test count 1364 -> **1391**:
`tests/test_analytics_api.py` adds 27.

### The routes

| Route | What it returns |
|---|---|
| `GET /v1/analytics/overview` | The six Command center KPIs, per-acuity medians and fired alerts; `?at=` optional |
| `GET /v1/analytics/flow-hourly` | Arrivals and the reconstructed census per hour; `?days=1..28` |
| `GET /v1/analytics/acuity-summary` | Median and p90 wait, walkouts, share over target, per acuity |
| `GET /v1/board` | The tracking board at a moment; `?at`, `?limit`, `?offset` |
| `GET /v1/meta` | Versions, data window, row counts, roster, and every alert rule |

Verified against the real extract: 57,179 visits, roster
`day=3, evening=3, night_early=2, night_late=2`, ESI 3 median 14.0 and p90 113.0
with 39.2% over target -- the same figures `artifacts/report.md` carries.

`/v1` rather than the bare path because the existing routes are unversioned and
must stay so; `/health` keeps its own unversioned path, and a test asserts
`/v1/health` is a 404.

### Cross-cutting

- **Request id** on every response, and now in the error envelope as well, taken
  from the same `request.state` the header is set from -- so a bug report cannot
  quote an id that does not match the log.
- **One error envelope** `{error, detail, request_id}` for a refusal, a validation
  failure and an unhandled exception alike. FastAPI's default validation body is a
  list of pydantic dicts, which is precise and unlike every other error this
  service returns; the handler flattens it into a sentence that names the field.
- **Caching keyed on the data version.** `flow-hourly` reconstructs the census at
  the top of every hour over a week, which is 169 truncations of a 57,000-row
  table. A retrain changes the key and the old payloads are dropped.
- **CORS restricted to the four loopback origins**, no credentials, `GET`/`POST`/
  `DELETE` only. Both services bind to 127.0.0.1, neither authenticates by default
  and the API accepts uploads, so a wildcard origin would let any page the reader
  happens to be browsing read this department's data.
- **gzip** above 500 bytes, below which the header costs more than it saves.
- **`protected_namespaces=()`** on 26 models across `api/schemas.py` and
  `src/serving/api.py`. The `model_version` warnings are gone; importing the app
  under `-W error::UserWarning` is now silent.

### A layering error of mine, caught here

`app/alerts.py` claimed the API imported it. **It cannot.** The Dockerfile copies
`src/` and not `app/` into the API image, and `app/` and not `src/` into the
dashboard image. The rules therefore exist twice -- `src/serving/analytics.py` for
the service, `app/alerts.py` for the dashboard -- mirrored exactly as this project
already mirrors `api.auth.API_KEY_HEADER` and the simulator's aging constants, and
three tests compare them key for key, threshold for threshold, title for title.
One of those tests pins the property that matters most: **neither copy will claim
"all clear" when it had nothing to evaluate**, so an empty alerts panel can always
be told apart from a quiet department.

### A real bug the cache test found

Eviction only ran on a cache **miss**. After a retrain, a route that keeps being hit
at the new version never misses, so the previous extract's payloads would have sat
in memory for the life of the process -- which is exactly what the cache's own
comment said would not happen. Eviction now runs on every call. The code was fixed
rather than the test.

### The dashboard consumes it

The Command center now renders the service's alert evaluation instead of its own, so
the dashboard and any other caller cannot disagree about whether an alert is firing.
The local reconstruction stays as the fallback, because this dashboard is useful
against saved tables with the service down and losing the whole panel to a
connection error would be the worse page. `app/api_client.py` gained five calls, and
the test stub learned all five routes so
`test_a_page_only_calls_endpoints_the_api_actually_has` still catches a typo'd URL.

### Not done

The board page still reconstructs locally rather than calling `/v1/board`. The
endpoint exists, is tested and is paginated, but the page's own path is already
leakage-safe and covered by `tests/test_flow_board.py`, and swapping it late in the
work would have put the one page with a correctness proof behind an untested change.

---

## Milestone 7 - Finish

### Documentation

The README's page table was replaced with all **eleven pages in the four new
groups**, each saying what it answers, and the five `/v1` routes were added to the
endpoint table with a note that they are read-only additions guarded by a contract
test. The stale references went with them: Shift replay now names the pages it
actually sits on, the screenshot list names the Patient flow board, and "Data and
analysis" is "Data upload" throughout. **No occurrence of an old page name remains
in either document.**

`SUBMISSION.md` gained the same page list and the **five-minute demo path**: Command
center, the Patient flow board at a busy hour, Staffing planner, Scenario lab,
Quality and equity -- each step naming what to look at rather than just where to
click.

### Stale claims the audits caught

Both documents still said **1246 tests passed, 11 skipped**. The real figures are
**1391 and 13**. The number audit had been agreeing with the stale README because
its own expectation was equally stale, which is the failure mode a mirrored constant
always has. Fixed in the documents, in the audit, and in
`tests/test_demo_launcher.py`, which pins the documented count so it cannot rot
unnoticed again.

The audit also grew: **78 claims, 0 mismatches**, now including the two new pages,
the renamed pages and all five new routes.

`package_check.py`: **no problems**, 41 of 41 required entries present, with the two
standing notes it has always had -- `.env` holds configuration and no credentials,
and `artifacts/pipeline.log` is gitignored by design.

### The package, verified from a clean unzip

```
dist\ed-project-submission.zip
14.1 MB compressed, 30.3 MB on disk, 347 files
```

It carries the redesign: `app/static/app.css`, `app/components.py`, `app/icons.py`,
`api/analytics.py`, `src/serving/analytics.py`.

| Step | Result |
|---|---|
| `Setup.cmd` on a tree with no `.venv` | exit 0 in 450 s; "fastapi, uvicorn, streamlit, httpx, lightgbm, lifelines, simpy all import" |
| `tests/test_headline_regression.py` | **10 passed** |
| Full suite in the package | **1391 passed, 13 skipped**, exit 0 (16m 02s) -- identical to the main project |
| `Start-Demo.cmd` from the unzipped folder | API healthy with models loaded, dashboard serving HTTP 200 |
| `/v1/meta`, `/v1/analytics/overview`, `/v1/board` in that instance | 200, 200, 200 |
| `Stop-Demo.cmd` | both PIDs stopped with their children, ports free |
| Temp folder | deleted |

### Docker: not rebuilt

**The container image predates the redesign.** The Docker engine was not running at
the end of this work, and a rebuild plus a trainer run is a twenty-minute round trip
that would have been verified at the very end of a long session rather than
carefully. The images therefore do not contain `app/static/app.css`,
`app/components.py`, `app/icons.py`, `api/analytics.py` or `src/serving/analytics.py`.

What this means in practice: `scripts\docker_up.ps1` will start the **previous**
dashboard and the **previous** API, without the new shell, the new pages or the
`/v1` routes. `Start-Demo.cmd` runs from the source tree and is unaffected, which is
why it is step 1 of the documented path. Rebuilding with
`.\scripts\docker_up.ps1 -Build` is the remedy and is expected to work -- the
Dockerfile copies `app/`, `api/` and `src/` wholesale, so the new files are inside
the existing COPY lines and no Dockerfile change is needed. **That expectation is
untested, and is recorded here as untested.**

### Final verification

**Gate: PASSED.** 1391 passed, 13 skipped, exit 0 (16m 03s) on the main project.
`Start-Demo.cmd` up with models loaded and the dashboard serving.

Every headline number, read back one last time:

| Figure | Required | Measured |
|---|---|---|
| wait MAE, model vs baseline | 11.52 vs 34.23 | **11.524 vs 34.233** |
| conformal, rolling vs static | 0.8875 vs 0.849 | **0.8875 vs 0.849** |
| forecast MAE vs seasonal-naive | 2.356 vs 2.979 | **2.3557 vs 2.9792** |
| cheapest plan | 3/3/2/2 at 49,280 | **day=3, evening=3, night_early=2, night_late=2 at 49,280** |
| plans feasible | 72 of 144 | **72 of 144** |

**Nothing moved.** No modelling code, config, threshold, artifact or result was
changed in this work: the only files touched under `src/` were `src/report.py`
(wording, in the earlier pass) and the two additions `src/serving/analytics.py` and
`protected_namespaces=()` on existing response models, none of which computes a
figure.

---

## Test count, milestone by milestone

| Milestone | Passed | Skipped | What moved it |
|---|---|---|---|
| start | 1246 | 11 | -- |
| 1 Design system | 1286 | 11 | +34 design-system tests, +6 parametrized cases for the new modules |
| 2 App shell | 1351 | 13 | parametrized cases for two new pages; `system_status.py` draws no charts, so two conditional rules skip it |
| 3 Command center | 1351 | 13 | nothing added |
| 4 Flow board | 1364 | 13 | +13 flow-board tests |
| 5 Remaining pages | 1364 | 13 | nothing added |
| 6 Backend | 1391 | 13 | +27 analytics and contract tests |
| 7 Finish | 1391 | 13 | nothing added |

Four existing tests were edited, all for renamed page labels, plus one pre-existing
flaky privacy test that was made precise. No test was deleted or weakened.

## Known gaps, in one place

1. **Per-item icons in the sidebar.** Group captions and page headers carry icons;
   the navigation items do not. A radio's option text is its value, and decorating
   it would have meant rewriting roughly four hundred parametrized test cases to
   carry presentation in their navigation calls.
2. **No continuous play on the flow board.** It steps and jumps. A page that
   advanced itself on a timer would make a recorded extract look like a ticking
   feed.
3. **The Docker image predates the redesign**, as described above. `Start-Demo.cmd`
   is unaffected.
4. **The flow board page still reconstructs locally** rather than calling
   `/v1/board`. The endpoint exists and is tested; the page's own path has a
   correctness proof and was left behind it.
5. **No human has opened the redesigned dashboard in a browser.** Every page is
   rendered and asserted through Streamlit's `AppTest`, and the live instance
   serves HTTP 200, but the visual result -- contrast, wrapping at 1366x768, the
   colour of a chip against a card -- has been reasoned about and not seen.
