# Robustness with the true generator, richer resources, real-data readiness, competing risks and RL

What was run, and what came back. **Every number below was read from an artifact file,
from test output, or from a running container.** Nothing is estimated. Failures and
negative results are in here with the rest.

Date: **2026-10-12**. Windows 11, PowerShell 5.1, `C:\dev\ed-project`, `.venv`
(Python 3.12), Docker 29.7.2. The OneDrive copy was not touched.

---

## The three stop conditions

| Condition | Status |
|---|---|
| A headline number moves | **Not triggered.** Checked after Parts 0+A, after the containerised trainer, and in the final run. |
| The original generator no longer reproduces `data/raw/ed_v2.csv` | **Not triggered.** Verified first, before any other work. |
| A part would need real patient data | **Not triggered.** Part B prepares for MIMIC-IV-ED without downloading it. |

### The generator reproduces ed_v2.csv

```
python scripts\original_generator\generate_ed_data.py --days 365 --seed 42 --scale 9.7 --out <tmp>
```

```
shapes            (57500, 10) == (57500, 10)
columns identical True
DataFrame.equals  True
```

The files are not byte-identical, and the reason is only line endings: the committed
file is LF, `to_csv` on Windows writes CRLF, the size differs by exactly 57,501 bytes
(one `\r` per line), and the two are equal after newline normalisation. **Content is
identical**, which is the condition as stated (`pandas .equals`).

### Headline numbers, before and after

Measured by a full pipeline run on `data/raw/ed_v2.csv`, at full precision.

| Figure | Required | After Parts 0+A | After the containerised trainer (all 16 stages) |
|---|---|---|---|
| wait MAE, model | 11.52 | **11.524** | **11.524** |
| wait MAE, baseline | 34.23 | **34.2327** | **34.233** |
| rolling conformal coverage | 0.8875 | **0.8875** | **0.8875** |
| static conformal coverage | 0.849 | **0.849** | **0.849** |
| forecast MAE | 2.356 | **2.3557** | **2.3557** |
| forecast, seasonal-naive | 2.979 | **2.9792** | **2.9792** |
| cheapest plan | 3/3/2/2 at 49,280 | **day=3, evening=3, night_early=2, night_late=2 at 49,280** | **identical** |
| plans feasible | 72 of 144 | **72 of 144** | **72 of 144** |

`tests/test_headline_regression.py` pins all of them and passed at every checkpoint.

---

## PART 0 — Robustness with the true generator

`scripts/robustness_study.py` loads the true generator with `importlib` **from its own
file**, never a copy, and runs the full pipeline on seeds 1-10 plus seed 42 as the
reference. Each seed owns `runs/robustness/seed_<S>/`; nothing was written to
`data/raw/`. Resumable (a seed with a `summary.json` is skipped) and run three at a
time as subprocesses.

**Worker threads are pinned to 2.** Three workers each grabbing every core thrash, and
a result that depends on what else was running is not reproducible. The reference-seed
check is what confirms the pin moves nothing: seed 42 reproduced all seven headline
figures under it.

### Per-seed results

| Seed | Rows | Wait MAE | Baseline MAE | Static | Rolling | Forecast | Naive | Gate | Cheapest plan | Cost | Feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 56,671 | 9.99 | 27.15 | 0.9121 | 0.9067 | 2.253 | 2.608 | pass | 3/3/1/2 | 43,890 | 80 |
| 2 | 56,778 | 10.92 | 32.97 | 0.8541 | 0.8854 | 2.315 | 2.969 | pass | 3/3/2/2 | 49,280 | 75 |
| 3 | 57,115 | 11.51 | 34.79 | 0.8657 | 0.8955 | 2.537 | 3.080 | pass | 3/3/2/1 | 48,510 | 73 |
| 4 | 57,129 | 10.69 | 32.80 | 0.8625 | 0.8850 | 2.486 | 2.816 | pass | 3/4/1/1 | 49,280 | 71 |
| 5 | 58,180 | 12.05 | 35.85 | 0.8790 | 0.8962 | 2.880 | 3.233 | pass | 3/4/2/1 | 54,670 | 61 |
| 6 | 57,258 | 10.95 | 34.28 | 0.8788 | 0.8965 | 2.397 | 3.010 | pass | 3/3/1/3 | 44,660 | 83 |
| 7 | 57,037 | 11.02 | 31.60 | 0.8951 | 0.8968 | 2.657 | 3.021 | **pass** | 3/3/1/2 | 43,890 | 80 |
| 8 | 57,562 | 10.14 | 30.33 | 0.9082 | 0.9000 | 2.292 | 2.941 | **FAIL** | refused | — | — |
| 9 | 57,094 | 9.92 | 31.40 | 0.8801 | 0.8946 | 2.588 | 3.035 | pass | 3/4/1/3 | 50,820 | 64 |
| 10 | 57,148 | 9.23 | 28.56 | 0.9426 | 0.9137 | 2.329 | 2.667 | **FAIL** | refused | — | — |
| **42 (ref)** | 57,500 | **11.524** | **34.233** | **0.849** | **0.8875** | **2.356** | **2.979** | pass | **3/3/2/2** | **49,280** | **72** |

### Mean, standard deviation and range across seeds 1-10

The reference seed is excluded from these: it *is* the dataset every headline number was
tuned on, so including it would flatter the spread.

| Metric | Mean | SD | Min | Max |
|---|---|---|---|---|
| wait MAE, model | 10.641 | 0.833 | 9.229 | 12.047 |
| wait MAE, baseline | 31.973 | 2.750 | 27.149 | 35.847 |
| conformal, static | 0.8878 | **0.0270** | 0.8541 | 0.9426 |
| conformal, rolling | 0.8970 | **0.0086** | 0.8850 | 0.9137 |
| forecast MAE | 2.474 | 0.197 | 2.253 | 2.880 |
| forecast, seasonal-naive | 2.938 | 0.191 | 2.608 | 3.233 |
| worst held-out relative error | 0.343 | 0.144 | 0.211 | 0.719 |
| cheapest plan cost | 48,125 | 3,795 | 43,890 | 54,670 |
| feasible plans | 73.4 | 7.8 | 61 | 83 |

### (a) Does the seed-7 gate failure reproduce?

**No.** Seed 7 **passes** with the true generator, with a worst held-out relative error
of 0.279.

The earlier finding came from the wrapper around `src/data/synthetic.py`, and the two
studies are not comparable even in principle: the wrapper's generator models total
post-contact time where the true one models physician-busy time (32/26/20/12/8 minutes
by ESI), so "seed 7" does not denote the same dataset in both. **That specific finding
was an artefact of the wrapper and is withdrawn.**

**What does reproduce is the general phenomenon.** 2 of 10 seeds fail the gate, for two
different reasons:

- **Seed 8** — abandonment rate ESI 3, simulated 0.028 against observed 0.016.
  Relative error 0.719 against a 0.50 tolerance, absolute error 0.012 against a 0.01
  floor. *Diagnosis: a tolerance artefact.* The observed rate is 1.6%, so a 1.2
  percentage-point miss is a 72% relative error while being clinically trivial. The
  absolute floor exists for exactly this and is set just too low to catch it.
- **Seed 10** — mean wait ESI 3, simulated 26.3 against observed 34.4. Relative error
  0.236 against 0.20, absolute error 8.1 minutes against a 3-minute floor.
  *Diagnosis: a genuine fit failure.* An 8-minute miss on a 34-minute mean wait is a
  real disagreement, not a small-sample effect, and the gate is right to refuse.

**Proposed, NOT applied:** raise `absolute_floor_rate` from 0.01 to about 0.015, or make
the abandonment floor scale with the observed rate — pass when
`|simulated - observed| <= max(0.01, 0.5 x observed)`. That would clear seed 8 and leave
seed 10 failing, which is the discrimination you want. It is not applied: loosening a
gate to make a failure go away is the move this project's design exists to prevent, and
the change should be argued for on its own.

A caution on the table above: `worst held-out relative error` is the largest relative
error among held-out checks, which is **not** the check that failed. Abandonment rows
carry a 0.50 tolerance and wait rows 0.20, so seed 10 fails at 0.236 while its largest
relative error is 0.327 on a row that passes.

### (b) Does the cheapest-plan cost spread reproduce?

**Yes.** Observed **43,890 to 54,670** (spread 10,780) against the earlier
43,120 to 55,440 (spread 12,320). The ranges overlap and are of the same size, and
**7 distinct rosters** appear across the 8 seeds that produced one — only seeds 1 and 7
agree. The day shift is 3 in every single case; everything else moves.

That count was wrong in the first draft of this report, and so was the artifact behind
it: `n_distinct_plans` counted the refusal string
`refused (simulation failed validation)` as an eighth plan. Fixed in `answer_questions`,
which now tallies plans only over seeds that produced a costed one and reports
`n_seeds_with_a_plan` beside it. Counting a refusal as a staffing plan is precisely the
error the refusal exists to prevent.

The conclusion stands and is strengthened by being reproduced with the true generator:
**the staffing method transfers, the specific roster does not.** 49,280 is a property of
`ed_v2.csv`.

### What held on every seed

- The wait model beats its baseline on all 11 datasets, by a factor of 2.7x to 3.1x.
- The forecast beats seasonal-naive on all 11.
- **Rolling conformal is three times more stable than static** (SD 0.0086 against
  0.0270). Static coverage ranges 0.854 to 0.943; rolling stays inside 0.885 to 0.914.
  The defensible claim is not "rolling is always closer to nominal" but "rolling lands
  in the same place whatever dataset you get".

### The `--no-noise` comparison

Seed 42 regenerated without the generator's realistic-dirt step (2% missing age, 3%
missing arrival mode, 5% missing complaint, 0.4% missing triage, 0.3% swapped
timestamps):

| Metric | With noise | No noise | Change |
|---|---|---|---|
| wait MAE, model | 11.524 | 11.542 | +0.16% |
| wait MAE, baseline | 34.233 | 34.140 | −0.27% |
| classifier PR-AUC | 0.9823 | 0.9826 | +0.03% |
| conformal, static | 0.8490 | 0.8484 | −0.07% |
| conformal, rolling | 0.8875 | 0.8871 | −0.05% |
| forecast MAE | 2.3557 | 2.4389 | **+3.53%** |
| cheapest plan cost | 49,280 | 49,280 | **0.00%** |
| feasible plans | 72 | 70 | −2.78% |

**The pipeline is essentially insensitive to the data-quality noise.** The validator
catches the missing values and swapped timestamps and the models carry on. The forecast
is the only thing that moves more than 1%, and the cheapest plan costs the same while
being a *different roster* (`day=3, evening=4, night_early=1, night_late=1`) — the cost
surface has ties.

### A bug this study found

Seed 1 originally crashed the survival stage with a bare
`TypeError: NaNs were detected in the dataset` from lifelines, 90 seconds in. One
patient arrived at 23:59 on the final day, was flagged LWBS, and had their departure
timestamp cleared by the extract window — so `is_lwbs` and `still_waiting_at_cutoff`
were both true and the duration was NaN. `ed_v2.csv` contains no such row.

Fixed in `build_waiting_room_events`: an abandonment with no departure timestamp has no
event time, so it is right-censored at the cutoff rather than counted as an observed
event. A guard now raises a message naming the cause rather than letting lifelines fail
several frames away. Two regression tests. Seed 1 re-ran clean and passes the gate.

---

## PART A — Richer resources in the simulator

Behind `simulation.resource_extensions`, **every field off by default**: exam
rooms/beds as a second dispatch constraint, triage nurses with reassessment, an
imaging/lab turnaround for ESI 1-3, and boarding that holds a bed (exit block).

### Off means off, proved rather than asserted

The output of 5 replications was captured **before** the DES was touched and compared
after:

```
patients   before 4915 | after 4915
sha256     before 9ac5068dae254a57... | after 9ac5068dae254a57...
PATIENT RECORDS IDENTICAL: True
KPIS IDENTICAL: True
```

The mechanism that makes this hold: extensions draw from their **own** RNG stream
(`extras_rng`), never `self.rng`, so the base model's sequence of draws is untouched
whether a flag is on or off. `tests/test_resource_extensions.py` asserts that no
extensions, disabled extensions and *enabled-but-unconstrained* extensions all produce
identical patients and identical KPIs.

### The scenarios

Bed counts are not invented: a probe run measures the calibrated model's own concurrent
bed occupancy (mean 12.6, p50 13, p90 19, max 27) and the scenarios use its median. 30
replications, common random numbers, paired intervals.

| Rank | Scenario | ESI 3 p90 | ESI 4 p90 | ESI 5 p90 | Mean absolute effect | Significant |
|---|---|---|---|---|---|---|
| 1 | `boarding_surge` | +199.6 | +176.7 | +167.5 | **181.3 min** | 3 of 3 |
| 2 | `imaging_delay_30` | +93.5 | +84.0 | +79.6 | **85.7 min** | 3 of 3 |
| 3 | `beds_limited` | +78.9 | +70.7 | +69.1 | **72.9 min** | 3 of 3 |

Each is one change on top of the one below it, so differencing isolates the lever:
beds alone cost 72.9 minutes, doubling boarding adds a further ~108, and imaging adds
~13. **Exit block dominates**, by a wide margin.

**A first version of this returned exactly 0.00 minutes for both bed scenarios.** The
probe measured bed-hours including boarding, but the model released the bed at the end
of treatment, so bed demand equalled the number of doctors — two or three — and no
realistic bed count ever bound. The scenarios now hold the bed until the patient
leaves, which is both what a bay is and what the probe measures.

**Everything here is labelled `SCENARIO ANALYSIS, NOT CALIBRATED`** — on the KPI rows,
in the artifacts, in the API response and on the dashboard. The data has no bed, nurse
or imaging column, the parameters are assumed, and the validation gate cannot test any
of it. The ranking is the usable result; the magnitudes are not.

---

## PART B — Real-data readiness

**No MIMIC-IV-ED data was downloaded, accessed or inspected.**

- `docs/real_data_readiness.md` — the join (keyed on `stay_id`, with the
  `subject_id` and `pyxis` traps named), what the mapping loses, and the five things
  that must happen before any real-data claim: PhysioNet credentialing and the DUA,
  de-identification discipline, retraining end to end, re-running the validation gate
  **without widening it**, and re-establishing the headline numbers from scratch.
- `configs/column_map_mimic_ed.yaml` — a template, named
  `mimic_iv_ed_TEMPLATE_UNVERIFIED`, with every column name flagged as a claim to check.

### The central problem

MIMIC-IV-ED has **no first-provider-contact timestamp**. The template maps `seen_time`
to a column that does not exist, so an unconfigured run fails loudly. The adapter's new
`proxy_wait` can derive it from a proxy, and **refuses to run without a proxy named in
words**, because every number downstream inherits the choice:

| Proxy | Error direction | Who is missing |
|---|---|---|
| first `vitalsign.charttime` | **under**-estimates: charted at triage, before a clinician | patients with no vitals |
| first `pyxis.charttime` | **over**-estimates: follows a clinical decision | everyone not given a drug — a biased subset correlated with acuity |

`tests/test_mimic_adapter.py` proves the two bracket the truth from opposite sides on a
30-row hand-written fixture with MIMIC's column names and vocabulary. Every result from
a proxy is labelled **PROXY WAIT (NOT TRUE DOOR-TO-PROVIDER)** in the adapter log, in
`metrics/dataset_shift.json` and at the top of the report.

A defect the fixture caught in my own template: `UNKNOWN` was in `na_values` *and* in
the `arrival_transport` value map, so a real category was being blanked before the map
could see it. Removed from `na_values`.

### Dataset-shift check

`src.data.shift_check` compares a loaded extract with the synthetic profile on the four
things that would change what the right model is, and runs automatically in
`stage_validate` for `--data real`:

| Comparison | Statistic | Flagged above | On `ed_v2` itself |
|---|---|---|---|
| arrival-hour shape | total variation distance | 0.10 | 0.0003 |
| acuity mix | total variation distance | 0.05 | 0.0001 |
| median wait (acuity-weighted) | relative change | 0.25 | 0.0000 |
| LWBS rate | absolute change | 0.02 | 0.0000 |

A warning, never a gate. **It initially reported 45% drift on the reference dataset
against itself**, because the stored profile holds per-acuity medians while the data has
a pooled median. Both sides are now the same statistic, and the identity case reads zero
on all four — which is the test that caught it.

---

## PART C — Competing risks at arrival

A LightGBM classifier per horizon predicting the **cumulative incidence** of leaving
without being seen — not a hazard. A patient seen before the horizon is a negative, not
a censoring: censoring them would answer the counterfactual "would they have left had
they kept waiting", which is not what anyone is asking.

### Test block, overall

| Horizon | n | Who left | AUC | Brier raw | Brier calibrated | ECE raw | ECE calibrated |
|---|---|---|---|---|---|---|---|
| 30 min | 8,619 | 23 | 0.825 | 0.002662 | 0.002660 | 0.001369 | **0.000071** |
| 60 min | 8,619 | 136 | 0.827 | 0.015442 | 0.015311 | 0.009465 | **0.003365** |
| 120 min | 8,619 | 319 | 0.862 | 0.033426 | 0.033117 | 0.013050 | **0.007068** |

Isotonic recalibration, fitted on the `calib` block alone, roughly halves the expected
calibration error at every horizon. Brier barely moves, which is what a rare event does
to a squared score.

### Against the cause-specific Cox model

| Horizon | Cox AUC | LightGBM AUC | Difference |
|---|---|---|---|
| 30 min | 0.7701 | 0.8272 | +0.057 |
| 60 min | 0.7293 | 0.8286 | +0.099 |
| 120 min | 0.6703 | 0.8621 | **+0.192** |

**Discrimination only, and that is a statement about the methods rather than a
convenience.** A cause-specific Cox model gives a risk *ranking*, not the cumulative
incidence of leaving by a horizon — reading one as the other is the classic
competing-risks error. Getting an incidence from a hazard model properly means
Fine-Gray, which is a different model. So the Cox has no Brier score or calibration
curve to compare. The gap widening with horizon is consistent with the
proportional-hazards violations already measured for `n_waiting` and `triage_level`.

### Against Aalen-Johansen, at group level

Absolute gaps are small (≤0.013) but relative gaps reach −55% at ESI 4 / 30 minutes,
where the incidence is 0.006. The model **under-predicts at short horizons**.

### ESI 1-2 are excluded, with the reason

Zero positive cases at every horizon on the test block. Their rows appear in the metrics
table with their counts and an explicit exclusion reason rather than being omitted. A
calibration curve over zero events is not a weak result; it is not a result.

A correction to my own reasoning, found by a test: I claimed isotonic calibration leaves
the AUC unchanged because it is monotone. It cannot *invert* a pair, but it pools
adjacent violators into ties, and tied pairs count as half — so the AUC does move
slightly (0.9802 to 0.9813 on a synthetic example). Discrimination is therefore reported
on the raw score, and the docstring now says why.

---

## PART D — Reinforcement learning for staffing

Tabular Q-learning over (queue length by acuity band, doctors on duty, hour-of-day
band), 600 states x 3 actions, stepping the real SimPy model one hour at a time through
a `dynamic_capacity` hook that is `None` on every other code path. No gymnasium, no
torch: a handful of methods, and nothing added to the dependency list.

Pre-declared budget, fixed before the result was seen: 60 training episodes, 20
evaluation episodes, 168-hour episodes, **disjoint random streams** for training and
evaluation.

| Policy | Mean total cost | Mean ESI 1-2 breaches | vs fixed roster | 95% CI | Significant |
|---|---|---|---|---|---|
| `reactive_rule` | 48,441 | 17.6 | **−1,438** | −3,175 to +298 | no |
| `fixed_roster` | 49,880 | 10.0 | — | — | — |
| `q_learning` | 67,712 | 25.7 | **+17,832** | 14,663 to 21,002 | **yes** |
| `random` | 71,952 | 16.6 | +22,072 | 19,468 to 24,675 | yes |

### The learned policy does not beat the fixed plan

It is **significantly worse**, and only marginally better than acting at random. It also
produces *more* high-acuity breaches than the random policy — it learned to under-staff.
**Reported as it stands. It has not been tuned until it won**, and the budget above was
declared before the run.

A diagnosis, offered without acting on it: 60 episodes x 168 hourly decisions is about
10,000 transitions spread over 1,800 table cells — roughly five visits per cell, far
short of what tabular Q-learning needs to converge. This is evidence about the training
budget, not about whether RL can do this.

### The circularity, stated plainly

The agent learns the simulator; the simulator was calibrated to reproduce `ed_v2.csv`;
`ed_v2.csv` came from a generator whose parameters were chosen by hand. Three things
make a win easier here than it could ever be in practice: there is no sim-to-real gap
because there is no real, the agent sees the true queue state rather than a stale
tracking board, and staffing is instantly adjustable with no roster, notice period or
overtime rules. **A win would be evidence about the method, not about any department.**

---

## PART E — Integration

### API

| Endpoint | Change |
|---|---|
| `POST /predict-risk` | **New.** Cumulative incidence of leaving by each horizon. Returns `p_still_present`, deliberately not "probability of being seen" — the model does not separate being seen from still waiting. Flags ESI 1-2 as `outside_calibrated_range` with a reason. 503 with the training command when no model is loaded. |
| `POST /simulate` | Accepts an optional `resource_extensions` block. Supplying any of it sets `uncalibrated: true` on the response and adds the not-calibrated notice to `caveats`. Omitting it is the calibrated model, byte for byte. |
| `GET /staffing-recommendation` | New `?include_rl=true`. **Off by default.** Returns the RL comparison and its verdict, labelled experimental. |

OpenAPI examples updated on every new field.

### Dashboard

A new **Research results** page under *Analyze*, with four tabs: **Robustness** (the
per-seed table, the spread, and a chart of static against rolling coverage per seed),
**Risk of leaving** (metrics, calibration curves before and after, the Cox and
Aalen-Johansen comparisons), **Scenario lab** (not-calibrated banner, the sensitivity
ranking), and **Experimental RL** (the verdict shown as an error when the agent loses,
which it does). Every section says plainly when its artifact has not been produced
rather than drawing an empty chart.

### Upload job engine

`competing_risks` added to `PIPELINE_STAGES`, and skipped with a plain-English reason
when the extract has fewer than `min_lwbs_events` patients who left, or when the split
is too small to train anything — the same gating `survival` already had.

### Docker

Rebuilt, started, and the trainer profile run end to end.

```
Pipeline finished: 16 stage(s) in 269.31s     (exit 0)
/health            models_loaded: true, forecaster_loaded: true, simulation_calibrated: true
/predict-risk      answers in the container (30/60/120 min)
/staffing-...      ?include_rl=true returns the verdict; plan still 3/3/2/2 at 49,280, 72 of 144
container user     uid 10001 (non-root)
published ports    127.0.0.1:8000, 127.0.0.1:8501
```

**The containerised trainer reproduced every headline number exactly.**

### The circular-import class of bug

`DECISION_SUPPORT_NOTICE` lived in `api/main.py`, which builds the FastAPI application
and loads a model bundle at import. `src/serving/api.py` needed it, and `api/main.py`
already imports `src/serving/` — so the import would have been circular. It now lives in
**`src/notices.py`, which imports nothing and does nothing at import time**, and both
sides import it from there. The same rule produced `app/config.py` for the dashboard
after the same bug.

---

## PART F — Verification

### The 11 skipped tests

All 11 are in `tests/test_dashboard_imports.py`, all are **conditional**, and **none can
be enabled** — each would require inventing something a module does not need.

| Count | Reason | Can it run? |
|---|---|---|
| 1 | `_shared.py` is a helper module, not a page, so the "every view has `render()`" rule skips it | **No.** It has no `render()` by design. |
| 5 | `__init__.py`, `about.py`, `data_analysis.py`, `model_quality.py`, `staffing.py` draw no Altair charts, so the "every chart is styled" rule skips them | **No.** Adding a chart to a page that does not need one to satisfy a test would be the wrong way round. |
| 5 | The same five, for the "every chart has a caption" rule | **No**, same reason. |

The skips are doing their job rather than hiding anything: the new `research.py` draws
charts, so it is **not** skipped by either rule, and both caught real violations in it —
unstyled charts and missing captions — which were fixed.

### Final test suite

Run on a quiet tree with no concurrent edits:

```
.\.venv\Scripts\python.exe -m pytest -q -rs

1218 passed, 11 skipped, 147 warnings in 905.12s (0:15:05)       exit 0
```

| | Tests | Skipped | Failed |
|---|---|---|---|
| Baseline at the start of the day | 1100 | 11 | 0 |
| **Final** | **1218** | **11** | **0** |
| Added | **+118** | 0 | 0 |

Where the 118 came from:

| File | Tests | Covers |
|---|---|---|
| `tests/test_staffing_rl.py` | 17 | Part D — environment bounds, CRN, reward signs, disjoint train/eval streams |
| `tests/test_competing_risks.py` | 16 | Part C — horizon labelling, leakage, calibration, monotone incidence |
| `tests/test_resource_extensions.py` | 15 | Part A — byte-identical output with flags off, each resource constraint |
| `tests/test_mimic_adapter.py` | 14 | Part B — the column map and both wait proxies on a fake fixture |
| `tests/test_api_predict_risk.py` | 12 | Part E — the new route, its refusals, its notices |
| `tests/test_shift_check.py` | 10 | Part B — the four shift statistics and the identity case |
| existing files | +34 | the survival regression, the 16th stage, the new dashboard page, upload gating |

**The two upload-API failures from the mid-day run did not recur.** They were an
artefact of my editing `src/pipeline.py` while that suite was running, as recorded below.

### Headline assertions, one last time

`tests/test_headline_regression.py` — **10 passed**, and the reference row of the
robustness study reads:

```
wait MAE model      11.524        conformal rolling   0.8875
wait MAE baseline   34.233        conformal static    0.849
forecast MAE        2.3557        cheapest plan       day=3, evening=3, night_early=2, night_late=2 @ 49,280
forecast naive      2.9792        feasible            72 of 144
```

**Nothing moved.**

---

## Everything that failed, was skipped, or was left undone

1. **Two upload-API tests failed in the Parts 0+A suite run** (`test_scoring_mode_...`
   and `test_the_whole_ed_v2_extract_runs_every_stage`). Cause: I edited
   `src/pipeline.py` while that suite was running, and the job subprocesses picked up a
   half-written file. Both pass in isolation, and the final suite above was run with no
   concurrent edits.
2. **Seeds 8 and 10 fail the simulation validation gate.** Not fixed, diagnosed, with a
   change proposed and deliberately not applied.
3. **The RL agent lost**, significantly. Not tuned.
4. **The gate tolerance change was not applied.** Proposing it is the whole of what was
   asked.
5. **No MIMIC data was touched**, so `configs/column_map_mimic_ed.yaml` and
   `docs/real_data_readiness.md` are unverified against the real tables. Every column
   name in them is a claim.
6. **The resource extensions cannot be validated at all** — not now, and not with MIMIC
   either, which records no bed, nurse or imaging data.
7. **No human has used the dashboard in a browser.** The Research page is covered by
   `AppTest`, which renders the component tree without a browser.
   `docs/upload_manual_test_checklist.md` remains marked NOT PERFORMED.
8. **One machine only.** Windows 11, Python 3.12, one CPU. No Linux host, no CI, no
   second Python version.
9. **The forecast study's winning variant was still not adopted** (carried over from the
   previous session): adopting it would move 2.356.
10. **A counting bug shipped in the Part 0 answers and in the first draft of this
    report**: `n_distinct_plans` counted the gate refusal as a staffing plan, giving 8
    where the honest count is 7. Found by checking the report's claims back against the
    artifact, fixed in `answer_questions`, and the tables re-aggregated. Nothing else
    read that field.

---

## Honest framing of what these results are

Everything above is a measurement on data from **one generator**, whose parameters were
chosen by hand to sit in published benchmark ranges. Agreement across ten seeds shows
the pipeline is not over-fitted to one draw of that generator. It does not show that
any model here would work in a real emergency department, and the four headline numbers
are properties of `data/raw/ed_v2.csv` rather than a benchmark.

Specifically labelled, in the code and in every artifact they touch:

- the resource extensions **recover an assumed department, not a measured one**;
- the competing-risks model **recovers the generator's patience distribution**, which
  was chosen rather than observed;
- the RL agent **recovers the simulator**, which was calibrated to the generator;
- any MIMIC result would be a **proxy wait**, not a door-to-provider time.

---

# Final submission pass

Date: **2026-10-13**. Windows 11, PowerShell 5.1, `C:\dev\ed-project`, `.venv`
(Python 3.12.10). No git was used. **No modelling code, config, threshold or result was
changed.**

**Headline numbers: unchanged.** `tests/test_headline_regression.py` passed 10/10 after
every step that could have moved them — including a full 16-stage pipeline re-run on the
main project (529.28 s, exit 0) and an independent 16-stage run inside the unzipped
package (428.20 s, exit 0), which reproduced `3/3/2/2 at 49,280`, 72 of 144 from its own
fresh virtual environment.

**Test suite: 1246 passed, 11 skipped, 0 failed** on the main project (18m 26s,
exit 0) and **1246 passed, 11 skipped** inside the unzipped package from a fresh
virtual environment (19m 45s, exit 0). **No tests were added**, so the count is
unchanged and the number audit's expectation of 1246 still holds.

## A. Robustness wording sweep

Four files still presented findings that Part 0 had withdrawn or superseded. The
generated report was the worst case: `_robustness_section` read
`robustness_summary.csv` — the **wrapper** study — and printed
`robust_seed7_base … refused (simulation failed validation)` into
`artifacts/report.md` as a current result.

| File | Old | New |
|---|---|---|
| `artifacts/checkpoints/hours_work.md` §A2 | *(no warning)* | **WITHDRAWN banner** on the whole section: wrapper-generated, not comparable, seed-7 failure withdrawn, pointing at `days_work.md` Part 0 |
| same, run table | `robust_seed7_base \| seed 7, base scale \| … \| **aborted at scenarios**` | `wrapper seed 7 … stopped at scenarios — **withdrawn, see above**` |
| same, gate bullet | "6 of 7 pass; `seed7_base` fails on ESI 4 abandonment" | "**superseded by Part 0:** gate passed on **8 of 10**; seed 8 rare-event tolerance artefact (1.6% base rate, 1.2 pp); seed 10 genuine 8-minute ESI 3 miss; the seed-7 claim **did not reproduce and is withdrawn**" |
| same, staffing bullet | "43,120 / 44,660 / 49,280 / 50,050 / 55,440 … 36 to 81 of 144" | "**43,890 to 54,670** across the **8 seeds that produced a plan**, **7 distinct rosters**, day shift 3 in every one; **3/3/2/2 at 49,280 is the reference-dataset answer**" |
| `src/report.py` `_robustness_section` | read `robustness_summary.csv` (wrapper), printed the seed-7 refusal | reads `robustness_seeds.csv` (true generator) and **refuses to fall back**, with the reason in the docstring; adds "gate passed on 8 of 10", a per-seed diagnosis for each refusal, the plan spread, what held, and the synthetic caveat |
| `src/report.py` `FUTURE_WORK_TAIL` | "**Reinforcement learning for dynamic staffing.** … the natural extension, and the simulation is already a usable environment for it." | "**A reinforcement learning staffing policy that works.** One was built and it lost. **+17,832** vs the fixed roster (95% CI 14,663–21,002), more ESI 1-2 breaches than random; the budget is a **hypothesis, not a defence**; circular" |
| same | "**Richer resource modelling.** Nurses, beds and diagnostics as explicit constrained resources…" | "**Calibrating the richer resource model.** Built behind `simulation.resource_extensions`, off by default; **cannot be calibrated** — no bed, nurse or imaging column; ranking usable, magnitudes not" |
| same | "**Competing-risks prediction**, not just description…" | "**Competing-risks prediction on real data.** Built at 30/60/120 min with isotonic recalibration and Aalen-Johansen; recovers the generator's patience distribution" |
| `README.md` limitations | "Nurses, beds, imaging and lab turnaround **are not modelled**." | "**are not calibrated**" + assumed parameters + the gate cannot test them + read the ranking |
| `README.md` *(added)* | — | "**The validation gate has a known blind spot.**" — a within-acuity permutation preserves every marginal KPI exactly and passes |
| `README.md` *(added)* | — | "**The gate refused a staffing plan on 2 of 10.**" passed 8 of 10; both diagnoses; no tolerance loosened |
| `README.md` *(added)* | — | "**The cheapest roster is dataset-specific.**" 43,890–54,670, 7 rosters, day shift 3; 3/3/2/2 is the reference answer |
| `README.md` not-built | "**Reinforcement learning and deep forecasters.** Future work; the simulation is already a usable RL environment." | split: deep forecasters deliberately out of scope, and the **quantified RL negative result** |
| `app/views/research.py` | per-seed table, no interpretation | new `_gate_and_roster_findings`, **computed from the table** so it cannot drift: renders "gate passed on 8 of 10", both seed diagnoses, "43,890 and 54,670 … 7 distinct rosters … day shift was 3", method-not-roster, and the synthetic caption |
| `artifacts/report.md` | wrapper robustness section, 8 rows incl. the seed-7 refusal | regenerated: 11-row true-generator table, 8 of 10, both diagnoses, the spread, what held |

Every synthetic-data and circular-recommendation caveat was kept; three were added.

**Number audit: 71 claims checked, 0 mismatches.** It was extended with 16 new checks
that fail if a seed-7 failure claim reappears *without withdrawal language in the same
sentence*, if the gate figure stops saying 8 of 10, or if the plan spread, the RL
figures, the reference-roster framing, the within-acuity blind spot, the proxy-wait
limitation, the generator command or the `.equals` check go missing.

## B. SUBMISSION.md

Extended, nothing removed:

- **What this is**, in five numbered lines.
- **Headline results** with each baseline named and the margin computed: wait MAE 11.52
  vs **34.23** (66.3% lower); rolling conformal 0.8875 vs static **0.849** (3.9 pp
  closer to the 0.90 target); forecast 2.356 vs **2.979** /h (**20.9% lower**); cheapest
  roster **3/3/2/2 at 49,280** (72 of 144 feasible). Plus the across-seed paragraph:
  8 of 10 gate passes, 43,890–54,670, 7 rosters.
- **Limitations**, eight numbered items: synthetic data; the circular staffing
  recommendation; the gate's within-acuity blind spot; the seed 8 and seed 10 refusals
  with the deliberately-unapplied tolerance change; the RL negative result; the
  uncalibrated resource extensions; the proxy wait with both proxies' error directions;
  and the standing measured limits.
- **Reproducibility**: the generator command and the `.equals` check, **re-run here, not
  restated** — `DataFrame.equals` **True**, shapes `(57500, 10)` both sides, sizes
  differing by exactly **57,501 bytes**, one `\r` per line.

## C. Stale data removed

`data\generated\` **moved** (not deleted) to `C:\dev\ed-project-old-data\generated\`:
14 files, **53 MB**, seven wrapper-generated datasets and their config sidecars. Each
sidecar carried a ready-to-run pipeline command, and one of them was
`ed_seed7_base.csv` — the dataset behind the withdrawn finding.

Confirmed afterwards: **no reference to `data/generated` anywhere** in `src/`,
`scripts/`, `tests/`, `app/`, `api/`, `configs/`, `docs/`, `README.md`, `SUBMISSION.md`
or `CLAUDE.md`, so nothing needed fixing. `data\raw\ed_v2.csv` is untouched. The
`.gitignore` pattern stays, with a comment saying where the data went and why, so a
stray re-run cannot put 53 MB of superseded data back into a package.

## D. One-click setup for a fresh machine

`Setup.cmd` created. `scripts\setup.ps1` already existed, so it was **extended rather
than replaced**:

- An interpreter probe before use. `py -3.12 -m venv` on a machine without 3.12 fails
  with a message that is clear in a terminal and invisible behind a double-click, so it
  now checks, and on failure names what to install, where, and the `-Python 'py -3.11'`
  alternative.
- **It installs all three pinned requirement sets, not one.** See "What failed" below.
- An import check for what the demo actually launches, before claiming success.
- The closing line is now exactly `Setup complete. Double-click Start-Demo.cmd.`

`scripts\start_demo.ps1` prints **"First run: double-click Setup.cmd, then
Start-Demo.cmd."** when `.venv` is missing, instead of naming a command that needs a
terminal. Setup is step 0 in `SUBMISSION.md` and at the top of the README quick start,
both labelled "only needed on a new machine".

## E. Package built and verified

`scripts\make_submission.ps1` stages an explicit copy, **verifies the exclusions held
before compressing**, and refuses to build if any of 21 required entries is missing or
if excluded content reached the staging tree.

```
dist\ed-project-submission.zip
14.0 MB compressed, 30.1 MB on disk, 335 files
```

Excluded: `.venv\`, `__pycache__\`, `.pytest_cache\`, `*.pyc`, `runs\`, `dist\`,
`data\generated\`, `.git\`, `.env`. Included: `data\raw\ed_v2.csv`, `artifacts\`,
`docs\`, `configs\`, `src\`, `api\`, `app\`, `tests\`, `scripts\`, the Docker files,
`Setup.cmd`, `Start-Demo.cmd`, `Stop-Demo.cmd`, `README.md`, `SUBMISSION.md`,
`requirements*.txt`. An empty `runs\logs\` ships with a `.gitkeep` so the first launch
has somewhere to redirect a log.

Verified by unzipping to `C:\dev\ed-pkg-verify` and running it as a recipient would:

| Step | Result |
|---|---|
| `Setup.cmd` on a tree with no `.venv` | **exit 0 in 53 s**, Python 3.12.10, and `fastapi, uvicorn, streamlit, httpx, lightgbm, lifelines, simpy all import` |
| Full test suite in the package | **1246 passed, 11 skipped**, exit 0, 19m 45s — identical to the main project |
| `tests/test_headline_regression.py` | **10 passed** |
| Pipeline end to end | **16 stages in 428.20 s**, exit 0 |
| Headline numbers from the package's own run | `3/3/2/2 at 49,280`, **72 of 144** — reproduced independently |
| `Start-Demo.cmd` from the unzipped folder | API healthy with models loaded, dashboard serving, browser opened; `/health` `models_loaded: true`; the page load fired `/forecast` and `/staffing-recommendation` |
| `Stop-Demo.cmd` | both PIDs stopped with their children, `runs\demo.pids` removed, **ports 8000 and 8501 free** |
| Temp folder | deleted (686 MB, mostly its own `.venv`) |

## F. Final smoke on the main project

Started with `Start-Demo.cmd`, every endpoint hit, stopped with `Stop-Demo.cmd`.

| Endpoint | Status | Keys |
|---|---|---|
| `GET /health` | 200 | all 6 present |
| `POST /predict-wait` | 200 | all 11 present |
| `GET /forecast?horizon_hours=24` | 200 | all 3 present |
| `POST /simulate` | 200 | all 2 present |
| `GET /staffing-recommendation` | 200 | all 12 present |
| `POST /predict-risk` | 200 | all 6 present |

Values read back rather than merely counted: `models_loaded: true`; predict-wait 92.0
min with a 90% interval of 66.8 to 117.3; 24 hourly forecast points; staffing
`{day: 3, evening: 3, night_early: 2, night_late: 2}` at **49,280**, **72 of 144**;
predict-risk 0.0026 / 0.0263 / 0.0554 at 30 / 60 / 120 minutes. The Command center
rendered against the live API with **zero errors** and all four KPI cards. `Stop-Demo`
left both ports free.

**Docker was not running** (`failed to connect to the docker API`), so there is no
container model version to compare against the local `base+queue-raw-fc574ffc`. Per the
instruction, nothing was rebuilt or started.

## What failed, and what it cost

1. **The package could not have run the demo, and the first clean-install verification
   is what caught it.** `requirements.txt` holds the pipeline only — no `fastapi`, no
   `uvicorn`, no `streamlit`, because the container images install
   `requirements-api.txt` and `requirements-app.txt` separately and neither image needs
   the other's packages. A recipient would have been told **"Setup complete"** and then
   watched `Start-Demo.cmd` fail. `httpx`, which FastAPI's `TestClient` needs, is in
   **none** of the three files, so a large part of the suite would have failed on import
   too. `scripts\setup.ps1` now installs all three plus `httpx==0.28.1` and checks the
   imports resolve before claiming success. **`requirements.txt` itself was left
   untouched**; adding `httpx` to a requirements file is the maintainer's call.
2. **I clobbered two run-record artifacts and had to restore them.** Regenerating the
   report with `--stages report` rewrote `pipeline_run.json` and `run_stages.json` to
   describe a one-stage run, so the report header read `Stages run: report`. Fixed by
   running the full 16-stage pipeline and restoring both records, so the header, the
   stage list and the timing now describe the same run. Caught by the number audit,
   which compares the report's claims against the artifacts.
3. **My smoke script was wrong before the API was.** It guessed
   `predicted_wait_minutes`, `n_feasible_plans` and a `/simulate` body of doctor counts.
   The real fields are `point_estimate_minutes`, `n_feasible`, and
   `{"scenario": "baseline"}`; `/simulate` returned 422 for my malformed body. The
   endpoints were correct throughout — the script was rewritten from the live OpenAPI
   schema.
4. **`model_version` moves without the model moving.** It went from
   `base+queue-raw-f91b7394` to `base+queue-raw-fc574ffc` across the re-run, because the
   hash incorporates the data version, which carries the run date
   (`real-57179rows-20261013`). Every metric is identical and all 10 headline assertions
   pass. Worth knowing before anyone treats that string as a content hash.
5. **The seed-7 question text is deliberately still in two places.**
   `scripts/robustness_study.py` and `artifacts/metrics/robustness_study.json` keep
   "Does the earlier seed-7 simulation-validation failure reproduce?" **with its answer
   of "does NOT reproduce"**. Those are recorded results, and deleting the question
   would delete the refutation. The audit allows a mention only when the same sentence
   withdraws it.
6. **The final-pass specification itself never arrived in the session** — only the
   update note that amends it. "The sections from step 3" could not be inferred, so
   SUBMISSION.md was extended with the four content items the update names explicitly.
   If step 3 asked for more sections, they are not here.

## Carried forward, unchanged

The gate tolerance change for seed 8 is still **proposed and not applied**. The RL agent
was **not retrained** until it won. The forecast study's winning variant is still **not
adopted**, because adopting it would move 2.356. No MIMIC data was downloaded or touched.
