# Robustness, high-acuity recall, sensitivity, forecasting and hardening

What was run, and what came back. **Every number below was read from an artifact
file, from test output, or from a scanner's output.** Nothing here is estimated, and
where something was not verified it says so in "What was NOT verified" at the end.

Date of this work: **2026-10-09 / 2026-10-10**. Windows 11, PowerShell 5.1, `.venv`
(Python 3.12), Docker 29.7.2.

All data involved is **synthetic**. `data/raw/ed_v2.csv` is a synthetic extract; every
dataset generated here comes from the same generator. None of it describes a real
department or a real patient.

---

## The reproduction gate

The four headline numbers for the default run on `data/raw/ed_v2.csv` are pinned by a
new test, `tests/test_headline_regression.py`:

| Figure | Value | Read from |
|---|---|---|
| wait MAE, best model | **11.524** | `metrics/wait_models.json` → `ablation.best_by_test_mae.mae_minutes` |
| wait MAE, rolling-mean baseline | **34.233** | `metrics/wait_models.json` → `baseline.blocks.test.regression.mae` |
| rolling conformal coverage, overall | **0.8875** | `metrics/wait_models.json` → `conformal.rolling.comparison` |
| forecast MAE, overall | **2.356** | `metrics/forecast.json` → `overall.model.mae` |
| cheapest feasible plan | **49,280**, 72 of 144 feasible | `metrics/staffing_search.json` |

The test fails if any of them moves. `ED_ALLOW_HEADLINE_CHANGE=1` skips the checks for
one deliberate update — and a separate test in the same file *fails* whenever that
variable is set, so a disabled guard cannot be left disabled unnoticed.

Both directions were verified rather than assumed: tampering `forecast.json` to 2.400
produced
`AssertionError: forecast_mae moved: expected 2.356 ... baseline artifacts now say 2.4`,
and the file was restored immediately afterwards.

Every experiment in this work wrote to `runs/<name>/` or to a new file under
`artifacts/`. Nothing overwrote a baseline artifact.

---

## PART A — Robustness across datasets

### A1. The generator, and a correction to the brief

**`scripts/generate_ed_data.py` did not exist.** `scripts/` contained only `.ps1`
files, and `C:\dev\ed-project` is not a git repository, so there was no history to
recover it from. On the instruction's terms this was the "skip Part A" case.

That was put to the user, who chose to **wrap the existing generator** rather than
skip. `scripts/generate_ed_data.py` was therefore written as a thin CLI over
`src.data.synthetic.generate_synthetic_visits` — the function that produced
`ed_v2.csv`. It contains **no generation logic**: it loads a config, applies the
overrides named on the command line, calls the generator, and writes the CSV plus a
derived config pointing `paths.raw_file` at it.

**A measurement that changed the design.** `configs/default.yaml` carries two doctor
rosters: `synthetic.shifts` (4/8/5) and `staffing_roster.shifts` (2/3/3/2, documented
as "the roster operating in `data/raw/ed_v2.csv`"). Generating against the published
roster was tried first and produces a department in collapse — at seed 42, **median
wait 192.4 min and 50.6% abandonment**, against ed_v2's 11.0 min and 3.6%. The two
blocks are not meant to agree: `synthetic.service_time` is total post-contact time,
and the simulation reconciles them by fitting `clinician_time_fraction`, which is
**0.207** on the baseline run. The v2 profile therefore leaves the synthetic block at
its configured default and changes only the visit count.

Agreement of the v2 profile with `ed_v2.csv`, at seed 42:

| | ed_v2.csv | v2 profile, seed 42 |
|---|---|---|
| median wait (seen) | 11.0 min | 16.5 min |
| mean wait | 34.9 min | 40.7 min |
| abandonment, overall | 3.58% | 3.84% |
| abandonment, ESI 5 | 8.96% | 8.0% |
| abandonment, ESI 1-2 | 0.0000 | 0.0000 |
| median post-contact | 99 min | 53.6 min |

Close, **not identical**. `ed_v2.csv`'s own generation parameters are recorded nowhere
in the repository and cannot be recovered: its 57,500 visits span 365 days, which
implies an arrival rate about 12% above what `base_rate_per_hour: 6.0` produces, and
its post-contact time is nearly twice this profile's. Two cosmetic differences are
deliberate and documented in the script: generated files code `sex` as
`female`/`male` rather than `F`/`M` (the column map accepts both), and they do not
reproduce ed_v2's ~0.36% missing `triage_level`, because manufacturing data corruption
is inventing data rather than generating it.

### A2. The runs

> **WITHDRAWN — read this before the rest of section A.** Every dataset in this section
> was produced by the older wrapper around `src/data/synthetic.py`, not by the true
> generator at `scripts/original_generator/generate_ed_data.py`. The two model different
> things: the wrapper's service times are total post-contact time where the true
> generator's are physician-busy time, so "seed 7" does not name the same dataset in the
> two studies and the runs below are not comparable with anything else in this
> repository. **The seed-7 gate failure recorded here did not reproduce and is
> withdrawn.** The robustness results that stand are in
> `artifacts/checkpoints/days_work.md`, Part 0: the gate passed on **8 of 10**
> regenerated datasets, refusing a staffing plan on seeds 8 and 10 only. The section is
> kept as a record of what was run and of a method error worth not repeating, not as
> evidence.

Five datasets as asked, plus `ed_v3` from Part F. Each ran the full 15-stage pipeline
into its own `runs/` folder.

| Run | Dataset | Visits | Wall time | Outcome |
|---|---|---|---|---|
| `robust_seed42_base` | seed 42, base scale | 57,500 | **288 s** | all 15 stages |
| `robust_seed7_base` | wrapper seed 7, base scale | 57,500 | **105 s** | stopped at `scenarios` — **withdrawn, see above** |
| `robust_seed99_base` | seed 99, base scale | 57,500 | **272 s** | all 15 stages |
| `robust_seed42_scale085` | seed 42, −15% | 48,875 | **254 s** | all 15 stages |
| `robust_seed42_scale115` | seed 42, +15% | 66,125 | **304 s** | all 15 stages |
| `robust_v3` | Part F dataset | 57,500 | **5,060 s** | all 15 stages |
| `robust_3year` | seed 42, 3 years | 155,422 | see Part B3 | see Part B3 |

**"Scale ±15%" was read as dataset size**, not load: the ±15% runs have the same
arrival rate and a proportionally longer or shorter span (344 and 456 days against
399). See "Interpretations worth disputing" below.

**`robust_v3` took 84 minutes against ~5 for the others.** It is the same number of
visits; what differs is congestion. v3's median wait is 25.9 min against 16.5, so every
discrete-event replication carries a longer queue and more events, and the stage that
pays for it is the 19-world sensitivity grid.

**`robust_seed7_base` stopped at `scenarios`, by design.** Its simulation failed
validation — abandonment rate ESI 4, simulated 0.024 against observed 0.053, relative
error **0.5421** against a tolerance of 0.50 — so the gate refused the scenario and
staffing stages and the pipeline exited 1. That is the gate doing its job on real data,
not a bug, and it is why that row of the table reads *refused* rather than blank.

`artifacts/tables/robustness_summary.csv` (7 rows including the baseline) and
`robustness_summary_across_seeds.csv` are built by the new
`scripts/robustness_summary.py`, which reads finished run folders and re-runs nothing.

### A3. What held on every dataset, and what did not

#### Held on every dataset

- **The wait model beats its baseline, everywhere.** MAE 11.5–25.6 against a
  rolling-mean baseline of 34.2–85.5; the ratio stays between 2.9× and 3.3× on all
  seven. Across the three seed replicates the model's MAE is 15.93 ± 0.88 (sd/mean
  0.055).
- **The ablation direction holds, everywhere.** `base+queue` beats `base` on all
  seven, by 14.6 to 49.0 minutes (17.06 ± 1.32 across seeds). The sign of the G3
  result is robust; its *magnitude* is not a property of the method, and the
  generator caveat still applies.
- **`fast_track_esi45` is the best scenario on every dataset that produced a
  ranking**, and the full order is byte-identical on all six:
  `fast_track_esi45 > plus_one_doctor_peak > plus_one_nurse_peak > baseline >
  surge_10pct > surge_20pct`. The ranking was also stable under all 19 perturbations
  in every one of those runs. This is the most robust conclusion in the project.
- **The forecast beats seasonal-naive overall on all seven** (MAE 1.90–2.58 against
  2.42–3.42).
- **Classifier quality is stable**: PR-AUC 0.9588 ± 0.0022 across seeds, recall
  0.866 ± 0.024.
- **ESI 1 is never predictable under the pooled target.** FNR = 1.000 on *every*
  dataset, with 1–5 positive cases. See Part B.

#### Did NOT hold

- **"Rolling conformal beats static" is not universal — but the reason matters.**
  Rolling lands closer to nominal on 6 of 7 datasets; on `seed42_base` static is
  closer (0.9031 against rolling's 0.8924). What *is* robust is the stability:

  | | mean | sd | sd/mean |
  |---|---|---|---|
  | `conformal_rolling_overall` | 0.8940 | **0.0034** | 0.004 |
  | `conformal_static_overall` | 0.8958 | **0.0213** | 0.024 |

  Static coverage swings from 0.849 to 0.913 across datasets; rolling stays between
  0.888 and 0.906. Per level the gap is wider still — at ESI 5, rolling's sd is 0.0006
  against static's 0.0082. So the defensible claim is not "rolling is always closer"
  but **"rolling lands in the same place every time, and static does not"**, which is
  what a coverage guarantee is supposed to mean. On one dataset static got lucky.

- **The simulation validation gate does not pass on every dataset.** The conclusion
  stands; the numbers behind it here do not, because they came from the wrapper
  generator. **Superseded by Part 0 of `days_work.md`:** with the true generator the
  gate passed on **8 of 10** datasets and refused to issue a staffing plan on two —
  **seed 8**, where ESI 3 abandonment missed a 1.6% base rate by 1.2 percentage points
  (a rare-event tolerance artefact), and **seed 10**, where ESI 3 mean wait came out
  8 minutes low against an observed 34.4 (a genuine fit miss). The specific claim made
  here, that `seed7_base` fails on ESI 4 abandonment, **did not reproduce and is
  withdrawn.** The general point — that a gate whose margin varies this much will fail
  occasionally on honest data — is what survived.

- **The staffing recommendation is NOT robust.** The conclusion stands and reproduced
  with the true generator; the figures here are the wrapper's. **Superseded by Part 0:**
  the cheapest plan spread **43,890 to 54,670** across the **8 seeds that produced a
  plan**, with **7 distinct rosters** and the day shift at 3 in every single one. The
  *method* transfers; the specific roster does not, and should not be quoted as a
  finding about anything but the dataset it was computed on. **3/3/2/2 at 49,280 is the
  reference-dataset answer**, not a recommendation for any department.

- **The peak-hour forecast weakness does not reproduce.** On `ed_v2` the model loses
  to seasonal-naive at peak (4.364 against 4.100). On **all six** generated datasets it
  wins: 3.07/3.97, 2.67/3.15, 2.99/3.59, 4.30/5.33, 2.74/2.84, 4.37/5.17. Combined with
  Part D's 26-origin result on ed_v2 itself, the weakness looks like a property of the
  4-fold backtest's 60 peak hours rather than of the forecaster.

#### The 3-year run, as an eighth row

`robust_3year` (155,422 visits) is in the same table and agrees with the rest: model
MAE 12.42 against a baseline of 31.86, ablation gap positive (10.38), forecast ahead
of seasonal-naive overall (2.03 against 2.73) **and at peak** (3.08 against 3.83),
simulation validation passes with the *best* margin of any run (worst held-out
relative error 0.1988), and the scenario ranking is the same. Its cheapest plan is
`day=4, evening=3, night_early=1, night_late=1`, 69 of 144 feasible — a different
roster from v2's at coincidentally the same cost of 49,280.

Two figures move in the direction more data would predict: the ablation gap is the
*smallest* of any run (10.38 against 14.6–49.0), and classifier recall is the lowest
(0.742). With three years of history the rolling-mean baseline has more to work with,
so the margin the queue features buy over it narrows.

#### Quantities flagged as varying widely across the three seeds

`sim_worst_relative_error` (sd/mean 0.45), `n_feasible_plans` (0.22) and
`forecast_mae_peak_seasonal_naive` (0.115). Everything else came in under 10%.

---

## PART B — ESI 1-2 recall

### B1-B2. Four approaches, compared on the same split

New module `src/models/high_acuity.py`, wired into the `train_wait` stage (no new
pipeline stage, so the stage list and the upload engine are unchanged). It writes
`tables/high_acuity_comparison.csv` with the caveat on line 1, and a `high_acuity`
block in `metrics/wait_models.json`.

The per-acuity targets live in `configs/default.yaml` under
`features.acuity_target_wait_minutes` — **ESI 1: 5, ESI 2: 15, ESI 3: 30, ESI 4: 60,
ESI 5: 120** — labelled in the config, in the module docstring and in every artifact as
**ASSUMED ILLUSTRATIVE VALUES, NOT CLINICAL STANDARDS**. ESI 1 uses 5 as the top of the
conventional "0-5 minutes" immediate-care range.

Measured on `data/raw/ed_v2.csv` (test block, thresholds chosen on validation only):

| Approach | ESI 1 positives | ESI 1 recall | ESI 1 FNR (95% CI) | ESI 2 positives | ESI 2 recall | ESI 2 FNR (95% CI) |
|---|---|---|---|---|---|---|
| (a) pooled | 1 | 0.000 | 1.000 — **not reportable** | 29 | 0.000 | 1.000 (1.000–1.000) |
| (b) acuity sample weights | 1 | 0.000 | 1.000 — **not reportable** | 29 | 0.000 | 1.000 (1.000–1.000) |
| (c) separate ESI 1-2 model | — | — | **NOT FITTED** | — | — | **NOT FITTED** |
| (d) acuity-specific target | 76 | 0.750 | 0.250 (0.145–0.363) | 113 | 0.903 | 0.097 (0.044–0.153) |

- **(a) reproduces the stated problem exactly**: FNR 1.000 for ESI 2 over 29 positives,
  and ESI 1 with a single positive case, which this module refuses to report a rate
  from at all.
- **(b) changed nothing.** Inverse-frequency weights were applied (ESI 1 ×19.997,
  ESI 2 ×3.112) and the false-negative rate stayed at 1.000 for both levels. The
  target, not the loss weighting, is what makes these levels unpredictable.
- **(c) refused to fit**, and the refusal is the finding: 40 positive cases in train
  and **10 in val**, against a minimum of 20. A booster trained on that many positives
  still emits scores, and those scores would look exactly like a result.
- **(d) is the only approach that produces a defensible high-acuity result**, and it
  does so by asking a different question rather than by manufacturing signal: an ESI 2
  patient who waits 20 minutes has breached a 15-minute target but not a 30-minute
  one, so positives exist inside the high-acuity group where the pooled label had
  almost none.

The cost is visible in the same table: precision at ESI 2 falls to 0.112 under (d)
against the pooled classifier's much higher precision on a much rarer event. A
target that fires more often is easier to recall and easier to over-call.

### B3. Does more history give ESI 2 enough positives?

A 3-year dataset at the same scale and seed 42 was generated
(`--n-days 1095`, **155,422 visits**, 2024-01-01 to 2026-12-30) and run through the
full pipeline in its own folder, `runs/robust_3year/`. Blocks: 93,191 train / 23,221
val / 15,617 calib / **23,393 test**.

Positive counts and results on the **test** block:

| Approach | ESI 1 positives | ESI 1 recall | Reportable? | ESI 2 positives | ESI 2 recall | Reportable? |
|---|---|---|---|---|---|---|
| pooled | **8** | 0.000 | **no** | **94** | 0.734 | **yes** |
| acuity sample weights | 8 | 0.000 | no | 94 | 0.755 | yes |
| separate ESI 1-2 model | 8 | 0.625 | **no** | 94 | 0.660 | yes |
| acuity-specific target | **141** | 0.709 | yes | **476** | 0.763 | yes |

**The answer is yes for ESI 2 and no for ESI 1.**

- **ESI 2: more history fixes it.** 29 positives on `ed_v2`'s test block became **94**
  over three years, which is comfortably above the 20-case reporting minimum and the
  25-case threshold-fitting minimum. The pooled classifier that had FNR 1.000 on
  `ed_v2` reaches **recall 0.734** here, with no change to the method at all. The
  separate ESI 1-2 model, which refused to fit on one year, now fits.
- **ESI 1: more history does not fix it.** 1 positive became **8** — still below the
  minimum, so pooled and acuity-weighted remain at FNR 1.000 and unreportable. The
  separate model produces recall 0.625 over those 8 cases and the module still marks it
  **not reportable**, which is the right call: a rate from eight patients has a
  confidence interval wider than the decision it would inform.
- **Extrapolating the counts** (1 → 8 positives for roughly 1 → 3 years of data), ESI 1
  would need something like **seven to eight years** at this department's scale to
  reach 20 positive cases under the pooled target. That is an extrapolation from very
  small counts and should be read as an order of magnitude, not a plan. A multi-site
  extract is the realistic route.
- Under the acuity-specific target, ESI 1 has 141 positives and is evaluable — again,
  by asking a different question, not by answering the same one better.

### B4. Conclusion

**(d), the acuity-specific target, is the only approach that gives a defensible
high-acuity result — and the only one that makes ESI 1 evaluable at all.**

Confirmed across all six generated datasets, test block, pooled target against
acuity-specific target:

| Dataset | ESI 1 pooled | ESI 1 acuity-target | ESI 2 pooled | ESI 2 acuity-target |
|---|---|---|---|---|
| | n+ / FNR | n+ / FNR | n+ / FNR | n+ / FNR |
| `seed42_base` | 2 / 1.000 ✗ | 62 / 0.065 | 44 / 0.227 | 207 / 0.063 |
| `seed7_base` | 5 / 1.000 ✗ | 76 / 0.118 | 45 / 0.133 | 252 / 0.123 |
| `seed99_base` | 2 / 1.000 ✗ | 61 / 0.033 | 54 / 0.296 | 213 / 0.183 |
| `scale085` | 1 / 1.000 ✗ | 54 / 0.222 | 33 / 0.333 | 195 / 0.246 |
| `scale115` | 1 / 1.000 ✗ | 90 / 0.411 | 41 / 0.195 | 214 / 0.318 |
| `v3` | 5 / 1.000 ✗ | 85 / 0.329 | 62 / 0.242 | 272 / 0.257 |

✗ = not reportable: fewer than 20 positive cases.

Reading that honestly:

1. **ESI 1 cannot be evaluated under the pooled target on any dataset.** One to five
   positive cases, FNR 1.000 every time. More data does not fix it at this scale,
   acuity weighting does not fix it, and a separate model cannot be fitted on it. Under
   the acuity-specific target ESI 1 gets 54–90 positives and becomes evaluable
   everywhere — but that is a different question being answered, not the same question
   answered better.

2. **The "ESI 2 FNR = 1.000" result is specific to `ed_v2`, not universal.** On the
   generated datasets ESI 2 has 33–62 positives under the pooled target and the pooled
   classifier reaches FNR 0.133–0.333. `ed_v2` sits just the wrong side of the
   threshold-fitting minimum (29 positives on test, 10 on validation), so its pooled
   threshold is never fitted and its recall collapses. The failure in the brief is
   real, and it is a **small-sample failure rather than a structural one**.

3. **The acuity-specific target does not always win on ESI 2.** It has a lower FNR on
   four datasets and a *higher* one on two (`scale115`: 0.318 against 0.195; `v3`:
   0.257 against 0.242). It reliably provides more positives; it does not reliably
   provide a better classifier.

4. **The precision cost is real and should be quoted with the recall.** On `ed_v2`,
   ESI 2 precision under the acuity target is 0.112. A target that fires on 20% of
   patients instead of 3% is easier to recall and easier to over-call.

5. **What is not defensible**: quoting any ESI 1 recall under the pooled target,
   quoting the `high_acuity_only` model's ESI 1 figures on `ed_v2` (recall 1.000 over
   2 positives — the module marks it unreportable and refuses to fit), or presenting
   the acuity-target numbers without the targets themselves, which are assumed.

### B5. Report generator

`src/report.py` gained `_high_acuity_block`, which renders the comparison table, the
"NOT FITTED" notes and a per-scope verdict **read from `verdict` in the artifact**, not
from a fixed sentence. `_future_work_section` gained `_high_acuity_future_bullet`,
which names the approaches that worked per acuity level — separately, because ESI 1
and ESI 2 rarely become evaluable under the same ones, and merging the lists would
credit an approach with a level it never reached.

---

## PART C — Wider sensitivity analysis

`src/simulation/sensitivity.py` keeps both original axes and adds five more. All
numbers come from `simulation.sensitivity` in the config; none is hard-coded.

| Axis | Worlds | Notes |
|---|---|---|
| service time | ×0.9, ×1.1, ×0.8, ×1.2 | unchanged |
| arrival rate | ×0.9, ×1.1, ×0.8, ×1.2 | unchanged |
| patience | ×0.7, ×1.3 | ×1.0 omitted: identical to the point estimate |
| clinician-time fraction | ×0.8, ×1.2 | length of stay held fixed, so not confounded with service time |
| acuity mix | ±5 points between ESI 2-3 and ESI 4-5 | ESI 1 held fixed |
| aging rate | ×0.5, ×2.0 on minutes-per-level | below 1 ages faster |
| aging floor | 2.5, 3.0 | **2.0 is refused and logged**, not run |

**19 worlds across 8 families** (including the point estimate), ~119 s for the stage.
Replications were **not** reduced: the user asked for full fidelity.

**The aging floor of 2.0 is never tested.** Any configured floor below the one in the
config is refused with a logged explanation. Measuring an unsafe policy to three
decimal places does not make it available.

### A noise control that was not asked for, and what it showed

`aging_floor_2.5` equals the configured floor, so that world is *configuration-identical*
to the point estimate and differs only in its random stream. It was kept rather than
skipped, because whatever it moves is Monte Carlo noise at this replication count.

On `ed_v2.csv` the noise floor is **3.546 min on mean wait, 4.796 min on p90 ESI 3 wait,
and 0.0048 on the walkout rate**. At that floor, **5 of 18 perturbations move mean wait
by less than the noise** — including both acuity-mix worlds and `aging_minutes_x0.5`.
Those are reported as effects that cannot be distinguished from noise, not as small
effects. The tornado figure shades the band and hatches the control bar.

### Results

- **The scenario ranking is stable across all 19 worlds** on mean wait. No scenario
  changed position.
- **The input the recommendation is most hostage to is the one that is least
  identifiable.** Largest effects on mean wait: `clinician_share_x1.2` **+27.6 min**,
  `service_x1.2` +25.9, `arrivals_x1.2` +24.9. The clinician-time fraction is a single
  free parameter fitted because the data never records when treatment ended, and it
  moves the headline KPI more than a 20% error in either observable input.
- On p90 ESI 3 wait: `service_x0.8` −74.2 min, `clinician_share_x0.8` −71.3,
  `arrivals_x0.8` −64.9.
- On the walkout rate: `clinician_share_x1.2` +0.0797, `service_x1.2` +0.0746.

`artifacts/figures/sensitivity_tornado.png` draws all three KPIs, coloured by axis.

**The upload job engine still skips sensitivity by design.** `sensitivity` remains
absent from `PIPELINE_STAGES` in `src/jobs/preconditions.py`; its stale "four demand
and service multipliers" wording was corrected to describe the real grid.

---

## PART D — Forecast weakness at peak hours

### The headline result: the weakness does not survive a larger sample

The brief's premise was verified first, from `metrics/forecast.json`: on the default
4-fold backtest, peak-hour MAE is **4.3636** against seasonal-naive **4.100**, and the
morning conformal band covers **0.806**. That backtest scores 288 hours, of which 60
are peak.

`src/models/forecast_study.py` and `scripts/forecast_study.py` re-run the comparison
over **26 weekly origins of 72 hours — 1,872 scored hours**, split in time into 13
validation and 13 test origins. Every tuning constant is fitted on the validation
origins; the test origins are scored once.

On the **test** origins, against seasonal-naive, with paired bootstrap intervals:

| Variant | Peak MAE | Naive | Difference (95% CI) | Beats naive? |
|---|---|---|---|---|
| `blend` *(selected)* | 3.2426 | 3.6692 | −0.4266 (−0.7286, −0.1434) | **yes, significant** |
| `poisson` (shipped) | 3.3760 | 3.6692 | −0.2932 (−0.6202, +0.0162) | yes, not significant |
| `same_hour_lags` | 3.3966 | 3.6692 | −0.2726 (−0.6120, +0.0647) | yes, not significant |
| `tweedie` | 3.6376 | 3.6692 | −0.0316 (−0.3519, +0.3070) | yes, not significant |
| `peak_corrected` | 3.6574 | 3.6692 | −0.0118 (−0.3182, +0.2917) | yes, not significant |

**The peak-hour loss in the four-fold backtest was a small-sample artefact.** Over 511
peak hours the shipped Poisson model is *ahead* of seasonal-naive at peak, not behind,
though not significantly so; the blend is ahead significantly.

Overall on the test origins: `blend` 2.2781 MAE / 35.43 WAPE, `poisson` 2.3471 / 36.50,
against seasonal-naive 2.8643. Every variant beats the baseline overall, significantly.

### Selection

Chosen on the **validation** origins at peak hours: **`blend`** (validation MAE 2.7880
against seasonal-naive 3.8347), with a weight of **0.85 on the model and 0.15 on
seasonal-naive**, fitted on the validation origins alone. The test figures above were
computed after that choice was frozen.

### A negative result worth keeping: the peak correction

The multiplicative peak correction was fitted at **×0.8522** — it scales predictions
*down*. That is the opposite of what the 4-fold diagnosis suggests, and the reason is
that the correction must key off the **prediction**, because at serving time the truth
is unknown:

- at **actual**-peak hours the model under-predicts (bias −2.44);
- at **predicted**-peak hours it over-predicts (bias **+1.52**).

Both are the same regression-to-the-mean effect seen from opposite ends, and only the
second is observable at serving time. Applying it drives aggregate bias on the
predicted-peak set to zero while making the error on *actual* peak hours worse
(peak MAE 3.6574, the worst of the five). It is kept in the comparison as a measured
negative result, and the module docstring was corrected to say so after the
measurement contradicted the original wording.

### Rolling conformal calibration, by hour band

Each interval is sized from the absolute residuals of forecasts issued at **earlier**
origins in the same hour band — only what had already been observed when the forecast
was made. The earliest origins get no interval rather than a borrowed one.

On the **test** origins, nominal 0.90, tolerance 0.04:

| Band | `poisson` | `blend` |
|---|---|---|
| overall | 0.8835 | 0.8910 |
| night | 0.8846 | 0.8889 |
| morning | 0.8761 | 0.8803 |
| afternoon | 0.8803 | 0.8932 |
| evening | 0.8932 | 0.9017 |

**The target is met on the test origins: every band is within 0.04 of nominal**, for
both variants. The specific weakness in the brief — morning coverage 0.806 — is gone:
morning covers **0.880** under rolling calibration.

**It is not met on the validation origins**, where `blend` covers 0.9444 in the morning
and 0.9583 at night, both over-covering by more than 0.04. That is the expected shape:
the earliest origins have little residual history, so their intervals start wide. Both
blocks are reported.

### What was NOT adopted, and why

**The shipped forecaster was not changed.** Adopting `blend` would move
`forecast.json → overall.model.mae` away from 2.356, which is one of the four numbers
this repository is required to reproduce. The study records the winner, and adoption is
left as an explicit, flagged change — the `adoption_note` field in
`metrics/forecast_study.json` says so, and `tests/test_headline_regression.py` is the
gate. This is a direct tension in the brief ("replace the 4-fold backtest" against
"must still reproduce 2.356"), resolved in favour of the reproduction guarantee with
the alternative measured and recorded.

### A leak found and fixed

The first version of `run_backtest` anchored its feature rows on the origin itself.
Because the same-hour lookback is anchored to the *target*, a 24-hour-ahead row could
then read the arrival count **in the origin hour — an hour that has not finished**.
`tests/test_forecast_study.py::test_same_hour_features_ignore_everything_after_the_anchor`
caught it (`7.25 != 2505.0` at the boundary row). The fix truncates the history at the
origin and anchors on the last completed hour, exactly as `forecast()` does, and
`test_backtest_predictions_cannot_see_past_their_origin` now runs the whole backtest
twice — once with the future poisoned — and asserts the predictions are identical.

The study results above were produced **after** that fix; the first run was stopped and
discarded.

Serving is unchanged: `forecast(horizon_hours, as_of)` and `/forecast` still truncate at
`as_of`, and the pre-existing leakage tests still pass.

---

## PART E — Forcing a simulation validation failure

### A limitation of the gate, found while building the fixture

The brief suggested shuffling `seen_time` offsets **within acuity**. That was built
first (`tests/upload_fixtures.decouple_waits_from_queue`, scope `within_acuity`) and
**it does not trip the gate**. Running `calibrate_sim` on it gives:

```
Simulation validation PASSED: all 12 held-out checks within tolerance
```

The reason is structural: the gate compares *marginal* KPIs — per-acuity mean and p90
wait, length of stay, abandonment — and a within-acuity permutation preserves every one
of them **exactly**. A dataset whose waits are completely independent of the queue
passes validation. Catching that would need a check on the dependence itself, for
example the correlation between queue length at arrival and the realised wait.

This is recorded in the fixture's docstring and asserted by
`test_within_acuity_decoupling_slips_past_the_gate`, which is written to fail loudly
(with instructions to delete it) if the gate is ever strengthened.

### E1-E2. The fixture that does fire the gate

Scope `across_acuity` permutes waits over the whole department, which also flattens the
acuity ordering — every ESI level ends up with the same mean wait, which a priority
queue cannot produce. Validation then fails:

```
SIMULATION VALIDATION FAILED: 3 of 12 held-out checks outside tolerance.
  mean wait ESI 1: simulated 5.445 vs observed 35.439 (relative error 0.846 > tolerance 0.20)
  mean wait ESI 2: simulated 5.639 vs observed 35.633 (relative error 0.842 > tolerance 0.20)
  mean wait ESI 5: simulated 46.026 vs observed 35.171 (relative error 0.309 > tolerance 0.20)
```

Both downstream stages refuse, naming every failed metric:

- `stage_scenarios` → `ValidationError: The simulation does not reproduce the observed
  department, so scenario results would be meaningless. Failures: mean wait ESI 1 ...`
  (exit code 1)
- `stage_staffing_search` → the same refusal (exit code 1)

**No override is reachable from the API or the dashboard.** `build_run_config` in
`src/jobs/storage.py` pins `simulation.allow_invalid_calibration = False` for every
job regardless of the config file, and `test_no_api_or_dashboard_code_turns_the_override_on`
scans every `.py` under `api/` and `app/` for the flag and fails if it appears.

### E3. Through the upload job engine, end to end

A 120-day decoupled extract uploaded through `JobEngine`:

- the job **completes** (`JobStatus.SUCCEEDED`) rather than failing;
- `calibrate_sim` is `DONE` — it is what produces the validation table;
- `scenarios` and `staffing_search` are `SKIPPED`, each carrying a reason containing
  "FAILED validation" and "refused";
- no traceback appears in the job error or in any stage reason;
- `tables/sim_validation.csv` is written, with the failing rows in it.

`tests/test_validation_gate.py`: **15 tests, all passing** (44.8 s for the calibration
set, 78.6 s for the two end-to-end upload tests). A new AppTest,
`test_a_refused_simulation_still_shows_its_validation_table`, asserts the page renders
the refusal **and** both evidence tables, with `mean wait ESI 1` and `mean wait ESI 2`
visible in the rendered dataframes and no traceback.

---

## PART F — More realistic wait levels (ed_v3)

New generator flags, all of them config overrides: `--profile v3`, `--doctor-delta`,
`--load-multiplier`, `--patience-scale`, `--boarding-scale`, `--calibration-report`.
**v2 behaviour is unchanged by default** — `--profile v2` is the default and touches
only the visit count.

Boarding turned out to be the lever that makes the calibration tractable: the generator
releases the doctor after treatment and adds boarding to the departure time only, so
`--boarding-scale` moves post-contact time and length of stay **without changing any
wait**. Verified directly: at boarding ×3.0/×3.2/×3.4/×3.6 the median wait stayed at
25.8 min while post-contact moved 84.1 → 87.1 → 90.1 → 93.0.

A 32-point grid over `doctor_delta`, `load_multiplier` and `patience_scale` found
exactly one combination meeting every target. Final settings: **load ×1.08, patience
×2.0, boarding ×3.4, no change to doctor counts, seed 42, 57,500 visits.**

`artifacts/tables/calibration_report.csv`, **6 of 6 targets met**:

| Target | Band | Achieved | Status |
|---|---|---|---|
| median wait, seen patients | 25–30 min | **25.85** | PASS |
| LWBS overall | 2–4% | **2.42%** | PASS |
| LWBS ESI 5 | < 8% | **5.23%** | PASS |
| LWBS ESI 1-2 | 0 | **0.0000** | PASS |
| median wait ESI 1 | < 12 min | **3.87** | PASS |
| median post-contact | ~90 min (80–100) | **90.1** | PASS |

**Removing a doctor is too coarse a lever to be useful here.** `doctor_delta=1` at
unchanged load jumps median wait from 16.5 to **45.9 min** with 8.1% abandonment. The
department sits near a congestion knee: load ×1.08 gives 25.8 min, load ×1.15 gives
42.7 min. Higher load, not tighter staffing, is what reaches the target band.

### What changes on v3 against v2

`robust_v3` ran all 15 stages and is row 7 of `robustness_summary.csv`.

**Unchanged.** The scenario ranking is byte-identical to every other dataset
(`fast_track_esi45` first), and stable under all 19 perturbations. The model still
beats its baseline by the same factor (25.6 against 85.5, a ratio of 3.34 against
2.97 on ed_v2). The forecast still beats seasonal-naive overall (2.58 against 3.42)
**and at peak** (4.37 against 5.17). Simulation validation passes. Rolling conformal
coverage is 0.8896, within 0.004 of its value on every other dataset.

**Changed.**

- **The ablation gap triples**: 49.0 minutes against 14.6–18.1 everywhere else. A more
  congested department makes queue state far more informative, which is the expected
  direction and makes the G3 caveat sharper rather than weaker — the measured value of
  queue features is a function of how congested the data is, so no single number for
  it means much.
- **The staffing recommendation moves to a different shape.** v2 recommends
  `day=3, evening=3, night_early=2, night_late=2` at 49,280; v3 recommends
  `day=4, evening=4, night_early=1, night_late=1` at **55,440**, with only **42 of 144**
  plans feasible against v2's 72. More doctors in the day and evening, fewer overnight.
  That is the staffing search responding correctly to a busier department, and it is
  also the clearest demonstration that **the cheapest-plan result is a property of the
  dataset, not a finding about emergency departments**.
- **The simulation gate passes with much less margin**: worst held-out relative error
  0.5588 (abandonment rate ESI 3) against 0.2810 on `seed42_base`. It passes only
  because the absolute floor (0.01 on a rate) catches it; the relative error alone is
  above the 0.50 tolerance. On a slightly different draw this dataset would fail
  validation the way `seed7_base` did.
- **ESI 2 becomes more predictable**: 62 pooled positives on the test block against
  `ed_v2`'s 29, with FNR 0.242. Longer waits mean more prolonged-wait positives at
  every acuity. ESI 1 does not change — 5 positives, FNR 1.000, still unreportable.

---

## PART G — Small items

### G1. Cox proportional-hazards checks

`src/models/survival.py` now fits each cause-specific model once and derives both the
hazard-ratio table and a Schoenfeld-residual proportional-hazards test from that single
fit, so the two cannot disagree. New artifact `tables/survival_cox_ph_test.csv`, new
`cox_proportional_hazards` and `cox_proportional_hazards_summary` keys in
`metrics/survival.json`.

On `data/raw/ed_v2.csv`, 12 covariate tests, **5 reject the assumption at p < 0.05**:

| Cause | Covariate | Test statistic | p | Violates? |
|---|---|---|---|---|
| seen | `n_waiting` | 1365.59 | 6.3e-299 | **yes** |
| seen | `hour_sin` | 718.98 | 2.2e-158 | **yes** |
| seen | `hour_cos` | 602.32 | 5.2e-133 | **yes** |
| seen | `triage_level` | 12.39 | 4.3e-04 | **yes** |
| seen | `age` | 1.46 | 0.227 | no |
| seen | `is_ambulance` | 3.54 | 0.060 | no |
| abandoned | `triage_level` | 4.60 | 0.032 | **yes** |
| abandoned | all others | ≤ 0.68 | ≥ 0.41 | no |

**What that limits.** A hazard ratio for a violating covariate is still a valid
*time-averaged* summary, weighted towards the times when most patients are still
waiting, but it is not the effect at any particular minute. `n_waiting` is the clearest
case: queue depth at arrival plainly matters more early in a wait than late. The caveat
is carried on the hazard-ratio rows themselves (`violates_proportional_hazards` and
`interpretation` columns) so reading `survival_cox.csv` alone still shows it. A test
that cannot be computed is reported as **untested**, never as passing.

### G2. Optional API authentication

New `api/auth.py`. With `ED_API_KEY` unset the service behaves exactly as before — the
default is unchanged, and every pre-existing API test still passes untouched. With it
set, every endpoint except `/health`, `/docs`, `/redoc`, `/openapi.json` and
`/docs/oauth2-redirect` requires a matching `X-API-Key` header.

- Comparison is `hmac.compare_digest`, including against a same-length dummy when the
  header is absent, so a missing header is not measurably faster to reject than a wrong
  one. A test asserts `compare_digest` appears in the comparison's source.
- A missing key and a wrong key return **byte-identical** responses.
- The key is **never logged**: a test drives both a valid and an invalid request at
  `DEBUG` level and asserts the key does not appear in the captured log.
- `/health` reports `authentication_required` so an operator can diagnose "everything
  returns 401" without reading the server's environment.
- The dashboard sends the header from the same `ED_API_KEY` variable. The header name
  is **duplicated** in `app/config.py` rather than imported from `api/`, because the
  dashboard image installs `requirements-app.txt`, which has no FastAPI; a test asserts
  the two constants stay equal.

Added to `.env.example` (commented out, with a paragraph on what it is and is not),
both services in `docker-compose.yml`, and `README.md` — where the "Not built" entry
was rewritten rather than deleted, because one shared secret with no rotation, no
revocation, no per-caller audit, no rate limiting and no TLS is not an access-control
system and the README now says exactly that.

`tests/test_api_auth.py`: **20 tests, all passing.**

### G3. Container scan

`docker scout` 1.24.0 was available; **Trivy was not installed on this machine**, so
these are Scout's findings only. Full write-up in `docs/container_scan.md`.

| Image | Critical | High | Medium | Low |
|---|---|---|---|---|
| `ed-intelligence-api` | 0 | 9 | 10 | 31 |
| `ed-intelligence-app` | 0 | 15 | 14 | 28 |

**No critical findings.** Every high was triaged:

| Package | Highs | Decision |
|---|---|---|
| `python-multipart` 0.0.20 | 3 | **bumped to 0.0.30** — parses the upload body, touches no model code |
| `pyarrow` 17.0.0 | 1 (both images) | **bumped to 23.0.1** — parquet round-trip only; float64 is exact |
| `pillow` 11.3.0 | 13 (app) | **blocked**: `streamlit==1.41.1` requires `pillow<12`. Needs a Streamlit upgrade plus a human browser pass. The API image already resolves pillow to 12.3.0 and is clean. All 13 are image-*decode* issues and the dashboard decodes no uploaded images. |
| `starlette` 0.41.3 | 3 | **blocked**: `fastapi==0.115.6` pins `starlette<0.42`; needs a FastAPI major upgrade with its own tested pass |
| `lightgbm` 4.5.0 | 1 | **deliberately not bumped**: CVE-2024-43598 is reached by parsing a malformed *model file*, and nothing here loads an untrusted one. Bumping it would move 11.52 / 0.8875 / 2.356. The procedure for doing it deliberately is in `docs/container_scan.md`. |
| `zlib` (Debian) | 1 | **no fix exists** — Scout reports "not fixed"; comes from `python:3.12-slim`, which Scout still reports as current |

### G4. Manual UI checklist

**`docs/ui_manual_test_checklist.md` does not exist.** The repository has
`docs/upload_manual_test_checklist.md`, which is the same document under a different
name, so that file was extended rather than a near-duplicate created under the name in
the brief.

Added: a prominent **"Status: NOT PERFORMED"** banner, a five-minute "short run" table
(what to click, what should appear, and what should make you stop), a four-minute
refusal-path run using the Part E decoupled fixture, and a two-minute API-key run.

**None of it has been performed.** This session has no browser and cannot see a screen.

---

## PART H — Final integration

### H1. Report generation

`src/report.py` gained five data-driven blocks, all of which read artifacts and all of
which degrade to nothing (or to a named absence) when the artifact is not there:

| Block | Reads | Appears in |
|---|---|---|
| `_high_acuity_block` | `high_acuity_comparison.csv`, `wait_models.json → high_acuity` | Wait-time prediction |
| `_cox_assumption_block` | `survival_cox_ph_test.csv`, `survival.json` | Survival analysis |
| `_sensitivity_block` | `sensitivity.json`, `sensitivity_tornado.csv`, the figure | Simulation and staffing |
| `_forecast_study_block` | `forecast_study.json`, its scores and coverage tables | Arrival forecasting |
| `_robustness_section` | `robustness_summary.csv` and its across-seed companion | **new top-level section** |

Two existing claims that had hardened into prose were replaced by generated text:

- the sensitivity sentence said "across +/-10% and +/-20% errors in service times and
  arrival rates", which became untrue the moment the grid widened. It now reads the
  world count and the axis list out of `sensitivity.json`.
- the future-work bullet on deep forecasters asserted a motivation. It now reads the
  study's test-block result and says *not currently justified*, *not clearly
  triggered* or *triggered* accordingly. On the current artifacts it says **not
  currently justified**, because the peak-hour deficit it existed to fix does not
  survive 26 origins.

The regenerated `artifacts/report.md` is 528 lines.

### H2. Containerised reproduction

Images rebuilt (`docker compose build --no-cache`, then a cached rebuild to pick up
the final config) and the stack started. `GET /health` returned
`status: ok, models_loaded: true, authentication_required: false`.

`data/raw/ed_v2.csv` was uploaded through the containerised API with
`scripts/smoke_upload.ps1 -Depth full`. The job **succeeded in 115.7 s**, 15 stages,
57,500 rows in / 57,179 valid / 321 dropped.

Read back out of the container's own run folder at full precision:

| Figure | Pinned baseline | Containerised run | Match |
|---|---|---|---|
| wait MAE, model | 11.524 | **11.524** | yes |
| wait MAE, baseline | 34.232666504025374 | **34.232666504025374** | yes |
| rolling conformal coverage | 0.8875 | **0.8875** | yes |
| forecast MAE | 2.3556913228196534 | **2.3556913228196534** | yes |
| cheapest feasible plan | 49,280 | **49,280** | yes |
| feasible plans | 72 of 144 | **72 of 144** | yes |
| plan roster | `day=3, evening=3, night_early=2, night_late=2` | **identical** | yes |

**All four headline numbers reproduce exactly inside the container, with the bumped
`pyarrow` 23.0.1 and `python-multipart` 0.0.30.** That is also the evidence that those
two version bumps are safe.

The same four were verified a second time from a fresh *host* run into
`runs/verify_baseline/` before the baseline tree was regenerated, so the final
`artifacts/` is a reproduction rather than a leftover.

**The optional API key was verified end to end in the running stack**, not only in
unit tests. With `ED_API_KEY` set in the environment and the API recreated:

| Request | Result |
|---|---|
| `GET /health` | **200**, `authentication_required: true` |
| `GET /forecast`, no header | **401** |
| `GET /forecast`, wrong key | **401** |
| `GET /forecast`, correct key | **200**, 2,540 bytes |
| after unsetting and recreating | `/forecast` **200** again, `authentication_required: false` |

One observation worth recording: deleting the smoke job afterwards returned **404**,
because recreating the API container cleared the in-memory job list. That is the
documented "no durable job store" limitation behaving as described — the run folder
survives in the `ed-intelligence-runs` volume, the job *record* does not.

**State as this was written:** Docker Desktop has since stopped (the machine idled
between sessions), so the stack is **not currently running**. Everything in this
section was measured while it was up and is recorded above rather than re-derived.
Bring it back with `.\scripts\docker_up.ps1`, or `docker compose up -d api app`. The
smoke job's run folder is still in the `ed-intelligence-runs` volume and is not
cleaned up by anything on the host.

### H3. Test suite

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

```
1100 passed, 11 skipped, 145 warnings in 832.18s (0:13:52)
exit code 0
```

Before this work the same command reported **994 passed, 11 skipped in 572.18 s**. So
**106 tests were added and none were removed or weakened.** The 11 skips are the
pre-existing conditional ones (5 views draw no Altair charts, 5 more for the chart
caption rule, and one helper module that is not a page).

New test files:

| File | Tests | Covers |
|---|---|---|
| `tests/test_headline_regression.py` | 10 | the four pinned numbers and the override flag |
| `tests/test_high_acuity.py` | 12 | acuity targets, weights, the refusal rules |
| `tests/test_sensitivity_axes.py` | 17 | the five new axes, the unsafe-floor refusal, the noise control |
| `tests/test_forecast_study.py` | 16 | same-hour leakage, selection discipline, rolling intervals |
| `tests/test_validation_gate.py` | 15 | the gate firing, both decoupling scopes, end-to-end upload |
| `tests/test_cox_assumptions.py` | 15 | PH diagnostics and the subsampling guard |
| `tests/test_api_auth.py` | 20 | the optional key, on and off |

plus one added to `tests/test_upload_dashboard.py` for the refusal screen.

---

## Bugs found and fixed during this work

1. **Forecast-study leakage** (found by a test I wrote, before any result was
   produced). `run_backtest` anchored feature rows on the origin, letting a
   24-hour-ahead row read the unfinished origin hour through the same-hour lookback.
   Fixed by truncating the history and anchoring on the last completed hour. The first
   study run was discarded and re-run.
2. **`high_acuity_comparison.csv` unreadable by the report.** The file carries its
   caveat as a leading `#` line, and `ArtifactReader.table` applied `comment="#"` only
   to `ablation.csv`, so the caveat became the header. Fixed with a `CAVEATED_TABLES`
   set.
3. **A stale fatal path.** `compare_high_acuity_approaches` raised `KeyError` when a
   config predated `features.acuity_target_wait_minutes`, which killed the whole
   `train_wait` stage for any run started from an older config snapshot. Against
   CLAUDE.md's "a stage that cannot run says why, in English", it now logs a reason and
   skips the comparison.
4. **CRLF line endings.** Rewriting files through Python on Windows converted several
   to CRLF, breaking `test_the_compose_files_have_no_crlf`. The files touched were
   normalised back to LF. (The repository was already mixed beforehand — `requirements.txt`,
   `schema.yaml`, `src/data/adapter.py` and others are CRLF and were not touched.)
5. **A wrong claim in a docstring.** `fit_peak_correction` asserted a negative bias at
   peak; the measurement showed **+1.52** at predicted-peak hours. The docstring was
   rewritten to describe what was measured.
6. **The new Cox diagnostic was a denial-of-service risk on large uploads.** The
   Schoenfeld test took ~24 s at 57,000 rows but **272 s** at 155,000 on the 3-year
   dataset — the survival stage went from ~25 s to 283 s, and the test, not the model
   fit, was all of the difference. The upload workflow accepts up to 500,000 rows
   against an 1,800 s stage timeout, so an unguarded test could have spent a job's
   entire budget on a diagnostic. Added `survival.ph_test_max_rows` (default
   **100,000**, chosen so `ed_v2`'s 57,179 rows are still tested in full and the
   baseline diagnostic is unchanged): above it the model is refitted on a seeded
   subsample and the test runs on that, with `n_tested`, `n_fitted` and
   `diagnostic_subsampled` recorded on every row. Hazard ratios are always fitted on
   all rows; only the assumption test is ever subsampled. Five tests cover it.

---

## Interpretations and assumptions worth disputing

- **"Scale ±15%" was read as dataset size**, not as load. The ±15% runs have the same
  arrival rate and a longer or shorter span. Load variation is covered by Part C's
  arrival-rate worlds (±10%, ±20%) and by v3's ×1.08. If the brief meant load, those
  two runs answer a different question than intended.
- **The per-acuity targets are invented.** They are plausible round numbers chosen so
  the alternative target could be measured at all. Every ESI 1-2 rate in Part B moves
  when they move.
- **The v2 profile is not `ed_v2.csv`.** Its parameters are unrecoverable; the profile
  is close on wait behaviour and visibly different on post-contact time.
- **The forecast winner was not adopted**, because adopting it breaks the reproduction
  guarantee. That is a judgement about which instruction wins.

---

## What was NOT verified

- **Any human browser interaction.** No page in `app/` was opened in a browser. Every
  dashboard claim here rests on Streamlit's `AppTest`, which renders the component tree
  without a browser. `docs/upload_manual_test_checklist.md` is marked NOT PERFORMED.
- **Any machine other than this one.** Windows 11, Python 3.12, one CPU. No Linux host,
  no other Python version, no CI.
- **Trivy.** Not installed; only Docker Scout was run. A second scanner would likely
  surface a different set, particularly on the OS layer.
- **The blocked security bumps.** `pillow` and `starlette` were not upgraded, so the
  images were not re-scanned with them. `lightgbm` 4.6.0 was not tested against the
  reproduction.
- **Licence review.** Scout's copyleft policy fails on both images (366 packages). Not
  examined.
- **Medium and low CVEs.** Not itemised.
- **`ed_v2.csv`'s provenance.** The generator that made it is gone; the v2 profile is a
  reconstruction, not a recovery.
- **Real data.** Everything here is synthetic. Nothing in this work is evidence about
  any real emergency department.
