# Real-data readiness: MIMIC-IV-ED

**Nothing in this document has been run against MIMIC-IV-ED.** No MIMIC data was
downloaded, accessed or inspected. Everything here is written from the published table
and column documentation, and every column name should be treated as a claim to verify
against the real extract rather than a fact.

The purpose is to make the remaining work explicit and to stop the first person who
gets access from discovering the hard parts by accident. There is one hard part, and
it is not a data-engineering problem.

---

## 1. The problem that does not have a clean solution

**MIMIC-IV-ED records no first-provider-contact timestamp.**

`edstays` gives `intime` (ED registration) and `outtime` (departure), and nothing in
between. There is no column for "a clinician saw this patient". Every wait-time model
in this repository predicts `seen_time - arrival_time`, so without that column the
central target does not exist.

This is not a mapping difficulty. It is a different quantity.

### The two proxies, and why each is wrong

| Proxy | Source | Direction of the error | Who it is missing |
|---|---|---|---|
| First vitals | earliest `vitalsign.charttime` | **Under**-estimates: vitals are usually charted *at triage*, by a nurse, before any clinician is involved | Patients with no vitals charted at all |
| First dispense | earliest `pyxis.charttime` | **Over**-estimates: a dispense happens only after a clinician has decided something | Everyone not given a medication — a biased subset, correlated with acuity |

They bracket the truth from opposite sides, and the gap between them is not noise. A
fast-triaged patient who then waited three hours has a first-vitals time minutes after
arrival; the same patient's first dispense may be after the wait ended. Choosing
between them changes the target, the 30-minute prolonged-wait threshold's meaning, the
conformal interval widths, and the simulation's fitted patience and clinician-time
parameters.

### What the code does about it

`configs/column_map_mimic_ed.yaml` maps `seen_time` to a column name that **does not
exist**, so an unconfigured run fails loudly rather than inventing a wait. To use the
dataset at all you must switch on `proxy_wait:` and name the proxy in words — the
adapter refuses an unnamed one, because every number downstream inherits the choice:

```yaml
proxy_wait:
  enabled: true
  source: first_vitalsign_charttime
  proxy_name: "first vitalsign charttime (usually recorded at triage, before a clinician)"
```

The adapter then logs `PROXY_WAIT_NOTICE` at WARNING on every load, and
`src.data.adapter.proxy_wait_status(config)` gives the report and the dashboard the
label they are required to show:

> **PROXY WAIT (NOT TRUE DOOR-TO-PROVIDER).** `seen_time` was derived from a proxy
> timestamp because the source dataset records no first-provider-contact time. Every
> wait, every prolonged-wait label and every simulation KPI derived from it measures
> the proxy, not the door-to-provider interval the models were designed for. The two
> differ systematically, not merely noisily.

**Recommendation.** Run both proxies and report both. If a conclusion holds under
either, it is probably about the department. If it only holds under one, it is about
the proxy. Do not average them: they are not two measurements of one thing.

---

## 2. Joining the tables

MIMIC-IV-ED is several tables; the adapter reads one. The join is a data-engineering
step performed before the pipeline, keyed on `stay_id`, and it must produce **one row
per ED stay**.

| Table | What it contributes | Note |
|---|---|---|
| `edstays` | `stay_id`, `subject_id`, `intime`, `outtime`, `gender`, `arrival_transport`, `disposition` | the spine |
| `triage` | `acuity`, `chiefcomplaint`, first vitals at triage | one row per stay |
| `vitalsign` | `charttime` — take the **minimum** per `stay_id` as `first_vitalsign_charttime` | many rows per stay |
| `pyxis` | `charttime` — take the **minimum** per `stay_id` as `first_pyxis_charttime` | many rows per stay; absent for many stays |
| `medrecon` | nothing usable here | reconciliation on arrival, no timestamp we need |
| `patients` (hosp module) | `anchor_age` | **not in the ED module**; a separate join |

Two traps in that join:

- **`subject_id` is not `stay_id`.** A patient may have many stays. Keying on
  `subject_id` silently collapses them.
- **A left join on `pyxis` drops nobody; an inner join drops the untreated.** If you
  use the dispense proxy, the patients with no dispense are not missing at random.

---

## 3. What the mapping loses

The template maps MIMIC's vocabulary onto this project's schema, and the schema is
coarser. These are real losses, asserted in `tests/test_mimic_adapter.py` so they are
visible rather than silent:

- **`ELOPED`, `LEFT WITHOUT BEING SEEN` and `LEFT AGAINST MEDICAL ADVICE` all become
  `lwbs`.** MIMIC distinguishes them and the distinction is clinically real — leaving
  before triage is not the same as leaving against advice after treatment. The
  project's schema has one category. Revisit this against the data.
- **`HELICOPTER` becomes `ambulance`**, and `UNKNOWN` becomes `other`.
- **Ages over 89 are shifted** in MIMIC for de-identification, so the oldest age band
  is not a true age. Every age-band fairness result inherits that.
- **A null `acuity`** (triage not completed) stays null and is rejected by the
  validator, which counts it. Do not impute it.

---

## 4. The dataset-shift check

`src.data.shift_check.compare_to_reference` compares a loaded extract with the
synthetic profile the models were tuned on, across the four things that would change
what the right model is:

| Comparison | Statistic | Flagged above |
|---|---|---|
| arrival-hour shape | total variation distance | 0.10 |
| acuity mix | total variation distance | 0.05 |
| median wait (acuity-weighted) | relative change | 0.25 |
| LWBS rate | absolute change | 0.02 |

It is a **warning, not a gate** — nothing stops the run. The thresholds are judgement
calls, stated in `THRESHOLDS` so they can be argued with. Verified against the
synthetic dataset itself, where all four comparisons read 0.

MIMIC will flag most of these. That is the expected and correct result: a real urban
academic ED is not the synthetic department, and the point of the check is that nobody
should be able to run this pipeline on real data without being told so.

---

## 5. Before any claim is made from real data

This is the part that is not code. All five, in order:

1. **Governance.** PhysioNet credentialing, the CITI "Data or Specimens Only Research"
   training, and the signed data use agreement for MIMIC-IV-ED. Institutional review
   as your institution requires it — MIMIC is de-identified and usually qualifies as
   non-human-subjects research, but *usually* is not a determination you get to make
   for yourself. Nothing below starts before this is finished.

2. **De-identification discipline on your side.** MIMIC arrives de-identified; keeping
   it that way is your responsibility. Do not export `chiefcomplaint` free text into
   anything shared — it is a re-identification surface, it is mapped only because the
   schema has the column, and no model here uses it. Do not put extracts in the
   repository, in `data/raw/`, or through the upload workflow, which writes files to
   disk and is built for synthetic demonstration data.

3. **Retrain everything, end to end.** Not "score the existing models". The LightGBM
   hyperparameters, the classifier's decision threshold, the per-acuity thresholds, the
   conformal calibration and the forecaster's lags were all selected against the
   synthetic data. Run the full pipeline with `--data real` and treat every number as
   new. The upload workflow's scoring-only mode is the wrong tool here and says so.

4. **Re-run the simulation validation gate and respect its answer.** The gate is the
   only thing standing between a mis-specified simulation and a confident staffing
   recommendation. Expect it to fail at first: the fitted clinician-time fraction
   (0.207 on synthetic data) encodes a department that may not resemble a real one, and
   `simulation.resource_extensions` models beds, nurses and imaging that MIMIC does not
   record either. **Do not widen the tolerances to make it pass.** If it fails, the
   scenario and staffing results are refused, and that refusal is the finding.

5. **Re-establish the headline numbers from scratch.** The figures this repository
   reproduces — wait MAE 11.52 against a 34.23 baseline, rolling conformal coverage
   0.8875, forecast MAE 2.356, a cheapest feasible plan at 49,280 — are properties of
   `data/raw/ed_v2.csv` and of nothing else. They are pinned by
   `tests/test_headline_regression.py` so that work on the synthetic pipeline cannot
   move them. They are **not** a benchmark for real data and must not be quoted
   alongside it.

---

## 6. What is still missing after all of that

Even with MIMIC loaded, retrained and validated, this project would still have:

- **no true door-to-provider wait** — the proxy problem does not go away with more
  work; it needs a dataset that records the timestamp;
- **no bed, nurse or imaging data**, so the resource extensions in
  `simulation.resource_extensions` stay uncalibrated scenario analysis on real data
  exactly as they are on synthetic;
- **one site**, so nothing about external validity;
- **retrospective data only**, so nothing about whether showing a prediction to a
  clinician changes what happens next, which is the question that matters for a
  decision-support tool.
