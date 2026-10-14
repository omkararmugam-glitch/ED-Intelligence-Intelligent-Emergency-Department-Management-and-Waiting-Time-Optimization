# MIMIC-IV-ED demo: adapter check and data profile

**No model was fitted and no model performance is reported here.** The open demo
holds **222 ED stays**. That is enough to prove the adapter reads real MIMIC tables
and enough to describe what those tables contain; it is nowhere near enough to fit or
evaluate anything. A figure computed from 222 stays would be quoted later as though
it meant something, so none is produced.

Data: MIMIC-IV-ED demo v2.2, ODbL v1.0, attribution in `data/external/README.md`.
Retrieved 2026-10-14 by `scripts\fetch_mimic_ed_demo.ps1`; the archive's own
`SHA256SUMS.txt` verified, 7 of 7 files.

Produced by `scripts\external_validation_mimic.py`.

---

## 1. Does the column map work on real data?

`configs/column_map_mimic_ed.yaml` was written from MIMIC's published schema **without
any data in hand**, and every column name in it was labelled a claim to be checked.
This is the check.

**Every column name in the template exists in the real tables.** The adapter mapped
all 222 stays to the schema's ten input columns using the project's own
`adapt_real_dataset`, unchanged — the same path the pipeline would take on any real
extract.

**One mapping was wrong, and is now corrected.** The template sent
`LEFT AGAINST MEDICAL ADVICE` to `lwbs`. It is not: leaving against advice means a
clinician saw the patient and advised them, so the patient *was* seen. Mapping it to
`lwbs` would have inflated the abandonment rate with patients who had already been
assessed — and abandonment is one of the quantities the simulation's validation gate
is judged on. It now maps to `other`.

Two other things the run surfaced, neither a defect:

- `anchor_age` is in MIMIC-IV's `patients` table, which the **ED** demo does not ship.
  The adapter reported all 222 ages missing, which is the right answer rather than an
  error.
- 15 of 222 stays have no triage acuity. The adapter counted them and carried on.

### Disposition mapping, as applied

| MIMIC value | Count | Mapped to |
|---|---|---|
| `ADMITTED` | 150 | admitted |
| `HOME` | 60 | discharged |
| `TRANSFER` | 5 | transferred |
| `LEFT WITHOUT BEING SEEN` | 2 | **lwbs** |
| `ELOPED` | 1 | **lwbs** |
| `LEFT AGAINST MEDICAL ADVICE` | 2 | other *(corrected)* |
| `OTHER` | 2 | other |

Giving `admitted` 150, `discharged` 60, `transferred` 5, `other` 4, `lwbs` 3.

---

## 2. The profile

222 stays. Acuity, as recorded in `triage.acuity`:

| ESI | Stays |
|---|---|
| 1 | 18 |
| 2 | 97 |
| 3 | 90 |
| 4 | 2 |
| 5 | 0 |
| not recorded | 15 |

**This is not an emergency department's acuity mix.** A real department is dominated
by ESI 3 to 5; this sample is dominated by ESI 2 and has no ESI 5 at all, because the
demo is a *sample of patients chosen for the demo*, not a window of operation. It is
the first and clearest reason nothing about volumes or case mix can be learned here.

**Length of stay** (arrival to departure, which MIMIC records directly and is *not* a
proxy): median **350.5 min**, IQR 253.2–526.1, p90 943.2, max 4,460.

**Arrival hour** is recorded for every stay, so the shape of the day is available —
but with 222 stays spread over the shifted dates it describes the sample, not a
department.

---

## 3. The proxy wait, measured on real data for the first time

MIMIC-IV-ED has **no first-provider-contact timestamp**. `edstays` gives `intime` and
`outtime` and nothing in between. This project's `proxy_wait` option fills `seen_time`
from a named proxy and labels everything downstream.

With `first_vitalsign_charttime`, the adapter derived `seen_time` for **206 of 222
stays (92.8%)** and emitted its own warning. Every figure in this section is a

> **proxy wait (not true door-to-provider)**

| Proxy | Stays covered | Median | p90 |
|---|---|---|---|
| First `vitalsign.charttime` | 206 (92.8%) | **2.0 min** | 190.5 min |
| First `pyxis.charttime` | 176 (79.3%) | **118.0 min** | 306.6 min |

**The two proxies bracket the truth from opposite sides, exactly as
`docs/real_data_readiness.md` predicted from the schema alone — and the gap is now
measured rather than argued.** Among the 167 stays that have both, the medication
dispense is later in **80.2%** of cases, by a median of **76 minutes**.

A median of 2 minutes for first vitals is the strongest possible confirmation that it
is recorded *at triage*: it is essentially the arrival time. Using it as
door-to-provider would report almost every patient as seen immediately. The 76-minute
median gap between the two proxies is the width of the disagreement about what is
being measured, and any wait figure from either carries that uncertainty.

---

## 4. Date shifting, and what it rules out

MIMIC-IV shifts every patient's dates into a random future year, **per patient**.
Intervals *within* one stay survive exactly — arrival to departure is real, which is
why the length-of-stay distribution above is usable. Two patients' timestamps are not
on a common clock.

The demo makes this visible in a way no documentation could: **222 stays whose arrival
times span 32,548 days — 89 years — at 0.01 stays per day.** That is not a quiet
department. It is the shift.

### What this project could validate on MIMIC-IV-ED

Even with full credentialed access:

| Can be validated | Cannot be validated |
|---|---|
| The adapter and the column mapping — which is what this run demonstrates | **Queue-state features**: `n_waiting`, `n_waiting_higher_priority`, `arrivals_last_1h` and the rest. They are counts over *concurrent* patients, and concurrency does not survive per-patient date shifting |
| Per-patient features using only that patient's own record: acuity, arrival mode, time of day, length of stay | **The simulation's validation gate.** It compares simulated against observed per-acuity mean and p90 wait, length of stay and abandonment for a department running as a queue, which needs a real window of operation |
| Disposition and acuity distributions | **The staffing search**, which is scored by that simulation |
| The proxy-wait distribution, labelled as proxy | **Door-to-provider wait itself** — the timestamp does not exist, with or without credentials |

**Credentialed access removes the sample-size problem and nothing else on that list.**
The date shifting and the missing provider-contact timestamp are properties of the
released data, not of the demo subset. This matters for the project's central claim:
the queue-state features are what make the wait model beat its baseline by a factor
of three, and **MIMIC-IV-ED cannot test them at all**.

---

## 5. What this run does and does not establish

**Establishes:** the adapter reads real MIMIC-IV-ED tables; the column map's names
are real; one of its value mappings was wrong and is fixed; the proxy-wait mechanism
works and labels itself; the two proxies differ by a median of 76 minutes on real
data; and the queue state cannot be reconstructed from this dataset.

**Does not establish:** anything whatsoever about this project's accuracy on real
patients. No model touched this data.
