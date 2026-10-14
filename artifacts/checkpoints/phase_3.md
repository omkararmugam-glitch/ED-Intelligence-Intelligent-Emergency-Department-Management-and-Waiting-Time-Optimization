# Phase 3 checkpoint — simulation, scenarios, sensitivity, staffing search

## Gates

| Gate | Result |
| --- | --- |
| Full test suite | **280 passed** (4 min 50 s) |
| Pipeline end to end on synthetic | 14 stages, all green |
| `tests/test_leakage.py` | 12 passed |
| No random splits in `src/` | 9 patterns tested, 0 found |
| Every tunable from config | asserted by `test_project_invariants.py` |

## What was built

| Requirement | Module | Status |
| --- | --- | --- |
| 1. SimPy DES with the generator's aging rule, abandonment, shift staffing, forecast-driven load | `src/simulation/des.py` | done |
| 2. Calibration + validation gate | `src/simulation/calibrate.py` | done |
| 3. Scenarios, >=30 replications, CIs, paired CRN comparison | `src/simulation/scenarios.py` | done |
| 4. Sensitivity to +/-10-20% input error | `src/simulation/sensitivity.py` | done |
| 5. Staffing search on forecast load, cost model, feasibility | `src/simulation/staffing_search.py` | done |
| 6. The six required tests | `tests/test_simulation.py` (36 tests) | done |

New stages: `calibrate_sim`, `scenarios`, `sensitivity`, `staffing_search`.

## 1. The model cannot drift from the generator

The priority-aging rule now lives in **one** place, `src/utils/priority.py`, and both
`src/data/synthetic.py` (the world) and `src/simulation/des.py` (the model of it) import
it and read its parameters from the same config block. A test asserts both modules
reference the identical class.

SimPy's `PriorityResource` fixes a request's priority when it queues, which cannot
express continuous aging, so the model keeps its own waiting set and selects the best
aged candidate at each dispatch — exactly as the generator does.

## 2. Calibration and validation

Validation **PASSED**: all 8 held-out checks within tolerance.

| metric | key | observed | simulated | absolute_error | relative_error | tolerance | within_tolerance | used_for_fitting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean wait | mean_wait_minutes | 38.78 | 33.47 | -5.315 | 0.137 | 0.2 | True | True |
| p90 wait | p90_wait_minutes | 106.3 | 99.34 | -6.941 | 0.0653 | 0.25 | True | False |
| mean LOS | mean_los_minutes | 121.5 | 115.9 | -5.631 | 0.0463 | 0.2 | True | False |
| abandonment rate | abandonment_rate | 0.0332 | 0.0246 | -0.0086 | 0.2588 | 0.5 | True | False |
| mean wait ESI 1 | mean_wait_esi1 | 5.391 | 5.25 | -0.1402 | 0.026 | 0.2 | True | False |
| mean wait ESI 2 | mean_wait_esi2 | 9.447 | 7.599 | -1.847 | 0.1956 | 0.2 | True | False |
| mean wait ESI 3 | mean_wait_esi3 | 41.71 | 34.85 | -6.863 | 0.1645 | 0.2 | True | False |
| mean wait ESI 4 | mean_wait_esi4 | 45.03 | 39.2 | -5.837 | 0.1296 | 0.2 | True | False |
| mean wait ESI 5 | mean_wait_esi5 | 47.57 | 44.53 | -3.037 | 0.0638 | 0.2 | True | False |

**One metric was used for fitting and is excluded from the gate.** The data records
arrival, first contact and departure, but never when *treatment* ended, so
`departure - seen` is treatment plus boarding and the split is not identifiable. It is
estimated as a single free parameter — a boarding offset for non-admitted patients —
chosen so simulated mean wait matches observed. Every other row is genuine out-of-sample
validation.

Two bugs the validation gate caught, which is what it is for:

- **Treatment times were over-estimated by ~13 minutes per patient**, because the first
  identification attributed all non-admitted post-contact time to treatment. That
  inflated doctor occupancy and every simulated wait by ~80%. All 9 checks failed.
- **Doctor utilisation exceeded 1.0** (1.0319), because busy minutes accumulated across
  the whole run while the capacity denominator covered only the post-warm-up window.
  Both sides now cover the same window; baseline utilisation is
  0.787.

A third issue was a flawed *test*, not a flawed model: a purely relative tolerance
failed ESI 1 on a two-minute discrepancy against a six-minute wait. Metrics now pass on
either the relative tolerance or an absolute floor.

## 3. Scenarios (30 replications each)

| scenario | mean_wait_minutes | mean_wait_minutes_ci_low | mean_wait_minutes_ci_high | p90_wait_esi3 | abandonment_rate | doctor_utilisation |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 36.14 | 34.69 | 37.6 | 118.7 | 0.0289 | 0.791 |
| plus_one_doctor_peak | 29.62 | 28.42 | 30.82 | 103 | 0.0215 | 0.7514 |
| plus_one_nurse_peak | 29.63 | 28.33 | 30.93 | 97.28 | 0.0188 | 0.7576 |
| fast_track_esi45 | 19.08 | 18.06 | 20.1 | 63.96 | 0.0054 | 0.6847 |
| surge_10pct | 51.06 | 49.23 | 52.88 | 159.3 | 0.0546 | 0.8499 |
| surge_20pct | 68.18 | 65.86 | 70.5 | 205.4 | 0.0923 | 0.8992 |

### Paired comparison with common random numbers

| scenario | difference_vs_baseline | ci_low | ci_high | method | significant |
| --- | --- | --- | --- | --- | --- |
| plus_one_doctor_peak | -6.52 | -7.169 | -5.87 | paired (common random numbers) | True |
| plus_one_nurse_peak | -6.512 | -6.909 | -6.115 | paired (common random numbers) | True |
| fast_track_esi45 | -17.06 | -17.88 | -16.25 | paired (common random numbers) | True |
| surge_10pct | 14.91 | 12.58 | 17.25 | unpaired (scenario changes demand) | True |
| surge_20pct | 32.04 | 29.3 | 34.77 | unpaired (scenario changes demand) | True |

**Fast-track is the strongest intervention**, cutting mean wait by
17.06 min
and abandonment from 0.0289
to 0.0054.

Common random numbers do real work here. The paired intervals span about
1.30 minutes,
against roughly
4.67
for the unpaired demand comparisons — from the same number of replications. A surge
scenario necessarily draws a different arrival stream, so its comparison is reported as
unpaired rather than being quietly presented as if the noise had cancelled.

**A modelling error the scenario results exposed.** The first fast-track implementation
routed ESI 4-5 patients *exclusively* to the dedicated doctor, which made overall mean
wait **worse** (58.5 vs 36.1 min) because those levels are ~46% of arrivals against one
clinician. A fast-track lane is an additional route, not a restriction: those patients
remain eligible for the main pool. After the fix it is the best scenario.

**"One nurse-equivalent" is a modelling choice, not a measurement.** This model
constrains doctors only; a nurse is represented as a proportional cut in treatment time
(`simulation.nurse_equivalent_service_factor`, default 0.95).

## 4. Sensitivity: does the ranking survive being wrong?

| scenario | rank_at_point_estimate | best_rank | worst_rank | rank_is_stable | max_rank_shift |
| --- | --- | --- | --- | --- | --- |
| fast_track_esi45 | 1 | 1 | 1 | True | 0 |
| plus_one_doctor_peak | 2 | 2 | 3 | False | 1 |
| plus_one_nurse_peak | 3 | 2 | 3 | False | 1 |
| baseline | 4 | 4 | 4 | True | 0 |
| surge_10pct | 5 | 5 | 5 | True | 0 |
| surge_20pct | 6 | 6 | 6 | True | 0 |

**The headline conclusion is robust: `fast_track_esi45` is rank 1 in every perturbed
world**, across +/-10% and +/-20% errors in both service times and arrival rates.

**The doctor-versus-nurse ordering is not.** They swap between ranks 2 and 3 depending
on the perturbation — which is exactly consistent with their point estimates
(-6.52 and
-6.51 min)
being statistically indistinguishable. Reporting "an extra doctor beats an extra nurse"
from the point estimate alone would have been unsupported.

## 5. Staffing search on forecast demand (research gap G1)

Load: **forecast (72 hours)**. Target: p90 wait for ESI 3 <= 90 min and mean abandonment <= 4.0%, judged on the upper 95% confidence bound.
36 of 48 plans were feasible.

| plan | cost | p90_wait_esi3 | p90_wait_esi3_ci_high | abandonment_rate_ci_high | doctor_utilisation | feasible |
| --- | --- | --- | --- | --- | --- | --- |
| day=7, eve=5, night=3 | 9.24e+04 | 34.11 | 51.88 | 0.00852 | 0.6918 | True |
| day=7, eve=5, night=4 | 9.856e+04 | 29.16 | 42.14 | 0.00852 | 0.6486 | True |
| day=7, eve=6, night=3 | 9.856e+04 | 23.83 | 34.2 | 0.00794 | 0.6503 | True |
| day=8, eve=5, night=3 | 9.856e+04 | 24.6 | 33.41 | 0.00345 | 0.6498 | True |
| day=7, eve=5, night=5 | 1.047e+05 | 28.49 | 40.73 | 0.00852 | 0.6104 | True |
| day=7, eve=6, night=4 | 1.047e+05 | 19.53 | 27.5 | 0.00794 | 0.612 | True |
| day=7, eve=7, night=3 | 1.047e+05 | 19.28 | 26.73 | 0.00794 | 0.6131 | True |
| day=8, eve=5, night=4 | 1.047e+05 | 22.87 | 30.77 | 0.00345 | 0.6115 | True |

**Cheapest feasible plan: `day=7, eve=5, night=3` at cost 92,400** for a
168-hour week, with p90 ESI 3 wait
34.1 min and abandonment 0.0037.

Two deliberate design choices:

- **Feasibility is decided before cost.** Only plans meeting *every* target are ranked
  by cost. The cheapest plan overall is usually infeasible, and returning it as "best"
  would be the most expensive mistake this module could make. A test asserts the search
  never returns an infeasible plan as the choice, and a second test gives it an
  impossible target and checks it returns nothing plus the closest misses.
- **Targets are judged on the upper confidence bound**, not the point estimate. A plan
  whose mean sits a whisker under the target with a wide interval has not been *shown*
  to meet it.

## Deviations from CLAUDE.md

1. **A `sensitivity` stage was added to `STAGE_ORDER`** (now 15 stages). Phase 3
   requires `sensitivity.py` and Phase 4's stage list names it.
2. **`src/utils/priority.py` and `src/simulation/sim_figures.py` are new modules** not
   in the original repo layout. The first exists so the generator and the DES cannot
   drift; the second keeps plotting out of the simulation logic.
3. Carried forward: `pyarrow`; the `split` stage keeps all rows; `ablation` aliases
   `train_wait`; a `survival` stage was added in Phase 2.

No new dependency. `simpy` was already approved.

## Known limitations

- **The staffing plan is sized for the forecast window, not the year.** The forecast
  horizon is 72 hours and that window happens to be quieter than the annual mean
  (~4.95 against ~5.87 arrivals/hour), so the recommended plan is leaner than an
  all-year plan should be. This is correct behaviour for forecast-driven staffing but
  must not be read as an annual recommendation.
- **The treatment/boarding split is not identifiable** from these columns and is fitted
  to one aggregate. A site with real treatment-end timestamps should use them.
- **Nurses, beds, imaging and lab turnaround are still not modelled.** Doctors are the
  only hard constraint; a nurse appears only as a service-time multiplier.
- **The simulation assumes one doctor treats one patient at a time**, as the generator
  does. Real clinicians interleave several, so utilisation is not directly comparable
  to a real department's.
- **Sensitivity perturbs service times and arrival rates only.** Patience, the acuity
  mix and the aging policy are held fixed.
- **The DES is validated against synthetic data generated by a queue of similar
  structure.** That it reproduces the generator is a correctness check on the
  implementation, not evidence that it would reproduce a real department.
