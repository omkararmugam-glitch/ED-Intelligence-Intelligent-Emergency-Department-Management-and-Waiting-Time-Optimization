# Phase 1a — deliberate leak check

Required by the global gates: introduce a deliberate leak, confirm the leakage tests
fail, revert, confirm they pass.

## The leak

Into `src/features/queue_state.py::add_queue_state_features`, folding the arriving
patient's **own** realised wait into the "recent waits" average — the classic target
leak, where the feature contains the answer:

```python
mean_wait, available = index.mean_wait_of_last_seen(instants, recent_k)
# ===== DELIBERATE LEAK (temporary gate check) =====
if "wait_minutes" in out.columns:
    own = pd.to_numeric(out["wait_minutes"], errors="coerce").to_numpy(dtype="float64")
    mean_wait = np.where(np.isfinite(own),
                         (np.nan_to_num(mean_wait) * available + own) / (available + 1),
                         mean_wait)
# ===== END DELIBERATE LEAK =====
out[f"mean_wait_last_{recent_k}_seen"] = mean_wait
```

## Result with the leak in place

```
$ python -m pytest tests/test_leakage.py -q
FAILED tests/test_leakage.py::test_features_ignore_the_derived_outcome_columns
1 failed, 11 passed
```

Failure detail: `DataFrame.iloc[:, 6] (column name="mean_wait_last_10_seen") values are
different (100.0 %)`.

## Result after reverting

```
$ python -m pytest tests/test_leakage.py -q
12 passed in 0.42s
```

## What this check actually exposed

**On the first attempt, all 11 leakage tests passed with the leak in place.** That was
a real gap in the suite, not a fluke.

Every test written in Phase 1a attacked *future-patient* leaks — truncate the future,
mutate later rows, delete later rows. An own-wait leak is a **self-leak**: it reads the
arriving patient's own outcome. Truncating or mutating *later* patients cannot reveal
it, because an earlier patient's own wait is identical in both runs. The suite was
therefore blind to an entire class of leak.

The gap was closed by adding `test_features_ignore_the_derived_outcome_columns`, which
overwrites every contract-derived outcome column (`wait_minutes`, `los_minutes`,
`is_lwbs`, `still_waiting_at_cutoff`) with nonsense for **all** rows and requires the
features to come out byte-identical. `seen_time` and `departure_time` are deliberately
left untouched, because the queue features legitimately compare those against the
arrival instant — scrambling them would make the test vacuous rather than strict.

The suite now covers both classes:

| Leak class | Test |
| --- | --- |
| reads a later patient | `test_truncating_the_future_does_not_change_past_features` |
| reads a later patient's outcome | `test_mutating_future_rows_does_not_change_past_features` |
| depends on later rows existing | `test_deleting_future_rows_does_not_change_past_features` |
| reads this patient's own outcome | `test_features_ignore_the_derived_outcome_columns` |
| counts the patient in their own queue | `test_a_patient_does_not_count_themselves_as_waiting` |
| never retires abandoners from the queue | `test_abandoning_patients_leave_the_waiting_room` |
| uses a wait not yet observed at `t` | `test_recent_wait_feature_uses_only_already_observed_waits` |
