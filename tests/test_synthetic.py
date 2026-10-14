"""Tests for the synthetic generator.

The generator is the project's stand-in for reality, so these tests check more than
"it runs": they check that the waits it produces actually behave like a queue.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import pytest

from src.data.schema import validate_visits
from src.data.synthetic import _capacity_by_hour, _run_queue, generate_synthetic_visits


def test_output_matches_the_contract_input_columns(tiny_visits, schema) -> None:
    """The generator emits the schema's input columns and no derived ones."""
    assert list(tiny_visits.columns) == schema.input_columns
    assert "wait_minutes" not in tiny_visits.columns
    assert "los_minutes" not in tiny_visits.columns

    assert tiny_visits["triage_level"].between(1, 5).all()
    assert set(tiny_visits["disposition"].unique()) <= set(schema.spec("disposition").allowed)
    assert set(tiny_visits["arrival_mode"].unique()) <= set(schema.spec("arrival_mode").allowed)
    assert tiny_visits["patient_id"].is_unique
    assert tiny_visits["arrival_time"].is_monotonic_increasing


def test_requested_size_is_honoured(config) -> None:
    """n_patients is exact, not approximate."""
    visits = generate_synthetic_visits(config, n_patients=777)
    assert len(visits) == 777


def test_n_days_generates_whole_days(config) -> None:
    """Asking for a calendar span gives whole days, which the forecaster relies on."""
    visits = generate_synthetic_visits(config, n_days=30)
    span = visits["arrival_time"].max() - visits["arrival_time"].min()
    assert pd.Timedelta(days=28) < span < pd.Timedelta(days=30)


def test_generation_is_deterministic_under_a_fixed_seed(config) -> None:
    """Same seed, same data -- byte for byte."""
    first = generate_synthetic_visits(config)
    second = generate_synthetic_visits(config)
    pd.testing.assert_frame_equal(first, second)


def test_changing_the_seed_changes_the_data(config) -> None:
    """The seed actually reaches the generator."""
    first = generate_synthetic_visits(config)
    other = copy.deepcopy(config)
    other["seed"] = config["seed"] + 1
    second = generate_synthetic_visits(other)

    first_wait = (first["seen_time"] - first["arrival_time"]).mean()
    second_wait = (second["seen_time"] - second["arrival_time"]).mean()
    assert first_wait != second_wait


def test_arrival_rate_varies_by_hour_and_weekday(config) -> None:
    """Arrivals follow a non-homogeneous process, not a flat rate."""
    visits = generate_synthetic_visits(config, n_days=120)
    arrivals = visits.set_index("arrival_time").resample("h").size()

    by_hour = arrivals.groupby(arrivals.index.hour).mean()
    # The configured profile peaks late morning and bottoms out before dawn.
    assert by_hour.loc[9:12].mean() > 2.0 * by_hour.loc[2:5].mean()

    by_dow = arrivals.groupby(arrivals.index.dayofweek).mean()
    assert by_dow.std() > 0
    assert by_dow.idxmax() in (0, 1)  # Monday/Tuesday peak per dow_profile


def test_higher_acuity_patients_are_seen_sooner(tiny_visits) -> None:
    """The priority queue must be visible in the data: ESI 1 waits least, ESI 5 most."""
    result = validate_visits(tiny_visits)
    mean_wait = result.modelling.groupby("triage_level")["wait_minutes"].mean()

    assert mean_wait.loc[1] < mean_wait.loc[3] < mean_wait.loc[5]
    assert mean_wait.loc[2] < mean_wait.loc[4]


def test_waits_emerge_from_congestion_not_noise(config) -> None:
    """A patient's wait must depend on how busy the department was when they arrived.

    This is the property the queue-state features exist to exploit (gap G3). We count,
    for each patient, how many earlier patients had arrived but not yet been seen, and
    require a clear positive association with the realised wait.
    """
    visits = generate_synthetic_visits(config, n_patients=6000)
    result = validate_visits(visits)
    everyone = result.all.sort_values("arrival_time").reset_index(drop=True)

    arrivals = everyone["arrival_time"].to_numpy()
    # A patient leaves the waiting room either by being seen or by giving up, so
    # "still waiting at time t" is (arrived before t) minus (resolved by t).
    resolved = everyone["seen_time"].fillna(everyone["departure_time"])
    resolved_sorted = np.sort(resolved.to_numpy())
    arrived_before = np.arange(len(everyone))
    resolved_before = np.searchsorted(resolved_sorted, arrivals, side="right")
    n_waiting = arrived_before - resolved_before

    seen_mask = everyone["wait_minutes"].notna().to_numpy()
    waits = everyone.loc[seen_mask, "wait_minutes"].to_numpy()
    congestion = n_waiting[seen_mask]

    correlation = float(np.corrcoef(congestion, waits)[0, 1])
    assert correlation > 0.4, f"wait is not congestion-driven (r={correlation:.2f})"

    # And the effect should be large, not merely detectable.
    busy = waits[congestion >= np.quantile(congestion, 0.75)].mean()
    quiet = waits[congestion <= np.quantile(congestion, 0.25)].mean()
    assert busy > 2.0 * quiet, f"busy mean {busy:.1f} vs quiet mean {quiet:.1f}"


def test_lwbs_patients_waited_longer_than_average(config) -> None:
    """LWBS must be driven by long waits, not drawn independently."""
    visits = generate_synthetic_visits(config, n_patients=6000)
    result = validate_visits(visits)

    lwbs_share = len(result.flagged) / len(result.all)
    assert 0.0 < lwbs_share < 0.20, f"implausible LWBS share {lwbs_share:.3f}"

    # Patients who left unseen stayed in the department longer than the median wait
    # of patients who were seen -- they left because they were still waiting.
    time_to_leave = (
        result.flagged["departure_time"] - result.flagged["arrival_time"]
    ).dt.total_seconds() / 60.0
    assert time_to_leave.median() > result.modelling["wait_minutes"].median()


def test_service_time_increases_with_acuity(tiny_visits) -> None:
    """Treatment duration is lognormal per ESI level, longest for the sickest."""
    seen = tiny_visits[tiny_visits["seen_time"].notna()].copy()
    # Treatment length is not published directly; LOS after being seen is its proxy.
    seen["post_seen_minutes"] = (
        seen["departure_time"] - seen["seen_time"]
    ).dt.total_seconds() / 60.0
    by_level = seen.groupby("triage_level")["post_seen_minutes"].median()
    assert by_level.loc[1] > by_level.loc[3] > by_level.loc[5]


def test_arrival_mode_and_age_correlate_with_acuity(tiny_visits) -> None:
    """Optional demographics are plausible, not uniform noise."""
    ambulance_share = (
        tiny_visits.assign(amb=tiny_visits["arrival_mode"] == "ambulance")
        .groupby("triage_level")["amb"]
        .mean()
    )
    assert ambulance_share.loc[1] > 0.6
    assert ambulance_share.loc[5] < 0.1

    mean_age = tiny_visits.groupby("triage_level")["age"].mean()
    assert mean_age.loc[1] > mean_age.loc[5]
    assert tiny_visits["age"].between(0, 120).all()


def test_triage_mix_matches_the_configured_proportions(config) -> None:
    """Acuity proportions follow triage_mix within sampling error."""
    visits = generate_synthetic_visits(config, n_patients=20000)
    observed = visits["triage_level"].value_counts(normalize=True)
    for level, expected in config["synthetic"]["triage_mix"].items():
        assert observed.loc[int(level)] == pytest.approx(expected, abs=0.015)


def test_capacity_by_hour_expands_shifts() -> None:
    """Shift definitions expand to 24 hourly capacities, including wrap-around."""
    shifts = {
        "night": {"start": 22, "end": 6, "doctors": 2},
        "day": {"start": 6, "end": 22, "doctors": 7},
    }
    capacity = _capacity_by_hour(shifts, key="doctors")
    assert capacity.tolist() == [2] * 6 + [7] * 16 + [2] * 2


def test_uncovered_shift_hours_raise() -> None:
    """A shift table with a gap is a config bug and must not pass silently."""
    with pytest.raises(ValueError, match="leaves hours uncovered"):
        _capacity_by_hour({"day": {"start": 8, "end": 16, "doctors": 5}}, key="doctors")


@pytest.mark.slow
def test_fifty_thousand_patients_generate_in_under_a_minute(default_config) -> None:
    """Performance budget from the spec: ~50k patients in under a minute."""
    cfg = copy.deepcopy(default_config)
    start = time.perf_counter()
    visits = generate_synthetic_visits(cfg, n_patients=50000)
    elapsed = time.perf_counter() - start

    assert len(visits) == 50000
    span_days = (visits["arrival_time"].max() - visits["arrival_time"].min()).days
    assert span_days >= 180, f"default size must span at least six months, got {span_days} days"
    assert elapsed < 60.0, f"generation took {elapsed:.1f}s"


# ------------------------------------------------------------------- priority aging


def test_aging_keeps_low_acuity_abandonment_within_target(default_config) -> None:
    """Priority aging must stop the queue starving its lowest-acuity patients.

    Targets agreed for the default configuration: roughly 3-5% overall abandonment and
    under about 8% for ESI 5.
    """
    import copy

    visits = generate_synthetic_visits(copy.deepcopy(default_config))
    result = validate_visits(visits)
    everyone = result.all

    never_seen = everyone["seen_time"].isna()
    abandoned = never_seen & everyone["is_lwbs"]
    overall = float(abandoned.mean())
    assert 0.03 <= overall <= 0.05, f"overall abandonment {overall:.4f} outside 3-5%"

    by_level = everyone.assign(gave_up=abandoned).groupby("triage_level")["gave_up"].mean()
    assert by_level.loc[5] < 0.08, f"ESI 5 abandonment {by_level.loc[5]:.4f} above 8%"
    # Abandonment should still rise with decreasing acuity -- aging softens it, not inverts it.
    assert by_level.loc[1] <= by_level.loc[3] <= by_level.loc[5]


def test_enough_abandonment_remains_for_survival_analysis(default_config) -> None:
    """Aging must not flatten the signal the survival module exists to model."""
    import copy

    result = validate_visits(generate_synthetic_visits(copy.deepcopy(default_config)))
    events = int((result.all["seen_time"].isna() & result.all["is_lwbs"]).sum())
    assert events > 500, f"only {events} abandonment events; survival analysis would be thin"


def test_the_aging_floor_protects_the_most_urgent_patients(tiny_visits) -> None:
    """An aged low-acuity patient must never outrank a freshly arrived ESI 1 or 2.

    Without the floor, aging produces a queue in which a resuscitation case waits
    behind an aged non-urgent one. The floor is a safety constraint, so this test
    guards the ordering rather than a tuned number.
    """
    result = validate_visits(tiny_visits)
    mean_wait = result.modelling.groupby("triage_level")["wait_minutes"].mean()

    assert mean_wait.loc[1] < mean_wait.loc[3]
    assert mean_wait.loc[2] < mean_wait.loc[3]
    # ESI 1 is seen promptly in absolute terms, not merely sooner than the rest.
    assert mean_wait.loc[1] < 15.0, f"ESI 1 mean wait {mean_wait.loc[1]:.1f} min is too long"


def test_stronger_aging_reduces_low_acuity_abandonment(config) -> None:
    """The aging rate must actually control the behaviour it claims to control."""
    import copy

    def esi5_abandonment(minutes_per_level: float) -> float:
        cfg = copy.deepcopy(config)
        cfg["synthetic"]["priority_aging_minutes_per_level"] = minutes_per_level
        cfg["synthetic"]["n_patients"] = 8000
        result = validate_visits(generate_synthetic_visits(cfg))
        everyone = result.all
        gave_up = everyone["seen_time"].isna() & everyone["is_lwbs"]
        return float(gave_up[everyone["triage_level"] == 5].mean())

    strong = esi5_abandonment(15.0)
    weak = esi5_abandonment(240.0)
    assert strong < weak, f"stronger aging did not help ESI 5 ({strong:.4f} vs {weak:.4f})"


def test_non_positive_aging_rate_raises() -> None:
    """A zero or negative aging rate is a config error, not a silent divide.

    The check lives in the shared AgingPolicy, which the generator and the SimPy model
    both construct, so one guard protects both.
    """
    arrays = (np.array([0.0]), np.array([3]), np.array([10.0]), np.array([99.0]))
    with pytest.raises(ValueError, match="minutes_per_level must be positive"):
        _run_queue(*arrays, np.full(24, 2), 0.0, 3.0, 1000.0)


# ------------------------------------------------------------- window censoring


def test_patients_unresolved_at_the_cutoff_are_censored(default_config) -> None:
    """The extract ends at a cutoff; whoever is still waiting gets no invented outcome."""
    import copy

    result = validate_visits(generate_synthetic_visits(copy.deepcopy(default_config)))
    everyone = result.all
    censored = everyone["still_waiting_at_cutoff"]

    assert censored.any(), "a year-long extract should end mid-queue for someone"
    # Censoring is rare and confined to the end of the window.
    assert float(censored.mean()) < 0.01
    last_arrival = everyone["arrival_time"].max()
    assert (everyone.loc[censored, "arrival_time"] > last_arrival - pd.Timedelta(hours=24)).all()

    assert everyone.loc[censored, "seen_time"].isna().all()
    assert everyone.loc[censored, "departure_time"].isna().all()


def test_no_event_is_recorded_after_the_cutoff(tiny_visits) -> None:
    """Nothing in the extract may happen after the window closes."""
    window_end = tiny_visits["arrival_time"].max().ceil("h")
    for column in ("seen_time", "departure_time"):
        assert (tiny_visits[column].dropna() <= window_end).all(), f"{column} exceeds the cutoff"


def test_aging_compresses_the_low_acuity_tails(default_config) -> None:
    """Aging converges the ESI 3-5 wait tails, and that is intended, not a defect.

    Relieving ESI 5 starvation requires letting an aged patient overtake someone
    fresher. With the floor at 2.5 the patients who may be overtaken are ESI 3 and 4,
    so their tails absorb the relief and the three p90s converge -- ESI 3's can even
    exceed ESI 5's. Mean waits still order correctly by acuity. This test pins that
    trade-off so a future change to the floor cannot alter it unnoticed.
    """
    import copy

    result = validate_visits(generate_synthetic_visits(copy.deepcopy(default_config)))
    waits = result.modelling.groupby("triage_level")["wait_minutes"]
    mean, p90 = waits.mean(), waits.quantile(0.90)

    # Means remain correctly ordered across every level.
    assert mean.loc[1] < mean.loc[2] < mean.loc[3] < mean.loc[4] < mean.loc[5]

    # ESI 1 and 2 are fully protected by the floor and stay far below the rest.
    assert p90.loc[2] < 0.3 * p90.loc[3]

    # The ESI 3-5 tails sit within a narrow band of each other, by design.
    low_acuity_p90 = p90.loc[[3, 4, 5]]
    spread = (low_acuity_p90.max() - low_acuity_p90.min()) / low_acuity_p90.mean()
    assert spread < 0.20, f"ESI 3-5 p90 spread {spread:.2f} is wider than the aging policy implies"
