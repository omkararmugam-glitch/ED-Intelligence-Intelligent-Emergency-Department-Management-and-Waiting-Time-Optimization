"""Synthetic ED visit generator.

The point of this module is that **waits are not drawn from a distribution** -- they
emerge from a queue. Patients arrive from a non-homogeneous Poisson process, join a
priority queue ordered by ESI level, and are served by a number of doctors that
changes with the shift. A patient whose wait exceeds their (random, acuity-dependent)
patience leaves without being seen. Consequently wait time genuinely depends on how
busy the department was at the moment of arrival, which is exactly the signal the
queue-state features are meant to capture.

This generator is deliberately *independent* of `src/simulation/des.py`. It plays the
role of the real world: the SimPy model is our model of it, and
`src/simulation/calibrate.py` checks whether that model reproduces these data.

Output conforms to `schema.yaml` input columns only; `wait_minutes` and `los_minutes`
are left for the loader to derive.
"""

from __future__ import annotations

import heapq
import math
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger
from src.utils.priority import AgingPolicy
from src.utils.seed import rng as make_rng

logger = get_logger(__name__)

#: Safety valve so a misconfigured rate cannot generate forever.
MAX_HORIZON_HOURS = 24 * 365 * 5

#: Presenting complaints by acuity band, used only to populate the optional column.
_COMPLAINTS: dict[str, list[str]] = {
    "high": ["chest pain", "shortness of breath", "altered mental status", "major trauma", "stroke symptoms"],
    "mid": ["abdominal pain", "fever", "vomiting", "back pain", "dizziness", "headache"],
    "low": ["sore throat", "minor laceration", "sprain", "rash", "dental pain", "ear pain"],
}


def _complaint_band(triage_level: int) -> str:
    """Map an ESI level to a complaint pool."""
    if triage_level <= 2:
        return "high"
    if triage_level == 3:
        return "mid"
    return "low"


def _lognormal(rng: np.random.Generator, median: float, sigma: float, size: int) -> np.ndarray:
    """Draw lognormal values with the requested median and log-scale sigma."""
    if median <= 0:
        raise ValueError(f"Lognormal median must be positive, got {median}")
    return np.exp(np.log(median) + sigma * rng.standard_normal(size))


def _capacity_by_hour(shifts: dict[str, dict[str, Any]], key: str = "doctors") -> np.ndarray:
    """Expand a shift definition into a 24-length array of server counts.

    Args:
        shifts: Mapping of shift name to {start, end, <key>}; `end` may wrap past 24.
        key: Which resource count to read from each shift.

    Returns:
        Array of length 24 giving the number of servers on duty in each clock hour.

    Raises:
        ValueError: If the shifts leave an hour of the day uncovered.
    """
    capacity = np.full(24, -1, dtype=np.int64)
    for name, spec in shifts.items():
        start, end = int(spec["start"]), int(spec["end"])
        count = int(spec[key])
        hours = [h % 24 for h in range(start, end if end > start else end + 24)]
        for hour in hours:
            capacity[hour] = count
        logger.debug("Shift %s covers hours %s with %d %s", name, hours, count, key)
    if (capacity < 0).any():
        uncovered = np.flatnonzero(capacity < 0).tolist()
        raise ValueError(f"Shift definition leaves hours uncovered: {uncovered}")
    return capacity


def _arrival_intensity(
    config: dict[str, Any],
    timestamps: pd.DatetimeIndex,
    surge_multiplier: np.ndarray,
) -> np.ndarray:
    """Compute the piecewise-constant hourly arrival rate.

    Rate = base * hour-of-day profile * day-of-week profile * annual seasonality * surge.
    """
    hour_profile = np.asarray(config["hour_profile"], dtype="float64")
    dow_profile = np.asarray(config["dow_profile"], dtype="float64")
    if hour_profile.size != 24:
        raise ValueError(f"hour_profile must have 24 entries, got {hour_profile.size}")
    if dow_profile.size != 7:
        raise ValueError(f"dow_profile must have 7 entries, got {dow_profile.size}")

    amplitude = float(config["seasonality_amplitude"])
    peak_doy = float(config["seasonality_peak_doy"])
    day_of_year = timestamps.dayofyear.to_numpy(dtype="float64")
    season = 1.0 + amplitude * np.cos(2.0 * np.pi * (day_of_year - peak_doy) / 365.25)

    return (
        float(config["base_rate_per_hour"])
        * hour_profile[timestamps.hour.to_numpy()]
        * dow_profile[timestamps.dayofweek.to_numpy()]
        * season
        * surge_multiplier
    )


def _generate_arrivals(
    config: dict[str, Any],
    rng: np.random.Generator,
    n_patients: int | None,
    n_days: int | None,
) -> tuple[np.ndarray, pd.Timestamp]:
    """Generate sorted arrival times from a non-homogeneous Poisson process.

    Hourly counts are Poisson with the hour's intensity, and arrival instants are
    uniform within the hour -- the standard exact realisation of a piecewise-constant
    NHPP.

    Args:
        config: The `synthetic` config block.
        rng: Seeded generator.
        n_patients: Target number of arrivals; generation stops once reached.
        n_days: If given, generate exactly this many whole days and ignore `n_patients`.

    Returns:
        Tuple of (arrival offsets in minutes from the start timestamp, start timestamp).
    """
    start = pd.Timestamp(config["start_date"])
    hour_profile = np.asarray(config["hour_profile"], dtype="float64")
    dow_profile = np.asarray(config["dow_profile"], dtype="float64")
    mean_rate = float(config["base_rate_per_hour"]) * hour_profile.mean() * dow_profile.mean()

    if n_days is not None:
        horizon_hours = int(n_days) * 24
    else:
        target = int(n_patients)
        # Generate ~15% more hours than the mean rate suggests, then truncate exactly.
        horizon_hours = int(math.ceil(target / max(mean_rate, 1e-6) * 1.15)) + 24
    horizon_hours = min(horizon_hours, MAX_HORIZON_HOURS)
    n_days_total = int(math.ceil(horizon_hours / 24))

    # Surges are decided per day, then expanded to the hours they cover.
    surge_cfg = config["surge"]
    surge = np.ones(n_days_total * 24, dtype="float64")
    surge_days = rng.random(n_days_total) < float(surge_cfg["prob_per_day"])
    duration = int(surge_cfg["duration_hours"])
    for day in np.flatnonzero(surge_days):
        start_hour = int(rng.integers(0, 24))
        begin = day * 24 + start_hour
        surge[begin : begin + duration] = float(surge_cfg["multiplier"])
    logger.debug("Generated %d surge day(s) over %d days", int(surge_days.sum()), n_days_total)

    timestamps = pd.date_range(start, periods=n_days_total * 24, freq="h")
    intensity = _arrival_intensity(config, timestamps, surge[: len(timestamps)])
    counts = rng.poisson(intensity)

    if n_days is None:
        cumulative = np.cumsum(counts)
        target = int(n_patients)
        if cumulative[-1] < target:
            raise RuntimeError(
                f"Generated only {cumulative[-1]} arrivals over {len(counts)} hours but "
                f"{target} were requested; increase synthetic.base_rate_per_hour"
            )
        # Keep whole hours up to the point where we first exceed the target.
        last_hour = int(np.searchsorted(cumulative, target, side="left"))
        counts = counts[: last_hour + 1]

    total = int(counts.sum())
    hour_index = np.repeat(np.arange(counts.size, dtype="float64"), counts)
    offsets = hour_index * 60.0 + rng.random(total) * 60.0
    offsets.sort(kind="mergesort")

    if n_days is None:
        offsets = offsets[: int(n_patients)]

    span_days = (offsets[-1] - offsets[0]) / (60.0 * 24.0) if offsets.size else 0.0
    logger.info(
        "Generated %d arrivals spanning %.1f days from %s",
        offsets.size,
        span_days,
        start.date(),
    )
    if span_days < 180:
        logger.warning(
            "Synthetic span is %.1f days (< 180). Increase synthetic.n_patients for a "
            "full six months of data; small spans are fine for tests only.",
            span_days,
        )
    return offsets, start


def _run_queue(
    ready_minutes: np.ndarray,
    triage: np.ndarray,
    service_minutes: np.ndarray,
    patience_minutes: np.ndarray,
    capacity_by_hour: np.ndarray,
    aging_minutes_per_level: float,
    aging_floor: float,
    window_end_minutes: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run an aging priority queue and return realised timings.

    Event-driven loop over queue entries, service completions, patience expiries and
    clock-hour boundaries (where the number of doctors on duty can change). Service in
    progress is never preempted when a shift shrinks; the department simply starts no
    new patients until it is back under capacity.

    **Priority aging.** A patient's effective priority is

        effective = max(triage_level - waited_minutes / aging_minutes_per_level, aging_floor)

    so a waiting patient gains one full ESI level of priority for every
    `aging_minutes_per_level` minutes waited, and the lowest-acuity patients are not
    starved indefinitely by a stream of new higher-acuity arrivals.

    `aging_floor` is a clinical safety constraint, not a tuning knob: aging can never
    promote a patient past it. With a floor of 3.0, a patient who has waited all day
    still never outranks a freshly arrived ESI 1 or ESI 2. Without it, aging produces a
    queue in which a resuscitation case waits behind an aged non-urgent one, which no
    real department would accept. Patients already more urgent than the floor do not
    age, because they are dispatched on acuity alone.

    Because aging is a monotone function of time it cannot be expressed as a static
    heap key, so the waiting room is held as a set and the best candidate is selected at
    each dispatch. Aging changes *who* is picked, never *when* a dispatch happens, so
    the event schedule is unaffected.

    **Window truncation.** The loop stops at `window_end_minutes`, the end of the data
    extract. Patients still in the waiting room at that moment are left unresolved:
    they neither were seen nor left, and the caller records them as censored by the
    window rather than inventing an outcome for them.

    Args:
        ready_minutes: Sorted offsets at which patients become eligible for a clinician
            (i.e. after triage and registration), in minutes.
        triage: ESI level per patient (1 = highest priority).
        service_minutes: Treatment duration per patient.
        patience_minutes: How long each patient will wait, measured from `ready_minutes`,
            before leaving unseen.
        capacity_by_hour: Doctors on duty for each of the 24 clock hours.
        aging_minutes_per_level: Minutes of waiting worth one ESI level of priority.
            Must be positive; a larger value means weaker aging.
        aging_floor: Best effective priority aging can reach. Patients at or above this
            acuity are never overtaken by an aged lower-acuity patient.
        window_end_minutes: End of the data window; nothing is simulated past it.

    Returns:
        Tuple of (seen_minutes, abandon_minutes). A seen patient has a finite
        start-of-treatment time, a patient who gave up has a finite abandonment time,
        and a patient still waiting when the window closed has neither.

    Raises:
        ValueError: If `aging_minutes_per_level` is not positive.
    """
    n = ready_minutes.size
    seen_minutes = np.full(n, np.nan)
    abandon_minutes = np.full(n, np.nan)

    in_service: list[float] = []  # heap of service completion times
    expiries: list[tuple[float, int]] = []  # heap of (patience deadline, index)
    still_waiting: set[int] = set()  # the waiting room

    # The aging rule lives in src/utils/priority.py so the SimPy model in
    # src/simulation/des.py applies exactly the same policy. If the two drifted, the
    # simulation would be calibrating against a department that does not exist.
    policy = AgingPolicy(minutes_per_level=aging_minutes_per_level, floor=aging_floor)

    def effective_priority(index: int, now: float) -> tuple[float, float]:
        """Aged priority, lower first, with arrival order as the tie-break."""
        return (
            policy.effective_priority(triage[index], now - ready_minutes[index]),
            ready_minutes[index],
        )

    next_arrival = 0
    now = float(ready_minutes[0])

    while now <= window_end_minutes:
        # 1. Free up doctors whose patients have finished.
        while in_service and in_service[0] <= now:
            heapq.heappop(in_service)

        # 2. Patients out of patience leave. Ties go to leaving, which keeps LWBS emergent.
        while expiries and expiries[0][0] <= now:
            deadline, index = heapq.heappop(expiries)
            if index in still_waiting:
                still_waiting.discard(index)
                abandon_minutes[index] = deadline

        # 3. New arrivals join the waiting room.
        while next_arrival < n and ready_minutes[next_arrival] <= now:
            index = next_arrival
            heapq.heappush(
                expiries, (float(ready_minutes[index] + patience_minutes[index]), index)
            )
            still_waiting.add(index)
            next_arrival += 1

        # 4. Start treatment for as many waiting patients as the current shift allows,
        #    always taking the best aged priority.
        capacity = int(capacity_by_hour[int(now // 60) % 24])
        while still_waiting and len(in_service) < capacity:
            index = min(still_waiting, key=lambda i: effective_priority(i, now))
            still_waiting.discard(index)
            seen_minutes[index] = now
            heapq.heappush(in_service, now + float(service_minutes[index]))

        # 5. Advance to the next event.
        next_time = math.inf
        if next_arrival < n:
            next_time = min(next_time, float(ready_minutes[next_arrival]))
        if in_service:
            next_time = min(next_time, in_service[0])
        if still_waiting:
            while expiries and expiries[0][1] not in still_waiting:
                heapq.heappop(expiries)
            if expiries:
                next_time = min(next_time, expiries[0][0])
            # Capacity can rise at the next clock hour, so that is an event too.
            next_time = min(next_time, (math.floor(now / 60.0) + 1.0) * 60.0)
        if not math.isfinite(next_time):
            break
        now = next_time

    if still_waiting:
        logger.info(
            "%d patient(s) were still in the waiting room when the data window closed; "
            "they are recorded as censored, not as having left",
            len(still_waiting),
        )
    return seen_minutes, abandon_minutes


def generate_synthetic_visits(
    config: dict[str, Any],
    n_patients: int | None = None,
    n_days: int | None = None,
) -> pd.DataFrame:
    """Generate a synthetic ED visits table conforming to the data contract.

    Args:
        config: The full pipeline config (uses the `synthetic` block and `seed`).
        n_patients: Override for `synthetic.n_patients`.
        n_days: Generate exactly this many days instead of a patient target. Useful
            when a downstream model needs a known calendar span.

    Returns:
        DataFrame with the schema's input columns, sorted by arrival time. Derived
        columns (`wait_minutes`, `los_minutes`) are deliberately not included; the
        loader computes them.
    """
    synth = config["synthetic"]
    rng = make_rng(config["seed"], "synthetic")

    target = int(n_patients if n_patients is not None else synth["n_patients"])
    arrival_minutes, start = _generate_arrivals(synth, rng, target, n_days)
    n = arrival_minutes.size

    # ---------------------------------------------------------------- patient attributes
    mix = {int(k): float(v) for k, v in synth["triage_mix"].items()}
    levels = np.array(sorted(mix), dtype="int16")
    probabilities = np.array([mix[int(level)] for level in levels], dtype="float64")
    probabilities = probabilities / probabilities.sum()
    triage = rng.choice(levels, size=n, p=probabilities).astype("int16")

    service_minutes = np.empty(n)
    patience_minutes = np.empty(n)
    triage_delay = np.empty(n)
    service_cfg = {int(k): v for k, v in synth["service_time"].items()}
    triage_delay_cfg = {int(k): v for k, v in synth["triage_delay_minutes"].items()}
    patience_cfg = synth["lwbs"]
    patience_medians = {int(k): float(v) for k, v in patience_cfg["patience_median_minutes"].items()}
    for level in levels:
        mask = triage == level
        size = int(mask.sum())
        if size == 0:
            continue
        spec = service_cfg[int(level)]
        service_minutes[mask] = _lognormal(rng, float(spec["median"]), float(spec["sigma"]), size)
        delay_spec = triage_delay_cfg[int(level)]
        triage_delay[mask] = _lognormal(
            rng, float(delay_spec["median"]), float(delay_spec["sigma"]), size
        )
        patience_minutes[mask] = _lognormal(
            rng, patience_medians[int(level)], float(patience_cfg["patience_sigma"]), size
        )

    # ---------------------------------------------------------------- queue dynamics
    # Triage and registration must finish before a clinician can pick the patient up,
    # so door-to-doctor time is this floor plus whatever queueing delay follows. The
    # floor is shorter for high-acuity patients, who are not held up by registration.
    ready_minutes = arrival_minutes + triage_delay

    # The queue engine needs its entries sorted by the time they join the queue, which
    # the random triage delay can reorder relative to arrival. Sort, run, then unsort.
    order = np.argsort(ready_minutes, kind="mergesort")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(n)
    # Patience runs from arrival, but the engine measures it from the queue entry.
    patience_from_ready = np.maximum(patience_minutes - triage_delay, 0.0)

    capacity_by_hour = _capacity_by_hour(synth["shifts"], key="doctors")
    logger.info("Doctor capacity by clock hour: %s", capacity_by_hour.tolist())

    # The extract ends at the close of the hour the last patient arrived in. Patients
    # unresolved at that moment are censored by the window, not given an outcome.
    window_end = (math.floor(float(arrival_minutes[-1]) / 60.0) + 1.0) * 60.0
    aging = float(synth["priority_aging_minutes_per_level"])
    aging_floor = float(synth["priority_aging_floor"])

    seen_sorted, abandon_sorted = _run_queue(
        ready_minutes[order],
        triage[order],
        service_minutes[order],
        patience_from_ready[order],
        capacity_by_hour,
        aging,
        aging_floor,
        window_end,
    )
    seen_minutes = seen_sorted[inverse]
    abandon_minutes = abandon_sorted[inverse]

    was_seen = np.isfinite(seen_minutes)
    abandoned = np.isfinite(abandon_minutes)
    still_waiting = ~was_seen & ~abandoned
    logger.info(
        "Queue simulated with priority aging (1 ESI level per %.0f min waited, "
        "never better than effective priority %.1f): %d seen, %d left without being "
        "seen (%.2f%%), %d still waiting at the cutoff",
        aging,
        aging_floor,
        int(was_seen.sum()),
        int(abandoned.sum()),
        100.0 * float(abandoned.mean()),
        int(still_waiting.sum()),
    )

    # ---------------------------------------------------------------- disposition & exit
    admit_prob = {int(k): float(v) for k, v in synth["admit_prob_by_triage"].items()}
    admit_draw = rng.random(n)
    admitted = was_seen & (admit_draw < np.array([admit_prob[int(t)] for t in triage]))
    transfer_draw = rng.random(n)
    transferred = was_seen & ~admitted & (transfer_draw < float(synth["transfer_prob"]))

    disposition = np.full(n, "discharged", dtype=object)
    disposition[admitted] = "admitted"
    disposition[transferred] = "transferred"
    disposition[abandoned] = "lwbs"
    # A patient still in the department when the extract ended has no outcome yet.
    disposition[still_waiting] = "other"

    boarding_cfg = synth["boarding_minutes"]
    boarding = np.zeros(n)
    admitted_spec, other_spec = boarding_cfg["admitted"], boarding_cfg["other"]
    if admitted.any():
        boarding[admitted] = _lognormal(
            rng, float(admitted_spec["median"]), float(admitted_spec["sigma"]), int(admitted.sum())
        )
    other_mask = was_seen & ~admitted
    if other_mask.any():
        boarding[other_mask] = _lognormal(
            rng, float(other_spec["median"]), float(other_spec["sigma"]), int(other_mask.sum())
        )

    departure_minutes = np.where(
        was_seen, seen_minutes + service_minutes + boarding, abandon_minutes
    )
    # Anything that would happen after the extract closed is simply not recorded, so a
    # patient still in treatment at the cutoff has a known wait but a censored LOS.
    departure_minutes = np.where(departure_minutes > window_end, np.nan, departure_minutes)
    logger.info(
        "Window closes at offset %.0f min; %d departure time(s) censored by it",
        window_end,
        int(np.isnan(departure_minutes).sum() - still_waiting.sum()),
    )

    # ---------------------------------------------------------------- demographics
    demo = synth["demographics"]
    age_means = {int(k): float(v) for k, v in demo["age_mean_by_triage"].items()}
    age = np.clip(
        rng.normal(
            loc=np.array([age_means[int(t)] for t in triage]), scale=float(demo["age_sd"]), size=n
        ),
        0,
        120,
    ).astype("int16")
    sex = np.where(rng.random(n) < float(demo["female_fraction"]), "female", "male")

    ambulance_prob = {int(k): float(v) for k, v in demo["ambulance_prob_by_triage"].items()}
    mode_draw = rng.random(n)
    p_ambulance = np.array([ambulance_prob[int(t)] for t in triage])
    arrival_mode = np.where(
        mode_draw < p_ambulance,
        "ambulance",
        np.where(mode_draw < p_ambulance + 0.97 * (1.0 - p_ambulance), "walk_in", "other"),
    )

    complaint = np.empty(n, dtype=object)
    for band, pool in _COMPLAINTS.items():
        mask = np.array([_complaint_band(int(t)) == band for t in triage])
        if mask.any():
            complaint[mask] = rng.choice(np.array(pool, dtype=object), size=int(mask.sum()))

    # ---------------------------------------------------------------- assemble
    def to_timestamp(minutes: np.ndarray) -> pd.Series:
        """Convert minute offsets (NaN-aware) to second-resolution timestamps."""
        stamps = start + pd.to_timedelta(minutes, unit="m")
        return pd.Series(stamps).dt.floor("s")

    visits = pd.DataFrame(
        {
            "patient_id": [f"P{i:07d}" for i in range(n)],
            "arrival_time": to_timestamp(arrival_minutes),
            "triage_level": triage.astype("int16"),
            "seen_time": to_timestamp(np.where(was_seen, seen_minutes, np.nan)),
            "departure_time": to_timestamp(departure_minutes),
            "disposition": disposition,
            "age": age,
            "sex": sex,
            "arrival_mode": arrival_mode,
            "chief_complaint": complaint,
        }
    )
    visits = visits.sort_values("arrival_time", kind="mergesort").reset_index(drop=True)

    realised_wait = (visits["seen_time"] - visits["arrival_time"]).dt.total_seconds() / 60.0
    logger.info(
        "Realised wait (seen patients): mean %.1f min, median %.1f min, p90 %.1f min",
        realised_wait.mean(),
        realised_wait.median(),
        realised_wait.quantile(0.90),
    )
    for level in levels:
        subset = realised_wait[visits["triage_level"] == level]
        if len(subset):
            logger.info(
                "  ESI %d: n=%5d mean %6.1f min p90 %6.1f min",
                level,
                len(subset),
                subset.mean(),
                subset.quantile(0.90),
            )
    return visits
