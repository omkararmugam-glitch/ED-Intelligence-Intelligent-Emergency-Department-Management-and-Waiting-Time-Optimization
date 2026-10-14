"""SimPy discrete-event model of the emergency department.

    arrival -> triage/registration -> aging priority queue -> doctor -> treatment -> departure

with patients abandoning the queue when their patience runs out.

**Why the dispatcher is hand-written.** SimPy's `PriorityResource` fixes a request's
priority at the moment it queues. This department ages priority continuously, so the
ordering changes while patients wait and a static key cannot express it. The model
therefore keeps its own waiting set and chooses the best candidate at each dispatch
opportunity, exactly as `src/data/synthetic.py` does — and both call the *same*
`AgingPolicy` from `src/utils/priority.py`, so the policy cannot drift between the
world and the model of it.

**Time-varying staffing.** Doctors on duty change by shift. Rather than mutating a
resource's capacity (which SimPy does not support cleanly), the dispatcher simply reads
the current capacity before each dispatch and a clock process wakes it at every shift
boundary. A shift that shrinks never preempts a patient already in treatment.

**Arrival load.** Either a configured hour-of-day profile, or an explicit hourly rate
series — which is how `forecast()` output drives the simulation, closing research gap
G1. All times inside the model are minutes from the start of the run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import simpy

from src.utils.logging import get_logger
from src.utils.metrics import quantile
from src.utils.priority import AgingPolicy
from src.utils.seed import rng as make_rng

logger = get_logger(__name__)

MINUTES_PER_HOUR = 60.0


@dataclass
class ServiceTimeModel:
    """Lognormal treatment durations and triage delays, per ESI level."""

    service_median: dict[int, float]
    service_sigma: dict[int, float]
    triage_delay_median: dict[int, float]
    triage_delay_sigma: dict[int, float]
    patience_median: dict[int, float]
    patience_sigma: float
    # Boarding is the exit-block delay after treatment ends. It occupies a bed, not a
    # doctor, so it lengthens LOS without lengthening anyone else's wait.
    boarding_median: dict[str, float] = field(default_factory=dict)
    boarding_sigma: dict[str, float] = field(default_factory=dict)
    admit_probability: dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ServiceTimeModel":
        """Build from the `synthetic` block, the department this model represents."""
        synthetic = config["synthetic"]
        service = {int(k): v for k, v in synthetic["service_time"].items()}
        delay = {int(k): v for k, v in synthetic["triage_delay_minutes"].items()}
        lwbs = synthetic["lwbs"]
        return cls(
            service_median={k: float(v["median"]) for k, v in service.items()},
            service_sigma={k: float(v["sigma"]) for k, v in service.items()},
            triage_delay_median={k: float(v["median"]) for k, v in delay.items()},
            triage_delay_sigma={k: float(v["sigma"]) for k, v in delay.items()},
            patience_median={
                int(k): float(v) for k, v in lwbs["patience_median_minutes"].items()
            },
            patience_sigma=float(lwbs["patience_sigma"]),
            boarding_median={
                k: float(v["median"]) for k, v in synthetic["boarding_minutes"].items()
            },
            boarding_sigma={
                k: float(v["sigma"]) for k, v in synthetic["boarding_minutes"].items()
            },
            admit_probability={
                int(k): float(v) for k, v in synthetic["admit_prob_by_triage"].items()
            },
        )

    def draw(
        self, rng: np.random.Generator, triage_level: int
    ) -> tuple[float, float, float, float]:
        """Draw (triage delay, service time, patience, boarding) in minutes."""
        level = int(triage_level)

        def lognormal(median: float, sigma: float) -> float:
            return float(np.exp(np.log(median) + sigma * rng.standard_normal()))

        boarding = 0.0
        if self.boarding_median:
            admitted = rng.random() < self.admit_probability.get(level, 0.0)
            key = "admitted" if admitted else "other"
            boarding = lognormal(self.boarding_median[key], self.boarding_sigma[key])

        return (
            lognormal(self.triage_delay_median[level], self.triage_delay_sigma[level]),
            lognormal(self.service_median[level], self.service_sigma[level]),
            lognormal(self.patience_median[level], self.patience_sigma),
            boarding,
        )

    def _replace(self, **changes: Any) -> "ServiceTimeModel":
        """Copy with some fields replaced, deep-copying the mappings.

        Every perturbation goes through here so that none of them can mutate the
        calibrated model other worlds are still using.
        """
        fields: dict[str, Any] = {
            "service_median": dict(self.service_median),
            "service_sigma": dict(self.service_sigma),
            "triage_delay_median": dict(self.triage_delay_median),
            "triage_delay_sigma": dict(self.triage_delay_sigma),
            "patience_median": dict(self.patience_median),
            "patience_sigma": self.patience_sigma,
            "boarding_median": dict(self.boarding_median),
            "boarding_sigma": dict(self.boarding_sigma),
            "admit_probability": dict(self.admit_probability),
        }
        fields.update(changes)
        return ServiceTimeModel(**fields)

    def scaled(self, service_factor: float = 1.0) -> "ServiceTimeModel":
        """Return a copy with every service-time median multiplied, for sensitivity runs.

        This lengthens treatment without shortening anything else, so length of stay
        moves with it. Use `with_clinician_share` instead to move time *between*
        clinician occupancy and the rest of the visit.
        """
        return self._replace(
            service_median={k: v * service_factor for k, v in self.service_median.items()}
        )

    def with_patience_scale(self, factor: float = 1.0) -> "ServiceTimeModel":
        """Return a copy with every acuity's patience median multiplied.

        Patience is the least observable input in the model: it is only seen for
        patients who ran out of it, and `calibrate.py` fits its scale against the
        observed abandonment rate. Perturbing it asks what a mis-fitted patience
        distribution would do to the recommendation.

        Args:
            factor: Multiplier on every level's patience median.

        Returns:
            A new model; `self` is unchanged.
        """
        return self._replace(
            patience_median={k: v * factor for k, v in self.patience_median.items()}
        )

    def with_boarding_scale(self, factor: float = 1.0) -> "ServiceTimeModel":
        """Return a copy with every boarding median multiplied.

        Boarding is the exit-block delay after treatment. On its own it lengthens
        length of stay and nothing else, because it holds a bed rather than a
        clinician -- so this changes anyone's *wait* only when beds are a modelled
        constraint and `boarding_holds_bed` is on. That pairing is what a boarding
        surge means, and it is why the surge scenario sets both.

        Args:
            factor: Multiplier on every boarding median.

        Returns:
            A new model; `self` is unchanged.
        """
        return self._replace(
            boarding_median={k: v * factor for k, v in self.boarding_median.items()}
        )

    def with_clinician_share(self, factor: float = 1.0) -> "ServiceTimeModel":
        """Return a copy where the clinician's share of the visit is scaled.

        `calibrate.py` splits each visit's post-contact time into clinician occupancy
        and everything else (tests, results, boarding) using one fitted fraction,
        because the data never records when treatment ended. That fraction is the
        single least identifiable number in the model, so the sensitivity grid has to
        move it.

        Moving it must not change length of stay, or the perturbation would be
        confounded with `scaled()`. Treatment is multiplied by `factor` and boarding is
        reduced by the *mean* change across acuity levels, which holds
        treatment + boarding roughly constant. It is a mean-preserving approximation,
        not an exact per-level identity, because boarding is keyed by disposition
        (admitted / other) rather than by acuity.

        Args:
            factor: Multiplier on the clinician's share, e.g. 1.2 for +20%.

        Returns:
            A new model; `self` is unchanged.

        Raises:
            ValueError: If `factor` is not positive.
        """
        if factor <= 0:
            raise ValueError(f"clinician-share factor must be positive, got {factor}")
        if factor == 1.0 or not self.service_median:
            return self._replace()

        treatment = {k: max(v * factor, 0.5) for k, v in self.service_median.items()}
        mean_delta = float(
            np.mean(
                [treatment[k] - self.service_median[k] for k in self.service_median]
            )
        )
        boarding = {
            key: max(value - mean_delta, 0.5) for key, value in self.boarding_median.items()
        }
        return self._replace(service_median=treatment, boarding_median=boarding)


@dataclass
class ArrivalModel:
    """Hourly arrival intensity and the acuity mix of those arrivals."""

    hourly_rate: np.ndarray  # arrivals per hour, indexed by hour of the run
    triage_probabilities: dict[int, float]

    @classmethod
    def from_config(
        cls, config: dict[str, Any], horizon_hours: int, multiplier: float = 1.0
    ) -> "ArrivalModel":
        """Build a repeating hour-of-day and day-of-week profile from config."""
        synthetic = config["synthetic"]
        hour_profile = np.asarray(synthetic["hour_profile"], dtype="float64")
        dow_profile = np.asarray(synthetic["dow_profile"], dtype="float64")
        base = float(synthetic["base_rate_per_hour"]) * multiplier

        hours = np.arange(horizon_hours)
        rate = base * hour_profile[hours % 24] * dow_profile[(hours // 24) % 7]
        mix = {int(k): float(v) for k, v in synthetic["triage_mix"].items()}
        total = sum(mix.values())
        return cls(hourly_rate=rate, triage_probabilities={k: v / total for k, v in mix.items()})

    @classmethod
    def from_forecast(
        cls,
        forecast_frame: pd.DataFrame,
        config: dict[str, Any],
        multiplier: float = 1.0,
    ) -> "ArrivalModel":
        """Drive the simulation directly from `forecast()` output.

        This is the link that closes research gap G1: predicted arrivals become the
        load the staffing search is evaluated against.

        Args:
            forecast_frame: Output of `src.models.forecast.forecast`, with `yhat`.
            config: Full pipeline config, for the acuity mix.
            multiplier: Scales every hour, for surge scenarios.

        Returns:
            An `ArrivalModel` whose rate series is the forecast.
        """
        if "yhat" not in forecast_frame.columns:
            raise KeyError("Forecast frame must carry a 'yhat' column")
        rate = forecast_frame["yhat"].to_numpy(dtype="float64") * multiplier

        mix = {int(k): float(v) for k, v in config["synthetic"]["triage_mix"].items()}
        total = sum(mix.values())
        logger.info(
            "Arrival model taken from a %d-hour forecast (mean %.2f arrivals/hour)",
            len(rate),
            float(rate.mean()),
        )
        return cls(hourly_rate=rate, triage_probabilities={k: v / total for k, v in mix.items()})

    @property
    def horizon_hours(self) -> int:
        """Length of the rate series."""
        return int(self.hourly_rate.size)

    def with_acuity_shift(
        self,
        points: float,
        gain_levels: Sequence[int],
        lose_levels: Sequence[int],
    ) -> "ArrivalModel":
        """Return a copy with probability mass moved between acuity bands.

        The acuity mix is an input taken from the data, but a department's case mix
        drifts -- with the season, with what the local primary-care service is doing,
        with a nearby ED closing. This asks whether the recommendation survives such a
        drift.

        `points` is in **percentage points of the whole mix**, not a multiplier, so
        "5 points more ESI 2-3" is unambiguous. The mass is added to `gain_levels` and
        removed from `lose_levels`, both in proportion to their current shares, and
        every level outside both lists is left exactly as it was -- ESI 1 in
        particular, whose share is a property of the catchment rather than something
        that drifts with case mix.

        Args:
            points: Percentage points to move, e.g. 5.0.
            gain_levels: Levels that gain the mass.
            lose_levels: Levels that give it up.

        Returns:
            A new `ArrivalModel`; `self` is unchanged.

        Raises:
            ValueError: If the donors cannot supply `points` without going negative,
                or either list contributes no probability mass at all.
        """
        mix = {int(k): float(v) for k, v in self.triage_probabilities.items()}
        fraction = float(points) / 100.0

        gain = [int(level) for level in gain_levels if int(level) in mix]
        lose = [int(level) for level in lose_levels if int(level) in mix]
        gain_mass = sum(mix[level] for level in gain)
        lose_mass = sum(mix[level] for level in lose)

        if not gain or gain_mass <= 0:
            raise ValueError(f"acuity shift has no mass to add to: levels {gain_levels}")
        if not lose or lose_mass <= 0:
            raise ValueError(f"acuity shift has no mass to take from: levels {lose_levels}")
        if fraction >= lose_mass:
            raise ValueError(
                f"cannot move {points:g} points out of levels {lose} which hold only "
                f"{100 * lose_mass:.1f} points between them"
            )

        shifted = dict(mix)
        for level in gain:
            shifted[level] = mix[level] + fraction * mix[level] / gain_mass
        for level in lose:
            shifted[level] = mix[level] - fraction * mix[level] / lose_mass

        total = sum(shifted.values())
        return ArrivalModel(
            hourly_rate=self.hourly_rate.copy(),
            triage_probabilities={k: v / total for k, v in shifted.items()},
        )

    def sample_arrivals(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Draw arrival instants and acuity levels for the whole run.

        Counts per hour are Poisson with that hour's rate and instants are uniform
        within the hour, which is the exact realisation of a piecewise-constant
        non-homogeneous Poisson process.
        """
        counts = rng.poisson(self.hourly_rate)
        total = int(counts.sum())
        hour_index = np.repeat(np.arange(counts.size, dtype="float64"), counts)
        minutes = hour_index * MINUTES_PER_HOUR + rng.random(total) * MINUTES_PER_HOUR
        order = np.argsort(minutes, kind="mergesort")

        levels = np.array(sorted(self.triage_probabilities), dtype="int64")
        probabilities = np.array([self.triage_probabilities[int(l)] for l in levels])
        triage = rng.choice(levels, size=total, p=probabilities)
        return minutes[order], triage[order]


@dataclass
class StaffingPlan:
    """Doctors on duty per shift, plus any fast-track lane."""

    shifts: dict[str, dict[str, int]]  # name -> {start, end, doctors}
    fast_track_doctors: int = 0
    fast_track_levels: tuple[int, ...] = ()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "StaffingPlan":
        """Build the baseline plan from the department's roster.

        `staffing_roster.shifts` is the department's published roster and takes
        precedence: it is an input, not something the model fits. The older
        `simulation.shifts` + `simulation.resources.doctors` pair is still honoured so
        existing configs keep working.

        Raises:
            ValueError: If neither source is present.
        """
        roster = config.get("staffing_roster", {}).get("shifts")
        if roster:
            plan = cls(
                shifts={
                    name: {
                        "start": int(spec["start"]),
                        "end": int(spec["end"]),
                        "doctors": int(spec["doctors"]),
                    }
                    for name, spec in roster.items()
                }
            )
            # Fail now rather than mid-simulation if an hour has nobody on duty.
            plan.capacity_by_hour()
            logger.info("Doctor roster from config: %s", plan.describe())
            return plan

        simulation = config.get("simulation", {})
        shifts = simulation.get("shifts")
        doctors = simulation.get("resources", {}).get("doctors")
        if not shifts or not doctors:
            raise ValueError(
                "No doctor roster in the config. Add a `staffing_roster.shifts` block "
                "giving doctors on duty by hour of day."
            )
        return cls(
            shifts={
                name: {
                    "start": int(spec["start"]),
                    "end": int(spec["end"]),
                    "doctors": int(doctors[name]),
                }
                for name, spec in shifts.items()
            }
        )

    def with_doctors(self, overrides: dict[str, int]) -> "StaffingPlan":
        """Return a copy with some shifts' doctor counts replaced."""
        shifts = {
            name: {**spec, "doctors": int(overrides.get(name, spec["doctors"]))}
            for name, spec in self.shifts.items()
        }
        return StaffingPlan(
            shifts=shifts,
            fast_track_doctors=self.fast_track_doctors,
            fast_track_levels=self.fast_track_levels,
        )

    def with_fast_track(self, doctors: int, levels: Sequence[int]) -> "StaffingPlan":
        """Return a copy carving out a dedicated low-acuity lane."""
        return StaffingPlan(
            shifts={name: dict(spec) for name, spec in self.shifts.items()},
            fast_track_doctors=int(doctors),
            fast_track_levels=tuple(int(level) for level in levels),
        )

    def capacity_by_hour(self) -> np.ndarray:
        """Expand the shifts into 24 hourly main-queue doctor counts."""
        capacity = np.full(24, -1, dtype="int64")
        for name, spec in self.shifts.items():
            start, end = int(spec["start"]), int(spec["end"])
            hours = [h % 24 for h in range(start, end if end > start else end + 24)]
            for hour in hours:
                capacity[hour] = int(spec["doctors"])
        if (capacity < 0).any():
            uncovered = np.flatnonzero(capacity < 0).tolist()
            raise ValueError(f"Staffing plan leaves hours uncovered: {uncovered}")
        return capacity

    def total_doctor_hours(self, horizon_hours: int) -> float:
        """Doctor-hours consumed over a run, including any fast-track lane."""
        capacity = self.capacity_by_hour()
        hours = np.arange(horizon_hours)
        main = float(capacity[hours % 24].sum())
        return main + float(self.fast_track_doctors * horizon_hours)

    def describe(self) -> str:
        """Compact human-readable summary."""
        parts = [f"{name}={spec['doctors']}" for name, spec in sorted(self.shifts.items())]
        if self.fast_track_doctors:
            parts.append(f"fast_track={self.fast_track_doctors}@ESI{list(self.fast_track_levels)}")
        return ", ".join(parts)



#: Stamped on every artifact produced with any resource extension active.
#:
#: The data this project calibrates against records arrival, first clinician contact
#: and departure. It records no bed, no nurse and no imaging order, so **none of the
#: extensions below can be validated against it**. The calibration gate cannot see
#: them, the 49,280 result does not involve them, and a number produced with them on
#: is a what-if about a department we have assumed, not a measurement of one.
UNCALIBRATED_NOTICE = (
    "SCENARIO ANALYSIS, NOT CALIBRATED. This result was produced with one or more "
    "resource extensions active (beds, nurses, imaging turnaround, bed-holding "
    "boarding). The dataset contains no bed, nurse or imaging columns, so these "
    "parameters are assumed rather than fitted and the simulation-validation gate "
    "cannot test them. Treat the direction of an effect as a hypothesis and the "
    "magnitude as arbitrary."
)


@dataclass(frozen=True)
class ResourceExtensions:
    """Second-order resources the department has but the data does not record.

    **Every field defaults to off.** With `enabled` false, `EDSimulation` takes exactly
    the code paths it took before this class existed, draws exactly the same random
    numbers in the same order, and produces byte-identical output --
    `tests/test_resource_extensions.py` asserts that against a stored digest.

    Attributes:
        enabled: Master switch. Nothing below has any effect while this is false.
        n_beds: Exam rooms/bays. A patient needs one to be treated, so this is a
            second constraint on dispatch alongside doctors. None leaves beds
            unlimited.
        n_triage_nurses: Nurses doing triage and reassessment. None leaves nursing
            unlimited, which is what the base model assumes.
        triage_nurse_minutes: Nurse time per arriving patient before they join the
            queue, on top of the existing registration delay.
        reassessment_every_minutes: How often a still-waiting patient is reassessed.
            0 disables reassessment.
        reassessment_minutes: Nurse time each reassessment costs.
        imaging_levels: Acuity levels that may be sent for imaging or labs.
        imaging_probability: Chance such a patient is sent.
        imaging_turnaround_minutes: Turnaround once sent. It holds the patient's bed
            and extends their stay; it does **not** hold the doctor, who moves on.
        boarding_holds_bed: When true, an admitted patient's boarding time occupies a
            bed, which is the usual mechanism behind exit block.
    """

    enabled: bool = False
    n_beds: int | None = None
    n_triage_nurses: int | None = None
    triage_nurse_minutes: float = 0.0
    reassessment_every_minutes: float = 0.0
    reassessment_minutes: float = 0.0
    imaging_levels: tuple[int, ...] = (1, 2, 3)
    imaging_probability: float = 0.0
    imaging_turnaround_minutes: float = 0.0
    boarding_holds_bed: bool = False

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ResourceExtensions":
        """Read `simulation.resource_extensions`, defaulting to every feature off.

        A config without the block at all produces a fully disabled instance, which is
        what keeps every existing config working unchanged.
        """
        block = (config.get("simulation") or {}).get("resource_extensions") or {}
        if not block.get("enabled", False):
            return cls()
        imaging = block.get("imaging") or {}
        nurses = block.get("nurses") or {}
        beds = block.get("beds") or {}
        return cls(
            enabled=True,
            n_beds=_optional_int(beds.get("count")),
            boarding_holds_bed=bool(beds.get("boarding_holds_bed", False)),
            n_triage_nurses=_optional_int(nurses.get("count")),
            triage_nurse_minutes=float(nurses.get("triage_minutes", 0.0)),
            reassessment_every_minutes=float(nurses.get("reassessment_every_minutes", 0.0)),
            reassessment_minutes=float(nurses.get("reassessment_minutes", 0.0)),
            imaging_levels=tuple(int(level) for level in imaging.get("levels", (1, 2, 3))),
            imaging_probability=float(imaging.get("probability", 0.0)),
            imaging_turnaround_minutes=float(imaging.get("turnaround_minutes", 0.0)),
        )

    def replace(self, **changes: Any) -> "ResourceExtensions":
        """A copy with some fields changed, for scenario definitions."""
        return replace(self, **changes)

    @property
    def beds_are_limited(self) -> bool:
        """Whether bed capacity actually constrains anything."""
        return self.enabled and self.n_beds is not None and self.n_beds > 0

    @property
    def nurses_are_limited(self) -> bool:
        """Whether nurse capacity actually constrains anything."""
        return self.enabled and self.n_triage_nurses is not None and self.n_triage_nurses > 0

    @property
    def imaging_is_active(self) -> bool:
        """Whether any patient can be delayed by imaging."""
        return (
            self.enabled
            and self.imaging_probability > 0.0
            and self.imaging_turnaround_minutes > 0.0
        )

    @property
    def reassessment_is_active(self) -> bool:
        """Whether waiting patients consume nurse time while they wait."""
        return (
            self.nurses_are_limited
            and self.reassessment_every_minutes > 0.0
            and self.reassessment_minutes > 0.0
        )

    @property
    def any_active(self) -> bool:
        """True when at least one extension changes the model's behaviour.

        This, not `enabled`, is what decides whether an output carries
        `UNCALIBRATED_NOTICE`: a config that switches the block on but constrains
        nothing is still the calibrated model.
        """
        return bool(
            self.beds_are_limited
            or self.nurses_are_limited
            or self.imaging_is_active
            or (self.boarding_holds_bed and self.beds_are_limited)
        )

    def describe(self) -> str:
        """One line naming what is active, for logs and artifacts."""
        if not self.any_active:
            return "none (calibrated model)"
        parts: list[str] = []
        if self.beds_are_limited:
            parts.append(f"beds={self.n_beds}")
        if self.boarding_holds_bed:
            parts.append("boarding holds a bed")
        if self.nurses_are_limited:
            parts.append(f"triage nurses={self.n_triage_nurses}")
        if self.reassessment_is_active:
            parts.append(
                f"reassessment every {self.reassessment_every_minutes:g} min"
            )
        if self.imaging_is_active:
            parts.append(
                f"imaging p={self.imaging_probability:g} for ESI "
                f"{list(self.imaging_levels)} at +{self.imaging_turnaround_minutes:g} min"
            )
        return ", ".join(parts)


def _optional_int(value: Any) -> int | None:
    """Parse an optional integer, treating None and empty string as absent."""
    if value is None or value == "":
        return None
    return int(value)

@dataclass
class SimulationResult:
    """Per-patient records and the aggregate KPIs derived from them."""

    patients: pd.DataFrame
    queue_length: pd.DataFrame
    kpis: dict[str, float]
    warmup_hours: float
    horizon_hours: float
    seed_label: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable KPI view."""
        return {
            "seed_label": self.seed_label,
            "warmup_hours": self.warmup_hours,
            "horizon_hours": self.horizon_hours,
            **self.kpis,
        }


class _Patient:
    """One patient's state inside a run."""

    __slots__ = (
        "index", "triage", "arrival", "ready", "patience", "service", "boarding",
        "seen_at", "left_at", "departed_at", "dispatch_event", "fast_track",
        # Extension-only state. Always present, always inert while the resource
        # extensions are off, so the base model's behaviour cannot depend on it.
        "imaging_minutes", "queued_at", "bed_minutes",
    )

    def __init__(
        self, index: int, triage: int, arrival: float, ready: float,
        patience: float, service: float, fast_track: bool, boarding: float = 0.0,
    ) -> None:
        self.index = index
        self.triage = triage
        self.arrival = arrival
        self.ready = ready
        self.patience = patience
        self.service = service
        self.boarding = boarding
        self.fast_track = fast_track
        self.seen_at: float | None = None
        self.left_at: float | None = None
        self.departed_at: float | None = None
        self.dispatch_event: simpy.Event | None = None
        self.imaging_minutes: float = 0.0
        self.queued_at: float | None = None
        self.bed_minutes: float = 0.0


class EDSimulation:
    """A single replication of the department.

    Args:
        config: Full pipeline config.
        arrivals: The arrival load.
        service: Service-time, triage-delay and patience distributions.
        plan: Staffing.
        seed_label: Names this replication's random stream, so runs are reproducible
            and so common random numbers can be shared across scenarios.
    """

    def __init__(
        self,
        config: dict[str, Any],
        arrivals: ArrivalModel,
        service: ServiceTimeModel,
        plan: StaffingPlan,
        seed_label: str = "replication_0",
        extensions: "ResourceExtensions | None" = None,
    ) -> None:
        self.config = config
        self.arrivals = arrivals
        self.service = service
        self.plan = plan
        self.seed_label = seed_label
        self.policy = AgingPolicy.from_config(config)
        self.rng = make_rng(config["seed"], f"des::{seed_label}")

        # Resource extensions draw from their OWN stream. The base model's draws must
        # come off `self.rng` in exactly the order they always did, or turning a flag
        # on would silently change the unrelated parts of every replication -- and
        # turning it off again would not restore them.
        self.extensions = extensions or ResourceExtensions.from_config(config)
        self.extras_rng = (
            make_rng(config["seed"], f"des_extensions::{seed_label}")
            if self.extensions.any_active
            else None
        )
        #: Set to an integer to override the roster for the current hour; see
        #: `_capacity_now`. None means "use the roster", which is every path except
        #: the RL environment.
        self.dynamic_capacity: int | None = None
        self.beds_in_use = 0
        self.bed_busy_minutes = 0.0
        self.nurse_busy_minutes = 0.0
        self.nurses: simpy.Resource | None = None

        self.env = simpy.Environment()
        self.capacity = plan.capacity_by_hour()
        self.patients: list[_Patient] = []
        self.waiting: set[int] = set()
        self.busy_main = 0
        self.busy_fast = 0
        self.busy_minutes = 0.0
        self.queue_samples: list[tuple[float, int, int]] = []
        if self.extensions.nurses_are_limited:
            self.nurses = simpy.Resource(
                self.env, capacity=int(self.extensions.n_triage_nurses)
            )

    # ------------------------------------------------------------------ processes

    def _capacity_now(self) -> int:
        """Main-queue doctors on duty at the current simulated minute.

        `dynamic_capacity` is the hook the reinforcement-learning environment uses to
        change staffing hour by hour. It is None everywhere else, in which case this is
        exactly the rostered lookup it always was.
        """
        if self.dynamic_capacity is not None:
            return int(self.dynamic_capacity)
        return int(self.capacity[int(self.env.now // MINUTES_PER_HOUR) % 24])

    def _best_waiting(self, candidates: set[int], now: float) -> int:
        """The waiting patient with the best aged priority, arrival order breaking ties."""
        return min(
            candidates,
            key=lambda i: (
                self.policy.effective_priority(
                    self.patients[i].triage, now - self.patients[i].ready
                ),
                self.patients[i].ready,
            ),
        )

    def _dispatch(self) -> None:
        """Start treatment for as many waiting patients as capacity allows.

        Called at every event that can change the answer: an arrival becoming eligible,
        a treatment finishing, a shift boundary. Aging changes *who* is chosen, never
        *when* a dispatch happens, so no extra events are needed for it.

        A fast-track lane is an **additional** route, not a restriction. Its doctors may
        only take fast-track-eligible patients, but those patients remain eligible for
        the main pool too. Confining them to the lane would starve them whenever the
        lane is smaller than their share of arrivals -- which, for ESI 4-5 at roughly
        46% of volume against one dedicated doctor, it badly is.
        """
        now = self.env.now

        # Dedicated lane first: it can serve nobody else, so using it costs nothing.
        eligible = {i for i in self.waiting if self.patients[i].fast_track}
        while eligible and self.busy_fast < self.plan.fast_track_doctors and self._bed_available():
            index = self._best_waiting(eligible, now)
            eligible.discard(index)
            self.waiting.discard(index)
            self._start_treatment(index, fast_lane=True)

        # Then the main pool, which serves anyone still waiting.
        capacity = self._capacity_now()
        while self.waiting and self.busy_main < capacity and self._bed_available():
            index = self._best_waiting(self.waiting, now)
            self.waiting.discard(index)
            self._start_treatment(index, fast_lane=False)

    def _bed_available(self) -> bool:
        """Whether a bed is free for the next patient.

        Always true when beds are not a modelled constraint, which is what keeps the
        base model's dispatch loop exactly as it was.
        """
        if not self.extensions.beds_are_limited:
            return True
        return self.beds_in_use < int(self.extensions.n_beds)

    def _start_treatment(self, index: int, fast_lane: bool) -> None:
        """Mark a patient as seen and occupy a doctor for their service time."""
        patient = self.patients[index]
        patient.seen_at = self.env.now
        if self.extensions.beds_are_limited:
            self.beds_in_use += 1
        if fast_lane:
            self.busy_fast += 1
        else:
            self.busy_main += 1
        if patient.dispatch_event is not None and not patient.dispatch_event.triggered:
            patient.dispatch_event.succeed()
        self.env.process(self._treat(index, fast_lane))

    def _treat(self, index: int, fast_lane: bool):
        """Occupy a doctor for the treatment, then release and board.

        The doctor is freed the moment treatment ends. Boarding that follows holds a
        bed, not a clinician, so it must not delay the next patient's dispatch.
        """
        patient = self.patients[index]
        bed_started = self.env.now
        yield self.env.timeout(patient.service)
        self.busy_minutes += patient.service
        if fast_lane:
            self.busy_fast -= 1
        else:
            self.busy_main -= 1

        # Imaging and labs hold the patient (and their bed), not the clinician, who
        # has already moved on. Dispatch therefore happens before the turnaround.
        if patient.imaging_minutes > 0:
            self._dispatch()
            yield self.env.timeout(patient.imaging_minutes)
        else:
            self._dispatch()

        # The bed is released either before boarding or after it, and that choice is
        # the whole mechanism behind exit block -- so it has to change when the bed is
        # actually free, not merely what gets recorded.
        holds_through_boarding = self.extensions.boarding_holds_bed
        if self.extensions.beds_are_limited and not holds_through_boarding:
            self._release_bed(patient, bed_started)

        if patient.boarding > 0:
            yield self.env.timeout(patient.boarding)
        patient.departed_at = self.env.now

        if self.extensions.beds_are_limited and holds_through_boarding:
            self._release_bed(patient, bed_started)

    def _release_bed(self, patient: "_Patient", occupied_from: float) -> None:
        """Free this patient's bed now, recording how long they held it."""
        patient.bed_minutes = max(self.env.now - occupied_from, 0.0)
        self.bed_busy_minutes += patient.bed_minutes
        self.beds_in_use -= 1
        self._dispatch()

    def _patient_process(self, index: int):
        """Triage delay, then wait for either a doctor or the end of patience."""
        patient = self.patients[index]
        yield self.env.timeout(patient.ready - self.env.now)

        if self.nurses is not None and self.extensions.triage_nurse_minutes > 0:
            # Triage itself needs a nurse. Queueing for one delays entry to the
            # doctor queue, and patience keeps running meanwhile -- which is the
            # point: a nursing bottleneck shows up as abandonment, not just as wait.
            with self.nurses.request() as request:
                yield request
                yield self.env.timeout(self.extensions.triage_nurse_minutes)
                self.nurse_busy_minutes += self.extensions.triage_nurse_minutes
            # Aging and tie-breaking run from the moment the patient actually joins.
            patient.ready = self.env.now

        patient.queued_at = self.env.now
        patient.dispatch_event = self.env.event()
        self.waiting.add(index)
        self._dispatch()

        if self.extensions.reassessment_is_active and patient.seen_at is None:
            self.env.process(self._reassess(index))

        if patient.seen_at is not None:
            return  # dispatched immediately

        # Patience is measured from arrival, so what remains is net of the triage delay.
        remaining = max(patient.arrival + patient.patience - self.env.now, 0.0)
        outcome = yield patient.dispatch_event | self.env.timeout(remaining)
        if patient.dispatch_event not in outcome:
            self.waiting.discard(index)
            patient.left_at = self.env.now
            patient.departed_at = self.env.now

    def _reassess(self, index: int):
        """Consume nurse time at intervals while a patient is still waiting.

        This is the coupling that makes a nurse constraint bite: reassessing the
        people already queued competes with triaging the people arriving, so a long
        queue slows the front door.
        """
        patient = self.patients[index]
        while patient.seen_at is None and patient.left_at is None:
            yield self.env.timeout(self.extensions.reassessment_every_minutes)
            if patient.seen_at is not None or patient.left_at is not None:
                return
            assert self.nurses is not None
            with self.nurses.request() as request:
                yield request
                if patient.seen_at is not None or patient.left_at is not None:
                    return
                yield self.env.timeout(self.extensions.reassessment_minutes)
                self.nurse_busy_minutes += self.extensions.reassessment_minutes

    def _arrival_process(self):
        """Create every patient at its drawn arrival instant."""
        minutes, levels = self.arrivals.sample_arrivals(self.rng)
        previous = 0.0
        for index, (instant, level) in enumerate(zip(minutes, levels)):
            yield self.env.timeout(max(instant - previous, 0.0))
            previous = instant

            delay, service, patience, boarding = self.service.draw(self.rng, int(level))
            fast_track = int(level) in self.plan.fast_track_levels and (
                self.plan.fast_track_doctors > 0
            )
            new_patient = _Patient(
                index=index, triage=int(level), arrival=instant,
                ready=instant + delay, patience=patience, service=service,
                boarding=boarding, fast_track=fast_track,
            )
            if self.extensions.imaging_is_active and int(level) in self.extensions.imaging_levels:
                # Drawn from the extensions stream, never from self.rng, so the base
                # model's sequence of draws is untouched whether this runs or not.
                assert self.extras_rng is not None
                if self.extras_rng.random() < self.extensions.imaging_probability:
                    new_patient.imaging_minutes = self.extensions.imaging_turnaround_minutes
            self.patients.append(new_patient)
            self.env.process(self._patient_process(index))

    def _shift_clock(self):
        """Wake the dispatcher at every clock hour, where capacity can rise."""
        while True:
            yield self.env.timeout(MINUTES_PER_HOUR)
            self._dispatch()

    def _monitor(self, interval_minutes: float = 30.0):
        """Sample queue length and doctors in use on a fixed grid."""
        while True:
            self.queue_samples.append(
                (
                    self.env.now,
                    len(self.waiting),
                    self.busy_main + self.busy_fast,
                )
            )
            yield self.env.timeout(interval_minutes)

    # ------------------------------------------------------------------- running

    def run(self) -> SimulationResult:
        """Execute one replication and return its records and KPIs."""
        horizon_minutes = self.arrivals.horizon_hours * MINUTES_PER_HOUR
        self.env.process(self._arrival_process())
        self.env.process(self._shift_clock())
        self.env.process(self._monitor())
        self.env.run(until=horizon_minutes)

        warmup_hours = float(self.config["simulation"]["warmup_hours"])
        return self._collect(warmup_hours, horizon_minutes)

    def _collect(self, warmup_hours: float, horizon_minutes: float) -> SimulationResult:
        """Turn raw patient state into a tidy frame and aggregate KPIs."""
        records = []
        for patient in self.patients:
            wait = (
                patient.seen_at - patient.arrival if patient.seen_at is not None else np.nan
            )
            los = (
                patient.departed_at - patient.arrival
                if patient.departed_at is not None and patient.seen_at is not None
                else np.nan
            )
            records.append(
                {
                    "patient_id": patient.index,
                    "triage_level": patient.triage,
                    "arrival_minutes": patient.arrival,
                    "seen_minutes": patient.seen_at,
                    "service_minutes": patient.service if patient.seen_at is not None else np.nan,
                    "wait_minutes": wait,
                    "los_minutes": los,
                    "seen": patient.seen_at is not None,
                    "abandoned": patient.left_at is not None,
                    "still_waiting": patient.seen_at is None and patient.left_at is None,
                    "fast_track": patient.fast_track,
                }
            )

        patients = pd.DataFrame(records)
        queue = pd.DataFrame(
            self.queue_samples, columns=["minutes", "queue_length", "doctors_busy"]
        )

        # Warm-up removal: the department starts empty, which no real one ever is.
        warmup_minutes = warmup_hours * MINUTES_PER_HOUR
        scored = patients[patients["arrival_minutes"] >= warmup_minutes]
        queue_scored = queue[queue["minutes"] >= warmup_minutes]

        kpis = self._kpis(scored, queue_scored, warmup_minutes, horizon_minutes)
        logger.debug(
            "Replication %s: %d patients scored, mean wait %.1f min, abandonment %.3f",
            self.seed_label,
            len(scored),
            kpis["mean_wait_minutes"],
            kpis["abandonment_rate"],
        )
        return SimulationResult(
            patients=patients,
            queue_length=queue,
            kpis=kpis,
            warmup_hours=warmup_hours,
            horizon_hours=self.arrivals.horizon_hours,
            seed_label=self.seed_label,
        )

    def _available_doctor_minutes(self, start_minutes: float, end_minutes: float) -> float:
        """Doctor-minutes on duty between two instants, including any fast-track lane.

        Summed hour by hour because the main-pool capacity changes with the shift.
        """
        total = 0.0
        hour = int(start_minutes // MINUTES_PER_HOUR)
        last_hour = int(math.ceil(end_minutes / MINUTES_PER_HOUR))
        while hour < last_hour:
            hour_start = hour * MINUTES_PER_HOUR
            overlap = min(hour_start + MINUTES_PER_HOUR, end_minutes) - max(
                hour_start, start_minutes
            )
            if overlap > 0:
                staffed = int(self.capacity[hour % 24]) + int(self.plan.fast_track_doctors)
                total += overlap * staffed
            hour += 1
        return total

    def _kpis(
        self,
        scored: pd.DataFrame,
        queue: pd.DataFrame,
        warmup_minutes: float,
        horizon_minutes: float,
    ) -> dict[str, float]:
        """Aggregate KPIs over the post-warm-up window."""
        waits = scored.loc[scored["seen"], "wait_minutes"]
        los = scored.loc[scored["seen"], "los_minutes"]

        # Both sides of the ratio must cover the same window. Treatment started inside
        # the scored window counts against capacity on duty inside it; warm-up work is
        # excluded from both.
        busy_minutes = float(
            scored.loc[scored["seen"], "service_minutes"].fillna(0.0).sum()
        )
        doctor_minutes = self._available_doctor_minutes(warmup_minutes, horizon_minutes)

        kpis = {
            "n_arrivals": int(len(scored)),
            "n_seen": int(scored["seen"].sum()),
            "n_abandoned": int(scored["abandoned"].sum()),
            "n_still_waiting": int(scored["still_waiting"].sum()),
            "mean_wait_minutes": float(waits.mean()) if len(waits) else float("nan"),
            "median_wait_minutes": float(waits.median()) if len(waits) else float("nan"),
            "p90_wait_minutes": quantile(waits, 0.90),
            "p95_wait_minutes": quantile(waits, 0.95),
            "mean_los_minutes": float(los.mean()) if len(los) else float("nan"),
            "p90_los_minutes": quantile(los, 0.90),
            "abandonment_rate": float(scored["abandoned"].mean()) if len(scored) else float("nan"),
            "mean_queue_length": float(queue["queue_length"].mean()) if len(queue) else float("nan"),
            "p90_queue_length": quantile(queue["queue_length"], 0.90) if len(queue) else float("nan"),
            "max_queue_length": int(queue["queue_length"].max()) if len(queue) else 0,
            "doctor_utilisation": float(busy_minutes / doctor_minutes)
            if doctor_minutes > 0
            else float("nan"),
        }

        for level in sorted(scored["triage_level"].unique()):
            subset = scored[scored["triage_level"] == level]
            level_waits = subset.loc[subset["seen"], "wait_minutes"]
            kpis[f"mean_wait_esi{int(level)}"] = (
                float(level_waits.mean()) if len(level_waits) else float("nan")
            )
            kpis[f"p90_wait_esi{int(level)}"] = quantile(level_waits, 0.90)
            kpis[f"abandonment_rate_esi{int(level)}"] = (
                float(subset["abandoned"].mean()) if len(subset) else float("nan")
            )

        # Extension KPIs are added ONLY when an extension is active, so a calibrated
        # run's KPI dict has exactly the keys it always had and nothing downstream
        # has to learn new ones.
        if self.extensions.any_active:
            window = max(horizon_minutes - warmup_minutes, 0.0)
            kpis["uncalibrated"] = True
            kpis["resource_extensions"] = self.extensions.describe()
            if self.extensions.beds_are_limited:
                bed_minutes = window * int(self.extensions.n_beds)
                kpis["bed_utilisation"] = (
                    float(self.bed_busy_minutes / bed_minutes)
                    if bed_minutes > 0 else float("nan")
                )
                kpis["n_beds"] = int(self.extensions.n_beds)
            if self.extensions.nurses_are_limited:
                nurse_minutes = window * int(self.extensions.n_triage_nurses)
                kpis["nurse_utilisation"] = (
                    float(self.nurse_busy_minutes / nurse_minutes)
                    if nurse_minutes > 0 else float("nan")
                )
                kpis["n_triage_nurses"] = int(self.extensions.n_triage_nurses)
            if self.extensions.imaging_is_active:
                imaged = [p for p in self.patients if p.imaging_minutes > 0]
                kpis["n_imaged"] = int(len(imaged))
        return kpis


def run_simulation(
    config: dict[str, Any],
    plan: StaffingPlan | None = None,
    arrivals: ArrivalModel | None = None,
    service: ServiceTimeModel | None = None,
    seed_label: str = "replication_0",
    horizon_hours: int | None = None,
    extensions: ResourceExtensions | None = None,
) -> SimulationResult:
    """Run one replication with sensible defaults drawn from the config.

    Args:
        config: Full pipeline config.
        plan: Staffing; defaults to the configured baseline.
        arrivals: Arrival load; defaults to the configured hour-of-day profile.
        service: Service-time model; defaults to the configured distributions.
        seed_label: Names the random stream for this replication.
        horizon_hours: Length of the run; defaults to `simulation.horizon_hours`.

    Returns:
        A `SimulationResult`.
    """
    horizon = int(horizon_hours or config["simulation"]["horizon_hours"])
    simulation = EDSimulation(
        config=config,
        arrivals=arrivals or ArrivalModel.from_config(config, horizon),
        service=service or ServiceTimeModel.from_config(config),
        plan=plan or StaffingPlan.from_config(config),
        seed_label=seed_label,
        extensions=extensions,
    )
    return simulation.run()


def run_replications(
    config: dict[str, Any],
    n_replications: int,
    plan: StaffingPlan | None = None,
    arrivals: ArrivalModel | None = None,
    service: ServiceTimeModel | None = None,
    seed_prefix: str = "baseline",
    horizon_hours: int | None = None,
    extensions: ResourceExtensions | None = None,
) -> tuple[pd.DataFrame, list[SimulationResult]]:
    """Run many replications and collect their KPIs.

    The seed label is `f"{seed_prefix}::{index}"`. Passing the **same** prefix to two
    scenarios gives them common random numbers: replication *i* of each sees the same
    arrival stream and service draws, so a paired comparison measures the policy
    difference rather than Monte Carlo noise.

    Args:
        config: Full pipeline config.
        n_replications: How many independent runs.
        plan: Staffing.
        arrivals: Arrival load.
        service: Service-time model.
        seed_prefix: Shared across scenarios to enable common random numbers.
        horizon_hours: Length of each run.

    Returns:
        Tuple of (one KPI row per replication, the raw results).
    """
    rows, results = [], []
    for index in range(int(n_replications)):
        result = run_simulation(
            config,
            plan=plan,
            arrivals=arrivals,
            service=service,
            seed_label=f"{seed_prefix}::{index}",
            horizon_hours=horizon_hours,
            extensions=extensions,
        )
        results.append(result)
        rows.append({"replication": index, **result.kpis})
    frame = pd.DataFrame(rows)
    logger.info(
        "Ran %d replications [%s]: mean wait %.2f min (sd %.2f), abandonment %.4f",
        n_replications,
        seed_prefix,
        frame["mean_wait_minutes"].mean(),
        frame["mean_wait_minutes"].std(ddof=1) if len(frame) > 1 else 0.0,
        frame["abandonment_rate"].mean(),
    )
    return frame, results
