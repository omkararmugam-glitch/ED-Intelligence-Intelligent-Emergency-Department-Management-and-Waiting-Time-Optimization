"""The doctor roster the simulation needs, taken from the user and validated.

Doctors are the only hard capacity constraint in this model, so the roster is not a
detail: every wait, every scenario and every staffing recommendation is conditional
on it. It is an **input**, never something the simulation infers, and the config's
roster describes one specific synthetic dataset.

That makes the default dangerous in a way a wrong number usually is not — it would
produce a complete, confident, entirely fictional staffing recommendation for someone
else's department. So the roster is asked for explicitly, the default is labelled as
belonging to the demo data, and every staffing result carries the roster it assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Hours in a day. A roster must account for all of them.
HOURS_IN_DAY = 24

#: Said on every staffing result, so a number cannot be quoted without its assumption.
ROSTER_ATTRIBUTION = "Based on the roster you entered."

#: Said when the demo roster was used unchanged.
DEFAULT_ROSTER_WARNING = (
    "This used the demo roster from the config, which describes the synthetic "
    "ed_v2 dataset and not your department. Simulation and staffing results depend "
    "on it entirely, so enter your own roster before acting on them."
)


class RosterError(ValueError):
    """The roster is unusable, with a message written for a person."""


@dataclass(frozen=True)
class RosterShift:
    """Doctors on duty over a half-open hour band `[start, end)`."""

    name: str
    start: int
    end: int
    doctors: int

    @property
    def hours(self) -> range:
        """The clock hours this shift covers."""
        return range(self.start, self.end)


@dataclass(frozen=True)
class Roster:
    """A full-day doctor roster."""

    shifts: tuple[RosterShift, ...]
    is_default: bool

    def to_config(self) -> dict[str, Any]:
        """The `staffing_roster` block this roster represents."""
        return {
            "shifts": {
                shift.name: {
                    "start": shift.start,
                    "end": shift.end,
                    "doctors": shift.doctors,
                }
                for shift in self.shifts
            }
        }

    def describe(self) -> str:
        """One line naming the cover, for a report or a results page."""
        parts = [
            f"{shift.start:02d}:00-{shift.end:02d}:00 -> {shift.doctors}"
            for shift in self.shifts
        ]
        return "  ".join(parts)

    def doctors_at(self, hour: int) -> int:
        """Doctors on duty at a given clock hour."""
        for shift in self.shifts:
            if shift.start <= hour < shift.end:
                return shift.doctors
        raise RosterError(f"Hour {hour} is not covered by the roster.")

    @property
    def total_doctor_hours(self) -> int:
        """Doctor-hours per day, a quick sanity figure for the user."""
        return sum(shift.doctors * len(shift.hours) for shift in self.shifts)


def default_roster(config: dict[str, Any]) -> Roster:
    """The roster from the config, flagged as the demo default."""
    return parse_roster(config["staffing_roster"]["shifts"], is_default=True)


def parse_roster(shifts: dict[str, Any], is_default: bool = False) -> Roster:
    """Validate a roster given as `{name: {start, end, doctors}}`.

    Args:
        shifts: The shift bands.
        is_default: Whether this is the config's demo roster rather than the user's.

    Returns:
        The validated `Roster`.

    Raises:
        RosterError: If a band is malformed, a count is negative or not a whole
            number, the bands overlap, or any hour of the day is uncovered. Each of
            these would make the simulation quietly wrong rather than fail, so all
            are rejected up front.
    """
    if not shifts:
        raise RosterError("The roster is empty. Give at least one shift covering 00:00-24:00.")

    parsed: list[RosterShift] = []
    for name, band in shifts.items():
        try:
            start = int(band["start"])
            end = int(band["end"])
            doctors = band["doctors"]
        except (KeyError, TypeError) as error:
            raise RosterError(
                f"Shift {name!r} must have start, end and doctors. Got: {band!r}"
            ) from error

        if isinstance(doctors, bool) or not float(doctors).is_integer():
            raise RosterError(
                f"Shift {name!r}: doctors must be a whole number, got {doctors!r}."
            )
        doctors = int(doctors)
        if doctors < 0:
            raise RosterError(f"Shift {name!r}: doctors cannot be negative, got {doctors}.")
        if not 0 <= start < HOURS_IN_DAY:
            raise RosterError(f"Shift {name!r}: start must be 0-23, got {start}.")
        if not start < end <= HOURS_IN_DAY:
            raise RosterError(
                f"Shift {name!r}: end must be after start and at most 24, got {end}. "
                "A shift crossing midnight is entered as two bands."
            )
        parsed.append(RosterShift(str(name), start, end, doctors))

    parsed.sort(key=lambda shift: shift.start)

    covered: dict[int, str] = {}
    for shift in parsed:
        for hour in shift.hours:
            if hour in covered:
                raise RosterError(
                    f"Hour {hour:02d}:00 is covered by both {covered[hour]!r} and "
                    f"{shift.name!r}. Shifts must not overlap."
                )
            covered[hour] = shift.name

    uncovered = [hour for hour in range(HOURS_IN_DAY) if hour not in covered]
    if uncovered:
        raise RosterError(
            "Every hour of the day must be covered. Missing: "
            + ", ".join(f"{hour:02d}:00" for hour in uncovered)
        )

    if all(shift.doctors == 0 for shift in parsed):
        raise RosterError("The roster has no doctors on duty at any hour.")

    return Roster(tuple(parsed), is_default=is_default)


def roster_from_options(
    roster_input: dict[str, Any] | None, config: dict[str, Any]
) -> Roster:
    """Take the roster from the upload options, or fall back to the demo default."""
    if not roster_input:
        logger.info("No roster supplied; using the demo roster from the config")
        return default_roster(config)
    return parse_roster(roster_input, is_default=False)


def apply_roster(config: dict[str, Any], roster: Roster) -> dict[str, Any]:
    """Write a roster into a run's config.

    Mutates and returns the config passed in, which is always a run-local copy: the
    roster must never reach the shared config or another job.
    """
    config["staffing_roster"] = roster.to_config()
    logger.info(
        "Run roster: %s (%d doctor-hours/day, default=%s)",
        roster.describe(),
        roster.total_doctor_hours,
        roster.is_default,
    )
    return config
