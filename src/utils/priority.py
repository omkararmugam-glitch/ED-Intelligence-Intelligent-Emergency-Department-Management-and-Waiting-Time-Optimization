"""The queue's priority-aging policy, in one place.

Both the synthetic generator (`src/data/synthetic.py`) and the SimPy model
(`src/simulation/des.py`) must apply **exactly** the same rule, or the simulation is
calibrating against a department that does not exist. Rather than implementing it twice
and hoping they stay in step, both import it from here and both read its parameters
from the same config block.

The rule:

    effective_priority = max(triage_level - waited_minutes / minutes_per_level, floor)

with no aging at all for patients already at or above the floor's urgency. Lower is
served first.

`floor` is a **clinical safety constraint, not a tuning knob**. At 2.5 — strictly
between ESI 2 and ESI 3 — an aged low-acuity patient may overtake a fresher ESI 3 or 4,
but never a freshly arrived ESI 1 or 2. See CLAUDE.md for the measured consequences of
moving it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class AgingPolicy:
    """Parameters of the priority-aging rule.

    Attributes:
        minutes_per_level: Minutes of waiting worth one ESI level of priority. Smaller
            means stronger aging.
        floor: Best effective priority aging can reach. Patients already this urgent do
            not age, and are never overtaken by an aged lower-acuity patient.
    """

    minutes_per_level: float
    floor: float

    def __post_init__(self) -> None:
        if self.minutes_per_level <= 0:
            raise ValueError(
                f"minutes_per_level must be positive, got {self.minutes_per_level}"
            )
        if self.floor <= 0:
            raise ValueError(f"floor must be positive, got {self.floor}")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AgingPolicy":
        """Read the policy from the `synthetic` config block.

        The generator's block is the single source of truth: the simulation is a model
        *of* that department, so it must not carry its own copy of the policy.
        """
        synthetic = config["synthetic"]
        return cls(
            minutes_per_level=float(synthetic["priority_aging_minutes_per_level"]),
            floor=float(synthetic["priority_aging_floor"]),
        )

    def effective_priority(self, triage_level: float, waited_minutes: float) -> float:
        """Aged priority for one patient. Lower is served first.

        Args:
            triage_level: ESI level, 1 (most urgent) to 5.
            waited_minutes: How long the patient has been eligible for a clinician.

        Returns:
            The effective priority, never better than `floor`.
        """
        level = float(triage_level)
        if level <= self.floor:
            return level  # already urgent; acuity alone decides
        aged = level - float(waited_minutes) / self.minutes_per_level
        return max(aged, self.floor)

    def effective_priority_array(
        self, triage_levels: np.ndarray, waited_minutes: np.ndarray
    ) -> np.ndarray:
        """Vectorised form of `effective_priority`."""
        levels = np.asarray(triage_levels, dtype="float64")
        waited = np.asarray(waited_minutes, dtype="float64")
        aged = np.maximum(levels - waited / self.minutes_per_level, self.floor)
        return np.where(levels <= self.floor, levels, aged)

    def describe(self) -> str:
        """One-line human-readable summary, for logs and reports."""
        return (
            f"one ESI level per {self.minutes_per_level:g} min waited, "
            f"never better than effective priority {self.floor:g}"
        )
