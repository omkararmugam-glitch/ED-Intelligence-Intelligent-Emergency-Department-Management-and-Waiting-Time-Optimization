"""The alert rules, in one place, as data.

**This module imports no Streamlit and does nothing at import time.** It holds the
rules and the function that evaluates them.

Why the rules are data rather than `if` statements scattered through a page: an
alert whose rule is hidden is indistinguishable from a guess. A charge nurse
reading "ESI 3-5 waits above target for 6 hours" is entitled to know what "above
target" meant and what fired it, so every rule here carries its own plain-language
statement, and `app/views/system_status.py` lists all of them from this same
dictionary. The Command center cannot show an alert that the System status page
does not explain, because both read this file.

**This file is the dashboard's copy; `src/serving/analytics.py` holds the
service's.** They are mirrored rather than shared because the two Docker images do
not overlap: the API image copies `src/` and not `app/`, and the dashboard image
copies `app/` and not `src/`, so neither can import the other's module -- the same
reason `app/config.py` already mirrors `api.auth.API_KEY_HEADER` and the simulator's
aging constants. `tests/test_analytics_api.py` asserts the two rule sets agree, key
for key and threshold for threshold, so the mirror cannot drift.

Every threshold here is a **presentation** threshold -- what is worth putting in
front of a reader -- not a modelling one. None of these values feeds a model, a
simulation or a staffing search, and changing one cannot move a headline number.
The clinical targets they refer to (the 30-minute door-to-doctor target, the 4.0%
abandonment target) come from the pipeline's own config and are quoted, not
redefined.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

#: The door-to-doctor target this department measures itself against, in minutes.
#: Quoted from the project's config rather than invented here.
WAIT_TARGET_MINUTES = 30

#: Acuity levels the target applies to. ESI 1 and 2 are seen almost immediately by
#: construction -- their mean wait is about 5 minutes -- so holding them to the same
#: 30-minute line would produce an alert that never fires and teaches nothing.
TARGETED_LEVELS = (3, 4, 5)


@dataclass(frozen=True)
class Rule:
    """One alert rule: what it checks, when it fires, and how to say so."""

    key: str
    title: str
    statement: str
    level: str
    threshold: str


#: Every rule, keyed as it appears in an alert's `rule` field.
RULES: dict[str, Rule] = {
    "wait_above_target": Rule(
        key="wait_above_target",
        title="Waits above target",
        statement=(
            "Fires when the median wait for ESI 3-5, over the hours in view, is "
            "above the department's door-to-doctor target."
        ),
        level="act",
        threshold=f"median wait > {WAIT_TARGET_MINUTES} min for ESI 3, 4 or 5",
    ),
    "congestion_high": Rule(
        key="congestion_high",
        title="Congestion risk high",
        statement=(
            "Fires when the arrivals forecast puts an hour in the next window above "
            "the busy threshold, which is the 80th percentile of the forecast "
            "itself rather than a fixed number of patients."
        ),
        level="watch",
        threshold="forecast arrivals in an hour > the forecast's own 80th percentile",
    ),
    "lwbs_elevated": Rule(
        key="lwbs_elevated",
        title="Left-without-being-seen rate elevated",
        statement=(
            "Fires when the share of patients who left before being seen is above "
            "the rate the simulation was validated against."
        ),
        level="watch",
        threshold="left-without-being-seen rate > 4.0% of arrivals",
    ),
    "longest_wait": Rule(
        key="longest_wait",
        title="A patient has waited a long time",
        statement=(
            "Fires when any single patient still waiting has been waiting more than "
            "three times the target. One patient is enough: an average hides them."
        ),
        level="act",
        threshold=f"longest current wait > {WAIT_TARGET_MINUTES * 3} min",
    ),
    "all_clear": Rule(
        key="all_clear",
        title="Nothing above threshold",
        statement=(
            "Shown when no other rule fired, so an empty alerts panel is never "
            "ambiguous between 'all clear' and 'not computed'."
        ),
        level="good",
        threshold="no other rule fired",
    ),
}


@dataclass(frozen=True)
class Alert:
    """One fired alert, ready to render."""

    rule: str
    title: str
    detail: str
    level: str
    at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """A plain mapping, for the API response and for a test to compare."""
        return {
            "rule": self.rule,
            "title": self.title,
            "detail": self.detail,
            "severity": self.level,
            "at": self.at,
        }


def _fmt(value: float, places: int = 0) -> str:
    """A number for prose, without pulling in the dashboard's formatters.

    Kept free of `app/theme.py` so this module stays importable on its own and can
    be compared against the service's copy of the same rules.
    """
    return f"{value:,.{places}f}"


def evaluate(
    *,
    waits_by_acuity: Mapping[int, float] | None = None,
    longest_wait_minutes: float | None = None,
    lwbs_rate: float | None = None,
    busy_hours: Sequence[str] = (),
    busy_threshold: float | None = None,
    at: str | None = None,
) -> list[Alert]:
    """Run every rule over what is known, and return what fired.

    Each argument is optional because the pages that call this know different
    things: the Command center has the forecast and the recorded waits, the board
    has the current queue. A rule whose inputs are missing does not fire and does
    not claim all-clear either -- it simply is not evaluated, and the caller's
    "nothing above threshold" line only appears when at least one rule could run.

    Args:
        waits_by_acuity: Median wait in minutes, keyed by ESI level.
        longest_wait_minutes: The longest single wait currently outstanding.
        lwbs_rate: Share of arrivals who left before being seen, as a fraction.
        busy_hours: Formatted hour labels the forecast marked busy.
        busy_threshold: Arrivals per hour above which an hour counts as busy.
        at: The dataset moment all of this describes.

    Returns:
        The fired alerts, most severe first. Never empty when at least one rule
        could be evaluated.
    """
    fired: list[Alert] = []
    evaluated = False

    if waits_by_acuity:
        evaluated = True
        over = {
            level: minutes
            for level, minutes in waits_by_acuity.items()
            if level in TARGETED_LEVELS and minutes is not None
            and minutes > WAIT_TARGET_MINUTES
        }
        if over:
            levels = ", ".join(f"ESI {level}" for level in sorted(over))
            worst = max(over.values())
            fired.append(
                Alert(
                    rule="wait_above_target",
                    title=f"Median wait above the {WAIT_TARGET_MINUTES}-minute target for {levels}",
                    detail=(
                        f"Worst median wait {_fmt(worst)} min against a "
                        f"{WAIT_TARGET_MINUTES} min target."
                    ),
                    level=RULES["wait_above_target"].level,
                    at=at,
                )
            )

    if longest_wait_minutes is not None:
        evaluated = True
        limit = WAIT_TARGET_MINUTES * 3
        if longest_wait_minutes > limit:
            fired.append(
                Alert(
                    rule="longest_wait",
                    title="One patient has waited more than three times the target",
                    detail=(
                        f"Longest current wait {_fmt(longest_wait_minutes)} min "
                        f"against a {limit} min threshold."
                    ),
                    level=RULES["longest_wait"].level,
                    at=at,
                )
            )

    if busy_hours:
        evaluated = True
        threshold = (
            f" above {_fmt(busy_threshold, 1)} arrivals/hour"
            if busy_threshold is not None
            else ""
        )
        shown = ", ".join(busy_hours[:3])
        more = f" and {len(busy_hours) - 3} more" if len(busy_hours) > 3 else ""
        fired.append(
            Alert(
                rule="congestion_high",
                title=f"Congestion risk high at {shown}{more}",
                detail=f"{len(busy_hours)} hour(s) forecast{threshold}.",
                level=RULES["congestion_high"].level,
                at=at,
            )
        )

    if lwbs_rate is not None:
        evaluated = True
        if lwbs_rate > 0.04:
            fired.append(
                Alert(
                    rule="lwbs_elevated",
                    title="Left-without-being-seen rate above 4.0%",
                    detail=f"{lwbs_rate * 100:.1f}% of arrivals left before being seen.",
                    level=RULES["lwbs_elevated"].level,
                    at=at,
                )
            )

    if evaluated and not fired:
        fired.append(
            Alert(
                rule="all_clear",
                title="Nothing above threshold",
                detail=(
                    "Every rule was checked against the figures above and none "
                    "fired."
                ),
                level=RULES["all_clear"].level,
                at=at,
            )
        )

    order = {"act": 0, "watch": 1, "info": 2, "good": 3}
    return sorted(fired, key=lambda alert: order.get(alert.level, 9))
