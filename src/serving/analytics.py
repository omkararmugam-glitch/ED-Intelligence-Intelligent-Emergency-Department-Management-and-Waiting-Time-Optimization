"""Read-only analytics over the saved artifacts, for the `/v1/analytics` routes.

Pure functions returning plain dictionaries. Nothing here fits a model, runs a
simulation, writes a file or mutates anything: every figure is read from what a
pipeline stage already wrote, or computed from the validated visits table by
counting. That is what makes these routes safe to add without touching a result.

**The reconstruction rule.** Anything "as at" a moment is computed from arrivals
strictly before that moment. A dashboard or an API that answered "who was waiting at
18:00" using a patient who walked in at 18:05 would be reporting hindsight as
operational fact, so the truncation happens once, here, in `_before`.

**On the duplication with `app/alerts.py`.** The alert rules exist twice: there, for
the dashboard, and here, for the service. The two Docker images do not overlap --
the API image copies `src/` and not `app/`, the dashboard image copies `app/` and
not `src/` -- so neither can import the other's module. They are mirrored for the
same reason `app/config.py` mirrors `api.auth.API_KEY_HEADER`, and
`tests/test_analytics_api.py` asserts the two agree key for key so the mirror cannot
drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import load_json, resolve_path
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: The door-to-doctor target, in minutes. Quoted from the analysis config, not
#: redefined: `alerts.WAIT_TARGET_MINUTES` in the dashboard holds the same number.
WAIT_TARGET_MINUTES = 30

#: Acuity levels the target applies to. ESI 1 and 2 are seen almost immediately by
#: construction, so holding them to the same line would produce a rule that never
#: fires and teaches nothing.
TARGETED_LEVELS = (3, 4, 5)

#: The abandonment rate the simulation was validated against.
LWBS_TARGET = 0.04

#: Columns needed from the validated visits table. Asked for by name so a schema
#: change fails here with a clear message rather than producing a quietly wrong count.
VISIT_COLUMNS = (
    "patient_id",
    "arrival_time",
    "seen_time",
    "departure_time",
    "triage_level",
    "wait_minutes",
    "arrival_mode",
    "is_lwbs",
)

#: The five alert rules, mirrored from `app/alerts.py`. Each carries the threshold
#: it fires on and a statement a reader can check it against.
ALERT_RULES: dict[str, dict[str, str]] = {
    "wait_above_target": {
        "title": "Waits above target",
        "severity": "act",
        "threshold": f"median wait > {WAIT_TARGET_MINUTES} min for ESI 3, 4 or 5",
    },
    "congestion_high": {
        "title": "Congestion risk high",
        "severity": "watch",
        "threshold": "forecast arrivals in an hour > the forecast's own 80th percentile",
    },
    "lwbs_elevated": {
        "title": "Left-without-being-seen rate elevated",
        "severity": "watch",
        "threshold": f"left-without-being-seen rate > {LWBS_TARGET * 100:.1f}% of arrivals",
    },
    "longest_wait": {
        "title": "A patient has waited a long time",
        "severity": "act",
        "threshold": f"longest current wait > {WAIT_TARGET_MINUTES * 3} min",
    },
    "all_clear": {
        "title": "Nothing above threshold",
        "severity": "good",
        "threshold": "no other rule fired",
    },
}


class AnalyticsUnavailable(RuntimeError):
    """The saved tables this needs have not been produced yet.

    Raised rather than returning empty figures, so a route can answer 503 with the
    command that fixes it. A zero that means "not computed" is worse than an error:
    it looks like a quiet department.
    """


# ------------------------------------------------------------------- loading


@dataclass(frozen=True)
class Extract:
    """The validated visits table, plus the window it covers."""

    visits: pd.DataFrame
    data_version: str

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(self.visits["arrival_time"].min())

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp(self.visits["arrival_time"].max())


def visits_path(config: dict[str, Any]) -> Path:
    """Where the validated visits table should be."""
    processed = config.get("paths", {}).get("processed_dir", "data/processed")
    return resolve_path(Path(processed) / "visits_validated.parquet")


def load_extract(config: dict[str, Any], data_version: str = "unknown") -> Extract:
    """The validated visits table, or `AnalyticsUnavailable`."""
    path = visits_path(config)
    if not path.is_file():
        raise AnalyticsUnavailable(
            "The validated visits table has not been produced. Run: "
            "python -m src.pipeline run --config configs/default.yaml --data real"
        )
    try:
        frame = pd.read_parquet(path, columns=list(VISIT_COLUMNS))
    except (OSError, ValueError, KeyError) as error:
        raise AnalyticsUnavailable(
            f"The validated visits table could not be read: {error}"
        ) from error

    for column in ("arrival_time", "seen_time", "departure_time"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame = frame.dropna(subset=["arrival_time"]).sort_values("arrival_time")
    return Extract(frame.reset_index(drop=True), data_version)


def _before(visits: pd.DataFrame, moment: pd.Timestamp) -> pd.DataFrame:
    """Arrivals strictly before `moment`. The one place truncation happens."""
    return visits.loc[visits["arrival_time"] < moment]


def _waiting(past: pd.DataFrame, moment: pd.Timestamp) -> pd.DataFrame:
    """Those who had arrived and were not yet seen or gone at `moment`."""
    if past.empty:
        return past
    resolved = past["seen_time"].fillna(past["departure_time"])
    return past.loc[resolved.isna() | (resolved > moment)]


def default_moment(extract: Extract) -> pd.Timestamp:
    """The moment to describe when the caller does not choose one.

    The midpoint of the extract rather than its end. At the final timestamp the
    waiting room is empty by construction -- everyone earlier has been seen and the
    last arrival is excluded by the strict before-the-cutoff rule -- so an
    end-of-extract default would answer with zeros and look like a quiet department
    rather than an arbitrary instant.
    """
    return extract.start + (extract.end - extract.start) / 2


# ------------------------------------------------------------------- overview


def overview(extract: Extract, at: pd.Timestamp | None = None) -> dict[str, Any]:
    """The Command center's figures and alerts, as at one moment."""
    moment = pd.Timestamp(at) if at is not None else default_moment(extract)
    past = _before(extract.visits, moment)
    waiting = _waiting(past, moment)

    longest = (
        float(((moment - waiting["arrival_time"]).dt.total_seconds() / 60).max())
        if not waiting.empty
        else 0.0
    )
    in_treatment = 0
    if not past.empty:
        seen = past["seen_time"]
        departed = past["departure_time"]
        in_treatment = int(
            (seen.notna() & (seen <= moment) & (departed.isna() | (departed > moment))).sum()
        )

    same_day = past.loc[past["arrival_time"].dt.date == moment.date()]
    waits = pd.to_numeric(same_day["wait_minutes"], errors="coerce").dropna()
    levels = pd.to_numeric(same_day["triage_level"], errors="coerce")
    lwbs = same_day["is_lwbs"].fillna(False).astype(bool) if not same_day.empty else pd.Series(dtype=bool)

    by_acuity = {
        int(level): float(group.median())
        for level, group in pd.to_numeric(
            same_day["wait_minutes"], errors="coerce"
        ).groupby(levels)
        if pd.notna(level) and not group.dropna().empty
    }

    kpis = {
        "waiting": int(len(waiting)),
        "in_treatment": in_treatment,
        "longest_wait_minutes": round(longest, 1),
        "median_wait_minutes_today": round(float(waits.median()), 1) if not waits.empty else None,
        "lwbs_rate_today": (
            round(float(lwbs.sum() / len(same_day)), 4) if len(same_day) and not lwbs.empty
            else None
        ),
        "arrivals_today": int(len(same_day)),
    }

    return {
        "as_of": moment.isoformat(),
        "data_version": extract.data_version,
        "kpis": kpis,
        "median_wait_by_acuity": by_acuity,
        "alerts": evaluate_alerts(
            waits_by_acuity=by_acuity,
            longest_wait_minutes=longest,
            lwbs_rate=kpis["lwbs_rate_today"],
            at=moment.isoformat(),
        ),
        "notice": (
            "Reconstructed from a recorded extract using arrivals before the stated "
            "moment. Synthetic data; decision support only, not for clinical use."
        ),
    }


def evaluate_alerts(
    *,
    waits_by_acuity: dict[int, float] | None = None,
    longest_wait_minutes: float | None = None,
    lwbs_rate: float | None = None,
    busy_hours: list[str] | None = None,
    at: str | None = None,
) -> list[dict[str, Any]]:
    """Run the rules and return what fired, most severe first.

    Mirrors `app.alerts.evaluate`. An empty list means no rule could be evaluated,
    which is deliberately different from "all clear": a caller must be able to tell
    a quiet department from a missing input.
    """
    fired: list[dict[str, Any]] = []
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
            fired.append({
                "rule": "wait_above_target",
                "title": f"Median wait above the {WAIT_TARGET_MINUTES}-minute target for {levels}",
                "detail": (
                    f"Worst median wait {max(over.values()):.0f} min against a "
                    f"{WAIT_TARGET_MINUTES} min target."
                ),
                "severity": "act",
                "at": at,
            })

    if longest_wait_minutes is not None:
        evaluated = True
        limit = WAIT_TARGET_MINUTES * 3
        if longest_wait_minutes > limit:
            fired.append({
                "rule": "longest_wait",
                "title": "One patient has waited more than three times the target",
                "detail": (
                    f"Longest current wait {longest_wait_minutes:.0f} min against a "
                    f"{limit} min threshold."
                ),
                "severity": "act",
                "at": at,
            })

    if busy_hours:
        evaluated = True
        fired.append({
            "rule": "congestion_high",
            "title": f"Congestion risk high at {', '.join(busy_hours[:3])}",
            "detail": f"{len(busy_hours)} hour(s) forecast above the busy threshold.",
            "severity": "watch",
            "at": at,
        })

    if lwbs_rate is not None:
        evaluated = True
        if lwbs_rate > LWBS_TARGET:
            fired.append({
                "rule": "lwbs_elevated",
                "title": f"Left-without-being-seen rate above {LWBS_TARGET * 100:.1f}%",
                "detail": f"{lwbs_rate * 100:.1f}% of arrivals left before being seen.",
                "severity": "watch",
                "at": at,
            })

    if evaluated and not fired:
        fired.append({
            "rule": "all_clear",
            "title": "Nothing above threshold",
            "detail": "Every rule was checked against the figures and none fired.",
            "severity": "good",
            "at": at,
        })

    order = {"act": 0, "watch": 1, "info": 2, "good": 3}
    return sorted(fired, key=lambda alert: order.get(alert["severity"], 9))


# --------------------------------------------------------------- flow by hour


def flow_hourly(extract: Extract, days: int = 7) -> dict[str, Any]:
    """Arrivals and the waiting-room census, hour by hour, over the last `days`.

    The census is counted at the top of each hour from arrivals before it, so the
    series is a reconstruction rather than a running total.
    """
    end = extract.end.floor("h")
    start = end - pd.Timedelta(days=days)
    visits = extract.visits

    arrivals = (
        visits.loc[visits["arrival_time"] >= start, "arrival_time"]
        .dt.floor("h")
        .value_counts()
        .sort_index()
    )

    points = []
    for hour in pd.date_range(start, end, freq="h"):
        past = _before(visits, hour)
        points.append({
            "hour": hour.isoformat(),
            "arrivals": int(arrivals.get(hour, 0)),
            "waiting": int(len(_waiting(past, hour))),
        })

    return {
        "from": start.isoformat(),
        "to": end.isoformat(),
        "data_version": extract.data_version,
        "points": points,
    }


# -------------------------------------------------------------- acuity summary


def acuity_summary(extract: Extract) -> dict[str, Any]:
    """Waits and outcomes per acuity, over the whole extract."""
    visits = extract.visits
    levels = pd.to_numeric(visits["triage_level"], errors="coerce")
    waits = pd.to_numeric(visits["wait_minutes"], errors="coerce")
    lwbs = visits["is_lwbs"].fillna(False).astype(bool)

    rows = []
    for level, index in levels.groupby(levels).groups.items():
        block = waits.loc[index].dropna()
        rows.append({
            "triage_level": int(level),
            "arrivals": int(len(index)),
            "left_without_being_seen": int(lwbs.loc[index].sum()),
            "median_wait_minutes": round(float(block.median()), 1) if not block.empty else None,
            "p90_wait_minutes": round(float(block.quantile(0.9)), 1) if not block.empty else None,
            "share_over_target": (
                round(float((block > WAIT_TARGET_MINUTES).mean()), 4) if not block.empty else None
            ),
        })

    return {
        "data_version": extract.data_version,
        "target_minutes": WAIT_TARGET_MINUTES,
        "levels": sorted(rows, key=lambda row: row["triage_level"]),
        "notice": (
            "ESI 1 and 2 are seen almost immediately by construction, so their "
            "share over target is near zero and says nothing about performance."
        ),
    }


# ----------------------------------------------------------------- the board


def board(
    extract: Extract,
    at: pd.Timestamp | None = None,
    *,
    limit: int = 25,
    offset: int = 0,
    aging_minutes_per_level: float = 25.0,
    aging_floor: float = 2.5,
) -> dict[str, Any]:
    """Who was waiting at `at`, in the department's own priority order, paginated.

    Ordered by the simulator's aging rule rather than by arrival time, because that
    is the order a tracking board has to show. The rule's parameters are passed in
    rather than hard-coded so the caller reads them from the config that the
    generator and the simulation also read.
    """
    moment = pd.Timestamp(at) if at is not None else default_moment(extract)
    past = _before(extract.visits, moment)
    waiting = _waiting(past, moment).copy()

    if waiting.empty:
        return {
            "as_of": moment.isoformat(),
            "data_version": extract.data_version,
            "total": 0,
            "limit": limit,
            "offset": offset,
            "patients": [],
        }

    waiting["waited_minutes"] = (
        (moment - waiting["arrival_time"]).dt.total_seconds() / 60
    ).round(0)
    levels = pd.to_numeric(waiting["triage_level"], errors="coerce")
    aged = levels - waiting["waited_minutes"] / aging_minutes_per_level
    waiting["effective_priority"] = aged.clip(lower=aging_floor).where(
        levels > aging_floor, levels
    )
    waiting = waiting.sort_values(
        ["effective_priority", "waited_minutes"], ascending=[True, False]
    )

    page = waiting.iloc[offset:offset + limit]
    patients = [
        {
            "patient_id": str(row.patient_id),
            "triage_level": (
                int(row.triage_level) if pd.notna(row.triage_level) else None
            ),
            "arrival_mode": (
                str(row.arrival_mode) if pd.notna(row.arrival_mode) else None
            ),
            "waited_minutes": float(row.waited_minutes),
            "effective_priority": round(float(row.effective_priority), 2),
        }
        for row in page.itertuples()
    ]

    return {
        "as_of": moment.isoformat(),
        "data_version": extract.data_version,
        "total": int(len(waiting)),
        "limit": limit,
        "offset": offset,
        "patients": patients,
    }


# ------------------------------------------------------------------- metadata


def meta(config: dict[str, Any], extract: Extract | None, versions: dict[str, Any]) -> dict[str, Any]:
    """Versions, the window the data covers, row counts and the roster."""
    metrics_dir = Path(config.get("paths", {}).get("metrics_dir", "artifacts/metrics"))
    roster = None
    try:
        calibration = load_json(resolve_path(metrics_dir / "sim_calibration.json"))
        roster = calibration.get("roster")
    except (OSError, ValueError):
        roster = None

    window = None
    rows = None
    if extract is not None:
        window = {"from": extract.start.isoformat(), "to": extract.end.isoformat()}
        rows = int(len(extract.visits))

    return {
        "model_version": versions.get("model_version"),
        "data_version": versions.get("data_version"),
        "api_version": versions.get("api_version"),
        "data_window": window,
        "n_visits": rows,
        "roster": roster,
        "wait_target_minutes": WAIT_TARGET_MINUTES,
        "notice": (
            "Synthetic data throughout. Decision support only, not for clinical use."
        ),
    }
