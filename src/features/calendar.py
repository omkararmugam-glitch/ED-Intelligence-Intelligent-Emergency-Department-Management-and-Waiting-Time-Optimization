"""Calendar features.

Everything here is a deterministic function of a single timestamp, so none of it can
leak: the calendar is knowable arbitrarily far in advance.

Cyclical quantities (hour of day, day of week, month) are encoded as sine/cosine pairs
as well as raw integers. The raw integer lets a tree split on a specific hour; the
sine/cosine pair tells any model that hour 23 and hour 0 are adjacent, which an integer
does not. Both are cheap, so both are provided.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Raw calendar columns produced by `add_calendar_features`.
CALENDAR_BASE_FEATURES = (
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "week_of_year",
    "is_weekend",
    "is_holiday",
    "minutes_since_midnight",
)

#: Cyclical encodings produced by `add_calendar_features`.
CALENDAR_CYCLICAL_FEATURES = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
)

#: Every calendar feature, in a stable order.
CALENDAR_FEATURES = CALENDAR_BASE_FEATURES + CALENDAR_CYCLICAL_FEATURES


def _cyclical(values: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    """Encode a cyclical integer as a (sin, cos) pair on the unit circle."""
    radians = 2.0 * np.pi * values.astype("float64") / period
    return np.sin(radians), np.cos(radians)


def parse_holidays(holidays: Iterable[str] | None) -> set[pd.Timestamp]:
    """Normalise a config list of holiday dates into a set of midnight timestamps.

    Args:
        holidays: Date strings such as "2024-12-25", or None.

    Returns:
        Set of normalised timestamps. Unparseable entries are skipped with a warning.
    """
    parsed: set[pd.Timestamp] = set()
    for entry in holidays or []:
        try:
            stamp = pd.Timestamp(entry)
        except (ValueError, TypeError):
            stamp = pd.NaT
        # `pd.Timestamp(None)` yields NaT rather than raising, so check the value too.
        if pd.isna(stamp):
            logger.warning("Ignoring unparseable holiday entry %r", entry)
            continue
        parsed.add(stamp.normalize())
    return parsed


def add_calendar_features(
    frame: pd.DataFrame,
    config: dict[str, Any],
    time_column: str = "arrival_time",
) -> pd.DataFrame:
    """Add calendar features derived from `time_column`.

    Args:
        frame: Table containing `time_column`.
        config: Full pipeline config; reads `features.holidays`.
        time_column: Timestamp column to derive from.

    Returns:
        A copy of `frame` with the columns in `CALENDAR_FEATURES` added.

    Raises:
        KeyError: If `time_column` is absent.
        ValueError: If `time_column` contains missing timestamps.
    """
    if time_column not in frame.columns:
        raise KeyError(f"Cannot build calendar features without {time_column!r}")

    out = frame.copy()
    times = pd.to_datetime(out[time_column])
    if times.isna().any():
        raise ValueError(
            f"{int(times.isna().sum())} rows have a missing {time_column}; validate the "
            "data before building features"
        )

    out["hour"] = times.dt.hour.astype("int16")
    out["day_of_week"] = times.dt.dayofweek.astype("int16")
    out["month"] = times.dt.month.astype("int16")
    out["day_of_year"] = times.dt.dayofyear.astype("int16")
    out["week_of_year"] = times.dt.isocalendar()["week"].to_numpy().astype("int16")
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("int8")
    out["minutes_since_midnight"] = (
        times.dt.hour * 60 + times.dt.minute
    ).astype("int16")

    holidays = parse_holidays(config.get("features", {}).get("holidays"))
    out["is_holiday"] = times.dt.normalize().isin(holidays).astype("int8")
    matched = int(out["is_holiday"].sum())

    # A short window legitimately contains no holiday, so only a long span with none is
    # worth warning about; otherwise every backtest fold emits a false alarm.
    span_days = float((times.max() - times.min()).total_seconds() / 86400.0)
    if holidays and matched == 0 and span_days >= 90:
        logger.warning(
            "None of the %d configured holidays fall inside the data range %s..%s; "
            "is_holiday is constant and will be useless. Update features.holidays.",
            len(holidays),
            times.min().date(),
            times.max().date(),
        )
    elif matched:
        logger.info("Calendar features: %d visits fall on a configured holiday", matched)
    else:
        logger.debug(
            "No configured holiday falls in this %.1f-day window; is_holiday is constant",
            span_days,
        )

    out["hour_sin"], out["hour_cos"] = _cyclical(out["hour"], 24.0)
    out["dow_sin"], out["dow_cos"] = _cyclical(out["day_of_week"], 7.0)
    out["month_sin"], out["month_cos"] = _cyclical(out["month"], 12.0)

    logger.info("Added %d calendar features", len(CALENDAR_FEATURES))
    return out
