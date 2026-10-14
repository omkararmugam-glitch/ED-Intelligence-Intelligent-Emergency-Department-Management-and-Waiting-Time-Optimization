"""Data contract loading and validation.

`schema.yaml` is the single source of truth for column names, dtypes and allowed
values. This module loads that contract and validates a visits DataFrame against it.

Validation philosophy:
  * A missing *required column* is a hard error -- the pipeline cannot proceed.
  * A bad *row* is dropped or flagged, never silently repaired, and every drop is
    counted by reason and logged.
  * Patients with no `seen_time` (left without being seen) are not dropped. They are
    kept in a separate flagged table: they still occupied the queue, so the feature
    layer needs them, but they have no wait-time target so the wait model excludes
    them. This exclusion is a known limitation (see README).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.utils.io import load_yaml, resolve_path
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Columns the loader computes; they must never be supplied as input.
DERIVED_COLUMNS = (
    "wait_minutes",
    "los_minutes",
    "is_lwbs",
    "still_waiting_at_cutoff",
)


class ColumnSpec(BaseModel):
    """Specification of one column in the data contract."""

    model_config = ConfigDict(extra="forbid")

    dtype: str
    description: str = ""
    nullable: bool = False
    min: float | None = None
    max: float | None = None
    allowed: list[str] | None = None
    formula: str | None = None


class Schema(BaseModel):
    """The parsed data contract."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    required: dict[str, ColumnSpec]
    optional: dict[str, ColumnSpec] = Field(default_factory=dict)
    derived: dict[str, ColumnSpec] = Field(default_factory=dict)
    leakage_blocklist: list[str] = Field(default_factory=list)

    @property
    def required_columns(self) -> list[str]:
        """Names of columns that must be present in the input."""
        return list(self.required)

    @property
    def optional_columns(self) -> list[str]:
        """Names of columns used when present and skipped when absent."""
        return list(self.optional)

    @property
    def input_columns(self) -> list[str]:
        """All columns the contract knows how to read from input."""
        return self.required_columns + self.optional_columns

    def spec(self, column: str) -> ColumnSpec:
        """Return the spec for a column, looking in required then optional then derived."""
        for group in (self.required, self.optional, self.derived):
            if column in group:
                return group[column]
        raise KeyError(f"Column {column!r} is not part of the data contract")

    def present_optional(self, columns: Iterable[str]) -> list[str]:
        """Return the optional columns that actually exist in `columns`."""
        available = set(columns)
        return [name for name in self.optional_columns if name in available]


def load_schema(path: str | Path = "schema.yaml") -> Schema:
    """Load and validate `schema.yaml` into a `Schema` object."""
    raw = load_yaml(path)
    schema = Schema(**raw)
    logger.info(
        "Loaded data contract v%d from %s (%d required, %d optional columns)",
        schema.version,
        resolve_path(path),
        len(schema.required),
        len(schema.optional),
    )
    return schema


@dataclass
class ValidationReport:
    """Counts produced by a validation pass. Serialised to artifacts/metrics/."""

    n_input: int = 0
    n_valid: int = 0
    n_modelling: int = 0
    n_flagged_no_seen_time: int = 0
    n_lwbs: int = 0
    n_still_waiting_at_cutoff: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    coerced: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    optional_columns_present: list[str] = field(default_factory=list)
    optional_columns_missing: list[str] = field(default_factory=list)

    @property
    def n_dropped(self) -> int:
        """Total number of rows removed."""
        return int(sum(self.dropped.values()))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the report."""
        return {
            "n_input": self.n_input,
            "n_valid": self.n_valid,
            "n_dropped": self.n_dropped,
            "n_modelling": self.n_modelling,
            "n_flagged_no_seen_time": self.n_flagged_no_seen_time,
            "n_lwbs": self.n_lwbs,
            "n_still_waiting_at_cutoff": self.n_still_waiting_at_cutoff,
            "dropped_by_reason": dict(self.dropped),
            "coerced_by_reason": dict(self.coerced),
            "warnings": list(self.warnings),
            "optional_columns_present": list(self.optional_columns_present),
            "optional_columns_missing": list(self.optional_columns_missing),
        }

    def log(self) -> None:
        """Emit the whole report through the logger at INFO level."""
        logger.info(
            "Validation: %d rows in, %d valid, %d dropped, %d never seen "
            "(%d left without being seen + %d still waiting at the cutoff), "
            "%d usable for the wait model",
            self.n_input,
            self.n_valid,
            self.n_dropped,
            self.n_flagged_no_seen_time,
            self.n_lwbs,
            self.n_still_waiting_at_cutoff,
            self.n_modelling,
        )
        for reason, count in sorted(self.dropped.items(), key=lambda kv: -kv[1]):
            if count:
                logger.info("  dropped %6d rows: %s", count, reason)
        for reason, count in sorted(self.coerced.items(), key=lambda kv: -kv[1]):
            if count:
                logger.info("  coerced %6d values: %s", count, reason)
        for message in self.warnings:
            logger.warning("  %s", message)


@dataclass
class ValidatedVisits:
    """The outcome of validation, in the three views downstream code needs.

    Attributes:
        all: Every row that passed validation, including patients never seen.
            The feature layer uses this so queue state counts real occupancy.
        modelling: Rows with a known `wait_minutes`; the wait-time model trains here.
        flagged: Rows with no `seen_time` (LWBS and equivalent), excluded from the
            wait-time model and reported as a limitation.
        report: Counts of everything dropped, coerced and flagged.
    """

    all: pd.DataFrame
    modelling: pd.DataFrame
    flagged: pd.DataFrame
    report: ValidationReport


def _coerce_datetime(series: pd.Series, fmt: str | None = None) -> pd.Series:
    """Parse a column to datetime64, turning unparseable values into NaT."""
    if fmt:
        return pd.to_datetime(series, format=fmt, errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def _normalise_category(series: pd.Series, allowed: list[str] | None, default: str) -> pd.Series:
    """Lower-case a text column and replace values outside `allowed` with `default`."""
    text = series.astype("string").str.strip().str.lower()
    if allowed is None:
        return text
    outside = text.notna() & ~text.isin(allowed)
    if outside.any():
        text = text.where(~outside, default)
    return text.fillna(default)


def validate_visits(
    frame: pd.DataFrame,
    schema: Schema | None = None,
    *,
    strict: bool = False,
) -> ValidatedVisits:
    """Validate a visits table against the data contract and derive wait/LOS.

    Args:
        frame: Raw visits, already mapped to schema column names (see adapter.py).
        schema: Parsed contract; loaded from `schema.yaml` when omitted.
        strict: When True, raise if any row has to be dropped. Useful in tests to
            assert that a generator produces clean data.

    Returns:
        A `ValidatedVisits` bundle with the all/modelling/flagged views and a report.

    Raises:
        ValueError: If a required column is missing, the table is empty, or `strict`
            is set and rows had to be dropped.
    """
    schema = schema or load_schema()
    report = ValidationReport(n_input=int(len(frame)))

    if frame.empty:
        raise ValueError("Cannot validate an empty visits table")

    missing_required = [c for c in schema.required_columns if c not in frame.columns]
    if missing_required:
        raise ValueError(
            "Input is missing required column(s): "
            + ", ".join(missing_required)
            + f". Expected columns: {schema.required_columns}. "
            "For a real dataset, map them in configs/column_map.yaml."
        )

    supplied_derived = [c for c in DERIVED_COLUMNS if c in frame.columns]
    if supplied_derived:
        report.warnings.append(
            f"Input supplied derived column(s) {supplied_derived}; recomputing from timestamps"
        )

    present_optional = schema.present_optional(frame.columns)
    report.optional_columns_present = present_optional
    report.optional_columns_missing = [
        c for c in schema.optional_columns if c not in present_optional
    ]
    if report.optional_columns_missing:
        logger.info(
            "Optional columns absent (degrading gracefully): %s",
            ", ".join(report.optional_columns_missing),
        )

    keep = schema.required_columns + present_optional
    extra = [c for c in frame.columns if c not in keep]
    if extra:
        logger.info("Ignoring %d column(s) not in the contract: %s", len(extra), extra)
    data = frame.loc[:, keep].copy()

    # ---------------------------------------------------------------- type coercion
    data["patient_id"] = data["patient_id"].astype("string").str.strip()

    for column in ("arrival_time", "seen_time", "departure_time"):
        before_na = data[column].isna().sum()
        data[column] = _coerce_datetime(data[column])
        unparseable = int(data[column].isna().sum() - before_na)
        if unparseable > 0:
            report.coerced[f"{column}: unparseable timestamp -> NaT"] = unparseable

    triage_spec = schema.spec("triage_level")
    data["triage_level"] = pd.to_numeric(data["triage_level"], errors="coerce")

    disposition_allowed = schema.spec("disposition").allowed or []
    raw_disposition = data["disposition"].astype("string").str.strip().str.lower()
    unmapped = raw_disposition.notna() & ~raw_disposition.isin(disposition_allowed)
    if unmapped.any():
        report.coerced["disposition: value outside contract -> other"] = int(unmapped.sum())
    data["disposition"] = _normalise_category(data["disposition"], disposition_allowed, "other")

    if "age" in data.columns:
        age_spec = schema.spec("age")
        data["age"] = pd.to_numeric(data["age"], errors="coerce")
        out_of_range = data["age"].notna() & (
            (data["age"] < (age_spec.min if age_spec.min is not None else -np.inf))
            | (data["age"] > (age_spec.max if age_spec.max is not None else np.inf))
        )
        if out_of_range.any():
            report.coerced["age: outside contract range -> missing"] = int(out_of_range.sum())
            data.loc[out_of_range, "age"] = np.nan

    if "sex" in data.columns:
        data["sex"] = _normalise_category(data["sex"], schema.spec("sex").allowed, "unknown")
    if "arrival_mode" in data.columns:
        data["arrival_mode"] = _normalise_category(
            data["arrival_mode"], schema.spec("arrival_mode").allowed, "other"
        )
    if "chief_complaint" in data.columns:
        data["chief_complaint"] = data["chief_complaint"].astype("string").str.strip()

    # ---------------------------------------------------------------- row-level drops
    drop_rules: list[tuple[str, pd.Series]] = [
        ("missing patient_id", data["patient_id"].isna() | (data["patient_id"] == "")),
        ("missing or unparseable arrival_time", data["arrival_time"].isna()),
        (
            "triage_level missing or outside 1-5",
            data["triage_level"].isna()
            | (data["triage_level"] < (triage_spec.min or 1))
            | (data["triage_level"] > (triage_spec.max or 5))
            | (data["triage_level"] % 1 != 0),
        ),
        ("seen_time before arrival_time (negative wait)", data["seen_time"] < data["arrival_time"]),
        (
            "departure_time before arrival_time (negative LOS)",
            data["departure_time"] < data["arrival_time"],
        ),
        ("seen_time after departure_time", data["seen_time"] > data["departure_time"]),
    ]

    drop_mask = pd.Series(False, index=data.index)
    for reason, rule in drop_rules:
        # Attribute each row to the first reason that matches, so counts sum to the total.
        rule = rule.fillna(False).astype(bool) & ~drop_mask
        count = int(rule.sum())
        report.dropped[reason] = count
        drop_mask |= rule

    data = data.loc[~drop_mask].copy()

    duplicated = data.duplicated(subset=["patient_id"], keep="first")
    report.dropped["duplicate patient_id"] = int(duplicated.sum())
    data = data.loc[~duplicated].copy()

    if data.empty:
        raise ValueError(
            f"All {report.n_input} rows were dropped during validation; "
            f"reasons: {report.dropped}"
        )

    data["triage_level"] = data["triage_level"].astype("int16")

    # ---------------------------------------------------------------- derived columns
    # Tie-break on patient_id, not on arrival_time alone. Timestamps are recorded to
    # the minute, so around 6% of visits in a year-long extract share one, and a
    # stable sort leaves those ties in whatever order the rows sat in the file. Queue
    # state is built from that order, so the same visits shuffled train a measurably
    # different model -- a result that cannot be reproduced from the data alone.
    # patient_id is unique by this point (duplicates were dropped above), which makes
    # this a total order.
    data = data.sort_values(
        ["arrival_time", "patient_id"], kind="mergesort"
    ).reset_index(drop=True)
    data["wait_minutes"] = (
        data["seen_time"] - data["arrival_time"]
    ).dt.total_seconds() / 60.0
    data["los_minutes"] = (
        data["departure_time"] - data["arrival_time"]
    ).dt.total_seconds() / 60.0

    # Two quite different reasons a patient has no seen_time, kept strictly apart.
    # Pooling them would bias any time-to-event estimate: a patient still in the
    # waiting room when the extract closed has not experienced the event.
    no_seen_time = data["seen_time"].isna()
    no_departure = data["departure_time"].isna()

    data["still_waiting_at_cutoff"] = (no_seen_time & no_departure).astype(bool)
    data["is_lwbs"] = (
        (data["disposition"] == "lwbs") | (no_seen_time & ~no_departure)
    ).astype(bool)

    inconsistent = (~no_seen_time) & (data["disposition"] == "lwbs")
    if inconsistent.any():
        report.warnings.append(
            f"{int(inconsistent.sum())} rows have disposition='lwbs' but a seen_time; "
            "they are kept for modelling and flagged via is_lwbs"
        )
    unlabelled_departure = no_seen_time & ~no_departure & (data["disposition"] != "lwbs")
    if unlabelled_departure.any():
        report.warnings.append(
            f"{int(unlabelled_departure.sum())} rows have no seen_time and a departure_time "
            "but disposition is not 'lwbs'; treated as left-without-being-seen"
        )
    if data["still_waiting_at_cutoff"].any():
        report.warnings.append(
            f"{int(data['still_waiting_at_cutoff'].sum())} rows have neither seen_time nor "
            "departure_time; treated as still present at the data cutoff and "
            "right-censored, not as abandonments"
        )

    report.n_valid = int(len(data))
    flagged = data.loc[no_seen_time].copy()
    modelling = data.loc[~no_seen_time].copy()
    report.n_flagged_no_seen_time = int(len(flagged))
    report.n_modelling = int(len(modelling))
    report.n_lwbs = int((data["is_lwbs"] & no_seen_time).sum())
    report.n_still_waiting_at_cutoff = int(data["still_waiting_at_cutoff"].sum())

    if report.n_modelling == 0:
        raise ValueError("No rows have a seen_time; the wait-time model has no target to learn")

    report.log()

    if strict and report.n_dropped > 0:
        raise ValueError(f"strict=True but {report.n_dropped} rows were dropped: {report.dropped}")

    return ValidatedVisits(all=data, modelling=modelling, flagged=flagged, report=report)


def assert_no_leakage(feature_columns: Iterable[str], schema: Schema | None = None) -> None:
    """Raise if any feature column is on the contract's leakage blocklist.

    Called by the feature builder and again before every model fit, so a forbidden
    column cannot reach a model even if it is introduced by accident later.

    Args:
        feature_columns: The column names about to be used as model inputs.
        schema: Parsed contract; loaded from `schema.yaml` when omitted.

    Raises:
        ValueError: If a blocklisted column (or a column derived by prefixing one)
            appears in `feature_columns`.
    """
    schema = schema or load_schema()
    columns = list(feature_columns)  # may be a generator; only consume it once
    offenders = sorted(set(columns) & set(schema.leakage_blocklist))
    if offenders:
        raise ValueError(
            "Leakage check failed: these columns are only knowable after arrival and "
            f"must not be features: {offenders}"
        )
    logger.debug("Leakage check passed for %d feature columns", len(columns))
