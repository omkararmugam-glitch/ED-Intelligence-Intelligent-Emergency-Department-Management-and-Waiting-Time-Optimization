"""Feature-matrix assembly.

Produces the two named feature sets the G3 ablation compares:

    base        calendar + patient attributes known at arrival
    base+queue  the same, plus queue-state features

The split boundary is the subtle part. Queue state is always measured against the
**whole** validated population, while features are *returned* only for the requested
rows. Without that, the first patient of the test set would appear to walk into an
empty department, because the patients still waiting from the validation period would
be invisible. Measuring against the full population is not leakage: at the moment that
patient arrived, those earlier patients really were in the waiting room.

Every assembled matrix passes through `assert_no_leakage` before it is returned, so a
blocklisted column cannot reach a model even if someone adds one later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

from src.data.schema import Schema, assert_no_leakage, load_schema
from src.features.calendar import CALENDAR_FEATURES, add_calendar_features
from src.features.queue_state import (
    QueueStateIndex,
    add_queue_state_features,
    queue_feature_names,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Patient attributes that are known the moment the patient arrives. `triage_level` is
#: included: triage happens at arrival, before any clinician contact, so it is an input
#: to the wait, not an outcome of it.
PATIENT_FEATURE_CANDIDATES = ("triage_level", "age", "is_ambulance", "is_female")

#: The named feature sets, by the feature groups they contain.
FEATURE_SETS = {
    "base": ("calendar", "patient"),
    "base+queue": ("calendar", "patient", "queue"),
}


@dataclass
class FeatureSpec:
    """The resolved column list for one named feature set."""

    name: str
    groups: tuple[str, ...]
    columns: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.columns)


def add_patient_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Encode the optional patient attributes that are available at arrival.

    Absent optional columns are simply not encoded, which is how the pipeline degrades
    gracefully on a dataset that lacks them.

    Args:
        frame: Validated visits.

    Returns:
        A copy with `is_ambulance` and/or `is_female` added where the source exists.
    """
    out = frame.copy()
    if "arrival_mode" in out.columns:
        out["is_ambulance"] = (out["arrival_mode"].astype("string") == "ambulance").astype("int8")
    if "sex" in out.columns:
        out["is_female"] = (out["sex"].astype("string") == "female").astype("int8")
    return out


def build_features(
    frame: pd.DataFrame,
    config: dict[str, Any],
    reference: pd.DataFrame | None = None,
    index: QueueStateIndex | None = None,
    time_column: str = "arrival_time",
) -> pd.DataFrame:
    """Attach every feature column to `frame`.

    Args:
        frame: Rows to featurise (a split, or the whole table).
        config: Full pipeline config.
        reference: Population to measure queue state against; should be the full
            validated table whenever `frame` is a subset of it.
        index: Pre-built queue-state index, to avoid rebuilding it per split.
        time_column: Arrival timestamp column.

    Returns:
        A copy of `frame` with calendar, patient and queue-state columns added.
    """
    out = add_calendar_features(frame, config, time_column=time_column)
    out = add_patient_features(out)
    out = add_queue_state_features(
        out, config, reference=reference, index=index, time_column=time_column
    )
    return out


def resolve_feature_set(
    name: str,
    available: Iterable[str],
    config: dict[str, Any],
) -> FeatureSpec:
    """Resolve a named feature set to the columns actually present.

    Args:
        name: "base" or "base+queue".
        available: Columns present in the featurised frame.
        config: Full pipeline config.

    Returns:
        A `FeatureSpec` listing the columns to use, in a stable order.

    Raises:
        ValueError: If `name` is not a known feature set, or resolves to no columns.
    """
    if name not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set {name!r}; known sets: {sorted(FEATURE_SETS)}")

    present = set(available)
    groups = FEATURE_SETS[name]
    columns: list[str] = []

    if "calendar" in groups:
        columns += [c for c in CALENDAR_FEATURES if c in present]
    if "patient" in groups:
        chosen = [c for c in PATIENT_FEATURE_CANDIDATES if c in present]
        skipped = [c for c in PATIENT_FEATURE_CANDIDATES if c not in present]
        if skipped:
            logger.info("Feature set %r: patient columns unavailable: %s", name, skipped)
        columns += chosen
    if "queue" in groups:
        columns += [c for c in queue_feature_names(config) if c in present]

    if not columns:
        raise ValueError(f"Feature set {name!r} resolved to zero columns")

    spec = FeatureSpec(name=name, groups=groups, columns=columns)
    logger.info("Feature set %r: %d columns", name, len(spec))
    return spec


def make_design_matrix(
    featurised: pd.DataFrame,
    feature_set: str,
    config: dict[str, Any],
    schema: Schema | None = None,
) -> tuple[pd.DataFrame, FeatureSpec]:
    """Select one feature set's columns and assert the result is leakage-free.

    Args:
        featurised: Output of `build_features`.
        feature_set: "base" or "base+queue".
        config: Full pipeline config.
        schema: Data contract; loaded from `schema.yaml` when omitted.

    Returns:
        Tuple of (design matrix, the resolved spec).

    Raises:
        ValueError: If any selected column is on the contract's leakage blocklist.
    """
    spec = resolve_feature_set(feature_set, featurised.columns, config)
    assert_no_leakage(spec.columns, schema or load_schema())
    return featurised.loc[:, spec.columns].copy(), spec


def add_prolonged_wait_label(
    frame: pd.DataFrame,
    config: dict[str, Any],
    target_column: str = "wait_minutes",
) -> pd.DataFrame:
    """Add the binary classification target for the prolonged-wait model.

    The label is an *outcome*, not a feature, and is on the leakage blocklist so it can
    never be selected into a design matrix.

    Args:
        frame: Table containing `target_column`.
        config: Full pipeline config; reads
            `features.prolonged_wait_threshold_minutes`.
        target_column: Column holding the realised wait.

    Returns:
        A copy with a `prolonged_wait` column (nullable Int8; null where the wait is
        unknown).

    Raises:
        KeyError: If `target_column` is absent.
    """
    if target_column not in frame.columns:
        raise KeyError(f"Cannot label prolonged waits without {target_column!r}")

    threshold = float(config["features"]["prolonged_wait_threshold_minutes"])
    out = frame.copy()
    wait = pd.to_numeric(out[target_column], errors="coerce")
    out["prolonged_wait"] = (wait > threshold).astype("Int8").where(wait.notna())

    known = out["prolonged_wait"].notna()
    logger.info(
        "Prolonged-wait label at >%.0f min: %d of %d known waits are positive (%.1f%%)",
        threshold,
        int(out.loc[known, "prolonged_wait"].sum()),
        int(known.sum()),
        100.0 * float(out.loc[known, "prolonged_wait"].mean()) if known.any() else 0.0,
    )
    return out
