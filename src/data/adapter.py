"""Real-dataset adapter.

Every piece of knowledge about a real dataset's quirks -- its column names, its coded
acuity values, its disposition vocabulary, whether it stores waits as timestamps or as
durations -- lives in `configs/column_map.yaml` and is applied here. Nothing else in
`src/` may reference a non-schema column name.

To onboard a new dataset you edit the YAML, not this file. See README.md,
"Adding a real dataset".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.data.schema import Schema, load_schema
from src.utils.io import load_yaml, resolve_path
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Raw file suffixes pandas can read without an extra dependency.
SUPPORTED_SUFFIXES = (".csv", ".csv.gz", ".tsv", ".txt", ".gz")


class ColumnMapping(BaseModel):
    """How one schema column is obtained from the raw dataset."""

    model_config = ConfigDict(extra="forbid")

    source: str
    datetime_format: str | None = None
    value_map: dict[str, Any] | None = None
    default: Any | None = None


class DurationFallback(BaseModel):
    """Reconstruct `seen_time`/`departure_time` when the source stores durations."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    wait_minutes_source: str | None = None
    los_minutes_source: str | None = None
    units: str = "minutes"

    @property
    def factor(self) -> float:
        """Multiplier that converts the source units to minutes."""
        units = self.units.lower()
        if units in ("minute", "minutes", "min"):
            return 1.0
        if units in ("hour", "hours", "hr", "h"):
            return 60.0
        raise ValueError(f"Unsupported duration unit {self.units!r}; use minutes or hours")


#: What any result computed from a proxy wait must be labelled with.
#:
#: MIMIC-IV-ED, the real dataset this project is being prepared for, has **no first
#: provider contact timestamp**. `edstays` gives arrival (`intime`) and departure
#: (`outtime`) and nothing in between. The closest available proxies are the first
#: `vitalsign.charttime` and the first `pyxis.charttime` (a medication dispense), and
#: neither is door-to-provider:
#:
#:   * first vitals is usually recorded **at triage**, often by a nurse before any
#:     clinician sees the patient, so it under-estimates the wait;
#:   * first medication dispense happens only after a clinician has decided something,
#:     so it over-estimates the wait, and it exists only for patients who were given a
#:     drug at all -- a biased subset, and one correlated with acuity.
#:
#: Either way the quantity is not the one the models were built on, so it must not be
#: reported under the same name.
PROXY_WAIT_NOTICE = (
    "PROXY WAIT (NOT TRUE DOOR-TO-PROVIDER). `seen_time` was derived from a proxy "
    "timestamp because the source dataset records no first-provider-contact time. "
    "Every wait, every prolonged-wait label and every simulation KPI derived from it "
    "measures the proxy, not the door-to-provider interval the models were designed "
    "for. The two differ systematically, not merely noisily."
)


class ProxyWait(BaseModel):
    """Derive `seen_time` from a proxy when the source has no provider-contact time.

    Off by default, and deliberately awkward to turn on: it must name the proxy in
    words, because every number downstream inherits that choice and a reader has to be
    able to see which proxy produced it.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    #: Raw column holding the proxy timestamp.
    source: str | None = None
    #: What the proxy actually is, in words, e.g. "first vitalsign charttime".
    #: Required when enabled: an unnamed proxy is an unreportable one.
    proxy_name: str | None = None
    #: Parsed with this format when given.
    datetime_format: str | None = None
    #: Only fill `seen_time` where it is missing, rather than overwriting a real one.
    only_when_missing: bool = True

    @property
    def is_active(self) -> bool:
        """Whether a proxy will actually be applied."""
        return bool(self.enabled and self.source)

    def describe(self) -> str:
        """One line naming the proxy, for logs, the report and the dashboard."""
        if not self.is_active:
            return "none: seen_time is a recorded provider-contact timestamp"
        return f"{self.proxy_name or self.source} (column {self.source!r})"


class MappingValidation(BaseModel):
    """Guardrails against a mapping that fails silently."""

    model_config = ConfigDict(extra="forbid")

    #: Warn when a value_map matches fewer than this share of non-null rows.
    min_value_map_match_rate: float = 0.05
    #: Raise when a schema column is unsourced but the raw file has that exact column.
    fail_on_unsourced_present_column: bool = True


class ColumnMap(BaseModel):
    """Parsed `configs/column_map.yaml`."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str = "unnamed_dataset"
    read_options: dict[str, Any] = Field(default_factory=dict)
    columns: dict[str, ColumnMapping]
    duration_fallback: DurationFallback = Field(default_factory=DurationFallback)
    proxy_wait: ProxyWait = Field(default_factory=ProxyWait)
    validation: MappingValidation = Field(default_factory=MappingValidation)


def load_column_map(path: str | Path = "configs/column_map.yaml") -> ColumnMap:
    """Load and validate the real-dataset column map."""
    column_map = ColumnMap(**load_yaml(path))
    logger.info(
        "Loaded column map for dataset %r from %s (%d mapped columns)",
        column_map.dataset_name,
        resolve_path(path),
        len(column_map.columns),
    )
    return column_map


def proxy_wait_status(config: dict[str, Any]) -> dict[str, Any]:
    """Whether this run's column map derives `seen_time` from a proxy.

    The report and the dashboard call this rather than inspecting the data, because
    the fact is a property of the mapping, not of any row: it is true for the whole
    dataset or for none of it.

    Args:
        config: Full pipeline config; reads `paths.column_map`.

    Returns:
        A dict with `active`, a human-readable `proxy`, and the `notice` to display.
        `active` is False when the map is absent or unreadable -- a missing map is not
        evidence that a proxy was used.
    """
    path = resolve_path(config["paths"]["column_map"])
    if not path.is_file():
        return {"active": False, "proxy": None, "notice": None}
    try:
        column_map = ColumnMap(**load_yaml(path))
    except Exception as error:  # noqa: BLE001 - a bad map must not break the report
        logger.warning("Could not read %s to check for a proxy wait: %s", path, error)
        return {"active": False, "proxy": None, "notice": None}

    proxy = column_map.proxy_wait
    return {
        "active": proxy.is_active,
        "proxy": proxy.describe() if proxy.is_active else None,
        "notice": PROXY_WAIT_NOTICE if proxy.is_active else None,
        "dataset_name": column_map.dataset_name,
    }


def read_raw(path: str | Path, read_options: dict[str, Any] | None = None) -> pd.DataFrame:
    """Read the raw dataset file with the options from the column map.

    Args:
        path: Path to the raw file (CSV/TSV, optionally gzipped).
        read_options: Keyword arguments forwarded to `pandas.read_csv`.

    Returns:
        The raw DataFrame, untouched apart from parsing.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the suffix is not a supported text format.
    """
    full = resolve_path(path)
    if not full.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found at {full}. Put your file there or change "
            "paths.raw_file in the config."
        )
    if not str(full).lower().endswith(SUPPORTED_SUFFIXES):
        raise ValueError(
            f"Unsupported raw file type {full.suffix!r}. Supported: {SUPPORTED_SUFFIXES}. "
            "Convert your extract to CSV first (keeping extra dependencies out of the stack)."
        )
    frame = pd.read_csv(full, **(read_options or {}))
    logger.info("Read raw dataset %s: %d rows x %d columns", full, len(frame), frame.shape[1])
    logger.debug("Raw columns: %s", list(frame.columns))
    return frame


#: Schema dtypes that are parsed numerically when no value_map is supplied.
NUMERIC_DTYPES = ("int", "float")


def _coerce_numeric(series: pd.Series, column: str) -> pd.Series:
    """Parse a numeric column, accepting ints, floats and their string forms.

    `1`, `1.0`, `"1"` and `"1.0"` all become 1.0. Anything else becomes missing and is
    counted, so an unparseable acuity code is visible here rather than appearing as an
    unexplained drop in the validator.
    """
    before_missing = int(series.isna().sum())
    numeric = pd.to_numeric(series, errors="coerce")
    unparseable = int(numeric.isna().sum()) - before_missing

    if unparseable > 0:
        offending = series[numeric.isna() & series.notna()]
        logger.warning(
            "Column %r: %d value(s) are not numeric and became missing. Examples: %s",
            column,
            unparseable,
            sorted({str(v) for v in offending.head(20)})[:8],
        )
    if before_missing:
        logger.info("Column %r: %d value(s) were already missing in the raw file",
                    column, before_missing)
    return numeric


def _apply_value_map(
    series: pd.Series,
    mapping: ColumnMapping,
    column: str,
    min_match_rate: float = 0.05,
) -> pd.Series:
    """Recode a column through its `value_map`, falling back to `default`.

    Matching is case-insensitive and whitespace-insensitive on the string form of the
    value, so "Walk-In " and "walk-in" both hit a "Walk-in" key.

    A value_map that matches almost nothing is treated as a configuration error rather
    than as data: the usual cause is a map written for a different extract, which would
    otherwise turn the whole column into its `default` and look like a data problem
    much later. `min_match_rate` controls how quiet that failure is allowed to be.

    Args:
        series: The raw column.
        mapping: Its mapping entry.
        column: Schema column name, for messages.
        min_match_rate: Warn below this share of non-null rows matched.

    Returns:
        The recoded column.
    """
    assert mapping.value_map is not None
    lookup = {str(key).strip().lower(): value for key, value in mapping.value_map.items()}
    keys = series.astype("string").str.strip().str.lower()
    mapped = keys.map(lookup)

    n_present = int(keys.notna().sum())
    n_matched = int((keys.notna() & mapped.notna()).sum())
    match_rate = n_matched / n_present if n_present else 1.0

    if n_present and match_rate < min_match_rate:
        logger.warning("=" * 78)
        logger.warning(
            "Column %r: the value_map matched only %d of %d non-null rows (%.2f%%). "
            "This is almost certainly a mapping written for a different extract.",
            column,
            n_matched,
            n_present,
            100.0 * match_rate,
        )
        logger.warning(
            "  value_map keys : %s", sorted(mapping.value_map)[:10]
        )
        logger.warning(
            "  actual values  : %s",
            sorted(keys.dropna().unique().tolist())[:10],
        )
        logger.warning(
            "  Every unmatched row becomes %r, so the column would be near-constant. "
            "Fix configs/column_map.yaml.",
            mapping.default,
        )
        logger.warning("=" * 78)
    elif n_present:
        logger.debug(
            "Column %r: value_map matched %.1f%% of rows", column, 100.0 * match_rate
        )

    unmapped = keys.notna() & mapped.isna()
    if unmapped.any():
        examples = sorted(keys[unmapped].dropna().unique().tolist())[:8]
        if mapping.default is None:
            logger.warning(
                "Column %r: %d value(s) are not in value_map and no default is set, so "
                "they become missing. Unmapped examples: %s",
                column,
                int(unmapped.sum()),
                examples,
            )
        else:
            logger.info(
                "Column %r: mapped %d value(s) to default %r. Unmapped examples: %s",
                column,
                int(unmapped.sum()),
                mapping.default,
                examples,
            )
            mapped = mapped.fillna(mapping.default)
    return mapped


def adapt_real_dataset(
    raw: pd.DataFrame,
    column_map: ColumnMap,
    schema: Schema | None = None,
) -> pd.DataFrame:
    """Rename and recode a raw dataset into the schema's input columns.

    This function does *not* validate the result; the pipeline passes its output to
    `src.data.schema.validate_visits`, so real and synthetic data take exactly the
    same validation path.

    Args:
        raw: The raw DataFrame as read from disk.
        column_map: Parsed column map describing the source dataset.
        schema: Parsed data contract; loaded from `schema.yaml` when omitted.

    Returns:
        DataFrame whose columns are schema input column names.

    Raises:
        ValueError: If a required schema column has no usable source column.
    """
    schema = schema or load_schema()
    out = pd.DataFrame(index=raw.index)
    missing_sources: list[str] = []

    for column, mapping in column_map.columns.items():
        if column not in schema.input_columns:
            logger.warning(
                "column_map entry %r is not a schema column (%s); ignoring it",
                column,
                schema.input_columns,
            )
            continue
        if mapping.source not in raw.columns:
            missing_sources.append(f"{column} (source {mapping.source!r})")
            continue

        series = raw[mapping.source]
        spec = schema.spec(column)

        if mapping.value_map:
            series = _apply_value_map(
                series, mapping, column, column_map.validation.min_value_map_match_rate
            )
        elif spec.dtype in NUMERIC_DTYPES:
            # No value_map on a numeric column means the source is already numeric.
            # Coerce here so 1, 1.0, "1" and "1.0" are all accepted identically, and
            # report anything that will not parse rather than leaving it to surface as
            # a mysterious drop in the validator.
            series = _coerce_numeric(series, column)

        if spec.dtype == "datetime":
            series = pd.to_datetime(
                series, format=mapping.datetime_format, errors="coerce"
            )
        out[column] = series

    fallback = column_map.duration_fallback
    if fallback.enabled:
        out = _apply_duration_fallback(out, raw, fallback)
        missing_sources = [
            entry
            for entry in missing_sources
            if not entry.startswith(("seen_time", "departure_time"))
        ]

    if column_map.proxy_wait.is_active:
        out = _apply_proxy_wait(out, raw, column_map.proxy_wait)
        missing_sources = [
            entry for entry in missing_sources if not entry.startswith("seen_time")
        ]

    unresolved_required = [
        entry for entry in missing_sources if entry.split(" ")[0] in schema.required_columns
    ]
    if unresolved_required:
        raise ValueError(
            "These required schema columns could not be sourced from the raw file: "
            + "; ".join(unresolved_required)
            + f". Raw columns available: {sorted(raw.columns)}. "
            "Fix the `source` entries in configs/column_map.yaml."
        )
    unresolved_optional = [entry for entry in missing_sources if entry not in unresolved_required]

    # A column that is "not sourced" while the raw file contains one of that exact name
    # is a typo in column_map.yaml, not a dataset that lacks the column. Degrading
    # gracefully there would quietly drop a feature the data actually has.
    if column_map.validation.fail_on_unsourced_present_column:
        recoverable = [
            name
            for name in schema.input_columns
            if name not in out.columns and name in set(raw.columns)
        ]
        if recoverable:
            raise ValueError(
                "These schema columns were not sourced, but the raw file contains a "
                f"column of the same name: {recoverable}. That is a mapping error, not "
                "a missing column. Add or fix their entries in configs/column_map.yaml "
                "(an identity mapping is `name: {source: name}`), or set "
                "validation.fail_on_unsourced_present_column: false to accept it."
            )

    if unresolved_optional:
        logger.info(
            "Optional columns not sourced (pipeline degrades gracefully): %s",
            "; ".join(unresolved_optional),
        )

    unmapped_columns = [c for c in schema.input_columns if c not in out.columns]
    if unmapped_columns:
        logger.info("Schema columns absent from this dataset: %s", unmapped_columns)

    logger.info(
        "Adapted %r to the schema: %d rows, columns %s",
        column_map.dataset_name,
        len(out),
        list(out.columns),
    )
    return out


def _apply_proxy_wait(
    out: pd.DataFrame, raw: pd.DataFrame, proxy: ProxyWait
) -> pd.DataFrame:
    """Fill `seen_time` from a proxy timestamp, loudly.

    This is the one place the project knowingly substitutes a different quantity for
    the one it models. It is allowed because without it MIMIC-IV-ED cannot be used at
    all, and it is noisy because every downstream number inherits the substitution.

    Args:
        out: The schema-shaped frame built so far.
        raw: The raw source frame.
        proxy: The configured proxy.

    Returns:
        `out` with `seen_time` filled from the proxy where it applies.

    Raises:
        ValueError: If the proxy is enabled without naming what it is, or its source
            column is not in the raw file.
    """
    if not proxy.proxy_name:
        raise ValueError(
            "proxy_wait is enabled but proxy_name is not set. The proxy has to be "
            "named in words -- 'first vitalsign charttime', 'first pyxis charttime' "
            "-- because every wait, label and KPI downstream is a measurement of it, "
            "and a reader cannot judge a number produced by an unnamed proxy."
        )
    if proxy.source not in raw.columns:
        raise ValueError(
            f"proxy_wait source {proxy.source!r} is not in the raw file. "
            f"Available columns: {sorted(raw.columns)}."
        )

    series = pd.to_datetime(
        raw[proxy.source], format=proxy.datetime_format, errors="coerce"
    )
    if "seen_time" in out.columns and proxy.only_when_missing:
        filled = out["seen_time"].isna() & series.notna()
        out["seen_time"] = out["seen_time"].where(~filled, series)
    else:
        filled = series.notna()
        out["seen_time"] = series

    logger.warning(
        "%s Proxy used: %s. %d of %d rows (%.1f%%) had seen_time derived from it.",
        PROXY_WAIT_NOTICE,
        proxy.describe(),
        int(filled.sum()),
        len(out),
        100.0 * float(filled.mean()) if len(out) else 0.0,
    )
    return out


def _apply_duration_fallback(
    out: pd.DataFrame, raw: pd.DataFrame, fallback: DurationFallback
) -> pd.DataFrame:
    """Rebuild `seen_time`/`departure_time` from duration columns plus `arrival_time`.

    Some extracts publish door-to-doctor and length-of-stay as numbers instead of
    timestamps. The rest of the pipeline is timestamp-based, so we reconstruct them.
    """
    if "arrival_time" not in out.columns:
        raise ValueError("duration_fallback needs arrival_time to be mapped first")
    factor = fallback.factor

    for target, source in (
        ("seen_time", fallback.wait_minutes_source),
        ("departure_time", fallback.los_minutes_source),
    ):
        if source is None:
            continue
        if source not in raw.columns:
            logger.warning(
                "duration_fallback source %r for %s is not in the raw file; skipping",
                source,
                target,
            )
            continue
        if target in out.columns and out[target].notna().any():
            logger.info("%s is already mapped from a timestamp; ignoring duration fallback", target)
            continue
        minutes = pd.to_numeric(raw[source], errors="coerce") * factor
        negative = minutes.notna() & (minutes < 0)
        if negative.any():
            logger.warning(
                "duration_fallback: %d negative value(s) in %r became missing",
                int(negative.sum()),
                source,
            )
            minutes = minutes.mask(negative)
        out[target] = out["arrival_time"] + pd.to_timedelta(minutes, unit="m")
        logger.info("Reconstructed %s from duration column %r (%s)", target, source, fallback.units)
    return out


def load_real_visits(
    config: dict[str, Any],
    schema: Schema | None = None,
) -> pd.DataFrame:
    """Read and adapt the configured real dataset in one call.

    Args:
        config: Full pipeline config; uses `paths.raw_file` and `paths.column_map`.
        schema: Parsed data contract; loaded from `schema.yaml` when omitted.

    Returns:
        Schema-shaped (but not yet validated) visits DataFrame.
    """
    column_map = load_column_map(config["paths"]["column_map"])
    raw = read_raw(config["paths"]["raw_file"], column_map.read_options)
    return adapt_real_dataset(raw, column_map, schema=schema)
