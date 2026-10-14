"""Work out how an uploaded file maps onto the schema -- automatically, never silently.

The rule this module exists to enforce: **a required column is never guessed.** A
wrong guess there is not a visible error, it is a plausible-looking model trained on
the wrong field, so detection is deliberately narrow. Three steps, in order, and
nothing after them:

1. the schema's own column name, exactly;
2. the same name ignoring case, spaces, hyphens and underscores (`Arrival Time`,
   `ARRIVAL_TIME` and `arrivaltime` are all `arrival_time`);
3. a documented alias from `uploads.column_aliases` in the config.

No edit distance, no substring scoring, no "closest match". If two source columns
compete for one schema column, or none matches, that column is **ambiguous** or
**missing** and the job stops to ask. Optional columns that cannot be found are
simply absent -- the pipeline already runs without them.

Value translation is equally explicit. Numeric acuity is coerced (`1`, `1.0`, `"1"`
are the same patient); textual acuity is translated only through the configured
label map; dispositions go through their own map with every unmapped value counted
and shown. A map that matches almost nothing is the failure mode that looks like a
data problem, so a match rate under the configured floor is a loud warning.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.schema import Schema
from src.jobs.settings import UploadSettings
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Bytes a UTF-8 file with a byte-order mark starts with. Excel writes these, and a
#: BOM read as data turns the first header into "﻿patient_id", which then fails
#: to match anything -- a confusing way to be told the file is fine.
UTF8_BOM = b"\xef\xbb\xbf"

#: How much of the file to read when sniffing the delimiter.
SNIFF_BYTES = 64 * 1024

#: Rows shown in the mapping preview.
PREVIEW_ROWS = 5

#: How a column was matched, in decreasing order of confidence.
MATCH_EXACT = "exact"
MATCH_NORMALISED = "normalised"
MATCH_ALIAS = "alias"
MATCH_AMBIGUOUS = "ambiguous"
MATCH_MISSING = "missing"
MATCH_MANUAL = "manual"

#: Matches that are good enough to proceed on without asking.
CONFIDENT_MATCHES = frozenset({MATCH_EXACT, MATCH_NORMALISED, MATCH_ALIAS, MATCH_MANUAL})

#: Required columns the duration fallback reconstructs from `arrival_time`. An
#: extract that stores waits in minutes has no column for either, so without this
#: they would be reported missing forever and the fallback could never be used.
DURATION_RECONSTRUCTS = ("seen_time", "departure_time")


class MappingError(ValueError):
    """The file cannot be read or mapped, with a message written for a person."""


def normalise(name: str) -> str:
    """Reduce a column name to its comparable form.

    Case, spaces, hyphens, dots and underscores carry no meaning in a column name, so
    they are removed before comparison: `Arrival Time`, `arrival-time` and
    `ARRIVAL_TIME` are one name.
    """
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


# ------------------------------------------------------------------- file dialect


@dataclass(frozen=True)
class FileDialect:
    """How to read the uploaded text file."""

    delimiter: str
    encoding: str
    has_bom: bool
    detected: bool
    note: str

    @property
    def delimiter_name(self) -> str:
        """The delimiter written the way a person would say it."""
        return {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}.get(
            self.delimiter, repr(self.delimiter)
        )

    def read_options(self) -> dict[str, Any]:
        """Keyword arguments for `pandas.read_csv`.

        `utf-8-sig` strips a byte-order mark if present and is harmless when absent,
        so it is always safe here.
        """
        return {"sep": self.delimiter, "encoding": self.encoding}


def sniff_dialect(path: str | Path, settings: UploadSettings) -> FileDialect:
    """Work out the delimiter and encoding of an uploaded file.

    `csv.Sniffer` is tried first and trusted only when it returns one of the
    configured delimiters. Otherwise the header line is scored: the delimiter that
    splits it into the most fields wins, which is reliable for a header row and does
    not depend on the data below it.

    Args:
        path: The uploaded file.
        settings: Upload settings, for the candidate delimiter list.

    Returns:
        The dialect to read the file with.

    Raises:
        MappingError: If the file is empty or cannot be decoded as text.
    """
    raw = Path(path).read_bytes()[:SNIFF_BYTES]
    if not raw.strip():
        raise MappingError(
            "The file is empty. Upload a CSV with a header row and at least one row of data."
        )

    has_bom = raw.startswith(UTF8_BOM)
    encoding = "utf-8-sig"
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError:
        # Latin-1 decodes any byte sequence, so this is a last resort that always
        # succeeds; the warning tells the user the file was not UTF-8.
        encoding = "latin-1"
        text = raw.decode(encoding, errors="replace")
        logger.warning("Upload is not valid UTF-8; read as latin-1")

    header = next((line for line in text.splitlines() if line.strip()), "")
    candidates = list(settings.delimiters)

    try:
        sniffed = csv.Sniffer().sniff(text[:SNIFF_BYTES], delimiters="".join(candidates))
        if sniffed.delimiter in candidates and len(header.split(sniffed.delimiter)) > 1:
            return FileDialect(
                delimiter=sniffed.delimiter,
                encoding=encoding,
                has_bom=has_bom,
                detected=True,
                note=f"Delimiter detected as {_name(sniffed.delimiter)}.",
            )
    except csv.Error:
        logger.info("csv.Sniffer could not determine the delimiter; scoring the header")

    counts = {delimiter: len(header.split(delimiter)) for delimiter in candidates}
    best = max(counts, key=lambda d: counts[d])
    if counts[best] < 2:
        raise MappingError(
            "Could not find a column separator in the first line. Supported separators "
            f"are {', '.join(_name(d) for d in candidates)}. The first line read was: "
            f"{header[:120]!r}"
        )
    return FileDialect(
        delimiter=best,
        encoding=encoding,
        has_bom=has_bom,
        detected=True,
        note=f"Delimiter detected as {_name(best)} ({counts[best]} columns in the header).",
    )


def _name(delimiter: str) -> str:
    """Human-readable name for a delimiter character."""
    return {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}.get(
        delimiter, repr(delimiter)
    )


def read_header(path: str | Path, dialect: FileDialect) -> list[str]:
    """Return the file's column names.

    Raises:
        MappingError: If the file has no header row.
    """
    try:
        frame = pd.read_csv(path, nrows=0, **dialect.read_options())
    except pd.errors.EmptyDataError as error:
        raise MappingError(
            "The file has no header row. The first line must name the columns."
        ) from error
    columns = [str(column).strip() for column in frame.columns]
    if not columns:
        raise MappingError("The file has no columns.")
    return columns


def read_preview(path: str | Path, dialect: FileDialect, rows: int = PREVIEW_ROWS):
    """Read the first few rows, for showing the user what was parsed."""
    return pd.read_csv(path, nrows=rows, dtype=str, **dialect.read_options())


def count_rows(path: str | Path, dialect: FileDialect) -> int:
    """Count data rows without holding the file in memory.

    Counts newlines rather than parsing, so a very large file is rejected by
    `max_rows` before pandas is ever asked to materialise it.
    """
    total = 0
    with open(path, "rb") as handle:
        for _ in handle:
            total += 1
    header_rows = 1
    return max(total - header_rows, 0)


# --------------------------------------------------------------- column detection


@dataclass(frozen=True)
class DetectedColumn:
    """How one schema column was located in the uploaded file."""

    schema_column: str
    source: str | None
    how: str
    candidates: tuple[str, ...] = ()
    required: bool = False

    @property
    def confident(self) -> bool:
        """Whether this match is good enough to proceed on without asking."""
        return self.how in CONFIDENT_MATCHES and self.source is not None

    def explain(self) -> str:
        """One line a person can read, saying how this column was decided."""
        if self.how == MATCH_EXACT:
            return f"{self.schema_column} <- {self.source} (exact name)"
        if self.how == MATCH_NORMALISED:
            return f"{self.schema_column} <- {self.source} (same name, different style)"
        if self.how == MATCH_ALIAS:
            return f"{self.schema_column} <- {self.source} (known alias)"
        if self.how == MATCH_MANUAL:
            return f"{self.schema_column} <- {self.source} (you chose this)"
        if self.how == MATCH_AMBIGUOUS:
            return (
                f"{self.schema_column}: more than one column could be it "
                f"({', '.join(self.candidates)}). Choose one."
            )
        return f"{self.schema_column}: no matching column found."


def detect_columns(
    source_columns: list[str], schema: Schema, settings: UploadSettings
) -> dict[str, DetectedColumn]:
    """Match each schema column to a column of the uploaded file.

    Args:
        source_columns: The file's column names.
        schema: The data contract.
        settings: Upload settings, for the alias list.

    Returns:
        One `DetectedColumn` per schema input column, in schema order.
    """
    by_normalised: dict[str, list[str]] = {}
    for column in source_columns:
        by_normalised.setdefault(normalise(column), []).append(column)

    claimed: set[str] = set()
    detected: dict[str, DetectedColumn] = {}

    # Exact and normalised matches are resolved for every schema column first, so an
    # alias can never steal a column that carries a schema name outright.
    for schema_column in schema.input_columns:
        required = schema_column in schema.required_columns
        if schema_column in source_columns:
            detected[schema_column] = DetectedColumn(
                schema_column, schema_column, MATCH_EXACT, required=required
            )
            claimed.add(schema_column)
            continue
        matches = by_normalised.get(normalise(schema_column), [])
        unclaimed = [m for m in matches if m not in claimed]
        if len(unclaimed) == 1:
            detected[schema_column] = DetectedColumn(
                schema_column, unclaimed[0], MATCH_NORMALISED, required=required
            )
            claimed.add(unclaimed[0])
        elif len(unclaimed) > 1:
            detected[schema_column] = DetectedColumn(
                schema_column, None, MATCH_AMBIGUOUS, tuple(unclaimed), required=required
            )

    for schema_column in schema.input_columns:
        if schema_column in detected:
            continue
        required = schema_column in schema.required_columns
        hits: list[str] = []
        for alias in settings.column_aliases.get(schema_column, ()):
            for candidate in by_normalised.get(normalise(alias), []):
                if candidate not in claimed and candidate not in hits:
                    hits.append(candidate)
        if len(hits) == 1:
            detected[schema_column] = DetectedColumn(
                schema_column, hits[0], MATCH_ALIAS, required=required
            )
            claimed.add(hits[0])
        elif len(hits) > 1:
            detected[schema_column] = DetectedColumn(
                schema_column, None, MATCH_AMBIGUOUS, tuple(hits), required=required
            )
        else:
            detected[schema_column] = DetectedColumn(
                schema_column, None, MATCH_MISSING, required=required
            )

    return {column: detected[column] for column in schema.input_columns}


# ------------------------------------------------------------------- the proposal


@dataclass
class MappingProposal:
    """What the file looks like, and whether the job may start without asking."""

    dialect: FileDialect
    source_columns: list[str]
    columns: dict[str, DetectedColumn]
    n_rows: int
    preview: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: True when the caller has said this extract stores durations rather than
    #: `seen_time` and `departure_time`, which the adapter then rebuilds from
    #: `arrival_time`.
    duration_fallback: bool = False

    @property
    def _reconstructed(self) -> frozenset[str]:
        """Required columns that do not need a source, because they are derived."""
        return frozenset(DURATION_RECONSTRUCTS) if self.duration_fallback else frozenset()

    @property
    def missing_required(self) -> list[str]:
        """Required schema columns with no match at all."""
        return [
            name
            for name, column in self.columns.items()
            if column.required
            and column.how == MATCH_MISSING
            and name not in self._reconstructed
        ]

    @property
    def ambiguous_required(self) -> list[str]:
        """Required schema columns where more than one source column could be it."""
        return [
            name
            for name, column in self.columns.items()
            if column.required
            and column.how == MATCH_AMBIGUOUS
            and name not in self._reconstructed
        ]

    @property
    def ready(self) -> bool:
        """True when every required column is confidently matched.

        The gate for starting a job without a human confirming the mapping.
        """
        return not self.missing_required and not self.ambiguous_required

    @property
    def matched_optional(self) -> list[str]:
        """Optional columns that were found, and so will be used."""
        return [
            name
            for name, column in self.columns.items()
            if not column.required and column.confident
        ]

    def mapping_table(self) -> list[dict[str, Any]]:
        """The mapping as rows, for display in a table."""
        return [
            {
                "schema_column": column.schema_column,
                "source_column": column.source,
                "required": column.required,
                "how": column.how,
                "explanation": column.explain(),
            }
            for column in self.columns.values()
        ]

    def blocking_reason(self) -> str | None:
        """Why the job cannot start automatically, in plain English."""
        if self.ready:
            return None
        parts: list[str] = []
        if self.missing_required:
            parts.append(
                "no column could be matched to " + ", ".join(self.missing_required)
            )
        if self.ambiguous_required:
            detail = "; ".join(
                f"{name} could be {' or '.join(self.columns[name].candidates)}"
                for name in self.ambiguous_required
            )
            parts.append(f"more than one column could be used for: {detail}")
        return (
            "The mapping needs confirming because " + " and ".join(parts) + "."
            " Choose the right column for each and start again."
        )


def propose_mapping(
    path: str | Path,
    schema: Schema,
    settings: UploadSettings,
    duration_fallback: bool = False,
) -> MappingProposal:
    """Inspect an uploaded file and propose how to read it.

    Args:
        path: The uploaded file.
        schema: The data contract.
        settings: Upload settings.
        duration_fallback: True when the caller has said this extract stores
            durations instead of `seen_time`/`departure_time`. Those two columns are
            then not required to be present, because the adapter rebuilds them.

    Returns:
        A `MappingProposal`. Check `.ready` before starting a job from it.

    Raises:
        MappingError: If the file cannot be read at all, with a message for a person.
    """
    dialect = sniff_dialect(path, settings)
    source_columns = read_header(path, dialect)
    n_rows = count_rows(path, dialect)

    if n_rows == 0:
        raise MappingError(
            "The file has a header row but no data rows. Upload an extract with visits in it."
        )

    columns = detect_columns(source_columns, schema, settings)
    preview_frame = read_preview(path, dialect)

    warnings: list[str] = []
    if dialect.has_bom:
        warnings.append(
            "The file starts with a UTF-8 byte-order mark (Excel adds this). It was "
            "stripped so the first column name reads correctly."
        )
    if dialect.delimiter != ",":
        warnings.append(
            f"Read as {dialect.delimiter_name}-separated, not comma-separated."
        )
    if duration_fallback:
        warnings.append(
            "Reading this extract as durations: seen_time and departure_time will be "
            "reconstructed from arrival_time plus the minute columns you chose."
        )
    missing_optional = [
        name
        for name, column in columns.items()
        if not column.required and not column.confident
    ]
    if missing_optional:
        warnings.append(
            "These optional columns were not found and will simply be absent: "
            + ", ".join(missing_optional)
            + ". The pipeline runs without them; fairness reports on what exists."
        )

    return MappingProposal(
        dialect=dialect,
        source_columns=source_columns,
        columns=columns,
        n_rows=n_rows,
        preview=preview_frame.fillna("").to_dict(orient="records"),
        warnings=warnings,
        duration_fallback=duration_fallback,
    )


def apply_overrides(
    proposal: MappingProposal, overrides: dict[str, str | None]
) -> MappingProposal:
    """Return a proposal with user-chosen source columns applied.

    Args:
        proposal: The automatic proposal.
        overrides: `{schema_column: source_column}` from the mapping form. A value of
            None or "" clears an optional column.

    Returns:
        A new proposal reflecting the choices.

    Raises:
        MappingError: If a chosen source column is not in the file, or a required
            column is cleared.
    """
    columns = dict(proposal.columns)
    for schema_column, source in overrides.items():
        if schema_column not in columns:
            raise MappingError(f"{schema_column!r} is not a column of the data contract.")
        current = columns[schema_column]
        if source in (None, ""):
            reconstructed = (
                proposal.duration_fallback and schema_column in DURATION_RECONSTRUCTS
            )
            if current.required and not reconstructed:
                raise MappingError(
                    f"{schema_column} is required and cannot be left unmapped."
                )
            columns[schema_column] = DetectedColumn(
                schema_column, None, MATCH_MISSING, required=current.required
            )
            continue
        if source not in proposal.source_columns:
            raise MappingError(
                f"{source!r} is not a column of the uploaded file. Available: "
                + ", ".join(proposal.source_columns)
            )
        columns[schema_column] = DetectedColumn(
            schema_column, source, MATCH_MANUAL, required=current.required
        )

    return MappingProposal(
        dialect=proposal.dialect,
        source_columns=proposal.source_columns,
        columns=columns,
        n_rows=proposal.n_rows,
        preview=proposal.preview,
        warnings=list(proposal.warnings),
        duration_fallback=proposal.duration_fallback,
    )


# -------------------------------------------------------- building the column map


def looks_textual(values: pd.Series) -> bool:
    """Whether an acuity column holds labels rather than numbers.

    Decided on the data, not the dtype: a column of `"1"` strings is numeric acuity,
    a column of `"Urgent"` is not.
    """
    sample = values.dropna().astype(str).str.strip()
    if sample.empty:
        return False
    numeric = pd.to_numeric(sample, errors="coerce")
    return bool(numeric.isna().mean() > 0.5)


@dataclass
class ValueMapReport:
    """How well a configured value map fitted the column it was applied to."""

    column: str
    matched: int
    total: int
    unmapped_values: dict[str, int]

    @property
    def match_rate(self) -> float:
        """Share of non-null values the map recognised."""
        return self.matched / self.total if self.total else 1.0


def summarise_value_map(
    values: pd.Series, label_map: dict[str, Any], column: str, top_n: int = 10
) -> ValueMapReport:
    """Count how many values a label map recognises, and which it does not.

    Matching ignores case and surrounding whitespace, exactly as the adapter does.
    """
    sample = values.dropna().astype(str).str.strip()
    lowered = sample.str.lower()
    known = set(label_map)
    recognised = lowered.isin(known)
    unmapped = sample[~recognised].value_counts().head(top_n)
    return ValueMapReport(
        column=column,
        matched=int(recognised.sum()),
        total=int(len(sample)),
        unmapped_values={str(k): int(v) for k, v in unmapped.items()},
    )


def build_column_map(
    proposal: MappingProposal,
    settings: UploadSettings,
    sample: pd.DataFrame,
    duration_fallback: dict[str, Any] | None = None,
    datetime_format: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Build the `column_map.yaml` content for this upload.

    Args:
        proposal: The confirmed mapping.
        settings: Upload settings, for the label maps.
        sample: A sample of the file, used to decide whether acuity is textual and to
            count unmapped disposition values.
        duration_fallback: Optional `{enabled, wait_minutes_source, los_minutes_source,
            units}` for a file that stores durations instead of timestamps.
        datetime_format: Explicit timestamp format, or None to let pandas infer it.

    Returns:
        `(column_map_dict, warnings)`. The dict is what `ColumnMap` parses.

    Raises:
        MappingError: If a required column is still unmapped.
    """
    if not proposal.ready:
        raise MappingError(proposal.blocking_reason() or "The mapping is incomplete.")

    warnings: list[str] = []
    columns: dict[str, dict[str, Any]] = {}

    for schema_column, column in proposal.columns.items():
        if not column.confident or column.source is None:
            continue
        entry: dict[str, Any] = {"source": column.source}

        if schema_column.endswith("_time"):
            if datetime_format:
                entry["datetime_format"] = datetime_format
            else:
                warnings.append(
                    f"No timestamp format was given for {schema_column}, so it is "
                    "inferred from the values. If your dates are day-first, state the "
                    "format to avoid a day/month mix-up."
                )

        if schema_column == "triage_level" and column.source in sample.columns:
            if looks_textual(sample[column.source]):
                report = summarise_value_map(
                    sample[column.source], settings.acuity_labels, column.source
                )
                entry["value_map"] = dict(settings.acuity_labels)
                warnings.extend(_map_warnings(report, "acuity", settings))
            # Numeric acuity needs no map: the adapter coerces 1, 1.0 and "1" alike.

        if schema_column == "disposition" and column.source in sample.columns:
            report = summarise_value_map(
                sample[column.source], settings.disposition_labels, column.source
            )
            entry["value_map"] = dict(settings.disposition_labels)
            entry["default"] = "other"
            warnings.extend(_map_warnings(report, "disposition", settings))

        if schema_column == "sex" and column.source in sample.columns:
            entry["value_map"] = {
                "f": "female", "m": "male", "female": "female", "male": "male",
                "w": "female", "other": "other", "unknown": "unknown",
            }
            entry["default"] = "unknown"

        if schema_column == "arrival_mode" and column.source in sample.columns:
            entry["value_map"] = {
                "ambulance": "ambulance", "ems": "ambulance", "emergency": "ambulance",
                "walk_in": "walk_in", "walk-in": "walk_in", "walkin": "walk_in",
                "self": "walk_in", "other": "other",
            }
            entry["default"] = "other"

        columns[schema_column] = entry

    fallback = duration_fallback or {"enabled": False}
    if fallback.get("enabled"):
        wait_source = fallback.get("wait_minutes_source")
        if not wait_source:
            raise MappingError(
                "Reading durations needs a waiting-time column in minutes. Choose one, "
                "or map seen_time to a timestamp column instead."
            )
        for key in ("wait_minutes_source", "los_minutes_source"):
            chosen = fallback.get(key)
            if chosen and chosen not in proposal.source_columns:
                raise MappingError(
                    f"{chosen!r} is not a column of the uploaded file. Available: "
                    + ", ".join(proposal.source_columns)
                )
        # Durations replace the timestamps they reconstruct, so those mappings must go
        # or the adapter would prefer a column the user said is not there.
        for timestamp_column in ("seen_time", "departure_time"):
            columns.pop(timestamp_column, None)
        warnings.append(
            "Using durations to reconstruct seen_time and departure_time from "
            "arrival_time. Any wait or length-of-stay column you mapped directly is "
            "ignored in favour of this."
        )

    column_map = {
        "dataset_name": "uploaded_extract",
        "read_options": {
            **proposal.dialect.read_options(),
            "na_values": ["", "NA", "N/A", "NULL", "null", "-", "unknown", "Unknown", "nan"],
        },
        "columns": columns,
        "duration_fallback": {
            "enabled": bool(fallback.get("enabled", False)),
            "wait_minutes_source": fallback.get("wait_minutes_source"),
            "los_minutes_source": fallback.get("los_minutes_source"),
            "units": fallback.get("units", "minutes"),
        },
        "validation": {
            "min_value_map_match_rate": settings.min_value_map_match_rate,
            # The uploaded file's columns were matched by this module, not written by
            # hand, so an unsourced-but-present column is a deliberate user choice
            # rather than the typo this guard was built to catch.
            "fail_on_unsourced_present_column": False,
        },
    }
    return column_map, warnings


def _map_warnings(
    report: ValueMapReport, label: str, settings: UploadSettings
) -> list[str]:
    """Warnings for a value map that fitted badly or left values behind."""
    warnings: list[str] = []
    if report.match_rate < settings.min_value_map_match_rate:
        warnings.append(
            f"WARNING: the {label} map matched only {report.match_rate:.1%} of values in "
            f"{report.column!r}. That almost always means the map is for a different "
            "extract, and the column will become nearly constant. Check the values: "
            f"{list(report.unmapped_values)[:5]}"
        )
    elif report.unmapped_values:
        total_unmapped = sum(report.unmapped_values.values())
        warnings.append(
            f"{total_unmapped} {label} value(s) were not recognised and fall through to "
            f"the default: {report.unmapped_values}"
        )
    return warnings
