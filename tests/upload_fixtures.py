"""Builders for the uploaded files the upload tests exercise.

Every awkward extract the mapping and the job engine claim to handle gets a builder
here: a renamed header, a semicolon separator, a byte-order mark, text acuity
labels, durations instead of timestamps. They are plain functions rather than
fixtures so a test can take two of them, or the same one twice.

All of it is derived from `data/raw/ed_v2.csv`, which is **synthetic data calibrated
to published benchmarks, not real patient data**.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from src.utils.io import project_root

#: The synthetic extract every fixture is cut from.
SOURCE = project_root() / "data" / "raw" / "ed_v2.csv"

#: Acuity labels the config's map recognises, by ESI level.
ACUITY_WORDS = {1: "Resuscitation", 2: "Emergent", 3: "Urgent", 4: "Less Urgent", 5: "Non-Urgent"}


def source_available() -> bool:
    """Whether the synthetic extract is on this machine."""
    return SOURCE.is_file()


def load_source() -> pd.DataFrame:
    """The full synthetic extract, with arrival_time parsed."""
    frame = pd.read_csv(SOURCE)
    frame["arrival_time"] = pd.to_datetime(frame["arrival_time"])
    return frame


def slice_days(days: int, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """The first `days` days of the extract.

    Used to build the short extracts that must make the forecast and survival
    stages skip rather than fail.
    """
    frame = load_source() if frame is None else frame
    cutoff = frame["arrival_time"].min() + pd.Timedelta(days=days)
    return frame[frame["arrival_time"] < cutoff].copy()


def to_csv_bytes(
    frame: pd.DataFrame,
    separator: str = ",",
    bom: bool = False,
    encoding: str = "utf-8",
) -> bytes:
    """Serialise a frame the way a real export would.

    Args:
        frame: The visits.
        separator: Field separator.
        bom: Prepend a UTF-8 byte-order mark, as Excel does.
        encoding: Text encoding.

    Returns:
        The file's bytes.
    """
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, sep=separator, date_format="%Y-%m-%d %H:%M:%S")
    text = buffer.getvalue()
    return ("﻿" + text).encode(encoding) if bom else text.encode(encoding)


# --------------------------------------------------------------- awkward headers


def rename_to_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename every column to a documented alias, in a different naming style.

    This is the file that must map itself without asking: no column carries a schema
    name, but each carries a name the alias list knows.
    """
    return frame.rename(
        columns={
            "patient_id": "Encounter ID",
            "arrival_time": "Registration Time",
            "triage_level": "ESI",
            "seen_time": "First Seen",
            "departure_time": "Discharge Time",
            "disposition": "Outcome",
            "age": "Patient Age",
            "sex": "Gender",
            "arrival_mode": "Mode of Arrival",
            "chief_complaint": "Presenting Complaint",
        }
    )


def make_ambiguous(frame: pd.DataFrame) -> pd.DataFrame:
    """Give two columns an equal claim on `arrival_time`.

    `arrival` and `registration_time` are both aliases of `arrival_time`, so neither
    may be chosen. The job must stop and ask.
    """
    out = frame.rename(columns={"arrival_time": "arrival"})
    out["registration_time"] = out["arrival"]
    return out


def drop_required(frame: pd.DataFrame, column: str = "triage_level") -> pd.DataFrame:
    """Remove a required column entirely."""
    return frame.drop(columns=[column])


def text_acuity(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace numeric acuity with the text labels the config knows."""
    out = frame.copy()
    out["triage_level"] = out["triage_level"].map(ACUITY_WORDS).fillna("Urgent")
    return out


def unknown_acuity_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Acuity as labels the config does *not* know, so the map matches almost nothing."""
    out = frame.copy()
    out["triage_level"] = "Kategorie " + out["triage_level"].astype(str)
    return out


def coded_dispositions(frame: pd.DataFrame) -> pd.DataFrame:
    """Dispositions written the way another hospital's export would write them."""
    out = frame.copy()
    out["disposition"] = out["disposition"].map(
        {
            "discharged": "Home",
            "admitted": "Inpatient",
            "transferred": "Transfer",
            "lwbs": "Did Not Wait",
            "other": "Deceased",
        }
    ).fillna("Home")
    return out


def durations_instead_of_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    """An extract that stores waits and length of stay rather than timestamps."""
    out = frame.copy()
    seen = pd.to_datetime(out["seen_time"], errors="coerce")
    departed = pd.to_datetime(out["departure_time"], errors="coerce")
    out["DoorToDoc_Min"] = (seen - out["arrival_time"]).dt.total_seconds() / 60
    out["LOS_Min"] = (departed - out["arrival_time"]).dt.total_seconds() / 60
    return out.drop(columns=["seen_time", "departure_time"])


def shuffle_rows(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Return the same visits in a scrambled order.

    The pipeline sorts by arrival time, so a shuffled file must produce identical
    results. `sample(frac=...)` is forbidden in `src/` by a project invariant, but
    this is a test fixture deliberately destroying order, not a data split.
    """
    rng = __import__("numpy").random.default_rng(seed)
    order = rng.permutation(len(frame))
    return frame.iloc[order].reset_index(drop=True)


#: How `decouple_waits_from_queue` breaks the link between queue and wait.
#:
#: "within_acuity" permutes waits inside each ESI level, which preserves every
#: marginal the validation gate checks -- and therefore PASSES it. That is a measured
#: finding about the gate, not a bug in this builder: see
#: `tests/test_validation_gate.py`.
#:
#: "across_acuity" permutes waits across the whole department, which additionally
#: destroys the acuity ordering of waits and does trip the gate.
DECOUPLE_SCOPES = ("within_acuity", "across_acuity")


def decouple_waits_from_queue(
    frame: pd.DataFrame, seed: int = 20240501, scope: str = "across_acuity"
) -> pd.DataFrame:
    """Break the link between how busy the department was and how long people waited.

    **What this is for.** The simulation validation gate exists to stop the project
    reporting scenario and staffing results from a model that cannot reproduce the
    department. A gate is only worth having if it fires, and on every honest dataset
    in this repository it passes -- which is no evidence that it would ever catch
    anything. This builder produces a dataset the gate must reject.

    **How it is broken.** Each patient's wait is replaced with another patient's wait,
    so the dependence between a patient's wait and the queue they actually arrived
    into is destroyed. No queueing model can reproduce that, because in a queueing
    model the wait *is* a function of the queue. Post-contact time travels with the
    patient rather than with the slot, so length of stay stays attached to a plausible
    visit, and patients who were never seen are left untouched -- they have no wait to
    permute, and inventing one would move the abandonment rate, which is one of the
    metrics the gate is being tested on.

    **The two scopes, and why both exist.**

    ``within_acuity`` permutes only inside an ESI level. It is the obvious way to
    decouple waits, and it does not work: it preserves the per-acuity mean, median and
    p90 wait, the length-of-stay distribution and the abandonment rate *exactly*, and
    those marginals are the only things the gate inspects. Measured on
    `data/raw/ed_v2.csv`, the gate passes all 12 held-out checks on this data. It is
    kept because that is a real limitation of the validation design and the test suite
    should state it rather than leave it to be rediscovered.

    ``across_acuity`` permutes over the whole department, which also flattens the
    acuity ordering of waits -- every ESI level ends up with the same mean wait. A
    priority queue cannot produce that, and the gate fails it on the per-acuity wait
    checks.

    Args:
        frame: Visits with `arrival_time`, `seen_time`, `departure_time` and
            `triage_level`.
        seed: Seed for the permutation, so the fixture is reproducible.
        scope: One of `DECOUPLE_SCOPES`.

    Returns:
        A copy with `seen_time` and `departure_time` rewritten.

    Raises:
        ValueError: If `scope` is not one of `DECOUPLE_SCOPES`.
    """
    import numpy as np

    if scope not in DECOUPLE_SCOPES:
        raise ValueError(f"scope must be one of {DECOUPLE_SCOPES}, got {scope!r}")

    out = frame.copy()
    for column in ("arrival_time", "seen_time", "departure_time"):
        out[column] = pd.to_datetime(out[column], errors="coerce")

    wait = (out["seen_time"] - out["arrival_time"]).dt.total_seconds()
    post_contact = (out["departure_time"] - out["seen_time"]).dt.total_seconds()
    observed = wait.notna().to_numpy()

    rng = np.random.default_rng(seed)
    shuffled = wait.to_numpy(dtype="float64").copy()

    if scope == "within_acuity":
        groups = [
            (out["triage_level"] == level).to_numpy() & observed
            for level in sorted(out["triage_level"].dropna().unique())
        ]
    else:
        groups = [observed]

    for mask in groups:
        count = int(mask.sum())
        if count < 2:
            continue
        values = shuffled[mask]
        shuffled[mask] = values[rng.permutation(count)]

    seen = out["arrival_time"] + pd.to_timedelta(shuffled, unit="s")
    out["seen_time"] = seen.where(wait.notna())
    out["departure_time"] = (
        out["seen_time"] + pd.to_timedelta(post_contact, unit="s")
    ).where(post_contact.notna())
    return out


# ----------------------------------------------------------------- broken files


def empty_file() -> bytes:
    """Zero bytes."""
    return b""


def header_only(frame: pd.DataFrame) -> bytes:
    """A header row and nothing else."""
    return to_csv_bytes(frame.head(0))


def oversize(limit_bytes: int) -> bytes:
    """A file just over the upload limit, cheap to build and never parsed."""
    header = b"patient_id,arrival_time,triage_level,seen_time,departure_time,disposition\n"
    filler = b"x" * (limit_bytes + 1024 - len(header))
    return header + filler


def write(path: Path, content: bytes) -> Path:
    """Write bytes to a path and return it."""
    path.write_bytes(content)
    return path
