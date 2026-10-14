"""Reading the pipeline's saved tables, for the pages the API does not serve.

Model quality, forecast reliability and the fairness breakdown describe how the
models were *evaluated*. They are properties of a completed pipeline run, not of
the department right now, so they come from the tables the pipeline wrote rather
than from a request. The pages that use them say so, because a reader otherwise
reasonably assumes every number on a dashboard is current.

The dashboard reads saved tables and metrics only. It never opens a model file:
loading a model into the dashboard process would make the page slow to start and
would duplicate the service's job.

Everything here returns `None` when the file is absent, so a dashboard pointed at
a fresh checkout shows an empty state with instructions instead of a traceback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.config import ARTIFACT_CACHE_TTL_SECONDS, ARTIFACTS_DIR


def artifacts_root() -> Path:
    """The artifacts directory, as configured."""
    return Path(ARTIFACTS_DIR)


def exists() -> bool:
    """Whether there is a completed pipeline run to read at all."""
    return (artifacts_root() / "metrics" / "wait_models.json").is_file()


@st.cache_data(ttl=ARTIFACT_CACHE_TTL_SECONDS, show_spinner=False)
def table(name: str, comment: str | None = None) -> pd.DataFrame | None:
    """One CSV from `artifacts/tables/`, or None if it is not there.

    Args:
        name: File name, e.g. `fairness.csv`.
        comment: Comment character, for the tables that carry a leading caveat
            line (`ablation.csv` does).
    """
    path = artifacts_root() / "tables" / name
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path, comment=comment) if comment else pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return None


@st.cache_data(ttl=ARTIFACT_CACHE_TTL_SECONDS, show_spinner=False)
def metrics(name: str) -> dict[str, Any] | None:
    """One JSON document from `artifacts/metrics/`, or None if it is not there."""
    path = artifacts_root() / "metrics" / name
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


# ----------------------------------------------------------- the derived readings


def wait_model_quality() -> dict[str, Any] | None:
    """The headline wait-model numbers, already picked out of the metrics file.

    Returns a dict of the few values the pages actually show, so no page has to
    know the shape of `wait_models.json`:

        model_mae, baseline_mae, improvement, coverage, nominal_coverage, n_test
    """
    document = metrics("wait_models.json")
    if not document:
        return None

    best = (document.get("ablation") or {}).get("best_by_test_mae") or {}
    baseline_test = (
        ((document.get("baseline") or {}).get("blocks") or {}).get("test") or {}
    ).get("regression") or {}
    conformal_overall = (document.get("conformal") or {}).get("overall") or {}

    model_mae = _number(best.get("mae_minutes"))
    baseline_mae = _number(baseline_test.get("mae"))
    improvement = None
    if model_mae is not None and baseline_mae not in (None, 0):
        improvement = (baseline_mae - model_mae) / baseline_mae

    return {
        "feature_set": best.get("feature_set"),
        "model_mae": model_mae,
        "baseline_mae": baseline_mae,
        "improvement": improvement,
        # `wait_models.json` calls it `coverage`; the CSV calls the same quantity
        # `empirical_coverage`. Both names are accepted so a rename in either
        # place shows up as a missing number rather than a silently wrong one.
        "coverage": _number(
            conformal_overall.get("coverage",
                                  conformal_overall.get("empirical_coverage"))
        ),
        "nominal_coverage": _number(conformal_overall.get("nominal_coverage")),
        "mean_width": _number(conformal_overall.get("mean_width")),
        "n_test": _number(baseline_test.get("n")),
        "ablation_caveat": (document.get("ablation") or {}).get("caveat"),
    }


def forecast_quality() -> dict[str, Any] | None:
    """Forecast error against the weekly-repeat baseline, overall and at peak.

    The peak comparison is the honest one and it is not flattering, so it is
    carried through to the page rather than averaged away.
    """
    frame = table("forecast_accuracy.csv")
    if frame is None or frame.empty:
        return None

    def pick(segment: str, model: str) -> float | None:
        rows = frame.loc[(frame["segment"] == segment) & (frame["model"] == model)]
        return _number(rows["mae"].iloc[0]) if not rows.empty else None

    overall_model = pick("overall", "lightgbm")
    overall_naive = pick("overall", "seasonal_naive")
    peak_model = pick("peak_hours", "lightgbm")
    peak_naive = pick("peak_hours", "seasonal_naive")

    beats_overall = (
        overall_model is not None
        and overall_naive is not None
        and overall_model < overall_naive
    )
    beats_peak = (
        peak_model is not None and peak_naive is not None and peak_model < peak_naive
    )
    return {
        "overall_model_mae": overall_model,
        "overall_naive_mae": overall_naive,
        "peak_model_mae": peak_model,
        "peak_naive_mae": peak_naive,
        "beats_baseline_overall": beats_overall,
        "beats_baseline_at_peak": beats_peak,
    }


def coverage_by_triage() -> pd.DataFrame | None:
    """Per-acuity interval coverage, for the "how often is the range right" table."""
    return table("conformal_coverage.csv")


def fairness_table() -> pd.DataFrame | None:
    """Subgroup breakdown, as the fairness stage wrote it."""
    return table("fairness.csv")


def triage_thresholds() -> pd.DataFrame | None:
    """Per-acuity decision thresholds, including the ones that could not be fitted."""
    return table("triage_thresholds.csv")


def data_quality_summary() -> pd.DataFrame | None:
    """How many rows survived validation, and why the rest did not."""
    return table("validation_summary.csv")


def split_bounds() -> dict[str, Any] | None:
    """The time split, so Shift replay knows where the held-out period starts."""
    return metrics("split.json")


def test_period_start() -> str | None:
    """First instant of the held-out test block, if the pipeline recorded it."""
    document = split_bounds()
    if not document:
        return None
    test = document.get("test") or {}
    start = test.get("start")
    return str(start) if start else None


def _number(value: Any) -> float | None:
    """A float, or None for anything that is not one -- including NaN."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number
