"""Tests for the dataset-shift warning.

The check exists so that nobody can point this pipeline at a new dataset without being
told that the models were tuned on a different one. Two things must hold: it must not
cry wolf on the reference dataset itself, and it must fire when something that would
change a modelling decision has changed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.shift_check import (
    RETRAIN_NOTICE,
    THRESHOLDS,
    ShiftReport,
    compare_to_reference,
    profile_dataset,
    reference_profile,
)
from src.utils.io import read_visits, resolve_path

BASELINE_TABLES = "artifacts/tables"
BASELINE_VISITS = "data/processed/visits_validated.parquet"

pytestmark = pytest.mark.skipif(
    not resolve_path(BASELINE_VISITS).is_file()
    or not (resolve_path(BASELINE_TABLES) / "profile_triage_mix.csv").is_file(),
    reason="needs a baseline synthetic run (pipeline --data real) on this machine",
)


@pytest.fixture(scope="module")
def baseline() -> pd.DataFrame:
    """The validated visits the reference profile was built from."""
    return read_visits(BASELINE_VISITS)


def test_the_reference_dataset_shows_no_shift_against_itself(baseline: pd.DataFrame) -> None:
    """The identity case. If this flags, the comparison is not like-for-like.

    It did flag, once: the stored profile holds per-acuity medians while the data has a
    pooled median, and comparing the two reported 45% drift on a dataset against
    itself. Both sides are now the acuity-weighted statistic.
    """
    report = compare_to_reference(baseline, BASELINE_TABLES)

    assert report.findings, "the reference profile should have been found"
    assert not report.any_flagged, (
        "the reference dataset must not be flagged against its own profile: "
        + "; ".join(f"{f.name} {f.difference:.4f}" for f in report.flagged)
    )
    for finding in report.findings:
        assert finding.difference == pytest.approx(0.0, abs=1e-3)


def test_a_shifted_acuity_mix_is_flagged(baseline: pd.DataFrame) -> None:
    """Move the ESI mix and the check has to notice."""
    shifted = baseline.copy()
    # Push a fifth of the ESI 3 patients down to ESI 5. ESI 3 is ~40% of arrivals, so
    # this moves ~8 points of the mix: a total variation distance of ~0.08, comfortably
    # over the 0.05 threshold. A tenth moves ~0.04 and correctly does NOT flag.
    esi3 = shifted.index[shifted["triage_level"] == 3][:: 5]
    shifted.loc[esi3, "triage_level"] = 5

    report = compare_to_reference(shifted, BASELINE_TABLES)
    flagged = {finding.name for finding in report.flagged}

    assert "acuity mix" in flagged
    assert RETRAIN_NOTICE in report.verdict()


def test_a_shifted_wait_scale_is_flagged(baseline: pd.DataFrame) -> None:
    """Doubling every wait is exactly the case the 30-minute threshold cares about."""
    shifted = baseline.copy()
    shifted["wait_minutes"] = shifted["wait_minutes"] * 2.0

    report = compare_to_reference(shifted, BASELINE_TABLES)

    assert "median wait" in {finding.name for finding in report.flagged}


def test_a_shifted_arrival_shape_is_flagged(baseline: pd.DataFrame) -> None:
    """Move arrivals to a different time of day."""
    shifted = baseline.copy()
    shifted["arrival_time"] = pd.to_datetime(shifted["arrival_time"]) + pd.Timedelta(hours=8)

    report = compare_to_reference(shifted, BASELINE_TABLES)

    assert "arrival-hour shape" in {finding.name for finding in report.flagged}


def test_a_shifted_lwbs_rate_is_flagged(baseline: pd.DataFrame) -> None:
    """The competing risk the survival models and the patience fit depend on."""
    shifted = baseline.copy()
    # Mark a further 5% of patients as having left without being seen.
    extra = shifted.index[~shifted["is_lwbs"].astype(bool)][:: 20]
    shifted.loc[extra, "is_lwbs"] = True

    report = compare_to_reference(shifted, BASELINE_TABLES)

    assert "LWBS rate" in {finding.name for finding in report.flagged}


def test_the_verdict_is_readable_either_way(baseline: pd.DataFrame) -> None:
    """Both verdicts have to be sentences a non-technical reader can act on."""
    clean = compare_to_reference(baseline, BASELINE_TABLES).verdict()
    assert "tuned on" in clean
    assert "never been validated on real data" in clean

    shifted = baseline.copy()
    shifted["wait_minutes"] = shifted["wait_minutes"] * 3.0
    dirty = compare_to_reference(shifted, BASELINE_TABLES).verdict()
    assert "Retrain" in dirty


def test_a_missing_reference_is_reported_as_unknown_not_as_no_shift(tmp_path) -> None:
    """Absence of evidence must not be rendered as evidence of absence."""
    report = compare_to_reference(pd.DataFrame(), tmp_path)

    assert report.findings == []
    assert report.any_flagged is False
    assert "No comparison could be made" in report.verdict()


def test_profile_reports_both_median_forms(baseline: pd.DataFrame) -> None:
    """The pooled median is the honest summary; the weighted one is for comparing."""
    profile = profile_dataset(baseline)

    assert np.isfinite(profile["median_wait_minutes"])
    assert np.isfinite(profile["median_wait_comparable"])
    # On this dataset they genuinely differ, which is why the distinction exists.
    assert profile["median_wait_minutes"] != profile["median_wait_comparable"]


def test_the_thresholds_are_stated_and_non_trivial() -> None:
    """A threshold of 1.0 would make the check decorative."""
    assert set(THRESHOLDS) == {
        "arrival_hour_tvd", "triage_mix_tvd", "median_wait_relative", "lwbs_rate_absolute"
    }
    for name, value in THRESHOLDS.items():
        assert 0.0 < value < 1.0, f"{name} is not a usable threshold"


def test_an_empty_report_is_not_an_all_clear() -> None:
    """A `ShiftReport` with nothing in it says it could not compare."""
    assert "No comparison could be made" in ShiftReport().verdict()
