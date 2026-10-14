"""Tests for rolling conformal calibration (Part A).

Three things must hold:
  * no residual from a patient seen at or after the target's arrival is ever used;
  * the intervals are reproducible;
  * coverage lands near nominal even when the load shifts mid-stream, which is the
    case static conformal gets wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.conformal import fit_split_conformal
from src.models.rolling_conformal import (
    SOURCE_POOLED,
    SOURCE_STATIC,
    SOURCE_WIDE,
    SOURCE_WINDOW,
    RollingConformalConfig,
    compare_static_and_rolling,
    fit_rolling_intervals,
)
from src.utils.metrics import interval_metrics


def _pool(seen_times, triage, residuals) -> pd.DataFrame:
    """Build a calibration pool frame."""
    return pd.DataFrame(
        {
            "seen_time": pd.to_datetime(seen_times),
            "triage_level": triage,
            "abs_residual": np.asarray(residuals, dtype="float64"),
        }
    )


def _targets(arrival_times, triage) -> pd.DataFrame:
    """Build a target frame."""
    return pd.DataFrame(
        {"arrival_time": pd.to_datetime(arrival_times), "triage_level": triage}
    )


@pytest.fixture()
def settings() -> RollingConformalConfig:
    """Small windows and counts so the fixtures stay readable."""
    return RollingConformalConfig(
        coverage=0.90, window_days=7.0, fallback_window_days=30.0, min_calibration=10
    )


# --------------------------------------------------------------------- leakage


def test_a_residual_seen_after_the_arrival_is_never_used(settings) -> None:
    """The defining boundary: a wait is only knowable once the patient has been seen.

    The pool here is split into small residuals observed before the query and enormous
    ones observed after. If any future residual leaked in, the half-width would jump.
    """
    query = pd.Timestamp("2024-06-15 12:00:00")

    past = pd.date_range(query - pd.Timedelta(days=5), periods=60, freq="2h")
    future = pd.date_range(query, periods=60, freq="2h")
    pool = pd.concat(
        [
            _pool(past, [3] * len(past), np.full(len(past), 5.0)),
            _pool(future, [3] * len(future), np.full(len(future), 10_000.0)),
        ],
        ignore_index=True,
    )

    result = fit_rolling_intervals(
        pool, _targets([query], [3]), np.array([30.0]), settings
    )
    # Only the 5.0 residuals were admissible, so the half-width must be 5.0.
    assert result.half_width[0] == pytest.approx(5.0)
    assert result.upper[0] == pytest.approx(35.0)
    assert result.n_calibration[0] == len(past)


def test_a_residual_seen_exactly_at_the_arrival_is_excluded(settings) -> None:
    """The boundary is strict: seen_time < arrival_time, not <=.

    A patient seen at the very instant another arrives has not had their wait recorded
    in any system the arriving patient's prediction could consult.
    """
    query = pd.Timestamp("2024-06-15 12:00:00")
    earlier = pd.date_range(query - pd.Timedelta(days=3), periods=30, freq="2h")

    pool = pd.concat(
        [
            _pool(earlier, [3] * len(earlier), np.full(len(earlier), 4.0)),
            _pool([query] * 30, [3] * 30, np.full(30, 999.0)),
        ],
        ignore_index=True,
    )
    result = fit_rolling_intervals(pool, _targets([query], [3]), np.array([20.0]), settings)

    assert result.n_calibration[0] == len(earlier)
    assert result.half_width[0] == pytest.approx(4.0)


def test_only_the_same_triage_level_is_used_when_available(settings) -> None:
    """Mondrian: a patient's interval comes from patients of their own acuity."""
    query = pd.Timestamp("2024-06-15 12:00:00")
    past = pd.date_range(query - pd.Timedelta(days=5), periods=40, freq="3h")

    pool = pd.concat(
        [
            _pool(past, [3] * len(past), np.full(len(past), 6.0)),
            _pool(past, [5] * len(past), np.full(len(past), 60.0)),
        ],
        ignore_index=True,
    )
    result = fit_rolling_intervals(
        pool, _targets([query, query], [3, 5]), np.array([20.0, 20.0]), settings
    )
    assert result.half_width[0] == pytest.approx(6.0)
    assert result.half_width[1] == pytest.approx(60.0)
    assert list(result.source) == [SOURCE_WINDOW, SOURCE_WINDOW]


def test_only_residuals_inside_the_window_are_used(settings) -> None:
    """Old residuals must age out, or the method cannot track drift."""
    query = pd.Timestamp("2024-06-15 12:00:00")
    recent = pd.date_range(query - pd.Timedelta(days=3), periods=30, freq="2h")
    ancient = pd.date_range(query - pd.Timedelta(days=200), periods=30, freq="2h")

    pool = pd.concat(
        [
            _pool(recent, [3] * len(recent), np.full(len(recent), 7.0)),
            _pool(ancient, [3] * len(ancient), np.full(len(ancient), 900.0)),
        ],
        ignore_index=True,
    )
    result = fit_rolling_intervals(pool, _targets([query], [3]), np.array([25.0]), settings)

    assert result.half_width[0] == pytest.approx(7.0)
    assert result.n_calibration[0] == len(recent)


# ------------------------------------------------------------------- fallbacks


def test_a_thin_window_widens_before_pooling(settings) -> None:
    """Too few recent same-acuity residuals must widen the window, not pool immediately."""
    query = pd.Timestamp("2024-06-15 12:00:00")
    # Only 3 inside the 7-day window, but 40 inside the 30-day one.
    recent = pd.date_range(query - pd.Timedelta(days=2), periods=3, freq="6h")
    older = pd.date_range(query - pd.Timedelta(days=20), periods=40, freq="6h")

    pool = pd.concat(
        [
            _pool(recent, [3] * len(recent), np.full(len(recent), 8.0)),
            _pool(older, [3] * len(older), np.full(len(older), 9.0)),
        ],
        ignore_index=True,
    )
    result = fit_rolling_intervals(pool, _targets([query], [3]), np.array([20.0]), settings)

    assert result.source[0] == SOURCE_WIDE
    assert result.n_calibration[0] > settings.min_calibration


def test_a_rare_acuity_falls_back_to_the_pooled_level(settings) -> None:
    """An acuity with almost no history borrows from every level rather than guessing."""
    query = pd.Timestamp("2024-06-15 12:00:00")
    past = pd.date_range(query - pd.Timedelta(days=5), periods=40, freq="3h")

    pool = pd.concat(
        [
            _pool(past, [3] * len(past), np.full(len(past), 12.0)),
            _pool(past[:2], [1] * 2, np.full(2, 3.0)),
        ],
        ignore_index=True,
    )
    result = fit_rolling_intervals(pool, _targets([query], [1]), np.array([10.0]), settings)

    assert result.source[0] == SOURCE_POOLED
    assert result.n_calibration[0] > settings.min_calibration


def test_an_empty_history_falls_back_to_the_static_half_width(settings) -> None:
    """With nothing to calibrate on, the static interval is the honest answer."""
    query = pd.Timestamp("2024-01-01 00:00:00")
    pool = _pool([], [], [])

    result = fit_rolling_intervals(
        pool, _targets([query], [3]), np.array([20.0]), settings, static_half_width=33.0
    )
    assert result.source[0] == SOURCE_STATIC
    assert result.half_width[0] == pytest.approx(33.0)
    assert result.upper[0] == pytest.approx(53.0)


def test_static_fallback_accepts_a_per_group_mapping(settings) -> None:
    """A Mondrian static calibrator supplies one half-width per acuity."""
    query = pd.Timestamp("2024-01-01 00:00:00")
    result = fit_rolling_intervals(
        _pool([], [], []),
        _targets([query, query], [2, 4]),
        np.array([10.0, 10.0]),
        settings,
        static_half_width={2: 5.0, 4: 50.0},
    )
    assert result.half_width.tolist() == [5.0, 50.0]


# -------------------------------------------------------------- reproducibility


def test_rolling_intervals_are_reproducible(settings) -> None:
    """The method is deterministic: same inputs, same intervals, every time."""
    rng = np.random.default_rng(0)
    start = pd.Timestamp("2024-01-01")
    seen = pd.date_range(start, periods=600, freq="3h")
    pool = _pool(seen, rng.integers(1, 6, 600), np.abs(rng.normal(20, 8, 600)))

    arrivals = pd.date_range(start + pd.Timedelta(days=40), periods=120, freq="5h")
    targets = _targets(arrivals, rng.integers(1, 6, 120))
    predictions = rng.uniform(10, 60, 120)

    first = fit_rolling_intervals(pool, targets, predictions, settings)
    second = fit_rolling_intervals(pool, targets, predictions, settings)

    assert np.array_equal(first.lower, second.lower)
    assert np.array_equal(first.upper, second.upper)
    assert np.array_equal(first.n_calibration, second.n_calibration)
    assert list(first.source) == list(second.source)


def test_row_order_does_not_change_a_rows_interval(settings) -> None:
    """Each interval depends only on that row's own timestamp and acuity."""
    rng = np.random.default_rng(1)
    start = pd.Timestamp("2024-01-01")
    pool = _pool(
        pd.date_range(start, periods=500, freq="3h"),
        rng.integers(1, 6, 500),
        np.abs(rng.normal(25, 10, 500)),
    )
    arrivals = pd.date_range(start + pd.Timedelta(days=40), periods=60, freq="7h")
    targets = _targets(arrivals, [3] * 60)
    predictions = np.linspace(10, 70, 60)

    forward = fit_rolling_intervals(pool, targets, predictions, settings)
    order = np.arange(59, -1, -1)
    reversed_result = fit_rolling_intervals(
        pool, targets.iloc[order].reset_index(drop=True), predictions[order], settings
    )
    assert np.allclose(forward.half_width[order], reversed_result.half_width)


# ------------------------------------------------------------ the drift benefit


@pytest.fixture()
def drifting_block() -> dict:
    """A stream whose error scale triples part-way through.

    This is the situation static conformal is built to get wrong: calibrate on the calm
    stretch and the intervals stay too narrow once the load shifts.
    """
    rng = np.random.default_rng(7)
    n_calm, n_busy = 2400, 1600
    start = pd.Timestamp("2024-01-01")

    times = pd.date_range(start, periods=n_calm + n_busy, freq="1h")
    triage = rng.integers(1, 6, n_calm + n_busy)
    # Residual scale triples at the shift; predictions stay unbiased.
    scale = np.concatenate([np.full(n_calm, 8.0), np.full(n_busy, 24.0)])
    truth = rng.normal(40.0, scale)
    predictions = np.full(truth.shape, 40.0)

    frame = pd.DataFrame(
        {
            "arrival_time": times,
            "seen_time": times,  # observed immediately, so the window is well populated
            "triage_level": triage,
            "wait": truth,
            "abs_residual": np.abs(truth - predictions),
        }
    )
    return {"frame": frame, "predictions": predictions, "shift_index": n_calm}


def test_rolling_beats_static_coverage_under_a_load_shift(drifting_block) -> None:
    """Coverage must land near nominal after the shift, where static under-covers."""
    frame = drifting_block["frame"]
    predictions = drifting_block["predictions"]
    shift = drifting_block["shift_index"]

    settings = RollingConformalConfig(
        coverage=0.90, window_days=14.0, fallback_window_days=60.0, min_calibration=50
    )

    # Static: calibrate on the calm stretch, as a fixed calibration block would.
    calm = frame.iloc[: shift // 2]
    static = fit_split_conformal(
        calm["wait"].to_numpy(),
        predictions[: shift // 2],
        coverage=settings.coverage,
        groups=calm["triage_level"].to_numpy(),
        min_group_size=50,
    )

    busy = frame.iloc[shift:].reset_index(drop=True)
    busy_predictions = predictions[shift:]

    static_lower, static_upper = static.intervals(
        busy_predictions, groups=busy["triage_level"].to_numpy()
    )
    rolling = fit_rolling_intervals(
        frame, busy, busy_predictions, settings,
        static_half_width=static.group_quantiles or static.global_quantile,
    )

    static_stats = interval_metrics(
        busy["wait"], static_lower, static_upper, nominal_coverage=settings.coverage
    )
    rolling_stats = interval_metrics(
        busy["wait"], rolling.lower, rolling.upper, nominal_coverage=settings.coverage
    )

    # Static under-covers badly once the scale triples.
    assert static_stats["coverage"] < 0.80, static_stats["coverage"]
    # Rolling tracks the shift and lands near nominal.
    assert abs(rolling_stats["coverage_gap"]) < 0.05, rolling_stats
    assert rolling_stats["coverage"] > static_stats["coverage"]
    # It pays for that in width, which is the honest trade.
    assert rolling_stats["mean_width"] > static_stats["mean_width"]


def test_comparison_table_covers_both_methods_overall_and_per_group(drifting_block) -> None:
    """The reported table must carry both methods at every scope."""
    frame = drifting_block["frame"].iloc[2400:].reset_index(drop=True)
    predictions = drifting_block["predictions"][2400:]
    groups = frame["triage_level"].to_numpy()

    settings = RollingConformalConfig(coverage=0.90, min_calibration=50)
    rolling = fit_rolling_intervals(
        drifting_block["frame"], frame, predictions, settings, static_half_width=20.0
    )
    table = compare_static_and_rolling(
        truth=frame["wait"].to_numpy(),
        groups=groups,
        static_lower=predictions - 20.0,
        static_upper=predictions + 20.0,
        rolling=rolling,
        coverage=0.90,
    )

    assert set(table["method"]) == {"static", "rolling"}
    assert "overall" in set(table["scope"])
    for level in np.unique(groups):
        assert f"ESI {level}" in set(table["scope"])
    assert (table["empirical_coverage"].between(0, 1)).all()


def test_mismatched_prediction_count_raises(settings) -> None:
    """A length mismatch is a programming error, not something to pad over."""
    with pytest.raises(ValueError, match="predictions for"):
        fit_rolling_intervals(
            _pool([], [], []), _targets(["2024-01-01"], [3]), np.array([1.0, 2.0]), settings
        )


def test_missing_columns_raise(settings) -> None:
    """Required columns must be named when absent."""
    with pytest.raises(KeyError, match="calibration pool"):
        fit_rolling_intervals(
            pd.DataFrame({"seen_time": []}), _targets(["2024-01-01"], [3]),
            np.array([1.0]), settings,
        )
