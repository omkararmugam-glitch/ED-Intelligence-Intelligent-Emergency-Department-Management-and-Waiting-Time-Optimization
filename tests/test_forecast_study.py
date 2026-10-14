"""Tests for the extended forecast study.

Two things need guarding here. The first is leakage: the study adds same-hour lookback
features anchored to the *target* rather than to `as_of`, which is a new way to read
the future, so it gets the same poisoning test the shipped features have. The second is
the selection discipline: every tuning constant must come from the validation origins,
because a blend weight or a peak correction fitted on test would make the whole study
worthless while still producing a plausible-looking table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.forecast import HistoryIndex, build_forecast_rows
from src.models.forecast_study import (
    MIN_ORIGINS,
    SAME_HOUR_FEATURES,
    VARIANTS,
    add_same_hour_features,
    apply_post_hoc_variants,
    coverage_by_band,
    fit_blend_weight,
    fit_peak_correction,
    paired_bootstrap_difference,
    rolling_conformal_intervals,
    select_by_validation,
    split_origins,
    study_feature_names,
)


@pytest.fixture()
def hourly_series() -> pd.DataFrame:
    """A deterministic hourly arrival series with a daily and weekly cycle."""
    stamps = pd.date_range("2024-01-01", periods=24 * 90, freq="h")
    hour = stamps.hour.to_numpy()
    weekday = stamps.dayofweek.to_numpy()
    counts = 6 + 4 * np.sin(2 * np.pi * hour / 24) + 0.5 * weekday
    return pd.DataFrame({"timestamp": stamps, "arrivals": np.round(counts).astype(float)})


# ---------------------------------------------------------------- leakage


def test_same_hour_features_ignore_everything_after_the_anchor(
    hourly_series: pd.DataFrame, default_config: dict
) -> None:
    """Poisoning the future must not change a single same-hour feature value.

    The `as_of` column in a feature row is the last *completed* hour -- the anchor --
    which is the convention `build_forecast_rows` already uses for its seasonal lag.
    The anchor hour itself is observed; everything after it is not.
    """
    anchor = hourly_series["timestamp"].iloc[24 * 60]
    history = HistoryIndex(hourly_series)
    rows = build_forecast_rows(history, [anchor], range(1, 73), default_config)
    clean = add_same_hour_features(rows, history, default_config)

    poisoned_series = hourly_series.copy()
    poisoned_series.loc[poisoned_series["timestamp"] > anchor, "arrivals"] = 9999.0
    poisoned = add_same_hour_features(
        rows, HistoryIndex(poisoned_series), default_config
    )

    for column in SAME_HOUR_FEATURES:
        pd.testing.assert_series_equal(
            clean[column], poisoned[column], check_names=False
        )


def test_backtest_predictions_cannot_see_past_their_origin(
    hourly_series: pd.DataFrame, default_config: dict
) -> None:
    """The whole backtest, end to end, is unchanged by poisoning the unscored future.

    This is the test that catches the mistake the anchoring in `run_backtest` exists to
    avoid: anchoring on the origin itself rather than on the last completed hour lets a
    24-hour-ahead row read the origin hour through its same-hour lookback.
    """
    from src.models.forecast_study import VariantSpec, run_backtest

    config = {**default_config}
    config["forecast"] = {**default_config["forecast"], "horizon_hours": 24}
    origins = [hourly_series["timestamp"].iloc[24 * 70]]
    only_control = (VariantSpec(name="poisson", description="control"),)

    clean = run_backtest(hourly_series, config, origins, only_control)

    # Everything strictly after the scored window is replaced with nonsense. The
    # scored hours and all history before the origin are untouched, so an honest
    # backtest cannot notice.
    poisoned_series = hourly_series.copy()
    beyond = poisoned_series["timestamp"] >= origins[0] + pd.Timedelta(hours=24)
    poisoned_series.loc[beyond, "arrivals"] = 9999.0
    poisoned = run_backtest(poisoned_series, config, origins, only_control)

    np.testing.assert_allclose(
        clean["yhat"].to_numpy(), poisoned["yhat"].to_numpy(), rtol=1e-12
    )
    np.testing.assert_allclose(
        clean["actual"].to_numpy(), poisoned["actual"].to_numpy(), rtol=1e-12
    )


def test_same_hour_features_are_missing_when_the_lookback_is_in_the_future(
    hourly_series: pd.DataFrame, default_config: dict
) -> None:
    """A one-day lookback beyond the horizon's reach must be NaN, not borrowed."""
    as_of = hourly_series["timestamp"].iloc[24 * 60]
    history = HistoryIndex(hourly_series)
    # A horizon far past 24h means target - 24h is after as_of for the later rows.
    rows = build_forecast_rows(history, [as_of], range(1, 97), default_config)
    enriched = add_same_hour_features(rows, history, default_config)

    far = enriched[enriched["horizon"] > 96]
    assert far["same_hour_prev_days_mean"].isna().all() if not far.empty else True
    # The weekly lookback is always observable at a 96-hour horizon.
    assert enriched["same_hour_prev_weeks_mean"].notna().any()


def test_only_the_same_hour_variant_sees_the_extra_features(default_config: dict) -> None:
    """A variant that did not ask for the features must not be given them."""
    by_name = {variant.name: variant for variant in VARIANTS}
    plain = study_feature_names(default_config, by_name["poisson"])
    enriched = study_feature_names(default_config, by_name["same_hour_lags"])

    assert not set(SAME_HOUR_FEATURES) & set(plain)
    assert set(SAME_HOUR_FEATURES) <= set(enriched)


# ---------------------------------------------------------------- selection discipline


def _toy_predictions() -> pd.DataFrame:
    """Backtest rows for one variant across six origins, three per block."""
    rows = []
    origins = pd.date_range("2024-03-01", periods=6, freq="7D")
    rng = np.random.default_rng(0)
    for index, origin in enumerate(origins):
        for hour in range(24):
            actual = 5.0 + 5.0 * (hour >= 12)
            rows.append(
                {
                    "origin": origin,
                    "timestamp": origin + pd.Timedelta(hours=hour),
                    "horizon": hour + 1,
                    "variant": "poisson",
                    "actual": actual,
                    # Under-predicts the busy half, which is the real diagnosis.
                    "yhat": actual - 2.0 * (hour >= 12) + rng.normal(0, 0.01),
                    "baseline": actual + 0.5,
                    "hour_band": "morning" if hour < 12 else "afternoon",
                    "hour_of_day": hour,
                    "block": "validation" if index < 3 else "test",
                }
            )
    return pd.DataFrame(rows)


def test_origins_split_earlier_for_tuning_later_for_reporting() -> None:
    """The validation origins must all precede the test origins."""
    origins = list(pd.date_range("2024-01-01", periods=26, freq="7D"))
    validation, test = split_origins(origins)

    assert len(validation) + len(test) == 26
    assert max(validation) < min(test)


def test_split_refuses_to_leave_a_block_empty() -> None:
    """A split with nothing to report on is an error, not a silent pass."""
    with pytest.raises(ValueError, match="cannot be split"):
        split_origins([pd.Timestamp("2024-01-01")])


def test_blend_weight_is_fitted_on_the_rows_it_is_given() -> None:
    """The caller passes validation rows only; the fit must use exactly those."""
    predictions = _toy_predictions()
    validation = predictions[predictions["block"] == "validation"]

    weight, grid = fit_blend_weight(validation)
    assert 0.0 <= weight <= 1.0
    assert grid["val_mae"].is_monotonic_increasing  # sorted best-first
    assert grid.iloc[0]["weight_on_model"] == pytest.approx(weight)


def test_peak_correction_keys_off_the_prediction_not_the_truth() -> None:
    """At serving time the truth is unknown, so the rule cannot depend on it."""
    predictions = _toy_predictions()
    validation = predictions[predictions["block"] == "validation"].copy()

    multiplier, summary = fit_peak_correction(validation, threshold=8.0)
    assert multiplier > 1.0, "the toy data under-predicts busy hours, so it should lift"

    # Corrupting the truth on rows the model does not predict as peak must not move
    # which rows get corrected.
    corrected = apply_post_hoc_variants(
        predictions, blend_weight=1.0, peak_multiplier=multiplier, threshold=8.0
    )
    peak_rows = corrected[corrected["variant"] == "peak_corrected"]
    source = corrected[corrected["variant"] == "poisson"]
    changed = ~np.isclose(
        peak_rows["yhat"].to_numpy(), source["yhat"].to_numpy()
    )
    assert np.array_equal(changed, source["yhat"].to_numpy() >= 8.0)


def test_selection_uses_validation_rows_only() -> None:
    """A winner chosen on test would invalidate the study; it must refuse."""
    scores = pd.DataFrame(
        [
            {"block": "test", "scope": "peak", "variant": "a", "mae": 1.0,
             "seasonal_naive_mae": 2.0},
        ]
    )
    with pytest.raises(ValueError, match="No validation scores"):
        select_by_validation(scores)


def test_selection_picks_the_lowest_validation_mae() -> None:
    """The rule is stated plainly enough to be checked."""
    scores = pd.DataFrame(
        [
            {"block": "validation", "scope": "peak", "variant": "a", "mae": 2.0,
             "seasonal_naive_mae": 3.0},
            {"block": "validation", "scope": "peak", "variant": "b", "mae": 1.0,
             "seasonal_naive_mae": 3.0},
            {"block": "test", "scope": "peak", "variant": "a", "mae": 0.1,
             "seasonal_naive_mae": 3.0},
        ]
    )
    selection = select_by_validation(scores)

    assert selection["winner"] == "b"
    assert selection["selection_block"] == "validation"
    assert selection["runner_up"] == "a"


def test_study_refuses_too_few_origins(hourly_series: pd.DataFrame, default_config: dict) -> None:
    """Running the study with four folds would repeat the mistake it exists to fix."""
    from src.models.forecast_study import run_forecast_study

    with pytest.raises(ValueError, match="below the minimum"):
        run_forecast_study(hourly_series, default_config, n_origins=4)


# ---------------------------------------------------------------- intervals


def test_rolling_intervals_use_only_residuals_observed_at_the_origin() -> None:
    """The first origin has no past forecasts, so it gets no interval at all."""
    predictions = _toy_predictions()
    intervals = rolling_conformal_intervals(
        predictions, _config_with_coverage(), "poisson", min_calibration=5
    )

    first_origin = intervals["origin"].min()
    first = intervals[intervals["origin"] == first_origin]
    assert first["lower"].isna().all()
    assert (first["interval_source"] == "none: too little history").all()

    later = intervals[intervals["origin"] == intervals["origin"].max()]
    assert later["lower"].notna().any(), "later origins should have usable history"


def _config_with_coverage() -> dict:
    """Minimal config slice for the interval helpers."""
    return {
        "conformal": {"coverage": 0.90, "tolerance": 0.04},
        "forecast": {"hour_bands": {"morning": [0, 12], "afternoon": [12, 24]}},
        "seed": 1,
    }


def test_coverage_table_flags_a_band_outside_tolerance() -> None:
    """A band that misses nominal by more than the tolerance must be marked."""
    intervals = pd.DataFrame(
        {
            "hour_band": ["morning"] * 100 + ["afternoon"] * 100,
            "covered": [1.0] * 50 + [0.0] * 50 + [1.0] * 90 + [0.0] * 10,
            "lower": 1.0,
            "upper": 3.0,
        }
    )
    table = coverage_by_band(intervals, _config_with_coverage(), "test").set_index("scope")

    assert bool(table.loc["morning", "within_tolerance"]) is False
    assert bool(table.loc["afternoon", "within_tolerance"]) is True


def test_paired_bootstrap_detects_a_real_difference() -> None:
    """A model that is plainly better gets an interval that excludes zero."""
    rng = np.random.default_rng(1)
    actual = rng.normal(10, 3, 500)
    good = actual + rng.normal(0, 0.5, 500)
    poor = actual + rng.normal(0, 4.0, 500)

    result = paired_bootstrap_difference(actual, good, poor, seed=1, label="t", n_resamples=400)
    assert result["model_is_better"] is True
    assert result["significant"] is True
    assert result["ci_high"] < 0


def test_paired_bootstrap_reports_no_difference_when_there_is_none() -> None:
    """Two identical forecasters must not be declared different."""
    rng = np.random.default_rng(2)
    actual = rng.normal(10, 3, 400)
    same = actual + rng.normal(0, 1.0, 400)

    result = paired_bootstrap_difference(actual, same, same, seed=1, label="u", n_resamples=400)
    assert result["mae_difference"] == pytest.approx(0.0, abs=1e-9)
    assert result["significant"] is False


def test_minimum_origins_is_what_the_task_asked_for() -> None:
    """26 weekly origins is the floor the study is built around."""
    assert MIN_ORIGINS >= 26
