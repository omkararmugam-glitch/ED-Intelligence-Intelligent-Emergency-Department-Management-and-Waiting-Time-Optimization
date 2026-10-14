"""Tests for hourly arrival forecasting (Phase 2).

The important one is `test_forecast_uses_only_data_before_as_of`: a forecast issued at
`as_of` must be bit-identical whether or not the future is present in the series it is
handed. Everything else follows from that.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.data.schema import validate_visits
from src.data.synthetic import generate_synthetic_visits
from src.models.forecast import (
    HistoryIndex,
    SeasonalNaiveForecaster,
    build_forecast_rows,
    build_hourly_series,
    calibrate_forecaster,
    evaluate_forecast_intervals,
    forecast,
    forecast_feature_names,
    rolling_origin_evaluate,
    summarise_forecast,
    train_forecaster,
)

FORECAST_N_DAYS = 160


@pytest.fixture(scope="module")
def forecast_config() -> dict:
    """Config sized so fits stay quick but the weekly cycle is still learnable."""
    from src.utils.io import load_config

    cfg = load_config("configs/default.yaml")
    cfg["forecast"]["model"]["n_estimators"] = 150
    cfg["forecast"]["train_stride_hours"] = 8
    cfg["forecast"]["rolling_origin"]["n_folds"] = 2
    return cfg


@pytest.fixture(scope="module")
def series(forecast_config) -> pd.DataFrame:
    """A gap-free hourly arrival series spanning several months."""
    visits = validate_visits(generate_synthetic_visits(forecast_config, n_days=FORECAST_N_DAYS)).all
    return build_hourly_series(visits)


@pytest.fixture(scope="module")
def fitted(series, forecast_config) -> dict:
    """A forecaster fitted and calibrated on disjoint, ordered windows."""
    start = series["timestamp"].min()
    span = series["timestamp"].max() - start
    train_end = (start + span * 0.6).floor("h")
    val_end = (start + span * 0.75).floor("h")
    calib_end = (start + span * 0.9).floor("h")

    forecaster = train_forecaster(series, forecast_config, train_end=train_end, val_end=val_end)
    calibrate_forecaster(forecaster, series, forecast_config, val_end, calib_end)
    return {"forecaster": forecaster, "calib_end": calib_end, "train_end": train_end}


# ------------------------------------------------------------------- hourly series


def test_hourly_series_is_contiguous_and_keeps_zero_hours(series) -> None:
    """Quiet hours must appear as explicit zeros, or every lag is corrupted."""
    gaps = series["timestamp"].diff().dropna().unique()
    assert len(gaps) == 1 and gaps[0] == pd.Timedelta(hours=1)

    assert (series["arrivals"] >= 0).all()
    assert series["arrivals"].dtype.kind in "iu"
    # A four-month synthetic span has quiet pre-dawn hours with no arrivals at all.
    assert (series["arrivals"] == 0).any()


def test_hourly_series_totals_match_the_visits(forecast_config) -> None:
    """Aggregation must neither lose nor invent an arrival."""
    visits = validate_visits(generate_synthetic_visits(forecast_config, n_days=30)).all
    built = build_hourly_series(visits)
    assert built["arrivals"].sum() == len(visits)


def test_empty_visits_raise() -> None:
    """An empty table is a configuration error, not an empty series."""
    with pytest.raises(ValueError, match="empty visits table"):
        build_hourly_series(pd.DataFrame({"arrival_time": pd.to_datetime([])}))


def test_history_index_lookups_are_exact(series) -> None:
    """The index must find real timestamps and return NaN for absent ones."""
    index = HistoryIndex(series)
    stamps = series["timestamp"].iloc[[0, 10, 100]]
    assert np.array_equal(index.value_at(stamps), series["arrivals"].iloc[[0, 10, 100]].to_numpy())

    outside = [series["timestamp"].min() - pd.Timedelta(hours=5)]
    assert np.isnan(index.value_at(outside)).all()


def test_history_rolling_mean_matches_a_direct_computation(series) -> None:
    """The cumulative-sum shortcut must agree with the obvious calculation."""
    index = HistoryIndex(series)
    stamp = series["timestamp"].iloc[200]
    window = 24
    expected = series["arrivals"].iloc[200 - window + 1 : 201].mean()
    assert index.rolling_mean_ending_at([stamp], window)[0] == pytest.approx(expected)


# ------------------------------------------------------------------------- leakage


def test_forecast_uses_only_data_before_as_of(series, fitted, forecast_config) -> None:
    """The Phase 2 leakage requirement, stated as an equality.

    A forecast issued at `as_of` is produced twice: once from the full series, and once
    from a copy truncated at `as_of`. If any feature read a value at or after `as_of`,
    the two would differ.
    """
    as_of = fitted["calib_end"]
    horizon = 48

    full = forecast(horizon, as_of, fitted["forecaster"], series, forecast_config)
    truncated_series = series.loc[series["timestamp"] < as_of].copy()
    truncated = forecast(horizon, as_of, fitted["forecaster"], truncated_series, forecast_config)

    pd.testing.assert_frame_equal(full, truncated, check_exact=True)


def test_corrupting_the_future_cannot_change_a_forecast(series, fitted, forecast_config) -> None:
    """Replacing every future observation with nonsense must not move the forecast."""
    as_of = fitted["calib_end"]
    poisoned = series.copy()
    future = poisoned["timestamp"] >= as_of
    assert future.any()
    poisoned.loc[future, "arrivals"] = 9999

    before = forecast(48, as_of, fitted["forecaster"], series, forecast_config)
    after = forecast(48, as_of, fitted["forecaster"], poisoned, forecast_config)
    pd.testing.assert_frame_equal(before, after, check_exact=True)


def test_seasonal_lag_is_dropped_once_it_would_need_the_future(series, forecast_config) -> None:
    """Beyond 168 hours the seasonal lag is unknowable and must be missing, not guessed."""
    index = HistoryIndex(series)
    as_of = series["timestamp"].iloc[500]
    rows = build_forecast_rows(index, [as_of], [1, 100, 169, 200], forecast_config)

    column = f"seasonal_lag_{forecast_config['forecast']['seasonal_lag_hours']}"
    by_horizon = rows.set_index("horizon")[column]
    assert np.isfinite(by_horizon.loc[1])
    assert np.isfinite(by_horizon.loc[100])
    assert np.isnan(by_horizon.loc[169])
    assert np.isnan(by_horizon.loc[200])


# -------------------------------------------------------------- public interface


def test_forecast_returns_the_documented_frame(series, fitted, forecast_config) -> None:
    """forecast(horizon_hours, as_of) -> DataFrame[timestamp, yhat, lower, upper]."""
    out = forecast(72, fitted["calib_end"], fitted["forecaster"], series, forecast_config)

    for column in ("timestamp", "yhat", "lower", "upper"):
        assert column in out.columns
    assert len(out) == 72
    assert out["timestamp"].min() >= fitted["calib_end"]
    assert out["timestamp"].is_monotonic_increasing

    # Counts cannot be negative, and the interval must bracket the point forecast.
    assert (out["yhat"] >= 0).all()
    assert (out["lower"] >= 0).all()
    assert (out["lower"] <= out["yhat"]).all()
    assert (out["yhat"] <= out["upper"]).all()


def test_forecast_rejects_a_non_positive_horizon(series, fitted, forecast_config) -> None:
    """A zero-hour forecast is a programming error."""
    with pytest.raises(ValueError, match="horizon_hours must be positive"):
        forecast(0, fitted["calib_end"], fitted["forecaster"], series, forecast_config)


def test_forecast_without_history_raises(series, fitted, forecast_config) -> None:
    """Forecasting from before the data starts cannot be done silently."""
    before_everything = series["timestamp"].min() - pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="No observations before"):
        forecast(24, before_everything, fitted["forecaster"], series, forecast_config)


def test_forecast_is_deterministic(series, fitted, forecast_config) -> None:
    """Same model, same as_of, same answer."""
    first = forecast(24, fitted["calib_end"], fitted["forecaster"], series, forecast_config)
    second = forecast(24, fitted["calib_end"], fitted["forecaster"], series, forecast_config)
    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_forecast_tracks_the_daily_demand_cycle(series, fitted, forecast_config) -> None:
    """A forecaster that ignored hour-of-day would be useless for staffing."""
    out = forecast(72, fitted["calib_end"], fitted["forecaster"], series, forecast_config)
    by_hour = out.assign(hour=out["timestamp"].dt.hour).groupby("hour")["yhat"].mean()

    busy = by_hour.loc[[9, 10, 11, 12]].mean()
    quiet = by_hour.loc[[2, 3, 4, 5]].mean()
    assert busy > 2.0 * quiet, f"daily cycle not learned: busy {busy:.2f} vs quiet {quiet:.2f}"


# ---------------------------------------------------------------------- baseline


def test_seasonal_naive_returns_the_value_one_week_earlier(series) -> None:
    """The baseline is exactly lag 168, with no smoothing."""
    baseline = SeasonalNaiveForecaster(lag_hours=168).fit(series)
    target = series["timestamp"].iloc[500]
    expected = series.set_index("timestamp")["arrivals"].loc[target - pd.Timedelta(hours=168)]
    assert baseline.predict([target])[0] == expected


def test_seasonal_naive_requires_fit(series) -> None:
    """Predicting before fitting is a programming error."""
    with pytest.raises(RuntimeError, match="Call fit"):
        SeasonalNaiveForecaster().predict(series["timestamp"].head(3))


# ----------------------------------------------------------- rolling-origin backtest


@pytest.fixture(scope="module")
def backtest(series, forecast_config) -> tuple[pd.DataFrame, dict]:
    """A two-fold rolling-origin backtest, reused by several assertions."""
    return rolling_origin_evaluate(series, forecast_config)


def test_backtest_folds_do_not_see_their_own_future(backtest) -> None:
    """Every scored hour must fall strictly after the fold's origin."""
    predictions, _ = backtest
    assert not predictions.empty
    assert (predictions["timestamp"] >= predictions["origin"]).all()
    assert predictions["fold"].nunique() >= 2


def test_model_beats_the_seasonal_naive_baseline(backtest) -> None:
    """CLAUDE.md rule 4: a model that cannot beat its baseline is reported as such."""
    _, summary = backtest
    model = summary["overall"]["model"]
    naive = summary["overall"]["seasonal_naive"]

    assert model["mae"] < naive["mae"], f"model MAE {model['mae']} vs naive {naive['mae']}"
    assert model["rmse"] < naive["rmse"]
    assert summary["mae_improvement_over_baseline_percent"] > 0


def test_backtest_reports_every_required_metric(backtest) -> None:
    """MAE, RMSE and MAPE are required; sMAPE and WAPE guard the zero-count hours."""
    _, summary = backtest
    for segment in ("overall", "peak_hours", "off_peak_hours"):
        for model_name in ("model", "seasonal_naive"):
            metrics = summary[segment][model_name]
            for key in ("mae", "rmse", "mape", "smape", "wape"):
                assert key in metrics and np.isfinite(metrics[key]), f"{segment}/{model_name}/{key}"


def test_peak_hours_are_reported_separately(backtest, forecast_config) -> None:
    """Peak hours are the top 20% of demand, and are harder than off-peak."""
    predictions, summary = backtest
    quantile = float(forecast_config["forecast"]["peak_quantile"])

    expected_threshold = float(np.quantile(predictions["actual"], quantile))
    assert summary["peak_threshold_arrivals"] == pytest.approx(expected_threshold)
    assert 0 < summary["n_peak_hours"] < summary["n_hours"]

    # Busy hours carry larger absolute errors than quiet ones, by construction.
    assert summary["peak_hours"]["model"]["mae"] > summary["off_peak_hours"]["model"]["mae"]


def test_model_beats_the_baseline_at_every_lead_time(backtest) -> None:
    """The forecaster must be useful across the whole horizon, not just the first day.

    Note what is deliberately *not* asserted: that MAE rises monotonically with lead
    time. It does on a full backtest, but each band here holds only a few dozen hours,
    where composition noise swamps the trend. The seasonal-naive baseline is also
    horizon-independent by construction -- it is always the value 168 hours earlier --
    so comparing its bands measures which hours landed in them, not forecast difficulty.
    Beating it at every lead time is the claim that survives a small sample.
    """
    _, summary = backtest
    bands = {row["horizon_band"]: row for row in summary["by_horizon"]}
    assert set(bands) == {"1-24h", "25-48h", "49-72h"}

    for name, row in bands.items():
        assert row["model_mae"] < row["naive_mae"], (
            f"model lost to the baseline at {name}: "
            f"{row['model_mae']:.3f} vs {row['naive_mae']:.3f}"
        )


def test_backtest_intervals_are_calibrated(backtest, forecast_config) -> None:
    """Each fold calibrates on held-out rows, so coverage should land near nominal."""
    predictions, _ = backtest
    coverage = evaluate_forecast_intervals(predictions, forecast_config)

    assert coverage, "no calibrated intervals were produced by the backtest"
    nominal = float(forecast_config["conformal"]["coverage"])
    tolerance = float(forecast_config["conformal"]["tolerance"])
    assert abs(coverage["overall"]["coverage"] - nominal) <= tolerance * 2


def test_interval_width_differs_by_hour_band(backtest, forecast_config) -> None:
    """Mondrian banding exists so quiet night hours get narrower intervals."""
    predictions, _ = backtest
    coverage = evaluate_forecast_intervals(predictions, forecast_config)
    by_band = coverage.get("by_group", {})

    assert "night" in by_band
    busiest = max(by_band, key=lambda band: by_band[band]["mean_width"])
    assert by_band["night"]["mean_width"] < by_band[busiest]["mean_width"]


def test_feature_names_are_stable(forecast_config) -> None:
    """The forecaster's design matrix columns must be reproducible and ordered."""
    first = forecast_feature_names(forecast_config)
    assert first == forecast_feature_names(forecast_config)
    assert first[0] == "horizon"
    assert any(name.startswith("lag_") for name in first)
    assert any(name.startswith("rolling_mean_") for name in first)
