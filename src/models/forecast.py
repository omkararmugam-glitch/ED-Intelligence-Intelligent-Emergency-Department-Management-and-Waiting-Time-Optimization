"""Hourly ED arrival forecasting.

**The design decision that makes this leakage-free.** A forecast issued at `as_of` for a
timestamp `h` hours later cannot use `y(target - 1)`: that value has not happened yet
when `h > 1`. The usual shortcut -- building lag features relative to the *target*
timestamp -- silently assumes it has, and produces a model that looks excellent in
backtest and fails in production.

This module therefore anchors every lag to **`as_of`**, not to the target:

    features(target, as_of) = {
        horizon            = target - as_of, in hours
        lag_k              = y(as_of - k + 1)        for k in forecast.lags
        rolling_mean_w     = mean of y over the w hours ending at as_of
        seasonal_lag       = y(target - 168)         usable only while horizon <= 168
        calendar(target)   = deterministic, knowable arbitrarily far ahead
    }

Horizon is itself a feature, so one model serves every lead time (a "direct
multi-horizon" forecaster) and accuracy is free to decay with horizon, as it should.
`forecast(horizon_hours, as_of)` and the backtest share this one function, so the
evaluation cannot be more optimistic than deployment.

Counts, not amounts: the model optimises a Poisson objective and predictions are
clipped at zero. MAPE is reported for comparability with the literature but is
undefined at zero-count hours, so sMAPE and WAPE are reported alongside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.calendar import CALENDAR_FEATURES, add_calendar_features
from src.models.conformal import SplitConformalCalibrator, evaluate_intervals, fit_split_conformal
from src.utils.logging import get_logger
from src.utils.metrics import count_metrics
from src.utils.seed import derive_seed

logger = get_logger(__name__)

TIME_COLUMN = "timestamp"
TARGET_COLUMN = "arrivals"


def build_hourly_series(visits: pd.DataFrame, time_column: str = "arrival_time") -> pd.DataFrame:
    """Aggregate individual arrivals into a gap-free hourly count series.

    Hours with no arrivals must appear as explicit zeros, not as missing rows: a quiet
    3 a.m. is information, and dropping it would corrupt every lag.

    Args:
        visits: Validated visits, one row per patient.
        time_column: Arrival timestamp column.

    Returns:
        DataFrame with `timestamp` and `arrivals`, hourly and contiguous.

    Raises:
        ValueError: If the table is empty.
    """
    if visits.empty:
        raise ValueError("Cannot build an hourly series from an empty visits table")

    times = pd.to_datetime(visits[time_column])
    counts = times.dt.floor("h").value_counts().sort_index()
    full_index = pd.date_range(counts.index.min(), counts.index.max(), freq="h")
    series = counts.reindex(full_index, fill_value=0)

    frame = pd.DataFrame({TIME_COLUMN: series.index, TARGET_COLUMN: series.to_numpy()})
    logger.info(
        "Hourly series: %d hours from %s to %s, mean %.2f arrivals/hour, %d zero hours",
        len(frame),
        frame[TIME_COLUMN].min(),
        frame[TIME_COLUMN].max(),
        frame[TARGET_COLUMN].mean(),
        int((frame[TARGET_COLUMN] == 0).sum()),
    )
    return frame


class HistoryIndex:
    """Fast lookup of the arrival series by timestamp, for as-of-anchored features.

    Lookups go through `np.searchsorted` on the sorted timestamp array rather than a
    dict. That is both vectorised and immune to the datetime64-versus-Timestamp key
    mismatch that makes a dict silently return nothing.
    """

    def __init__(self, series: pd.DataFrame) -> None:
        frame = series.sort_values(TIME_COLUMN, kind="mergesort").reset_index(drop=True)
        self.timestamps = pd.to_datetime(frame[TIME_COLUMN]).to_numpy(dtype="datetime64[ns]")
        self.values = frame[TARGET_COLUMN].to_numpy(dtype="float64")
        self.cumulative = np.concatenate([[0.0], np.cumsum(self.values)])

    @staticmethod
    def _as_array(stamps: Iterable[pd.Timestamp]) -> np.ndarray:
        """Normalise any timestamp-like input to a datetime64[ns] array."""
        return pd.to_datetime(pd.Series(list(stamps))).to_numpy(dtype="datetime64[ns]")

    def _exact_position(self, stamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Index of each timestamp in the history, and whether it was found exactly."""
        position = np.searchsorted(self.timestamps, stamps, side="left")
        clipped = np.clip(position, 0, len(self.timestamps) - 1)
        found = (position < len(self.timestamps)) & (self.timestamps[clipped] == stamps)
        return clipped, found

    def value_at(self, stamps: Iterable[pd.Timestamp]) -> np.ndarray:
        """Observed count at each timestamp, NaN where outside the history."""
        array = self._as_array(stamps)
        if array.size == 0:
            return np.array([], dtype="float64")
        position, found = self._exact_position(array)
        return np.where(found, self.values[position], np.nan)

    def rolling_mean_ending_at(self, stamps: Iterable[pd.Timestamp], window: int) -> np.ndarray:
        """Mean of the `window` hours ending at (and including) each timestamp."""
        array = self._as_array(stamps)
        if array.size == 0:
            return np.array([], dtype="float64")

        # The last observation at or before each stamp bounds the window.
        end = np.searchsorted(self.timestamps, array, side="right") - 1
        usable = end >= 0
        safe_end = np.where(usable, end, 0)
        start = np.maximum(safe_end + 1 - window, 0)
        span = (safe_end + 1) - start
        totals = self.cumulative[safe_end + 1] - self.cumulative[start]
        return np.where(usable, totals / np.maximum(span, 1), np.nan)


def build_forecast_rows(
    history: HistoryIndex,
    as_of_values: Iterable[pd.Timestamp],
    horizons: Iterable[int],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build (as_of, horizon) feature rows with every lag anchored to `as_of`.

    Args:
        history: Index over the full arrival series.
        as_of_values: Forecast issue times.
        horizons: Lead times in hours.
        config: Full pipeline config; reads the `forecast` block.

    Returns:
        Long DataFrame with one row per (as_of, horizon), carrying the features, the
        target timestamp and -- where the target is inside the history -- the truth.
    """
    settings = config["forecast"]
    lags = list(settings["lags"])
    windows = list(settings["rolling_windows"])
    seasonal = int(settings["seasonal_lag_hours"])

    as_of_list = list(as_of_values)
    horizon_list = list(horizons)
    issue = np.repeat(np.array(as_of_list, dtype="datetime64[ns]"), len(horizon_list))
    horizon = np.tile(np.array(horizon_list, dtype="int64"), len(as_of_list))
    target = issue + horizon.astype("timedelta64[h]")

    frame = pd.DataFrame(
        {"as_of": pd.to_datetime(issue), "horizon": horizon, TIME_COLUMN: pd.to_datetime(target)}
    )

    # Lags anchored to as_of: lag_1 is the last hour that had already completed.
    for lag in lags:
        stamps = frame["as_of"] - pd.to_timedelta(lag - 1, unit="h")
        frame[f"lag_{lag}"] = history.value_at(stamps)
    for window in windows:
        frame[f"rolling_mean_{window}"] = history.rolling_mean_ending_at(frame["as_of"], window)

    # The seasonal lag is anchored to the TARGET, which is legitimate only while the
    # horizon is short enough that target - 168h is still before as_of.
    seasonal_stamps = frame[TIME_COLUMN] - pd.to_timedelta(seasonal, unit="h")
    seasonal_values = history.value_at(seasonal_stamps)
    knowable = seasonal_stamps <= frame["as_of"]
    frame[f"seasonal_lag_{seasonal}"] = np.where(knowable, seasonal_values, np.nan)
    if (~knowable).any():
        logger.debug(
            "%d row(s) have a horizon beyond the seasonal lag; that feature is left missing",
            int((~knowable).sum()),
        )

    frame = add_calendar_features(frame, config, time_column=TIME_COLUMN)
    frame[TARGET_COLUMN] = history.value_at(frame[TIME_COLUMN])
    return frame


def forecast_feature_names(config: dict[str, Any]) -> list[str]:
    """Columns used as forecaster inputs, in a stable order."""
    settings = config["forecast"]
    return [
        "horizon",
        *[f"lag_{lag}" for lag in settings["lags"]],
        *[f"rolling_mean_{w}" for w in settings["rolling_windows"]],
        f"seasonal_lag_{int(settings['seasonal_lag_hours'])}",
        *CALENDAR_FEATURES,
    ]


def minimum_history_hours(config: dict[str, Any]) -> int:
    """Hours of history the forecaster needs before it can produce a single row.

    The longest lag must fit behind the first usable `as_of`, and there must still be
    somewhere for a training target to land after it.
    """
    settings = config["forecast"]
    longest_lag = max(max(settings["lags"]), max(settings["rolling_windows"]))
    return int(longest_lag + int(settings["seasonal_lag_hours"]))


def history_is_sufficient(
    series: pd.DataFrame, config: dict[str, Any], train_end: pd.Timestamp
) -> tuple[bool, str]:
    """Check whether there is enough history before `train_end` to fit a forecaster.

    Args:
        series: The hourly arrival series.
        config: Full pipeline config.
        train_end: Exclusive upper bound on training targets.

    Returns:
        Tuple of (is sufficient, human-readable reason when it is not).
    """
    required = minimum_history_hours(config)
    available = int(
        (pd.Timestamp(train_end) - pd.Timestamp(series[TIME_COLUMN].min()))
        / pd.Timedelta(hours=1)
    )
    if available >= required:
        return True, ""
    reason = (
        f"Only {available} hours of history before {train_end}, but the configured lags "
        f"need at least {required} (longest lag plus one seasonal cycle of "
        f"{config['forecast']['seasonal_lag_hours']}h). Arrival forecasting needs "
        f"several weeks of data; shorten forecast.lags or supply a longer extract."
    )
    return False, reason


@dataclass
class SeasonalNaiveForecaster:
    """Baseline: the count at the same hour of the previous week.

    The standard baseline for hourly ED arrivals, and a strong one -- it carries both
    the daily and the weekly cycle for free.
    """

    lag_hours: int = 168
    history: HistoryIndex | None = field(default=None, repr=False)

    def fit(self, series: pd.DataFrame) -> "SeasonalNaiveForecaster":
        """Index the series; there are no parameters to estimate."""
        self.history = HistoryIndex(series)
        logger.info("Seasonal-naive baseline ready (lag %d hours)", self.lag_hours)
        return self

    def predict(self, timestamps: Iterable[pd.Timestamp]) -> np.ndarray:
        """Predict each timestamp as the value one seasonal lag earlier."""
        if self.history is None:
            raise RuntimeError("Call fit() before predict()")
        stamps = pd.to_datetime(pd.Series(list(timestamps))) - pd.to_timedelta(
            self.lag_hours, unit="h"
        )
        return self.history.value_at(stamps)


@dataclass
class ArrivalForecaster:
    """LightGBM direct multi-horizon forecaster with as-of-anchored features."""

    model: lgb.LGBMRegressor
    feature_names: list[str]
    history: HistoryIndex | None = field(default=None, repr=False)
    trained_through: pd.Timestamp | None = None
    calibrator: SplitConformalCalibrator | None = None
    band_edges: dict[str, tuple[int, int]] = field(default_factory=dict)

    def predict_frame(self, rows: pd.DataFrame) -> np.ndarray:
        """Predict from pre-built feature rows, clipped at zero."""
        return np.maximum(self.model.predict(rows.loc[:, self.feature_names]), 0.0)

    def hour_band(self, timestamps: Iterable[pd.Timestamp]) -> np.ndarray:
        """Map timestamps to their configured hour-of-day band."""
        hours = pd.to_datetime(pd.Series(list(timestamps))).dt.hour.to_numpy()
        labels = np.array(["unbanded"] * len(hours), dtype=object)
        for name, (start, end) in self.band_edges.items():
            labels[(hours >= start) & (hours < end)] = name
        return labels


def hour_band_edges(config: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """Read the hour-of-day bands used as Mondrian conformal groups."""
    return {
        name: (int(bounds[0]), int(bounds[1]))
        for name, bounds in config["forecast"]["hour_bands"].items()
    }


def train_forecaster(
    series: pd.DataFrame,
    config: dict[str, Any],
    train_end: pd.Timestamp,
    val_end: pd.Timestamp | None = None,
) -> ArrivalForecaster:
    """Fit the arrival forecaster on history up to `train_end`.

    Args:
        series: The full hourly series. Only hours before `train_end` are used for
            fitting; later hours remain in the index purely so that `forecast()` can
            look up seasonal lags when issuing a forecast later in time.
        config: Full pipeline config.
        train_end: Exclusive upper bound on target timestamps used for fitting.
        val_end: Optional early-stopping block end; when given, targets in
            [train_end, val_end) are used as the eval set.

    Returns:
        A fitted `ArrivalForecaster`.
    """
    settings = config["forecast"]
    horizon = int(settings["horizon_hours"])
    stride = int(settings["train_stride_hours"])
    history = HistoryIndex(series)

    # `as_of` values are strided to keep the design matrix manageable; every as_of must
    # leave room for the longest lag behind it.
    longest_lag = max(max(settings["lags"]), max(settings["rolling_windows"]))
    first_as_of = series[TIME_COLUMN].min() + pd.to_timedelta(longest_lag, unit="h")

    def rows_between(lower: pd.Timestamp, upper: pd.Timestamp) -> pd.DataFrame:
        """Build training rows whose TARGET lands in [lower, upper)."""
        as_of_values = pd.date_range(
            max(first_as_of, lower - pd.to_timedelta(horizon, unit="h")),
            upper - pd.to_timedelta(1, unit="h"),
            freq=f"{stride}h",
        )
        if len(as_of_values) == 0:
            return pd.DataFrame()
        built = build_forecast_rows(history, as_of_values, range(1, horizon + 1), config)
        inside = (built[TIME_COLUMN] >= lower) & (built[TIME_COLUMN] < upper)
        return built.loc[inside & built[TARGET_COLUMN].notna()].copy()

    train_rows = rows_between(series[TIME_COLUMN].min(), train_end)
    if train_rows.empty:
        raise ValueError(f"No training rows with a target before {train_end}")

    features = forecast_feature_names(config)
    params = dict(settings["model"])
    early_stopping_rounds = int(params.pop("early_stopping_rounds", 100))
    seed = derive_seed(config["seed"], "arrival_forecaster")
    model = lgb.LGBMRegressor(random_state=seed, verbose=-1, n_jobs=-1, **params)

    fit_kwargs: dict[str, Any] = {}
    val_rows = rows_between(train_end, val_end) if val_end is not None else pd.DataFrame()
    if not val_rows.empty:
        fit_kwargs["eval_set"] = [
            (val_rows.loc[:, features], val_rows[TARGET_COLUMN].to_numpy(dtype="float64"))
        ]
        fit_kwargs["callbacks"] = [
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ]

    model.fit(
        train_rows.loc[:, features],
        train_rows[TARGET_COLUMN].to_numpy(dtype="float64"),
        **fit_kwargs,
    )
    logger.info(
        "Fitted arrival forecaster on %d rows (targets before %s), %d validation rows, "
        "best iteration %s",
        len(train_rows),
        train_end,
        len(val_rows),
        getattr(model, "best_iteration_", None),
    )

    return ArrivalForecaster(
        model=model,
        feature_names=features,
        history=history,
        trained_through=pd.Timestamp(train_end),
        band_edges=hour_band_edges(config),
    )


def calibrate_forecaster(
    forecaster: ArrivalForecaster,
    series: pd.DataFrame,
    config: dict[str, Any],
    calib_start: pd.Timestamp,
    calib_end: pd.Timestamp,
) -> SplitConformalCalibrator:
    """Fit Mondrian conformal intervals for the forecaster, grouped by hour band.

    Calibration uses a block the model was neither fitted nor early-stopped on. Night
    hours are quiet and easy; midday hours are busy and variable. Pooling them gives
    intervals too wide at night and too narrow at the peak, which is backwards for
    staffing decisions.
    """
    rows = build_forecast_rows(
        HistoryIndex(series),
        pd.date_range(calib_start, calib_end, freq=f"{int(config['forecast']['train_stride_hours'])}h"),
        range(1, int(config["forecast"]["horizon_hours"]) + 1),
        config,
    )
    rows = rows.loc[
        (rows[TIME_COLUMN] >= calib_start)
        & (rows[TIME_COLUMN] < calib_end)
        & rows[TARGET_COLUMN].notna()
    ]
    if rows.empty:
        raise ValueError(f"No calibration rows between {calib_start} and {calib_end}")

    predictions = forecaster.predict_frame(rows)
    calibrator = fit_split_conformal(
        rows[TARGET_COLUMN].to_numpy(dtype="float64"),
        predictions,
        coverage=float(config["conformal"]["coverage"]),
        groups=forecaster.hour_band(rows[TIME_COLUMN]),
        min_group_size=int(config["conformal"]["min_group_size"]),
        lower_bound=0.0,  # an arrival count cannot be negative
    )
    forecaster.calibrator = calibrator
    return calibrator


def forecast(
    horizon_hours: int,
    as_of: pd.Timestamp,
    forecaster: ArrivalForecaster,
    series: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Public forecasting interface.

    Produces a forecast for the `horizon_hours` hours after `as_of`, using **only**
    observations strictly before `as_of`. The history is truncated at `as_of` before any
    feature is built, so a later observation cannot influence the result even if it is
    present in `series`. `tests/test_forecast.py` asserts this by comparing a forecast
    made from the full series against one made from a truncated copy.

    Args:
        horizon_hours: How many hours ahead to forecast.
        as_of: The moment the forecast is issued.
        forecaster: A fitted forecaster.
        series: The hourly arrival series; anything at or after `as_of` is ignored.
        config: Full pipeline config.

    Returns:
        DataFrame with `timestamp`, `yhat`, `lower`, `upper`, plus `horizon` and the
        hour band used for calibration.

    Raises:
        ValueError: If `horizon_hours` is not positive or the history is too short.
    """
    if horizon_hours <= 0:
        raise ValueError(f"horizon_hours must be positive, got {horizon_hours}")

    as_of = pd.Timestamp(as_of)
    # Everything at or after as_of is discarded before a single feature is computed.
    past = series.loc[pd.to_datetime(series[TIME_COLUMN]) < as_of]
    if past.empty:
        raise ValueError(f"No observations before as_of={as_of}; cannot forecast")

    longest_lag = max(
        max(config["forecast"]["lags"]), max(config["forecast"]["rolling_windows"])
    )
    if len(past) < longest_lag:
        logger.warning(
            "Only %d hours of history before %s, shorter than the longest lag (%d); "
            "early lag features will be missing",
            len(past),
            as_of,
            longest_lag,
        )

    history = HistoryIndex(past)
    # The last completed hour is the anchor, so lag_1 is a real observation.
    anchor = pd.Timestamp(past[TIME_COLUMN].max())
    lead = int((as_of - anchor) / pd.Timedelta(hours=1))
    rows = build_forecast_rows(
        history, [anchor], range(lead, lead + horizon_hours), config
    )
    rows = rows.loc[rows[TIME_COLUMN] >= as_of].head(horizon_hours)

    yhat = forecaster.predict_frame(rows)
    bands = forecaster.hour_band(rows[TIME_COLUMN])

    if forecaster.calibrator is not None:
        lower, upper = forecaster.calibrator.intervals(yhat, groups=bands)
    else:
        logger.warning("Forecaster has no calibrator; intervals will be NaN")
        lower = np.full(yhat.shape, np.nan)
        upper = np.full(yhat.shape, np.nan)

    out = pd.DataFrame(
        {
            "timestamp": rows[TIME_COLUMN].to_numpy(),
            "yhat": yhat,
            "lower": lower,
            "upper": upper,
            "horizon": rows["horizon"].to_numpy(),
            "hour_band": bands,
        }
    ).reset_index(drop=True)
    logger.info(
        "Forecast issued at %s for %d hours: mean %.2f arrivals/hour",
        as_of,
        len(out),
        float(out["yhat"].mean()),
    )
    return out


def rolling_origin_evaluate(
    series: pd.DataFrame,
    config: dict[str, Any],
    n_folds: int | None = None,
    step_hours: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Backtest by repeatedly moving the forecast origin forward.

    Each fold refits on everything before its origin and forecasts the next
    `horizon_hours`, so no fold ever sees its own future. Both the model and the
    seasonal-naive baseline are scored on identical rows.

    Args:
        series: The full hourly series.
        config: Full pipeline config.
        n_folds: Override for `forecast.rolling_origin.n_folds`.
        step_hours: Override for `forecast.rolling_origin.step_hours`.

    Returns:
        Tuple of (per-fold predictions, metrics summary).
    """
    settings = config["forecast"]
    folds = int(n_folds if n_folds is not None else settings["rolling_origin"]["n_folds"])
    step = int(step_hours if step_hours is not None else settings["rolling_origin"]["step_hours"])
    horizon = int(settings["horizon_hours"])

    last = pd.Timestamp(series[TIME_COLUMN].max())
    origins = [
        last - pd.to_timedelta(horizon + step * index, unit="h") for index in range(folds)
    ][::-1]

    calibration_hours = int(settings.get("calibration_hours", 336))

    predictions: list[pd.DataFrame] = []
    for fold, origin in enumerate(origins, start=1):
        # Within a fold the history is split again: the model is fitted strictly before
        # the calibration window, so the conformal residuals come from rows the model
        # never saw. Calibrating on fitted rows would make the intervals far too narrow.
        calib_start = origin - pd.to_timedelta(calibration_hours, unit="h")
        forecaster = train_forecaster(series, config, train_end=calib_start)
        baseline = SeasonalNaiveForecaster(
            lag_hours=int(settings["seasonal_lag_hours"])
        ).fit(series.loc[series[TIME_COLUMN] < origin])

        try:
            calibrate_forecaster(forecaster, series, config, calib_start, origin)
        except ValueError as error:
            logger.warning("Fold %d could not be calibrated (%s); intervals will be NaN", fold, error)

        issued = forecast(horizon, origin, forecaster, series, config)
        truth = series.set_index(TIME_COLUMN)[TARGET_COLUMN]
        issued["actual"] = issued["timestamp"].map(truth)
        issued["baseline"] = baseline.predict(issued["timestamp"])
        issued["fold"] = fold
        issued["origin"] = origin
        predictions.append(issued.dropna(subset=["actual"]))
        logger.info("Rolling-origin fold %d/%d at %s: %d hours scored", fold, folds, origin, len(issued))

    combined = pd.concat(predictions, ignore_index=True)
    summary = summarise_forecast(combined, config)
    return combined, summary


def summarise_forecast(predictions: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Score a backtest, overall, at peak hours, and by horizon band.

    Peak hours are the busiest `1 - forecast.peak_quantile` share of *observed* hours.
    They are reported separately because that is when a staffing decision actually
    bites: being wrong at 3 a.m. costs little.
    """
    actual = predictions["actual"].to_numpy(dtype="float64")
    model = predictions["yhat"].to_numpy(dtype="float64")
    baseline = predictions["baseline"].to_numpy(dtype="float64")

    peak_threshold = float(np.quantile(actual, float(config["forecast"]["peak_quantile"])))
    peak = actual >= peak_threshold

    summary: dict[str, Any] = {
        "n_hours": int(len(predictions)),
        "n_folds": int(predictions["fold"].nunique()),
        "peak_threshold_arrivals": peak_threshold,
        "n_peak_hours": int(peak.sum()),
        "overall": {
            "model": count_metrics(actual, model),
            "seasonal_naive": count_metrics(actual, baseline),
        },
        "peak_hours": {
            "model": count_metrics(actual[peak], model[peak]),
            "seasonal_naive": count_metrics(actual[peak], baseline[peak]),
        },
        "off_peak_hours": {
            "model": count_metrics(actual[~peak], model[~peak]),
            "seasonal_naive": count_metrics(actual[~peak], baseline[~peak]),
        },
    }

    by_horizon = []
    for lower, upper in ((1, 24), (25, 48), (49, 72)):
        mask = (predictions["horizon"] >= lower) & (predictions["horizon"] <= upper)
        if not mask.any():
            continue
        by_horizon.append(
            {
                "horizon_band": f"{lower}-{upper}h",
                "n": int(mask.sum()),
                **{f"model_{k}": v for k, v in count_metrics(actual[mask], model[mask]).items()},
                **{
                    f"naive_{k}": v
                    for k, v in count_metrics(actual[mask], baseline[mask]).items()
                },
            }
        )
    summary["by_horizon"] = by_horizon

    improvement = 100.0 * (
        1.0 - summary["overall"]["model"]["mae"] / summary["overall"]["seasonal_naive"]["mae"]
    )
    summary["mae_improvement_over_baseline_percent"] = round(improvement, 2)
    logger.info(
        "Forecast backtest: model MAE %.3f vs seasonal-naive %.3f (%.1f%% better); "
        "peak-hour MAE %.3f vs %.3f",
        summary["overall"]["model"]["mae"],
        summary["overall"]["seasonal_naive"]["mae"],
        improvement,
        summary["peak_hours"]["model"]["mae"],
        summary["peak_hours"]["seasonal_naive"]["mae"],
    )
    return summary


def evaluate_forecast_intervals(
    predictions: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    """Empirical interval coverage overall and by hour-of-day band."""
    usable = predictions.dropna(subset=["lower", "upper", "actual"])
    if usable.empty:
        logger.warning("No calibrated intervals to evaluate")
        return {}
    return evaluate_intervals(
        usable["actual"].to_numpy(dtype="float64"),
        usable["lower"].to_numpy(dtype="float64"),
        usable["upper"].to_numpy(dtype="float64"),
        coverage=float(config["conformal"]["coverage"]),
        groups=usable["hour_band"].to_numpy(),
    )
