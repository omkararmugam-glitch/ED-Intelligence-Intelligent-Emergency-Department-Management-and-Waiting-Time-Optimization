"""Why the forecaster is worse than seasonal-naive at the peak, and what fixes it.

**The weakness.** On the default 4-fold backtest the LightGBM forecaster beats the
seasonal-naive baseline overall -- MAE 2.356 against 2.979 -- but *loses* at peak hours,
4.364 against 4.100, and its intervals under-cover in the morning band (0.806 against a
nominal 0.90). Peak hours are the only hours where the forecast drives a staffing
decision, so losing there is the result that matters.

**Why the default backtest cannot settle it.** Four folds of 72 hours is 288 scored
hours, of which about 60 are peak. A 0.26-arrival MAE difference over 60 hours is not
something four folds can resolve. The first thing this module does is replace that with
a rolling-origin backtest over at least 26 weekly origins, which is roughly 1,900 scored
hours and 380 peak hours, and attach paired bootstrap intervals to every comparison so
the reader can see whether a difference is real.

**What it compares.** Five candidates, every one of them scored on identical rows:

===================  =======================================================
``poisson``          the shipped configuration; the control
``tweedie``          a Tweedie objective instead of Poisson
``same_hour_lags``   Poisson plus same-hour-previous-day and
                     same-hour-previous-week rolling means
``blend``            a convex blend of ``poisson`` and seasonal-naive, weight
                     fitted on the validation origins only
``peak_corrected``   ``poisson`` with a multiplicative correction applied to
                     predicted-peak hours, fitted on the validation origins only
===================  =======================================================

**The selection rule.** The origins are split in time: the earlier half are
*validation* origins and the later half are *test* origins. Every tuning decision --
the blend weight, the peak correction factor, and the choice of winner -- is made on
the validation origins alone, and the test origins are scored once.

**What this module deliberately does not do.** It does not change the shipped
forecaster. `artifacts/` reproduces a documented set of headline numbers, forecast MAE
2.356 among them, and a module that silently adopted a new objective would move one of
them without anybody choosing to. If a variant wins on validation, this module says so
in its artifacts and the decision to adopt it is a separate, explicit change -- see
`tests/test_headline_regression.py`.

No new libraries: everything here is LightGBM, NumPy and pandas, all already required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.models.forecast import (
    TARGET_COLUMN,
    TIME_COLUMN,
    HistoryIndex,
    SeasonalNaiveForecaster,
    build_forecast_rows,
    forecast_feature_names,
    hour_band_edges,
)
from src.utils.logging import get_logger
from src.utils.seed import derive_seed, rng as make_rng

logger = get_logger(__name__)

#: Minimum number of weekly origins. Four folds could not resolve the peak-hour
#: question; this is the number that makes the comparison worth running.
MIN_ORIGINS = 26

#: Lookbacks for the same-hour features, in days and weeks respectively.
SAME_HOUR_DAY_LAGS = (1, 2, 3, 4)
SAME_HOUR_WEEK_LAGS = (1, 2, 3, 4)

#: Feature names those lookbacks produce.
SAME_HOUR_FEATURES = ("same_hour_prev_days_mean", "same_hour_prev_weeks_mean")


@dataclass(frozen=True)
class VariantSpec:
    """One forecaster candidate.

    Attributes:
        name: Identifier used in every table.
        description: One line for the report.
        objective: LightGBM objective, when this variant refits a model.
        same_hour_features: Whether to add the same-hour lookback features.
        derives_from: For post-hoc variants, the refitted variant they transform.
            Post-hoc variants cost no extra fits, which is why the blend and the
            peak correction are built this way.
    """

    name: str
    description: str
    objective: str = "poisson"
    same_hour_features: bool = False
    derives_from: str | None = None

    @property
    def needs_refit(self) -> bool:
        """True when this variant fits its own model at every origin."""
        return self.derives_from is None


#: The candidates, control first.
VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec(
        name="poisson",
        description="the shipped configuration: LightGBM with a Poisson objective",
    ),
    VariantSpec(
        name="tweedie",
        description="a Tweedie objective, which allows more spread than Poisson",
        objective="tweedie",
    ),
    VariantSpec(
        name="same_hour_lags",
        description=(
            "Poisson plus same-hour-previous-day and same-hour-previous-week "
            "rolling means"
        ),
        same_hour_features=True,
    ),
    VariantSpec(
        name="blend",
        description=(
            "convex blend of the Poisson model and seasonal-naive, weight fitted "
            "on the validation origins only"
        ),
        derives_from="poisson",
    ),
    VariantSpec(
        name="peak_corrected",
        description=(
            "the Poisson model with a multiplicative correction on predicted-peak "
            "hours, fitted on the validation origins only"
        ),
        derives_from="poisson",
    ),
)


def add_same_hour_features(
    frame: pd.DataFrame, history: HistoryIndex, config: dict[str, Any]
) -> pd.DataFrame:
    """Add same-hour-previous-day and same-hour-previous-week rolling means.

    For a row forecasting target time ``T`` from issue time ``as_of``, these are the
    mean count at ``T - 24h, T - 48h, ...`` and at ``T - 168h, T - 336h, ...``.

    **Leakage boundary.** Both lookbacks are anchored to the *target*, not to `as_of`,
    which is only legitimate while the observation is already in the past at issue
    time. Every lookback is therefore masked to ``<= as_of`` and contributes nothing
    when it is not, exactly as the existing seasonal-lag feature is. With the default
    72-hour horizon the one-day lookbacks are partly masked near the end of the horizon
    and the one-week lookbacks are not masked at all.

    Args:
        frame: Rows from `build_forecast_rows`, carrying `as_of` and the target time.
        history: Index over the arrival series.
        config: Full pipeline config (unused today; kept so the lookbacks can move to
            the config without changing every caller).

    Returns:
        A copy of `frame` with `SAME_HOUR_FEATURES` added.
    """
    out = frame.copy()
    target = pd.to_datetime(out[TIME_COLUMN])
    as_of = pd.to_datetime(out["as_of"])

    for column, lags, unit_hours in (
        ("same_hour_prev_days_mean", SAME_HOUR_DAY_LAGS, 24),
        ("same_hour_prev_weeks_mean", SAME_HOUR_WEEK_LAGS, 168),
    ):
        collected = np.full((len(out), len(lags)), np.nan)
        for index, multiple in enumerate(lags):
            stamps = target - pd.to_timedelta(multiple * unit_hours, unit="h")
            values = history.value_at(stamps)
            # Anything not yet observed at issue time is not available to the model.
            collected[:, index] = np.where(stamps <= as_of, values, np.nan)
        with np.errstate(invalid="ignore"):
            out[column] = np.nanmean(collected, axis=1)

    return out


def study_feature_names(config: dict[str, Any], variant: VariantSpec) -> list[str]:
    """Feature columns for one variant, in a stable order."""
    names = list(forecast_feature_names(config))
    if variant.same_hour_features:
        names.extend(SAME_HOUR_FEATURES)
    return names


def weekly_origins(
    series: pd.DataFrame, config: dict[str, Any], n_origins: int = MIN_ORIGINS
) -> list[pd.Timestamp]:
    """Forecast issue times, one per week, ending at the last fully-scorable origin.

    Args:
        series: The hourly arrival series.
        config: Full pipeline config.
        n_origins: How many origins to produce.

    Returns:
        Origins in chronological order.

    Raises:
        ValueError: If the series is too short to supply `n_origins` origins with
            enough history behind the first one to fit a model.
    """
    from src.models.forecast import minimum_history_hours

    settings = config["forecast"]
    horizon = int(settings["horizon_hours"])
    step = int(settings["rolling_origin"]["step_hours"])

    first_observation = pd.Timestamp(series[TIME_COLUMN].min())
    last = pd.Timestamp(series[TIME_COLUMN].max())
    origins = [
        last - pd.to_timedelta(horizon + step * index, unit="h")
        for index in range(int(n_origins))
    ][::-1]

    # Each origin refits on everything before its calibration window, so the earliest
    # origin still needs a model's worth of history behind it.
    needed = minimum_history_hours(config) + int(settings.get("calibration_hours", 336))
    earliest_usable = first_observation + pd.to_timedelta(needed, unit="h")
    usable = [origin for origin in origins if origin >= earliest_usable]

    if len(usable) < int(n_origins):
        raise ValueError(
            f"The series spans {first_observation.date()} to {last.date()}, which "
            f"supports {len(usable)} weekly origin(s) with {needed}h of history behind "
            f"each, but {n_origins} were requested. Use a longer extract or fewer "
            "origins; this module will not score an origin it cannot fit honestly."
        )
    return usable


def split_origins(
    origins: Sequence[pd.Timestamp], validation_fraction: float = 0.5
) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    """Split origins in time: earlier ones tune, later ones report.

    Args:
        origins: Origins in chronological order.
        validation_fraction: Share of origins used for tuning.

    Returns:
        Tuple of (validation origins, test origins).

    Raises:
        ValueError: If either side would be empty.
    """
    cut = int(round(len(origins) * float(validation_fraction)))
    validation, test = list(origins[:cut]), list(origins[cut:])
    if not validation or not test:
        raise ValueError(
            f"{len(origins)} origin(s) cannot be split into a validation and a test "
            f"half at fraction {validation_fraction}"
        )
    logger.info(
        "Origins split in time: %d validation (%s..%s), %d test (%s..%s)",
        len(validation), validation[0].date(), validation[-1].date(),
        len(test), test[0].date(), test[-1].date(),
    )
    return validation, test


def _fit_one(
    series: pd.DataFrame,
    config: dict[str, Any],
    variant: VariantSpec,
    train_end: pd.Timestamp,
    history: HistoryIndex,
) -> tuple[Any, list[str]]:
    """Fit one variant's model on targets strictly before `train_end`.

    Mirrors `src.models.forecast.train_forecaster`, but parameterised by objective and
    by whether the same-hour features are included, and returning the raw booster
    because this module scores predictions rather than serving them.

    Args:
        series: The full hourly series.
        config: Full pipeline config.
        variant: The candidate being fitted.
        train_end: Exclusive upper bound on training targets.
        history: Index over `series`.

    Returns:
        Tuple of (fitted model, feature names).

    Raises:
        ValueError: If no training rows have a target before `train_end`.
    """
    import lightgbm as lgb

    settings = config["forecast"]
    horizon = int(settings["horizon_hours"])
    stride = int(settings["train_stride_hours"])
    longest_lag = max(max(settings["lags"]), max(settings["rolling_windows"]))
    first_as_of = series[TIME_COLUMN].min() + pd.to_timedelta(longest_lag, unit="h")

    as_of_values = pd.date_range(
        first_as_of, train_end - pd.to_timedelta(1, unit="h"), freq=f"{stride}h"
    )
    if len(as_of_values) == 0:
        raise ValueError(f"No usable as_of values before {train_end}")

    rows = build_forecast_rows(history, as_of_values, range(1, horizon + 1), config)
    if variant.same_hour_features:
        rows = add_same_hour_features(rows, history, config)
    rows = rows.loc[(rows[TIME_COLUMN] < train_end) & rows[TARGET_COLUMN].notna()]
    if rows.empty:
        raise ValueError(f"No training rows with a target before {train_end}")

    features = study_feature_names(config, variant)
    params = dict(settings["model"])
    params.pop("early_stopping_rounds", None)
    params["objective"] = variant.objective

    model = lgb.LGBMRegressor(
        random_state=derive_seed(config["seed"], f"forecast_study::{variant.name}"),
        verbose=-1,
        n_jobs=-1,
        **params,
    )
    model.fit(rows.loc[:, features], rows[TARGET_COLUMN].to_numpy(dtype="float64"))
    return model, features


def run_backtest(
    series: pd.DataFrame,
    config: dict[str, Any],
    origins: Sequence[pd.Timestamp],
    variants: Sequence[VariantSpec] = VARIANTS,
) -> pd.DataFrame:
    """Score every refitting variant and the seasonal-naive baseline at every origin.

    Each origin refits from scratch on history strictly before its calibration window,
    so no origin ever sees its own future, and every variant is scored on identical
    rows.

    Post-hoc variants (those with `derives_from`) are not fitted here; they are built
    from these predictions by `apply_post_hoc_variants` once the tuning constants have
    been fitted on the validation origins.

    Args:
        series: The full hourly series.
        config: Full pipeline config.
        origins: Forecast issue times.
        variants: Candidates to score.

    Returns:
        Long frame: one row per (origin, target hour, variant), with `actual`,
        `yhat`, `baseline` and the hour band.
    """
    settings = config["forecast"]
    horizon = int(settings["horizon_hours"])
    calibration_hours = int(settings.get("calibration_hours", 336))
    bands = hour_band_edges(config)
    history = HistoryIndex(series)
    truth = series.set_index(TIME_COLUMN)[TARGET_COLUMN]

    refitting = [variant for variant in variants if variant.needs_refit]
    collected: list[pd.DataFrame] = []

    for index, origin in enumerate(origins, start=1):
        # Same discipline as the shipped backtest: fit strictly before the conformal
        # calibration window, so interval residuals come from unseen rows.
        train_end = origin - pd.to_timedelta(calibration_hours, unit="h")

        # Features are built from a history truncated at the origin, exactly as
        # `src.models.forecast.forecast` does, and anchored on the last *completed*
        # hour. Anchoring on the origin itself would let a 24-hour-ahead row read the
        # origin hour through its same-hour lookback, and that hour has not finished.
        past = series.loc[pd.to_datetime(series[TIME_COLUMN]) < origin]
        if past.empty:
            logger.warning("Origin %s has no history before it; skipping", origin)
            continue
        past_history = HistoryIndex(past)
        anchor = pd.Timestamp(past[TIME_COLUMN].max())
        lead = int((origin - anchor) / pd.Timedelta(hours=1))

        rows = build_forecast_rows(
            past_history, [anchor], range(lead, lead + horizon), config
        )
        rows = add_same_hour_features(rows, past_history, config)
        rows = rows.loc[rows[TIME_COLUMN] >= origin].head(horizon).copy()
        rows["actual"] = rows[TIME_COLUMN].map(truth)
        rows = rows.dropna(subset=["actual"])
        if rows.empty:
            logger.warning("Origin %s scored no hours; skipping", origin)
            continue

        baseline = SeasonalNaiveForecaster(
            lag_hours=int(settings["seasonal_lag_hours"])
        ).fit(series.loc[series[TIME_COLUMN] < origin])
        baseline_prediction = baseline.predict(rows[TIME_COLUMN])

        for variant in refitting:
            model, features = _fit_one(series, config, variant, train_end, history)
            prediction = np.maximum(model.predict(rows.loc[:, features]), 0.0)
            frame = pd.DataFrame(
                {
                    "origin": origin,
                    "timestamp": rows[TIME_COLUMN].to_numpy(),
                    "horizon": rows["horizon"].to_numpy(),
                    "variant": variant.name,
                    "actual": rows["actual"].to_numpy(dtype="float64"),
                    "yhat": prediction,
                    "baseline": baseline_prediction,
                }
            )
            collected.append(frame)

        logger.info(
            "Backtest origin %d/%d at %s: %d hour(s) x %d variant(s)",
            index, len(origins), origin, len(rows), len(refitting),
        )

    if not collected:
        return pd.DataFrame()

    combined = pd.concat(collected, ignore_index=True)
    hours = pd.to_datetime(combined["timestamp"]).dt.hour.to_numpy()
    band = np.array(["unbanded"] * len(combined), dtype=object)
    for name, (start, end) in bands.items():
        band[(hours >= start) & (hours < end)] = name
    combined["hour_band"] = band
    combined["hour_of_day"] = hours
    return combined


def peak_threshold(predictions: pd.DataFrame, config: dict[str, Any]) -> float:
    """The arrival count above which an hour counts as peak.

    Taken from the *observed* counts on the given rows, exactly as
    `src.models.forecast.summarise_forecast` does, so the two agree on what "peak"
    means.
    """
    actual = predictions["actual"].to_numpy(dtype="float64")
    return float(np.quantile(actual, float(config["forecast"]["peak_quantile"])))


def fit_blend_weight(
    validation: pd.DataFrame, grid: Iterable[float] | None = None
) -> tuple[float, pd.DataFrame]:
    """Choose the convex blend weight that minimises MAE on the validation origins.

    Args:
        validation: Rows for the `poisson` variant on the validation origins only.
        grid: Candidate weights on the model (1.0 = model alone).

    Returns:
        Tuple of (chosen weight, the full grid with its validation MAE).
    """
    weights = list(grid if grid is not None else np.round(np.arange(0.0, 1.01, 0.05), 2))
    actual = validation["actual"].to_numpy(dtype="float64")
    model = validation["yhat"].to_numpy(dtype="float64")
    baseline = validation["baseline"].to_numpy(dtype="float64")

    rows = []
    for weight in weights:
        blended = weight * model + (1.0 - weight) * baseline
        rows.append({"weight_on_model": float(weight), "val_mae": float(np.mean(np.abs(actual - blended)))})
    table = pd.DataFrame(rows).sort_values("val_mae", ignore_index=True)
    chosen = float(table.iloc[0]["weight_on_model"])
    logger.info(
        "Blend weight chosen on validation origins: %.2f on the model, %.2f on "
        "seasonal-naive (validation MAE %.4f)",
        chosen, 1.0 - chosen, float(table.iloc[0]["val_mae"]),
    )
    return chosen, table


def fit_peak_correction(
    validation: pd.DataFrame, threshold: float
) -> tuple[float, dict[str, Any]]:
    """Fit one multiplicative correction for hours the model predicts to be busy.

    The correction is applied where the *prediction* is above the threshold, never
    where the truth is: at serving time the truth is not known. That distinction turns
    out to matter more than it looks.

    **Conditioning on the prediction flips the sign of the bias.** Measured on
    `data/raw/ed_v2.csv` over 26 origins, the model under-predicts hours that turn out
    to be busy (bias -2.44 at actual-peak hours) but *over*-predicts hours it expects
    to be busy (bias +1.52 at predicted-peak hours, so the fitted multiplier is 0.85).
    Both are the same regression-to-the-mean effect seen from opposite ends, and only
    the second is observable at serving time.

    The consequence, which this study measures rather than assumes: correcting on the
    predicted-peak set drives aggregate bias on that set to zero while making the error
    on *actual* peak hours worse, because it scales down exactly the predictions that
    were already too low for the hours that mattered. It is kept in the comparison as a
    measured negative result.

    Args:
        validation: Rows for the `poisson` variant on the validation origins only.
        threshold: Arrival count above which an hour counts as peak.

    Returns:
        Tuple of (multiplier, a summary of how it was fitted).
    """
    predicted_peak = validation["yhat"].to_numpy(dtype="float64") >= threshold
    actual = validation["actual"].to_numpy(dtype="float64")[predicted_peak]
    model = validation["yhat"].to_numpy(dtype="float64")[predicted_peak]

    if model.size == 0 or model.sum() == 0:
        logger.warning(
            "No predicted-peak hours on the validation origins; the peak correction "
            "is left at 1.0 (no correction)"
        )
        return 1.0, {"n_validation_peak_hours": 0, "reason": "no predicted-peak hours"}

    # Ratio of totals rather than mean of ratios: it is the multiplier that makes the
    # corrected predictions unbiased in aggregate over the peak hours.
    multiplier = float(actual.sum() / model.sum())
    summary = {
        "n_validation_peak_hours": int(predicted_peak.sum()),
        "threshold_arrivals": round(float(threshold), 4),
        "multiplier": round(multiplier, 4),
        "validation_peak_bias_before": round(float(np.mean(model - actual)), 4),
        "validation_peak_bias_after": round(float(np.mean(multiplier * model - actual)), 4),
    }
    logger.info(
        "Peak correction fitted on validation origins: x%.4f over %d predicted-peak "
        "hour(s) (bias %+.3f -> %+.3f)",
        multiplier,
        summary["n_validation_peak_hours"],
        summary["validation_peak_bias_before"],
        summary["validation_peak_bias_after"],
    )
    return multiplier, summary


def apply_post_hoc_variants(
    predictions: pd.DataFrame,
    blend_weight: float,
    peak_multiplier: float,
    threshold: float,
    variants: Sequence[VariantSpec] = VARIANTS,
) -> pd.DataFrame:
    """Build the blend and peak-corrected variants from the fitted base predictions.

    Args:
        predictions: Output of `run_backtest`.
        blend_weight: Weight on the model, fitted on validation origins.
        peak_multiplier: Peak correction, fitted on validation origins.
        threshold: Peak threshold in arrivals.
        variants: The full variant list.

    Returns:
        `predictions` with the post-hoc variants' rows appended.
    """
    derived: list[pd.DataFrame] = []
    for variant in variants:
        if variant.needs_refit:
            continue
        source = predictions[predictions["variant"] == variant.derives_from]
        if source.empty:
            logger.warning(
                "Variant %r derives from %r, which produced no rows; skipping",
                variant.name, variant.derives_from,
            )
            continue

        frame = source.copy()
        frame["variant"] = variant.name
        if variant.name == "blend":
            frame["yhat"] = (
                blend_weight * source["yhat"].to_numpy(dtype="float64")
                + (1.0 - blend_weight) * source["baseline"].to_numpy(dtype="float64")
            )
        elif variant.name == "peak_corrected":
            prediction = source["yhat"].to_numpy(dtype="float64")
            frame["yhat"] = np.where(
                prediction >= threshold, prediction * peak_multiplier, prediction
            )
        else:
            logger.warning("No post-hoc rule implemented for %r; skipping", variant.name)
            continue
        derived.append(frame)

    if not derived:
        return predictions
    return pd.concat([predictions, *derived], ignore_index=True)


def _mae(actual: np.ndarray, prediction: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(actual - prediction))) if actual.size else float("nan")


def _wape(actual: np.ndarray, prediction: np.ndarray) -> float:
    """Weighted absolute percentage error: total absolute error over total volume.

    Preferred to MAPE for counts, because MAPE is undefined on a zero-arrival hour and
    explodes on a one-arrival hour.
    """
    total = float(np.sum(np.abs(actual)))
    if total == 0:
        return float("nan")
    return 100.0 * float(np.sum(np.abs(actual - prediction))) / total


def paired_bootstrap_difference(
    actual: np.ndarray,
    model: np.ndarray,
    reference: np.ndarray,
    seed: int,
    label: str,
    n_resamples: int = 2000,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap interval for the MAE difference between two forecasters.

    Paired and resampled by *hour*: both forecasters are scored on the same resampled
    rows every time, so the interval is about the difference between them rather than
    about how hard the resampled hours happened to be.

    A negative difference means the model has the lower error -- it is better.

    Args:
        actual: Observed counts.
        model: The candidate's predictions.
        reference: The comparator's predictions, usually seasonal-naive.
        seed: Global seed, for a derived stream.
        label: Mixed into the stream so each comparison is independent.
        n_resamples: Bootstrap resamples.
        confidence: Interval level.

    Returns:
        A dict with both MAEs, the difference, its interval, and whether the interval
        excludes zero.
    """
    if actual.size == 0:
        return {"n": 0, "note": "no rows to compare"}

    model_error = np.abs(actual - model)
    reference_error = np.abs(actual - reference)
    observed = float(model_error.mean() - reference_error.mean())

    generator = make_rng(seed, f"forecast_study::bootstrap::{label}")
    draws = np.empty(n_resamples)
    for index in range(n_resamples):
        picked = generator.integers(0, actual.size, actual.size)
        draws[index] = model_error[picked].mean() - reference_error[picked].mean()

    tail = (1.0 - confidence) / 2.0
    low = float(np.percentile(draws, 100 * tail))
    high = float(np.percentile(draws, 100 * (1 - tail)))
    return {
        "n": int(actual.size),
        "model_mae": round(float(model_error.mean()), 4),
        "reference_mae": round(float(reference_error.mean()), 4),
        "mae_difference": round(observed, 4),
        "ci_low": round(low, 4),
        "ci_high": round(high, 4),
        "significant": bool(low > 0 or high < 0),
        "model_is_better": bool(observed < 0),
    }


def score_variants(
    predictions: pd.DataFrame,
    config: dict[str, Any],
    threshold: float,
    block: str,
) -> pd.DataFrame:
    """Score every variant overall and at peak hours, against seasonal-naive.

    Args:
        predictions: Backtest rows for one block (validation or test).
        config: Full pipeline config.
        threshold: Peak threshold in arrivals, fixed across blocks so the two are
            comparable.
        block: Name recorded on every row.

    Returns:
        One row per (variant, scope).
    """
    seed = int(config["seed"])
    rows: list[dict[str, Any]] = []

    for variant_name, group in predictions.groupby("variant", sort=False):
        actual = group["actual"].to_numpy(dtype="float64")
        model = group["yhat"].to_numpy(dtype="float64")
        baseline = group["baseline"].to_numpy(dtype="float64")
        is_peak = actual >= threshold

        for scope, mask in (("overall", np.ones(actual.size, dtype=bool)), ("peak", is_peak)):
            if not mask.any():
                continue
            comparison = paired_bootstrap_difference(
                actual[mask], model[mask], baseline[mask], seed, f"{block}:{variant_name}:{scope}"
            )
            rows.append(
                {
                    "block": block,
                    "variant": variant_name,
                    "scope": scope,
                    "n_hours": int(mask.sum()),
                    "mae": round(_mae(actual[mask], model[mask]), 4),
                    "wape": round(_wape(actual[mask], model[mask]), 4),
                    "bias": round(float(np.mean(model[mask] - actual[mask])), 4),
                    "rmse": round(
                        float(np.sqrt(np.mean((model[mask] - actual[mask]) ** 2))), 4
                    ),
                    "seasonal_naive_mae": round(_mae(actual[mask], baseline[mask]), 4),
                    "seasonal_naive_wape": round(_wape(actual[mask], baseline[mask]), 4),
                    "mae_vs_naive": comparison.get("mae_difference"),
                    "mae_vs_naive_ci_low": comparison.get("ci_low"),
                    "mae_vs_naive_ci_high": comparison.get("ci_high"),
                    "beats_naive": comparison.get("model_is_better"),
                    "difference_is_significant": comparison.get("significant"),
                }
            )

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(["scope", "mae"], ignore_index=True)
    return table


def rolling_conformal_intervals(
    predictions: pd.DataFrame,
    config: dict[str, Any],
    variant: str,
    min_calibration: int = 60,
) -> pd.DataFrame:
    """Calibrate forecast intervals from residuals already observed at issue time.

    The same idea as `src.models.rolling_conformal` for waits, transplanted to the
    forecast: an interval issued at origin *t* is sized from the absolute residuals of
    forecasts issued at **earlier** origins, within the same hour-of-day band.

    **The leakage boundary.** A residual for target hour *T* is only knowable once *T*
    has happened. A forecast issued at origin *t* may therefore use residuals whose
    target time is strictly before *t*, and nothing else. The earliest origins have no
    such history and are reported with no interval rather than with a borrowed one.

    Fallbacks, in order, recorded per row:
      1. same hour band, targets before this origin;
      2. all bands pooled, targets before this origin;
      3. no interval -- too little history to size one honestly.

    Args:
        predictions: Backtest rows, all variants.
        config: Full pipeline config.
        variant: Which variant's intervals to build.
        min_calibration: Residuals needed before a quantile is trusted.

    Returns:
        The variant's rows with `lower`, `upper`, `covered` and `interval_source`.
    """
    coverage = float(config["conformal"]["coverage"])
    frame = predictions[predictions["variant"] == variant].copy()
    if frame.empty:
        return frame

    frame = frame.sort_values(["origin", "timestamp"], ignore_index=True)
    frame["abs_residual"] = np.abs(
        frame["actual"].to_numpy(dtype="float64") - frame["yhat"].to_numpy(dtype="float64")
    )

    origins = sorted(frame["origin"].unique())
    lower = np.full(len(frame), np.nan)
    upper = np.full(len(frame), np.nan)
    source = np.array(["none: too little history"] * len(frame), dtype=object)

    for origin in origins:
        current = frame["origin"].to_numpy() == origin
        # Only residuals whose target hour had already happened when this origin was
        # issued are available to size the interval.
        observed = frame["timestamp"].to_numpy() < np.datetime64(pd.Timestamp(origin))
        if not observed.any():
            continue

        for band in frame.loc[current, "hour_band"].unique():
            rows = current & (frame["hour_band"].to_numpy() == band)
            same_band = observed & (frame["hour_band"].to_numpy() == band)
            if int(same_band.sum()) >= min_calibration:
                residuals = frame.loc[same_band, "abs_residual"].to_numpy(dtype="float64")
                label = f"same hour band ({band}), {int(same_band.sum())} residuals"
            elif int(observed.sum()) >= min_calibration:
                residuals = frame.loc[observed, "abs_residual"].to_numpy(dtype="float64")
                label = f"pooled bands, {int(observed.sum())} residuals"
            else:
                continue

            # Conformal quantile with the finite-sample correction.
            n = residuals.size
            level = min(1.0, np.ceil((n + 1) * coverage) / n)
            half_width = float(np.quantile(residuals, level))
            prediction = frame.loc[rows, "yhat"].to_numpy(dtype="float64")
            lower[rows] = np.maximum(prediction - half_width, 0.0)
            upper[rows] = prediction + half_width
            source[rows] = label

    frame["lower"] = lower
    frame["upper"] = upper
    frame["interval_source"] = source
    actual = frame["actual"].to_numpy(dtype="float64")
    frame["covered"] = np.where(
        np.isnan(lower), np.nan, (actual >= lower) & (actual <= upper)
    )
    return frame


def coverage_by_band(
    intervals: pd.DataFrame, config: dict[str, Any], block: str
) -> pd.DataFrame:
    """Empirical interval coverage overall and per hour-of-day band.

    Args:
        intervals: Output of `rolling_conformal_intervals`.
        config: Full pipeline config.
        block: Name recorded on every row.

    Returns:
        One row per scope, with the gap against nominal and whether it is inside the
        configured tolerance.
    """
    nominal = float(config["conformal"]["coverage"])
    tolerance = float(config["conformal"]["tolerance"])
    scored = intervals.dropna(subset=["covered"])

    rows: list[dict[str, Any]] = []

    def add(scope: str, frame: pd.DataFrame) -> None:
        """Score one slice."""
        if frame.empty:
            return
        empirical = float(frame["covered"].astype(float).mean())
        rows.append(
            {
                "block": block,
                "scope": scope,
                "n": int(len(frame)),
                "nominal_coverage": nominal,
                "empirical_coverage": round(empirical, 4),
                "coverage_gap": round(empirical - nominal, 4),
                "within_tolerance": bool(abs(empirical - nominal) <= tolerance),
                "tolerance": tolerance,
                "mean_width": round(float((frame["upper"] - frame["lower"]).mean()), 4),
            }
        )

    add("overall", scored)
    for band in sorted(scored["hour_band"].unique()):
        add(band, scored[scored["hour_band"] == band])

    table = pd.DataFrame(rows)
    if not table.empty:
        missed = table[~table["within_tolerance"]]
        if missed.empty:
            logger.info(
                "Rolling conformal: every hour band is within %.2f of nominal %.2f",
                tolerance, nominal,
            )
        else:
            logger.warning(
                "Rolling conformal: %d band(s) outside the %.2f tolerance: %s",
                len(missed),
                tolerance,
                ", ".join(
                    f"{row.scope} {row.empirical_coverage:.3f}" for row in missed.itertuples()
                ),
            )
    return table


@dataclass
class StudyResult:
    """Everything the study produced, ready to be written out."""

    predictions: pd.DataFrame
    scores: pd.DataFrame
    coverage: pd.DataFrame
    blend_grid: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)


def select_by_validation(scores: pd.DataFrame, scope: str = "peak") -> dict[str, Any]:
    """Pick the winner on the validation block alone.

    The selection scope is the peak hours by default, because peak-hour accuracy is the
    weakness being investigated and the only place where the forecast drives a staffing
    decision. The overall ranking is reported alongside, so a variant that wins the
    peak by wrecking everything else is visible.

    Args:
        scores: Output of `score_variants` for both blocks.
        scope: Which scope the choice is made on.

    Returns:
        A dict naming the winner, its validation figures, and the runner-up.

    Raises:
        ValueError: If there are no validation rows at the requested scope.
    """
    validation = scores[(scores["block"] == "validation") & (scores["scope"] == scope)]
    if validation.empty:
        raise ValueError(
            f"No validation scores at scope {scope!r}; the winner cannot be chosen "
            "without them, and choosing on test would invalidate the whole study."
        )

    ordered = validation.sort_values("mae", ignore_index=True)
    winner = ordered.iloc[0]
    result = {
        "selection_scope": scope,
        "selection_block": "validation",
        "winner": str(winner["variant"]),
        "winner_val_mae": float(winner["mae"]),
        "winner_val_seasonal_naive_mae": float(winner["seasonal_naive_mae"]),
        "ranking": ordered[["variant", "mae", "seasonal_naive_mae"]].to_dict(orient="records"),
    }
    if len(ordered) > 1:
        result["runner_up"] = str(ordered.iloc[1]["variant"])
        result["margin_over_runner_up"] = round(
            float(ordered.iloc[1]["mae"] - winner["mae"]), 4
        )
    logger.info(
        "Validation winner at %s hours: %r (MAE %.4f vs seasonal-naive %.4f)",
        scope, result["winner"], result["winner_val_mae"],
        result["winner_val_seasonal_naive_mae"],
    )
    return result


def run_forecast_study(
    series: pd.DataFrame,
    config: dict[str, Any],
    n_origins: int = MIN_ORIGINS,
    variants: Sequence[VariantSpec] = VARIANTS,
) -> StudyResult:
    """Run the whole study: backtest, tune on validation, report on test.

    Args:
        series: The full hourly arrival series.
        config: Full pipeline config.
        n_origins: Weekly origins to score; at least `MIN_ORIGINS`.
        variants: Candidates to compare.

    Returns:
        A `StudyResult`.

    Raises:
        ValueError: If `n_origins` is below `MIN_ORIGINS`, or the backtest scored
            nothing.
    """
    if int(n_origins) < MIN_ORIGINS:
        raise ValueError(
            f"{n_origins} origins is below the minimum of {MIN_ORIGINS}. The whole "
            "point of this study is that four folds could not resolve the peak-hour "
            "question; running it with too few origins repeats that mistake."
        )

    origins = weekly_origins(series, config, n_origins)
    validation_origins, test_origins = split_origins(origins)

    base = run_backtest(series, config, origins, variants)
    if base.empty:
        raise ValueError("The backtest scored no hours")

    # The peak threshold is fixed once, on the validation block, so that moving between
    # blocks cannot move the definition of a peak hour.
    is_validation = base["origin"].isin(validation_origins)
    threshold = peak_threshold(base[is_validation], config)

    control = VARIANTS[0].name
    validation_control = base[is_validation & (base["variant"] == control)]
    blend_weight, blend_grid = fit_blend_weight(validation_control)
    peak_multiplier, peak_summary = fit_peak_correction(validation_control, threshold)

    predictions = apply_post_hoc_variants(
        base, blend_weight, peak_multiplier, threshold, variants
    )
    predictions["block"] = np.where(
        predictions["origin"].isin(validation_origins), "validation", "test"
    )

    scores = pd.concat(
        [
            score_variants(
                predictions[predictions["block"] == block], config, threshold, block
            )
            for block in ("validation", "test")
        ],
        ignore_index=True,
    )
    selection = select_by_validation(scores)

    # Intervals are reported for the control and, when it differs, the winner.
    coverage_frames = []
    interval_variants = {control, selection["winner"]}
    for name in sorted(interval_variants):
        intervals = rolling_conformal_intervals(predictions, config, name)
        if intervals.empty:
            continue
        for block in ("validation", "test"):
            table = coverage_by_band(
                intervals[intervals["block"] == block], config, block
            )
            if not table.empty:
                table.insert(0, "variant", name)
                coverage_frames.append(table)
    coverage = (
        pd.concat(coverage_frames, ignore_index=True) if coverage_frames else pd.DataFrame()
    )

    test_scores = scores[scores["block"] == "test"]
    winner_test = test_scores[test_scores["variant"] == selection["winner"]]
    control_test = test_scores[test_scores["variant"] == control]

    def _as_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        """Rows as plain dicts, for the metrics JSON."""
        return frame.to_dict(orient="records")

    peak_rows = test_scores[test_scores["scope"] == "peak"]
    still_losing = [
        str(row["variant"])
        for _, row in peak_rows.iterrows()
        if not bool(row["beats_naive"])
    ]

    summary: dict[str, Any] = {
        "ran": True,
        "n_origins": int(len(origins)),
        "horizon_hours": int(config["forecast"]["horizon_hours"]),
        "step_hours": int(config["forecast"]["rolling_origin"]["step_hours"]),
        "n_validation_origins": int(len(validation_origins)),
        "n_test_origins": int(len(test_origins)),
        "n_scored_hours": int(len(base[base["variant"] == control])),
        "peak_threshold_arrivals": round(float(threshold), 4),
        "blend_weight_on_model": blend_weight,
        "peak_correction": peak_summary,
        "selection": selection,
        "validation_scores": _as_records(scores[scores["block"] == "validation"]),
        "test_scores": _as_records(test_scores),
        "winner_test_scores": _as_records(winner_test),
        "control_test_scores": _as_records(control_test),
        "variants_still_losing_to_seasonal_naive_at_peak": still_losing,
        "coverage": _as_records(coverage) if not coverage.empty else [],
        "adoption_note": (
            "This study does not change the shipped forecaster. artifacts/ reproduces "
            "a documented forecast MAE of 2.356 from the default four-fold backtest, "
            "and adopting a variant here would move it. Adoption is a separate, "
            "explicit change: see tests/test_headline_regression.py."
        ),
    }
    return StudyResult(
        predictions=predictions,
        scores=scores,
        coverage=coverage,
        blend_grid=blend_grid,
        summary=summary,
    )
