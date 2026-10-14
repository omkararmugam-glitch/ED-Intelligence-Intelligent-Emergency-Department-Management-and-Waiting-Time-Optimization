"""Run this project's arrival forecaster on real hourly ED arrivals.

The dataset is `uihc_ed_arrivals` from the R package **fastTS** (GPL >= 3): 41,640
hourly arrival counts from the University of Iowa Hospitals and Clinics emergency
department, 2013-07-01 to 2018-03-31. It is real data from a real department, which
is the whole point -- every other number in this repository comes from a generator.

**What this can show.** Whether the forecaster, unchanged, beats the baselines it is
supposed to beat on a real series. That is a claim about the method.

**What it cannot show.** Anything about this project's wait model, its simulation or
its staffing search. Those need per-patient records with timestamps, and this is a
count series. It also says nothing about *this* department's performance in any
other year or any other hospital.

Two rules the script keeps, because breaking either would make the result flattering
and wrong:

* **Nothing is retuned.** The same model class, the same lag set including the
  168-hour seasonal lag, the same rolling-origin design, the same metrics and the
  same conformal calibration, read from `configs/default.yaml`. If the forecaster
  loses on real data, that is the finding.
* **Only features known in advance.** Calendar and holiday indicators are known
  before the hour starts; **the temperature column is not**, and using it would be
  predicting the past. The dataset ships `temp` for the same hour, so it is used only
  in a clearly labelled side experiment, lagged by the forecast horizon so the value
  would genuinely have been available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Run as a script from anywhere, like the other drivers in this directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.forecast import (  # noqa: E402
    TARGET_COLUMN,
    TIME_COLUMN,
    hour_band_edges,
    rolling_origin_evaluate,
)
from src.utils.io import (  # noqa: E402
    ensure_parent,
    load_config,
    resolve_path,
    save_json,
)
from src.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

#: Where the extracted series lives.
DEFAULT_SOURCE = Path("data/external/uihc_ed_arrivals.csv")

#: Where everything this script writes goes.
DEFAULT_OUT = Path("artifacts/external_validation/uihc")

#: Holiday and event indicators the package ships. All are known in advance -- a
#: hospital knows next Christmas's date and the football fixture list -- so these
#: are legitimate forecast features in a way the temperature is not.
EVENT_COLUMNS = ("xmas", "xmas2", "nye", "nyd", "thx", "thx1", "ind", "game_day")


# --------------------------------------------------------------------- loading


def load_series(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The hourly series in this project's shape, plus the event indicators.

    Returns:
        `(series, events)` where `series` has exactly `timestamp` and `arrivals`,
        which is what `src/models/forecast.py` expects, and `events` carries the
        extra columns keyed by timestamp for the side experiments.
    """
    frame = pd.read_csv(resolve_path(path))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    series = pd.DataFrame({
        TIME_COLUMN: frame["timestamp"],
        TARGET_COLUMN: pd.to_numeric(frame["Arrivals"], errors="coerce"),
    })
    events = frame[["timestamp", "temp", *EVENT_COLUMNS]].copy()
    return series, events


def validate_series(series: pd.DataFrame, events: pd.DataFrame) -> dict[str, Any]:
    """Check the series is what it claims to be, before anything is fitted.

    A forecaster will happily fit a series with a duplicated hour or a silent gap
    and produce confident nonsense, because a gap corrupts every lag that spans it.
    This runs first and the findings go in the report whether they are clean or not.
    """
    stamps = series[TIME_COLUMN]
    expected = pd.date_range(stamps.min(), stamps.max(), freq="h")
    present = pd.DatetimeIndex(stamps)

    counts = pd.to_numeric(series[TARGET_COLUMN], errors="coerce")
    integral = counts.dropna().apply(lambda value: float(value).is_integer()).all()

    findings = {
        "rows": int(len(series)),
        "first_hour": str(stamps.min()),
        "last_hour": str(stamps.max()),
        "expected_hours_in_range": int(len(expected)),
        "missing_hours": int(len(expected.difference(present))),
        "duplicate_hours": int(int(present.duplicated().sum())),
        "missing_counts": int(counts.isna().sum()),
        "negative_counts": int((counts < 0).sum()),
        "all_counts_integral": bool(integral),
        "mean_arrivals_per_hour": round(float(counts.mean()), 3),
        "max_arrivals_in_an_hour": int(counts.max()),
        "zero_hours": int((counts == 0).sum()),
        "event_hours": {
            column: int(pd.to_numeric(events[column], errors="coerce").fillna(0).gt(0).sum())
            for column in EVENT_COLUMNS
        },
        "temperature_present": bool(events["temp"].notna().any()),
    }

    # A daylight-saving spring-forward leaves a missing hour and an autumn
    # fall-back a duplicate, in local time. Naming them separately matters: a DST
    # artefact is a property of the clock, while a gap anywhere else is missing data.
    findings["gap_timestamps"] = [str(value) for value in expected.difference(present)[:24]]
    findings["dst_note"] = (
        "A spring-forward shows up as a missing local hour and an autumn fall-back "
        "as a duplicate. Any gap or duplicate listed here that is not at 02:00 on a "
        "DST changeover is missing data rather than a clock artefact."
    )
    return findings


# ------------------------------------------------------------------- baselines


def four_week_same_hour(series: pd.DataFrame, timestamps: pd.Series) -> np.ndarray:
    """The mean of the same hour over the previous four weeks.

    The second baseline the brief asks for, and a harder one than seasonal-naive:
    averaging four weeks cancels some of the noise that a single week carries, so a
    forecaster that only just beats seasonal-naive can lose to this.
    """
    truth = series.set_index(TIME_COLUMN)[TARGET_COLUMN]
    values = []
    for stamp in pd.DatetimeIndex(timestamps):
        weeks = [truth.get(stamp - pd.Timedelta(weeks=week)) for week in (1, 2, 3, 4)]
        known = [value for value in weeks if value is not None and not pd.isna(value)]
        values.append(float(np.mean(known)) if known else np.nan)
    return np.asarray(values, dtype=float)


# --------------------------------------------------------------------- metrics


def _smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric mean absolute percentage error, in percent.

    Computed only where the denominator is non-zero. A quiet 3 a.m. with zero
    arrivals and a forecast of 0.4 would otherwise contribute an undefined or
    infinite term, and dropping those hours is more honest than inventing an
    epsilon that quietly decides the answer.
    """
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    usable = denominator > 0
    if not usable.any():
        return float("nan")
    return float(
        100 * np.mean(np.abs(actual[usable] - predicted[usable]) / denominator[usable])
    )


def score(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    """MAE, RMSE, sMAPE and WAPE over the rows where both are present."""
    frame = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    if frame.empty:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"),
                "smape_pct": float("nan"), "wape_pct": float("nan")}

    truth = frame["actual"].to_numpy(dtype=float)
    guess = frame["predicted"].to_numpy(dtype=float)
    errors = np.abs(truth - guess)
    total = truth.sum()
    return {
        "n": int(len(frame)),
        "mae": round(float(errors.mean()), 4),
        "rmse": round(float(np.sqrt(np.mean((truth - guess) ** 2))), 4),
        "smape_pct": round(_smape(truth, guess), 3),
        # WAPE is the error as a share of total volume, which is the one a manager
        # can read directly: "we were out by 12% of the patients who arrived".
        "wape_pct": round(float(100 * errors.sum() / total), 3) if total else float("nan"),
    }


def score_everything(
    predictions: pd.DataFrame, series: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    """Every metric, for the model and both baselines, overall and by hour band."""
    frame = predictions.copy()
    frame["four_week"] = four_week_same_hour(series, frame["timestamp"])
    frame["hour"] = pd.DatetimeIndex(frame["timestamp"]).hour

    peak_threshold = float(frame["actual"].quantile(config["forecast"]["peak_quantile"]))
    frame["is_peak"] = frame["actual"] >= peak_threshold

    # `yhat` is the forecaster's own column name, `baseline` the seasonal-naive it
    # scores itself against, and `four_week` the second baseline added here.
    columns = {"model": "yhat", "seasonal_naive": "baseline", "four_week_mean": "four_week"}
    results: dict[str, Any] = {
        "peak_threshold_arrivals": round(peak_threshold, 2),
        "overall": {}, "peak": {}, "off_peak": {}, "by_hour_band": {},
    }
    for label, column in columns.items():
        results["overall"][label] = score(frame["actual"], frame[column])
        results["peak"][label] = score(
            frame.loc[frame["is_peak"], "actual"], frame.loc[frame["is_peak"], column]
        )
        results["off_peak"][label] = score(
            frame.loc[~frame["is_peak"], "actual"], frame.loc[~frame["is_peak"], column]
        )

    for band, (start, end) in hour_band_edges(config).items():
        inside = (frame["hour"] >= start) & (frame["hour"] < end)
        results["by_hour_band"][band] = {
            label: score(frame.loc[inside, "actual"], frame.loc[inside, column])
            for label, column in columns.items()
        }

    # The project's own interval check, on real data: how often the actual count
    # fell inside the forecast's range.
    if {"lower", "upper"}.issubset(frame.columns):
        inside_interval = (frame["actual"] >= frame["lower"]) & (frame["actual"] <= frame["upper"])
        results["conformal_coverage"] = {
            "overall": round(float(inside_interval.mean()), 4),
            "by_hour_band": {
                band: round(
                    float(
                        inside_interval[
                            (frame["hour"] >= start) & (frame["hour"] < end)
                        ].mean()
                    ),
                    4,
                )
                for band, (start, end) in hour_band_edges(config).items()
            },
        }
    return results, frame


# ------------------------------------------------- the labelled side experiment


def lagged_temperature_experiment(
    frame: pd.DataFrame, events: pd.DataFrame, horizon_hours: int
) -> dict[str, Any]:
    """Does yesterday's temperature help? Asked honestly, and answered as measured.

    The dataset ships the temperature for the same hour, which a forecaster issuing
    a 72-hour forecast cannot know. Using it would be predicting the past and would
    make the model look better than it is. So the column is lagged by the forecast
    horizon -- the value that *was* available when the forecast was issued -- and the
    result is reported as a correlation, not folded into the model.
    """
    temperature = events.set_index("timestamp")["temp"]
    lagged = frame["timestamp"].map(
        lambda stamp: temperature.get(stamp - pd.Timedelta(hours=horizon_hours))
    )
    usable = pd.DataFrame({
        "residual": frame["actual"] - frame["yhat"],
        "temp_lagged": pd.to_numeric(lagged, errors="coerce"),
        "actual": frame["actual"],
    }).dropna()

    if len(usable) < 100:
        return {"ran": False, "reason": "too few hours with a lagged temperature"}

    return {
        "ran": True,
        "lag_hours": horizon_hours,
        "n": int(len(usable)),
        "correlation_with_arrivals": round(
            float(usable["temp_lagged"].corr(usable["actual"])), 4
        ),
        "correlation_with_residual": round(
            float(usable["temp_lagged"].corr(usable["residual"])), 4
        ),
        "label": (
            "LAGGED temperature, not the same hour's. Same-hour temperature is not "
            "known when a forecast is issued, so it is not a usable feature; this is "
            "the value from the moment the forecast was made."
        ),
    }


# ---------------------------------------------------------------------- figure


def write_figure(frame: pd.DataFrame, out: Path) -> Path | None:
    """One figure: a fortnight of forecast against actual and both baselines."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    window = frame.sort_values("timestamp").tail(336)
    figure, axes = plt.subplots(figsize=(11, 4.2))
    axes.plot(window["timestamp"], window["actual"], color="#0e8a8a", linewidth=1.6, label="Actual")
    axes.plot(window["timestamp"], window["yhat"], color="#2a78d6", linewidth=1.4, label="Forecast")
    axes.plot(window["timestamp"], window["baseline"], color="#6b7280", linewidth=1.0,
              linestyle="--", label="Seasonal-naive (168 h)")
    axes.plot(window["timestamp"], window["four_week"], color="#c77700", linewidth=1.0,
              linestyle=":", label="Mean of same hour, 4 weeks")
    if {"lower", "upper"}.issubset(window.columns):
        axes.fill_between(window["timestamp"], window["lower"], window["upper"],
                          color="#9db8d8", alpha=0.25, label="Forecast range")
    axes.set_ylabel("Arrivals per hour")
    axes.set_title("UIHC emergency department: last 14 days of the backtest")
    axes.legend(loc="upper left", fontsize=8, ncol=2)
    figure.autofmt_xdate()
    figure.tight_layout()

    path = ensure_parent(out / "uihc_forecast_vs_baselines.png")
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


# ------------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python scripts/external_validation_uihc.py",
        description=(
            "Run the project's arrival forecaster, unchanged, on real UIHC hourly "
            "arrivals. Forecasting only: this dataset cannot validate the wait "
            "model, the simulation or the staffing search."
        ),
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", type=int, default=None,
                        help="rolling-origin folds; default is the config's")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    config = load_config(args.config)
    out = resolve_path(Path(args.out))
    out.mkdir(parents=True, exist_ok=True)

    series, events = load_series(Path(args.source))
    checks = validate_series(series, events)
    logger.info(
        "UIHC series: %d hours, %s to %s, %d missing, %d duplicate",
        checks["rows"], checks["first_hour"], checks["last_hour"],
        checks["missing_hours"], checks["duplicate_hours"],
    )

    predictions, summary = rolling_origin_evaluate(series, config, n_folds=args.folds)
    metrics, frame = score_everything(predictions, series, config)
    temperature = lagged_temperature_experiment(
        frame, events, int(config["forecast"]["horizon_hours"])
    )

    payload = {
        "dataset": "uihc_ed_arrivals (fastTS, GPL >= 3)",
        "what_this_can_show": (
            "Whether this project's arrival forecaster, unchanged, beats its "
            "baselines on a real hourly series."
        ),
        "what_this_cannot_show": (
            "Anything about the wait model, the simulation or the staffing search: "
            "those need per-patient records and this is a count series. It is also "
            "one department over one period."
        ),
        "data_checks": checks,
        "config_used": {
            "lags": list(config["forecast"]["lags"]),
            "rolling_windows": list(config["forecast"]["rolling_windows"]),
            "seasonal_lag_hours": config["forecast"]["seasonal_lag_hours"],
            "horizon_hours": config["forecast"]["horizon_hours"],
            "rolling_origin": config["forecast"]["rolling_origin"],
            "model": config["forecast"]["model"],
            "note": "Read from configs/default.yaml and not modified for this run.",
        },
        "metrics": metrics,
        "lagged_temperature_experiment": temperature,
        "synthetic_comparison": {
            "synthetic_forecast_mae": 2.3557,
            "synthetic_seasonal_naive_mae": 2.9792,
            "note": (
                "The synthetic figures are the project's headline numbers on "
                "data/raw/ed_v2.csv. They are not comparable in absolute terms -- a "
                "different department with a different arrival rate -- so only the "
                "direction and the size of the gap to the baseline are comparable."
            ),
        },
    }

    save_json(payload, out / "metrics.json")
    frame.to_csv(out / "backtest_predictions.csv", index=False)
    figure = write_figure(frame, out)

    logger.info(
        "UIHC: model MAE %.4f | seasonal-naive %.4f | 4-week mean %.4f",
        metrics["overall"]["model"]["mae"],
        metrics["overall"]["seasonal_naive"]["mae"],
        metrics["overall"]["four_week_mean"]["mae"],
    )
    if figure:
        logger.info("Wrote %s", figure)
    print(json.dumps(metrics["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
