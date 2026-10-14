"""Two analyses the upload workflow adds to the pipeline: drift, and scoring.

**Drift** answers the question that decides whether any of the rest means anything
for this user: is the uploaded department like the one the baseline models were
trained on? Population Stability Index and a Kolmogorov-Smirnov statistic, per
feature, against the baseline training block.

**Scoring** is the alternative to retraining. It runs the baseline models over the
uploaded visits and reports how well they do, which is the honest way to answer
"would your models work here" without pretending a model trained elsewhere is a
model trained here. Every number it produces is paired with the drift table,
because an error figure without a drift figure invites the wrong conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.jobs.storage import RunPaths
from src.utils.io import load_json, read_visits, resolve_path, save_json, save_table
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: PSI bands in common use. Below 0.1 a shift is not worth acting on; above 0.25 the
#: populations are different enough that a model trained on one is suspect on the other.
PSI_MINOR = 0.10
PSI_MAJOR = 0.25

#: Bins used for the PSI histogram. Ten is the convention and is stable at the row
#: counts this workflow accepts.
PSI_BINS = 10

#: Columns compared. Deliberately the inputs a model sees plus the outcome, not every
#: column: drift in `patient_id` is meaningless.
DRIFT_COLUMNS = (
    "triage_level",
    "age",
    "wait_minutes",
    "los_minutes",
    "n_waiting",
    "n_in_treatment",
    "hour_of_day",
)

#: Categorical columns compared by share rather than by histogram.
DRIFT_CATEGORICAL = ("sex", "arrival_mode", "disposition")


class DriftUnavailable(RuntimeError):
    """There is no baseline to compare against."""


def psi(reference: pd.Series, candidate: pd.Series, bins: int = PSI_BINS) -> float:
    """Population Stability Index between two numeric samples.

    Bins are quantiles of the *reference*, so the index measures how the candidate
    redistributes itself across the shape the baseline had.

    Args:
        reference: The baseline sample.
        candidate: The uploaded sample.
        bins: Number of quantile bins.

    Returns:
        The PSI. 0 is identical; higher is more different.
    """
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    candidate = pd.to_numeric(candidate, errors="coerce").dropna()
    if len(reference) < bins or len(candidate) < bins:
        return float("nan")

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    reference_share = np.histogram(reference, bins=edges)[0] / len(reference)
    candidate_share = np.histogram(candidate, bins=edges)[0] / len(candidate)

    # An empty bin makes the log term infinite; the usual correction is a floor at
    # one observation's worth of mass.
    floor = 1.0 / max(len(reference), len(candidate))
    reference_share = np.clip(reference_share, floor, None)
    candidate_share = np.clip(candidate_share, floor, None)

    return float(
        np.sum((candidate_share - reference_share) * np.log(candidate_share / reference_share))
    )


def ks_statistic(reference: pd.Series, candidate: pd.Series) -> float:
    """Kolmogorov-Smirnov statistic: the largest gap between two CDFs.

    Computed directly rather than through scipy, which is not a dependency of this
    project.
    """
    reference = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    candidate = pd.to_numeric(candidate, errors="coerce").dropna().to_numpy()
    if len(reference) == 0 or len(candidate) == 0:
        return float("nan")

    grid = np.sort(np.unique(np.concatenate([reference, candidate])))
    reference_cdf = np.searchsorted(np.sort(reference), grid, side="right") / len(reference)
    candidate_cdf = np.searchsorted(np.sort(candidate), grid, side="right") / len(candidate)
    return float(np.max(np.abs(reference_cdf - candidate_cdf)))


def categorical_drift(reference: pd.Series, candidate: pd.Series) -> float:
    """Total variation distance between two categorical distributions.

    Half the sum of absolute share differences: 0 identical, 1 disjoint.
    """
    reference_share = reference.dropna().astype(str).value_counts(normalize=True)
    candidate_share = candidate.dropna().astype(str).value_counts(normalize=True)
    categories = reference_share.index.union(candidate_share.index)
    return float(
        0.5
        * np.abs(
            reference_share.reindex(categories, fill_value=0.0).to_numpy()
            - candidate_share.reindex(categories, fill_value=0.0).to_numpy()
        ).sum()
    )


def interpret(value: float) -> str:
    """The plain-English reading of a PSI."""
    if not np.isfinite(value):
        return "not comparable"
    if value < PSI_MINOR:
        return "no meaningful shift"
    if value < PSI_MAJOR:
        return "moderate shift"
    return "large shift"


def baseline_training_frame(config: dict[str, Any]) -> pd.DataFrame:
    """The training block the baseline models were fitted on.

    Raises:
        DriftUnavailable: If the baseline pipeline has not been run here.
    """
    # The run config points `processed_dir` at the run folder, so the baseline is
    # addressed explicitly rather than through the config being used.
    candidates = [
        resolve_path(Path("data/processed/features_train.parquet")),
        resolve_path(Path("data/processed/split_train.parquet")),
        resolve_path(Path("data/processed/visits_validated.parquet")),
    ]
    for path in candidates:
        if path.is_file():
            logger.info("Drift reference: %s", path)
            return read_visits(path)
    raise DriftUnavailable(
        "There is no baseline training data on this machine to compare against. Run "
        "the pipeline once (scripts\\run_pipeline.ps1) to create it."
    )


def write_column_profile(paths: RunPaths, frame: pd.DataFrame) -> dict[str, Any]:
    """Record how complete each column is, and what is in it.

    A missing rate is the first thing to look at when a model behaves oddly on a new
    extract: a column that is 90% empty was probably mapped to the wrong field, and
    nothing downstream will say so.

    Writes `metrics/column_profile.json` and returns it.
    """
    rows = []
    for column in frame.columns:
        values = frame[column]
        missing = int(values.isna().sum())
        rows.append(
            {
                "column": column,
                "missing": missing,
                "missing_rate": round(missing / len(frame), 4) if len(frame) else 0.0,
                "distinct": int(values.nunique(dropna=True)),
                "dtype": str(values.dtype),
            }
        )

    profile = {
        "n_rows": int(len(frame)),
        "columns": rows,
        # A column that is almost entirely empty is worth saying out loud.
        "mostly_empty": [
            row["column"] for row in rows if row["missing_rate"] > 0.5
        ],
    }
    save_json(profile, paths.artifacts / "metrics" / "column_profile.json")
    return profile


def run_drift(paths: RunPaths, config: dict[str, Any]) -> dict[str, Any]:
    """Compare the uploaded data with the baseline training block.

    Also profiles the uploaded columns, so a completeness problem is visible even
    when there is no baseline on this machine to compare against.

    Writes `tables/drift.csv`, `metrics/drift.json` and
    `metrics/column_profile.json` into the run folder.

    Args:
        paths: The run's folder layout.
        config: The run config.

    Returns:
        The drift summary.
    """
    candidate = read_visits(paths.processed / "visits_validated.parquet")
    write_column_profile(paths, candidate)

    try:
        reference = baseline_training_frame(config)
    except DriftUnavailable as error:
        summary = {"ran": False, "reason": str(error)}
        save_json(summary, paths.artifacts / "metrics" / "drift.json")
        return summary

    # Queue-state and calendar columns exist only after featurisation, so derive the
    # handful that are cheap rather than requiring the features stage to have run.
    candidate = candidate.copy()
    if "arrival_time" in candidate.columns:
        candidate["hour_of_day"] = pd.to_datetime(
            candidate["arrival_time"], errors="coerce"
        ).dt.hour
    if "arrival_time" in reference.columns and "hour_of_day" not in reference.columns:
        reference = reference.copy()
        reference["hour_of_day"] = pd.to_datetime(
            reference["arrival_time"], errors="coerce"
        ).dt.hour

    rows: list[dict[str, Any]] = []
    for column in DRIFT_COLUMNS:
        if column not in reference.columns or column not in candidate.columns:
            continue
        value = psi(reference[column], candidate[column])
        rows.append(
            {
                "feature": column,
                "kind": "numeric",
                "psi": round(value, 4) if np.isfinite(value) else None,
                "ks": round(ks_statistic(reference[column], candidate[column]), 4),
                "baseline_mean": _safe_mean(reference[column]),
                "uploaded_mean": _safe_mean(candidate[column]),
                "reading": interpret(value),
            }
        )

    for column in DRIFT_CATEGORICAL:
        if column not in reference.columns or column not in candidate.columns:
            continue
        distance = categorical_drift(reference[column], candidate[column])
        rows.append(
            {
                "feature": column,
                "kind": "categorical",
                "psi": None,
                "ks": round(distance, 4),
                "baseline_mean": None,
                "uploaded_mean": None,
                "reading": (
                    "no meaningful shift"
                    if distance < PSI_MINOR
                    else "moderate shift"
                    if distance < PSI_MAJOR
                    else "large shift"
                ),
            }
        )

    table = pd.DataFrame(rows)
    if not table.empty:
        save_table(table, paths.artifacts / "tables" / "drift.csv")

    large = [row["feature"] for row in rows if row["reading"] == "large shift"]
    summary = {
        "ran": True,
        "n_features_compared": len(rows),
        "n_large_shifts": len(large),
        "large_shift_features": large,
        "baseline_rows": int(len(reference)),
        "uploaded_rows": int(len(candidate)),
        "psi_minor_threshold": PSI_MINOR,
        "psi_major_threshold": PSI_MAJOR,
        "verdict": (
            "The uploaded data looks like the data the baseline models were trained on."
            if not large
            else "The uploaded data differs substantially from the baseline training "
            "data in: " + ", ".join(large) + "."
        ),
    }
    save_json(summary, paths.artifacts / "metrics" / "drift.json")
    logger.info("Drift: %d feature(s) compared, %d large shift(s)", len(rows), len(large))
    return summary


def _safe_mean(series: pd.Series) -> float | None:
    """Mean of a numeric column, or None when it is not numeric."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    return round(float(values.mean()), 3) if len(values) else None


# ------------------------------------------------------------------ scoring mode


@dataclass
class ScoringResult:
    """How the baseline models did on the uploaded visits."""

    n_scored: int
    mae_minutes: float
    rmse_minutes: float
    median_absolute_error: float
    interval_coverage: float
    nominal_coverage: float
    baseline_mae_minutes: float | None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form."""
        return {
            "ran": True,
            "n_scored": self.n_scored,
            "mae_minutes": round(self.mae_minutes, 2),
            "rmse_minutes": round(self.rmse_minutes, 2),
            "median_absolute_error_minutes": round(self.median_absolute_error, 2),
            "interval_coverage": round(self.interval_coverage, 4),
            "nominal_coverage": self.nominal_coverage,
            "baseline_mae_minutes": (
                round(self.baseline_mae_minutes, 2)
                if self.baseline_mae_minutes is not None
                else None
            ),
        }


def run_scoring(paths: RunPaths, config: dict[str, Any]) -> dict[str, Any]:
    """Score the uploaded visits with the baseline models, without retraining.

    The models were fitted on a different department. That is the point of the mode
    and also its hazard, so the result carries the warning and the drift table is
    produced alongside it.

    Writes `tables/scoring.csv` and `metrics/scoring.json` into the run folder.

    Raises:
        FileNotFoundError: If no baseline models exist on this machine.
    """
    from src.features.build import make_design_matrix
    from src.serving import load_bundle

    bundle = load_bundle("configs/default.yaml")

    features_path = paths.processed / "features_test.parquet"
    if not features_path.is_file():
        features_path = paths.processed / "features_train.parquet"
    frame = read_visits(features_path)

    actual_column = "wait_minutes"
    scored = frame[frame[actual_column].notna()].copy()
    if scored.empty:
        summary = {
            "ran": False,
            "reason": (
                "No visit in the uploaded file records both an arrival time and a time "
                "the patient was seen, so there is nothing to score against."
            ),
        }
        save_json(summary, paths.artifacts / "metrics" / "scoring.json")
        return summary

    design, _ = make_design_matrix(scored, bundle.feature_set, bundle.config)
    predictions = bundle.regressor.predict(design, kind=bundle.prediction_kind)
    lower, upper = bundle.calibrator.intervals(
        np.asarray(predictions), groups=scored["triage_level"].astype(int).tolist()
    )

    actual = scored[actual_column].to_numpy(dtype=float)
    errors = predictions - actual
    covered = (actual >= lower) & (actual <= upper)

    baseline_mae = _rolling_baseline_mae(scored, config)

    result = ScoringResult(
        n_scored=int(len(scored)),
        mae_minutes=float(np.mean(np.abs(errors))),
        rmse_minutes=float(np.sqrt(np.mean(errors**2))),
        median_absolute_error=float(np.median(np.abs(errors))),
        interval_coverage=float(np.mean(covered)),
        nominal_coverage=float(config["conformal"]["coverage"]),
        baseline_mae_minutes=baseline_mae,
    )

    per_triage = (
        pd.DataFrame(
            {
                "triage_level": scored["triage_level"].to_numpy(),
                "actual_minutes": actual,
                "predicted_minutes": predictions,
                "absolute_error": np.abs(errors),
                "within_interval": covered,
            }
        )
        .groupby("triage_level", observed=True)
        .agg(
            n=("actual_minutes", "size"),
            mae_minutes=("absolute_error", "mean"),
            mean_actual=("actual_minutes", "mean"),
            mean_predicted=("predicted_minutes", "mean"),
            interval_coverage=("within_interval", "mean"),
        )
        .round(3)
        .reset_index()
    )
    save_table(per_triage, paths.artifacts / "tables" / "scoring.csv")

    summary = result.to_dict()
    summary["per_triage"] = per_triage.to_dict(orient="records")
    summary["model_version_note"] = (
        "These models were trained on the baseline dataset, not on your upload. The "
        "error figures above say how well someone else's model does on your data; "
        "read them next to the drift table, which says how different your data is."
    )
    summary["feature_set"] = bundle.feature_set
    summary["estimator"] = str(bundle.best["prediction_kind"])
    save_json(summary, paths.artifacts / "metrics" / "scoring.json")
    logger.info(
        "Scored %d visit(s) with the baseline model: MAE %.2f min, coverage %.1f%%",
        result.n_scored, result.mae_minutes, 100 * result.interval_coverage,
    )
    return summary


def _rolling_baseline_mae(frame: pd.DataFrame, config: dict[str, Any]) -> float | None:
    """MAE of the rolling-mean baseline on the same rows, for context.

    A model's error means little without the trivial predictor's error beside it.
    """
    column = f"rolling_mean_wait_{config['features']['rolling_baseline_k']}"
    if column not in frame.columns or "wait_minutes" not in frame.columns:
        return None
    usable = frame[[column, "wait_minutes"]].dropna()
    if usable.empty:
        return None
    return float(np.mean(np.abs(usable[column] - usable["wait_minutes"])))
