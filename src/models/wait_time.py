"""Per-patient wait-time models: a regressor and a prolonged-wait classifier.

Three things in here are easy to get subtly wrong, so each is handled explicitly.

**Back-transforming a log target.** Fitting `log1p(wait)` and applying `expm1` to the
prediction does *not* estimate the mean wait. Because `expm1` is convex, the result
estimates something closer to the conditional **median**. To estimate the conditional
**mean** you need a correction for the spread of the residuals; this module uses Duan's
smearing estimator, which averages the back-transformed prediction over the empirical
residual distribution instead of assuming it is log-normal. Both are reported, each
labelled with what it estimates, because MAE rewards a median and RMSE rewards a mean.

**Choosing the decision threshold.** The classifier's threshold is picked on the `val`
block and then frozen. Picking it on `test` would make the reported recall optimistic
by construction. The chosen threshold and the block it came from are both recorded.

**Which rows can carry a regression target.** A patient who left without being seen, or
who was still waiting when the extract closed, has no observed wait. A patient who was
seen but whose departure was censored *does* have an observed wait, but is excluded too
so that every training row is a fully-observed visit. Each exclusion is counted and
logged, because silently dropping rows is how a cohort stops matching its description.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.utils.logging import get_logger
from src.utils.metrics import classification_metrics, regression_metrics
from src.utils.seed import derive_seed

logger = get_logger(__name__)

TargetTransform = Literal["raw", "log1p"]

#: Number of residual quantiles used to approximate Duan's smearing factor. The exact
#: estimator averages over all n calibration residuals, which is O(n_pred * n_resid);
#: a few hundred quantiles reproduce it to well under a minute of wait time.
SMEARING_QUANTILES = 256

#: How each prediction flavour should be read.
PREDICTION_KINDS = {
    "raw": "conditional median (L1 objective on the minute scale)",
    "log1p_median": "conditional median (expm1 back-transform, no spread correction)",
    "log1p_smearing": "conditional mean (Duan smearing over empirical residuals)",
}


@dataclass
class RegressionCohort:
    """A fully-observed wait-time cohort, with the exclusions that produced it."""

    frame: pd.DataFrame
    exclusions: dict[str, int] = field(default_factory=dict)
    n_input: int = 0

    @property
    def n_excluded(self) -> int:
        """Total rows removed."""
        return int(sum(self.exclusions.values()))

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "n_input": self.n_input,
            "n_kept": int(len(self.frame)),
            "n_excluded": self.n_excluded,
            "excluded_by_reason": dict(self.exclusions),
        }


def select_regression_cohort(frame: pd.DataFrame, block_name: str = "") -> RegressionCohort:
    """Keep only rows whose wait is fully observed, counting every exclusion.

    Exclusions, applied in order so the counts sum to the total:
      1. left without being seen -- no clinician contact, so no wait
      2. still waiting at the data cutoff -- right-censored, not an outcome
      3. missing `wait_minutes` for any other reason
      4. censored departure (seen, but never recorded as leaving) -- the wait *is*
         observed here, so this is a conservative choice that keeps the cohort to
         complete visits only

    Args:
        frame: A featurised block.
        block_name: Name used in log messages.

    Returns:
        A `RegressionCohort`.
    """
    label = f" [{block_name}]" if block_name else ""
    remaining = pd.Series(True, index=frame.index)
    exclusions: dict[str, int] = {}

    def apply(reason: str, condition: pd.Series) -> None:
        """Drop rows matching `condition`, attributing them to the first reason that hits."""
        nonlocal remaining
        hit = condition.fillna(False).astype(bool) & remaining
        exclusions[reason] = int(hit.sum())
        remaining &= ~hit

    if "is_lwbs" in frame.columns:
        apply("left without being seen", frame["is_lwbs"] & frame["seen_time"].isna())
    if "still_waiting_at_cutoff" in frame.columns:
        apply("still waiting at the data cutoff", frame["still_waiting_at_cutoff"])
    apply("wait_minutes missing", frame["wait_minutes"].isna())
    if "los_minutes" in frame.columns:
        apply("departure censored by the data window", frame["los_minutes"].isna())

    cohort = RegressionCohort(
        frame=frame.loc[remaining].copy(), exclusions=exclusions, n_input=int(len(frame))
    )
    logger.info(
        "Regression cohort%s: %d rows in, %d excluded, %d kept",
        label,
        cohort.n_input,
        cohort.n_excluded,
        len(cohort.frame),
    )
    for reason, count in exclusions.items():
        if count:
            logger.info("  excluded %5d rows: %s", count, reason)
        else:
            logger.debug("  excluded     0 rows: %s", reason)
    return cohort


@dataclass
class WaitTimeRegressor:
    """A LightGBM wait-time regressor with an explicit target transform.

    Attributes:
        model: The fitted booster.
        transform: "raw" or "log1p".
        feature_names: Design-matrix columns, in order.
        feature_set: Name of the feature set used ("base" or "base+queue").
        smearing_residuals: Log-space residual quantiles from the validation block,
            used for Duan's smearing. Empty for the raw target.
        best_iteration: Iteration chosen by early stopping.
    """

    model: lgb.LGBMRegressor
    transform: TargetTransform
    feature_names: list[str]
    feature_set: str
    smearing_residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    best_iteration: int | None = None

    def _raw_prediction(self, features: pd.DataFrame) -> np.ndarray:
        """Model output in the space the model was fitted in."""
        return self.model.predict(features.loc[:, self.feature_names])

    def predict(self, features: pd.DataFrame, kind: str = "default") -> np.ndarray:
        """Predict wait in minutes.

        Args:
            features: Design matrix containing `feature_names`.
            kind: "default" gives the median-style estimate for both transforms;
                "smearing" gives the mean estimate and requires the log1p transform.

        Returns:
            Predicted minutes, floored at zero.

        Raises:
            ValueError: If smearing is requested for a raw-target model.
        """
        prediction = self._raw_prediction(features)

        if self.transform == "raw":
            if kind == "smearing":
                raise ValueError(
                    "Smearing applies to a log-transformed target; this model was fitted "
                    "on the raw minute scale, where the L1 objective already estimates "
                    "the conditional median."
                )
            return np.maximum(prediction, 0.0)

        if kind == "smearing":
            if self.smearing_residuals.size == 0:
                raise ValueError("No smearing residuals were stored; refit with a validation set")
            # Duan's smearing: average the back-transform over the empirical residual
            # distribution rather than assuming log-normality.
            grid = prediction[:, None] + self.smearing_residuals[None, :]
            return np.maximum(np.expm1(grid).mean(axis=1), 0.0)

        return np.maximum(np.expm1(prediction), 0.0)

    def prediction_kinds(self) -> dict[str, str]:
        """Available prediction flavours for this model, and what each estimates."""
        if self.transform == "raw":
            return {"raw": PREDICTION_KINDS["raw"]}
        return {
            "log1p_median": PREDICTION_KINDS["log1p_median"],
            "log1p_smearing": PREDICTION_KINDS["log1p_smearing"],
        }


def _transform_target(values: np.ndarray, transform: TargetTransform) -> np.ndarray:
    """Map the target into the space the model is fitted in."""
    if transform == "raw":
        return values
    if transform == "log1p":
        if (values < 0).any():
            raise ValueError("log1p target requires non-negative waits")
        return np.log1p(values)
    raise ValueError(f"Unknown target transform {transform!r}; expected 'raw' or 'log1p'")


def train_wait_regressor(
    train_features: pd.DataFrame,
    train_target: np.ndarray,
    val_features: pd.DataFrame,
    val_target: np.ndarray,
    config: dict[str, Any],
    transform: TargetTransform = "raw",
    feature_set: str = "base+queue",
) -> WaitTimeRegressor:
    """Fit the wait-time regressor, early-stopping on the validation block.

    Hyperparameters and the seed come from the config, unchanged between feature sets
    and transforms, so the ablation isolates the feature effect.

    Args:
        train_features: Training design matrix.
        train_target: Training waits in minutes.
        val_features: Validation design matrix, for early stopping only.
        val_target: Validation waits in minutes.
        config: Full pipeline config.
        transform: "raw" or "log1p".
        feature_set: Name of the feature set, recorded on the fitted object.

    Returns:
        A fitted `WaitTimeRegressor`.
    """
    params = dict(config["wait_model"]["regressor"])
    early_stopping_rounds = int(params.pop("early_stopping_rounds", 100))
    seed = derive_seed(config["seed"], f"wait_regressor::{feature_set}::{transform}")

    model = lgb.LGBMRegressor(random_state=seed, verbose=-1, n_jobs=-1, **params)
    model.fit(
        train_features,
        _transform_target(np.asarray(train_target, dtype="float64"), transform),
        eval_set=[
            (val_features, _transform_target(np.asarray(val_target, dtype="float64"), transform))
        ],
        eval_metric="l1",
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    regressor = WaitTimeRegressor(
        model=model,
        transform=transform,
        feature_names=list(train_features.columns),
        feature_set=feature_set,
        best_iteration=getattr(model, "best_iteration_", None),
    )

    if transform == "log1p":
        # Smearing residuals come from the validation block, which the model did not
        # fit on. Using training residuals would understate the spread.
        log_prediction = model.predict(val_features)
        residuals = _transform_target(np.asarray(val_target, dtype="float64"), transform) - (
            log_prediction
        )
        residuals = residuals[np.isfinite(residuals)]
        quantiles = np.linspace(0.0, 1.0, SMEARING_QUANTILES)
        regressor.smearing_residuals = np.quantile(residuals, quantiles)
        logger.info(
            "Stored %d residual quantiles for smearing (residual sd %.4f in log space)",
            SMEARING_QUANTILES,
            float(np.std(residuals)),
        )

    logger.info(
        "Fitted wait regressor [%s / %s]: %d features, best iteration %s",
        feature_set,
        transform,
        len(regressor.feature_names),
        regressor.best_iteration,
    )
    return regressor


def evaluate_regressor(
    regressor: WaitTimeRegressor,
    features: pd.DataFrame,
    target: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Score every prediction flavour on the original minute scale.

    Both MAE and RMSE are always computed on minutes, never in log space, so the
    numbers are comparable across transforms.

    Args:
        regressor: Fitted model.
        features: Design matrix.
        target: Observed waits in minutes.

    Returns:
        Mapping of prediction-kind name to its metric bundle, with an `estimates` key
        stating what that flavour estimates.
    """
    results: dict[str, dict[str, Any]] = {}
    for kind, description in regressor.prediction_kinds().items():
        predict_kind = "smearing" if kind.endswith("smearing") else "default"
        prediction = regressor.predict(features, kind=predict_kind)
        metrics = regression_metrics(target, prediction)
        metrics["estimates"] = description
        results[kind] = metrics
        logger.info(
            "  %-16s MAE %6.2f  RMSE %6.2f  bias %+6.2f   (%s)",
            kind,
            metrics["mae"],
            metrics["rmse"],
            metrics["bias"],
            description,
        )
    return results


@dataclass
class ProlongedWaitClassifier:
    """A LightGBM prolonged-wait classifier with a threshold chosen off-test.

    Attributes:
        model: The fitted booster.
        feature_names: Design-matrix columns, in order.
        feature_set: Name of the feature set used.
        threshold: Decision threshold on the predicted probability.
        threshold_source: Which block the threshold was selected on.
        threshold_objective: The rule used to pick it.
        best_iteration: Iteration chosen by early stopping.
    """

    model: lgb.LGBMClassifier
    feature_names: list[str]
    feature_set: str
    threshold: float = 0.5
    threshold_source: str = "config"
    threshold_objective: str = "config default"
    best_iteration: int | None = None

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Positive-class probability."""
        return self.model.predict_proba(features.loc[:, self.feature_names])[:, 1]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Hard prediction at the frozen threshold."""
        return (self.predict_proba(features) >= self.threshold).astype("int8")


def choose_decision_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    config: dict[str, Any],
) -> tuple[float, str]:
    """Pick the decision threshold on a non-test block.

    The objective comes from the config. The default, `min_recall`, picks the highest
    threshold that still achieves a target recall, because the costly error for ED
    decision support is telling a patient they will be seen soon when they will not.
    `f1` and `fixed` are also available.

    Args:
        y_true: Binary labels from the selection block.
        y_score: Predicted probabilities for the same rows.
        config: Full pipeline config; reads `wait_model.classifier`.

    Returns:
        Tuple of (threshold, human-readable description of the rule used).

    Raises:
        ValueError: If the configured objective is unknown.
    """
    settings = config["wait_model"]["classifier"]
    objective = str(settings.get("threshold_objective", "min_recall")).lower()
    truth = np.asarray(y_true, dtype="float64")
    score = np.asarray(y_score, dtype="float64")

    if objective == "fixed":
        threshold = float(settings.get("decision_threshold", 0.5))
        return threshold, f"fixed threshold {threshold:.3f} from config"

    candidates = np.unique(np.round(score, 4))
    if candidates.size > 2000:
        candidates = np.quantile(score, np.linspace(0.0, 1.0, 2000))

    positives = truth.sum()
    if positives == 0:
        logger.warning("Selection block has no positive labels; falling back to 0.5")
        return 0.5, "fallback 0.5 (no positives in the selection block)"

    if objective == "min_recall":
        target_recall = float(settings.get("target_recall", 0.80))
        best: tuple[float, float] | None = None
        for threshold in candidates:
            predicted = score >= threshold
            recall = float((predicted & (truth > 0)).sum() / positives)
            if recall >= target_recall:
                # Higher threshold means fewer false alarms at the same recall floor.
                if best is None or threshold > best[0]:
                    best = (float(threshold), recall)
        if best is None:
            logger.warning(
                "No threshold reaches recall %.2f on the selection block; using the "
                "lowest candidate to maximise recall",
                target_recall,
            )
            threshold = float(candidates.min())
            return threshold, f"recall target {target_recall:.2f} unreachable; used minimum score"
        return (
            best[0],
            f"highest threshold with recall >= {target_recall:.2f} (achieved {best[1]:.3f})",
        )

    if objective == "f1":
        best_threshold, best_f1 = 0.5, -1.0
        for threshold in candidates:
            predicted = score >= threshold
            true_positive = float((predicted & (truth > 0)).sum())
            predicted_positive = float(predicted.sum())
            precision = true_positive / predicted_positive if predicted_positive else 0.0
            recall = true_positive / positives
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            if f1 > best_f1:
                best_threshold, best_f1 = float(threshold), f1
        return best_threshold, f"maximised F1 (achieved {best_f1:.3f})"

    raise ValueError(
        f"Unknown wait_model.classifier.threshold_objective {objective!r}; "
        "expected 'min_recall', 'f1' or 'fixed'"
    )


def train_prolonged_classifier(
    train_features: pd.DataFrame,
    train_labels: np.ndarray,
    val_features: pd.DataFrame,
    val_labels: np.ndarray,
    config: dict[str, Any],
    feature_set: str = "base+queue",
) -> ProlongedWaitClassifier:
    """Fit the prolonged-wait classifier and freeze its threshold off-test.

    Args:
        train_features: Training design matrix.
        train_labels: Binary training labels.
        val_features: Validation design matrix, used for early stopping *and* threshold
            selection -- never the test block.
        val_labels: Binary validation labels.
        config: Full pipeline config.
        feature_set: Name of the feature set, recorded on the fitted object.

    Returns:
        A fitted `ProlongedWaitClassifier` with its threshold already set.
    """
    params = dict(config["wait_model"]["classifier"])
    early_stopping_rounds = int(params.pop("early_stopping_rounds", 100))
    for key in ("decision_threshold", "threshold_objective", "target_recall"):
        params.pop(key, None)
    seed = derive_seed(config["seed"], f"wait_classifier::{feature_set}")

    model = lgb.LGBMClassifier(random_state=seed, verbose=-1, n_jobs=-1, **params)
    model.fit(
        train_features,
        np.asarray(train_labels, dtype="int8"),
        eval_set=[(val_features, np.asarray(val_labels, dtype="int8"))],
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    classifier = ProlongedWaitClassifier(
        model=model,
        feature_names=list(train_features.columns),
        feature_set=feature_set,
        best_iteration=getattr(model, "best_iteration_", None),
    )

    val_score = classifier.predict_proba(val_features)
    threshold, rule = choose_decision_threshold(val_labels, val_score, config)
    classifier.threshold = threshold
    classifier.threshold_source = "val"
    classifier.threshold_objective = rule

    logger.info(
        "Fitted prolonged-wait classifier [%s]: threshold %.4f chosen on the val block (%s)",
        feature_set,
        threshold,
        rule,
    )
    return classifier


def seen_only_bias_report(
    visits: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Quantify the bias from training only on patients who were seen.

    Patients who left without being seen are not missing at random: they are the ones
    who waited longest. Excluding them therefore biases any estimate of the wait
    *downwards*. Their true waits are unknown -- they never happened -- but their
    time-to-abandon is a hard **lower bound** on how long they would have waited, so
    substituting it bounds the bias from below.

    Args:
        visits: The full validated population, including patients never seen.
        config: Full pipeline config.

    Returns:
        Dict with per-level and overall means under both views, plus the implied shift.
    """
    everyone = visits.copy()
    never_seen = everyone["seen_time"].isna()
    abandoned = never_seen & everyone["is_lwbs"]

    time_to_abandon = (
        pd.to_datetime(everyone["departure_time"]) - pd.to_datetime(everyone["arrival_time"])
    ).dt.total_seconds() / 60.0

    # (a) seen patients only -- the cohort the model trains on.
    seen_only = everyone["wait_minutes"]
    # (b) seen patients, plus abandoners entered at their time-to-abandon lower bound.
    with_bound = seen_only.where(~abandoned, time_to_abandon)

    rows = []
    for level, group in everyone.groupby("triage_level"):
        mask = group.index
        a = seen_only.loc[mask].dropna()
        b = with_bound.loc[mask].dropna()
        n_abandoned = int(abandoned.loc[mask].sum())
        rows.append(
            {
                "triage_level": int(level),
                "n_seen": int(len(a)),
                "n_abandoned": n_abandoned,
                "abandoned_share": round(n_abandoned / max(len(group), 1), 4),
                "mean_wait_seen_only": round(float(a.mean()), 2),
                "mean_wait_with_abandon_lower_bound": round(float(b.mean()), 2),
                "bias_minutes": round(float(b.mean() - a.mean()), 2),
                "bias_percent": round(100.0 * float(b.mean() - a.mean()) / float(a.mean()), 2),
            }
        )

    table = pd.DataFrame(rows)
    overall_a = float(seen_only.dropna().mean())
    overall_b = float(with_bound.dropna().mean())
    report = {
        "direction": "downward -- excluding abandoners understates the wait",
        "interpretation": (
            "Patients who left without being seen waited longer than average, so a "
            "model trained only on seen patients under-predicts. Their true waits are "
            "unobservable; time-to-abandon is a lower bound, so the figures below are "
            "the MINIMUM size of the bias, not an estimate of it."
        ),
        "overall": {
            "mean_wait_seen_only": round(overall_a, 2),
            "mean_wait_with_abandon_lower_bound": round(overall_b, 2),
            "bias_minutes_at_least": round(overall_b - overall_a, 2),
            "bias_percent_at_least": round(100.0 * (overall_b - overall_a) / overall_a, 2),
            "n_seen": int(seen_only.notna().sum()),
            "n_abandoned": int(abandoned.sum()),
        },
        "by_triage_level": rows,
    }

    logger.info(
        "Seen-only bias: overall mean wait %.2f min (seen only) vs %.2f min with "
        "abandoners at their lower bound -- at least %+.2f min (%+.1f%%), biased downward",
        overall_a,
        overall_b,
        overall_b - overall_a,
        report["overall"]["bias_percent_at_least"],
    )
    for row in rows:
        logger.info(
            "  ESI %d: %.1f -> %.1f min (at least %+.1f min, %+.1f%%), %d abandoned",
            row["triage_level"],
            row["mean_wait_seen_only"],
            row["mean_wait_with_abandon_lower_bound"],
            row["bias_minutes"],
            row["bias_percent"],
            row["n_abandoned"],
        )
    return report, table
