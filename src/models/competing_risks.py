"""At arrival: will this patient be seen, or will they leave first -- and when?

The wait-time models in this project answer "how long until a clinician?" **for
patients who were seen**, which is a question conditional on an outcome that has not
happened yet. This module asks the unconditional one: standing at the door, what is the
chance this patient leaves without being seen, by 30, 60 or 120 minutes?

**Two estimators, deliberately.**

``cause_specific_cox``
    The hazard model already in `src/models/survival.py`, which gives a continuous
    time-to-event for each cause and a hazard ratio per covariate. It is the
    conventional analysis and it is interpretable.

``discrete_time_lightgbm``
    A LightGBM binary classifier per horizon, predicting *cumulative incidence*
    directly: "did this patient abandon by h minutes". It carries no proportional
    hazards assumption, it uses the same leakage-safe queue features as the wait
    models, and it is what you would actually deploy.

Both are compared against the **Aalen-Johansen** cumulative incidence at the group
level, which is the non-parametric answer and the thing a competing-risks estimate is
supposed to agree with.

**The target is cumulative incidence, not a hazard.** A patient seen before the horizon
is labelled 0: they did not abandon, and they are no longer at risk of it. Treating
them as censored would answer a different question -- "would they have abandoned had
they kept waiting" -- which is counterfactual and not what a triage nurse is asking.

**ESI 1 and 2 are excluded from every calibration result**, and the reason is in the
data rather than in the method: the generator never lets them abandon and the real
extract records almost none. A calibration curve over zero positive cases is not a
weak result, it is not a result.

Decision support only: this estimates a probability for a person to weigh, and
nothing here triages anyone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.data.schema import Schema
from src.features.build import make_design_matrix
from src.utils.logging import get_logger
from src.utils.seed import derive_seed

logger = get_logger(__name__)

#: Horizons, in minutes, at which the cumulative incidence of leaving is predicted.
DEFAULT_HORIZONS = (30, 60, 120)

#: Acuity levels the calibration is reported for. ESI 1-2 are excluded because they
#: supply essentially no positive cases -- see the module docstring.
CALIBRATED_LEVELS = (3, 4, 5)

#: Bins in the calibration curve.
CALIBRATION_BINS = 10

#: Fewer positives than this and a calibration curve is not reported for that slice.
MIN_POSITIVES_FOR_CALIBRATION = 20

#: Carried into every artifact.
SYNTHETIC_NOTICE = (
    "Fitted on SYNTHETIC data. The abandonment behaviour this model recovers is the "
    "generator's patience distribution, which was chosen, not measured. Treat the "
    "method as demonstrated and the probabilities as uncalibrated for any real "
    "department."
)


@dataclass
class HorizonLabels:
    """Cumulative-incidence labels at one horizon, and who was at risk."""

    horizon_minutes: int
    #: 1 when the patient left without being seen within the horizon.
    abandoned: np.ndarray
    #: 1 when the patient was seen within the horizon.
    seen: np.ndarray
    #: Rows whose outcome by the horizon is known at all.
    observed: np.ndarray

    @property
    def n_positive(self) -> int:
        """Patients who abandoned within the horizon, among observed rows."""
        return int(self.abandoned[self.observed].sum())

    @property
    def n_observed(self) -> int:
        """Rows with a known outcome by this horizon."""
        return int(self.observed.sum())


def build_horizon_labels(
    events: pd.DataFrame, horizon_minutes: int, seen_code: int, abandoned_code: int
) -> HorizonLabels:
    """Label cumulative incidence of abandonment at one horizon.

    Args:
        events: Output of `src.models.survival.build_waiting_room_events`, carrying
            `duration_minutes` and `event`.
        horizon_minutes: The horizon.
        seen_code: The event code meaning "seen".
        abandoned_code: The event code meaning "left without being seen".

    Returns:
        The labels, and a mask of rows whose outcome by the horizon is known.

    The censoring rule is the one that makes this a cumulative-incidence problem: a
    patient **still waiting** at the horizon has a known answer (they had not abandoned
    *yet*), but a patient administratively censored *before* the horizon does not, and
    is excluded.
    """
    duration = events["duration_minutes"].to_numpy(dtype="float64")
    event = events["event"].to_numpy()

    abandoned = ((event == abandoned_code) & (duration <= horizon_minutes)).astype("int8")
    seen = ((event == seen_code) & (duration <= horizon_minutes)).astype("int8")

    # Known by the horizon: anyone whose event happened by then, plus anyone still
    # under observation at the horizon. Unknown: censored before the horizon.
    resolved_by_horizon = duration <= horizon_minutes
    censored = (event != seen_code) & (event != abandoned_code)
    observed = ~(censored & resolved_by_horizon)

    return HorizonLabels(
        horizon_minutes=horizon_minutes,
        abandoned=abandoned,
        seen=seen,
        observed=observed,
    )


def calibration_curve(
    truth: np.ndarray, probability: np.ndarray, bins: int = CALIBRATION_BINS
) -> pd.DataFrame:
    """Observed frequency against predicted probability, in equal-width bins.

    Equal-width rather than equal-count bins, because the question is "when the model
    says 20%, does it happen 20% of the time" and that is a question about the
    probability axis.

    Args:
        truth: Binary outcomes.
        probability: Predicted probabilities.
        bins: Number of bins.

    Returns:
        One row per non-empty bin, with the bin's mean prediction, observed rate and
        count.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probability, edges[1:-1], right=False), 0, bins - 1)

    rows: list[dict[str, Any]] = []
    for bin_index in range(bins):
        mask = index == bin_index
        if not mask.any():
            continue
        rows.append(
            {
                "bin": bin_index,
                "bin_low": round(float(edges[bin_index]), 4),
                "bin_high": round(float(edges[bin_index + 1]), 4),
                "n": int(mask.sum()),
                "mean_predicted": round(float(probability[mask].mean()), 6),
                "observed_rate": round(float(truth[mask].mean()), 6),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["gap"] = (frame["observed_rate"] - frame["mean_predicted"]).round(6)
    return frame


def expected_calibration_error(curve: pd.DataFrame) -> float:
    """Count-weighted mean absolute gap between predicted and observed.

    Args:
        curve: Output of `calibration_curve`.

    Returns:
        The ECE, or NaN for an empty curve.
    """
    if curve.empty:
        return float("nan")
    weights = curve["n"].to_numpy(dtype="float64")
    return float(np.average(curve["gap"].abs().to_numpy(), weights=weights))


def brier_score(truth: np.ndarray, probability: np.ndarray) -> float:
    """Mean squared error of a probabilistic prediction."""
    if truth.size == 0:
        return float("nan")
    return float(np.mean((probability - truth) ** 2))


def roc_auc(truth: np.ndarray, score: np.ndarray) -> float:
    """Area under the ROC curve, or NaN when one class is absent.

    Computed from the rank statistic rather than by sweeping thresholds, which is both
    exact and immune to tie handling.
    """
    positives = truth > 0
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank().to_numpy()
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ------------------------------------------------------------------ calibration


@dataclass
class ProbabilityCalibrator:
    """Isotonic recalibration of a probability, fitted on the calibration block only.

    Isotonic rather than Platt: the raw LightGBM scores here are already roughly
    monotone in the truth but badly scaled at the low end, where almost all the mass
    is, and a two-parameter sigmoid cannot fix a shape problem. Isotonic is
    non-parametric and monotone, so it cannot reorder patients -- it only moves the
    numbers onto the probability scale.

    Fitted on `calib`, which the model was neither trained nor early-stopped on. Fitting
    it on train would recalibrate against the model's own over-fitting and look
    excellent while changing nothing useful.
    """

    thresholds: np.ndarray
    values: np.ndarray
    block: str = "calib"
    n_fitted: int = 0

    @classmethod
    def fit(
        cls, truth: np.ndarray, probability: np.ndarray, block: str = "calib"
    ) -> "ProbabilityCalibrator":
        """Fit the isotonic map.

        Args:
            truth: Binary outcomes on the calibration block.
            probability: The model's raw probabilities for those rows.
            block: Name of the block, recorded for audit.

        Returns:
            A fitted calibrator. With fewer than two distinct probabilities, or no
            positives, it returns an identity map rather than a degenerate one.
        """
        from sklearn.isotonic import IsotonicRegression

        if truth.size == 0 or len(np.unique(probability)) < 2 or truth.sum() == 0:
            logger.warning(
                "Not enough signal on the %s block to fit a calibrator (%d rows, %d "
                "positives); leaving probabilities unchanged",
                block, int(truth.size), int(truth.sum()),
            )
            return cls(thresholds=np.array([0.0, 1.0]), values=np.array([0.0, 1.0]),
                       block=block, n_fitted=0)

        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(probability, truth)
        grid = np.linspace(0.0, 1.0, 201)
        return cls(
            thresholds=grid,
            values=np.clip(model.predict(grid), 0.0, 1.0),
            block=block,
            n_fitted=int(truth.size),
        )

    def apply(self, probability: np.ndarray) -> np.ndarray:
        """Map raw probabilities onto the calibrated scale."""
        return np.interp(probability, self.thresholds, self.values)

    @property
    def is_identity(self) -> bool:
        """True when the calibrator could not be fitted and leaves inputs alone."""
        return self.n_fitted == 0


# ------------------------------------------------------------------ the model


@dataclass
class DiscreteTimeRiskModel:
    """One LightGBM classifier per horizon, predicting cumulative incidence of LWBS."""

    models: dict[int, Any] = field(default_factory=dict)
    calibrators: dict[int, ProbabilityCalibrator] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    feature_set: str = "base+queue"
    horizons: tuple[int, ...] = DEFAULT_HORIZONS

    def predict_raw(self, features: pd.DataFrame, horizon: int) -> np.ndarray:
        """Uncalibrated P(left without being seen by `horizon`)."""
        model = self.models[horizon]
        return model.predict_proba(features.loc[:, self.feature_names])[:, 1]

    def predict(self, features: pd.DataFrame, horizon: int) -> np.ndarray:
        """Calibrated P(left without being seen by `horizon`)."""
        raw = self.predict_raw(features, horizon)
        calibrator = self.calibrators.get(horizon)
        return raw if calibrator is None else calibrator.apply(raw)

    def predict_all(self, features: pd.DataFrame) -> pd.DataFrame:
        """Calibrated risk at every horizon, plus the complementary P(seen).

        `p_seen` is `1 - p_lwbs` by construction: with only two competing outcomes and
        a cumulative-incidence target, the patient either leaves by the horizon or has
        not, and "has not" includes both being seen and still waiting. The column is
        named `p_not_lwbs` rather than `p_seen` for exactly that reason.
        """
        out = pd.DataFrame(index=features.index)
        for horizon in self.horizons:
            risk = self.predict(features, horizon)
            out[f"p_lwbs_{horizon}min"] = risk
            out[f"p_not_lwbs_{horizon}min"] = 1.0 - risk
        return out


def train_discrete_time_model(
    frames: dict[str, pd.DataFrame],
    events_by_block: dict[str, pd.DataFrame],
    config: dict[str, Any],
    schema: Schema | None = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    feature_set: str = "base+queue",
    seen_code: int = 1,
    abandoned_code: int = 2,
) -> DiscreteTimeRiskModel:
    """Fit one classifier per horizon on the train block, calibrate on `calib`.

    Args:
        frames: The **full** featurised rows per block -- every patient, not the
            seen-only regression cohort. Abandoners are the positive class here, so
            training on the regression cohort would train on a dataset from which
            every positive case had been removed.
        events_by_block: Waiting-room events per block, aligned row-for-row with
            `frames[name]`.
        config: Full pipeline config.
        schema: Data contract, for the leakage assertion.
        horizons: Horizons in minutes.
        feature_set: Feature set name; must be leakage-safe.
        seen_code: Event code for "seen".
        abandoned_code: Event code for "left without being seen".

    Returns:
        A fitted `DiscreteTimeRiskModel`.
    """
    import lightgbm as lgb

    designs = {
        name: make_design_matrix(frames[name], feature_set, config, schema)[0]
        for name in ("train", "val", "calib", "test")
        if name in frames
    }
    params = dict(config["wait_model"]["classifier"])
    early_stopping_rounds = int(params.pop("early_stopping_rounds", 100))
    for key in (
        "decision_threshold", "threshold_objective", "target_recall",
        "min_positives_per_level", "acuity_sample_weights",
    ):
        params.pop(key, None)

    model = DiscreteTimeRiskModel(
        feature_names=list(designs["train"].columns),
        feature_set=feature_set,
        horizons=tuple(int(h) for h in horizons),
    )

    for horizon in model.horizons:
        train_labels = build_horizon_labels(
            events_by_block["train"], horizon, seen_code, abandoned_code
        )
        val_labels = build_horizon_labels(
            events_by_block["val"], horizon, seen_code, abandoned_code
        )
        if train_labels.n_positive < MIN_POSITIVES_FOR_CALIBRATION:
            logger.warning(
                "Horizon %d min has only %d positive case(s) in train; not fitting a "
                "model for it. A classifier from that many events is noise.",
                horizon, train_labels.n_positive,
            )
            continue

        booster = lgb.LGBMClassifier(
            random_state=derive_seed(config["seed"], f"competing_risks::{horizon}"),
            verbose=-1, n_jobs=-1, **params,
        )
        train_mask, val_mask = train_labels.observed, val_labels.observed
        booster.fit(
            designs["train"].loc[train_mask],
            train_labels.abandoned[train_mask],
            eval_set=[
                (designs["val"].loc[val_mask], val_labels.abandoned[val_mask])
            ],
            eval_metric="average_precision",
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        model.models[horizon] = booster

        # Calibrate on `calib` only: a block the model never saw, in fitting or in
        # early stopping.
        calib_labels = build_horizon_labels(
            events_by_block["calib"], horizon, seen_code, abandoned_code
        )
        calib_mask = calib_labels.observed
        model.calibrators[horizon] = ProbabilityCalibrator.fit(
            calib_labels.abandoned[calib_mask],
            booster.predict_proba(designs["calib"].loc[calib_mask])[:, 1],
        )
        logger.info(
            "Horizon %d min: fitted on %d rows (%d positives), calibrated on %d rows",
            horizon, int(train_mask.sum()), train_labels.n_positive,
            model.calibrators[horizon].n_fitted,
        )
    return model


# ------------------------------------------------------------------ evaluation


def evaluate_horizon(
    truth: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray,
    horizon: int,
    scope: str,
) -> dict[str, Any]:
    """Discrimination and calibration for one horizon and one slice.

    Discrimination is reported once, on the **raw** score, because that is the
    ordering the model actually learned. Isotonic calibration is monotone and so cannot
    invert any pair, but it is not strictly monotone: it pools adjacent violators into
    flat blocks, which turns distinct scores into ties, and tied pairs count as half in
    the AUC. The calibrated AUC therefore moves slightly -- measured here at 0.9802 to
    0.9813 on a synthetic example -- and reporting it would invite reading a tie
    artefact as a change in discrimination.

    Calibration is reported twice, before and after, which is the point of the
    exercise.

    Args:
        truth: Binary outcomes.
        raw: Uncalibrated probabilities.
        calibrated: Calibrated probabilities.
        horizon: Horizon in minutes.
        scope: What this slice is, e.g. "overall" or "ESI 4".

    Returns:
        One row of the metrics table.
    """
    n_positive = int(truth.sum())
    reportable = n_positive >= MIN_POSITIVES_FOR_CALIBRATION
    row: dict[str, Any] = {
        "horizon_minutes": horizon,
        "scope": scope,
        "n": int(truth.size),
        "n_positive": n_positive,
        "positive_rate": round(float(truth.mean()), 6) if truth.size else None,
        "reportable": reportable,
        "auc": round(roc_auc(truth, raw), 4),
        "brier_raw": round(brier_score(truth, raw), 6),
        "brier_calibrated": round(brier_score(truth, calibrated), 6),
        "ece_raw": round(expected_calibration_error(calibration_curve(truth, raw)), 6),
        "ece_calibrated": round(
            expected_calibration_error(calibration_curve(truth, calibrated)), 6
        ),
    }
    if not reportable:
        row["unreportable_reason"] = (
            f"{n_positive} positive case(s) < {MIN_POSITIVES_FOR_CALIBRATION}: a "
            "calibration curve over this many events is not an estimate"
        )
    else:
        row["unreportable_reason"] = None
    if row["brier_raw"] is not None and row["brier_calibrated"] is not None:
        row["brier_improvement"] = round(row["brier_raw"] - row["brier_calibrated"], 6)
    return row


def evaluate_discrete_time_model(
    model: DiscreteTimeRiskModel,
    test_features: pd.DataFrame,
    test_events: pd.DataFrame,
    test_levels: np.ndarray,
    seen_code: int = 1,
    abandoned_code: int = 2,
    levels: Sequence[int] = CALIBRATED_LEVELS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the model on the test block, overall and per acuity.

    Args:
        model: The fitted model.
        test_features: Test design matrix.
        test_events: Test waiting-room events, aligned with `test_features`.
        test_levels: Acuity per test row.
        seen_code: Event code for "seen".
        abandoned_code: Event code for "left without being seen".
        levels: Acuity levels to report calibration for.

    Returns:
        Tuple of (metrics table, calibration curves table).
    """
    metric_rows: list[dict[str, Any]] = []
    curve_rows: list[pd.DataFrame] = []

    for horizon in model.horizons:
        if horizon not in model.models:
            continue
        labels = build_horizon_labels(test_events, horizon, seen_code, abandoned_code)
        mask = labels.observed
        features = test_features.loc[mask]
        truth = labels.abandoned[mask].astype("float64")
        raw = model.predict_raw(features, horizon)
        calibrated = model.predict(features, horizon)
        acuity = np.asarray(test_levels)[mask]

        metric_rows.append(evaluate_horizon(truth, raw, calibrated, horizon, "overall"))
        for frame, label in (
            (calibration_curve(truth, raw), "raw"),
            (calibration_curve(truth, calibrated), "calibrated"),
        ):
            if frame.empty:
                continue
            frame = frame.assign(horizon_minutes=horizon, scope="overall", stage=label)
            curve_rows.append(frame)

        for level in levels:
            level_mask = acuity == level
            if not level_mask.any():
                continue
            level_truth = truth[level_mask]
            metric_rows.append(
                evaluate_horizon(
                    level_truth, raw[level_mask], calibrated[level_mask],
                    horizon, f"ESI {level}",
                )
            )
            for frame, label in (
                (calibration_curve(level_truth, raw[level_mask]), "raw"),
                (calibration_curve(level_truth, calibrated[level_mask]), "calibrated"),
            ):
                if frame.empty:
                    continue
                frame = frame.assign(
                    horizon_minutes=horizon, scope=f"ESI {level}", stage=label
                )
                curve_rows.append(frame)

        # ESI 1-2 are reported as excluded, with their counts, rather than omitted.
        for level in (1, 2):
            level_mask = acuity == level
            if not level_mask.any():
                continue
            metric_rows.append(
                {
                    "horizon_minutes": horizon,
                    "scope": f"ESI {level}",
                    "n": int(level_mask.sum()),
                    "n_positive": int(truth[level_mask].sum()),
                    "positive_rate": round(float(truth[level_mask].mean()), 6),
                    "reportable": False,
                    "auc": None, "brier_raw": None, "brier_calibrated": None,
                    "ece_raw": None, "ece_calibrated": None, "brier_improvement": None,
                    "unreportable_reason": (
                        "excluded by design: high-acuity patients effectively never "
                        "leave without being seen, so there is no incidence to "
                        "calibrate against"
                    ),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    curves = (
        pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    )
    return metrics, curves


def aalen_johansen_comparison(
    model: DiscreteTimeRiskModel,
    test_features: pd.DataFrame,
    test_events: pd.DataFrame,
    test_levels: np.ndarray,
    abandoned_code: int = 2,
    seen_code: int = 1,
    levels: Sequence[int] = CALIBRATED_LEVELS,
) -> pd.DataFrame:
    """Compare the model's mean predicted risk with the non-parametric incidence.

    Aalen-Johansen is the estimator that handles competing risks correctly: it answers
    "what share of this group had left by h", accounting for the fact that being seen
    removes a patient from risk. A model whose average prediction for a group does not
    match it is miscalibrated at the group level, whatever its per-patient metrics say.

    Args:
        model: The fitted model.
        test_features: Test design matrix.
        test_events: Test waiting-room events.
        test_levels: Acuity per test row.
        abandoned_code: Event code for "left without being seen".
        seen_code: Event code for "seen".
        levels: Acuity levels to compare.

    Returns:
        One row per (horizon, group), with the predicted mean, the Aalen-Johansen
        incidence and the gap.
    """
    from lifelines import AalenJohansenFitter

    acuity = np.asarray(test_levels)
    duration = test_events["duration_minutes"].to_numpy(dtype="float64")
    event = test_events["event"].to_numpy()

    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, np.ndarray]] = [("overall", np.ones(len(acuity), dtype=bool))]
    groups += [(f"ESI {level}", acuity == level) for level in levels]

    for scope, group_mask in groups:
        if int(group_mask.sum()) < MIN_POSITIVES_FOR_CALIBRATION:
            continue
        group_duration = duration[group_mask]
        # lifelines wants 0 = censored, and one code per competing cause.
        group_event = np.where(
            event[group_mask] == abandoned_code, 1,
            np.where(event[group_mask] == seen_code, 2, 0),
        )
        if (group_event == 1).sum() < MIN_POSITIVES_FOR_CALIBRATION:
            continue

        fitter = AalenJohansenFitter(calculate_variance=False, seed=0)
        try:
            fitter.fit(group_duration, group_event, event_of_interest=1)
        except Exception as error:  # noqa: BLE001 - a failed comparison is reported
            logger.warning(
                "Aalen-Johansen could not be fitted for %s (%s: %s)",
                scope, type(error).__name__, error,
            )
            continue

        curve = fitter.cumulative_density_
        for horizon in model.horizons:
            if horizon not in model.models:
                continue
            incidence = float(
                curve.loc[curve.index <= horizon].iloc[-1, 0]
                if (curve.index <= horizon).any() else 0.0
            )
            predicted = float(
                model.predict(test_features.loc[group_mask], horizon).mean()
            )
            rows.append(
                {
                    "scope": scope,
                    "horizon_minutes": horizon,
                    "n": int(group_mask.sum()),
                    "aalen_johansen_incidence": round(incidence, 6),
                    "mean_predicted_risk": round(predicted, 6),
                    "gap": round(predicted - incidence, 6),
                    "relative_gap": (
                        round((predicted - incidence) / incidence, 4) if incidence else None
                    ),
                }
            )

    table = pd.DataFrame(rows)
    if not table.empty:
        worst = table.loc[table["gap"].abs().idxmax()]
        logger.info(
            "Largest group-level gap against Aalen-Johansen: %s at %d min, predicted "
            "%.4f against %.4f observed",
            worst["scope"], worst["horizon_minutes"],
            worst["mean_predicted_risk"], worst["aalen_johansen_incidence"],
        )
    return table


def cox_discrimination_comparison(
    train_events: pd.DataFrame,
    test_events: pd.DataFrame,
    model: DiscreteTimeRiskModel,
    test_features: pd.DataFrame,
    config: dict[str, Any],
    seen_code: int = 1,
    abandoned_code: int = 2,
) -> pd.DataFrame:
    """Compare the cause-specific Cox model with the discrete-time classifier.

    **The comparison is on discrimination only, and that is a statement about the two
    methods rather than a convenience.** A cause-specific Cox model gives a hazard
    ratio and a risk *ranking*; it does not give the cumulative incidence of leaving by
    a horizon, because that depends on both cause hazards together and on the competing
    risk removing patients from the risk set. Reading a cause-specific hazard as if it
    were an incidence is the classic competing-risks error -- it overstates the risk,
    and the overstatement grows with the competing event's rate. Getting an incidence
    out of a hazard model properly means Fine-Gray, which is a different model.

    So: both are scored by AUC at each horizon, on identical rows, and only the
    classifier is quoted for Brier and calibration, because only the classifier
    estimates the quantity those measure.

    Args:
        train_events: Waiting-room events for the training block, with covariates.
        test_events: Waiting-room events for the test block, with covariates.
        model: The fitted discrete-time model.
        test_features: Test design matrix for the classifier.
        config: Full pipeline config.
        seen_code: Event code for "seen".
        abandoned_code: Event code for "left without being seen".

    Returns:
        One row per horizon, with both AUCs and the difference. Empty when the Cox
        model cannot be fitted.
    """
    from lifelines import CoxPHFitter

    from src.models.survival import _available_covariates

    covariates = _available_covariates(train_events, config)
    if not covariates:
        logger.warning("No usable Cox covariates; skipping the Cox comparison")
        return pd.DataFrame()

    def design(frame: pd.DataFrame) -> pd.DataFrame:
        """Numeric covariates plus duration and the abandonment indicator."""
        out = frame.loc[:, [*covariates, "duration_minutes", "event"]].copy()
        for name in covariates:
            out[name] = pd.to_numeric(out[name], errors="coerce")
        out = out.dropna()
        out["observed"] = (out["event"] == abandoned_code).astype(int)
        return out.drop(columns=["event"])

    train_design = design(train_events)
    if train_design["observed"].sum() < MIN_POSITIVES_FOR_CALIBRATION:
        logger.warning(
            "Only %d abandonment event(s) in train; not fitting a Cox comparison",
            int(train_design["observed"].sum()),
        )
        return pd.DataFrame()

    fitter = CoxPHFitter(penalizer=float(config["survival"].get("cox_penalizer", 0.1)))
    fitter.fit(train_design, duration_col="duration_minutes", event_col="observed")

    test_design = design(test_events)
    # A higher partial hazard means a higher rate of leaving, so it ranks the same way
    # a risk probability does. Only the ordering is used.
    cox_score = fitter.predict_partial_hazard(test_design).to_numpy()
    aligned = test_events.index.isin(test_design.index)

    rows: list[dict[str, Any]] = []
    for horizon in model.horizons:
        if horizon not in model.models:
            continue
        labels = build_horizon_labels(test_events, horizon, seen_code, abandoned_code)
        mask = labels.observed & aligned
        truth = labels.abandoned[mask].astype("float64")
        if truth.sum() < MIN_POSITIVES_FOR_CALIBRATION:
            continue

        cox_mask = test_design.index.isin(test_events.index[mask])
        classifier_score = model.predict(test_features.loc[mask], horizon)
        cox_auc = roc_auc(truth, cox_score[cox_mask])
        lgb_auc = roc_auc(truth, classifier_score)
        rows.append(
            {
                "horizon_minutes": horizon,
                "n": int(mask.sum()),
                "n_positive": int(truth.sum()),
                "cox_auc": round(cox_auc, 4),
                "lightgbm_auc": round(lgb_auc, 4),
                "auc_difference": round(lgb_auc - cox_auc, 4),
                "better": "lightgbm" if lgb_auc > cox_auc else "cox",
                "note": (
                    "AUC only. The Cox model ranks risk; it does not estimate the "
                    "cumulative incidence of leaving by this horizon, so it has no "
                    "Brier score or calibration curve to compare."
                ),
            }
        )

    table = pd.DataFrame(rows)
    for row in table.itertuples():
        logger.info(
            "Horizon %d min: Cox AUC %.4f, LightGBM AUC %.4f (%s ahead by %.4f)",
            row.horizon_minutes, row.cox_auc, row.lightgbm_auc,
            row.better, abs(row.auc_difference),
        )
    return table
