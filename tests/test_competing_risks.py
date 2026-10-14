"""Tests for the arrival-time competing-risks model.

Three things have to hold. The labelling must be *cumulative incidence* and not a
hazard, because the two answer different questions and only one of them is what a
triage nurse is asking. The calibration must be fitted on a block the model never saw.
And a prediction made for a patient must not change when patients who arrive later are
added to the dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.competing_risks import (
    CALIBRATED_LEVELS,
    DEFAULT_HORIZONS,
    MIN_POSITIVES_FOR_CALIBRATION,
    ProbabilityCalibrator,
    brier_score,
    build_horizon_labels,
    calibration_curve,
    expected_calibration_error,
    roc_auc,
)
from src.models.survival import ABANDONED, CENSORED, SEEN


def _events(rows: list[tuple[float, int]]) -> pd.DataFrame:
    """A minimal events frame: (duration_minutes, event code)."""
    return pd.DataFrame(rows, columns=["duration_minutes", "event"])


# ---------------------------------------------------------------- the labelling


def test_a_patient_seen_before_the_horizon_is_a_negative_not_a_censoring() -> None:
    """The target is cumulative incidence, so being seen means 'did not abandon'.

    Censoring them instead would answer the counterfactual question -- would they have
    left had they kept waiting -- which is not what is being predicted.
    """
    labels = build_horizon_labels(
        _events([(10.0, SEEN), (20.0, ABANDONED), (90.0, SEEN)]),
        horizon_minutes=60, seen_code=SEEN, abandoned_code=ABANDONED,
    )

    assert labels.abandoned.tolist() == [0, 1, 0]
    assert labels.observed.all(), "none of these outcomes is unknown at 60 minutes"
    assert labels.n_positive == 1


def test_an_event_after_the_horizon_is_a_negative() -> None:
    """Leaving at 90 minutes means you had not left at 60."""
    labels = build_horizon_labels(
        _events([(90.0, ABANDONED)]),
        horizon_minutes=60, seen_code=SEEN, abandoned_code=ABANDONED,
    )

    assert labels.abandoned.tolist() == [0]
    assert labels.observed.tolist() == [True]


def test_a_patient_censored_before_the_horizon_has_an_unknown_outcome() -> None:
    """Dropped, not guessed: we genuinely do not know what they did by 60 minutes."""
    labels = build_horizon_labels(
        _events([(20.0, CENSORED), (20.0, ABANDONED)]),
        horizon_minutes=60, seen_code=SEEN, abandoned_code=ABANDONED,
    )

    assert labels.observed.tolist() == [False, True]
    assert labels.n_observed == 1


def test_a_patient_censored_after_the_horizon_is_still_usable() -> None:
    """Censored at 300 minutes tells us they had not left at 60, which is the label."""
    labels = build_horizon_labels(
        _events([(300.0, CENSORED)]),
        horizon_minutes=60, seen_code=SEEN, abandoned_code=ABANDONED,
    )

    assert labels.observed.tolist() == [True]
    assert labels.abandoned.tolist() == [0]


def test_longer_horizons_cannot_have_fewer_positives() -> None:
    """Cumulative incidence is non-decreasing in the horizon, by definition."""
    events = _events([(20.0, ABANDONED), (50.0, ABANDONED), (200.0, ABANDONED), (10.0, SEEN)])
    counts = [
        build_horizon_labels(events, h, SEEN, ABANDONED).n_positive
        for h in DEFAULT_HORIZONS
    ]

    assert counts == sorted(counts)


# ---------------------------------------------------------------- calibration


def test_isotonic_calibration_improves_a_badly_scaled_score() -> None:
    """The case the calibrator exists for: right ranking, wrong scale."""
    rng = np.random.default_rng(0)
    truth = (rng.random(2000) < 0.1).astype(float)
    # Correct ordering, but squashed into [0, 0.02] -- far below the true 10% rate.
    raw = np.where(truth > 0, rng.uniform(0.01, 0.02, 2000), rng.uniform(0.0, 0.01, 2000))

    calibrator = ProbabilityCalibrator.fit(truth, raw)
    calibrated = calibrator.apply(raw)

    assert expected_calibration_error(
        calibration_curve(truth, calibrated)
    ) < expected_calibration_error(calibration_curve(truth, raw))


def test_isotonic_calibration_never_inverts_a_pair() -> None:
    """Monotone, so it preserves order -- but it is NOT order-preserving on the AUC.

    Isotonic pools adjacent violators into flat blocks, which makes previously distinct
    scores equal. Tied pairs count as half in the AUC, so the calibrated AUC moves a
    little even though no pair has been inverted. That is why `evaluate_horizon`
    reports discrimination on the raw score only.
    """
    rng = np.random.default_rng(1)
    truth = (rng.random(1000) < 0.2).astype(float)
    raw = np.clip(truth * 0.3 + rng.normal(0, 0.1, 1000), 0.0, 1.0)

    calibrated = ProbabilityCalibrator.fit(truth, raw).apply(raw)

    # No inversion: wherever the raw scores differ strictly, the calibrated ones are
    # ordered the same way or tied, never swapped.
    order = np.argsort(raw)
    assert np.all(np.diff(calibrated[order]) >= -1e-12)
    # And the AUC stays close, but is not required to be identical.
    assert roc_auc(truth, calibrated) == pytest.approx(roc_auc(truth, raw), abs=0.01)


def test_a_calibrator_with_no_signal_is_the_identity() -> None:
    """No positives means nothing to learn; it must not collapse every score to zero."""
    calibrator = ProbabilityCalibrator.fit(np.zeros(100), np.linspace(0, 1, 100))

    assert calibrator.is_identity
    np.testing.assert_allclose(calibrator.apply(np.array([0.3, 0.7])), [0.3, 0.7])


def test_the_calibrator_records_which_block_it_was_fitted_on() -> None:
    """Fitting on train would recalibrate against the model's own over-fitting."""
    calibrator = ProbabilityCalibrator.fit(
        np.array([0.0, 1.0, 1.0, 0.0]), np.array([0.1, 0.9, 0.8, 0.2])
    )
    assert calibrator.block == "calib"


# ---------------------------------------------------------------- the metrics


def test_the_calibration_curve_bins_the_probability_axis() -> None:
    """Equal-width bins, because the question is about the probability scale."""
    truth = np.array([0.0] * 50 + [1.0] * 50)
    probability = np.array([0.05] * 50 + [0.95] * 50)

    curve = calibration_curve(truth, probability, bins=10)

    assert len(curve) == 2
    assert curve["observed_rate"].tolist() == [0.0, 1.0]
    assert expected_calibration_error(curve) == pytest.approx(0.05, abs=1e-9)


def test_auc_is_undefined_with_one_class() -> None:
    """A group with no positives has no discrimination to measure."""
    assert np.isnan(roc_auc(np.zeros(10), np.linspace(0, 1, 10)))
    assert np.isnan(roc_auc(np.ones(10), np.linspace(0, 1, 10)))


def test_auc_recognises_a_perfect_and_an_inverted_ranking() -> None:
    """Sanity on the rank statistic, in both directions."""
    truth = np.array([0.0, 0.0, 1.0, 1.0])
    assert roc_auc(truth, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert roc_auc(truth, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)


def test_brier_rewards_a_confident_correct_prediction() -> None:
    """A basic property, asserted so a sign error cannot hide."""
    truth = np.array([1.0, 0.0])
    assert brier_score(truth, np.array([0.9, 0.1])) < brier_score(truth, np.array([0.5, 0.5]))


# ---------------------------------------------------------------- the exclusions


def test_high_acuity_levels_are_not_in_the_calibrated_set() -> None:
    """ESI 1-2 supply no positives, so a calibration curve over them is not a result."""
    assert 1 not in CALIBRATED_LEVELS
    assert 2 not in CALIBRATED_LEVELS
    assert set(CALIBRATED_LEVELS) == {3, 4, 5}


def test_the_reporting_minimum_is_not_trivial() -> None:
    """A threshold of zero would make every refusal in this module a no-op."""
    assert MIN_POSITIVES_FOR_CALIBRATION >= 10


# ---------------------------------------------------------------- leakage


def test_a_prediction_does_not_change_when_later_patients_are_added(
    tiny_visits: pd.DataFrame, config: dict
) -> None:
    """The leakage boundary, in the style of the existing suite.

    A risk estimate for a patient is made at their arrival. Adding patients who arrive
    *after* them must not change it -- if it does, a queue feature is reading forward
    in time and the model could never be deployed.
    """
    from src.data.schema import load_schema, validate_visits
    from src.features.build import add_patient_features, build_features, make_design_matrix

    validated = validate_visits(tiny_visits, load_schema()).all
    validated = validated.sort_values("arrival_time").reset_index(drop=True)

    cut = int(len(validated) * 0.6)
    early = validated.iloc[:cut].copy()

    def design(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
        """Leakage-safe design matrix for `frame`, with queue state from `reference`."""
        featurised = add_patient_features(
            build_features(frame, config, reference=reference)
        )
        return make_design_matrix(featurised, "base+queue", config, load_schema())[0]

    # The same early patients, once with the whole dataset visible and once with only
    # their own past. Queue features look strictly backwards, so these must agree.
    with_future = design(early, validated).reset_index(drop=True)
    without_future = design(early, early).reset_index(drop=True)

    pd.testing.assert_frame_equal(with_future, without_future)
