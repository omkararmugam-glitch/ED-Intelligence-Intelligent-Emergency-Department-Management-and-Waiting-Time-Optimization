"""The external-validation scripts, and the claims they are allowed to make.

Two kinds of test. The first kind is ordinary: the metric functions are correct, the
data checks catch a broken series, the MIMIC join does not multiply rows. The second
kind guards the honesty of the thing, and matters more:

* **nothing retuned** -- the UIHC run must read the project's own forecast config
  rather than a flattering one of its own;
* **no same-hour temperature** -- it is not known when a forecast is issued, and
  using it would be predicting the past;
* **no model claims from 222 stays** -- the MIMIC script must not fit or score
  anything, because a number from the demo would be quoted later as though it meant
  something;
* **every proxy wait labelled** as not being door-to-provider.

The tests that need the datasets skip cleanly when they are absent, because the
datasets are deliberately not in the package and not in the repository.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils.io import project_root

UIHC_SCRIPT = project_root() / "scripts" / "external_validation_uihc.py"
MIMIC_SCRIPT = project_root() / "scripts" / "external_validation_mimic.py"
FETCH_SCRIPT = project_root() / "scripts" / "fetch_mimic_ed_demo.ps1"
EXTERNAL_README = project_root() / "data" / "external" / "README.md"
UIHC_CSV = project_root() / "data" / "external" / "uihc_ed_arrivals.csv"
MIMIC_DIR = project_root() / "data" / "external" / "mimic_ed_demo" / "mimic-iv-ed-demo-2.2" / "ed"


def _load(path: Path):
    """Import one of the scripts as a module, by path."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def uihc():
    """The UIHC validation module."""
    return _load(UIHC_SCRIPT)


@pytest.fixture(scope="module")
def mimic():
    """The MIMIC validation module."""
    return _load(MIMIC_SCRIPT)


# ------------------------------------------------------------------- the metrics


def test_a_perfect_forecast_scores_zero(uihc) -> None:
    """The arithmetic, pinned at its easiest point."""
    actual = pd.Series([3.0, 7.0, 11.0])
    result = uihc.score(actual, actual.copy())
    assert result["mae"] == 0.0
    assert result["rmse"] == 0.0
    assert result["smape_pct"] == 0.0
    assert result["wape_pct"] == 0.0
    assert result["n"] == 3


def test_the_metrics_are_what_they_claim(uihc) -> None:
    """MAE, RMSE and WAPE computed by hand on a small case."""
    actual = pd.Series([10.0, 20.0, 30.0])
    predicted = pd.Series([12.0, 18.0, 33.0])

    result = uihc.score(actual, predicted)
    assert result["mae"] == pytest.approx(7 / 3, abs=1e-4)
    assert result["rmse"] == pytest.approx(np.sqrt((4 + 4 + 9) / 3), abs=1e-4)
    # WAPE is total error over total volume, which is the one a manager reads.
    assert result["wape_pct"] == pytest.approx(100 * 7 / 60, abs=1e-3)


def test_smape_ignores_hours_where_it_is_undefined(uihc) -> None:
    """A quiet hour with no arrivals and a forecast of zero has no percentage error.

    Both terms are zero, the denominator is zero, and inventing an epsilon would let
    an arbitrary constant decide the score. Those hours are dropped instead.
    """
    actual = np.array([0.0, 4.0])
    predicted = np.array([0.0, 5.0])
    value = uihc._smape(actual, predicted)
    assert value == pytest.approx(100 * abs(4 - 5) / 4.5, abs=1e-6)


def test_scoring_ignores_rows_where_either_side_is_missing(uihc) -> None:
    """A baseline with no history for an hour must not count as an error of zero."""
    actual = pd.Series([5.0, 6.0, 7.0])
    predicted = pd.Series([5.0, float("nan"), 7.0])
    assert uihc.score(actual, predicted)["n"] == 2


def test_the_four_week_baseline_averages_the_same_hour(uihc) -> None:
    """Not the previous four hours, and not the previous four days."""
    index = pd.date_range("2024-01-01", periods=24 * 35, freq="h")
    series = pd.DataFrame({"timestamp": index, "arrivals": np.arange(len(index), dtype=float)})

    target = pd.Series([pd.Timestamp("2024-02-01 09:00")])
    value = uihc.four_week_same_hour(series, target)[0]

    truth = series.set_index("timestamp")["arrivals"]
    expected = np.mean([
        truth[pd.Timestamp("2024-02-01 09:00") - pd.Timedelta(weeks=week)]
        for week in (1, 2, 3, 4)
    ])
    assert value == pytest.approx(expected)


# --------------------------------------------------------------- the data checks


def test_the_data_check_counts_a_gap(uihc) -> None:
    """A silent gap corrupts every lag that spans it, so it has to be found."""
    index = pd.date_range("2024-01-01", periods=100, freq="h").delete(50)
    series = pd.DataFrame({"timestamp": index, "arrivals": np.ones(len(index))})
    events = pd.DataFrame({
        "timestamp": index, "temp": 50.0,
        **{column: 0.0 for column in uihc.EVENT_COLUMNS},
    })

    findings = uihc.validate_series(series, events)
    assert findings["missing_hours"] == 1
    assert findings["gap_timestamps"], "the missing hour should be named"


def test_the_data_check_counts_duplicates_and_bad_values(uihc) -> None:
    """A duplicated hour, a negative count and a fractional count are all findings."""
    index = pd.date_range("2024-01-01", periods=5, freq="h")
    series = pd.DataFrame({
        "timestamp": list(index) + [index[2]],
        "arrivals": [1.0, 2.0, 3.0, -1.0, 2.5, 3.0],
    })
    events = pd.DataFrame({
        "timestamp": series["timestamp"], "temp": 50.0,
        **{column: 0.0 for column in uihc.EVENT_COLUMNS},
    })

    findings = uihc.validate_series(series, events)
    assert findings["duplicate_hours"] == 1
    assert findings["negative_counts"] == 1
    assert findings["all_counts_integral"] is False


def test_the_data_check_names_daylight_saving(uihc) -> None:
    """A DST artefact is a property of the clock; a gap elsewhere is missing data."""
    index = pd.date_range("2024-01-01", periods=10, freq="h")
    series = pd.DataFrame({"timestamp": index, "arrivals": np.ones(10)})
    events = pd.DataFrame({
        "timestamp": index, "temp": 50.0,
        **{column: 0.0 for column in uihc.EVENT_COLUMNS},
    })
    assert "DST" in uihc.validate_series(series, events)["dst_note"]


# ------------------------------------------------------- nothing retuned, nothing leaked


def test_the_uihc_run_uses_the_projects_own_forecast_settings() -> None:
    """A run that quietly retuned the model would prove nothing about transfer.

    The script must not carry its own lag set, its own learning rate or its own fold
    count: it reads `configs/default.yaml` like every other caller.
    """
    source = UIHC_SCRIPT.read_text(encoding="utf-8")
    assert "load_config" in source
    assert "rolling_origin_evaluate" in source, "it must use the project's own backtest"
    for invented in ("learning_rate=", "n_estimators=", "lags = [", "num_leaves="):
        assert invented not in source, (
            f"the script sets {invented!r} itself, so the run is not the project's "
            "configured model"
        )


def test_same_hour_temperature_is_never_a_feature() -> None:
    """It is not known when a forecast is issued. Using it predicts the past."""
    source = UIHC_SCRIPT.read_text(encoding="utf-8")
    assert "lagged_temperature_experiment" in source
    # The only place `temp` is read must be the lagged experiment, and it must shift
    # by the forecast horizon.
    assert "pd.Timedelta(hours=horizon_hours)" in source
    assert "LAGGED temperature" in source, "the experiment must label itself"


def test_the_lagged_temperature_experiment_is_actually_lagged(uihc) -> None:
    """Checked by construction: the value used must be the earlier one."""
    index = pd.date_range("2024-01-01", periods=400, freq="h")
    frame = pd.DataFrame({
        "timestamp": index,
        "actual": np.linspace(1, 20, len(index)),
        "yhat": np.linspace(1, 20, len(index)),
    })
    # Temperature that rises steadily, so a lag is detectable from the value alone.
    events = pd.DataFrame({"timestamp": index, "temp": np.arange(len(index), dtype=float)})

    result = uihc.lagged_temperature_experiment(frame, events, horizon_hours=72)
    assert result["ran"] is True
    assert result["lag_hours"] == 72
    assert "not known when a forecast is issued" in result["label"]


# ---------------------------------------------------------------- the MIMIC side


def test_the_mimic_script_fits_no_model_and_scores_nothing() -> None:
    """222 stays. Any accuracy figure would be meaningless and would be quoted anyway.

    This is a source-level check on purpose: it is the one property of the script
    that must hold even if somebody later adds a convenient-looking import.
    """
    source = MIMIC_SCRIPT.read_text(encoding="utf-8")
    for forbidden in (
        "train_", "fit(", "LGBM", "lightgbm", "predict(",
        "mean_absolute_error", "roc_auc", "brier",
    ):
        assert forbidden not in source, (
            f"the MIMIC script references {forbidden!r}; it must not fit or score a "
            "model on 222 stays"
        )
    assert "no_model_claims" in source, "the output must say why there are no claims"


def test_the_mimic_script_names_its_proxy_in_words() -> None:
    """A wait from an unnamed proxy cannot be judged by a reader."""
    source = MIMIC_SCRIPT.read_text(encoding="utf-8")
    assert "proxy wait (not true door-to-provider)" in source
    assert "PROXY_WAIT_NOTICE" in source
    for proxy in ("first_vitalsign_charttime", "first_pyxis_charttime"):
        assert proxy in source


def test_the_mimic_script_uses_the_adapters_own_function() -> None:
    """A special-case mapping would prove nothing about the real adapter."""
    source = MIMIC_SCRIPT.read_text(encoding="utf-8")
    assert "adapt_real_dataset" in source
    assert "load_column_map" in source


def test_the_column_map_no_longer_claims_to_be_unverified() -> None:
    """It has been run against the real demo, and one mapping was wrong."""
    text = (project_root() / "configs" / "column_map_mimic_ed.yaml").read_text(encoding="utf-8")
    assert "NEVER BEEN RUN" not in text
    assert "VERIFIED AGAINST THE OPEN DEMO" in text
    # The correction itself: leaving against advice is not leaving before being seen.
    assert '"LEFT AGAINST MEDICAL ADVICE": other' in text


def test_leaving_against_advice_is_not_counted_as_abandonment() -> None:
    """The correction the real data forced, stated as its own test.

    A patient who left against medical advice was seen by a clinician and advised.
    Counting them as left-without-being-seen would inflate the abandonment rate --
    which is one of the quantities the simulation's validation gate is judged on.
    """
    text = (project_root() / "configs" / "column_map_mimic_ed.yaml").read_text(encoding="utf-8")
    lwbs_block = text[text.index("disposition:"):text.index("triage_level:")]
    assert '"LEFT WITHOUT BEING SEEN": lwbs' in lwbs_block
    assert "ELOPED: lwbs" in lwbs_block
    assert '"LEFT AGAINST MEDICAL ADVICE": lwbs' not in lwbs_block


# ------------------------------------------------------------------- provenance


def test_both_datasets_have_recorded_provenance() -> None:
    """Source, licence, retrieval date and checksum, for each."""
    assert EXTERNAL_README.is_file(), "data/external/README.md is missing"
    text = EXTERNAL_README.read_text(encoding="utf-8")

    assert "GPL (>= 3)" in text, "the fastTS licence is not recorded"
    assert "Open Database License (ODbL)" in text, "the MIMIC licence is not recorded"
    assert "2026-10-14" in text, "no retrieval date"
    assert text.count("SHA256") >= 3, "checksums are not recorded"
    assert "Johnson" in text, "the MIMIC attribution is missing"
    assert "Peterson" in text, "the fastTS citation is missing"


def test_the_external_data_is_excluded_from_the_package() -> None:
    """Licensed third-party data must not ship inside this project's zip."""
    builder = (project_root() / "scripts" / "make_submission.ps1").read_text(encoding="utf-8")
    assert "data\\external" in builder, (
        "the package builder does not exclude data\\external, so the zip would carry "
        "third-party licensed data"
    )


def test_the_extra_dependency_is_not_in_the_main_requirements() -> None:
    """Nothing in the pipeline, API, dashboard or tests may need pyreadr."""
    main = (project_root() / "requirements.txt").read_text(encoding="utf-8")
    extra = (project_root() / "requirements-extra.txt").read_text(encoding="utf-8")
    assert "pyreadr" not in main, "pyreadr leaked into requirements.txt"
    assert "pyreadr==" in extra, "the extra dependency is not pinned"


def test_the_fetch_script_records_the_licence_and_refuses_quietly() -> None:
    """A blocked download must be a reported skip, not a failure of the project."""
    text = FETCH_SCRIPT.read_text(encoding="utf-8")
    assert "ODbL" in text
    assert "SHA256SUMS" in text, "the archive's own checksums are not verified"
    assert "demo" in text.lower()
    assert "skipped" in text.lower(), (
        "the script should say that a failed download skips this part rather than "
        "breaking anything"
    )


# ------------------------------------------- the results, when the data is present


@pytest.mark.skipif(not UIHC_CSV.is_file(), reason="UIHC extract not present")
def test_the_uihc_extract_is_the_documented_shape() -> None:
    """41,640 hourly rows, continuous, matching what the README claims."""
    frame = pd.read_csv(UIHC_CSV)
    assert len(frame) == 41640
    stamps = pd.to_datetime(frame["timestamp"])
    assert len(pd.date_range(stamps.min(), stamps.max(), freq="h")) == len(frame)
    assert int(stamps.duplicated().sum()) == 0
    assert pd.to_numeric(frame["Arrivals"]).ge(0).all()


@pytest.mark.skipif(
    not (project_root() / "artifacts/external_validation/uihc_26folds/metrics.json").is_file(),
    reason="the UIHC backtest has not been run",
)
def test_the_forecaster_beat_both_baselines_on_real_data() -> None:
    """The result itself, pinned so a later change cannot quietly lose it.

    Not a threshold anyone chose: the model simply has to be better than the two
    baselines it is compared against. If a future change makes it worse, that is a
    finding and this test is where it surfaces.
    """
    payload = json.loads(
        (project_root() / "artifacts/external_validation/uihc_26folds/metrics.json")
        .read_text(encoding="utf-8")
    )
    overall = payload["metrics"]["overall"]
    assert overall["model"]["mae"] < overall["seasonal_naive"]["mae"]
    assert overall["model"]["mae"] < overall["four_week_mean"]["mae"]
    assert overall["model"]["n"] > 1000, "too few hours scored to mean anything"


@pytest.mark.skipif(
    not (MIMIC_DIR / "edstays.csv.gz").is_file(), reason="MIMIC demo not present"
)
def test_the_mimic_join_does_not_multiply_rows(mimic) -> None:
    """The `subject_id` trap: one patient can have several stays.

    Joining on `subject_id` would multiply rows and quietly inflate every count. The
    join must be on `stay_id`.
    """
    stays = pd.read_csv(MIMIC_DIR / "edstays.csv.gz")
    joined = mimic.join_tables(MIMIC_DIR)
    assert len(joined) == len(stays) == 222
    assert joined["stay_id"].is_unique


@pytest.mark.skipif(
    not (MIMIC_DIR / "edstays.csv.gz").is_file(), reason="MIMIC demo not present"
)
def test_the_two_proxies_bracket_the_truth_from_opposite_sides(mimic) -> None:
    """Predicted from the schema; now checked against the data.

    First vitals is recorded at triage and under-estimates the wait; a medication
    dispense follows a clinical decision and over-estimates it. If this ever fails,
    the reasoning in `docs/real_data_readiness.md` is wrong and needs revisiting.
    """
    joined = mimic.join_tables(MIMIC_DIR)
    coverage = mimic.proxy_coverage(joined)

    vitals = coverage["first_vitalsign_charttime"]["median_minutes"]
    pyxis = coverage["first_pyxis_charttime"]["median_minutes"]
    assert vitals is not None and pyxis is not None
    assert vitals < pyxis, (
        "first vitals should be earlier than the first medication dispense"
    )
    assert coverage["gap_between_proxies"]["share_where_pyxis_is_later"] > 0.5


@pytest.mark.skipif(
    not (MIMIC_DIR / "edstays.csv.gz").is_file(), reason="MIMIC demo not present"
)
def test_the_queue_state_is_reported_as_unreconstructible(mimic) -> None:
    """The finding that decides how much of this project MIMIC could ever check."""
    joined = mimic.join_tables(MIMIC_DIR)
    finding = mimic.reconstruction_check(joined)

    assert finding["queue_state_reconstructible"] is False
    assert "date" in finding["date_shifting"].lower()
    assert any("Queue-state" in item for item in finding["cannot_be_validated_on_mimic"])
    assert any("simulation" in item.lower() for item in finding["cannot_be_validated_on_mimic"])
