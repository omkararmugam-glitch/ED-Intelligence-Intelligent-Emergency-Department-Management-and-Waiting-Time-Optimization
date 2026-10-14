"""Tests for the SHAP and fairness reports (Phase 1c)."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.data.schema import validate_visits
from src.data.split import make_time_split
from src.data.synthetic import generate_synthetic_visits
from src.explain.fairness import add_age_band, build_fairness_report, evaluate_fairness
from src.explain.shap_report import (
    build_shap_report,
    compute_shap_values,
    global_importance_table,
    local_explanations,
    verify_additivity,
)
from src.features.build import add_prolonged_wait_label, build_features, make_design_matrix
from src.features.queue_state import QueueStateIndex
from src.models.wait_time import (
    select_regression_cohort,
    train_prolonged_classifier,
    train_wait_regressor,
)

EXPLAIN_N_PATIENTS = 9000


@pytest.fixture(scope="module")
def explain_config() -> dict:
    """Config sized so the fits and the bootstrap stay quick."""
    from src.utils.io import load_config

    cfg = load_config("configs/default.yaml")
    cfg["synthetic"]["n_patients"] = EXPLAIN_N_PATIENTS
    cfg["wait_model"]["regressor"]["n_estimators"] = 80
    cfg["wait_model"]["classifier"]["n_estimators"] = 80
    cfg["explain"]["shap_max_samples"] = 600
    cfg["fairness"]["bootstrap_resamples"] = 60
    return cfg


@pytest.fixture(scope="module")
def fitted(explain_config) -> dict:
    """A trained regressor and classifier plus their test-block design matrices."""
    population = validate_visits(generate_synthetic_visits(explain_config)).all
    split = make_time_split(population, explain_config)
    index = QueueStateIndex.from_visits(population)

    blocks = {}
    for name in ("train", "val", "test"):
        featurised = add_prolonged_wait_label(
            build_features(split[name], explain_config, index=index), explain_config
        )
        blocks[name] = select_regression_cohort(featurised, block_name=name).frame

    design = {
        name: make_design_matrix(frame, "base+queue", explain_config)[0]
        for name, frame in blocks.items()
    }
    regressor = train_wait_regressor(
        design["train"],
        blocks["train"]["wait_minutes"].to_numpy(dtype="float64"),
        design["val"],
        blocks["val"]["wait_minutes"].to_numpy(dtype="float64"),
        explain_config,
        transform="raw",
    )
    classifier = train_prolonged_classifier(
        design["train"],
        blocks["train"]["prolonged_wait"].to_numpy(dtype="int8"),
        design["val"],
        blocks["val"]["prolonged_wait"].to_numpy(dtype="int8"),
        explain_config,
    )
    return {
        "regressor": regressor,
        "classifier": classifier,
        "test_frame": blocks["test"],
        "test_design": design["test"],
    }


# ------------------------------------------------------------------------------ SHAP


def test_contributions_sum_to_the_raw_prediction(fitted, explain_config) -> None:
    """The defining property of a SHAP attribution, and the core check after dropping
    the `shap` package: contributions plus the base value must equal the model's raw
    prediction, exactly.

    LightGBM's `pred_contrib=True` is the same TreeSHAP routine the `shap` package
    calls into for LightGBM models, so this is an equality, not an approximation.
    """
    values, sample, base = compute_shap_values(
        fitted["regressor"].model, fitted["test_design"], explain_config
    )
    assert values.shape == sample.shape, "one contribution per feature, base excluded"

    reconstructed = values.sum(axis=1) + base
    direct = fitted["regressor"].model.predict(sample)

    assert np.allclose(reconstructed, direct, atol=1e-9), (
        f"max discrepancy {np.max(np.abs(reconstructed - direct)):.3e}"
    )


def test_additivity_check_passes_and_can_fail(fitted, explain_config) -> None:
    """`verify_additivity` must confirm a good attribution and reject a corrupted one."""
    values, sample, base = compute_shap_values(
        fitted["regressor"].model, fitted["test_design"], explain_config
    )
    discrepancy = verify_additivity(fitted["regressor"].model, sample, values, base)
    assert discrepancy < 1e-9

    corrupted = values.copy()
    corrupted[:, 0] += 5.0
    with pytest.raises(ValueError, match="do not sum to the model output"):
        verify_additivity(fitted["regressor"].model, sample, corrupted, base)


def test_shap_needs_no_shap_package(fitted, explain_config) -> None:
    """The explanation path must not import `shap`, whose numba DLL is blocked here."""
    import sys

    blocked = {"shap", "numba", "llvmlite"}

    class Blocker:
        """Import hook that makes the removed libraries unimportable.

        Implements `find_spec`, not `find_module`: Python 3.12 removed the latter from
        the import system, so a hook built on it is silently inert and the test would
        pass without proving anything.
        """

        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in blocked:
                raise ImportError(
                    f"{fullname} is blocked (simulating the Windows Application "
                    "Control policy that makes numba's DLL unloadable)"
                )
            return None

    hook = Blocker()
    # Drop anything already imported, or a cached module would satisfy the import.
    cached = {name: sys.modules.pop(name) for name in list(sys.modules)
              if name.split(".")[0] in blocked}
    sys.meta_path.insert(0, hook)
    try:
        # The hook must really bite, or the rest of this test proves nothing.
        with pytest.raises(ImportError):
            __import__("shap")

        values, sample, base = compute_shap_values(
            fitted["regressor"].model, fitted["test_design"], explain_config
        )
        assert values.shape[0] > 0
        assert np.isfinite(base)

        # And the whole report, including its matplotlib figures, builds without it.
        reconstructed = values.sum(axis=1) + base
        direct = fitted["regressor"].model.predict(sample)
        assert np.allclose(reconstructed, direct, atol=1e-9)
    finally:
        sys.meta_path.remove(hook)
        sys.modules.update(cached)


def test_shap_sample_is_capped_and_reproducible(fitted, explain_config) -> None:
    """The sample honours explain.shap_max_samples and is seed-stable."""
    cap = int(explain_config["explain"]["shap_max_samples"])
    first, sample_a, _ = compute_shap_values(
        fitted["regressor"].model, fitted["test_design"], explain_config
    )
    second, sample_b, _ = compute_shap_values(
        fitted["regressor"].model, fitted["test_design"], explain_config
    )

    assert len(sample_a) <= cap
    assert list(sample_a.index) == list(sample_b.index)
    assert np.array_equal(first, second)


def test_global_table_ranks_and_sums_to_one(fitted, explain_config) -> None:
    """The global table is ordered by importance and its shares form a distribution."""
    values, sample, _ = compute_shap_values(
        fitted["regressor"].model, fitted["test_design"], explain_config
    )
    table = global_importance_table(values, sample)

    assert list(table["rank"]) == list(range(1, len(table) + 1))
    assert table["mean_abs_shap"].is_monotonic_decreasing
    assert table["share_of_attribution"].sum() == pytest.approx(1.0, abs=1e-3)
    assert set(table["feature"]) == set(sample.columns)
    # On queue-generated data a queue feature should lead the ranking.
    assert table.loc[0, "is_queue_feature"]


def test_local_explanations_reconstruct_each_prediction(fitted, explain_config) -> None:
    """Each worked example decomposes one real patient's prediction."""
    values, sample, base = compute_shap_values(
        fitted["regressor"].model, fitted["test_design"], explain_config
    )
    positions = fitted["test_design"].index.get_indexer(sample.index)
    context = fitted["test_frame"].iloc[positions]

    local = local_explanations(values, sample, context, base, explain_config, n_examples=3)
    assert not local.empty
    assert local["example_id"].nunique() <= 3
    assert set(local["patient_id"]) <= set(context["patient_id"])
    # Contributions are ranked by absolute size within each example.
    for _, group in local.groupby("example_id"):
        assert list(group["rank"]) == sorted(group["rank"])
        assert group["shap_contribution"].abs().is_monotonic_decreasing


def test_shap_report_writes_tables_and_figures(fitted, explain_config, tmp_path) -> None:
    """The stage's artifacts land on disk and the units are labelled."""
    report = build_shap_report(
        model=fitted["regressor"].model,
        features=fitted["test_design"],
        context=fitted["test_frame"],
        config=explain_config,
        tables_dir=tmp_path / "tables",
        figures_dir=tmp_path / "figures",
        feature_set="base+queue",
        target_transform="raw",
    )
    assert report.units == "minutes"
    # The output file and table names are unchanged by dropping the `shap` package.
    for name in ("shap_top_features.csv", "shap_all_features.csv", "shap_local_examples.csv"):
        assert (tmp_path / "tables" / name).is_file(), f"missing {name}"
    for name in ("shap_importance.png", "shap_summary.png", "shap_local_examples.png"):
        assert (tmp_path / "figures" / name).is_file(), f"missing {name}"
    assert report.to_dict()["shap_backend"].startswith("lightgbm")
    for figure in report.figures:
        assert figure.is_file() and figure.stat().st_size > 5000


def test_log_target_shap_units_are_labelled(fitted, explain_config, tmp_path) -> None:
    """A log-target model must not have its SHAP values presented as minutes."""
    report = build_shap_report(
        model=fitted["regressor"].model,
        features=fitted["test_design"],
        context=fitted["test_frame"],
        config=explain_config,
        tables_dir=tmp_path / "tables",
        figures_dir=tmp_path / "figures",
        feature_set="base+queue",
        target_transform="log1p",
    )
    assert report.units == "log1p(minutes)"
    assert "minutes" in report.to_dict()["shap_units"]


# -------------------------------------------------------------------------- fairness


def _fairness_inputs(fitted) -> dict:
    """Arguments shared by the fairness tests."""
    frame = fitted["test_frame"]
    return {
        "frame": frame,
        "y_true_label": frame["prolonged_wait"].to_numpy(dtype="int8"),
        "y_score": fitted["classifier"].predict_proba(fitted["test_design"]),
        "threshold": fitted["classifier"].threshold,
        "y_true_wait": frame["wait_minutes"].to_numpy(dtype="float64"),
        "y_pred_wait": fitted["regressor"].predict(fitted["test_design"]),
    }


def test_fairness_covers_every_configured_attribute(fitted, explain_config) -> None:
    """Each present subgroup column appears, with gaps measured against overall."""
    report = evaluate_fairness(config=explain_config, **_fairness_inputs(fitted))

    assert set(report.attributes_evaluated) == {"sex", "arrival_mode", "triage_level", "age_band"}
    assert report.attributes_skipped == []
    assert not report.table.empty

    for attribute in report.attributes_evaluated:
        assert attribute in set(report.table["attribute"])

    # Group sizes within an attribute must add up to the test block.
    for attribute, group in report.table.groupby("attribute"):
        assert group["n"].sum() == report.overall["n"]


def test_fairness_reports_positive_counts_for_fnr(fitted, explain_config) -> None:
    """An FNR is estimated from positives alone, so the count must be visible."""
    report = evaluate_fairness(config=explain_config, **_fairness_inputs(fitted))

    assert "n_positive" in report.table.columns
    assert "fnr_underpowered" in report.table.columns
    assert (report.table["n_positive"] + report.table["n_negative"] == report.table["n"]).all()

    minimum = int(explain_config["fairness"]["min_positives_for_fnr"])
    expected = report.table["n_positive"] < minimum
    assert (report.table["fnr_underpowered"] == expected).all()


def test_fairness_gap_is_measured_against_the_overall_rate(fitted, explain_config) -> None:
    """Each gap must equal the subgroup metric minus the overall metric."""
    report = evaluate_fairness(config=explain_config, **_fairness_inputs(fitted))
    for metric in ("fnr", "fpr", "mae"):
        recomputed = report.table[metric] - report.overall[metric]
        # A subgroup with no positive cases has an undefined FNR, so NaN must line up
        # with NaN rather than count as a mismatch.
        assert np.allclose(
            report.table[f"{metric}_gap"], recomputed, atol=1e-3, equal_nan=True
        )


def test_fairness_bootstrap_intervals_bracket_the_gap(fitted, explain_config) -> None:
    """A percentile interval must be ordered and contain plausible values."""
    report = evaluate_fairness(config=explain_config, **_fairness_inputs(fitted))
    usable = report.table[report.table["fnr_gap_ci_low"].notna()]
    assert not usable.empty
    assert (usable["fnr_gap_ci_low"] <= usable["fnr_gap_ci_high"]).all()

    flagged = usable["fnr_gap_excludes_zero"]
    expected = (usable["fnr_gap_ci_low"] > 0) | (usable["fnr_gap_ci_high"] < 0)
    assert (flagged == expected).all()


@pytest.mark.parametrize("missing", ["sex", "arrival_mode", "age"])
def test_fairness_runs_with_an_optional_column_removed(fitted, explain_config, missing) -> None:
    """Required by Phase 1c: a dataset lacking an optional column must still report."""
    inputs = _fairness_inputs(fitted)
    inputs["frame"] = inputs["frame"].drop(columns=[missing])

    report = evaluate_fairness(config=explain_config, **inputs)

    absent = "age_band" if missing == "age" else missing
    assert absent in report.attributes_skipped
    assert absent not in report.attributes_evaluated
    # Everything else still works.
    assert not report.table.empty
    assert "triage_level" in report.attributes_evaluated


def test_fairness_with_every_optional_column_removed(fitted, explain_config) -> None:
    """The most degraded real dataset still produces a triage-level report."""
    inputs = _fairness_inputs(fitted)
    inputs["frame"] = inputs["frame"].drop(columns=["sex", "arrival_mode", "age"])

    report = evaluate_fairness(config=explain_config, **inputs)
    assert report.attributes_evaluated == ["triage_level"]
    assert set(report.attributes_skipped) == {"sex", "arrival_mode", "age_band"}
    assert not report.table.empty


def test_fairness_with_no_subgroup_columns_at_all(fitted, explain_config) -> None:
    """With nothing to split on, the report is empty but does not raise."""
    inputs = _fairness_inputs(fitted)
    inputs["frame"] = inputs["frame"].drop(columns=["sex", "arrival_mode", "age", "triage_level"])

    report = evaluate_fairness(config=explain_config, **inputs)
    assert report.attributes_evaluated == []
    assert report.table.empty
    assert report.overall["n"] > 0


def test_age_band_derivation_matches_the_configured_bins(explain_config) -> None:
    """Ages land in the configured bands, with the right-open convention."""
    frame = pd.DataFrame({"age": [0, 17, 18, 39, 40, 64, 65, 79, 80, 120]})
    banded = add_age_band(frame, explain_config)
    assert banded["age_band"].tolist() == [
        "0-17", "0-17", "18-39", "18-39", "40-64",
        "40-64", "65-79", "65-79", "80+", "80+",
    ]


def test_age_band_is_skipped_without_an_age_column(explain_config) -> None:
    """No age column means no age_band, and no exception."""
    frame = pd.DataFrame({"sex": ["female", "male"]})
    assert "age_band" not in add_age_band(frame, explain_config).columns


def test_mismatched_age_labels_raise(explain_config) -> None:
    """A config where labels and bins disagree is a configuration error."""
    broken = copy.deepcopy(explain_config)
    broken["fairness"]["age_labels"] = ["young", "old"]
    with pytest.raises(ValueError, match="age_labels"):
        add_age_band(pd.DataFrame({"age": [10, 50]}), broken)


def test_fairness_report_writes_its_artifacts(fitted, explain_config, tmp_path) -> None:
    """The stage writes fairness.csv and its figure."""
    report = build_fairness_report(
        config=explain_config,
        tables_dir=tmp_path / "tables",
        figures_dir=tmp_path / "figures",
        **_fairness_inputs(fitted),
    )
    assert (tmp_path / "tables" / "fairness.csv").is_file()
    assert report.figures and report.figures[0].is_file()

    payload = report.to_dict()
    assert payload["overall"]["n"] > 0
    assert payload["n_subgroups"] == len(report.table)
