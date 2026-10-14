"""Tests for the report writer, which is prose generated from artifacts.

The numbers in the report are already covered by the stage tests that produce them.
What is tested here is the narrative: a sentence that interprets a table must agree
with that table, and must keep agreeing when the table changes. These are the claims
that would otherwise rot silently - "future work" for something already measured, a
hardcoded count beside a column that counts itself, a deliberate skip listed as a
fault.

Each test builds a small artifact directory by hand, so a wording regression fails
here in a second rather than after a full pipeline run.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.report import build_report, save_run_stages

#: The per-acuity threshold comparison, shaped like `triage_thresholds.csv`: ESI 3-5
#: fitted, ESI 1-2 refused for want of positives, and a pooled-vs-per-acuity pair at
#: `overall` whose recalls differ by far less than their intervals are wide.
THRESHOLDS = pd.DataFrame(
    [
        ("overall", "pooled", 0.6982, 8198, 3695, 0.8755, 0.8640, 0.8862, 0.1245,
         0.1143, 0.1361, 0.9577, "pooled"),
        ("overall", "per_triage", None, 8198, 3695, 0.8747, 0.8640, 0.8856, 0.1253,
         0.1151, 0.1377, 0.9573, "mixed"),
        ("ESI 1", "pooled", 0.6982, 188, 1, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, None,
         "too few positives; using the pooled threshold"),
        ("ESI 1", "per_triage", 0.6982, 188, 1, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, None,
         "too few positives; using the pooled threshold"),
        ("ESI 2", "pooled", 0.6982, 1171, 29, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, None,
         "too few positives; using the pooled threshold"),
        ("ESI 2", "per_triage", 0.6982, 1171, 29, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, None,
         "too few positives; using the pooled threshold"),
        ("ESI 3", "pooled", 0.6982, 3345, 1620, 0.8932, 0.8790, 0.9077, 0.1068,
         0.0929, 0.1226, 0.9513, "fitted on the validation block"),
        ("ESI 3", "per_triage", 0.7143, 3345, 1620, 0.8846, 0.8681, 0.8987, 0.1154,
         0.1022, 0.1325, 0.9528, "fitted on the validation block"),
        ("ESI 4", "pooled", 0.6982, 2555, 1465, 0.8655, 0.8468, 0.8818, 0.1345,
         0.1170, 0.1531, 0.9650, "fitted on the validation block"),
        ("ESI 4", "per_triage", 0.6677, 2555, 1465, 0.8765, 0.8593, 0.8930, 0.1235,
         0.1068, 0.1427, 0.9625, "fitted on the validation block"),
        ("ESI 5", "pooled", 0.6982, 939, 580, 0.8966, 0.8699, 0.9202, 0.1034,
         0.0791, 0.1283, 0.9576, "fitted on the validation block"),
        ("ESI 5", "per_triage", 0.7175, 939, 580, 0.8879, 0.8628, 0.9124, 0.1121,
         0.0865, 0.1378, 0.9572, "fitted on the validation block"),
    ],
    columns=[
        "scope", "rule", "threshold", "n", "n_positive", "recall", "recall_ci_low",
        "recall_ci_high", "fnr", "fnr_ci_low", "fnr_ci_high", "precision",
        "threshold_source",
    ],
)

#: The simulation gate, shaped like `sim_validation.csv`. Two rows are marked as
#: fitted, which is the number the report must count rather than assume.
SIM_VALIDATION = pd.DataFrame(
    [
        ("mean wait", "mean_wait_minutes", 35.0116, 32.5352, -2.4764, 0.0707, 0.20,
         3.0, True, "relative", True),
        ("p90 wait", "p90_wait_minutes", 100.0, 89.2142, -10.7858, 0.1079, 0.25,
         3.0, True, "relative", False),
        ("mean LOS", "mean_los_minutes", 160.6365, 152.038, -8.5985, 0.0535, 0.20,
         3.0, True, "relative", False),
        ("abandonment rate", "abandonment_rate", 0.0358, 0.0298, -0.0060, 0.1668,
         0.50, 0.01, True, "relative", True),
        ("abandonment rate ESI 4", "abandonment_rate_esi4", 0.0557, 0.0384, -0.0174,
         0.3116, 0.50, 0.01, True, "relative", False),
    ],
    columns=[
        "metric", "key", "observed", "simulated", "absolute_error", "relative_error",
        "tolerance", "absolute_floor", "within_tolerance", "passed_on",
        "used_for_fitting",
    ],
)

#: Subgroup results, shaped like `fairness.csv`. ESI 2 is the finding: a complete
#: false-negative rate over enough positives to believe it.
FAIRNESS = pd.DataFrame(
    [
        ("triage_level", "2", 1171, 29, 1.0, 0.8755, 0.8657, 0.8853, False, -7.9743),
        ("triage_level", "3", 3345, 1620, 0.1068, -0.0177, -0.0351, -0.0019, False,
         1.7709),
        ("triage_level", "1", 188, 1, 1.0, 0.8755, 0.8657, 0.8850, True, -7.9221),
    ],
    columns=[
        "attribute", "group", "n", "n_positive", "fnr", "fnr_gap", "fnr_gap_ci_low",
        "fnr_gap_ci_high", "fnr_underpowered", "mae_gap",
    ],
)

ABLATION = pd.DataFrame(
    [
        ("base", "raw", "raw", 8198, 33.930, 53.073),
        ("base+queue", "raw", "raw", 8198, 11.524, 19.743),
        ("base+queue", "log1p", "log1p_median", 8198, 11.712, 19.821),
    ],
    columns=["feature_set", "target", "prediction_kind", "n_test", "mae_minutes",
             "rmse_minutes"],
)

CONFORMAL_STATIC = pd.DataFrame(
    [
        ("overall", 8198, 0.9, 0.8490, -0.0510, 39.04, 43.80),
        ("1", 188, 0.9, 0.8404, -0.0596, 10.65, 11.35),
        ("4", 2555, 0.9, 0.8329, -0.0671, 41.78, 50.42),
        ("5", 939, 0.9, 0.8839, -0.0161, 51.81, 61.61),
    ],
    columns=["triage_level", "n", "nominal_coverage", "empirical_coverage",
             "coverage_gap", "mean_width_minutes", "median_width_minutes"],
)

CONFORMAL_ROLLING = pd.DataFrame(
    [
        ("overall", "static", 8198, 0.9, 0.8490, -0.0510, 39.04, 43.80),
        ("overall", "rolling", 8198, 0.9, 0.8875, -0.0125, 46.49, 50.26),
        ("ESI 4", "static", 2555, 0.9, 0.8329, -0.0671, 41.78, 50.42),
        ("ESI 4", "rolling", 2555, 0.9, 0.8834, -0.0166, 52.45, 56.25),
    ],
    columns=["scope", "method", "n", "nominal_coverage", "empirical_coverage",
             "coverage_gap", "mean_width_minutes", "median_width_minutes"],
)

WAIT_MODELS: dict[str, Any] = {
    "ablation": {
        "best_by_test_mae": {
            "feature_set": "base+queue",
            "target": "raw",
            "prediction_kind": "raw",
            "mae_minutes": 11.524,
            "rmse_minutes": 19.743,
        },
        "caveat": "CAVEAT: on synthetic data this comparison overstates queue features.",
    },
    "baseline": {"k": 20, "blocks": {"test": {"regression": {"mae": 33.9, "rmse": 53.1}}}},
}

FAIRNESS_METRICS: dict[str, Any] = {
    "overall": {
        "n": 8198, "fnr": 0.1245, "fpr": 0.0318, "recall": 0.8755, "mae": 11.524,
    }
}

FORECAST: dict[str, Any] = {
    "ran": True,
    "n_hours": 288,
    "n_folds": 4,
    "n_peak_hours": 60,
    "peak_threshold_arrivals": 11.0,
    "mae_improvement_over_baseline_percent": 20.93,
    "peak_hours": {
        "model": {"mae": 4.3636, "n": 60},
        "seasonal_naive": {"mae": 4.1000, "n": 60},
    },
}


@pytest.fixture()
def artifacts(tmp_path: Path) -> dict[str, Any]:
    """A config pointing at a hand-built artifact directory, with the tables written.

    Deliberately partial: enough for the sections under test, so a missing artifact
    is also exercised on every call.
    """
    tables, metrics, figures = (tmp_path / name for name in ("tables", "metrics", "figures"))
    for directory in (tables, metrics, figures):
        directory.mkdir()

    THRESHOLDS.to_csv(tables / "triage_thresholds.csv", index=False)
    SIM_VALIDATION.to_csv(tables / "sim_validation.csv", index=False)
    FAIRNESS.to_csv(tables / "fairness.csv", index=False)
    ABLATION.to_csv(tables / "ablation.csv", index=False)
    CONFORMAL_STATIC.to_csv(tables / "conformal_coverage.csv", index=False)
    CONFORMAL_ROLLING.to_csv(tables / "conformal_static_vs_rolling.csv", index=False)
    (metrics / "wait_models.json").write_text(json.dumps(WAIT_MODELS), encoding="utf-8")
    (metrics / "fairness.json").write_text(json.dumps(FAIRNESS_METRICS), encoding="utf-8")
    (metrics / "forecast.json").write_text(json.dumps(FORECAST), encoding="utf-8")

    return {
        "paths": {
            "artifacts_dir": str(tmp_path),
            "tables_dir": str(tables),
            "metrics_dir": str(metrics),
            "figures_dir": str(figures),
            "report": str(tmp_path / "report.md"),
        }
    }


def _bullets(report: str, heading: str) -> list[str]:
    """The bullets of one `## heading` section, each joined onto a single line."""
    body = report.split(f"## {heading}\n", 1)[1].split("\n## ", 1)[0]
    bullets: list[str] = []
    for line in body.splitlines():
        if line.startswith("- "):
            bullets.append(line[2:].strip())
        elif line.startswith("  ") and bullets:
            bullets[-1] += " " + line.strip()
    return bullets


# ------------------------------------------------- (a) nothing already done is "future work"


def test_future_work_does_not_claim_per_acuity_thresholds_were_never_tried(
    artifacts: dict[str, Any]
) -> None:
    """The artifacts show the mitigation was fitted, so it cannot be future work.

    This is the regression that matters most in this file: the threshold module was
    built, fitted on the validation block and scored on test, while the report still
    said mitigation "is future work". A reader who believes the prose then thinks an
    untried fix is available.
    """
    report = build_report(artifacts)

    # The exact regression, and the family it belongs to.
    assert "is future work" not in report
    assert "Per-class fairness mitigation" not in report
    assert re.search(
        r"(?i)per-acuity thresholds?[^.]*\b(is|are|remains?)\b[^.]*(future work|still to)", report
    ) is None

    # Any mention of the mitigation in a forward-looking place must say it was done.
    for bullet in _bullets(report, "Future work"):
        if re.search(r"(?i)per-acuity threshold", bullet):
            assert "already fitted and measured" in bullet, (
                "the report lists per-acuity thresholds as work still to do, but "
                f"triage_thresholds.csv shows them fitted: {bullet!r}"
            )
        assert "would address it directly" not in bullet

    # And the fairness section says so positively, naming the levels that were fitted.
    assert "Mitigation was tried" in report
    assert "ESI 3, ESI 4, ESI 5" in report


def test_future_work_names_what_is_actually_open(artifacts: dict[str, Any]) -> None:
    """What remains is a method for scarce high-acuity positives, not a threshold."""
    bullets = _bullets(build_report(artifacts), "Future work")
    scarcity = [b for b in bullets if "very few high-acuity positives" in b]
    assert len(scarcity) == 1, bullets
    assert "acuity-aware training objective" in scarcity[0]
    assert "ESI 1 (n_positive = 1)" in scarcity[0]
    assert "ESI 2 (n_positive = 29)" in scarcity[0]


def test_threshold_verdict_follows_the_table(artifacts: dict[str, Any]) -> None:
    """A recall change smaller than the interval must be called no improvement."""
    report = build_report(artifacts)
    assert "There was no meaningful improvement" in report
    assert "Pooled recall **0.8755** against per-acuity **0.8747**" in report


def test_threshold_verdict_changes_when_the_table_does(
    artifacts: dict[str, Any], tmp_path: Path
) -> None:
    """A real gain must not be reported as noise. The sentence is data, not boilerplate."""
    improved = THRESHOLDS.copy()
    overall_per_triage = (improved["scope"] == "overall") & (improved["rule"] == "per_triage")
    improved.loc[overall_per_triage, ["recall", "recall_ci_low", "recall_ci_high"]] = [
        0.9500, 0.9400, 0.9600,
    ]
    improved.to_csv(Path(artifacts["paths"]["tables_dir"]) / "triage_thresholds.csv", index=False)

    report = build_report(artifacts)
    assert "no meaningful improvement" not in report
    assert "it is a real improvement" in report


# ------------------------------------------------------- (b) the fitted-metric count


def test_fitted_metric_count_matches_the_validation_table(artifacts: dict[str, Any]) -> None:
    """The count in the prose must come from `used_for_fitting`, not from memory."""
    expected = int(SIM_VALIDATION["used_for_fitting"].sum())
    assert expected == 2  # guards the fixture itself

    report = build_report(artifacts)
    assert f"**{expected} metrics**" in report
    assert "(mean wait, abandonment rate) are used to fit" in report
    assert f"fitted to {expected} aggregate metrics" in report
    assert "fitted to one aggregate" not in report


@pytest.mark.parametrize("n_fitted", [1, 3])
def test_fitted_metric_count_tracks_the_column(
    artifacts: dict[str, Any], n_fitted: int
) -> None:
    """Change the column, and both the section and the limitation follow it."""
    table = SIM_VALIDATION.copy()
    table["used_for_fitting"] = [index < n_fitted for index in range(len(table))]
    table.to_csv(Path(artifacts["paths"]["tables_dir"]) / "sim_validation.csv", index=False)

    report = build_report(artifacts)
    unit = "metric" if n_fitted == 1 else "metrics"
    assert f"**{n_fitted} {unit}**" in report
    assert f"fitted to {n_fitted} aggregate {unit}" in report


def test_abandonment_tolerance_and_worst_error_are_in_the_limitations(
    artifacts: dict[str, Any]
) -> None:
    """The loosest gate in the table must be stated where limitations are read."""
    limitations = " ".join(_bullets(build_report(artifacts), "Limitations"))
    assert "relative tolerance of 0.50" in limitations
    assert "0.20, 0.25" in limitations  # the wait rows, for comparison
    assert "**0.3116**" in limitations
    assert "abandonment rate ESI 4" in limitations


# ------------------------------------------- (c) deliberate skips are not faults


def test_skipped_by_design_artifacts_are_not_listed_as_missing(
    artifacts: dict[str, Any]
) -> None:
    """An upload job never runs the sensitivity analysis; that is not a problem.

    The job path writes the stage manifest before the report stage, so the report
    can name the reason instead of listing the table beside a genuine fault.
    """
    save_run_stages(
        artifacts,
        run_type="uploaded-file analysis",
        performed=[
            "load", "validate", "train_wait", "fairness", "forecast", "calibrate_sim",
            "report",
        ],
        not_performed={
            "sensitivity": "the sensitivity analysis is offline research work",
            "staffing_search": "the simulation did not reproduce your department",
        },
    )
    report = build_report(artifacts)

    by_design, problems = _missing_lists(report)
    assert "tables/sensitivity_ranking.csv" in by_design
    assert "metrics/staffing_search.json" in by_design
    assert "metrics/pipeline_run.json" in by_design
    assert not {"tables/sensitivity_ranking.csv", "metrics/staffing_search.json"} & problems
    assert "offline research work" in report
    assert "did not reproduce your department" in report
    assert "by design" in report

    # A stage that did run and wrote nothing stays a problem.
    assert "tables/forecast_accuracy.csv" in problems


def test_missing_artifacts_are_problems_when_their_stage_ran(
    artifacts: dict[str, Any]
) -> None:
    """Nothing is excused merely by being absent."""
    save_run_stages(
        artifacts,
        run_type="pipeline run",
        performed=["load", "validate", "sensitivity", "calibrate_sim", "report"],
        not_performed={},
    )
    _, problems = _missing_lists(build_report(artifacts))
    assert "tables/sensitivity_ranking.csv" in problems
    assert "metrics/validation.json" in problems


def test_without_a_manifest_no_absence_is_called_deliberate(
    artifacts: dict[str, Any]
) -> None:
    """A report built from a bare directory must not invent a reason for an absence."""
    report = build_report(artifacts)
    by_design, problems = _missing_lists(report)
    assert by_design == {"metrics/pipeline_run.json"}  # structural, and the only one
    assert "tables/sensitivity_ranking.csv" in problems


def _missing_lists(report: str) -> tuple[set[str], set[str]]:
    """The two halves of the "Artifacts not available" section, as path sets."""
    if "## Artifacts not available" not in report:
        return set(), set()
    body = report.split("## Artifacts not available\n", 1)[1].split("\n## ", 1)[0]
    halves = body.split("**Missing unexpectedly.**")
    def paths(text: str) -> set[str]:
        return set(re.findall(r"^- `([^`]+)`", text, re.M))
    return paths(halves[0]), paths(halves[1]) if len(halves) > 1 else set()


# ------------------------------------------------------------------ other wording


def test_the_best_model_is_named_readably(artifacts: dict[str, Any]) -> None:
    """`base+queue / raw / raw` names three config keys, not a model."""
    report = build_report(artifacts)
    assert "base+queue features, raw target" in report
    assert "base+queue / raw / raw" not in report


def test_conformal_leads_with_rolling_and_states_the_static_gap(
    artifacts: dict[str, Any]
) -> None:
    """Rolling first, static as the comparison, and no per-level guarantee claimed."""
    report = build_report(artifacts)
    section = report.split("### Conformal prediction intervals", 1)[1]
    rolling_lead, static_mention = (
        section.index("Rolling recalibration is the result to read"),
        section.index("Static Mondrian calibration is kept as the comparison"),
    )
    assert rolling_lead < static_mention
    assert "the guarantee holds per level" not in report
    assert "worst per-level gap is **-0.0671** at ESI 4" in report


def test_forecast_pairs_the_peak_with_the_overall_improvement(
    artifacts: dict[str, Any]
) -> None:
    """A model that loses at the peak must say so next to the headline number."""
    report = build_report(artifacts)
    paragraph = report.split("## Arrival forecasting\n\n", 1)[1].split("\n\n", 1)[0]
    assert "20.93% better on MAE overall" in paragraph
    assert "peak MAE **4.364**" in paragraph
    assert "baseline's 4.100" in paragraph
    assert "6.43% worse" in paragraph


def test_forecast_peak_verdict_follows_the_artifact(artifacts: dict[str, Any]) -> None:
    """When the model wins at the peak, the same sentence says that instead."""
    forecast = copy.deepcopy(FORECAST)
    forecast["peak_hours"]["model"]["mae"] = 3.2800
    (Path(artifacts["paths"]["metrics_dir"]) / "forecast.json").write_text(
        json.dumps(forecast), encoding="utf-8"
    )
    paragraph = build_report(artifacts).split("## Arrival forecasting\n\n", 1)[1]
    assert "and it holds at the busiest hours" in paragraph
    assert "20.00% better**." in paragraph or "20.00% better" in paragraph
