"""The four headline numbers, pinned.

These are the figures the README and `artifacts/report.md` quote for the default
pipeline run on `data/raw/ed_v2.csv`:

================================================  ==========
wait MAE, best model vs rolling-mean baseline     11.52 / 34.23 min
rolling conformal coverage, overall               0.8875
forecast MAE, overall                             2.356 arrivals/hour
cheapest feasible staffing plan                   49,280 (72 of 144 plans feasible)
================================================  ==========

Every experiment this repository runs writes to its own run folder, so none of them
may move these. This test reads the **baseline** `artifacts/` tree and fails if any of
them has shifted, which catches the two ways that happens by accident: an experiment
that forgot `--artifacts-dir`/`--run-name` and overwrote the baseline, and a change to
a default in `configs/default.yaml` or in the code that silently moves the result.

**The explicit flag.** When a change to one of these numbers is intended, set
``ED_ALLOW_HEADLINE_CHANGE=1`` and this test skips. Re-run the default pipeline, check
the new numbers are what you meant, and update ``EXPECTED`` below in the same commit.
The flag is deliberately an environment variable rather than a config key: it has to be
impossible to set it by accident while running the pipeline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from src.utils.io import load_json, resolve_path

#: Environment variable that turns this guard off for one deliberate change.
OVERRIDE_ENV = "ED_ALLOW_HEADLINE_CHANGE"

#: Baseline artifacts. Experiments must never write here.
ARTIFACTS = resolve_path("artifacts")

#: metric name -> (expected value, decimal places compared, where it is quoted)
#:
#: The decimal places are the precision the report quotes, so this fails on a real
#: change and not on a last-bit floating-point difference between machines.
EXPECTED: dict[str, tuple[float, int, str]] = {
    "wait_mae_model": (11.524, 3, "report.md: 'test MAE 11.52 min'"),
    "wait_mae_baseline": (34.233, 3, "report.md: 'Baseline ... MAE 34.23 min'"),
    "rolling_conformal_coverage": (0.8875, 4, "report.md: 'Empirical coverage 0.8875'"),
    "forecast_mae": (2.356, 3, "report.md forecast accuracy table, overall lightgbm"),
    "staffing_cheapest_cost": (49280.0, 2, "report.md: 'at cost 49,280'"),
    "staffing_n_feasible": (72.0, 0, "report.md: '72 of 144 plans were feasible'"),
    "staffing_n_plans": (144.0, 0, "report.md: '72 of 144 plans were feasible'"),
}


def _require(path: Path) -> dict[str, Any]:
    """Load a baseline metrics file, failing with a message that says what to do."""
    if not path.is_file():
        pytest.fail(
            f"{path} is missing, so the headline numbers cannot be checked. The "
            "baseline artifacts come from:\n"
            "    python -m src.pipeline run --data real\n"
            "Run that, or set "
            f"{OVERRIDE_ENV}=1 if you are deliberately rebuilding them."
        )
    return load_json(path)


def _rolling_overall_coverage(wait_models: dict[str, Any]) -> float:
    """Pull the overall rolling-conformal coverage out of the comparison table."""
    comparison = wait_models["conformal"]["rolling"]["comparison"]
    matches = [
        row
        for row in comparison
        if row.get("scope") == "overall" and row.get("method") == "rolling"
    ]
    if len(matches) != 1:
        pytest.fail(
            "expected exactly one overall/rolling row in "
            f"conformal.rolling.comparison, found {len(matches)}"
        )
    return float(matches[0]["empirical_coverage"])


def observed_headline_numbers() -> dict[str, float]:
    """Read all seven pinned figures out of the baseline artifacts.

    Returns:
        Mapping with the same keys as `EXPECTED`.
    """
    wait_models = _require(ARTIFACTS / "metrics" / "wait_models.json")
    forecast = _require(ARTIFACTS / "metrics" / "forecast.json")
    staffing = _require(ARTIFACTS / "metrics" / "staffing_search.json")

    return {
        "wait_mae_model": float(wait_models["ablation"]["best_by_test_mae"]["mae_minutes"]),
        "wait_mae_baseline": float(
            wait_models["baseline"]["blocks"]["test"]["regression"]["mae"]
        ),
        "rolling_conformal_coverage": _rolling_overall_coverage(wait_models),
        "forecast_mae": float(forecast["overall"]["model"]["mae"]),
        "staffing_cheapest_cost": float(staffing["cheapest_feasible_plan"]["cost"]),
        "staffing_n_feasible": float(staffing["n_feasible"]),
        "staffing_n_plans": float(staffing["n_plans_evaluated"]),
    }


@pytest.fixture(scope="module")
def observed() -> dict[str, float]:
    """The headline numbers as the baseline artifacts currently report them."""
    if os.environ.get(OVERRIDE_ENV):
        pytest.skip(
            f"{OVERRIDE_ENV} is set: the headline numbers are allowed to move. "
            "Update EXPECTED in this file to the new values."
        )
    if not ARTIFACTS.is_dir():
        pytest.skip(
            f"{ARTIFACTS} does not exist, so there is no baseline run to check. "
            "Run `python -m src.pipeline run --data real` first."
        )
    return observed_headline_numbers()


@pytest.mark.parametrize("metric", sorted(EXPECTED))
def test_headline_number_has_not_moved(metric: str, observed: dict[str, float]) -> None:
    """Each pinned figure still matches, to the precision the report quotes it at."""
    expected, places, quoted_in = EXPECTED[metric]
    actual = observed[metric]

    assert round(actual, places) == round(expected, places), (
        f"{metric} moved: expected {expected} ({quoted_in}), baseline artifacts now "
        f"say {actual}.\n"
        "Nothing in this repository is supposed to change the default run's results. "
        "Either an experiment wrote into artifacts/ instead of its own run folder "
        "(pass --artifacts-dir or --run-name), or a default changed. If the change is "
        f"intended, set {OVERRIDE_ENV}=1, re-run the default pipeline, and update "
        "EXPECTED in this file."
    )


def test_cheapest_plan_is_still_the_same_roster(observed: dict[str, float]) -> None:
    """The 49,280 plan is a specific roster, not just a cost; pin the roster too."""
    staffing = _require(ARTIFACTS / "metrics" / "staffing_search.json")
    plan = staffing["cheapest_feasible_plan"]

    assert plan["plan"] == "day=3, evening=3, night_early=2, night_late=2", (
        f"the cheapest feasible plan changed to {plan['plan']!r}. See "
        f"{OVERRIDE_ENV} in this module's docstring if that is intended."
    )
    assert plan["feasible"] is True, "the cheapest 'feasible' plan is not marked feasible"


def test_expected_table_covers_every_observed_number() -> None:
    """Guard against a key being added to one side of the comparison only."""
    # Does not read artifacts, so it runs even on a fresh clone.
    assert set(EXPECTED) == {
        "wait_mae_model",
        "wait_mae_baseline",
        "rolling_conformal_coverage",
        "forecast_mae",
        "staffing_cheapest_cost",
        "staffing_n_feasible",
        "staffing_n_plans",
    }, "EXPECTED changed shape; update observed_headline_numbers() to match"


def test_the_override_flag_is_not_set_in_this_environment() -> None:
    """A set override must be visible, not quietly disabling the guard forever.

    This is the one test in the module that does not honour the flag: if the flag is
    left set in a shell or a CI job, every other check above skips, and a skipped
    guard protects nothing. Failing here makes that state impossible to miss.
    """
    assert not os.environ.get(OVERRIDE_ENV), (
        f"{OVERRIDE_ENV} is set, so the headline-number checks are skipping. That is "
        "fine for one deliberate update, but it must not be left set: unset it once "
        "EXPECTED has been updated."
    )
