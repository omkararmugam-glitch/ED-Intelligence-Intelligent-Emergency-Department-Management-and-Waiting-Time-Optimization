"""Tests for the discrete-event simulation, calibration, scenarios and staffing search.

The five required checks:
  * more servers reduce mean wait;
  * doubling arrivals increases it;
  * a fixed seed gives identical output;
  * the DES driven by the generator's own parameters reproduces the generator's KPIs;
  * the staffing search never returns an infeasible plan as "best".
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.data.schema import validate_visits
from src.data.synthetic import generate_synthetic_visits
from src.simulation.calibrate import (
    CalibrationResult,
    ValidationError,
    calibrate_and_validate,
    observed_kpis,
    require_valid_calibration,
)
from src.simulation.des import (
    ArrivalModel,
    ServiceTimeModel,
    StaffingPlan,
    run_replications,
    run_simulation,
)
from src.simulation.scenarios import (
    ScenarioSpec,
    load_scenarios,
    paired_comparison,
    run_all_scenarios,
    summarise_scenarios,
)
from src.simulation.sensitivity import (
    Perturbation,
    assess_ranking_stability,
    run_sensitivity,
)
from src.simulation.staffing_search import (
    CostModel,
    StaffingTarget,
    candidate_plans,
    search_staffing,
)
from src.utils.priority import AgingPolicy

SIM_HORIZON_HOURS = 24 * 10


@pytest.fixture(scope="module")
def sim_config() -> dict:
    """Config sized so simulation tests stay quick."""
    from src.utils.io import load_config

    cfg = load_config("configs/default.yaml")
    cfg["simulation"]["horizon_hours"] = SIM_HORIZON_HOURS
    cfg["simulation"]["n_replications"] = 8
    cfg["simulation"]["calibration"]["replications"] = 4
    # Bound the nested calibration search: these tests check that it converges and
    # gates correctly, not that it finds the last decimal place.
    cfg["simulation"]["calibration"]["search_steps"] = 6
    cfg["simulation"]["calibration"]["search_rounds"] = 4
    cfg["staffing_search"]["n_replications"] = 3
    cfg["synthetic"]["n_patients"] = 12000
    return cfg


# ------------------------------------------------------------------ shared policy


def test_generator_and_simulation_share_one_aging_policy(sim_config) -> None:
    """The rule must live in one place, or the model drifts from the world."""
    from src.data import synthetic as generator_module
    from src.simulation import des as simulation_module

    assert generator_module.AgingPolicy is AgingPolicy
    assert simulation_module.AgingPolicy is AgingPolicy

    policy = AgingPolicy.from_config(sim_config)
    assert policy.minutes_per_level == sim_config["synthetic"]["priority_aging_minutes_per_level"]
    assert policy.floor == sim_config["synthetic"]["priority_aging_floor"]


def test_aging_floor_protects_the_most_urgent(sim_config) -> None:
    """No amount of waiting lets a low-acuity patient outrank a fresh ESI 1 or 2."""
    policy = AgingPolicy.from_config(sim_config)

    assert policy.effective_priority(5, waited_minutes=10_000) >= policy.floor
    assert policy.effective_priority(5, waited_minutes=10_000) > policy.effective_priority(2, 0)
    assert policy.effective_priority(1, waited_minutes=0) == 1.0
    # Patients already at or above the floor's urgency do not age at all.
    assert policy.effective_priority(2, waited_minutes=500) == 2.0


def test_aging_policy_rejects_bad_parameters() -> None:
    """A non-positive rate or floor is a config error."""
    with pytest.raises(ValueError, match="minutes_per_level must be positive"):
        AgingPolicy(minutes_per_level=0.0, floor=2.5)
    with pytest.raises(ValueError, match="floor must be positive"):
        AgingPolicy(minutes_per_level=25.0, floor=0.0)


# ----------------------------------------------------------------- DES behaviour


def test_more_doctors_reduce_the_mean_wait(sim_config) -> None:
    """The first required sanity check."""
    baseline = StaffingPlan.from_config(sim_config)
    richer = baseline.with_doctors({name: spec["doctors"] + 2 for name, spec in baseline.shifts.items()})

    lean = run_simulation(sim_config, plan=baseline, seed_label="servers").kpis
    generous = run_simulation(sim_config, plan=richer, seed_label="servers").kpis

    assert generous["mean_wait_minutes"] < lean["mean_wait_minutes"]
    assert generous["p90_wait_minutes"] < lean["p90_wait_minutes"]
    assert generous["doctor_utilisation"] < lean["doctor_utilisation"]


def test_doubling_arrivals_increases_the_wait(sim_config) -> None:
    """The second required sanity check."""
    normal = ArrivalModel.from_config(sim_config, SIM_HORIZON_HOURS, multiplier=1.0)
    doubled = ArrivalModel.from_config(sim_config, SIM_HORIZON_HOURS, multiplier=2.0)

    quiet = run_simulation(sim_config, arrivals=normal, seed_label="load").kpis
    busy = run_simulation(sim_config, arrivals=doubled, seed_label="load").kpis

    assert busy["n_arrivals"] > 1.5 * quiet["n_arrivals"]
    assert busy["mean_wait_minutes"] > quiet["mean_wait_minutes"]
    assert busy["abandonment_rate"] >= quiet["abandonment_rate"]


def test_a_fixed_seed_gives_identical_output(sim_config) -> None:
    """The third required sanity check: runs must be exactly reproducible."""
    first = run_simulation(sim_config, seed_label="reproducible")
    second = run_simulation(sim_config, seed_label="reproducible")

    assert first.kpis == second.kpis
    pd.testing.assert_frame_equal(first.patients, second.patients, check_exact=True)


def test_different_seeds_give_different_runs(sim_config) -> None:
    """Replications must actually differ, or the confidence intervals are fiction."""
    first = run_simulation(sim_config, seed_label="replication_a").kpis
    second = run_simulation(sim_config, seed_label="replication_b").kpis
    assert first["mean_wait_minutes"] != second["mean_wait_minutes"]


def test_acuity_ordering_holds_in_the_simulation(sim_config) -> None:
    """The priority queue must be visible in simulated waits, as it is in the data."""
    kpis = run_simulation(sim_config, seed_label="ordering").kpis
    assert kpis["mean_wait_esi1"] < kpis["mean_wait_esi3"]
    assert kpis["mean_wait_esi2"] < kpis["mean_wait_esi5"]


def test_utilisation_is_a_proportion(sim_config) -> None:
    """Utilisation above 1 means the window accounting is broken."""
    for label in ("util_a", "util_b", "util_c"):
        kpis = run_simulation(sim_config, seed_label=label).kpis
        assert 0.0 <= kpis["doctor_utilisation"] <= 1.0


def test_boarding_lengthens_los_but_not_the_wait(sim_config) -> None:
    """Boarding holds a bed, not a clinician, so it must not delay anyone else."""
    service = ServiceTimeModel.from_config(sim_config)
    no_boarding = ServiceTimeModel(
        service_median=dict(service.service_median),
        service_sigma=dict(service.service_sigma),
        triage_delay_median=dict(service.triage_delay_median),
        triage_delay_sigma=dict(service.triage_delay_sigma),
        patience_median=dict(service.patience_median),
        patience_sigma=service.patience_sigma,
        boarding_median={"admitted": 0.01, "other": 0.01},
        boarding_sigma=dict(service.boarding_sigma),
        admit_probability=dict(service.admit_probability),
    )

    with_boarding = run_simulation(sim_config, service=service, seed_label="board").kpis
    without = run_simulation(sim_config, service=no_boarding, seed_label="board").kpis

    assert with_boarding["mean_los_minutes"] > without["mean_los_minutes"]
    assert with_boarding["mean_wait_minutes"] == pytest.approx(
        without["mean_wait_minutes"], rel=0.02
    )


def test_fast_track_is_an_additional_route_not_a_restriction(sim_config) -> None:
    """Fast-track patients must stay eligible for the main pool.

    Confining ESI 4-5 to one dedicated doctor starves them, because they are roughly
    46% of arrivals. Adding the lane must help them, not hurt them.
    """
    baseline = StaffingPlan.from_config(sim_config)
    with_lane = baseline.with_fast_track(doctors=1, levels=(4, 5))

    without = run_simulation(sim_config, plan=baseline, seed_label="fast").kpis
    lane = run_simulation(sim_config, plan=with_lane, seed_label="fast").kpis

    assert lane["mean_wait_esi4"] < without["mean_wait_esi4"]
    assert lane["mean_wait_esi5"] < without["mean_wait_esi5"]
    assert lane["mean_wait_minutes"] < without["mean_wait_minutes"]


def test_staffing_plan_rejects_uncovered_hours(sim_config) -> None:
    """A shift table with a gap is a config bug."""
    broken = StaffingPlan(shifts={"day": {"start": 8, "end": 16, "doctors": 5}})
    with pytest.raises(ValueError, match="leaves hours uncovered"):
        broken.capacity_by_hour()


def test_arrival_model_can_be_driven_by_a_forecast(sim_config) -> None:
    """The G1 link: forecast output becomes the simulation's load."""
    forecast_frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-06-01", periods=48, freq="h"),
            "yhat": np.linspace(2.0, 9.0, 48),
        }
    )
    arrivals = ArrivalModel.from_forecast(forecast_frame, sim_config)

    assert arrivals.horizon_hours == 48
    assert np.allclose(arrivals.hourly_rate, forecast_frame["yhat"].to_numpy())

    result = run_simulation(sim_config, arrivals=arrivals, seed_label="fc", horizon_hours=48)
    assert result.kpis["n_arrivals"] > 0


def test_forecast_arrival_model_requires_yhat(sim_config) -> None:
    """A frame without the forecast column is a programming error."""
    with pytest.raises(KeyError, match="yhat"):
        ArrivalModel.from_forecast(pd.DataFrame({"timestamp": []}), sim_config)


# ------------------------------------------------------------------- calibration


@pytest.fixture(scope="module")
def population(sim_config) -> pd.DataFrame:
    """Validated visits used as the ground truth for calibration."""
    return validate_visits(generate_synthetic_visits(sim_config)).all


@pytest.fixture(scope="module")
def calibration(population, sim_config) -> CalibrationResult:
    """A calibrated, validated simulation."""
    return calibrate_and_validate(population, sim_config)


def test_calibrated_simulation_reproduces_the_generator(calibration, population) -> None:
    """The fourth required check: the DES must reproduce the department it models."""
    assert calibration.passed, f"validation failed: {calibration.failures}"

    table = calibration.validation
    held_out = table[~table["used_for_fitting"]]
    assert len(held_out) >= 7, "most metrics must be held out of the fit"
    assert held_out["within_tolerance"].all()

    # And the headline numbers really are close, not merely inside a loose tolerance.
    observed = observed_kpis(population)
    assert calibration.simulated["p90_wait_minutes"] == pytest.approx(
        observed["p90_wait_minutes"], rel=0.25
    )


def test_calibration_records_its_identification_assumption(calibration) -> None:
    """The unobservable treatment/boarding split must be stated, not buried."""
    assumptions = " ".join(calibration.assumptions).lower()
    assert "treatment" in assumptions and "boarding" in assumptions
    assert "not identifiable" in assumptions or "assumption" in assumptions
    # The metric used for fitting must be marked so it cannot be read as validation.
    # Two parameters are fitted -- the clinician-time fraction against mean wait,
    # and the patience scale against the overall abandonment rate -- so exactly two
    # metrics are consumed by the fit and cannot also validate it.
    assert calibration.validation["used_for_fitting"].sum() == 2
    fitted = set(
        calibration.validation.loc[calibration.validation["used_for_fitting"], "key"]
    )
    assert fitted == {"mean_wait_minutes", "abandonment_rate"}


def test_failed_validation_blocks_downstream_analysis(calibration) -> None:
    """Scenarios must refuse to run on a simulation that does not reproduce reality."""
    broken = CalibrationResult(
        arrivals=calibration.arrivals,
        service=calibration.service,
        plan=calibration.plan,
        validation=calibration.validation,
        passed=False,
        failures=["mean wait: simulated 200 vs observed 40"],
    )

    with pytest.raises(ValidationError, match="does not reproduce the observed department"):
        require_valid_calibration(broken)

    # ...unless explicitly overridden.
    require_valid_calibration(broken, allow_override=True)


def test_scenarios_refuse_an_invalid_calibration(calibration, sim_config) -> None:
    """The gate is enforced at the scenario entry point, not just in the helper."""
    broken = CalibrationResult(
        arrivals=calibration.arrivals,
        service=calibration.service,
        plan=calibration.plan,
        validation=calibration.validation,
        passed=False,
        failures=["p90 wait out of tolerance"],
    )
    with pytest.raises(ValidationError):
        run_all_scenarios(sim_config, broken, n_replications=2)


# --------------------------------------------------------------------- scenarios


@pytest.fixture(scope="module")
def scenario_run(sim_config, calibration):
    """Scenario replications, summary and paired comparison."""
    return run_all_scenarios(sim_config, calibration, n_replications=8)


def test_every_configured_scenario_runs(scenario_run, sim_config) -> None:
    """All six named scenarios must appear in the summary."""
    _, summary, _ = scenario_run
    expected = {spec.name for spec in load_scenarios(sim_config)}
    assert set(summary["scenario"]) == expected
    assert (summary["n_replications"] == 8).all()


def test_common_random_numbers_pair_the_staffing_scenarios(scenario_run) -> None:
    """A staffing change must not alter the arrival stream, so pairing is valid."""
    replications, _, comparison = scenario_run

    staffing = comparison[comparison["scenario"].str.startswith("plus_one")]
    assert (staffing["method"] == "paired (common random numbers)").all()

    # Same demand means the same number of arrivals in matched replications.
    baseline = replications[replications["scenario"] == "baseline"].set_index("replication")
    plus = replications[replications["scenario"] == "plus_one_doctor_peak"].set_index("replication")
    assert (baseline["n_arrivals"] == plus["n_arrivals"]).all()


def test_demand_scenarios_are_reported_as_unpaired(scenario_run) -> None:
    """A surge changes the arrival draws, so the comparison cannot claim pairing."""
    replications, _, comparison = scenario_run

    surge = comparison[comparison["scenario"].str.startswith("surge")]
    assert (surge["method"] == "unpaired (scenario changes demand)").all()

    baseline = replications[replications["scenario"] == "baseline"]
    surged = replications[replications["scenario"] == "surge_20pct"]
    assert surged["n_arrivals"].mean() > baseline["n_arrivals"].mean()


def test_pairing_tightens_the_confidence_interval(sim_config, calibration) -> None:
    """The point of common random numbers: the paired interval must be narrower.

    The same scenario is compared to the baseline twice -- once sharing the random
    stream, once not -- and the paired interval should be clearly tighter.
    """
    spec = ScenarioSpec(name="plus_one_doctor_peak", doctors={"day": 9})
    from src.simulation.scenarios import run_scenario

    paired_frames = [
        run_scenario(ScenarioSpec(name="baseline"), sim_config, calibration, 8, seed_prefix="shared"),
        run_scenario(spec, sim_config, calibration, 8, seed_prefix="shared"),
    ]
    paired = paired_comparison(pd.concat(paired_frames, ignore_index=True), [ScenarioSpec("baseline"), spec])
    width_paired = float(paired["ci_high"].iloc[0] - paired["ci_low"].iloc[0])

    independent = [
        run_scenario(ScenarioSpec(name="baseline"), sim_config, calibration, 8, seed_prefix="stream_a"),
        run_scenario(spec, sim_config, calibration, 8, seed_prefix="stream_b"),
    ]
    frame = pd.concat(independent, ignore_index=True)
    # Treat it as unpaired by declaring a demand change.
    unpaired_spec = ScenarioSpec(name="plus_one_doctor_peak", doctors={"day": 9}, arrival_multiplier=1.0001)
    frame["scenario"] = frame["scenario"].replace({"plus_one_doctor_peak": unpaired_spec.name})
    unpaired = paired_comparison(frame, [ScenarioSpec("baseline"), unpaired_spec])
    width_unpaired = float(unpaired["ci_high"].iloc[0] - unpaired["ci_low"].iloc[0])

    assert width_paired < width_unpaired, (
        f"pairing did not tighten the interval: {width_paired:.3f} vs {width_unpaired:.3f}"
    )


def test_surges_make_the_department_worse(scenario_run) -> None:
    """A demand increase must raise waits and abandonment."""
    _, summary, _ = scenario_run
    indexed = summary.set_index("scenario")

    assert indexed.loc["surge_20pct", "mean_wait_minutes"] > indexed.loc["surge_10pct", "mean_wait_minutes"]
    assert indexed.loc["surge_10pct", "mean_wait_minutes"] > indexed.loc["baseline", "mean_wait_minutes"]
    assert indexed.loc["surge_20pct", "abandonment_rate"] > indexed.loc["baseline", "abandonment_rate"]


def test_extra_capacity_helps(scenario_run) -> None:
    """Both capacity scenarios must beat the baseline, with intervals excluding zero."""
    _, _, comparison = scenario_run
    helpers = comparison[comparison["scenario"].isin({"plus_one_doctor_peak", "fast_track_esi45"})]

    assert (helpers["difference_vs_baseline"] < 0).all()
    assert helpers["significant"].all()


def test_summary_carries_confidence_intervals(scenario_run) -> None:
    """Every reported KPI needs an interval, or the comparison is not interpretable."""
    _, summary, _ = scenario_run
    for kpi in ("mean_wait_minutes", "p90_wait_esi3", "abandonment_rate"):
        assert f"{kpi}_ci_low" in summary.columns
        assert (summary[f"{kpi}_ci_low"] <= summary[kpi]).all()
        assert (summary[kpi] <= summary[f"{kpi}_ci_high"]).all()


# ------------------------------------------------------------------- sensitivity


def test_sensitivity_reruns_every_world(sim_config, calibration) -> None:
    """Each perturbation must produce a full set of scenario results."""
    specs = [ScenarioSpec("baseline"), ScenarioSpec("plus_one_doctor_peak", doctors={"day": 9})]
    worlds = [Perturbation("point_estimate"), Perturbation("service_x1.2", service_factor=1.2)]

    results, stability = run_sensitivity(
        sim_config, calibration, specs=specs, perturbations=worlds, n_replications=4
    )
    assert set(results["perturbation"]) == {"point_estimate", "service_x1.2"}
    assert set(results["scenario"]) == {"baseline", "plus_one_doctor_peak"}
    assert set(stability["scenario"]) == {"baseline", "plus_one_doctor_peak"}


def test_longer_service_times_make_things_worse(sim_config, calibration) -> None:
    """A perturbation must actually change the world it claims to."""
    specs = [ScenarioSpec("baseline")]
    worlds = [Perturbation("point_estimate"), Perturbation("service_x1.2", service_factor=1.2)]
    results, _ = run_sensitivity(
        sim_config, calibration, specs=specs, perturbations=worlds, n_replications=4
    )
    means = results.groupby("perturbation")["mean_wait_minutes"].mean()
    assert means["service_x1.2"] > means["point_estimate"]


def test_ranking_stability_detects_a_reordering() -> None:
    """The stability summary must flag a scenario whose rank moves."""
    frame = pd.DataFrame(
        {
            "perturbation": ["point_estimate"] * 2 + ["service_x1.2"] * 2,
            "scenario": ["a", "b", "a", "b"],
            "mean_wait_minutes": [10.0, 20.0, 25.0, 15.0],
        }
    )
    stability = assess_ranking_stability(frame)
    assert not stability["rank_is_stable"].all()
    assert set(stability["max_rank_shift"]) == {1}


# --------------------------------------------------------------- staffing search


def test_search_never_returns_an_infeasible_plan_as_best(sim_config, calibration) -> None:
    """The fifth required check, asserted both ways."""
    result = search_staffing(sim_config, calibration, n_replications=3)

    if result.any_feasible:
        assert result.best is not None
        assert result.best["feasible"] is True
        assert result.best["meets_wait_target"] and result.best["meets_abandonment_target"]
        # And it really is the cheapest among the feasible ones.
        feasible = result.plans[result.plans["feasible"]]
        assert result.best["cost"] == pytest.approx(feasible["cost"].min())
    else:
        assert result.best is None


def test_an_impossible_target_yields_no_plan(sim_config, calibration) -> None:
    """With an unreachable target the search must say so, not return the least-bad plan."""
    impossible = copy.deepcopy(sim_config)
    impossible["staffing_search"]["target"]["p90_wait_esi3_minutes"] = 0.001
    impossible["staffing_search"]["target"]["abandonment_rate"] = 0.0

    result = search_staffing(impossible, calibration, n_replications=3)

    assert not result.any_feasible
    assert result.best is None
    assert result.feasible_count == 0
    # The closest misses are reported instead, so the user can see how far off they are.
    assert not result.near_misses.empty
    assert "worst_shortfall" in result.near_misses.columns


def test_feasibility_is_judged_on_the_upper_confidence_bound(sim_config, calibration) -> None:
    """A plan must be shown to meet the target, not merely estimated to."""
    result = search_staffing(sim_config, calibration, n_replications=3)
    target = StaffingTarget.from_config(sim_config)

    feasible = result.plans[result.plans["feasible"]]
    assert (feasible["p90_wait_esi3_ci_high"] <= target.p90_wait_esi3_max).all()
    assert (feasible["abandonment_rate_ci_high"] <= target.abandonment_rate_max).all()


def test_more_doctors_cost_more(sim_config) -> None:
    """The cost model must be monotone in staffing."""
    cost_model = CostModel.from_config(sim_config)
    baseline = StaffingPlan.from_config(sim_config)
    richer = baseline.with_doctors({name: spec["doctors"] + 1 for name, spec in baseline.shifts.items()})

    assert cost_model.plan_cost(richer) > cost_model.plan_cost(baseline)


def test_candidate_plans_cover_the_configured_grid(sim_config) -> None:
    """The grid strategy must enumerate every combination."""
    baseline = StaffingPlan.from_config(sim_config)
    plans = candidate_plans(sim_config, baseline)

    grid = sim_config["staffing_search"]["doctor_grid"]
    expected = int(np.prod([len(values) for values in grid.values()]))
    assert len(plans) == expected

    descriptions = {plan.describe() for plan in plans}
    assert len(descriptions) == expected, "grid produced duplicate plans"


def test_random_strategy_samples_the_grid(sim_config) -> None:
    """Random search must return the configured sample size, reproducibly."""
    random_config = copy.deepcopy(sim_config)
    random_config["staffing_search"]["strategy"] = "random"
    random_config["staffing_search"]["n_random_samples"] = 5

    baseline = StaffingPlan.from_config(random_config)
    first = [plan.describe() for plan in candidate_plans(random_config, baseline)]
    second = [plan.describe() for plan in candidate_plans(random_config, baseline)]

    assert len(first) == 5
    assert first == second


def test_unknown_strategy_raises(sim_config) -> None:
    """A typo must not silently fall back to a different search."""
    broken = copy.deepcopy(sim_config)
    broken["staffing_search"]["strategy"] = "bayesian"
    with pytest.raises(ValueError, match="Unknown staffing_search.strategy"):
        candidate_plans(broken, StaffingPlan.from_config(broken))


def test_search_uses_forecast_arrivals_when_given(sim_config, calibration) -> None:
    """The G1 link must be visible in the output, not merely available."""
    forecast_frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-06-01", periods=72, freq="h"),
            "yhat": np.tile(np.linspace(2.0, 9.0, 24), 3),
        }
    )
    result = search_staffing(
        sim_config, calibration, forecast_frame=forecast_frame, n_replications=2
    )
    assert (result.plans["load_source"].str.startswith("forecast")).all()


def test_calibration_persists_its_fitted_parameters(calibration) -> None:
    """The two fitted numbers must reach the artifact.

    The clinician-time fraction and the patience scale are what the whole simulation
    rests on. If they are not written down, a run cannot be reproduced or audited and a
    report has nowhere to read them from.
    """
    payload = calibration.to_dict()
    parameters = payload["fitted_parameters"]

    assert 0.0 < parameters["clinician_time_fraction"] <= 1.0
    assert parameters["patience_scale"] > 0.0
    assert parameters == calibration.fitted_parameters

    # The derived distributions are recorded too, so the fit is fully inspectable.
    assert set(payload["fitted_service_medians"]) == {"1", "2", "3", "4", "5"}
    assert set(payload["fitted_patience_medians"]) == {"1", "2", "3", "4", "5"}
    assert payload["roster_doctor_hours_per_day"] > 0

    import json

    json.dumps(payload)  # must stay serialisable
