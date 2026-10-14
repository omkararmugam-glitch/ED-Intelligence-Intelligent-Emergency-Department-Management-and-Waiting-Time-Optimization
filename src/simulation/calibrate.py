"""Fit the simulation's inputs from data, then check that it reproduces reality.

Calibration without validation is wishful thinking, so this module does both and the
second part is a **gate**: `scenarios.py` refuses to run on a simulation that failed
validation unless an explicit override is passed. A staffing recommendation produced by
a model that cannot reproduce today's department is worse than no recommendation.

**One identification problem, stated rather than hidden.** The data records arrival,
first clinician contact and departure. It does not record when *treatment* ended, so
`departure - seen` is treatment **plus** boarding. The simulation needs them apart,
because treatment occupies a doctor and boarding does not. The assumption used here:

    treatment duration ~ post-contact time of patients who were NOT admitted
    boarding duration  ~ the excess post-contact time of patients who WERE admitted

That is defensible — discharged patients leave soon after their care finishes, while
admitted patients wait for an inpatient bed — but it is an assumption, not a
measurement, and it is recorded in the calibration report. A site with real
treatment-end timestamps should use them instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.simulation.des import (
    ArrivalModel,
    ServiceTimeModel,
    StaffingPlan,
    run_replications,
)
from src.utils.io import save_table
from src.utils.logging import get_logger
from src.utils.metrics import quantile

logger = get_logger(__name__)

#: Identification assumptions recorded with every calibration report.
TREATMENT_IDENTIFICATION = (
    "The data records arrival, first clinician contact and departure, but never when "
    "treatment ended, so `departure - seen` is clinician time PLUS everything else the "
    "visit contains (tests, results, boarding). The split is not identifiable from "
    "these columns. It is estimated as a single free parameter -- the fraction of "
    "post-contact time during which a doctor is occupied -- chosen so that simulated "
    "mean wait matches observed. A multiplicative fraction rather than a constant "
    "offset, because with this roster the department could not function if a doctor "
    "were occupied for the whole visit: all post-contact time would require 4.95x the "
    "rostered capacity. The split preserves treatment + boarding = post_contact "
    "exactly, so length of stay remains an independent validation metric."
)

PATIENCE_IDENTIFICATION = (
    "Patience is only observed for patients who ran out of it; everyone who was seen "
    "had patience longer than their wait, which is censoring. The shape is taken from "
    "the observed time-to-abandon per acuity level and the scale is then fitted so the "
    "simulated abandonment rate matches the observed one."
)

FITTED_ON = (
    "Mean wait and the overall abandonment rate were USED to fit the two free "
    "parameters, so their agreement is guaranteed by construction and is not evidence. "
    "Every other row of the validation table -- p90 wait, length of stay, the per-acuity "
    "waits and the per-acuity abandonment rates -- was held out of the fit and is "
    "genuine out-of-sample validation."
)

#: Metrics consumed by the fit, and therefore excluded from the pass/fail gate.
FITTED_METRICS = ("mean_wait_minutes", "abandonment_rate")


@dataclass
class CalibrationResult:
    """Fitted inputs, the validation table, and whether it passed."""

    arrivals: ArrivalModel
    service: ServiceTimeModel
    plan: StaffingPlan
    validation: pd.DataFrame
    passed: bool
    failures: list[str] = field(default_factory=list)
    observed: dict[str, float] = field(default_factory=dict)
    simulated: dict[str, float] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    #: The two fitted parameters. Recorded so a run can be reproduced and audited, and
    #: so a report never has to quote them from anywhere but the artifact.
    fitted_parameters: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "passed": self.passed,
            "failures": list(self.failures),
            "assumptions": list(self.assumptions),
            "fitted_parameters": dict(self.fitted_parameters),
            "fitted_service_medians": {
                str(level): round(float(median), 3)
                for level, median in sorted(self.service.service_median.items())
            },
            "fitted_patience_medians": {
                str(level): round(float(median), 1)
                for level, median in sorted(self.service.patience_median.items())
            },
            "roster_doctor_hours_per_day": round(self.plan.total_doctor_hours(24), 1),
            "roster": self.plan.describe(),
            "observed": self.observed,
            "simulated": self.simulated,
            "validation": self.validation.to_dict(orient="records"),
        }


def _lognormal_fit(values: np.ndarray) -> tuple[float, float]:
    """Fit a lognormal by its log-scale moments, returning (median, sigma)."""
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size < 2:
        return float("nan"), float("nan")
    logs = np.log(positive)
    return float(np.exp(np.mean(logs))), float(np.std(logs, ddof=1))


def fit_arrival_model(
    visits: pd.DataFrame, config: dict[str, Any], horizon_hours: int, multiplier: float = 1.0
) -> ArrivalModel:
    """Estimate the hourly arrival rate from the data's own hour-of-day/weekday pattern.

    Using the empirical profile rather than the configured one is what lets the
    simulation reproduce the department it is supposed to model: the configured profile
    carries no seasonality or surges, so it understates mean load.

    Args:
        visits: Validated visits.
        config: Full pipeline config.
        horizon_hours: Length of the runs this model will drive.
        multiplier: Scales every hour, for surge scenarios.

    Returns:
        An `ArrivalModel` whose rates repeat the observed weekly pattern.
    """
    times = pd.to_datetime(visits["arrival_time"])
    hourly = times.dt.floor("h").value_counts()
    full = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    counts = hourly.reindex(full, fill_value=0)

    frame = pd.DataFrame(
        {"count": counts.to_numpy(), "hour": full.hour, "dow": full.dayofweek}
    )
    profile = frame.groupby(["dow", "hour"])["count"].mean()

    # Repeat the fitted weekly pattern across the run's horizon.
    hours = np.arange(horizon_hours)
    rate = np.array(
        [float(profile.loc[((h // 24) % 7, h % 24)]) for h in hours]
    ) * multiplier

    mix = visits["triage_level"].value_counts(normalize=True)
    probabilities = {int(level): float(share) for level, share in mix.items()}

    logger.info(
        "Fitted arrival model from %d visits: mean %.2f arrivals/hour (observed %.2f)",
        len(visits),
        float(rate.mean() / max(multiplier, 1e-9)),
        float(frame["count"].mean()),
    )
    return ArrivalModel(hourly_rate=rate, triage_probabilities=probabilities)


def fit_service_model(visits: pd.DataFrame, config: dict[str, Any]) -> ServiceTimeModel:
    """Fit treatment, triage-delay, patience and boarding distributions from the data.

    See `TREATMENT_IDENTIFICATION` for the assumption that separates treatment from
    boarding. Any quantity the data cannot support falls back to the configured value,
    and the fallback is logged.
    """
    fallback = ServiceTimeModel.from_config(config)
    seen = visits[visits["seen_time"].notna()].copy()
    seen["post_contact"] = (
        pd.to_datetime(seen["departure_time"]) - pd.to_datetime(seen["seen_time"])
    ).dt.total_seconds() / 60.0
    seen["wait"] = (
        pd.to_datetime(seen["seen_time"]) - pd.to_datetime(seen["arrival_time"])
    ).dt.total_seconds() / 60.0

    not_admitted = seen[seen["disposition"] != "admitted"]
    admitted = seen[seen["disposition"] == "admitted"]

    service_median, service_sigma = {}, {}
    delay_median, delay_sigma = {}, {}
    for level in sorted(visits["triage_level"].unique()):
        level = int(level)
        subset = not_admitted[not_admitted["triage_level"] == level]["post_contact"]
        median, sigma = _lognormal_fit(subset.to_numpy(dtype="float64"))
        if not np.isfinite(median):
            logger.warning("No usable treatment times for ESI %d; keeping the config value", level)
            median, sigma = fallback.service_median[level], fallback.service_sigma[level]
        service_median[level], service_sigma[level] = median, sigma

        # The triage delay is unobservable on its own: the shortest waits at a level are
        # dominated by it, so its lower quantile is the best available estimate.
        level_waits = seen[seen["triage_level"] == level]["wait"].to_numpy(dtype="float64")
        floor_estimate = quantile(level_waits, 0.05)
        delay_median[level] = (
            float(max(floor_estimate, 0.5))
            if np.isfinite(floor_estimate)
            else fallback.triage_delay_median[level]
        )
        delay_sigma[level] = fallback.triage_delay_sigma[level]

    boarding_median, boarding_sigma = dict(fallback.boarding_median), dict(fallback.boarding_sigma)
    if not admitted.empty and not not_admitted.empty:
        admitted_median, admitted_sigma = _lognormal_fit(
            admitted["post_contact"].to_numpy(dtype="float64")
        )
        typical_treatment = float(np.median(list(service_median.values())))
        excess = max(admitted_median - typical_treatment, 1.0)
        boarding_median["admitted"] = excess
        boarding_sigma["admitted"] = admitted_sigma
        logger.info(
            "Boarding for admitted patients estimated at %.1f min (median post-contact "
            "%.1f minus typical treatment %.1f)",
            excess,
            admitted_median,
            typical_treatment,
        )

    admit_probability = {}
    for level in sorted(visits["triage_level"].unique()):
        subset = seen[seen["triage_level"] == int(level)]
        admit_probability[int(level)] = (
            float((subset["disposition"] == "admitted").mean()) if len(subset) else 0.0
        )

    # Patience is only observed for patients who ran out of it: everyone who was seen
    # had patience longer than their wait, which is censoring. The *shape* per acuity
    # level is taken from the observed time-to-abandon; the *scale* is fitted later
    # against the observed abandonment rate, which is what corrects the censoring.
    patience_median = dict(fallback.patience_median)
    patience_sigma = fallback.patience_sigma
    if "is_lwbs" in visits.columns:
        abandoned = visits[visits["is_lwbs"] & visits["seen_time"].isna()].copy()
        if not abandoned.empty:
            abandoned["time_to_abandon"] = (
                pd.to_datetime(abandoned["departure_time"])
                - pd.to_datetime(abandoned["arrival_time"])
            ).dt.total_seconds() / 60.0
            sigmas = []
            for level in sorted(visits["triage_level"].unique()):
                subset = abandoned[abandoned["triage_level"] == int(level)]["time_to_abandon"]
                median, sigma = _lognormal_fit(subset.to_numpy(dtype="float64"))
                if np.isfinite(median) and len(subset) >= 20:
                    patience_median[int(level)] = median
                    sigmas.append(sigma)
                else:
                    logger.info(
                        "ESI %d has only %d observed abandonment(s); keeping the "
                        "configured patience for it",
                        int(level),
                        len(subset),
                    )
            if sigmas:
                patience_sigma = float(np.mean(sigmas))
            logger.info(
                "Observed time-to-abandon medians by acuity: %s (sigma %.2f)",
                {k: round(v, 1) for k, v in patience_median.items()},
                patience_sigma,
            )

    logger.info(
        "Raw post-contact medians (treatment + boarding) %s",
        {k: round(v, 1) for k, v in service_median.items()},
    )
    return ServiceTimeModel(
        service_median=service_median,
        service_sigma=service_sigma,
        triage_delay_median=delay_median,
        triage_delay_sigma=delay_sigma,
        patience_median=patience_median,
        patience_sigma=patience_sigma,
        boarding_median=boarding_median,
        boarding_sigma=boarding_sigma,
        admit_probability=admit_probability,
    )


def _rescaled(
    service: ServiceTimeModel, fraction: float, patience_scale: float
) -> ServiceTimeModel:
    """Apply a doctor-time fraction and a patience scale to a fitted service model.

    `fraction` splits each level's post-contact time into clinician occupancy and
    everything else, so treatment + boarding always reproduces the observed
    post-contact time. `patience_scale` stretches the observed time-to-abandon
    distribution to correct for the censoring in it.
    """
    treatment = {
        level: max(median * fraction, 0.5)
        for level, median in service.service_median.items()
    }
    # Whatever is not clinician time is boarding, so length of stay is preserved.
    typical_remainder = float(
        np.mean([median * (1.0 - fraction) for median in service.service_median.values()])
    )
    boarding = dict(service.boarding_median)
    boarding["other"] = max(typical_remainder, 0.5)
    boarding["admitted"] = max(
        boarding.get("admitted", typical_remainder) * (1.0 - fraction) + typical_remainder, 0.5
    )

    return ServiceTimeModel(
        service_median=treatment,
        service_sigma=dict(service.service_sigma),
        triage_delay_median=dict(service.triage_delay_median),
        triage_delay_sigma=dict(service.triage_delay_sigma),
        patience_median={
            level: median * patience_scale
            for level, median in service.patience_median.items()
        },
        patience_sigma=service.patience_sigma,
        boarding_median=boarding,
        boarding_sigma=dict(service.boarding_sigma),
        admit_probability=dict(service.admit_probability),
    )


def fit_service_parameters(
    visits: pd.DataFrame,
    config: dict[str, Any],
    arrivals: ArrivalModel,
    service: ServiceTimeModel,
    plan: StaffingPlan,
    n_replications: int = 3,
) -> tuple[float, float, ServiceTimeModel]:
    """Fit the two unobservable parameters: doctor-time fraction and patience scale.

    Both are one-dimensional and monotone, so each is found by bisection:

      * a larger **fraction** means more clinician occupancy, hence longer waits;
      * a larger **patience scale** means patients tolerate longer, hence fewer
        abandonments.

    They interact -- longer waits cause more abandonment -- so the two bisections are
    alternated for a couple of rounds rather than run once.

    Args:
        visits: Validated visits.
        config: Full pipeline config.
        arrivals: Fitted arrival model.
        service: Service model whose medians are still raw post-contact times.
        plan: The department's rostered staffing.
        n_replications: Replications per trial.

    Returns:
        Tuple of (fraction, patience scale, the rescaled service model).
    """
    observed = observed_kpis(visits)
    target_wait = float(observed["mean_wait_minutes"])
    target_abandonment = float(observed["abandonment_rate"])
    horizon = int(config["simulation"]["horizon_hours"])
    settings = config["simulation"]["calibration"]
    inner_steps = int(settings.get("search_steps", 12))
    outer_steps = int(settings.get("search_rounds", 10))

    def simulate(fraction: float, patience_scale: float) -> tuple[float, float]:
        """Mean wait and abandonment rate the simulation produces at these parameters."""
        frame, _ = run_replications(
            config,
            n_replications=n_replications,
            plan=plan,
            arrivals=arrivals,
            service=_rescaled(service, fraction, patience_scale),
            seed_prefix="parameter_search",
            horizon_hours=horizon,
        )
        return (
            float(frame["mean_wait_minutes"].mean()),
            float(frame["abandonment_rate"].mean()),
        )

    def fit_fraction(patience_scale: float) -> tuple[float, float]:
        """Bisect the doctor-time fraction so simulated mean wait matches observed.

        Mean wait rises monotonically with the fraction -- more clinician occupancy
        means more congestion -- so a bisection is safe.

        Returns:
            Tuple of (fraction, the mean wait it produces).
        """
        low, high = 0.01, 1.0
        best, best_error, best_wait = 0.2, float("inf"), float("nan")
        for _ in range(inner_steps):
            middle = 0.5 * (low + high)
            wait, _ = simulate(middle, patience_scale)
            error = abs(wait - target_wait)
            if error < best_error:
                best, best_error, best_wait = middle, error, wait
            if wait > target_wait:
                high = middle  # too congested: the doctor is busy for less of the visit
            else:
                low = middle
            if error / max(target_wait, 1e-9) < 0.02:
                break
        return best, best_wait

    # Outer problem: find the patience scale whose *wait-matched* fraction also
    # reproduces the observed abandonment rate. Every evaluation re-fits the fraction,
    # so the mean-wait constraint holds throughout and the outer search is genuinely
    # one-dimensional.
    cache: dict[float, tuple[float, float, float]] = {}

    def evaluate(patience_scale: float) -> tuple[float, float, float]:
        """Return (fraction, mean wait, abandonment rate) at this patience scale."""
        key = round(patience_scale, 6)
        if key not in cache:
            fraction, wait = fit_fraction(patience_scale)
            _, abandonment = simulate(fraction, patience_scale)
            cache[key] = (fraction, wait, abandonment)
        return cache[key]

    low, high = 0.2, 12.0
    best_scale, best_error = 1.0, float("inf")
    for _ in range(outer_steps):
        middle = 0.5 * (low + high)
        fraction, wait, abandonment = evaluate(middle)
        error = abs(abandonment - target_abandonment)
        if error < best_error:
            best_scale, best_error = middle, error
        logger.debug(
            "  patience x%.3f -> fraction %.4f, wait %.2f, abandonment %.4f",
            middle, fraction, wait, abandonment,
        )
        # More patience means fewer people give up, so abandonment falls as the scale
        # rises even after the fraction is re-matched.
        if abandonment > target_abandonment:
            low = middle
        else:
            high = middle
        if error <= 0.08 * max(target_abandonment, 1e-9):
            break

    patience_scale = best_scale
    fraction, wait, abandonment = evaluate(patience_scale)

    logger.info(
        "Calibration converged: clinician-time fraction %.4f, patience x%.2f",
        fraction,
        patience_scale,
    )
    logger.info(
        "  mean wait   %.2f min vs observed %.2f (%+.1f%%)",
        wait, target_wait, 100.0 * (wait - target_wait) / target_wait,
    )
    logger.info(
        "  abandonment %.4f vs observed %.4f (%+.1f%%)",
        abandonment, target_abandonment,
        100.0 * (abandonment - target_abandonment) / max(target_abandonment, 1e-9),
    )
    logger.info(
        "  A clinician is occupied for about %.0f%% of a patient's post-contact time; "
        "the rest is tests, results and boarding.",
        100.0 * fraction,
    )
    return fraction, patience_scale, _rescaled(service, fraction, patience_scale)


def observed_kpis(visits: pd.DataFrame) -> dict[str, float]:
    """KPIs computed from the real data, in the same shape the simulation reports."""
    seen = visits[visits["seen_time"].notna()]
    waits = (
        pd.to_datetime(seen["seen_time"]) - pd.to_datetime(seen["arrival_time"])
    ).dt.total_seconds() / 60.0
    los = (
        pd.to_datetime(seen["departure_time"]) - pd.to_datetime(seen["arrival_time"])
    ).dt.total_seconds() / 60.0
    abandoned = visits["is_lwbs"] & visits["seen_time"].isna()

    kpis = {
        "mean_wait_minutes": float(waits.mean()),
        "median_wait_minutes": float(waits.median()),
        "p90_wait_minutes": quantile(waits, 0.90),
        "mean_los_minutes": float(los.mean()),
        "p90_los_minutes": quantile(los, 0.90),
        "abandonment_rate": float(abandoned.mean()),
    }
    for level in sorted(visits["triage_level"].unique()):
        subset = visits[visits["triage_level"] == level]
        level_waits = (
            pd.to_datetime(subset["seen_time"]) - pd.to_datetime(subset["arrival_time"])
        ).dt.total_seconds() / 60.0
        kpis[f"mean_wait_esi{int(level)}"] = float(level_waits.mean())
        kpis[f"p90_wait_esi{int(level)}"] = quantile(level_waits, 0.90)
        kpis[f"abandonment_rate_esi{int(level)}"] = float(
            (subset["is_lwbs"] & subset["seen_time"].isna()).mean()
        )
    return kpis


def calibrate_and_validate(
    visits: pd.DataFrame,
    config: dict[str, Any],
    n_replications: int | None = None,
    tables_dir: Path | None = None,
) -> CalibrationResult:
    """Fit the simulation inputs from data and check it reproduces observed KPIs.

    Args:
        visits: Validated visits, including patients who were never seen.
        config: Full pipeline config; reads `simulation.calibration.tolerance`.
        n_replications: Replications for the validation run.
        tables_dir: Where `sim_validation.csv` is written.

    Returns:
        A `CalibrationResult` whose `passed` flag gates the scenario stage.
    """
    settings = config["simulation"]
    horizon = int(settings["horizon_hours"])
    replications = int(n_replications or settings["calibration"].get("replications", 10))

    arrivals = fit_arrival_model(visits, config, horizon)
    raw_service = fit_service_model(visits, config)
    plan = StaffingPlan.from_config(config)
    fraction, patience_scale, service = fit_service_parameters(
        visits, config, arrivals, raw_service, plan
    )
    fitted_parameters = {
        "clinician_time_fraction": round(float(fraction), 6),
        "patience_scale": round(float(patience_scale), 6),
    }
    logger.info(
        "Fitted clinician-time medians: %s (patience scaled x%.2f)",
        {k: round(v, 1) for k, v in service.service_median.items()},
        patience_scale,
    )

    frame, _ = run_replications(
        config,
        n_replications=replications,
        plan=plan,
        arrivals=arrivals,
        service=service,
        seed_prefix="calibration",
        horizon_hours=horizon,
    )
    simulated = {column: float(frame[column].mean()) for column in frame.columns}
    observed = observed_kpis(visits)

    tolerance = settings["calibration"]["tolerance"]
    # A relative tolerance alone is the wrong test where the absolute scale is small:
    # ESI 1 waits are a few minutes, so a clinically irrelevant two-minute discrepancy
    # is a 30% relative error. A metric passes if it is within EITHER the relative
    # tolerance or this absolute floor.
    wait_floor = float(tolerance.get("absolute_floor_minutes", 3.0))
    rate_floor = float(tolerance.get("absolute_floor_rate", 0.01))

    checks = [
        ("mean_wait_minutes", "mean wait", float(tolerance["mean_wait_rel"]), wait_floor),
        ("p90_wait_minutes", "p90 wait", float(tolerance["p90_wait_rel"]), wait_floor),
        ("mean_los_minutes", "mean LOS", float(tolerance["mean_los_rel"]), wait_floor),
        (
            "abandonment_rate",
            "abandonment rate",
            float(tolerance.get("abandonment_rel", 0.50)),
            rate_floor,
        ),
    ]
    for level in sorted(visits["triage_level"].unique()):
        checks.append(
            (
                f"mean_wait_esi{int(level)}",
                f"mean wait ESI {int(level)}",
                float(tolerance["mean_wait_rel"]),
                wait_floor,
            )
        )
    # Only the OVERALL abandonment rate is fitted, so the per-acuity rates are genuine
    # held-out checks and are added to the gate rather than merely reported.
    for level in sorted(visits["triage_level"].unique()):
        checks.append(
            (
                f"abandonment_rate_esi{int(level)}",
                f"abandonment rate ESI {int(level)}",
                float(tolerance.get("abandonment_rel", 0.50)),
                rate_floor,
            )
        )

    rows, failures = [], []
    for key, label, limit, floor in checks:
        real = observed.get(key, float("nan"))
        model = simulated.get(key, float("nan"))
        absolute = abs(model - real)
        relative = absolute / abs(real) if real not in (0, None) and np.isfinite(real) else float("nan")
        within_relative = bool(np.isfinite(relative) and relative <= limit)
        within_absolute = bool(np.isfinite(absolute) and absolute <= floor)
        within = within_relative or within_absolute
        rows.append(
            {
                "metric": label,
                "key": key,
                "observed": round(real, 4),
                "simulated": round(model, 4),
                "absolute_error": round(model - real, 4),
                "relative_error": round(relative, 4) if np.isfinite(relative) else float("nan"),
                "tolerance": limit,
                "absolute_floor": floor,
                "within_tolerance": within,
                "passed_on": (
                    "relative" if within_relative else ("absolute floor" if within_absolute else "neither")
                ),
            }
        )
        rows[-1]["used_for_fitting"] = key in FITTED_METRICS
        # A metric used to fit a parameter cannot also validate that fit.
        if not within and key not in FITTED_METRICS:
            failures.append(
                f"{label}: simulated {model:.3f} vs observed {real:.3f} "
                f"(relative error {relative:.3f} > tolerance {limit:.2f}, "
                f"absolute error {absolute:.3f} > floor {floor:g})"
            )

    validation = pd.DataFrame(rows)
    passed = not failures
    n_gated = sum(1 for row in rows if not row["used_for_fitting"])

    if passed:
        logger.info(
            "Simulation validation PASSED: all %d held-out checks within tolerance "
            "(%d further metric used for fitting)",
            n_gated,
            len(rows) - n_gated,
        )
    else:
        logger.warning("=" * 78)
        logger.warning(
            "SIMULATION VALIDATION FAILED: %d of %d held-out checks outside tolerance.",
            len(failures),
            n_gated,
        )
        for failure in failures:
            logger.warning("  %s", failure)
        logger.warning(
            "Scenario and staffing-search results from this model are NOT trustworthy. "
            "They will refuse to run unless explicitly overridden."
        )
        logger.warning("=" * 78)

    if tables_dir is not None:
        save_table(validation, Path(tables_dir) / "sim_validation.csv")

    return CalibrationResult(
        arrivals=arrivals,
        service=service,
        plan=plan,
        validation=validation,
        passed=passed,
        failures=failures,
        observed=observed,
        simulated=simulated,
        assumptions=[TREATMENT_IDENTIFICATION, PATIENCE_IDENTIFICATION, FITTED_ON],
        fitted_parameters=fitted_parameters,
    )


class ValidationError(RuntimeError):
    """Raised when scenarios are run on a simulation that failed validation."""


def require_valid_calibration(result: CalibrationResult, allow_override: bool = False) -> None:
    """Gate downstream analysis on a passing validation.

    Args:
        result: The calibration outcome.
        allow_override: When True, log loudly and continue anyway.

    Raises:
        ValidationError: If validation failed and no override was given.
    """
    if result.passed:
        return
    if allow_override:
        logger.warning(
            "Proceeding on a simulation that FAILED validation because an override was "
            "passed. Every downstream number is suspect: %s",
            "; ".join(result.failures),
        )
        return
    raise ValidationError(
        "The simulation does not reproduce the observed department, so scenario results "
        "would be meaningless. Failures: "
        + "; ".join(result.failures)
        + ". Re-calibrate, widen simulation.calibration.tolerance if the gap is "
        "acceptable for your purpose, or pass the explicit override flag."
    )
