"""Competing-risks survival analysis of the ED waiting room.

A patient who arrives and waits leaves that state in exactly one of three ways:

    1. they are seen by a clinician        (the event we care about)
    2. they give up and leave unseen       (a *competing* event, not censoring)
    3. the data window closes on them      (genuine right-censoring)

Treating (2) as censoring is the common shortcut and it is **wrong in a specific,
directional way**: censoring assumes the patient would eventually have been seen on the
same hazard as everyone still waiting, when in fact they have left and can never be
seen at all. That assumption inflates the estimated probability of being seen, and it
inflates it most for exactly the low-acuity patients who abandon most often.

So this module reports both:

  * **Kaplan-Meier** for time-to-seen, with abandonment treated as censoring. Included
    because it is what the literature usually reports, and labelled as an approximation
    whose bias is upward.
  * **Aalen-Johansen** cumulative incidence for "seen" and "abandoned" separately,
    which handles the competing risk correctly and whose two curves sum to the
    probability of having left the waiting room at all.

and fits **cause-specific Cox models** for each event: one for the hazard of being seen
(abandonment censored) and one for the hazard of abandoning (being seen censored). A
covariate can push the two in opposite directions, which is the point of fitting both.

The module skips cleanly, with a log message, when the dataset carries no abandonment
information at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.io import save_figure, save_table
from src.utils.logging import get_logger
from src.utils.seed import rng as make_rng

logger = get_logger(__name__)

#: Event codes used throughout this module.
CENSORED, SEEN, ABANDONED = 0, 1, 2
EVENT_LABELS = {CENSORED: "still waiting at cutoff", SEEN: "seen", ABANDONED: "abandoned"}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#6b6b6b"
GRID = "#e3e3e0"
#: Validated categorical slots, in fixed order, for the five ESI levels.
LEVEL_COLOURS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100", 5: "#e87ba4"}
CAUSE_COLOURS = {"seen": "#2a78d6", "abandoned": "#eb6834"}


@dataclass
class SurvivalReport:
    """Everything the survival stage produces, or the reason it was skipped."""

    ran: bool
    reason: str = ""
    event_counts: dict[str, int] = field(default_factory=dict)
    km_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    incidence_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    km_bias_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    cox_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Per-covariate proportional-hazards test, the caveat behind every ratio above.
    cox_ph_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    figures: list[Path] = field(default_factory=list)
    covariates: list[str] = field(default_factory=list)
    administrative_censor_minutes: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "ran": self.ran,
            "reason": self.reason,
            "event_counts": self.event_counts,
            "covariates": self.covariates,
            "administrative_censor_minutes": self.administrative_censor_minutes,
            "km_median_minutes": self.km_table.to_dict(orient="records")
            if not self.km_table.empty
            else [],
            "cumulative_incidence": self.incidence_table.to_dict(orient="records")
            if not self.incidence_table.empty
            else [],
            "km_vs_aalen_johansen": self.km_bias_table.to_dict(orient="records")
            if not self.km_bias_table.empty
            else [],
            "cox": self.cox_table.to_dict(orient="records")
            if not self.cox_table.empty
            else [],
            "cox_proportional_hazards": self.cox_ph_table.to_dict(orient="records")
            if not self.cox_ph_table.empty
            else [],
            "cox_proportional_hazards_summary": summarise_ph_violations(self.cox_ph_table),
            "figures": [str(p) for p in self.figures],
        }


def has_abandonment_information(visits: pd.DataFrame) -> bool:
    """True when the dataset records at least one patient who left without being seen."""
    if "is_lwbs" not in visits.columns or "seen_time" not in visits.columns:
        return False
    return bool((visits["is_lwbs"] & visits["seen_time"].isna()).any())


def build_waiting_room_events(
    visits: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, float]:
    """Turn visits into (duration, event code) for the waiting-room state.

    Durations are measured from arrival. Patients still waiting when the extract closed
    are censored at the cutoff, and everyone is additionally censored administratively
    at `survival.max_followup_minutes` so that one 14-hour outlier cannot dominate a
    tail estimate.

    Args:
        visits: Validated visits, including patients never seen.
        config: Full pipeline config; reads `survival.max_followup_minutes`.

    Returns:
        Tuple of (events frame with `duration_minutes` and `event`, the cutoff timestamp
        expressed as minutes of follow-up allowed).
    """
    frame = visits.copy()
    arrival = pd.to_datetime(frame["arrival_time"])
    seen = pd.to_datetime(frame["seen_time"])
    departure = pd.to_datetime(frame["departure_time"])

    # The extract closes at the last moment anything was observed.
    cutoff = pd.Series(
        [seen.max(), departure.max(), arrival.max()]
    ).max()

    # An abandonment is only an *observed event* if we know when it happened. A row
    # flagged LWBS with no departure timestamp -- which happens when a patient arrives
    # just before the extract closes and the generator's window truncation clears
    # their departure -- has no event time, so it is right-censored at the cutoff like
    # anyone else still present. Taking the `is_lwbs` flag at face value here produced
    # a NaN duration and a bare `TypeError` out of lifelines on generator seed 1.
    abandoned = (
        frame["is_lwbs"].to_numpy(dtype=bool)
        & seen.isna().to_numpy()
        & departure.notna().to_numpy()
    )
    flagged_without_departure = int(
        (frame["is_lwbs"].to_numpy(dtype=bool) & seen.isna().to_numpy() & departure.isna().to_numpy()).sum()
    )
    if flagged_without_departure:
        logger.warning(
            "%d patient(s) are flagged as having left without being seen but have no "
            "departure timestamp, so the moment they left is unknown. They are "
            "right-censored at the cutoff rather than counted as abandonments: an "
            "event with no event time is not an observation.",
            flagged_without_departure,
        )
    still_waiting = frame["still_waiting_at_cutoff"].to_numpy(dtype=bool)
    was_seen = seen.notna().to_numpy()

    duration = np.where(
        was_seen,
        (seen - arrival).dt.total_seconds().to_numpy() / 60.0,
        np.where(
            abandoned,
            (departure - arrival).dt.total_seconds().to_numpy() / 60.0,
            (cutoff - arrival).dt.total_seconds().to_numpy() / 60.0,
        ),
    )
    event = np.where(was_seen, SEEN, np.where(abandoned, ABANDONED, CENSORED))

    horizon = float(config["survival"]["max_followup_minutes"])
    beyond = duration > horizon
    if beyond.any():
        logger.info(
            "Administratively censoring %d patient(s) whose event fell beyond %.0f minutes",
            int(beyond.sum()),
            horizon,
        )
        event = np.where(beyond, CENSORED, event)
        duration = np.minimum(duration, horizon)

    # A zero-length duration breaks the risk set; nudge to a half-second.
    duration = np.maximum(duration, 1.0 / 120.0)

    # Anything still not finite would reach lifelines as a bare TypeError several
    # frames away from the cause. Fail here instead, naming the rows.
    if not np.isfinite(duration).all():
        unusable = int((~np.isfinite(duration)).sum())
        raise ValueError(
            f"{unusable} of {len(duration)} waiting-room durations are not finite, so "
            "a survival model cannot be fitted. This means a row was classified as an "
            "observed event without the timestamp that event needs; check "
            "`seen_time`, `departure_time` and `is_lwbs` for mutual consistency."
        )

    events = frame.assign(duration_minutes=duration, event=event)
    counts = pd.Series(event).value_counts().to_dict()
    logger.info(
        "Waiting-room events: %d seen, %d abandoned, %d censored (cutoff %s, horizon %.0f min)",
        counts.get(SEEN, 0),
        counts.get(ABANDONED, 0),
        counts.get(CENSORED, 0) + int(still_waiting.sum() == 0) * 0,
        cutoff,
        horizon,
    )
    return events, horizon


def kaplan_meier_time_to_seen(
    events: pd.DataFrame, figures_dir: Path
) -> tuple[pd.DataFrame, Path]:
    """KM curves for time-to-seen by triage level, abandonment treated as censoring.

    Reported because it is the conventional analysis, and labelled as an approximation:
    censoring an abandonment pretends the patient remained at risk of being seen, which
    biases the estimated probability of being seen **upwards**.
    """
    from lifelines import KaplanMeierFitter

    figure, axes = plt.subplots(figsize=(8.0, 5.0), facecolor=SURFACE)
    rows = []

    for level in sorted(events["triage_level"].unique()):
        subset = events[events["triage_level"] == level]
        fitter = KaplanMeierFitter(label=f"ESI {int(level)}")
        fitter.fit(
            subset["duration_minutes"],
            event_observed=(subset["event"] == SEEN).astype(int),
        )
        fitter.plot_survival_function(
            ax=axes, color=LEVEL_COLOURS.get(int(level), INK_MUTED), linewidth=2, ci_alpha=0.12
        )
        median = fitter.median_survival_time_
        rows.append(
            {
                "triage_level": int(level),
                "n": int(len(subset)),
                "n_seen": int((subset["event"] == SEEN).sum()),
                "median_minutes_to_seen": round(float(median), 2)
                if np.isfinite(median)
                else float("nan"),
                "prob_unseen_at_60min": round(float(fitter.predict(60.0)), 4),
                "prob_unseen_at_120min": round(float(fitter.predict(120.0)), 4),
            }
        )

    axes.set_facecolor(SURFACE)
    axes.set_xlabel("minutes since arrival", color=INK_MUTED, fontsize=9)
    axes.set_ylabel("probability not yet seen", color=INK_MUTED, fontsize=9)
    axes.set_title(
        "Kaplan-Meier: time to first clinician contact, by acuity",
        color=INK, fontsize=11, loc="left", pad=10,
    )
    axes.grid(color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    axes.tick_params(colors=INK_MUTED, labelsize=9)
    axes.legend(frameon=False, fontsize=9, labelcolor=INK)
    figure.text(
        0.5, -0.04,
        "Abandonment is treated as censoring here, which is an approximation: it assumes "
        "a patient who left\nwould otherwise have been seen on the same hazard, so it "
        "overstates the probability of being seen.",
        ha="center", fontsize=8, color=INK_MUTED,
    )

    path = save_figure(figure, figures_dir / "survival_km_time_to_seen.png")
    return pd.DataFrame(rows), path


def cumulative_incidence(
    events: pd.DataFrame, figures_dir: Path, horizon: float
) -> tuple[pd.DataFrame, Path]:
    """Aalen-Johansen cumulative incidence of each cause, by triage level.

    This is the competing-risks-correct view: the two curves are the probability of
    having been seen, and of having abandoned, by each elapsed time. Unlike the KM
    complement they cannot exceed, in total, the probability of having left the waiting
    room at all.
    """
    from lifelines import AalenJohansenFitter

    levels = sorted(events["triage_level"].unique())
    figure, panels = plt.subplots(
        1, len(levels), figsize=(3.1 * len(levels), 3.8), facecolor=SURFACE, sharey=True
    )
    panels = np.atleast_1d(panels)
    rows: list[dict[str, Any]] = []
    grid = np.array([30.0, 60.0, 120.0, 240.0])

    for panel, level in zip(panels, levels):
        subset = events[events["triage_level"] == level]
        for cause_name, cause_code in (("seen", SEEN), ("abandoned", ABANDONED)):
            if (subset["event"] == cause_code).sum() == 0:
                logger.info("ESI %s has no '%s' events; skipping that curve", level, cause_name)
                continue
            fitter = AalenJohansenFitter(calculate_variance=False, seed=0)
            # jitter resolves tied event times, which Aalen-Johansen requires
            fitter.fit(
                subset["duration_minutes"],
                subset["event"],
                event_of_interest=cause_code,
            )
            curve = fitter.cumulative_density_
            times = curve.index.to_numpy(dtype="float64")
            values = curve.iloc[:, 0].to_numpy(dtype="float64")
            panel.step(
                times, values, where="post",
                color=CAUSE_COLOURS[cause_name], linewidth=2, label=cause_name,
            )
            at = np.interp(grid, times, values, left=0.0)
            rows.append(
                {
                    "triage_level": int(level),
                    "cause": cause_name,
                    "n": int(len(subset)),
                    "n_events": int((subset["event"] == cause_code).sum()),
                    **{f"incidence_at_{int(t)}min": round(float(v), 4) for t, v in zip(grid, at)},
                    f"incidence_at_{int(horizon)}min": round(float(values[-1]), 4),
                }
            )

        panel.set_title(f"ESI {int(level)}", color=INK, fontsize=10, loc="left", pad=8)
        panel.set_facecolor(SURFACE)
        panel.set_xlim(0, min(horizon, 300))
        panel.set_ylim(0, 1.02)
        panel.grid(color=GRID, linewidth=0.8)
        panel.set_axisbelow(True)
        for side in ("top", "right"):
            panel.spines[side].set_visible(False)
        panel.tick_params(colors=INK_MUTED, labelsize=8.5)
        panel.set_xlabel("minutes since arrival", color=INK_MUTED, fontsize=8.5)

    panels[0].set_ylabel("cumulative incidence", color=INK_MUTED, fontsize=9)
    handles, labels = panels[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9,
        bbox_to_anchor=(0.5, -0.08), labelcolor=INK,
    )
    figure.suptitle(
        "Aalen-Johansen cumulative incidence: competing risks of being seen and abandoning",
        color=INK, fontsize=11, x=0.02, ha="left",
    )
    path = save_figure(figure, figures_dir / "survival_cumulative_incidence.png")
    return pd.DataFrame(rows), path


def compare_km_against_aalen_johansen(
    km_table: pd.DataFrame, incidence_table: pd.DataFrame
) -> pd.DataFrame:
    """Quantify how much treating abandonment as censoring overstates "seen".

    Kaplan-Meier's complement, `1 - S(t)`, is routinely read as "probability of having
    been seen by t". Under a competing risk it is not: censoring an abandonment assumes
    the patient stayed at risk of being seen, when they had left. Aalen-Johansen gives
    the quantity people actually mean. The difference is the bias, and it is always
    non-negative -- KM can only overstate.

    Args:
        km_table: Output of `kaplan_meier_time_to_seen`.
        incidence_table: Output of `cumulative_incidence`.

    Returns:
        One row per (triage level, horizon) with both estimates and their gap.
    """
    if km_table.empty or incidence_table.empty:
        return pd.DataFrame()

    seen = incidence_table[incidence_table["cause"] == "seen"].set_index("triage_level")
    rows = []
    for horizon in (60, 120):
        km_column = f"prob_unseen_at_{horizon}min"
        aj_column = f"incidence_at_{horizon}min"
        if km_column not in km_table.columns or aj_column not in seen.columns:
            continue
        for record in km_table.itertuples():
            level = int(record.triage_level)
            if level not in seen.index:
                continue
            km_seen = 1.0 - float(getattr(record, km_column))
            aj_seen = float(seen.loc[level, aj_column])
            rows.append(
                {
                    "triage_level": level,
                    "horizon_minutes": horizon,
                    "km_prob_seen": round(km_seen, 4),
                    "aalen_johansen_prob_seen": round(aj_seen, 4),
                    "km_overstatement": round(km_seen - aj_seen, 4),
                    "km_overstatement_percent": round(
                        100.0 * (km_seen - aj_seen) / aj_seen, 2
                    )
                    if aj_seen > 0
                    else float("nan"),
                }
            )

    table = pd.DataFrame(rows)
    if not table.empty:
        worst = table.loc[table["km_overstatement"].idxmax()]
        logger.info(
            "Treating abandonment as censoring overstates P(seen) by up to %.4f "
            "(%.1f%%), at ESI %d and %d minutes",
            worst["km_overstatement"],
            worst["km_overstatement_percent"],
            int(worst["triage_level"]),
            int(worst["horizon_minutes"]),
        )
    return table


def _available_covariates(events: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    """Configured Cox covariates that are actually present and non-degenerate."""
    requested = list(config["survival"]["cox_covariates"])
    usable: list[str] = []
    for name in requested:
        if name not in events.columns:
            logger.info("Cox covariate %r is absent from this dataset; skipping it", name)
            continue
        column = pd.to_numeric(events[name], errors="coerce")
        if column.notna().sum() == 0 or column.nunique(dropna=True) < 2:
            logger.info("Cox covariate %r is constant or empty; skipping it", name)
            continue
        usable.append(name)
    return usable


#: What a hazard ratio means once its covariate has failed the proportional-hazards
#: test. Carried into the artifacts so the caveat travels with the number.
PH_VIOLATION_NOTE = (
    "A Cox hazard ratio is one number for the whole of the waiting period. That is "
    "only a complete description when the covariate's effect is constant over time, "
    "which is what the proportional-hazards assumption asserts. Where the test below "
    "rejects it, the reported hazard ratio is still a valid TIME-AVERAGED summary -- "
    "weighted towards the times when most patients are still waiting -- but it is not "
    "the effect at any particular minute, and it should not be read as one. A "
    "covariate whose effect strengthens or fades as the wait goes on will show a "
    "ratio somewhere between its early and late values."
)

#: Default significance level for the proportional-hazards test.
PH_TEST_ALPHA = 0.05

#: Rows above which the proportional-hazards test runs on a random subsample.
#:
#: The Schoenfeld-residual test is far more expensive than the Cox fit it diagnoses.
#: Measured on this project: ~24 s at 57,000 rows (`data/raw/ed_v2.csv`) but **272 s**
#: at 155,000 rows, so it grows faster than linearly. The upload workflow accepts up
#: to `uploads.limits.max_rows` = 500,000 rows against a 1,800 s stage timeout, and an
#: unguarded test at that size would spend the whole budget on a diagnostic.
#:
#: Above this threshold the *diagnostic* is computed on a seeded subsample, which
#: costs statistical power and is recorded on every row of the table. The hazard
#: ratios themselves are always fitted on the full data; only the assumption test is
#: subsampled.
#:
#: 100,000 is chosen so that `data/raw/ed_v2.csv` (57,179 validated rows) is still
#: tested in full and the baseline diagnostic is unchanged, while a 500,000-row upload
#: cannot spend its entire stage budget here.
PH_TEST_MAX_ROWS = 100_000


def _fit_cause_models(
    events: pd.DataFrame, config: dict[str, Any]
) -> list[tuple[str, Any, pd.DataFrame, list[str]]]:
    """Fit one Cox model per cause and return them with the data they were fitted on.

    Both the hazard-ratio table and the proportional-hazards test need the fitted
    model *and* its design matrix, and fitting twice would be wasteful and could
    silently diverge. This is the single place the models are fitted.

    Args:
        events: Output of `build_waiting_room_events`, with covariate columns attached.
        config: Full pipeline config; reads `survival.cox_covariates`.

    Returns:
        One tuple per fitted cause: (cause name, fitter, design matrix, covariates).
        Causes with too few events to fit are omitted, with a warning.
    """
    from lifelines import CoxPHFitter

    covariates = _available_covariates(events, config)
    if not covariates:
        logger.warning("No usable Cox covariates; skipping the cause-specific models")
        return []

    fitted: list[tuple[str, Any, pd.DataFrame, list[str]]] = []
    for cause_name, cause_code in (("seen", SEEN), ("abandoned", ABANDONED)):
        design = events.loc[:, [*covariates, "duration_minutes", "event"]].copy()
        for name in covariates:
            design[name] = pd.to_numeric(design[name], errors="coerce")
        design = design.dropna()
        design["observed"] = (design["event"] == cause_code).astype(int)
        design = design.drop(columns=["event"])

        if design["observed"].sum() < 10:
            logger.warning(
                "Only %d '%s' events after dropping missing covariates; skipping that model",
                int(design["observed"].sum()),
                cause_name,
            )
            continue

        fitter = CoxPHFitter(penalizer=float(config["survival"].get("cox_penalizer", 0.1)))
        fitter.fit(design, duration_col="duration_minutes", event_col="observed")
        logger.info(
            "Cause-specific Cox for %r: %d events, concordance %.4f",
            cause_name,
            int(design["observed"].sum()),
            float(fitter.concordance_index_),
        )
        fitted.append((cause_name, fitter, design, covariates))
    return fitted


def _diagnostic_sample(
    design: pd.DataFrame, max_rows: int, seed: int, label: str
) -> tuple[pd.DataFrame, bool]:
    """Shrink a design matrix for the assumption test, when it is too large to test.

    A subsample is taken only when the full matrix would make the diagnostic cost more
    than the analysis it describes -- see `PH_TEST_MAX_ROWS`. The draw is seeded, so a
    re-run reports the same p-values.

    Args:
        design: The matrix the Cox model was fitted on.
        max_rows: Threshold above which to subsample.
        seed: Global seed, for a derived stream.
        label: Mixed into the stream so each cause draws independently.

    Returns:
        Tuple of (matrix to test, whether it was subsampled).
    """
    if max_rows <= 0 or len(design) <= max_rows:
        return design, False

    generator = make_rng(seed, f"ph_test::{label}")
    chosen = generator.choice(len(design), size=int(max_rows), replace=False)
    logger.warning(
        "Proportional-hazards test for %r runs on a seeded subsample of %d of %d "
        "rows: the Schoenfeld test costs more than the Cox fit it diagnoses and "
        "grows faster than linearly. The hazard ratios are still fitted on all %d "
        "rows; only this diagnostic is subsampled, and it therefore has less power.",
        label, int(max_rows), len(design), len(design),
    )
    return design.iloc[np.sort(chosen)], True


def proportional_hazards_table(
    fitted: list[tuple[str, Any, pd.DataFrame, list[str]]],
    alpha: float = PH_TEST_ALPHA,
    max_rows: int = PH_TEST_MAX_ROWS,
    seed: int = 0,
    penalizer: float = 0.1,
) -> pd.DataFrame:
    """Test the proportional-hazards assumption for every covariate of every cause.

    Uses lifelines' `proportional_hazard_test`, which is the Schoenfeld-residual test:
    it regresses the scaled Schoenfeld residuals of each covariate on transformed time
    and asks whether the slope is non-zero. A small p-value means the covariate's
    effect changes over the waiting period, so a single hazard ratio is a time-average
    rather than a description.

    The rank time-transform is used, which is the lifelines default and is robust to
    the long right tail of a waiting-time distribution.

    **On large datasets the diagnostic is subsampled** (see `PH_TEST_MAX_ROWS`). The
    test is then a proper test on a random subsample -- the model is refitted on that
    subsample rather than the full-data residuals being filtered, which would not be
    the same thing -- and every row records that it happened.

    Args:
        fitted: Output of `_fit_cause_models`.
        alpha: Significance level below which the assumption is called violated.
        max_rows: Rows above which to subsample for the test; 0 disables subsampling.
        seed: Global seed, for the subsample draw.
        penalizer: Penalizer for the refit on a subsample; matches the main fit.

    Returns:
        One row per (cause, covariate), with the test statistic, the p-value, how many
        rows were tested and a plain verdict. Empty when no model could be fitted.
    """
    if not fitted:
        return pd.DataFrame()

    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test

    rows: list[dict[str, Any]] = []
    for cause_name, fitter, design, covariates in fitted:
        tested, subsampled = _diagnostic_sample(design, max_rows, seed, cause_name)
        try:
            if subsampled:
                # Refit on the subsample: the Schoenfeld residuals must come from the
                # model that was fitted on the rows being tested.
                fitter = CoxPHFitter(penalizer=penalizer)
                fitter.fit(tested, duration_col="duration_minutes", event_col="observed")
            result = proportional_hazard_test(fitter, tested, time_transform="rank")
        except Exception as error:  # noqa: BLE001 - diagnostics must never fail a run
            # A failed diagnostic is reported as unavailable, not as a passing test:
            # silently dropping it would read as "the assumption holds".
            logger.warning(
                "Proportional-hazards test could not be computed for %r (%s: %s); "
                "the hazard ratios for that cause are reported untested",
                cause_name,
                type(error).__name__,
                error,
            )
            for covariate in covariates:
                rows.append(
                    {
                        "cause": cause_name,
                        "covariate": covariate,
                        "test": "Schoenfeld residuals (rank time-transform)",
                        "test_statistic": None,
                        "p_value": None,
                        "violates_proportional_hazards": None,
                        "n_tested": int(len(tested)),
                        "n_fitted": int(len(design)),
                        "diagnostic_subsampled": subsampled,
                        "verdict": f"not available ({type(error).__name__})",
                    }
                )
            continue

        summary = result.summary.reset_index()
        name_column = summary.columns[0]
        for _, record in summary.iterrows():
            p_value = float(record["p"])
            violates = bool(p_value < alpha)
            rows.append(
                {
                    "cause": cause_name,
                    "covariate": str(record[name_column]),
                    "test": "Schoenfeld residuals (rank time-transform)",
                    "test_statistic": round(float(record["test_statistic"]), 4),
                    "p_value": p_value,
                    "violates_proportional_hazards": violates,
                    "n_tested": int(len(tested)),
                    "n_fitted": int(len(design)),
                    "diagnostic_subsampled": subsampled,
                    "verdict": (
                        "effect changes over the wait; the hazard ratio is a "
                        "time-average only"
                        if violates
                        else "no evidence against proportional hazards"
                    ),
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    violating = table[table["violates_proportional_hazards"] == True]  # noqa: E712
    if violating.empty:
        logger.info(
            "Proportional hazards: no covariate rejects the assumption at p < %.2f", alpha
        )
    else:
        logger.warning(
            "Proportional hazards REJECTED at p < %.2f for %d covariate(s): %s. %s",
            alpha,
            len(violating),
            ", ".join(f"{row.cause}/{row.covariate}" for row in violating.itertuples()),
            PH_VIOLATION_NOTE,
        )
    return table


def summarise_ph_violations(ph_table: pd.DataFrame) -> dict[str, Any]:
    """Turn the proportional-hazards table into something the report can quote.

    Args:
        ph_table: Output of `proportional_hazards_table`.

    Returns:
        A dict naming which covariates violated the assumption and what that limits,
        written as data so the report does not hardcode a claim.
    """
    if ph_table.empty:
        return {"ran": False, "reason": "no Cox model could be fitted"}

    violating = ph_table[ph_table["violates_proportional_hazards"] == True]  # noqa: E712
    untested = ph_table[ph_table["violates_proportional_hazards"].isna()]
    return {
        "ran": True,
        "test": "Schoenfeld residuals (rank time-transform)",
        "alpha": PH_TEST_ALPHA,
        "n_covariate_tests": int(len(ph_table)),
        "n_violations": int(len(violating)),
        "n_untested": int(len(untested)),
        "violations": [
            {
                "cause": row.cause,
                "covariate": row.covariate,
                "p_value": row.p_value,
            }
            for row in violating.itertuples()
        ],
        "all_assumptions_hold": bool(violating.empty and untested.empty),
        "limitation_note": PH_VIOLATION_NOTE,
    }


def cause_specific_cox(
    events: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Fit one Cox model per cause, treating the other cause as censoring.

    A cause-specific hazard answers "among patients still waiting, what multiplies the
    rate of *this* outcome?". Fitting both is what reveals a covariate that raises one
    hazard while lowering the other.

    Args:
        events: Output of `build_waiting_room_events`, with covariate columns attached.
        config: Full pipeline config; reads `survival.cox_covariates`.

    Returns:
        Long table of hazard ratios with confidence intervals, one block per cause.
    """
    return cox_tables(events, config)[0]


def cox_tables(
    events: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hazard ratios and their proportional-hazards diagnostics, from one fit.

    Args:
        events: Output of `build_waiting_room_events`, with covariate columns attached.
        config: Full pipeline config.

    Returns:
        Tuple of (hazard-ratio table, proportional-hazards test table).
    """
    fitted = _fit_cause_models(events, config)
    if not fitted:
        return pd.DataFrame(), pd.DataFrame()

    ph_table = proportional_hazards_table(
        fitted,
        alpha=float(config["survival"].get("ph_test_alpha", PH_TEST_ALPHA)),
        max_rows=int(config["survival"].get("ph_test_max_rows", PH_TEST_MAX_ROWS)),
        seed=int(config.get("seed", 0)),
        penalizer=float(config["survival"].get("cox_penalizer", 0.1)),
    )
    violated = set()
    if not ph_table.empty:
        violated = {
            (row.cause, row.covariate)
            for row in ph_table.itertuples()
            if row.violates_proportional_hazards
        }

    rows: list[dict[str, Any]] = []
    for cause_name, fitter, design, covariates in fitted:
        summary = fitter.summary
        for covariate in covariates:
            if covariate not in summary.index:
                continue
            record = summary.loc[covariate]
            breaks_ph = (cause_name, covariate) in violated
            rows.append(
                {
                    "cause": cause_name,
                    "covariate": covariate,
                    "coef": round(float(record["coef"]), 4),
                    "hazard_ratio": round(float(record["exp(coef)"]), 4),
                    "hr_lower_95": round(float(record["exp(coef) lower 95%"]), 4),
                    "hr_upper_95": round(float(record["exp(coef) upper 95%"]), 4),
                    "p_value": float(record["p"]),
                    "significant_at_5pct": bool(record["p"] < 0.05),
                    # Carried on the hazard-ratio row itself, so the caveat cannot be
                    # lost by reading one table without the other.
                    "violates_proportional_hazards": breaks_ph,
                    "interpretation": (
                        "time-averaged only: the effect changes over the wait"
                        if breaks_ph
                        else "constant-effect reading is supported by the test"
                    ),
                    "n": int(len(design)),
                    "n_events": int(design["observed"].sum()),
                }
            )
    return pd.DataFrame(rows), ph_table


def run_survival_analysis(
    visits: pd.DataFrame,
    config: dict[str, Any],
    tables_dir: Path,
    figures_dir: Path,
) -> SurvivalReport:
    """Run the full competing-risks analysis, or skip cleanly when it does not apply.

    Args:
        visits: Validated visits with queue-state features attached, including patients
            who were never seen.
        config: Full pipeline config.
        tables_dir: Where CSVs go.
        figures_dir: Where PNGs go.

    Returns:
        A `SurvivalReport`. `ran` is False, with a reason, when the dataset carries no
        abandonment information.
    """
    if not has_abandonment_information(visits):
        reason = (
            "This dataset records no patient who left without being seen, so there is "
            "no competing risk to model and the survival analysis is skipped."
        )
        logger.info("Survival analysis skipped: %s", reason)
        return SurvivalReport(ran=False, reason=reason)

    events, horizon = build_waiting_room_events(visits, config)
    counts = {
        EVENT_LABELS[code]: int((events["event"] == code).sum())
        for code in (SEEN, ABANDONED, CENSORED)
    }

    km_table, km_figure = kaplan_meier_time_to_seen(events, figures_dir)
    incidence_table, incidence_figure = cumulative_incidence(events, figures_dir, horizon)
    cox_table, cox_ph_table = cox_tables(events, config)
    km_bias_table = compare_km_against_aalen_johansen(km_table, incidence_table)

    save_table(km_table, tables_dir / "survival_km_time_to_seen.csv")
    save_table(incidence_table, tables_dir / "survival_cumulative_incidence.csv")
    if not km_bias_table.empty:
        save_table(km_bias_table, tables_dir / "survival_km_vs_aalen_johansen.csv")
    if not cox_table.empty:
        save_table(cox_table, tables_dir / "survival_cox.csv")
    if not cox_ph_table.empty:
        save_table(cox_ph_table, tables_dir / "survival_cox_ph_test.csv")

    report = SurvivalReport(
        ran=True,
        event_counts=counts,
        km_table=km_table,
        incidence_table=incidence_table,
        km_bias_table=km_bias_table,
        cox_table=cox_table,
        cox_ph_table=cox_ph_table,
        figures=[km_figure, incidence_figure],
        covariates=_available_covariates(events, config),
        administrative_censor_minutes=horizon,
    )
    logger.info(
        "Survival analysis complete: %s",
        ", ".join(f"{label} {count}" for label, count in counts.items()),
    )
    return report
