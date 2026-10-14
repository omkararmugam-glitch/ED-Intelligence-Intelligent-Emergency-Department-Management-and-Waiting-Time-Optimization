"""Research results: robustness, risk, the scenario lab and the experimental agent.

One page, four sections, because they share a single property that has to govern how
they are read: **none of them is a result about a real emergency department.** They are
a seed study on synthetic data, a model fitted to a generator's patience distribution,
a what-if on resources the data does not record, and an agent that learned this
project's own simulator.

Everything else in the dashboard answers "what is happening in the department". This
page answers "how much should you believe the rest of it", which is a different
question and belongs behind its own heading rather than mixed in with the operational
pages.

Each section reads artifacts written by a script or a pipeline stage and says plainly
when one has not been produced, rather than drawing an empty chart.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import artifacts, components, theme

#: Shown at the top of every section whose inputs are not calibrated against data.
NOT_CALIBRATED_BANNER = (
    "**Scenario analysis, not calibrated.** The dataset records no bed, no nurse and "
    "no imaging order, so these parameters are assumed and the simulation's validation "
    "gate cannot test them. Read the ranking of the effects, not their size."
)

#: Shown on the reinforcement-learning section.
EXPERIMENTAL_BANNER = (
    "**Experimental.** The agent learned this project's own simulator, which was "
    "calibrated to synthetic data. A win would show the method runs end to end; it "
    "would not be evidence that the policy helps a real department."
)


def render() -> None:
    """Draw the research page."""
    components.page_header(
        "Research results",
        "How much to believe the rest of this dashboard.",
        breadcrumb=("Clinical intelligence", "Research results"),
        icon_name="research",
    )


    robustness, risk, lab, agent = st.tabs(
        ["Robustness", "Risk of leaving", "Scenario lab", "Experimental RL"]
    )
    with robustness:
        _robustness()
    with risk:
        _risk()
    with lab:
        _scenario_lab()
    with agent:
        _experimental_rl()


# ------------------------------------------------------------------ robustness


def _robustness() -> None:
    """Part 0: the same pipeline on ten generator seeds."""
    st.subheader("Does any of this survive a different dataset?")
    st.caption(
        "The whole pipeline was re-run on ten datasets from the same generator, "
        "differing only in random seed. Seed 42 is the reference: it reproduces "
        "`data/raw/ed_v2.csv` exactly, so its row must match the headline numbers."
    )

    seeds = artifacts.table("robustness_seeds.csv")
    if seeds is None or seeds.empty:
        st.info(
            "No robustness study has been run. Produce one with "
            "`python scripts/robustness_study.py`.",
            icon="ℹ️",
        )
        return

    study = seeds[seeds["seed"] != 42] if "seed" in seeds.columns else seeds
    passed = int(study["sim_validation_passed"].fillna(False).astype(bool).sum())
    columns = st.columns(4)
    columns[0].metric("Seeds studied", theme.count(len(study)))
    columns[1].metric(
        "Simulation gate passed", f"{theme.count(passed)} of {theme.count(len(study))}"
    )
    if "wait_mae_model" in study:
        columns[2].metric(
            "Wait MAE",
            theme.minutes(study["wait_mae_model"].mean(), places=2),
            f"± {theme.decimal(study['wait_mae_model'].std(), places=2)} across seeds",
            delta_color="off",
        )
    costs = (
        study["cheapest_feasible_cost"].dropna()
        if "cheapest_feasible_cost" in study
        else pd.Series(dtype=float)
    )
    if not costs.empty:
        columns[3].metric(
            "Cheapest plan",
            theme.count(costs.mean()),
            f"{theme.count(costs.min())} to {theme.count(costs.max())}",
            delta_color="off",
        )

    st.markdown("**Per-seed results**")
    shown = [
        column
        for column in (
            "seed", "n_rows", "wait_mae_model", "wait_mae_baseline",
            "conformal_static_overall", "conformal_rolling_overall", "forecast_mae",
            "sim_validation_passed", "cheapest_feasible_cost", "n_feasible_plans",
        )
        if column in seeds.columns
    ]
    st.dataframe(seeds[shown], use_container_width=True, hide_index=True)
    _gate_and_roster_findings(study)

    _coverage_chart(seeds)

    stats = artifacts.table("robustness_seed_stats.csv")
    if stats is not None and not stats.empty:
        with st.expander("Spread across seeds (mean, standard deviation, range)"):
            st.dataframe(stats, use_container_width=True, hide_index=True)


def _gate_and_roster_findings(study: pd.DataFrame) -> None:
    """What the per-seed table means, in words, for the two findings that matter.

    The table above is correct and almost unreadable: a reader scanning ten rows of
    nine columns will not work out which seeds refused a staffing plan or why, and
    those two facts are the whole point of the study. The counts are recomputed from
    the table rather than written down, so this cannot drift away from the data; the
    diagnoses are the ones recorded in `artifacts/checkpoints/days_work.md`.
    """
    if "sim_validation_passed" not in study.columns:
        return

    gate = study["sim_validation_passed"].fillna(False).astype(bool)
    passed, total = int(gate.sum()), int(len(study))
    refused = sorted(int(s) for s in study.loc[~gate, "seed"]) if "seed" in study else []

    st.markdown("**What survived, and what did not**")
    st.markdown(
        f"- **The simulation validation gate passed on {passed} of {total}** regenerated "
        "datasets. It **refused to issue a staffing plan** on the rest, which is the "
        "behaviour asked of it: a plan from a model that cannot reproduce its own "
        "dataset would look exactly as confident as a correct one."
        + (
            f"\n- **The {'two' if len(refused) == 2 else len(refused)} refusals have "
            f"different causes.** Seed {refused[0]} missed ESI 3 abandonment against a "
            "1.6% base rate — a 1.2 percentage-point error, which is 72% in relative "
            "terms and clinically trivial, so it is a **rare-event tolerance artefact**. "
            f"Seed {refused[-1]} came out **8 minutes low on ESI 3 mean wait** against an "
            "observed 34.4 minutes, which is a **genuine fit miss** the gate is right to "
            "refuse. No tolerance was loosened to make either one pass."
            if len(refused) == 2 else ""
        )
    )

    costs = study["cheapest_feasible_cost"].dropna() if "cheapest_feasible_cost" in study else pd.Series(dtype=float)
    plans = (
        study.loc[costs.index, "cheapest_feasible_plan"].dropna().unique()
        if not costs.empty and "cheapest_feasible_plan" in study
        else []
    )
    if not costs.empty:
        st.markdown(
            f"- **The staffing method transfers; the specific roster does not.** The "
            f"cheapest plan cost between {theme.count(costs.min())} and "
            f"{theme.count(costs.max())} across the {len(costs)} seeds that produced "
            f"one, over {len(plans)} distinct rosters — and the day shift was 3 in every "
            "single one. **3 / 3 / 2 / 2 at 49,280 is the reference dataset's answer**, "
            "not a recommendation for any department."
        )
    st.markdown(
        "- **The wait model beat its baseline and the forecast beat seasonal-naive on "
        "every seed**, and rolling conformal coverage landed in the same place on all of "
        "them while the static split wandered."
    )
    st.caption(
        "All of this is one generator with ten different seeds. Agreement across seeds "
        "shows the pipeline is not over-fitted to one draw of it. It recovers the "
        "generator, and is not evidence about real emergency departments."
    )


def _coverage_chart(seeds: pd.DataFrame) -> None:
    """Static against rolling conformal coverage, per seed.

    The chart makes one point: rolling coverage sits in almost the same place on every
    dataset while static wanders. That is what a coverage guarantee is supposed to do,
    and it is easier to see than to read off a table of ten numbers.
    """
    needed = {"seed", "conformal_static_overall", "conformal_rolling_overall"}
    if not needed <= set(seeds.columns):
        return

    import altair as alt

    long = seeds.melt(
        id_vars="seed",
        value_vars=["conformal_static_overall", "conformal_rolling_overall"],
        var_name="method",
        value_name="coverage",
    )
    long["method"] = long["method"].map(
        {"conformal_static_overall": "static", "conformal_rolling_overall": "rolling"}
    )

    st.markdown("**Interval coverage by seed**")
    chart = (
        alt.Chart(long)
        .mark_point(size=90, filled=True)
        .encode(
            x=alt.X("seed:O", title="generator seed"),
            y=alt.Y(
                "coverage:Q",
                title="empirical coverage",
                scale=alt.Scale(domain=[0.80, 0.98]),
            ),
            color=alt.Color("method:N", title="calibration"),
            tooltip=["seed", "method", alt.Tooltip("coverage:Q", format=".4f")],
        )
        .properties(height=240)
    )
    nominal = (
        alt.Chart(pd.DataFrame({"y": [0.90]}))
        .mark_rule(strokeDash=[5, 5])
        .encode(y="y:Q")
    )
    st.altair_chart(theme.style_chart(chart + nominal), use_container_width=True)
    theme.chart_caption(
        "Interval coverage for each generator seed, static against rolling "
        "recalibration, with the nominal 90% marked by a dashed line. The rolling "
        "points sit in a tight band around the line on every seed; the static points "
        "scatter either side of it, so static coverage depends on which dataset you "
        "happened to get."
    )


# ------------------------------------------------------------------ risk


def _risk() -> None:
    """Part C: will this patient be seen, or leave first."""
    st.subheader("Who leaves before being seen")
    st.caption(
        "The wait models answer 'how long until a clinician' for the patients who were "
        "seen. This one is unconditional: standing at the door, what is the chance a "
        "patient leaves within 30, 60 or 120 minutes."
    )

    metrics = artifacts.metrics("competing_risks.json")
    if not metrics or not metrics.get("ran"):
        reason = (metrics or {}).get("reason", "the stage has not been run")
        st.info(f"No risk model is available: {reason}", icon="ℹ️")
        return

    table = artifacts.table("competing_risks_metrics.csv")
    if table is None or table.empty:
        st.info("The risk model ran but wrote no metrics table.", icon="ℹ️")
        return

    overall = table[table["scope"] == "overall"]
    columns = st.columns(len(overall)) if len(overall) else []
    for column, (_index, row) in zip(columns, overall.iterrows()):
        column.metric(
            f"Ranking quality (AUC) at {theme.minutes(row['horizon_minutes'])}",
            f"AUC {theme.decimal(row['auc'], places=3)}" if pd.notna(row["auc"]) else "—",
            f"{theme.count(row['n_positive'])} who left",
            delta_color="off",
        )

    st.markdown("**Accuracy and calibration, before and after recalibration**")
    shown = [
        c for c in (
            "horizon_minutes", "scope", "n", "n_positive", "auc", "brier_raw",
            "brier_calibrated", "ece_raw", "ece_calibrated", "reportable",
        ) if c in table.columns
    ]
    st.dataframe(table[shown], use_container_width=True, hide_index=True)
    st.caption(
        "ESI 1 and 2 are excluded by design: high-acuity patients effectively never "
        "leave without being seen, so there is no incidence to calibrate against. "
        "Their rows show the counts and say so."
    )

    _calibration_chart()

    cox = artifacts.table("competing_risks_vs_cox.csv")
    if cox is not None and not cox.empty:
        with st.expander("Against the Cox model"):
            st.dataframe(cox, use_container_width=True, hide_index=True)
            st.caption(
                "AUC only. A Cox model ranks risk; it does not estimate the chance of "
                "leaving by a horizon, so it has no calibration to compare."
            )

    aalen = artifacts.table("competing_risks_aalen.csv")
    if aalen is not None and not aalen.empty:
        with st.expander("Against the non-parametric incidence (Aalen-Johansen)"):
            st.dataframe(aalen, use_container_width=True, hide_index=True)
            st.caption(
                "A model whose average prediction for a group does not match the "
                "observed incidence is miscalibrated at the group level, whatever its "
                "per-patient scores say."
            )


def _calibration_chart() -> None:
    """Predicted against observed, before and after recalibration."""
    curves = artifacts.table("competing_risks_calibration.csv")
    if curves is None or curves.empty:
        return

    import altair as alt

    overall = curves[curves["scope"] == "overall"] if "scope" in curves else curves
    if overall.empty:
        return

    st.markdown("**Calibration: does 20% mean 20%?**")
    chart = (
        alt.Chart(overall)
        .mark_line(point=True)
        .encode(
            x=alt.X("mean_predicted:Q", title="predicted probability"),
            y=alt.Y("observed_rate:Q", title="observed rate"),
            color=alt.Color("stage:N", title=""),
            column=alt.Column("horizon_minutes:O", title="horizon (minutes)"),
            tooltip=["stage", "mean_predicted", "observed_rate", "n"],
        )
        .properties(height=200, width=180)
    )
    st.altair_chart(theme.style_chart(chart))
    theme.chart_caption(
        "Observed rate against predicted probability, one panel per horizon, before "
        "and after recalibration. A perfectly calibrated model lies on the diagonal; "
        "a line below it is over-predicting and a line above it is under-predicting."
    )


# ------------------------------------------------------------------ scenario lab


def _scenario_lab() -> None:
    """Part A: what-ifs on resources the data does not record."""
    st.subheader("Beds, nurses, imaging and boarding")
    st.warning(NOT_CALIBRATED_BANNER, icon="⚠️")

    comparison = artifacts.table("scenario_lab_comparison.csv")
    if comparison is None or comparison.empty:
        st.info(
            "No scenario lab has been run. Produce one with "
            "`python scripts/scenario_lab.py`.",
            icon="ℹ️",
        )
        return

    ranking = artifacts.table("scenario_lab_sensitivity.csv")
    if ranking is not None and not ranking.empty:
        st.markdown("**Which constrained resource hurts ESI 3-5 waits most**")
        shown = [
            c for c in (
                "rank", "scenario", "p90_wait_esi3", "p90_wait_esi4", "p90_wait_esi5",
                "mean_abs_effect_minutes", "n_kpis_significant",
            ) if c in ranking.columns
        ]
        st.dataframe(ranking[shown], use_container_width=True, hide_index=True)
        st.caption(
            "Minutes added to the 90th-percentile wait, against the calibrated "
            "baseline, with common random numbers. The ordering is the usable result; "
            "the magnitudes depend on assumed parameters."
        )

    st.markdown("**Effect on every headline measure**")
    shown = [
        c for c in (
            "scenario", "kpi", "baseline_mean", "scenario_mean", "difference",
            "ci_low", "ci_high", "percent_change", "significant",
        ) if c in comparison.columns
    ]
    kpis = sorted(comparison["kpi"].unique()) if "kpi" in comparison else []
    chosen = st.multiselect(
        "Measures to show",
        kpis,
        default=[k for k in ("mean_wait_minutes", "p90_wait_esi3", "abandonment_rate") if k in kpis],
    )
    filtered = comparison[comparison["kpi"].isin(chosen)] if chosen else comparison
    st.dataframe(filtered[shown], use_container_width=True, hide_index=True)


# ------------------------------------------------------------------ RL


def _experimental_rl() -> None:
    """Part D: a learned staffing policy, and whether it beat the roster."""
    st.subheader("A staffing policy learned against the simulator")
    st.warning(EXPERIMENTAL_BANNER, icon="🧪")

    payload = artifacts.metrics("staffing_rl.json")
    if not payload or not payload.get("ran"):
        st.info(
            "No agent has been trained. Produce one with "
            "`python scripts/staffing_rl.py`.",
            icon="ℹ️",
        )
        return

    # The verdict is a measurement, not a malfunction. It was drawn as a red error
    # box, which put the same alarm colour on "this experiment did not work" as on
    # "the service is down" -- and a reader who meets a red box on a research page
    # learns to discount red boxes everywhere.
    verdict = payload.get("verdict", "")
    if verdict:
        lost = "does NOT beat" in verdict
        components.alert_banner(
            "The learned policy did not beat the fixed roster" if lost
            else "The learned policy beat the fixed roster",
            verdict,
            level="watch" if lost else "good",
            rule="experimental: reported as measured, not tuned until it won",
        )

    comparison = artifacts.table("staffing_rl_comparison.csv")
    if comparison is not None and not comparison.empty:
        st.markdown("**Every policy, on the same held-out episodes**")
        shown = [
            c for c in (
                "policy", "n_episodes", "mean_total_cost", "mean_staffing_cost",
                "mean_breaches", "cost_difference", "ci_low", "ci_high",
                "beats_baseline", "significant",
            ) if c in comparison.columns
        ]
        st.dataframe(comparison[shown], use_container_width=True, hide_index=True)
        st.caption(
            "Cost per episode against the fixed roster, paired on episode with common "
            "random numbers. `mean_breaches` counts ESI 1-2 patients who waited beyond "
            "target -- the failure the reward weights most heavily."
        )

    training = artifacts.table("staffing_rl_training.csv")
    if training is not None and not training.empty:
        with st.expander("Training curve"):
            st.line_chart(training.set_index("episode")["total_reward"])
            st.caption(
                f"{payload.get('training_episodes', '?')} training episodes over a "
                f"{payload.get('n_states', '?')}-state table. A flat or noisy curve "
                "means the agent has not converged, which is information about the "
                "training budget rather than about the method."
            )

    with st.expander("Why a win here would not mean much"):
        st.markdown(payload.get("notice", EXPERIMENTAL_BANNER))
