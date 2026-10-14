"""Command Center: today at a glance.

The landing page, and the only page most readers will look at. It answers four
questions in four cards -- how many are coming, when the worst of it is, how long
someone waits now, and whether the next six hours look like trouble -- and then
says in two or three sentences what the service suggests doing about it.

Two rules shaped it. **No alert without evidence**: every line in the alerts list
is derived from a number on this page, and when the data does not support an alert
the list says so rather than inventing one. **Nothing claims to be live**: the data
is synthetic and the forecast is issued from the end of a recorded history, so the
wording is "expected" and "estimated", never "current" or "real-time".
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import streamlit as st

from app import alerts, api_client, artifacts, components, replay, theme
from app.config import INTERVAL_COVERAGE
from app.views import _shared

#: Hours ahead the congestion judgement looks.
RISK_HORIZON_HOURS = 6

#: Arrivals per hour above which an hour counts as busy. Taken from the data
#: rather than chosen: it is the 80th percentile of the forecast itself, so
#: "busy" means busy for this department rather than busy in the abstract.
BUSY_PERCENTILE = 0.80

#: How many of the next `RISK_HORIZON_HOURS` must be busy for each risk level.
RISK_RULES = (
    (0.60, "act", "High"),
    (0.30, "watch", "Moderate"),
    (0.00, "good", "Low"),
)

#: Default horizon of the chart on this page.
CHART_HOURS = 48


def render() -> None:
    """Draw the Command Center."""
    moment = _replay_control()
    census = _census(moment)

    components.page_header(
        "Command center",
        "Where the department stands, what is coming, and what needs a decision.",
        breadcrumb=("Operations", "Command center"),
        updated=(
            theme.clock(census["at"], "%a %d %b %H:%M")
            if census.get("at") is not None else None
        ),
        icon_name="command_center",
    )

    forecast, ok = _shared.load(
        lambda: api_client.forecast(CHART_HOURS, as_of=moment),
        spinner="Calculating the arrivals forecast...",
        error_title="The forecast did not load",
        retry_key="cc_retry_forecast",
        placeholder_height=110,
        placeholder_rows=2,
    )
    if not ok:
        return

    frame = _as_frame(forecast)
    if frame.empty:
        theme.empty_state(
            "The forecast came back empty",
            "The service answered but returned no hours. Run the analysis once to "
            "produce an arrivals model.",
            icon="📈",
        )
        return

    risk = _congestion(frame)
    _kpi_row(frame, risk, moment)

    st.divider()
    forecast_column, mix = st.columns([3, 2], gap="medium")
    with forecast_column:
        components.section_header(
            "Expected arrivals",
            f"Next {CHART_HOURS} hours. The band is the "
            f"{theme.percent(INTERVAL_COVERAGE, places=0)} range; busy hours are "
            "marked, and the dashed line is what repeating last week would have "
            "predicted.",
        )
        _forecast_chart(frame)

    with mix:
        components.section_header(
            "Who is waiting", "By acuity, at the replay moment."
        )
        _acuity_mix(census)

    st.divider()
    waits_column, heat_column = st.columns(2, gap="medium")
    with waits_column:
        components.section_header(
            "Waiting time by acuity",
            f"Against the {alerts.WAIT_TARGET_MINUTES}-minute door-to-doctor target.",
        )
        _waits_by_acuity(census)
    with heat_column:
        components.section_header(
            "When the department is busy", "Arrivals per day of week and hour."
        )
        _congestion_heatmap()

    st.divider()
    components.section_header(
        "Alerts", "Each one names the rule that produced it."
    )
    _alerts_panel(frame, risk, census)

    st.divider()
    left, right = st.columns([3, 2], gap="medium")
    with left:
        components.section_header("What the service suggests", "Evidence one click away.")
        _suggestions(frame, risk)
    with right:
        components.section_header("Forecast caveats", "Only what the data supports.")
        _alerts(frame, risk)

    if moment is not None:
        st.divider()
        _replay_comparison(frame, moment)

    st.divider()
    _summary_download(frame, risk, moment)



def _replay_comparison(frame: pd.DataFrame, moment: datetime) -> None:
    """The forecast issued at `moment` against what actually arrived afterwards.

    The point of replay. Everything above this panel is what the service would
    have said at the time; this is the only place that looks forward, and it says
    so in the heading, because a reader who confuses the two sides has learned the
    opposite of what the panel is for.
    """
    visits = replay.load_visits()
    if visits is None:
        return

    theme.section_header(
        "What actually happened next",
        "Hindsight, for comparison only. Nothing above this line used it.",
    )

    hours = min(len(frame), 24)
    actual = replay.actual_arrivals_after(visits, moment, hours)
    if actual.empty:
        theme.info_callout(
            "The recorded data ends before this window, so there is nothing to "
            "compare against. Pick an earlier moment.",
            level="watch",
            title="No comparison available.",
        )
        return

    predicted = frame.head(hours).reset_index(drop=True)
    merged = predicted.merge(actual, on="timestamp", how="inner")
    if merged.empty:
        theme.info_callout(
            "The forecast hours and the recorded hours do not line up, so no "
            "comparison is shown rather than a misleading one.",
            level="watch",
            title="No overlap.",
        )
        return

    error = (merged["yhat"] - merged["arrivals"]).abs()
    inside = (merged["arrivals"] >= merged["lower"]) & (
        merged["arrivals"] <= merged["upper"]
    )

    columns = st.columns(3, gap="small")
    with columns[0]:
        theme.kpi_card(
            "Arrivals forecast",
            theme.count(merged["yhat"].sum()),
            help=f"Total expected over the {len(merged)} hours after the chosen moment.",
        )
    with columns[1]:
        theme.kpi_card(
            "Arrivals that came",
            theme.count(merged["arrivals"].sum()),
            help="What the recorded data actually contains for those hours.",
        )
    with columns[2]:
        theme.kpi_card(
            "Hours inside the range",
            f"{int(inside.sum())} of {len(merged)}",
            help=(
                "How often the true count fell inside the forecast's range. Over "
                f"many windows this should be about {theme.percent(INTERVAL_COVERAGE, places=0)}."
            ),
        )
        share = inside.mean()
        theme.render_pill(
            "good" if share >= INTERVAL_COVERAGE - 0.1 else "watch",
            "About as often as advertised" if share >= INTERVAL_COVERAGE - 0.1
            else "Less often than advertised",
        )

    _comparison_chart(merged)
    st.caption(
        f"Average miss over these {len(merged)} hours: "
        f"{theme.decimal(error.mean())} arrivals an hour."
    )


def _comparison_chart(merged: pd.DataFrame) -> None:
    """Forecast against actual, in the two colours those concepts always use."""
    import altair as alt

    long = pd.concat(
        [
            pd.DataFrame(
                {
                    "timestamp": merged["timestamp"],
                    "arrivals": merged["yhat"],
                    "series": "Forecast at the time",
                }
            ),
            pd.DataFrame(
                {
                    "timestamp": merged["timestamp"],
                    "arrivals": merged["arrivals"],
                    "series": "What actually arrived",
                }
            ),
        ]
    )

    band = (
        alt.Chart(merged)
        .mark_area(opacity=0.2, color=theme.BAND)
        .encode(
            x=alt.X("timestamp:T", title=theme.axis("Time")),
            y=alt.Y("lower:Q", title=theme.axis("Arrivals", "per hour")),
            y2="upper:Q",
        )
    )
    lines = (
        alt.Chart(long)
        .mark_line(strokeWidth=2)
        .encode(
            x="timestamp:T",
            y="arrivals:Q",
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(
                    domain=["Forecast at the time", "What actually arrived"],
                    range=[theme.BLUE, theme.TEAL],
                ),
            ),
            strokeDash=alt.StrokeDash(
                "series:N",
                title=None,
                scale=alt.Scale(
                    domain=["Forecast at the time", "What actually arrived"],
                    range=[[1, 0], [4, 2]],
                ),
            ),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Hour"),
                alt.Tooltip("series:N", title=None),
                alt.Tooltip("arrivals:Q", title="Arrivals", format=".1f"),
            ],
        )
    )
    st.altair_chart(theme.style_chart(band + lines), use_container_width=True)
    theme.chart_caption(
        f"Two lines over {len(merged)} hours: the forecast issued at the chosen "
        f"moment (solid blue, totalling {theme.count(merged['yhat'].sum())} "
        f"arrivals) and what the recorded data actually contains (dashed teal, "
        f"totalling {theme.count(merged['arrivals'].sum())}). The shaded band is "
        "the forecast's range. The two lines are distinguished by dash pattern as "
        "well as colour."
    )


# -------------------------------------------------------------------- replay mode


def _replay_control() -> datetime | None:
    """The Shift replay picker. Returns the chosen moment, or None for the default.

    Off by default. On, the page shows what the service would have said at a
    moment in the recorded history, which is the only honest way to demonstrate a
    forecast without a live feed.
    """
    with st.expander("Shift replay — see a past moment", icon="🕰️"):
        st.caption(
            "Replay of historical data. Pick a moment and the page shows "
            "what the service would have known then, using only data recorded "
            "before it. Nothing here is a live feed."
        )
        on = st.toggle("Replay a past moment", key="cc_replay_on")
        if not on:
            return None

        visits = replay.load_visits()
        if visits is None:
            theme.info_callout(
                "Replay needs the processed visits table, which a completed "
                "analysis produces.",
                level="watch",
                title="Not available yet.",
            )
            return None

        window = replay.replay_window(visits, artifacts.test_period_start())
        day = st.date_input(
            "Date",
            value=window.start.date(),
            min_value=window.start.date(),
            max_value=window.end.date(),
            key="cc_replay_day",
            help=f"Any day in the {window.days}-day held-out period.",
        )
        hour = st.slider("Hour", 0, 23, 17, key="cc_replay_hour", help="24-hour clock.")
        moment = datetime.combine(day, datetime.min.time()) + timedelta(hours=int(hour))
        st.caption(f"Showing the service's view as at {moment:%a %d %b %Y, %H:%M}.")
        return moment


# ----------------------------------------------------------------------- the cards


def _typical_wait(moment: datetime | None) -> tuple[float | None, float | None, float | None]:
    """The usual mid-acuity wait, from the recorded period.

    Deliberately not called "current": there is no live feed, and a number labelled
    current that came from a saved table would be a lie. In replay mode the window
    is the hours *before* the chosen moment, so it stays leakage-safe.
    """
    visits = replay.load_visits()
    if visits is None:
        return None, None, None

    if moment is not None:
        recent = replay.before(visits, moment).tail(2000)
    else:
        recent = visits.tail(2000)

    levels = pd.to_numeric(recent["triage_level"], errors="coerce")
    waits = pd.to_numeric(recent.loc[levels == 3, "wait_minutes"], errors="coerce").dropna()
    if waits.empty:
        return None, None, None
    return (
        float(waits.median()),
        float(waits.quantile(0.25)),
        float(waits.quantile(0.75)),
    )


# ------------------------------------------------------------------ the judgement


def _congestion(frame: pd.DataFrame) -> dict[str, Any]:
    """Low / Moderate / High for the next few hours, with the rule it used.

    The rule is shown to the reader rather than hidden, because a risk level
    nobody can check is a risk level nobody should act on.
    """
    threshold = float(frame["yhat"].quantile(BUSY_PERCENTILE))
    window = frame.head(RISK_HORIZON_HOURS)
    busy = int((window["yhat"] >= threshold).sum())
    share = busy / max(len(window), 1)

    level, label = "good", "Low"
    for minimum, candidate_level, candidate_label in RISK_RULES:
        if share >= minimum:
            level, label = candidate_level, candidate_label
            break

    return {
        "level": level,
        "label": label,
        "busy_hours": busy,
        "threshold": threshold,
        "detail": f"{busy} of the next {len(window)} hours look busy",
        "rule": (
            "An hour counts as busy when expected arrivals reach the 80th "
            f"percentile of this forecast ({threshold:.1f} per hour). "
            f"High: more than 60% of the next {RISK_HORIZON_HOURS} hours are busy. "
            "Moderate: more than 30%. Low: fewer."
        ),
    }


def _suggestions(frame: pd.DataFrame, risk: dict[str, Any]) -> None:
    """Two or three plain sentences built from the staffing and scenario results."""
    staffing, error = api_client.staffing_recommendation()

    lines: list[str] = []
    peak = frame.loc[frame["yhat"].idxmax()]
    lines.append(
        f"Demand peaks around {theme.day_and_clock(peak['timestamp'])}, at roughly "
        f"{theme.decimal(peak['yhat'])} arrivals an hour."
    )

    if error:
        lines.append(
            "The staffing recommendation is not available right now, so there is "
            "nothing to suggest about the roster."
        )
    elif staffing and staffing.get("feasible"):
        lines.append(
            f"The cheapest roster that still meets the target "
            f"({staffing.get('target', 'the wait target')}) is "
            f"{staffing.get('description') or 'the plan shown on the Staffing page'}, "
            f"costing {theme.count(staffing.get('cost'))} a week."
        )
        if risk["level"] in ("watch", "act"):
            lines.append(
                "Given the busy hours above, check that the roster for those hours "
                "matches this plan before the shift starts."
            )
        else:
            lines.append(
                "The next few hours look ordinary, so no staffing change is "
                "suggested on the strength of this forecast."
            )
    else:
        lines.append(
            "No roster among the plans searched met the wait target, so the "
            "suggestion is to look at the target itself rather than the roster."
        )

    for line in lines:
        st.markdown(f"- {line}")

    with theme.technical_details("the evidence behind this"):
        if error:
            st.caption("The staffing recommendation could not be read.")
            st.code(error, language=None)
        st.caption("Congestion rule applied on this page:")
        st.code(risk["rule"], language=None)


def _alerts(frame: pd.DataFrame, risk: dict[str, Any]) -> None:
    """Only alerts the data on this page supports."""
    alerts: list[tuple[str, str]] = []

    busy = frame.head(RISK_HORIZON_HOURS)
    busy = busy.loc[busy["yhat"] >= risk["threshold"]]
    if not busy.empty:
        first = theme.clock(busy["timestamp"].iloc[0])
        last = theme.clock(busy["timestamp"].iloc[-1] + pd.Timedelta(hours=1))
        alerts.append(
            (
                "watch",
                f"Busy stretch expected {first}–{last}, above "
                f"{risk['threshold']:.1f} arrivals an hour.",
            )
        )

    quality = artifacts.forecast_quality()
    if quality and quality.get("beats_baseline_at_peak") is False:
        alerts.append(
            (
                "watch",
                "At peak hours this forecast is no better than simply repeating "
                "last week, so treat the peak numbers with extra caution.",
            )
        )

    widest = frame.assign(width=frame["upper"] - frame["lower"]).nlargest(1, "width")
    if not widest.empty:
        row = widest.iloc[0]
        if row["width"] > 2 * float((frame["upper"] - frame["lower"]).median()):
            alerts.append(
                (
                    "watch",
                    f"The forecast is unusually uncertain around "
                    f"{theme.day_and_clock(row['timestamp'])}.",
                )
            )

    if not alerts:
        theme.info_callout(
            "Nothing in this forecast crosses a threshold worth flagging.",
            level="good",
            title="No alerts.",
        )
        return

    for level, text in alerts:
        st.markdown(
            f"{theme.status_pill(level, theme.LEVELS[level]['word'])} {text}",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------- the chart


def _seasonal_naive(frame: pd.DataFrame) -> pd.DataFrame:
    """What "just repeat last week" would have predicted for the same hours.

    The baseline the forecast is measured against, drawn beside it so a reader can
    see the gap rather than being told about it. It is computed from arrivals one
    week before each forecast hour, which is data that already existed when the
    forecast was issued -- so putting it on the chart adds no hindsight.

    Returns an empty frame when the extract does not reach back a week from the
    forecast window, and the chart then simply omits the line.
    """
    visits = replay.load_visits()
    if visits is None or frame.empty:
        return pd.DataFrame(columns=["timestamp", "naive"])

    arrivals = visits["arrival_time"]
    hourly = arrivals.dt.floor("h").value_counts()
    rows = []
    for stamp in frame["timestamp"]:
        earlier = pd.Timestamp(stamp).floor("h") - pd.Timedelta(days=7)
        if earlier < arrivals.min():
            continue
        rows.append({"timestamp": stamp, "naive": float(hourly.get(earlier, 0))})
    return pd.DataFrame(rows)


def _forecast_chart(frame: pd.DataFrame) -> None:
    """Forecast line, interval band, the seasonal-naive baseline, busy hours."""
    import altair as alt

    threshold = float(frame["yhat"].quantile(BUSY_PERCENTILE))
    plot = frame.assign(busy=frame["yhat"] >= threshold)
    naive = _seasonal_naive(frame)

    band = (
        alt.Chart(plot)
        .mark_area(opacity=0.25, color=theme.BAND)
        .encode(
            x=alt.X("timestamp:T", title=theme.axis("Time")),
            y=alt.Y("lower:Q", title=theme.axis("Arrivals", "per hour")),
            y2="upper:Q",
        )
    )
    line = (
        alt.Chart(plot)
        .mark_line(color=theme.BLUE, strokeWidth=2)
        .encode(x="timestamp:T", y="yhat:Q")
    )
    marks = (
        alt.Chart(plot.loc[plot["busy"]])
        .mark_point(color=theme.AMBER, size=45, filled=True, shape="triangle-up")
        .encode(
            x="timestamp:T",
            y="yhat:Q",
            tooltip=[
                alt.Tooltip("timestamp:T", title="Hour"),
                alt.Tooltip("yhat:Q", title="Expected", format=".1f"),
            ],
        )
    )

    layers = band + line
    naive_note = ""
    if not naive.empty:
        baseline = (
            alt.Chart(naive)
            .mark_line(color=theme.GREY, strokeWidth=1.5, strokeDash=[5, 4])
            .encode(
                x="timestamp:T",
                y=alt.Y("naive:Q"),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Hour"),
                    alt.Tooltip("naive:Q", title="Same hour last week", format=".0f"),
                ],
            )
        )
        layers = layers + baseline
        naive_note = (
            " The grey dashed line is what repeating the same hour last week would "
            "have predicted, which is the baseline this forecast is measured against."
        )

    st.altair_chart(
        theme.style_chart(layers + marks), use_container_width=True
    )
    theme.chart_caption(
        f"Expected arrivals per hour over the next {len(frame)} hours. The blue line "
        f"is the forecast, the shaded band its "
        f"{theme.percent(INTERVAL_COVERAGE, places=0)} range, and "
        f"amber triangles mark the {int(plot['busy'].sum())} hours expected to reach "
        f"{threshold:.1f} arrivals an hour or more. The busiest hour is "
        f"{theme.day_and_clock(frame.loc[frame['yhat'].idxmax(), 'timestamp'])}."
        f"{naive_note}"
    )


# ------------------------------------------------------------------- the download


def _summary_download(
    frame: pd.DataFrame, risk: dict[str, Any], moment: datetime | None
) -> None:
    """Offer the one-page summary as a download."""
    st.download_button(
        "Download shift summary",
        data=_summary_markdown(frame, risk, moment).encode("utf-8"),
        file_name="ed_shift_summary.md",
        mime="text/markdown",
        help="A one-page markdown summary of this page.",
    )


def _summary_markdown(
    frame: pd.DataFrame, risk: dict[str, Any], moment: datetime | None
) -> str:
    """A one-page summary a reader can take into a handover meeting.

    Separate from the button so its wording can be tested: AppTest does not model
    `st.download_button`, so the only way to check that this note is readable --
    and that no internal identifier leaked into it -- is to build it directly.
    """
    next_24 = frame.head(24)
    peak = frame.loc[frame["yhat"].idxmax()]
    issued = "the end of the recorded history" if moment is None else f"{moment:%d %b %Y %H:%M}"

    lines = [
        "# ED Operations — shift summary",
        "",
        "**Demo data (synthetic).** Decision support, not clinical advice.",
        "",
        f"- Forecast issued from: {issued}",
        f"- Expected arrivals, next 24 h: {theme.count(next_24['yhat'].sum())} "
        f"(range {theme.count(next_24['lower'].sum())}–{theme.count(next_24['upper'].sum())})",
        f"- Busiest hour expected: {theme.day_and_clock(peak['timestamp'])} "
        f"(about {theme.decimal(peak['yhat'])} an hour)",
        f"- Congestion risk, next {RISK_HORIZON_HOURS} h: {risk['label']} "
        f"({risk['detail']})",
        "",
        "## How the risk level was decided",
        "",
        risk["rule"],
        "",
        "## Caveats",
        "",
        "- The data is synthetic, calibrated to published benchmarks. It is not a "
        "real department.",
        "- Ranges are 90% intervals and have been measured to cover slightly less "
        "than that in testing.",
        "- Forecast accuracy is weaker at peak hours than overall.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------- data shape


def _as_frame(forecast: dict[str, Any]) -> pd.DataFrame:
    """The service's forecast as a frame, sorted, with timestamps parsed."""
    points = (forecast or {}).get("points") or []
    if not points:
        return pd.DataFrame(columns=["timestamp", "yhat", "lower", "upper"])
    frame = pd.DataFrame(points)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame.sort_values("timestamp").reset_index(drop=True)


# ======================================================================= the rows
#
# Four bands, in the order a charge nurse reads them: where the department is now,
# what is coming, where the waiting actually is, and what needs a decision.


def _service_overview(moment: datetime | None) -> dict[str, Any] | None:
    """The same figures, computed by the service instead of here.

    Asked for rather than re-derived, because a figure computed in two places
    eventually disagrees with itself and the service is the copy that has to be
    right -- other callers read it too. The local reconstruction stays as the
    fallback: this dashboard is useful against saved tables with the service down,
    and losing the whole top row to a connection error would be a worse page than
    a slightly slower one.

    Returns None when the service cannot answer, and the caller computes locally.
    """
    at = None
    if moment is not None:
        at = pd.Timestamp(moment).isoformat()
    payload, error = api_client.analytics_overview(at)
    if error or not isinstance(payload, dict) or "kpis" not in payload:
        return None
    return payload


def _census(moment: datetime | None) -> dict[str, Any]:
    """Where the department stands at `moment`, from earlier arrivals only.

    Every figure here is computed from the truncated visits table, so nothing on
    the top row of this page can have been influenced by a patient who had not yet
    walked in. Returns an empty mapping when the extract is not available, and the
    cards then say so rather than printing a zero that looks like good news.
    """
    visits = replay.load_visits_detailed()
    if visits is None:
        return {}

    cutoff = pd.Timestamp(moment) if moment is not None else _default_moment(visits)
    snapshot = replay.knowledge_at(visits, cutoff)
    past = replay.before(visits, cutoff)
    if past.empty:
        return {"at": cutoff, "waiting": 0, "in_treatment": 0}

    left_waiting_at = past["seen_time"].fillna(past["departure_time"])
    waiting = past.loc[left_waiting_at.isna() | (left_waiting_at > cutoff)]
    longest = (
        float(((cutoff - waiting["arrival_time"]).dt.total_seconds() / 60).max())
        if not waiting.empty
        else 0.0
    )

    # "Today" is the calendar day of the chosen moment, up to the moment itself.
    same_day = past.loc[past["arrival_time"].dt.date == cutoff.date()]
    waits_today = pd.to_numeric(same_day.get("wait_minutes"), errors="coerce").dropna()
    levels_today = pd.to_numeric(same_day.get("triage_level"), errors="coerce")

    seen_or_left = same_day
    lwbs = pd.Series(dtype=bool)
    if "is_lwbs" in same_day.columns:
        lwbs = same_day["is_lwbs"].fillna(False).astype(bool)

    return {
        "at": cutoff,
        "waiting": int(snapshot.n_waiting),
        "in_treatment": int(snapshot.n_in_treatment),
        "longest_wait": longest,
        "median_wait_today": float(waits_today.median()) if not waits_today.empty else None,
        "lwbs_rate_today": (
            float(lwbs.sum() / len(seen_or_left)) if len(seen_or_left) and not lwbs.empty
            else None
        ),
        "waiting_levels": pd.to_numeric(
            waiting.get("triage_level"), errors="coerce"
        ).dropna(),
        "levels_today": levels_today.dropna(),
        "waits_by_level_today": (
            same_day.assign(
                level=levels_today,
                wait=pd.to_numeric(same_day.get("wait_minutes"), errors="coerce"),
            )
            .dropna(subset=["level", "wait"])
        ),
    }


def _default_moment(visits: pd.DataFrame) -> pd.Timestamp:
    """The moment to describe when the reader has not chosen one.

    The middle of the held-out period, not its end. At the very last timestamp the
    waiting room is empty by construction -- everyone who arrived earlier has been
    seen and the final arrival is excluded by the strict "before the cutoff" rule --
    so an end-of-extract default would open this page on a row of zeros and look
    like a broken dashboard rather than a quiet one.

    The midpoint is representative rather than flattering: it is not chosen for
    being busy, and the reader can move the replay control to any other instant.
    """
    window = replay.replay_window(visits, artifacts.test_period_start())
    if window.end <= window.start:
        return pd.Timestamp(visits["arrival_time"].max())
    return window.start + (window.end - window.start) / 2


def _week_ago(moment: datetime | None) -> dict[str, Any]:
    """The same figures a week earlier, for the deltas.

    A week back rather than an hour back because an emergency department's load is
    strongly weekly: comparing 18:00 Friday with 17:00 Friday says almost nothing,
    while comparing it with 18:00 the previous Friday is the comparison a charge
    nurse actually makes. When a week earlier falls outside the extract the caller
    falls back to the overall average and **says so in the card**, because a delta
    whose baseline is unstated is worse than no delta.
    """
    if moment is None:
        return {}
    earlier = pd.Timestamp(moment) - pd.Timedelta(days=7)
    visits = replay.load_visits_detailed()
    if visits is None or earlier < pd.Timestamp(visits["arrival_time"].min()):
        return {}
    return _census(earlier)


def _delta(now: float | None, before: float | None) -> tuple[str | None, str]:
    """A formatted delta and its direction, or nothing when it cannot be computed.

    The direction is returned separately from the sign because which way is bad
    depends on the metric, and the card colours itself from the direction rather
    than guessing.
    """
    if now is None or before is None:
        return None, "flat"
    change = now - before
    if abs(change) < 0.5:
        return "no change", "flat"
    return f"{change:+,.0f}", "up" if change > 0 else "down"


def _kpi_row(frame: pd.DataFrame, risk: dict[str, Any], moment: datetime | None) -> None:
    """Row 1: six numbers that say where the department is.

    Each card names what its delta is measured against, so the comparison is never
    left implicit. Where a week earlier is inside the extract the delta is against
    the same hour last week; where it is not, the card says "vs overall average"
    instead of silently changing meaning.
    """
    census = _census(moment)
    if not census:
        theme.empty_state(
            "The extract is not available",
            "These figures come from the saved visits table, which the analysis has "
            "not produced yet.",
            icon="🏥",
        )
        return

    previous = _week_ago(moment)
    comparison = "vs same hour last week" if previous else "vs overall average"
    stamp = theme.clock(census["at"], "%a %d %b %H:%M")

    row = st.columns(3, gap="small")
    second = st.columns(3, gap="small")

    waiting = census["waiting"]
    delta, direction = _delta(waiting, previous.get("waiting"))
    with row[0]:
        components.kpi_card(
            "Patients waiting now",
            theme.count(waiting),
            delta=delta,
            delta_direction=direction,
            status="act" if waiting > 40 else "watch" if waiting > 20 else "good",
            sparkline=_recent_waiting(moment),
            comparison=comparison if delta else None,
            help=(
                "Arrived and not yet seen, counted at the replay moment from "
                "patients who had already arrived."
            ),
        )

    treating = census["in_treatment"]
    delta, direction = _delta(treating, previous.get("in_treatment"))
    with row[1]:
        components.kpi_card(
            "In treatment",
            theme.count(treating),
            delta=delta,
            # More patients in treatment is the department working, not a problem,
            # so this one is never coloured as bad.
            delta_direction="flat",
            status="good",
            status_text="Capacity in use",
            comparison=comparison if delta else None,
            help="Seen by a clinician and not yet departed.",
        )

    longest = census.get("longest_wait")
    delta, direction = _delta(longest, previous.get("longest_wait"))
    with row[2]:
        components.kpi_card(
            "Longest wait",
            theme.minutes(longest) if longest is not None else "no data",
            delta=delta,
            delta_direction=direction,
            status=(
                "act" if (longest or 0) > alerts.WAIT_TARGET_MINUTES * 3
                else "watch" if (longest or 0) > alerts.WAIT_TARGET_MINUTES
                else "good"
            ),
            comparison=comparison if delta else None,
            help=(
                "The single longest outstanding wait. One patient is enough to "
                "matter here, which an average would hide."
            ),
        )

    median_today = census.get("median_wait_today")
    delta, direction = _delta(median_today, previous.get("median_wait_today"))
    with second[0]:
        components.kpi_card(
            f"Median door-to-doctor, {theme.clock(census['at'], '%a')}",
            theme.minutes(median_today) if median_today is not None else "no data",
            delta=delta,
            delta_direction=direction,
            status=(
                "good" if (median_today or 0) <= alerts.WAIT_TARGET_MINUTES
                else "act"
            ),
            status_text=(
                f"Target {alerts.WAIT_TARGET_MINUTES} min"
                if median_today is not None else None
            ),
            comparison=comparison if delta else None,
            help=(
                "Middle wait for patients who arrived on the replay moment's own "
                f"day, up to that moment. The target is "
                f"{alerts.WAIT_TARGET_MINUTES} minutes."
            ),
        )

    lwbs = census.get("lwbs_rate_today")
    with second[1]:
        components.kpi_card(
            "Left without being seen",
            theme.percent(lwbs) if lwbs is not None else "no data",
            status=("act" if (lwbs or 0) > 0.04 else "good"),
            status_text="Above 4.0%" if (lwbs or 0) > 0.04 else "Within 4.0%",
            help=(
                "Share of that day's arrivals who left before a clinician saw "
                "them. 4.0% is the rate the simulation was validated against."
            ),
        )

    with second[2]:
        components.kpi_card(
            f"Congestion risk, next {RISK_HORIZON_HOURS} h",
            str(risk["label"]),
            status={"High": "act", "Moderate": "watch"}.get(risk["label"], "good"),
            status_text=str(risk["label"]),
            help=str(risk["rule"]),
            comparison="from the arrivals forecast",
        )

    st.caption(f"Figures as of {stamp}, reconstructed from the recorded extract.")


def _recent_waiting(moment: datetime | None, hours: int = 12) -> list[float]:
    """How many were waiting over the last few hours, for the card's sparkline.

    Context for the number above it, not a chart: twelve points, no axes, and no
    value a reader is expected to read off.
    """
    visits = replay.load_visits_detailed()
    if visits is None:
        return []
    cutoff = pd.Timestamp(moment) if moment is not None else _default_moment(visits)
    points: list[float] = []
    for back in range(hours, 0, -1):
        at = cutoff - pd.Timedelta(hours=back)
        points.append(float(replay.knowledge_at(visits, at).n_waiting))
    return points


# ----------------------------------------------------------------- row 2: mix


def _acuity_mix(census: dict[str, Any]) -> None:
    """Who is waiting, by acuity, in the palette the rest of the app uses."""
    levels = census.get("waiting_levels")
    if levels is None or len(levels) == 0:
        theme.empty_state(
            "Nobody was waiting at this moment",
            "The waiting room was empty, so there is no mix to show.",
            icon="🧑‍⚕️",
        )
        return

    import altair as alt

    counts = (
        levels.astype(int)
        .value_counts()
        .rename_axis("level")
        .reset_index(name="patients")
        .sort_values("level")
    )
    counts["Acuity"] = counts["level"].map(theme.ESI_LABELS).fillna("Not recorded")
    counts["share"] = counts["patients"] / counts["patients"].sum()

    chart = (
        alt.Chart(counts)
        .mark_arc(innerRadius=55, stroke=None)
        .encode(
            theta=alt.Theta("patients:Q", stack=True),
            color=alt.Color(
                "Acuity:N",
                scale=alt.Scale(
                    domain=list(theme.ESI_LABELS.values()),
                    range=list(theme.ESI_SCALE),
                ),
                legend=alt.Legend(title="Acuity", orient="right"),
            ),
            tooltip=[
                "Acuity",
                alt.Tooltip("patients:Q", title="Waiting"),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
            ],
        )
        .properties(height=theme.CHART_HEIGHT)
    )
    st.altair_chart(theme.style_chart(chart), use_container_width=True)
    theme.chart_caption(
        "A ring divided by acuity, in the same colours the triage chips use. The "
        "largest segment is where the queue actually is, which is usually not where "
        "the clinical risk is."
    )


# ------------------------------------------------- row 3: waits and congestion


def _waits_by_acuity(census: dict[str, Any]) -> None:
    """Middle and 90th-percentile wait per acuity, against the target line."""
    frame = census.get("waits_by_level_today")
    if frame is None or frame.empty:
        theme.empty_state(
            "No completed waits on this day yet",
            "Waits appear here once patients from this day have been seen.",
            icon="⏱️",
        )
        return

    import altair as alt

    grouped = (
        frame.groupby(frame["level"].astype(int))["wait"]
        .agg(median="median", p90=lambda values: values.quantile(0.9))
        .reset_index()
        .rename(columns={"level": "level"})
    )
    grouped["Acuity"] = grouped["level"].map(theme.ESI_LABELS).fillna("Not recorded")
    long = grouped.melt(
        id_vars=["level", "Acuity"],
        value_vars=["median", "p90"],
        var_name="measure",
        value_name="minutes",
    )
    long["measure"] = long["measure"].map(
        {"median": "Middle wait", "p90": "9 in 10 waited less than"}
    )

    bars = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("Acuity:N", title=None, sort=list(theme.ESI_LABELS.values())),
            y=alt.Y("minutes:Q", title=theme.axis("Wait", "minutes")),
            color=alt.Color(
                "measure:N",
                title=None,
                scale=alt.Scale(
                    domain=["Middle wait", "9 in 10 waited less than"],
                    range=[theme.TEAL, theme.BAND],
                ),
            ),
            xOffset="measure:N",
            tooltip=[
                "Acuity",
                "measure",
                alt.Tooltip("minutes:Q", title="Minutes", format=".0f"),
            ],
        )
    )
    target = (
        alt.Chart(pd.DataFrame({"minutes": [float(alerts.WAIT_TARGET_MINUTES)]}))
        .mark_rule(strokeDash=[6, 4], size=2, color=theme.RED)
        .encode(y="minutes:Q")
    )
    st.altair_chart(
        theme.style_chart((bars + target).properties(height=theme.CHART_HEIGHT)),
        use_container_width=True,
    )
    theme.chart_caption(
        f"Two bars per acuity -- the middle wait and the wait nine in ten patients "
        f"came in under -- against the dashed {alerts.WAIT_TARGET_MINUTES}-minute "
        "target. The dashes are the line that matters: a middle wait inside target "
        "with a 90th percentile far outside it means a few people are waiting a "
        "very long time."
    )


def _congestion_heatmap() -> None:
    """Arrivals by day of week and hour, over the whole extract."""
    visits = replay.load_visits()
    if visits is None:
        return

    import altair as alt

    frame = pd.DataFrame(
        {
            "day": visits["arrival_time"].dt.day_name(),
            "hour": visits["arrival_time"].dt.hour,
        }
    )
    counts = (
        frame.groupby(["day", "hour"]).size().reset_index(name="arrivals")
    )
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    per_week = counts["arrivals"].sum() / max(len(visits) / len(counts), 1)
    counts["per day"] = counts["arrivals"] / max(
        visits["arrival_time"].dt.date.nunique() / 7, 1
    )

    chart = (
        alt.Chart(counts)
        .mark_rect()
        .encode(
            x=alt.X("hour:O", title=theme.axis("Hour of day")),
            y=alt.Y("day:N", title=None, sort=days),
            color=alt.Color(
                "per day:Q",
                title="Arrivals",
                scale=alt.Scale(scheme="teals"),
                legend=alt.Legend(orient="right"),
            ),
            tooltip=[
                "day",
                alt.Tooltip("hour:O", title="Hour"),
                alt.Tooltip("per day:Q", title="Arrivals", format=".1f"),
            ],
        )
        .properties(height=theme.CHART_HEIGHT)
    )
    st.altair_chart(theme.style_chart(chart), use_container_width=True)
    theme.chart_caption(
        "A grid of day against hour, darker where more patients arrive. It is the "
        "shape the roster has to cover: the dark band runs across the afternoon and "
        "early evening every day of the week, and is not a weekend effect."
    )


# ------------------------------------------------------------- row 4: the alerts


def _alerts_panel(
    frame: pd.DataFrame, risk: dict[str, Any], census: dict[str, Any]
) -> None:
    """Row 4: what fired, why it fired, and when.

    The rules come from `app/alerts.py` so this panel cannot show a judgement the
    System status page does not explain. Nothing here is invented on the page.
    """
    busy = frame.head(RISK_HORIZON_HOURS)
    busy = busy.loc[busy["yhat"] >= risk["threshold"]]
    busy_hours = [theme.day_and_clock(value) for value in busy["timestamp"]]

    waits = census.get("waits_by_level_today")
    waits_by_acuity: dict[int, float] = {}
    if waits is not None and not waits.empty:
        waits_by_acuity = {
            int(level): float(group["wait"].median())
            for level, group in waits.groupby(waits["level"].astype(int))
        }

    stamp = (
        theme.clock(census["at"], "%Y-%m-%d %H:%M") if census.get("at") is not None else None
    )

    # The service evaluates the same rules over the same figures. Prefer its answer
    # so the dashboard and any other caller cannot disagree about whether an alert
    # is firing; fall back to the local rules when it cannot be reached.
    served = _service_overview(census.get("at"))
    if served and served.get("alerts"):
        for item in served["alerts"]:
            components.alert_banner(
                str(item.get("title", "")),
                str(item.get("detail") or "") or None,
                level=str(item.get("severity", "info")),
                rule=str(item.get("rule", "")) or None,
                timestamp=str(item.get("at") or "") or None,
            )
        st.caption(
            "Evaluated by the analysis service. Every rule, its threshold and what "
            "it means are listed on the System status page."
        )
        return

    fired = alerts.evaluate(
        waits_by_acuity=waits_by_acuity or None,
        longest_wait_minutes=census.get("longest_wait"),
        lwbs_rate=census.get("lwbs_rate_today"),
        busy_hours=busy_hours,
        busy_threshold=float(risk["threshold"]),
        at=stamp,
    )

    if not fired:
        theme.empty_state(
            "No rule could be evaluated",
            "The figures these rules read are not available, so this panel is "
            "empty rather than claiming all is well.",
            icon="🔔",
        )
        return

    for alert in fired:
        components.alert_banner(
            alert.title,
            alert.detail,
            level=alert.level,
            rule=alert.rule,
            timestamp=alert.at,
        )
    st.caption(
        "Every rule, its threshold and what it means are listed on the System "
        "status page. An alert that cannot be traced to a rule is not shown."
    )
