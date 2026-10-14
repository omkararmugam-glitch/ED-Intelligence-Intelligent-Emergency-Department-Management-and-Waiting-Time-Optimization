"""Demand forecast: how many are coming, and how much to trust it.

The reliability panel is the point of this page as much as the chart is. A forecast
without its measured error is a number someone will plan a roster around, so the
backtest comparison against "just repeat last week" is on the page rather than
buried, including the part that does not flatter the model: at peak hours it does
not beat that baseline.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app import api_client, artifacts, components, theme
from app.config import FORECAST_HORIZONS, INTERVAL_COVERAGE
from app.views import _shared


def render() -> None:
    """Draw the Demand forecast page."""
    components.page_header(
        "Demand forecast",
        "Expected arrivals per hour, with the range around each estimate.",
        breadcrumb=("Planning", "Demand forecast"),
        icon_name="forecast",
    )

    horizon = st.radio(
        "How far ahead",
        FORECAST_HORIZONS,
        index=1,
        horizontal=True,
        format_func=lambda hours: f"{hours} hours",
        key="forecast_horizon",
        help="Beyond 72 hours the weekly pattern the model leans on runs out.",
    )

    forecast, ok = _shared.load(
        lambda: api_client.forecast(int(horizon)),
        spinner="Calculating forecast...",
        error_title="The forecast did not load",
        retry_key="fc_retry",
        placeholder_height=110,
        placeholder_rows=2,
    )
    if not ok:
        return

    frame = _as_frame(forecast)
    if frame.empty:
        theme.empty_state(
            "No hours came back",
            "The service answered but the forecast was empty. Run the analysis once "
            "to produce an arrivals model.",
            icon="📈",
        )
        return

    _headline(frame)
    st.divider()

    theme.section_header(
        "Hour by hour",
        f"The band is the {theme.percent(INTERVAL_COVERAGE, places=0)} range: the true "
        "count should fall "
        "inside it about that often.",
    )
    _chart(frame)

    st.divider()
    theme.section_header("How reliable is this?", "Measured on held-out data.")
    _reliability()

    st.divider()
    theme.section_header("The numbers", "One row per hour.")
    _table(frame, int(horizon))




def _headline(frame: pd.DataFrame) -> None:
    """Three cards: total, busiest hour, quietest hour."""
    peak = frame.loc[frame["yhat"].idxmax()]
    trough = frame.loc[frame["yhat"].idxmin()]

    columns = st.columns(3, gap="small")
    with columns[0]:
        theme.kpi_card(
            f"Expected arrivals, next {len(frame)} h",
            theme.count(frame["yhat"].sum()),
            help="The hourly forecast added up across the whole window shown.",
            sub=f"Range {theme.count(frame['lower'].sum())}–{theme.count(frame['upper'].sum())}",
        )
    with columns[1]:
        theme.kpi_card(
            "Busiest hour",
            theme.day_and_clock(peak["timestamp"]),
            help="The hour with the highest expected arrivals in this window.",
            sub=f"About {theme.decimal(peak['yhat'])} an hour",
        )
    with columns[2]:
        theme.kpi_card(
            "Quietest hour",
            theme.day_and_clock(trough["timestamp"]),
            help="The hour with the lowest expected arrivals in this window.",
            sub=f"About {theme.decimal(trough['yhat'])} an hour",
        )


def _chart(frame: pd.DataFrame) -> None:
    """Forecast line with its interval band."""
    import altair as alt

    band = (
        alt.Chart(frame)
        .mark_area(opacity=0.25, color=theme.BAND)
        .encode(
            x=alt.X("timestamp:T", title=theme.axis("Time")),
            y=alt.Y("lower:Q", title=theme.axis("Arrivals", "per hour")),
            y2="upper:Q",
        )
    )
    line = (
        alt.Chart(frame)
        .mark_line(color=theme.BLUE, strokeWidth=2)
        .encode(
            x="timestamp:T",
            y="yhat:Q",
            tooltip=[
                alt.Tooltip("timestamp:T", title="Hour"),
                alt.Tooltip("yhat:Q", title="Expected", format=".1f"),
                alt.Tooltip("lower:Q", title="Low", format=".1f"),
                alt.Tooltip("upper:Q", title="High", format=".1f"),
            ],
        )
    )
    st.altair_chart(theme.style_chart(band + line), use_container_width=True)

    peak = frame.loc[frame["yhat"].idxmax()]
    theme.chart_caption(
        f"Expected arrivals per hour across {len(frame)} hours, from "
        f"{theme.day_and_clock(frame['timestamp'].iloc[0])} to "
        f"{theme.day_and_clock(frame['timestamp'].iloc[-1])}. The line averages "
        f"{theme.decimal(frame['yhat'].mean())} arrivals an hour and peaks at "
        f"{theme.decimal(peak['yhat'])} around {theme.day_and_clock(peak['timestamp'])}. "
        "The shaded band shows the range around each hour."
    )


def _reliability() -> None:
    """The backtest, against repeating last week.

    Every number here is read from `artifacts/tables/forecast_accuracy.csv`; none
    is computed on this page, and none is invented.
    """
    quality = artifacts.forecast_quality()
    if quality is None:
        _shared.needs_analysis("reliability measurements")
        return

    overall_model = quality["overall_model_mae"]
    overall_naive = quality["overall_naive_mae"]
    peak_model = quality["peak_model_mae"]
    peak_naive = quality["peak_naive_mae"]

    columns = st.columns(2, gap="medium")
    with columns[0]:
        theme.kpi_card(
            "Typical error, all hours",
            theme.per_hour(overall_model),
            help=(
                "Mean absolute error on held-out data: on average the forecast is "
                "out by this many arrivals in an hour."
            ),
            sub=f"Repeating last week: {theme.per_hour(overall_naive)}",
        )
        if quality["beats_baseline_overall"]:
            theme.render_pill("good", "Better than repeating last week")
        else:
            theme.render_pill("watch", "No better than repeating last week")

    with columns[1]:
        theme.kpi_card(
            "Typical error, peak hours",
            theme.per_hour(peak_model),
            help=(
                "The same measurement, restricted to the busiest hours of the day "
                "-- the hours a roster decision actually turns on."
            ),
            sub=f"Repeating last week: {theme.per_hour(peak_naive)}",
        )
        if quality["beats_baseline_at_peak"]:
            theme.render_pill("good", "Better than repeating last week")
        else:
            theme.render_pill("watch", "No better than repeating last week")

    if quality["beats_baseline_at_peak"] is False:
        theme.info_callout(
            "Across all hours the model is more accurate than simply repeating last "
            f"week ({theme.decimal(overall_model)} versus "
            f"{theme.decimal(overall_naive)} arrivals an hour). At peak hours it is "
            f"not ({theme.decimal(peak_model)} versus {theme.decimal(peak_naive)}). "
            "Peak hours are the ones a staffing decision turns on, so treat the "
            "peak of this forecast as a rough guide and weigh your own experience "
            "of the department alongside it.",
            level="watch",
            title="Read the peak hours with caution.",
        )

    with theme.technical_details():
        st.caption(
            "Mean absolute error by segment and model, from the forecast backtest "
            "."
        )
        raw = artifacts.table("forecast_accuracy.csv")
        if raw is not None:
            st.dataframe(raw, use_container_width=True, hide_index=True)
        horizons = artifacts.table("forecast_by_horizon.csv")
        if horizons is not None:
            st.caption("Error by how far ahead the forecast was issued:")
            st.dataframe(
                horizons[["horizon_band", "model_mae", "naive_mae"]],
                use_container_width=True,
                hide_index=True,
            )


def _display_table(frame: pd.DataFrame) -> pd.DataFrame:
    """The forecast as a reader-facing table.

    Separate from the rendering so the export can be tested: AppTest does not
    model `st.download_button`, so checking the frame is the only way to know a
    raw column name never reaches someone's spreadsheet.
    """
    return pd.DataFrame(
        {
            "Hour": frame["timestamp"].dt.strftime("%a %d %b, %H:%M"),
            "Expected arrivals": frame["yhat"].round(1),
            "Low end of range": frame["lower"].round(1),
            "High end of range": frame["upper"].round(1),
        }
    )


def _table(frame: pd.DataFrame, horizon: int) -> None:
    """The hour-by-hour table, with a download."""
    display = _display_table(frame)
    st.dataframe(display, use_container_width=True, hide_index=True, height=320)

    st.download_button(
        "Download as CSV",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name=f"ed_forecast_{horizon}h.csv",
        mime="text/csv",
        help="The table above, for a spreadsheet.",
    )


def _as_frame(forecast: dict[str, Any]) -> pd.DataFrame:
    """The service's forecast as a frame, sorted, with timestamps parsed."""
    points = (forecast or {}).get("points") or []
    if not points:
        return pd.DataFrame(columns=["timestamp", "yhat", "lower", "upper"])
    frame = pd.DataFrame(points)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame.sort_values("timestamp").reset_index(drop=True)
