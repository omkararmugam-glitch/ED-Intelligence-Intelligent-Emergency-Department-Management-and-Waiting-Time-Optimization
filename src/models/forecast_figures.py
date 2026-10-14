"""Tables and figures for the arrival forecast.

Same conventions as `src/models/figures.py`: validated categorical hues in fixed order,
one measure per axis, legends always present, recessive grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.io import save_figure
from src.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_COLOUR = "#2a78d6"
NAIVE_COLOUR = "#eb6834"
ACTUAL_COLOUR = "#0b0b0b"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#6b6b6b"
GRID = "#e3e3e0"
REFERENCE = "#d03b3b"

#: Metrics carried into the summary table, in reporting order.
REPORTED_METRICS = ("mae", "rmse", "mape", "smape", "wape", "bias", "n")


def forecast_summary_table(summary: dict[str, Any]) -> pd.DataFrame:
    """Flatten the backtest summary into one row per (segment, model)."""
    rows = []
    for segment in ("overall", "peak_hours", "off_peak_hours"):
        for model_name in ("model", "seasonal_naive"):
            metrics = summary[segment][model_name]
            rows.append(
                {
                    "segment": segment,
                    "model": "lightgbm" if model_name == "model" else "seasonal_naive",
                    **{
                        key: round(float(metrics[key]), 4)
                        for key in REPORTED_METRICS
                        if key in metrics
                    },
                    "n_zero_truth_hours": metrics.get("n_zero_truth_hours"),
                }
            )
    return pd.DataFrame(rows)


def forecast_coverage_table(coverage: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    """Interval coverage overall and per hour-of-day band."""
    nominal = float(config["conformal"]["coverage"])
    rows = [
        {
            "hour_band": "overall",
            "n": coverage["overall"]["n"],
            "nominal_coverage": nominal,
            "empirical_coverage": round(coverage["overall"]["coverage"], 4),
            "coverage_gap": round(coverage["overall"]["coverage_gap"], 4),
            "mean_width_arrivals": round(coverage["overall"]["mean_width"], 3),
        }
    ]
    for band, stats in sorted(coverage.get("by_group", {}).items()):
        rows.append(
            {
                "hour_band": band,
                "n": stats["n"],
                "nominal_coverage": nominal,
                "empirical_coverage": round(stats["coverage"], 4),
                "coverage_gap": round(stats["coverage_gap"], 4),
                "mean_width_arrivals": round(stats["mean_width"], 3),
            }
        )
    return pd.DataFrame(rows)


def _style(axes: plt.Axes, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    """Shared recessive styling."""
    axes.set_facecolor(SURFACE)
    axes.grid(color=GRID, linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID)
    axes.spines["left"].set_color(GRID)
    axes.tick_params(colors=INK_MUTED, labelsize=8.5, length=0)
    if xlabel:
        axes.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    if ylabel:
        axes.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    if title:
        axes.set_title(title, color=INK, fontsize=11, loc="left", pad=10)


def plot_forecast_window(issued: pd.DataFrame, figures_dir: Path) -> Path:
    """The issued forecast with its conformal band."""
    figure, axes = plt.subplots(figsize=(11.0, 3.8), facecolor=SURFACE)

    axes.fill_between(
        issued["timestamp"], issued["lower"], issued["upper"],
        color=MODEL_COLOUR, alpha=0.18, zorder=2, label="90% conformal interval",
    )
    axes.plot(
        issued["timestamp"], issued["yhat"], color=MODEL_COLOUR, linewidth=2,
        zorder=4, label="forecast",
    )
    _style(
        axes,
        xlabel="",
        ylabel="arrivals per hour",
        title=f"Issued forecast: {len(issued)} hours from {issued['timestamp'].min():%Y-%m-%d %H:%M}",
    )
    axes.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK)
    figure.autofmt_xdate()
    return save_figure(figure, figures_dir / "forecast_window.png")


def plot_backtest(predictions: pd.DataFrame, figures_dir: Path) -> Path:
    """Actual against forecast for the most recent backtest fold."""
    fold = predictions["fold"].max()
    subset = predictions[predictions["fold"] == fold].sort_values("timestamp")

    figure, axes = plt.subplots(figsize=(11.0, 3.8), facecolor=SURFACE)
    if subset["lower"].notna().any():
        axes.fill_between(
            subset["timestamp"], subset["lower"], subset["upper"],
            color=MODEL_COLOUR, alpha=0.16, zorder=2, label="90% conformal interval",
        )
    axes.plot(
        subset["timestamp"], subset["actual"], color=ACTUAL_COLOUR, linewidth=2,
        zorder=5, label="actual",
    )
    axes.plot(
        subset["timestamp"], subset["yhat"], color=MODEL_COLOUR, linewidth=2,
        zorder=4, label="LightGBM",
    )
    axes.plot(
        subset["timestamp"], subset["baseline"], color=NAIVE_COLOUR, linewidth=1.6,
        linestyle="--", zorder=3, label="seasonal naive",
    )
    _style(
        axes,
        ylabel="arrivals per hour",
        title=f"Rolling-origin backtest, final fold (origin {subset['origin'].iloc[0]:%Y-%m-%d})",
    )
    axes.legend(frameon=False, fontsize=9, ncol=4, loc="upper right", labelcolor=INK)
    figure.autofmt_xdate()
    return save_figure(figure, figures_dir / "forecast_backtest.png")


def plot_error_by_horizon(summary: dict[str, Any], figures_dir: Path) -> Path:
    """MAE by horizon band, model against baseline."""
    table = pd.DataFrame(summary["by_horizon"])
    if table.empty:
        logger.warning("No horizon bands to plot")
        return Path()

    figure, axes = plt.subplots(figsize=(7.2, 3.6), facecolor=SURFACE)
    positions = np.arange(len(table), dtype="float64")
    width = 0.36

    for offset, (column, label, colour) in zip(
        (-0.5, 0.5),
        (("model_mae", "LightGBM", MODEL_COLOUR), ("naive_mae", "seasonal naive", NAIVE_COLOUR)),
    ):
        bars = axes.bar(
            positions + offset * width, table[column], width=width * 0.92,
            color=colour, label=label, zorder=3,
        )
        for bar, value in zip(bars, table[column]):
            axes.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{value:.2f}", ha="center", fontsize=8.5, color=INK,
            )

    axes.set_xticks(positions)
    axes.set_xticklabels(table["horizon_band"], fontsize=9, color=INK)
    _style(axes, xlabel="forecast lead time", ylabel="MAE (arrivals/hour)",
           title="Forecast error grows with lead time, and more slowly than the baseline")
    axes.legend(frameon=False, fontsize=9, labelcolor=INK)
    return save_figure(figure, figures_dir / "forecast_error_by_horizon.png")


def plot_coverage_by_band(coverage: dict[str, Any], config: dict[str, Any], figures_dir: Path) -> Path:
    """Interval coverage per hour-of-day band, as a dot plot against nominal."""
    table = forecast_coverage_table(coverage, config)
    nominal = float(config["conformal"]["coverage"])

    figure, panels = plt.subplots(1, 2, figsize=(11.0, 3.2), facecolor=SURFACE)
    positions = np.arange(len(table), dtype="float64")
    colours = [
        MODEL_COLOUR if band == "overall" else NAIVE_COLOUR for band in table["hour_band"]
    ]

    values = table["empirical_coverage"].to_numpy()
    low = min(float(values.min()), nominal) - 0.04
    high = max(float(values.max()), nominal) + 0.04
    panels[0].axvline(nominal, color=REFERENCE, linewidth=2, linestyle="--", zorder=2,
                      label=f"nominal {nominal:.2f}")
    panels[0].hlines(positions, low, values, color=GRID, linewidth=1.5, zorder=3)
    panels[0].scatter(values, positions, s=90, color=colours, zorder=4,
                      edgecolor=SURFACE, linewidth=1.5, label="empirical")
    panels[0].set_xlim(low, high)
    for position, value in zip(positions, values):
        offset = 0.006 if value < nominal else -0.006
        panels[0].text(value + offset, position - 0.3, f"{value:.3f}", fontsize=8.5,
                       color=INK, ha="left" if value < nominal else "right")
    _style(panels[0], xlabel="coverage", title="Forecast interval coverage by hour band")
    panels[0].legend(frameon=False, fontsize=9, loc="upper center", ncol=2,
                     bbox_to_anchor=(0.5, -0.22), labelcolor=INK)

    bars = panels[1].barh(positions, table["mean_width_arrivals"], height=0.62,
                          color=colours, zorder=3)
    span = float(table["mean_width_arrivals"].max()) * 1.2
    panels[1].set_xlim(0, span)
    for bar, value in zip(bars, table["mean_width_arrivals"]):
        panels[1].text(bar.get_width() + span * 0.012, bar.get_y() + bar.get_height() / 2,
                       f"{value:.2f}", va="center", fontsize=8.5, color=INK)
    _style(panels[1], xlabel="arrivals", title="Mean interval width")

    for panel in panels:
        panel.set_yticks(positions)
        panel.set_yticklabels(table["hour_band"], fontsize=9, color=INK)
        panel.invert_yaxis()
    return save_figure(figure, figures_dir / "forecast_coverage.png")


def write_forecast_figures(
    predictions: pd.DataFrame,
    summary: dict[str, Any],
    issued: pd.DataFrame,
    figures_dir: Path,
    config: dict[str, Any] | None = None,
) -> list[Path]:
    """Write every forecast figure and return the paths."""
    paths = [
        plot_backtest(predictions, figures_dir),
        plot_error_by_horizon(summary, figures_dir),
        plot_forecast_window(issued, figures_dir),
    ]
    coverage = summary.get("interval_coverage")
    if coverage and config is not None:
        paths.append(plot_coverage_by_band(coverage, config, figures_dir))
    logger.info("Wrote %d forecast figures to %s", len(paths), figures_dir)
    return paths
