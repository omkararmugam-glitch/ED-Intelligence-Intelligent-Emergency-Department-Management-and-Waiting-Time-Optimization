"""Figures for the wait-time models.

Static PNGs for `artifacts/figures/`, so a reader can see the Phase 1b results without
running anything.

Design choices, applied consistently:
  * Horizontal bars for every magnitude comparison -- the category labels here are
    long ("log1p_smearing"), and horizontal bars give them room without rotation.
  * Two validated categorical hues for the only categorical split that matters
    (feature set), assigned in fixed order and never cycled.
  * Identity is never colour-alone: every chart carries a legend and direct value
    labels on the bars.
  * Never two y-scales on one panel. Coverage and interval width are different
    measures, so they are separate panels.
  * Recessive grid and axes; data ink carries the emphasis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: these are written to disk, never displayed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.io import save_figure
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Validated categorical slots 1 and 2 (light surface). All six checks pass:
#: adjacent-pair CVD delta-E 24.7 (protan), normal-vision 33.6, contrast >= 3:1.
SERIES = {"base": "#2a78d6", "base+queue": "#eb6834"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#6b6b6b"
GRID = "#e3e3e0"
REFERENCE = "#d03b3b"  # status:critical, used only for target/reference lines


def _style_axes(axes: plt.Axes, xlabel: str = "", title: str = "") -> None:
    """Apply the recessive grid/axis treatment shared by every figure."""
    axes.set_facecolor(SURFACE)
    axes.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID)
    axes.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    if xlabel:
        axes.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    if title:
        axes.set_title(title, color=INK, fontsize=11, loc="left", pad=10)


def _label_bars(axes: plt.Axes, bars: Any, values: np.ndarray, fmt: str = "{:.1f}") -> None:
    """Write each value at the end of its bar, so no number is colour-only."""
    span = axes.get_xlim()[1] - axes.get_xlim()[0]
    for bar, value in zip(bars, values):
        axes.text(
            bar.get_width() + span * 0.012,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(value),
            va="center",
            ha="left",
            fontsize=8.5,
            color=INK,
        )


def plot_ablation(table: pd.DataFrame, figures_dir: Path, baseline_mae: float | None) -> Path:
    """Grouped horizontal bars of test MAE and RMSE by feature set and target.

    Args:
        table: The ablation table.
        figures_dir: Destination directory.
        baseline_mae: Baseline test MAE, drawn as a reference line when given.

    Returns:
        Path to the written figure.
    """
    kinds = list(dict.fromkeys(table["prediction_kind"]))
    figure, panels = plt.subplots(1, 2, figsize=(11.5, 3.6), facecolor=SURFACE)

    for panel, (metric, label) in zip(
        panels, (("mae_minutes", "MAE (minutes)"), ("rmse_minutes", "RMSE (minutes)"))
    ):
        positions = np.arange(len(kinds), dtype="float64")
        height = 0.36
        for offset, (feature_set, colour) in zip((-0.5, 0.5), SERIES.items()):
            subset = table[table["feature_set"] == feature_set].set_index("prediction_kind")
            values = np.array(
                [subset.loc[kind, metric] if kind in subset.index else np.nan for kind in kinds]
            )
            bars = panel.barh(
                positions + offset * height,
                values,
                height=height * 0.92,  # 2px-equivalent gap between adjacent bars
                color=colour,
                label=feature_set,
                zorder=3,
            )
            panel.set_xlim(0, float(np.nanmax(table[metric])) * 1.28)
            _label_bars(panel, bars, values)

        panel.set_yticks(positions)
        panel.set_yticklabels(kinds, fontsize=9, color=INK)
        panel.invert_yaxis()
        _style_axes(panel, xlabel=label)

        if metric == "mae_minutes" and baseline_mae is not None:
            panel.axvline(
                baseline_mae, color=REFERENCE, linewidth=2, linestyle="--", zorder=4,
                label=f"baseline ({baseline_mae:.1f})",
            )

    panels[0].set_title(
        "Wait-time model error on the test block, by feature set and target",
        color=INK, fontsize=11, loc="left", pad=10,
    )
    handles, labels = panels[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", ncol=3, frameon=False,
        fontsize=9, bbox_to_anchor=(0.5, -0.07), labelcolor=INK,
    )
    figure.text(
        0.5, -0.17,
        "Synthetic data inflates the queue-feature gap: the generator produces waits by "
        "simulating a queue.\nTreat the direction as supported and the magnitude as an upper bound.",
        ha="center", fontsize=8, color=INK_MUTED,
    )
    return save_figure(figure, figures_dir / "ablation_error.png")


def plot_conformal_coverage(table: pd.DataFrame, figures_dir: Path) -> Path:
    """Coverage against nominal, and interval width, as two separate panels.

    Coverage is a probability and width is in minutes, so they never share an axis.
    """
    rows = table.copy()
    rows["label"] = rows["triage_level"].astype(str).replace({"overall": "overall"})
    nominal = float(rows["nominal_coverage"].iloc[0])

    figure, panels = plt.subplots(1, 2, figsize=(11.5, 3.4), facecolor=SURFACE)
    positions = np.arange(len(rows), dtype="float64")
    # The pooled row is the summary, so it is tinted differently from the per-level rows.
    colours = [
        SERIES["base"] if label == "overall" else SERIES["base+queue"] for label in rows["label"]
    ]

    # Coverage is a position against a reference, not a magnitude from zero, so it is
    # drawn as a dot plot. That lets the axis zoom to the region where the differences
    # live without the truncated-baseline distortion a zoomed bar chart would cause.
    coverage = rows["empirical_coverage"].to_numpy()
    span_low = min(float(coverage.min()), nominal) - 0.04
    span_high = max(float(coverage.max()), nominal) + 0.04
    panels[0].axvline(
        nominal, color=REFERENCE, linewidth=2, linestyle="--", zorder=2,
        label=f"nominal {nominal:.2f}",
    )
    panels[0].hlines(
        positions, span_low, coverage, color=GRID, linewidth=1.5, zorder=3
    )
    panels[0].scatter(
        coverage, positions, s=90, color=colours, zorder=4,
        edgecolor=SURFACE, linewidth=1.5, label="empirical",
    )
    panels[0].set_xlim(span_low, span_high)
    for position, value in zip(positions, coverage):
        # Label on whichever side of the marker has room, so nothing lands on the line.
        offset = 0.006 if value < nominal else -0.006
        panels[0].text(
            value + offset, position - 0.30, f"{value:.3f}",
            va="center", ha="left" if value < nominal else "right",
            fontsize=8.5, color=INK,
        )
    panels[0].set_title(
        "Empirical coverage of 90% conformal intervals (test block)",
        color=INK, fontsize=11, loc="left", pad=10,
    )
    panels[0].legend(
        frameon=False, fontsize=9, loc="upper center", ncol=2,
        bbox_to_anchor=(0.5, -0.22), labelcolor=INK,
    )

    width_bars = panels[1].barh(
        positions, rows["mean_width_minutes"], height=0.62, color=colours, zorder=3
    )
    panels[1].set_xlim(0, float(rows["mean_width_minutes"].max()) * 1.22)
    _label_bars(panels[1], width_bars, rows["mean_width_minutes"].to_numpy())
    panels[1].set_title(
        "Mean interval width (minutes)", color=INK, fontsize=11, loc="left", pad=10
    )

    for panel in panels:
        panel.set_yticks(positions)
        panel.set_yticklabels(
            [f"ESI {label}" if label != "overall" else "overall" for label in rows["label"]],
            fontsize=9, color=INK,
        )
        panel.invert_yaxis()
        _style_axes(panel)
    panels[0].set_xlabel("coverage", color=INK_MUTED, fontsize=9)
    panels[1].set_xlabel("minutes", color=INK_MUTED, fontsize=9)
    return save_figure(figure, figures_dir / "conformal_coverage.png")


def plot_seen_only_bias(table: pd.DataFrame, figures_dir: Path) -> Path:
    """Mean wait per ESI level under both views of the abandoners."""
    figure, axes = plt.subplots(figsize=(8.0, 3.6), facecolor=SURFACE)
    positions = np.arange(len(table), dtype="float64")
    height = 0.36

    series = (
        ("mean_wait_seen_only", "seen patients only", SERIES["base"]),
        (
            "mean_wait_with_abandon_lower_bound",
            "+ abandoners at their lower bound",
            SERIES["base+queue"],
        ),
    )
    for offset, (column, label, colour) in zip((-0.5, 0.5), series):
        bars = axes.barh(
            positions + offset * height,
            table[column],
            height=height * 0.92,
            color=colour,
            label=label,
            zorder=3,
        )
        axes.set_xlim(0, float(table["mean_wait_with_abandon_lower_bound"].max()) * 1.25)
        _label_bars(axes, bars, table[column].to_numpy())

    axes.set_yticks(positions)
    axes.set_yticklabels([f"ESI {int(level)}" for level in table["triage_level"]],
                         fontsize=9, color=INK)
    axes.invert_yaxis()
    _style_axes(
        axes,
        xlabel="mean wait (minutes)",
        title="Training only on seen patients understates the wait",
    )
    axes.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)
    figure.text(
        0.5, -0.08,
        "Time-to-abandon is a LOWER bound on how long those patients would have waited, "
        "so the gap is the minimum bias.",
        ha="center", fontsize=8, color=INK_MUTED,
    )
    return save_figure(figure, figures_dir / "seen_only_bias.png")


def plot_prediction_quality(
    observed: np.ndarray,
    predicted: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    figures_dir: Path,
    n_bins: int = 20,
) -> Path:
    """Observed against predicted wait, binned, with the conformal band.

    A scatter of thousands of patients is an unreadable blob, so predictions are
    binned by decile of predicted wait and the bin's mean observed wait is plotted
    against it. A perfectly calibrated model sits on the diagonal.
    """
    frame = pd.DataFrame(
        {"observed": observed, "predicted": predicted, "lower": lower, "upper": upper}
    ).dropna()
    frame["bin"] = pd.qcut(frame["predicted"], q=n_bins, duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(
        predicted=("predicted", "mean"),
        observed=("observed", "mean"),
        lower=("lower", "mean"),
        upper=("upper", "mean"),
        n=("observed", "size"),
    )

    figure, axes = plt.subplots(figsize=(6.4, 5.2), facecolor=SURFACE)
    limit = float(max(grouped["predicted"].max(), grouped["observed"].max())) * 1.08

    axes.plot(
        [0, limit], [0, limit], color=INK_MUTED, linewidth=1.2, linestyle="--",
        zorder=2, label="perfect calibration",
    )
    axes.fill_between(
        grouped["predicted"], grouped["lower"], grouped["upper"],
        color=SERIES["base+queue"], alpha=0.16, zorder=1,
        label="mean 90% conformal band",
    )
    axes.plot(
        grouped["predicted"], grouped["observed"], color=SERIES["base"], linewidth=2,
        marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2,
        zorder=4, label="observed mean per bin",
    )

    axes.set_xlim(0, limit)
    axes.set_ylim(0, limit)
    axes.set_aspect("equal")
    _style_axes(
        axes,
        xlabel="predicted wait (minutes)",
        title="Predicted against observed wait, by prediction decile",
    )
    axes.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    axes.set_ylabel("observed mean wait (minutes)", color=INK_MUTED, fontsize=9)
    axes.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    return save_figure(figure, figures_dir / "prediction_quality.png")


def write_wait_model_figures(
    ablation: pd.DataFrame,
    conformal: pd.DataFrame,
    bias: pd.DataFrame,
    figures_dir: Path,
    baseline_mae: float | None = None,
    quality: dict[str, np.ndarray] | None = None,
) -> list[Path]:
    """Write every Phase 1b figure and return the paths."""
    paths = [
        plot_ablation(ablation, figures_dir, baseline_mae),
        plot_conformal_coverage(conformal, figures_dir),
        plot_seen_only_bias(bias, figures_dir),
    ]
    if quality is not None:
        paths.append(
            plot_prediction_quality(
                quality["observed"], quality["predicted"],
                quality["lower"], quality["upper"], figures_dir,
            )
        )
    logger.info("Wrote %d wait-model figures to %s", len(paths), figures_dir)
    return paths
