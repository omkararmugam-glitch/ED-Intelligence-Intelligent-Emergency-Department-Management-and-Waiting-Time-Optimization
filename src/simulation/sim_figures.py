"""Figures for the simulation scenarios.

Same conventions as the other figure modules: validated categorical hues in fixed
order, one measure per axis, a legend always present, recessive grid.
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

BETTER = "#1baf7a"
WORSE = "#eb6834"
NEUTRAL = "#2a78d6"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#6b6b6b"
GRID = "#e3e3e0"
REFERENCE = "#d03b3b"


def _style(axes: plt.Axes, xlabel: str = "", title: str = "") -> None:
    """Shared recessive styling."""
    axes.set_facecolor(SURFACE)
    axes.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID)
    axes.tick_params(colors=INK_MUTED, labelsize=8.5, length=0)
    if xlabel:
        axes.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    if title:
        axes.set_title(title, color=INK, fontsize=11, loc="left", pad=10)


def plot_scenario_kpis(summary: pd.DataFrame, figures_dir: Path) -> Path:
    """Mean wait and ESI-3 p90 per scenario, with confidence intervals."""
    frame = summary.copy()
    figure, panels = plt.subplots(
        1, 2, figsize=(12.0, 0.45 * len(frame) + 2.0), facecolor=SURFACE
    )
    positions = np.arange(len(frame))[::-1]

    for panel, (kpi, label) in zip(
        panels,
        (("mean_wait_minutes", "Mean wait (minutes)"), ("p90_wait_esi3", "p90 wait, ESI 3 (minutes)")),
    ):
        values = frame[kpi].to_numpy()
        low = frame[f"{kpi}_ci_low"].to_numpy()
        high = frame[f"{kpi}_ci_high"].to_numpy()
        baseline_value = float(frame.loc[frame["scenario"] == "baseline", kpi].iloc[0])
        colours = [
            NEUTRAL if name == "baseline" else (BETTER if value < baseline_value else WORSE)
            for name, value in zip(frame["scenario"], values)
        ]

        errors = np.vstack([values - low, high - values])
        panel.barh(positions, values, height=0.62, color=colours, zorder=3)
        panel.errorbar(
            values, positions, xerr=np.abs(errors), fmt="none",
            ecolor=INK_MUTED, elinewidth=1.2, capsize=3, zorder=4,
        )
        panel.axvline(
            baseline_value, color=REFERENCE, linewidth=1.6, linestyle="--", zorder=2,
            label="baseline",
        )
        span = float(high.max()) * 1.18
        panel.set_xlim(0, span)
        for position, value in zip(positions, values):
            panel.text(value + span * 0.015, position, f"{value:.1f}", va="center",
                       fontsize=8.5, color=INK)

        panel.set_yticks(positions)
        panel.set_yticklabels(frame["scenario"], fontsize=9, color=INK)
        _style(panel, xlabel=label)

    panels[0].set_title(
        "Scenario KPIs with 95% confidence intervals", color=INK, fontsize=11,
        loc="left", pad=10,
    )
    panels[0].legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)
    return save_figure(figure, figures_dir / "scenario_kpis.png")


def plot_paired_comparison(comparison: pd.DataFrame, figures_dir: Path) -> Path:
    """Difference from baseline per scenario, showing which comparisons are paired."""
    frame = comparison.copy()
    figure, axes = plt.subplots(figsize=(8.5, 0.45 * len(frame) + 2.0), facecolor=SURFACE)
    positions = np.arange(len(frame))[::-1]

    values = frame["difference_vs_baseline"].to_numpy()
    low = frame["ci_low"].to_numpy()
    high = frame["ci_high"].to_numpy()
    colours = [BETTER if value < 0 else WORSE for value in values]
    # Unpaired comparisons are drawn hollow, so the weaker evidence is visible.
    paired = frame["method"].str.startswith("paired").to_numpy()

    axes.errorbar(
        values, positions, xerr=np.abs(np.vstack([values - low, high - values])),
        fmt="none", ecolor=INK_MUTED, elinewidth=1.4, capsize=4, zorder=3,
    )
    for position, value, colour, is_paired in zip(positions, values, colours, paired):
        axes.scatter(
            value, position, s=110, zorder=4,
            color=colour if is_paired else SURFACE,
            edgecolor=colour, linewidth=2,
        )
    axes.axvline(0, color=INK_MUTED, linewidth=1.4, zorder=2)

    axes.set_yticks(positions)
    axes.set_yticklabels(frame["scenario"], fontsize=9, color=INK)
    _style(
        axes,
        xlabel="change in mean wait vs baseline (minutes; negative is better)",
        title="Scenario effect, with common random numbers where demand is unchanged",
    )
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=9, color=BETTER),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=9,
                   markerfacecolor=SURFACE, markeredgecolor=WORSE, markeredgewidth=2),
    ]
    axes.legend(
        handles, ["paired (common random numbers)", "unpaired (demand changed)"],
        frameon=False, fontsize=9, loc="lower right", labelcolor=INK,
    )
    return save_figure(figure, figures_dir / "scenario_comparison.png")


def plot_sensitivity(by_world: pd.DataFrame, figures_dir: Path) -> Path:
    """Mean wait per scenario across every perturbed world."""
    pivot = by_world.pivot_table(
        index="scenario", columns="perturbation", values="mean_wait_minutes"
    )
    figure, axes = plt.subplots(figsize=(10.0, 0.45 * len(pivot) + 2.2), facecolor=SURFACE)
    positions = np.arange(len(pivot))[::-1]

    for column in pivot.columns:
        emphasis = column == "point_estimate"
        axes.scatter(
            pivot[column], positions,
            s=110 if emphasis else 55,
            color=NEUTRAL if emphasis else GRID,
            edgecolor=SURFACE if emphasis else INK_MUTED,
            linewidth=1.4 if emphasis else 0.8,
            zorder=4 if emphasis else 3,
            label="point estimate" if emphasis else None,
        )

    axes.set_yticks(positions)
    axes.set_yticklabels(pivot.index, fontsize=9, color=INK)
    _style(
        axes,
        xlabel="mean wait (minutes) across perturbed worlds",
        title="Does the scenario ranking survive +/-10% and +/-20% input errors?",
    )
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=10, color=NEUTRAL),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=7, color=GRID),
    ]
    axes.legend(handles, ["point estimate", "perturbed world"], frameon=False,
                fontsize=9, loc="lower right", labelcolor=INK)
    return save_figure(figure, figures_dir / "sensitivity.png")


def plot_staffing_search(plans: pd.DataFrame, target: Any, figures_dir: Path) -> Path:
    """Cost against service level, with the feasible region marked."""
    figure, axes = plt.subplots(figsize=(8.0, 5.0), facecolor=SURFACE)

    feasible = plans[plans["feasible"]]
    infeasible = plans[~plans["feasible"]]

    axes.scatter(
        infeasible["p90_wait_esi3"], infeasible["cost"], s=55, color=GRID,
        edgecolor=INK_MUTED, linewidth=0.8, zorder=3, label="infeasible",
    )
    axes.scatter(
        feasible["p90_wait_esi3"], feasible["cost"], s=70, color=BETTER,
        edgecolor=SURFACE, linewidth=1.2, zorder=4, label="meets every target",
    )
    if not feasible.empty:
        best = feasible.loc[feasible["cost"].idxmin()]
        axes.scatter(
            [best["p90_wait_esi3"]], [best["cost"]], s=240, marker="*",
            color=NEUTRAL, edgecolor=SURFACE, linewidth=1.4, zorder=5,
            label=f"cheapest feasible: {best['plan']}",
        )

    axes.axvline(
        target.p90_wait_esi3_max, color=REFERENCE, linewidth=1.8, linestyle="--",
        zorder=2, label=f"target p90 <= {target.p90_wait_esi3_max:g} min",
    )

    axes.set_facecolor(SURFACE)
    axes.grid(color=GRID, linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    axes.tick_params(colors=INK_MUTED, labelsize=8.5)
    axes.set_xlabel("p90 wait for ESI 3 (minutes)", color=INK_MUTED, fontsize=9)
    axes.set_ylabel("weekly doctor cost", color=INK_MUTED, fontsize=9)
    axes.set_title(
        "Staffing plans under forecast demand: cost against service level",
        color=INK, fontsize=11, loc="left", pad=10,
    )
    axes.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK)
    return save_figure(figure, figures_dir / "staffing_search.png")


def write_scenario_figures(
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    figures_dir: Path,
    config: dict[str, Any],
) -> list[Path]:
    """Write the scenario figures."""
    paths = [
        plot_scenario_kpis(summary, figures_dir),
        plot_paired_comparison(comparison, figures_dir),
    ]
    logger.info("Wrote %d scenario figures to %s", len(paths), figures_dir)
    return paths
