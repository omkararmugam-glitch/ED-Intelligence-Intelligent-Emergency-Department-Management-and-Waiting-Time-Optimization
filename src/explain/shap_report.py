"""SHAP explanations for the best wait-time model, computed by LightGBM itself.

**No `shap` dependency.** LightGBM implements TreeSHAP natively:
`model.predict(X, pred_contrib=True)` returns one column per feature plus a final
column holding the base value. These are the same exact Shapley values the `shap`
package's `TreeExplainer` returns for a LightGBM model -- `shap` calls into this very
routine -- so dropping the package costs nothing in fidelity and removes a dependency
on `numba`, whose compiled DLL is blocked by Windows Application Control on some
machines.

The defining property, which the tests assert: **contributions plus the base value sum
exactly to the model's raw prediction.** That is what makes an attribution an
explanation rather than a heuristic ranking.

Two kinds of output, because they answer different questions:

  * **Global** -- which features drive the model overall: mean absolute contribution per
    feature, as a table, a ranked bar chart and a beeswarm. This is a statement about
    what this model uses, not a causal claim.
  * **Local** -- why *this* patient got *this* prediction, decomposed into contributions
    that sum to it. This is the form a clinician can interrogate.

A note on the log target: when the best model was fitted on `log1p(wait)`, contributions
are in log space and do **not** convert additively back to minutes. Every table, axis
label and JSON field carries the unit rather than implying minutes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import save_figure, save_table
from src.utils.logging import get_logger
from src.utils.seed import rng as make_rng

logger = get_logger(__name__)

#: Palette, matching the other figure modules (validated categorical slots).
SERIES_POSITIVE = "#eb6834"
SERIES_NEGATIVE = "#2a78d6"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#6b6b6b"
GRID = "#e3e3e0"

#: Single-hue sequential ramp for feature value in the beeswarm, light to dark.
#: Sequential because a feature value is a magnitude, not a polarity.
VALUE_RAMP = ("#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b")

#: Prefixes identifying queue-state features, used to quantify their share.
QUEUE_FEATURE_PREFIXES = (
    "n_waiting",
    "n_in_treatment",
    "arrivals_last",
    "mean_wait_last",
    "n_seen_last",
)


@dataclass
class ShapReport:
    """Everything the SHAP stage produces."""

    global_table: pd.DataFrame
    local_examples: pd.DataFrame
    units: str
    n_explained: int
    feature_set: str
    target_transform: str
    base_value: float
    figures: list[Path] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "feature_set": self.feature_set,
            "target_transform": self.target_transform,
            "shap_units": self.units,
            "shap_backend": "lightgbm native TreeSHAP (pred_contrib=True)",
            "n_explained": self.n_explained,
            "base_value": self.base_value,
            "top_features": self.global_table.head(10).to_dict(orient="records"),
            "n_local_examples": int(self.local_examples["example_id"].nunique())
            if not self.local_examples.empty
            else 0,
            "figures": [str(p) for p in self.figures],
        }


def _sample_rows(
    features: pd.DataFrame, max_samples: int, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Take a reproducible sample of rows to explain."""
    n = len(features)
    if n <= max_samples:
        return features, np.arange(n)

    generator = make_rng(seed, "shap_sample")
    positions = np.sort(generator.choice(n, size=max_samples, replace=False))
    logger.info("Sampling %d of %d rows for SHAP", max_samples, n)
    return features.iloc[positions], positions


def compute_shap_values(
    model: Any,
    features: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[np.ndarray, pd.DataFrame, float]:
    """Compute exact TreeSHAP contributions using LightGBM's own implementation.

    Args:
        model: A fitted LightGBM estimator (sklearn wrapper or raw Booster).
        features: Design matrix to explain.
        config: Full pipeline config; reads `explain.shap_max_samples` and `seed`.

    Returns:
        Tuple of (contributions of shape `(n_rows, n_features)`, the explained rows,
        the base value). Contributions plus the base value sum to the raw prediction.

    Raises:
        TypeError: If the model does not support `pred_contrib`.
    """
    max_samples = int(config["explain"]["shap_max_samples"])
    sample, _ = _sample_rows(features, max_samples, int(config["seed"]))

    try:
        raw = model.predict(sample, pred_contrib=True)
    except TypeError as error:  # pragma: no cover - guards a wrong model type
        raise TypeError(
            "This model does not support LightGBM's pred_contrib=True. SHAP values are "
            "computed by LightGBM natively; pass a fitted LightGBM estimator."
        ) from error

    contributions = np.asarray(raw)
    if contributions.ndim != 2 or contributions.shape[1] != sample.shape[1] + 1:
        raise ValueError(
            f"Expected pred_contrib output of shape (n, {sample.shape[1] + 1}), got "
            f"{contributions.shape}. For a multiclass model, extract one class first."
        )

    # LightGBM puts the base value in the final column.
    values = contributions[:, :-1]
    base = float(contributions[:, -1][0])

    logger.info(
        "Computed SHAP values for %d rows x %d features via LightGBM TreeSHAP "
        "(base value %.4f)",
        values.shape[0],
        values.shape[1],
        base,
    )
    return values, sample, base


def verify_additivity(
    model: Any, features: pd.DataFrame, values: np.ndarray, base: float, tolerance: float = 1e-6
) -> float:
    """Check that contributions plus the base value reproduce the raw prediction.

    This is the property that makes the attribution an explanation. It is checked here
    rather than assumed, because a silent mismatch would make every local explanation
    wrong in a way no plot would reveal.

    Args:
        model: The fitted estimator.
        features: The explained rows.
        values: Contributions, excluding the base column.
        base: The base value.
        tolerance: Maximum acceptable absolute discrepancy.

    Returns:
        The largest absolute discrepancy observed.

    Raises:
        ValueError: If the discrepancy exceeds `tolerance`.
    """
    reconstructed = values.sum(axis=1) + base
    direct = np.asarray(model.predict(features), dtype="float64")
    discrepancy = float(np.max(np.abs(reconstructed - direct)))

    if discrepancy > tolerance:
        raise ValueError(
            f"SHAP contributions do not sum to the model output (max discrepancy "
            f"{discrepancy:.3e} > {tolerance:.1e}). The explanation would be wrong."
        )
    logger.debug("SHAP additivity verified (max discrepancy %.3e)", discrepancy)
    return discrepancy


def is_queue_feature(name: str) -> bool:
    """True when a feature name belongs to the queue-state group."""
    return str(name).startswith(QUEUE_FEATURE_PREFIXES)


def global_importance_table(values: np.ndarray, features: pd.DataFrame) -> pd.DataFrame:
    """Rank features by mean absolute contribution.

    Also reports the mean signed contribution, which says whether a feature typically
    pushes the prediction up or down, and each feature's share of total attribution.
    """
    mean_abs = np.abs(values).mean(axis=0)
    table = pd.DataFrame(
        {
            "feature": features.columns,
            "mean_abs_shap": mean_abs,
            "mean_shap": values.mean(axis=0),
            "max_abs_shap": np.abs(values).max(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)

    total = table["mean_abs_shap"].sum()
    table["share_of_attribution"] = (
        table["mean_abs_shap"] / total if total > 0 else np.nan
    ).round(4)
    table["rank"] = np.arange(1, len(table) + 1)
    table["is_queue_feature"] = table["feature"].map(is_queue_feature)
    for column in ("mean_abs_shap", "mean_shap", "max_abs_shap"):
        table[column] = table[column].round(4)
    return table[
        [
            "rank",
            "feature",
            "mean_abs_shap",
            "mean_shap",
            "max_abs_shap",
            "share_of_attribution",
            "is_queue_feature",
        ]
    ]


def local_explanations(
    values: np.ndarray,
    features: pd.DataFrame,
    context: pd.DataFrame,
    base_value: float,
    config: dict[str, Any],
    n_examples: int = 4,
    top_n: int = 8,
) -> pd.DataFrame:
    """Decompose a few individual predictions into feature contributions.

    Examples span the prediction range rather than being sampled at random: the point of
    a local explanation is to show why a *short* wait and a *long* wait differ, which
    four typical patients would not illustrate.

    Args:
        values: Contributions for `features`.
        features: The explained rows.
        context: Row-aligned patient context (ids, triage, observed wait).
        base_value: The model's base value.
        config: Full pipeline config.
        n_examples: How many patients to explain.
        top_n: How many contributions to list per patient.

    Returns:
        Long-format table, at most `n_examples * top_n` rows.
    """
    totals = values.sum(axis=1) + base_value
    if totals.size == 0:
        return pd.DataFrame()

    quantiles = np.linspace(0.02, 0.98, n_examples)
    chosen = np.unique(
        [int(np.argmin(np.abs(totals - np.quantile(totals, q)))) for q in quantiles]
    )

    rows: list[dict[str, Any]] = []
    for example_id, position in enumerate(chosen, start=1):
        contributions = values[position]
        order = np.argsort(np.abs(contributions))[::-1][:top_n]
        patient = context.iloc[position]
        for rank, feature_index in enumerate(order, start=1):
            rows.append(
                {
                    "example_id": example_id,
                    "patient_id": patient.get("patient_id"),
                    "triage_level": patient.get("triage_level"),
                    "observed_wait_minutes": round(float(patient.get("wait_minutes", np.nan)), 2),
                    "model_output": round(float(totals[position]), 4),
                    "base_value": round(base_value, 4),
                    "rank": rank,
                    "feature": features.columns[feature_index],
                    "feature_value": round(float(features.iloc[position, feature_index]), 4),
                    "shap_contribution": round(float(contributions[feature_index]), 4),
                }
            )
    logger.info("Built %d local explanations", len(chosen))
    return pd.DataFrame(rows)


def _style_axes(axes: Any, xlabel: str = "", title: str = "") -> None:
    """Shared recessive styling for the SHAP figures."""
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


def _beeswarm_offsets(x: np.ndarray, n_bins: int = 60, spread: float = 0.34) -> np.ndarray:
    """Vertical offsets that spread overlapping points, as a beeswarm does.

    Points are binned along x and the members of each bin are fanned symmetrically
    about the row, so dense regions read as thick bands and sparse tails as single
    points. Without this every row would be an unreadable line.
    """
    if x.size == 0:
        return np.zeros(0)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros(x.size)

    low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros(x.size)

    edges = np.linspace(low, high, n_bins + 1)
    bin_index = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)

    offsets = np.zeros(x.size)
    for b in range(n_bins):
        members = np.flatnonzero(bin_index == b)
        if members.size <= 1:
            continue
        # Fan symmetrically: 0, +1, -1, +2, -2, ... scaled into the row's half-height.
        ranks = np.arange(members.size) - (members.size - 1) / 2.0
        offsets[members] = ranks / max(members.size, 2) * 2.0 * spread
    return offsets


def plot_global_summary(
    values: np.ndarray, features: pd.DataFrame, figures_dir: Path, units: str, top_n: int = 15
) -> Path:
    """Beeswarm summary: contribution distribution per feature, coloured by value.

    Built with matplotlib rather than `shap.summary_plot`, so the project carries no
    `shap` dependency. Colour encodes the feature's own value on a single-hue
    light-to-dark ramp, because a feature value is a magnitude rather than a polarity.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    order = np.argsort(np.abs(values).mean(axis=0))[::-1][:top_n][::-1]
    colormap = LinearSegmentedColormap.from_list("feature_value", VALUE_RAMP)

    figure, axes = plt.subplots(
        figsize=(9.0, 0.42 * len(order) + 2.0), facecolor=SURFACE
    )
    scatter = None

    for row, feature_index in enumerate(order):
        contribution = values[:, feature_index]
        raw_value = pd.to_numeric(features.iloc[:, feature_index], errors="coerce").to_numpy(
            dtype="float64"
        )

        # Normalise each feature independently; absolute scales are not comparable.
        finite = raw_value[np.isfinite(raw_value)]
        if finite.size and finite.max() > finite.min():
            normalised = Normalize(vmin=finite.min(), vmax=finite.max())(raw_value)
        else:
            normalised = np.full(raw_value.shape, 0.5)

        scatter = axes.scatter(
            contribution,
            row + _beeswarm_offsets(contribution),
            c=normalised,
            cmap=colormap,
            s=11,
            alpha=0.75,
            linewidths=0,
            zorder=3,
        )

    axes.axvline(0, color=INK_MUTED, linewidth=1.1, zorder=2)
    axes.set_yticks(np.arange(len(order)))
    axes.set_yticklabels([features.columns[i] for i in order], fontsize=8.5, color=INK)
    axes.set_ylim(-0.7, len(order) - 0.3)
    _style_axes(
        axes,
        xlabel=f"contribution to the prediction ({units})",
        title="Which features move the wait-time prediction, and in which direction",
    )

    if scatter is not None:
        bar = figure.colorbar(scatter, ax=axes, pad=0.02, aspect=28)
        bar.set_label("feature value (low to high)", color=INK_MUTED, fontsize=8.5)
        bar.ax.tick_params(colors=INK_MUTED, labelsize=7.5, length=0)
        bar.set_ticks([0.0, 1.0])
        bar.set_ticklabels(["low", "high"])
        bar.outline.set_visible(False)

    return save_figure(figure, figures_dir / "shap_summary.png")


def plot_global_bar(table: pd.DataFrame, figures_dir: Path, units: str, top_n: int = 20) -> Path:
    """Horizontal bar of mean absolute contribution, with queue features called out."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = table.head(top_n).iloc[::-1]
    figure, axes = plt.subplots(figsize=(8.0, 0.32 * len(top) + 1.8), facecolor=SURFACE)

    colours = [SERIES_POSITIVE if flag else SERIES_NEGATIVE for flag in top["is_queue_feature"]]
    positions = np.arange(len(top))
    bars = axes.barh(positions, top["mean_abs_shap"], height=0.72, color=colours, zorder=3)

    span = float(top["mean_abs_shap"].max()) * 1.18
    axes.set_xlim(0, span)
    for bar, value in zip(bars, top["mean_abs_shap"]):
        axes.text(
            bar.get_width() + span * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3g}",
            va="center",
            fontsize=8.5,
            color=INK,
        )

    axes.set_yticks(positions)
    axes.set_yticklabels(top["feature"], fontsize=8.5, color=INK)
    _style_axes(
        axes,
        xlabel=f"mean |contribution| ({units})",
        title="Global feature importance for the best wait-time model",
    )

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=9, color=SERIES_POSITIVE),
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=9, color=SERIES_NEGATIVE),
    ]
    axes.legend(
        handles,
        ["queue-state feature", "calendar / patient feature"],
        frameon=False,
        fontsize=9,
        loc="lower right",
        labelcolor=INK,
    )
    return save_figure(figure, figures_dir / "shap_importance.png")


def plot_local_example(local: pd.DataFrame, figures_dir: Path, units: str) -> Path:
    """Waterfall-style bars for each worked local explanation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    examples = sorted(local["example_id"].unique())
    figure, panels = plt.subplots(
        1, len(examples), figsize=(4.2 * len(examples), 4.4), facecolor=SURFACE, squeeze=False
    )

    for panel, example_id in zip(panels[0], examples):
        rows = local[local["example_id"] == example_id].sort_values(
            "shap_contribution", key=abs
        )
        colours = [
            SERIES_POSITIVE if value > 0 else SERIES_NEGATIVE
            for value in rows["shap_contribution"]
        ]
        positions = np.arange(len(rows))
        panel.barh(positions, rows["shap_contribution"], height=0.7, color=colours, zorder=3)
        panel.axvline(0, color=INK_MUTED, linewidth=1)

        panel.set_yticks(positions)
        panel.set_yticklabels(
            [f"{f} = {v:g}" for f, v in zip(rows["feature"], rows["feature_value"])],
            fontsize=8,
            color=INK,
        )
        head = rows.iloc[0]
        panel.set_title(
            f"ESI {int(head['triage_level'])}, observed {head['observed_wait_minutes']:.0f} min\n"
            f"model output {head['model_output']:.2f} ({units})",
            color=INK,
            fontsize=9.5,
            loc="left",
            pad=8,
        )
        _style_axes(panel, xlabel=f"contribution ({units})")

    figure.suptitle(
        "Why individual patients received their predictions",
        color=INK,
        fontsize=11,
        x=0.02,
        ha="left",
    )
    return save_figure(figure, figures_dir / "shap_local_examples.png")


def build_shap_report(
    model: Any,
    features: pd.DataFrame,
    context: pd.DataFrame,
    config: dict[str, Any],
    tables_dir: Path,
    figures_dir: Path,
    feature_set: str,
    target_transform: str,
) -> ShapReport:
    """Produce the full SHAP report: global table, figures and local examples.

    Args:
        model: The fitted booster of the best wait-time model.
        features: Design matrix (test block).
        context: Row-aligned patient context for labelling local examples.
        config: Full pipeline config.
        tables_dir: Where CSVs go.
        figures_dir: Where PNGs go.
        feature_set: Name of the feature set the model used.
        target_transform: "raw" or "log1p"; determines the units.

    Returns:
        A `ShapReport`.
    """
    units = "minutes" if target_transform == "raw" else "log1p(minutes)"
    if target_transform != "raw":
        logger.warning(
            "The best model was fitted on %s, so contributions are in %s and do NOT "
            "convert additively to minutes. Reported units are labelled accordingly.",
            target_transform,
            units,
        )

    values, sample, base = compute_shap_values(model, features, config)
    verify_additivity(model, sample, values, base)

    sample_positions = features.index.get_indexer(sample.index)
    sample_context = context.iloc[sample_positions]

    table = global_importance_table(values, sample)
    top_n = int(config["explain"]["top_n_features"])
    save_table(table.head(top_n), tables_dir / "shap_top_features.csv")
    save_table(table, tables_dir / "shap_all_features.csv")

    local = local_explanations(values, sample, sample_context, base, config)
    save_table(local, tables_dir / "shap_local_examples.csv")

    figures = [
        plot_global_bar(table, figures_dir, units, top_n=top_n),
        plot_global_summary(values, sample, figures_dir, units),
    ]
    if not local.empty:
        figures.append(plot_local_example(local, figures_dir, units))

    queue_share = float(table.loc[table["is_queue_feature"], "share_of_attribution"].sum())
    logger.info(
        "SHAP: top feature %r (mean |contribution| %.4g %s); queue features carry "
        "%.1f%% of total attribution",
        table.loc[0, "feature"],
        table.loc[0, "mean_abs_shap"],
        units,
        100.0 * queue_share,
    )

    return ShapReport(
        global_table=table,
        local_examples=local,
        units=units,
        n_explained=int(len(sample)),
        feature_set=feature_set,
        target_transform=target_transform,
        base_value=base,
        figures=figures,
    )
