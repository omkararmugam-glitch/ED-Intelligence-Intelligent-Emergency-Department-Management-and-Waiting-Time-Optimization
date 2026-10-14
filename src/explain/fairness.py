"""Subgroup error analysis for the wait-time models (research gap G2).

Fairness here means **equality of error**, not equality of prediction. Patients at
different acuity levels *should* receive different predicted waits; what would be
indefensible is the model being reliably *worse* for one group than another, because
that group then receives worse decision support.

The headline quantity is the **false-negative rate** of the prolonged-wait classifier.
A false negative tells a patient they will be seen soon when in fact they will wait a
long time. If that error lands disproportionately on one subgroup, that subgroup is
systematically under-warned, which is the concrete harm this report exists to detect.

Three things this module is careful about:

  * **Gaps need uncertainty.** A subgroup of 60 patients will differ from the overall
    rate by chance alone, so every gap carries a bootstrap confidence interval and a
    flag for whether it excludes zero. Reporting a bare gap invites over-reading noise.
  * **Small groups are reported, not silently dropped.** Groups below
    `fairness.min_group_size` are kept in the table and marked, because "too small to
    judge" is itself a finding worth seeing.
  * **Missing columns are skipped with a log message**, never an error. A real extract
    may have no recorded sex or arrival mode, and the pipeline must still run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.io import save_figure, save_table
from src.utils.logging import get_logger
from src.utils.metrics import classification_metrics, regression_metrics
from src.utils.seed import rng as make_rng

logger = get_logger(__name__)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#6b6b6b"
GRID = "#e3e3e0"
SERIES = "#2a78d6"
SERIES_ALT = "#eb6834"
WARN = "#d03b3b"

#: Metrics compared across subgroups. FNR leads because it is the costly error.
SUBGROUP_METRICS = ("fnr", "fpr", "recall", "precision", "mae")


@dataclass
class FairnessReport:
    """Subgroup table plus the summary a reader needs to interpret it."""

    table: pd.DataFrame
    overall: dict[str, float]
    attributes_evaluated: list[str]
    attributes_skipped: list[str] = field(default_factory=list)
    flagged: pd.DataFrame = field(default_factory=pd.DataFrame)
    figures: list[Path] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "overall": self.overall,
            "attributes_evaluated": self.attributes_evaluated,
            "attributes_skipped": self.attributes_skipped,
            "n_subgroups": int(len(self.table)),
            "n_flagged": int(len(self.flagged)),
            "n_underpowered_for_fnr": int(self.table["fnr_underpowered"].sum())
            if "fnr_underpowered" in self.table.columns
            else 0,
            "flagged": self.flagged.to_dict(orient="records") if not self.flagged.empty else [],
            "subgroups": self.table.to_dict(orient="records"),
            "figures": [str(p) for p in self.figures],
        }


def add_age_band(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add an `age_band` column from the configured bin edges.

    Args:
        frame: Table that may contain an `age` column.
        config: Full pipeline config; reads `fairness.age_bins` and `age_labels`.

    Returns:
        A copy with `age_band` added when `age` exists, otherwise unchanged.
    """
    out = frame.copy()
    if "age" not in out.columns:
        logger.info("No `age` column; skipping age_band derivation")
        return out

    bins = list(config["fairness"]["age_bins"])
    labels = list(config["fairness"]["age_labels"])
    if len(labels) != len(bins) - 1:
        raise ValueError(
            f"fairness.age_labels has {len(labels)} entries but age_bins implies "
            f"{len(bins) - 1} bands"
        )
    out["age_band"] = pd.cut(out["age"], bins=bins, labels=labels, right=False).astype("object")
    return out


def _bootstrap_gap(
    group_mask: np.ndarray,
    statistic: Any,
    n_resamples: int,
    seed: int,
    label: str,
) -> tuple[float, float]:
    """Percentile bootstrap CI for (group statistic minus overall statistic).

    Both the group and the overall population are resampled on each draw, so the
    interval reflects uncertainty in the comparison rather than in the group alone.
    """
    generator = make_rng(seed, f"fairness_bootstrap::{label}")
    n_total = group_mask.size
    group_positions = np.flatnonzero(group_mask)
    if group_positions.size == 0:
        return float("nan"), float("nan")

    gaps = np.empty(n_resamples)
    for draw in range(n_resamples):
        overall_idx = generator.integers(0, n_total, n_total)
        group_idx = generator.choice(group_positions, group_positions.size, replace=True)
        group_value = statistic(group_idx)
        overall_value = statistic(overall_idx)
        gaps[draw] = group_value - overall_value

    finite = gaps[np.isfinite(gaps)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def _subgroup_row(
    attribute: str,
    value: Any,
    mask: np.ndarray,
    y_true_label: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    y_true_wait: np.ndarray,
    y_pred_wait: np.ndarray,
    overall: dict[str, float],
    config: dict[str, Any],
    n_resamples: int,
) -> dict[str, Any]:
    """Compute one subgroup's metrics and their gaps against the overall rate."""
    min_size = int(config["fairness"]["min_group_size"])
    n = int(mask.sum())

    classification = classification_metrics(
        y_true_label[mask], y_score[mask], threshold=threshold
    )
    regression = regression_metrics(y_true_wait[mask], y_pred_wait[mask])

    # An FNR is computed over positives alone, so a group can be large overall and
    # still carry almost no information about it. ESI 1 patients, for instance, rarely
    # wait long enough to be a positive case at all. Reporting the positive count
    # alongside the rate is what stops a 1.0 FNR computed from two patients being read
    # as a finding.
    n_positive = int((y_true_label[mask] > 0).sum())
    min_positives = int(config["fairness"].get("min_positives_for_fnr", 20))

    row: dict[str, Any] = {
        "attribute": attribute,
        "group": str(value),
        "n": n,
        "n_positive": n_positive,
        "n_negative": int(n - n_positive),
        "share_of_test": round(n / len(mask), 4),
        "positive_rate": round(classification.get("positive_rate", float("nan")), 4),
        "below_min_group_size": n < min_size,
        "fnr_underpowered": n_positive < min_positives,
    }
    for metric in SUBGROUP_METRICS:
        source = regression if metric == "mae" else classification
        row[metric] = round(float(source.get(metric, float("nan"))), 4)
        row[f"{metric}_gap"] = round(row[metric] - overall[metric], 4)

    # Bootstrap only the two headline gaps; the full set would dominate runtime.
    seed = int(config["seed"])
    for metric in ("fnr", "mae"):
        if n < 2:
            row[f"{metric}_gap_ci_low"], row[f"{metric}_gap_ci_high"] = float("nan"), float("nan")
            row[f"{metric}_gap_excludes_zero"] = False
            continue

        if metric == "fnr":

            def statistic(idx: np.ndarray) -> float:
                """False-negative rate over the resampled positions."""
                truth = y_true_label[idx]
                predicted = y_score[idx] >= threshold
                positives = truth.sum()
                return float(((truth > 0) & ~predicted).sum() / positives) if positives else np.nan

        else:

            def statistic(idx: np.ndarray) -> float:
                """Mean absolute error over the resampled positions."""
                return float(np.nanmean(np.abs(y_true_wait[idx] - y_pred_wait[idx])))

        low, high = _bootstrap_gap(mask, statistic, n_resamples, seed, f"{attribute}={value}:{metric}")
        row[f"{metric}_gap_ci_low"] = round(low, 4)
        row[f"{metric}_gap_ci_high"] = round(high, 4)
        row[f"{metric}_gap_excludes_zero"] = bool(
            np.isfinite(low) and np.isfinite(high) and (low > 0 or high < 0)
        )
    return row


def evaluate_fairness(
    frame: pd.DataFrame,
    y_true_label: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    y_true_wait: np.ndarray,
    y_pred_wait: np.ndarray,
    config: dict[str, Any],
) -> FairnessReport:
    """Compare error rates across subgroups of every configured attribute.

    Args:
        frame: Test-block rows carrying the subgroup columns.
        y_true_label: Binary prolonged-wait labels.
        y_score: Predicted probabilities from the classifier.
        threshold: The classifier's frozen decision threshold.
        y_true_wait: Observed waits in minutes.
        y_pred_wait: Predicted waits in minutes.
        config: Full pipeline config; reads the `fairness` block.

    Returns:
        A `FairnessReport`. Attributes absent from `frame` are skipped and recorded.
    """
    settings = config["fairness"]
    attributes = list(settings["subgroup_columns"])
    n_resamples = int(settings.get("bootstrap_resamples", 400))

    enriched = add_age_band(frame, config)
    evaluated, skipped = [], []
    for attribute in attributes:
        if attribute in enriched.columns and enriched[attribute].notna().any():
            evaluated.append(attribute)
        else:
            skipped.append(attribute)
    if skipped:
        logger.info(
            "Fairness: skipping %d attribute(s) absent from this dataset: %s",
            len(skipped),
            ", ".join(skipped),
        )
    if not evaluated:
        logger.warning(
            "Fairness: none of the configured subgroup columns are present; the report "
            "will contain only the overall row"
        )

    overall_classification = classification_metrics(y_true_label, y_score, threshold=threshold)
    overall_regression = regression_metrics(y_true_wait, y_pred_wait)
    overall = {
        metric: float(
            (overall_regression if metric == "mae" else overall_classification).get(
                metric, float("nan")
            )
        )
        for metric in SUBGROUP_METRICS
    }
    overall["n"] = int(len(frame))
    overall["positive_rate"] = float(overall_classification.get("positive_rate", float("nan")))
    logger.info(
        "Fairness overall: FNR %.4f, FPR %.4f, recall %.4f, MAE %.2f min (n=%d)",
        overall["fnr"],
        overall["fpr"],
        overall["recall"],
        overall["mae"],
        overall["n"],
    )

    rows: list[dict[str, Any]] = []
    for attribute in evaluated:
        values = enriched[attribute]
        for value in sorted(values.dropna().unique(), key=str):
            mask = (values == value).to_numpy()
            if mask.sum() == 0:
                continue
            rows.append(
                _subgroup_row(
                    attribute,
                    value,
                    mask,
                    y_true_label,
                    y_score,
                    threshold,
                    y_true_wait,
                    y_pred_wait,
                    overall,
                    config,
                    n_resamples,
                )
            )

    table = pd.DataFrame(rows)
    flagged = pd.DataFrame()
    if not table.empty:
        warn_threshold = float(settings["gap_warn_threshold"])
        # A subgroup is flagged only when the gap is both material AND distinguishable
        # from noise -- either condition alone over-reports.
        material = table["fnr_gap"].abs() > warn_threshold
        powered = ~table["below_min_group_size"] & ~table["fnr_underpowered"]
        flagged = table[material & table["fnr_gap_excludes_zero"] & powered].copy()

        for row in flagged.itertuples():
            logger.warning(
                "Fairness flag: %s=%s has FNR gap %+.4f (95%% CI %+.4f to %+.4f) over "
                "%d positive case(s) in a group of %d",
                row.attribute,
                row.group,
                row.fnr_gap,
                row.fnr_gap_ci_low,
                row.fnr_gap_ci_high,
                row.n_positive,
                row.n,
            )

        # A large gap that cannot be judged is still worth saying out loud, but it is
        # reported as under-powered rather than as a finding.
        underpowered = table[material & ~powered]
        for row in underpowered.itertuples():
            logger.warning(
                "Fairness: %s=%s shows an FNR gap of %+.4f but it rests on only %d "
                "positive case(s) (threshold %d); reported as under-powered, not as a "
                "finding. The bootstrap interval is tight because resampling %d cases "
                "reproduces the same answer, not because the estimate is reliable.",
                row.attribute,
                row.group,
                row.fnr_gap,
                row.n_positive,
                int(config["fairness"].get("min_positives_for_fnr", 20)),
                row.n_positive,
            )
        if flagged.empty:
            logger.info(
                "Fairness: no subgroup exceeds the %.2f FNR gap threshold with a "
                "confidence interval excluding zero",
                warn_threshold,
            )

    return FairnessReport(
        table=table,
        overall=overall,
        attributes_evaluated=evaluated,
        attributes_skipped=skipped,
    )


def plot_fairness(report: FairnessReport, figures_dir: Path, config: dict[str, Any]) -> list[Path]:
    """Plot FNR and MAE gaps per subgroup with their bootstrap intervals."""
    if report.table.empty:
        logger.warning("No subgroups to plot")
        return []

    table = report.table.copy()
    table["label"] = table["attribute"] + " = " + table["group"]
    warn_threshold = float(config["fairness"]["gap_warn_threshold"])

    figure, panels = plt.subplots(
        1, 2, figsize=(12.5, 0.34 * len(table) + 2.2), facecolor=SURFACE
    )
    positions = np.arange(len(table))[::-1]

    for panel, (metric, label, unit) in zip(
        panels,
        (("fnr", "False-negative-rate gap vs overall", ""), ("mae", "MAE gap vs overall", " min")),
    ):
        gaps = table[f"{metric}_gap"].to_numpy()
        low = table.get(f"{metric}_gap_ci_low", pd.Series(np.nan, index=table.index)).to_numpy()
        high = table.get(f"{metric}_gap_ci_high", pd.Series(np.nan, index=table.index)).to_numpy()

        significant = table.get(
            f"{metric}_gap_excludes_zero", pd.Series(False, index=table.index)
        ).to_numpy()
        small = (
            table["below_min_group_size"] | table.get("fnr_underpowered", False)
        ).to_numpy() if metric == "fnr" else table["below_min_group_size"].to_numpy()
        colours = [
            GRID if is_small else (WARN if sig else SERIES)
            for sig, is_small in zip(significant, small)
        ]

        errors = np.vstack([gaps - low, high - gaps])
        errors = np.where(np.isfinite(errors), np.abs(errors), 0.0)
        panel.errorbar(
            gaps, positions, xerr=errors, fmt="none", ecolor=INK_MUTED, elinewidth=1.2, capsize=3,
            zorder=3,
        )
        panel.scatter(gaps, positions, s=70, color=colours, zorder=4, edgecolor=SURFACE,
                      linewidth=1.2)
        panel.axvline(0, color=INK_MUTED, linewidth=1.4, zorder=2)

        if metric == "fnr":
            for sign in (-1, 1):
                panel.axvline(
                    sign * warn_threshold, color=WARN, linewidth=1, linestyle=":", zorder=2
                )

        panel.set_yticks(positions)
        panel.set_yticklabels(table["label"], fontsize=8.5, color=INK)
        panel.set_facecolor(SURFACE)
        panel.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        panel.set_axisbelow(True)
        for side in ("top", "right", "left"):
            panel.spines[side].set_visible(False)
        panel.spines["bottom"].set_color(GRID)
        panel.tick_params(colors=INK_MUTED, labelsize=8.5, length=0)
        panel.set_xlabel(f"gap{unit} (positive = worse than overall)", color=INK_MUTED, fontsize=9)
        panel.set_title(label, color=INK, fontsize=11, loc="left", pad=10)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=8, color=WARN),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=8, color=SERIES),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=8, color=GRID),
    ]
    figure.legend(
        handles,
        ["CI excludes zero", "CI includes zero", "too few cases to judge"],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
        labelcolor=INK,
    )
    path = save_figure(figure, figures_dir / "fairness_gaps.png")
    report.figures = [path]
    return [path]


def build_fairness_report(
    frame: pd.DataFrame,
    y_true_label: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    y_true_wait: np.ndarray,
    y_pred_wait: np.ndarray,
    config: dict[str, Any],
    tables_dir: Path,
    figures_dir: Path,
) -> FairnessReport:
    """Run the fairness analysis and write its table and figure."""
    report = evaluate_fairness(
        frame, y_true_label, y_score, threshold, y_true_wait, y_pred_wait, config
    )
    if not report.table.empty:
        save_table(report.table, tables_dir / "fairness.csv")
        plot_fairness(report, figures_dir, config)
    else:
        logger.warning("Fairness table is empty; no CSV or figure written")
    return report
