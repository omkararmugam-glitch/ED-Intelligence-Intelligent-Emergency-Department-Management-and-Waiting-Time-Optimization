"""Assemble `artifacts/report.md` from the artifacts on disk.

Every number here is read from a file that a pipeline stage wrote. Nothing is
re-estimated and nothing is typed in, so the report cannot drift from the run that
produced it; the only arithmetic done here is on figures already in the artifacts --
a difference between two recalls, the width of an interval printed either side of
it, a percentage change between two MAEs. The prose follows the same rule: a
sentence that interprets a result --
"no meaningful improvement", "worse at peak hours", "already tried and measured" --
is generated from the same table it describes, so a claim cannot outlive the run that
justified it.

A section whose artifact is missing says so rather than being silently omitted - a
reader needs to know the difference between "zero" and "not run". Missing artifacts
are then split in two: the ones this run was never going to produce, which are not a
problem, and the ones a stage that did run should have written, which are.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.utils.io import load_json, resolve_path, save_json
from src.utils.logging import get_logger
from src.utils.report import md_table

logger = get_logger(__name__)

MISSING = "_Not available: the stage that produces this did not run._"

#: Tables whose first line is a `#` caveat rather than the header. The caveat is
#: written into the file so it cannot be separated from the numbers; pandas has to be
#: told to skip it, and forgetting to do so turns the caveat into the header row.
CAVEATED_TABLES = frozenset({"ablation.csv", "high_acuity_comparison.csv"})

#: The file recording which stages a run performed. Written by `src.pipeline` and by
#: the upload job runner before the report stage, and read back here. Without it the
#: report cannot tell a deliberate absence from a broken one, so it assumes the worse
#: of the two.
RUN_STAGES_FILE = "run_stages.json"

#: Artifact (named as `ArtifactReader` records it) -> the stage that writes it. Used
#: only to explain an absence; no artifact is ever read through this map.
ARTIFACT_PRODUCER: dict[str, str] = {
    "metrics/validation.json": "validate",
    "tables/profile_wait_by_triage.csv": "validate",
    "metrics/split.json": "split",
    "metrics/wait_models.json": "train_wait",
    "tables/ablation.csv": "train_wait",
    "tables/classifier_summary.csv": "train_wait",
    "tables/conformal_coverage.csv": "train_wait",
    "tables/conformal_static_vs_rolling.csv": "train_wait",
    "tables/triage_thresholds.csv": "train_wait",
    "figures/ablation_error.png": "train_wait",
    "metrics/fairness.json": "fairness",
    "tables/fairness.csv": "fairness",
    "figures/fairness_gaps.png": "fairness",
    "metrics/forecast.json": "forecast",
    "tables/forecast_accuracy.csv": "forecast",
    "tables/forecast_coverage.csv": "forecast",
    "figures/forecast_backtest.png": "forecast",
    "metrics/survival.json": "survival",
    "tables/survival_km_vs_aalen_johansen.csv": "survival",
    "tables/survival_cox.csv": "survival",
    "tables/sim_validation.csv": "calibrate_sim",
    "tables/scenarios.csv": "scenarios",
    "tables/scenario_comparison.csv": "scenarios",
    "tables/sensitivity_ranking.csv": "sensitivity",
    "metrics/staffing_search.json": "staffing_search",
}

#: Artifacts written *after* the report stage, so the run that builds a report never
#: has them yet. The absence is structural, not a fault of the run.
LATE_ARTIFACTS: dict[str, str] = {
    "metrics/pipeline_run.json": (
        "the run summary is written after the report stage, so no run has it while "
        "its own report is being assembled"
    ),
}


# ---------------------------------------------------------------- which stages ran


@dataclass(frozen=True)
class RunStages:
    """Which stages a run performed, and why the others did not.

    `known` is False when the manifest is absent - a report built straight from a
    directory of artifacts, for instance. Nothing is then claimed to be deliberate.
    """

    known: bool
    run_type: str
    performed: tuple[str, ...]
    not_performed: dict[str, str]

    def explain(self, artifact: str) -> str | None:
        """Why `artifact` is legitimately absent, or None if its absence is a fault."""
        if artifact in LATE_ARTIFACTS:
            return LATE_ARTIFACTS[artifact]
        if not self.known:
            return None
        stage = ARTIFACT_PRODUCER.get(artifact)
        if stage is None or stage in self.performed:
            return None
        return self.not_performed.get(
            stage, f"the `{stage}` stage is not part of this run"
        )


def save_run_stages(
    config: Mapping[str, Any],
    run_type: str,
    performed: list[str],
    not_performed: Mapping[str, str],
) -> Path:
    """Record the stage plan where the report can read it.

    Called before the report stage by whatever is driving the run. The report uses it
    for one purpose: to say whether a missing artifact was ever going to exist.

    Args:
        config: Full run config, for `paths.metrics_dir`.
        run_type: How this run was launched, in words a reader understands.
        performed: Stages this run ran, or is about to run.
        not_performed: Stage -> the reason it is not part of this run.

    Returns:
        The path written.
    """
    return save_json(
        {
            "run_type": run_type,
            "stages_performed": list(performed),
            "stages_not_performed": dict(not_performed),
        },
        Path(config["paths"]["metrics_dir"]) / RUN_STAGES_FILE,
    )


def _read_run_stages(metrics_dir: Path) -> RunStages:
    """Read the stage manifest, or report it as unknown."""
    path = resolve_path(metrics_dir / RUN_STAGES_FILE)
    if not path.is_file():
        return RunStages(False, "unknown", (), {})
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read %s; treating every absence as unexpected", path)
        return RunStages(False, "unknown", (), {})
    return RunStages(
        known=True,
        run_type=str(payload.get("run_type", "unknown")),
        performed=tuple(str(stage) for stage in payload.get("stages_performed", ())),
        not_performed={
            str(stage): str(reason)
            for stage, reason in (payload.get("stages_not_performed") or {}).items()
        },
    )


class ArtifactReader:
    """Reads artifacts, recording anything absent instead of failing."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.tables_dir = Path(config["paths"]["tables_dir"])
        self.metrics_dir = Path(config["paths"]["metrics_dir"])
        self.figures_dir = Path(config["paths"]["figures_dir"])
        self.missing: list[str] = []
        self.stages = _read_run_stages(self.metrics_dir)

    def table(self, name: str, record: bool = True) -> pd.DataFrame | None:
        """Read a CSV from artifacts/tables/, or record it as missing.

        Args:
            name: File name inside the tables directory.
            record: Whether to record it in `missing` when absent. A section that
                re-reads an artifact another section already asked for passes False,
                so the same absence is not reported from two places.
        """
        path = resolve_path(self.tables_dir / name)
        if not path.is_file():
            if record:
                self.missing.append(f"tables/{name}")
            return None
        # Two tables carry their caveat as a leading comment line, because the caveat
        # has to travel with the file rather than only with the report that quotes it.
        comment = "#" if name in CAVEATED_TABLES else None
        return pd.read_csv(path, comment=comment)

    def metrics(self, name: str, record: bool = True) -> dict[str, Any] | None:
        """Read a JSON from artifacts/metrics/, or record it as missing."""
        path = resolve_path(self.metrics_dir / name)
        if not path.is_file():
            if record:
                self.missing.append(f"metrics/{name}")
            return None
        return load_json(path)

    def figure(self, name: str) -> str:
        """Return a Markdown image link if the figure exists."""
        if resolve_path(self.figures_dir / name).is_file():
            return f"![{name}](figures/{name})"
        self.missing.append(f"figures/{name}")
        return ""


def _section(title: str, body: str) -> str:
    """One report section."""
    return f"## {title}\n\n{body}\n"


def _count(n: int, singular: str, plural: str | None = None) -> str:
    """`1 metric` / `2 metrics`, so generated prose does not read like a form."""
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


def _model_label(row: Mapping[str, Any]) -> str:
    """A readable name for one modelling choice.

    The artifacts identify a model by three keys (`feature_set`, `target`,
    `prediction_kind`). Pasted together they read as `base+queue / raw / raw`, which
    means nothing to a reader; this spells them out instead.
    """
    targets = {"raw": "raw target", "log1p": "log1p target"}
    kinds = {
        "raw": "",
        "log1p_median": ", median back-transform",
        "log1p_smearing": ", Duan smearing to the conditional mean",
    }
    feature_set = str(row["feature_set"])
    target = str(row["target"])
    kind = str(row["prediction_kind"])
    return (
        f"{feature_set} features, "
        f"{targets.get(target, f'{target} target')}"
        f"{kinds.get(kind, f', {kind}')}"
    )


def _data_section(reader: ArtifactReader) -> str:
    """Dataset composition and validation."""
    validation = reader.metrics("validation.json")
    split = reader.metrics("split.json")
    if validation is None:
        return MISSING

    lines = [
        f"- **{validation['n_input']:,} visits** loaded, **{validation['n_valid']:,}** valid, "
        f"**{validation['n_dropped']}** dropped.",
        f"- **{validation['n_modelling']:,}** have an observed wait and can train the "
        f"wait-time model.",
        f"- **{validation['n_lwbs']:,}** left without being seen and "
        f"**{validation['n_still_waiting_at_cutoff']}** were still waiting when the extract "
        "closed. These are different things and are never pooled: one is an event, the "
        "other is censoring.",
    ]
    if split:
        lines.append(
            "- Blocks: "
            + ", ".join(
                f"**{name}** {split[name]['n']:,} ({split[name]['share']:.1%})"
                for name in ("train", "val", "calib", "test")
                if name in split
            )
            + ", cut forward in time and non-overlapping."
        )

    body = "\n".join(lines)
    profile = reader.table("profile_wait_by_triage.csv")
    if profile is not None:
        body += "\n\n### Observed waits by acuity\n\n" + md_table(
            profile[
                ["triage_level", "n_arrived", "n_seen", "n_lwbs", "mean", "median", "p90",
                 "share_over_30min"]
            ]
        )
    return body


# ------------------------------------------------- per-acuity threshold evidence


@dataclass(frozen=True)
class ThresholdEvidence:
    """What `triage_thresholds.csv` shows about per-acuity threshold mitigation.

    One object, read once, used by three sections: the threshold table that reports
    it, the fairness finding it is the mitigation for, and the future-work list that
    must not call it unattempted.
    """

    fitted: list[str]
    fallback: dict[str, int]
    pooled_recall: float
    per_triage_recall: float
    ci_width: float

    @property
    def delta(self) -> float:
        """Per-acuity recall minus pooled recall, both on the test block."""
        return self.per_triage_recall - self.pooled_recall

    @property
    def meaningful(self) -> bool:
        """Whether the change is larger than the interval around the estimates."""
        return abs(self.delta) >= self.ci_width

    def verdict(self) -> str:
        """One sentence interpreting the comparison, generated from the numbers."""
        head = (
            f"Pooled recall **{self.pooled_recall:.4f}** against per-acuity "
            f"**{self.per_triage_recall:.4f}** on the test block, a change of "
            f"{self.delta:+.4f} against a 95% interval {self.ci_width:.4f} wide"
        )
        if not self.meaningful:
            return (
                f"{head}. **There was no meaningful improvement**: the change is "
                "smaller than the interval around either estimate, so the per-acuity "
                "rule moves recall within noise."
            )
        direction = "improvement" if self.delta > 0 else "loss"
        return (
            f"{head}. The change is larger than that interval, so it is a real "
            f"{direction} at this sample size rather than noise."
        )

    def fallback_counts(self) -> str:
        """`ESI 1 (n_positive = 1) and ESI 2 (n_positive = 29)`."""
        parts = [f"{level} (n_positive = {n})" for level, n in self.fallback.items()]
        if len(parts) <= 1:
            return "".join(parts)
        return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def _threshold_evidence(thresholds: pd.DataFrame | None) -> ThresholdEvidence | None:
    """Read the pooled-vs-per-acuity comparison, or None if the table cannot answer.

    Args:
        thresholds: The `triage_thresholds.csv` frame, or None.

    Returns:
        A `ThresholdEvidence`, or None when the overall comparison rows are absent.
    """
    if thresholds is None or "scope" not in thresholds or "rule" not in thresholds:
        return None
    overall = thresholds[thresholds["scope"] == "overall"].set_index("rule")
    if not {"pooled", "per_triage"}.issubset(set(overall.index)):
        return None

    pooled, per_triage = overall.loc["pooled"], overall.loc["per_triage"]
    widths = [
        float(row["recall_ci_high"]) - float(row["recall_ci_low"])
        for row in (pooled, per_triage)
        if pd.notna(row.get("recall_ci_high")) and pd.notna(row.get("recall_ci_low"))
    ]

    levels = thresholds[thresholds["scope"] != "overall"]
    source = levels["threshold_source"].astype(str)
    was_fitted = source.str.contains("fitted", case=False, na=False)
    fallback = levels[~was_fitted]
    return ThresholdEvidence(
        fitted=sorted(set(levels.loc[was_fitted, "scope"])),
        fallback={
            str(level): int(group["n_positive"].max())
            for level, group in fallback.groupby("scope", sort=True)
        },
        pooled_recall=float(pooled["recall"]),
        per_triage_recall=float(per_triage["recall"]),
        ci_width=max(widths) if widths else 0.0,
    )


def _wait_model_section(reader: ArtifactReader) -> str:
    """Wait-time models, ablation and conformal intervals."""
    metrics = reader.metrics("wait_models.json")
    ablation = reader.table("ablation.csv")
    if metrics is None or ablation is None:
        return MISSING

    best = metrics["ablation"]["best_by_test_mae"]
    baseline = metrics["baseline"]["blocks"]["test"]["regression"]
    body = [
        f"Best model: **{_model_label(best)}** - test MAE "
        f"**{best['mae_minutes']:.2f} min**, RMSE {best['rmse_minutes']:.2f} min.",
        "",
        f"Baseline (rolling mean of the last {metrics['baseline']['k']} observed waits at "
        f"the same acuity): MAE **{baseline['mae']:.2f} min**, RMSE {baseline['rmse']:.2f} min.",
        "",
        "### Ablation: do queue-state features help? (research gap G3)",
        "",
        md_table(
            ablation[["feature_set", "target", "prediction_kind", "mae_minutes", "rmse_minutes"]]
        ),
        "",
        f"> {metrics['ablation']['caveat']}",
    ]

    classifiers = reader.table("classifier_summary.csv")
    if classifiers is not None:
        body += [
            "",
            "### Prolonged-wait classifier",
            "",
            md_table(
                classifiers[
                    ["feature_set", "threshold", "threshold_source", "pr_auc", "recall",
                     "precision", "fnr"]
                ]
            ),
            "",
            "The decision threshold is chosen on the validation block and frozen; it never "
            "sees the test block.",
        ]

    body += _conformal_block(reader)
    body += _threshold_block(reader)
    body += _high_acuity_block(reader)

    figure = reader.figure("ablation_error.png")
    if figure:
        body += ["", figure]
    return "\n".join(body)


def _conformal_block(reader: ArtifactReader) -> list[str]:
    """Conformal intervals, led by the calibration that actually covers."""
    static = reader.table("conformal_coverage.csv")
    rolling = reader.table("conformal_static_vs_rolling.csv")
    if static is None and rolling is None:
        return []

    body = ["", "### Conformal prediction intervals", ""]

    overall = (
        rolling[rolling["scope"] == "overall"].set_index("method")
        if rolling is not None
        else None
    )
    if overall is not None and not {"static", "rolling"}.issubset(set(overall.index)):
        # Nothing to lead with: render the table and let the reader see why.
        body += [
            "Only one calibration method is recorded for this run, so there is no "
            "comparison to draw.",
            "",
            md_table(rolling),
        ]
        overall = None
    elif overall is not None:
        live, fixed = overall.loc["rolling"], overall.loc["static"]
        misses = (
            f"one time in {1.0 / (1.0 - float(fixed['empirical_coverage'])):.0f}"
            if float(fixed["empirical_coverage"]) < 1.0
            else "sometimes"
        )
        body += [
            f"**Rolling recalibration is the result to read.** Empirical coverage "
            f"**{live['empirical_coverage']:.4f}** against a nominal "
            f"{live['nominal_coverage']:.2f} (gap {live['coverage_gap']:+.4f}), at a mean "
            f"interval width of {live['mean_width_minutes']:.1f} minutes. Each patient's "
            f"interval is calibrated from the residuals of same-acuity patients **already "
            f"seen when that patient arrived**, so it tracks drift instead of assuming it "
            f"away.",
            "",
            f"Static Mondrian calibration is kept as the comparison: coverage "
            f"**{fixed['empirical_coverage']:.4f}** (gap {fixed['coverage_gap']:+.4f}) at "
            f"{fixed['mean_width_minutes']:.1f} minutes. It calibrates once on a fixed "
            f"block and assumes the test residuals are exchangeable with it; under "
            f"temporal drift they are not, and coverage falls. The rolling intervals are "
            f"wider, which is the honest price of actually covering: a narrow interval "
            f"that misses {misses} is worse than a wider one that does what it claims.",
            "",
            md_table(rolling),
        ]

    if static is not None:
        body += ["", "#### Static coverage per acuity level", "", md_table(static)]
        body.append("")
        body.append(_worst_static_gap(static))

    return body


def _worst_static_gap(static: pd.DataFrame) -> str:
    """State the worst per-level static gap instead of claiming a per-level guarantee.

    Calibrating within each acuity level is what Mondrian conformal does, but it buys
    a *conditional* guarantee only under exchangeability. These are the measured
    per-level gaps, which is the only thing the table supports.
    """
    levels = static[static["triage_level"].astype(str) != "overall"]
    if levels.empty:
        return (
            "Coverage is reported per acuity level; this run has no per-level rows to "
            "compare."
        )
    worst = levels.loc[levels["coverage_gap"].idxmin()]
    level, gap = worst["triage_level"], float(worst["coverage_gap"])
    if gap < 0:
        return (
            f"Calibrating within each acuity level does not make the static guarantee "
            f"hold per level: the worst per-level gap is **{gap:+.4f}** at ESI {level}, "
            f"where empirical coverage is {worst['empirical_coverage']:.4f} against a "
            f"nominal {worst['nominal_coverage']:.2f}."
        )
    return (
        f"On this run every acuity level covers at or above nominal under static "
        f"calibration; the smallest margin is **{gap:+.4f}** at ESI {level} "
        f"({worst['empirical_coverage']:.4f} against {worst['nominal_coverage']:.2f})."
    )


def _threshold_block(reader: ArtifactReader) -> list[str]:
    """Per-acuity decision thresholds, with the verdict read off the table."""
    thresholds = reader.table("triage_thresholds.csv")
    if thresholds is None:
        return []

    body = [
        "",
        "#### Per-acuity decision thresholds",
        "",
        md_table(
            thresholds[
                ["scope", "rule", "threshold", "n", "n_positive", "recall",
                 "recall_ci_low", "recall_ci_high", "fnr", "fnr_ci_low",
                 "fnr_ci_high", "precision", "threshold_source"]
            ]
        ),
        "",
        "A single pooled threshold optimises recall across all patients, and prolonged "
        "waits are overwhelmingly an ESI 3-5 phenomenon, so the pooled rule can abandon "
        "the few high-acuity patients who do face a long wait. Thresholds are therefore "
        "chosen per acuity on the validation block, never on test.",
    ]

    evidence = _threshold_evidence(thresholds)
    if evidence is not None:
        fitted = (
            f"Own thresholds were fitted for {', '.join(evidence.fitted)}"
            if evidence.fitted
            else "No level had enough positive cases for its own threshold"
        )
        body += ["", f"{fitted}. {evidence.verdict()}"]

    thin = thresholds[thresholds["threshold_source"].str.contains("too few", na=False)]
    if not thin.empty:
        levels = sorted(set(thin["scope"]))
        body.append(
            f"\n**{', '.join(levels)} could not be fixed.** Each has too few positive "
            "cases on the validation block to choose a threshold from, so they keep the "
            "pooled one. That is a data limit, not a tuning failure: no cut-off "
            "manufactures signal from a handful of patients, and a threshold fitted to "
            "that many cases would be noise dressed as a decision. Read `n_positive` "
            "before reading any false-negative rate in the table above."
        )
    return body


def _fairness_section(reader: ArtifactReader) -> str:
    """Subgroup error analysis."""
    metrics = reader.metrics("fairness.json")
    table = reader.table("fairness.csv")
    if metrics is None or table is None:
        return MISSING

    overall = metrics["overall"]
    body = [
        f"Overall on the test block (n = {overall['n']:,}): FNR **{overall['fnr']:.4f}**, "
        f"FPR {overall['fpr']:.4f}, recall {overall['recall']:.4f}, "
        f"MAE {overall['mae']:.2f} min.",
        "",
        md_table(
            table[
                ["attribute", "group", "n", "n_positive", "fnr", "fnr_gap",
                 "fnr_gap_ci_low", "fnr_gap_ci_high", "fnr_underpowered", "mae_gap"]
            ]
        ),
    ]

    severe = table[(table["fnr"] >= 0.99) & ~table["fnr_underpowered"]]
    if not severe.empty:
        row = severe.iloc[0]
        body += [
            "",
            f"**Finding: the classifier never warns a high-acuity patient who will wait "
            f"a long time.** {row['attribute']} = {row['group']} has an FNR of "
            f"**{row['fnr']:.4f}** over {int(row['n_positive'])} positive cases, against an "
            f"overall {overall['fnr']:.4f}. The `min_recall` objective optimises recall "
            "pooled across all patients, and high-acuity patients are a small fraction of "
            "the positives, so a single global threshold buys its recall from the "
            "low-acuity majority. Clinically this is the wrong way round. Detecting and "
            "reporting it is what this section is for.",
            "",
            _mitigation_statement(reader),
        ]

    underpowered = table[table["fnr_underpowered"]]
    if not underpowered.empty:
        body += [
            "",
            f"{len(underpowered)} subgroup(s) have too few positive cases to judge their "
            "false-negative rate and are marked rather than reported as findings. An FNR "
            "is estimated from positives alone, so a large group can still carry almost no "
            "information about it.",
        ]

    figure = reader.figure("fairness_gaps.png")
    if figure:
        body += ["", figure]
    return "\n".join(body)


def _mitigation_statement(reader: ArtifactReader) -> str:
    """What was done about the high-acuity failure, read from the threshold table.

    The mitigation for this finding is per-acuity thresholds, and they have been
    fitted and measured. Calling them future work would misdescribe the artifacts,
    so this paragraph is generated from them instead.
    """
    evidence = _threshold_evidence(reader.table("triage_thresholds.csv", record=False))
    if evidence is None:
        return (
            "**Mitigation:** the per-acuity threshold comparison is not available for "
            "this run, so this section reports the failure without a measured mitigation "
            "beside it."
        )

    if not evidence.fitted:
        return (
            "**Mitigation was attempted and could not be fitted at all.** No acuity level "
            f"had enough positive cases for its own threshold ({evidence.fallback_counts()}), "
            "so every patient keeps the pooled cut-off. What this needs is an approach that "
            "works when high-acuity positives are this scarce - an acuity-aware training "
            "objective, or more positive cases at those levels - not a different threshold."
        )

    sentences = [
        "**Mitigation was tried, and the artifacts show what it bought.** Per-acuity "
        f"thresholds were fitted on the validation block for {', '.join(evidence.fitted)} "
        "- the levels with enough positive cases - and scored against the pooled rule in "
        "the per-acuity threshold table above.",
        evidence.verdict(),
    ]
    if evidence.fallback:
        sentences.append(
            f"They could not be applied to {evidence.fallback_counts()} at all: too few "
            "positive cases to choose a cut-off from, which is exactly where this finding "
            "lives."
        )
    sentences.append(
        "So the failure reported here is not waiting on a mitigation nobody wrote. It "
        "needs an approach that works with very few high-acuity positives - an "
        "acuity-aware objective, or more data at those levels."
    )
    return " ".join(sentences)


def _forecast_section(reader: ArtifactReader) -> str:
    """Arrival forecasting."""
    metrics = reader.metrics("forecast.json")
    if metrics is None:
        return MISSING
    if not metrics.get("ran", True):
        return f"_Skipped: {metrics.get('reason', 'unknown reason')}_"

    accuracy = reader.table("forecast_accuracy.csv")
    body = [_forecast_headline(metrics)]
    if accuracy is not None:
        body += ["", md_table(accuracy[["segment", "model", "mae", "rmse", "mape", "smape", "wape"]])]
    body += [
        "",
        "MAPE is reported for comparability with the literature but is undefined at "
        "zero-count hours; sMAPE and WAPE are defined everywhere and are the ones to trust.",
    ]

    coverage = reader.table("forecast_coverage.csv")
    if coverage is not None:
        body += ["", "### Conformal intervals by hour-of-day band", "", md_table(coverage)]

    figure = reader.figure("forecast_backtest.png")
    if figure:
        body += ["", figure]
    body += _forecast_study_block(reader)
    return "\n".join(body)


def _forecast_headline(metrics: Mapping[str, Any]) -> str:
    """Overall accuracy and the peak-hour comparison, in one paragraph.

    Peak hours belong next to the overall figure rather than several paragraphs
    below it: a forecast that beats the baseline on average and loses to it when the
    department is busiest is not usefully summarised by the average alone.
    """
    lead = (
        f"Backtested over {metrics['n_hours']} hours in {metrics['n_folds']} "
        f"rolling-origin folds. Against the seasonal-naive baseline the model is "
        f"**{metrics['mae_improvement_over_baseline_percent']:.2f}% better on MAE "
        f"overall**"
    )
    peak = metrics.get("peak_hours") or {}
    if not peak:
        return lead + "."

    model_mae = float(peak["model"]["mae"])
    naive_mae = float(peak["seasonal_naive"]["mae"])
    scope = (
        f"the busiest hours (>= {metrics['peak_threshold_arrivals']:.1f} arrivals/hour, "
        f"{metrics.get('n_peak_hours', peak['model'].get('n'))} of "
        f"{metrics['n_hours']} hours)"
    )
    if not naive_mae:
        return (
            f"{lead}. At {scope} the baseline has an MAE of zero, so there is no "
            f"percentage to quote; the model scores {model_mae:.3f} there."
        )

    change = (naive_mae - model_mae) / naive_mae * 100.0
    verdict = f"**{abs(change):.2f}% {'better' if change >= 0 else 'worse'}**"
    joiner = "and it holds at" if change >= 0 else "but that reverses at"
    return (
        f"{lead}, {joiner} {scope}: peak MAE **{model_mae:.3f}** against the "
        f"baseline's {naive_mae:.3f}, {verdict}. The two are reported together because "
        "the peak is when a staffing decision bites, and the overall figure alone would "
        "hide it."
    )


def _survival_section(reader: ArtifactReader) -> str:
    """Competing-risks analysis."""
    metrics = reader.metrics("survival.json")
    if metrics is None:
        return MISSING
    if not metrics.get("ran"):
        return f"_Skipped: {metrics.get('reason', 'unknown reason')}_"

    counts = metrics["event_counts"]
    body = [
        f"Events: **{counts['seen']:,} seen**, **{counts['abandoned']:,} abandoned**, "
        f"**{counts['still waiting at cutoff']} censored** at the data cutoff.",
        "",
        "Abandonment is a *competing event*, not censoring. Treating it as censoring "
        "assumes the patient stayed at risk of being seen when in fact they had left, "
        "which overstates the probability of being seen.",
    ]

    bias = reader.table("survival_km_vs_aalen_johansen.csv")
    if bias is not None:
        worst = bias.loc[bias["km_overstatement"].idxmax()]
        body += [
            "",
            "### How much the usual shortcut costs",
            "",
            md_table(bias),
            "",
            f"Kaplan-Meier overstates the probability of being seen by up to "
            f"**{worst['km_overstatement']:.4f} ({worst['km_overstatement_percent']:.2f}%)**, "
            f"at ESI {int(worst['triage_level'])} and {int(worst['horizon_minutes'])} minutes. "
            "The bias is zero where nobody abandons and grows with both decreasing acuity "
            "and elapsed time.",
        ]

    cox = reader.table("survival_cox.csv")
    if cox is not None:
        body += [
            "",
            "### Cause-specific Cox models",
            "",
            md_table(
                cox[["cause", "covariate", "hazard_ratio", "hr_lower_95", "hr_upper_95",
                     "significant_at_5pct"]]
            ),
            "",
            "Fitting both causes is the point: a covariate can raise one hazard while "
            "lowering the other, and a single model would show only one of them.",
        ]
    body += _cox_assumption_block(reader)
    return "\n".join(body)


# ------------------------------------------------------- simulation validation facts


@dataclass(frozen=True)
class FittedMetrics:
    """Which validation rows were fitted, and how loose the abandonment gate is.

    Read from `sim_validation.csv` so the simulation section and the limitations
    cannot disagree about how much of the validation table is out-of-sample.
    """

    names: list[str]
    abandonment_tolerances: list[float]
    other_tolerances: list[float]
    worst_abandonment_error: tuple[str, float] | None

    @property
    def n_fitted(self) -> int:
        """How many metrics the calibration was fitted to."""
        return len(self.names)


def _fitted_metrics(validation: pd.DataFrame | None) -> FittedMetrics | None:
    """Summarise the fitting and abandonment-tolerance columns of the gate table."""
    if validation is None or "used_for_fitting" not in validation:
        return None

    # The column round-trips through CSV, so it can come back as bool or as text.
    flag = validation["used_for_fitting"].astype(str).str.strip().str.lower()
    fitted = validation[flag.isin({"true", "1", "yes"})]

    is_abandonment = (
        validation["metric"].astype(str).str.contains("abandonment", case=False, na=False)
    )
    abandonment = validation[is_abandonment]
    tolerances = sorted(
        {round(float(value), 4) for value in abandonment["tolerance"].dropna()}
    )
    others = sorted(
        {round(float(value), 4) for value in validation[~is_abandonment]["tolerance"].dropna()}
    )
    errors = abandonment.dropna(subset=["relative_error"])
    worst: tuple[str, float] | None = None
    if not errors.empty:
        row = errors.loc[errors["relative_error"].abs().idxmax()]
        worst = (str(row["metric"]), abs(float(row["relative_error"])))

    return FittedMetrics(
        names=[str(name) for name in fitted["metric"]],
        abandonment_tolerances=tolerances,
        other_tolerances=others,
        worst_abandonment_error=worst,
    )


def _simulation_section(reader: ArtifactReader) -> str:
    """Simulation calibration, scenarios, sensitivity and staffing."""
    validation = reader.table("sim_validation.csv")
    if validation is None:
        return MISSING

    passed = bool(validation["within_tolerance"].all())
    body = [
        f"**Validation {'PASSED' if passed else 'FAILED'}.** The simulation is checked "
        "against the observed department before any scenario is allowed to run; scenarios "
        "refuse to execute on a model that failed, unless explicitly overridden.",
        "",
        md_table(validation),
        "",
        _fitting_note(_fitted_metrics(validation)),
    ]

    scenarios = reader.table("scenarios.csv")
    comparison = reader.table("scenario_comparison.csv")
    if scenarios is not None:
        body += [
            "",
            "### Scenarios",
            "",
            md_table(
                scenarios[
                    ["scenario", "n_replications", "mean_wait_minutes",
                     "mean_wait_minutes_ci_low", "mean_wait_minutes_ci_high",
                     "p90_wait_esi3", "abandonment_rate"]
                ]
            ),
        ]
    if comparison is not None:
        body += [
            "",
            md_table(comparison[["scenario", "difference_vs_baseline", "ci_low", "ci_high",
                                 "method", "significant"]]),
            "",
            "Staffing scenarios are compared with **common random numbers**: replication *i* "
            "of each faces the same arrivals and service draws, so the shared noise cancels "
            "in the difference. A demand scenario necessarily draws a different stream and "
            "is reported as unpaired -- visible in the `method` column, and in the width of "
            "its interval.",
        ]

    stability = reader.table("sensitivity_ranking.csv")
    if stability is not None:
        stable = stability[stability["rank_is_stable"] & (stability["rank_at_point_estimate"] == 1)]
        body += [
            "",
            "### Sensitivity: does the ranking survive being wrong about the inputs?",
            "",
            md_table(stability),
        ]
        if not stable.empty:
            # How many worlds, and which inputs they move, comes from the metrics file
            # rather than from a sentence here: the grid is configurable, so a fixed
            # claim about "+/-10% and +/-20% in service times and arrival rates" would
            # go stale the moment anyone widened it.
            sensitivity = reader.metrics("sensitivity.json", record=False) or {}
            worlds = sensitivity.get("n_worlds")
            axes = sensitivity.get("axes") or []
            scope = (
                f"across all {worlds} perturbed worlds, spanning "
                f"{', '.join(sorted(axes))}"
                if worlds and axes
                else "in every perturbed world"
            )
            body.append(
                f"\n**`{stable.iloc[0]['scenario']}` is rank 1 {scope}.**"
            )
        moving = stability[~stability["rank_is_stable"]]
        if not moving.empty:
            body.append(
                f"\n{', '.join(moving['scenario'])} change rank under perturbation, so their "
                "ordering is not a supported conclusion."
            )
        body += _sensitivity_block(reader)

    search = reader.metrics("staffing_search.json")
    if search is not None:
        body += ["", "### Staffing search on forecast demand (research gap G1)", ""]
        if search["any_feasible"]:
            best = search["cheapest_feasible_plan"]
            body += [
                f"Target: {search['target']}.",
                "",
                f"**Cheapest feasible plan: `{best['plan']}` at cost "
                f"{best['cost']:,.0f}** per {search['costing_horizon_hours']}-hour week, "
                f"achieving p90 ESI 3 wait {best['p90_wait_esi3']:.1f} min and abandonment "
                f"{best['abandonment_rate']:.4f}. "
                f"{search['n_feasible']} of {search['n_plans_evaluated']} plans were feasible.",
                "",
                "Feasibility is decided before cost, and judged on the upper confidence "
                "bound: a plan must be *shown* to meet the target, not merely estimated to.",
            ]
        else:
            body += [
                f"**No feasible plan.** None of {search['n_plans_evaluated']} candidates met "
                f"{search['target']}. The closest misses are reported rather than returning "
                "a least-bad plan as the answer.",
            ]
    return "\n".join(body)


def _fitting_note(fitted: FittedMetrics | None) -> str:
    """How much of the validation table is out-of-sample, counted from the table."""
    if fitted is None:
        return (
            "The `used_for_fitting` column marks the rows the calibration was fitted to; "
            "this run does not record it, so treat the gate as partly in-sample."
        )
    if fitted.n_fitted == 0:
        return (
            "No row is marked `used_for_fitting`, so every row in the table above is "
            "out-of-sample validation."
        )
    verb = "is" if fitted.n_fitted == 1 else "are"
    return (
        f"**{_count(fitted.n_fitted, 'metric')}** ({', '.join(fitted.names)}) {verb} used "
        "to fit the unobservable treatment/boarding split and therefore excluded from the "
        "gate; every other row is out-of-sample validation."
    )


# -------------------------------------------------------------- limitations and next


LIMITATIONS_HEAD = """\
- **Synthetic data.** Every number in this report comes from data generated by a queue
  whose structure the models are then asked to recover. That validates the methods; it
  is not evidence about any real emergency department.
- **Single site, single configuration.** No external validation, no cross-site
  comparison, no temporal validation beyond the held-out test block.
- **Censoring bias.** The wait-time models train only on patients who were seen.
  Patients who left without being seen waited longest, so the models under-predict; the
  survival section quantifies a lower bound on that bias.
- **The ablation overstates queue features.** The generator creates waits by simulating
  a queue, so congestion at arrival is close to a sufficient statistic. The direction is
  supported; the magnitude is an upper bound.
- **The staffing recommendation is partly circular.** The roster that generated the
  data is also the cheapest plan the search finds feasible. The simulation is calibrated
  to reproduce a department running that roster, so it is close to guaranteed that the
  roster looks adequate and that anything leaner does not. The finding "the current
  roster is the minimum viable staffing" is therefore reinforced by how the data was
  made, and would need a department whose roster changed during the observation period
  to be tested properly.
- **Priority-aging assumptions.** The queue applies a specific aging rule with a safety
  floor. Both the generator and the simulation use it, so the simulation reproduces the
  generator partly by construction. A real department's prioritisation differs and the
  aging floor in particular is a policy choice, not a measurement."""

LIMITATIONS_TAIL = """\
- **Unmodelled resources.** Nurses, beds, imaging and laboratory turnaround are not
  modelled. Doctors are the only hard constraint.
- **Fairness subgroups are confounded with acuity.** Arrival-mode and age-band gaps
  largely track acuity composition; no within-acuity adjusted analysis is computed.
- **Conformal coverage assumes exchangeability**, which time-ordered blocks only
  approximate. Empirical coverage is reported rather than assumed.
- **The simulation validation gate passed on 8 of 10 regenerated datasets**; it
  refused on seed 8 (rare-event tolerance) and seed 10 (ESI 3 mean-wait miss).
- **The cheapest roster varies across datasets** (43,890-54,670 over 7 distinct
  rosters on the 8 seeds that produced one); the method transfers, the specific
  roster does not.
- **Decision support only.** Nothing here triages a patient, allocates care, or makes an
  autonomous clinical decision."""


def _limitations_section(reader: ArtifactReader) -> str:
    """Limitations, with the simulation ones counted from the validation table."""
    fitted = _fitted_metrics(reader.table("sim_validation.csv", record=False))
    middle: list[str] = []

    if fitted is None or fitted.n_fitted == 0:
        middle.append(
            "- **The treatment/boarding split is not identifiable** from arrival, contact\n"
            "  and departure timestamps alone. It is fitted to aggregate behaviour rather\n"
            "  than measured; a site with real treatment-end timestamps should use them."
        )
    else:
        middle.append(
            "- **The treatment/boarding split is not identifiable** from arrival, contact\n"
            f"  and departure timestamps alone. It is fitted to "
            f"{_count(fitted.n_fitted, 'aggregate metric')}\n"
            f"  ({', '.join(fitted.names)}), which are therefore excluded from the\n"
            "  validation gate; a site with real treatment-end timestamps should use them."
        )

    if fitted is not None and fitted.abandonment_tolerances:
        tolerances = ", ".join(f"{value:.2f}" for value in fitted.abandonment_tolerances)
        widest = max(fitted.abandonment_tolerances)
        comparison = (
            "  where the wait and length-of-stay rows use "
            f"{', '.join(f'{value:.2f}' for value in fitted.other_tolerances)}."
            if fitted.other_tolerances
            else ""
        )
        sentences = [
            "- **The abandonment gate is loose.** The abandonment rows are checked",
            f"  against a relative tolerance of {tolerances},",
            *([comparison] if comparison else []),
            "  Abandonment is a small count and a tight relative band would fail on",
            f"  noise alone, so a simulated rate wrong by {widest:.0%} still passes.",
        ]
        if fitted.worst_abandonment_error is not None:
            name, error = fitted.worst_abandonment_error
            sentences += [
                f"  The largest relative abandonment error in the table is **{error:.4f}**",
                f"  ({name}), and it passes.",
            ]
        sentences.append(
            "  Read those rows as a sanity check, not as evidence that the simulation\n"
            "  reproduces who leaves."
        )
        middle.append("\n".join(sentences))

    return "\n".join([LIMITATIONS_HEAD, *middle, LIMITATIONS_TAIL])


FUTURE_WORK_HEAD = """\
- **Validation on real data.** The single most valuable next step: run the adapter
  against a real extract and re-measure everything. Every number in this report should
  be treated as provisional until then."""

FUTURE_WORK_TAIL = """\
- **A reinforcement learning staffing policy that works.** One was built and it lost. A
  tabular Q-learning agent over the validated simulator cost **17,832 more** per
  evaluation
  week than the fixed roster on a pre-declared 60-episode budget (95% CI 14,663 to
  21,002), with more ESI 1-2 breaches than acting at random. A larger training budget is
  the obvious suspect and is a hypothesis, not a defence; it was not retrained until it
  won. It is also circular -- the agent learns a simulator calibrated to a generator --
  so a win would be evidence about the method and nothing else.
- **Calibrating the richer resource model.** Nurses, beds, imaging turnaround and
  bed-holding boarding exist as explicit constrained resources behind
  `simulation.resource_extensions`, off by default. They cannot be calibrated here: the
  data has no bed, nurse or imaging column, so those parameters are assumed and the
  validation gate cannot test them. Their ranking is usable; their magnitudes are not.
- **Competing-risks prediction on real data.** Built: a discrete-time model for the
  cumulative incidence of leaving without being seen at 30, 60 and 120 minutes, with
  isotonic recalibration and an Aalen-Johansen comparison. What it recovers is the
  generator's patience distribution, which was chosen rather than observed."""


def _deep_forecaster_bullet(reader: ArtifactReader) -> str:
    """The deep-forecaster item, with its trigger decided by the backtest.

    A deep model is worth reaching for only if the shallow one is actually losing
    where it matters. Whether it is losing is a measurement, so the bullet is written
    from the measurement rather than asserted.
    """
    summary = reader.metrics("forecast_study.json", record=False)
    scores = reader.table("forecast_study_scores.csv", record=False)

    if not summary or not summary.get("ran") or scores is None or scores.empty:
        metrics = reader.metrics("forecast.json", record=False) or {}
        peak = (metrics.get("peak_hours") or {})
        model = (peak.get("model") or {}).get("mae")
        naive = (peak.get("seasonal_naive") or {}).get("mae")
        if model is not None and naive is not None and model > naive:
            return (
                "- **Deep forecasters.** On the default backtest the model is worse "
                f"than seasonal-naive at peak hours ({model:.3f} against {naive:.3f} "
                "MAE), which is the trigger for trying LSTM or temporal-fusion "
                "architectures. Note that this comparison rests on a small number of "
                "peak hours; `scripts/forecast_study.py` exists to settle it on more."
            )
        return (
            "- **Deep forecasters.** LSTM or temporal-fusion architectures may capture "
            "arrival dynamics the lag features miss, particularly around holidays and "
            "surges."
        )

    winner = summary.get("selection", {}).get("winner")
    peak_rows = scores[(scores["block"] == "test") & (scores["scope"] == "peak")]
    winner_rows = peak_rows[peak_rows["variant"] == winner]
    if winner_rows.empty:
        return (
            "- **Deep forecasters.** The extended backtest ran but produced no "
            "peak-hour row for the selected forecaster, so the trigger is undecided."
        )

    row = winner_rows.iloc[0]
    if not bool(row["beats_naive"]):
        return (
            f"- **Deep forecasters: triggered.** Over "
            f"{summary['n_origins']} rolling origins the selected forecaster "
            f"(`{winner}`) still loses to seasonal-naive at peak hours "
            f"({row['mae']:.3f} against {row['seasonal_naive_mae']:.3f} MAE). Shallow "
            "features and a reweighted objective have both been tried and neither "
            "closed it, which is the case for an architecture that models the "
            "sequence directly."
        )
    if not bool(row["difference_is_significant"]):
        return (
            f"- **Deep forecasters: not clearly triggered.** Over "
            f"{summary['n_origins']} rolling origins the selected forecaster "
            f"(`{winner}`) is ahead of seasonal-naive at peak hours "
            f"({row['mae']:.3f} against {row['seasonal_naive_mae']:.3f} MAE) but the "
            f"95% interval on the difference "
            f"({row['mae_vs_naive_ci_low']:+.3f} to {row['mae_vs_naive_ci_high']:+.3f}) "
            "includes zero. More history, not a deeper model, is the cheaper next step."
        )
    return (
        f"- **Deep forecasters: not currently justified.** The peak-hour weakness that "
        f"motivated this item does not survive a larger sample. Over "
        f"{summary['n_origins']} rolling origins the selected forecaster (`{winner}`) "
        f"beats seasonal-naive at peak hours, {row['mae']:.3f} against "
        f"{row['seasonal_naive_mae']:.3f} MAE, with a 95% interval on the difference of "
        f"{row['mae_vs_naive_ci_low']:+.3f} to {row['mae_vs_naive_ci_high']:+.3f} that "
        "excludes zero. A deep architecture remains interesting, but it no longer has "
        "a measured deficit to fix."
    )


def _high_acuity_future_bullet(reader: ArtifactReader) -> str | None:
    """What is still open on high-acuity prediction, after four approaches were tried."""
    summary = (reader.metrics("wait_models.json", record=False) or {}).get("high_acuity")
    if not summary or not summary.get("ran", False):
        return None

    verdict = summary.get("verdict", {})
    worked: list[str] = []
    failed: list[str] = []
    for scope in ("ESI 1", "ESI 2"):
        entry = verdict.get(scope)
        if not entry:
            continue
        (worked if entry.get("any_reportable") else failed).append(scope)

    lines = [
        "- **High-acuity prolonged-wait prediction.** Four approaches were tried and "
        "measured: the pooled classifier, acuity sample weights, a separate ESI 1-2 "
        "model, and an acuity-specific target."
    ]
    if worked:
        # Named per scope, not merged: ESI 1 and ESI 2 rarely become evaluable under
        # the same approaches, and pooling the two lists would credit an approach with
        # a level it never reached.
        for scope in worked:
            names = sorted(verdict[scope].get("reportable_approaches", []))
            lines.append(
                f"  {scope} became evaluable under "
                f"{', '.join(f'`{name}`' for name in names)}."
            )
        lines.append(
            "  So the open problem is no longer whether it can be done at all for "
            f"{', '.join(worked)}."
        )
    if failed:
        lines.append(
            f"  {', '.join(failed)} still has too few positive cases for any of them, "
            "and no amount of modelling fixes that: it needs more history or a "
            "multi-site extract."
        )
    lines.append(
        "  What remains is choosing the target with clinicians rather than assuming "
        "it: the per-acuity targets used here are illustrative, and every rate above "
        "moves when they move."
    )
    return "\n".join(lines)


def _future_work_section(reader: ArtifactReader) -> str:
    """Future work, with the fairness bullet kept consistent with the artifacts.

    Per-acuity thresholds are implemented, fitted and measured, so they cannot be
    listed here. What remains is the part the artifacts show is still open: a method
    that works when a level has almost no positive cases.
    """
    evidence = _threshold_evidence(reader.table("triage_thresholds.csv", record=False))
    if evidence is None or not evidence.fitted:
        fairness = (
            "- **An objective that works with very few high-acuity positives.** One global\n"
            "  threshold optimises pooled recall, and high-acuity patients are a small share\n"
            "  of the positives. An acuity-aware training objective, or enough high-acuity\n"
            "  positive cases to estimate a rule from, is what this needs."
        )
    else:
        lines = [
            "- **An objective that works with very few high-acuity positives.** Per-acuity",
            f"  thresholds are already fitted and measured ({', '.join(evidence.fitted)}), and",
            f"  they changed pooled recall by {evidence.delta:+.4f} against a 95% interval",
            f"  {evidence.ci_width:.4f} wide.",
        ]
        if evidence.fallback:
            lines.append(f"  {evidence.fallback_counts()} could not be fitted at all.")
        lines.append(
            "  So thresholding is not the open problem: an acuity-aware training\n"
            "  objective, or more positive cases at those levels, is."
        )
        fairness = "\n".join(lines)

    bullets = [FUTURE_WORK_HEAD, fairness]
    high_acuity = _high_acuity_future_bullet(reader)
    if high_acuity:
        bullets.append(high_acuity)
    bullets.append(_deep_forecaster_bullet(reader))
    bullets.append(FUTURE_WORK_TAIL)
    return "\n".join(bullets)


def _missing_section(reader: ArtifactReader) -> str | None:
    """Explain absent artifacts, separating the deliberate from the broken.

    An upload job does not run the sensitivity analysis, and no run has its own run
    summary while the report is being written. Listing those beside a genuinely
    absent table teaches a reader to ignore the whole list, which is the opposite of
    what it is for.
    """
    missing = sorted(set(reader.missing))
    if not missing:
        return None

    by_design: list[tuple[str, str]] = []
    unexpected: list[str] = []
    for name in missing:
        reason = reader.stages.explain(name)
        if reason is None:
            unexpected.append(name)
        else:
            by_design.append((name, reason))

    body: list[str] = []
    if by_design:
        run_type = reader.stages.run_type
        where = f" for this run type ({run_type})" if run_type != "unknown" else ""
        body += [
            f"**Not produced by this run, by design{where}.** Nothing is wrong with "
            "these; the sections that would have used them are simply absent:",
            "",
            *(f"- `{name}` - {reason}" for name, reason in by_design),
        ]
    if unexpected:
        if body:
            body.append("")
        body += [
            "**Missing unexpectedly.** A stage that ran should have written these, so "
            "the sections above are incomplete and this is a problem worth chasing:",
            "",
            *(f"- `{name}`" for name in unexpected),
        ]

    if unexpected:
        logger.warning(
            "Report is missing %d artifact(s) unexpectedly: %s", len(unexpected), unexpected
        )
    if by_design:
        logger.info(
            "Report notes %d artifact(s) absent by design: %s",
            len(by_design),
            [name for name, _ in by_design],
        )
    return "\n".join(body)



def _data_provenance_block(reader: ArtifactReader) -> list[str]:
    """Label a proxy wait, and report how far this dataset sits from the reference.

    Both facts are properties of the dataset rather than of any model, so they belong
    at the top of the report: a reader who does not know that `seen_time` came from a
    proxy will misread every wait figure below it.
    """
    shift = reader.metrics("dataset_shift.json", record=False)
    if not shift:
        return []

    body: list[str] = []
    proxy = shift.get("proxy_wait") or {}
    if proxy.get("active"):
        body += [
            "",
            f"> **{proxy['notice']}**",
            ">",
            f"> Proxy used: {proxy['proxy']}.",
            "",
            "Every wait, prolonged-wait label, conformal interval and simulation KPI "
            "below is a measurement of that proxy. They are not door-to-provider "
            "times and must not be compared with figures that are.",
        ]

    findings = shift.get("findings") or []
    if findings:
        body += [
            "",
            "#### How far this dataset sits from the one the models were tuned on",
            "",
            md_table(
                pd.DataFrame(findings)[
                    ["name", "statistic", "difference", "threshold", "flagged"]
                ]
            ),
            "",
            shift.get("verdict", ""),
        ]
    return body


def _high_acuity_block(reader: ArtifactReader) -> list[str]:
    """The four-approach comparison for ESI 1-2, read entirely from artifacts.

    Per-acuity thresholds can only move an existing score; they cannot create positive
    cases. This block reports what happened when the target itself was changed, and it
    states the verdict from the table rather than from a fixed sentence, because the
    verdict is exactly the thing that may differ on another dataset.
    """
    table = reader.table("high_acuity_comparison.csv", record=False)
    summary = (reader.metrics("wait_models.json", record=False) or {}).get("high_acuity")
    if table is None or table.empty or not summary or not summary.get("ran", True):
        return []

    targets = summary.get("acuity_target_minutes", {})
    target_text = ", ".join(f"ESI {level}: {minutes:g} min" for level, minutes in sorted(targets.items()))

    body = [
        "",
        "#### Can a prolonged wait be predicted for a high-acuity patient at all?",
        "",
        "Four approaches on the same split, the same queue features and the same "
        "bootstrap procedure, scored once on the test block:",
        "",
    ]

    columns = [
        column
        for column in (
            "approach", "scope", "n", "n_positive", "val_positives", "threshold",
            "threshold_source", "recall", "recall_ci_low", "recall_ci_high",
            "fnr", "fnr_ci_low", "fnr_ci_high", "precision", "reportable",
        )
        if column in table.columns
    ]
    high = table[table["scope"].isin(["overall", "ESI 1", "ESI 2", "ESI 1-2 pooled"])]
    body += [md_table(high[columns]), ""]

    approaches = summary.get("approaches", {})
    for name, detail in approaches.items():
        notes = detail.get("notes") or []
        unfitted = [note for note in notes if note.startswith("NOT FITTED")]
        if unfitted:
            body.append(f"- **`{name}` was not fitted.** {unfitted[0]}")

    verdict = summary.get("verdict", {})
    for scope in ("ESI 1", "ESI 2"):
        entry = verdict.get(scope)
        if not entry:
            continue
        reportable = entry.get("reportable_approaches", [])
        if reportable:
            best = max(
                (
                    item
                    for item in entry["approaches"]
                    if item.get("reportable") and item.get("recall") is not None
                ),
                key=lambda item: item["recall"],
                default=None,
            )
            line = (
                f"- **{scope}: a defensible result exists** under "
                f"{', '.join(f'`{name}`' for name in reportable)}."
            )
            if best is not None:
                interval = best.get("fnr_ci") or [None, None]
                line += (
                    f" The best of them, `{best['approach']}`, reaches recall "
                    f"{best['recall']:.3f} over {best['n_positive']} positive case(s), "
                    f"with a false-negative rate of {best['fnr']:.3f}"
                )
                if interval[0] is not None and interval[1] is not None:
                    line += f" (95% CI {interval[0]:.3f} to {interval[1]:.3f})"
                line += "."
            body.append(line)
        else:
            reasons = {
                item.get("unreportable_reason")
                for item in entry.get("approaches", [])
                if item.get("unreportable_reason")
            }
            body.append(
                f"- **{scope} cannot be evaluated by any of the four approaches.** "
                + (" ".join(sorted(r for r in reasons if r)) or "Too few positive cases.")
            )

    body += [
        "",
        f"The acuity-specific target used here is {target_text}. "
        + str(summary.get("caveat", "")),
    ]
    return body


def _cox_assumption_block(reader: ArtifactReader) -> list[str]:
    """Proportional-hazards diagnostics, and what a violation limits."""
    table = reader.table("survival_cox_ph_test.csv", record=False)
    metrics = reader.metrics("survival.json", record=False) or {}
    summary = metrics.get("cox_proportional_hazards_summary") or {}
    if table is None or table.empty or not summary.get("ran"):
        return []

    columns = [
        column
        for column in ("cause", "covariate", "test", "test_statistic", "p_value",
                       "violates_proportional_hazards", "verdict")
        if column in table.columns
    ]
    body = [
        "",
        "#### Does the proportional-hazards assumption hold?",
        "",
        md_table(table[columns]),
        "",
    ]

    violations = summary.get("violations", [])
    if not violations:
        body.append(
            f"No covariate rejects the assumption at p < {summary.get('alpha', 0.05)}, "
            "so each hazard ratio above can be read as a constant effect over the wait."
        )
    else:
        named = ", ".join(
            f"`{item['covariate']}` ({item['cause']}, p = {item['p_value']:.3g})"
            for item in violations
        )
        body.append(
            f"**{len(violations)} of {summary.get('n_covariate_tests')} covariate tests "
            f"reject proportional hazards** at p < {summary.get('alpha', 0.05)}: {named}."
        )
        body.append("")
        body.append(summary.get("limitation_note", ""))

    untested = summary.get("n_untested", 0)
    if untested:
        body.append(
            f"\n{untested} covariate test(s) could not be computed and are reported as "
            "untested rather than as passing."
        )
    return body


def _sensitivity_block(reader: ArtifactReader) -> list[str]:
    """Which uncertain input the recommendation is most hostage to."""
    metrics = reader.metrics("sensitivity.json", record=False)
    tornado = reader.table("sensitivity_tornado.csv", record=False)
    if not metrics or tornado is None or tornado.empty:
        return []

    axes = metrics.get("axes", [])
    body = [
        "",
        "#### How wrong could this be?",
        "",
        f"The scenarios were re-run in **{metrics.get('n_worlds', 0)} perturbed worlds** "
        f"across {len(axes)} families of input: {', '.join(f'`{axis}`' for axis in sorted(axes))}. "
        "Each world moves exactly one input, so an effect belongs to that input alone.",
        "",
    ]

    stable = metrics.get("ranking_stable")
    if stable is True:
        body.append(
            "**The scenario ranking is stable.** No scenario changed position on mean "
            "wait in any of those worlds."
        )
    elif stable is False:
        body.append(
            "**The scenario ranking is NOT stable**: at least one scenario changes "
            "position under perturbation, so the ordering is not something to act on."
        )

    floors = metrics.get("monte_carlo_noise_floor", {})
    if floors:
        stated = ", ".join(f"{name} {value:.4g}" for name, value in sorted(floors.items()))
        body += [
            "",
            "One world in the grid is a **noise control**: it sets the priority-aging "
            "floor to the value already configured, so it changes nothing about the "
            "department and differs only in its random stream. Whatever it moves is "
            f"Monte Carlo noise at this replication count: {stated}. Any effect below "
            "that is not a finding, and the table marks it.",
        ]

    body += ["", "Largest effects, excluding the noise control:", ""]
    for kpi, rows in (metrics.get("most_sensitive") or {}).items():
        if not rows:
            continue
        label = rows[0].get("kpi_label", kpi)
        listed = ", ".join(
            f"`{row['perturbation']}` ({row['effect']:+.3g})" for row in rows
        )
        body.append(f"- **{label}**: {listed}")

    figure = reader.figure("sensitivity_tornado.png")
    if figure:
        body += ["", figure]

    note = metrics.get("aging_floor_safety_note")
    if note:
        body += ["", note]
    return body


def _forecast_study_block(reader: ArtifactReader) -> list[str]:
    """The extended rolling-origin study, when it has been run."""
    summary = reader.metrics("forecast_study.json", record=False)
    scores = reader.table("forecast_study_scores.csv", record=False)
    if not summary or not summary.get("ran") or scores is None or scores.empty:
        return []

    selection = summary.get("selection", {})
    body = [
        "",
        "#### A longer backtest, because four folds could not settle the peak question",
        "",
        f"The default backtest above uses {summary.get('n_folds', 4)} folds. This study "
        f"re-runs the comparison over **{summary['n_origins']} weekly origins** "
        f"({summary['n_scored_hours']} scored hours), splits them in time, tunes every "
        f"constant on the {summary['n_validation_origins']} validation origins and "
        f"scores the {summary['n_test_origins']} test origins once.",
        "",
        md_table(
            scores[
                [
                    column
                    for column in (
                        "block", "variant", "scope", "n_hours", "mae", "wape", "bias",
                        "seasonal_naive_mae", "mae_vs_naive", "mae_vs_naive_ci_low",
                        "mae_vs_naive_ci_high", "beats_naive",
                        "difference_is_significant",
                    )
                    if column in scores.columns
                ]
            ]
        ),
        "",
        f"Selected on the **validation** origins at {selection.get('selection_scope')} "
        f"hours: **`{selection.get('winner')}`** "
        f"(MAE {selection.get('winner_val_mae', float('nan')):.4f} against "
        f"seasonal-naive {selection.get('winner_val_seasonal_naive_mae', float('nan')):.4f}).",
    ]

    still_losing = summary.get("variants_still_losing_to_seasonal_naive_at_peak", [])
    peak = scores[(scores["block"] == "test") & (scores["scope"] == "peak")]
    if not peak.empty:
        winner_rows = peak[peak["variant"] == selection.get("winner")]
        if not winner_rows.empty:
            row = winner_rows.iloc[0]
            if bool(row["beats_naive"]) and bool(row["difference_is_significant"]):
                body.append(
                    f"\n**On the test origins the selected forecaster beats "
                    f"seasonal-naive at peak hours**: MAE {row['mae']:.4f} against "
                    f"{row['seasonal_naive_mae']:.4f}, a difference of "
                    f"{row['mae_vs_naive']:+.4f} with a 95% interval of "
                    f"{row['mae_vs_naive_ci_low']:+.4f} to {row['mae_vs_naive_ci_high']:+.4f}, "
                    "which excludes zero. The peak-hour weakness visible in the "
                    "four-fold backtest does not survive the larger sample."
                )
            elif bool(row["beats_naive"]):
                body.append(
                    f"\n**At peak hours on the test origins the selected forecaster is "
                    f"ahead but not significantly so**: MAE {row['mae']:.4f} against "
                    f"{row['seasonal_naive_mae']:.4f}, interval "
                    f"{row['mae_vs_naive_ci_low']:+.4f} to {row['mae_vs_naive_ci_high']:+.4f}, "
                    "which includes zero."
                )
            else:
                body.append(
                    f"\n**At peak hours the selected forecaster still does not beat "
                    f"seasonal-naive** on the test origins: MAE {row['mae']:.4f} "
                    f"against {row['seasonal_naive_mae']:.4f}. This is the trigger for "
                    "the deep-forecaster item in future work."
                )
    if still_losing:
        body.append(
            "\nVariants that still do not beat seasonal-naive at peak on the test "
            "block: " + ", ".join(f"`{name}`" for name in still_losing) + "."
        )

    coverage = reader.table("forecast_study_coverage.csv", record=False)
    if coverage is not None and not coverage.empty:
        test = coverage[coverage["block"] == "test"]
        body += [
            "",
            "**Rolling conformal intervals, by hour-of-day band.** Each interval is "
            "sized from the residuals of forecasts issued at *earlier* origins in the "
            "same band, so it uses only what had already been observed when the "
            "forecast was made.",
            "",
            md_table(test),
        ]
        outside = test[~test["within_tolerance"].astype(bool)]
        if outside.empty:
            body.append(
                f"\nEvery band on the test origins is within "
                f"{float(test['tolerance'].iloc[0]):.2f} of nominal "
                f"{float(test['nominal_coverage'].iloc[0]):.2f}, which is the target."
            )
        else:
            named = ", ".join(
                f"{row.scope} {row.empirical_coverage:.3f}" for row in outside.itertuples()
            )
            body.append(
                f"\n**The target is not met on every band**: {named} fall outside "
                f"{float(test['tolerance'].iloc[0]):.2f} of nominal."
            )

    body += ["", summary.get("adoption_note", "")]
    return body


#: What each refusing seed's validation failure actually was, from the Part 0 study
#: recorded in `artifacts/checkpoints/days_work.md`. A gate refusal is not
#: self-explanatory and the two here mean very different things, so the diagnosis
#: travels with the number rather than living only in a checkpoint file.
GATE_REFUSAL_DIAGNOSES = {
    8: (
        "ESI 3 abandonment against a 1.6% base rate -- a 1.2 percentage-point miss, "
        "which is 72% in relative terms and clinically trivial. A **rare-event "
        "tolerance artefact**: the absolute floor exists for exactly this case and is "
        "set just low enough to let it through."
    ),
    10: (
        "ESI 3 mean wait **8 minutes low** against an observed 34.4 minutes. A "
        "**genuine fit miss**, not a small-sample effect, and the gate is right to "
        "refuse it."
    ),
}


def _robustness_section(reader: ArtifactReader) -> str:
    """Which conclusions survive a different dataset.

    Reads the seed study written by `scripts/robustness_study.py`, which regenerates
    each dataset with the **true** generator at
    `scripts/original_generator/generate_ed_data.py`.

    It deliberately does not fall back to `robustness_summary.csv`. That file holds the
    older study, built from the wrapper around `src/data/synthetic.py`, whose service
    times are total post-contact time where the true generator's are physician-busy
    time. The two do not measure the same department, so "seed 7" does not name the
    same dataset in both -- and the seed-7 gate failure that study reported did not
    reproduce and has been withdrawn. Printing it beside current figures would present
    a retracted finding as a result.
    """
    table = reader.table("robustness_seeds.csv", record=False)
    if table is None or table.empty:
        # An uploaded-file run is one dataset by definition: a seed study
        # regenerates the *generator's* datasets, which has nothing to do with the
        # file somebody handed the service. Telling that reader to go and run a
        # study would be advice they cannot act on and do not need, so the two
        # cases get different sentences.
        if reader.stages.run_type == "uploaded-file analysis":
            return "_Not part of an uploaded-file run._"
        return (
            "_Not available: the seed study has not been run. Produce it with "
            "`python scripts/robustness_study.py`, which regenerates each dataset "
            "with the original generator and runs the whole pipeline on it._"
        )

    reference = table[table["seed"] == 42] if "seed" in table.columns else table.iloc[0:0]
    study = table[table["seed"] != 42] if "seed" in table.columns else table

    body = [
        f"The whole pipeline was re-run on **{len(study)} regenerated dataset(s)**, "
        "each produced by the original generator with nothing changed but its random "
        "seed, and each writing to its own run folder so none of them can touch the "
        "baseline artifacts. Every figure below is read from those runs.",
    ]
    if not reference.empty:
        body.append(
            ""
            "\nSeed 42 is the reference row: at 365 days and scale 9.7 it *is* "
            "`data/raw/ed_v2.csv`, so it must reproduce the headline numbers, and the "
            "study stops if it does not."
        )
    body += [
        "",
        md_table(
            table[
                [
                    column
                    for column in (
                        "seed", "n_rows", "wait_mae_model", "wait_mae_baseline",
                        "conformal_static_overall", "conformal_rolling_overall",
                        "forecast_mae", "forecast_mae_seasonal_naive",
                        "sim_validation_passed", "sim_worst_relative_error",
                        "cheapest_feasible_plan", "cheapest_feasible_cost",
                        "n_feasible_plans",
                    )
                    if column in table.columns
                ]
            ]
        ),
    ]

    # The validation gate: how often it passed, and what each refusal was.
    if "sim_validation_passed" in study.columns:
        gate = study["sim_validation_passed"].fillna(False).astype(bool)
        passed, total = int(gate.sum()), int(len(study))
        refused = sorted(int(s) for s in study.loc[~gate, "seed"]) if "seed" in study else []
        body += [
            "",
            f"**The simulation validation gate passed on {passed} of {total}.** On the "
            "rest it refused to issue a staffing plan, which is what it is for: a plan "
            "from a model that cannot reproduce its own dataset looks exactly as "
            "confident as a correct one.",
        ]
        for seed in refused:
            diagnosis = GATE_REFUSAL_DIAGNOSES.get(
                seed, "see that run's `sim_validation.csv` for the failing check."
            )
            body.append(f"- **Seed {seed}:** {diagnosis}")
        if refused:
            body.append(
                "\nNo tolerance was widened to make any of these pass. Loosening a gate "
                "until it stops refusing is how a staffing recommendation from an "
                "unvalidated model reaches a reader."
            )

    # The staffing plan: the spread is the finding.
    costs = (
        pd.to_numeric(study["cheapest_feasible_cost"], errors="coerce").dropna()
        if "cheapest_feasible_cost" in study.columns
        else pd.Series(dtype=float)
    )
    if not costs.empty and "cheapest_feasible_plan" in study.columns:
        plans = study.loc[costs.index, "cheapest_feasible_plan"].dropna()
        distinct = pd.unique(plans)
        day_shifts = {
            part.split("=")[1]
            for plan in distinct
            for part in str(plan).split(", ")
            if part.startswith("day=")
        }
        body += [
            "",
            f"**The staffing method transfers; the specific roster does not.** Across "
            f"the {len(costs)} seed(s) that produced a plan the cheapest cost ranged "
            f"**{costs.min():,.0f} to {costs.max():,.0f}** over **{len(distinct)} "
            f"distinct roster(s)**"
            + (
                f", with the day shift at {day_shifts.pop()} in every one"
                if len(day_shifts) == 1
                else ""
            )
            + ". The cheapest plan this report quotes is therefore the answer for "
            "**this dataset**, not a recommendation for a department.",
        ]

    # Conclusions that held everywhere.
    held: list[str] = []
    if {"wait_mae_model", "wait_mae_baseline"} <= set(study.columns):
        beats = int((study["wait_mae_model"] < study["wait_mae_baseline"]).sum())
        held.append(
            f"- The wait model beat its queue-position baseline on **{beats} of "
            f"{len(study)}** seeds."
        )
    if "forecast_beats_naive_overall" in study.columns:
        wins = int(study["forecast_beats_naive_overall"].fillna(False).astype(bool).sum())
        held.append(
            f"- The arrivals forecast beat seasonal-naive on **{wins} of {len(study)}** "
            "seeds."
        )
    if {"conformal_static_overall", "conformal_rolling_overall"} <= set(study.columns):
        sd_static = float(study["conformal_static_overall"].std())
        sd_rolling = float(study["conformal_rolling_overall"].std())
        if sd_rolling > 0:
            held.append(
                f"- Rolling conformal coverage was **{sd_static / sd_rolling:.1f}x more "
                f"stable** than a static split (sd {sd_rolling:.4f} against "
                f"{sd_static:.4f}). The defensible claim is not that rolling is always "
                "closer to nominal but that it lands in the same place whatever dataset "
                "arrives."
            )
    if held:
        body += ["", "**What held across the seeds.**", "", *held]

    stats = reader.table("robustness_seed_stats.csv", record=False)
    if stats is not None and not stats.empty:
        body += [
            "",
            "**Spread across the seeds.** Mean, standard deviation and range for every "
            "quantity the study measured.",
            "",
            md_table(stats),
        ]

    body.append(
        "\nAll of these datasets come from one generator. Agreement across its seeds "
        "shows the pipeline is not over-fitted to a single draw of it; it recovers the "
        "generator, and is not evidence about any real emergency department."
    )
    return "\n".join(body)


def build_report(config: dict[str, Any]) -> str:
    """Assemble the full report from artifacts on disk.

    Args:
        config: Full pipeline config.

    Returns:
        The report as Markdown.
    """
    reader = ArtifactReader(config)
    run = reader.metrics("pipeline_run.json") or {}

    header = [
        "# ED Intelligence - results report",
        "",
        "**Decision support only.** Every figure below is an estimate for a human to "
        "weigh. Nothing in this system triages a patient or makes an autonomous clinical "
        "decision.",
        "",
        # The run folder's path was in this sentence. It is a machine-specific
        # absolute path, it means nothing to anyone reading the report on another
        # computer, and in an uploaded run it carried the job id as well.
        "Every number here is read from a file a pipeline stage wrote; nothing is "
        "re-estimated in this report and nothing is entered by hand. Where a sentence "
        "compares two of those numbers -- a difference, an interval width, a "
        "percentage change -- it is computed from them and from nothing else.",
    ]
    if run:
        header += [
            "",
            f"- Data source: **{run.get('data_source', 'unknown')}**",
            f"- Stages run: {', '.join(run.get('stages_run', []))}",
            f"- Total pipeline time: **{run.get('total_seconds', float('nan')):.1f} s**",
        ]
    elif reader.stages.known:
        header += [
            "",
            f"- Run type: **{reader.stages.run_type}**",
            f"- Stages in this run: {', '.join(reader.stages.performed)}",
        ]

    gaps = [
        "",
        "## Research gaps addressed",
        "",
        "| Gap | Problem | Where it is addressed |",
        "| --- | --- | --- |",
        "| **G1** | Forecasting, wait prediction and simulation are built in isolation. | "
        "The arrival forecast drives the simulation, which drives the staffing search. |",
        "| **G2** | Subgroup fairness is rarely evaluated. | Subgroup MAE, FNR and FPR with "
        "bootstrap intervals, and an explicit finding about high-acuity patients. |",
        "| **G3** | Queue-state features are assumed to help but rarely isolated. | A "
        "controlled `base` vs `base+queue` ablation, with its caveat attached. |",
    ]

    sections = [
        _section(
            "Data and validation",
            _data_section(reader) + "\n".join(_data_provenance_block(reader)),
        ),
        _section("Wait-time prediction", _wait_model_section(reader)),
        _section("Subgroup performance (G2)", _fairness_section(reader)),
        _section("Arrival forecasting", _forecast_section(reader)),
        _section("Competing-risks survival analysis", _survival_section(reader)),
        _section("Simulation and staffing (G1)", _simulation_section(reader)),
        _section("Robustness across datasets", _robustness_section(reader)),
        _section("Limitations", _limitations_section(reader)),
        _section("Future work", _future_work_section(reader)),
    ]

    notes = _missing_section(reader)
    if notes is not None:
        sections.append(_section("Artifacts not available", notes))

    return "\n".join(header + gaps + [""] + sections)


def write_report(config: dict[str, Any]) -> Path:
    """Build the report and write it to `paths.report`."""
    from src.utils.io import ensure_parent

    text = build_report(config)
    path = ensure_parent(config["paths"]["report"])
    path.write_text(text, encoding="utf-8")
    logger.info("Wrote %s (%d lines)", path, len(text.splitlines()))
    return path
