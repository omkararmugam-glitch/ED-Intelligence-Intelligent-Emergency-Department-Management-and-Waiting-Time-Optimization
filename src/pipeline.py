"""Pipeline CLI orchestrator.

    python -m src.pipeline run --config configs/default.yaml --data synthetic
    python -m src.pipeline run --config configs/default.yaml --data real
    python -m src.pipeline run --stages load,validate,split
    python -m src.pipeline list-stages

Design rules that keep this usable from a future FastAPI layer:
  * Every stage is a plain function of a `PipelineContext`, with no hidden globals.
  * Every stage reads its inputs from disk and writes its outputs to disk, so any
    stage can be run on its own without re-running the ones before it.
  * Every stage is deterministic given the config's seed, so re-running it overwrites
    its outputs with identical content -- that is what "idempotent" means here.

Stages are registered in `STAGE_ORDER` with the project phase that delivers them
(see CLAUDE.md). Asking for a stage from a later phase raises a clear error naming
that phase, rather than silently doing nothing.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

# Heavy third-party libraries (lightgbm, simpy, lifelines, matplotlib) are imported
# inside the stage that needs them, not here. That keeps `--help` and `list-stages`
# working when an optional library is missing or broken -- a real situation on locked
# down Windows machines -- and means one unusable library disables one stage rather
# than the whole CLI.
from src.data.schema import Schema, load_schema, validate_visits
from src.data.split import BLOCK_NAMES
from src.utils.io import (
    ensure_parent,
    load_config,
    load_json,
    load_model,
    read_visits,
    resolve_path,
    save_json,
    save_model,
    save_table,
    write_visits,
)
from src.utils.logging import get_logger, setup_logging
from src.utils.seed import set_global_seeds

logger = get_logger(__name__)

DATA_SOURCES = ("synthetic", "real")

#: Intermediate files the stages hand to each other, relative to paths.processed_dir.
PROCESSED_FILES = {
    "visits": "visits.parquet",
    "features_train": "features_train.parquet",
    "features_val": "features_val.parquet",
    "features_calib": "features_calib.parquet",
    "features_test": "features_test.parquet",
    "hourly_series": "hourly_series.parquet",
    "forecast": "forecast.parquet",
    "validated": "visits_validated.parquet",
    "flagged": "visits_flagged_not_seen.parquet",
    "train": "split_train.parquet",
    "val": "split_val.parquet",
    "calib": "split_calib.parquet",
    "test": "split_test.parquet",
}


@dataclass
class PipelineContext:
    """Everything a stage needs: the config, the data source and the schema.

    Stages communicate through files rather than through this object, so the context
    stays small and a single stage can be run from a cold start.
    """

    config: dict[str, Any]
    data_source: str
    schema: Schema
    stage_timings: dict[str, float] = field(default_factory=dict)

    @property
    def paths(self) -> dict[str, str]:
        """The config's `paths` block."""
        return self.config["paths"]

    def processed(self, key: str) -> Path:
        """Absolute path of a named intermediate file under the processed directory."""
        if key not in PROCESSED_FILES:
            raise KeyError(f"Unknown processed file {key!r}; known: {sorted(PROCESSED_FILES)}")
        return resolve_path(Path(self.paths["processed_dir"]) / PROCESSED_FILES[key])

    def metrics_path(self, name: str) -> Path:
        """Absolute path for a metrics JSON artifact."""
        return resolve_path(Path(self.paths["metrics_dir"]) / name)

    def require(self, key: str) -> pd.DataFrame:
        """Read an intermediate file, with a message naming the stage that creates it."""
        path = self.processed(key)
        if not path.is_file():
            producer = {
                "visits": "load",
                "validated": "validate",
                "flagged": "validate",
                "train": "split",
                "val": "split",
                "calib": "split",
                "test": "split",
                "features_train": "features",
                "features_val": "features",
                "features_calib": "features",
                "features_test": "features",
                "hourly_series": "forecast",
                "forecast": "forecast",
            }.get(key, "an earlier")
            raise FileNotFoundError(
                f"{path} is missing. Run the '{producer}' stage first, for example:\n"
                f"  python -m src.pipeline run --stages {producer} --data {self.data_source}"
            )
        return read_visits(path)


# --------------------------------------------------------------------------- stages


def stage_load(context: PipelineContext) -> None:
    """Obtain the visits table from the configured source and stage it for validation.

    For `--data synthetic` the generator runs and its output is also kept under
    `data/synthetic/` as the canonical dataset. For `--data real` the raw extract is
    read and mapped through `configs/column_map.yaml`.
    """
    from src.data.adapter import load_real_visits
    from src.data.synthetic import generate_synthetic_visits

    if context.data_source == "synthetic":
        visits = generate_synthetic_visits(context.config)
        write_visits(visits, context.paths["synthetic_file"])
    else:
        visits = load_real_visits(context.config, schema=context.schema)

    write_visits(visits, context.processed("visits"))
    logger.info("Stage 'load' produced %d visits from the %s source", len(visits), context.data_source)


def stage_validate(context: PipelineContext) -> None:
    """Validate against the data contract, derive wait/LOS, and split off flagged rows."""
    visits = context.require("visits")
    result = validate_visits(visits, context.schema)

    write_visits(result.all, context.processed("validated"))
    write_visits(result.flagged, context.processed("flagged"))
    save_json(result.report.to_dict(), context.metrics_path("validation.json"))
    save_table(
        _validation_summary_table(result.report),
        Path(context.paths["tables_dir"]) / "validation_summary.csv",
    )
    _write_dataset_profile(context, result.all)
    _check_dataset_shift(context, result.all)

    if result.report.n_flagged_no_seen_time:
        logger.warning(
            "%d patients (%.2f%%) were never seen and are excluded from the wait-time "
            "model. This is a known limitation; see README.",
            result.report.n_flagged_no_seen_time,
            100.0 * result.report.n_flagged_no_seen_time / max(result.report.n_valid, 1),
        )


def _check_dataset_shift(context: PipelineContext, validated: pd.DataFrame) -> None:
    """Warn when a real extract differs from the synthetic profile the models assume.

    Runs only for `--data real`, and only against a *baseline* synthetic profile --
    comparing a run with the profile it is about to overwrite would compare a dataset
    with itself and always read zero.

    This never stops a run and never changes a metric. It exists so that nobody can
    point the pipeline at a new dataset without being told that every model in it was
    tuned on a different one.

    Args:
        context: The pipeline context.
        validated: The validated visits.
    """
    if context.data_source != "real":
        return

    from src.data.adapter import proxy_wait_status
    from src.data.shift_check import compare_to_reference

    # The baseline tables, not this run's: a run's own profile is written moments
    # before this and would make every comparison trivially zero.
    baseline_tables = resolve_path("artifacts/tables")
    report = compare_to_reference(validated, baseline_tables)

    proxy = proxy_wait_status(context.config)
    if proxy.get("active"):
        logger.warning("%s Proxy: %s", proxy["notice"], proxy["proxy"])

    payload = report.to_dict()
    payload["proxy_wait"] = proxy
    save_json(payload, context.metrics_path("dataset_shift.json"))


def _validation_summary_table(report: Any) -> pd.DataFrame:
    """Flatten the validation report into one auditable row per check.

    Every row the validator removed or repaired appears here with its reason and
    count, so data quality can be reviewed without reading JSON.
    """
    payload = report.to_dict()
    rows: list[dict[str, Any]] = [
        {"category": "input", "detail": "rows read from source", "count": payload["n_input"]},
        {"category": "outcome", "detail": "rows valid after checks", "count": payload["n_valid"]},
        {"category": "outcome", "detail": "rows dropped", "count": payload["n_dropped"]},
        {
            "category": "outcome",
            "detail": "usable for the wait-time model",
            "count": payload["n_modelling"],
        },
        {
            "category": "censoring",
            "detail": "never seen: left without being seen",
            "count": payload["n_lwbs"],
        },
        {
            "category": "censoring",
            "detail": "never seen: still waiting at the data cutoff",
            "count": payload["n_still_waiting_at_cutoff"],
        },
    ]
    for reason, count in sorted(payload["dropped_by_reason"].items(), key=lambda kv: -kv[1]):
        rows.append({"category": "dropped", "detail": reason, "count": count})
    for reason, count in sorted(payload["coerced_by_reason"].items(), key=lambda kv: -kv[1]):
        rows.append({"category": "coerced", "detail": reason, "count": count})
    for name in payload["optional_columns_present"]:
        rows.append({"category": "optional column", "detail": f"{name}: present", "count": 1})
    for name in payload["optional_columns_missing"]:
        rows.append({"category": "optional column", "detail": f"{name}: ABSENT", "count": 0})

    table = pd.DataFrame(rows)
    total = max(int(payload["n_input"]), 1)
    table["share_of_input"] = (table["count"] / total).round(6)
    return table


def _profile_arrivals_by_hour(validated: pd.DataFrame) -> pd.DataFrame:
    """Mean arrivals per clock hour, which exposes the hour-of-day demand profile."""
    hourly = validated.set_index("arrival_time").resample("h").size()
    by_hour = hourly.groupby(hourly.index.hour).agg(["mean", "std", "max"])
    by_hour.index.name = "hour_of_day"
    by_hour["share_of_arrivals"] = by_hour["mean"] / by_hour["mean"].sum()
    return by_hour.round(4).reset_index()


def _profile_arrivals_by_weekday(validated: pd.DataFrame) -> pd.DataFrame:
    """Mean arrivals per hour by weekday, to confirm the day-of-week profile is real."""
    hourly = validated.set_index("arrival_time").resample("h").size()
    by_dow = hourly.groupby(hourly.index.dayofweek).agg(["mean", "std"])
    by_dow.index = by_dow.index.map(
        dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]))
    )
    by_dow.index.name = "day_of_week"
    return by_dow.round(4).reset_index()


def _profile_triage_mix(validated: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Observed acuity mix, next to the configured target mix when one exists."""
    counts = validated["triage_level"].value_counts().sort_index()
    mix = pd.DataFrame({"triage_level": counts.index, "n": counts.to_numpy()})
    mix["observed_share"] = (mix["n"] / mix["n"].sum()).round(4)

    configured = {int(k): float(v) for k, v in config.get("synthetic", {}).get("triage_mix", {}).items()}
    if configured:
        mix["configured_share"] = mix["triage_level"].map(configured)
        mix["difference"] = (mix["observed_share"] - mix["configured_share"]).round(4)
    return mix


def _profile_wait_by_triage(validated: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Wait-time distribution per acuity level, plus the never-seen share.

    The percentiles come only from patients who were actually seen, because a patient
    without a `seen_time` has no observed wait. Their share is reported alongside so
    the censoring is visible rather than hidden.
    """
    threshold = float(config["features"]["prolonged_wait_threshold_minutes"])
    rows = []
    for level, group in validated.groupby("triage_level"):
        seen = group["wait_minutes"].dropna()
        never_seen = group["seen_time"].isna()
        # The two reasons for a missing wait are reported separately: one is an event
        # (the patient gave up), the other is censoring by the extract window.
        lwbs = never_seen & group["is_lwbs"]
        censored = never_seen & group["still_waiting_at_cutoff"]
        rows.append(
            {
                "triage_level": int(level),
                "n_arrived": int(len(group)),
                "n_seen": int(len(seen)),
                "n_lwbs": int(lwbs.sum()),
                "n_still_waiting_at_cutoff": int(censored.sum()),
                "never_seen_share": round(1.0 - len(seen) / len(group), 4),
                "lwbs_share": round(float(lwbs.mean()), 4),
                "still_waiting_share": round(float(censored.mean()), 6),
                "mean": round(float(seen.mean()), 1),
                "median": round(float(seen.median()), 1),
                "p75": round(float(seen.quantile(0.75)), 1),
                "p90": round(float(seen.quantile(0.90)), 1),
                "p95": round(float(seen.quantile(0.95)), 1),
                "max": round(float(seen.max()), 1),
                f"share_over_{int(threshold)}min": round(float((seen > threshold).mean()), 4),
            }
        )
    return pd.DataFrame(rows)


def _write_dataset_profile(context: PipelineContext, validated: pd.DataFrame) -> None:
    """Write the descriptive profile of the validated dataset to artifacts/tables/.

    These tables are how a reader checks that the data behaves like an ED before any
    model is fitted: demand varies by hour and weekday, the acuity mix matches the
    configured one, and waits grow with ESI level.
    """
    tables_dir = Path(context.paths["tables_dir"])
    for name, frame in {
        "profile_arrivals_by_hour.csv": _profile_arrivals_by_hour(validated),
        "profile_arrivals_by_weekday.csv": _profile_arrivals_by_weekday(validated),
        "profile_triage_mix.csv": _profile_triage_mix(validated, context.config),
        "profile_wait_by_triage.csv": _profile_wait_by_triage(validated, context.config),
    }.items():
        save_table(frame, tables_dir / name)


def stage_split(context: PipelineContext) -> None:
    """Cut the validated visits into train/val/test forward in time.

    The bounds are derived from the modellable rows and then applied unchanged to the
    full table, so later stages can reuse exactly the same dates for the feature
    matrix and the hourly arrival series.
    """
    from src.data.split import assert_blocks_disjoint, make_time_split

    validated = context.require("validated")

    # Every validated row goes into a block, including patients who were never seen.
    # Each model stage then selects its own cohort and logs what it excluded: the
    # wait-time regressor needs fully observed waits, while the Phase 2 survival model
    # needs exactly the rows the regressor drops.
    split = make_time_split(validated, context.config)
    for name, frame in split.frames.items():
        write_visits(frame, context.processed(name))

    summary = split.summary()
    summary["n_without_observed_wait"] = int(validated["wait_minutes"].isna().sum())
    save_json(summary, context.metrics_path("split.json"))
    assert_blocks_disjoint(split)


def stage_features(context: PipelineContext) -> None:
    """Build the feature matrix for each split.

    Queue state is measured against the whole validated population, not against each
    split in isolation: a patient arriving on the first day of the test period really
    did walk into a department holding patients who arrived during the validation
    period. Building one shared index also means the sort happens once, not three times.
    """
    from src.features.build import (
        FEATURE_SETS,
        add_prolonged_wait_label,
        build_features,
        make_design_matrix,
    )
    from src.features.queue_state import QueueStateIndex

    validated = context.require("validated")
    index = QueueStateIndex.from_visits(validated)

    summary: dict[str, Any] = {"feature_sets": {}, "splits": {}}
    for split_name in BLOCK_NAMES:
        rows = context.require(split_name)
        started = time.perf_counter()
        featurised = build_features(rows, context.config, index=index)
        featurised = add_prolonged_wait_label(featurised, context.config)
        elapsed = time.perf_counter() - started

        write_visits(featurised, context.processed(f"features_{split_name}"))
        summary["splits"][split_name] = {
            "n_rows": int(len(featurised)),
            "seconds": round(elapsed, 3),
            "rows_per_second": int(len(featurised) / elapsed) if elapsed > 0 else None,
            "positive_label_rate": round(float(featurised["prolonged_wait"].mean()), 4)
            if featurised["prolonged_wait"].notna().any()
            else None,
        }

    # Record the resolved column lists so the ablation can be audited later.
    reference_frame = context.require("features_train")
    for set_name in FEATURE_SETS:
        _, spec = make_design_matrix(reference_frame, set_name, context.config, context.schema)
        summary["feature_sets"][set_name] = {"n_columns": len(spec), "columns": spec.columns}

    save_json(summary, context.metrics_path("features.json"))
    logger.info(
        "Feature sets: base=%d columns, base+queue=%d columns",
        summary["feature_sets"]["base"]["n_columns"],
        summary["feature_sets"]["base+queue"]["n_columns"],
    )


def stage_train_wait(context: PipelineContext) -> None:
    """Train the wait-time models, run the ablation, and calibrate conformal intervals.

    Produces, under artifacts/:
        tables/ablation.csv                base vs base+queue x raw vs log1p target
        tables/classifier_summary.csv      PR-AUC, recall, precision, FNR per set
        tables/conformal_coverage.csv      coverage and width, overall and per ESI
        tables/high_acuity_comparison.csv  four ESI 1-2 approaches, with CIs
        tables/seen_only_bias.csv          the bias from excluding abandoners
        metrics/wait_models.json           everything above, machine-readable
        models/*.joblib                    every fitted model and the calibrator
    """
    from src.models.figures import write_wait_model_figures
    from src.models.train_wait_stage import (
        ABLATION_CAVEAT,
        REPORT_BLOCK,
        calibrate_best_model,
        classifier_table,
        evaluate_per_triage_thresholds,
        evaluate_rolling_conformal,
        evaluate_baseline,
        prepare_blocks,
        run_ablation,
        select_best_cell,
    )
    from src.models.high_acuity import TARGET_CAVEAT, compare_high_acuity_approaches
    from src.models.wait_time import seen_only_bias_report
    from src.features.build import make_design_matrix

    featurised = {name: context.require(f"features_{name}") for name in BLOCK_NAMES}
    blocks = prepare_blocks(featurised)
    validated = context.require("validated")

    baseline_results, baseline_model = evaluate_baseline(blocks, validated, context.config)
    table, artifacts = run_ablation(blocks, context.config, context.schema)
    best = select_best_cell(table)
    logger.info(
        "Best cell by test MAE: feature_set=%s target=%s kind=%s (MAE %.2f min)",
        best["feature_set"],
        best["target"],
        best["prediction_kind"],
        best["mae_minutes"],
    )

    calibrator, conformal_report, conformal_rows = calibrate_best_model(
        blocks, artifacts, best, context.config, context.schema
    )
    # Static conformal above is kept as the baseline; this adds the rolling comparison.
    rolling_table, rolling_summary = evaluate_rolling_conformal(
        blocks, artifacts, best, calibrator, context.config, context.schema
    )
    conformal_report["rolling"] = rolling_summary

    thresholds, threshold_table, threshold_summary = evaluate_per_triage_thresholds(
        blocks, artifacts, context.config, context.schema
    )
    # Per-acuity thresholds can only move an existing score around; they cannot create
    # positive cases where ESI 1 and 2 have almost none. This compares four approaches
    # that might, including an acuity-specific target, and reports the positive counts
    # and intervals that decide whether any of them is defensible.
    #
    # Skipped rather than fatal when the config predates the feature: a run started
    # from an older config snapshot (the upload workflow and
    # scripts/generate_ed_data.py both write one) must still train its wait models.
    if context.config["features"].get("acuity_target_wait_minutes"):
        high_acuity_table, high_acuity_summary = compare_high_acuity_approaches(
            blocks, artifacts, context.config, context.schema
        )
    else:
        reason = (
            "features.acuity_target_wait_minutes is not set in this run's config, so "
            "the acuity-aware target cannot be defined and the four-approach "
            "comparison was not run. Add the block (see configs/default.yaml) to "
            "enable it."
        )
        logger.warning("High-acuity comparison skipped: %s", reason)
        high_acuity_table = pd.DataFrame()
        high_acuity_summary = {"ran": False, "skipped_reason": reason}
    bias_report, bias_table = seen_only_bias_report(validated, context.config)

    tables_dir = Path(context.paths["tables_dir"])
    # The caveat rides along as a comment line so nobody reads the table without it.
    # Read it back with pandas.read_csv(path, comment="#").
    ablation_path = ensure_parent(tables_dir / "ablation.csv")
    with ablation_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# " + ABLATION_CAVEAT + "\n")
        table.to_csv(handle, index=False)
    logger.info("Wrote %s (%d rows, caveat on line 1)", ablation_path, len(table))

    classifiers = classifier_table(artifacts)
    save_table(classifiers, tables_dir / "classifier_summary.csv")
    save_table(conformal_rows, tables_dir / "conformal_coverage.csv")
    save_table(rolling_table, tables_dir / "conformal_static_vs_rolling.csv")
    save_table(threshold_table, tables_dir / "triage_thresholds.csv")
    # Same convention as the ablation: the caveat rides along on line 1, because the
    # per-acuity targets behind half this table are assumed, not clinical.
    high_acuity_path = ensure_parent(tables_dir / "high_acuity_comparison.csv")
    with high_acuity_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# " + TARGET_CAVEAT + "\n")
        high_acuity_table.to_csv(handle, index=False)
    logger.info(
        "Wrote %s (%d rows, caveat on line 1)", high_acuity_path, len(high_acuity_table)
    )
    save_table(bias_table, tables_dir / "seen_only_bias.csv")

    # Figures need the best model's test-block predictions and its intervals.
    best_cell = artifacts[f"{best['feature_set']}::{best['target']}"]
    best_regressor = best_cell["regressor"]
    predict_kind = "smearing" if best["prediction_kind"].endswith("smearing") else "default"
    test_design, _ = make_design_matrix(
        blocks[REPORT_BLOCK].frame, best["feature_set"], context.config, context.schema
    )
    test_prediction = best_regressor.predict(test_design, kind=predict_kind)
    interval_lower, interval_upper = calibrator.intervals(
        test_prediction, groups=blocks[REPORT_BLOCK].triage
    )
    write_wait_model_figures(
        ablation=table,
        conformal=conformal_rows,
        bias=bias_table,
        figures_dir=Path(context.paths["figures_dir"]),
        baseline_mae=baseline_results["blocks"][REPORT_BLOCK]["regression"]["mae"],
        quality={
            "observed": blocks[REPORT_BLOCK].wait,
            "predicted": test_prediction,
            "lower": interval_lower,
            "upper": interval_upper,
        },
    )

    models_dir = Path(context.paths["models_dir"])
    save_model(baseline_model, models_dir / "baseline_rolling_mean.joblib")
    save_model(calibrator, models_dir / "conformal_calibrator.joblib")
    save_model(thresholds, models_dir / "triage_thresholds.joblib")
    for key, payload in artifacts.items():
        safe = key.replace("::", "__").replace("+", "_plus_")
        if "regressor" in payload:
            save_model(payload["regressor"], models_dir / f"wait_regressor__{safe}.joblib")
        if "classifier" in payload:
            save_model(payload["classifier"], models_dir / f"classifier__{safe}.joblib")

    save_json(
        {
            "blocks": {
                name: {
                    "n_featurised": int(len(block.featurised)),
                    "cohort": block.cohort.to_dict(),
                }
                for name, block in blocks.items()
            },
            "baseline": baseline_results,
            "ablation": {
                "caveat": ABLATION_CAVEAT,
                "rows": table.to_dict(orient="records"),
                "best_by_test_mae": best,
            },
            "classifiers": classifiers.to_dict(orient="records"),
            "conformal": conformal_report,
            "triage_thresholds": threshold_summary,
            "high_acuity": high_acuity_summary,
            "seen_only_bias": bias_report,
        },
        context.metrics_path("wait_models.json"),
    )


def stage_explain(context: PipelineContext) -> None:
    """SHAP explanations for the best wait-time model (Phase 1c, part 1).

    Explains the model the ablation selected, on the test block, so the explanation
    describes the model that is actually being reported.
    """
    from src.explain.shap_report import build_shap_report

    bundle = _load_best_wait_model(context)
    report = build_shap_report(
        model=bundle["regressor"].model,
        features=bundle["test_design"],
        context=bundle["test_frame"],
        config=context.config,
        tables_dir=Path(context.paths["tables_dir"]),
        figures_dir=Path(context.paths["figures_dir"]),
        feature_set=bundle["best"]["feature_set"],
        target_transform=bundle["best"]["target"],
    )
    save_json(report.to_dict(), context.metrics_path("shap.json"))


def stage_fairness(context: PipelineContext) -> None:
    """Subgroup error analysis for the prolonged-wait classifier (Phase 1c, part 2)."""
    from src.explain.fairness import build_fairness_report

    bundle = _load_best_wait_model(context)
    classifier = bundle["classifier"]

    report = build_fairness_report(
        frame=bundle["test_frame"],
        y_true_label=bundle["test_frame"]["prolonged_wait"].to_numpy(dtype="int8"),
        y_score=classifier.predict_proba(bundle["classifier_design"]),
        threshold=classifier.threshold,
        y_true_wait=bundle["test_frame"]["wait_minutes"].to_numpy(dtype="float64"),
        y_pred_wait=bundle["test_prediction"],
        config=context.config,
        tables_dir=Path(context.paths["tables_dir"]),
        figures_dir=Path(context.paths["figures_dir"]),
    )
    save_json(report.to_dict(), context.metrics_path("fairness.json"))


def _load_best_wait_model(context: PipelineContext) -> dict[str, Any]:
    """Reload the winning model and its test-block design matrix from artifacts.

    Stages must be runnable on their own, so this reads the saved models rather than
    depending on `train_wait` having run in the same process.
    """
    from src.features.build import make_design_matrix
    from src.models.wait_time import select_regression_cohort

    metrics = load_json(context.metrics_path("wait_models.json"))
    best = metrics["ablation"]["best_by_test_mae"]
    models_dir = Path(context.paths["models_dir"])

    def artifact(stem: str) -> Path:
        return resolve_path(models_dir / stem)

    safe = f"{best['feature_set']}__{best['target']}".replace("+", "_plus_")
    regressor = load_model(artifact(f"wait_regressor__{safe}.joblib"))
    classifier_safe = f"{best['feature_set']}__classifier".replace("+", "_plus_")
    classifier = load_model(artifact(f"classifier__{classifier_safe}.joblib"))
    calibrator = load_model(artifact("conformal_calibrator.joblib"))

    featurised = context.require("features_test")
    cohort = select_regression_cohort(featurised, block_name="test")
    test_frame = cohort.frame

    test_design, _ = make_design_matrix(
        test_frame, best["feature_set"], context.config, context.schema
    )
    classifier_design, _ = make_design_matrix(
        test_frame, classifier.feature_set, context.config, context.schema
    )
    predict_kind = "smearing" if best["prediction_kind"].endswith("smearing") else "default"

    return {
        "best": best,
        "regressor": regressor,
        "classifier": classifier,
        "calibrator": calibrator,
        "test_frame": test_frame,
        "test_design": test_design,
        "classifier_design": classifier_design,
        "test_prediction": regressor.predict(test_design, kind=predict_kind),
    }


def stage_forecast(context: PipelineContext) -> None:
    """Hourly arrival forecasting with rolling-origin backtest (Phase 2).

    The forecaster is fitted on the `train` block, early-stopped on `val`, and its
    conformal intervals calibrated on `calib` -- the same block discipline the wait-time
    models use, for the same reason.
    """
    from src.models.forecast import (
        build_hourly_series,
        calibrate_forecaster,
        evaluate_forecast_intervals,
        forecast,
        history_is_sufficient,
        rolling_origin_evaluate,
        train_forecaster,
    )
    from src.models.forecast_figures import (
        forecast_coverage_table,
        forecast_summary_table,
        write_forecast_figures,
    )

    validated = context.require("validated")
    series = build_hourly_series(validated)
    write_visits(series, context.processed("hourly_series"))

    bounds = load_json(context.metrics_path("split.json"))["bounds"]
    train_end = pd.Timestamp(bounds["train_end"])
    val_end = pd.Timestamp(bounds["val_end"])
    calib_end = pd.Timestamp(bounds["calib_end"])

    sufficient, reason = history_is_sufficient(series, context.config, train_end)
    if not sufficient:
        logger.warning("Forecast stage skipped: %s", reason)
        save_json(
            {"ran": False, "reason": reason, "n_hours_available": int(len(series))},
            context.metrics_path("forecast.json"),
        )
        return

    forecaster = train_forecaster(series, context.config, train_end=train_end, val_end=val_end)
    calibrate_forecaster(forecaster, series, context.config, val_end, calib_end)

    predictions, summary = rolling_origin_evaluate(series, context.config)
    summary["ran"] = True
    coverage = evaluate_forecast_intervals(predictions, context.config)
    summary["interval_coverage"] = coverage

    # A worked forecast from the end of the calibration block, which is also what the
    # simulation consumes as its arrival load in Phase 3.
    issued = forecast(
        int(context.config["forecast"]["horizon_hours"]),
        calib_end,
        forecaster,
        series,
        context.config,
    )
    write_visits(issued, context.processed("forecast"))

    tables_dir = Path(context.paths["tables_dir"])
    save_table(forecast_summary_table(summary), tables_dir / "forecast_accuracy.csv")
    save_table(pd.DataFrame(summary["by_horizon"]), tables_dir / "forecast_by_horizon.csv")
    if coverage:
        save_table(
            forecast_coverage_table(coverage, context.config),
            tables_dir / "forecast_coverage.csv",
        )
    save_table(predictions, tables_dir / "forecast_backtest_predictions.csv")

    write_forecast_figures(
        predictions=predictions,
        summary=summary,
        issued=issued,
        figures_dir=Path(context.paths["figures_dir"]),
        config=context.config,
    )

    save_model(forecaster, Path(context.paths["models_dir"]) / "arrival_forecaster.joblib")
    save_json(summary, context.metrics_path("forecast.json"))


def stage_survival(context: PipelineContext) -> None:
    """Competing-risks analysis of the waiting room (Phase 2).

    Runs only when the dataset records patients who left without being seen; otherwise
    it logs why it is skipping and writes a report saying so.
    """
    from src.features.build import add_patient_features, build_features
    from src.models.survival import run_survival_analysis

    validated = context.require("validated")
    # Cox covariates include queue state measured at arrival, so the full population is
    # featurised here rather than reusing a single block.
    featurised = build_features(validated, context.config, reference=validated)
    featurised = add_patient_features(featurised)

    report = run_survival_analysis(
        featurised,
        context.config,
        tables_dir=Path(context.paths["tables_dir"]),
        figures_dir=Path(context.paths["figures_dir"]),
    )
    save_json(report.to_dict(), context.metrics_path("survival.json"))


def stage_competing_risks(context: PipelineContext) -> None:
    """At arrival: will this patient be seen, or leave first, and by when?

    Additive only. It writes new artifacts and touches nothing an earlier stage
    produced, so none of the project's headline numbers can move because of it.

    Produces, under artifacts/:
        tables/competing_risks_metrics.csv      AUC, Brier and ECE per horizon/acuity
        tables/competing_risks_calibration.csv  the 10-bin curves, raw and calibrated
        tables/competing_risks_aalen.csv        group-level check against Aalen-Johansen
        metrics/competing_risks.json            all of the above
        models/competing_risks.joblib           the model `/predict-risk` serves
    """
    from src.features.build import add_patient_features, build_features, make_design_matrix
    from src.models.competing_risks import (
        DEFAULT_HORIZONS,
        SYNTHETIC_NOTICE,
        aalen_johansen_comparison,
        cox_discrimination_comparison,
        evaluate_discrete_time_model,
        train_discrete_time_model,
    )
    from src.models.survival import (
        ABANDONED,
        SEEN,
        build_waiting_room_events,
        has_abandonment_information,
    )

    validated = context.require("validated")
    if not has_abandonment_information(validated):
        reason = (
            "This dataset records no patients who left without being seen, so there is "
            "no competing risk to model. Nothing was fitted."
        )
        logger.warning("Competing-risks stage skipped: %s", reason)
        save_json({"ran": False, "reason": reason}, context.metrics_path("competing_risks.json"))
        return

    # The FULL featurised block, not the regression cohort: abandoners are the positive
    # class, and the regression cohort is exactly the patients who did not abandon.
    frames = {name: context.require(f"features_{name}") for name in BLOCK_NAMES}
    events = {
        name: build_waiting_room_events(frame, context.config)[0]
        for name, frame in frames.items()
    }

    horizons = tuple(
        int(h) for h in context.config.get("competing_risks", {}).get("horizons", DEFAULT_HORIZONS)
    )
    model = train_discrete_time_model(
        frames, events, context.config, schema=context.schema, horizons=horizons
    )
    if not model.models:
        reason = (
            "No horizon had enough abandonment events to fit a classifier. The "
            "counts are in the metrics file."
        )
        logger.warning("Competing-risks stage produced no model: %s", reason)
        save_json(
            {"ran": False, "reason": reason, "horizons": list(horizons)},
            context.metrics_path("competing_risks.json"),
        )
        return

    test_design, _ = make_design_matrix(
        frames["test"], model.feature_set, context.config, context.schema
    )
    levels = frames["test"]["triage_level"].to_numpy()
    metrics, curves = evaluate_discrete_time_model(
        model, test_design, events["test"], levels,
        seen_code=SEEN, abandoned_code=ABANDONED,
    )
    aalen = aalen_johansen_comparison(
        model, test_design, events["test"], levels,
        abandoned_code=ABANDONED, seen_code=SEEN,
    )

    cox = cox_discrimination_comparison(
        events["train"], events["test"], model, test_design, context.config,
        seen_code=SEEN, abandoned_code=ABANDONED,
    )

    tables_dir = Path(context.paths["tables_dir"])
    save_table(metrics, tables_dir / "competing_risks_metrics.csv")
    if not cox.empty:
        save_table(cox, tables_dir / "competing_risks_vs_cox.csv")
    if not curves.empty:
        save_table(curves, tables_dir / "competing_risks_calibration.csv")
    if not aalen.empty:
        save_table(aalen, tables_dir / "competing_risks_aalen.csv")
    save_model(model, Path(context.paths["models_dir"]) / "competing_risks.joblib")

    save_json(
        {
            "ran": True,
            "notice": SYNTHETIC_NOTICE,
            "horizons": list(model.horizons),
            "fitted_horizons": sorted(model.models),
            "feature_set": model.feature_set,
            "calibration_block": "calib",
            "calibration_method": "isotonic",
            "excluded_levels": [1, 2],
            "excluded_reason": (
                "ESI 1-2 effectively never leave without being seen, so there is no "
                "incidence to calibrate against."
            ),
            "metrics": metrics.to_dict(orient="records"),
            "aalen_johansen": aalen.to_dict(orient="records"),
            "cox_comparison": cox.to_dict(orient="records") if not cox.empty else [],
        },
        context.metrics_path("competing_risks.json"),
    )


def _load_calibration(context: PipelineContext) -> Any:
    """Load the calibrated simulation, or calibrate it now if it has not been saved.

    Stages must run standalone, so this reads the saved calibration rather than
    depending on `calibrate_sim` having run in the same process.
    """
    path = resolve_path(Path(context.paths["models_dir"]) / "sim_calibration.joblib")
    if path.is_file():
        return load_model(path)
    logger.info("No saved calibration found; calibrating now")
    return _calibrate(context)


def _calibrate(context: PipelineContext) -> Any:
    """Fit the simulation inputs from data and validate them."""
    from src.simulation.calibrate import calibrate_and_validate

    validated = context.require("validated")
    calibration = calibrate_and_validate(
        validated, context.config, tables_dir=Path(context.paths["tables_dir"])
    )
    save_model(calibration, Path(context.paths["models_dir"]) / "sim_calibration.joblib")
    save_json(calibration.to_dict(), context.metrics_path("sim_calibration.json"))
    return calibration


def stage_calibrate_sim(context: PipelineContext) -> None:
    """Fit the DES inputs from data and check it reproduces the observed department."""
    _calibrate(context)


def stage_scenarios(context: PipelineContext) -> None:
    """Run the named staffing and demand scenarios with common random numbers.

    Refuses to run on a simulation that failed validation unless
    `simulation.allow_invalid_calibration` is set, because a staffing recommendation
    from a model that cannot reproduce today's department is worse than none.
    """
    from src.simulation.scenarios import run_all_scenarios, write_scenario_artifacts
    from src.simulation.sim_figures import write_scenario_figures

    calibration = _load_calibration(context)
    allow_invalid = bool(context.config["simulation"].get("allow_invalid_calibration", False))

    replications, summary, comparison = run_all_scenarios(
        context.config, calibration, allow_invalid=allow_invalid
    )
    write_scenario_artifacts(
        summary, comparison, replications, Path(context.paths["tables_dir"])
    )
    write_scenario_figures(
        summary, comparison, Path(context.paths["figures_dir"]), context.config
    )
    save_json(
        {
            "n_replications": int(context.config["simulation"]["n_replications"]),
            "calibration_passed": calibration.passed,
            "scenarios": summary.to_dict(orient="records"),
            "comparison": comparison.to_dict(orient="records"),
        },
        context.metrics_path("scenarios.json"),
    )


def stage_sensitivity(context: PipelineContext) -> None:
    """Re-run the scenarios with perturbed inputs and check the ranking survives.

    Produces, under artifacts/:
        tables/sensitivity.csv          every KPI in every perturbed world
        tables/sensitivity_ranking.csv  whether each scenario's rank held
        tables/sensitivity_tornado.csv  signed effect of each input on three KPIs
        figures/sensitivity_tornado.png the same, as a tornado plot
    """
    from src.simulation.sensitivity import (
        AGING_FLOOR_SAFETY_NOTE,
        run_sensitivity,
        summarise_axis_sensitivity,
        write_sensitivity_artifacts,
        write_tornado_figure,
    )

    calibration = _load_calibration(context)
    results, stability = run_sensitivity(context.config, calibration)
    tables_dir = Path(context.paths["tables_dir"])
    by_world = write_sensitivity_artifacts(results, stability, tables_dir)

    # "Which input is this recommendation most hostage to" is a different question
    # from "does the ranking hold", and needs the movement of each KPI, not the order.
    axis_summary = summarise_axis_sensitivity(results)
    save_table(axis_summary, tables_dir / "sensitivity_tornado.csv")
    write_tornado_figure(axis_summary, Path(context.paths["figures_dir"]))

    save_json(
        {
            "ranking_stable": bool(stability["rank_is_stable"].all()),
            "n_worlds": int(results["perturbation"].nunique()),
            "axes": sorted(results["perturbation_axis"].dropna().unique().tolist())
            if "perturbation_axis" in results.columns
            else [],
            "aging_floor_safety_note": AGING_FLOOR_SAFETY_NOTE,
            "stability": stability.to_dict(orient="records"),
            "by_world": by_world.to_dict(orient="records"),
            "tornado": axis_summary.to_dict(orient="records"),
            # Noise-control worlds are excluded: they change nothing about the
            # department, so ranking them among the sensitivities would be misleading.
            "most_sensitive": {
                kpi: group[~group["noise_control"]].head(3).to_dict(orient="records")
                for kpi, group in axis_summary.groupby("kpi")
            }
            if not axis_summary.empty
            else {},
            "monte_carlo_noise_floor": {
                str(kpi): float(group["noise_floor"].dropna().iloc[0])
                for kpi, group in axis_summary.groupby("kpi")
                if not group["noise_floor"].dropna().empty
            }
            if not axis_summary.empty
            else {},
        },
        context.metrics_path("sensitivity.json"),
    )


def stage_staffing_search(context: PipelineContext) -> None:
    """Search staffing plans against forecast demand and cost the feasible ones.

    This is the G1 link: `forecast()` output becomes the load the DES evaluates each
    candidate plan against.
    """
    from src.simulation.staffing_search import search_staffing, write_search_artifacts

    calibration = _load_calibration(context)
    allow_invalid = bool(context.config["simulation"].get("allow_invalid_calibration", False))

    forecast_frame = None
    forecast_path = context.processed("forecast")
    if forecast_path.is_file():
        forecast_frame = read_visits(forecast_path)
    else:
        logger.warning(
            "No forecast on disk; the staffing search will use the historical arrival "
            "pattern. Run the 'forecast' stage first to close the forecast-to-staffing link."
        )

    result = search_staffing(
        context.config, calibration, forecast_frame=forecast_frame, allow_invalid=allow_invalid
    )
    write_search_artifacts(result, Path(context.paths["tables_dir"]))
    save_json(result.to_dict(), context.metrics_path("staffing_search.json"))


def stage_report(context: PipelineContext) -> None:
    """Assemble artifacts/report.md from everything the other stages produced."""
    from src.report import write_report

    write_report(context.config)


#: Stage name -> (callable, phase that implements it). A `None` callable means the
#: stage is planned but not yet built; asking for it raises a clear error.
#: Config `paths` entries that name where a run writes. `--artifacts-dir` and
#: `--run-name` move these; the inputs (`schema`, `column_map`, `raw_file`) stay put.
OUTPUT_PATH_KEYS = (
    "artifacts_dir",
    "models_dir",
    "metrics_dir",
    "figures_dir",
    "tables_dir",
    "report",
)


def redirect_outputs(
    config: dict[str, Any],
    artifacts_dir: str | Path,
    processed_dir: str | Path | None = None,
    log_file: str | Path | None = None,
) -> dict[str, Any]:
    """Point a run's outputs somewhere other than the shared `artifacts/`.

    This is how one pipeline run is kept from overwriting another: the upload
    workflow gives every job its own folder, and `--artifacts-dir` gives the same
    ability to anyone running the CLI.

    Absolute paths are written. `resolve_path` resolves a relative path against the
    repository root, so writing absolute paths here means a stage cannot land back
    in `artifacts/` whatever directory it was launched from.

    Args:
        config: The config to modify. Mutated and returned, so pass a copy unless
            you mean to change the caller's.
        artifacts_dir: Where models, metrics, figures, tables and the report go.
        processed_dir: Where intermediate parquet goes. Left alone when omitted --
            which is what `--artifacts-dir` alone does, so two runs over the same
            inputs can share the intermediates and still write their results apart.
        log_file: Where this run's log goes.

    Returns:
        The same config, for chaining.
    """
    base = Path(artifacts_dir)
    config["paths"]["artifacts_dir"] = str(resolve_path(base))
    config["paths"]["models_dir"] = str(resolve_path(base / "models"))
    config["paths"]["metrics_dir"] = str(resolve_path(base / "metrics"))
    config["paths"]["figures_dir"] = str(resolve_path(base / "figures"))
    config["paths"]["tables_dir"] = str(resolve_path(base / "tables"))
    config["paths"]["report"] = str(resolve_path(base / "report.md"))

    if processed_dir is not None:
        config["paths"]["processed_dir"] = str(resolve_path(Path(processed_dir)))
    if log_file is not None:
        config["logging"] = dict(config.get("logging", {}))
        config["logging"]["file"] = str(resolve_path(Path(log_file)))
    return config


def run_directory(config: dict[str, Any], run_name: str) -> Path:
    """The folder a named run owns, under the configured runs directory."""
    runs_dir = (config.get("uploads") or {}).get("runs_dir", "runs")
    return resolve_path(Path(runs_dir) / run_name)


def redirect_to_run(config: dict[str, Any], run_root: str | Path) -> dict[str, Any]:
    """Point *everything* a run writes -- intermediates included -- into one folder.

    Used by `--run-name` and by the upload workflow, which needs the stronger
    guarantee: a job must not share even a parquet file with another run.
    """
    root = Path(run_root)
    return redirect_outputs(
        config,
        artifacts_dir=root / "artifacts",
        processed_dir=root / "data" / "processed",
        log_file=root / "pipeline.log",
    )


STAGE_ORDER: dict[str, tuple[Callable[[PipelineContext], None] | None, int]] = {
    "load": (stage_load, 0),
    "validate": (stage_validate, 0),
    "split": (stage_split, 0),
    "features": (stage_features, 1),
    "train_wait": (stage_train_wait, 1),
    "ablation": (stage_train_wait, 1),
    "explain": (stage_explain, 1),
    "fairness": (stage_fairness, 1),
    "forecast": (stage_forecast, 2),
    "survival": (stage_survival, 2),
    "competing_risks": (stage_competing_risks, 2),
    "calibrate_sim": (stage_calibrate_sim, 3),
    "scenarios": (stage_scenarios, 3),
    "sensitivity": (stage_sensitivity, 3),
    "staffing_search": (stage_staffing_search, 3),
    "report": (stage_report, 4),
}


def implemented_stages() -> list[str]:
    """Names of the stages that can run today, in pipeline order."""
    return [name for name, (fn, _) in STAGE_ORDER.items() if fn is not None]


def resolve_stages(requested: str | None) -> list[str]:
    """Turn a `--stages` argument into a validated, ordered list of stage names.

    Args:
        requested: Comma-separated stage names, or None for every implemented stage.

    Returns:
        Stage names in pipeline order.

    Raises:
        ValueError: If a name is unknown, or names a stage from a later phase.
    """
    if not requested:
        return implemented_stages()

    names = [part.strip() for part in requested.split(",") if part.strip()]
    unknown = [name for name in names if name not in STAGE_ORDER]
    if unknown:
        raise ValueError(
            f"Unknown stage(s) {unknown}. Known stages, in order: {list(STAGE_ORDER)}"
        )
    not_built = [(name, STAGE_ORDER[name][1]) for name in names if STAGE_ORDER[name][0] is None]
    if not_built:
        detail = ", ".join(f"{name} (phase {phase})" for name, phase in not_built)
        raise ValueError(
            f"These stages are not implemented yet: {detail}. "
            f"Available now: {implemented_stages()}"
        )
    return [name for name in STAGE_ORDER if name in set(names)]


def run_pipeline(
    config_path: str = "configs/default.yaml",
    data_source: str = "synthetic",
    stages: str | None = None,
    artifacts_dir: str | Path | None = None,
    run_name: str | None = None,
) -> PipelineContext:
    """Run the requested stages end to end.

    Args:
        config_path: Path to the YAML config.
        data_source: Either "synthetic" or "real".
        stages: Comma-separated subset of stages; all implemented stages when omitted.
        artifacts_dir: Write models, metrics, figures, tables and the report here
            instead of the configured `artifacts/`. Intermediates are left where
            they are.
        run_name: Write everything, intermediates included, to
            `<uploads.runs_dir>/<run_name>/`. Use this to keep a run wholly separate
            from the baseline.

    Returns:
        The context, whose `stage_timings` records how long each stage took.

    Raises:
        ValueError: If `data_source` is not a known source, a stage name is invalid,
            or both `artifacts_dir` and `run_name` are given.
    """
    if data_source not in DATA_SOURCES:
        raise ValueError(f"--data must be one of {DATA_SOURCES}, got {data_source!r}")
    if artifacts_dir and run_name:
        raise ValueError(
            "Pass --artifacts-dir or --run-name, not both: --run-name already "
            "chooses where everything goes."
        )

    config = load_config(config_path)
    if run_name:
        root = run_directory(config, run_name)
        redirect_to_run(config, root)
        logger.info("Run %r writes everything to %s", run_name, root)
    elif artifacts_dir:
        redirect_outputs(config, artifacts_dir)
        logger.info("Outputs redirected to %s", config["paths"]["artifacts_dir"])
    setup_logging(config["logging"]["level"], config["logging"].get("file"))
    set_global_seeds(config["seed"])
    logger.info("Global seed set to %s", config["seed"])

    context = PipelineContext(
        config=config, data_source=data_source, schema=load_schema(config["paths"]["schema"])
    )
    selected = resolve_stages(stages)
    logger.info("Running stages: %s", " -> ".join(selected))

    # Written before the first stage, because the report stage runs inside this loop
    # and needs it: without it the report cannot tell an artifact this run was never
    # going to produce from one a stage should have written and did not.
    from src.report import save_run_stages

    save_run_stages(
        config,
        run_type="pipeline run" if stages is None else f"pipeline run of {stages}",
        performed=selected,
        not_performed={
            name: "not in the stage list this run was asked for"
            for name, (stage_fn, _) in STAGE_ORDER.items()
            if stage_fn is not None and name not in set(selected)
        },
    )

    for name in selected:
        stage_fn, _ = STAGE_ORDER[name]
        assert stage_fn is not None  # resolve_stages already rejected unbuilt stages
        logger.info("--- stage %s: start ---", name)
        started = time.perf_counter()
        stage_fn(context)
        elapsed = time.perf_counter() - started
        context.stage_timings[name] = elapsed
        logger.info("--- stage %s: done in %.2fs ---", name, elapsed)

    total = sum(context.stage_timings.values())
    logger.info("Pipeline finished: %d stage(s) in %.2fs", len(context.stage_timings), total)
    save_json(
        {
            "data_source": data_source,
            "config": config["_config_path"],
            "stages_run": selected,
            "stage_seconds": context.stage_timings,
            "total_seconds": total,
            "stages_not_yet_implemented": {
                name: f"phase {phase}" for name, (fn, phase) in STAGE_ORDER.items() if fn is None
            },
        },
        context.metrics_path("pipeline_run.json"),
    )
    return context


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="ED forecasting, wait-time prediction and staffing simulation pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run pipeline stages")
    run.add_argument("--config", default="configs/default.yaml", help="path to the YAML config")
    run.add_argument(
        "--data",
        default="synthetic",
        choices=DATA_SOURCES,
        help="synthetic generator, or the real extract mapped via column_map.yaml",
    )
    run.add_argument(
        "--stages",
        default=None,
        help="comma-separated subset of stages (default: every implemented stage)",
    )
    run.add_argument(
        "--artifacts-dir",
        default=None,
        help=(
            "write models, metrics, figures, tables and the report here instead of "
            "artifacts/ (intermediates stay where they are)"
        ),
    )
    run.add_argument(
        "--run-name",
        default=None,
        help=(
            "write everything, intermediates included, to runs/<name>/ -- a run "
            "wholly separate from the baseline artifacts"
        ),
    )

    subparsers.add_parser("list-stages", help="show the stage order and what is implemented")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    if args.command == "list-stages":
        setup_logging("INFO")
        for index, (name, (fn, phase)) in enumerate(STAGE_ORDER.items(), start=1):
            status = "ready" if fn is not None else f"planned (phase {phase})"
            logger.info("%2d. %-16s %s", index, name, status)
        return 0

    try:
        run_pipeline(
            config_path=args.config,
            data_source=args.data,
            stages=args.stages,
            artifacts_dir=args.artifacts_dir,
            run_name=args.run_name,
        )
    except (ValueError, FileNotFoundError) as error:
        setup_logging("INFO")
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
