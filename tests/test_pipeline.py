"""Tests for the pipeline CLI and its stage contract.

Every test redirects all output paths into a temporary directory, so the suite never
touches the real `data/` or `artifacts/` trees.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.pipeline import (
    STAGE_ORDER,
    implemented_stages,
    main,
    resolve_stages,
    run_pipeline,
)
from src.utils.io import load_json, resolve_path

TINY_N = 1500


@pytest.fixture()
def sandbox_config(default_config: dict, tmp_path: Path) -> Path:
    """Write a copy of the config whose every output path points inside tmp_path."""
    cfg = copy.deepcopy(default_config)
    cfg.pop("_config_path", None)
    cfg["synthetic"]["n_patients"] = TINY_N
    cfg["logging"]["file"] = str(tmp_path / "pipeline.log")

    # Shrink every expensive stage. These tests check that the stages wire together
    # and write what they claim to; the statistical behaviour of each is covered by
    # its own module's tests, where the fixtures are sized for it.
    cfg["wait_model"]["regressor"]["n_estimators"] = 40
    cfg["wait_model"]["classifier"]["n_estimators"] = 40
    cfg["explain"]["shap_max_samples"] = 200
    cfg["fairness"]["bootstrap_resamples"] = 20
    cfg["simulation"]["horizon_hours"] = 72
    cfg["simulation"]["warmup_hours"] = 6
    cfg["simulation"]["n_replications"] = 4
    cfg["simulation"]["calibration"]["replications"] = 2
    cfg["simulation"]["calibration"]["search_steps"] = 4
    cfg["simulation"]["calibration"]["search_rounds"] = 3
    cfg["simulation"]["sensitivity"] = {"factors": [1.2], "n_replications": 2}
    cfg["staffing_search"]["n_replications"] = 2
    cfg["staffing_search"]["strategy"] = "random"
    cfg["staffing_search"]["n_random_samples"] = 4
    # A 10-day, 1500-patient extract cannot calibrate a simulation, and the gate
    # correctly refuses it. These tests check the stages wire together, so they take
    # the override; `test_simulation.py` covers the gate itself, both ways, and
    # `test_scenarios_blocked_without_the_override` below checks it fires here too.
    cfg["simulation"]["allow_invalid_calibration"] = True

    for key in ("raw_dir", "processed_dir", "synthetic_dir", "artifacts_dir"):
        cfg["paths"][key] = str(tmp_path / key)
    for key in ("models_dir", "metrics_dir", "figures_dir", "tables_dir"):
        cfg["paths"][key] = str(tmp_path / "artifacts_dir" / key.replace("_dir", ""))
    cfg["paths"]["synthetic_file"] = str(tmp_path / "synthetic_dir" / "ed_visits.parquet")
    cfg["paths"]["raw_file"] = str(tmp_path / "raw_dir" / "ed_visits.csv")
    cfg["paths"]["report"] = str(tmp_path / "artifacts_dir" / "report.md")

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def test_stage_order_declares_every_spec_stage() -> None:
    """The stage list must match the sequence the project specification fixes."""
    assert list(STAGE_ORDER) == [
        "load",
        "validate",
        "split",
        "features",
        "train_wait",
        "ablation",
        "explain",
        "fairness",
        "forecast",
        "survival",
        "competing_risks",
        "calibrate_sim",
        "scenarios",
        "sensitivity",
        "staffing_search",
        "report",
    ]


def test_implemented_stages_are_those_built_so_far() -> None:
    """Phases 0 and 1a ship these; later stages are declared, not faked."""
    assert implemented_stages() == [
        "load",
        "validate",
        "split",
        "features",
        "train_wait",
        "ablation",
        "explain",
        "fairness",
        "forecast",
        "survival",
        "competing_risks",
        "calibrate_sim",
        "scenarios",
        "sensitivity",
        "staffing_search",
        "report",
    ]


def test_resolve_stages_defaults_to_everything_implemented() -> None:
    """Omitting --stages runs the whole implemented pipeline, in order."""
    assert resolve_stages(None) == [
        "load",
        "validate",
        "split",
        "features",
        "train_wait",
        "ablation",
        "explain",
        "fairness",
        "forecast",
        "survival",
        "competing_risks",
        "calibrate_sim",
        "scenarios",
        "sensitivity",
        "staffing_search",
        "report",
    ]
    # Requested stages are returned in pipeline order, not the order they were typed.
    assert resolve_stages("split,load") == ["load", "split"]


def test_resolve_stages_rejects_unknown_names() -> None:
    """A typo must not silently skip work."""
    with pytest.raises(ValueError, match="Unknown stage"):
        resolve_stages("lod")


def test_resolve_stages_names_the_phase_for_unbuilt_stages(monkeypatch) -> None:
    """Asking for a stage that is declared but not built says which phase owns it.

    Every stage is implemented now, so the mechanism is exercised by temporarily
    un-building one rather than by leaving a real gap in the pipeline.
    """
    import src.pipeline as pipeline_module

    patched = dict(pipeline_module.STAGE_ORDER)
    patched["report"] = (None, 4)
    monkeypatch.setattr(pipeline_module, "STAGE_ORDER", patched)

    with pytest.raises(ValueError, match=r"report \(phase 4\)"):
        resolve_stages("load,report")


def test_full_run_produces_every_foundation_artifact(sandbox_config: Path, tmp_path: Path) -> None:
    """A complete run writes the processed tables and the metrics JSONs."""
    context = run_pipeline(config_path=str(sandbox_config), data_source="synthetic")

    assert set(context.stage_timings) == {
        "load",
        "validate",
        "split",
        "features",
        "train_wait",
        "ablation",
        "explain",
        "fairness",
        "forecast",
        "survival",
        "competing_risks",
        "calibrate_sim",
        "scenarios",
        "sensitivity",
        "staffing_search",
        "report",
    }

    for name in ("visits", "validated", "flagged", "train", "val", "calib", "test"):
        assert context.processed(name).is_file(), f"missing processed file for {name}"
    for name in ("validation.json", "split.json", "pipeline_run.json"):
        assert context.metrics_path(name).is_file(), f"missing metrics artifact {name}"

    validation = load_json(context.metrics_path("validation.json"))
    assert validation["n_input"] == TINY_N
    assert validation["n_valid"] + validation["n_dropped"] == TINY_N
    assert validation["n_modelling"] + validation["n_flagged_no_seen_time"] == validation["n_valid"]

    split = load_json(context.metrics_path("split.json"))
    assert split["bounds"]["train_end"] < split["bounds"]["val_end"]
    assert split["bounds"]["val_end"] < split["bounds"]["calib_end"]
    assert split["train"]["n"] > split["val"]["n"]

    run = load_json(context.metrics_path("pipeline_run.json"))
    assert run["data_source"] == "synthetic"
    assert run["stages_run"] == [
        "load",
        "validate",
        "split",
        "features",
        "train_wait",
        "ablation",
        "explain",
        "fairness",
        "forecast",
        "survival",
        "competing_risks",
        "calibrate_sim",
        "scenarios",
        "sensitivity",
        "staffing_search",
        "report",
    ]
    # Every declared stage is now built, so nothing should be pending.
    assert run["stages_not_yet_implemented"] == {}


def test_intermediate_tables_are_parquet_and_round_trip_dtypes(sandbox_config: Path) -> None:
    """Parquet is used so timestamps and the boolean flags survive a write/read cycle."""
    context = run_pipeline(config_path=str(sandbox_config), data_source="synthetic")

    for name in ("visits", "validated", "flagged", "train", "val", "calib", "test"):
        assert context.processed(name).suffix == ".parquet"

    import pandas as pd

    validated = context.require("validated")
    assert pd.api.types.is_datetime64_any_dtype(validated["arrival_time"])
    assert pd.api.types.is_datetime64_any_dtype(validated["seen_time"])
    assert validated["is_lwbs"].dtype == bool
    assert validated["still_waiting_at_cutoff"].dtype == bool
    assert validated["triage_level"].dtype == "int16"


def test_validate_writes_the_dataset_profile(sandbox_config: Path) -> None:
    """The descriptive profile is a pipeline artifact, not a one-off script."""
    import pandas as pd

    context = run_pipeline(config_path=str(sandbox_config), data_source="synthetic")
    tables_dir = Path(context.paths["tables_dir"])

    expected = {
        "profile_arrivals_by_hour.csv": ("hour_of_day", 24),
        "profile_arrivals_by_weekday.csv": ("day_of_week", 7),
        "profile_triage_mix.csv": ("triage_level", 5),
        "profile_wait_by_triage.csv": ("triage_level", 5),
    }
    for name, (index_column, n_rows) in expected.items():
        frame = pd.read_csv(tables_dir / name)
        assert len(frame) == n_rows, f"{name} has {len(frame)} rows"
        assert index_column in frame.columns


def test_profile_reflects_the_generators_structure(sandbox_config: Path) -> None:
    """The profile must show the demand curve, the configured mix and the acuity ordering."""
    import pandas as pd

    context = run_pipeline(config_path=str(sandbox_config), data_source="synthetic")
    tables_dir = Path(context.paths["tables_dir"])

    by_hour = pd.read_csv(tables_dir / "profile_arrivals_by_hour.csv").set_index("hour_of_day")
    assert by_hour.loc[9:12, "mean"].mean() > 2.0 * by_hour.loc[2:5, "mean"].mean()
    assert by_hour["share_of_arrivals"].sum() == pytest.approx(1.0, abs=1e-3)

    mix = pd.read_csv(tables_dir / "profile_triage_mix.csv")
    assert (mix["difference"].abs() < 0.03).all(), "observed mix drifted from the configured mix"

    waits = pd.read_csv(tables_dir / "profile_wait_by_triage.csv").set_index("triage_level")
    assert waits.loc[1, "mean"] < waits.loc[3, "mean"] < waits.loc[5, "mean"]
    # Censoring is reported, and it concentrates in the low-acuity levels.
    assert waits.loc[5, "never_seen_share"] > waits.loc[1, "never_seen_share"]
    assert (waits["n_seen"] <= waits["n_arrived"]).all()

    # The two reasons for a missing wait are reported separately, and they add up.
    assert (waits["n_seen"] + waits["n_lwbs"] + waits["n_still_waiting_at_cutoff"]
            == waits["n_arrived"]).all()
    import numpy as np

    assert np.allclose(
        waits["lwbs_share"] + waits["still_waiting_share"],
        waits["never_seen_share"],
        atol=1e-3,
    )


def test_rerunning_is_idempotent(sandbox_config: Path) -> None:
    """Stages are deterministic, so a second run reproduces byte-identical outputs."""
    first = run_pipeline(config_path=str(sandbox_config), data_source="synthetic")
    snapshots = {
        name: first.processed(name).read_bytes()
        for name in ("visits", "validated", "flagged", "train", "val", "calib", "test")
    }
    metrics_before = load_json(first.metrics_path("validation.json"))

    second = run_pipeline(config_path=str(sandbox_config), data_source="synthetic")
    for name, content in snapshots.items():
        assert second.processed(name).read_bytes() == content, f"{name} changed on re-run"
    assert load_json(second.metrics_path("validation.json")) == metrics_before


def test_each_stage_runs_independently(sandbox_config: Path) -> None:
    """Running stages one at a time, in separate invocations, must work."""
    for stage in ("load", "validate", "split", "features", "train_wait"):
        context = run_pipeline(
            config_path=str(sandbox_config), data_source="synthetic", stages=stage
        )
        assert list(context.stage_timings) == [stage]

    assert context.processed("test").is_file()


def test_a_stage_without_its_inputs_explains_what_to_run(sandbox_config: Path) -> None:
    """Running 'split' from a cold start names the stage that produces its input."""
    with pytest.raises(FileNotFoundError, match="Run the 'validate' stage first"):
        run_pipeline(config_path=str(sandbox_config), data_source="synthetic", stages="split")


def test_split_outputs_are_disjoint_and_ordered(sandbox_config: Path) -> None:
    """The on-disk splits carry the same guarantees as the in-memory ones."""
    context = run_pipeline(config_path=str(sandbox_config), data_source="synthetic")
    blocks = {name: context.require(name) for name in ("train", "val", "calib", "test")}

    for earlier, later in (("train", "val"), ("val", "calib"), ("calib", "test")):
        assert (
            blocks[earlier]["arrival_time"].max() < blocks[later]["arrival_time"].min()
        ), f"{earlier} overlaps {later}"

    # Every validated row lands in exactly one block, including patients never seen.
    assert sum(len(f) for f in blocks.values()) == len(context.require("validated"))
    identifiers = pd.concat([f["patient_id"] for f in blocks.values()])
    assert identifiers.is_unique


def test_never_seen_patients_reach_blocks_but_not_the_regression_cohort(
    sandbox_config: Path,
) -> None:
    """Patients never seen stay in the blocks; the model stage excludes them itself.

    Phase 2's survival model needs exactly the rows the wait regressor must drop, so
    the split keeps everyone and each model stage selects its own cohort.
    """
    from src.models.wait_time import select_regression_cohort

    context = run_pipeline(config_path=str(sandbox_config), data_source="synthetic")
    flagged_ids = set(context.require("flagged")["patient_id"])
    assert flagged_ids, "fixture should contain patients who were never seen"

    reached: set = set()
    for name in ("train", "val", "calib", "test"):
        block = context.require(name)
        reached |= flagged_ids & set(block["patient_id"])

        cohort = select_regression_cohort(block, block_name=name)
        assert cohort.frame["wait_minutes"].notna().all()
        assert not flagged_ids & set(cohort.frame["patient_id"])

    assert reached == flagged_ids, "every never-seen patient should land in some block"


def test_real_source_without_an_extract_fails_clearly(sandbox_config: Path) -> None:
    """--data real with no file present must say where to put it."""
    with pytest.raises(FileNotFoundError, match="Raw dataset not found"):
        run_pipeline(config_path=str(sandbox_config), data_source="real", stages="load")


def test_unknown_data_source_raises(sandbox_config: Path) -> None:
    """Only the two declared sources exist."""
    with pytest.raises(ValueError, match="--data must be one of"):
        run_pipeline(config_path=str(sandbox_config), data_source="imaginary")


def test_cli_run_returns_zero(sandbox_config: Path) -> None:
    """The documented command line works and reports success."""
    code = main(["run", "--config", str(sandbox_config), "--data", "synthetic"])
    assert code == 0


def test_cli_reports_failure_without_a_traceback(sandbox_config: Path) -> None:
    """Configuration mistakes exit non-zero with a logged message, not a stack trace."""
    code = main(["run", "--config", str(sandbox_config), "--stages", "forecast"])
    assert code == 1


def test_cli_list_stages_returns_zero() -> None:
    """`list-stages` is the documented way to see what is built."""
    assert main(["list-stages"]) == 0


def test_metrics_artifacts_are_valid_json(sandbox_config: Path) -> None:
    """Artifacts are consumed by the report stage and by humans; they must parse."""
    context = run_pipeline(config_path=str(sandbox_config), data_source="synthetic")
    metrics_dir = context.metrics_path("validation.json").parent

    written = sorted(metrics_dir.glob("*.json"))
    assert [p.name for p in written] == [
        "competing_risks.json",
        "fairness.json",
        "features.json",
        "forecast.json",
        "pipeline_run.json",
        "run_stages.json",
        "scenarios.json",
        "sensitivity.json",
        "shap.json",
        "sim_calibration.json",
        "split.json",
        "staffing_search.json",
        "survival.json",
        "validation.json",
        "wait_models.json",
    ]
    for path in written:
        json.loads(path.read_text(encoding="utf-8"))


def test_features_stage_writes_a_matrix_per_split(sandbox_config: Path) -> None:
    """The features stage produces one featurised table per split, plus its metrics."""
    context = run_pipeline(config_path=str(sandbox_config), data_source="synthetic")
    summary = load_json(context.metrics_path("features.json"))

    assert set(summary["splits"]) == {"train", "val", "calib", "test"}
    assert set(summary["feature_sets"]) == {"base", "base+queue"}
    # base+queue must strictly contain base, which is what makes the ablation valid.
    base = set(summary["feature_sets"]["base"]["columns"])
    full = set(summary["feature_sets"]["base+queue"]["columns"])
    assert base < full

    for split_name in ("train", "val", "calib", "test"):
        frame = context.require(f"features_{split_name}")
        assert len(frame) == summary["splits"][split_name]["n_rows"]
        assert "n_waiting" in frame.columns
        assert "prolonged_wait" in frame.columns


def test_features_stage_needs_the_split_stage_first(sandbox_config: Path) -> None:
    """Running features from a cold start names the stage that produces its input."""
    with pytest.raises(FileNotFoundError, match="Run the 'validate' stage first"):
        run_pipeline(config_path=str(sandbox_config), data_source="synthetic", stages="features")


def test_train_wait_stage_writes_every_artifact(sandbox_config: Path) -> None:
    """Phase 1b must leave tables, figures, models and metrics on disk."""
    context = run_pipeline(config_path=str(sandbox_config), data_source="synthetic")

    tables_dir = Path(context.paths["tables_dir"])
    for name in (
        "ablation.csv",
        "classifier_summary.csv",
        "conformal_coverage.csv",
        "seen_only_bias.csv",
    ):
        assert (tables_dir / name).is_file(), f"missing table {name}"

    figures_dir = Path(context.paths["figures_dir"])
    for name in (
        "ablation_error.png",
        "conformal_coverage.png",
        "seen_only_bias.png",
        "prediction_quality.png",
    ):
        figure = figures_dir / name
        assert figure.is_file(), f"missing figure {name}"
        assert figure.stat().st_size > 5000, f"{name} looks empty"

    models_dir = Path(context.paths["models_dir"])
    saved = {p.name for p in models_dir.glob("*.joblib")}
    assert "baseline_rolling_mean.joblib" in saved
    assert "conformal_calibrator.joblib" in saved
    assert any(n.startswith("wait_regressor__") for n in saved)
    assert any(n.startswith("classifier__") for n in saved)


def test_ablation_csv_carries_its_caveat(sandbox_config: Path) -> None:
    """The synthetic-data caveat must travel with the table, not live only in prose."""
    context = run_pipeline(config_path=str(sandbox_config), data_source="synthetic")
    path = Path(context.paths["tables_dir"]) / "ablation.csv"

    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    assert first_line.startswith("# CAVEAT")
    assert "synthetic" in first_line.lower()

    # And the table still parses when the comment is honoured.
    table = pd.read_csv(path, comment="#")
    assert set(table["feature_set"]) == {"base", "base+queue"}
    assert set(table["target"]) == {"raw", "log1p"}


def test_wait_models_metrics_record_the_blocks_used(sandbox_config: Path) -> None:
    """The metrics JSON must make the block discipline auditable after the fact."""
    context = run_pipeline(config_path=str(sandbox_config), data_source="synthetic")
    metrics = load_json(context.metrics_path("wait_models.json"))

    assert metrics["conformal"]["calibration_block"] == "calib"
    assert set(metrics["blocks"]) == {"train", "val", "calib", "test"}
    for block in metrics["blocks"].values():
        cohort = block["cohort"]
        assert cohort["n_input"] == cohort["n_kept"] + cohort["n_excluded"]

    for row in metrics["classifiers"]:
        assert row["threshold_source"] == "val"

    assert metrics["seen_only_bias"]["direction"].startswith("downward")
    assert "CAVEAT" in metrics["ablation"]["caveat"]


def test_forecast_skips_cleanly_on_a_short_extract(sandbox_config: Path) -> None:
    """A 10-day extract cannot support a 168-hour seasonal lag, and must say so.

    Graceful degradation matters here: a site piloting the pipeline on three weeks of
    data should get an explanation, not a stack trace.
    """
    context = run_pipeline(config_path=str(sandbox_config), data_source="synthetic")
    payload = load_json(context.metrics_path("forecast.json"))

    assert payload["ran"] is False
    assert "hours of history" in payload["reason"]
    assert "seasonal" in payload["reason"]


def test_forecast_runs_when_the_extract_is_long_enough(
    default_config: dict, tmp_path: Path
) -> None:
    """With a few months of data the forecast stage produces its full artifacts."""
    import copy

    import yaml

    cfg = copy.deepcopy(default_config)
    cfg.pop("_config_path", None)
    cfg["synthetic"]["n_patients"] = 16000
    cfg["forecast"]["model"]["n_estimators"] = 80
    cfg["forecast"]["train_stride_hours"] = 12
    cfg["forecast"]["rolling_origin"]["n_folds"] = 2
    cfg["logging"]["file"] = str(tmp_path / "pipeline.log")
    for key in ("raw_dir", "processed_dir", "synthetic_dir", "artifacts_dir"):
        cfg["paths"][key] = str(tmp_path / key)
    for key in ("models_dir", "metrics_dir", "figures_dir", "tables_dir"):
        cfg["paths"][key] = str(tmp_path / "artifacts_dir" / key.replace("_dir", ""))
    cfg["paths"]["synthetic_file"] = str(tmp_path / "synthetic_dir" / "ed_visits.parquet")
    cfg["paths"]["raw_file"] = str(tmp_path / "raw_dir" / "ed_visits.csv")
    cfg["paths"]["report"] = str(tmp_path / "artifacts_dir" / "report.md")

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    context = run_pipeline(
        config_path=str(path), data_source="synthetic",
        stages="load,validate,split,forecast",
    )
    payload = load_json(context.metrics_path("forecast.json"))

    assert payload.get("ran") is True
    assert payload["overall"]["model"]["mae"] < payload["overall"]["seasonal_naive"]["mae"]
    assert (Path(context.paths["tables_dir"]) / "forecast_accuracy.csv").is_file()
    assert context.processed("forecast").is_file()


def test_scenarios_are_blocked_without_the_override(sandbox_config: Path) -> None:
    """The validation gate must fire on a real pipeline run, not just in unit tests.

    The sandbox is deliberately too small to calibrate, so with the override removed
    the scenario stage must refuse rather than produce staffing advice from a model
    that cannot reproduce the department.
    """
    import yaml

    from src.simulation.calibrate import ValidationError

    config = yaml.safe_load(sandbox_config.read_text(encoding="utf-8"))
    config["simulation"]["allow_invalid_calibration"] = False
    strict = sandbox_config.parent / "strict_config.yaml"
    strict.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValidationError, match="does not reproduce the observed department"):
        run_pipeline(
            config_path=str(strict),
            data_source="synthetic",
            stages="load,validate,split,calibrate_sim,scenarios",
        )


def test_simulation_stages_write_their_artifacts(sandbox_config: Path) -> None:
    """Phase 3 stages must leave their tables and the calibration on disk."""
    context = run_pipeline(config_path=str(sandbox_config), data_source="synthetic")
    tables = Path(context.paths["tables_dir"])

    for name in (
        "sim_validation.csv",
        "scenarios.csv",
        "scenario_comparison.csv",
        "sensitivity.csv",
        "sensitivity_ranking.csv",
        "staffing_search.csv",
    ):
        assert (tables / name).is_file(), f"missing {name}"

    assert (Path(context.paths["models_dir"]) / "sim_calibration.joblib").is_file()

    search = load_json(context.metrics_path("staffing_search.json"))
    # Whatever the outcome, an infeasible plan must never be returned as the choice.
    if search["any_feasible"]:
        assert search["cheapest_feasible_plan"]["feasible"] is True
    else:
        assert search["cheapest_feasible_plan"] is None


def test_end_to_end_smoke_on_a_tiny_dataset(sandbox_config: Path) -> None:
    """Every stage runs on a tiny synthetic dataset and leaves the report on disk.

    This is the guard that the fifteen stages still compose. It deliberately uses the
    smallest viable dataset: stages that cannot work at that size (forecasting needs
    weeks of history) must degrade with an explanation rather than crash, and this test
    asserts that they do.
    """
    context = run_pipeline(config_path=str(sandbox_config), data_source="synthetic")

    assert list(context.stage_timings) == implemented_stages()

    report_path = resolve_path(context.config["paths"]["report"])
    assert report_path.is_file()
    report = report_path.read_text(encoding="utf-8")
    assert "# ED Intelligence" in report
    assert "Decision support only" in report
    assert "## Limitations" in report
    assert "## Future work" in report

    # The forecast stage cannot run on ten days of history and must say why.
    forecast = load_json(context.metrics_path("forecast.json"))
    assert forecast["ran"] is False
    assert "hours of history" in forecast["reason"]
    assert "Not available" in report or "Skipped" in report


def test_every_stage_is_individually_runnable(sandbox_config: Path) -> None:
    """Each stage must run on its own from a cold start, in order."""
    for stage in implemented_stages():
        context = run_pipeline(
            config_path=str(sandbox_config), data_source="synthetic", stages=stage
        )
        assert list(context.stage_timings) == [stage]
