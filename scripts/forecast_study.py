"""Run the extended forecast study and write its artifacts.

This is offline research work, like the sensitivity analysis: it refits a forecaster at
every one of 26+ weekly origins for three model variants, which takes minutes rather
than seconds, so it is a script rather than a pipeline stage. Nothing in the default
pipeline calls it and nothing it writes changes the default run's results.

    python scripts/forecast_study.py --run-name forecast_study

Outputs, under the chosen artifacts directory:

    tables/forecast_study_scores.csv       every variant, both blocks, both scopes
    tables/forecast_study_coverage.csv     rolling conformal coverage by hour band
    tables/forecast_study_blend_grid.csv   the blend weights considered on validation
    tables/forecast_study_predictions.csv  the raw backtest rows
    metrics/forecast_study.json            all of the above, machine-readable

All data involved is SYNTHETIC.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/forecast_study.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.forecast import build_hourly_series  # noqa: E402
from src.models.forecast_study import MIN_ORIGINS, run_forecast_study  # noqa: E402
from src.utils.io import (  # noqa: E402
    load_config,
    read_visits,
    resolve_path,
    save_json,
    save_table,
)
from src.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python scripts/forecast_study.py",
        description=(
            "Rolling-origin forecast study: 26+ weekly origins, five variants, "
            "rolling conformal intervals. Writes to its own run folder."
        ),
    )
    parser.add_argument("--config", default="configs/default.yaml", help="pipeline config")
    parser.add_argument(
        "--visits",
        default=None,
        help=(
            "validated visits parquet to build the hourly series from; defaults to "
            "the run folder's own, else the configured processed directory"
        ),
    )
    parser.add_argument(
        "--run-name",
        default="forecast_study",
        help="write under runs/<name>/artifacts (keeps the baseline artifacts/ clean)",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="write here instead of runs/<run-name>/artifacts",
    )
    parser.add_argument(
        "--origins",
        type=int,
        default=MIN_ORIGINS,
        help=f"weekly origins to score (minimum {MIN_ORIGINS})",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    config = load_config(args.config)
    runs_dir = Path(config["uploads"]["runs_dir"])

    if args.artifacts_dir:
        artifacts = resolve_path(args.artifacts_dir)
    else:
        artifacts = resolve_path(runs_dir / args.run_name / "artifacts")

    if args.visits:
        visits_path = resolve_path(args.visits)
    else:
        candidate = resolve_path(runs_dir / args.run_name / "data" / "processed" / "visits_validated.parquet")
        visits_path = (
            candidate
            if candidate.is_file()
            else resolve_path(Path(config["paths"]["processed_dir"]) / "visits_validated.parquet")
        )

    if not visits_path.is_file():
        logger.error(
            "No validated visits at %s. Run the pipeline first, for example:\n"
            "    python -m src.pipeline run --data real --stages load,validate "
            "--run-name %s",
            visits_path,
            args.run_name,
        )
        return 1

    logger.info("Building the hourly series from %s", visits_path)
    series = build_hourly_series(read_visits(visits_path))
    logger.info(
        "Series spans %s to %s (%d hours)",
        series["timestamp"].min(),
        series["timestamp"].max(),
        len(series),
    )

    result = run_forecast_study(series, config, n_origins=args.origins)

    tables = artifacts / "tables"
    save_table(result.scores, tables / "forecast_study_scores.csv")
    save_table(result.blend_grid, tables / "forecast_study_blend_grid.csv")
    save_table(result.predictions, tables / "forecast_study_predictions.csv")
    if not result.coverage.empty:
        save_table(result.coverage, tables / "forecast_study_coverage.csv")
    save_json(result.summary, artifacts / "metrics" / "forecast_study.json")

    selection = result.summary["selection"]
    logger.info(
        "Study complete: %d origins, %d scored hours. Validation winner at peak: %r.",
        result.summary["n_origins"],
        result.summary["n_scored_hours"],
        selection["winner"],
    )
    still_losing = result.summary["variants_still_losing_to_seasonal_naive_at_peak"]
    if still_losing:
        logger.warning(
            "On the TEST block these variants still do not beat seasonal-naive at "
            "peak hours: %s",
            ", ".join(still_losing),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
