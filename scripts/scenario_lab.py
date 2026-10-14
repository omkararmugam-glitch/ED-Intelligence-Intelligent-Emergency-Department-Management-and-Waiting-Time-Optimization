"""Run the uncalibrated resource what-ifs and write their tables.

    python scripts/scenario_lab.py --artifacts-dir artifacts

Outputs, all three stamped with `UNCALIBRATED_NOTICE`:

    tables/scenario_lab_replications.csv   every replication of every scenario
    tables/scenario_lab_comparison.csv     paired effect per KPI, with intervals
    tables/scenario_lab_sensitivity.csv    which resource moves ESI 3-5 p90 most
    metrics/scenario_lab.json              all of the above, machine-readable

**None of this is calibrated.** The dataset has no bed, nurse or imaging column, so
these parameters are assumed and the validation gate cannot test them. Read the
ranking, not the magnitudes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.calibrate import calibrate_and_validate, require_valid_calibration  # noqa: E402
from src.simulation.des import UNCALIBRATED_NOTICE  # noqa: E402
from src.simulation.scenario_lab import (  # noqa: E402
    DEFAULT_REPLICATIONS,
    build_scenarios,
    probe_bed_demand,
    run_scenario_lab,
    sensitivity_ranking,
)
from src.utils.io import (  # noqa: E402
    load_config,
    load_model,
    read_visits,
    resolve_path,
    save_table,
)
from src.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


def obtain_calibration(config: dict, models_dir: Path, visits_path: Path):
    """Load the saved calibration, or fit one if it has not been saved.

    The what-ifs are perturbations *of the calibrated department*, so they are only
    meaningful if the department they start from passed validation. That is checked
    here rather than assumed.
    """
    saved = resolve_path(models_dir / "sim_calibration.joblib")
    if saved.is_file():
        calibration = load_model(saved)
        logger.info("Loaded the saved calibration from %s", saved)
    else:
        logger.info("No saved calibration; fitting one from %s", visits_path)
        calibration = calibrate_and_validate(read_visits(visits_path), config)

    require_valid_calibration(calibration)
    return calibration


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python scripts/scenario_lab.py",
        description=(
            "Uncalibrated what-ifs on beds, nurses, imaging and boarding. "
            "Scenario analysis only -- none of it is validated against the data."
        ),
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--artifacts-dir", default="artifacts", help="where the tables are written"
    )
    parser.add_argument(
        "--replications", type=int, default=DEFAULT_REPLICATIONS,
        help=f"replications per scenario (minimum {DEFAULT_REPLICATIONS})",
    )
    parser.add_argument("--horizon-hours", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    if args.replications < DEFAULT_REPLICATIONS:
        logger.error(
            "%d replications is below the minimum of %d; the paired intervals would "
            "be too wide to rank anything.",
            args.replications, DEFAULT_REPLICATIONS,
        )
        return 1

    config = load_config(args.config)
    artifacts = resolve_path(args.artifacts_dir)
    calibration = obtain_calibration(
        config,
        Path(config["paths"]["models_dir"]),
        resolve_path(Path(config["paths"]["processed_dir"]) / "visits_validated.parquet"),
    )

    logger.warning("%s", UNCALIBRATED_NOTICE)

    horizon = args.horizon_hours or int(config["simulation"]["horizon_hours"])
    demand = probe_bed_demand(config, calibration, horizon_hours=horizon)
    scenarios = build_scenarios(demand)
    replications, comparison = run_scenario_lab(
        config, calibration, scenarios=scenarios,
        n_replications=args.replications, horizon_hours=horizon,
    )
    ranking = sensitivity_ranking(comparison)

    tables = artifacts / "tables"
    save_table(replications, tables / "scenario_lab_replications.csv")
    save_table(comparison, tables / "scenario_lab_comparison.csv")
    if not ranking.empty:
        save_table(ranking, tables / "scenario_lab_sensitivity.csv")

    summary = {
        "ran": True,
        "uncalibrated": True,
        "notice": UNCALIBRATED_NOTICE,
        "n_replications": args.replications,
        "horizon_hours": horizon,
        "bed_demand_probe": demand,
        "bed_count_note": (
            "Bed counts are quantiles of the calibrated model's own concurrent "
            "occupancy, not a measurement: the data has no bed column."
        ),
        "scenarios": [
            {
                "name": scenario.name,
                "description": scenario.description,
                "extensions": scenario.extensions.describe(),
                "boarding_scale": scenario.boarding_scale,
            }
            for scenario in scenarios
        ],
        "comparison": comparison.to_dict(orient="records"),
        "sensitivity_ranking": ranking.to_dict(orient="records") if not ranking.empty else [],
    }
    path = artifacts / "metrics" / "scenario_lab.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
