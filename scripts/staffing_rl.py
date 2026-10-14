"""Train and evaluate the experimental staffing agent.

    python scripts/staffing_rl.py --episodes 60 --eval-episodes 20

Outputs, all stamped EXPERIMENTAL:

    tables/staffing_rl_training.csv    reward per training episode
    tables/staffing_rl_episodes.csv    every policy on every evaluation episode
    tables/staffing_rl_comparison.csv  paired comparison against the fixed roster
    metrics/staffing_rl.json           all of the above

Training and evaluation use **disjoint random streams**, so the agent is scored on
worlds it did not train on. The circularity that remains is structural and is stated in
`src.simulation.staffing_rl`: the agent learns the simulator, and the simulator was
calibrated to a generator. A win is evidence about the method, not about any real
department.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.simulation.calibrate import calibrate_and_validate, require_valid_calibration  # noqa: E402
from src.simulation.staffing_rl import (  # noqa: E402
    EXPERIMENTAL_NOTICE,
    EnvConfig,
    FixedRosterPolicy,
    QLearningPolicy,
    RandomPolicy,
    ReactiveRulePolicy,
    StaffingEnv,
    compare_policies,
    evaluate_policy,
    train_q_learning,
)
from src.utils.io import (  # noqa: E402
    load_config,
    load_model,
    read_visits,
    resolve_path,
    save_model,
    save_table,
)
from src.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


def obtain_calibration(config: dict, models_dir: Path, visits_path: Path):
    """Load the saved calibration, or fit one, and insist it passed validation."""
    saved = resolve_path(models_dir / "sim_calibration.joblib")
    if saved.is_file():
        calibration = load_model(saved)
        logger.info("Loaded the saved calibration from %s", saved)
    else:
        calibration = calibrate_and_validate(read_visits(visits_path), config)
    require_valid_calibration(calibration)
    return calibration


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python scripts/staffing_rl.py",
        description="Experimental tabular Q-learning staffing agent.",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--episodes", type=int, default=60, help="training episodes")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--hours", type=int, default=168, help="hours per episode")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    logger.warning("%s", EXPERIMENTAL_NOTICE)

    config = load_config(args.config)
    artifacts = resolve_path(args.artifacts_dir)
    calibration = obtain_calibration(
        config,
        Path(config["paths"]["models_dir"]),
        resolve_path(Path(config["paths"]["processed_dir"]) / "visits_validated.parquet"),
    )

    env_config = EnvConfig(hours=args.hours)
    env = StaffingEnv(config, calibration, env_config=env_config)
    logger.info(
        "Environment: %d states x %d actions, %d-hour episodes",
        env.n_states, env.n_actions, env_config.hours,
    )

    agent, training_log = train_q_learning(
        env, episodes=args.episodes, seed=int(config["seed"]), seed_prefix="rl::train"
    )

    policies = [
        FixedRosterPolicy(calibration.plan.capacity_by_hour(), env_config),
        ReactiveRulePolicy(),
        RandomPolicy(seed=int(config["seed"])),
        agent,
    ]
    # Evaluation episodes use a different prefix from training, so these are worlds
    # the agent has not played.
    seeds = list(range(args.eval_episodes))
    results = pd.concat(
        [evaluate_policy(env, policy, seeds, seed_prefix="rl::eval") for policy in policies],
        ignore_index=True,
    )
    comparison = compare_policies(results)

    tables = artifacts / "tables"
    save_table(training_log, tables / "staffing_rl_training.csv")
    save_table(results, tables / "staffing_rl_episodes.csv")
    save_table(comparison, tables / "staffing_rl_comparison.csv")
    save_model(agent, Path(config["paths"]["models_dir"]) / "staffing_rl_policy.joblib")

    learned = comparison[comparison["policy"] == "q_learning"]
    verdict = "the learned policy was not evaluated"
    if not learned.empty:
        row = learned.iloc[0]
        if bool(row.get("beats_baseline")) and bool(row.get("significant")):
            verdict = (
                f"The learned policy beats the fixed roster by "
                f"{abs(row['cost_difference']):.0f} per episode "
                f"(95% CI {row['ci_low']:.0f} to {row['ci_high']:.0f}), which excludes "
                "zero. It learned the simulator; that is all this shows."
            )
        elif bool(row.get("beats_baseline")):
            verdict = (
                f"The learned policy is ahead of the fixed roster by "
                f"{abs(row['cost_difference']):.0f} per episode, but the 95% interval "
                f"({row['ci_low']:.0f} to {row['ci_high']:.0f}) includes zero, so the "
                "difference is not established."
            )
        else:
            verdict = (
                f"The learned policy does NOT beat the fixed roster: it costs "
                f"{row['cost_difference']:.0f} more per episode "
                f"(95% CI {row['ci_low']:.0f} to {row['ci_high']:.0f}). Reported as it "
                "stands; it has not been tuned until it won."
            )
    logger.info("%s", verdict)

    payload = {
        "ran": True,
        "experimental": True,
        "notice": EXPERIMENTAL_NOTICE,
        "verdict": verdict,
        "n_states": env.n_states,
        "n_actions": env.n_actions,
        "training_episodes": args.episodes,
        "evaluation_episodes": args.eval_episodes,
        "hours_per_episode": env_config.hours,
        "train_seed_prefix": "rl::train",
        "eval_seed_prefix": "rl::eval",
        "policies": [
            {"name": policy.name, "description": policy.description} for policy in policies
        ],
        "comparison": comparison.to_dict(orient="records"),
    }
    path = artifacts / "metrics" / "staffing_rl.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
