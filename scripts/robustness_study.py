"""Run the whole pipeline across generator seeds, and see which conclusions survive.

**This study uses the true generator**, `scripts/original_generator/generate_ed_data.py`
-- the one that reproduces `data/raw/ed_v2.csv` exactly. It is loaded with `importlib`
straight from that file, never copied or re-implemented, so the datasets here are the
same kind of thing the baseline is.

That matters because an earlier robustness study used a *wrapper* around
`src/data/synthetic.py` instead, which models total post-contact time where the true
generator models physician-busy time. Two of that study's findings are re-tested here
and are the questions this script exists to answer:

    (a) does the simulation-validation gate still fail on seed 7?
    (b) does the cheapest-plan cost still spread across roughly 43,120 - 55,440?

Usage::

    python scripts/robustness_study.py                      # seeds 1..10 + 42, resumable
    python scripts/robustness_study.py --seeds 3 4          # just these
    python scripts/robustness_study.py --force              # ignore completed seeds
    python scripts/robustness_study.py --aggregate-only     # rebuild the tables

Each seed owns `runs/robustness/seed_<S>/`: its CSV, its derived config, its artifacts
and its `summary.json`. Nothing is written to `data/raw/` and nothing touches the
baseline `artifacts/` tree.

**Seed 42 is the reference**: at `--days 365 --scale 9.7` it *is* `ed_v2.csv`, so its
row of the table must reproduce the project's headline numbers. That is a built-in
check on the study itself, and `--check-reference` asserts it.

All data here is SYNTHETIC.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.robustness_summary import summarise_run  # noqa: E402
from src.utils.io import load_config, resolve_path, save_table  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

#: The true generator. Loaded from this path, never imported as a package, so there is
#: exactly one copy of it in the repository and this script cannot drift from it.
GENERATOR_PATH = resolve_path("scripts/original_generator/generate_ed_data.py")

#: Where seed folders live. Deliberately under runs/, never data/raw/.
DEFAULT_ROOT = Path("runs/robustness")

#: Seeds to study. 42 is the reference: it reproduces data/raw/ed_v2.csv.
DEFAULT_SEEDS = tuple(range(1, 11))
REFERENCE_SEED = 42

#: Generator settings that reproduce ed_v2.csv.
DEFAULT_DAYS = 365
DEFAULT_SCALE = 9.7

#: Threads each worker subprocess may use.
#:
#: Pinned rather than left to LightGBM's default of "all cores", for two reasons: three
#: workers each grabbing every core thrash, and a result that depends on how many other
#: seeds happened to be running at the time is not reproducible. The reference-seed
#: check is what confirms this value does not move any headline number.
WORKER_THREADS = "2"

#: The headline numbers the reference seed must reproduce. Same figures as
#: `tests/test_headline_regression.py`, repeated here because this script is the other
#: place they are checked, and it checks them against a freshly generated dataset
#: rather than against the stored artifacts.
REFERENCE_EXPECTED: dict[str, tuple[float, int]] = {
    "wait_mae_model": (11.524, 3),
    "wait_mae_baseline": (34.233, 3),
    "conformal_rolling_overall": (0.8875, 4),
    "conformal_static_overall": (0.849, 4),
    "forecast_mae": (2.356, 3),
    "cheapest_feasible_cost": (49280.0, 2),
    "n_feasible_plans": (72.0, 0),
}


def load_generator():
    """Load the true generator module from its file, without copying it.

    Returns:
        The imported module, exposing `to_frame`.

    Raises:
        FileNotFoundError: If the generator is not where it is supposed to be.
    """
    if not GENERATOR_PATH.is_file():
        raise FileNotFoundError(
            f"The true generator is missing from {GENERATOR_PATH}. This study will not "
            "fall back to another generator: the point of it is that the datasets come "
            "from the same code that produced data/raw/ed_v2.csv."
        )
    spec = importlib.util.spec_from_file_location("ed_true_generator", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load a module spec from {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_dir(root: Path, seed: int, noise: bool = True) -> Path:
    """Folder a seed owns."""
    suffix = "" if noise else "_nonoise"
    return resolve_path(root / f"seed_{seed}{suffix}")


def is_complete(folder: Path) -> bool:
    """Whether this seed has already been studied, for resuming."""
    return (folder / "summary.json").is_file()


def generate_dataset(seed: int, days: int, scale: float, path: Path, noise: bool) -> int:
    """Write one seed's dataset with the true generator.

    Args:
        seed: Generator seed.
        days: Days of data.
        scale: Arrival-rate scale.
        path: CSV to write.
        noise: False applies `--no-noise` (no missing values, no swapped timestamps).

    Returns:
        Number of rows written.
    """
    generator = load_generator()
    frame = generator.to_frame(days, seed, scale, noise=noise)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logger.info(
        "seed %d%s: %d rows -> %s", seed, "" if noise else " (no noise)", len(frame), path
    )
    return len(frame)


def write_config(base_config_path: str, csv_path: Path, out_path: Path) -> Path:
    """Write the derived config the pipeline runs this seed under.

    A full snapshot rather than an overlay, so a seed's result stays reproducible even
    if `configs/default.yaml` changes later.
    """
    config = load_config(base_config_path)
    config.pop("_config_path", None)
    config["paths"] = dict(config["paths"])
    try:
        config["paths"]["raw_file"] = csv_path.relative_to(resolve_path(".")).as_posix()
    except ValueError:
        config["paths"]["raw_file"] = str(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("# Written by scripts/robustness_study.py. Dataset is SYNTHETIC.\n")
        yaml.safe_dump(config, handle, sort_keys=False)
    return out_path


def run_one_seed(
    seed: int,
    root: Path,
    days: int,
    scale: float,
    base_config: str,
    noise: bool = True,
) -> dict[str, Any]:
    """Generate one seed, run the full pipeline on it, and summarise what came out.

    A pipeline that exits non-zero is **not** an error here: the simulation-validation
    gate is designed to stop the run, and whether it does is one of the things being
    measured. Whatever artifacts exist are summarised either way.

    Args:
        seed: Generator seed.
        root: Study root, normally `runs/robustness`.
        days: Days of data to generate.
        scale: Arrival-rate scale.
        base_config: Config to derive from.
        noise: Whether the generator applies its realistic-dirt step.

    Returns:
        The seed's summary row, also written to `summary.json` in its folder.
    """
    folder = seed_dir(root, seed, noise)
    folder.mkdir(parents=True, exist_ok=True)
    label = f"seed_{seed}" + ("" if noise else "_nonoise")

    csv_path = folder / f"{label}.csv"
    n_rows = generate_dataset(seed, days, scale, csv_path, noise)
    config_path = write_config(base_config, csv_path, folder / "config.yaml")

    environment = dict(os.environ)
    environment.update(
        OMP_NUM_THREADS=WORKER_THREADS,
        OPENBLAS_NUM_THREADS=WORKER_THREADS,
        MKL_NUM_THREADS=WORKER_THREADS,
        NUMEXPR_NUM_THREADS=WORKER_THREADS,
    )

    started = time.perf_counter()
    completed = subprocess.run(
        [
            sys.executable, "-m", "src.pipeline", "run",
            "--config", str(config_path),
            "--data", "real",
            "--run-name", f"{root.name}/{label}",
        ],
        cwd=str(resolve_path(".")),
        env=environment,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started
    (folder / "pipeline_stdout.log").write_text(
        completed.stdout + "\n" + completed.stderr, encoding="utf-8"
    )

    if completed.returncode != 0:
        logger.warning(
            "%s: pipeline exited %d after %.0fs -- summarising the artifacts it did "
            "produce. This is expected when the simulation-validation gate refuses.",
            label, completed.returncode, elapsed,
        )

    row = summarise_run(folder, label=label)
    row.update(
        seed=seed,
        noise=noise,
        n_rows=n_rows,
        pipeline_exit_code=completed.returncode,
        pipeline_completed=completed.returncode == 0,
        wall_seconds=round(elapsed, 1),
        generator="scripts/original_generator/generate_ed_data.py",
        days=days,
        scale=scale,
    )
    (folder / "summary.json").write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
    logger.info(
        "%s done in %.0fs (exit %d): wait MAE %s, gate %s, plan %s",
        label, elapsed, completed.returncode,
        row.get("wait_mae_model"), row.get("sim_validation_passed"),
        row.get("cheapest_feasible_plan"),
    )
    return row


# ------------------------------------------------------------------ parallel driver


def run_seeds(
    seeds: list[int],
    root: Path,
    days: int,
    scale: float,
    base_config: str,
    max_parallel: int,
    force: bool,
    noise: bool = True,
) -> None:
    """Run each seed in its own subprocess, at most `max_parallel` at a time.

    Each worker is this same script invoked with `--worker-seed`, which keeps the
    parallelism to one mechanism: no shared state, no threads in this process, and a
    crashed worker takes nothing else down.

    Args:
        seeds: Seeds to run.
        root: Study root.
        days: Days per dataset.
        scale: Arrival-rate scale.
        base_config: Config to derive from.
        max_parallel: Workers allowed at once.
        force: Re-run seeds that already have a summary.
        noise: Whether the generator applies its dirt step.
    """
    pending = []
    for seed in seeds:
        folder = seed_dir(root, seed, noise)
        if is_complete(folder) and not force:
            logger.info("seed %d already complete; skipping (use --force to redo)", seed)
            continue
        pending.append(seed)

    if not pending:
        logger.info("Nothing to run: every requested seed is already complete")
        return

    logger.info(
        "Running %d seed(s) %s at up to %d in parallel, %s threads each",
        len(pending), pending, max_parallel, WORKER_THREADS,
    )

    running: list[tuple[int, subprocess.Popen]] = []
    queue = list(pending)
    failures: list[int] = []

    while queue or running:
        while queue and len(running) < max_parallel:
            seed = queue.pop(0)
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--worker-seed", str(seed),
                "--root", str(root),
                "--days", str(days),
                "--scale", str(scale),
                "--config", base_config,
            ]
            if not noise:
                command.append("--no-noise")
            logger.info("starting seed %d", seed)
            running.append((seed, subprocess.Popen(command, cwd=str(resolve_path(".")))))

        time.sleep(2.0)
        for seed, process in list(running):
            if process.poll() is None:
                continue
            running.remove((seed, process))
            if process.returncode != 0:
                # The worker itself failed, which is different from the pipeline
                # refusing: that case is handled inside the worker and exits 0.
                failures.append(seed)
                logger.error("worker for seed %d exited %d", seed, process.returncode)

    if failures:
        logger.error("worker(s) failed for seed(s) %s; their rows will be missing", failures)


# ------------------------------------------------------------------ aggregation


#: Metrics averaged across seeds. Yes/no columns are handled separately, because the
#: mean of a boolean is not a number anyone should read as one.
NUMERIC_METRICS = (
    "wait_mae_model", "wait_mae_baseline", "ablation_gap_minutes",
    "classifier_pr_auc", "classifier_recall",
    "conformal_static_overall", "conformal_rolling_overall",
    "conformal_static_esi1", "conformal_rolling_esi1",
    "conformal_static_esi2", "conformal_rolling_esi2",
    "conformal_static_esi3", "conformal_rolling_esi3",
    "conformal_static_esi4", "conformal_rolling_esi4",
    "conformal_static_esi5", "conformal_rolling_esi5",
    "forecast_mae", "forecast_mae_seasonal_naive",
    "forecast_mae_peak", "forecast_mae_peak_seasonal_naive",
    "sim_worst_relative_error", "cheapest_feasible_cost", "n_feasible_plans",
    "n_rows", "wall_seconds",
)


def collect(root: Path, noise: bool = True) -> pd.DataFrame:
    """Read every finished seed's summary into one table."""
    rows: list[dict[str, Any]] = []
    for path in sorted(resolve_path(root).glob("seed_*/summary.json")):
        if (not noise) != path.parent.name.endswith("_nonoise"):
            continue
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows).sort_values("seed", ignore_index=True)
    leading = [c for c in ("seed", "run", "n_rows", "pipeline_completed", "wall_seconds") if c in table]
    return table.loc[:, leading + [c for c in table.columns if c not in leading]]


def describe(table: pd.DataFrame, exclude_seeds: tuple[int, ...] = (REFERENCE_SEED,)) -> pd.DataFrame:
    """Mean, std, min and max for every numeric metric across the study seeds.

    The reference seed is excluded by default: it is `ed_v2.csv`, the dataset every
    headline number was tuned and reported on, so including it in a spread that is
    meant to say "what happens on data we have not seen" would be double-counting.

    Args:
        table: Output of `collect`.
        exclude_seeds: Seeds left out of the statistics.

    Returns:
        One row per metric.
    """
    subset = table[~table["seed"].isin(exclude_seeds)] if "seed" in table else table
    rows: list[dict[str, Any]] = []
    for metric in NUMERIC_METRICS:
        if metric not in subset.columns:
            continue
        values = pd.to_numeric(subset[metric], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "metric": metric,
                "n_seeds": int(len(values)),
                "mean": round(float(values.mean()), 4),
                "std": round(float(values.std(ddof=1)), 4) if len(values) > 1 else 0.0,
                "min": round(float(values.min()), 4),
                "max": round(float(values.max()), 4),
                "spread": round(float(values.max() - values.min()), 4),
            }
        )
    return pd.DataFrame(rows)


def answer_questions(table: pd.DataFrame) -> dict[str, Any]:
    """Answer the two questions this study exists to settle.

    Args:
        table: Output of `collect`.

    Returns:
        A dict with a verdict and the evidence for each question, written so the
        report can quote it without restating a conclusion the data may not support.
    """
    study = table[table["seed"] != REFERENCE_SEED] if "seed" in table else table
    answers: dict[str, Any] = {}

    # (a) does the seed-7 simulation-validation failure reproduce?
    gate = study[["seed", "sim_validation_passed", "sim_worst_relative_error", "sim_worst_metric"]]
    failed = gate[gate["sim_validation_passed"] == False]  # noqa: E712
    seed7 = gate[gate["seed"] == 7]
    seed7_failed = (
        bool((seed7["sim_validation_passed"] == False).any()) if not seed7.empty else None  # noqa: E712
    )
    answers["a_seed7_gate_failure"] = {
        "question": "Does the earlier seed-7 simulation-validation failure reproduce "
                    "with the true generator?",
        "seed_7_present": not seed7.empty,
        "seed_7_failed": seed7_failed,
        "seeds_that_failed_the_gate": [int(s) for s in failed["seed"].tolist()],
        "n_failed": int(len(failed)),
        "n_seeds": int(len(study)),
        "worst_relative_error_by_seed": {
            int(row.seed): row.sim_worst_relative_error for row in gate.itertuples()
        },
        "verdict": (
            "reproduces" if seed7_failed
            else "does NOT reproduce: the earlier finding came from the wrapper "
                 "generator and was an artefact of it, not a property of seed 7"
        ),
    }

    # (b) does the cheapest-plan cost spread reproduce?
    costs = pd.to_numeric(study["cheapest_feasible_cost"], errors="coerce").dropna()
    earlier_low, earlier_high = 43120.0, 55440.0

    # A seed whose simulation failed validation has the refusal string in the plan
    # column, not a roster. Counting that string as a distinct plan inflates the
    # count by one and reads as if the refusal were a staffing recommendation, so
    # the plan tally is taken over the seeds that actually produced a costed plan.
    costed_plans = study.loc[costs.index, "cheapest_feasible_plan"].dropna().to_numpy()
    if costs.empty:
        answers["b_plan_cost_spread"] = {
            "question": "Does the cheapest-plan cost spread 43,120 - 55,440 reproduce?",
            "verdict": "no feasible plan was costed on any study seed",
        }
    else:
        observed_low, observed_high = float(costs.min()), float(costs.max())
        overlaps = not (observed_high < earlier_low or observed_low > earlier_high)
        answers["b_plan_cost_spread"] = {
            "question": "Does the cheapest-plan cost spread 43,120 - 55,440 reproduce?",
            "earlier_range": [earlier_low, earlier_high],
            "observed_range": [observed_low, observed_high],
            "observed_spread": round(observed_high - observed_low, 2),
            "earlier_spread": round(earlier_high - earlier_low, 2),
            "n_seeds_with_a_plan": int(costed_plans.size),
            "n_distinct_plans": int(pd.unique(costed_plans).size),
            "plans": sorted(pd.unique(costed_plans).tolist()),
            "overlaps_earlier_range": overlaps,
            "verdict": (
                "reproduces: the cheapest plan is dataset-specific"
                if observed_high - observed_low > 0.25 * (earlier_high - earlier_low)
                else "does NOT reproduce: the spread is far narrower with the true "
                     "generator, so the earlier range was an artefact of the wrapper"
            ),
        }
    return answers


def check_reference(table: pd.DataFrame) -> dict[str, Any]:
    """Assert the reference seed reproduces the project's headline numbers.

    Seed 42 at 365 days and scale 9.7 *is* `data/raw/ed_v2.csv`, so this is a check on
    the study itself: if the reference row disagrees with the headline numbers, the
    study's plumbing is wrong and none of its other rows mean anything.

    Args:
        table: Output of `collect`.

    Returns:
        Per-metric comparison and an overall verdict.
    """
    reference = table[table["seed"] == REFERENCE_SEED] if "seed" in table else pd.DataFrame()
    if reference.empty:
        return {"ran": False, "reason": f"seed {REFERENCE_SEED} has not been run"}

    row = reference.iloc[0]
    checks = []
    for metric, (expected, places) in REFERENCE_EXPECTED.items():
        got = row.get(metric)
        matches = got is not None and round(float(got), places) == round(expected, places)
        checks.append(
            {"metric": metric, "expected": expected, "observed": got, "matches": bool(matches)}
        )
    passed = all(check["matches"] for check in checks)
    if passed:
        logger.info("Reference seed %d reproduces every headline number", REFERENCE_SEED)
    else:
        for check in checks:
            if not check["matches"]:
                logger.error(
                    "Reference seed %d: %s expected %s, got %s",
                    REFERENCE_SEED, check["metric"], check["expected"], check["observed"],
                )
    return {"ran": True, "passed": passed, "checks": checks}


# ------------------------------------------------------------------ the CLI


def write_outputs(root: Path, tables_dir: Path) -> dict[str, Any]:
    """Build every table and JSON the study produces.

    Args:
        root: Study root.
        tables_dir: Where the CSVs go.

    Returns:
        The study summary, also written as JSON.
    """
    table = collect(root, noise=True)
    if table.empty:
        logger.error("No finished seeds under %s", root)
        return {"ran": False, "reason": "no finished seeds"}

    save_table(table, tables_dir / "robustness_seeds.csv")
    stats = describe(table)
    save_table(stats, tables_dir / "robustness_seed_stats.csv")

    answers = answer_questions(table)
    reference = check_reference(table)

    no_noise = collect(root, noise=False)
    noise_comparison: list[dict[str, Any]] = []
    if not no_noise.empty:
        for _, clean in no_noise.iterrows():
            seed = clean["seed"]
            dirty_rows = table[table["seed"] == seed]
            if dirty_rows.empty:
                continue
            dirty = dirty_rows.iloc[0]
            for metric in NUMERIC_METRICS:
                if metric not in table.columns:
                    continue
                left, right = dirty.get(metric), clean.get(metric)
                if left is None or right is None or pd.isna(left) or pd.isna(right):
                    continue
                noise_comparison.append(
                    {
                        "seed": int(seed),
                        "metric": metric,
                        "with_noise": round(float(left), 4),
                        "no_noise": round(float(right), 4),
                        "difference": round(float(right) - float(left), 4),
                        "percent_change": (
                            round(100.0 * (float(right) - float(left)) / float(left), 2)
                            if float(left) else None
                        ),
                    }
                )
        if noise_comparison:
            save_table(pd.DataFrame(noise_comparison), tables_dir / "robustness_no_noise.csv")

    summary = {
        "ran": True,
        "generator": str(GENERATOR_PATH.relative_to(resolve_path("."))),
        "generator_note": (
            "The true generator, loaded by importlib from its own file. It reproduces "
            "data/raw/ed_v2.csv exactly (pandas DataFrame.equals)."
        ),
        "n_seeds": int(len(table[table["seed"] != REFERENCE_SEED])),
        "reference_seed": REFERENCE_SEED,
        "worker_threads": WORKER_THREADS,
        "reference_check": reference,
        "answers": answers,
        "per_seed": table.to_dict(orient="records"),
        "statistics": stats.to_dict(orient="records"),
        "no_noise_comparison": noise_comparison,
        "synthetic_data_note": (
            "Every dataset here is synthetic and comes from one generator. Agreement "
            "across seeds shows the pipeline is not over-fitted to one draw; it says "
            "nothing about any real emergency department."
        ),
    }
    path = resolve_path(tables_dir).parent / "metrics" / "robustness_study.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python scripts/robustness_study.py",
        description=(
            "Run the full pipeline across generator seeds using the TRUE generator, "
            "and report which conclusions survive. All data is synthetic."
        ),
    )
    parser.add_argument(
        "--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS),
        help="seeds to study (default 1..10)",
    )
    parser.add_argument(
        "--reference-seed", type=int, default=REFERENCE_SEED,
        help="seed that reproduces ed_v2.csv; run and checked unless --no-reference",
    )
    parser.add_argument("--no-reference", action="store_true", help="skip the reference seed")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="study root under runs/")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="re-run completed seeds")
    parser.add_argument(
        "--no-noise", action="store_true",
        help="generate without the realistic-dirt step (missing values, swapped times)",
    )
    parser.add_argument(
        "--no-noise-seed", type=int, default=REFERENCE_SEED,
        help="seed to additionally run with --no-noise for comparison; -1 to skip",
    )
    parser.add_argument("--aggregate-only", action="store_true", help="rebuild tables only")
    parser.add_argument(
        "--tables-dir", default="artifacts/tables",
        help="where the study's tables go",
    )
    parser.add_argument(
        "--worker-seed", type=int, default=None,
        help="internal: run exactly this one seed in this process and exit",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    root = Path(args.root)

    # Worker mode: one seed, in this process, then exit.
    if args.worker_seed is not None:
        run_one_seed(
            args.worker_seed, root, args.days, args.scale, args.config,
            noise=not args.no_noise,
        )
        return 0

    if not args.aggregate_only:
        seeds = list(args.seeds)
        if not args.no_reference and args.reference_seed not in seeds:
            seeds.append(args.reference_seed)
        run_seeds(
            seeds, root, args.days, args.scale, args.config,
            max_parallel=args.max_parallel, force=args.force, noise=True,
        )
        if args.no_noise_seed is not None and args.no_noise_seed >= 0:
            run_seeds(
                [args.no_noise_seed], root, args.days, args.scale, args.config,
                max_parallel=1, force=args.force, noise=False,
            )

    summary = write_outputs(root, Path(args.tables_dir))
    if not summary.get("ran"):
        return 1

    reference = summary["reference_check"]
    if reference.get("ran") and not reference.get("passed"):
        logger.error(
            "The reference seed does NOT reproduce the headline numbers. Every other "
            "row of this study is suspect until that is explained."
        )
        return 1

    for key, answer in summary["answers"].items():
        logger.info("%s -> %s", key, answer["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
