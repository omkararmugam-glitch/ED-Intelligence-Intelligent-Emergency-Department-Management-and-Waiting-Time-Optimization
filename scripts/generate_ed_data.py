"""Command-line front end for the project's synthetic ED visit generator.

**This script contains no generation logic of its own.** Every visit it writes comes
from :func:`src.data.synthetic.generate_synthetic_visits`, the generator that produced
`data/raw/ed_v2.csv`. All this file does is:

1. load a base config (``configs/default.yaml`` by default),
2. apply the overrides asked for on the command line,
3. call the generator,
4. write the visits as CSV and write a *derived config* that points
   ``paths.raw_file`` at that CSV,

so the pipeline can then be run on the new dataset with::

    python -m src.pipeline run --config <derived.yaml> --data real --run-name <name>

Why a derived config rather than new flags on the pipeline: the baseline run reads
`data/raw/ed_v2.csv` through ``--data real``, and a robustness comparison is only
apples-to-apples if every generated dataset travels the same code path. Writing a
config also leaves an auditable record of exactly which parameters produced a file.

Profiles
--------
``--profile v2`` (the default) reproduces the *generator settings* behind
`data/raw/ed_v2.csv`: the department's published roster as the generator's doctor
capacity, 57,500 visits, starting 2024-01-01. It is not byte-identical to that file --
see "Known differences from ed_v2.csv" below -- but it is the same generator run on the
same department.

``--profile v3`` is the longer-wait dataset: tighter doctor staffing and/or higher load
with longer patience, so median wait of seen patients lands nearer published national
figures than ed_v2's 11 minutes. v3 changes nothing unless it is asked for by name.

Known differences from ed_v2.csv
--------------------------------
`ed_v2.csv` was written by a script that is no longer in the repository, and it carries
two cosmetic artefacts this wrapper deliberately does **not** reproduce, because
manufacturing them would be inventing data rather than generating it:

* ``sex`` is coded ``F``/``M`` in ed_v2.csv; the generator emits ``female``/``male``.
  ``configs/column_map.yaml`` maps both, so either loads identically.
* ~0.36% of ed_v2.csv rows have a missing ``triage_level``, which the validator counts
  and rejects. Generated files are clean, so their validation drop counts are ~0.

All data produced here is SYNTHETIC. It is not real patient data and carries no
information about any real person or hospital.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Allow `python scripts/generate_ed_data.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.synthetic import generate_synthetic_visits  # noqa: E402
from src.utils.io import load_config, resolve_path  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

#: Visit count of `data/raw/ed_v2.csv`, the baseline dataset.
V2_N_PATIENTS = 57_500

#: Columns written, in this order: the schema's input columns and nothing else.
#: `wait_minutes` and `los_minutes` stay derived, so a generated file cannot leak them.
OUTPUT_COLUMNS = (
    "patient_id",
    "arrival_time",
    "triage_level",
    "seen_time",
    "departure_time",
    "disposition",
    "age",
    "sex",
    "arrival_mode",
    "chief_complaint",
)


def _describe_roster(shifts: dict[str, dict[str, Any]]) -> str:
    """Compact ``name=doctors`` summary of a roster block."""
    parts = [f"{name}={spec['doctors']}" for name, spec in sorted(shifts.items())]
    return ", ".join(parts)


def apply_v2_profile(config: dict[str, Any]) -> list[str]:
    """Keep the generator exactly as `configs/default.yaml` configures it.

    The only change is the visit count, set to ed_v2.csv's 57,500 so a generated
    dataset is the same size as the baseline one.

    **Why nothing else is touched.** `configs/default.yaml` carries two different
    doctor rosters -- ``synthetic.shifts`` (4/8/5) and ``staffing_roster.shifts``
    (2/3/3/2) -- and they are not meant to agree. ``synthetic.service_time`` is *total
    post-contact time*, not doctor occupancy; the simulation reconciles the two by
    fitting ``clinician_time_fraction`` (0.207 on the baseline run), which is why a
    2/3/3/2 roster can reproduce a department generated with 4/8/5. Forcing the
    generator onto the published roster instead produces a department in permanent
    collapse: measured at seed 42, median wait 192 min and 50.6% abandonment, against
    ed_v2's 11 min and 3.6%.

    Measured agreement of this profile with `data/raw/ed_v2.csv` at seed 42: median
    wait 16.5 min vs 11.0, mean 40.7 vs 34.9, abandonment 3.84% vs 3.58%, ESI 1-2
    abandonment 0.000 in both, ESI 5 abandonment 8.0% vs 9.0%. Close, not identical --
    ed_v2's own generation parameters are not recorded anywhere in the repository, so
    they cannot be recovered exactly. Its median post-contact time is 99 min against
    this profile's 53.6, so v2's boarding or service settings differed.

    Args:
        config: Config dict, mutated in place.

    Returns:
        Human-readable notes describing what was changed.
    """
    config["synthetic"]["n_patients"] = V2_N_PATIENTS
    return [
        f"synthetic.n_patients = {V2_N_PATIENTS} (ed_v2.csv's visit count); "
        "every other synthetic parameter left at the config default"
    ]


def apply_v3_profile(
    config: dict[str, Any],
    doctor_delta: int,
    load_multiplier: float,
    patience_scale: float,
    boarding_scale: float = 1.0,
) -> list[str]:
    """Lengthen waits towards published national figures, keeping abandonment sane.

    Three levers, all of them existing config keys:

    * ``doctor_delta`` removes doctors from every rostered shift (floored at 1) -- the
      "tighter doctor staffing" lever;
    * ``load_multiplier`` scales ``synthetic.base_rate_per_hour`` -- the "higher load"
      lever;
    * ``patience_scale`` stretches every ESI level's patience median, so a longer wait
      does not simply convert into more people leaving without being seen;
    * ``boarding_scale`` stretches the post-treatment exit-block delay. Boarding
      occupies a bed, not a clinician -- the generator releases the doctor after
      treatment and adds boarding to the departure time only -- so this lengthens
      post-contact time and length of stay **without** touching anyone's wait. That
      independence is what makes it usable as a separate calibration lever.

    The roster the *simulation* is told about moves in step, because the roster is an
    input describing the department, not a parameter to be fitted.

    Args:
        config: Config dict, mutated in place.
        doctor_delta: Doctors to remove from each shift.
        load_multiplier: Multiplier on the arrival rate.
        patience_scale: Multiplier on every patience median.
        boarding_scale: Multiplier on every boarding median.

    Returns:
        Human-readable notes describing what was changed.
    """
    notes = apply_v2_profile(config)
    synthetic = config["synthetic"]

    if doctor_delta:
        for shifts in (synthetic["shifts"], config["staffing_roster"]["shifts"]):
            for spec in shifts.values():
                spec["doctors"] = max(1, int(spec["doctors"]) - int(doctor_delta))
        notes.append(
            f"doctors reduced by {doctor_delta} on every shift (floor 1) -> "
            f"{_describe_roster(synthetic['shifts'])}"
        )

    if load_multiplier != 1.0:
        synthetic["base_rate_per_hour"] = float(synthetic["base_rate_per_hour"]) * float(
            load_multiplier
        )
        notes.append(
            f"synthetic.base_rate_per_hour x{load_multiplier:g} -> "
            f"{synthetic['base_rate_per_hour']:.4f}"
        )

    if patience_scale != 1.0:
        medians = synthetic["lwbs"]["patience_median_minutes"]
        synthetic["lwbs"]["patience_median_minutes"] = {
            level: float(value) * float(patience_scale) for level, value in medians.items()
        }
        notes.append(f"synthetic.lwbs.patience_median_minutes x{patience_scale:g}")

    if boarding_scale != 1.0:
        boarding = synthetic["boarding_minutes"]
        for spec in boarding.values():
            spec["median"] = float(spec["median"]) * float(boarding_scale)
        notes.append(
            f"synthetic.boarding_minutes medians x{boarding_scale:g} -> "
            + ", ".join(f"{k}={v['median']:.1f}" for k, v in sorted(boarding.items()))
        )

    return notes



#: Calibration targets for the v3 profile, with the reason each one exists.
#:
#: The point of v3 is that ed_v2's median wait of 11 minutes is well below published
#: national figures for door-to-doctor time, which cluster nearer 25-30 minutes. The
#: other five targets are guardrails: a dataset is not more realistic if it reaches a
#: realistic median wait by making a tenth of the department walk out, or by letting
#: ESI 1 patients queue.
#:
#: These are ILLUSTRATIVE calibration goals for a SYNTHETIC dataset, not clinical
#: standards and not a claim about any real department. Each entry is
#: (name, low, high, unit, why).
V3_TARGETS: tuple[tuple[str, float, float, str, str], ...] = (
    (
        "median_wait_minutes_seen", 25.0, 30.0, "minutes",
        "door-to-doctor for patients who were seen; ed_v2 sits at 11, published "
        "national figures are nearer 25-30",
    ),
    (
        "lwbs_rate_overall", 0.02, 0.04, "rate",
        "longer waits must not be bought with a walkout rate no department would "
        "accept",
    ),
    (
        "lwbs_rate_esi5", 0.0, 0.08, "rate",
        "the least urgent patients abandon most; this caps how far that may go",
    ),
    (
        "lwbs_rate_esi1_esi2", 0.0, 0.0, "rate",
        "a high-acuity patient leaving without being seen is a never event, so the "
        "generator must produce none at all",
    ),
    (
        "median_wait_minutes_esi1", 0.0, 12.0, "minutes",
        "ESI 1 is a resuscitation case; the priority queue has to keep their wait "
        "short even under the heavier load",
    ),
    (
        "median_post_contact_minutes", 80.0, 100.0, "minutes",
        "treatment plus boarding, targeted at about 90 minutes; ed_v2 sits at 99",
    ),
)


def measure_calibration(visits: pd.DataFrame) -> dict[str, float]:
    """Measure every quantity `V3_TARGETS` names, from generated visits.

    Args:
        visits: The generator's output.

    Returns:
        Mapping of target name to the achieved value.
    """
    frame = visits.copy()
    for column in ("arrival_time", "seen_time", "departure_time"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")

    wait = (frame["seen_time"] - frame["arrival_time"]).dt.total_seconds() / 60.0
    post_contact = (frame["departure_time"] - frame["seen_time"]).dt.total_seconds() / 60.0
    lwbs = frame["disposition"] == "lwbs"
    level = pd.to_numeric(frame["triage_level"], errors="coerce")

    high_acuity = level.isin([1, 2])
    return {
        "median_wait_minutes_seen": float(wait.median()),
        "lwbs_rate_overall": float(lwbs.mean()),
        "lwbs_rate_esi5": float(lwbs[level == 5].mean()),
        "lwbs_rate_esi1_esi2": float(lwbs[high_acuity].mean()),
        "median_wait_minutes_esi1": float(wait[level == 1].median()),
        "median_post_contact_minutes": float(post_contact.median()),
    }


def calibration_report(visits: pd.DataFrame) -> pd.DataFrame:
    """Score the generated dataset against every v3 target.

    Args:
        visits: The generator's output.

    Returns:
        One row per target: the band asked for, the value achieved, and PASS or MISS.
        A missed target is reported as missed; it is never quietly widened.
    """
    achieved = measure_calibration(visits)
    rows = []
    for name, low, high, unit, why in V3_TARGETS:
        value = achieved[name]
        passed = bool(low <= value <= high) if pd.notna(value) else False
        rows.append(
            {
                "target": name,
                "unit": unit,
                "target_low": low,
                "target_high": high,
                "target_description": (
                    f"{low:g}" if low == high else f"{low:g} to {high:g}"
                ),
                "achieved": round(value, 4) if pd.notna(value) else None,
                "status": "PASS" if passed else "MISS",
                "why_this_target": why,
            }
        )
    table = pd.DataFrame(rows)

    missed = table[table["status"] == "MISS"]
    if missed.empty:
        logger.info("Calibration: all %d target(s) met", len(table))
    else:
        for row in missed.itertuples():
            logger.warning(
                "Calibration MISS: %s achieved %s, target %s %s",
                row.target, row.achieved, row.target_description, row.unit,
            )
    return table

def build_config(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    """Load the base config and apply every requested override.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Tuple of (config dict, notes describing the overrides applied).

    Raises:
        ValueError: If a ``--doctors`` assignment is malformed or names no such shift.
    """
    config = copy.deepcopy(load_config(args.config))
    notes: list[str] = []

    if args.profile == "v2":
        notes.extend(apply_v2_profile(config))
    else:
        notes.extend(
            apply_v3_profile(
                config,
                doctor_delta=args.doctor_delta,
                load_multiplier=args.load_multiplier,
                patience_scale=args.patience_scale,
                boarding_scale=args.boarding_scale,
            )
        )

    if args.seed is not None:
        config["seed"] = int(args.seed)
        notes.append(f"seed = {args.seed}")

    for assignment in args.doctors or []:
        name, separator, value = assignment.partition("=")
        name, value = name.strip(), value.strip()
        if not separator or not value.isdigit():
            raise ValueError(f"--doctors expects SHIFT=COUNT, got {assignment!r}")
        # The generator's shifts (night/day/eve) and the published roster
        # (night_early/day/evening/night_late) are different blocks with different
        # names, so a shift name is expected to appear in one of them, not both.
        blocks = {
            "synthetic.shifts": config["synthetic"]["shifts"],
            "staffing_roster.shifts": config["staffing_roster"]["shifts"],
        }
        matched = [label for label, shifts in blocks.items() if name in shifts]
        if not matched:
            known = sorted({key for shifts in blocks.values() for key in shifts})
            raise ValueError(
                f"--doctors names shift {name!r}, which is in neither roster "
                f"(known shifts: {known})"
            )
        for label in matched:
            blocks[label][name]["doctors"] = int(value)
        notes.append(f"{' and '.join(matched)}: {name} set to {value} doctor(s)")

    if args.n_patients is not None:
        config["synthetic"]["n_patients"] = int(args.n_patients)
        notes.append(f"synthetic.n_patients = {args.n_patients}")

    if args.scale != 1.0:
        scaled = int(round(int(config["synthetic"]["n_patients"]) * float(args.scale)))
        config["synthetic"]["n_patients"] = scaled
        notes.append(f"synthetic.n_patients scaled x{args.scale:g} -> {scaled}")

    if args.n_days is not None:
        notes.append(f"generated as {args.n_days} whole days (visit target ignored)")

    return config, notes


def derived_config(
    config: dict[str, Any], csv_path: Path, notes: list[str]
) -> dict[str, Any]:
    """Build the config the pipeline should use to read the generated CSV.

    The returned config is the generation config with ``paths.raw_file`` repointed, so
    a pipeline run on it uses exactly the parameters the data was generated under --
    which matters for the simulation, whose roster must match the generator's.

    Args:
        config: The config the data was generated with.
        csv_path: Where the CSV was written.
        notes: Override notes, recorded in the file as a comment block.

    Returns:
        A YAML-serialisable config dict.
    """
    out = copy.deepcopy(config)
    out.pop("_config_path", None)
    out["paths"] = dict(out["paths"])
    try:
        out["paths"]["raw_file"] = csv_path.relative_to(resolve_path(".")).as_posix()
    except ValueError:
        out["paths"]["raw_file"] = str(csv_path)
    out["_generated_by"] = {
        "script": "scripts/generate_ed_data.py",
        "overrides": list(notes),
        "data_is_synthetic": True,
    }
    return out


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python scripts/generate_ed_data.py",
        description=(
            "Generate a synthetic ED visits CSV with the project's existing generator, "
            "plus a derived config the pipeline can run on. All output is synthetic."
        ),
    )
    parser.add_argument("--config", default="configs/default.yaml", help="base config")
    parser.add_argument(
        "--profile",
        choices=("v2", "v3"),
        default="v2",
        help=(
            "v2 (default): the generator settings behind data/raw/ed_v2.csv. "
            "v3: tighter staffing and longer patience, for longer waits."
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="override the config seed")
    parser.add_argument("--n-patients", type=int, default=None, help="target visit count")
    parser.add_argument(
        "--n-days",
        type=int,
        default=None,
        help="generate exactly this many whole days instead of a visit target",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="multiply the visit target (e.g. 0.85 or 1.15); ignored with --n-days",
    )
    parser.add_argument(
        "--doctors",
        action="append",
        metavar="SHIFT=N",
        help="set a rostered shift's doctor count exactly; repeatable",
    )
    parser.add_argument(
        "--doctor-delta",
        type=int,
        default=1,
        help="v3 only: doctors to remove from every shift (default 1)",
    )
    parser.add_argument(
        "--load-multiplier",
        type=float,
        default=1.0,
        help="v3 only: multiplier on synthetic.base_rate_per_hour",
    )
    parser.add_argument(
        "--patience-scale",
        type=float,
        default=1.0,
        help="v3 only: multiplier on every ESI level's patience median",
    )
    parser.add_argument(
        "--boarding-scale",
        type=float,
        default=1.0,
        help=(
            "v3 only: multiplier on the post-treatment boarding medians. Lengthens "
            "post-contact time and length of stay without changing any wait."
        ),
    )
    parser.add_argument("--out", required=True, help="CSV path to write")
    parser.add_argument(
        "--config-out",
        default=None,
        help="where to write the derived config (default: alongside --out, as .yaml)",
    )
    parser.add_argument(
        "--calibration-report",
        default=None,
        help=(
            "write a target/achieved/status CSV for the v3 calibration targets "
            "(see V3_TARGETS). Useful for any profile, not only v3."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    config, notes = build_config(args)

    csv_path = resolve_path(args.out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    visits = generate_synthetic_visits(config, n_days=args.n_days)
    missing = [column for column in OUTPUT_COLUMNS if column not in visits.columns]
    if missing:
        raise RuntimeError(
            f"The generator did not return the expected input columns: {missing}. "
            "This wrapper writes the schema's input columns and nothing else."
        )
    visits[list(OUTPUT_COLUMNS)].to_csv(
        csv_path, index=False, date_format="%Y-%m-%d %H:%M:%S"
    )
    logger.info("Wrote %d synthetic visits to %s", len(visits), csv_path)

    if args.calibration_report:
        report = calibration_report(visits)
        report_path = resolve_path(args.calibration_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(report_path, index=False)
        logger.info(
            "Wrote %s (%d target(s), %d met)",
            report_path,
            len(report),
            int((report["status"] == "PASS").sum()),
        )

    config_path = (
        resolve_path(args.config_out) if args.config_out else csv_path.with_suffix(".yaml")
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "# Derived config written by scripts/generate_ed_data.py.\n"
            "# The dataset it points at is SYNTHETIC. Run the pipeline with:\n"
            "#   python -m src.pipeline run --config "
            f"{config_path.as_posix()} --data real --run-name <name>\n"
        )
        for note in notes:
            handle.write(f"# override: {note}\n")
        yaml.safe_dump(derived_config(config, csv_path, notes), handle, sort_keys=False)
    logger.info("Wrote derived config to %s", config_path)

    for note in notes:
        logger.info("override applied: %s", note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
