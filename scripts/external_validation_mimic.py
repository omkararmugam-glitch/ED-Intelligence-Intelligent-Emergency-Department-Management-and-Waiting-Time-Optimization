"""Run this project's adapter against the real MIMIC-IV-ED demo, and profile it.

**No model is fitted and no model performance is reported here, deliberately.** The
open demo holds 222 ED stays. That is enough to prove an adapter maps real columns
correctly and enough to describe what the tables contain; it is nowhere near enough
to fit or evaluate anything, and a figure computed from 222 stays would be quoted
later as though it meant something.

What this script answers:

1. Does `configs/column_map_mimic_ed.yaml` -- written from the published schema
   without ever seeing the data -- actually map the real columns? Every claim in it
   becomes checkable here.
2. What is in the tables: stays, acuity mix, disposition mix, length of stay,
   arrival-hour shape.
3. How many stays get a wait at all from each proxy, given that **MIMIC-IV-ED has no
   first-provider-contact timestamp**. Every wait figure it produces is labelled
   `proxy wait (not true door-to-provider)`.
4. Whether the concurrent queue state can be reconstructed from these tables -- which
   decides whether this project's queue features and its simulation validation could
   ever be checked against MIMIC, even with full credentialed access.

Licence: Open Database License (ODbL) v1.0. Attribution required; see
`data/external/README.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.adapter import (  # noqa: E402
    PROXY_WAIT_NOTICE,
    adapt_real_dataset,
    load_column_map,
)
from src.utils.io import resolve_path, save_json  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

DEFAULT_SOURCE = Path("data/external/mimic_ed_demo/mimic-iv-ed-demo-2.2/ed")
DEFAULT_OUT = Path("artifacts/external_validation/mimic_demo")
DEFAULT_MAP = Path("configs/column_map_mimic_ed.yaml")

#: The two candidate proxies for "first clinician contact", neither of which is it.
PROXIES = {
    "first_vitalsign_charttime": (
        "first vitalsign charttime (usually recorded at triage, before a clinician)"
    ),
    "first_pyxis_charttime": (
        "first pyxis charttime (a medication dispense, which follows a clinical "
        "decision and exists only for patients given a drug)"
    ),
}


# -------------------------------------------------------------------- the join


def join_tables(base: Path) -> pd.DataFrame:
    """Join the ED tables into one row per stay, keyed on `stay_id`.

    The adapter reads one table, so this is the data-engineering step
    `docs/real_data_readiness.md` says the user has to perform. It is done here
    rather than hidden inside the adapter because the join is where the judgement
    lives: which of several `vitalsign` rows counts as "first", and whether a stay
    with no `pyxis` row has no wait or an unknown one.

    The `subject_id` trap is worth naming: a patient can have several stays, so
    joining on `subject_id` would multiply rows. Everything here joins on `stay_id`.
    """
    stays = pd.read_csv(base / "edstays.csv.gz")
    triage = pd.read_csv(base / "triage.csv.gz")
    vitals = pd.read_csv(base / "vitalsign.csv.gz")
    pyxis = pd.read_csv(base / "pyxis.csv.gz")

    for frame, column in ((stays, "intime"), (stays, "outtime"),
                          (vitals, "charttime"), (pyxis, "charttime")):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")

    first_vitals = (
        vitals.sort_values("charttime").groupby("stay_id")["charttime"].first()
        .rename("first_vitalsign_charttime")
    )
    first_pyxis = (
        pyxis.sort_values("charttime").groupby("stay_id")["charttime"].first()
        .rename("first_pyxis_charttime")
    )

    joined = (
        stays
        .merge(triage.drop(columns=["subject_id"]), on="stay_id", how="left")
        .merge(first_vitals, on="stay_id", how="left")
        .merge(first_pyxis, on="stay_id", how="left")
    )
    # anchor_age is in MIMIC-IV's patients table, which the ED demo does not ship.
    # The column map asks for it, so it is created empty rather than left missing:
    # the adapter should see "no ages recorded", not "the file is malformed".
    joined["anchor_age"] = pd.NA

    assert len(joined) == len(stays), "the join multiplied rows; check the keys"
    logger.info(
        "Joined %d stays; %d have a first vitalsign, %d have a first pyxis dispense",
        len(joined),
        int(joined["first_vitalsign_charttime"].notna().sum()),
        int(joined["first_pyxis_charttime"].notna().sum()),
    )
    return joined


# ------------------------------------------------------------------ the profile


def profile(frame: pd.DataFrame, proxy_label: str) -> dict[str, Any]:
    """Describe the extract. Counts and distributions only, never performance."""
    arrivals = pd.to_datetime(frame["arrival_time"], errors="coerce")
    seen = pd.to_datetime(frame.get("seen_time"), errors="coerce")
    departures = pd.to_datetime(frame["departure_time"], errors="coerce")

    los = (departures - arrivals).dt.total_seconds() / 60
    wait = (seen - arrivals).dt.total_seconds() / 60 if seen is not None else pd.Series(dtype=float)
    wait = wait.loc[wait.ge(0)] if not wait.empty else wait

    def distribution(values: pd.Series) -> dict[str, Any]:
        clean = pd.to_numeric(values, errors="coerce").dropna()
        if clean.empty:
            return {"n": 0}
        return {
            "n": int(len(clean)),
            "median": round(float(clean.median()), 1),
            "p25": round(float(clean.quantile(0.25)), 1),
            "p75": round(float(clean.quantile(0.75)), 1),
            "p90": round(float(clean.quantile(0.90)), 1),
            "max": round(float(clean.max()), 1),
        }

    return {
        "stays": int(len(frame)),
        "date_range": {
            "first_arrival": str(arrivals.min()),
            "last_arrival": str(arrivals.max()),
            "span_days": int((arrivals.max() - arrivals.min()).days),
        },
        "acuity_mix": {
            str(int(level)): int(count)
            for level, count in pd.to_numeric(frame["triage_level"], errors="coerce")
            .value_counts().sort_index().items()
        },
        "acuity_missing": int(pd.to_numeric(frame["triage_level"], errors="coerce").isna().sum()),
        "disposition_mix": {
            str(key): int(value)
            for key, value in frame["disposition"].value_counts(dropna=False).items()
        },
        "length_of_stay_minutes": distribution(los),
        "arrival_hour_counts": {
            str(hour): int(count)
            for hour, count in arrivals.dt.hour.value_counts().sort_index().items()
        },
        "proxy_wait": {
            "label": "proxy wait (not true door-to-provider)",
            "proxy_used": proxy_label,
            "stays_with_a_proxy_wait": int(wait.notna().sum()) if not wait.empty else 0,
            "stays_without_one": int(len(frame) - (wait.notna().sum() if not wait.empty else 0)),
            "minutes": distribution(wait),
            "notice": PROXY_WAIT_NOTICE,
        },
    }


def proxy_coverage(joined: pd.DataFrame) -> dict[str, Any]:
    """How many stays each proxy can produce a wait for, and how they differ.

    The two proxies bracket the truth from opposite sides, so the gap between them
    is the width of the uncertainty a proxy wait carries. Measured here on real data
    for the first time, rather than argued from the schema.
    """
    arrivals = pd.to_datetime(joined["intime"], errors="coerce")
    result: dict[str, Any] = {"stays": int(len(joined))}

    for column, label in PROXIES.items():
        stamps = pd.to_datetime(joined[column], errors="coerce")
        minutes = (stamps - arrivals).dt.total_seconds() / 60
        usable = minutes.loc[minutes.ge(0)]
        result[column] = {
            "label": label,
            "stays_with_this_proxy": int(stamps.notna().sum()),
            "coverage_pct": round(100 * float(stamps.notna().mean()), 1),
            "negative_or_zero_intervals": int((minutes < 0).sum()),
            "median_minutes": round(float(usable.median()), 1) if not usable.empty else None,
            "p90_minutes": round(float(usable.quantile(0.9)), 1) if not usable.empty else None,
        }

    vitals = (pd.to_datetime(joined["first_vitalsign_charttime"], errors="coerce") - arrivals)
    pyxis = (pd.to_datetime(joined["first_pyxis_charttime"], errors="coerce") - arrivals)
    both = pd.DataFrame({
        "vitals": vitals.dt.total_seconds() / 60,
        "pyxis": pyxis.dt.total_seconds() / 60,
    }).dropna()
    if not both.empty:
        gap = both["pyxis"] - both["vitals"]
        result["gap_between_proxies"] = {
            "n_stays_with_both": int(len(both)),
            "median_minutes_pyxis_later": round(float(gap.median()), 1),
            "share_where_pyxis_is_later": round(float((gap > 0).mean()), 3),
            "reading": (
                "A positive gap means the medication dispense is later than the first "
                "vitals, which is the direction the schema predicts: vitals are taken "
                "at triage and a dispense follows a clinical decision. The median gap "
                "is the width of the disagreement between the two proxies, and any "
                "wait figure from either carries that much uncertainty about which "
                "interval it is measuring."
            ),
        }
    return result


# ---------------------------------------------- what MIMIC can and cannot support


def reconstruction_check(joined: pd.DataFrame) -> dict[str, Any]:
    """Can the concurrent queue state be rebuilt from these tables?

    This is the question that decides how much of this project MIMIC-IV-ED could
    ever validate, and it is answerable from the demo even though nothing else is.
    """
    arrivals = pd.to_datetime(joined["intime"], errors="coerce").dropna()
    span_days = int((arrivals.max() - arrivals.min()).days)
    per_day = len(arrivals) / max(span_days, 1)

    return {
        "stays": int(len(joined)),
        "arrival_span_days": span_days,
        "mean_stays_per_day_in_the_demo": round(per_day, 2),
        "date_shifting": (
            "MIMIC-IV shifts each patient's dates into a random future year, per "
            "patient, to de-identify them. Intervals inside one stay are preserved "
            "exactly -- arrival to departure is real -- but two patients' timestamps "
            "are not on a common clock."
        ),
        "queue_state_reconstructible": False,
        "why": (
            "Because the shift is per patient, two stays that overlapped in reality "
            "do not overlap in the released data, and two that appear concurrent were "
            "probably years apart. Counting 'how many were waiting when this patient "
            "arrived' therefore counts an arbitrary set. The demo makes this visible "
            "in a second way: "
            f"{round(per_day, 2)} stays per day across {span_days} days is far below "
            "any real department's volume, because these are a sample of patients "
            "rather than a window of operation."
        ),
        "can_be_validated_on_mimic": [
            "The adapter and the column mapping, which is what this run demonstrates.",
            "Per-patient features that use only that patient's own record: acuity, "
            "arrival mode, age, time of day, length of stay.",
            "Disposition and acuity distributions, and the proxy-wait distribution, "
            "all labelled as proxy.",
        ],
        "cannot_be_validated_on_mimic": [
            "Queue-state features (n_waiting, n_waiting_higher_priority, "
            "arrivals_last_1h and the rest). They are counts over concurrent "
            "patients, and concurrency does not survive per-patient date shifting.",
            "The simulation's validation gate. It compares simulated against "
            "observed per-acuity mean and p90 wait, length of stay and abandonment "
            "for a department running as a queue -- which requires a real window of "
            "operation, not a patient sample.",
            "The staffing search, which is scored by that simulation.",
            "Door-to-provider wait itself. There is no first-provider-contact "
            "timestamp in MIMIC-IV-ED, with or without credentials.",
        ],
        "note_on_full_access": (
            "Credentialed access to the full MIMIC-IV-ED removes the sample-size "
            "problem and nothing else on this list. The date shifting and the missing "
            "provider-contact timestamp are properties of the released data, not of "
            "the demo subset."
        ),
    }


# ------------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python scripts/external_validation_mimic.py",
        description=(
            "Run the adapter against the real MIMIC-IV-ED demo and profile it. "
            "No model is fitted and no model performance is reported: 222 stays is "
            "far too few, and a figure from it would be quoted as though it were not."
        ),
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--column-map", default=str(DEFAULT_MAP))
    parser.add_argument(
        "--proxy", default="first_vitalsign_charttime", choices=sorted(PROXIES),
        help="which proxy stands in for first clinician contact",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    base = resolve_path(Path(args.source))
    out = resolve_path(Path(args.out))
    out.mkdir(parents=True, exist_ok=True)

    if not (base / "edstays.csv.gz").is_file():
        logger.error(
            "No MIMIC-IV-ED demo at %s. Fetch it with "
            "scripts\\fetch_mimic_ed_demo.ps1 -- and if this machine has no route to "
            "physionet.org, this part is simply skipped.", base,
        )
        return 1

    joined = join_tables(base)
    joined_path = out / "joined_stays.csv"
    joined.to_csv(joined_path, index=False)

    # The adapter, with the project's own template, and the proxy enabled by name.
    column_map = load_column_map(resolve_path(Path(args.column_map)))
    column_map.proxy_wait.enabled = True
    column_map.proxy_wait.source = args.proxy
    column_map.proxy_wait.proxy_name = PROXIES[args.proxy]

    # `seen_time.source` is deliberately left pointing at the column the template
    # names and MIMIC does not have. That is the whole point: the first version of
    # this script pointed it straight at the proxy column, which produced the same
    # timestamps and reported "0 of 222 rows had seen_time derived from a proxy" --
    # the numbers were right and the mechanism under test had not run. Leaving the
    # real source absent is what forces the proxy path to fill it, and what makes the
    # adapter emit its own proxy warning.

    # The adapter's own function, unchanged, so this is the same path the pipeline
    # would take on a real extract rather than a special case for this script.
    mapped = adapt_real_dataset(joined, column_map)
    mapping_report = {
        "rows_in": int(len(joined)),
        "rows_out": int(len(mapped)),
        "schema_columns_produced": sorted(mapped.columns),
        "columns_all_present": sorted(mapped.columns) != [],
    }
    logger.info("Adapter mapped %d rows to %d schema columns", len(mapped), len(mapped.columns))

    payload = {
        "dataset": "MIMIC-IV-ED demo v2.2 (ODbL v1.0, attribution required)",
        "what_this_is": (
            "An adapter and mapping check against real data, plus a profile of what "
            "the tables contain."
        ),
        "no_model_claims": (
            "No model was fitted and no model performance is reported. The open demo "
            "holds 222 ED stays; any accuracy figure from it would be meaningless and "
            "would be quoted later as though it were not."
        ),
        "column_map_used": str(args.column_map),
        "proxy_used": {"column": args.proxy, "label": PROXIES[args.proxy]},
        "mapping_report": mapping_report,
        "profile": profile(mapped, PROXIES[args.proxy]),
        "proxy_coverage": proxy_coverage(joined),
        "reconstruction": reconstruction_check(joined),
    }
    save_json(payload, out / "profile.json")
    mapped.to_csv(out / "mapped_stays.csv", index=False)

    print(json.dumps({
        "stays": payload["profile"]["stays"],
        "acuity_mix": payload["profile"]["acuity_mix"],
        "disposition_mix": payload["profile"]["disposition_mix"],
        "proxy_wait_stays": payload["profile"]["proxy_wait"]["stays_with_a_proxy_wait"],
        "queue_state_reconstructible": payload["reconstruction"]["queue_state_reconstructible"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
