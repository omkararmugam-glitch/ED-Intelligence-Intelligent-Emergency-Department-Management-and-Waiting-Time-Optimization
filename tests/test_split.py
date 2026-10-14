"""Tests for time-based block splitting.

Covers the requirement that the four blocks -- train / val / calib / test -- are
strictly ordered in time and non-overlapping. Random splitting must be unreachable
from the config.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.schema import validate_visits
from src.data.split import (
    BLOCK_NAMES,
    BlockBounds,
    assert_blocks_disjoint,
    assert_blocks_ordered,
    make_time_split,
    split_by_bounds,
)


@pytest.fixture()
def visits(tiny_visits) -> pd.DataFrame:
    """Full validated population; every block receives rows of every kind."""
    return validate_visits(tiny_visits).all


def test_block_order_is_the_documented_one() -> None:
    """The order is meaningful: calibration must come after early stopping."""
    assert BLOCK_NAMES == ("train", "val", "calib", "test")


def test_blocks_are_strictly_ordered_in_time(visits, config) -> None:
    """The core invariant: each block ends strictly before the next one begins."""
    split = make_time_split(visits, config)

    for earlier, later in zip(BLOCK_NAMES, BLOCK_NAMES[1:]):
        assert (
            split[earlier]["arrival_time"].max() < split[later]["arrival_time"].min()
        ), f"{earlier} overlaps {later}"
    assert_blocks_ordered(split)


def test_blocks_do_not_overlap(visits, config) -> None:
    """No patient may appear in two blocks."""
    split = make_time_split(visits, config)
    assert_blocks_disjoint(split)

    seen: set[str] = set()
    for name in BLOCK_NAMES:
        ids = set(split[name]["patient_id"])
        assert not ids & seen, f"{name} shares patients with an earlier block"
        seen |= ids


def test_blocks_partition_the_data_exactly(visits, config) -> None:
    """No row is lost and no row is duplicated."""
    split = make_time_split(visits, config)
    total = sum(len(frame) for frame in split.frames.values())
    assert total == len(visits)

    ids = pd.concat([split[name]["patient_id"] for name in BLOCK_NAMES])
    assert ids.is_unique
    assert set(ids) == set(visits["patient_id"])


def test_fraction_mode_respects_the_configured_proportions(config) -> None:
    """Fractions are honoured to within the one-day granularity of the midnight cuts."""
    from src.data.synthetic import generate_synthetic_visits

    visits = validate_visits(generate_synthetic_visits(config, n_days=200)).all
    split = make_time_split(visits, config)

    n = len(visits)
    expected = {"train": 0.60, "val": 0.15, "calib": 0.10, "test": 0.15}
    for name, want in expected.items():
        assert len(split[name]) / n == pytest.approx(want, abs=0.02), name


def test_calibration_block_sits_between_val_and_test(visits, config) -> None:
    """Conformal calibration must follow early stopping and precede the test block.

    If calibration preceded early stopping, the residuals would not be exchangeable
    with the test residuals and the coverage guarantee would not hold.
    """
    split = make_time_split(visits, config)
    assert split.val["arrival_time"].max() < split.calib["arrival_time"].min()
    assert split.calib["arrival_time"].max() < split.test["arrival_time"].min()


def test_cuts_land_on_midnight_so_no_day_spans_two_blocks(visits, config) -> None:
    """Keeping whole days together keeps each day's queue state internally coherent."""
    config["split"]["floor_to_day"] = True
    split = make_time_split(visits, config)

    for cut in split.bounds.cuts:
        assert cut == cut.normalize()

    days = {name: set(split[name]["arrival_time"].dt.date) for name in BLOCK_NAMES}
    for earlier, later in zip(BLOCK_NAMES, BLOCK_NAMES[1:]):
        assert not days[earlier] & days[later]


def test_dates_mode_uses_the_configured_boundaries(visits, config) -> None:
    """Fixed dates make runs comparable across data refreshes."""
    times = visits["arrival_time"]
    span = times.max() - times.min()
    cuts = [(times.min() + span * share).normalize() for share in (0.5, 0.7, 0.85)]

    config["split"]["mode"] = "dates"
    config["split"]["dates"] = {
        "train_end": str(cuts[0].date()),
        "val_end": str(cuts[1].date()),
        "calib_end": str(cuts[2].date()),
    }

    split = make_time_split(visits, config)
    assert list(split.bounds.cuts) == cuts
    assert split.train["arrival_time"].max() < cuts[0]
    assert split.test["arrival_time"].min() >= cuts[2]
    assert_blocks_ordered(split)


def test_dates_mode_requires_every_cut(visits, config) -> None:
    """A missing cut date must name the key that is absent."""
    config["split"]["mode"] = "dates"
    config["split"]["dates"] = {"train_end": "2024-02-01", "val_end": "2024-03-01"}
    with pytest.raises(ValueError, match="calib_end"):
        make_time_split(visits, config)


def test_unknown_mode_raises(visits, config) -> None:
    """There is no 'random' mode, and a typo must not fall back to one."""
    config["split"]["mode"] = "random"
    with pytest.raises(ValueError, match="Unknown split.mode"):
        make_time_split(visits, config)


def test_configured_dates_outside_the_data_raise(visits, config) -> None:
    """An empty block is a silent disaster downstream, so it fails here instead."""
    config["split"]["mode"] = "dates"
    config["split"]["dates"] = {
        "train_end": "2099-01-01",
        "val_end": "2099-02-01",
        "calib_end": "2099-03-01",
    }
    with pytest.raises(ValueError, match="empty block"):
        make_time_split(visits, config)


def test_fractions_leaving_no_test_block_raise(visits, config) -> None:
    """The leading fractions must leave a positive remainder for the test block."""
    config["split"]["fractions"] = {"train": 0.7, "val": 0.2, "calib": 0.15}
    with pytest.raises(ValueError, match="must sum to less than 1"):
        make_time_split(visits, config)


def test_non_positive_fraction_raises(visits, config) -> None:
    """A zero-size block is a configuration error."""
    config["split"]["fractions"] = {"train": 0.7, "val": 0.0, "calib": 0.1}
    with pytest.raises(ValueError, match="must be positive"):
        make_time_split(visits, config)


def test_missing_fraction_raises(visits, config) -> None:
    """Every block except the last needs an explicit fraction."""
    config["split"]["fractions"] = {"train": 0.7, "val": 0.15}
    with pytest.raises(ValueError, match="must give a fraction"):
        make_time_split(visits, config)


def test_unknown_block_name_raises(visits, config) -> None:
    """A typo in a block name must not be silently ignored."""
    config["split"]["fractions"] = {"train": 0.6, "val": 0.15, "calib": 0.1, "holdout": 0.1}
    with pytest.raises(ValueError, match="Unknown block"):
        make_time_split(visits, config)


def test_bounds_must_strictly_increase() -> None:
    """Cut timestamps out of order cannot produce a valid split."""
    with pytest.raises(ValueError, match="strictly increase"):
        BlockBounds(
            names=BLOCK_NAMES,
            cuts=(
                pd.Timestamp("2024-06-01"),
                pd.Timestamp("2024-05-01"),
                pd.Timestamp("2024-07-01"),
            ),
        )


def test_bounds_need_one_fewer_cut_than_blocks() -> None:
    """Four blocks need exactly three cuts."""
    with pytest.raises(ValueError, match="need 3 cuts"):
        BlockBounds(names=BLOCK_NAMES, cuts=(pd.Timestamp("2024-06-01"),))


def test_the_same_bounds_cut_every_table_identically(visits, config) -> None:
    """Reusing bounds keeps the visits table and the hourly series aligned."""
    split = make_time_split(visits, config)

    hourly = (
        visits.set_index("arrival_time").resample("h").size().rename("arrivals").reset_index()
    )
    hourly_split = split_by_bounds(hourly, split.bounds, time_column="arrival_time")

    assert hourly_split.train["arrival_time"].max() < split.bounds.cuts[0]
    assert hourly_split.test["arrival_time"].min() >= split.bounds.cuts[-1]
    assert_blocks_ordered(hourly_split)


def test_split_on_a_missing_column_raises(visits, config) -> None:
    """Splitting needs a time column; asking for a missing one is a programming error."""
    split = make_time_split(visits, config)
    with pytest.raises(KeyError, match="seen_at"):
        split_by_bounds(visits, split.bounds, time_column="seen_at")


def test_assert_blocks_ordered_catches_a_violation(visits, config) -> None:
    """The guard must actually fire when the invariant is broken."""
    split = make_time_split(visits, config)
    split.blocks["calib"] = pd.concat(
        [split.calib, split.train.head(1)], ignore_index=True
    )
    with pytest.raises(ValueError, match="Time ordering violated"):
        assert_blocks_ordered(split)


def test_assert_blocks_disjoint_catches_a_duplicate(visits, config) -> None:
    """A patient leaking into two blocks must be detected by identifier, not by time."""
    split = make_time_split(visits, config)
    split.blocks["test"] = pd.concat([split.test, split.calib.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="appears in both"):
        assert_blocks_disjoint(split)


def test_summary_is_json_serialisable(visits, config) -> None:
    """The summary is written to artifacts/metrics/, so it must serialise."""
    import json

    summary = make_time_split(visits, config).summary()
    payload = json.loads(json.dumps(summary))
    assert payload["block_order"] == list(BLOCK_NAMES)
    assert payload["train"]["n"] > 0
    assert payload["bounds"]["train_end"] < payload["bounds"]["val_end"]
    assert payload["bounds"]["val_end"] < payload["bounds"]["calib_end"]
