"""Time-based splitting into ordered blocks.

Random splitting is forbidden in this project. An ED queue is autocorrelated: two
patients who arrived minutes apart share almost all of their queue state, so a random
split leaks the test period into training and inflates every metric. Every split here
is a forward-in-time cut, and `assert_blocks_ordered` is asserted in the test suite.

The pipeline uses four blocks, in chronological order:

    train   fitting the models
    val     early stopping, and choosing the classifier's decision threshold
    calib   conformal calibration -- held out from both fitting and tuning
    test    reported once, touched by nothing else

The separation of `val` from `calib` matters. Split conformal is only valid when the
calibration residuals are exchangeable with the test residuals, which fails if the
model was early-stopped on the same rows. And the decision threshold must come from
`val`, never from `test`, or the reported recall is optimistic.

Cuts land on midnight boundaries by default so that a single day is never divided
between two blocks, which keeps the queue-state features of a day internally coherent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

#: The blocks, in chronological order. Order is meaningful and is relied upon.
BLOCK_NAMES: tuple[str, ...] = ("train", "val", "calib", "test")


@dataclass(frozen=True)
class BlockBounds:
    """The cut timestamps that define a sequence of forward-in-time blocks.

    For blocks `(b0, b1, ..., bk)` there are `k` cuts. A row belongs to block `i` when
    `cuts[i-1] <= time < cuts[i]`, with the first and last blocks unbounded on their
    outer side.
    """

    names: tuple[str, ...]
    cuts: tuple[pd.Timestamp, ...]

    def __post_init__(self) -> None:
        if len(self.cuts) != len(self.names) - 1:
            raise ValueError(
                f"{len(self.names)} blocks need {len(self.names) - 1} cuts, got {len(self.cuts)}"
            )
        for earlier, later in zip(self.cuts, self.cuts[1:]):
            if earlier >= later:
                raise ValueError(f"Cuts must strictly increase; {earlier} is not before {later}")

    def to_dict(self) -> dict[str, str]:
        """JSON-serialisable view, naming each cut by the block it ends."""
        return {
            f"{name}_end": str(cut) for name, cut in zip(self.names[:-1], self.cuts)
        }


@dataclass
class TimeSplit:
    """Chronologically ordered, disjoint slices of a table."""

    blocks: dict[str, pd.DataFrame]
    bounds: BlockBounds
    time_column: str = "arrival_time"

    def __iter__(self) -> Iterator[tuple[str, pd.DataFrame]]:
        """Iterate as (name, frame) pairs in chronological order."""
        return iter((name, self.blocks[name]) for name in self.bounds.names)

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.blocks[name]

    @property
    def names(self) -> tuple[str, ...]:
        """Block names in chronological order."""
        return self.bounds.names

    @property
    def frames(self) -> dict[str, pd.DataFrame]:
        """Mapping of block name to frame."""
        return self.blocks

    @property
    def train(self) -> pd.DataFrame:
        """The fitting block."""
        return self.blocks["train"]

    @property
    def val(self) -> pd.DataFrame:
        """The early-stopping and threshold-selection block."""
        return self.blocks["val"]

    @property
    def calib(self) -> pd.DataFrame:
        """The conformal calibration block."""
        return self.blocks["calib"]

    @property
    def test(self) -> pd.DataFrame:
        """The held-out reporting block."""
        return self.blocks["test"]

    def summary(self) -> dict[str, Any]:
        """Row counts and date ranges per block, for the metrics artifact."""
        total = sum(len(frame) for frame in self.blocks.values())
        out: dict[str, Any] = {
            "bounds": self.bounds.to_dict(),
            "time_column": self.time_column,
            "block_order": list(self.names),
            "n_total": total,
        }
        for name, frame in self.blocks.items():
            if frame.empty:
                out[name] = {"n": 0, "share": 0.0, "start": None, "end": None, "days": 0.0}
                continue
            times = frame[self.time_column]
            out[name] = {
                "n": int(len(frame)),
                "share": round(len(frame) / max(total, 1), 4),
                "start": str(times.min()),
                "end": str(times.max()),
                "days": round(float((times.max() - times.min()).total_seconds() / 86400.0), 2),
            }
        return out

    def log(self) -> None:
        """Log the block composition at INFO level."""
        total = sum(len(f) for f in self.blocks.values())
        for name in self.names:
            frame = self.blocks[name]
            if frame.empty:
                logger.warning("Block %-6s is EMPTY", name)
                continue
            times = frame[self.time_column]
            logger.info(
                "Block %-6s n=%6d (%4.1f%%)  %s -> %s",
                name,
                len(frame),
                100.0 * len(frame) / max(total, 1),
                times.min(),
                times.max(),
            )


def _block_fractions(config: dict[str, Any]) -> dict[str, float]:
    """Read and validate the per-block fractions from the config.

    The final block's fraction is implied by the others, so it may be omitted. All
    fractions must be positive and the non-final ones must leave room for it.

    Raises:
        ValueError: If a fraction is non-positive, a block is unknown, or the
            non-final fractions do not leave a positive remainder.
    """
    split_cfg = config["split"]
    raw = dict(split_cfg.get("fractions") or {})

    unknown = set(raw) - set(BLOCK_NAMES)
    if unknown:
        raise ValueError(f"Unknown block(s) in split.fractions: {sorted(unknown)}")

    leading = BLOCK_NAMES[:-1]
    missing = [name for name in leading if name not in raw]
    if missing:
        raise ValueError(f"split.fractions must give a fraction for each of {list(leading)}")

    for name in leading:
        value = float(raw[name])
        if value <= 0.0:
            raise ValueError(f"split.fractions[{name}] must be positive, got {value}")

    used = sum(float(raw[name]) for name in leading)
    remainder = 1.0 - used
    if remainder <= 0.0:
        raise ValueError(
            f"split.fractions for {list(leading)} sum to {used:.3f}, leaving nothing for "
            f"'{BLOCK_NAMES[-1]}'; they must sum to less than 1"
        )

    declared_last = raw.get(BLOCK_NAMES[-1])
    if declared_last is not None and abs(float(declared_last) - remainder) > 1e-6:
        logger.warning(
            "split.fractions['%s'] is %.3f but the remainder is %.3f; using the remainder",
            BLOCK_NAMES[-1],
            float(declared_last),
            remainder,
        )

    fractions = {name: float(raw[name]) for name in leading}
    fractions[BLOCK_NAMES[-1]] = remainder
    return fractions


def _fraction_bounds(
    times: pd.Series, fractions: dict[str, float], floor_to_day: bool
) -> BlockBounds:
    """Derive cut timestamps from quantiles of the time column."""
    cumulative = 0.0
    quantiles: list[float] = []
    for name in BLOCK_NAMES[:-1]:
        cumulative += fractions[name]
        quantiles.append(cumulative)

    cuts = [pd.Timestamp(times.quantile(q)) for q in quantiles]
    if floor_to_day:
        floored = [cut.normalize() for cut in cuts]
        if all(a < b for a, b in zip(floored, floored[1:])):
            return BlockBounds(names=BLOCK_NAMES, cuts=tuple(floored))
        logger.warning(
            "Flooring block bounds to midnight would collapse a block (the data spans "
            "too few days); using exact quantile timestamps instead."
        )
    return BlockBounds(names=BLOCK_NAMES, cuts=tuple(cuts))


def _date_bounds(config: dict[str, Any]) -> BlockBounds:
    """Read fixed cut dates from the config's `split.dates` block."""
    dates = config["split"].get("dates") or {}
    cuts = []
    for name in BLOCK_NAMES[:-1]:
        key = f"{name}_end"
        if key not in dates:
            raise ValueError(
                f"split.mode is 'dates' but split.dates is missing {key!r}; "
                f"expected keys: {[f'{n}_end' for n in BLOCK_NAMES[:-1]]}"
            )
        cuts.append(pd.Timestamp(dates[key]))
    return BlockBounds(names=BLOCK_NAMES, cuts=tuple(cuts))


def split_by_bounds(
    frame: pd.DataFrame,
    bounds: BlockBounds,
    time_column: str = "arrival_time",
) -> TimeSplit:
    """Apply existing bounds to a table.

    Used to cut the feature matrix and the hourly arrival series on exactly the same
    dates as the visits table, so no boundary ever differs between artifacts.

    Args:
        frame: Table containing `time_column`.
        bounds: Pre-computed block bounds.
        time_column: Name of the datetime column to cut on.

    Returns:
        A `TimeSplit`.

    Raises:
        KeyError: If `time_column` is absent.
    """
    if time_column not in frame.columns:
        raise KeyError(f"Cannot split on missing column {time_column!r}")

    times = pd.to_datetime(frame[time_column])
    edges = [None, *bounds.cuts, None]
    blocks: dict[str, pd.DataFrame] = {}
    for position, name in enumerate(bounds.names):
        lower, upper = edges[position], edges[position + 1]
        mask = pd.Series(True, index=frame.index)
        if lower is not None:
            mask &= times >= lower
        if upper is not None:
            mask &= times < upper
        blocks[name] = frame.loc[mask].copy()

    return TimeSplit(blocks=blocks, bounds=bounds, time_column=time_column)


def make_time_split(
    frame: pd.DataFrame,
    config: dict[str, Any],
    time_column: str = "arrival_time",
) -> TimeSplit:
    """Split a table into forward-in-time blocks per the config's `split` block.

    Two modes are supported:
      * `fraction` -- cut at quantiles of the time column (default), so the split
        adapts to however much data is present.
      * `dates` -- cut at fixed dates from `split.dates`, which is what you want when
        comparing runs across data refreshes.

    Args:
        frame: Table containing `time_column`.
        config: Full pipeline config; reads the `split` block.
        time_column: Name of the datetime column to cut on.

    Returns:
        A `TimeSplit` whose blocks are chronologically ordered and disjoint.

    Raises:
        ValueError: If the mode is unknown, or the bounds leave a block empty.
    """
    split_cfg = config["split"]
    mode = str(split_cfg.get("mode", "fraction")).lower()
    times = pd.to_datetime(frame[time_column])

    if mode == "fraction":
        bounds = _fraction_bounds(
            times,
            _block_fractions(config),
            floor_to_day=bool(split_cfg.get("floor_to_day", True)),
        )
    elif mode == "dates":
        bounds = _date_bounds(config)
    else:
        raise ValueError(f"Unknown split.mode {mode!r}; expected 'fraction' or 'dates'")

    logger.info(
        "Time split (mode=%s) on %r into %s at cuts %s",
        mode,
        time_column,
        " < ".join(BLOCK_NAMES),
        [str(cut) for cut in bounds.cuts],
    )
    split = split_by_bounds(frame, bounds, time_column=time_column)
    split.log()

    empty = [name for name, part in split.blocks.items() if part.empty]
    if empty:
        raise ValueError(
            f"Split produced empty block(s): {empty}. Data spans {times.min()} to "
            f"{times.max()}; cuts are {bounds.to_dict()}. Adjust the split block in the config."
        )
    assert_blocks_ordered(split)
    return split


def assert_blocks_ordered(split: TimeSplit) -> None:
    """Verify the blocks are strictly forward in time and non-overlapping.

    Checks that the last timestamp of each block strictly precedes the first timestamp
    of the next, which is the invariant the leakage and conformal guarantees depend on.

    Args:
        split: The split to check.

    Raises:
        ValueError: If any ordering is violated.
    """
    column = split.time_column
    present = [name for name in split.names if not split.blocks[name].empty]

    for earlier, later in zip(present, present[1:]):
        max_earlier = pd.to_datetime(split.blocks[earlier][column]).max()
        min_later = pd.to_datetime(split.blocks[later][column]).min()
        if not max_earlier < min_later:
            raise ValueError(
                f"Time ordering violated: max({earlier}.{column}) = {max_earlier} is not "
                f"strictly before min({later}.{column}) = {min_later}"
            )
    logger.debug("Block ordering verified for %s", present)


def assert_blocks_disjoint(split: TimeSplit, id_column: str = "patient_id") -> None:
    """Verify no row appears in more than one block.

    Args:
        split: The split to check.
        id_column: Column holding a unique row identifier.

    Raises:
        ValueError: If any identifier appears in two blocks.
    """
    seen: dict[Any, str] = {}
    for name in split.names:
        frame = split.blocks[name]
        if id_column not in frame.columns:
            raise KeyError(f"Cannot check disjointness without {id_column!r}")
        for identifier in frame[id_column]:
            previous = seen.get(identifier)
            if previous is not None:
                raise ValueError(
                    f"{id_column}={identifier!r} appears in both {previous!r} and {name!r}"
                )
            seen[identifier] = name
    logger.debug("Block disjointness verified over %d rows", len(seen))
