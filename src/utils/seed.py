"""Deterministic seeding.

A single integer seed in the config drives everything. Modules that need their own
independent stream call `derive_seed(base, "name")`, which is stable across runs and
platforms (unlike Python's salted `hash()`).
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np

_MAX_UINT32 = 2**32 - 1


def set_global_seeds(seed: int) -> None:
    """Seed Python's `random`, NumPy's legacy global state and PYTHONHASHSEED."""
    seed = int(seed) % _MAX_UINT32
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def derive_seed(base_seed: int, name: str) -> int:
    """Derive a stable child seed from a base seed and a label.

    Uses BLAKE2b so the result does not depend on PYTHONHASHSEED or interpreter version.

    Args:
        base_seed: The global seed from the config.
        name: A label identifying the consumer, e.g. "synthetic" or "replication_3".

    Returns:
        An integer in [0, 2**32).
    """
    digest = hashlib.blake2b(f"{int(base_seed)}::{name}".encode("utf-8"), digest_size=8)
    return int.from_bytes(digest.digest(), "big") % _MAX_UINT32


def rng(base_seed: int, name: str) -> np.random.Generator:
    """Return an independent NumPy Generator for the named consumer."""
    return np.random.default_rng(derive_seed(base_seed, name))
