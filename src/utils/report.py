"""Markdown helpers for the checkpoint and report writers.

A tiny GitHub-flavoured table writer, so the project does not take a dependency on
`tabulate` just to render a few tables.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _cell(value: Any) -> str:
    """Render one cell, keeping floats readable and escaping the column separator."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        text = f"{value:.4g}"
    else:
        text = str(value)
    return text.replace("|", r"\|").replace("\n", " ")


def md_table(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table.

    Args:
        frame: The table to render.
        columns: Optional subset/order of columns.

    Returns:
        The Markdown table, or a placeholder line when the frame is empty.
    """
    data = frame if columns is None else frame.loc[:, list(columns)]
    if data.empty:
        return "_(no rows)_"

    headers = [str(c) for c in data.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in data.itertuples(index=False):
        lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
    return "\n".join(lines)


def md_kv_table(mapping: Mapping[str, Any], key_header: str = "Item", value_header: str = "Value") -> str:
    """Render a mapping as a two-column Markdown table."""
    frame = pd.DataFrame({key_header: list(mapping), value_header: list(mapping.values())})
    return md_table(frame)


def md_list(items: Iterable[str]) -> str:
    """Render an iterable as a Markdown bullet list."""
    rendered = [f"- {item}" for item in items]
    return "\n".join(rendered) if rendered else "_(none)_"
