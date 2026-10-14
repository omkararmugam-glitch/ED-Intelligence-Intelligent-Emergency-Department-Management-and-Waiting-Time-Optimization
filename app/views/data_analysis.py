"""Data and analysis: the guided upload flow.

A thin wrapper. The flow itself -- upload, column confirmation, progress, results
as tabs, the baseline comparison, cancel, resume and delete -- lives in
`app/upload_page.py`, which is a large and well-tested module; rewriting it to move
it into this package would have thrown away that coverage for no gain. What this
module adds is the page's place in the navigation, so the entry script treats every
view the same way.
"""

from __future__ import annotations


def render() -> None:
    """Draw the Data and analysis page."""
    from app.upload_page import page_upload

    page_upload()
