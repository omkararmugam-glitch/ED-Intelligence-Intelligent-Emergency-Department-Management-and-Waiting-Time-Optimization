"""Inline SVG icons. No network, no font, no colour of their own.

**This module imports nothing and does nothing at import time.** It holds strings.

Three decisions worth recording:

1. **Inline SVG rather than Streamlit's `:material/...:` syntax.** Streamlit does
   bundle Material Symbols locally, so the material syntax would also work
   offline -- but it is only expanded in Streamlit's own markdown and label
   slots, not inside the HTML this project emits for cards, chips and table
   rows. Half the icons on a page would render and half would show as literal
   `:material/x:` text. One mechanism that works everywhere is worth more than
   the larger icon set.

2. **Every path uses `currentColor`.** No icon carries a colour, so the same
   glyph can sit on a navy sidebar, a white card and an amber alert without a
   variant for each -- and `app/theme.py` stays the only Python module with a
   colour literal in it.

3. **`aria-hidden` on all of them.** Each icon here sits next to a text label
   that says the same thing, so announcing the icon as well would make a screen
   reader repeat itself. An icon that is ever the *only* label needs a `title`
   instead, and `icon(..., label=...)` does that.

The geometry is a 24x24 viewBox with a 2px stroke, which is the Material
Symbols metric, so these sit correctly beside Streamlit's own icons.
"""

from __future__ import annotations

#: Each entry is the inner markup of a 24x24 `<svg>`: no `svg` wrapper, no size,
#: no colour. `icon()` supplies those.
#:
#: Named for what the page uses them for, not for what they draw, so a page asks
#: for "forecast" rather than for "a line going up".
PATHS: dict[str, str] = {
    # --- navigation ------------------------------------------------------
    "command_center": (
        '<path d="M3 13h4l2.5 6 3-13 2.5 7h6" />'
    ),
    "patient_flow": (
        '<circle cx="7" cy="6" r="2.4" />'
        '<path d="M3.2 20v-2.2a3.8 3.8 0 0 1 3.8-3.8h0a3.8 3.8 0 0 1 3.8 3.8V20" />'
        '<path d="M14 9h7" /><path d="M14 13h7" /><path d="M14 17h4" />'
    ),
    "forecast": (
        '<path d="M3 17l5-6 4 3 5-7 4 4" />'
        '<path d="M3 21h18" />'
    ),
    "staffing": (
        '<circle cx="8.5" cy="7.5" r="2.6" />'
        '<path d="M3 20v-2a4 4 0 0 1 4-4h3a4 4 0 0 1 4 4v2" />'
        '<path d="M17 11h4" /><path d="M19 9v4" />'
    ),
    "simulation": (
        '<path d="M9 3h6" /><path d="M10 3v5.5L5.5 17A2.5 2.5 0 0 0 7.8 21h8.4a2.5 2.5 0 0 0 2.3-4L14 8.5V3" />'
        '<path d="M7.5 15h9" />'
    ),
    "risk": (
        '<path d="M12 3.5l8.5 15.5H3.5L12 3.5z" />'
        '<path d="M12 10v4" /><path d="M12 17h.01" />'
    ),
    "quality": (
        '<path d="M12 3l2.6 5.4 5.9.8-4.3 4.1 1.1 5.9-5.3-2.9-5.3 2.9 1.1-5.9L3.5 9.2l5.9-.8L12 3z" />'
    ),
    "upload": (
        '<path d="M12 16V4" /><path d="M7.5 8.5L12 4l4.5 4.5" />'
        '<path d="M4 16v2.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V16" />'
    ),
    "system": (
        '<circle cx="12" cy="12" r="3" />'
        '<path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3M5.2 5.2l2.1 2.1M16.7 16.7l2.1 2.1M18.8 5.2l-2.1 2.1M7.3 16.7l-2.1 2.1" />'
    ),
    "research": (
        '<circle cx="10.5" cy="10.5" r="6" />'
        '<path d="M15 15l5.5 5.5" />'
    ),
    "about": (
        '<circle cx="12" cy="12" r="8.5" />'
        '<path d="M12 11v5.5" /><path d="M12 7.6h.01" />'
    ),
    # --- furniture -------------------------------------------------------
    "hospital": (
        '<path d="M4 21V9.5L12 4l8 5.5V21" />'
        '<path d="M12 9.5v5M9.5 12h5" />'
        '<path d="M9 21v-4.5h6V21" />'
    ),
    "clock": (
        '<circle cx="12" cy="12" r="8.5" />'
        '<path d="M12 7.5V12l3.2 2" />'
    ),
    "refresh": (
        '<path d="M20 11a8 8 0 1 0-2.3 5.7" />'
        '<path d="M20 5.5V11h-5.5" />'
    ),
    "warning": (
        '<path d="M12 3.8l8.2 14.9H3.8L12 3.8z" />'
        '<path d="M12 9.8v4.2" /><path d="M12 16.8h.01" />'
    ),
    "check": '<path d="M4.5 12.5l5 5 10-10.5" />',
    "cross": '<path d="M6 6l12 12M18 6L6 18" />',
    "info": (
        '<circle cx="12" cy="12" r="8.5" />'
        '<path d="M12 11v5.5" /><path d="M12 7.6h.01" />'
    ),
    "arrow_up": '<path d="M12 19V5" /><path d="M6 11l6-6 6 6" />',
    "arrow_down": '<path d="M12 5v14" /><path d="M6 13l6 6 6-6" />',
    "arrow_flat": '<path d="M5 12h14" />',
    "bed": (
        '<path d="M3 18v-6h12a4 4 0 0 1 4 4v2" />'
        '<path d="M3 18h18" /><path d="M3 12V8" />'
        '<circle cx="8" cy="9.5" r="1.8" />'
    ),
    "ambulance": (
        '<path d="M2.5 16.5V8.5h10v8" /><path d="M12.5 11h4l3 3v2.5h-7" />'
        '<circle cx="6.5" cy="18" r="1.6" /><circle cx="16.5" cy="18" r="1.6" />'
        '<path d="M6 11.5h3M7.5 10v3" />'
    ),
    "walk": (
        '<circle cx="12.5" cy="4.6" r="1.7" />'
        '<path d="M11 8l-2 4.5 2 1.5 1 6" />'
        '<path d="M12 14l3 2 1 4" /><path d="M9 12.5L6.5 11" />'
    ),
    "target": (
        '<circle cx="12" cy="12" r="8.5" /><circle cx="12" cy="12" r="4" />'
        '<circle cx="12" cy="12" r="0.8" />'
    ),
}

#: Arrival modes, mapped to the icon that stands for each. The dataset's values
#: come straight from the schema; anything unmapped falls back to a person.
ARRIVAL_ICONS: dict[str, str] = {
    "ambulance": "ambulance",
    "walk_in": "walk",
    "walk-in": "walk",
    "walkin": "walk",
    "public_transport": "walk",
    "private_vehicle": "walk",
    "other": "walk",
}


def icon(name: str, size: int = 16, *, label: str | None = None,
         classes: str = "") -> str:
    """One icon as an inline `<svg>` string, ready to drop into markup.

    Args:
        name: A key of `PATHS`. An unknown name returns an empty string rather
            than raising: a missing icon should never be the thing that takes a
            page down, and the text label beside it still says what it is.
        size: Width and height in pixels.
        label: Accessible name. Supply this only when the icon is the *only*
            label; with it the icon becomes `role="img"` with a `<title>`,
            without it the icon is hidden from assistive technology because the
            adjacent text already says the same thing.
        classes: Extra CSS classes.

    Returns:
        An `<svg>` element, or `""` for an unknown name.
    """
    body = PATHS.get(name)
    if body is None:
        return ""

    if label:
        accessibility = f'role="img" aria-label="{label}"'
        title = f"<title>{label}</title>"
    else:
        accessibility = 'aria-hidden="true" focusable="false"'
        title = ""

    css = f"ed-icon {classes}".strip()
    return (
        f'<svg class="{css}" xmlns="http://www.w3.org/2000/svg" '
        f'width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round" {accessibility}>{title}{body}</svg>'
    )


def arrival_icon(mode: str | None, size: int = 14) -> str:
    """The icon for an arrival mode, with a sensible fallback.

    A missing arrival mode is common in this dataset -- the generator drops 3% of
    them on purpose -- so an unknown value is a normal case, not an error.
    """
    key = (mode or "").strip().lower().replace(" ", "_")
    return icon(ARRIVAL_ICONS.get(key, "walk"), size=size, label=None)
