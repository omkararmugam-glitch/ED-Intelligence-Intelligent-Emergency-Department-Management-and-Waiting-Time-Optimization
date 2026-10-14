"""Shared dashboard constants, importable from anywhere without side effects.

This module exists because of a crash. `app/dashboard.py` is the Streamlit entry
script: importing it *runs* it, and the first thing it runs is
`st.set_page_config()`. A page module that did `from app.dashboard import API_URL`
therefore re-executed the entry script mid-render and Streamlit raised
`StreamlitSetPageConfigMustBeFirstCommandError`.

So the rule this module enforces by existing: **nothing imports the entry script.**
Shared values live here, the entry script and every view import them from here, and
this module imports no Streamlit page code, no Streamlit at all, and calls no `st.*`
command at import time. Importing it is free, which is the whole point.

Colours are not here; they are in `app/theme.py`, which owns presentation.
`tests/test_dashboard_imports.py` enforces the rules above.
"""

from __future__ import annotations

import os

# ------------------------------------------------------------------- the service

#: Which API the dashboard calls. Two accepted names, in this order:
#:
#: * `ED_API_BASE_URL` -- what `docker-compose.yml` sets, to `http://api:8000`: the
#:   compose service name, never localhost, because inside the container network
#:   localhost is the dashboard itself.
#: * `ED_API_URL` -- the original name. `scripts/run_app.ps1` sets it and
#:   `tests/test_local_run_scripts.py` pins its default against the launcher so the
#:   banner cannot print a URL nothing is listening on.
#:
#: Both exist because the containerised deployment arrived after the launchers, and
#: renaming the variable would have broken every shortcut already written down. The
#: default stays localhost, which is right for a developer running both by hand.
API_URL = (
    os.environ.get("ED_API_BASE_URL")
    or os.environ.get("ED_API_URL")
    or "http://localhost:8000"
).rstrip("/")

#: The queue's priority-aging rule, mirrored from `synthetic.priority_aging_*` in
#: `configs/default.yaml`.
#:
#: Mirrored rather than read, because the dashboard image installs
#: `requirements-app.txt`, which has no PyYAML in it -- a dashboard that opened the
#: analysis config would crash the container on start. `tests/test_design_system.py`
#: asserts these two numbers still equal the config's, so the mirror cannot drift.
#:
#: The rule, which the generator and the simulator both apply:
#:
#:     effective_priority = max(triage_level - waited / minutes_per_level, floor)
#:
#: Lower is served first. The floor is a clinical safety constraint, not a tuning
#: knob: at 2.5 an aged low-acuity patient may overtake a fresher ESI 3 or 4, and
#: can never overtake a freshly arrived ESI 1 or 2.
AGING_MINUTES_PER_LEVEL = 25.0
AGING_FLOOR = 2.5

#: Session-state key holding the replay moment the reader has chosen.
#:
#: It lives here rather than in a view because two pages set it and the shell
#: reads it: the Command center's replay control, the Patient flow board's time
#: slider, and the top bar that labels whichever moment is current. A constant two
#: modules share belongs in a module that imports neither of them.
REPLAY_MOMENT_KEY = "ed_replay_moment"

#: Header the key travels in. Duplicated from `api.auth.API_KEY_HEADER` rather than
#: imported: the dashboard image installs `requirements-app.txt`, which has no
#: FastAPI in it, so importing anything under `api/` would crash the container on
#: start. `tests/test_api_auth.py` asserts the two constants stay equal.
API_KEY_HEADER = "X-API-Key"

#: Shared secret the API requires when it is started with `ED_API_KEY` set. The
#: dashboard reads the SAME variable name, so one value in the environment -- or one
#: line in `.env` -- configures both sides. Unset means the API is open and the
#: dashboard sends no header, which is the default posture.
#:
#: Never render this. It is a credential, and a dashboard is a screen other people
#: can see; `app/api_client.py` puts it in a header and nothing else reads it.
API_KEY = os.environ.get("ED_API_KEY", "") or None

#: Seconds before an API call is abandoned. Generous: a simulation run is slow.
REQUEST_TIMEOUT = float(os.environ.get("ED_API_TIMEOUT", "60"))

#: Seconds a read-only answer may be reused. Short, because the point of a
#: dashboard is to be current; long enough that moving between pages does not
#: re-ask the API for the same thing.
CACHE_TTL_SECONDS = 30

#: A slower cache for things that change only when the pipeline is re-run.
ARTIFACT_CACHE_TTL_SECONDS = 300


# ------------------------------------------------------------------- the product

#: What the product is called, in the sidebar and the browser tab.
APP_NAME = "ED Operations"

#: One line under the name, saying what it is for.
APP_PURPOSE = "Anticipate demand, waits and staffing pressure"

#: Shown in About and the sidebar footer. Bumped by hand with the dashboard.
APP_VERSION = "2.0"

#: The badge every page carries. The data is generated, and a dashboard that looks
#: like a hospital feed has to keep saying that it is not one.
DEMO_BADGE = "Demo data (synthetic)"

#: The standing disclaimer. This is decision support: a person decides.
DECISION_SUPPORT_NOTICE = (
    "Decision support, not clinical advice. Every figure is an estimate for a "
    "qualified person to weigh against what else they know."
)


# -------------------------------------------------------------------- local files

#: Where the pipeline's artifact tables are, for the pages that read them rather
#: than asking the API. Reading a saved table is not model loading: the dashboard
#: never opens a model file.
ARTIFACTS_DIR = os.environ.get("ED_ARTIFACTS_DIR", "artifacts")

#: Where the validated visits table lives, which is what Shift replay reads.
PROCESSED_DIR = os.environ.get("ED_PROCESSED_DIR", "data/processed")


# ------------------------------------------------------------------ presentation

#: Hours of forecast the Demand forecast page offers. The API accepts 1-168; the
#: config's horizon is 72h, beyond which the weekly seasonal feature runs out.
FORECAST_HORIZONS = (24, 48, 72)

#: Replications per scenario the What-if page offers. The API caps this itself;
#: these are the choices a person is given.
SCENARIO_REPLICATIONS = (4, 8, 12)

#: Nominal coverage of every interval the service returns, from
#: `configs/default.yaml`. Used only to word the explanation of a range.
INTERVAL_COVERAGE = 0.90
