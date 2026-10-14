"""Standing notices that cross module boundaries.

**This module imports nothing and does nothing at import time.** That is its whole
purpose: it is the place a constant can live when both `api/` and `src/` need it,
without either importing the other.

The bug this prevents is specific and has bitten this project before in the dashboard
(`app/config.py` exists for the same reason). `DECISION_SUPPORT_NOTICE` was defined in
`api/main.py`, which builds a FastAPI application, loads a model bundle at startup and
registers routes. Anything under `src/serving/` that wanted the constant would have had
to import that module and would have paid for all of it -- and `api/main.py` imports
`src/serving/` already, so the import would have been circular.

The rule, in one line: **a constant two packages share belongs in a module that
imports neither of them.**
"""

from __future__ import annotations

#: Attached to every prediction this service returns.
#:
#: A number served over HTTP outlives the context it was produced in: it gets pasted
#: into a message, quoted in a meeting, screenshotted. The notice travels with it.
DECISION_SUPPORT_NOTICE = (
    "Decision support only. This is an estimate for a clinician or manager to weigh, "
    "not a clinical decision, a triage instruction, or a guarantee of waiting time. "
    "It is produced from synthetic data calibrated to published benchmarks."
)
