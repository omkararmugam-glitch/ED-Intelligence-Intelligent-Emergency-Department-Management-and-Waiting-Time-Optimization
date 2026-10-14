# Container image scan

Scanner: **Docker Scout 1.24.0** (`docker scout quickview` and `docker scout cves`),
run against locally built images. Trivy was not installed on this machine, so these
results come from Scout alone.

Date of scan: **2026-10-10**. Images: `ed-intelligence-api:latest` (1.05 GB) and
`ed-intelligence-app:latest` (753 MB), both on base image `python:3.12-slim`
(Debian 13 "trixie"). Scout reports the base image as **not outdated**.

Vulnerability databases move. Re-run the scan rather than trusting this page:

```powershell
docker compose build
docker scout quickview ed-intelligence-api:latest
docker scout cves --only-severity critical,high ed-intelligence-api:latest
```

## Summary as scanned

| Image | Critical | High | Medium | Low | Scout health score |
|---|---|---|---|---|---|
| `ed-intelligence-api` | 0 | 9 | 10 | 31 | D (33%) |
| `ed-intelligence-app` | 0 | 15 | 14 | 28 | C (56%) |
| base `python:3.12-slim` | 0 | 1 | 6 | 27 | — |

**No critical findings in either image.** Every high finding is listed below with what
was done about it.

## Summary after the bumps

Images rebuilt with `docker compose build --no-cache` and re-scanned:

| Image | Critical | High | Medium | Low | Change in highs |
|---|---|---|---|---|---|
| `ed-intelligence-api` | 0 | **5** | 9 | 29 | 9 → 5 |
| `ed-intelligence-app` | 0 | **14** | 14 | 28 | 15 → 14 |

The four highs cleared on the API are the three `python-multipart` issues and the
`pyarrow` one; the app's single clearance is `pyarrow`. What remains is exactly the
blocked and deliberate set below: `starlette` (3) + `lightgbm` (1) + `zlib` (1) on the
API, and `pillow` (13) + `zlib` (1) on the app. Verified in the rebuilt image:
`pyarrow 23.0.1`, `python-multipart 0.0.30`, `lightgbm 4.5.0` (unchanged, deliberately).

## High findings, and the decision on each

| Package | Image | Highs | Fixed in | Action |
|---|---|---|---|---|
| `python-multipart` 0.0.20 | api | 3 | 0.0.30 | **Bumped to 0.0.30** |
| `pyarrow` 17.0.0 | api, app | 1 each | 23.0.1 | **Bumped to 23.0.1** |
| `pillow` 11.3.0 | app | 13 | 12.3.0 | **Blocked** — see below |
| `starlette` 0.41.3 | api | 3 | 0.49.1 / 1.1.0 / 1.3.1 | **Blocked** — see below |
| `lightgbm` 4.5.0 | api | 1 | 4.6.0 | **Deliberately not bumped** — see below |
| `zlib` (Debian) | api, app | 1 | not fixed | **No fix exists** — see below |

### Bumped

**`python-multipart` 0.0.20 → 0.0.30.** CVE-2026-24486 (path traversal),
CVE-2026-42561 and CVE-2026-53539 (resource exhaustion). This library parses the
multipart body of `POST /jobs`, which is exactly the attack surface those CVEs
describe, and it touches no model code, so the bump is both the most valuable and the
safest on the list.

**`pyarrow` 17.0.0 → 23.0.1.** CVE-2026-25087 (use-after-free). pyarrow only
round-trips the intermediate parquet files. Float64 survives that exactly, so no
metric can move; the containerised reproduction of the four headline numbers is the
check that it did not.

### Blocked by a dependency, not applied

**`pillow` 11.3.0 → 12.3.0 (13 highs, the largest single win).** Cannot be done on its
own: `streamlit==1.41.1` requires `pillow<12,>=7.1.0`, so pip refuses the pin. Note
the API image already resolves `pillow` to 12.3.0 and is clean — this is a Streamlit
constraint and nothing else.

Fixing it means upgrading Streamlit, which changes the dashboard's rendering library.
That is not a change to make without a human running
`docs/upload_manual_test_checklist.md` and `docs/dashboard_manual_test_checklist.md`
in a real browser afterwards, which was not done here. **Recommended as the next
security change**, paired with that manual pass.

The exposure in the meantime is narrow: all 13 are decode-side issues (out-of-bounds
reads and writes, integer overflows, unbounded allocation) reached by *parsing an
image*. The dashboard decodes no uploaded images — it uploads CSVs and renders charts
Streamlit produces itself — so there is no path from user input to the vulnerable code.

**`starlette` 0.41.3 → 0.49.1 or later (3 highs: one SSRF, two resource exhaustion).**
`fastapi==0.115.6` pins `starlette<0.42`, so this needs a FastAPI major upgrade, which
is a change to the whole request/response layer and wants its own tested pass. The
service binds to `127.0.0.1` by default and now supports an optional API key
(`ED_API_KEY`), which narrows but does not remove the exposure. **Recommended, with
the API test suite as the gate.**

### Deliberately not bumped

**`lightgbm` 4.5.0 → 4.6.0.** CVE-2024-43598 is a heap-based buffer overflow reached by
**parsing a malformed model file**. Nothing in this project loads a model file from an
untrusted source: the boosters under `artifacts/models/` are produced by this
repository's own pipeline, and the upload workflow accepts a CSV, never a model.

Against that, LightGBM *is* the model. A version change alters tree construction and
would move `11.52`, `0.8875` and `2.356` — the numbers this repository exists to
reproduce and that `tests/test_headline_regression.py` pins. Upgrading it is therefore
a deliberate, flagged change, not a security patch:

```powershell
# the procedure, if the bump is wanted
$env:ED_ALLOW_HEADLINE_CHANGE = "1"
python -m src.pipeline run --config configs/default.yaml --data real
# then compare the new numbers, update EXPECTED in
# tests/test_headline_regression.py, and unset the variable
```

### No fix available

**`zlib` 1:1.3.dfsg+really1.3.1-1 (Debian trixie).** CVE-2026-85091. Scout reports
"Fixed version: not fixed" — there is no patched Debian package to move to. It comes
from `python:3.12-slim`, which Scout still reports as the current base image. Nothing
to do but re-check when Debian ships a fix.

## What this scan does not cover

- **No second scanner.** Trivy was not installed, so these are Scout's findings only.
  A second scanner would likely surface a different set, particularly for the OS layer.
- **Licences.** Scout's "Copyleft licensed packages found" policy fails on both images
  (366 packages). That is a licensing question, not a vulnerability, and has not been
  reviewed here.
- **Medium and low findings.** 10M/31L on the API and 14M/28L on the app are not
  itemised above. None are fixable-and-ignored high-severity issues; they were left
  for a routine dependency refresh.
- **Runtime behaviour.** This is a static scan of installed packages. It says nothing
  about how the services behave.
