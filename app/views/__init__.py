"""The dashboard's pages, one module each.

Every module here exposes a single `render()` and nothing else the entry script
needs. Two rules, both enforced by `tests/test_dashboard_imports.py`:

* **Nothing here imports `app.dashboard`.** That module is the Streamlit entry
  script, so importing it re-runs it and its `st.set_page_config()` call raises
  `StreamlitSetPageConfigMustBeFirstCommandError` mid-render. Shared values come
  from `app.config`, shared parts from `app.theme`, and the service from
  `app.api_client`.
* **No `st.*` call at import time.** Everything a page draws happens inside
  `render()`, which runs when that page is the one being shown.

The directory is `views/` rather than `pages/` on purpose: Streamlit treats a
`pages/` directory beside the entry script as an automatic multipage app and would
build its own navigation from it, competing with the sidebar this app draws.
"""
