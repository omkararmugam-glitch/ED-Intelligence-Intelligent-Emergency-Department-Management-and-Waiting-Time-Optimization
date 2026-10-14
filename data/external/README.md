# External datasets

Two real datasets, obtained to test this project against something it did not
generate. **Neither is included in `dist\ed-project-submission.zip`** — the package
builder excludes `data\external\` — and neither is needed by the pipeline, the demo
or the test suite. Fetch them only to re-run the external validation.

Retrieved **2026-10-14** on Windows 11, into `C:\dev\ed-project`.

---

## 1. `uihc_ed_arrivals.csv` — real hourly ED arrivals

| | |
|---|---|
| **Source** | `uihc_ed_arrivals`, a dataset shipped in the R package **fastTS** |
| **Obtained from** | `https://cran.r-project.org/src/contrib/fastTS_1.0.3.tar.gz` |
| **Why that route** | This machine has no R installation, so the dataset was lifted out of the package's own source tarball rather than exported by `Rscript`. `data/uihc_ed_arrivals.RData` was extracted and read with `pyreadr` (see `requirements-extra.txt`); nothing else in the tarball was extracted or executed. |
| **Rows** | 41,640 hourly observations |
| **Covering** | 2013-07-01 00:00 to 2018-03-31 23:00, hourly, no gaps and no duplicates |
| **Institution** | University of Iowa Hospitals and Clinics emergency department |
| **Licence** | **GPL (>= 3)**, the package's licence |
| **Package author** | Ryan Andrew Peterson (`ryan-peterson@uiowa.edu`) |
| **Tarball SHA256** | `bee1648bafcebbcd657d510e5fd1eb66d759812ce4204afd10c4c829745324f8` |
| **CSV SHA256** | `ee268ffed9c66ff8c066953454f18c295d89166f5dd5f8b07c153dd322a939d8` |

### Citation

> Peterson RA. *fastTS: Fast Time Series Modeling for Seasonal Series with Exogenous
> Variables.* R package version 1.0.3. <https://cran.r-project.org/package=fastTS>

### Columns

`Year, Quarter, Month, Day, Hour, Arrivals, Date, Weekday, temp, xmas, xmas2, nye,
nyd, thx, thx1, ind, game_day`, plus a `timestamp` column added here by combining
`Date` and `Hour` so the series has a single index.

`Arrivals` is the hourly count. `temp` is the temperature **for the same hour**, which
is *not* known when a forecast is issued — see the note in
`artifacts/external_validation/uihc/report.md` about why it is not used as a feature.
`xmas … game_day` are holiday and event indicators, which *are* known in advance.

---

## 2. `mimic_ed_demo/` — MIMIC-IV-ED demo v2.2

| | |
|---|---|
| **Source** | MIMIC-IV-ED **demo**, the openly published subset |
| **Obtained from** | `https://physionet.org/content/mimic-iv-ed-demo/get-zip/2.2/` |
| **Fetched by** | `scripts\fetch_mimic_ed_demo.ps1` |
| **Size** | 97,782 bytes (95.5 KB) |
| **Contents** | 222 ED stays across `edstays`, `triage`, `vitalsign`, `pyxis`, `medrecon`, `diagnosis` |
| **Licence** | **Open Database License (ODbL) v1.0** — attribution required |
| **Credentialing** | **None.** This is the open demo. The full MIMIC-IV-ED requires PhysioNet credentialing and a signed data use agreement, and was **not** obtained. |
| **Zip SHA256** | `698e1b20ad337ff35b889569ad7ffe6b0d1557d8d3d28d67ee22d6fd3e948417` |
| **Integrity** | The archive's own `SHA256SUMS.txt` was verified: 7 of 7 files matched. |

### Required attribution

> Johnson A, Bulgarelli L, Pollard T, Celi LA, Mark R, Horng S. *MIMIC-IV-ED Demo*
> (version 2.2). PhysioNet (2023). <https://doi.org/10.13026/jd97-se30>
>
> Johnson AEW, Bulgarelli L, Shen L, et al. *MIMIC-IV, a freely accessible electronic
> health record dataset.* Scientific Data 10, 1 (2023).
>
> Goldberger AL, Amaral LAN, Glass L, et al. *PhysioBank, PhysioToolkit, and PhysioNet:
> Components of a New Research Resource for Complex Physiologic Signals.*
> Circulation 101(23):e215–e220 (2000).

### What it was used for, and what it was not

It was used to check that `configs/column_map_mimic_ed.yaml` maps the **real** column
names, that the adapter reads the real tables, and to profile what the tables contain.

**No model was fitted on it and no model performance is reported from it.** 222 stays
is far too few, and any accuracy figure would be quoted later as though it were not.
See `artifacts/external_validation/mimic_demo/report.md`.

---

## Privacy and scope

Neither dataset identifies a patient. The UIHC data is hourly counts with no patient
records at all. MIMIC-IV-ED is de-identified under HIPAA Safe Harbor, with dates
shifted per patient — which is itself a finding, and the reason the queue state cannot
be reconstructed from it.

Nothing here changes any published figure in this project. Every headline number still
comes from `data/raw/ed_v2.csv`, which is synthetic.
