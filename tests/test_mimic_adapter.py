"""Adapter tests on a hand-written, clearly fake MIMIC-shaped extract.

**No MIMIC-IV-ED data was downloaded, accessed or used.** The fixture below is thirty
rows typed out by hand to have MIMIC's *column names and value vocabulary*, with
invented timestamps and invented identifiers. It is not a sample of MIMIC, it is not
derived from MIMIC, and its numbers mean nothing.

What it does let us test is the part that will break first on a real extract: whether
the column map reads MIMIC's names and coded values correctly, and whether the proxy
wait behaves the way the documentation promises.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from src.data.adapter import (
    PROXY_WAIT_NOTICE,
    ColumnMap,
    ProxyWait,
    adapt_real_dataset,
    load_column_map,
    proxy_wait_status,
    read_raw,
)
from src.data.schema import load_schema
from src.utils.io import project_root

#: A fake extract with MIMIC-IV-ED's column names, joined the way the real one would
#: have to be. Thirty stays. Every value is invented.
#:
#: Deliberate awkwardness, all of which a real extract would contain:
#:   - stay 11 has no first vitals and no dispense (nothing to proxy from);
#:   - stay 12 eloped; stay 13 left without being seen (two codes, one meaning);
#:   - stay 14 has a null acuity (triage not completed);
#:   - stay 15 died in the department;
#:   - arrival_transport includes HELICOPTER and UNKNOWN.
FAKE_MIMIC_CSV = textwrap.dedent(
    """\
    stay_id,subject_id,intime,outtime,disposition,acuity,gender,arrival_transport,chiefcomplaint,anchor_age,first_vitalsign_charttime,first_pyxis_charttime
    30000001,10000001,2180-01-01 00:14:00,2180-01-01 04:02:00,HOME,3,F,WALK IN,abdominal pain,41,2180-01-01 00:31:00,2180-01-01 01:22:00
    30000002,10000002,2180-01-01 01:05:00,2180-01-01 09:40:00,ADMITTED,2,M,AMBULANCE,chest pain,67,2180-01-01 01:09:00,2180-01-01 01:51:00
    30000003,10000003,2180-01-01 02:30:00,2180-01-01 05:11:00,HOME,4,F,WALK IN,sore throat,23,2180-01-01 03:02:00,
    30000004,10000004,2180-01-01 03:47:00,2180-01-01 12:05:00,ADMITTED,1,M,AMBULANCE,cardiac arrest,74,2180-01-01 03:49:00,2180-01-01 03:55:00
    30000005,10000005,2180-01-01 06:12:00,2180-01-01 08:30:00,HOME,5,F,WALK IN,medication refill,33,2180-01-01 07:05:00,
    30000006,10000006,2180-01-01 07:20:00,2180-01-01 13:44:00,TRANSFER,2,M,HELICOPTER,major trauma,52,2180-01-01 07:23:00,2180-01-01 07:40:00
    30000007,10000007,2180-01-01 08:55:00,2180-01-01 11:02:00,HOME,4,F,WALK IN,sprain,29,2180-01-01 09:30:00,
    30000008,10000008,2180-01-01 09:10:00,2180-01-01 18:22:00,ADMITTED,3,M,AMBULANCE,shortness of breath,81,2180-01-01 09:18:00,2180-01-01 10:02:00
    30000009,10000009,2180-01-01 10:40:00,2180-01-01 12:50:00,HOME,4,F,UNKNOWN,rash,45,2180-01-01 11:20:00,
    30000010,10000010,2180-01-01 11:25:00,2180-01-01 16:08:00,HOME,3,M,WALK IN,fever,38,2180-01-01 11:58:00,2180-01-01 12:40:00
    30000011,10000011,2180-01-01 12:00:00,2180-01-01 13:15:00,HOME,5,F,WALK IN,suture removal,27,,
    30000012,10000012,2180-01-01 12:30:00,2180-01-01 14:05:00,ELOPED,4,M,WALK IN,back pain,36,2180-01-01 13:10:00,
    30000013,10000013,2180-01-01 13:05:00,2180-01-01 15:30:00,LEFT WITHOUT BEING SEEN,5,F,WALK IN,cold symptoms,31,,
    30000014,10000014,2180-01-01 14:20:00,2180-01-01 17:45:00,HOME,,M,WALK IN,dizziness,59,2180-01-01 14:55:00,2180-01-01 15:30:00
    30000015,10000015,2180-01-01 15:00:00,2180-01-01 19:12:00,EXPIRED,1,F,AMBULANCE,unresponsive,88,2180-01-01 15:02:00,2180-01-01 15:08:00
    30000016,10000016,2180-01-02 00:40:00,2180-01-02 03:30:00,HOME,4,M,WALK IN,laceration,24,2180-01-02 01:12:00,
    30000017,10000017,2180-01-02 02:15:00,2180-01-02 10:50:00,ADMITTED,2,F,AMBULANCE,stroke symptoms,70,2180-01-02 02:19:00,2180-01-02 02:48:00
    30000018,10000018,2180-01-02 04:05:00,2180-01-02 06:22:00,HOME,5,M,WALK IN,insect bite,19,2180-01-02 04:50:00,
    30000019,10000019,2180-01-02 05:30:00,2180-01-02 09:14:00,HOME,3,F,WALK IN,headache,47,2180-01-02 06:05:00,2180-01-02 06:55:00
    30000020,10000020,2180-01-02 07:45:00,2180-01-02 15:30:00,ADMITTED,3,M,AMBULANCE,sepsis concern,76,2180-01-02 07:52:00,2180-01-02 08:35:00
    30000021,10000021,2180-01-02 08:20:00,2180-01-02 10:40:00,HOME,4,F,WALK IN,urinary symptoms,34,2180-01-02 08:58:00,
    30000022,10000022,2180-01-02 09:35:00,2180-01-02 14:02:00,HOME,3,M,WALK IN,fracture,55,2180-01-02 10:10:00,2180-01-02 10:48:00
    30000023,10000023,2180-01-02 10:50:00,2180-01-02 12:15:00,LEFT AGAINST MEDICAL ADVICE,4,F,WALK IN,dental pain,28,2180-01-02 11:30:00,
    30000024,10000024,2180-01-02 11:40:00,2180-01-02 20:05:00,ADMITTED,2,M,AMBULANCE,major bleeding,63,2180-01-02 11:44:00,2180-01-02 12:10:00
    30000025,10000025,2180-01-02 13:10:00,2180-01-02 15:48:00,HOME,4,F,WALK IN,minor injury,42,2180-01-02 13:45:00,
    30000026,10000026,2180-01-02 14:25:00,2180-01-02 18:30:00,HOME,3,M,WALK IN,vomiting,50,2180-01-02 15:00:00,2180-01-02 15:42:00
    30000027,10000027,2180-01-02 16:00:00,2180-01-02 17:20:00,HOME,5,F,WALK IN,minor rash,22,2180-01-02 16:40:00,
    30000028,10000028,2180-01-02 17:35:00,2180-01-03 01:10:00,ADMITTED,1,M,AMBULANCE,respiratory failure,79,2180-01-02 17:37:00,2180-01-02 17:44:00
    30000029,10000029,2180-01-02 19:15:00,2180-01-02 21:40:00,HOME,4,F,WALK IN,ear pain,26,2180-01-02 19:52:00,
    30000030,10000030,2180-01-02 20:50:00,2180-01-03 02:35:00,TRANSFER,2,M,AMBULANCE,severe shortness of breath,68,2180-01-02 20:54:00,2180-01-02 21:20:00
    """
)


@pytest.fixture()
def fake_extract(tmp_path: Path) -> Path:
    """Write the invented extract to disk and return its path."""
    path = tmp_path / "fake_mimic_ed.csv"
    path.write_text(FAKE_MIMIC_CSV, encoding="utf-8")
    return path


@pytest.fixture()
def mimic_map() -> ColumnMap:
    """The shipped MIMIC template."""
    return load_column_map("configs/column_map_mimic_ed.yaml")


def test_the_map_says_how_far_it_has_been_verified(mimic_map: ColumnMap) -> None:
    """The label has to match what has actually been checked -- no more, no less.

    This test used to require the words TEMPLATE and UNVERIFIED, because the map was
    written from MIMIC's published schema with no data in hand. It has since been run
    against the open MIMIC-IV-ED demo (222 stays): every column name was confirmed to
    exist and one value mapping was found to be wrong and corrected. So "unverified"
    became false, and a test demanding a false label is worse than no test.

    What must stay true is that the label does not overclaim. The demo is a sample of
    patients, not a window of operation, so it verifies the *schema* and nothing about
    volumes, queue state or the simulation -- and the map's own header says so.
    """
    assert "demo" in mimic_map.dataset_name.lower(), (
        "the name should say the verification was against the demo, not the full dataset"
    )
    assert "UNVERIFIED" not in mimic_map.dataset_name.upper()

    header = (project_root() / "configs" / "column_map_mimic_ed.yaml").read_text(
        encoding="utf-8"
    )[:2000]
    assert "VERIFIED AGAINST THE OPEN DEMO, NOT AGAINST THE FULL DATASET" in header
    assert "real_data_readiness" in header, (
        "the header must still point at what has to happen before a real-data claim"
    )


def test_the_template_ships_with_the_proxy_off(mimic_map: ColumnMap) -> None:
    """Using a proxy has to be a deliberate act, not a default."""
    assert mimic_map.proxy_wait.enabled is False


def test_without_a_proxy_the_missing_seen_time_column_is_fatal(
    fake_extract: Path, mimic_map: ColumnMap
) -> None:
    """MIMIC has no provider-contact time, and the adapter must say so, not guess.

    `seen_time` is required by the schema, so an extract that cannot supply it has to
    stop the run with a message naming the column.
    """
    raw = read_raw(fake_extract, mimic_map.read_options)

    with pytest.raises(ValueError, match="seen_time"):
        adapt_real_dataset(raw, mimic_map, schema=load_schema())


def _with_proxy(mimic_map: ColumnMap, **changes) -> ColumnMap:
    """The template with its proxy switched on."""
    fields = mimic_map.model_dump()
    fields["proxy_wait"] = {
        **fields["proxy_wait"],
        "enabled": True,
        **changes,
    }
    return ColumnMap(**fields)


def test_the_vitals_proxy_fills_seen_time(fake_extract: Path, mimic_map: ColumnMap) -> None:
    """With the proxy on, the extract becomes usable -- as a different quantity."""
    raw = read_raw(fake_extract, mimic_map.read_options)
    adapted = adapt_real_dataset(raw, _with_proxy(mimic_map), schema=load_schema())

    assert "seen_time" in adapted.columns
    # 30 stays, of which two have no vitals timestamp at all.
    assert adapted["seen_time"].notna().sum() == 28
    first = adapted.iloc[0]
    assert first["seen_time"] == pd.Timestamp("2180-01-01 00:31:00")


def test_a_stay_with_no_proxy_timestamp_stays_unseen(
    fake_extract: Path, mimic_map: ColumnMap
) -> None:
    """Two invented stays have neither vitals nor a dispense; they must not be faked."""
    raw = read_raw(fake_extract, mimic_map.read_options)
    adapted = adapt_real_dataset(raw, _with_proxy(mimic_map), schema=load_schema())

    unseen = adapted[adapted["seen_time"].isna()]["patient_id"].tolist()
    assert sorted(unseen) == [30000011, 30000013]


def test_the_two_proxies_give_different_waits(
    fake_extract: Path, mimic_map: ColumnMap
) -> None:
    """The headline caveat, as an executable statement.

    Vitals are charted at triage and dispenses happen after a decision, so the two
    proxies bracket the true door-to-provider time from opposite sides. If they agreed,
    the choice would not matter -- and it does.
    """
    raw = read_raw(fake_extract, mimic_map.read_options)
    vitals = adapt_real_dataset(raw, _with_proxy(mimic_map), schema=load_schema())
    pyxis = adapt_real_dataset(
        raw,
        _with_proxy(
            mimic_map,
            source="first_pyxis_charttime",
            proxy_name="first pyxis charttime (after a clinician decision)",
        ),
        schema=load_schema(),
    )

    vitals_wait = (vitals["seen_time"] - vitals["arrival_time"]).dt.total_seconds() / 60
    pyxis_wait = (pyxis["seen_time"] - pyxis["arrival_time"]).dt.total_seconds() / 60
    both = vitals_wait.notna() & pyxis_wait.notna()

    assert both.sum() > 5
    assert (pyxis_wait[both] > vitals_wait[both]).all(), (
        "a medication dispense always follows the first vitals in this fixture, so the "
        "pyxis proxy must give the longer wait"
    )
    # And the dispense proxy is only available for a subset -- which is the bias.
    assert pyxis_wait.notna().sum() < vitals_wait.notna().sum()


def test_an_unnamed_proxy_is_refused(fake_extract: Path, mimic_map: ColumnMap) -> None:
    """Every number downstream inherits the choice, so the proxy must be named."""
    raw = read_raw(fake_extract, mimic_map.read_options)

    with pytest.raises(ValueError, match="named in words"):
        adapt_real_dataset(
            raw, _with_proxy(mimic_map, proxy_name=None), schema=load_schema()
        )


def test_a_missing_proxy_column_is_refused(fake_extract: Path, mimic_map: ColumnMap) -> None:
    """A typo in the proxy source must stop the run, not silently produce no waits."""
    raw = read_raw(fake_extract, mimic_map.read_options)

    with pytest.raises(ValueError, match="not in the raw file"):
        adapt_real_dataset(
            raw, _with_proxy(mimic_map, source="no_such_column"), schema=load_schema()
        )


def test_mimic_disposition_codes_map_onto_the_schema(
    fake_extract: Path, mimic_map: ColumnMap
) -> None:
    """MIMIC's vocabulary differs from ours, and the mapping must be complete."""
    raw = read_raw(fake_extract, mimic_map.read_options)
    adapted = adapt_real_dataset(raw, _with_proxy(mimic_map), schema=load_schema())

    counts = adapted["disposition"].value_counts()
    assert counts.get("discharged", 0) > 0
    assert counts.get("admitted", 0) > 0
    assert counts.get("transferred", 0) == 2

    # ELOPED and LEFT WITHOUT BEING SEEN collapse to lwbs -- a real loss of
    # information, since MIMIC distinguishes them and this schema has one category.
    #
    # LEFT AGAINST MEDICAL ADVICE does NOT. This test previously required it to, and
    # that was wrong: leaving against advice means a clinician saw the patient and
    # advised them, so the patient *was* seen. Counting it as abandonment would
    # inflate the abandonment rate with patients who had already been assessed -- and
    # abandonment is one of the quantities the simulation's validation gate is judged
    # on. It maps to `other`, and the fixture's EXPIRED row joins it there.
    assert counts.get("lwbs", 0) == 2
    assert counts.get("other", 0) == 2


def test_mimic_arrival_transport_maps_onto_three_categories(
    fake_extract: Path, mimic_map: ColumnMap
) -> None:
    """HELICOPTER is an ambulance to us; UNKNOWN is not."""
    raw = read_raw(fake_extract, mimic_map.read_options)
    adapted = adapt_real_dataset(raw, _with_proxy(mimic_map), schema=load_schema())

    assert set(adapted["arrival_mode"].dropna().unique()) <= {"ambulance", "walk_in", "other"}
    # 9 AMBULANCE rows plus the one HELICOPTER, which is an ambulance arrival to us.
    assert (adapted["arrival_mode"] == "ambulance").sum() == 10
    assert (adapted["arrival_mode"] == "other").sum() == 1  # the single UNKNOWN


def test_a_null_acuity_survives_as_null(fake_extract: Path, mimic_map: ColumnMap) -> None:
    """Triage not completed is a real state; it must not become a guessed level."""
    raw = read_raw(fake_extract, mimic_map.read_options)
    adapted = adapt_real_dataset(raw, _with_proxy(mimic_map), schema=load_schema())

    assert adapted["triage_level"].isna().sum() == 1


def test_the_proxy_status_helper_reports_the_proxy(tmp_path: Path, mimic_map: ColumnMap) -> None:
    """The report and dashboard read this rather than inspecting rows."""
    import yaml

    path = tmp_path / "map.yaml"
    path.write_text(
        yaml.safe_dump(_with_proxy(mimic_map).model_dump()), encoding="utf-8"
    )
    status = proxy_wait_status({"paths": {"column_map": str(path)}})

    assert status["active"] is True
    assert "vitalsign" in status["proxy"]
    assert status["notice"] == PROXY_WAIT_NOTICE


def test_the_notice_says_the_quantity_is_different() -> None:
    """A caveat that only says 'approximate' would be too weak to be useful."""
    assert "NOT TRUE DOOR-TO-PROVIDER" in PROXY_WAIT_NOTICE
    assert "systematically" in PROXY_WAIT_NOTICE


def test_proxy_wait_is_inert_by_default() -> None:
    """The type itself defaults to off, independently of any config file."""
    assert ProxyWait().is_active is False
    assert ProxyWait(enabled=True).is_active is False  # no source named
