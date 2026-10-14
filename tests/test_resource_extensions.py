"""Tests for the uncalibrated resource extensions.

The headline requirement is negative: **with the flags off, nothing changes.** The
extensions add a second constrained resource, a nursing stage and an imaging delay to
a discrete-event model whose output the whole project's staffing result depends on, so
the first thing to prove is that none of it runs unless asked.

The second requirement is that anything produced with them on is labelled
uncalibrated, because the data has no bed, nurse or imaging column and the validation
gate cannot test any of it.
"""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from src.simulation.des import (
    UNCALIBRATED_NOTICE,
    ArrivalModel,
    ResourceExtensions,
    ServiceTimeModel,
    StaffingPlan,
    run_replications,
)

#: Short runs: these tests are about identity and direction, not about precision.
HORIZON_HOURS = 72
REPLICATIONS = 3


def _run(config: dict, extensions: ResourceExtensions | None, prefix: str = "ext_test"):
    """Run a few replications and return (KPI frame, concatenated patient records)."""
    arrivals = ArrivalModel.from_config(config, HORIZON_HOURS)
    frame, results = run_replications(
        config,
        n_replications=REPLICATIONS,
        plan=StaffingPlan.from_config(config),
        arrivals=arrivals,
        service=ServiceTimeModel.from_config(config),
        seed_prefix=prefix,
        horizon_hours=HORIZON_HOURS,
        extensions=extensions,
    )
    patients = pd.concat(
        [r.patients.assign(replication=i) for i, r in enumerate(results)],
        ignore_index=True,
    )
    return frame, patients


# ---------------------------------------------------------------- off means off


def test_the_shipped_config_has_every_extension_off(default_config: dict) -> None:
    """The state the repository ships in, asserted rather than assumed."""
    extensions = ResourceExtensions.from_config(default_config)

    assert extensions.enabled is False
    assert extensions.any_active is False
    assert extensions.describe() == "none (calibrated model)"


def test_a_config_without_the_block_at_all_is_valid(default_config: dict) -> None:
    """Every pre-existing config must keep working untouched."""
    config = copy.deepcopy(default_config)
    config["simulation"].pop("resource_extensions", None)

    assert ResourceExtensions.from_config(config).any_active is False


def test_output_is_identical_whether_extensions_are_absent_or_inert(
    default_config: dict,
) -> None:
    """The guarantee the staffing result rests on.

    Passing no extensions, passing a disabled one, and passing one that is *enabled
    but constrains nothing* must all produce the same patients and the same KPIs --
    byte for byte. If any of them diverged, turning a flag on would perturb unrelated
    parts of the model through the shared random stream, and turning it off again
    would not put them back.
    """
    absent_kpis, absent_patients = _run(default_config, None)
    disabled_kpis, disabled_patients = _run(default_config, ResourceExtensions())
    inert_kpis, inert_patients = _run(
        default_config, ResourceExtensions(enabled=True, n_beds=None, n_triage_nurses=None)
    )

    pd.testing.assert_frame_equal(absent_patients, disabled_patients)
    pd.testing.assert_frame_equal(absent_patients, inert_patients)
    pd.testing.assert_frame_equal(absent_kpis, disabled_kpis)
    pd.testing.assert_frame_equal(absent_kpis, inert_kpis)


def test_an_inactive_run_gains_no_new_kpi_keys(default_config: dict) -> None:
    """A calibrated run's KPI dict must have exactly the keys it always had."""
    kpis, _ = _run(default_config, ResourceExtensions())

    for key in ("uncalibrated", "bed_utilisation", "nurse_utilisation", "n_imaged"):
        assert key not in kpis.columns, f"{key} leaked into a calibrated run"


def test_turning_a_constraint_on_does_change_the_result(default_config: dict) -> None:
    """The mirror of the test above: if nothing changed, nothing is being modelled."""
    _, baseline = _run(default_config, ResourceExtensions())
    _, constrained = _run(default_config, ResourceExtensions(enabled=True, n_beds=1))

    assert not baseline.equals(constrained), (
        "capping beds at 1 must change the simulation, or the constraint is not wired in"
    )


# ---------------------------------------------------------------- the constraints


def test_beds_cap_how_many_patients_are_treated_at_once(default_config: dict) -> None:
    """A bed limit is a limit, not a suggestion."""
    _, patients = _run(default_config, ResourceExtensions(enabled=True, n_beds=2))
    seen = patients.dropna(subset=["seen_minutes"])

    # With two beds, no third patient can start while two are still in treatment.
    for replication, group in seen.groupby("replication"):
        starts = group["seen_minutes"].to_numpy()
        ends = starts + group["service_minutes"].to_numpy()
        for start in starts:
            concurrent = int(((starts <= start) & (ends > start)).sum())
            assert concurrent <= 2, (
                f"replication {replication}: {concurrent} patients in treatment at "
                "once with only 2 beds"
            )


def test_fewer_beds_cannot_shorten_the_queue(default_config: dict) -> None:
    """Removing capacity must not improve the wait; that would be a sign error."""
    loose, _ = _run(default_config, ResourceExtensions(enabled=True, n_beds=12))
    tight, _ = _run(default_config, ResourceExtensions(enabled=True, n_beds=3))

    assert tight["mean_wait_minutes"].mean() >= loose["mean_wait_minutes"].mean()


def test_imaging_lengthens_stay_without_lengthening_treatment(
    default_config: dict,
) -> None:
    """Imaging holds the patient and their bed; it must not occupy the doctor."""
    plain, plain_patients = _run(default_config, ResourceExtensions())
    imaged_kpis, imaged_patients = _run(
        default_config,
        ResourceExtensions(
            enabled=True, imaging_probability=1.0, imaging_turnaround_minutes=30.0,
            imaging_levels=(1, 2, 3),
        ),
    )

    # Treatment durations are untouched.
    assert plain_patients["service_minutes"].sum() == pytest.approx(
        imaged_patients["service_minutes"].sum()
    )
    # But ESI 1-3 patients stay longer.
    high = imaged_patients["triage_level"].isin([1, 2, 3])
    plain_high = plain_patients["triage_level"].isin([1, 2, 3])
    assert imaged_patients.loc[high, "los_minutes"].mean() > (
        plain_patients.loc[plain_high, "los_minutes"].mean()
    )


def test_imaging_only_touches_the_configured_levels(default_config: dict) -> None:
    """ESI 4-5 are not sent for imaging, so their stay must not move."""
    plain, plain_patients = _run(default_config, ResourceExtensions())
    _, imaged = _run(
        default_config,
        ResourceExtensions(
            enabled=True, imaging_probability=1.0, imaging_turnaround_minutes=30.0,
            imaging_levels=(1, 2, 3),
        ),
    )
    low = imaged["triage_level"].isin([4, 5])
    plain_low = plain_patients["triage_level"].isin([4, 5])

    assert imaged.loc[low, "los_minutes"].mean() == pytest.approx(
        plain_patients.loc[plain_low, "los_minutes"].mean(), rel=1e-9
    )


def test_boarding_holds_a_bed_only_when_asked(default_config: dict) -> None:
    """Exit block is a choice about when the bed is freed, and it must bite."""
    released_early, _ = _run(
        default_config, ResourceExtensions(enabled=True, n_beds=4, boarding_holds_bed=False)
    )
    held, _ = _run(
        default_config, ResourceExtensions(enabled=True, n_beds=4, boarding_holds_bed=True)
    )

    assert held["mean_wait_minutes"].mean() > released_early["mean_wait_minutes"].mean(), (
        "holding beds through boarding must lengthen the queue, or exit block is not "
        "being modelled"
    )


def test_a_nurse_bottleneck_delays_entry_to_the_queue(default_config: dict) -> None:
    """One triage nurse with slow triage has to show up as a longer wait."""
    unlimited, _ = _run(default_config, ResourceExtensions(enabled=True))
    bottleneck, _ = _run(
        default_config,
        ResourceExtensions(enabled=True, n_triage_nurses=1, triage_nurse_minutes=20.0),
    )

    assert bottleneck["mean_wait_minutes"].mean() > unlimited["mean_wait_minutes"].mean()


# ---------------------------------------------------------------- the labelling


def test_an_active_run_is_labelled_uncalibrated(default_config: dict) -> None:
    """Nothing produced with an extension on may read as a calibrated result."""
    kpis, _ = _run(default_config, ResourceExtensions(enabled=True, n_beds=3))

    assert kpis["uncalibrated"].all()
    assert "bed_utilisation" in kpis.columns
    assert (kpis["bed_utilisation"] >= 0).all()


def test_the_notice_says_what_is_wrong_with_the_numbers() -> None:
    """The caveat has to be usable, not decorative."""
    assert "NOT CALIBRATED" in UNCALIBRATED_NOTICE
    assert "assumed" in UNCALIBRATED_NOTICE
    for word in ("bed", "nurse", "imaging"):
        assert word in UNCALIBRATED_NOTICE.lower()


def test_describe_names_every_active_constraint() -> None:
    """A reader must be able to tell from one line what was switched on."""
    extensions = ResourceExtensions(
        enabled=True, n_beds=8, n_triage_nurses=2, triage_nurse_minutes=5.0,
        imaging_probability=0.5, imaging_turnaround_minutes=30.0,
        boarding_holds_bed=True,
    )
    described = extensions.describe()

    assert "beds=8" in described
    assert "triage nurses=2" in described
    assert "imaging" in described
    assert "boarding holds a bed" in described


def test_scenarios_in_the_shipped_config_are_all_calibrated(default_config: dict) -> None:
    """None of the existing scenarios may silently acquire an extension."""
    from src.simulation.scenarios import load_scenarios

    for spec in load_scenarios(default_config):
        assert spec.resource_extensions is None, (
            f"scenario {spec.name} gained resource extensions; the shipped scenario "
            "set must stay calibrated"
        )
