"""A unison the parse holds as one voice: not a fault, not nothing, not silent.

Older choral engraving prints a unison as one notehead serving both parts, so a
reference with one staff per part writes it twice and homr can only ever write
it once. `collapse_unisons` folds the reference's pair so the extra head is not
reported as a lost note -- and until now the match was recorded green, kept out
of the score, and that was the end of it.

It is not the end of it. Nothing was misread, and a part the page means is still
absent from the file: a choir sings it, and everything downstream reads its parts
out of that file. So it is a **warning** -- outside the note percentage, because
no note is wrong, and inside `perfect`, because the five committed fixtures are
held to "nothing to look at here" rather than to "nothing measurably wrong".

The two halves are pinned separately here: that a warning does not move the
score, and that it does stop a fixture being perfect.
"""

from fixturecheck import series
from fixturecheck.compare import Result


def result(**counts) -> Result:
    return Result(**counts)


# --- what a warning is worth ---


def test_a_unison_is_counted_as_a_warning():
    assert result(agree=10, unison=3).warnings == 3


def test_a_warning_is_not_a_fault():
    assert result(agree=10, unison=3).faults == 0


def test_a_warning_stays_out_of_the_score():
    """No note is wrong, so no percentage may move: every figure since
    2026-09-05 has to stay comparable with the ones after this."""
    with_warnings = result(agree=10, unison=3)
    without = result(agree=10)

    assert with_warnings.scored == without.scored == 10
    assert with_warnings.score == without.score == 100.0


def test_a_warning_stops_a_fixture_being_perfect():
    """system4 reads every note right and holds three unisons."""
    assert result(agree=35).perfect
    assert not result(agree=35, unison=3).perfect


def test_a_case_with_nothing_judged_is_still_not_perfect():
    """An unread parse must not pass the gate by having no warnings either."""
    assert not result().perfect


# --- and how the gate says so ---


def test_a_fixture_failing_only_on_warnings_is_named_apart():
    counts = {"agree": 35, "unison": 3, "staves_page": 2, "staves_homr": 2}

    assert series.warnings_only(counts)


def test_a_fixture_with_a_real_fault_is_not_a_warnings_only_one():
    """It fails for the fault; saying "warnings" would send a reader past it."""
    counts = {"agree": 35, "unison": 3, "size": 1, "staves_page": 2, "staves_homr": 2}

    assert not series.warnings_only(counts)


def test_a_fixture_with_no_warnings_is_not_a_warnings_only_one():
    counts = {"agree": 35, "unison": 0, "size": 1, "staves_page": 2, "staves_homr": 2}

    assert not series.warnings_only(counts)


def test_a_staff_disagreement_is_not_a_warning_either():
    counts = {"agree": 35, "unison": 3, "staves_page": 3, "staves_homr": 2}

    assert not series.warnings_only(counts)


def test_the_gate_names_the_two_reasons_apart(tmp_path):
    """`below 100%` is untrue of a fixture whose every note is right."""
    from fixturecheck import quality

    path = tmp_path / "series.jsonl"
    series.record_run(
        "fixturecheck", "one",
        [series.CaseRecord("system4", counts={"agree": 35, "unison": 3, "perfect": False,
                                              "staves_page": 2, "staves_homr": 2}),
         series.CaseRecord("hanget-soi", counts={"agree": 29, "pitch": 1, "perfect": False,
                                                 "staves_page": 2, "staves_homr": 2})],
        references="r", gate={"fixtures": 2, "perfect": 0},
        extra={"committed": ["hanget-soi", "system4"]}, path=path)

    written = quality.render(path)

    assert "below 100%: `hanget-soi`" in written
    assert "holding a unison as one voice: `system4`" in written
