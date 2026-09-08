"""A unison written as one voice: counted and said, never scored, never a gate.

Older choral engraving prints a unison as one notehead serving both parts, so a
reference with one staff per part writes it twice and homr can only write it
once. Nothing there is misread — which is exactly why it went unsaid for so
long, and why a case whose second part is missing from the file could read as a
case with nothing to look at.

Two halves, and the second is the one that keeps this honest: what a warning is
worth, and everything a warning must **not** move.
"""

from __future__ import annotations

from fixturecheck import references, report
from fixturecheck.compare import Result, unison_note

# --- what a warning is worth ---------------------------------------------


def test_a_unison_is_counted_as_a_warning() -> None:
    assert Result(agree=10, unison=3).warnings == 3


def test_a_case_with_no_unisons_warns_about_nothing() -> None:
    assert Result(agree=10).warnings == 0


def test_a_warning_is_not_a_fault() -> None:
    assert Result(agree=10, unison=3).faults == 0


def test_a_warning_is_not_on_the_list_of_what_is_still_wrong() -> None:
    """`remaining` is faults, and a unison is not one. The page says it beside."""
    assert Result(agree=10, unison=3).remaining == []


def test_the_sentence_says_both_halves() -> None:
    """Not misread, and still missing a part — either half alone misleads."""
    said = unison_note(2).lower()

    assert "2 unison" in said
    assert "one voice" in said
    assert "no note is misread" in said
    assert "counted against the score" in said
    assert "silence" in said


# --- and everything it must not move -------------------------------------


def test_a_warning_stays_out_of_the_score() -> None:
    """Every figure quoted since 2026-09-05 has to keep meaning what it meant."""
    without = Result(agree=10)
    with_unisons = Result(agree=10, unison=3)

    assert with_unisons.scored == without.scored == 10
    assert with_unisons.score == without.score == 100.0


def test_a_warning_is_not_in_the_numbers_a_case_is_remembered_by() -> None:
    counts = {
        "agree": 10,
        "voice": 0,
        "pitch": 0,
        "size": 0,
        "timing": 0,
        "staves_page": 2,
        "staves_homr": 2,
        "meter": 0,
    }

    assert references.marks(counts) == references.marks({**counts, "unison": 4})


def test_a_case_that_gained_unisons_does_not_fall_below_its_memory() -> None:
    """The card asked for the operator's eye, not for a gate — see `MEMORY`."""
    was = references.marks({"agree": 10, "staves_page": 2, "staves_homr": 2})
    now = references.marks({"agree": 10, "unison": 4, "staves_page": 2, "staves_homr": 2})

    assert references.worse(now, was) == []


def test_unison_is_not_one_of_the_remembered_numbers() -> None:
    assert "unison" not in references.MEMORY


# --- and what the page shows ---------------------------------------------


def test_a_unison_row_is_amber_rather_than_green() -> None:
    """It was green, which is what made a missing part read as nothing to see."""
    assert report._verdict_class("agree") == "ok"
    assert report._verdict_class("unison") == "warn"
    assert report._verdict_class("pitch") == "no"


def test_a_case_holding_unisons_says_so_beside_its_faults() -> None:
    page = report._still_wrong(Result(agree=10, unison=2), None)

    assert "class='warned'" in page
    assert "2 unison(s) here" in page
    # And it is said beside the faults rather than among them: the list above it
    # is still empty, because no note here is wrong.
    assert "Nothing this check can name is wrong with it" in page


def test_a_calm_case_says_nothing_about_unisons() -> None:
    page = report._still_wrong(Result(agree=10), None)

    assert "class='warned'" not in page
    assert "unison" not in page
