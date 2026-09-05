"""The comparison's own rules, which have been wrong more than the code they judge.

Four reports were filed in one day claiming homr had misread music when what had
been compared was the detector. A count rule once called three genuinely missing
notes a beat read differently. A structural disagreement was written up as homr
losing a staff on three systems where the page agreed with homr every time. Each
of those was a rule in `compare.py`, and none of them had a test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fixturecheck.compare import (
    collapse_unisons,
    compare_output,
    printed_staves,
    read_score,
    staves_a_person_counted,
)

PRINTED = Path(__file__).resolve().parent.parent / "fixturecheck" / "printed.json"


def score(staves: list[list[list[tuple]]], path: Path, name: str) -> Path:
    """One MusicXML file: staves x measures x (step, octave, voice, duration)."""
    parts = []
    listing = []
    for index, measures in enumerate(staves, start=1):
        listing.append(f'<score-part id="P{index}"><part-name>V</part-name></score-part>')
        bars = []
        for number, notes in enumerate(measures, start=1):
            attributes = ('<attributes><divisions>1</divisions>'
                          '<clef><sign>G</sign><line>2</line></clef></attributes>'
                          if number == 1 else "")
            written = "".join(
                f"<note><pitch><step>{step}</step><octave>{octave}</octave></pitch>"
                f"<duration>{duration}</duration><voice>{voice}</voice></note>"
                for step, octave, voice, duration in notes)
            bars.append(f'<measure number="{number}">{attributes}{written}</measure>')
        parts.append(f'<part id="P{index}">{"".join(bars)}</part>')
    target = path / f"{name}.musicxml"
    target.write_text(f'<?xml version="1.0"?><score-partwise>'
                      f'<part-list>{"".join(listing)}</part-list>{"".join(parts)}'
                      f'</score-partwise>')
    return target


ONE_NOTE = [[[("C", 4, "1", 1)]]]


def test_a_staff_count_nobody_has_checked_blames_nobody(tmp_path: Path) -> None:
    reference = score(ONE_NOTE, tmp_path, "ref")
    parsed = score(ONE_NOTE * 2, tmp_path, "homr")

    result = compare_output(reference, parsed, "a-system-nobody-looked-at")

    assert result.structure == 1
    assert result.at_fault == ""


def test_the_page_can_say_the_reference_is_the_wrong_one(tmp_path: Path, monkeypatch) -> None:
    """The failure this exists for: three systems blamed on homr, all the reference's."""
    reference = score(ONE_NOTE, tmp_path, "ref")
    parsed = score(ONE_NOTE * 2, tmp_path, "homr")
    monkeypatch.setattr("fixturecheck.compare.staves_a_person_counted", lambda case: 2)

    result = compare_output(reference, parsed, "a-system-somebody-looked-at")

    assert result.at_fault == "reference"
    assert "REFERENCE is wrong" in result.rows[0].verdict


def test_the_page_can_say_homr_is_the_wrong_one(tmp_path: Path, monkeypatch) -> None:
    reference = score(ONE_NOTE * 2, tmp_path, "ref")
    parsed = score(ONE_NOTE, tmp_path, "homr")
    monkeypatch.setattr("fixturecheck.compare.staves_a_person_counted", lambda case: 2)

    result = compare_output(reference, parsed, "a-system-somebody-looked-at")

    assert result.at_fault == "homr"
    assert "HOMR is wrong" in result.rows[0].verdict


def test_the_page_can_agree_with_neither(tmp_path: Path, monkeypatch) -> None:
    reference = score(ONE_NOTE, tmp_path, "ref")
    parsed = score(ONE_NOTE * 2, tmp_path, "homr")
    monkeypatch.setattr("fixturecheck.compare.staves_a_person_counted", lambda case: 3)

    result = compare_output(reference, parsed, "a-system-somebody-looked-at")

    assert result.at_fault == "both"


def test_staves_that_agree_are_nobody_s_fault(tmp_path: Path, monkeypatch) -> None:
    reference = score(ONE_NOTE, tmp_path, "ref")
    parsed = score(ONE_NOTE, tmp_path, "homr")
    monkeypatch.setattr("fixturecheck.compare.staves_a_person_counted", lambda case: 9)

    result = compare_output(reference, parsed, "a-system-somebody-looked-at")

    assert result.structure == 0
    assert result.at_fault == ""


def test_an_unrecorded_system_reads_as_unknown_and_not_as_zero() -> None:
    assert staves_a_person_counted("no-such-system-anywhere") == 0


def test_every_recorded_count_says_what_was_seen() -> None:
    """A number with no reason cannot be checked against the crop, so it is not evidence."""
    recorded = json.loads(PRINTED.read_text())["systems"]
    assert recorded, "the file exists to hold readings; an empty one is a mistake"
    for case, entry in recorded.items():
        assert entry["staves"] >= 1, case
        assert entry.get("why", "").strip(), f"{case} records a count with no reading"


def test_a_bar_short_of_heads_has_lost_notes_and_not_moved_them(tmp_path: Path) -> None:
    """The rule that was too loose: homr writes one head where the page prints two."""
    reference = score([[[("C", 4, "1", 1), ("D", 4, "1", 1)]]], tmp_path, "ref")
    parsed = score([[[("C", 4, "1", 1)]]], tmp_path, "homr")

    result = compare_output(reference, parsed, "case")

    assert result.size == 1
    assert result.timing == 0


def test_a_bar_holding_its_notes_at_other_beats_has_moved_them(tmp_path: Path) -> None:
    reference = score([[[("C", 4, "1", 1), ("D", 4, "1", 1)]]], tmp_path, "ref")
    parsed = score([[[("C", 4, "1", 2), ("D", 4, "1", 1)]]], tmp_path, "homr")

    result = compare_output(reference, parsed, "case")

    # The page sounds something at beat 1 and homr does not, but the bar still
    # holds two heads, so they are late rather than lost.
    assert result.timing == 1
    assert result.size == 0


def test_a_unison_is_one_printed_head_and_counts_once() -> None:
    found = {("1", 1, 0.0): [{"position": 5, "voice": "1", "name": "C4", "stem": "", "chord": False},
                             {"position": 5, "voice": "2", "name": "C4", "stem": "", "chord": False}]}

    assert len(collapse_unisons(found)[("1", 1, 0.0)]) == 1


def test_a_resting_staff_still_counts_as_printed(tmp_path: Path) -> None:
    """Counted off the file, not off the notes: a staff that rests is still a row."""
    target = tmp_path / "s.musicxml"
    target.write_text('<?xml version="1.0"?><score-partwise><part-list>'
                      '<score-part id="P1"><part-name>V</part-name></score-part></part-list>'
                      '<part id="P1"><measure number="1">'
                      '<attributes><divisions>1</divisions><staves>2</staves></attributes>'
                      '</measure></part></score-partwise>')

    assert printed_staves(target) == 2
    assert read_score(target) == {}


@pytest.mark.parametrize("missing", [Path("/nowhere/printed.json")])
def test_a_missing_record_file_is_not_an_error(missing: Path) -> None:
    assert staves_a_person_counted("anything", missing) == 0
