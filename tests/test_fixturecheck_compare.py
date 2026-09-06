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
    _voice_rank,
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


def staff_of_two_voices(path: Path, name: str, notes: list[tuple]) -> Path:
    """One staff, one bar, the notes sounding together — each in its own voice."""
    written = []
    for index, (step, octave, voice) in enumerate(notes):
        if index:
            written.append("<backup><duration>1</duration></backup>")
        written.append(
            f"<note><pitch><step>{step}</step><octave>{octave}</octave></pitch>"
            f"<duration>1</duration><voice>{voice}</voice></note>")
    target = path / f"{name}.musicxml"
    target.write_text(
        '<?xml version="1.0"?><score-partwise><part-list>'
        '<score-part id="P1"><part-name>V</part-name></score-part></part-list>'
        '<part id="P1"><measure number="1"><attributes><divisions>1</divisions>'
        '<clef><sign>G</sign><line>2</line></clef></attributes>'
        f'{"".join(written)}</measure></part></score-partwise>')
    return target


UNISON = [("C", 4, "1"), ("C", 4, "2")]


def test_a_unison_written_into_both_voices_is_agreement(tmp_path: Path) -> None:
    """One printed head carrying two stems is what the page draws.

    Both sides collapse to one head, and both know it serves two parts, so this
    is the moment agreeing rather than a head too many.
    """
    reference = staff_of_two_voices(tmp_path, "ref", UNISON)
    parsed = staff_of_two_voices(tmp_path, "homr", UNISON)

    result = compare_output(reference, parsed, "case")

    assert (result.agree, result.unison, result.size) == (1, 0, 0)


def test_a_unison_the_parse_holds_as_one_voice_is_still_warned(tmp_path: Path) -> None:
    reference = staff_of_two_voices(tmp_path, "ref", UNISON)
    parsed = staff_of_two_voices(tmp_path, "homr", [("C", 4, "1")])

    result = compare_output(reference, parsed, "case")

    assert (result.agree, result.unison, result.size) == (0, 1, 0)


def test_a_head_doubled_where_one_part_sings_it_is_a_fault(tmp_path: Path) -> None:
    """The mark can be read off a head that carries only one stem, and is."""
    reference = staff_of_two_voices(tmp_path, "ref", [("C", 4, "1")])
    parsed = staff_of_two_voices(tmp_path, "homr", UNISON)

    result = compare_output(reference, parsed, "case")

    assert (result.agree, result.unison, result.size) == (0, 0, 1)


def voices_over_bars(path: Path, name: str,
                     bars: list[list[tuple]]) -> Path:
    """One staff, bar by bar, each note `(beat, step, octave, voice)`.

    Beats rather than document order, because that is what the failure is made
    of: a voice entering at a beat the other file has nothing at.
    """
    measures = []
    for number, notes in enumerate(bars, start=1):
        written, at = [], 0.0
        for beat, step, octave, voice in notes:
            if beat < at:
                written.append(f"<backup><duration>{int(at - beat)}"
                               f"</duration></backup>")
            elif beat > at:
                written.append(f"<forward><duration>{int(beat - at)}"
                               f"</duration></forward>")
            written.append(
                f"<note><pitch><step>{step}</step><octave>{octave}</octave>"
                f"</pitch><duration>1</duration><voice>{voice}</voice></note>")
            at = beat + 1
        attributes = ("<attributes><divisions>1</divisions><clef><sign>G</sign>"
                      "<line>2</line></clef></attributes>" if number == 1 else "")
        measures.append(f'<measure number="{number}">{attributes}'
                        f'{"".join(written)}</measure>')
    target = path / f"{name}.musicxml"
    target.write_text(
        '<?xml version="1.0"?><score-partwise><part-list>'
        '<score-part id="P1"><part-name>V</part-name></score-part></part-list>'
        f'<part id="P1">{"".join(measures)}</part></score-partwise>')
    return target


#: The staff both files agree about: the upper line above the lower one, twice.
TOGETHER = [[(1.0, "G", 4, "1"), (1.0, "C", 4, "2")],
            [(0.0, "A", 4, "1"), (0.0, "D", 4, "2")]]


def test_a_voice_entering_early_does_not_reverse_the_staff(tmp_path: Path) -> None:
    """The failure this rule was rewritten for, in the shape it arrived in.

    homr has the lower line enter a beat early, so the first moment of its
    staff holds one note and it is the *lower* voice. Reading the ranking off
    that moment made the lower line rank 1 for the whole page and reported
    every later note as being in the other voice -- 30 of them on Heraa Suomi
    p1, in bars that are note for note right.

    And the moment that decided it is **not in the reference at all**, so the
    loop below never compares it to anything: the ranking turned on a note the
    measurement does not score.
    """
    reference = voices_over_bars(tmp_path, "ref", TOGETHER)
    parsed = voices_over_bars(tmp_path, "homr",
                              [[(0.0, "C", 4, "2")] + TOGETHER[0], TOGETHER[1]])

    result = compare_output(reference, parsed, "case")

    assert (result.agree, result.voice) == (4, 0)


def test_the_line_that_is_higher_in_more_bars_is_the_first_one(tmp_path: Path) -> None:
    """Crossing lines are settled a bar at a time, not by a mean over the staff.

    The upper line dives far below the lower one for one bar of three. Its mean
    height over the staff then comes out the lower of the two, while it is
    still the line above in two bars out of three -- which is what the rank is
    read as saying. `heraa-suomi-final-s10` is this, and a mean reports 6
    faults on a system read correctly.
    """
    crossing = [[(0.0, "G", 5, "1"), (0.0, "C", 4, "2")],
                [(0.0, "A", 5, "1"), (0.0, "D", 4, "2")],
                [(0.0, "C", 1, "1"), (0.0, "E", 4, "2")]]
    ranks = _voice_rank(collapse_unisons(read_score(
        voices_over_bars(tmp_path, "ref", crossing))))

    assert ranks[1] == {"1": 1, "2": 2}


def test_a_line_sung_in_one_bar_does_not_outrank_one_sung_throughout(
        tmp_path: Path) -> None:
    """A scrap of a voice high on the staff is one bar's evidence, not a page's.

    This is the other half of the same failure: on the pre-#153 parses of Heraa
    Suomi p1 the second voice was three notes that happened to sit high, and any
    rule that weighs them against a line sung all the way down gets the staff
    backwards.
    """
    reference = voices_over_bars(tmp_path, "ref", [
        [(0.0, "G", 4, "1"), (0.0, "C", 4, "2")],
        [(0.0, "A", 4, "1"), (0.0, "D", 4, "2")],
        [(0.0, "B", 4, "1"), (0.0, "E", 4, "2")]])
    parsed = voices_over_bars(tmp_path, "homr", [
        [(0.0, "G", 4, "1"), (0.0, "C", 4, "2"), (2.0, "C", 7, "2")],
        [(0.0, "A", 4, "1"), (0.0, "D", 4, "2")],
        [(0.0, "B", 4, "1"), (0.0, "E", 4, "2")]])

    result = compare_output(reference, parsed, "case")

    assert (result.agree, result.voice) == (6, 0)
